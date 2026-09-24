"""
src/medstat/data/missing.py: Headless Missing Data Handling, Imputation & Rubin's Rules Pooling.

Implements:
1. MissingStrategyRequiredError: Hard clinical safety gate preventing silent listwise deletion.
2. Approved strategies: 'complete-case', 'mice', 'knn', 'indicator'.
3. MICEImputer: Multiple Imputation by Chained Equations with sample_posterior=True.
4. Rubin's rules pooling: pool_estimates and pool_regression_results (Barnard-Rubin 1999 df).
5. KNN imputation with automatic feature standardization.
6. Missing indicator method with median/zero filling.
7. prepare_data_for_analysis: Standardized data preparation gate with SampleFlowTracker integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import StandardScaler

from medstat.logging import get_logger

logger = get_logger(__name__)

APPROVED_STRATEGIES = {"complete-case", "mice", "knn", "indicator"}


# ==============================================================================
# 1. Error Hierarchy
# ==============================================================================


class MissingDataError(ValueError):
    """Base exception for missing data handling errors."""


class MissingStrategyRequiredError(MissingDataError):
    """
    CLINICAL SAFETY MANDATE:
    Raised when missing data is detected in required analysis columns without
    an explicit, clinically justified strategy.
    """

    def __init__(
        self,
        message: str | None = None,
        missing_counts: dict[str, int] | None = None,
        missing_pct: dict[str, float] | None = None,
        required_cols: list[str] | None = None,
    ):
        self.missing_counts = missing_counts or {}
        self.missing_pct = missing_pct or {}
        self.required_cols = required_cols or []

        if message is None:
            formatted_vars = "\n".join(
                f"  - '{col}': {count} missing ({self.missing_pct.get(col, 0.0):.1f}%)"
                for col, count in self.missing_counts.items()
                if count > 0
            )
            message = (
                f"[CLINICAL SAFETY ERROR: MissingStrategyRequiredError] Dataset contains missing values in required variables:\n"
                f"{formatted_vars}\n"
                f"Silent listwise deletion is prohibited under clinical safety directives.\n"
                f"You must explicitly specify a missing data strategy and clinical justification:\n"
                f"  --strategy complete-case   (Requires justification explaining MCAR assumption)\n"
                f"  --strategy mice            (Multiple Imputation by Chained Equations; recommended)\n"
                f"  --strategy knn             (K-Nearest Neighbors imputation)\n"
                f"  --strategy indicator       (Missing indicator method with dummy variable)\n"
            )
        super().__init__(message)


# ==============================================================================
# 2. Result Data Structures
# ==============================================================================


@dataclass
class PooledEstimate:
    """Pooled estimate from multiple imputation using Rubin's rules."""

    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    df: float
    p_value: float
    n_imputations: int
    within_variance: float
    between_variance: float
    total_variance: float
    fmi: float  # Fraction of missing information
    lambda_: float  # Proportion of variance due to missingness
    relative_increase_variance: float = 0.0


@dataclass
class PooledRegressionResults:
    """Pooled regression results across m imputations."""

    coefficients: dict[str, PooledEstimate]
    n_observations: int
    n_imputations: int
    model_type: str
    pooled_r_squared: float | None = None
    odds_ratios: dict[str, tuple[float, float, float]] | None = None
    hazard_ratios: dict[str, tuple[float, float, float]] | None = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert pooled results to a DataFrame (parity with legacy tests)."""
        rows = []
        for var_name, est in self.coefficients.items():
            row = {
                "Variable": var_name,
                "Coefficient": est.estimate,
                "SE": est.se,
                "95% CI Lower": est.ci_lower,
                "95% CI Upper": est.ci_upper,
                "P-value": est.p_value,
                "FMI": est.fmi,
            }
            if self.odds_ratios and var_name in self.odds_ratios:
                or_val, or_low, or_high = self.odds_ratios[var_name]
                row["OR"] = or_val
                row["OR 95% CI Lower"] = or_low
                row["OR 95% CI Upper"] = or_high
            if self.hazard_ratios and var_name in self.hazard_ratios:
                hr_val, hr_low, hr_high = self.hazard_ratios[var_name]
                row["HR"] = hr_val
                row["HR 95% CI Lower"] = hr_low
                row["HR 95% CI Upper"] = hr_high
            rows.append(row)
        return pd.DataFrame(rows)


@dataclass
class MICEResult:
    """Result of MICE multiple imputation."""

    imputed_datasets: list[pd.DataFrame]
    n_imputations: int
    columns_imputed: list[str]
    original_missing_mask: pd.DataFrame
    summary_stats: pd.DataFrame = field(default_factory=pd.DataFrame)


# ==============================================================================
# 3. Missing Value Normalization & Inspection Utilities
# ==============================================================================


def apply_missing_values_to_df(
    df: pd.DataFrame,
    var_meta: dict[str, Any] | None = None,
    missing_codes: list[Any] | dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Replace user-specified missing value codes (e.g. -99, 999, 'NA') with NaN."""
    df_copy = df.copy()
    var_meta = var_meta or {}

    for col in df_copy.columns:
        codes = []
        if col in var_meta and "missing_values" in var_meta[col]:
            codes = var_meta[col]["missing_values"]
        elif isinstance(missing_codes, dict):
            codes = missing_codes.get(col, [])
        elif missing_codes:
            codes = missing_codes

        if not isinstance(codes, (list, tuple, set, np.ndarray)):
            codes = [codes]

        normalized_codes = set()
        for code in codes:
            if pd.isna(code):
                continue
            normalized_codes.add(code)
            normalized_codes.add(str(code))
            try:
                normalized_codes.add(float(code))
            except (ValueError, TypeError):
                pass

        if normalized_codes:
            df_copy[col] = df_copy[col].replace(list(normalized_codes), np.nan)

    return df_copy


def detect_missing_in_variable(
    series: pd.Series,
    missing_codes: list[Any] | None = None,
    already_normalized: bool = False,
) -> dict[str, Any]:
    """Detect and count missing values in a single series."""
    total_count = len(series)
    missing_nan_count = int(series.isna().sum())
    missing_coded_count = 0

    if missing_codes and not already_normalized:
        for code in missing_codes:
            if pd.isna(code):
                continue
            missing_coded_count += int((series == code).sum())

    total_missing = missing_nan_count + missing_coded_count
    missing_pct = (
        round((total_missing / total_count * 100), 2) if total_count > 0 else 0.0
    )

    return {
        "total_count": total_count,
        "missing_count": total_missing,
        "missing_pct": missing_pct,
        "missing_coded_count": missing_coded_count,
        "missing_nan_count": missing_nan_count,
        "valid_count": total_count - total_missing,
    }


def get_missing_summary_df(
    df: pd.DataFrame,
    var_meta: dict[str, Any] | None = None,
    missing_codes: list[Any] | None = None,
    already_normalized: bool = False,
) -> pd.DataFrame:
    """Build a sorted summary DataFrame of missingness per variable."""
    var_meta = var_meta or {}
    rows = []
    for col in df.columns:
        var_type = (
            "Continuous" if pd.api.types.is_numeric_dtype(df[col]) else "Categorical"
        )
        if col in var_meta and "type" in var_meta[col]:
            var_type = var_meta[col]["type"]

        stats_dict = detect_missing_in_variable(
            df[col], missing_codes=missing_codes, already_normalized=already_normalized
        )
        rows.append(
            {
                "Variable": col,
                "Type": var_type,
                "N_Total": stats_dict["total_count"],
                "N_Valid": stats_dict["valid_count"],
                "N_Missing": stats_dict["missing_count"],
                "Pct_Missing": f"{stats_dict['missing_pct']}%",
            }
        )

    summary_df = pd.DataFrame(rows)
    return summary_df.sort_values(by="N_Missing", ascending=False).reset_index(
        drop=True
    )


# ==============================================================================
# 4. Imputation Algorithms
# ==============================================================================


class MICEImputer:
    """
    Multiple Imputation by Chained Equations (MICE) Imputer.
    Uses IterativeImputer with sample_posterior=True.
    """

    def __init__(
        self,
        n_imputations: int = 5,
        max_iter: int = 10,
        random_state: int = 42,
    ):
        if n_imputations < 1:
            raise ValueError("n_imputations must be >= 1")
        self.n_imputations = n_imputations
        self.max_iter = max_iter
        self.random_state = random_state

    def fit_transform(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> MICEResult:
        df_work = df.copy()

        if columns is None:
            numeric_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
        else:
            invalid_cols = [
                c
                for c in columns
                if c not in df_work.columns
                or not pd.api.types.is_numeric_dtype(df_work[c])
            ]
            if invalid_cols:
                raise ValueError(f"Invalid columns for MICE: {invalid_cols}")
            numeric_cols = [
                c for c in columns if pd.api.types.is_numeric_dtype(df_work[c])
            ]

        missing_mask = df_work[numeric_cols].isna()
        cols_with_missing = [c for c in numeric_cols if missing_mask[c].any()]

        if not cols_with_missing:
            return MICEResult(
                imputed_datasets=[df_work.copy()],
                n_imputations=1,
                columns_imputed=[],
                original_missing_mask=missing_mask,
            )

        imputed_datasets = []
        for m in range(self.n_imputations):
            rng_seed = self.random_state + m * 1000
            imputer = IterativeImputer(
                max_iter=self.max_iter,
                random_state=rng_seed,
                sample_posterior=True,
            )
            imputed_vals = imputer.fit_transform(df_work[numeric_cols])
            df_imp = df_work.copy()
            df_imp[numeric_cols] = imputed_vals
            imputed_datasets.append(df_imp)

        result = MICEResult(
            imputed_datasets=imputed_datasets,
            n_imputations=self.n_imputations,
            columns_imputed=cols_with_missing,
            original_missing_mask=missing_mask,
        )
        result.summary_stats = get_imputation_summary(result)
        return result


def impute_knn(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    n_neighbors: int = 5,
    weights: Literal["uniform", "distance"] = "uniform",
    scale: bool = True,
    metric: str = "nan_euclidean",
) -> pd.DataFrame:
    """
    K-Nearest Neighbors imputation with optional feature standardization.
    """
    df_out = df.copy()
    if columns is None:
        numeric_cols = df_out.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df_out[c])]

    if not numeric_cols:
        return df_out

    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights, metric=metric)

    if scale:
        scaler = StandardScaler()
        scaled_vals = scaler.fit_transform(df_out[numeric_cols])
        imputed_scaled = imputer.fit_transform(scaled_vals)
        df_out[numeric_cols] = scaler.inverse_transform(imputed_scaled)
    else:
        df_out[numeric_cols] = imputer.fit_transform(df_out[numeric_cols])

    return df_out


def impute_indicator(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    continuous_fill: Literal["median", "mean", "zero"] = "median",
    suffix: str = "_missing",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Missing indicator method: appends binary flag columns and fills continuous values.
    """
    df_out = df.copy()
    target_cols = columns or df_out.columns.tolist()
    added_cols = []

    for col in target_cols:
        if df_out[col].isna().any():
            indicator_col = f"{col}{suffix}"
            df_out[indicator_col] = df_out[col].isna().astype(int)
            added_cols.append(indicator_col)

            if pd.api.types.is_numeric_dtype(df_out[col]):
                if continuous_fill == "median":
                    fill_val = float(df_out[col].median())
                elif continuous_fill == "mean":
                    fill_val = float(df_out[col].mean())
                else:
                    fill_val = 0.0
                df_out[col] = df_out[col].fillna(fill_val)
            else:
                df_out[col] = df_out[col].fillna("Missing")

    return df_out, added_cols


def get_imputation_summary(mice_result: MICEResult) -> pd.DataFrame:
    """Generate summary statistics of imputed values across imputations."""
    if not mice_result.imputed_datasets or not mice_result.columns_imputed:
        return pd.DataFrame()

    summary_rows = []
    for col in mice_result.columns_imputed:
        imputed_vals = []
        for df_imp in mice_result.imputed_datasets:
            mask = mice_result.original_missing_mask[col]
            imputed_vals.extend(df_imp.loc[mask, col].values)

        arr = np.array(imputed_vals, dtype=float)
        summary_rows.append(
            {
                "Variable": col,
                "N_Imputed": len(arr) // mice_result.n_imputations,
                "Mean": float(np.mean(arr)),
                "SD": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "Min": float(np.min(arr)),
                "Max": float(np.max(arr)),
            }
        )

    return pd.DataFrame(summary_rows)


def create_imputation_diagnostics(
    mice_result: MICEResult,
    original_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Create diagnostic summaries or plots for multiple imputation.
    Returns a dictionary of figures or summary objects.
    """
    figures: dict[str, Any] = {}
    if not mice_result.columns_imputed:
        return figures

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        n_cols = len(mice_result.columns_imputed)
        fig_density = make_subplots(
            rows=1,
            cols=min(n_cols, 3),
            subplot_titles=mice_result.columns_imputed[:3],
        )

        for i, col in enumerate(mice_result.columns_imputed[:3]):
            col_idx = i + 1
            mask = (
                mice_result.original_missing_mask[col]
                if mice_result.original_missing_mask is not None
                else pd.Series(False, index=original_df.index)
            )
            observed = original_df.loc[~mask, col].dropna()

            if len(observed) > 0:
                fig_density.add_trace(
                    go.Histogram(
                        x=observed,
                        name=f"{col} (Observed)",
                        histnorm="probability density",
                        opacity=0.7,
                    ),
                    row=1,
                    col=col_idx,
                )

            if mice_result.imputed_datasets:
                imputed_values = mice_result.imputed_datasets[0].loc[mask, col]
                if len(imputed_values) > 0:
                    fig_density.add_trace(
                        go.Histogram(
                            x=imputed_values,
                            name=f"{col} (Imputed)",
                            histnorm="probability density",
                            opacity=0.5,
                        ),
                        row=1,
                        col=col_idx,
                    )

        figures["density_comparison"] = fig_density
    except ImportError:
        # Plotly not available; return structured summary dictionary
        figures["summary"] = get_imputation_summary(mice_result).to_dict(
            orient="records"
        )

    return figures


# ==============================================================================
# 5. Rubin's Rules Pooling Utilities
# ==============================================================================


def pool_estimates(
    estimates: list[float] | np.ndarray,
    variances: list[float] | np.ndarray,
    n_obs: int | None = None,
    alpha: float = 0.05,
    k: int = 1,
    df_complete: int | None = None,
) -> PooledEstimate:
    """
    Pool estimates and variances across m multiple imputations using Rubin's rules
    with Barnard-Rubin (1999) small-sample adjusted degrees of freedom.
    """
    m = len(estimates)
    if m != len(variances):
        raise ValueError("Number of estimates must match number of variances")

    theta = np.asarray(estimates, dtype=float)
    V = np.asarray(variances, dtype=float)

    if m < 2:
        est = float(theta[0])
        var = float(V[0])
        se = float(np.sqrt(var))
        df_single = np.inf
        t_crit = float(stats.t.ppf(1 - alpha / 2, df_single))
        p_val = float(2 * stats.t.sf(abs(est / se), df_single)) if se > 0 else 1.0
        return PooledEstimate(
            estimate=est,
            se=se,
            ci_lower=est - t_crit * se,
            ci_upper=est + t_crit * se,
            df=df_single,
            p_value=p_val,
            n_imputations=1,
            within_variance=var,
            between_variance=0.0,
            total_variance=var,
            fmi=0.0,
            lambda_=0.0,
            relative_increase_variance=0.0,
        )

    theta_bar = float(np.mean(theta))
    W_bar = float(np.mean(V))
    B = float(np.var(theta, ddof=1))
    T = float(W_bar + (1.0 + 1.0 / m) * B)
    se = float(np.sqrt(T))

    if B > 0 and W_bar > 0:
        r = float((1.0 + 1.0 / m) * B / W_bar)
        df_old = (m - 1) * (1.0 + 1.0 / r) ** 2
        if n_obs is not None or df_complete is not None:
            nu_com = (
                float(df_complete)
                if df_complete is not None
                else max(1.0, float(n_obs - k))
            )
            lam = r / (r + 1.0)
            df_obs = ((nu_com + 1.0) / (nu_com + 3.0)) * nu_com * (1.0 - lam)
            df = float((df_old * df_obs) / (df_old + df_obs))
        else:
            df = float(df_old)
    elif B > 0 and W_bar <= 0:
        r = np.inf
        df = float(m - 1)
    else:
        r = 0.0
        if n_obs is not None or df_complete is not None:
            nu_com = (
                float(df_complete)
                if df_complete is not None
                else max(1.0, float(n_obs - k))
            )
            df = float(((nu_com + 1.0) / (nu_com + 3.0)) * nu_com)
        else:
            df = np.inf

    t_stat = theta_bar / se if se > 0 else 0.0
    if np.isfinite(df) and df > 0:
        p_value = float(2.0 * stats.t.sf(abs(t_stat), df))
        t_crit = float(stats.t.ppf(1 - alpha / 2, df))
    else:
        p_value = float(2.0 * stats.norm.sf(abs(t_stat)))
        t_crit = float(stats.norm.ppf(1 - alpha / 2))

    ci_lower = float(theta_bar - t_crit * se)
    ci_upper = float(theta_bar + t_crit * se)

    lambda_ = float((B + B / m) / T) if T > 0 else 0.0
    if np.isinf(r):
        fmi = 1.0
    elif r > 0 and np.isfinite(df):
        fmi = float((r + 2.0 / (df + 3.0)) / (r + 1.0))
    else:
        fmi = 0.0

    return PooledEstimate(
        estimate=theta_bar,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        df=df,
        p_value=p_value,
        n_imputations=m,
        within_variance=W_bar,
        between_variance=B,
        total_variance=T,
        fmi=fmi,
        lambda_=lambda_,
        relative_increase_variance=r,
    )


def pool_regression_results(
    models: list[Any],
    model_type: Literal[
        "linear", "logistic", "firth_logistic", "cox", "firth_cox", "poisson", "negbin"
    ] = "linear",
    get_params: Callable[[Any], tuple[pd.Series, pd.Series]] | None = None,
    alpha: float = 0.05,
) -> PooledRegressionResults:
    """Pool regression model coefficients across m multiple imputations."""
    m = len(models)
    if m == 0:
        raise ValueError("No models provided for pooling")

    first_model = models[0]
    if get_params is not None:
        p0, _ = get_params(first_model)
    elif hasattr(first_model, "params") and hasattr(first_model, "bse"):
        p0 = first_model.params
    elif hasattr(first_model, "params_") and hasattr(first_model, "standard_errors_"):
        p0 = first_model.params_
    else:
        raise ValueError("Cannot extract parameters from model. Provide get_params.")

    var_names = (
        list(p0.index) if hasattr(p0, "index") else [f"x{i}" for i in range(len(p0))]
    )
    n_obs = getattr(first_model, "nobs", getattr(first_model, "n_obs", None))

    pooled_coefs: dict[str, PooledEstimate] = {}
    odds_ratios: dict[str, tuple[float, float, float]] = {}
    hazard_ratios: dict[str, tuple[float, float, float]] = {}

    for var in var_names:
        estimates = []
        variances = []
        for model in models:
            if get_params is not None:
                p, b = get_params(model)
            elif hasattr(model, "params") and hasattr(model, "bse"):
                p, b = model.params, model.bse
            else:
                p, b = model.params_, model.standard_errors_

            idx = (
                list(p.index).index(var)
                if hasattr(p, "index")
                else var_names.index(var)
            )
            est_val = float(p.iloc[idx]) if hasattr(p, "iloc") else float(p[idx])
            se_val = float(b.iloc[idx]) if hasattr(b, "iloc") else float(b[idx])

            estimates.append(est_val)
            variances.append(se_val**2)

        pooled = pool_estimates(
            estimates, variances, n_obs=int(n_obs) if n_obs else None, alpha=alpha
        )
        pooled_coefs[var] = pooled

        if model_type in ("logistic", "firth_logistic"):
            odds_ratios[var] = (
                float(np.exp(pooled.estimate)),
                float(np.exp(pooled.ci_lower)),
                float(np.exp(pooled.ci_upper)),
            )
        elif model_type in ("cox", "firth_cox"):
            hazard_ratios[var] = (
                float(np.exp(pooled.estimate)),
                float(np.exp(pooled.ci_lower)),
                float(np.exp(pooled.ci_upper)),
            )

    r_squared = None
    if hasattr(first_model, "rsquared"):
        r_sq_vals = [m_obj.rsquared for m_obj in models if hasattr(m_obj, "rsquared")]
        if r_sq_vals:
            r_squared = float(np.mean(r_sq_vals))

    return PooledRegressionResults(
        coefficients=pooled_coefs,
        n_observations=int(n_obs) if n_obs else 0,
        n_imputations=m,
        model_type=model_type,
        pooled_r_squared=r_squared,
        odds_ratios=odds_ratios if odds_ratios else None,
        hazard_ratios=hazard_ratios if hazard_ratios else None,
    )


# ==============================================================================
# 6. Primary Ingestion Gate: prepare_data_for_analysis
# ==============================================================================


def prepare_data_for_analysis(
    df: pd.DataFrame,
    required_cols: list[str],
    numeric_cols: list[str] | None = None,
    handle_missing: str | None = None,
    missing_justification: str | None = None,
    tracker: Any | None = None,
    var_meta: dict[str, Any] | None = None,
    missing_codes: list[Any] | dict[str, Any] | None = None,
    strategy_params: dict[str, Any] | None = None,
    return_info: bool = True,
    **kwargs: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Standardized clinical data preparation gate enforcing explicit missingness strategy.

    Parameters:
        df: Input DataFrame.
        required_cols: Columns required for downstream model or calculation.
        numeric_cols: Columns required to be numeric.
        handle_missing: One of 'complete-case', 'mice', 'knn', 'indicator'.
        missing_justification: Documented clinical rationale for strategy.
        tracker: Optional SampleFlowTracker.
        var_meta: Optional variable metadata mapping.
        missing_codes: Custom missing value codes.
        strategy_params: Optional hyperparameters for imputation.
        return_info: Whether to return info dict.
        **kwargs: Additional hyperparameters passed to strategy_params (e.g. n_imputations, n_neighbors).

    Returns:
        tuple[pd.DataFrame, dict[str, Any]]: Cleaned data and metadata summary.

    Raises:
        ValueError: If input DataFrame is empty or required columns missing.
        MissingStrategyRequiredError: If missing data exists without strategy or justification.
    """
    strategy_params = dict(strategy_params or {})
    strategy_params.update(kwargs)

    # 1. Basic validation
    if len(df) == 0:
        raise ValueError("Input DataFrame is empty (0 rows).")

    # Deduplicate required_cols preserving order
    required_cols = list(dict.fromkeys(required_cols))
    missing_in_df = set(required_cols) - set(df.columns)
    if missing_in_df:
        raise ValueError(f"Required columns missing from DataFrame: {missing_in_df}")

    df_subset = df[required_cols].copy()

    # 2. Normalize user missing codes to NaN
    if var_meta or missing_codes:
        df_subset = apply_missing_values_to_df(df_subset, var_meta, missing_codes)

    # 3. Numeric coercion for numeric_cols
    if numeric_cols:
        for col in numeric_cols:
            if col in df_subset.columns:
                df_subset[col] = pd.to_numeric(df_subset[col], errors="coerce")

    # 4. Detect 100% missing columns
    for col in required_cols:
        if df_subset[col].isna().all():
            raise ValueError(
                f"Column '{col}' is 100% missing and cannot be analyzed or imputed."
            )

    # 5. Missingness audit across required columns
    missing_counts = {col: int(df_subset[col].isna().sum()) for col in required_cols}
    missing_pct = {
        col: round(missing_counts[col] / len(df_subset) * 100, 2)
        for col in required_cols
    }
    total_missing_cells = sum(missing_counts.values())

    original_rows = len(df_subset)

    # 6. Safety Gate: Missing values present
    if total_missing_cells > 0:
        if handle_missing is None:
            raise MissingStrategyRequiredError(
                missing_counts=missing_counts,
                missing_pct=missing_pct,
                required_cols=required_cols,
            )

        strategy_norm = handle_missing.lower().replace("_", "-")
        if strategy_norm not in APPROVED_STRATEGIES:
            raise ValueError(
                f"Unknown missing data strategy: '{handle_missing}'. "
                f"Approved clinical strategies: {sorted(APPROVED_STRATEGIES)}"
            )

        if not missing_justification or not str(missing_justification).strip():
            raise MissingStrategyRequiredError(
                message=(
                    f"[CLINICAL SAFETY ERROR] Strategy '{handle_missing}' selected for dataset "
                    f"with {total_missing_cells} missing values, but no clinical justification was provided. "
                    f"A documented clinical justification is required under STROBE/CONSORT standards."
                ),
                missing_counts=missing_counts,
                missing_pct=missing_pct,
                required_cols=required_cols,
            )
    else:
        strategy_norm = (handle_missing or "none").lower().replace("_", "-")

    # 7. Execute Strategy
    imputed_datasets = None
    mice_result = None
    added_columns: list[str] = []

    if total_missing_cells == 0 or strategy_norm == "none":
        df_clean = df_subset
        rows_excluded = 0

    elif strategy_norm == "complete-case":
        df_clean = df_subset.dropna()
        rows_excluded = original_rows - len(df_clean)
        if len(df_clean) == 0:
            raise ValueError(
                "No complete observations remaining after complete-case deletion."
            )

        if tracker is not None and hasattr(tracker, "record_stage"):
            tracker.record_stage(
                stage_name="missing_covariates",
                n_remaining=len(df_clean),
                n_excluded=rows_excluded,
                reason=f"Complete-case missing data exclusion: {missing_justification}",
            )

    elif strategy_norm == "mice":
        non_numeric_missing = [
            col
            for col in required_cols
            if missing_counts.get(col, 0) > 0
            and not pd.api.types.is_numeric_dtype(df_subset[col])
        ]
        if non_numeric_missing:
            raise ValueError(
                f"MICE imputation requires numeric columns; non-numeric columns with missing data: {non_numeric_missing}"
            )

        m_imputations = strategy_params.get(
            "n_imputations", strategy_params.get("m", 5)
        )
        max_iter = strategy_params.get("max_iter", 10)
        random_state = strategy_params.get("random_state", 42)

        imputer = MICEImputer(
            n_imputations=m_imputations, max_iter=max_iter, random_state=random_state
        )
        mice_result = imputer.fit_transform(df_subset)
        df_clean = mice_result.imputed_datasets[0]
        imputed_datasets = mice_result.imputed_datasets
        rows_excluded = 0

        if tracker is not None and hasattr(tracker, "record_stage"):
            tracker.record_stage(
                stage_name="missing_covariates",
                n_remaining=original_rows,
                n_excluded=0,
                reason=f"MICE multiple imputation ({m_imputations} imputations): {missing_justification}",
            )

    elif strategy_norm == "knn":
        non_numeric_missing = [
            col
            for col in required_cols
            if missing_counts.get(col, 0) > 0
            and not pd.api.types.is_numeric_dtype(df_subset[col])
        ]
        if non_numeric_missing:
            raise ValueError(
                f"KNN imputation requires numeric columns; non-numeric columns with missing data: {non_numeric_missing}"
            )

        k_neighbors = strategy_params.get("n_neighbors", strategy_params.get("k", 5))
        df_clean = impute_knn(df_subset, n_neighbors=k_neighbors, scale=True)
        rows_excluded = 0

        if tracker is not None and hasattr(tracker, "record_stage"):
            tracker.record_stage(
                stage_name="missing_covariates",
                n_remaining=original_rows,
                n_excluded=0,
                reason=f"KNN imputation (k={k_neighbors}): {missing_justification}",
            )

    elif strategy_norm == "indicator":
        df_clean, added_columns = impute_indicator(df_subset, continuous_fill="median")
        rows_excluded = 0

        if tracker is not None and hasattr(tracker, "record_stage"):
            tracker.record_stage(
                stage_name="missing_covariates",
                n_remaining=original_rows,
                n_excluded=0,
                reason=f"Missing indicator method: {missing_justification}",
            )

    # Validate that df_clean contains no missing values in required_cols
    remaining_missing = [
        col
        for col in required_cols
        if col in df_clean.columns and df_clean[col].isna().any()
    ]
    if remaining_missing:
        raise ValueError(
            f"Missing data handling failed to resolve missing values in required columns: {remaining_missing}"
        )

    info: dict[str, Any] = {
        "strategy": strategy_norm,
        "missing_justification": missing_justification,
        "rows_original": original_rows,
        "rows_analyzed": len(df_clean),
        "rows_excluded": rows_excluded,
        "missing_counts": missing_counts,
        "missing_pct": missing_pct,
        "added_columns": added_columns,
        "imputed_datasets": imputed_datasets,
        "mice_result": mice_result,
    }

    return df_clean, info


def handle_missing_for_analysis(
    df: pd.DataFrame,
    var_meta: dict[str, Any] | None = None,
    strategy: str | None = None,
    return_counts: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, int]]:
    """
    Backward-compatible legacy wrapper for complete-case filtering.
    Requires an explicit strategy.
    """
    if strategy is None:
        raise MissingStrategyRequiredError(
            "An explicit missing-data strategy must be specified for handle_missing_for_analysis."
        )
    df_work = df.copy()
    if var_meta:
        df_work = apply_missing_values_to_df(df_work, var_meta)

    original_rows = len(df_work)
    if strategy in ("complete-case", "complete_case", "drop"):
        df_clean = df_work.dropna()
    else:
        df_clean = df_work

    final_rows = len(df_clean)
    rows_removed = original_rows - final_rows

    if return_counts:
        return df_clean, {
            "original_rows": original_rows,
            "final_rows": final_rows,
            "rows_removed": rows_removed,
        }
    return df_clean

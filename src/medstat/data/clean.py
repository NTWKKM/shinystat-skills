"""
src/medstat/data/clean.py: Clinically Sound Data Cleaning, Sanitization,
Missingness Audit & Little's MCAR Test.

Decoupled pure headless biostatistical calculation engine with zero Shiny/UI dependencies.
Apache-2.0 License.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from medstat.data.missing import (
    MissingDataError,
    MissingStrategyRequiredError,
    apply_missing_values_to_df,
    detect_missing_in_variable,
    get_missing_summary_df,
)
from medstat.logging import get_logger

logger = get_logger(__name__)


# ==============================================================================
# Exceptions
# ==============================================================================


class DataCleaningError(Exception):
    """Custom exception for data cleaning and numeric coercion errors."""


class DataValidationError(ValueError):
    """Custom exception for data validation failures (inherits from ValueError)."""


# ==============================================================================
# Structured Result Dataclasses
# ==============================================================================


@dataclass(frozen=True)
class VariableMissingAudit:
    """Missingness audit metrics for an individual variable."""

    variable: str
    dtype: str
    inferred_type: str  # "Continuous", "Categorical", "Binary", "ID"
    total_count: int
    valid_count: int
    missing_count: int
    missing_pct: float
    risk_tier: (
        str  # "Low (<5%)", "Moderate (5-20%)", "High (20-40%)", "Critical (>40%)"
    )
    recommendation: str
    missing_coded_count: int = 0
    missing_nan_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MissingPattern:
    """Missingness pattern across observations."""

    pattern_id: int
    observed_vars: list[str]
    missing_vars: list[str]
    row_count: int
    pct: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MissingnessAudit:
    """Comprehensive missingness audit report."""

    total_rows: int
    total_columns: int
    total_cells: int
    total_missing_cells: int
    overall_missing_pct: float
    variables: dict[str, VariableMissingAudit]
    patterns: list[MissingPattern]
    co_occurrence: dict[str, dict[str, float]]
    has_critical_missing: bool
    summary_df: pd.DataFrame

    def to_dict(self) -> dict[str, Any]:
        base = {
            "summary": {
                "total_rows": self.total_rows,
                "total_columns": self.total_columns,
                "total_cells": self.total_cells,
                "total_missing_cells": self.total_missing_cells,
                "overall_missing_pct": self.overall_missing_pct,
                "has_critical_missing": self.has_critical_missing,
            },
            "variables": {k: v.to_dict() for k, v in self.variables.items()},
            "patterns": [p.to_dict() for p in self.patterns],
            "co_occurrence": self.co_occurrence,
        }
        for k, v in self.variables.items():
            base[k] = v.to_dict()
        return base

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def __getitem__(self, key: str) -> Any:
        d = self.to_dict()
        if key in d:
            return d[key]
        if key in self.variables:
            return self.variables[key].to_dict()
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return key in self.to_dict() or key in self.variables


@dataclass
class OutlierResult:
    """Outlier detection and treatment result."""

    treated_series: pd.Series
    outlier_mask: pd.Series
    method: str
    action: str
    threshold: float
    outlier_count: int
    outlier_pct: float
    lower_bound: float
    upper_bound: float
    stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "action": self.action,
            "threshold": self.threshold,
            "outlier_count": self.outlier_count,
            "outlier_pct": self.outlier_pct,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "stats": self.stats,
        }


@dataclass
class LittlesMCARResult:
    """Statistical outcome of Little's MCAR test."""

    statistic: float  # Little's d2 chi-square statistic
    df: int  # Degrees of freedom: sum(p_s) - P
    p_value: float  # Asymptotic p-value from chi2.sf(d2, df)
    is_mcar: bool  # True if p_value > 0.05
    n_patterns: int  # Number of unique observed patterns
    n_observations: int  # Total observations analyzed
    n_variables: int  # Number of continuous variables tested
    pattern_counts: dict[str, int]
    em_converged: bool
    em_iterations: int
    mu_mle: dict[str, float]
    sigma_mle: dict[str, dict[str, float]]
    message: str

    @property
    def pvalue(self) -> float:
        return self.p_value

    def to_dict(self) -> dict[str, Any]:
        return {
            "statistic": float(self.statistic),
            "df": int(self.df),
            "p_value": float(self.p_value),
            "pvalue": float(self.p_value),
            "is_mcar": bool(self.is_mcar),
            "n_patterns": int(self.n_patterns),
            "n_observations": int(self.n_observations),
            "n_variables": int(self.n_variables),
            "pattern_counts": self.pattern_counts,
            "em_converged": bool(self.em_converged),
            "em_iterations": int(self.em_iterations),
            "mu_mle": self.mu_mle,
            "sigma_mle": self.sigma_mle,
            "message": self.message,
        }

    def __getitem__(self, key: str) -> Any:
        d = self.to_dict()
        if key in d:
            return d[key]
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return key in self.to_dict()


# ==============================================================================
# 1. Numeric Sanitization
# ==============================================================================


_SPECIAL_CHARS_RE = re.compile(r"[<>=,%$€£฿¥]")


def clean_numeric(
    val: Any,
    handle_special_chars: bool = True,
    remove_whitespace: bool = True,
) -> float:
    """
    Sanitize a single scalar value into a clean float.

    Handles:
    - Comparison operators: '>', '<', '>=', '<='
    - Currencies: '$', '€', '£', '฿', '¥'
    - Thousand separators: ','
    - Percentages: '10%' -> 10.0
    - Financial parentheses negative notation: '(100)' -> -100.0
    - Unicode minus symbols: '−' (\u2212), en-dash '–' (\u2013)
    - Empty, None, or unparseable text -> np.nan
    """
    if val is None or pd.isna(val):
        return np.nan

    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)

    s = str(val)
    if remove_whitespace:
        s = s.strip()

    if not s:
        return np.nan

    # Reject ambiguous clinical values containing ±, ~, |, or whitespace between digits
    if any(char in s for char in ("±", "~", "|")):
        return np.nan
    if re.search(r"\d\s+\d", s):
        return np.nan

    # Standardize unicode minus characters
    s = s.replace("−", "-").replace("–", "-")

    # Financial parentheses negative notation: '(100)' -> '-100'
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1].strip()

    if handle_special_chars:
        # Strip comparison operators and currency symbols
        s = _SPECIAL_CHARS_RE.sub("", s)

    if not s:
        return np.nan

    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan


def clean_numeric_vector(
    series: pd.Series | np.ndarray | Sequence[Any],
) -> pd.Series:
    """
    Vectorized sanitization of a Series or array into a float Series.
    """
    if not isinstance(series, pd.Series):
        s = pd.Series(series)
    else:
        s = series.copy()

    # Fast path if already numeric
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)

    # Convert to string and apply regex cleaning
    str_s = s.astype(str).str.strip()
    str_s = str_s.str.replace("−", "-", regex=False).str.replace("–", "-", regex=False)

    # Detect ambiguous values containing ±, ~, |, or whitespace between digits
    ambiguous_mask = str_s.str.contains(r"[±~|]", regex=True) | str_s.str.contains(
        r"\d\s+\d", regex=True
    )

    # Handle financial parentheses '(100)' -> '-100'
    parens_mask = str_s.str.startswith("(") & str_s.str.endswith(")")
    str_s.loc[parens_mask] = "-" + str_s.loc[parens_mask].str.slice(1, -1).str.strip()

    # Strip symbols
    cleaned_str = str_s.str.replace(r"[<>=,%$€£฿¥]", "", regex=True)

    # Coerce to float, invalid entries and ambiguous become NaN
    result = pd.to_numeric(cleaned_str, errors="coerce").astype(float)
    result.loc[ambiguous_mask] = np.nan
    return result


# ==============================================================================
# 2. Outlier Detection & Treatment
# ==============================================================================


def detect_outliers(
    series: pd.Series,
    method: str = "iqr",
    threshold: float | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """
    Detect statistical outliers in a continuous numeric Series.

    Methods:
    - 'iqr': Tukey's IQR rule [Q1 - k*IQR, Q3 + k*IQR], default threshold k=1.5
    - 'zscore': Standard normal deviations [mean - z*std, mean + z*std], default threshold z=3.0
    - 'modified_zscore' / 'mad': Median Absolute Deviation (MAD), default threshold 3.5

    Returns:
    - mask: pd.Series of booleans (True = outlier)
    - stats: summary dictionary with threshold, bounds, and counts
    """
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()

    method_norm = method.lower().strip()
    n_valid = len(valid)

    if n_valid < 3:
        mask = pd.Series(False, index=series.index)
        return mask, {
            "method": method_norm,
            "threshold": threshold,
            "lower_bound": -np.inf,
            "upper_bound": np.inf,
            "outlier_count": 0,
            "outlier_pct": 0.0,
        }

    if method_norm == "iqr":
        k = 1.5 if threshold is None else float(threshold)
        q25 = float(np.percentile(valid, 25))
        q75 = float(np.percentile(valid, 75))
        iqr = q75 - q25
        lower = q25 - k * iqr
        upper = q75 + k * iqr

    elif method_norm == "zscore":
        z = 3.0 if threshold is None else float(threshold)
        mean = float(valid.mean())
        std = float(valid.std(ddof=1))
        if std == 0 or np.isnan(std):
            lower = mean
            upper = mean
        else:
            lower = mean - z * std
            upper = mean + z * std

    elif method_norm in ("modified_zscore", "mad"):
        m = 3.5 if threshold is None else float(threshold)
        med = float(valid.median())
        mad = float(np.median(np.abs(valid - med)))
        if mad == 0:
            lower = med
            upper = med
        else:
            # 0.6745 is the consistency constant for normal distribution
            lower = med - (m * mad / 0.6745)
            upper = med + (m * mad / 0.6745)

    else:
        raise ValueError(
            f"Unknown outlier detection method: '{method}'. "
            f"Approved methods: 'iqr', 'zscore', 'modified_zscore'"
        )

    mask = (s < lower) | (s > upper)
    # NaNs in original series are not outliers
    mask = mask.fillna(False)

    outlier_count = int(mask.sum())
    total_len = len(series)
    outlier_pct = round((outlier_count / total_len * 100), 2) if total_len > 0 else 0.0

    stats_out = {
        "method": method_norm,
        "threshold": threshold
        if threshold is not None
        else (
            1.5 if method_norm == "iqr" else (3.0 if method_norm == "zscore" else 3.5)
        ),
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "outlier_count": outlier_count,
        "outlier_pct": outlier_pct,
    }

    return mask, stats_out


def handle_outliers(
    series: pd.Series,
    method: str = "iqr",
    action: str = "flag",
    threshold: float | None = None,
    **kwargs: Any,
) -> pd.Series:
    """
    Detect and treat statistical outliers in a continuous numeric Series.

    Actions:
    - 'flag' / 'remove': Replaces outlier values with np.nan.
    - 'winsorize' / 'cap': Clamps extreme values to bounds (lower_bound, upper_bound).
    - 'mask': Returns the boolean outlier mask.
    """
    mask, stats_info = detect_outliers(series, method=method, threshold=threshold)
    action_norm = action.lower().strip()

    s_out = pd.to_numeric(series.copy(), errors="coerce")

    if action_norm in ("flag", "remove"):
        s_out.loc[mask] = np.nan
        return s_out

    elif action_norm in ("winsorize", "cap"):
        lower = stats_info["lower_bound"]
        upper = stats_info["upper_bound"]
        return s_out.clip(lower=lower, upper=upper)

    elif action_norm == "mask":
        return mask

    else:
        raise ValueError(
            f"Unknown outlier action: '{action}'. "
            f"Approved actions: 'flag', 'remove', 'winsorize', 'cap', 'mask'"
        )


# ==============================================================================
# 3. Defensive Predictor Checks
# ==============================================================================


def detect_zero_variance(
    df: pd.DataFrame,
    cols: list[str] | None = None,
) -> list[str]:
    """
    Detect constant or zero-variance columns in a DataFrame.
    """
    target_cols = cols if cols is not None else df.columns.tolist()
    zero_var_cols = []

    for col in target_cols:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) <= 1:
            zero_var_cols.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            if float(s.std()) == 0.0 or s.nunique() <= 1:
                zero_var_cols.append(col)
        else:
            if s.nunique() <= 1:
                zero_var_cols.append(col)

    return zero_var_cols


def check_missing_data_impact(
    df_original: pd.DataFrame,
    df_clean: pd.DataFrame,
    var_meta: dict[str, Any] | None = None,
    missing_codes: dict[str, Any] | list[Any] | None = None,
) -> dict[str, Any]:
    """
    Quantify observations removed and variables affected by missing data treatment.
    """
    rows_original = len(df_original)
    rows_clean = len(df_clean)
    rows_removed = rows_original - rows_clean
    pct_removed = (
        round((rows_removed / rows_original * 100), 2) if rows_original > 0 else 0.0
    )

    # Identify which variables had missing data
    variables_affected = []
    observations_lost: dict[str, dict[str, Any]] = {}

    df_norm = apply_missing_values_to_df(df_original, var_meta, missing_codes)

    for col in df_original.columns:
        missing_count = int(df_norm[col].isna().sum())
        if missing_count > 0:
            variables_affected.append(col)
            observations_lost[col] = {
                "count": missing_count,
                "pct": round(missing_count / rows_original * 100, 2),
            }

    return {
        "rows_original": rows_original,
        "rows_clean": rows_clean,
        "rows_removed": rows_removed,
        "pct_removed": pct_removed,
        "variables_affected": variables_affected,
        "observations_lost": observations_lost,
    }


# ==============================================================================
# 5. Missingness Audit & Little's MCAR Test
# ==============================================================================


def audit_missingness(
    df: pd.DataFrame,
    var_meta: dict[str, Any] | None = None,
    missing_codes: list[Any] | dict[str, Any] | None = None,
) -> MissingnessAudit:
    """
    Produce a rigorous MissingnessAudit report across all variables.

    Evaluates:
    1. Per-variable counts, percentages, and clinical risk tiers:
       - Low (<5%): Listwise deletion usually acceptable under MCAR.
       - Moderate (5-20%): Multiple imputation (MICE) recommended.
       - High (20-40%): Imputation required; sensitivity analysis necessary.
       - Critical (>40%): High risk of bias; consider indicator or dropping.
    2. Missingness patterns across observations.
    3. Pairwise missingness co-occurrence correlation matrix.
    """
    df_work = apply_missing_values_to_df(df, var_meta, missing_codes)
    total_rows, total_cols = df_work.shape
    total_cells = total_rows * total_cols
    total_missing_cells = int(df_work.isna().sum().sum())
    overall_missing_pct = (
        round((total_missing_cells / total_cells * 100), 2) if total_cells > 0 else 0.0
    )

    variables: dict[str, VariableMissingAudit] = {}
    has_critical = False

    for col in df_work.columns:
        s = df_work[col]
        missing_count = int(s.isna().sum())
        valid_count = total_rows - missing_count
        pct = round((missing_count / total_rows * 100), 2) if total_rows > 0 else 0.0

        if pct < 5.0:
            tier = "Low (<5%)"
            rec = "Complete-case analysis acceptable if MCAR holds."
        elif pct <= 20.0:
            tier = "Moderate (5-20%)"
            rec = "Multiple imputation (MICE) strongly recommended."
        elif pct <= 40.0:
            tier = "High (20-40%)"
            rec = "Imputation required; conduct sensitivity analysis."
        else:
            tier = "Critical (>40%)"
            rec = "Severe risk of bias. Consider missing indicator or domain review."
            has_critical = True

        inferred_type = (
            "Continuous" if pd.api.types.is_numeric_dtype(s) else "Categorical"
        )
        if s.dropna().nunique() == 2:
            inferred_type = "Binary"

        variables[col] = VariableMissingAudit(
            variable=col,
            dtype=str(s.dtype),
            inferred_type=inferred_type,
            total_count=total_rows,
            valid_count=valid_count,
            missing_count=missing_count,
            missing_pct=pct,
            risk_tier=tier,
            recommendation=rec,
            missing_coded_count=0,
            missing_nan_count=missing_count,
        )

    # Missingness pattern analysis
    missing_bool = df_work.isna()
    pattern_groups = missing_bool.groupby(list(df_work.columns), observed=False).size()

    patterns: list[MissingPattern] = []
    pid = 1
    for mask_tuple, count in pattern_groups.items():
        if not isinstance(mask_tuple, tuple):
            mask_tuple = (mask_tuple,)
        obs_vars = [col for col, is_m in zip(df_work.columns, mask_tuple) if not is_m]
        miss_vars = [col for col, is_m in zip(df_work.columns, mask_tuple) if is_m]
        patterns.append(
            MissingPattern(
                pattern_id=pid,
                observed_vars=obs_vars,
                missing_vars=miss_vars,
                row_count=int(count),
                pct=round(count / total_rows * 100, 2) if total_rows > 0 else 0.0,
            )
        )
        pid += 1

    patterns.sort(key=lambda p: p.row_count, reverse=True)

    # Co-occurrence correlation matrix
    co_occurrence: dict[str, dict[str, float]] = {}
    missing_num = missing_bool.astype(int)
    corr = missing_num.corr().fillna(0.0)

    for c1 in df_work.columns:
        co_occurrence[c1] = {}
        for c2 in df_work.columns:
            co_occurrence[c1][c2] = round(float(corr.loc[c1, c2]), 3)

    summary_df = get_missing_summary_df(
        df_work, var_meta=var_meta, already_normalized=True
    )

    return MissingnessAudit(
        total_rows=total_rows,
        total_columns=total_cols,
        total_cells=total_cells,
        total_missing_cells=total_missing_cells,
        overall_missing_pct=overall_missing_pct,
        variables=variables,
        patterns=patterns,
        co_occurrence=co_occurrence,
        has_critical_missing=has_critical,
        summary_df=summary_df,
    )


def littles_mcar_test(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    max_iter: int = 100,
    tol: float = 1e-5,
    ridge_eps: float = 1e-6,
) -> LittlesMCARResult:
    """
    Roderick J. A. Little's (1988) Test of Missing Completely at Random (MCAR).

    Reference:
    Little, R. J. A. (1988). A Test of Missing Completely at Random for Multivariate Data
    with Missing Values. Journal of the American Statistical Association, 83(404), 1198-1202.

    Implements:
    - Pattern-grouped Expectation-Maximization (EM) algorithm for ML parameter estimation
      under multivariate normality.
    - Ridge regularization to prevent singular covariance inversions.
    - Little's d2 chi-square test statistic with degrees of freedom:
      df = sum(p_s) - P, where p_s is observed variables in pattern s, P is total variables.
    """
    if len(df) == 0:
        raise ValueError("Input DataFrame is empty (0 rows).")

    # Select continuous numeric columns
    if cols is not None:
        target_cols = [c for c in cols if c in df.columns]
        invalid = set(cols) - set(target_cols)
        if invalid:
            raise ValueError(f"Specified columns not in DataFrame: {invalid}")
    else:
        target_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not target_cols:
        raise ValueError("No numeric columns available for Little's MCAR test.")

    P = len(target_cols)
    N = len(df)
    Y = df[target_cols].to_numpy(dtype=float, copy=True)

    # Check for 100% missing columns
    nan_mask = np.isnan(Y)
    col_all_nan = nan_mask.all(axis=0)
    if np.any(col_all_nan):
        bad_cols = [target_cols[i] for i, b in enumerate(col_all_nan) if b]
        raise ValueError(f"Column(s) 100% missing: {bad_cols}")

    # Check if data is completely observed
    if not np.any(nan_mask):
        cov_mat = np.atleast_2d(np.cov(Y.T))
        return LittlesMCARResult(
            statistic=0.0,
            df=0,
            p_value=1.0,
            is_mcar=True,
            n_patterns=1,
            n_observations=N,
            n_variables=P,
            pattern_counts={"complete": N},
            em_converged=True,
            em_iterations=0,
            mu_mle={col: float(np.mean(Y[:, i])) for i, col in enumerate(target_cols)},
            sigma_mle={
                c1: {c2: float(cov_mat[i, j]) for j, c2 in enumerate(target_cols)}
                for i, c1 in enumerate(target_cols)
            },
            message="Dataset is fully observed (0 missing values). MCAR holds trivially.",
        )

    # Drop rows that are completely missing (p_s = 0)
    row_all_nan = nan_mask.all(axis=1)
    if np.any(row_all_nan):
        Y = Y[~row_all_nan]
        nan_mask = nan_mask[~row_all_nan]
        N = len(Y)

    # Group into unique missingness patterns
    # Pattern representation: boolean tuple where True = observed, False = missing
    obs_mask = ~nan_mask
    patterns_dict: dict[tuple[bool, ...], list[int]] = {}
    for i, row in enumerate(obs_mask):
        key = tuple(row)
        if key not in patterns_dict:
            patterns_dict[key] = []
        patterns_dict[key].append(i)

    # Pre-parse patterns
    parsed_patterns = []
    total_p_s = 0
    pattern_counts = {}

    for key, row_indices in patterns_dict.items():
        O_s = np.array([j for j, obs in enumerate(key) if obs], dtype=int)
        M_s = np.array([j for j, obs in enumerate(key) if not obs], dtype=int)
        p_s = len(O_s)
        n_s = len(row_indices)
        total_p_s += p_s
        pat_name = f"obs_{p_s}_of_{P}"
        pattern_counts[pat_name] = pattern_counts.get(pat_name, 0) + n_s
        parsed_patterns.append(
            {
                "indices": np.array(row_indices, dtype=int),
                "O_s": O_s,
                "M_s": M_s,
                "p_s": p_s,
                "n_s": n_s,
            }
        )

    df_stat = total_p_s - P

    # If df <= 0, test cannot be evaluated asymptotically
    if df_stat <= 0:
        return LittlesMCARResult(
            statistic=0.0,
            df=max(0, df_stat),
            p_value=1.0,
            is_mcar=True,
            n_patterns=len(parsed_patterns),
            n_observations=N,
            n_variables=P,
            pattern_counts=pattern_counts,
            em_converged=True,
            em_iterations=0,
            mu_mle={
                col: float(np.nanmean(Y[:, i])) for i, col in enumerate(target_cols)
            },
            sigma_mle={c1: {c2: 0.0 for c2 in target_cols} for c1 in target_cols},
            message=f"Degrees of freedom ({df_stat}) <= 0; insufficient missingness patterns for asymptotic test.",
        )

    # EM Algorithm initialization
    mu = np.nanmean(Y, axis=0)
    # Initial sample covariance with mean imputation
    Y_init = Y.copy()
    for j in range(P):
        nan_j = np.isnan(Y_init[:, j])
        Y_init[nan_j, j] = mu[j]

    Sigma = np.cov(Y_init, rowvar=False, ddof=1)
    if Sigma.ndim == 0:
        Sigma = np.array([[float(Sigma)]])
    # Add initial ridge
    scale_ridge = ridge_eps * (np.trace(Sigma) / P if np.trace(Sigma) > 0 else 1.0)
    Sigma += np.eye(P) * scale_ridge

    converged = False
    em_iter = 0

    for iteration in range(1, max_iter + 1):
        em_iter = iteration
        # E-step
        sum_y = np.zeros(P)
        sum_yy = np.zeros((P, P))

        for pat in parsed_patterns:
            O_s = pat["O_s"]
            M_s = pat["M_s"]
            n_s = pat["n_s"]
            idx = pat["indices"]
            Y_pat = Y[idx]

            if len(M_s) == 0:
                # Complete observation pattern
                sum_y += np.sum(Y_pat, axis=0)
                sum_yy += Y_pat.T @ Y_pat
            else:
                # Invert Sigma_OO with ridge regularization
                Sigma_OO = Sigma[np.ix_(O_s, O_s)]
                Sigma_MO = Sigma[np.ix_(M_s, O_s)]
                Sigma_MM = Sigma[np.ix_(M_s, M_s)]

                try:
                    # Beta regression weights: Sigma_MO @ inv(Sigma_OO)
                    # Solved via Sigma_OO @ W.T = Sigma_MO.T
                    W = np.linalg.solve(
                        Sigma_OO + np.eye(len(O_s)) * scale_ridge, Sigma_MO.T
                    ).T
                except np.linalg.LinAlgError:
                    W = Sigma_MO @ np.linalg.pinv(Sigma_OO)

                # Conditional covariance
                C_MM = Sigma_MM - W @ Sigma_MO.T
                # Symmetrize
                C_MM = 0.5 * (C_MM + C_MM.T)

                # Impute missing values for each row in pattern
                Y_imputed = Y_pat.copy()
                diff_obs = Y_pat[:, O_s] - mu[O_s]
                Y_imputed[:, M_s] = mu[M_s] + diff_obs @ W.T

                sum_y += np.sum(Y_imputed, axis=0)
                # Outer product sum
                sum_yy += Y_imputed.T @ Y_imputed
                # Add conditional covariance block
                sum_yy[np.ix_(M_s, M_s)] += n_s * C_MM

        # M-step
        mu_next = sum_y / N
        Sigma_next = (sum_yy / N) - np.outer(mu_next, mu_next)
        Sigma_next = 0.5 * (Sigma_next + Sigma_next.T)
        Sigma_next += np.eye(P) * scale_ridge

        # Check convergence
        diff_mu = np.max(np.abs(mu_next - mu))
        diff_sigma = np.max(np.abs(Sigma_next - Sigma))

        mu = mu_next
        Sigma = Sigma_next

        if max(diff_mu, diff_sigma) < tol:
            converged = True
            break

    # Calculate Little's d2 statistic
    d2 = 0.0
    for pat in parsed_patterns:
        O_s = pat["O_s"]
        n_s = pat["n_s"]
        idx = pat["indices"]
        Y_pat = Y[idx]

        y_bar_obs = np.mean(Y_pat[:, O_s], axis=0)
        diff = y_bar_obs - mu[O_s]

        Sigma_OO = Sigma[np.ix_(O_s, O_s)]
        try:
            inv_Sigma_OO = np.linalg.inv(Sigma_OO + np.eye(len(O_s)) * scale_ridge)
        except np.linalg.LinAlgError:
            inv_Sigma_OO = np.linalg.pinv(Sigma_OO)

        d2_s = float(diff.T @ inv_Sigma_OO @ diff)
        d2 += n_s * d2_s

    d2 = max(0.0, float(d2))
    p_value = float(stats.chi2.sf(d2, df_stat))
    is_mcar = bool(p_value > 0.05)

    msg = (
        f"Little's MCAR test: d2 = {d2:.4f}, df = {df_stat}, p = {p_value:.4f}. "
        f"{'Fail to reject H0: Data is consistent with MCAR (p > 0.05).' if is_mcar else 'Reject H0: Missingness departs from MCAR (p <= 0.05); MAR or MNAR likely.'}"
    )

    return LittlesMCARResult(
        statistic=d2,
        df=df_stat,
        p_value=p_value,
        is_mcar=is_mcar,
        n_patterns=len(parsed_patterns),
        n_observations=N,
        n_variables=P,
        pattern_counts=pattern_counts,
        em_converged=converged,
        em_iterations=em_iter,
        mu_mle={col: float(mu[i]) for i, col in enumerate(target_cols)},
        sigma_mle={
            c1: {c2: float(Sigma[i, j]) for j, c2 in enumerate(target_cols)}
            for i, c1 in enumerate(target_cols)
        },
        message=msg,
    )


# ==============================================================================
# 6. Full DataFrame Cleaning & Summary
# ==============================================================================


def clean_dataframe(
    df: pd.DataFrame,
    numeric_threshold: float = 0.30,
    handle_outliers_flag: bool = False,
    outlier_method: str = "iqr",
    outlier_action: str = "flag",
    validate_quality: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Clean and coerce columns in a DataFrame.

    Parameters:
    - numeric_threshold: minimum ratio of convertible numeric values to convert an object column to float.
    - handle_outliers_flag: if True, applies outlier handling to numeric columns.
    - outlier_method: 'iqr', 'zscore', or 'modified_zscore'.
    - outlier_action: 'flag', 'remove', 'winsorize', 'cap'.
    """
    df_clean = df.copy()
    total_rows = len(df_clean)

    report: dict[str, Any] = {
        "original_shape": df.shape,
        "converted_numeric_cols": [],
        "outlier_stats": {},
        "string_cols": [],
        "total_rows": total_rows,
    }

    if total_rows == 0:
        report["final_shape"] = (0, len(df.columns))
        return df_clean, report

    for col in df_clean.columns:
        series = df_clean[col]

        if pd.api.types.is_numeric_dtype(series):
            # Already numeric
            if handle_outliers_flag:
                df_clean[col] = handle_outliers(
                    series, method=outlier_method, action=outlier_action
                )
                _, out_stats = detect_outliers(series, method=outlier_method)
                report["outlier_stats"][col] = out_stats
            continue

        # Try numeric conversion
        coerced = clean_numeric_vector(series)
        valid_ratio = coerced.notna().sum() / total_rows

        if valid_ratio >= numeric_threshold:
            df_clean[col] = coerced
            report["converted_numeric_cols"].append(col)
            if handle_outliers_flag:
                df_clean[col] = handle_outliers(
                    coerced, method=outlier_method, action=outlier_action
                )
                _, out_stats = detect_outliers(coerced, method=outlier_method)
                report["outlier_stats"][col] = out_stats
        else:
            # Leave as string-like, preserving missing values as NaN
            mask = series.isna()
            df_clean[col] = series.astype(str).mask(mask, np.nan)
            report["string_cols"].append(col)

    report["final_shape"] = df_clean.shape
    return df_clean, report


def get_cleaning_summary(report: dict[str, Any]) -> str:
    """Generate human-readable summary of data cleaning report."""
    orig = report.get("original_shape", (0, 0))
    final = report.get("final_shape", (0, 0))
    converted = report.get("converted_numeric_cols", [])
    outliers = report.get("outlier_stats", {})

    lines = [
        "========================================",
        "         DATA CLEANING SUMMARY          ",
        "========================================",
        f"Original shape: {orig}",
        f"Cleaned shape:  {final}",
        f"Converted to numeric ({len(converted)}): {', '.join(converted) if converted else 'None'}",
    ]

    if outliers:
        lines.append("Outlier treatment:")
        for col, stats_i in outliers.items():
            lines.append(
                f"  - {col}: {stats_i.get('outlier_count', 0)} outliers detected ({stats_i.get('outlier_pct', 0.0)}%)"
            )

    lines.append("========================================")
    return "\n".join(lines)


# Cross-exported for seamless API access (implemented in medstat.data.missing)
from medstat.data.missing import prepare_data_for_analysis  # noqa: E402

__all__ = [
    "DataCleaningError",
    "DataValidationError",
    "MissingDataError",
    "MissingStrategyRequiredError",
    "VariableMissingAudit",
    "MissingPattern",
    "MissingnessAudit",
    "OutlierResult",
    "LittlesMCARResult",
    "clean_numeric",
    "clean_numeric_vector",
    "detect_outliers",
    "handle_outliers",
    "detect_zero_variance",
    "apply_missing_values_to_df",
    "detect_missing_in_variable",
    "get_missing_summary_df",
    "check_missing_data_impact",
    "audit_missingness",
    "littles_mcar_test",
    "clean_dataframe",
    "get_cleaning_summary",
    "prepare_data_for_analysis",
]

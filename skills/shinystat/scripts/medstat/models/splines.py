"""
Restricted Cubic Splines (RCS) for Non-linear Clinical Relationships.

Builds Harrell's restricted cubic spline basis (Harrell 2015, quantile knots)
for Cox PH and logistic models, tests non-linearity by likelihood ratio, and
returns reference-centred HR / OR contrast curves. Recipe 22 in shinystat-cloud
uses the same basis and fitting procedure (ADR 78).
"""

from __future__ import annotations

import keyword
from typing import Any

import numpy as np
import pandas as pd
import patsy
import plotly.graph_objects as go
from lifelines import CoxPHFitter
from scipy import stats

from medstat.theme.palette import get_color_palette

COLORS = get_color_palette()


def _quote_col(name: str) -> str:
    """Quote column names for Patsy formulas if not a standard identifier."""
    if not name.isidentifier() or keyword.iskeyword(name):
        return f"Q({name!r})"
    return name


class CoxRCSResult(tuple):
    """Container supporting both 3-tuple unpacking and dictionary-style key access."""

    def __new__(
        cls, fig: go.Figure, contrast_df: pd.DataFrame, stats_meta: dict[str, Any]
    ):
        return super().__new__(cls, (fig, contrast_df, stats_meta))

    def __init__(
        self, fig: go.Figure, contrast_df: pd.DataFrame, stats_meta: dict[str, Any]
    ):
        self.fig = fig
        self.contrast_df = contrast_df
        self.stats_meta = stats_meta
        self._dict = {
            "fig": fig,
            "contrast_df": contrast_df,
            "stats_meta": stats_meta,
            "model": stats_meta.get("cph"),
            "spline_columns": stats_meta.get("spline_columns", []),
            **stats_meta,
        }

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, (int, slice)):
            return super().__getitem__(item)
        return self._dict[item]

    def __contains__(self, item: object) -> bool:
        return item in self._dict or super().__contains__(item)

    def get(self, key: str, default: Any = None) -> Any:
        return self._dict.get(key, default)


_HARRELL_QUANTILES = {
    3: (0.10, 0.50, 0.90),
    4: (0.05, 0.35, 0.65, 0.95),
    5: (0.05, 0.275, 0.50, 0.725, 0.95),
}


def build_harrell_rcs_basis(
    x: np.ndarray | pd.Series,
    knots: int | list[float] | np.ndarray = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Harrell restricted cubic spline basis (k - 1 columns, first = x) and knot vector.

    An integer ``knots`` places k knots at Harrell's quantiles (k = 3, 4, 5; evenly
    spaced 5th-95th percentiles otherwise); an array gives the knot locations.
    """
    x_arr = np.asarray(x, dtype=float).ravel()
    if isinstance(knots, (int, np.integer)):
        k = int(knots)
        if k < 3:
            raise ValueError("knots must be >= 3 for Restricted Cubic Splines.")
        q = _HARRELL_QUANTILES.get(k, tuple(np.linspace(0.05, 0.95, k)))
        knot_vals = np.quantile(x_arr[~np.isnan(x_arr)], q)
        if len(np.unique(knot_vals)) < k:
            # Merging tied quantiles would silently fit fewer knots and fewer df.
            raise ValueError(
                f"{k} Harrell knots requested but only {len(np.unique(knot_vals))} quantile "
                "locations are distinct; lower `knots` or pass explicit knot locations."
            )
    else:
        knot_vals = np.sort(np.asarray(knots, dtype=float))
        if len(knot_vals) < 3:
            raise ValueError("knots must be >= 3 for Restricted Cubic Splines.")

    knot_vals = np.unique(knot_vals)
    if len(knot_vals) < 3:
        raise ValueError("Insufficient distinct knot locations for RCS basis.")
    k = len(knot_vals)
    t_k, t_km1 = knot_vals[-1], knot_vals[-2]
    scale = (t_k - knot_vals[0]) ** 2
    denom_tail = t_k - t_km1

    def _pos3(u: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, u) ** 3

    cols = [x_arr]
    for j in range(k - 2):
        t_j = knot_vals[j]
        cols.append(
            (
                _pos3(x_arr - t_j)
                - _pos3(x_arr - t_km1) * ((t_k - t_j) / denom_tail)
                + _pos3(x_arr - t_k) * ((t_km1 - t_j) / denom_tail)
            )
            / scale
        )
    return np.column_stack(cols), knot_vals


def _orthonormalize(basis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centred, QR-orthonormalised spline columns Z = (B - c) @ T.

    The Harrell columns are nearly collinear (r > 0.99 at k = 5), which stalls
    Newton-Raphson; fitting on Z and mapping back with beta_B = T @ beta_Z gives
    the same likelihood and the coefficients on the original basis.
    """
    center = basis.mean(axis=0)
    q, r = np.linalg.qr(basis - center)
    if np.any(np.abs(np.diag(r)) <= 1e-10 * max(1.0, float(np.abs(r).max()))):
        raise ValueError("RCS basis is rank deficient: too few distinct values of the spline variable.")
    t = np.linalg.inv(r) * np.sqrt(basis.shape[0])
    return (basis - center) @ t, t, center


def _nonlinearity_test(
    ll_spline: float, ll_linear: float, df_diff: int
) -> tuple[float, float, str | None]:
    """LR test of non-linearity; a spline fit worse than its nested linear fit is a
    convergence failure, reported as NaN with a reason rather than P = 1."""
    lr = 2.0 * (ll_spline - ll_linear)
    if lr < -1e-6 * max(1.0, abs(ll_linear)):
        return (
            float("nan"),
            float("nan"),
            "spline model log-likelihood below the nested linear model (non-convergence)",
        )
    lr = max(0.0, lr)
    return lr, float(stats.chi2.sf(lr, df=df_diff)), None


def _assert_full_rank(design: pd.DataFrame, label: str) -> None:
    """Fail closed on a collinear design (e.g. a covariate duplicating the spline variable)."""
    arr = design.to_numpy(dtype=float)
    if np.linalg.matrix_rank(arr) < arr.shape[1]:
        raise ValueError(
            f"{label} design matrix is rank deficient: drop collinear covariates before the RCS fit."
        )


def _fit_binomial_glm(y: np.ndarray, X: pd.DataFrame) -> tuple[Any, str | None]:
    """Binomial GLM (IRLS) plus a reason when its inference is undefined.

    statsmodels warns on perfect separation yet still reports ``converged=True``,
    so both signals are checked.
    """
    import warnings

    import statsmodels.api as sm

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    if any("PerfectSeparation" in type(w.message).__name__ for w in caught):
        return res, "perfect or quasi-complete separation in the binomial GLM"
    if not bool(getattr(res, "converged", True)):
        return res, "binomial GLM did not converge"
    return res, None


def _contrast_curve(
    x: np.ndarray,
    knot_vals: np.ndarray,
    beta_b: np.ndarray,
    cov_b: np.ndarray,
    ref_value: float,
    n_grid: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.linspace(float(np.min(x)), float(np.max(x)), n_grid)
    grid_b, _ = build_harrell_rcs_basis(grid, knots=knot_vals)
    ref_b, _ = build_harrell_rcs_basis(np.array([ref_value]), knots=knot_vals)
    diff = grid_b - ref_b
    log_ratio = diff @ beta_b
    se = np.sqrt(np.maximum(np.sum((diff @ cov_b) * diff, axis=1), 0.0))
    return grid, log_ratio, se


def _covariate_matrix(clean_df: pd.DataFrame, adjust: list[str]) -> pd.DataFrame:
    """Adjustment covariates via patsy (categoricals dummy-coded), no intercept."""
    if not adjust:
        return pd.DataFrame(index=clean_df.index)
    rhs = " + ".join(_quote_col(c) for c in adjust)
    mat = patsy.dmatrix(rhs, clean_df, return_type="dataframe")
    return mat.drop(columns=["Intercept"], errors="ignore")


def fit_cox_rcs(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    rcs_var: str | None = None,
    adjust_cols: list[str] | None = None,
    knots: int | None = None,
    ref_value: float | None = None,
    n_knots: int | None = None,
    **kwargs: Any,
) -> CoxRCSResult:
    """
    Perform Restricted Cubic Spline (RCS) analysis for Cox Proportional Hazards.

    Parameters:
        df: Input DataFrame.
        duration_col: Name of time duration column.
        event_col: Name of binary event column.
        rcs_var: Continuous variable to model with splines (alias: spline_var).
        adjust_cols: Optional list of additional covariates to adjust for (alias: covariates).
        knots: Number of Harrell knots k (>= 3, default 4), placed at Harrell's quantiles;
            the spline spends k - 1 degrees of freedom. Also accepts n_knots.
        ref_value: Reference value for hazard ratio comparisons (defaults to median of rcs_var).

    Returns:
        CoxRCSResult: Tuple/dict supporting (plotly_figure, contrast_dataframe, model_stats_dict)
    """
    if rcs_var is None:
        rcs_var = kwargs.pop("spline_var", None)
    if adjust_cols is None:
        adjust_cols = kwargs.pop("covariates", None)
    kwargs.pop("spline_var", None)
    kwargs.pop("covariates", None)
    if kwargs:
        unexpected = ", ".join(repr(k) for k in kwargs)
        raise TypeError(f"fit_cox_rcs got unexpected keyword argument(s): {unexpected}")
    if not rcs_var:
        raise ValueError(
            "`rcs_var` (or `spline_var`) must be specified for spline modeling."
        )

    num_knots = n_knots if n_knots is not None else (knots if knots is not None else 4)
    if num_knots < 3:
        raise ValueError("knots must be >= 3 for Restricted Cubic Splines.")

    adjust = adjust_cols or []
    adjust = [
        c for c in dict.fromkeys(adjust) if c not in {rcs_var, duration_col, event_col}
    ]

    req_cols = [duration_col, event_col, rcs_var] + adjust
    clean_df = df[req_cols].dropna()

    if len(clean_df) < 10:
        raise ValueError("Insufficient data points for RCS Cox model (need >= 10).")

    if ref_value is None:
        ref_value = float(clean_df[rcs_var].median())

    x = clean_df[rcs_var].to_numpy(dtype=float)
    basis, knot_vals = build_harrell_rcs_basis(x, knots=num_knots)
    spline_cols = [f"{rcs_var}_rcs{i}" for i in range(basis.shape[1])]
    z, t_mat, _ = _orthonormalize(basis)
    z_cols = [f"_z{i}" for i in range(z.shape[1])]
    cov_df = _covariate_matrix(clean_df, adjust)

    def _fit(design: pd.DataFrame) -> CoxPHFitter:
        data = design.copy()
        data["_dur"] = clean_df[duration_col].to_numpy(dtype=float)
        data["_ev"] = clean_df[event_col].to_numpy(dtype=float)
        return CoxPHFitter().fit(data, duration_col="_dur", event_col="_ev")

    design = pd.concat(
        [pd.DataFrame(z, index=clean_df.index, columns=z_cols), cov_df], axis=1
    )
    cph = _fit(design)
    lin_design = pd.concat(
        [pd.DataFrame({rcs_var: x}, index=clean_df.index), cov_df], axis=1
    )
    cph_linear = _fit(lin_design)

    # Back-transform to the Harrell basis (covariates unchanged).
    n_s, n_c = len(z_cols), cov_df.shape[1]
    m = np.eye(n_s + n_c)
    m[:n_s, :n_s] = t_mat
    names = spline_cols + list(cov_df.columns)
    coef = m @ cph.params_.to_numpy()
    cov_full = m @ cph.variance_matrix_.to_numpy() @ m.T
    se = np.sqrt(np.maximum(np.diag(cov_full), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        zstat = coef / se
    summary_df = pd.DataFrame(
        {
            "coef": coef,
            "se(coef)": se,
            "z": zstat,
            "p": 2.0 * stats.norm.sf(np.abs(zstat)),
        },
        index=pd.Index(names, name="covariate"),
    )
    beta_b, cov_b = coef[:n_s], cov_full[:n_s, :n_s]

    lr_stat, non_linear_p, non_linear_reason = _nonlinearity_test(
        float(cph.log_likelihood_), float(cph_linear.log_likelihood_), n_s - 1
    )
    wald = float(beta_b @ np.linalg.pinv(cov_b) @ beta_b)
    p_overall = float(stats.chi2.sf(wald, df=n_s))

    grid_vals, log_hr, se_log_hr = _contrast_curve(x, knot_vals, beta_b, cov_b, ref_value)
    hr_est = np.exp(log_hr)
    hr_low = np.exp(log_hr - 1.96 * se_log_hr)
    hr_high = np.exp(log_hr + 1.96 * se_log_hr)

    contrast_df = pd.DataFrame(
        {
            rcs_var: grid_vals,
            "HR": hr_est,
            "HR_lower": hr_low,
            "HR_upper": hr_high,
            "log_HR": log_hr,
            "se_log_HR": se_log_hr,
        }
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=grid_vals,
            y=hr_est,
            mode="lines",
            line=dict(color=COLORS["primary"], width=2.5),
            name="Hazard Ratio",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([grid_vals, grid_vals[::-1]]),
            y=np.concatenate([hr_high, hr_low[::-1]]),
            fill="toself",
            fillcolor="rgba(15, 23, 42, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
            name="95% CI",
        )
    )
    fig.add_hline(
        y=1.0, line_dash="dash", line_color="gray", annotation_text="HR = 1.0"
    )
    fig.update_layout(
        title=f"Restricted Cubic Splines: Hazard Ratio by {rcs_var} (Ref = {ref_value:.2f})",
        xaxis_title=rcs_var,
        yaxis_title="Hazard Ratio (95% CI)",
        template="plotly_white",
        yaxis_type="log",
    )

    stats_meta = {
        "model": cph,
        "cph": cph,
        "summary_df": summary_df,
        "ref_value": ref_value,
        "knots": num_knots,
        "knot_locations": knot_vals.tolist(),
        "spline_columns": spline_cols,
        "c_index": float(cph.concordance_index_),
        "log_likelihood": float(cph.log_likelihood_),
        "lr_nonlinearity": lr_stat,
        "non_linear_pvalue": non_linear_p,
        "non_linear_reason": non_linear_reason,
        "p_overall": p_overall,
    }

    return CoxRCSResult(fig, contrast_df, stats_meta)


def fit_logistic_rcs(
    df: pd.DataFrame,
    outcome: str,
    spline_var: str,
    covariates: list[str] | None = None,
    n_knots: int = 4,
    ref_value: float | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Fit multivariable Logistic Regression with Restricted Cubic Splines (RCS).

    Parameters:
        df: Input DataFrame.
        outcome: Binary outcome column.
        spline_var: Continuous variable to model with splines.
        covariates: Optional baseline confounders.
        n_knots: Number of Harrell knots k (>= 3, default 4); k - 1 spline degrees of freedom.
        ref_value: Reference value for Odds Ratio comparisons (default: median).

    Returns:
        dict containing model, summary_df, contrast_df, knots, knot_locations,
        non_linear_pvalue (with non_linear_reason), p_overall and log_likelihood.
        Under separation or non-convergence ``model`` is None, every estimate, P value
        and contrast is NaN and ``fit_failure_reason`` says why; a rank-deficient
        design raises.
    """
    import statsmodels.api as sm

    if kwargs:
        unexpected = ", ".join(repr(k) for k in kwargs)
        raise TypeError(
            f"fit_logistic_rcs got unexpected keyword argument(s): {unexpected}"
        )

    adjust = covariates or []
    all_cols = [outcome, spline_var] + adjust
    clean_df = df[all_cols].dropna().copy()

    if len(clean_df) < 10:
        raise ValueError("Insufficient observations for logistic RCS analysis.")

    if n_knots < 3:
        raise ValueError("n_knots must be >= 3 for Restricted Cubic Splines.")
    num_knots = n_knots
    if ref_value is None:
        ref_value = float(clean_df[spline_var].median())

    x = clean_df[spline_var].to_numpy(dtype=float)
    y = clean_df[outcome].to_numpy(dtype=float)
    basis, knot_vals = build_harrell_rcs_basis(x, knots=num_knots)
    spline_cols = [f"{spline_var}_rcs{i}" for i in range(basis.shape[1])]
    z, t_mat, center = _orthonormalize(basis)
    cov_df = _covariate_matrix(clean_df, adjust)

    z_df = pd.DataFrame(z, index=clean_df.index, columns=[f"_z{i}" for i in range(z.shape[1])])
    X = sm.add_constant(pd.concat([z_df, cov_df], axis=1), has_constant="add")
    X_linear = sm.add_constant(
        pd.concat([pd.DataFrame({spline_var: x}, index=clean_df.index), cov_df], axis=1),
        has_constant="add",
    )
    _assert_full_rank(X, "Spline")
    _assert_full_rank(X_linear, "Linear")
    # GLM Binomial (IRLS) as in Recipe 22, so Wald and LR statistics match it exactly.
    logit_model, spline_fail = _fit_binomial_glm(y, X)
    linear_model, linear_fail = _fit_binomial_glm(y, X_linear)
    fit_failure_reason = (
        f"spline model: {spline_fail}" if spline_fail
        else (f"linear model: {linear_fail}" if linear_fail else None)
    )

    # Affine map back to [Intercept, Harrell basis, covariates]:
    # a + (B - c) T b_z = (a - c . T b_z) + B (T b_z).
    n_s, n_c = z.shape[1], cov_df.shape[1]
    m = np.eye(1 + n_s + n_c)
    m[1 : 1 + n_s, 1 : 1 + n_s] = t_mat
    m[0, 1 : 1 + n_s] = -(center @ t_mat)
    names = ["Intercept"] + spline_cols + list(cov_df.columns)
    coef = m @ logit_model.params.to_numpy()
    cov_full = m @ logit_model.cov_params().to_numpy() @ m.T
    if fit_failure_reason is not None:
        # Fail closed: estimates from a separated or non-converged fit are not reported.
        coef = np.full_like(coef, np.nan)
        cov_full = np.full_like(cov_full, np.nan)
    se = np.sqrt(np.maximum(np.diag(cov_full), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        zstat = coef / se
    pvals = 2.0 * stats.norm.sf(np.abs(zstat))
    lo, hi = coef - 1.96 * se, coef + 1.96 * se

    # Individual basis coefficients and the intercept are not odds ratios.
    is_basis_or_intercept = np.array([n in spline_cols or n == "Intercept" for n in names])
    or_vals = np.where(is_basis_or_intercept, np.nan, np.exp(coef))
    summary_df = pd.DataFrame(
        {
            "coef": coef,
            "std_error": se,
            "z_score": zstat,
            "p_value": pvals,
            "odds_ratio": or_vals,
            "ci_lower": np.where(is_basis_or_intercept, np.nan, np.exp(lo)),
            "ci_upper": np.where(is_basis_or_intercept, np.nan, np.exp(hi)),
        },
        index=names,
    )
    beta_b, cov_b = coef[1 : 1 + n_s], cov_full[1 : 1 + n_s, 1 : 1 + n_s]

    if fit_failure_reason is not None:
        lr_stat = non_linear_p = p_overall = float("nan")
        non_linear_reason = fit_failure_reason
    else:
        lr_stat, non_linear_p, non_linear_reason = _nonlinearity_test(
            float(logit_model.llf), float(linear_model.llf), n_s - 1
        )
        wald = float(beta_b @ np.linalg.pinv(cov_b) @ beta_b)
        p_overall = float(stats.chi2.sf(wald, df=n_s))

    grid_vals, log_or, se_log_or = _contrast_curve(x, knot_vals, beta_b, cov_b, ref_value)
    contrast_df = pd.DataFrame(
        {
            spline_var: grid_vals,
            "OR": np.exp(log_or),
            "OR_lower": np.exp(log_or - 1.96 * se_log_or),
            "OR_upper": np.exp(log_or + 1.96 * se_log_or),
            "log_OR": log_or,
            "se_log_OR": se_log_or,
        }
    )

    return {
        # Withheld on failure, like coef / cov_full: no estimates from an invalid fit.
        "model": None if fit_failure_reason else logit_model,
        "summary_df": summary_df,
        "contrast_df": contrast_df,
        "ref_value": ref_value,
        "knots": num_knots,
        "knot_locations": knot_vals.tolist(),
        "spline_columns": spline_cols,
        "lr_nonlinearity": lr_stat,
        "non_linear_pvalue": non_linear_p,
        "non_linear_reason": non_linear_reason,
        "p_overall": p_overall,
        "fit_failure_reason": fit_failure_reason,
        "log_likelihood": float("nan") if fit_failure_reason else float(logit_model.llf),
    }


# Backwards compatibility and agent aliases
fit_rcs_cox = fit_cox_rcs
fit_rcs_logistic = fit_logistic_rcs

"""
Restricted Cubic Splines (RCS) for Non-linear Clinical Relationships.

Uses patsy's natural cubic spline basis `cr()` to model non-linear effects
in survival analysis (Cox PH) and generate Hazard Ratio contrast curves.
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

from medstat.logging import get_logger
from medstat.theme.palette import get_color_palette

logger = get_logger(__name__)
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


def fit_cox_rcs(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    rcs_var: str | None = None,
    adjust_cols: list[str] | None = None,
    knots: int | None = None,
    ref_value: float | None = None,
    n_knots: int | None = None,
    constraints: str | None = "center",
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
        knots: Number of spline knots (must be >= 3, default 4). Also accepts n_knots.
        ref_value: Reference value for hazard ratio comparisons (defaults to median of rcs_var).
        constraints: Centering/orthogonality constraint passed to patsy cr (default: 'center').

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

    # Determine reference value
    if ref_value is None:
        ref_value = float(clean_df[rcs_var].median())

    constraint_str = f", constraints='{constraints}'" if constraints else ""
    formula_rhs = f"cr({_quote_col(rcs_var)}, df={num_knots}{constraint_str})"
    if adjust:
        formula_rhs += " + " + " + ".join(_quote_col(c) for c in adjust)

    # Build design matrix with patsy
    y, X = patsy.dmatrices(
        f"{_quote_col(duration_col)} + {_quote_col(event_col)} ~ {formula_rhs}",
        clean_df,
        return_type="dataframe",
    )
    design_info = X.design_info

    # Drop intercept for CoxPHFitter
    if "Intercept" in X.columns:
        X = X.drop(columns=["Intercept"])

    # Prepare DataFrame for lifelines CoxPHFitter
    data_for_cph = X.copy()
    data_for_cph[duration_col] = clean_df[duration_col].values
    data_for_cph[event_col] = clean_df[event_col].values

    cph = CoxPHFitter()
    cph.fit(data_for_cph, duration_col=duration_col, event_col=event_col)

    # Generate prediction grid across range of rcs_var
    var_min = float(clean_df[rcs_var].min())
    var_max = float(clean_df[rcs_var].max())
    grid_vals = np.linspace(var_min, var_max, 100)

    # Create synthetic prediction dataset
    pred_data = pd.DataFrame({rcs_var: grid_vals})
    for c in adjust:
        pred_data[c] = (
            clean_df[c].mean()
            if pd.api.types.is_numeric_dtype(clean_df[c])
            else clean_df[c].mode()[0]
        )

    # Reference row
    ref_data = pd.DataFrame({rcs_var: [ref_value]})
    for c in adjust:
        ref_data[c] = pred_data[c].iloc[0]

    # Transform using saved design_info
    X_pred = patsy.build_design_matrices([design_info], pred_data)[0]
    X_ref = patsy.build_design_matrices([design_info], ref_data)[0]

    X_pred_df = pd.DataFrame(X_pred, columns=design_info.column_names)
    X_ref_df = pd.DataFrame(X_ref, columns=design_info.column_names)

    if "Intercept" in X_pred_df.columns:
        X_pred_df = X_pred_df.drop(columns=["Intercept"])
        X_ref_df = X_ref_df.drop(columns=["Intercept"])

    # Contrast matrix
    contrast = X_pred_df.values - X_ref_df.values
    coefs = cph.params_.values
    cov_mat = cph.variance_matrix_.values

    # log(HR) = contrast * beta
    log_hr = contrast @ coefs
    # Var(log HR) = diag(contrast * cov * contrast^T)
    var_log_hr = np.sum((contrast @ cov_mat) * contrast, axis=1)
    se_log_hr = np.sqrt(np.maximum(var_log_hr, 0.0))

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

    # Generate Plotly visualization
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
    # Reference line at HR=1
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

    # Fit linear Cox model to compare for non-linearity (Likelihood Ratio Test)
    try:
        linear_rhs = f"{_quote_col(rcs_var)}"
        if adjust:
            linear_rhs += " + " + " + ".join(_quote_col(c) for c in adjust)
        X_linear = patsy.dmatrix(linear_rhs, clean_df, return_type="dataframe")
        if "Intercept" in X_linear.columns:
            X_linear = X_linear.drop(columns=["Intercept"])
        data_for_linear = X_linear.copy()
        data_for_linear[duration_col] = clean_df[duration_col].values
        data_for_linear[event_col] = clean_df[event_col].values
        cph_linear = CoxPHFitter()
        cph_linear.fit(data_for_linear, duration_col=duration_col, event_col=event_col)

        ll_spline = float(cph.log_likelihood_)
        ll_linear = float(cph_linear.log_likelihood_)
        df_diff = len(cph.params_) - len(cph_linear.params_)
        if df_diff > 0:
            lr_stat = max(0.0, 2.0 * (ll_spline - ll_linear))
            non_linear_p = float(stats.chi2.sf(lr_stat, df=df_diff))
        else:
            non_linear_p = np.nan
    except Exception as e:
        logger.warning(f"Could not compute non-linear LR test for splines: {e}")
        non_linear_p = np.nan

    spline_cols = [c for c in X.columns if rcs_var in c]
    stats_meta = {
        "model": cph,
        "cph": cph,
        "summary_df": cph.summary,
        "ref_value": ref_value,
        "knots": num_knots,
        "spline_columns": spline_cols,
        "c_index": float(cph.concordance_index_),
        "log_likelihood": float(cph.log_likelihood_),
        "non_linear_pvalue": non_linear_p,
    }

    return CoxRCSResult(fig, contrast_df, stats_meta)

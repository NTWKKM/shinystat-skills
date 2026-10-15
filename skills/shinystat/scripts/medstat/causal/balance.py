"""
Covariate Balance Diagnostics & Love Plots for Causal Inference.

Computes Standardized Mean Differences (SMDs) and generates publication Love plots
aligned with Austin (2009) criteria (SMD < 0.10 indicates acceptable balance).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from medstat.theme.palette import get_color_palette

COLORS = get_color_palette()


def calculate_smd(
    treated: Any,
    control: Any = None,
    weights_treated: Any = None,
    weights_control: pd.Series | np.ndarray | None = None,
) -> float:
    """
    Calculate Standardized Mean Difference (SMD) for a continuous or binary variable.

    SMD = (mean_treated - mean_control) / sqrt((var_treated + var_control) / 2)
    Supports both:
    - calculate_smd(treated_series, control_series)
    - calculate_smd(df, treatment_col, covariate_col)
    """
    if isinstance(treated, pd.DataFrame):
        df = treated
        treatment_col = str(control)
        covar_col = str(weights_treated)
        t_mask = df[treatment_col] == 1
        c_mask = df[treatment_col] == 0
        t_s = df.loc[t_mask, covar_col]
        c_s = df.loc[c_mask, covar_col]
        return calculate_smd(t_s, c_s, weights_treated=weights_control)

    t_arr = np.asarray(treated, dtype=float)
    c_arr = np.asarray(control, dtype=float)

    t_valid = t_arr[~np.isnan(t_arr)]
    c_valid = c_arr[~np.isnan(c_arr)]

    if len(t_valid) == 0 or len(c_valid) == 0:
        return np.nan

    if weights_treated is not None:
        wt = np.asarray(weights_treated, dtype=float)[~np.isnan(t_arr)]
        sum_wt = float(np.sum(wt))
        sum_wt2 = float(np.sum(wt**2))
        mean_t = float(np.average(t_valid, weights=wt))
        if sum_wt**2 > sum_wt2:
            # Bessel-corrected weighted sample variance (Austin 2009 / Kish formula)
            bessel_factor = (sum_wt**2) / (sum_wt**2 - sum_wt2)
            raw_var = float(np.average((t_valid - mean_t) ** 2, weights=wt))
            var_t = bessel_factor * raw_var
        else:
            var_t = 0.0
    else:
        mean_t = float(np.mean(t_valid))
        var_t = float(np.var(t_valid, ddof=1)) if len(t_valid) > 1 else 0.0

    if weights_control is not None:
        wc = np.asarray(weights_control, dtype=float)[~np.isnan(c_arr)]
        sum_wc = float(np.sum(wc))
        sum_wc2 = float(np.sum(wc**2))
        mean_c = float(np.average(c_valid, weights=wc))
        if sum_wc**2 > sum_wc2:
            bessel_factor = (sum_wc**2) / (sum_wc**2 - sum_wc2)
            raw_var = float(np.average((c_valid - mean_c) ** 2, weights=wc))
            var_c = bessel_factor * raw_var
        else:
            var_c = 0.0
    else:
        mean_c = float(np.mean(c_valid))
        var_c = float(np.var(c_valid, ddof=1)) if len(c_valid) > 1 else 0.0

    pooled_sd = np.sqrt((var_t + var_c) / 2.0)
    if pooled_sd == 0:
        return 0.0 if mean_t == mean_c else np.nan

    return float((mean_t - mean_c) / pooled_sd)


def check_balance(
    df: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    weights: pd.Series | np.ndarray | str | None = None,
) -> pd.DataFrame:
    """
    Calculate SMDs across multiple covariates in a cohort.
    """
    treated_mask = df[treatment] == 1
    control_mask = df[treatment] == 0

    if isinstance(weights, str):
        w_series = df[weights]
    else:
        w_series = weights

    results = []
    for cov in covariates:
        if not pd.api.types.is_numeric_dtype(df[cov]):
            continue

        smd = calculate_smd(
            df.loc[treated_mask, cov],
            df.loc[control_mask, cov],
            weights_treated=w_series[treated_mask] if w_series is not None else None,
            weights_control=w_series[control_mask] if w_series is not None else None,
        )

        is_finite = pd.notna(smd) and np.isfinite(smd)
        status = "Balanced" if (is_finite and abs(smd) < 0.10) else "Imbalanced"
        results.append(
            {
                "Covariate": cov,
                "Variable": cov,
                "SMD": float(smd) if is_finite else None,
                "Absolute_SMD": abs(float(smd)) if is_finite else None,
                "Status": status,
            }
        )

    return pd.DataFrame(results)


def compute_balance_table(
    df: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    matched_df: pd.DataFrame | None = None,
    weights_col: str | None = None,
) -> pd.DataFrame:
    """
    Generate comprehensive covariate balance comparison table supporting unadjusted,
    matched, and IPTW-weighted cohorts.
    """
    unadj = check_balance(df, treatment, covariates)
    unadj_map = dict(zip(unadj["Covariate"], unadj["SMD"]))

    matched_map = {}
    if matched_df is not None:
        matched_res = check_balance(matched_df, treatment, covariates)
        matched_map = dict(zip(matched_res["Covariate"], matched_res["SMD"]))

    weighted_map = {}
    if weights_col is not None and weights_col in df.columns:
        weighted_res = check_balance(df, treatment, covariates, weights=weights_col)
        weighted_map = dict(zip(weighted_res["Covariate"], weighted_res["SMD"]))

    rows = []
    for cov in covariates:
        u_smd = _normalize_smd(unadj_map.get(cov))
        row = {
            "Covariate": cov,
            "Unadjusted_SMD": u_smd,
            "Unadjusted_Abs_SMD": abs(u_smd) if np.isfinite(u_smd) else np.nan,
        }
        if matched_df is not None:
            m_smd = _normalize_smd(matched_map.get(cov))
            row["Matched_SMD"] = m_smd
            row["Matched_Abs_SMD"] = abs(m_smd) if np.isfinite(m_smd) else np.nan
        if weights_col is not None:
            w_smd = _normalize_smd(weighted_map.get(cov))
            row["Weighted_SMD"] = w_smd
            row["Weighted_Abs_SMD"] = abs(w_smd) if np.isfinite(w_smd) else np.nan

        eval_smd = (
            row.get("Weighted_Abs_SMD")
            or row.get("Matched_Abs_SMD")
            or row["Unadjusted_Abs_SMD"]
        )
        row["Status"] = (
            "Balanced" if (np.isfinite(eval_smd) and eval_smd < 0.10) else "Imbalanced"
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _normalize_smd(val: Any) -> float:
    if val is None or pd.isna(val):
        return np.nan
    try:
        f = float(val)
        return f if np.isfinite(f) else np.nan
    except (ValueError, TypeError):
        return np.nan


def compare_pre_post_balance(
    raw_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    treatment: str,
    covariates: list[str],
) -> pd.DataFrame:
    """
    Compare covariate balance before and after matching.
    """
    pre_bal = check_balance(raw_df, treatment, covariates)
    post_bal = check_balance(matched_df, treatment, covariates)

    combined = []
    for cov in covariates:
        pre_row = pre_bal[pre_bal["Covariate"] == cov]
        post_row = post_bal[post_bal["Covariate"] == cov]

        pre_smd = (
            _normalize_smd(pre_row["SMD"].iloc[0]) if not pre_row.empty else np.nan
        )
        post_smd = (
            _normalize_smd(post_row["SMD"].iloc[0]) if not post_row.empty else np.nan
        )

        combined.append(
            {
                "Covariate": cov,
                "Pre_SMD": pre_smd,
                "Post_SMD": post_smd,
                "Pre_Abs_SMD": abs(pre_smd) if np.isfinite(pre_smd) else np.nan,
                "Post_Abs_SMD": abs(post_smd) if np.isfinite(post_smd) else np.nan,
                "Balanced_Post": bool(abs(post_smd) < 0.10)
                if np.isfinite(post_smd)
                else False,
            }
        )

    return pd.DataFrame(combined)


def create_love_plot(
    balance_comparison_df: pd.DataFrame,
    threshold: float = 0.10,
) -> go.Figure:
    """
    Generate an interactive Love Plot comparing Pre- vs Post-match or Weighted absolute SMDs.
    """
    sort_col = (
        "Pre_Abs_SMD"
        if "Pre_Abs_SMD" in balance_comparison_df.columns
        else balance_comparison_df.columns[1]
    )
    df = balance_comparison_df.sort_values(by=sort_col, ascending=True)

    fig = go.Figure()

    # Pre-matching points
    pre_col = "Pre_Abs_SMD" if "Pre_Abs_SMD" in df.columns else "Pre_SMD"
    if pre_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[pre_col].abs(),
                y=df["Covariate"],
                mode="markers",
                marker=dict(color=COLORS["danger"], size=9, symbol="circle"),
                name="Unadjusted (Pre-match)",
            )
        )

    # Post-matching points
    post_col = "Post_Abs_SMD" if "Post_Abs_SMD" in df.columns else "Post_SMD"
    if post_col in df.columns and df[post_col].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df[post_col].abs(),
                y=df["Covariate"],
                mode="markers",
                marker=dict(color=COLORS["success"], size=9, symbol="diamond"),
                name="Adjusted (Matched)",
            )
        )

    # Weighted points
    w_col = "Weighted_Abs_SMD" if "Weighted_Abs_SMD" in df.columns else "Weighted_SMD"
    if w_col in df.columns and df[w_col].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df[w_col].abs(),
                y=df["Covariate"],
                mode="markers",
                marker=dict(color="#2563EB", size=9, symbol="square"),
                name="Adjusted (IPTW Weighted)",
            )
        )

    # Threshold line at SMD = threshold
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color=COLORS["warning"],
        annotation_text=f"Threshold ({threshold:.2f})",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Love Plot: Covariate Balance Before and After Matching",
        xaxis_title="Absolute Standardized Mean Difference (SMD)",
        yaxis_title="Covariates",
        template="plotly_white",
        height=max(400, len(df) * 30),
    )

    return fig

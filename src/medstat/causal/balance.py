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
        mean_t = np.average(t_valid, weights=wt)
        var_t = np.average((t_valid - mean_t) ** 2, weights=wt)
    else:
        mean_t = np.mean(t_valid)
        var_t = np.var(t_valid, ddof=1) if len(t_valid) > 1 else 0.0

    if weights_control is not None:
        wc = np.asarray(weights_control, dtype=float)[~np.isnan(c_arr)]
        mean_c = np.average(c_valid, weights=wc)
        var_c = np.average((c_valid - mean_c) ** 2, weights=wc)
    else:
        mean_c = np.mean(c_valid)
        var_c = np.var(c_valid, ddof=1) if len(c_valid) > 1 else 0.0

    pooled_sd = np.sqrt((var_t + var_c) / 2.0)
    if pooled_sd == 0:
        return 0.0

    return float((mean_t - mean_c) / pooled_sd)


def check_balance(
    df: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    weights: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Calculate SMDs across multiple covariates in a cohort.
    """
    treated_mask = df[treatment] == 1
    control_mask = df[treatment] == 0

    results = []
    for cov in covariates:
        if not pd.api.types.is_numeric_dtype(df[cov]):
            continue

        smd = calculate_smd(
            df.loc[treated_mask, cov],
            df.loc[control_mask, cov],
            weights_treated=weights[treated_mask] if weights is not None else None,
            weights_control=weights[control_mask] if weights is not None else None,
        )

        status = "Balanced" if abs(smd) < 0.10 else "Imbalanced"
        results.append(
            {
                "Covariate": cov,
                "Variable": cov,
                "SMD": smd,
                "Absolute_SMD": abs(smd),
                "Status": status,
            }
        )

    return pd.DataFrame(results)


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

        pre_smd = float(pre_row["SMD"].iloc[0]) if not pre_row.empty else np.nan
        post_smd = float(post_row["SMD"].iloc[0]) if not post_row.empty else np.nan

        combined.append(
            {
                "Covariate": cov,
                "Pre_SMD": pre_smd,
                "Post_SMD": post_smd,
                "Pre_Abs_SMD": abs(pre_smd),
                "Post_Abs_SMD": abs(post_smd),
                "Balanced_Post": abs(post_smd) < 0.10,
            }
        )

    return pd.DataFrame(combined)


def create_love_plot(balance_comparison_df: pd.DataFrame) -> go.Figure:
    """
    Generate an interactive Love Plot comparing Pre- vs Post-match absolute SMDs.
    """
    df = balance_comparison_df.sort_values(by="Pre_Abs_SMD", ascending=True)

    fig = go.Figure()

    # Pre-matching points
    fig.add_trace(
        go.Scatter(
            x=df["Pre_Abs_SMD"],
            y=df["Covariate"],
            mode="markers",
            marker=dict(color=COLORS["danger"], size=9, symbol="circle"),
            name="Unadjusted (Pre-match)",
        )
    )

    # Post-matching points
    fig.add_trace(
        go.Scatter(
            x=df["Post_Abs_SMD"],
            y=df["Covariate"],
            mode="markers",
            marker=dict(color=COLORS["success"], size=9, symbol="diamond"),
            name="Adjusted (Matched)",
        )
    )

    # Threshold line at SMD = 0.10
    fig.add_vline(
        x=0.10,
        line_dash="dash",
        line_color=COLORS["warning"],
        annotation_text="Threshold (0.10)",
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

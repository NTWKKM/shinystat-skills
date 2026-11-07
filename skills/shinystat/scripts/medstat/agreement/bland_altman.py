"""
Bland-Altman Agreement Analysis.

Calculates mean differences (bias), limits of agreement (LoA),
confidence intervals via Carkeet (2015) / Bland & Altman (1999),
and optional headless Plotly visualizations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats

from medstat.theme.palette import get_color_palette

COLORS = get_color_palette()


def calculate_bland_altman(
    df: pd.DataFrame,
    method1: str,
    method2: str,
    ci: float = 0.95,
) -> dict[str, Any]:
    """
    Calculate Bland-Altman agreement statistics between two measurement methods.

    Parameters:
        df: Input DataFrame containing measurement columns.
        method1: Column name for first method (method1 - method2).
        method2: Column name for second method.
        ci: Confidence level for confidence intervals (default 0.95).

    Returns:
        Dictionary with n, mean_diff, ci_mean_diff, sd_diff,
        upper_loa, lower_loa, ci_upper_loa, ci_lower_loa.
    """
    sub = df[[method1, method2]].dropna()
    n = len(sub)
    if n < 2:
        raise ValueError(
            f"Bland-Altman requires at least 2 non-missing pairs (got {n})."
        )

    m1 = sub[method1].to_numpy(dtype=float)
    m2 = sub[method2].to_numpy(dtype=float)

    diffs = m1 - m2
    means = (m1 + m2) / 2.0

    mean_diff = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1))
    se_mean_diff = sd_diff / np.sqrt(n)

    # Critical t-value for CI around mean difference
    t_crit = stats.t.ppf(1.0 - (1.0 - ci) / 2.0, df=n - 1)
    ci_md_low = float(mean_diff - t_crit * se_mean_diff)
    ci_md_high = float(mean_diff + t_crit * se_mean_diff)

    # Standard Limits of Agreement (1.96 * SD)
    z_loa = stats.norm.ppf(1.0 - (1.0 - ci) / 2.0)
    loa_range = z_loa * sd_diff
    upper_loa = float(mean_diff + loa_range)
    lower_loa = float(mean_diff - loa_range)

    # Standard error of the limits of agreement (Bland & Altman 1999)
    # var(LoA) approx = (1/n + z^2 / (2(n-1))) * s^2
    se_loa = float(np.sqrt((1.0 / n + (z_loa**2) / (2.0 * (n - 1))) * (sd_diff**2)))

    ci_upper_loa_low = float(upper_loa - t_crit * se_loa)
    ci_upper_loa_high = float(upper_loa + t_crit * se_loa)

    ci_lower_loa_low = float(lower_loa - t_crit * se_loa)
    ci_lower_loa_high = float(lower_loa + t_crit * se_loa)

    return {
        "n": n,
        "mean_diff": mean_diff,
        "ci_mean_diff": [ci_md_low, ci_md_high],
        "sd_diff": sd_diff,
        "upper_loa": upper_loa,
        "lower_loa": lower_loa,
        "ci_upper_loa": [ci_upper_loa_low, ci_upper_loa_high],
        "ci_lower_loa": [ci_lower_loa_low, ci_lower_loa_high],
        "means": means,
        "diffs": diffs,
    }


def create_bland_altman_plot(
    stats_res: dict[str, Any],
    method1: str,
    method2: str,
    show_ci_bands: bool = True,
) -> go.Figure:
    """
    Generate a Plotly figure representing the Bland-Altman analysis.
    """
    means = stats_res["means"]
    diffs = stats_res["diffs"]
    mean_diff = stats_res["mean_diff"]
    upper_loa = stats_res["upper_loa"]
    lower_loa = stats_res["lower_loa"]

    fig = go.Figure()

    # Scatter points
    fig.add_trace(
        go.Scatter(
            x=means,
            y=diffs,
            mode="markers",
            marker=dict(color=COLORS["primary"], size=7, opacity=0.7),
            name="Pairs",
            hovertemplate="Mean: %{x:.2f}<br>Diff: %{y:.2f}<extra></extra>",
        )
    )

    # Mean difference line
    fig.add_hline(
        y=mean_diff,
        line_dash="solid",
        line_color=COLORS["primary"],
        annotation_text=f"Mean: {mean_diff:.2f}",
        annotation_position="top right",
    )

    # Upper LoA
    fig.add_hline(
        y=upper_loa,
        line_dash="dash",
        line_color=COLORS["danger"],
        annotation_text=f"+1.96 SD: {upper_loa:.2f}",
        annotation_position="top right",
    )

    # Lower LoA
    fig.add_hline(
        y=lower_loa,
        line_dash="dash",
        line_color=COLORS["danger"],
        annotation_text=f"-1.96 SD: {lower_loa:.2f}",
        annotation_position="bottom right",
    )

    if show_ci_bands:
        ci_md = stats_res["ci_mean_diff"]
        ci_u = stats_res["ci_upper_loa"]
        ci_l = stats_res["ci_lower_loa"]

        fig.add_hrect(
            y0=ci_md[0],
            y1=ci_md[1],
            line_width=0,
            fillcolor=COLORS["primary"],
            opacity=0.15,
        )
        fig.add_hrect(
            y0=ci_u[0],
            y1=ci_u[1],
            line_width=0,
            fillcolor=COLORS["danger"],
            opacity=0.10,
        )
        fig.add_hrect(
            y0=ci_l[0],
            y1=ci_l[1],
            line_width=0,
            fillcolor=COLORS["danger"],
            opacity=0.10,
        )

    fig.update_layout(
        title=f"Bland-Altman Plot: {method1} vs {method2}",
        xaxis_title=f"Mean of {method1} & {method2}",
        yaxis_title=f"Difference ({method1} - {method2})",
        template="plotly_white",
        height=500,
        hovermode="closest",
    )

    return fig


def bland_altman_analysis(
    m1_or_df: Any,
    m2_or_col1: Any = None,
    method2_col: str | None = None,
    ci: float = 0.95,
) -> dict[str, Any]:
    """
    Flexible wrapper supporting:
    - bland_altman_analysis(df, 'col1', 'col2', ci=0.95)
    - bland_altman_analysis(series1, series2, ci=0.95)
    """
    if (
        isinstance(m1_or_df, pd.DataFrame)
        and isinstance(m2_or_col1, str)
        and method2_col is not None
    ):
        return calculate_bland_altman(m1_or_df, m2_or_col1, method2_col, ci=ci)
    elif (
        isinstance(m1_or_df, pd.DataFrame)
        and method2_col is None
        and m2_or_col1 is None
    ):
        cols = [
            c for c in m1_or_df.columns if pd.api.types.is_numeric_dtype(m1_or_df[c])
        ][:2]
        return calculate_bland_altman(m1_or_df, cols[0], cols[1], ci=ci)
    else:
        s1 = pd.Series(m1_or_df).reset_index(drop=True)
        s2 = pd.Series(m2_or_col1).reset_index(drop=True)
        tdf = pd.DataFrame({"_m1": s1, "_m2": s2})
        return calculate_bland_altman(tdf, "_m1", "_m2", ci=ci)

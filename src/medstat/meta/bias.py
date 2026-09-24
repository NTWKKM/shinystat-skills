"""
Publication Bias and Small-Study Effects Suite for Meta-Analysis.

Provides Egger's linear regression test, Begg's rank test, and
contour-enhanced Funnel Plots.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
import statsmodels.api as sm

from medstat.theme.palette import get_color_palette

COLORS = get_color_palette()


def eggers_test(
    df_or_theta: Any,
    se_or_effect_col: Any = None,
    se_col: str | None = None,
) -> dict[str, Any]:
    """
    Egger's linear regression test for funnel plot asymmetry.

    Regresses standardized effect (log_effect / se) on precision (1 / se).
    An intercept significantly different from 0 indicates funnel plot asymmetry.
    Supports:
    - eggers_test(df)
    - eggers_test(theta, se)
    - eggers_test(df, effect_col, se_col)
    """
    if isinstance(df_or_theta, pd.DataFrame):
        df = df_or_theta
        if se_or_effect_col and se_col:
            theta = np.asarray(df[se_or_effect_col], dtype=float)
            se = np.asarray(df[se_col], dtype=float)
        else:
            eff_col = (
                "log_effect"
                if "log_effect" in df.columns
                else ("effect" if "effect" in df.columns else "effect_size")
            )
            theta = np.asarray(df[eff_col], dtype=float)
            se = np.asarray(df["se"], dtype=float)
    else:
        theta = np.asarray(df_or_theta, dtype=float)
        se = np.asarray(se_or_effect_col, dtype=float)

    k = len(theta)
    if k < 3:
        raise ValueError(f"Egger's test requires at least 3 studies (got {k}).")

    # Standardized effect z = theta / se, precision = 1 / se
    z = theta / se
    prec = 1.0 / se
    X = sm.add_constant(prec)

    model = sm.OLS(z, X).fit()
    intercept = float(model.params[0])
    se_inter = float(model.bse[0])
    t_stat = float(model.tvalues[0])
    p_val = float(model.pvalues[0])

    has_bias = bool(
        p_val < 0.10
    )  # Standard 0.10 threshold for publication bias screening

    return {
        "test": "Egger's linear regression",
        "intercept": intercept,
        "std_error": se_inter,
        "t_statistic": t_stat,
        "p_value": p_val,
        "asymmetry_detected": has_bias,
        "k_studies": k,
    }


def beggs_test(df: pd.DataFrame) -> dict[str, Any]:
    """
    Begg and Mazumdar rank correlation test between standardized effect size and variance.
    """
    k = len(df)
    if k < 3:
        raise ValueError(f"Begg's test requires at least 3 studies (got {k}).")

    theta = df["log_effect"].to_numpy(dtype=float)
    se = df["se"].to_numpy(dtype=float)

    # Variance-weighted mean effect for standardization
    v = se**2
    w = 1.0 / v
    sum_w = np.sum(w)
    theta_bar = np.sum(w * theta) / sum_w
    var_pooled = 1.0 / sum_w
    std_effects = (theta - theta_bar) / np.sqrt(v - var_pooled)

    tau, p_val = stats.kendalltau(std_effects, v)

    return {
        "test": "Begg and Mazumdar rank correlation",
        "kendall_tau": float(tau),
        "p_value": float(p_val),
        "asymmetry_detected": bool(p_val < 0.10),
        "k_studies": k,
    }


def run_publication_bias_tests(df: pd.DataFrame) -> dict[str, Any]:
    """
    Run complete publication bias evaluation suite (Egger and Begg tests).
    """
    egger = eggers_test(df)
    begg = beggs_test(df)

    return {
        "egger": egger,
        "begg": begg,
        "summary": "Potential publication bias detected (p < 0.10)"
        if (egger["asymmetry_detected"] or begg["asymmetry_detected"])
        else "No significant funnel plot asymmetry detected",
    }


def create_funnel_plot(
    df: pd.DataFrame,
    pooled_effect: float | None = None,
    is_ratio: bool = True,
) -> go.Figure:
    """
    Generate an interactive Plotly Funnel Plot with pseudo 95% confidence intervals.
    """
    effects = df["effect_size"].to_numpy(dtype=float)
    ses = df["se"].to_numpy(dtype=float)

    if pooled_effect is None:
        pooled_effect = float(np.mean(effects))

    fig = go.Figure()

    # Individual study points
    fig.add_trace(
        go.Scatter(
            x=effects,
            y=ses,
            mode="markers",
            marker=dict(color=COLORS["primary"], size=8, opacity=0.8),
            name="Studies",
            text=df["study"] if "study" in df.columns else None,
            hovertemplate="Study: %{text}<br>Effect: %{x:.2f}<br>SE: %{y:.2f}<extra></extra>",
        )
    )

    # Vertical pooled effect line
    fig.add_vline(
        x=pooled_effect,
        line_dash="solid",
        line_color=COLORS["danger"],
        annotation_text="Pooled Effect",
    )

    # Funnel boundaries (95% CI: pooled_effect +/- 1.96 * SE)
    se_grid = np.linspace(0.001, max(float(np.max(ses)) * 1.15, 0.5), 100)
    if is_ratio:
        log_pooled = np.log(pooled_effect)
        left_bound = np.exp(log_pooled - 1.96 * se_grid)
        right_bound = np.exp(log_pooled + 1.96 * se_grid)
    else:
        left_bound = pooled_effect - 1.96 * se_grid
        right_bound = pooled_effect + 1.96 * se_grid

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([left_bound, right_bound[::-1]]),
            y=np.concatenate([se_grid, se_grid[::-1]]),
            fill="toself",
            fillcolor="rgba(15, 23, 42, 0.05)",
            line=dict(color="rgba(15, 23, 42, 0.3)", dash="dash"),
            name="Pseudo 95% CI Funnel",
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title="Funnel Plot for Publication Bias Assessment",
        xaxis_title="Observed Effect Size" + (" (log scale)" if is_ratio else ""),
        yaxis_title="Standard Error (SE)",
        yaxis=dict(autorange="reversed"),  # Invert y-axis: small SE at top
        xaxis_type="log" if is_ratio else "linear",
        template="plotly_white",
    )

    return fig

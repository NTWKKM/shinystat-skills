"""
Model Calibration and Goodness-of-Fit Validation.

Provides calibration curves, Brier score, calibration slope & intercept,
Integrated Calibration Index (ICI / E50 / E90 / Emax), and Hosmer-Lemeshow test.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from medstat.theme.palette import get_color_palette

COLORS = get_color_palette()


def calculate_brier_score(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """
    Calculate Brier Score and scaled Brier score relative to incidence.
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()

    valid = ~(np.isnan(y_t) | np.isnan(y_p))
    y_t = y_t[valid]
    y_p = y_p[valid]

    brier = float(brier_score_loss(y_t, y_p))
    prevalence = float(np.mean(y_t))
    brier_ref = prevalence * (1.0 - prevalence)
    brier_scaled = float(1.0 - (brier / brier_ref)) if brier_ref > 0 else np.nan

    interpretation = (
        "Excellent"
        if brier < 0.10
        else "Good"
        if brier < 0.20
        else "Acceptable"
        if brier < 0.25
        else "Poor"
    )

    return {
        "brier_score": brier,
        "brier_scaled": brier_scaled,
        "brier_reference": brier_ref,
        "interpretation": interpretation,
        "n": len(y_t),
    }


def calculate_calibration_curve(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """
    Compute calibration curve: observed event proportions vs predicted mean probabilities.
    """
    y_t = np.asarray(y_true, dtype=int).ravel()
    y_p = np.clip(np.asarray(y_pred, dtype=float).ravel(), 1e-6, 1.0 - 1e-6)

    prob_true, prob_pred = calibration_curve(y_t, y_p, n_bins=n_bins, strategy=strategy)

    return pd.DataFrame(
        {
            "bin": range(1, len(prob_true) + 1),
            "prob_pred": prob_pred,
            "prob_observed": prob_true,
        }
    )


def calculate_calibration_slope_and_intercept(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """
    Calculate calibration intercept (calibration-in-the-large) and calibration slope.
    Ideal: intercept = 0, slope = 1.
    """
    y_t = np.asarray(y_true, dtype=int).ravel()
    y_p = np.clip(np.asarray(y_pred, dtype=float).ravel(), 1e-6, 1.0 - 1e-6)

    # Log-odds of predictions
    log_odds = stats.logit(y_p)

    # 1. Calibration Slope: logit(y) = a + b * logit(p)
    X_slope = sm.add_constant(log_odds)
    try:
        model_slope = sm.Logit(y_t, X_slope).fit(disp=False)
        slope = float(model_slope.params[1])
        slope_ci = [float(c) for c in model_slope.conf_int()[1]]
        slope_p = float(model_slope.pvalues[1])
    except Exception:
        slope, slope_ci, slope_p = np.nan, [np.nan, np.nan], np.nan

    # 2. Calibration-in-the-large (Intercept with slope offset = 1)
    # logit(y) = a + offset(logit(p))
    try:
        model_inter = sm.Logit(y_t, np.ones_like(y_t), offset=log_odds).fit(disp=False)
        intercept = float(model_inter.params[0])
        inter_ci = [float(c) for c in model_inter.conf_int()[0]]
        inter_p = float(model_inter.pvalues[0])
    except Exception:
        intercept, inter_ci, inter_p = np.nan, [np.nan, np.nan], np.nan

    return {
        "calibration_slope": slope,
        "slope_ci": slope_ci,
        "slope_pvalue": slope_p,
        "calibration_intercept": intercept,
        "intercept_ci": inter_ci,
        "intercept_pvalue": inter_p,
    }


def calculate_ici(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """
    Calculate Integrated Calibration Index (ICI), E50, E90, and Emax
    according to Austin & Steyerberg (2019).
    """
    y_t = np.asarray(y_true, dtype=int).ravel()
    y_p = np.clip(np.asarray(y_pred, dtype=float).ravel(), 1e-6, 1.0 - 1e-6)

    # Binning approximation or spline smoothing
    # Sort by predicted probability
    order = np.argsort(y_p)
    y_p_sorted = y_p[order]
    y_t_sorted = y_t[order]

    # Use rolling window or deciles
    window_size = max(len(y_t) // 10, 5)
    smooth_obs = (
        pd.Series(y_t_sorted)
        .rolling(window=window_size, center=True, min_periods=3)
        .mean()
        .bfill()
        .ffill()
        .to_numpy()
    )

    abs_errors = np.abs(y_p_sorted - smooth_obs)

    ici = float(np.mean(abs_errors))
    e50 = float(np.median(abs_errors))
    e90 = float(np.percentile(abs_errors, 90))
    emax = float(np.max(abs_errors))

    return {
        "ici": ici,
        "e50": e50,
        "e90": e90,
        "emax": emax,
    }


def hosmer_lemeshow_test(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    g: int = 10,
) -> dict[str, Any]:
    """
    Hosmer-Lemeshow Goodness-of-Fit test for binary logistic models.
    """
    y_t = np.asarray(y_true, dtype=int).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()

    n = len(y_t)
    if n < g:
        raise ValueError(f"Sample size {n} is less than number of groups {g}.")

    # Quantile bins based on predicted probabilities
    df = pd.DataFrame({"y": y_t, "p": y_p})
    df["group"] = pd.qcut(df["p"], q=g, duplicates="drop")

    chi2 = 0.0
    groups = df["group"].unique()

    for grp in groups:
        sub = df[df["group"] == grp]
        n_k = len(sub)
        o_k = np.sum(sub["y"])
        e_k = np.sum(sub["p"])
        e_k_neg = n_k - e_k

        if e_k > 0 and e_k_neg > 0:
            term_pos = ((o_k - e_k) ** 2) / e_k
            term_neg = (((n_k - o_k) - e_k_neg) ** 2) / e_k_neg
            chi2 += term_pos + term_neg

    df_dof = len(groups) - 2
    df_dof = max(df_dof, 1)
    p_val = float(stats.chi2.sf(chi2, df=df_dof))

    return {
        "test": "Hosmer-Lemeshow",
        "statistic": float(chi2),
        "df": int(df_dof),
        "p_value": p_val,
        "is_calibrated": bool(p_val >= 0.05),
    }


def create_calibration_plot(
    curve_df: pd.DataFrame,
    slope: float | None = None,
    intercept: float | None = None,
) -> go.Figure:
    """
    Generate Plotly calibration plot with 45-degree ideal line.
    """
    fig = go.Figure()

    # Ideal calibration line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name="Ideal Calibration",
        )
    )

    # Observed vs Predicted points
    fig.add_trace(
        go.Scatter(
            x=curve_df["prob_pred"],
            y=curve_df["prob_observed"],
            mode="lines+markers",
            line=dict(color=COLORS["primary"], width=2),
            marker=dict(size=8, color=COLORS["primary"]),
            name="Model",
        )
    )

    title_text = "Calibration Curve"
    if slope is not None and intercept is not None:
        title_text += f" (Slope = {slope:.2f}, Intercept = {intercept:.2f})"

    fig.update_layout(
        title=title_text,
        xaxis_title="Predicted Probability",
        yaxis_title="Observed Proportion",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
    )

    return fig

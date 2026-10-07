"""
Decision Curve Analysis (DCA) and Clinical Net Benefit.

Implements Vickers & Elkin (2006) decision curve analysis comparing model
net benefit against 'treat all' and 'treat none' across clinical decision thresholds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from medstat.theme.palette import get_color_palette

COLORS = get_color_palette()


def calculate_dca(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    thresholds: list[float] | np.ndarray | None = None,
    model_name: str = "Model",
) -> pd.DataFrame:
    """
    Calculate Net Benefit for model, Treat All, and Treat None strategies across threshold probabilities.

    Parameters:
        y_true: Binary ground-truth vector (0/1).
        y_pred: Predicted risk probabilities (between 0 and 1).
        thresholds: Array of decision threshold probabilities pt (default 0.01 to 0.99).
        model_name: Label for the predictive model.

    Returns:
        pd.DataFrame with columns: threshold, net_benefit, strategy, tp_rate, fp_rate
    """
    y_t = np.asarray(y_true, dtype=int).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()

    valid = ~(np.isnan(y_t) | np.isnan(y_p))
    y_t = y_t[valid]
    y_p = y_p[valid]

    n = len(y_t)
    if n == 0:
        raise ValueError("Cannot perform DCA on empty data.")

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    results = []

    for pt in thresholds:
        pt_val = float(pt)
        weight = pt_val / (1.0 - pt_val) if pt_val < 1.0 else 0.0

        # Model strategy
        pred_pos = (y_p >= pt_val).astype(int)
        tp = np.sum((pred_pos == 1) & (y_t == 1))
        fp = np.sum((pred_pos == 1) & (y_t == 0))
        nb_model = (tp / n) - (fp / n) * weight

        results.append(
            {
                "threshold": pt_val,
                "net_benefit": float(nb_model),
                "strategy": model_name,
                "tp_rate": float(tp / n),
                "fp_rate": float(fp / n),
            }
        )

        # Treat All strategy
        tp_all = np.sum(y_t == 1)
        fp_all = np.sum(y_t == 0)
        nb_all = (tp_all / n) - (fp_all / n) * weight
        results.append(
            {
                "threshold": pt_val,
                "net_benefit": float(nb_all),
                "strategy": "Treat All",
                "tp_rate": float(tp_all / n),
                "fp_rate": float(fp_all / n),
            }
        )

        # Treat None strategy
        results.append(
            {
                "threshold": pt_val,
                "net_benefit": 0.0,
                "strategy": "Treat None",
                "tp_rate": 0.0,
                "fp_rate": 0.0,
            }
        )

    return pd.DataFrame(results)


def calculate_net_benefit(
    df: pd.DataFrame,
    truth_col: str,
    prob_col: str,
    thresholds: list[float] | np.ndarray | None = None,
    model_name: str = "Model",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    DataFrame wrapper for calculate_dca.
    """
    sub = df[[truth_col, prob_col]].dropna()
    dca_df = calculate_dca(
        sub[truth_col], sub[prob_col], thresholds=thresholds, model_name=model_name
    )
    meta = {
        "n": len(sub),
        "prevalence": float(sub[truth_col].mean()),
        "model_name": model_name,
    }
    return dca_df, meta


def create_dca_plot(dca_df: pd.DataFrame) -> go.Figure:
    """
    Generate interactive Plotly DCA curves.
    """
    fig = go.Figure()
    strategies = dca_df["strategy"].unique()

    color_map = {
        "Treat All": "gray",
        "Treat None": "black",
    }

    for strat in strategies:
        strat_df = dca_df[dca_df["strategy"] == strat]
        color = color_map.get(strat, COLORS["primary"])
        dash = "dash" if strat in color_map else "solid"
        width = 1.5 if strat in color_map else 2.5

        fig.add_trace(
            go.Scatter(
                x=strat_df["threshold"],
                y=strat_df["net_benefit"],
                mode="lines",
                name=strat,
                line=dict(color=color, dash=dash, width=width),
            )
        )

    # Calculate y-limits
    y_max = max(float(dca_df["net_benefit"].max()), 0.1) if not dca_df.empty else 0.5
    y_min = -0.05

    fig.update_layout(
        title="Decision Curve Analysis (Net Benefit)",
        xaxis_title="Threshold Probability",
        yaxis_title="Net Benefit",
        yaxis=dict(range=[y_min, y_max * 1.1]),
        template="plotly_white",
        hovermode="x unified",
    )

    return fig

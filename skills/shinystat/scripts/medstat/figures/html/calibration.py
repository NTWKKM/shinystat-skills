"""
src/medstat/figures/html/calibration.py: Dual-view (Plotly + 300 DPI Matplotlib) Calibration curve generator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from medstat.figures.calibration import (
    plot_bootstrap_optimism as plot_bootstrap_optimism_static,
)
from medstat.figures.calibration import (
    plot_calibration,
)
from medstat.figures.calibration import (
    plot_recalibration_comparison as plot_recalibration_comparison_static,
)
from medstat.figures.html.base import DualFigureResult


def plot_dual_calibration(
    calib_bins: dict[str, np.ndarray],
    calib_curve: dict[str, np.ndarray] | None = None,
    brier_score: float | None = None,
    ici: float | None = None,
    title: str = "Model Calibration Curve",
    out_path: str | Path = "figures/calibration_plot.png",
    target: Literal["docx", "pptx", "web"] = "docx",
) -> DualFigureResult:
    """
    Renders both publication-grade 300 DPI Matplotlib figure and interactive Plotly calibration curve.
    """
    res = plot_calibration(
        calib_bins=calib_bins,
        calib_curve=calib_curve,
        brier_score=brier_score,
        ici=ici,
        title=title,
        out_path=out_path,
        target=target,
    )

    p_means = np.asarray(calib_bins["pred_mean"], dtype=float)
    o_rates = np.asarray(calib_bins["obs_rate"], dtype=float)
    ci_low = np.asarray(calib_bins.get("ci_lower", o_rates), dtype=float)
    ci_high = np.asarray(calib_bins.get("ci_upper", o_rates), dtype=float)

    fig = go.Figure()

    # Ideal calibration line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="#94A3B8", width=1.5),
            name="Ideal Calibration",
            hoverinfo="skip",
        )
    )

    # Smoothed trajectory if present
    if calib_curve is not None:
        p_smooth = np.asarray(calib_curve["pred_smooth"], dtype=float)
        o_smooth = np.asarray(calib_curve["obs_smooth"], dtype=float)
        fig.add_trace(
            go.Scatter(
                x=p_smooth,
                y=o_smooth,
                mode="lines",
                line=dict(color="#0284C7", width=2),
                name="Smoothed Calibration",
                hovertemplate="Pred: %{x:.3f}<br>Obs: %{y:.3f}<extra></extra>",
            )
        )

    # Observed deciles with error bars
    error_y_minus = np.maximum(0, o_rates - ci_low)
    error_y_plus = np.maximum(0, ci_high - o_rates)

    fig.add_trace(
        go.Scatter(
            x=p_means,
            y=o_rates,
            customdata=np.column_stack([ci_low, ci_high]).tolist(),
            mode="markers",
            marker=dict(size=9, color="#0F172A", symbol="circle"),
            error_y=dict(
                type="data",
                symmetric=False,
                array=error_y_plus,
                arrayminus=error_y_minus,
                color="#0F172A",
                width=1.5,
            ),
            name="Decile Bins (95% CI)",
            hovertemplate=(
                "Pred Mean: %{x:.3f}<br>"
                "Obs Rate: %{y:.3f}<br>"
                "95% CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<extra></extra>"
            ),
        )
    )

    # Title & metrics
    title_text = title
    sub_metrics = []
    if brier_score is not None:
        sub_metrics.append(f"Brier = {brier_score:.3f}")
    if ici is not None:
        sub_metrics.append(f"ICI = {ici:.3f}")
    if sub_metrics:
        title_text += f" ({', '.join(sub_metrics)})"

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=15, color="#0F172A")),
        xaxis=dict(
            title="Predicted Probability",
            range=[0, 1.02],
            showgrid=True,
            gridcolor="#F1F5F9",
        ),
        yaxis=dict(
            title="Observed Proportion",
            range=[0, 1.02],
            showgrid=True,
            gridcolor="#F1F5F9",
        ),
        template="plotly_white",
        legend=dict(
            yanchor="bottom",
            y=0.02,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="#E2E8F0",
            borderwidth=1,
        ),
        margin=dict(l=50, r=30, t=50, b=50),
        height=500,
    )

    return DualFigureResult(
        png_path=res.png_path,
        alt_text=res.alt_text,
        caption=res.caption,
        source_df=res.source_df,
        csv_path=res.csv_path,
        plotly_fig=fig,
    )


def plot_dual_recalibration_comparison(
    y_true: pd.Series | np.ndarray | dict[str, Any],
    p_orig: pd.Series | np.ndarray | None = None,
    p_recal: pd.Series | np.ndarray | None = None,
    n_bins: int = 10,
    title: str = "Model Recalibration Comparison",
    out_path: str | Path | None = None,
    output_path: str | Path | None = None,
    target: Literal["docx", "pptx", "web"] = "docx",
) -> DualFigureResult:
    """
    Renders publication 300 DPI Matplotlib figure and interactive Plotly recalibration comparison.
    """
    res = plot_recalibration_comparison_static(
        y_true=y_true,
        p_orig=p_orig,
        p_recal=p_recal,
        n_bins=n_bins,
        title=title,
        out_path=out_path,
        output_path=output_path,
        target=target,
    )

    fig = go.Figure()

    # Ideal calibration line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="#94A3B8", width=1.5),
            name="Ideal Calibration",
            hoverinfo="skip",
        )
    )

    # Pre-recalibration trace
    df_src = res.source_df.dropna(subset=["Pre_Pred_Mean", "Pre_Obs_Rate"])
    fig.add_trace(
        go.Scatter(
            x=df_src["Pre_Pred_Mean"],
            y=df_src["Pre_Obs_Rate"],
            mode="lines+markers",
            line=dict(dash="dash", color="#64748B", width=2),
            marker=dict(
                symbol="circle-open", size=9, color="#64748B", line=dict(width=2)
            ),
            name="Pre-Recalibration",
            hovertemplate="Pred Mean: %{x:.3f}<br>Obs Rate: %{y:.3f}<extra>Pre-Recalibration</extra>",
        )
    )

    # Post-recalibration trace
    df_src_r = res.source_df.dropna(subset=["Post_Pred_Mean", "Post_Obs_Rate"])
    fig.add_trace(
        go.Scatter(
            x=df_src_r["Post_Pred_Mean"],
            y=df_src_r["Post_Obs_Rate"],
            mode="lines+markers",
            line=dict(color="#0F172A", width=2.5),
            marker=dict(symbol="square", size=9, color="#0F172A"),
            name="Post-Recalibration",
            hovertemplate="Pred Mean: %{x:.3f}<br>Obs Rate: %{y:.3f}<extra>Post-Recalibration</extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#0F172A")),
        xaxis=dict(
            title="Predicted Probability",
            range=[0, 1.02],
            showgrid=True,
            gridcolor="#F1F5F9",
        ),
        yaxis=dict(
            title="Observed Proportion",
            range=[0, 1.02],
            showgrid=True,
            gridcolor="#F1F5F9",
        ),
        template="plotly_white",
        legend=dict(
            yanchor="bottom",
            y=0.02,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="#E2E8F0",
            borderwidth=1,
        ),
        margin=dict(l=50, r=30, t=50, b=50),
        height=500,
    )

    return DualFigureResult(
        png_path=res.png_path,
        alt_text=res.alt_text,
        caption=res.caption,
        source_df=res.source_df,
        csv_path=res.csv_path,
        plotly_fig=fig,
    )


def plot_dual_bootstrap_optimism(
    validation_res: dict[str, Any],
    title: str = "Bootstrap Internal Validation (Optimism)",
    out_path: str | Path | None = None,
    output_path: str | Path | None = None,
    target: Literal["docx", "pptx", "web"] = "docx",
) -> DualFigureResult:
    """
    Renders publication 300 DPI Matplotlib figure and interactive Plotly bootstrap optimism chart.
    """
    res = plot_bootstrap_optimism_static(
        validation_res=validation_res,
        title=title,
        out_path=out_path,
        output_path=output_path,
        target=target,
    )

    df_src = res.source_df
    s_hat = float(validation_res.get("shrinkage_factor", 1.0))

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=df_src["Metric"],
            x=df_src["Apparent"],
            orientation="h",
            name="Apparent Performance",
            marker=dict(color="#94A3B8"),
            hovertemplate="%{y}<br>Apparent: %{x:.3f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            y=df_src["Metric"],
            x=df_src["Optimism_Corrected"],
            orientation="h",
            name="Optimism-Corrected",
            marker=dict(color="#0F172A"),
            hovertemplate="%{y}<br>Optimism-Corrected: %{x:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        barmode="group",
        title=dict(
            text=f"{title} (Shrinkage Factor ŝ = {s_hat:.3f})",
            font=dict(size=15, color="#0F172A"),
        ),
        xaxis=dict(
            title="Metric Value",
            showgrid=True,
            gridcolor="#F1F5F9",
        ),
        yaxis=dict(
            title="",
            autorange="reversed",
        ),
        template="plotly_white",
        legend=dict(
            yanchor="bottom",
            y=0.02,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="#E2E8F0",
            borderwidth=1,
        ),
        margin=dict(l=140, r=30, t=50, b=50),
        height=450,
    )

    return DualFigureResult(
        png_path=res.png_path,
        alt_text=res.alt_text,
        caption=res.caption,
        source_df=res.source_df,
        csv_path=res.csv_path,
        plotly_fig=fig,
    )


# Alias parity
plot_recalibration_comparison = plot_dual_recalibration_comparison
plot_bootstrap_optimism = plot_dual_bootstrap_optimism

"""
src/medstat/figures/html/roc.py: Dual-view (Plotly + 300 DPI Matplotlib) ROC curve generator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go

from medstat.figures.html.base import DualFigureResult
from medstat.figures.roc import plot_roc_curve


def plot_dual_roc_curve(
    roc_data: dict[str, Any],
    paired_roc_data: dict[str, Any] | None = None,
    paired_p_value: float | None = None,
    title: str = "Receiver Operating Characteristic (ROC) Curve",
    out_path: str | Path = "figures/roc_curve.png",
    target: Literal["docx", "pptx", "web"] = "docx",
) -> DualFigureResult:
    """
    Renders both publication-grade 300 DPI Matplotlib figure and interactive Plotly ROC curve.
    """
    res = plot_roc_curve(
        roc_data=roc_data,
        paired_roc_data=paired_roc_data,
        paired_p_value=paired_p_value,
        title=title,
        out_path=out_path,
        target=target,
    )

    fpr = np.asarray(roc_data["fpr"], dtype=float)
    tpr = np.asarray(roc_data["tpr"], dtype=float)
    auc = float(roc_data.get("auc", 0.5))
    ci_l = roc_data.get("ci_lower")
    ci_u = roc_data.get("ci_upper")
    model_name = str(roc_data.get("name", "Model 1"))

    if ci_l is not None and ci_u is not None:
        label_1 = f"{model_name}: AUC = {auc:.3f} (95% CI, {ci_l:.3f}–{ci_u:.3f})"
    else:
        label_1 = f"{model_name}: AUC = {auc:.3f}"

    fig = go.Figure()

    # 45-degree chance diagonal
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="#94A3B8", width=1.5),
            name="Chance Diagonal (AUC = 0.500)",
            hoverinfo="skip",
        )
    )

    # Primary model ROC curve
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            line=dict(color="#0284C7", width=2.5),
            name=label_1,
            hovertemplate=(
                "<b>" + model_name + "</b><br>"
                "1 - Specificity (FPR): %{x:.3f}<br>"
                "Sensitivity (TPR): %{y:.3f}<extra></extra>"
            ),
        )
    )

    # Optimal cutoff marker
    opt_pt = roc_data.get("optimal_point")
    if opt_pt:
        opt_fpr = float(opt_pt["fpr"])
        opt_tpr = float(opt_pt["tpr"])
        opt_cut = float(opt_pt["cutoff"])
        opt_sens = float(roc_data.get("sensitivity", opt_tpr))
        opt_spec = float(roc_data.get("specificity", 1.0 - opt_fpr))

        fig.add_trace(
            go.Scatter(
                x=[opt_fpr],
                y=[opt_tpr],
                mode="markers",
                marker=dict(size=10, color="#DC2626", symbol="circle"),
                name=f"Optimal Cutpoint ({opt_cut:.2f})",
                hovertemplate=(
                    f"<b>Optimal Cutpoint: {opt_cut:.3f}</b><br>"
                    f"Sensitivity: {opt_sens:.1%}<br>"
                    f"Specificity: {opt_spec:.1%}<br>"
                    f"FPR: {opt_fpr:.3f}<extra></extra>"
                ),
            )
        )

    # Paired model curve if available
    if paired_roc_data is not None:
        fpr_2 = np.asarray(paired_roc_data["fpr"], dtype=float)
        tpr_2 = np.asarray(paired_roc_data["tpr"], dtype=float)
        auc_2 = float(paired_roc_data.get("auc", 0.5))
        ci_l2 = paired_roc_data.get("ci_lower")
        ci_u2 = paired_roc_data.get("ci_upper")
        model_name_2 = str(paired_roc_data.get("name", "Model 2"))

        if ci_l2 is not None and ci_u2 is not None:
            label_2 = (
                f"{model_name_2}: AUC = {auc_2:.3f} (95% CI, {ci_l2:.3f}–{ci_u2:.3f})"
            )
        else:
            label_2 = f"{model_name_2}: AUC = {auc_2:.3f}"

        if paired_p_value is not None:
            label_2 += f" (p = {paired_p_value:.3f})"

        fig.add_trace(
            go.Scatter(
                x=fpr_2,
                y=tpr_2,
                mode="lines",
                line=dict(color="#D97706", width=2.2, dash="dot"),
                name=label_2,
                hovertemplate=(
                    "<b>" + model_name_2 + "</b><br>"
                    "1 - Specificity (FPR): %{x:.3f}<br>"
                    "Sensitivity (TPR): %{y:.3f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#0F172A")),
        xaxis=dict(
            title="False Positive Rate (1 - Specificity)",
            range=[0, 1.02],
            showgrid=True,
            gridcolor="#F1F5F9",
        ),
        yaxis=dict(
            title="True Positive Rate (Sensitivity)",
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

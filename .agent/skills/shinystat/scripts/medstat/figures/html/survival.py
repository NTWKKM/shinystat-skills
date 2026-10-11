"""
src/medstat/figures/html/survival.py: Dual-view (Plotly + 300 DPI Matplotlib) Kaplan-Meier generator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from medstat.figures.html.base import DualFigureResult
from medstat.figures.survival import plot_kaplan_meier

COLORS = ["#0284C7", "#D97706", "#059669", "#7C3AED", "#DC2626", "#475569"]


def plot_dual_kaplan_meier(
    km_curves: dict[str, dict[str, Any]],
    risk_table: pd.DataFrame | None = None,
    log_rank_p: float | None = None,
    time_unit: str = "Days",
    title: str = "Kaplan-Meier Overall Survival",
    out_path: str | Path = "figures/kaplan_meier.png",
    target: Literal["docx", "pptx", "web"] = "docx",
) -> DualFigureResult:
    """
    Renders both publication-grade 300 DPI Matplotlib figure (with risk table)
    and interactive Plotly step survival curves with confidence intervals.
    """
    res = plot_kaplan_meier(
        km_curves=km_curves,
        risk_table=risk_table,
        log_rank_p=log_rank_p,
        time_unit=time_unit,
        title=title,
        out_path=out_path,
        target=target,
    )

    fig = go.Figure()

    for idx, (grp_name, data) in enumerate(km_curves.items()):
        color = COLORS[idx % len(COLORS)]
        t = np.asarray(data["timeline"], dtype=float)
        s = np.asarray(data["survival"], dtype=float)
        ci_l = np.asarray(data.get("ci_lower", s), dtype=float)
        ci_u = np.asarray(data.get("ci_upper", s), dtype=float)

        # Upper CI bound (invisible line for fill)
        fig.add_trace(
            go.Scatter(
                x=t,
                y=ci_u,
                mode="lines",
                line=dict(width=0, shape="hv"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Lower CI bound (shaded region)
        fig.add_trace(
            go.Scatter(
                x=t,
                y=ci_l,
                mode="lines",
                line=dict(width=0, shape="hv"),
                fill="tonexty",
                fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.15)",
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Main step survival curve
        fig.add_trace(
            go.Scatter(
                x=t,
                y=s,
                mode="lines",
                line=dict(color=color, width=2.5, shape="hv"),
                name=str(grp_name),
                hovertemplate=(
                    f"<b>{grp_name}</b><br>"
                    f"Time: %{{x:.1f}} {time_unit}<br>"
                    "Survival: %{y:.1%}<extra></extra>"
                ),
            )
        )

    title_text = title
    if log_rank_p is not None:
        p_str = "< 0.001" if log_rank_p < 0.001 else f"= {log_rank_p:.3f}"
        title_text += f" (Log-rank p {p_str})"

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=15, color="#0F172A")),
        xaxis=dict(
            title=f"Time ({time_unit})",
            showgrid=True,
            gridcolor="#F1F5F9",
        ),
        yaxis=dict(
            title="Survival Probability",
            range=[0, 1.05],
            showgrid=True,
            gridcolor="#F1F5F9",
            tickformat=".0%",
        ),
        template="plotly_white",
        legend=dict(
            yanchor="bottom",
            y=0.02,
            xanchor="left",
            x=0.02,
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

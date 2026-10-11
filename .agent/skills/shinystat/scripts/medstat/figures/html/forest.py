"""
src/medstat/figures/html/forest.py: Dual-view (Plotly + 300 DPI Matplotlib) Forest plot generator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from medstat.figures.forest import plot_forest
from medstat.figures.html.base import DualFigureResult


def plot_dual_forest(
    data: list[dict[str, Any]] | pd.DataFrame,
    scale: Literal["OR", "HR", "RR", "Beta"] = "OR",
    title: str = "Forest Plot of Effect Estimates",
    out_path: str | Path = "figures/forest_plot.png",
    target: Literal["docx", "pptx", "web"] = "docx",
) -> DualFigureResult:
    """
    Renders both publication-grade 300 DPI Matplotlib figure and interactive Plotly forest plot.
    """
    res = plot_forest(
        data=data,
        scale=scale,
        title=title,
        out_path=out_path,
        target=target,
    )

    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data.copy()

    if "term" not in df.columns:
        for alias in ["study", "Variable", "covariate", "label", "name"]:
            if alias in df.columns:
                df["term"] = df[alias]
                break

    terms = df["term"].tolist()
    estimates = df["estimate"].to_numpy(dtype=float)
    ci_lows = df["ci_lower"].to_numpy(dtype=float)
    ci_highs = df["ci_upper"].to_numpy(dtype=float)

    is_log = scale in ["OR", "HR", "RR"]
    null_val = 1.0 if is_log else 0.0

    y_pos = list(range(len(terms) - 1, -1, -1))

    fig = go.Figure()

    # Null reference line
    fig.add_vline(
        x=null_val,
        line_dash="dash",
        line_color="#94A3B8",
        line_width=1.5,
    )

    # Error bars & markers
    err_plus = np.maximum(0, ci_highs - estimates)
    err_minus = np.maximum(0, estimates - ci_lows)

    fig.add_trace(
        go.Scatter(
            x=estimates,
            y=y_pos,
            mode="markers",
            marker=dict(symbol="square", size=9, color="#0284C7"),
            error_x=dict(
                type="data",
                symmetric=False,
                array=err_plus,
                arrayminus=err_minus,
                color="#0284C7",
                width=2,
            ),
            text=terms,
            hovertemplate=(
                "<b>%{text}</b><br>"
                + f"{scale}: %{{x:.2f}}<br>"
                + "95% CI: [%{customdata[0]:.2f}, %{customdata[1]:.2f}]<extra></extra>"
            ),
            customdata=np.column_stack([ci_lows, ci_highs]),
            showlegend=False,
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#0F172A")),
        xaxis=dict(
            title=f"{scale} (95% Confidence Interval)",
            type="log" if is_log else "linear",
            showgrid=True,
            gridcolor="#F1F5F9",
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=y_pos,
            ticktext=terms,
            autorange=True,
        ),
        template="plotly_white",
        margin=dict(l=120, r=40, t=50, b=50),
        height=max(400, len(terms) * 35 + 120),
    )

    return DualFigureResult(
        png_path=res.png_path,
        alt_text=res.alt_text,
        caption=res.caption,
        source_df=res.source_df,
        csv_path=res.csv_path,
        plotly_fig=fig,
    )

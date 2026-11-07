"""
src/medstat/figures/html/dca.py: Dual-view (Plotly + 300 DPI Matplotlib) Decision Curve Analysis generator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from medstat.diagnostic.dca import create_dca_plot
from medstat.figures.dca import plot_dca
from medstat.figures.html.base import DualFigureResult


def plot_dual_dca(
    dca_data: pd.DataFrame | None = None,
    dca_df: pd.DataFrame | None = None,
    title: str = "Decision Curve Analysis (Clinical Net Benefit)",
    out_path: str | Path = "figures/dca_curve.png",
    target: Literal["docx", "pptx", "web"] = "docx",
) -> DualFigureResult:
    """
    Renders both publication-grade 300 DPI Matplotlib figure and interactive Plotly DCA curves.
    """
    df = dca_data if dca_data is not None else dca_df
    if df is None:
        raise ValueError("dca_data or dca_df must be provided")

    res = plot_dca(
        dca_data=df,
        title=title,
        out_path=out_path,
        target=target,
    )

    fig = create_dca_plot(df)
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#0F172A")),
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

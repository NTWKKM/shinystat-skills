"""
src/medstat/figures/html/balance.py: Dual-view (Plotly + 300 DPI Matplotlib) Love Plot generator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from medstat.causal.balance import create_love_plot
from medstat.figures.balance import plot_love
from medstat.figures.html.base import DualFigureResult


def plot_dual_love(
    love_data: dict[str, Any] | pd.DataFrame,
    threshold: float = 0.10,
    title: str = "Covariate Balance Before and After Propensity Score Matching",
    out_path: str | Path = "figures/love_plot.png",
    target: Literal["docx", "pptx", "web"] = "docx",
) -> DualFigureResult:
    """
    Renders both publication-grade 300 DPI Matplotlib figure and interactive Plotly Love plot.
    """
    res = plot_love(
        love_data=love_data,
        threshold=threshold,
        title=title,
        out_path=out_path,
        target=target,
    )

    if isinstance(love_data, dict):
        covariates = love_data["covariates"]
        pre_smd = np.asarray(love_data["smd_raw"], dtype=float)
        post_smd = np.asarray(love_data["smd_matched"], dtype=float)
    else:
        covariates = love_data["Variable"].tolist()
        pre_smd = np.asarray(love_data["Pre_Match_SMD"], dtype=float)
        post_smd = np.asarray(love_data["Post_Match_SMD"], dtype=float)

    comp_df = pd.DataFrame(
        {
            "Covariate": covariates,
            "Pre_SMD": pre_smd,
            "Post_SMD": post_smd,
            "Pre_Abs_SMD": np.abs(pre_smd),
            "Post_Abs_SMD": np.abs(post_smd),
            "Balanced_Post": np.abs(post_smd) < threshold,
        }
    )

    try:
        fig = create_love_plot(comp_df, threshold=threshold)
    except TypeError:
        fig = create_love_plot(comp_df)
        for shape in fig.layout.shapes:
            if getattr(shape, "type", None) == "line" and shape.x0 == 0.10:
                shape.x0 = threshold
                shape.x1 = threshold
        for ann in fig.layout.annotations:
            if "Threshold" in getattr(ann, "text", ""):
                ann.x = threshold
                ann.text = f"Threshold ({threshold:.2f})"
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#0F172A")),
        margin=dict(l=120, r=40, t=50, b=50),
        height=max(400, len(covariates) * 35 + 120),
    )

    return DualFigureResult(
        png_path=res.png_path,
        alt_text=res.alt_text,
        caption=res.caption,
        source_df=res.source_df,
        csv_path=res.csv_path,
        plotly_fig=fig,
    )

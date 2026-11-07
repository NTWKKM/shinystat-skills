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
        covariates = (
            love_data.get("covariates")
            or love_data.get("Variable")
            or love_data.get("Covariate")
        )
        pre_smd = np.asarray(
            love_data.get("smd_raw")
            if "smd_raw" in love_data
            else love_data.get("Pre_Match_SMD", love_data.get("Unadjusted_SMD")),
            dtype=float,
        )
        has_post = (
            "smd_matched" in love_data
            or "Post_Match_SMD" in love_data
            or "Matched_SMD" in love_data
        )
        post_smd = (
            np.asarray(
                love_data.get("smd_matched")
                if "smd_matched" in love_data
                else love_data.get("Post_Match_SMD", love_data.get("Matched_SMD")),
                dtype=float,
            )
            if has_post
            else None
        )
        has_weighted = (
            "smd_weighted" in love_data
            or "Weighted_SMD" in love_data
            or "Post_Weight_SMD" in love_data
        )
        weighted_smd = (
            np.asarray(
                love_data.get("smd_weighted")
                if "smd_weighted" in love_data
                else love_data.get("Weighted_SMD", love_data.get("Post_Weight_SMD")),
                dtype=float,
            )
            if has_weighted
            else None
        )
    else:
        var_col = (
            "Variable"
            if "Variable" in love_data.columns
            else (
                "Covariate"
                if "Covariate" in love_data.columns
                else love_data.columns[0]
            )
        )
        covariates = love_data[var_col].tolist()
        pre_col = next(
            (
                c
                for c in ["Pre_Match_SMD", "Unadjusted_SMD", "Pre_SMD", "smd_raw"]
                if c in love_data.columns
            ),
            None,
        )
        pre_smd = (
            np.asarray(love_data[pre_col], dtype=float)
            if pre_col
            else np.zeros(len(covariates))
        )

        post_col = next(
            (
                c
                for c in ["Post_Match_SMD", "Matched_SMD", "Post_SMD", "smd_matched"]
                if c in love_data.columns
            ),
            None,
        )
        post_smd = np.asarray(love_data[post_col], dtype=float) if post_col else None

        weight_col = next(
            (
                c
                for c in [
                    "Weighted_SMD",
                    "Post_Weight_SMD",
                    "Weighted_Abs_SMD",
                    "smd_weighted",
                ]
                if c in love_data.columns
            ),
            None,
        )
        weighted_smd = (
            np.asarray(love_data[weight_col], dtype=float) if weight_col else None
        )

    data_dict: dict[str, Any] = {
        "Covariate": covariates,
        "Pre_SMD": pre_smd,
        "Pre_Abs_SMD": np.abs(pre_smd),
    }
    if post_smd is not None:
        data_dict["Post_SMD"] = post_smd
        data_dict["Post_Abs_SMD"] = np.abs(post_smd)
        data_dict["Balanced_Post"] = np.abs(post_smd) < threshold
    if weighted_smd is not None:
        data_dict["Weighted_SMD"] = weighted_smd
        data_dict["Weighted_Abs_SMD"] = np.abs(weighted_smd)
        data_dict["Balanced_Weighted"] = np.abs(weighted_smd) < threshold

    comp_df = pd.DataFrame(data_dict)

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

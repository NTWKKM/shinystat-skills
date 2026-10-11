"""
src/medstat/figures/html/agreement.py: Dual-view (Plotly + 300 DPI Matplotlib) Bland-Altman generator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from medstat.agreement.bland_altman import create_bland_altman_plot
from medstat.figures.agreement import plot_bland_altman
from medstat.figures.html.base import DualFigureResult


def plot_dual_bland_altman(
    ba_data: dict[str, Any],
    units: str = "mmHg",
    title: str = "Bland-Altman Limits of Agreement",
    out_path: str | Path = "figures/bland_altman.png",
    target: Literal["docx", "pptx", "web"] = "docx",
) -> DualFigureResult:
    """
    Renders both publication-grade 300 DPI Matplotlib figure and interactive Plotly Bland-Altman plot.
    """
    res = plot_bland_altman(
        ba_data=ba_data,
        units=units,
        title=title,
        out_path=out_path,
        target=target,
    )

    # Ensure required CI keys exist for create_bland_altman_plot
    plot_dict = dict(ba_data)
    mean_diff = float(ba_data["mean_diff"])
    upper_loa = float(ba_data["loa_upper"])
    lower_loa = float(ba_data["loa_lower"])

    if "ci_mean_diff" not in plot_dict:
        plot_dict["ci_mean_diff"] = [mean_diff, mean_diff]
    if "ci_upper_loa" not in plot_dict:
        plot_dict["ci_upper_loa"] = ba_data.get("ci_loa_upper", [upper_loa, upper_loa])
    if "ci_lower_loa" not in plot_dict:
        plot_dict["ci_lower_loa"] = ba_data.get("ci_loa_lower", [lower_loa, lower_loa])
    if "upper_loa" not in plot_dict:
        plot_dict["upper_loa"] = upper_loa
    if "lower_loa" not in plot_dict:
        plot_dict["lower_loa"] = lower_loa

    m1_name = ba_data.get("method1", "Method 1")
    m2_name = ba_data.get("method2", "Method 2")

    fig = create_bland_altman_plot(
        stats_res=plot_dict,
        method1=m1_name,
        method2=m2_name,
        show_ci_bands=True,
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#0F172A")),
        xaxis=dict(title=f"Mean of {m1_name} & {m2_name} ({units})"),
        yaxis=dict(title=f"Difference ({m1_name} - {m2_name}) ({units})"),
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

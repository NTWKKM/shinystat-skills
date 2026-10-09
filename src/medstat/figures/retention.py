"""
src/medstat/figures/retention.py: STROBE / CONSORT Participant Retention Flow Diagram generator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import pandas as pd

from medstat.figures.base import FigureResult, export_source_data
from medstat.figures.styles import (
    CLINICAL_PALETTE,
    save_publication_figure,
    set_clinical_figure_style,
)


def plot_retention_flow(
    retention_data: dict[str, Any],
    title: str = "Participant Retention Flow Diagram (STROBE / CONSORT)",
    out_path: str | Path = "figures/retention_flow.png",
    target: Literal["docx", "pptx", "web"] = "docx",
) -> FigureResult:
    """
    Renders pure-Matplotlib STROBE/CONSORT flowchart showing cohort attrition:
    Initial Cohort -> Excluded (with reasons) -> Final Analyzed Cohort.
    """
    w, h, dpi = set_clinical_figure_style(target)
    fig, ax = plt.subplots(figsize=(w, max(h, 4.8)))

    n_init = int(retention_data.get("n_initial", 0))
    n_excl = int(retention_data.get("n_excluded", 0))
    n_analyzed = int(retention_data.get("n_analyzed", n_init - n_excl))
    reasons = retention_data.get("exclusion_reasons", [])

    if isinstance(reasons, str):
        reasons_list = [reasons]
    elif isinstance(reasons, list):
        reasons_list = [
            f"• {r}" if not str(r).startswith("•") else str(r) for r in reasons
        ]
    else:
        reasons_list = ["Criteria not met"]

    reasons_str = "\n".join(reasons_list)

    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Box styles
    box_blue = dict(
        boxstyle="square,pad=0.8", facecolor="#EFF6FF", edgecolor="#2563EB", lw=1.5
    )
    box_red = dict(
        boxstyle="square,pad=0.8", facecolor="#FEF2F2", edgecolor="#DC2626", lw=1.5
    )
    box_green = dict(
        boxstyle="square,pad=0.8", facecolor="#ECFDF5", edgecolor="#059669", lw=1.5
    )

    # 1. Initial Box (Top center)
    ax.text(
        3.5,
        8.5,
        f"Assessed for Eligibility\n(N = {n_init:,})",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="600",
        bbox=box_blue,
    )

    # 2. Excluded Box (Right side)
    excl_text = f"Excluded (N = {n_excl:,})\n{reasons_str}"
    ax.text(
        7.8,
        5.5,
        excl_text,
        ha="center",
        va="center",
        fontsize=9.5,
        bbox=box_red,
    )

    # 3. Final Analyzed Box (Bottom center)
    ax.text(
        3.5,
        2.0,
        f"Included in Final Analysis\n(N = {n_analyzed:,})",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="600",
        bbox=box_green,
    )

    # Connecting arrows
    # Vertical line from top box down to middle
    ax.annotate(
        "",
        xy=(3.5, 5.5),
        xytext=(3.5, 7.3),
        arrowprops=dict(arrowstyle="-", color=CLINICAL_PALETTE["primary"], lw=1.5),
    )
    # Horizontal arrow to excluded box
    ax.annotate(
        "",
        xy=(6.0, 5.5),
        xytext=(3.5, 5.5),
        arrowprops=dict(arrowstyle="->", color=CLINICAL_PALETTE["danger"], lw=1.5),
    )
    # Vertical arrow from middle down to analyzed box
    ax.annotate(
        "",
        xy=(3.5, 3.2),
        xytext=(3.5, 5.5),
        arrowprops=dict(arrowstyle="->", color=CLINICAL_PALETTE["accent"], lw=1.5),
    )

    ax.set_title(title, loc="left", pad=12, fontsize=12, fontweight="bold")
    fig.tight_layout()

    source_df = pd.DataFrame(
        [
            {
                "Stage": "Assessed for Eligibility",
                "Count": n_init,
                "Details": "Initial screening",
            },
            {"Stage": "Excluded", "Count": n_excl, "Details": reasons_str},
            {
                "Stage": "Final Analyzed Cohort",
                "Count": n_analyzed,
                "Details": "Complete-case / primary analytic dataset",
            },
        ]
    )
    png_path = save_publication_figure(fig, out_path, dpi=dpi)
    csv_path = export_source_data(source_df, png_path)

    alt_text = f"Flow diagram showing cohort attrition from {n_init:,} assessed down to {n_analyzed:,} analyzed."
    caption = f"Figure. {title}. Sample attrition flow documenting exclusions and final analytic cohort size in accordance with ICMJE/STROBE standards."

    return FigureResult(
        png_path=png_path,
        alt_text=alt_text,
        caption=caption,
        source_df=source_df,
        csv_path=csv_path,
    )

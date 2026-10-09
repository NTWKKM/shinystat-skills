"""
src/medstat/figures/forest.py: Publication-grade Forest Plot generator (OR / HR).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from medstat.figures.base import FigureResult, export_source_data
from medstat.figures.styles import (
    CLINICAL_PALETTE,
    save_publication_figure,
    set_clinical_figure_style,
)


def plot_forest(
    data: list[dict[str, Any]] | pd.DataFrame,
    scale: Literal["OR", "HR", "RR", "Beta"] = "OR",
    title: str = "Forest Plot of Effect Estimates",
    out_path: str | Path = "figures/forest_plot.png",
    target: Literal["docx", "pptx", "web"] = "docx",
) -> FigureResult:
    """
    Renders an ICMJE publication-standard Forest Plot with log-scale axis,
    95% CI error bars, and aligned right-hand numerical estimates.
    """
    w, h, dpi = set_clinical_figure_style(target)

    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data.copy()

    # Map column aliases to 'term'
    if "term" not in df.columns:
        for alias in ["study", "Variable", "covariate", "label", "name"]:
            if alias in df.columns:
                df["term"] = df[alias]
                break

    required_cols = {"term", "estimate", "ci_lower", "ci_upper"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Forest plot data missing required columns: {missing}")

    # Reverse order so top row appears at top of plot
    df = df.iloc[::-1].reset_index(drop=True)
    n_terms = len(df)

    # Adjust height based on number of terms
    dynamic_height = max(h, 0.45 * n_terms + 2.0)
    fig, (ax_plot, ax_text) = plt.subplots(
        1, 2, figsize=(w, dynamic_height), gridspec_kw={"width_ratios": [3.2, 2.0]}
    )

    y_positions = np.arange(n_terms)
    estimates = df["estimate"].to_numpy(dtype=float)
    ci_lowers = df["ci_lower"].to_numpy(dtype=float)
    ci_uppers = df["ci_upper"].to_numpy(dtype=float)
    terms = df["term"].astype(str).tolist()

    # Determine reference line value
    ref_val = 0.0 if scale == "Beta" else 1.0

    # Plot confidence interval lines and point markers
    for idx, y in enumerate(y_positions):
        est = estimates[idx]
        lo = ci_lowers[idx]
        hi = ci_uppers[idx]

        # Horizontal error bar
        ax_plot.plot(
            [lo, hi],
            [y, y],
            color=CLINICAL_PALETTE["primary"],
            lw=1.8,
            solid_capstyle="round",
        )
        # Estimate marker
        ax_plot.plot(
            est,
            y,
            marker="s",
            markersize=6.5,
            color=CLINICAL_PALETTE["secondary"],
            zorder=5,
        )

    # Vertical reference line of no effect
    ax_plot.axvline(
        ref_val,
        color=CLINICAL_PALETTE["danger"],
        ls="--",
        lw=1.2,
        alpha=0.85,
        label=f"Null effect ({ref_val})",
    )

    # X-axis configuration
    if scale in ("OR", "HR", "RR"):
        ax_plot.set_xscale("log")
        ax_plot.set_xlabel(f"{scale} (log scale)", labelpad=6)
    else:
        ax_plot.set_xlabel(f"{scale} Effect Size", labelpad=6)

    ax_plot.set_yticks(y_positions)
    ax_plot.set_yticklabels(terms, fontweight="500")
    ax_plot.set_ylim(-0.8, n_terms - 0.2)
    ax_plot.set_title(title, loc="left", pad=12)

    # Right text column: numerical summary
    ax_text.axis("off")
    ax_text.set_ylim(-0.8, n_terms - 0.2)

    # Header in text pane
    header_str = f"{scale} (95% CI)"
    ax_text.text(
        0.05,
        n_terms - 0.2,
        header_str,
        fontweight="bold",
        fontsize=10,
        va="bottom",
        ha="left",
    )
    if "p_value" in df.columns:
        ax_text.text(
            0.85,
            n_terms - 0.2,
            "P-value",
            fontweight="bold",
            fontsize=10,
            va="bottom",
            ha="right",
        )

    for idx, y in enumerate(y_positions):
        est_str = f"{estimates[idx]:.2f} [{ci_lowers[idx]:.2f}, {ci_uppers[idx]:.2f}]"
        ax_text.text(
            0.05,
            y,
            est_str,
            va="center",
            ha="left",
            fontsize=9.5,
            fontfamily="monospace",
        )
        if "p_value" in df.columns:
            pval = df.loc[idx, "p_value"]
            p_str = (
                f"{pval:.3f}"
                if (isinstance(pval, (int, float)) and pval >= 0.001)
                else ("<0.001" if isinstance(pval, (int, float)) else str(pval))
            )
            ax_text.text(0.85, y, p_str, va="center", ha="right", fontsize=9.5)

    fig.tight_layout()

    png_path = save_publication_figure(fig, out_path, dpi=dpi)
    csv_path = export_source_data(df.iloc[::-1].reset_index(drop=True), png_path)

    alt_text = f"Forest plot displaying {n_terms} effect estimates ({scale}) with 95% confidence intervals."
    caption = f"Figure. {title}. Squares represent point estimates ({scale}) with horizontal lines indicating 95% confidence intervals. The dashed vertical line denotes null effect ({ref_val})."

    return FigureResult(
        png_path=png_path,
        alt_text=alt_text,
        caption=caption,
        source_df=df,
        csv_path=csv_path,
    )

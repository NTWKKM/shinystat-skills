"""
src/medstat/figures/agreement.py: Publication-grade Bland-Altman agreement plot generator.
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


def plot_bland_altman(
    ba_data: dict[str, Any],
    units: str = "mmHg",
    title: str = "Bland-Altman Limits of Agreement",
    out_path: str | Path = "figures/bland_altman.png",
    target: Literal["docx", "pptx", "web"] = "docx",
) -> FigureResult:
    """
    Renders Bland-Altman difference plot with mean bias, 95% Limits of Agreement,
    and Bland-Altman (1999) large-sample confidence interval bands.
    """
    w, h, dpi = set_clinical_figure_style(target)
    fig, ax = plt.subplots(figsize=(w, h))

    means = np.asarray(ba_data["means"], dtype=float)
    diffs = np.asarray(ba_data["diffs"], dtype=float)
    mean_diff = float(ba_data["mean_diff"])
    upper_loa = float(ba_data["loa_upper"])
    lower_loa = float(ba_data["loa_lower"])

    ci_mean = ba_data.get("ci_mean_diff", (mean_diff, mean_diff))
    ci_upper = ba_data.get("ci_loa_upper", (upper_loa, upper_loa))
    ci_lower = ba_data.get("ci_loa_lower", (lower_loa, lower_loa))

    # Scatter points
    ax.scatter(
        means,
        diffs,
        alpha=0.55,
        color=CLINICAL_PALETTE["secondary"],
        edgecolors="none",
        s=28,
        zorder=3,
    )

    x_min, x_max = float(np.min(means)), float(np.max(means))
    x_pad = (x_max - x_min) * 0.05
    x_range = [x_min - x_pad, x_max + x_pad]

    # Mean difference (bias)
    ax.axhline(
        mean_diff,
        color=CLINICAL_PALETTE["primary"],
        lw=1.8,
        label=f"Mean Bias ({mean_diff:+.2f})",
    )
    ax.axhspan(
        ci_mean[0],
        ci_mean[1],
        color=CLINICAL_PALETTE["primary"],
        alpha=0.12,
        label="Mean Bias 95% CI",
    )

    # Upper LoA
    ax.axhline(
        upper_loa,
        color=CLINICAL_PALETTE["danger"],
        ls="--",
        lw=1.5,
        label=f"+1.96 SD ({upper_loa:+.2f})",
    )
    ax.axhspan(ci_upper[0], ci_upper[1], color=CLINICAL_PALETTE["danger"], alpha=0.10)

    # Lower LoA
    ax.axhline(
        lower_loa,
        color=CLINICAL_PALETTE["danger"],
        ls="--",
        lw=1.5,
        label=f"–1.96 SD ({lower_loa:+.2f})",
    )
    ax.axhspan(ci_lower[0], ci_lower[1], color=CLINICAL_PALETTE["danger"], alpha=0.10)

    # Zero difference reference line
    ax.axhline(0.0, color=CLINICAL_PALETTE["neutral"], ls=":", lw=1.0)

    # Annotation callouts on right edge
    right_x = x_max + x_pad * 0.2
    ax.text(
        right_x,
        upper_loa,
        f"+1.96 SD: {upper_loa:+.1f}",
        va="center",
        ha="left",
        fontsize=9,
        color=CLINICAL_PALETTE["danger"],
        fontweight="600",
    )
    ax.text(
        right_x,
        mean_diff,
        f"Mean: {mean_diff:+.1f}",
        va="center",
        ha="left",
        fontsize=9,
        color=CLINICAL_PALETTE["primary"],
        fontweight="600",
    )
    ax.text(
        right_x,
        lower_loa,
        f"–1.96 SD: {lower_loa:+.1f}",
        va="center",
        ha="left",
        fontsize=9,
        color=CLINICAL_PALETTE["danger"],
        fontweight="600",
    )

    unit_str = f" ({units})" if units else ""
    ax.set_xlabel(f"Mean of Paired Measurements{unit_str}", labelpad=6)
    ax.set_ylabel(f"Difference (Method 1 – Method 2){unit_str}", labelpad=6)
    ax.set_xlim(x_range[0], x_range[1] + x_pad * 2.0)
    ax.set_title(title, loc="left", pad=10)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))

    fig.tight_layout()

    source_df = pd.DataFrame({"Mean": means, "Difference": diffs})
    png_path = save_publication_figure(fig, out_path, dpi=dpi)
    csv_path = export_source_data(source_df, png_path)

    alt_text = f"Bland-Altman plot indicating mean bias of {mean_diff:+.2f} with limits of agreement."
    caption = f"Figure. {title}. Differences between paired measurements plotted against mean values. Solid line indicates mean bias with shaded 95% confidence interval; dashed lines indicate 95% Limits of Agreement (mean ± 1.96 SD) with large-sample confidence bounds."

    return FigureResult(
        png_path=png_path,
        alt_text=alt_text,
        caption=caption,
        source_df=source_df,
        csv_path=csv_path,
    )

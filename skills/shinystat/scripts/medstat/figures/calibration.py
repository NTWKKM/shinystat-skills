"""
src/medstat/figures/calibration.py: Publication-grade Model Calibration Curve generator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from medstat.figures.base import FigureResult, export_source_data
from medstat.figures.styles import (
    CLINICAL_PALETTE,
    save_publication_figure,
    set_clinical_figure_style,
)


def plot_calibration(
    calib_bins: dict[str, np.ndarray],
    calib_curve: dict[str, np.ndarray] | None = None,
    brier_score: float | None = None,
    ici: float | None = None,
    title: str = "Model Calibration Curve",
    out_path: str | Path = "figures/calibration_plot.png",
    target: Literal["docx", "pptx", "web"] = "docx",
) -> FigureResult:
    """
    Renders decile calibration points with 95% Wilson confidence intervals,
    45-degree ideal diagonal, smoothed calibration trajectory, and goodness metrics.
    """
    w, h, dpi = set_clinical_figure_style(target)
    fig, ax = plt.subplots(figsize=(w, h))

    p_means = np.asarray(calib_bins["pred_mean"], dtype=float)
    o_rates = np.asarray(calib_bins["obs_rate"], dtype=float)
    ci_low = np.asarray(calib_bins.get("ci_lower", o_rates), dtype=float)
    ci_high = np.asarray(calib_bins.get("ci_upper", o_rates), dtype=float)
    counts = np.asarray(calib_bins.get("counts", np.zeros_like(p_means)), dtype=int)

    # 45-degree reference line
    ax.plot(
        [0, 1],
        [0, 1],
        color=CLINICAL_PALETTE["neutral"],
        ls="--",
        lw=1.2,
        label="Perfect Calibration",
    )

    # Smoothed calibration trajectory
    if calib_curve is not None:
        p_smooth = np.asarray(calib_curve["pred_smooth"], dtype=float)
        o_smooth = np.asarray(calib_curve["obs_smooth"], dtype=float)
        ax.plot(
            p_smooth,
            o_smooth,
            color=CLINICAL_PALETTE["secondary"],
            lw=2.0,
            label="Loess Recalibration",
        )

    # Decile bin points with error bars
    valid = ~(np.isnan(p_means) | np.isnan(o_rates))
    yerr_lower = np.maximum(o_rates[valid] - ci_low[valid], 0.0)
    yerr_upper = np.maximum(ci_high[valid] - o_rates[valid], 0.0)

    ax.errorbar(
        p_means[valid],
        o_rates[valid],
        yerr=[yerr_lower, yerr_upper],
        fmt="o",
        color=CLINICAL_PALETTE["primary"],
        ecolor=CLINICAL_PALETTE["neutral"],
        elinewidth=1.2,
        capsize=3.0,
        markersize=6.0,
        zorder=5,
        label="Decile Observed (95% CI)",
    )

    # Metrics callout box
    metrics_lines = []
    if brier_score is not None:
        metrics_lines.append(f"Brier Score: {brier_score:.3f}")
    if ici is not None:
        metrics_lines.append(f"ICI: {ici:.3f}")

    if metrics_lines:
        ax.text(
            0.05,
            0.82,
            "\n".join(metrics_lines),
            transform=ax.transAxes,
            fontsize=9.5,
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="#F8FAFC",
                edgecolor="#CBD5E1",
                alpha=0.9,
            ),
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xlabel("Predicted Probability", labelpad=6)
    ax.set_ylabel("Observed Proportion", labelpad=6)
    ax.set_title(title, loc="left", pad=10)
    ax.legend(loc="lower right")

    fig.tight_layout()

    source_df = pd.DataFrame(
        {
            "Bin_Predicted_Mean": p_means,
            "Bin_Observed_Rate": o_rates,
            "Bin_CI_Lower": ci_low,
            "Bin_CI_Upper": ci_high,
            "Bin_Count": counts,
        }
    )
    png_path = save_publication_figure(fig, out_path, dpi=dpi)
    csv_path = export_source_data(source_df, png_path)

    alt_text = "Model calibration curve comparing predicted risk deciles against observed event rates."
    caption = f"Figure. {title}. Decile points denote mean predicted versus observed proportions with 95% Wilson confidence intervals. The dashed diagonal represents perfect agreement."

    return FigureResult(
        png_path=png_path,
        alt_text=alt_text,
        caption=caption,
        source_df=source_df,
        csv_path=csv_path,
    )

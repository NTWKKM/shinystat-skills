"""
src/medstat/figures/dca.py: Vickers Decision Curve Analysis (DCA) plot generator.
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


def plot_dca(
    dca_data: pd.DataFrame,
    title: str = "Decision Curve Analysis (Clinical Net Benefit)",
    out_path: str | Path = "figures/dca_plot.png",
    target: Literal["docx", "pptx", "web"] = "docx",
) -> FigureResult:
    """
    Renders Vickers Decision Curve Analysis evaluating net clinical benefit
    across a range of decision threshold probabilities (pt).
    """
    w, h, dpi = set_clinical_figure_style(target)
    fig, ax = plt.subplots(figsize=(w, h))

    df = dca_data.copy()
    strategies = df["strategy"].unique()

    # Colors and styles
    style_map = {
        "Model": {"color": CLINICAL_PALETTE["secondary"], "lw": 2.2, "ls": "-"},
        "Treat All": {"color": CLINICAL_PALETTE["neutral"], "lw": 1.5, "ls": "--"},
        "Treat None": {"color": CLINICAL_PALETTE["primary"], "lw": 1.2, "ls": ":"},
    }

    max_nb = 0.05
    min_nb = -0.05
    for strat in strategies:
        sub = df[df["strategy"] == strat].sort_values("threshold")
        cfg = style_map.get(
            strat, {"color": CLINICAL_PALETTE["accent"], "lw": 1.5, "ls": "-"}
        )
        ax.plot(sub["threshold"], sub["net_benefit"], label=strat, **cfg)
        valid_strat_nb = sub["net_benefit"].dropna()
        if not valid_strat_nb.empty:
            strat_max = float(valid_strat_nb.max())
            if np.isfinite(strat_max):
                max_nb = max(max_nb, strat_max)
            strat_min = float(valid_strat_nb.min())
            if np.isfinite(strat_min):
                min_nb = min(min_nb, strat_min)

    # Treat None horizontal reference at 0
    ax.axhline(0.0, color=CLINICAL_PALETTE["primary"], ls=":", lw=0.8, alpha=0.5)

    valid_thresh = df["threshold"].dropna()
    t_min = float(valid_thresh.min()) if not valid_thresh.empty else 0.0
    t_max = float(valid_thresh.max()) if not valid_thresh.empty else 1.0
    x_min = t_min if np.isfinite(t_min) else 0.0
    x_max = t_max if np.isfinite(t_max) else 1.0
    if x_min >= x_max:
        x_min, x_max = 0.0, 1.0

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(min_nb, max_nb * 1.15)
    ax.set_xlabel("Threshold Probability (pt)", labelpad=6)
    ax.set_ylabel("Net Benefit", labelpad=6)
    ax.set_title(title, loc="left", pad=10)
    ax.legend(loc="upper right")

    fig.tight_layout()

    png_path = save_publication_figure(fig, out_path, dpi=dpi)
    csv_path = export_source_data(df, png_path)

    alt_text = "Decision curve analysis demonstrating net clinical benefit across threshold probabilities."
    caption = f"Figure. {title}. Net benefit curves for model-guided intervention versus default strategies ('Treat All' and 'Treat None') across clinical threshold probabilities."

    return FigureResult(
        png_path=png_path,
        alt_text=alt_text,
        caption=caption,
        source_df=df,
        csv_path=csv_path,
    )

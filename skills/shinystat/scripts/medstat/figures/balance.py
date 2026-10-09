"""
src/medstat/figures/balance.py: Austin (2009) Love Plot for Covariate Balance before and after PSM.
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


def plot_love(
    love_data: dict[str, Any] | pd.DataFrame,
    threshold: float = 0.10,
    title: str = "Covariate Balance Before and After Propensity Score Matching",
    out_path: str | Path = "figures/love_plot.png",
    target: Literal["docx", "pptx", "web"] = "docx",
) -> FigureResult:
    """
    Renders Austin (2009) Love Plot displaying Standardized Mean Differences
    before versus after propensity score matching relative to the 0.10 balance threshold.
    """
    w, h, dpi = set_clinical_figure_style(target)

    if isinstance(love_data, dict):
        df = pd.DataFrame(
            {
                "Variable": love_data["covariates"],
                "Pre_Match_SMD": love_data["smd_raw"],
                "Post_Match_SMD": love_data["smd_matched"],
            }
        )
    else:
        df = love_data.copy()

    # Invert so top variable appears at top of plot
    df = df.iloc[::-1].reset_index(drop=True)
    n_vars = len(df)

    dynamic_h = max(h, 0.45 * n_vars + 2.0)
    fig, ax = plt.subplots(figsize=(w, dynamic_h))

    y_pos = np.arange(n_vars)
    pre_smd = np.abs(df["Pre_Match_SMD"].to_numpy(dtype=float))
    post_smd = np.abs(df["Post_Match_SMD"].to_numpy(dtype=float))
    vars_list = df["Variable"].astype(str).tolist()

    # Balanced zone shading (SMD < 0.10)
    ax.axvspan(
        0.0,
        threshold,
        color=CLINICAL_PALETTE["accent"],
        alpha=0.10,
        label=f"Balanced Zone (<{threshold:.2f})",
    )
    ax.axvline(threshold, color=CLINICAL_PALETTE["accent"], ls="--", lw=1.2)

    # Connector lines between raw and matched
    for idx, y in enumerate(y_pos):
        if not (np.isnan(pre_smd[idx]) or np.isnan(post_smd[idx])):
            ax.plot(
                [pre_smd[idx], post_smd[idx]],
                [y, y],
                color=CLINICAL_PALETTE["light_neutral"],
                lw=1.5,
                zorder=2,
            )

    # Points
    ax.scatter(
        pre_smd,
        y_pos,
        color=CLINICAL_PALETTE["warning"],
        s=55,
        zorder=4,
        label="Raw Cohort",
        marker="o",
    )
    ax.scatter(
        post_smd,
        y_pos,
        color=CLINICAL_PALETTE["accent"],
        s=55,
        zorder=5,
        label="Matched Cohort",
        marker="D",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(vars_list, fontweight="500")
    ax.set_xlabel("Absolute Standardized Mean Difference (|SMD|)", labelpad=6)
    ax.set_xlim(
        0.0,
        max(
            0.40,
            float(
                np.nanmax(pre_smd) * 1.15
                if len(pre_smd) > 0 and not np.isnan(pre_smd).all()
                else 0.40
            ),
        ),
    )
    ax.set_ylim(-0.8, n_vars - 0.2)
    ax.set_title(title, loc="left", pad=12)
    ax.legend(loc="lower right")

    fig.tight_layout()

    source_df = df.iloc[::-1].reset_index(drop=True)
    png_path = save_publication_figure(fig, out_path, dpi=dpi)
    csv_path = export_source_data(source_df, png_path)

    alt_text = "Love plot showing covariate standardized mean differences before and after propensity score matching."
    caption = f"Figure. {title}. Absolute Standardized Mean Differences (|SMD|) for baseline covariates before (circles) and after (diamonds) 1:1 propensity score matching. Shaded region indicates negligible imbalance (|SMD| < {threshold:.2f})."

    return FigureResult(
        png_path=png_path,
        alt_text=alt_text,
        caption=caption,
        source_df=source_df,
        csv_path=csv_path,
    )

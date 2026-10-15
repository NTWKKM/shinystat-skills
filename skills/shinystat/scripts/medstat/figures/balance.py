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
        var_col = (
            love_data.get("covariates")
            or love_data.get("Variable")
            or love_data.get("Covariate")
        )
        data_dict: dict[str, Any] = {"Variable": var_col}
        if (
            "smd_raw" in love_data
            or "Pre_Match_SMD" in love_data
            or "Unadjusted_SMD" in love_data
        ):
            data_dict["Pre_Match_SMD"] = (
                love_data.get("smd_raw")
                if "smd_raw" in love_data
                else love_data.get("Pre_Match_SMD", love_data.get("Unadjusted_SMD"))
            )
        if (
            "smd_matched" in love_data
            or "Post_Match_SMD" in love_data
            or "Matched_SMD" in love_data
        ):
            data_dict["Post_Match_SMD"] = (
                love_data.get("smd_matched")
                if "smd_matched" in love_data
                else love_data.get("Post_Match_SMD", love_data.get("Matched_SMD"))
            )
        if (
            "smd_weighted" in love_data
            or "Weighted_SMD" in love_data
            or "Post_Weight_SMD" in love_data
        ):
            data_dict["Weighted_SMD"] = (
                love_data.get("smd_weighted")
                if "smd_weighted" in love_data
                else love_data.get("Weighted_SMD", love_data.get("Post_Weight_SMD"))
            )
        df = pd.DataFrame(data_dict)
    else:
        df = love_data.copy()

    # Map variable column
    if "Variable" not in df.columns:
        if "Covariate" in df.columns:
            df["Variable"] = df["Covariate"]
        else:
            df["Variable"] = [f"Var_{i + 1}" for i in range(len(df))]

    # Map Pre-match SMD
    if "Pre_Match_SMD" not in df.columns:
        for c in ["Unadjusted_SMD", "Pre_SMD", "smd_raw", "Unadjusted_Abs_SMD"]:
            if c in df.columns:
                df["Pre_Match_SMD"] = df[c]
                break

    # Map Matched SMD
    has_matched = False
    if "Post_Match_SMD" in df.columns:
        has_matched = True
    else:
        for c in ["Matched_SMD", "Post_SMD", "smd_matched", "Matched_Abs_SMD"]:
            if c in df.columns and df[c].notna().any():
                df["Post_Match_SMD"] = df[c]
                has_matched = True
                break

    # Map Weighted SMD
    has_weighted = False
    if "Weighted_SMD" in df.columns:
        has_weighted = True
    else:
        for c in ["Post_Weight_SMD", "Weighted_Abs_SMD", "smd_weighted"]:
            if c in df.columns and df[c].notna().any():
                df["Weighted_SMD"] = df[c]
                has_weighted = True
                break

    # Invert so top variable appears at top of plot
    df = df.iloc[::-1].reset_index(drop=True)
    n_vars = len(df)

    dynamic_h = max(h, 0.45 * n_vars + 2.0)
    fig, ax = plt.subplots(figsize=(w, dynamic_h))

    y_pos = np.arange(n_vars)
    pre_smd = (
        np.abs(df["Pre_Match_SMD"].to_numpy(dtype=float))
        if "Pre_Match_SMD" in df.columns
        else np.zeros(n_vars)
    )
    post_smd = (
        np.abs(df["Post_Match_SMD"].to_numpy(dtype=float)) if has_matched else None
    )
    weight_smd = (
        np.abs(df["Weighted_SMD"].to_numpy(dtype=float)) if has_weighted else None
    )
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

    # Connector lines
    for idx, y in enumerate(y_pos):
        if (
            has_matched
            and post_smd is not None
            and not (np.isnan(pre_smd[idx]) or np.isnan(post_smd[idx]))
        ):
            ax.plot(
                [pre_smd[idx], post_smd[idx]],
                [y, y],
                color=CLINICAL_PALETTE["light_neutral"],
                lw=1.5,
                zorder=2,
            )
        elif (
            has_weighted
            and weight_smd is not None
            and not (np.isnan(pre_smd[idx]) or np.isnan(weight_smd[idx]))
        ):
            ax.plot(
                [pre_smd[idx], weight_smd[idx]],
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
    if has_matched and post_smd is not None:
        ax.scatter(
            post_smd,
            y_pos,
            color=CLINICAL_PALETTE["accent"],
            s=55,
            zorder=5,
            label="Matched Cohort",
            marker="D",
        )
    if has_weighted and weight_smd is not None:
        ax.scatter(
            weight_smd,
            y_pos,
            color=CLINICAL_PALETTE["primary"],
            s=55,
            zorder=6,
            label="IPTW Weighted Cohort",
            marker="s",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(vars_list, fontweight="500")
    ax.set_xlabel("Absolute Standardized Mean Difference (|SMD|)", labelpad=6)
    all_smds = [pre_smd]
    if post_smd is not None:
        all_smds.append(post_smd)
    if weight_smd is not None:
        all_smds.append(weight_smd)
    concat_smd = np.concatenate(all_smds) if all_smds else np.array([0.4])
    valid_max = (
        np.nanmax(concat_smd)
        if len(concat_smd) > 0 and not np.isnan(concat_smd).all()
        else 0.40
    )
    ax.set_xlim(0.0, max(0.40, float(valid_max * 1.15)))
    ax.set_ylim(-0.8, n_vars - 0.2)
    ax.set_title(title, loc="left", pad=12)
    ax.legend(loc="lower right")

    fig.tight_layout()

    source_df = df.iloc[::-1].reset_index(drop=True)
    png_path = save_publication_figure(fig, out_path, dpi=dpi)
    csv_path = export_source_data(source_df, png_path)

    alt_text = "Love plot showing covariate standardized mean differences before and after adjustment."
    caption = f"Figure. {title}. Absolute Standardized Mean Differences (|SMD|) for baseline covariates before and after propensity adjustment. Shaded region indicates negligible imbalance (|SMD| < {threshold:.2f})."

    return FigureResult(
        png_path=png_path,
        alt_text=alt_text,
        caption=caption,
        source_df=source_df,
        csv_path=csv_path,
    )

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
    journal: str | None = None,
    canvas: str = "manuscript",
    export_vector: bool = True,
) -> FigureResult:
    """
    Renders Austin (2009) Love Plot displaying Standardized Mean Differences
    before versus after propensity score matching relative to the 0.10 balance threshold.
    """
    if journal is not None:
        from medstat.figures.journal import apply_journal_theme

        spec = apply_journal_theme(journal, canvas, target)
        palette = spec["palette"]
        color_unadj = palette[1 % len(palette)]
        color_match = palette[0]
        color_weight = palette[2 % len(palette)]
        w = 6.5
        h = 5.0
        dpi = 300
    else:
        w, h, dpi = set_clinical_figure_style(target)
        color_unadj = CLINICAL_PALETTE["warning"]
        color_match = CLINICAL_PALETTE["accent"]
        color_weight = CLINICAL_PALETTE["primary"]

    if isinstance(love_data, dict):
        var_col = None
        for k in [
            "covariates",
            "Variable",
            "Covariate",
            "variable",
            "covariate",
            "term",
            "feature",
            "var",
        ]:
            if k in love_data:
                var_col = love_data[k]
                break
        data_dict: dict[str, Any] = {"Variable": var_col if var_col is not None else []}
        for k in [
            "smd_raw",
            "Pre_Match_SMD",
            "Unadjusted_SMD",
            "smd_unadjusted",
            "raw_smd",
            "unadjusted",
        ]:
            if k in love_data:
                data_dict["Pre_Match_SMD"] = love_data[k]
                break
        for k in [
            "smd_matched",
            "Post_Match_SMD",
            "Matched_SMD",
            "smd_post",
            "matched",
        ]:
            if k in love_data:
                data_dict["Post_Match_SMD"] = love_data[k]
                break
        for k in ["smd_weighted", "Weighted_SMD", "Post_Weight_SMD", "weighted"]:
            if k in love_data:
                data_dict["Weighted_SMD"] = love_data[k]
                break
        df = pd.DataFrame(data_dict)
    else:
        df = love_data.copy()

    # Map variable column
    if "Variable" not in df.columns:
        found_var = False
        for c in ["Covariate", "covariate", "variable", "term", "feature", "var"]:
            if c in df.columns:
                df["Variable"] = df[c]
                found_var = True
                break
        if not found_var:
            df["Variable"] = [f"Var_{i + 1}" for i in range(len(df))]

    # Map Pre-match SMD
    if "Pre_Match_SMD" not in df.columns:
        for c in [
            "Unadjusted_SMD",
            "Pre_SMD",
            "smd_raw",
            "Unadjusted_Abs_SMD",
            "smd_unadjusted",
            "raw_smd",
            "unadjusted",
        ]:
            if c in df.columns:
                df["Pre_Match_SMD"] = df[c]
                break

    # Map Matched SMD
    has_matched = False
    if "Post_Match_SMD" in df.columns:
        has_matched = True
    else:
        for c in [
            "Matched_SMD",
            "Post_SMD",
            "smd_matched",
            "Matched_Abs_SMD",
            "matched",
        ]:
            if c in df.columns and df[c].notna().any():
                df["Post_Match_SMD"] = df[c]
                has_matched = True
                break

    # Map Weighted SMD
    has_weighted = False
    if "Weighted_SMD" in df.columns:
        has_weighted = True
    else:
        for c in ["Post_Weight_SMD", "Weighted_Abs_SMD", "smd_weighted", "weighted"]:
            if c in df.columns and df[c].notna().any():
                df["Weighted_SMD"] = df[c]
                has_weighted = True
                break

    if "Pre_Match_SMD" not in df.columns and not has_matched and not has_weighted:
        raise ValueError(
            "Could not identify any SMD columns in love_data. Checked Pre_Match_SMD, Post_Match_SMD, Weighted_SMD and aliases."
        )

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
        color=color_unadj,
        s=55,
        zorder=4,
        label="Raw Cohort",
        marker="o",
    )
    if has_matched and post_smd is not None:
        ax.scatter(
            post_smd,
            y_pos,
            color=color_match,
            s=55,
            zorder=5,
            label="Matched Cohort",
            marker="D",
        )
    if has_weighted and weight_smd is not None:
        ax.scatter(
            weight_smd,
            y_pos,
            color=color_weight,
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

    from medstat.figures.qa import qa_layout

    qa = qa_layout(fig, check_outside=False)

    source_df = df.iloc[::-1].reset_index(drop=True)
    if export_vector:
        from medstat.figures.styles import save_figure_bundle

        png_path, vec_path, _ = save_figure_bundle(
            fig, out_path, dpi=dpi, export_vector=True
        )
    else:
        png_path = save_publication_figure(fig, out_path, dpi=dpi)
        vec_path = None
    csv_path = export_source_data(source_df, png_path)

    alt_text = "Love plot showing covariate standardized mean differences before and after adjustment."
    caption = f"Figure. {title}. Absolute Standardized Mean Differences (|SMD|) for baseline covariates before and after propensity adjustment. Shaded region indicates negligible imbalance (|SMD| < {threshold:.2f})."

    return FigureResult(
        png_path=png_path,
        alt_text=alt_text,
        caption=caption,
        source_df=source_df,
        csv_path=csv_path,
        vector_path=vec_path,
        qa=qa,
    )

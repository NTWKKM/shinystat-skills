"""
src/medstat/figures/roc.py: Publication-grade Receiver Operating Characteristic (ROC) curve generator.
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


def plot_roc_curve(
    roc_data: dict[str, Any],
    paired_roc_data: dict[str, Any] | None = None,
    paired_p_value: float | None = None,
    title: str = "Receiver Operating Characteristic (ROC) Curve",
    out_path: str | Path = "figures/roc_curve.png",
    target: Literal["docx", "pptx", "web"] = "docx",
) -> FigureResult:
    """
    Renders an empirical ROC curve with 45-degree chance diagonal, Youden's J optimal
    cutpoint marker, DeLong 95% CI label, and optional paired comparison overlay.
    """
    w, h, dpi = set_clinical_figure_style(target)
    fig, ax = plt.subplots(figsize=(w, h))

    # Primary model curve
    fpr = np.asarray(roc_data["fpr"], dtype=float)
    tpr = np.asarray(roc_data["tpr"], dtype=float)
    auc = float(roc_data.get("auc", 0.5))
    ci_l = roc_data.get("ci_lower")
    ci_u = roc_data.get("ci_upper")

    model_name = str(roc_data.get("name", "Model 1"))
    if ci_l is not None and ci_u is not None:
        label_1 = f"{model_name}: AUC = {auc:.3f} (95% CI, {ci_l:.3f}–{ci_u:.3f})"
    else:
        label_1 = f"{model_name}: AUC = {auc:.3f}"

    ax.plot(fpr, tpr, color=CLINICAL_PALETTE["secondary"], lw=2.2, label=label_1)

    # Optimal cutoff annotation
    opt_pt = roc_data.get("optimal_point")
    if opt_pt:
        opt_fpr = float(opt_pt["fpr"])
        opt_tpr = float(opt_pt["tpr"])
        opt_cut = float(opt_pt["cutoff"])
        opt_sens = float(roc_data.get("sensitivity", opt_tpr))
        opt_spec = float(roc_data.get("specificity", 1.0 - opt_fpr))

        ax.scatter(
            [opt_fpr],
            [opt_tpr],
            color=CLINICAL_PALETTE["danger"],
            s=55,
            zorder=6,
            label=f"Optimal Cutpoint ({opt_cut:.2f})",
        )
        ax.annotate(
            f"Cutoff: {opt_cut:.2f}\nSens: {opt_sens:.1%}\nSpec: {opt_spec:.1%}",
            xy=(opt_fpr, opt_tpr),
            xytext=(opt_fpr + 0.12, opt_tpr - 0.18),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color=CLINICAL_PALETTE["primary"], lw=1.0),
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="#FFFBEB",
                edgecolor="#FDE68A",
                alpha=0.95,
            ),
        )

    # Optional paired model
    if paired_roc_data is not None:
        fpr_2 = np.asarray(paired_roc_data["fpr"], dtype=float)
        tpr_2 = np.asarray(paired_roc_data["tpr"], dtype=float)
        auc_2 = float(paired_roc_data.get("auc", 0.5))
        ci_l2 = paired_roc_data.get("ci_lower")
        ci_u2 = paired_roc_data.get("ci_upper")
        name_2 = str(paired_roc_data.get("name", "Model 2"))

        if ci_l2 is not None and ci_u2 is not None:
            label_2 = f"{name_2}: AUC = {auc_2:.3f} (95% CI, {ci_l2:.3f}–{ci_u2:.3f})"
        else:
            label_2 = f"{name_2}: AUC = {auc_2:.3f}"

        ax.plot(
            fpr_2,
            tpr_2,
            color=CLINICAL_PALETTE["accent"],
            lw=2.0,
            ls="--",
            label=label_2,
        )

        if paired_p_value is not None:
            diff_text = (
                f"Paired DeLong ΔAUC P = {paired_p_value:.3f}"
                if paired_p_value >= 0.001
                else "Paired DeLong P < 0.001"
            )
            ax.text(
                0.04,
                0.20,
                diff_text,
                transform=ax.transAxes,
                fontsize=9.5,
                fontweight="600",
                bbox=dict(
                    boxstyle="round,pad=0.3", facecolor="#F8FAFC", edgecolor="#CBD5E1"
                ),
            )

    # 45-degree reference diagonal
    ax.plot(
        [0, 1],
        [0, 1],
        color=CLINICAL_PALETTE["neutral"],
        ls=":",
        lw=1.2,
        label="Chance (AUC = 0.500)",
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xlabel("1 – Specificity (False Positive Rate)", labelpad=6)
    ax.set_ylabel("Sensitivity (True Positive Rate)", labelpad=6)
    ax.set_title(title, loc="left", pad=10)
    ax.legend(loc="lower right")

    fig.tight_layout()

    if paired_roc_data is not None:
        primary_name = str(roc_data.get("name", "Primary Model"))
        paired_name = str(paired_roc_data.get("name", "Model 2"))
        paired_thresh = paired_roc_data.get("thresholds", np.full_like(fpr_2, np.nan))
        df_primary = pd.DataFrame(
            {
                "Model": primary_name,
                "False_Positive_Rate": fpr,
                "True_Positive_Rate": tpr,
                "Threshold": roc_data.get("thresholds", np.full_like(fpr, np.nan)),
            }
        )
        df_paired = pd.DataFrame(
            {
                "Model": paired_name,
                "False_Positive_Rate": fpr_2,
                "True_Positive_Rate": tpr_2,
                "Threshold": paired_thresh,
            }
        )
        source_df = pd.concat([df_primary, df_paired], ignore_index=True)
    else:
        source_df = pd.DataFrame(
            {
                "False_Positive_Rate": fpr,
                "True_Positive_Rate": tpr,
                "Threshold": roc_data.get("thresholds", np.full_like(fpr, np.nan)),
            }
        )
    png_path = save_publication_figure(fig, out_path, dpi=dpi)
    csv_path = export_source_data(source_df, png_path)

    alt_text = f"ROC curve showing discrimination accuracy with AUC of {auc:.3f}."
    caption = f"Figure. {title}. Receiver Operating Characteristic curve displaying diagnostic discrimination (AUC = {auc:.3f}). Solid line represents primary empirical curve with 95% DeLong confidence intervals."

    return FigureResult(
        png_path=png_path,
        alt_text=alt_text,
        caption=caption,
        source_df=source_df,
        csv_path=csv_path,
    )

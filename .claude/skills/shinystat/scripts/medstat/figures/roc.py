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
        primary_name = model_name
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


def plot_time_dependent_roc(
    td_roc_data: dict[str, Any],
    mode: Literal["curves", "auroc_trajectory"] = "curves",
    title: str | None = None,
    out_path: str | Path = "figures/td_roc.png",
    target: Literal["docx", "pptx", "web"] = "docx",
) -> FigureResult:
    """
    Renders publication-grade Time-Dependent ROC curves or AUROC(t) discrimination trajectories.

    Modes:
    - "curves": Multi-horizon cumulative/dynamic ROC curves at t_1, t_2, ... with 95% CIs in legend.
    - "auroc_trajectory": Longitudinal AUROC(t) trajectory curve across time with 95% CI bands.
    """
    w, h, dpi = set_clinical_figure_style(target)
    fig, ax = plt.subplots(figsize=(w, h))

    horizons = td_roc_data.get("time_horizons", [])
    results_by_horizon = td_roc_data.get("results_by_horizon", {})

    palette = [
        CLINICAL_PALETTE["secondary"],
        CLINICAL_PALETTE["primary"],
        CLINICAL_PALETTE["accent"],
        "#0D9488",  # Teal
        "#8B5CF6",  # Purple
        "#DC2626",  # Red
    ]

    if mode == "curves":
        fig_title = title or "Time-Dependent Receiver Operating Characteristic Curves"
        source_rows = []

        # 45-degree reference diagonal
        ax.plot(
            [0, 1],
            [0, 1],
            color=CLINICAL_PALETTE["neutral"],
            ls=":",
            lw=1.2,
            label="Chance (AUC = 0.500)",
        )

        for idx, t in enumerate(horizons):
            res_h = results_by_horizon.get(t, results_by_horizon.get(str(t), {}))
            if not res_h:
                continue
            fpr = np.asarray(res_h.get("fpr", [0, 1]), dtype=float)
            tpr = np.asarray(res_h.get("tpr", [0, 1]), dtype=float)
            auc_val = float(res_h.get("auroc", 0.5))
            ci_l = res_h.get("ci_lower")
            ci_u = res_h.get("ci_upper")
            thresh = res_h.get("thresholds", [np.nan] * len(fpr))

            color = palette[idx % len(palette)]
            if (
                ci_l is not None
                and not np.isnan(ci_l)
                and ci_u is not None
                and not np.isnan(ci_u)
            ):
                label = (
                    f"t = {t:g}: AUC = {auc_val:.3f} (95% CI, {ci_l:.3f}–{ci_u:.3f})"
                )
            else:
                label = f"t = {t:g}: AUC = {auc_val:.3f}"

            ax.plot(fpr, tpr, color=color, lw=2.0, label=label)

            for f_val, t_val, th_val in zip(fpr, tpr, thresh):
                source_rows.append(
                    {
                        "Horizon": t,
                        "False_Positive_Rate": float(f_val),
                        "True_Positive_Rate": float(t_val),
                        "Threshold": float(th_val) if not np.isinf(th_val) else np.nan,
                    }
                )

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.04)
        ax.set_xlabel("1 – Specificity (False Positive Rate)", labelpad=6)
        ax.set_ylabel("Sensitivity (True Positive Rate)", labelpad=6)
        ax.set_title(fig_title, loc="left", pad=10)
        ax.legend(loc="lower right", fontsize=8.5)
        source_df = pd.DataFrame(source_rows)
        alt_text = f"Time-dependent ROC curves across horizons {horizons}."
        caption = f"Figure. {fig_title}. Cumulative/dynamic IPCW ROC curves across follow-up horizons with 95% confidence intervals."

    else:  # auroc_trajectory
        fig_title = title or "Longitudinal Discrimination Trajectory (AUROC(t))"
        source_rows = []
        comp_dict = td_roc_data.get("comparison")

        if comp_dict:
            # Dual model comparison trajectory
            h_vals = sorted(list(comp_dict.keys()), key=lambda x: float(x))
            x_vals = [float(x) for x in h_vals]
            old_aucs = [float(comp_dict[x]["auroc_old"]) for x in h_vals]
            new_aucs = [float(comp_dict[x]["auroc_new"]) for x in h_vals]

            ax.plot(
                x_vals,
                old_aucs,
                marker="o",
                color=CLINICAL_PALETTE["neutral"],
                lw=2.0,
                ls="--",
                label="Baseline Model",
            )
            ax.plot(
                x_vals,
                new_aucs,
                marker="s",
                color=CLINICAL_PALETTE["secondary"],
                lw=2.2,
                label="Updated Model",
            )

            for x_t, a_old, a_new in zip(x_vals, old_aucs, new_aucs):
                source_rows.append(
                    {"Model": "Baseline", "Horizon": x_t, "AUROC": a_old}
                )
                source_rows.append({"Model": "Updated", "Horizon": x_t, "AUROC": a_new})
        else:
            traj = td_roc_data.get("trajectory", {})
            x_vals = [float(x) for x in traj.get("horizons", horizons)]
            y_vals = [float(y) for y in traj.get("auroc", [])]
            ci_l = [
                float(val_l) if val_l is not None and not np.isnan(val_l) else y_vals[i]
                for i, val_l in enumerate(traj.get("ci_lower", []))
            ]
            ci_u = [
                float(u) if u is not None and not np.isnan(u) else y_vals[i]
                for i, u in enumerate(traj.get("ci_upper", []))
            ]

            ax.plot(
                x_vals,
                y_vals,
                marker="o",
                color=CLINICAL_PALETTE["secondary"],
                lw=2.2,
                label="AUROC(t)",
            )
            ax.fill_between(
                x_vals,
                ci_l,
                ci_u,
                color=CLINICAL_PALETTE["secondary"],
                alpha=0.18,
                label="95% CI",
            )

            for x_t, y_t, l_t, u_t in zip(x_vals, y_vals, ci_l, ci_u):
                source_rows.append(
                    {
                        "Horizon": x_t,
                        "AUROC": y_t,
                        "CI_Lower": l_t,
                        "CI_Upper": u_t,
                    }
                )

        ax.axhline(0.50, color="#94A3B8", ls=":", lw=1.2, label="Chance (0.50)")
        ax.set_ylim(0.45, 1.02)
        ax.set_xlabel("Follow-up Time Horizon", labelpad=6)
        ax.set_ylabel("Cumulative/Dynamic AUROC", labelpad=6)
        ax.set_title(fig_title, loc="left", pad=10)
        ax.legend(loc="lower left", fontsize=8.5)
        source_df = pd.DataFrame(source_rows)
        alt_text = f"Longitudinal AUROC trajectory plot over horizons {x_vals}."
        caption = f"Figure. {fig_title}. Longitudinal cumulative/dynamic discrimination performance (AUROC(t)) plotted across follow-up time."

    fig.tight_layout()
    png_path = save_publication_figure(fig, out_path, dpi=dpi)
    csv_path = export_source_data(source_df, png_path)

    return FigureResult(
        png_path=png_path,
        alt_text=alt_text,
        caption=caption,
        source_df=source_df,
        csv_path=csv_path,
    )


def plot_reclassification(
    reclass_data: dict[str, Any] | None = None,
    p_old: pd.Series | np.ndarray | None = None,
    p_new: pd.Series | np.ndarray | None = None,
    y_true: pd.Series | np.ndarray | None = None,
    thresholds: list[float] | None = None,
    title: str = "Risk Reclassification (Baseline vs Updated Model)",
    out_path: str | Path = "figures/reclassification.png",
    target: Literal["docx", "pptx", "web"] = "docx",
) -> FigureResult:
    """
    Renders publication-grade Risk Reclassification scatter and calibration-shift plot.
    Displays p_old on x-axis vs p_new on y-axis, colored by Y=1 Events vs Y=0 Non-events,
    with diagonal equality line and optional risk strata threshold grid lines.
    """
    w, h, dpi = set_clinical_figure_style(target)
    fig, ax = plt.subplots(figsize=(w, h))

    # Extract arrays
    if p_old is None or p_new is None or y_true is None:
        if (
            reclass_data is not None
            and "p_old" in reclass_data
            and "p_new" in reclass_data
        ):
            p_0 = pd.Series(reclass_data["p_old"]).to_numpy(dtype=float)
            p_1 = pd.Series(reclass_data["p_new"]).to_numpy(dtype=float)
            y_t = pd.Series(reclass_data["y_true"]).to_numpy(dtype=float)
            if thresholds is None:
                thresholds = reclass_data.get("thresholds")
        else:
            raise ValueError(
                "plot_reclassification requires p_old, p_new, and y_true (or reclass_data containing them)."
            )
    else:
        p_0 = pd.Series(p_old).to_numpy(dtype=float)
        p_1 = pd.Series(p_new).to_numpy(dtype=float)
        y_t = pd.Series(y_true).to_numpy(dtype=float)

    valid = ~(np.isnan(p_0) | np.isnan(p_1) | np.isnan(y_t))
    p_0 = p_0[valid]
    p_1 = p_1[valid]
    y_t = y_t[valid].astype(int)

    # Non-events (Y=0)
    idx_0 = y_t == 0
    ax.scatter(
        p_0[idx_0],
        p_1[idx_0],
        color=CLINICAL_PALETTE["secondary"],
        alpha=0.50,
        s=26,
        edgecolors="none",
        label=f"Non-Events (Y=0, n={int(np.sum(idx_0))})",
    )

    # Events (Y=1)
    idx_1 = y_t == 1
    ax.scatter(
        p_0[idx_1],
        p_1[idx_1],
        color=CLINICAL_PALETTE["danger"],
        alpha=0.75,
        s=36,
        marker="^",
        edgecolors="#991B1B",
        linewidths=0.5,
        label=f"Events (Y=1, n={int(np.sum(idx_1))})",
    )

    # Equality diagonal (p_new = p_old)
    ax.plot(
        [0, 1],
        [0, 1],
        color=CLINICAL_PALETTE["neutral"],
        ls="--",
        lw=1.2,
        label="No Shift (p_new = p_old)",
    )

    # Threshold grid lines
    if thresholds:
        sorted_th = sorted([float(t) for t in thresholds])
        for th in sorted_th:
            ax.axvline(x=th, color="#CBD5E1", ls=":", lw=1.0)
            ax.axhline(y=th, color="#CBD5E1", ls=":", lw=1.0)

    # Summary metrics annotation
    if reclass_data:
        nri_c = reclass_data.get("nri_continuous", {}).get("total", {})
        idi_d = reclass_data.get("idi", {})
        txt_lines = []
        if "estimate" in nri_c:
            p_nri = nri_c.get("p_value", 1.0)
            p_nri_str = f"P={p_nri:.3f}" if p_nri >= 0.001 else "P<0.001"
            txt_lines.append(f"Continuous NRI: {nri_c['estimate']:.3f} ({p_nri_str})")
        if "estimate" in idi_d:
            p_idi = idi_d.get("p_value", 1.0)
            p_idi_str = f"P={p_idi:.3f}" if p_idi >= 0.001 else "P<0.001"
            txt_lines.append(f"IDI: {idi_d['estimate']:.3f} ({p_idi_str})")
        if txt_lines:
            ax.text(
                0.04,
                0.86,
                "\n".join(txt_lines),
                transform=ax.transAxes,
                fontsize=9.0,
                fontweight="600",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="#F8FAFC",
                    edgecolor="#CBD5E1",
                    alpha=0.92,
                ),
            )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Baseline Model Predicted Risk (p_old)", labelpad=6)
    ax.set_ylabel("Updated Model Predicted Risk (p_new)", labelpad=6)
    ax.set_title(title, loc="left", pad=10)
    ax.legend(loc="lower right", fontsize=8.5)

    fig.tight_layout()

    source_df = pd.DataFrame(
        {
            "Patient_ID": [f"ID_{i + 1}" for i in range(len(p_0))],
            "Y_True": y_t,
            "P_Old": p_0,
            "P_New": p_1,
            "Risk_Difference": p_1 - p_0,
        }
    )
    png_path = save_publication_figure(fig, out_path, dpi=dpi)
    csv_path = export_source_data(source_df, png_path)

    alt_text = f"Risk reclassification scatter plot comparing p_old vs p_new across {len(p_0)} patients."
    caption = f"Figure. {title}. Scatter plot displaying individual predicted probabilities under the baseline versus updated model stratified by clinical outcome events."

    return FigureResult(
        png_path=png_path,
        alt_text=alt_text,
        caption=caption,
        source_df=source_df,
        csv_path=csv_path,
    )

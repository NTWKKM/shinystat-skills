"""
src/medstat/figures/calibration.py: Publication-grade Model Calibration Curve generator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from medstat._phi import assert_zero_phi_text
from medstat.diagnostic.calibration import (
    calculate_brier_score,
    calculate_calibration_slope_and_intercept,
)
from medstat.figures.base import FigureResult, export_source_data
from medstat.figures.journal import apply_journal_theme, format_num, style_axis
from medstat.figures.qa import qa_layout
from medstat.figures.styles import (
    CLINICAL_PALETTE,
    save_figure_bundle,
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
    journal: str | None = None,
    canvas: str = "manuscript",
    export_vector: bool = True,
) -> FigureResult:
    """
    Renders decile calibration points with 95% Wilson confidence intervals,
    45-degree ideal diagonal, smoothed calibration trajectory, and goodness metrics.
    """
    if journal is not None:
        spec = apply_journal_theme(journal, canvas, target)
        w, h = 4.5, 4.2
        dpi = 300
        primary_color = spec["palette"][0]
        secondary_color = spec["palette"][1]
        neutral_color = "#666666"
    else:
        w, h, dpi = set_clinical_figure_style(target)
        primary_color = CLINICAL_PALETTE["primary"]
        secondary_color = CLINICAL_PALETTE["secondary"]
        neutral_color = CLINICAL_PALETTE["neutral"]

    assert_zero_phi_text([title])
    fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)
    if journal is not None:
        style_axis(ax, journal)

    p_means = np.asarray(calib_bins["pred_mean"], dtype=float)
    o_rates = np.asarray(calib_bins["obs_rate"], dtype=float)
    ci_low = np.asarray(calib_bins.get("ci_lower", o_rates), dtype=float)
    ci_high = np.asarray(calib_bins.get("ci_upper", o_rates), dtype=float)
    counts = np.asarray(calib_bins.get("counts", np.zeros_like(p_means)), dtype=int)

    # 45-degree reference line
    ax.plot(
        [0, 1],
        [0, 1],
        color=neutral_color,
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
            color=secondary_color,
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
        color=primary_color,
        ecolor=neutral_color,
        elinewidth=1.2,
        capsize=3.0,
        markersize=6.0,
        zorder=5,
        label="Decile Observed (95% CI)",
    )

    # Metrics callout box
    metrics_lines = []
    if brier_score is not None:
        val_brier = (
            format_num(brier_score, 3, journal)
            if journal is not None
            else f"{brier_score:.3f}"
        )
        metrics_lines.append(f"Brier Score: {val_brier}")
    if ici is not None:
        val_ici = format_num(ici, 3, journal) if journal is not None else f"{ici:.3f}"
        metrics_lines.append(f"ICI: {val_ici}")

    if metrics_lines:
        y_pos = 0.68 if journal is not None else 0.82
        f_size = 8.0 if journal is not None else 9.5
        ax.text(
            0.04,
            y_pos,
            "\n".join(metrics_lines),
            transform=ax.transAxes,
            fontsize=f_size,
            bbox=dict(
                boxstyle="round,pad=0.35",
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
    if journal is not None:
        ax.legend(loc="upper left", frameon=False, fontsize=8.0)
    else:
        ax.legend(loc="lower right")

    fig.tight_layout()

    qa = qa_layout(fig, check_outside=False)

    source_df = pd.DataFrame(
        {
            "Bin_Predicted_Mean": p_means,
            "Bin_Observed_Rate": o_rates,
            "Bin_CI_Lower": ci_low,
            "Bin_CI_Upper": ci_high,
            "Bin_Count": counts,
        }
    )
    if export_vector:
        png_path, vec_path, _ = save_figure_bundle(
            fig, out_path, dpi=dpi, export_vector=True
        )
    else:
        png_path = save_publication_figure(fig, out_path, dpi=dpi)
        vec_path = None
    csv_path = export_source_data(source_df, png_path)

    alt_text = "Model calibration curve comparing predicted risk deciles against observed event rates."
    caption = f"Figure. {title}. Decile points denote mean predicted versus observed proportions with 95% Wilson confidence intervals. The dashed diagonal represents perfect agreement."

    return FigureResult(
        png_path=png_path,
        alt_text=alt_text,
        caption=caption,
        source_df=source_df,
        csv_path=csv_path,
        vector_path=vec_path,
        qa=qa,
    )


def plot_recalibration_comparison(
    y_true: pd.Series | np.ndarray | dict[str, Any],
    p_orig: pd.Series | np.ndarray | None = None,
    p_recal: pd.Series | np.ndarray | None = None,
    n_bins: int = 10,
    title: str = "Model Recalibration Comparison",
    out_path: str | Path | None = None,
    output_path: str | Path | None = None,
    target: Literal["docx", "pptx", "web"] = "docx",
    journal: str | None = None,
    canvas: str = "manuscript",
    export_vector: bool = True,
) -> FigureResult:
    """
    Plot calibration curves pre- vs post-recalibration.

    Displays:
    - 45-degree ideal agreement line
    - Pre-recalibration curve: dashed line with open markers (annotated with alpha, beta, Brier)
    - Post-recalibration curve: solid line with filled markers (annotated with updated alpha, beta, Brier)
    """
    save_target = out_path if out_path is not None else output_path
    if save_target is None:
        save_target = "figures/recalibration_comparison.png"

    if journal is not None:
        spec = apply_journal_theme(journal, canvas, target)
        w, h = 4.5, 4.2
        dpi = 300
        primary_color = spec["palette"][0]
        neutral_color = "#666666"
    else:
        w, h, dpi = set_clinical_figure_style(target)
        primary_color = CLINICAL_PALETTE["primary"]
        neutral_color = CLINICAL_PALETTE["neutral"]

    assert_zero_phi_text([title])
    fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)
    if journal is not None:
        style_axis(ax, journal)

    if isinstance(y_true, dict):
        d = y_true
        y_data = d.get("y_true", d.get("y"))
        p_o_data = (
            p_orig
            if p_orig is not None
            else d.get("p_orig", d.get("p_pred", d.get("p")))
        )
        p_r_data = (
            p_recal
            if p_recal is not None
            else d.get("p_recalibrated", d.get("p_recal"))
        )
    else:
        y_data = y_true
        p_o_data = p_orig
        p_r_data = p_recal

    if y_data is None or p_o_data is None or p_r_data is None:
        raise ValueError(
            "plot_recalibration_comparison requires y_true, p_orig, and p_recal "
            "(or a recalibration result dictionary containing them)."
        )

    y_t = np.asarray(y_data, dtype=int).ravel()
    p_o = np.clip(np.asarray(p_o_data, dtype=float).ravel(), 1e-6, 1.0 - 1e-6)
    p_r = np.clip(np.asarray(p_r_data, dtype=float).ravel(), 1e-6, 1.0 - 1e-6)

    # Calculate metrics
    pre_cal = calculate_calibration_slope_and_intercept(y_t, p_o)
    pre_br = calculate_brier_score(y_t, p_o)["brier_score"]
    post_cal = calculate_calibration_slope_and_intercept(y_t, p_r)
    post_br = calculate_brier_score(y_t, p_r)["brier_score"]

    # Calibration curves (quantiles or uniform bins)
    try:
        prob_true_o, prob_pred_o = calibration_curve(
            y_t, p_o, n_bins=n_bins, strategy="quantile"
        )
    except Exception:
        prob_true_o, prob_pred_o = calibration_curve(
            y_t, p_o, n_bins=n_bins, strategy="uniform"
        )

    try:
        prob_true_r, prob_pred_r = calibration_curve(
            y_t, p_r, n_bins=n_bins, strategy="quantile"
        )
    except Exception:
        prob_true_r, prob_pred_r = calibration_curve(
            y_t, p_r, n_bins=n_bins, strategy="uniform"
        )

    # 45-degree line
    ax.plot(
        [0, 1],
        [0, 1],
        color=neutral_color,
        ls="--",
        lw=1.2,
        label="Ideal Calibration",
    )

    # Pre-recalibration curve
    if journal is not None:
        p_c_int = format_num(pre_cal["calibration_intercept"], 2, journal)
        p_c_slp = format_num(pre_cal["calibration_slope"], 2, journal)
        p_c_br = format_num(pre_br, 3, journal)
    else:
        p_c_int = f"{pre_cal['calibration_intercept']:.2f}"
        p_c_slp = f"{pre_cal['calibration_slope']:.2f}"
        p_c_br = f"{pre_br:.3f}"

    lbl_pre = (
        f"Pre-Recalibration (Intercept={p_c_int}, Slope={p_c_slp}, Brier={p_c_br})"
    )
    ax.plot(
        prob_pred_o,
        prob_true_o,
        "o--",
        color="#64748B",
        fillstyle="none",
        lw=1.8,
        markersize=7.0,
        label=lbl_pre,
    )

    # Post-recalibration curve
    if journal is not None:
        po_c_int = format_num(post_cal["calibration_intercept"], 2, journal)
        po_c_slp = format_num(post_cal["calibration_slope"], 2, journal)
        po_c_br = format_num(post_br, 3, journal)
    else:
        po_c_int = f"{post_cal['calibration_intercept']:.2f}"
        po_c_slp = f"{post_cal['calibration_slope']:.2f}"
        po_c_br = f"{post_br:.3f}"

    lbl_post = (
        f"Post-Recalibration (Intercept={po_c_int}, Slope={po_c_slp}, Brier={po_c_br})"
    )
    ax.plot(
        prob_pred_r,
        prob_true_r,
        "s-",
        color=primary_color,
        lw=2.2,
        markersize=7.0,
        label=lbl_post,
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xlabel("Predicted Probability", labelpad=6)
    ax.set_ylabel("Observed Proportion", labelpad=6)
    ax.set_title(title, loc="left", pad=10)
    ax.legend(
        loc="lower right",
        framealpha=0.9 if journal is None else None,
        frameon=journal is None,
    )

    fig.tight_layout()

    qa = qa_layout(fig, check_outside=False)

    # Create coordinate table
    max_len = max(len(prob_pred_o), len(prob_pred_r))
    df_rows = []
    for i in range(max_len):
        df_rows.append(
            {
                "Bin": i + 1,
                "Pre_Pred_Mean": prob_pred_o[i] if i < len(prob_pred_o) else np.nan,
                "Pre_Obs_Rate": prob_true_o[i] if i < len(prob_true_o) else np.nan,
                "Post_Pred_Mean": prob_pred_r[i] if i < len(prob_pred_r) else np.nan,
                "Post_Obs_Rate": prob_true_r[i] if i < len(prob_true_r) else np.nan,
            }
        )
    source_df = pd.DataFrame(df_rows)

    if export_vector:
        png_path, vec_path, _ = save_figure_bundle(
            fig, save_target, dpi=dpi, export_vector=True
        )
    else:
        png_path = save_publication_figure(fig, save_target, dpi=dpi)
        vec_path = None
    csv_path = export_source_data(source_df, png_path)

    alt_text = "Model recalibration comparison comparing pre- and post-recalibration calibration curves against ideal diagonal."
    caption = (
        f"Figure. {title}. Dashed line with open circles represents pre-recalibration model calibration; "
        f"solid line with squares represents updated model calibration. Ideal calibration is denoted by the dashed diagonal."
    )

    return FigureResult(
        png_path=png_path,
        alt_text=alt_text,
        caption=caption,
        source_df=source_df,
        csv_path=csv_path,
        vector_path=vec_path,
        qa=qa,
    )


def plot_bootstrap_optimism(
    validation_res: dict[str, Any],
    title: str = "Bootstrap Internal Validation (Optimism)",
    out_path: str | Path | None = None,
    output_path: str | Path | None = None,
    target: Literal["docx", "pptx", "web"] = "docx",
    journal: str | None = None,
    canvas: str = "manuscript",
    export_vector: bool = True,
) -> FigureResult:
    """
    Renders grouped horizontal bar chart or dumbbell plot comparing
    Apparent vs Optimism-Corrected model performance metrics with uniform shrinkage factor callout.
    """
    save_target = out_path if out_path is not None else output_path
    if save_target is None:
        save_target = "figures/bootstrap_optimism.png"

    if journal is not None:
        spec = apply_journal_theme(journal, canvas, target)
        w, h = 6.0, 4.5
        dpi = 300
        primary_color = spec["palette"][0]
        secondary_color = spec["palette"][1]
    else:
        w, h, dpi = set_clinical_figure_style(target)
        primary_color = CLINICAL_PALETTE["primary"]
        secondary_color = "#94A3B8"

    assert_zero_phi_text([title])
    fig, ax = plt.subplots(figsize=(w, max(h, 4.2)), dpi=dpi)
    if journal is not None:
        style_axis(ax, journal)

    metrics = validation_res.get("metrics", {})
    s_hat = float(validation_res.get("shrinkage_factor", 1.0))
    epp = float(validation_res.get("events_per_predictor", np.nan))
    n_succ = validation_res.get(
        "n_successful_replicates", validation_res.get("n_bootstrap", 200)
    )
    n_boot = validation_res.get("n_bootstrap", 200)

    # Plot metrics: AUROC, Calibration Slope, Nagelkerke R2, 1 - Brier Score
    metric_order = [
        ("auroc", "AUROC"),
        ("calibration_slope", "Calibration Slope"),
        ("nagelkerke_r2", "Nagelkerke R²"),
        ("brier_score", "1 - Brier Score"),
    ]

    labels = []
    apparent_vals = []
    corrected_vals = []
    optimism_vals = []

    for key, display_name in metric_order:
        m_info = metrics.get(key, {})
        app = float(m_info.get("apparent", np.nan))
        corr = float(m_info.get("optimism_corrected", np.nan))
        opt_val = float(m_info.get("optimism", np.nan))

        if key == "brier_score":
            app = 1.0 - app
            corr = 1.0 - corr

        labels.append(display_name)
        apparent_vals.append(app)
        corrected_vals.append(corr)
        optimism_vals.append(opt_val)

    y_pos = np.arange(len(labels))
    bar_height = 0.32

    # Grouped horizontal bars
    ax.barh(
        y_pos + bar_height / 2,
        apparent_vals,
        height=bar_height,
        color=secondary_color,
        label="Apparent Performance",
        zorder=3,
    )
    ax.barh(
        y_pos - bar_height / 2,
        corrected_vals,
        height=bar_height,
        color=primary_color,
        label="Optimism-Corrected",
        zorder=3,
    )

    # Annotate value labels on bars
    for idx, (app, corr) in enumerate(zip(apparent_vals, corrected_vals)):
        if np.isfinite(app):
            app_str = (
                format_num(app, 3, journal) if journal is not None else f"{app:.3f}"
            )
            ax.text(
                app + 0.02,
                idx + bar_height / 2,
                app_str,
                va="center",
                fontsize=8.5,
                color="#475569",
            )
        if np.isfinite(corr):
            corr_str = (
                format_num(corr, 3, journal) if journal is not None else f"{corr:.3f}"
            )
            ax.text(
                corr + 0.02,
                idx - bar_height / 2,
                corr_str,
                va="center",
                fontsize=8.5,
                color="#0F172A",
                fontweight="bold",
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("Metric Value", labelpad=6)
    ax.set_title(title, loc="left", pad=12)
    max_val = max([v for v in apparent_vals + corrected_vals if np.isfinite(v)] + [1.0])
    ax.set_xlim(0.0, max_val * 1.35)
    ax.legend(
        loc="lower right",
        framealpha=0.9 if journal is None else None,
        frameon=journal is None,
    )
    if journal is None:
        ax.grid(axis="x", ls=":", alpha=0.6, zorder=0)

    # Callout annotation for shrinkage factor and replicates
    s_hat_str = format_num(s_hat, 3, journal) if journal is not None else f"{s_hat:.3f}"
    callout_text = f"Uniform Shrinkage Factor (ŝ): {s_hat_str}"
    if np.isfinite(epp):
        epp_str = format_num(epp, 1, journal) if journal is not None else f"{epp:.1f}"
        callout_text += f"\nEvents Per Predictor (EPP): {epp_str}"
    callout_text += f"\nSuccessful Replicates: {n_succ}/{n_boot}"

    ax.text(
        0.04,
        0.12,
        callout_text,
        transform=ax.transAxes,
        fontsize=8.5,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#F8FAFC",
            edgecolor="#CBD5E1",
            alpha=0.9,
        ),
        zorder=5,
    )

    fig.tight_layout()

    qa = qa_layout(fig, check_outside=False)

    source_df = pd.DataFrame(
        {
            "Metric": labels,
            "Apparent": apparent_vals,
            "Optimism_Corrected": corrected_vals,
            "Optimism": optimism_vals,
            "Uniform_Shrinkage_Factor": [s_hat] * len(labels),
        }
    )

    if export_vector:
        png_path, vec_path, _ = save_figure_bundle(
            fig, save_target, dpi=dpi, export_vector=True
        )
    else:
        png_path = save_publication_figure(fig, save_target, dpi=dpi)
        vec_path = None
    csv_path = export_source_data(source_df, png_path)

    alt_text = "Grouped horizontal bar chart comparing apparent versus optimism-corrected prediction performance metrics."
    caption = (
        f"Figure. {title}. Apparent performance is compared against TRIPOD+AI optimism-corrected performance "
        f"via {n_boot} bootstrap replicates. Uniform shrinkage factor ŝ = {s_hat:.3f}."
    )

    return FigureResult(
        png_path=png_path,
        alt_text=alt_text,
        caption=caption,
        source_df=source_df,
        csv_path=csv_path,
        vector_path=vec_path,
        qa=qa,
    )

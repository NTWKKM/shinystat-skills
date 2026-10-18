"""
src/medstat/figures/survival.py: Publication-grade Kaplan-Meier curves with aligned risk table.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from medstat._phi import assert_zero_phi_text
from medstat.figures.base import FigureResult, export_source_data
from medstat.figures.journal import (
    apply_journal_theme,
    format_p_value,
    style_axis,
)
from medstat.figures.qa import qa_layout
from medstat.figures.styles import (
    GROUP_COLORS,
    save_figure_bundle,
    save_publication_figure,
    set_clinical_figure_style,
)


def plot_kaplan_meier(
    km_curves: dict[str, dict[str, np.ndarray]],
    risk_table: pd.DataFrame | None = None,
    log_rank_p: float | None = None,
    time_unit: str = "Days",
    title: str = "Kaplan-Meier Overall Survival",
    out_path: str | Path = "figures/kaplan_meier.png",
    target: Literal["docx", "pptx", "web"] = "docx",
    journal: str | None = None,
    canvas: str = "manuscript",
    export_vector: bool = True,
    nejm_inset_ylim: tuple[float, float] | None = None,
) -> FigureResult:
    """
    Renders publication Kaplan-Meier step-curves with Greenwood 95% CI bands,
    censored event tick marks, and an aligned numbers-at-risk table below the plot.
    """
    if journal is not None:
        spec = apply_journal_theme(journal, canvas, target)
        palette = spec["palette"]
        w = 6.5
        h = 4.8
        dpi = 300
    else:
        palette = GROUP_COLORS
        w, h, dpi = set_clinical_figure_style(target)

    has_risk_table = risk_table is not None and not risk_table.empty
    if has_risk_table:
        fig = plt.figure(figsize=(w, h + 1.2), dpi=dpi)
        gs = fig.add_gridspec(2, 1, height_ratios=[3.5, 1.0], hspace=0.28)
        ax_km = fig.add_subplot(gs[0])
        ax_risk = fig.add_subplot(gs[1])
    else:
        fig, ax_km = plt.subplots(figsize=(w, h), dpi=dpi)
        ax_risk = None

    if journal is not None:
        style_axis(ax_km, journal)

    curve_rows = []
    groups = list(km_curves.keys())

    max_t = 0.0
    for idx, (grp_name, data) in enumerate(km_curves.items()):
        color = palette[idx % len(palette)]
        t = np.asarray(data["timeline"], dtype=float)
        s = np.asarray(data["survival"], dtype=float)
        ci_l = np.asarray(data.get("ci_lower", s), dtype=float)
        ci_u = np.asarray(data.get("ci_upper", s), dtype=float)

        if len(t) > 0:
            max_t = max(max_t, float(np.max(t)))

        # Step survival curve
        ax_km.step(
            t,
            s,
            where="post",
            color=color,
            lw=1.8 if journal is not None else 2.0,
            label=grp_name,
        )
        # 95% CI shaded band
        ax_km.fill_between(t, ci_l, ci_u, step="post", color=color, alpha=0.15)

        # Censored tick marks
        cens_t = np.asarray(data.get("censored_times", []), dtype=float)
        cens_s = np.asarray(data.get("censored_survival", []), dtype=float)
        if len(cens_t) > 0 and len(cens_s) > 0:
            marker_sym = "|" if journal is not None else "+"
            ax_km.scatter(
                cens_t,
                cens_s,
                marker=marker_sym,
                s=28,
                color=color,
                alpha=0.85,
                zorder=4,
            )

        for ti, si, li, ui in zip(t, s, ci_l, ci_u):
            curve_rows.append(
                {
                    "Group": grp_name,
                    "Time": ti,
                    "Survival": si,
                    "CI_Lower": li,
                    "CI_Upper": ui,
                }
            )

    if journal is not None:
        ax_km.set_ylim(0.0, 1.02)
        ax_km.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    else:
        ax_km.set_ylim(-0.02, 1.04)
    ax_km.set_xlim(0, max(max_t, 1.0))
    ax_km.set_ylabel("Survival Probability", labelpad=6)
    ax_km.set_title(title, loc="left", pad=10)

    # Optional zoomed inset break
    if nejm_inset_ylim is not None:
        ins = ax_km.inset_axes([0.10, 0.47, 0.50, 0.46])
        for idx, (grp_name, data) in enumerate(km_curves.items()):
            color = palette[idx % len(palette)]
            t = np.asarray(data["timeline"], dtype=float)
            s = np.asarray(data["survival"], dtype=float)
            ins.step(t, s, where="post", color=color, lw=1.6)
        if journal is not None:
            style_axis(ins, journal)
        ins.set_xlim(ax_km.get_xlim())
        ins.set_ylim(nejm_inset_ylim[0], nejm_inset_ylim[1])
        ins.tick_params(labelsize=7.5)
        for dy in (0.0, 0.035):
            ins.plot(
                [-0.045, 0.045],
                [0.80 + dy - 0.02, 0.80 + dy + 0.02],
                transform=ins.transAxes,
                clip_on=False,
                color="#111111",
                lw=1.3,
            )

    # Annotated Log-Rank p-value
    if log_rank_p is not None:
        if journal is not None:
            p_text = f"Log-rank {format_p_value(log_rank_p, journal)}"
            if nejm_inset_ylim:
                ax_km.text(
                    0.97,
                    0.40,
                    p_text,
                    transform=ax_km.transAxes,
                    fontsize=8.5,
                    va="top",
                    ha="right",
                )
            else:
                ax_km.text(
                    0.04,
                    0.12,
                    p_text,
                    transform=ax_km.transAxes,
                    fontsize=8.5,
                    fontweight="600",
                )
        else:
            p_text = (
                f"Log-rank P = {log_rank_p:.3f}"
                if log_rank_p >= 0.001
                else "Log-rank P < 0.001"
            )
            ax_km.text(
                0.04,
                0.12,
                p_text,
                transform=ax_km.transAxes,
                fontsize=10,
                fontweight="600",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="#FFFFFF",
                    edgecolor="#CBD5E1",
                    alpha=0.9,
                ),
            )

    if nejm_inset_ylim is not None:
        ax_km.legend(loc="center right", bbox_to_anchor=(1.0, 0.60), frameon=False)
    elif journal is not None:
        ax_km.legend(loc="upper right", frameon=False)
    else:
        ax_km.legend(loc="upper right")

    # Aligned Risk Table
    if has_risk_table and ax_risk is not None:
        import matplotlib as mpl

        ax_risk.axis("off")

        time_cols = [c for c in risk_table.columns if c != "Group"]
        n_grps = len(risk_table)
        ax_risk.set_xlim(ax_km.get_xlim())

        # Parse milestone times from columns (e.g. "t=30.0")
        milestone_vals = []
        for c in time_cols:
            try:
                milestone_vals.append(float(c.replace("t=", "")))
            except ValueError:
                milestone_vals.append(0.0)

        if journal is not None:
            ax_risk.text(
                0.0,
                1.05,
                "No. at Risk",
                transform=ax_risk.transAxes,
                fontsize=8.5,
                fontweight="bold",
                ha="right",
                va="bottom",
            )
            for g_idx, (_, r) in enumerate(
                risk_table.reset_index(drop=True).iterrows()
            ):
                grp_label = str(r["Group"])
                y_pos = 0.80 - (g_idx * (0.80 / max(n_grps, 1)))
                grp_color = palette[g_idx % len(palette)]
                ax_risk.text(
                    0,
                    y_pos,
                    grp_label,
                    transform=ax_risk.get_yaxis_transform()
                    + mpl.transforms.ScaledTranslation(-0.24, 0, fig.dpi_scale_trans),
                    fontsize=8.0,
                    fontweight="bold",
                    color=grp_color,
                    ha="right",
                    va="center",
                )
                for m_val, col in zip(milestone_vals, time_cols):
                    ax_risk.text(
                        m_val,
                        y_pos,
                        str(r[col]),
                        fontsize=8.0,
                        color="#222222",
                        ha="center",
                        va="center",
                    )
            ax_risk.set_ylim(-0.15, 1.15)
        else:
            ax_risk.set_ylim(-0.5, n_grps + 0.5)
            # Header
            ax_risk.text(
                0,
                n_grps,
                "Number at risk",
                fontweight="bold",
                fontsize=9.5,
                ha="left",
                va="center",
            )
            for g_idx, (_, r) in enumerate(risk_table.iterrows()):
                grp_label = str(r["Group"])
                y_pos = n_grps - 1 - g_idx
                ax_risk.text(
                    -max_t * 0.02,
                    y_pos,
                    grp_label,
                    fontsize=9,
                    fontweight="500",
                    ha="right",
                    va="center",
                )
                for m_val, col in zip(milestone_vals, time_cols):
                    count_val = str(r[col])
                    ax_risk.text(
                        m_val, y_pos, count_val, fontsize=9, ha="center", va="center"
                    )

    ax_km.set_xlabel(f"Time ({time_unit})", labelpad=6)
    if not (has_risk_table and ax_risk is not None):
        fig.tight_layout()

    assert_zero_phi_text([title] + groups)
    qa = qa_layout(fig, check_outside=False)

    source_df = pd.DataFrame(curve_rows)
    if export_vector:
        png_path, vec_path, _ = save_figure_bundle(
            fig, out_path, dpi=dpi, export_vector=True
        )
    else:
        png_path = save_publication_figure(fig, out_path, dpi=dpi)
        vec_path = None
    csv_path = export_source_data(source_df, png_path)

    if has_risk_table:
        risk_table.to_csv(os.path.splitext(png_path)[0] + "_risktable.csv", index=False)

    alt_text = f"Kaplan-Meier survival curves comparing {len(groups)} groups over time."
    cens_desc = "Vertical ticks ('|')" if journal is not None else "'+' markers"
    caption = f"Figure. {title}. Step lines indicate cumulative survival probabilities with shaded 95% Greenwood confidence intervals. {cens_desc} denote censored observations."
    if log_rank_p is not None:
        if journal is not None:
            caption += f" Log-rank test {format_p_value(log_rank_p, journal)}."
        else:
            p_str = f"{log_rank_p:.3f}" if log_rank_p >= 0.001 else "< 0.001"
            caption += f" Log-rank test p-value: {p_str}."

    return FigureResult(
        png_path=png_path,
        alt_text=alt_text,
        caption=caption,
        source_df=source_df,
        csv_path=csv_path,
        vector_path=vec_path,
        qa=qa,
    )


def plot_cif(
    cif_data: Any,
    primary_label: str = "Event of Interest",
    competing_label: str = "Competing Event",
    show_naive_km: bool = True,
    time_unit: str = "Months",
    title: str = "Cumulative Incidence Function Under Competing Risks",
    out_path: str | Path = "figures/cif_plot.png",
    target: Literal["docx", "pptx", "web"] = "docx",
    journal: str | None = None,
    canvas: str = "manuscript",
    export_vector: bool = True,
) -> FigureResult:
    """
    Renders publication-grade Cumulative Incidence Function (CIF) step curves
    under competing risks with Greenwood 95% confidence intervals and Naive 1 - KM comparison.
    """
    if journal is not None:
        spec = apply_journal_theme(journal, canvas, target)
        palette = spec["palette"]
        w = 6.0
        h = 4.5
        dpi = 300
        color_primary = palette[0]
        color_competing = palette[1]
        color_naive = "#B24745"
    else:
        w, h, dpi = set_clinical_figure_style(target)
        color_primary = GROUP_COLORS[0]
        color_competing = GROUP_COLORS[1]
        color_naive = "#DC2626"

    # Extract DataFrame
    if hasattr(cif_data, "cif_table"):
        df = cif_data.cif_table.copy()
    elif isinstance(cif_data, pd.DataFrame):
        df = cif_data.copy()
    elif isinstance(cif_data, dict) and "cif_table" in cif_data:
        df = cif_data["cif_table"].copy()
    else:
        raise ValueError(
            "cif_data must be a CompetingRisksCIFResult, DataFrame, or dict with 'cif_table'."
        )

    fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)
    if journal is not None:
        style_axis(ax, journal)

    t = df["time"].to_numpy(dtype=float)
    cif1 = df["cif_primary"].to_numpy(dtype=float)
    cif1_l = (
        df["cif_primary_ci_lower"].to_numpy(dtype=float)
        if "cif_primary_ci_lower" in df.columns
        else cif1
    )
    cif1_u = (
        df["cif_primary_ci_upper"].to_numpy(dtype=float)
        if "cif_primary_ci_upper" in df.columns
        else cif1
    )

    cif2 = df["cif_competing"].to_numpy(dtype=float)
    cif2_l = (
        df["cif_competing_ci_lower"].to_numpy(dtype=float)
        if "cif_competing_ci_lower" in df.columns
        else cif2
    )
    cif2_u = (
        df["cif_competing_ci_upper"].to_numpy(dtype=float)
        if "cif_competing_ci_upper" in df.columns
        else cif2
    )

    # Conversion guard: if survival CI was passed (band at ~95%), convert and swap bounds
    if len(cif1_l) > 0 and np.nanmean(cif1_l) > 0.5 and np.nanmean(cif1) < 0.5:
        cif1_l, cif1_u = 1.0 - cif1_u, 1.0 - cif1_l
    if len(cif2_l) > 0 and np.nanmean(cif2_l) > 0.5 and np.nanmean(cif2) < 0.5:
        cif2_l, cif2_u = 1.0 - cif2_u, 1.0 - cif2_l

    if np.any(cif1_l > cif1 + 1e-6):
        raise ValueError(
            "Primary CIF lower confidence bound exceeds point estimate (ci_lower > cif)"
        )
    if np.any(cif2_l > cif2 + 1e-6):
        raise ValueError(
            "Competing CIF lower confidence bound exceeds point estimate (ci_lower > cif)"
        )

    # Primary CIF
    ax.step(
        t,
        cif1,
        where="post",
        color=color_primary,
        lw=1.8,
        label=f"CIF: {primary_label}",
    )
    ax.fill_between(t, cif1_l, cif1_u, step="post", alpha=0.15, color=color_primary)

    # Competing CIF
    ax.step(
        t,
        cif2,
        where="post",
        color=color_competing,
        lw=1.8,
        label=f"CIF: {competing_label}",
    )
    ax.fill_between(t, cif2_l, cif2_u, step="post", alpha=0.15, color=color_competing)

    # Naive 1 - KM
    if show_naive_km and "naive_1_minus_km" in df.columns:
        naive = df["naive_1_minus_km"].to_numpy(dtype=float)
        ax.step(
            t,
            naive,
            where="post",
            color=color_naive,
            lw=1.5,
            ls="--",
            label="Naive 1 - KM (Ignoring Competing Risks)",
        )

    max_cif = float(max(np.nanmax(cif1), np.nanmax(cif2)))
    if show_naive_km and "naive_1_minus_km" in df.columns:
        max_cif = max(max_cif, float(np.nanmax(df["naive_1_minus_km"])))

    ax.set_ylim(0.0, max(0.20, min(1.02, max_cif * 1.18)))
    ax.set_xlim(0.0, float(np.nanmax(t) * 1.02))
    ax.set_xlabel(f"Time ({time_unit})", labelpad=6)
    ax.set_ylabel("Cumulative Incidence Probability", labelpad=6)
    ax.set_title(title, loc="left", pad=12)
    ax.legend(loc="upper left", frameon=journal is None)

    fig.tight_layout()

    assert_zero_phi_text([title, primary_label, competing_label])
    qa = qa_layout(fig, check_outside=False)

    source_df = df.copy()
    if export_vector:
        png_path, vec_path, _ = save_figure_bundle(
            fig, out_path, dpi=dpi, export_vector=True
        )
    else:
        png_path = save_publication_figure(fig, out_path, dpi=dpi)
        vec_path = None
    csv_path = export_source_data(source_df, png_path)

    alt_text = f"Cumulative Incidence Function curve showing {primary_label} and {competing_label}."
    caption = f"Figure. {title}. Cumulative incidence estimated via Aalen-Johansen method with 95% confidence interval bands."
    if show_naive_km and "naive_1_minus_km" in df.columns:
        caption += " Dashed line demonstrates the overestimation bias of naive 1 - Kaplan-Meier estimation."

    return FigureResult(
        png_path=png_path,
        alt_text=alt_text,
        caption=caption,
        source_df=source_df,
        csv_path=csv_path,
        vector_path=vec_path,
        qa=qa,
    )

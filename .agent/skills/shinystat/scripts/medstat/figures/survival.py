"""
src/medstat/figures/survival.py: Publication-grade Kaplan-Meier curves with aligned risk table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from medstat.figures.base import FigureResult, export_source_data
from medstat.figures.styles import (
    GROUP_COLORS,
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
) -> FigureResult:
    """
    Renders publication Kaplan-Meier step-curves with Greenwood 95% CI bands,
    censored event tick marks, and an aligned numbers-at-risk table below the plot.
    """
    w, h, dpi = set_clinical_figure_style(target)

    has_risk_table = risk_table is not None and not risk_table.empty
    if has_risk_table:
        fig = plt.figure(figsize=(w, h + 1.2))
        gs = fig.add_gridspec(2, 1, height_ratios=[3.5, 1.0], hspace=0.25)
        ax_km = fig.add_subplot(gs[0])
        ax_risk = fig.add_subplot(gs[1])
    else:
        fig, ax_km = plt.subplots(figsize=(w, h))
        ax_risk = None

    curve_rows = []
    groups = list(km_curves.keys())

    max_t = 0.0
    for idx, (grp_name, data) in enumerate(km_curves.items()):
        color = GROUP_COLORS[idx % len(GROUP_COLORS)]
        t = np.asarray(data["timeline"], dtype=float)
        s = np.asarray(data["survival"], dtype=float)
        ci_l = np.asarray(data.get("ci_lower", s), dtype=float)
        ci_u = np.asarray(data.get("ci_upper", s), dtype=float)

        if len(t) > 0:
            max_t = max(max_t, float(np.max(t)))

        # Step survival curve
        ax_km.step(t, s, where="post", color=color, lw=2.0, label=grp_name)
        # 95% CI shaded band
        ax_km.fill_between(t, ci_l, ci_u, step="post", color=color, alpha=0.15)

        # Censored tick marks
        cens_t = np.asarray(data.get("censored_times", []), dtype=float)
        cens_s = np.asarray(data.get("censored_survival", []), dtype=float)
        if len(cens_t) > 0 and len(cens_s) > 0:
            ax_km.scatter(
                cens_t, cens_s, marker="+", s=28, color=color, alpha=0.85, zorder=4
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

    ax_km.set_ylim(-0.02, 1.04)
    ax_km.set_xlim(0, max(max_t, 1.0))
    ax_km.set_ylabel("Survival Probability", labelpad=6)
    ax_km.set_title(title, loc="left", pad=10)

    # Annotated Log-Rank p-value
    if log_rank_p is not None:
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

    ax_km.legend(loc="upper right")

    # Aligned Risk Table
    if has_risk_table and ax_risk is not None:
        ax_risk.axis("off")
        ax_km.set_xlabel("")  # Omit x label on main plot when table is present

        time_cols = [c for c in risk_table.columns if c != "Group"]
        n_grps = len(risk_table)
        ax_risk.set_ylim(-0.5, n_grps + 0.5)
        ax_risk.set_xlim(ax_km.get_xlim())

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

        # Parse milestone times from columns (e.g. "t=30.0")
        milestone_vals = []
        for c in time_cols:
            try:
                milestone_vals.append(float(c.replace("t=", "")))
            except ValueError:
                milestone_vals.append(0.0)

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

        # Set time axis on the lower risk table
        ax_risk.set_xlabel(f"Time ({time_unit})", labelpad=4)
    else:
        ax_km.set_xlabel(f"Time ({time_unit})", labelpad=6)
        fig.tight_layout()

    source_df = pd.DataFrame(curve_rows)
    png_path = save_publication_figure(fig, out_path, dpi=dpi)
    csv_path = export_source_data(source_df, png_path)

    alt_text = f"Kaplan-Meier survival curves comparing {len(groups)} groups over time."
    caption = f"Figure. {title}. Step lines indicate cumulative survival probabilities with shaded 95% Greenwood confidence intervals. '+' markers denote censored observations."
    if log_rank_p is not None:
        p_str = f"{log_rank_p:.3f}" if log_rank_p >= 0.001 else "< 0.001"
        caption += f" Log-rank test p-value: {p_str}."

    return FigureResult(
        png_path=png_path,
        alt_text=alt_text,
        caption=caption,
        source_df=source_df,
        csv_path=csv_path,
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
) -> FigureResult:
    """
    Renders publication-grade Cumulative Incidence Function (CIF) step curves
    under competing risks with Greenwood 95% confidence intervals and Naive 1 - KM comparison.
    """
    w, h, dpi = set_clinical_figure_style(target)

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

    fig, ax = plt.subplots(figsize=(w, h))

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

    color_primary = GROUP_COLORS[0]
    color_competing = GROUP_COLORS[1]
    color_naive = "#DC2626"

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
    ax.legend(loc="upper left")

    fig.tight_layout()

    source_df = df.copy()
    png_path = save_publication_figure(fig, out_path, dpi=dpi)
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
    )

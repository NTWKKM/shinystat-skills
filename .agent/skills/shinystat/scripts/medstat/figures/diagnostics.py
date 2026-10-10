"""
src/medstat/figures/diagnostics.py: Diagnostic and audit plots
(Missingness maps, Schoenfeld residuals, and MICE density overlays).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

from medstat.figures.base import FigureResult, export_source_data
from medstat.figures.styles import (
    CLINICAL_PALETTE,
    save_publication_figure,
    set_clinical_figure_style,
)


def plot_missingness_map(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    title: str = "Missing Data Pattern Matrix",
    out_path: str | Path = "figures/missingness_map.png",
    target: Literal["docx", "pptx", "web"] = "docx",
) -> FigureResult:
    """
    Renders missingness matrix heatmap across variables (Dark = Observed, Light = Missing).
    """
    w, h, dpi = set_clinical_figure_style(target)
    cols = columns or list(df.columns)
    sub = df[cols].copy()

    # Matrix: 0 = observed, 1 = missing
    matrix = sub.isna().to_numpy().astype(int)

    fig, ax = plt.subplots(figsize=(w, max(h, 4.0)))
    ax.imshow(matrix, aspect="auto", cmap="Blues", interpolation="nearest")

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9.5)
    ax.set_ylabel("Patient Index", labelpad=6)
    ax.set_title(title, loc="left", pad=10)

    fig.tight_layout()

    source_df = pd.DataFrame(
        [
            {
                "Variable": col,
                "Missing_Count": int(sub[col].isna().sum()),
                "Missing_Pct": float(sub[col].isna().mean() * 100),
            }
            for col in cols
        ]
    )
    png_path = save_publication_figure(fig, out_path, dpi=dpi)
    csv_path = export_source_data(source_df, png_path)

    alt_text = f"Heatmap showing missing data patterns across {len(cols)} variables."
    caption = f"Figure. {title}. Blue bands indicate missing data values across study participants."

    return FigureResult(
        png_path=png_path,
        alt_text=alt_text,
        caption=caption,
        source_df=source_df,
        csv_path=csv_path,
    )


def plot_schoenfeld_residuals(
    times: np.ndarray,
    residuals: np.ndarray,
    var_name: str,
    p_value: float | None = None,
    title: str = "Schoenfeld Residual Proportional Hazards Test",
    out_path: str | Path = "figures/schoenfeld_residuals.png",
    target: Literal["docx", "pptx", "web"] = "docx",
) -> FigureResult:
    """
    Plots scaled Schoenfeld residuals over time with a loess smoother to verify
    the Proportional Hazards assumption (flat slope indicates non-violation).
    """
    w, h, dpi = set_clinical_figure_style(target)
    fig, ax = plt.subplots(figsize=(w, h))

    t = np.asarray(times, dtype=float)
    r = np.asarray(residuals, dtype=float)

    # Scatter of residuals
    ax.scatter(
        t,
        r,
        alpha=0.45,
        color=CLINICAL_PALETTE["neutral"],
        s=22,
        label="Scaled Residuals",
    )

    # Loess smoother
    if len(t) > 5:
        smoothed = lowess(r, t, frac=0.6, it=2)
        ax.plot(
            smoothed[:, 0],
            smoothed[:, 1],
            color=CLINICAL_PALETTE["danger"],
            lw=2.0,
            label="Loess Trend",
        )

    # Horizontal zero line
    ax.axhline(0.0, color=CLINICAL_PALETTE["primary"], ls="--", lw=1.2)

    # Annotation
    if p_value is not None:
        p_str = (
            f"Grambsch-Therneau P = {p_value:.3f}"
            if p_value >= 0.001
            else "Grambsch-Therneau P < 0.001"
        )
        status_str = (
            "No PH violation detected (p > 0.05)"
            if p_value > 0.05
            else "Potential PH violation (p <= 0.05)"
        )
        ax.text(
            0.04,
            0.12,
            f"{p_str}\n{status_str}",
            transform=ax.transAxes,
            fontsize=9.5,
            fontweight="500",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="#FFFFFF", edgecolor="#CBD5E1"
            ),
        )

    ax.set_xlabel("Time (Follow-up Duration)", labelpad=6)
    ax.set_ylabel(f"Scaled Schoenfeld Residuals ({var_name})", labelpad=6)
    ax.set_title(f"{title}: {var_name}", loc="left", pad=10)
    ax.legend(loc="upper right")

    fig.tight_layout()

    source_df = pd.DataFrame({"Time": t, "Residual": r})
    png_path = save_publication_figure(fig, out_path, dpi=dpi)
    csv_path = export_source_data(source_df, png_path)

    alt_text = f"Schoenfeld residual plot for {var_name} evaluating proportional hazards assumption."
    caption = f"Figure. {title} for {var_name}. A horizontal loess trend is consistent with proportional hazards over time."

    return FigureResult(
        png_path=png_path,
        alt_text=alt_text,
        caption=caption,
        source_df=source_df,
        csv_path=csv_path,
    )


def plot_mice_diagnostics(
    observed_vals: np.ndarray,
    imputed_vals: list[np.ndarray] | np.ndarray,
    var_name: str,
    title: str = "MICE Imputation Diagnostic Density Overlay",
    out_path: str | Path = "figures/mice_density.png",
    target: Literal["docx", "pptx", "web"] = "docx",
) -> FigureResult:
    """
    Overlays observed distribution against MICE imputed distributions to verify
    plausibility and absence of extreme imputation distortion.
    """
    w, h, dpi = set_clinical_figure_style(target)
    fig, ax = plt.subplots(figsize=(w, h))

    obs = np.asarray(observed_vals, dtype=float)
    obs = obs[~np.isnan(obs)]

    # Plot observed density (histogram + KDE-style line)
    ax.hist(
        obs,
        bins=25,
        density=True,
        alpha=0.35,
        color=CLINICAL_PALETTE["primary"],
        label="Observed",
        edgecolor="none",
    )

    # Overlay imputed sets
    if isinstance(imputed_vals, list):
        imp_list = imputed_vals
    else:
        imp_list = [imputed_vals]

    cleaned_imputed: list[np.ndarray] = []
    for m_idx, imp in enumerate(imp_list):
        imp_clean = np.asarray(imp, dtype=float)
        imp_clean = imp_clean[~np.isnan(imp_clean)]
        cleaned_imputed.append(imp_clean)
        if len(imp_clean) > 0:
            lbl = (
                f"Imputed Set {m_idx + 1}"
                if len(imp_list) <= 3
                else ("Imputed Sets" if m_idx == 0 else None)
            )
            ax.hist(
                imp_clean,
                bins=25,
                density=True,
                histtype="step",
                lw=1.5,
                color=CLINICAL_PALETTE["secondary"],
                label=lbl,
            )

    ax.set_xlabel(f"{var_name} Value", labelpad=6)
    ax.set_ylabel("Probability Density", labelpad=6)
    ax.set_title(f"{title}: {var_name}", loc="left", pad=10)
    ax.legend(loc="upper right")

    fig.tight_layout()

    source_data: dict[str, pd.Series] = {"Observed": pd.Series(obs)}
    for m_idx, imp_clean in enumerate(cleaned_imputed):
        source_data[f"Imputed Set {m_idx + 1}"] = pd.Series(imp_clean)
    source_df = pd.DataFrame(source_data)
    png_path = save_publication_figure(fig, out_path, dpi=dpi)
    csv_path = export_source_data(source_df, png_path)

    alt_text = f"MICE density overlay comparing observed versus imputed distributions for {var_name}."
    caption = f"Figure. {title} for {var_name}. Overlapping density profiles suggest preservation of distributional properties without aberrant outliers."

    return FigureResult(
        png_path=png_path,
        alt_text=alt_text,
        caption=caption,
        source_df=source_df,
        csv_path=csv_path,
    )

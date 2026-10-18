from __future__ import annotations

import warnings
from pathlib import Path
from textwrap import fill as _wrap
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from medstat._phi import assert_zero_phi_text
from medstat.figures.base import FigureResult, export_source_data
from medstat.figures.journal import (
    apply_journal_theme,
    format_num,
    format_p_value,
)
from medstat.figures.qa import qa_layout
from medstat.figures.styles import (
    CLINICAL_PALETTE,
    save_figure_bundle,
    save_publication_figure,
    set_clinical_figure_style,
)


def plot_forest(
    data: list[dict[str, Any]] | pd.DataFrame,
    scale: Literal["OR", "HR", "RR", "Beta", "SHR", "CSHR", "MD"] = "OR",
    title: str = "Forest Plot of Effect Estimates",
    out_path: str | Path = "figures/forest_plot.png",
    target: Literal["docx", "pptx", "web"] = "docx",
    journal: str | None = None,
    canvas: str = "manuscript",
    export_vector: bool = True,
    null_value: float | None = None,
    favor_left_label: str = "Favors Intervention",
    favor_right_label: str = "Favors Control",
) -> FigureResult:
    """
    Renders an ICMJE/journal-standard Forest Plot with log-scale axis,
    95% CI error bars, and aligned right-hand numerical estimates.
    """
    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data.copy()
    df = df.reset_index(drop=True)

    # Map column aliases to 'term'
    if "term" not in df.columns:
        for alias in ["study", "Variable", "covariate", "label", "name"]:
            if alias in df.columns:
                df["term"] = df[alias]
                break

    required_cols = {"term", "estimate", "ci_lower", "ci_upper"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Forest plot data missing required columns: {missing}")

    is_ratio = scale.upper() in {"OR", "HR", "RR", "SHR", "CSHR", "IRR"}
    ref_val = (1.0 if is_ratio else 0.0) if null_value is None else null_value

    if is_ratio:
        for col in ["estimate", "ci_lower", "ci_upper"]:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if (vals <= 0).any():
                raise ValueError(
                    f"Ratio scale '{scale}' requires strictly positive {col} values (> 0)"
                )

    if journal is not None:
        spec = apply_journal_theme(journal, canvas, target)
        primary_color = spec["palette"][0]
        n_terms = len(df)
        w = 6.5
        if w < 5.0:
            warnings.warn(
                "single_col forest is too narrow for a 3-zone layout; using one_half_col (5.0 in)"
            )
            w = 5.0
        h = max(2.8, 1.2 + 0.36 * n_terms)
        dpi = 300
        fs = max(6.0, 8.5 * min(1.0, w / 6.5))

        fig = plt.figure(figsize=(w, h), dpi=dpi)
        gs = gridspec.GridSpec(1, 3, width_ratios=[2.6, 2.4, 2.8], wspace=0.04)
        ax_left = fig.add_subplot(gs[0])
        ax_mid = fig.add_subplot(gs[1], sharey=ax_left)
        ax_right = fig.add_subplot(gs[2], sharey=ax_left)

        for a in (ax_left, ax_right):
            a.axis("off")

        ax_mid.spines["top"].set_visible(False)
        ax_mid.spines["right"].set_visible(False)
        ax_mid.spines["left"].set_visible(False)
        ax_mid.yaxis.set_visible(False)

        if is_ratio:
            ax_mid.set_xscale("log")

        y_top = n_terms + 0.6
        for a in (ax_left, ax_mid, ax_right):
            a.axhline(n_terms + 0.15, color="#222222", linewidth=0.9)
            a.axhline(n_terms + 1.05, color="#222222", linewidth=1.2)

        has_events = "events_text" in df.columns
        ax_left.text(
            0.0,
            y_top,
            "Subgroup" if has_events else "Variable / Subgroup",
            fontweight="bold",
            fontsize=fs,
            va="center",
            ha="left",
        )
        if has_events:
            ax_left.text(
                0.98,
                y_top,
                "No./Total\n(%)",
                fontweight="bold",
                fontsize=fs,
                va="center",
                ha="right",
                linespacing=1.0,
            )

        scale_header = {
            "SHR": "Subdistribution HR (95% CI)",
            "CSHR": "Cause-Specific HR (95% CI)",
            "HR": "Hazard Ratio (95% CI)",
            "OR": "Odds Ratio (95% CI)",
            "RR": "Risk Ratio (95% CI)",
        }.get(scale.upper(), f"{scale} (95% CI)")

        ax_right.text(
            0.02,
            y_top,
            _wrap(scale_header, 15, break_long_words=False),
            fontweight="bold",
            fontsize=fs,
            va="center",
            ha="left",
            linespacing=1.0,
        )
        p_hdr = "p value" if journal.upper() == "LANCET" else "P Value"
        ax_right.text(
            0.98, y_top, p_hdr, fontweight="bold", fontsize=fs, va="center", ha="right"
        )

        ax_mid.axvline(
            ref_val, color="#555555", linestyle="--", linewidth=0.9, zorder=1
        )

        weights = (
            df["weight"].fillna(1.0).values
            if "weight" in df.columns
            else np.ones(n_terms)
        )
        w_norm = 4.5 + 3.5 * (weights / max(np.max(weights), 1e-6))

        for idx, row in df.iterrows():
            y = n_terms - idx - 0.5
            is_hdr = (
                bool(row["is_header"])
                if ("is_header" in row and pd.notna(row["is_header"]))
                else False
            )
            if idx % 2 == 1:
                for a in (ax_left, ax_mid, ax_right):
                    a.axhspan(
                        y - 0.45,
                        y + 0.45,
                        color="#F2EFE9" if canvas == "editorial" else "#F5F7F8",
                        zorder=0,
                    )

            term_str = str(row["term"])
            ax_left.text(
                0.0,
                y,
                term_str,
                fontweight="bold" if is_hdr else "normal",
                fontsize=fs,
                va="center",
                ha="left",
            )
            if has_events and pd.notna(row.get("events_text")):
                ax_left.text(
                    0.98,
                    y,
                    str(row["events_text"]),
                    fontsize=fs - 0.5,
                    va="center",
                    ha="right",
                )

            if not is_hdr and pd.notna(row.get("estimate")):
                est = float(row["estimate"])
                lo = float(row["ci_lower"])
                hi = float(row["ci_upper"])
                ax_mid.plot([lo, hi], [y, y], color="#222222", linewidth=1.2, zorder=3)
                ax_mid.plot(
                    est,
                    y,
                    marker="s",
                    color=primary_color,
                    markersize=w_norm[idx],
                    zorder=4,
                )

                j_upper = journal.upper() if journal else "NEJM"
                if lo < 0:
                    sep = " to "
                elif j_upper in ("NEJM", "BMJ"):
                    sep = " to "
                elif j_upper == "JAMA":
                    sep = "-"
                else:
                    sep = "–"

                est_txt = f"{format_num(est, 2, journal)} ({format_num(lo, 2, journal)}{sep}{format_num(hi, 2, journal)})"
                ax_right.text(
                    0.02, y, est_txt, fontsize=fs - 0.3, va="center", ha="left"
                )

                if "p_value" in row and pd.notna(row["p_value"]):
                    p_val_num = float(row["p_value"])
                    p_str = format_p_value(p_val_num, journal, bare=True)
                    ax_right.text(
                        0.98, y, p_str, fontsize=fs - 0.3, va="center", ha="right"
                    )

        ax_left.set_xlim(0, 1)
        ax_right.set_xlim(0, 1)
        ax_left.set_ylim(-0.4, n_terms + 1.3)

        if is_ratio:
            valid_lo = df["ci_lower"].dropna()
            valid_hi = df["ci_upper"].dropna()
            xmin = max(0.05, float(valid_lo.min()) * 0.8) if len(valid_lo) else 0.25
            xmax = min(20.0, float(valid_hi.max()) * 1.25) if len(valid_hi) else 4.0
            ax_mid.set_xlim(xmin, xmax)
            ticks = [
                t for t in [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0] if xmin <= t <= xmax
            ]
            if 1.0 not in ticks:
                ticks = sorted(ticks + [1.0])
            ax_mid.set_xticks(ticks)
            ax_mid.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
            ax_mid.get_xaxis().set_major_formatter(
                mpl.ticker.FuncFormatter(
                    lambda x, _: format_num(
                        x, 2 if x < 0.5 else (1 if x < 1 else 0), journal
                    )
                )
            )

        ax_mid.set_xlabel(scale_header, fontsize=fs, fontweight="bold")
        _x_tr = ax_mid.get_xaxis_transform()
        _shift = mpl.transforms.ScaledTranslation(3 / 72, 0, fig.dpi_scale_trans)
        _shift_inv = mpl.transforms.ScaledTranslation(-3 / 72, 0, fig.dpi_scale_trans)
        ax_mid.text(
            ref_val,
            -0.34,
            _wrap("\u2190 " + favor_left_label, 16),
            transform=_x_tr + _shift_inv,
            fontsize=fs - 1.2,
            ha="right",
            va="top",
            multialignment="right",
            linespacing=1.0,
        )
        ax_mid.text(
            ref_val,
            -0.34,
            _wrap(favor_right_label + " \u2192", 16),
            transform=_x_tr + _shift,
            fontsize=fs - 1.2,
            ha="left",
            va="top",
            multialignment="left",
            linespacing=1.0,
        )

        phi_texts = [title, favor_left_label, favor_right_label] + df["term"].astype(
            str
        ).tolist()
        if "events_text" in df.columns:
            phi_texts.extend(df["events_text"].dropna().astype(str).tolist())
        assert_zero_phi_text(phi_texts)
        qa = qa_layout(fig, check_outside=False)

        if export_vector:
            png_path, vec_path, _ = save_figure_bundle(
                fig, out_path, dpi=dpi, export_vector=True
            )
        else:
            png_path = save_publication_figure(fig, out_path, dpi=dpi)
            vec_path = None
        csv_path = export_source_data(df, png_path)

        alt_text = (
            f"{journal}-styled forest plot of {scale} estimates across {n_terms} terms."
        )
        caption = f"{title}. Points represent {scale} with horizontal 95% confidence intervals."

        return FigureResult(
            png_path=png_path,
            alt_text=alt_text,
            caption=caption,
            source_df=df,
            csv_path=csv_path,
            vector_path=vec_path,
            qa=qa,
        )

    # Backward compatible default branch (journal is None)
    w, h, dpi = set_clinical_figure_style(target)

    # Reverse order so top row appears at top of plot
    df_rev = df.iloc[::-1].reset_index(drop=True)
    n_terms = len(df_rev)

    # Adjust height based on number of terms
    dynamic_height = max(h, 0.45 * n_terms + 2.0)
    fig, (ax_plot, ax_text) = plt.subplots(
        1, 2, figsize=(w, dynamic_height), gridspec_kw={"width_ratios": [3.2, 2.0]}
    )

    y_positions = np.arange(n_terms)
    estimates = df_rev["estimate"].to_numpy(dtype=float)
    ci_lowers = df_rev["ci_lower"].to_numpy(dtype=float)
    ci_uppers = df_rev["ci_upper"].to_numpy(dtype=float)
    terms = df_rev["term"].astype(str).tolist()

    # Plot confidence interval lines and point markers
    for idx, y in enumerate(y_positions):
        est = estimates[idx]
        lo = ci_lowers[idx]
        hi = ci_uppers[idx]

        # Horizontal error bar
        ax_plot.plot(
            [lo, hi],
            [y, y],
            color=CLINICAL_PALETTE["primary"],
            lw=1.8,
            solid_capstyle="round",
        )
        # Estimate marker
        ax_plot.plot(
            est,
            y,
            marker="s",
            markersize=6.5,
            color=CLINICAL_PALETTE["secondary"],
            zorder=5,
        )

    # Vertical reference line of no effect
    ax_plot.axvline(
        ref_val,
        color=CLINICAL_PALETTE["danger"],
        ls="--",
        lw=1.2,
        alpha=0.85,
        label=f"Null effect ({ref_val})",
    )

    # X-axis configuration
    if is_ratio:
        ax_plot.set_xscale("log")
        ax_plot.set_xlabel(f"{scale} (log scale)", labelpad=6)
    else:
        ax_plot.set_xlabel(f"{scale} Effect Size", labelpad=6)

    ax_plot.set_yticks(y_positions)
    ax_plot.set_yticklabels(terms, fontweight="500")
    ax_plot.set_ylim(-0.8, n_terms - 0.2)
    ax_plot.set_title(title, loc="left", pad=12)

    # Right text column: numerical summary
    ax_text.axis("off")
    ax_text.set_ylim(-0.8, n_terms - 0.2)

    # Header in text pane
    header_str = f"{scale} (95% CI)"
    ax_text.text(
        0.05,
        n_terms - 0.2,
        header_str,
        fontweight="bold",
        fontsize=10,
        va="bottom",
        ha="left",
    )
    if "p_value" in df_rev.columns:
        ax_text.text(
            0.85,
            n_terms - 0.2,
            "P-value",
            fontweight="bold",
            fontsize=10,
            va="bottom",
            ha="right",
        )

    for idx, y in enumerate(y_positions):
        est_str = f"{estimates[idx]:.2f} [{ci_lowers[idx]:.2f}, {ci_uppers[idx]:.2f}]"
        ax_text.text(
            0.05,
            y,
            est_str,
            va="center",
            ha="left",
            fontsize=9.5,
            fontfamily="monospace",
        )
        if "p_value" in df_rev.columns:
            pval = df_rev.loc[idx, "p_value"]
            if pd.isna(pval):
                p_str = "—"
            elif isinstance(pval, (int, float, np.number)):
                val_f = float(pval)
                p_str = f"{val_f:.3f}" if val_f >= 0.001 else "<0.001"
            else:
                p_str = str(pval)
            ax_text.text(0.85, y, p_str, va="center", ha="right", fontsize=9.5)

    fig.tight_layout()

    assert_zero_phi_text([title] + terms)
    qa = qa_layout(fig, check_outside=False)

    rev_df = df_rev.iloc[::-1].reset_index(drop=True)
    if export_vector:
        png_path, vec_path, _ = save_figure_bundle(
            fig, out_path, dpi=dpi, export_vector=True
        )
    else:
        png_path = save_publication_figure(fig, out_path, dpi=dpi)
        vec_path = None
    csv_path = export_source_data(rev_df, png_path)

    alt_text = f"Forest plot displaying {n_terms} effect estimates ({scale}) with 95% confidence intervals."
    caption = f"Figure. {title}. Squares represent point estimates ({scale}) with horizontal lines indicating 95% confidence intervals. The dashed vertical line denotes null effect ({ref_val})."

    return FigureResult(
        png_path=png_path,
        alt_text=alt_text,
        caption=caption,
        source_df=rev_df,
        csv_path=csv_path,
        vector_path=vec_path,
        qa=qa,
    )

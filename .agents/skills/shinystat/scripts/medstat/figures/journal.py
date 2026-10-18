"""
src/medstat/figures/journal.py: Journal House Style Engine, Formatters, and Sizing Rules.
Supports NEJM, JAMA, LANCET, BMJ, and CLINICAL (shinystat canonical) styles.
"""

from __future__ import annotations

import re
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

JOURNAL_SPECS: dict[str, dict[str, Any]] = {
    "NEJM": {
        "palette": [
            "#BC3C29",
            "#0072B5",
            "#E18727",
            "#20854E",
            "#7876B1",
            "#6F99AD",
            "#FFDC91",
            "#EE4C97",
        ],
        "bg_editorial": "#FBF7F0",
        "bg_manuscript": "#FFFFFF",
        "grid_y": False,
        "spine_lw": 0.8,
        "tick_dir": "out",
        "decimal_sep": ".",
        "p_prefix": "P",
        "leading_zero_p": True,
    },
    "JAMA": {
        "palette": [
            "#374E55",
            "#DF8F44",
            "#00A1D5",
            "#B24745",
            "#79AF97",
            "#6A6599",
            "#80796B",
            "#0073C2",
        ],
        "bg_editorial": "#FFFFFF",
        "bg_manuscript": "#FFFFFF",
        "grid_y": True,
        "spine_lw": 0.8,
        "tick_dir": "out",
        "decimal_sep": ".",
        "p_prefix": "P",
        "leading_zero_p": False,  # AMA style: P = .03, P < .001
    },
    "LANCET": {
        "palette": [
            "#00468B",
            "#ED0000",
            "#42B540",
            "#0099B4",
            "#925E9F",
            "#FDAF91",
            "#AD002A",
            "#ADB6B6",
        ],
        "bg_editorial": "#FFFFFF",
        "bg_manuscript": "#FFFFFF",
        "grid_y": False,
        "spine_lw": 0.8,
        "tick_dir": "out",
        "decimal_sep": "\u00b7",  # Lancet mid-dot: 0·85
        "p_prefix": "p",
        "leading_zero_p": True,
    },
    "BMJ": {
        "palette": [
            "#003366",
            "#D52B1E",
            "#008080",
            "#E87722",
            "#6F2C91",
            "#2D68C4",
            "#708090",
            "#118AB2",
        ],
        "bg_editorial": "#FFFFFF",
        "bg_manuscript": "#FFFFFF",
        "grid_y": False,
        "spine_lw": 0.8,
        "tick_dir": "out",
        "decimal_sep": ".",
        "p_prefix": "P",
        "leading_zero_p": True,
    },
    "CLINICAL": {
        "palette": [
            "#2563EB",
            "#DC2626",
            "#059669",
            "#D97706",
            "#7C3AED",
            "#0891B2",
            "#475569",
            "#0F172A",
        ],
        "bg_editorial": "#FFFFFF",
        "bg_manuscript": "#FFFFFF",
        "grid_y": True,
        "spine_lw": 0.8,
        "tick_dir": "out",
        "decimal_sep": ".",
        "p_prefix": "P",
        "leading_zero_p": True,
    },
    "OKABE_ITO": {
        "palette": [
            "#0072B2",
            "#D55E00",
            "#009E73",
            "#CC79A7",
            "#E69F00",
            "#56B4E9",
            "#F0E442",
            "#000000",
        ],
        "bg_editorial": "#FFFFFF",
        "bg_manuscript": "#FFFFFF",
        "grid_y": False,
        "spine_lw": 0.8,
        "tick_dir": "out",
        "decimal_sep": ".",
        "p_prefix": "P",
        "leading_zero_p": True,
    },
}

TARGET_WIDTHS: dict[str, float] = {
    "single_col": 3.5,
    "one_half_col": 5.0,
    "double_col": 7.0,
    "docx": 6.5,
    "pptx": 11.5,
    "web": 7.5,
}


def apply_journal_theme(
    journal: str = "NEJM",
    canvas: str = "manuscript",
    target: str | None = None,
) -> dict[str, Any]:
    """
    Configures matplotlib rcParams for NEJM, JAMA, LANCET, BMJ, or CLINICAL style.
    Returns the resolved journal specification dictionary.
    """
    j_key = journal.upper() if journal and journal.upper() in JOURNAL_SPECS else "NEJM"
    spec = JOURNAL_SPECS[j_key]
    bg = spec["bg_editorial"] if canvas == "editorial" else spec["bg_manuscript"]

    font_chain = [
        "Arial",
        "Helvetica",
        "DejaVu Sans",
        "Sarabun",
        "Thonburi",
        "Sukhumvit Set",
        "Tahoma",
        "sans-serif",
    ]

    base_font = 9.0
    if target == "pptx":
        base_font = 13.0
    elif target in ("web", "docx"):
        base_font = 10.0

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": font_chain,
            "axes.unicode_minus": False,
            "axes.edgecolor": "#222222",
            "axes.linewidth": spec["spine_lw"],
            "axes.facecolor": bg,
            "figure.facecolor": bg,
            "savefig.facecolor": bg,
            "xtick.direction": spec["tick_dir"],
            "ytick.direction": spec["tick_dir"],
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": spec["spine_lw"],
            "ytick.major.width": spec["spine_lw"],
            "xtick.labelsize": base_font - 1.0,
            "ytick.labelsize": base_font - 1.0,
            "axes.labelsize": base_font + 0.5,
            "axes.titlesize": base_font + 1.0,
            "legend.fontsize": base_font - 1.0,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return spec


def format_num(
    val: float,
    decimals: int = 2,
    journal: str = "NEJM",
    drop_leading_zero: bool = False,
) -> str:
    """Format numeric value according to NEJM, JAMA (AMA), BMJ, or Lancet house rules."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "NA"
    s = f"{val:.{decimals}f}"
    if drop_leading_zero and abs(val) < 1.0:
        s = re.sub(r"^(-?)0\.", r"\1.", s)
    if journal and journal.upper() == "LANCET":
        s = s.replace(".", "\u00b7")
    return s


def format_p_value(p: float, journal: str = "NEJM", bare: bool = False) -> str:
    """
    Format P-value per journal conventions.
    bare=True drops the 'P'/'p' prefix and '=' for table columns (keeps '<' or '>').
    """
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    j = journal.upper() if journal else "NEJM"

    if j == "LANCET":
        if p < 0.0001:
            s_ = "<0\u00b70001"
        elif p < 0.001:
            s_ = f"{p:.4f}".replace(".", "\u00b7")
        elif p < 0.01:
            s_ = f"{p:.3f}".replace(".", "\u00b7")
        else:
            s_ = f"{p:.2f}".replace(".", "\u00b7")
        if bare:
            return s_
        return f"p{s_}" if s_.startswith("<") else f"p={s_}"

    if j == "JAMA":
        if p < 0.001:
            s_ = "<.001"
        elif p < 0.01:
            s_ = format_num(p, 3, "JAMA", drop_leading_zero=True)
        else:
            s_ = format_num(p, 2, "JAMA", drop_leading_zero=True)
        if p > 0.99:
            s_ = ">.99"
        if bare:
            return s_
        if s_[0] in "<>":
            return f"P {s_[0]} {s_[1:]}"
        return f"P = {s_}"

    if j == "BMJ":
        if p < 0.001:
            s_ = "< 0.001"
            return s_ if bare else f"P {s_}"
        if p < 0.01:
            s_ = f"{p:.3f}"
        else:
            s_ = f"{p:.2f}"
        return s_ if bare else f"P = {s_}"

    # NEJM and CLINICAL / DEFAULT
    if p < 0.001:
        s_ = "<0.001"
    elif p < 0.01:
        s_ = f"{p:.3f}"
    else:
        s_ = f"{p:.2f}"
    if bare:
        return s_
    return f"P{s_}" if s_.startswith("<") else f"P={s_}"


def format_estimate(
    est: float,
    lcl: float,
    ucl: float,
    metric: str = "HR",
    decimals: int = 2,
    journal: str = "NEJM",
) -> str:
    """Formats an effect estimate with 95% CI according to journal style."""
    j = journal.upper() if journal else "NEJM"

    if j == "LANCET":
        e_s = format_num(est, decimals, "LANCET")
        l_s = format_num(lcl, decimals, "LANCET")
        u_s = format_num(ucl, decimals, "LANCET")
        return f"{metric} {e_s} (95% CI {l_s}–{u_s})"

    if j == "JAMA":
        e_s = format_num(est, decimals, "JAMA", drop_leading_zero=False)
        l_s = format_num(lcl, decimals, "JAMA", drop_leading_zero=False)
        u_s = format_num(ucl, decimals, "JAMA", drop_leading_zero=False)
        return f"{metric}, {e_s} (95% CI, {l_s}-{u_s})"

    if j == "BMJ":
        e_s = format_num(est, decimals, "BMJ")
        l_s = format_num(lcl, decimals, "BMJ")
        u_s = format_num(ucl, decimals, "BMJ")
        return f"{metric} {e_s} (95% CI {l_s} to {u_s})"

    if j == "NEJM":
        e_s = format_num(est, decimals, "NEJM")
        l_s = format_num(lcl, decimals, "NEJM")
        u_s = format_num(ucl, decimals, "NEJM")
        return f"{metric}, {e_s} (95% CI, {l_s} to {u_s})"

    # CLINICAL / DEFAULT
    e_s = format_num(est, decimals, "CLINICAL")
    l_s = format_num(lcl, decimals, "CLINICAL")
    u_s = format_num(ucl, decimals, "CLINICAL")
    return f"{metric} {e_s} (95% CI: {l_s}–{u_s})"


def style_axis(ax: plt.Axes, journal: str = "NEJM") -> None:
    """Apply journal L-frame spine and grid rules to a matplotlib Axes."""
    j_key = journal.upper() if journal and journal.upper() in JOURNAL_SPECS else "NEJM"
    spec = JOURNAL_SPECS[j_key]
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if spec["grid_y"]:
        ax.yaxis.grid(True, linestyle=":", linewidth=0.6, color="#D9D9D9", zorder=0)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)
    if j_key == "LANCET":
        ax.xaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda x, _: f"{x:g}".replace(".", "\u00b7"))
        )
        ax.yaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda y, _: f"{y:g}".replace(".", "\u00b7"))
        )

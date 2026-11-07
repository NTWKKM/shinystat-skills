# Journal House Styles & Artwork Reference (Cloud Sandbox Edition)

Reached from `SKILL.md` Pillar 5 and **Output Format & House Style Alignment Gate** (#5). Provides self-contained Python recipes for applying journal **house styles** (`NEJM`, `JAMA`, `LANCET`, `BMJ`, `CLINICAL`, `OKABE_ITO`) in cloud sandboxes (`matplotlib`, `numpy`, and `pandas` only; zero `medstat` imports).

> **Provenance**: The color palettes follow the widely used `ggsci` journal-inspired sets (plus Okabe-Ito for universal color-vision-deficiency accessibility), not official publisher color mandates. Journals specify resolution, vector typography, column widths, and file formats rather than fixed hex palettes; check the target journal's current author artwork guidelines before submission.

---

## 1. Journal Style Specifications, Palettes & Target Widths

```python
from pathlib import Path
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

JOURNAL_SPECS = {
    "NEJM": {
        "palette": ["#BC3C29", "#0072B5", "#E18727", "#20854E", "#7876B1", "#6F99AD", "#FFDC91", "#EE4C97"],
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
        "palette": ["#374E55", "#DF8F44", "#00A1D5", "#B24745", "#79AF97", "#6A6599", "#80796B", "#0073C2"],
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
        "palette": ["#00468B", "#ED0000", "#42B540", "#0099B4", "#925E9F", "#FDAF91", "#AD002A", "#ADB6B6"],
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
        "palette": ["#003366", "#D52B1E", "#008080", "#E87722", "#6F2C91", "#2D68C4", "#708090", "#118AB2"],
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
        "palette": ["#2563EB", "#DC2626", "#059669", "#D97706", "#7C3AED", "#0891B2", "#475569", "#0F172A"],
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
        "palette": ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"],
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

TARGET_WIDTHS = {
    "single_col": 3.5,
    "one_half_col": 5.0,
    "double_col": 7.0,
    "docx": 6.5,
    "pptx": 11.5,
    "web": 7.5,
}
```

---

## 2. Recipe: Applying Journal Theme & L-Frame Axis Rules

```python
def apply_standalone_journal_theme(
    journal: str = "NEJM", canvas: str = "manuscript", target: str | None = None
) -> dict:
    """Configures global matplotlib rcParams to match the selected house style."""
    j_key = journal.upper() if journal and journal.upper() in JOURNAL_SPECS else "NEJM"
    spec = JOURNAL_SPECS[j_key]
    bg = spec["bg_editorial"] if canvas == "editorial" else spec["bg_manuscript"]
    base_font = 13.0 if target == "pptx" else (10.0 if target in ("web", "docx") else 9.0)

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Sarabun", "Thonburi", "Sukhumvit Set", "Tahoma", "sans-serif"],
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
        "xtick.labelsize": base_font - 1.0,
        "ytick.labelsize": base_font - 1.0,
        "axes.labelsize": base_font + 0.5,
        "axes.titlesize": base_font + 1.0,
        "legend.fontsize": base_font - 1.0,
        "legend.frameon": False,
        "pdf.fonttype": 42,  # TrueType vector text
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    return spec


def style_standalone_axis(ax, journal: str = "NEJM") -> None:
    """Applies L-frame spine, dotted grid (JAMA/CLINICAL), and Lancet mid-dot ticks."""
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
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{x:g}".replace(".", "\u00b7")))
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: f"{y:g}".replace(".", "\u00b7")))
```

---

## 3. Recipe: Number, Estimate & P-Value Formatters

```python
def format_standalone_p(p: float, journal: str = "NEJM") -> str:
    j = journal.upper() if journal else "NEJM"
    if j == "LANCET":
        if p < 0.0001:
            return "p<0\u00b70001"
        fmt = f"{p:.4f}" if p < 0.001 else (f"{p:.3f}" if p < 0.01 else f"{p:.2f}")
        return "p=" + fmt.replace(".", "\u00b7")
    if j == "JAMA":
        if p < 0.001:
            return "P < .001"
        if p > 0.99:
            return "P > .99"
        fmt = f"{p:.3f}" if p < 0.01 else f"{p:.2f}"
        return "P = " + re.sub(r"^0\.", ".", fmt)
    if j == "BMJ":
        if p < 0.001:
            return "P < 0.001"
        return f"P = {p:.3f}" if p < 0.01 else f"P = {p:.2f}"
    if p < 0.001:
        return "P<0.001"
    return f"P={p:.3f}" if p < 0.01 else f"P={p:.2f}"
```

---

## 4. Recipe: Exporting Multi-Format Bundle (PNG, TrueType PDF, SVG, CSV)

```python
def save_standalone_bundle(fig, out_path: str, source_df=None, dpi: int = 300) -> dict:
    """Saves 300/600 DPI PNG, TrueType vector PDF (fonttype=42), SVG, and traceable _source.csv."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    png_path = p.with_suffix(".png")
    pdf_path = p.with_suffix(".pdf")
    svg_path = p.with_suffix(".svg")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(svg_path, format="svg", bbox_inches="tight", facecolor=fig.get_facecolor())
    csv_path = p.with_name(p.stem + "_source.csv")
    if source_df is not None:
        source_df.to_csv(csv_path, index=False)
    return {"png_path": str(png_path), "vector_path": str(pdf_path), "svg_path": str(svg_path), "csv_path": str(csv_path)}
```

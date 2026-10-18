# Journal House Styles & Artwork Reference (Cloud Sandbox Edition)

This reference provides self-contained, standalone Python recipes for applying publication house styles (NEJM, JAMA, Lancet, BMJ, Clinical) in cloud sandboxes without external library dependencies (`matplotlib`, `numpy`, and `pandas` only).

---

## 1. Journal Style Specifications & Palettes

```python
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

JOURNAL_SPECS = {
    "NEJM": {
        "palette": ["#941100", "#1A365D", "#2D3748", "#4A5568", "#718096"],
        "font_family": ["Arial", "Helvetica", "Sarabun", "sans-serif"],
        "tick_direction": "out",
        "tick_length": 3.5,
        "l_frame": True,
        "bg_manuscript": "#FFFFFF",
        "bg_editorial": "#FDFBF7",
    },
    "JAMA": {
        "palette": ["#002D62", "#BA0C2F", "#5C768D", "#C59B27", "#1D252D"],
        "font_family": ["Arial", "Helvetica", "Sarabun", "sans-serif"],
        "tick_direction": "out",
        "tick_length": 3.0,
        "l_frame": True,
        "bg_manuscript": "#FFFFFF",
        "bg_editorial": "#FFFFFF",
    },
    "LANCET": {
        "palette": ["#D62728", "#1F77B4", "#FF7F0E", "#2CA02C", "#9467BD"],
        "font_family": ["Arial", "Helvetica", "Sarabun", "sans-serif"],
        "tick_direction": "out",
        "tick_length": 3.5,
        "l_frame": True,
        "bg_manuscript": "#FFFFFF",
        "bg_editorial": "#FAFAFA",
    },
    "BMJ": {
        "palette": ["#004F9F", "#E05A47", "#2E8B57", "#F4A261", "#6C757D"],
        "font_family": ["Arial", "Helvetica", "Sarabun", "sans-serif"],
        "tick_direction": "out",
        "tick_length": 3.0,
        "l_frame": True,
        "bg_manuscript": "#FFFFFF",
        "bg_editorial": "#F8F9FA",
    },
    "CLINICAL": {
        "palette": ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"],
        "font_family": ["Arial", "Helvetica", "Sarabun", "sans-serif"],
        "tick_direction": "out",
        "tick_length": 3.0,
        "l_frame": False,
        "bg_manuscript": "#FFFFFF",
        "bg_editorial": "#FFFFFF",
    },
}
```

---

## 2. Recipe: Applying Journal Theme to Matplotlib

```python
def apply_standalone_journal_theme(journal: str = "NEJM", canvas: str = "manuscript"):
    """Configures global matplotlib rcParams to match journal house style."""
    j = journal.upper() if journal and journal.upper() in JOURNAL_SPECS else "NEJM"
    spec = JOURNAL_SPECS[j]
    bg = spec["bg_editorial"] if canvas == "editorial" else spec["bg_manuscript"]

    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = spec["font_family"]
    mpl.rcParams["figure.facecolor"] = bg
    mpl.rcParams["axes.facecolor"] = bg
    mpl.rcParams["axes.edgecolor"] = "#222222"
    mpl.rcParams["axes.linewidth"] = 0.8
    mpl.rcParams["xtick.direction"] = spec["tick_direction"]
    mpl.rcParams["ytick.direction"] = spec["tick_direction"]
    mpl.rcParams["xtick.major.size"] = spec["tick_length"]
    mpl.rcParams["ytick.major.size"] = spec["tick_length"]
    mpl.rcParams["pdf.fonttype"] = 42  # TrueType vector fonts for editable text
    mpl.rcParams["ps.fonttype"] = 42
```

---

## 3. Recipe: L-Frame Spine Styling

```python
def style_standalone_axis(ax, journal: str = "NEJM"):
    """Applies crisp L-frame spine (left and bottom only) for top-tier clinical journals."""
    j = journal.upper() if journal and journal.upper() in JOURNAL_SPECS else "NEJM"
    spec = JOURNAL_SPECS[j]
    if spec["l_frame"]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_position(("outward", 3.0))
        ax.spines["bottom"].set_position(("outward", 3.0))
    ax.tick_params(axis="both", which="both", direction="out", color="#222222", length=spec["tick_length"])
```

---

## 4. Recipe: Exporting Multi-Format Bundle (PNG, PDF, SVG, CSV)

```python
def save_standalone_bundle(fig, out_path: str, source_df=None, dpi: int = 300):
    """Saves high-DPI raster, vector PDF (TrueType), and source data CSV."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    # High-DPI PNG
    png_path = p.with_suffix(".png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    
    # TrueType Vector PDF
    pdf_path = p.with_suffix(".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
    
    # Source CSV for numerical traceability
    csv_path = p.with_name(p.stem + "_source.csv")
    if source_df is not None:
        source_df.to_csv(csv_path, index=False)
        
    return {"png": str(png_path), "pdf": str(pdf_path), "csv": str(csv_path)}
```

"""
src/medstat/figures/styles.py: Centralized clinical styling, colorblind palettes,
and font fallback chains for publication-grade figure generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt

# Use Agg backend for headless generation
matplotlib.use("Agg")

# Colorblind-safe publication palette (Okabe-Ito / Clinical Slate)
CLINICAL_PALETTE = {
    "primary": "#0F172A",  # Slate 900
    "secondary": "#2563EB",  # Blue 600
    "accent": "#059669",  # Emerald 600
    "warning": "#D97706",  # Amber 600
    "danger": "#DC2626",  # Red 600
    "purple": "#7C3AED",  # Purple 600
    "neutral": "#64748B",  # Slate 500
    "light_neutral": "#E2E8F0",  # Slate 200
    "bg_subtle": "#F8FAFC",  # Slate 50
    "text": "#0F172A",  # Slate 900
}

# Distinct colorblind palette for multi-group curves
GROUP_COLORS = [
    "#2563EB",  # Blue
    "#DC2626",  # Red
    "#059669",  # Green
    "#D97706",  # Amber
    "#7C3AED",  # Purple
    "#0891B2",  # Cyan
    "#475569",  # Slate
]

# Standard Publication Dimensions (in inches at 300 DPI)
DIMENSIONS = {
    "docx": {"width": 6.5, "height": 4.5, "dpi": 300},
    "pptx": {"width": 12.0, "height": 6.75, "dpi": 300},
    "web": {"width": 8.0, "height": 5.0, "dpi": 150},
}


def configure_matplotlib_fonts() -> list[str]:
    """
    Configures Matplotlib font family with graceful Thai and Unicode fallback chain.
    """
    font_chain = [
        "Sarabun",
        "Thonburi",
        "Sukhumvit Set",
        "Ayuthaya",
        "Tahoma",
        "Arial Unicode MS",
        "DejaVu Sans",
        "sans-serif",
    ]
    plt.rcParams["font.sans-serif"] = font_chain
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    return font_chain


def set_clinical_figure_style(
    target: Literal["docx", "pptx", "web"] = "docx",
) -> tuple[float, float, int]:
    """
    Applies publication-grade typography, clean spines, and sizing rules.
    Returns (width, height, dpi).
    """
    configure_matplotlib_fonts()

    dim = DIMENSIONS.get(target, DIMENSIONS["docx"])
    base_font_size = 11.0 if target == "docx" else 13.0

    plt.rcParams.update(
        {
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#94A3B8",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#F1F5F9",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "axes.titlesize": base_font_size + 2.0,
            "axes.titleweight": "600",
            "axes.labelsize": base_font_size,
            "axes.labelweight": "500",
            "xtick.labelsize": base_font_size - 1.5,
            "ytick.labelsize": base_font_size - 1.5,
            "legend.fontsize": base_font_size - 1.5,
            "legend.frameon": True,
            "legend.facecolor": "#FFFFFF",
            "legend.edgecolor": "#E2E8F0",
            "legend.framealpha": 0.95,
            "figure.dpi": dim["dpi"],
            "savefig.dpi": dim["dpi"],
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )

    return dim["width"], dim["height"], dim["dpi"]


def save_publication_figure(
    fig: plt.Figure,
    out_path: str | Path,
    dpi: int = 300,
) -> str:
    """
    Saves figure cleanly, ensuring parent directories exist, and closes the figure.
    """
    p = Path(out_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(p), dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(p)

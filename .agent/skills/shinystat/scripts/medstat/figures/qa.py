"""
src/medstat/figures/qa.py: Layout QA Gate, Colorblind Simulation (CVD), and Zero-PHI verification.
"""

from __future__ import annotations

from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from medstat._phi import PHI_PATTERNS, assert_zero_phi_text, scan_phi_violations

# Default must-differ tissue & clinical pairs
TISSUE_MUST_DIFFER = [
    ("epithelial", "granulation"),
    ("granulation", "slough"),
    ("slough", "necrotic"),
    ("granulation", "necrotic"),
    ("epithelial", "slough"),
    ("rbc", "bacteria"),
    ("rbc", "platelet"),
    ("lipid", "cap"),
]

# Color vision deficiency matrices (Machado et al. 2009, severity 1.0)
_CVD = {
    "deuteranopia": np.array(
        [
            [0.367322, 0.860646, -0.227968],
            [0.280085, 0.672501, 0.047413],
            [-0.011820, 0.042940, 0.968881],
        ]
    ),
    "protanopia": np.array(
        [
            [0.152286, 1.052583, -0.204868],
            [0.114503, 0.786281, 0.099216],
            [-0.003882, -0.048116, 1.051998],
        ]
    ),
    "tritanopia": np.array(
        [
            [1.255528, -0.076749, -0.178779],
            [-0.078411, 0.930809, 0.147602],
            [0.004733, 0.691367, 0.303900],
        ]
    ),
}


def _lin(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)])


def _rgb_to_lab(rgb_lin: np.ndarray) -> np.ndarray:
    M = np.array(
        [
            [0.4124, 0.3576, 0.1805],
            [0.2126, 0.7152, 0.0722],
            [0.0193, 0.1192, 0.9505],
        ]
    )
    xyz = M @ rgb_lin / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    return np.array([116.0 * f[1] - 16.0, 500.0 * (f[0] - f[1]), 200.0 * (f[1] - f[2])])


def cvd_check(
    pairs: Sequence[tuple[str, str]] | None = None,
    palette: dict[str, str] | None = None,
    min_de: float = 12.0,
) -> pd.DataFrame:
    """
    Computes Delta-E76 color difference between must-differ color pairs under normal
    vision + 3 CVD types (deuteranopia, protanopia, tritanopia).
    Pairs with deltaE < min_de are flagged with needs_cue=True (require hatch, shape, or label).
    """
    if pairs is None:
        pairs = TISSUE_MUST_DIFFER
    if palette is None:
        from medstat.figures.styles import CLINICAL_PALETTE

        palette = CLINICAL_PALETTE

    rows: list[dict[str, Any]] = []
    for a, b in pairs:
        if a not in palette or b not in palette:
            continue
        c_a = _hex_to_rgb(palette[a])
        c_b = _hex_to_rgb(palette[b])

        for mode in ("normal", *_CVD):
            la, lb = _lin(c_a), _lin(c_b)
            if mode != "normal":
                la = np.clip(_CVD[mode] @ la, 0, 1)
                lb = np.clip(_CVD[mode] @ lb, 0, 1)
            delta_e = round(float(np.linalg.norm(_rgb_to_lab(la) - _rgb_to_lab(lb))), 1)
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "vision": mode,
                    "deltaE": delta_e,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["needs_cue"] = df["deltaE"] < min_de
    else:
        df = pd.DataFrame(columns=["a", "b", "vision", "deltaE", "needs_cue"])
    return df


def qa_layout(
    fig: plt.Figure,
    check_outside: bool = False,
    tol_px: float = 2.0,
    min_font: float = 6.0,
) -> dict[str, Any]:
    """
    Layout gate for figures: detects overlapping text, boundary-clipped text,
    and text below min_font size.
    check_outside=False (default) is recommended for figures saved with bbox_inches='tight',
    as matplotlib automatically expands the canvas to encompass outer tick labels.
    Returns: {"text_overlaps": list, "text_outside_canvas": list, "fonts_below_min": list, "passed": bool}.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    fb = fig.bbox

    texts: list[mpl.text.Text] = []

    def collect_axes_texts(ax: plt.Axes) -> None:
        texts.extend(ax.texts)
        if getattr(ax, "axison", True):
            if ax.xaxis.get_visible():
                texts.extend(ax.xaxis.get_majorticklabels())
                texts.extend(ax.xaxis.get_minorticklabels())
                texts.append(ax.xaxis.label)
            if ax.yaxis.get_visible():
                texts.extend(ax.yaxis.get_majorticklabels())
                texts.extend(ax.yaxis.get_minorticklabels())
                texts.append(ax.yaxis.label)
            texts.append(ax.title)
        leg = ax.get_legend()
        if leg is not None:
            texts.extend(leg.get_texts())
        for ch in getattr(ax, "child_axes", []):
            collect_axes_texts(ch)

    for a in fig.axes:
        collect_axes_texts(a)
    texts.extend(fig.texts)

    items: list[tuple[mpl.text.Text, Any]] = []
    font_sizes: list[float] = []

    for t in texts:
        if t.get_visible() and t.get_text() and t.get_text().strip():
            items.append((t, t.get_window_extent(rend)))
            font_sizes.append(t.get_fontsize())

    overlaps: list[tuple[str, str]] = []
    outside: list[str] = []

    for i, (ti, bi) in enumerate(items):
        if check_outside:
            if (
                bi.x0 < fb.x0 - 1
                or bi.x1 > fb.x1 + 1
                or bi.y0 < fb.y0 - 1
                or bi.y1 > fb.y1 + 1
            ):
                outside.append(ti.get_text()[:40])
        for tj, bj in items[i + 1 :]:
            w = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
            h = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
            if w > tol_px and h > tol_px:
                overlaps.append((ti.get_text()[:30], tj.get_text()[:30]))

    small_fonts = sorted({s for s in font_sizes if s < min_font})
    passed = len(overlaps) == 0 and len(outside) == 0 and len(small_fonts) == 0

    return {
        "text_overlaps": overlaps,
        "text_outside_canvas": outside,
        "fonts_below_min": small_fonts,
        "passed": passed,
    }


__all__ = [
    "assert_zero_phi_text",
    "scan_phi_violations",
    "PHI_PATTERNS",
    "cvd_check",
    "qa_layout",
    "TISSUE_MUST_DIFFER",
]

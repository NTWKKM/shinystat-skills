"""
src/medstat/figures/illustration.py: Vector-first scientific schematics and clinical illustrations
(cells, tissues, vessels, pathways) using matplotlib and numpy only.
All output enforces Zero-PHI, layout collision checks, CVD color cues, and schematic disclosures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon
from matplotlib.path import Path as MplPath

from medstat.figures.base import FigureResult
from medstat.figures.journal import TARGET_WIDTHS, apply_journal_theme
from medstat.figures.qa import assert_zero_phi_text, qa_layout

# Semantic art palette: one meaning = one color across figures
SEM: dict[str, str] = {
    "epidermis": "#E9B99D",
    "dermis": "#F3D3BC",
    "hypodermis": "#F7E7B0",
    "muscle": "#B8504F",
    "bone": "#EFE9DA",
    "skin_surface": "#F6DCC8",
    "blood": "#B5282E",
    "plasma": "#F9E6DC",
    "fibrin": "#8A7F72",
    "clot": "#8E2B31",
    "epithelial": "#F2BDB5",
    "granulation": "#D8575A",
    "slough": "#E4CF74",
    "necrotic": "#3C302B",
    "erythema": "#F0B7A6",
    "collagen": "#E8DCC0",
    "scar": "#EADFD6",
    "nucleus": "#5A4A9C",
    "nucleus_dark": "#3F3480",
    "cyto": "#F4DDEA",
    "neutrophil": "#F0E6F3",
    "lymphocyte": "#DCE6F4",
    "macrophage": "#EBCBA6",
    "rbc": "#D9433C",
    "rbc_pallor": "#EE8A84",
    "platelet": "#B58BD6",
    "bacteria": "#3E8E5B",
    "lipid": "#F1D88A",
    "cap": "#E9DCCB",
    "lumen": "#FBEDEA",
    "vessel_wall": "#E7A2A0",
    "adventitia": "#EFE3CF",
    "endothelium": "#E8B0A8",
    "outline": "#3A2E2E",
    "text": "#222222",
    "guide": "#6B6B6B",
}


def shoelace(p: np.ndarray) -> float:
    """Calculates polygon area via shoelace formula."""
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def ellipse_pts(
    cx: float, cy: float, rx: float, ry: float, rot: float = 0.0, n: int = 72
) -> np.ndarray:
    """Generates 2D ellipse perimeter points."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(rot), np.sin(rot)
    x, y = rx * np.cos(t), ry * np.sin(t)
    return np.c_[cx + c * x - s * y, cy + s * x + c * y]


def shrink(
    p: np.ndarray, k: float, toward: Sequence[float] | None = None
) -> np.ndarray:
    """Shrinks 2D polygon points toward centroid or specified target point."""
    p_arr = np.asarray(p, float)
    c = p_arr.mean(0) if toward is None else np.asarray(toward, float)
    return c + (p_arr - c) * k


class Scene:
    """
    Standardized canvas in relative coordinates:
    x in [0, 100], y in [0, 100 * aspect]; 1 unit ~ width / 100 inches.
    """

    def __init__(
        self,
        width: str | float = "double_col",
        aspect: float = 0.6,
        journal: str = "NEJM",
        canvas: str = "manuscript",
        seed: int = 7,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.journal = journal
        w = TARGET_WIDTHS.get(width, width) if isinstance(width, str) else float(width)

        apply_journal_theme(journal, canvas)
        self.fig = plt.figure(figsize=(w, w * aspect), dpi=300)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.W, self.H = 100.0, 100.0 * aspect
        self.ax.set_xlim(0, self.W)
        self.ax.set_ylim(0, self.H)
        self.ax.set_aspect("equal")
        self.ax.axis("off")

        self.log: list[dict[str, Any]] = []
        self._z = 1.0
        self.texts: list[str] = []
        self.font_sizes: list[float] = []

    def z(self) -> float:
        self._z += 0.001
        return self._z

    def note(self, element: str, label: str = "", **params: Any) -> None:
        """Records drawn element coordinates for auditability and manifest CSV."""
        self.log.append(
            {
                "element": element,
                "label": label,
                "params": json.dumps(params, default=float),
                "seed": self.seed,
            }
        )

    # -- Primitives
    def poly(
        self,
        pts: Sequence[Sequence[float]] | np.ndarray,
        fc: str,
        ec: str | None = None,
        lw: float = 0.5,
        alpha: float = 1.0,
        hatch: str | None = None,
        closed: bool = True,
        z: float | None = None,
    ) -> Polygon:
        p = Polygon(
            np.asarray(pts),
            closed=closed,
            facecolor=fc,
            edgecolor=ec or "none",
            linewidth=lw,
            alpha=alpha,
            hatch=hatch,
            zorder=z or self.z(),
            joinstyle="round",
        )
        self.ax.add_patch(p)
        return p

    def line(
        self,
        pts: Sequence[Sequence[float]] | np.ndarray,
        color: str = SEM["outline"],
        lw: float = 0.6,
        alpha: float = 1.0,
        z: float | None = None,
        ls: str = "-",
        clip: np.ndarray | None = None,
        cap: str = "round",
    ) -> Any:
        pts_arr = np.asarray(pts)
        (line,) = self.ax.plot(
            pts_arr[:, 0],
            pts_arr[:, 1],
            color=color,
            lw=lw,
            alpha=alpha,
            ls=ls,
            zorder=z or self.z(),
            solid_capstyle=cap,
        )
        if clip is not None:
            line.set_clip_path(Polygon(clip, transform=self.ax.transData))
        return line

    def harmonic(
        self,
        th: np.ndarray,
        irregularity: float = 0.1,
        ks: Sequence[int] = (2, 3, 4, 5),
        phase: np.ndarray | None = None,
    ) -> np.ndarray:
        ph = self.rng.uniform(0, 2 * np.pi, len(ks)) if phase is None else phase
        r = np.ones_like(th)
        for k, f in zip(ks, ph):
            r += (
                irregularity / (k - 1) * np.cos(k * th + f) * self.rng.uniform(0.6, 1.0)
            )
        return r

    def blob(
        self,
        cx: float,
        cy: float,
        rx: float,
        ry: float | None = None,
        irregularity: float = 0.1,
        rot: float = 0.0,
        ks: Sequence[int] = (2, 3, 4, 5),
        n: int = 120,
    ) -> np.ndarray:
        ry_val = rx if ry is None else ry
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        r = self.harmonic(th, irregularity, ks)
        c, s = np.cos(rot), np.sin(rot)
        x, y = rx * r * np.cos(th), ry_val * r * np.sin(th)
        return np.c_[cx + c * x - s * y, cy + s * x + c * y]

    def volumetric(
        self,
        pts: np.ndarray,
        fc: str,
        ec: str = SEM["outline"],
        lw: float = 0.5,
        n: int = 4,
        light: tuple[float, float] = (-0.25, 0.25),
        strength: float = 0.18,
        z: float | None = None,
    ) -> None:
        """Vector-friendly soft shading: base + shrunken, light-shifted transparent layers."""
        z_val = z or self.z()
        self.poly(pts, fc, ec, lw, z=z_val)
        size = np.sqrt(shoelace(np.asarray(pts)))
        for i in range(n):
            k = 0.78 - i * (0.55 / n)
            q = shrink(pts, k) + np.array(light) * size * (1 - k) * 0.9
            self.poly(
                q, "#FFFFFF", None, alpha=strength / (1 + i * 0.35), z=z_val + 1e-5
            )

    def inside_points(
        self,
        pts: np.ndarray,
        n: int,
        margin: float = 0.0,
        min_dist: float = 0.0,
        avoid: Sequence[Any] = (),
    ) -> np.ndarray:
        """Random uniformly spaced points inside polygon, kept off the perimeter."""
        pts_arr = np.asarray(pts)
        path = MplPath(pts_arr)
        lo, hi = pts_arr.min(0), pts_arr.max(0)
        out: list[np.ndarray] = []
        inner = MplPath(shrink(pts_arr, max(1.0 - margin, 0.2))) if margin > 0 else path
        tries = 0
        while len(out) < n and tries < n * 300:
            tries += 1
            q = self.rng.uniform(lo, hi)
            if not (path.contains_point(q) and inner.contains_point(q)):
                continue
            if min_dist > 0 and any(
                np.hypot(*(q - o)) < min_dist for o in [*out, *avoid]
            ):
                continue
            out.append(q)
        return np.array(out).reshape(-1, 2)

    def dots(
        self,
        pts: np.ndarray,
        r: float,
        color: str,
        alpha: float = 1.0,
        z: float | None = None,
    ) -> None:
        z_val = z or self.z()
        for q in np.atleast_2d(pts):
            self.poly(
                ellipse_pts(q[0], q[1], r, r, n=14), color, None, alpha=alpha, z=z_val
            )

    def where(self, element: str) -> list[tuple[float, float]]:
        """Positions of previously drawn elements (useful for non-overlapping leader lines)."""
        out = []
        for row in self.log:
            if row["element"] == element:
                p = json.loads(row["params"])
                if "cx" in p and "cy" in p:
                    out.append((p["cx"], p["cy"]))
        return out

    # -- Text and Annotations
    def text(
        self,
        x: float,
        y: float,
        s: str,
        size: float = 7.0,
        ha: str = "center",
        va: str = "center",
        weight: str = "normal",
        color: str = SEM["text"],
        z: float | None = None,
        **kw: Any,
    ) -> mpl.text.Text:
        self.texts.append(s)
        self.font_sizes.append(size)
        return self.ax.text(
            x,
            y,
            s,
            fontsize=size,
            ha=ha,
            va=va,
            fontweight=weight,
            color=color,
            zorder=z or 50,
            **kw,
        )

    def badge(
        self, x: float, y: float, letter: str, size: float = 10.0
    ) -> mpl.text.Text:
        return self.text(x, y, letter, size, "left", "top", "bold")

    def callout(
        self,
        s: str,
        target: tuple[float, float],
        text_xy: tuple[float, float],
        size: float = 6.5,
        ha: str | None = None,
        dot: bool = True,
    ) -> mpl.text.Text:
        ha_val = ha or ("left" if text_xy[0] >= target[0] else "right")
        self.ax.annotate(
            "",
            xy=target,
            xytext=text_xy,
            zorder=48,
            arrowprops=dict(
                arrowstyle="-", lw=0.5, color=SEM["guide"], shrinkA=1.5, shrinkB=0
            ),
        )
        if dot:
            self.dots(np.array([target]), 0.45, SEM["outline"], z=49)
        return self.text(text_xy[0], text_xy[1], s, size, ha_val, "center")

    def label_column(
        self,
        items: list[tuple[str, tuple[float, float]]],
        x: float,
        y_lo: float,
        y_hi: float,
        side: str = "right",
        size: float = 6.5,
        gap: float = 3.4,
    ) -> None:
        """Spreads label items vertically without overlap and connects leaders."""
        sorted_items = sorted(items, key=lambda t: -t[1][1])
        ys: list[float] = []
        for _, (_, ty) in sorted_items:
            ys.append(min(ty, (ys[-1] - gap) if ys else y_hi))
        if ys and ys[-1] < y_lo:
            ys = [
                y_hi - i * (y_hi - y_lo) / max(len(ys) - 1, 1) for i in range(len(ys))
            ]
        for (s, tgt), y in zip(sorted_items, ys):
            self.callout(s, tgt, (x, y), size, "left" if side == "right" else "right")

    def scale_bar(
        self, x: float, y: float, length: float, label: str, size: float = 6.5
    ) -> None:
        self.line([(x, y), (x + length, y)], SEM["outline"], 1.2, z=48)
        self.text(x + length / 2, y - 1.8, label, size)

    def legend_swatches(
        self,
        entries: list[tuple[str, str]],
        x: float,
        y: float,
        dy: float = 3.6,
        size: float = 6.5,
        hatch_map: dict[str, str] | None = None,
    ) -> None:
        for i, (name, col) in enumerate(entries):
            h = (hatch_map or {}).get(name)
            self.poly(
                [
                    (x, y - i * dy - 1),
                    (x + 3, y - i * dy - 1),
                    (x + 3, y - i * dy + 1.2),
                    (x, y - i * dy + 1.2),
                ],
                col,
                SEM["outline"],
                0.4,
                hatch=h,
            )
            self.text(x + 4.2, y - i * dy + 0.1, name, size, "left")

    # ------------------------------------------------------------ CELLS
    def rbc(
        self,
        cx: float,
        cy: float,
        r: float = 1.3,
        rot: float = 0.0,
        tilt: float = 0.55,
    ) -> None:
        self.note("rbc", "", cx=cx, cy=cy, r=r)
        tilt_factor = 1.0 if tilt is None else max(tilt, 0.55)
        o = ellipse_pts(cx, cy, r, r * tilt_factor, rot)
        self.poly(o, SEM["rbc"], "#8E2A25", 0.35)
        self.poly(
            ellipse_pts(cx, cy, r * 0.5, r * 0.5 * tilt_factor, rot),
            SEM["rbc_pallor"],
            None,
            0.85,
        )

    def platelet(self, cx: float, cy: float, r: float = 0.7) -> None:
        self.note("platelet", "", cx=cx, cy=cy)
        self.poly(
            self.blob(cx, cy, r, r * 0.6, 0.12, self.rng.uniform(0, 3), n=30),
            SEM["platelet"],
            "#6E4A8E",
            0.3,
        )
        self.dots(
            self.inside_points(self.blob(cx, cy, r * 0.8, r * 0.45, 0, 0, n=20), 2),
            r * 0.12,
            "#4B2E6B",
        )

    def neutrophil(
        self,
        cx: float,
        cy: float,
        r: float = 2.2,
        lobes: int = 3,
        squeeze: float = 1.0,
        rot: float = 0.0,
    ) -> None:
        self.note("neutrophil", "", cx=cx, cy=cy, r=r)
        cyt = self.blob(cx, cy, r, r / squeeze, 0.05, rot, n=90)
        self.poly(cyt, SEM["neutrophil"], SEM["outline"], 0.5)
        a0 = self.rng.uniform(0, 2 * np.pi)
        cs = []
        for k in range(lobes):
            a = a0 + k * 2.2 / max(lobes - 1, 1) * (1 if lobes > 1 else 0) - 1.1
            c = (
                cx + 0.5 * r * np.cos(a + rot),
                cy + 0.5 * r / squeeze * np.sin(a + rot),
            )
            cs.append(c)
        self.line(cs, SEM["nucleus"], 0.9, z=self.z())
        for c in cs:
            self.poly(
                self.blob(c[0], c[1], r * 0.27, r * 0.24, 0.08, n=30),
                SEM["nucleus"],
                SEM["nucleus_dark"],
                0.3,
            )
        self.dots(self.inside_points(cyt, 10, 0.15), r * 0.045, "#D9B5D6", 0.9)

    def lymphocyte(self, cx: float, cy: float, r: float = 1.8) -> None:
        self.note("lymphocyte", "", cx=cx, cy=cy, r=r)
        self.poly(
            self.blob(cx, cy, r, r, 0.04, n=70), SEM["lymphocyte"], SEM["outline"], 0.5
        )
        nuc = self.blob(cx, cy, r * 0.8, r * 0.8, 0.05, n=60)
        self.poly(nuc, SEM["nucleus"], SEM["nucleus_dark"], 0.4)
        self.dots(self.inside_points(nuc, 7, 0.1), r * 0.06, "#8C7DC4", 0.8)

    def macrophage(
        self,
        cx: float,
        cy: float,
        r: float = 3.2,
        vacuoles: int = 4,
        engulfed: int = 0,
    ) -> None:
        self.note("macrophage", "", cx=cx, cy=cy, r=r, engulfed=engulfed)
        body = self.blob(cx, cy, r, r * 0.85, 0.2, ks=(2, 3, 4, 5, 6), n=140)
        self.poly(body, SEM["macrophage"], SEM["outline"], 0.5, z=self.z())
        nuc = self.blob(
            cx + r * 0.2,
            cy + r * 0.1,
            r * 0.38,
            r * 0.28,
            0.15,
            self.rng.uniform(0, 3),
            n=50,
        )
        self.poly(nuc, SEM["nucleus"], SEM["nucleus_dark"], 0.4)
        for q in self.inside_points(body, vacuoles, 0.25):
            self.poly(
                ellipse_pts(q[0], q[1], r * 0.09, r * 0.09, n=16),
                "#FBF1E6",
                "#B98F68",
                0.3,
            )
        for q in self.inside_points(body, engulfed, 0.3):
            self.bacterium(q[0], q[1], r * 0.22, "bacillus", self.rng.uniform(0, 3))

    def fibroblast(
        self, cx: float, cy: float, L: float = 5.0, rot: float = 0.0
    ) -> None:
        self.note("fibroblast", "", cx=cx, cy=cy, L=L)
        u = np.linspace(-1, 1, 40)
        hw = 0.11 * L * (1 - u**2) ** 0.7
        x = np.r_[u, u[::-1]] * L / 2
        y = np.r_[hw, -hw[::-1]]
        c, s = np.cos(rot), np.sin(rot)
        p = np.c_[cx + c * x - s * y, cy + s * x + c * y]
        self.poly(p, "#E2C9A8", SEM["outline"], 0.4)
        self.poly(
            ellipse_pts(cx, cy, L * 0.17, L * 0.07, rot),
            SEM["nucleus"],
            SEM["nucleus_dark"],
            0.3,
        )

    def bacterium(
        self,
        cx: float,
        cy: float,
        L: float = 1.6,
        kind: str = "bacillus",
        rot: float = 0.0,
    ) -> None:
        self.note("bacterium", kind, cx=cx, cy=cy, L=L)
        c, s = np.cos(rot), np.sin(rot)

        def at(dx: float, dy: float) -> tuple[float, float]:
            return cx + c * dx - s * dy, cy + s * dx + c * dy

        col, ec = SEM["bacteria"], "#26603B"
        if kind == "bacillus":
            t = np.linspace(-np.pi / 2, np.pi / 2, 14)
            w = L * 0.17
            pts = [(-(L / 2 - w) - w * np.cos(a), w * np.sin(a)) for a in t] + [
                ((L / 2 - w) + w * np.cos(a), -w * np.sin(a)) for a in t
            ]
            self.poly([at(*q) for q in pts], col, ec, 0.35)
        else:
            offs = {
                "coccus": [(0, 0)],
                "diplococcus": [(-0.2 * L, 0), (0.2 * L, 0)],
                "chain": [(k * 0.3 * L, 0) for k in range(-2, 3)],
                "cluster": [
                    (0, 0),
                    (0.3 * L, 0.05 * L),
                    (-0.28 * L, 0.1 * L),
                    (0.1 * L, 0.3 * L),
                    (-0.1 * L, -0.28 * L),
                    (0.25 * L, -0.25 * L),
                ],
            }[kind]
            for dx, dy in offs:
                x0, y0 = at(dx, dy)
                self.poly(ellipse_pts(x0, y0, 0.17 * L, 0.17 * L, n=20), col, ec, 0.35)

    def eukaryote(
        self, cx: float, cy: float, r: float = 9.0, organelles: bool = True
    ) -> None:
        self.note("eukaryotic_cell", "", cx=cx, cy=cy, r=r)
        body = self.blob(cx, cy, r, r * 0.85, 0.06)
        self.poly(body, SEM["cyto"], SEM["outline"], 0.8)
        nx, ny = cx - r * 0.12, cy + r * 0.05
        self.poly(
            ellipse_pts(nx, ny, r * 0.34, r * 0.3), "#D7CDEB", SEM["nucleus_dark"], 0.8
        )
        self.poly(
            ellipse_pts(nx - r * 0.05, ny + r * 0.02, r * 0.09, r * 0.09),
            SEM["nucleus"],
            None,
        )
        if organelles:
            for q in self.inside_points(body, 5, 0.35):
                if np.hypot(q[0] - nx, q[1] - ny) < r * 0.45:
                    continue
                rot = self.rng.uniform(0, np.pi)
                self.poly(
                    ellipse_pts(q[0], q[1], r * 0.1, r * 0.055, rot),
                    "#F2B58A",
                    "#9A5B34",
                    0.4,
                )
                self.line(
                    [
                        (
                            q[0] - r * 0.06 * np.cos(rot),
                            q[1] - r * 0.06 * np.sin(rot),
                        ),
                        (
                            q[0] + r * 0.06 * np.cos(rot),
                            q[1] + r * 0.06 * np.sin(rot),
                        ),
                    ],
                    "#9A5B34",
                    0.3,
                )
            for k in range(3):
                a = np.linspace(-0.9, 0.9, 20) + 0.1 * k
                rr = r * (0.42 + 0.05 * k)
                self.line(
                    np.c_[
                        nx + rr * np.cos(a + 2.4) + 0.01 * r * np.sin(a * 25),
                        ny + rr * np.sin(a + 2.4),
                    ],
                    "#9B86B8",
                    0.5,
                )
            self.dots(self.inside_points(body, 25, 0.1), r * 0.012, "#7A6A98")

    # ------------------------------------------------------------ TISSUE
    def skin(
        self,
        x0: float,
        x1: float,
        ytop: float,
        thick: tuple[float, float, float, float] = (3.0, 11.0, 9.0, 9.0),
        rete: float = 0.9,
    ) -> dict[str, Any]:
        """Layered skin block. thick=(epidermis, dermis, hypodermis, muscle)."""
        xs = np.linspace(x0, x1, 240)
        e, d, h, m = thick
        dej = ytop - e - rete * np.abs(np.sin((xs - x0) * 0.9)) * (rete > 0)
        y_d = np.full_like(xs, ytop - e - d) + 0.4 * np.sin(xs * 0.4)
        y_h = y_d - h + 0.5 * np.sin(xs * 0.25 + 1)
        y_m = y_h - m
        geo = dict(
            x0=x0,
            x1=x1,
            ytop=ytop,
            thick=thick,
            xs=xs,
            dej=dej,
            y_dermis=y_d,
            y_hypo=y_h,
            y_bottom=y_m,
        )

        def band(top: np.ndarray, bot: np.ndarray, col: str) -> None:
            self.poly(np.r_[np.c_[xs, top], np.c_[xs[::-1], bot[::-1]]], col, None)

        band(np.full_like(xs, ytop), dej, SEM["epidermis"])
        band(dej, y_d, SEM["dermis"])
        band(y_d, y_h, SEM["hypodermis"])
        band(y_h, y_m, SEM["muscle"])

        for q in self.inside_points(
            np.r_[np.c_[xs, y_d], np.c_[xs[::-1], y_h[::-1]]],
            int((x1 - x0) * 0.4),
            0.12,
            min_dist=2.7,
        ):
            self.poly(ellipse_pts(q[0], q[1], 1.2, 1.1), "#FBF0C9", "#D9C27A", 0.3)

        for yy in np.linspace(y_h.mean() - 0.4 * m, y_m.mean() + 0.4 * m, 4):
            self.line([(x0, yy), (x1, yy + 0.8)], "#8E3B3A", 0.4, alpha=0.6)

        self.line(np.c_[xs, np.full_like(xs, ytop)], SEM["outline"], 0.7)
        self.line(np.c_[xs, y_m], SEM["outline"], 0.5)
        self.line([(x0, ytop), (x0, y_m[0])], SEM["outline"], 0.5)
        self.line([(x1, ytop), (x1, y_m[-1])], SEM["outline"], 0.5)
        self.note("skin_block", "", x0=x0, x1=x1, ytop=ytop, thick=list(thick))
        return geo

    def crater(
        self,
        geo: dict[str, Any],
        xc: float,
        width: float,
        depth: float,
        flatness: float = 2.6,
        fill: str = SEM["plasma"],
        outline: bool = True,
    ) -> np.ndarray:
        """Wound crater cut into a skin block; returns outline polygon."""
        t = np.linspace(0, np.pi, 80)
        ex = 2.0 / flatness
        x = xc + width / 2 * np.sign(np.cos(t)) * np.abs(np.cos(t)) ** ex
        y = (
            geo["ytop"]
            - depth * np.abs(np.sin(t)) ** ex
            + 0.18 * self.rng.normal(0, 1, t.size) * np.sin(t)
        )
        pts = np.c_[x, y]
        self.poly(pts, fill, None, z=self.z())
        if outline:
            self.line(pts, SEM["outline"], 0.8, z=self.z())
        self.note("crater", "", xc=xc, width=width, depth=depth)
        return pts

    def fibers(
        self,
        clip_pts: np.ndarray,
        angle: float = 0.0,
        spacing: float = 1.2,
        wiggle: float = 0.25,
        color: str = SEM["collagen"],
        lw: float = 0.6,
        ec: str = "#B7A57F",
    ) -> None:
        p = np.asarray(clip_pts)
        lo, hi = p.min(0), p.max(0)
        c, s = np.cos(angle), np.sin(angle)
        R = np.hypot(*(hi - lo))
        ctr = (lo + hi) / 2
        for off in np.arange(-R / 2, R / 2, spacing):
            u = np.linspace(-R / 2, R / 2, 60)
            v = off + wiggle * np.sin(u * 1.3 + off)
            pts = np.c_[ctr[0] + c * u - s * v, ctr[1] + s * u + c * v]
            self.line(pts, ec, lw + 0.3, z=self.z(), clip=p)
            self.line(pts, color, lw, z=self.z(), clip=p)

    def fibrin(
        self, clip_pts: np.ndarray, n: int = 70, color: str = SEM["fibrin"]
    ) -> None:
        p = np.asarray(clip_pts)
        lo, hi = p.min(0), p.max(0)
        for _ in range(n):
            q = self.rng.uniform(lo, hi)
            a = self.rng.uniform(0, np.pi)
            length = self.rng.uniform(2.5, 6.0)
            self.line(
                [
                    (q[0], q[1]),
                    (q[0] + length * np.cos(a), q[1] + length * np.sin(a)),
                ],
                color,
                0.35,
                alpha=0.8,
                clip=p,
            )

    def capillary_loop(
        self, x: float, y: float, h: float = 3.2, w: float = 1.1
    ) -> None:
        t = np.linspace(np.pi, 0, 26)
        arc = np.c_[x + w * np.cos(t), y + h * 0.45 + h * 0.55 * np.sin(t) ** 0.8]
        pts = np.r_[[(x - w, y)], arc, [(x + w, y)]]
        self.line(pts, "#7C1D22", 1.9, z=self.z())
        self.line(pts, SEM["blood"], 1.1, z=self.z())

    def wound_phase(
        self,
        geo: dict[str, Any],
        crater: np.ndarray,
        xc: float,
        width: float,
        depth: float,
        phase: str,
    ) -> None:
        """phase in {'hemostasis', 'inflammation', 'proliferation', 'remodeling'}."""
        cp = np.asarray(crater)
        lo, hi = cp.min(0), cp.max(0)
        y0, y1 = lo[1], geo["ytop"]
        self.note("wound_phase", phase, xc=xc, width=width, depth=depth)

        if phase == "hemostasis":
            self.poly(cp, SEM["clot"], None, alpha=0.8, z=self.z())
            self.fibrin(cp, 90, "#CDBFB2")
            rb = self.inside_points(cp, 15, 0.16, min_dist=2.7)
            for q in rb:
                self.rbc(q[0], q[1], 1.1, self.rng.uniform(0, 3))
            for q in self.inside_points(cp, 8, 0.12, min_dist=1.8, avoid=rb):
                self.platelet(q[0], q[1], 0.7)
            scab = np.array(
                [
                    [xc - width / 2 * 0.9, y1 + 0.1],
                    [xc - width / 2 * 0.5, y1 + 1.1],
                    [xc + width / 2 * 0.5, y1 + 1.2],
                    [xc + width / 2 * 0.9, y1 + 0.1],
                ]
            )
            self.poly(scab, "#5E1F24", SEM["outline"], 0.4)
        elif phase == "inflammation":
            self.poly(cp, "#FBEEDF", None, z=self.z())
            self.fibrin(cp, 30)
            pts = self.inside_points(cp, 8, 0.14, min_dist=4.3)
            for q in pts[:2]:
                self.macrophage(q[0], q[1], 2.2, 3, engulfed=1)
            for q in pts[2:]:
                self.neutrophil(
                    q[0],
                    q[1],
                    1.4,
                    int(self.rng.integers(3, 5)),
                    rot=self.rng.uniform(0, 3),
                )
            for q in self.inside_points(cp, 3, 0.14, min_dist=3.0, avoid=pts):
                self.bacterium(q[0], q[1], 1.1, "bacillus", self.rng.uniform(0, 3))
        elif phase == "proliferation":
            lvl = y0 + (y1 - y0) * 0.72
            xs = np.linspace(lo[0] - 1, hi[0] + 1, 80)
            top = lvl + 0.45 * np.sin(xs * 2.2) + 0.25 * np.sin(xs * 5)
            fill = np.r_[[(xs[0], y0 - 2)], np.c_[xs, top], [(xs[-1], y0 - 2)]]
            self.poly(cp, SEM["plasma"], None, z=self.z())
            pb = Polygon(cp, transform=self.ax.transData)
            g = self.poly(fill, SEM["granulation"], None, z=self.z())
            g.set_clip_path(pb)
            for xx in np.linspace(lo[0] + width * 0.17, hi[0] - width * 0.17, 5):
                yb = np.interp(xx, xs, top) - 4.2
                if MplPath(cp).contains_point((xx, yb)):
                    self.capillary_loop(xx, yb, 3.0, 0.9)
            bed = np.r_[np.c_[xs, top - 3.0], [(xs[-1], y0), (xs[0], y0)]]
            inc = MplPath(shrink(cp, 0.93))
            gr = [
                q
                for q in self.inside_points(bed, 140, 0.02, min_dist=1.2)
                if inc.contains_point(q)
            ]
            self.dots(np.array(gr).reshape(-1, 2), 0.28, "#B8383C", 0.55)
            for q in self.inside_points(bed, 3, 0.2, min_dist=6.0):
                self.fibroblast(q[0], q[1], 3.6, self.rng.uniform(-0.5, 0.5))
            for sgn in (-1, 1):
                x_e = xc + sgn * width / 2
                xx = np.linspace(x_e, x_e - sgn * width * 0.22, 20)
                yy = np.interp(xx, xs, top) + 0.2
                self.line(np.c_[xx, yy], "#7B3F39", 2.6, clip=cp)
                self.line(np.c_[xx, yy], SEM["epithelial"], 1.6, clip=cp)
        elif phase == "remodeling":
            self.poly(cp, SEM["scar"], None, z=self.z())
            self.fibers(cp, 0.0, 1.0, 0.2)
            cap = np.array(
                [
                    [lo[0], y1 - 0.2],
                    [hi[0], y1 - 0.2],
                    [hi[0], y1 - 2.4],
                    [lo[0], y1 - 2.4],
                ]
            )
            g = self.poly(cap, SEM["epidermis"], None, z=self.z())
            g.set_clip_path(Polygon(cp, transform=self.ax.transData))
            self.line(cp, "#B7A57F", 0.5, ls=(0, (2, 2)))
        else:
            raise ValueError(f"Unknown wound phase: {phase}")

    def pressure_injury(
        self, geo: dict[str, Any], xc: float, stage: int
    ) -> np.ndarray | None:
        """NPIAP-style staging: stage in {1, 2, 3, 4}."""
        th = geo["thick"]
        e, d, h, m = th
        self.note("pressure_injury", f"stage{stage}")
        if stage == 1:
            p = ellipse_pts(xc, geo["ytop"] - e / 2, 9, e * 0.55, n=60)
            self.poly(p, "#E97D6D", None, alpha=0.55, z=self.z())
            return None
        depth = {2: e + d * 0.35, 3: e + d + h * 0.95, 4: e + d + h + m * 0.8}[stage]
        width = {2: 10, 3: 14, 4: 12}[stage]
        cr = self.crater(
            geo,
            xc,
            width,
            depth,
            flatness=2.4 if stage != 4 else 3.0,
            fill=SEM["plasma"],
        )
        if stage == 2:
            self.poly(cr, "#F0A79A", None, alpha=0.6, z=self.z())
        if stage == 3:
            self.poly(cr, SEM["slough"], None, alpha=0.55, z=self.z())
        if stage == 4:
            self.poly(cr, SEM["slough"], None, alpha=0.35, z=self.z())
            lo = cr.min(0)
            self.poly(
                np.array(
                    [
                        [xc - 3.4, lo[1] + 0.2],
                        [xc + 3.4, lo[1] + 0.2],
                        [xc + 3.0, lo[1] - 1.6],
                        [xc - 3.0, lo[1] - 1.6],
                    ]
                ),
                SEM["bone"],
                SEM["outline"],
                0.5,
                z=self.z(),
            )
        return cr

    def wound_topdown(
        self,
        cx: float,
        cy: float,
        r: float,
        fractions: dict[str, float],
        order: Sequence[str] = (
            "epithelial",
            "granulation",
            "slough",
            "necrotic",
        ),
        irregularity: float = 0.10,
        hatch: bool = False,
        tol: float = 0.01,
    ) -> dict[str, float]:
        """Top-down wound bed; ring areas match fractions (verified & logged)."""
        unknown = set(fractions.keys()) - set(order)
        if unknown:
            raise ValueError(
                f"Unknown tissue fraction keys: {sorted(unknown)}; allowed keys are {order}"
            )
        for k, v in fractions.items():
            if not np.isfinite(v) or v < 0:
                raise ValueError(
                    f"Invalid tissue fraction for '{k}': {v} (must be finite and non-negative)"
                )
        tot = sum(fractions.values())
        if tot <= 0 or not np.isfinite(tot):
            raise ValueError(f"Total fraction must be positive and finite, got {tot}")
        fr = {k: float(fractions.get(k, 0)) / tot for k in order}
        th = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        base = self.harmonic(th, irregularity)
        eps = [self.harmonic(th, 0.06, (2, 3, 5)) - 1 for _ in order]

        def poly_s(s: float, e: np.ndarray) -> np.ndarray:
            rr = s * r * base * (1 + e)
            return np.c_[cx + rr * np.cos(th), cy + rr * np.sin(th)]

        A = shoelace(poly_s(1.0, 0 * th))
        hmap = (
            {
                "epithelial": None,
                "granulation": "..",
                "slough": "///",
                "necrotic": "xxx",
            }
            if hatch
            else {}
        )
        for s_, c_, a_ in (
            (1.95, SEM["skin_surface"], 1.0),
            (1.55, SEM["erythema"], 0.85),
            (1.22, "#EBA695", 0.85),
        ):
            self.poly(poly_s(s_, 0 * th), c_, None, alpha=a_)
        cum, bounds = 0.0, [poly_s(1.0, 0 * th)]
        for k, name in enumerate(order[:-1]):
            if fr[name] == 0.0:
                bounds.append(bounds[-1].copy())
                continue
            cum += fr[name]
            target = A * (1 - cum)
            lo_s, hi_s = 0.0, 1.0
            for _ in range(40):
                mid = (lo_s + hi_s) / 2
                if shoelace(poly_s(mid, eps[k + 1])) > target:
                    hi_s = mid
                else:
                    lo_s = mid
            new_bound = poly_s((lo_s + hi_s) / 2, eps[k + 1])
            outer_path = MplPath(bounds[k])
            if not outer_path.contains_points(new_bound).all():
                raise ValueError(
                    f"Generated inner boundary for '{name}' crosses outer boundary"
                )
            bounds.append(new_bound)
        ach: dict[str, float] = {}
        for k, name in enumerate(order):
            self.poly(bounds[k], SEM[name], SEM["outline"], 0.4, hatch=hmap.get(name))
            inner = shoelace(bounds[k + 1]) if k + 1 < len(bounds) else 0.0
            ach[name] = (shoelace(bounds[k]) - inner) / A
        err = max(abs(ach[k] - fr[k]) for k in order)
        if err > tol:
            raise ValueError(f"Wound tissue area mismatch: {err:.3f} > tol {tol}")
        self.line(np.r_[bounds[0], bounds[0][:1]], SEM["outline"], 0.9)
        self.note(
            "wound_topdown",
            "tissue_fractions",
            requested=fr,
            achieved={k: round(v, 4) for k, v in ach.items()},
            max_abs_err=round(err, 5),
        )
        return ach

    # ------------------------------------------------------------ VESSELS
    def artery_cross_section(
        self,
        cx: float,
        cy: float,
        R: float = 14.0,
        stenosis: float = 0.6,
        lipid_core: bool = True,
        rupture: bool = False,
        thrombus: float = 0.0,
    ) -> dict[str, Any]:
        """Eccentric plaque. Area stenosis is exact by geometric construction."""
        if not np.isfinite(stenosis) or not (0.0 <= stenosis < 1.0):
            raise ValueError(
                f"stenosis must be a finite float in [0.0, 1.0), got {stenosis}"
            )
        Rl = R * 0.62
        rr = Rl * np.sqrt(1 - stenosis)
        d = Rl - rr
        v = np.array([0.0, 1.0])
        self.poly(
            ellipse_pts(cx, cy, R * 1.28, R * 1.28, n=120),
            SEM["adventitia"],
            SEM["outline"],
            0.6,
        )
        for a in np.linspace(0, 2 * np.pi, 40, endpoint=False):
            self.line(
                [
                    (cx + R * 1.02 * np.cos(a), cy + R * 1.02 * np.sin(a)),
                    (cx + R * 1.2 * np.cos(a + 0.1), cy + R * 1.2 * np.sin(a + 0.1)),
                ],
                "#C9B48C",
                0.35,
            )
        self.poly(
            ellipse_pts(cx, cy, R, R, n=120), SEM["vessel_wall"], SEM["outline"], 0.5
        )
        for k in range(14):
            self.line(
                ellipse_pts(
                    cx, cy, R * (0.72 + 0.017 * k), R * (0.72 + 0.017 * k), n=90
                )[: 70 + k],
                "#B96A68",
                0.3,
                alpha=0.7,
            )
        self.poly(
            ellipse_pts(cx, cy, Rl * 1.06, Rl * 1.06, n=100),
            SEM["endothelium"],
            None,
        )
        outer = ellipse_pts(cx, cy, Rl, Rl, n=140)
        inner = ellipse_pts(cx + v[0] * d, cy + v[1] * d, rr, rr, n=140)[::-1]
        codes = (
            [MplPath.MOVETO]
            + [MplPath.LINETO] * (len(outer) - 1)
            + [MplPath.CLOSEPOLY]
            + [MplPath.MOVETO]
            + [MplPath.LINETO] * (len(inner) - 1)
            + [MplPath.CLOSEPOLY]
        )
        verts = np.r_[outer, outer[:1], inner, inner[:1]]
        self.ax.add_patch(
            PathPatch(
                MplPath(verts, codes),
                fc=SEM["cap"],
                ec=SEM["outline"],
                lw=0.6,
                zorder=self.z(),
            )
        )
        self.poly(
            ellipse_pts(cx + v[0] * d, cy + v[1] * d, rr, rr, n=140),
            SEM["lumen"],
            SEM["outline"],
            0.5,
        )
        T = 2 * Rl - 2 * rr
        if lipid_core and T > 0.5:
            c0 = np.array([cx, cy]) - v * (Rl - 0.55 * T)
            core = ellipse_pts(c0[0], c0[1], min(Rl * 0.62, T * 1.1), T * 0.27, 0, 50)
            core = core[(np.hypot(core[:, 0] - cx, core[:, 1] - cy) <= Rl * 0.98)]
            self.poly(core, SEM["lipid"], "#B79A3A", 0.5)
            for i in range(5):
                self.line(
                    [
                        (c0[0] - 3 + i * 1.5, c0[1] - 0.3),
                        (c0[0] - 2.4 + i * 1.5, c0[1] + 0.6),
                    ],
                    "#8C7425",
                    0.4,
                )
        if rupture:
            x_r = cx + 0.12 * Rl
            y_r = cy - (Rl - T) + 0.1
            self.poly(
                ellipse_pts(x_r, y_r, 1.3, 0.7), SEM["lumen"], SEM["outline"], 0.5
            )
        if thrombus > 0:
            lc = np.array([cx + v[0] * d, cy + v[1] * d])
            rt = rr * (0.35 + 0.5 * thrombus)
            tb = self.blob(lc[0], lc[1] - rr * 0.12, rt, rt * 0.92, 0.12, n=60)
            clipc = ellipse_pts(lc[0], lc[1], rr * 0.985, rr * 0.985, n=100)
            tp = self.poly(tb, SEM["clot"], SEM["outline"], 0.4, alpha=0.92)
            tp.set_clip_path(Polygon(clipc, transform=self.ax.transData))
            self.fibrin(tb, 25, "#D8C9BD")
            for q in self.inside_points(tb, int(8 * thrombus) + 2, 0.22, min_dist=1.7):
                self.rbc(q[0], q[1], 0.75)
        ach = 1.0 - (rr / Rl) ** 2
        self.note(
            "artery",
            "plaque",
            requested_stenosis=stenosis,
            achieved_stenosis=round(ach, 4),
            lipid_core=lipid_core,
            rupture=rupture,
            thrombus=thrombus,
        )
        return dict(Rl=Rl, rr=rr, d=d, T=T, plaque_center=(cx, cy - Rl + 0.5 * T))

    def vessel_longitudinal(
        self,
        x0: float,
        x1: float,
        y_floor: float,
        lumen_h: float = 14.0,
        wall: float = 3.0,
        tissue_h: float = 18.0,
    ) -> dict[str, Any]:
        """Post-capillary venule wall and lumen."""
        self.poly(
            [
                (x0, y_floor - wall - tissue_h),
                (x1, y_floor - wall - tissue_h),
                (x1, y_floor - wall),
                (x0, y_floor - wall),
            ],
            "#F5E4DA",
            None,
        )
        self.poly(
            [
                (x0, y_floor - wall),
                (x1, y_floor - wall),
                (x1, y_floor),
                (x0, y_floor),
            ],
            SEM["endothelium"],
            None,
        )
        self.poly(
            [
                (x0, y_floor),
                (x1, y_floor),
                (x1, y_floor + lumen_h),
                (x0, y_floor + lumen_h),
            ],
            SEM["plasma"],
            None,
        )
        self.poly(
            [
                (x0, y_floor + lumen_h),
                (x1, y_floor + lumen_h),
                (x1, y_floor + lumen_h + wall),
                (x0, y_floor + lumen_h + wall),
            ],
            SEM["endothelium"],
            None,
        )
        for yy in (y_floor, y_floor + lumen_h):
            self.line([(x0, yy), (x1, yy)], SEM["outline"], 0.7)
        for xj in np.arange(x0 + 8, x1 - 4, 11):
            self.poly(
                ellipse_pts(xj + 4, y_floor - wall / 2, 2.6, 0.8), "#B98F9A", None
            )
            self.line(
                [(xj + 8.5, y_floor), (xj + 8.5, y_floor - wall)],
                SEM["outline"],
                0.4,
            )
        self.note("vessel_longitudinal", "", x0=x0, x1=x1)
        return dict(floor=y_floor, top=y_floor + lumen_h, wall=wall)

    def selectin_lollipop(
        self, x: float, y: float, h: float = 1.8, color: str = "#1F6F5C"
    ) -> None:
        self.line([(x, y), (x, y + h)], color, 0.6)
        self.dots(np.array([(x, y + h + 0.35)]), 0.38, color)

    # ------------------------------------------------------------ PATHWAY
    def node(
        self,
        id: str,
        label: str,
        xy: tuple[float, float],
        kind: str = "process",
        size: float = 6.5,
        pad: tuple[float, float] = (1.6, 1.0),
        fc: str | None = None,
    ) -> FancyBboxPatch:
        sty = {
            "process": (
                "#FFFFFF",
                SEM["outline"],
                0.7,
                "round,pad=0.0,rounding_size=1.0",
            ),
            "molecule": (
                "#EEF3F8",
                "#3D5A80",
                0.6,
                "round,pad=0.0,rounding_size=2.2",
            ),
            "stimulus": (
                "#ECECEC",
                "#555555",
                0.7,
                "round,pad=0.0,rounding_size=0.6",
            ),
            "outcome": (
                "#F7DADA",
                SEM["blood"],
                1.3,
                "round,pad=0.0,rounding_size=1.0",
            ),
            "cell": (
                "#F4E9F6",
                SEM["nucleus"],
                0.8,
                "round,pad=0.0,rounding_size=1.6",
            ),
        }[kind]
        t = self.ax.text(
            xy[0],
            xy[1],
            label,
            fontsize=size,
            ha="center",
            va="center",
            zorder=52,
            fontweight="bold" if kind == "outcome" else "normal",
            linespacing=1.15,
        )
        self.texts.append(label)
        self.font_sizes.append(size)
        renderer = self.fig.canvas.get_renderer()
        bb = t.get_window_extent(renderer).transformed(self.ax.transData.inverted())
        w, h = bb.width + 2 * pad[0], bb.height + 2 * pad[1]
        box = FancyBboxPatch(
            (xy[0] - w / 2, xy[1] - h / 2),
            w,
            h,
            boxstyle=sty[3],
            fc=fc or sty[0],
            ec=sty[1],
            lw=sty[2],
            zorder=51,
        )
        self.ax.add_patch(box)
        if not hasattr(self, "nodes"):
            self.nodes: dict[str, dict[str, Any]] = {}
        self.nodes[id] = dict(box=box, xy=xy, w=w, h=h, label=label)
        self.note("pathway_node", label, id=id, kind=kind, xy=list(xy))
        return box

    def edge(
        self,
        a: str,
        b: str,
        kind: str = "activate",
        label: str | None = None,
        rad: float = 0.0,
        label_xy: tuple[float, float] | None = None,
        size: float = 6.0,
    ) -> None:
        A, B = self.nodes[a], self.nodes[b]
        st = {
            "activate": ("-|>,head_length=3.2,head_width=1.6", "#2B2B2B", "-"),
            "inhibit": ("-[,widthB=1.7,lengthB=0.0", SEM["blood"], "-"),
            "indirect": (
                "-|>,head_length=3.2,head_width=1.6",
                "#7A7A7A",
                (0, (3, 2)),
            ),
            "flow": ("-|>,head_length=3.2,head_width=1.6", "#3D5A80", "-"),
        }[kind]
        p = FancyArrowPatch(
            A["xy"],
            B["xy"],
            arrowstyle=st[0],
            connectionstyle=f"arc3,rad={rad}",
            patchA=A["box"],
            patchB=B["box"],
            shrinkA=0.8,
            shrinkB=1.2,
            mutation_scale=1.0,
            lw=0.9,
            color=st[1],
            ls=st[2],
            zorder=49,
        )
        self.ax.add_patch(p)
        if label:
            mx, my = label_xy or (
                (A["xy"][0] + B["xy"][0]) / 2,
                (A["xy"][1] + B["xy"][1]) / 2,
            )
            t = self.text(
                mx,
                my,
                label,
                size,
                color=st[1] if kind == "inhibit" else "#333333",
                z=53,
            )
            t.set_bbox(dict(fc=mpl.rcParams["figure.facecolor"], ec="none", pad=0.6))
        self.note("pathway_edge", label or "", src=a, dst=b, kind=kind)

    def cascade(
        self,
        rows: list[list[tuple[str, str, str]]],
        y_top: float,
        y_bot: float,
        x_lo: float = 6.0,
        x_hi: float = 94.0,
        **kw: Any,
    ) -> None:
        """Evenly distributes pathway nodes across rows."""
        ys = np.linspace(y_top, y_bot, len(rows)) if len(rows) > 1 else [y_top]
        for row, y in zip(rows, ys):
            xs = (
                np.linspace(x_lo, x_hi, len(row) + 2)[1:-1]
                if len(row) > 1
                else [(x_lo + x_hi) / 2]
            )
            for (i, lab, kind), x in zip(row, xs):
                self.node(i, lab, (x, y), kind, **kw)

    # ------------------------------------------------------------ QA & EXPORT
    def qa(self, min_font: float = 6.0) -> dict[str, Any]:
        """Runs layout QA gate on scene."""
        return qa_layout(self.fig, check_outside=True, tol_px=2.0, min_font=min_font)

    def save(
        self,
        out_path: str | Path,
        caption_title: str,
        alt_text: str,
        notes: str = "Schematic illustration; not to scale; not derived from patient images.",
        export_vector: bool = True,
        export_svg: bool = True,
        dpi: int = 600,
        strict: bool = True,
    ) -> FigureResult:
        """
        Saves illustration bundle:
        - PNG at 600 DPI (high-resolution publication line art)
        - Vector PDF with TrueType fonts (fonttype=42)
        - SVG vector
        - _source.csv log
        Validates Zero-PHI and Layout collision QA.
        """
        ax_texts = [t.get_text() for t in self.ax.texts]
        fig_texts = [t.get_text() for t in self.fig.texts]
        assert_zero_phi_text(
            list(self.texts) + ax_texts + fig_texts + [caption_title, alt_text, notes]
        )
        q = self.qa()
        if strict and not q["passed"]:
            raise RuntimeError(f"Layout QA failed for illustration: {q}")

        out_p = Path(out_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        base = out_p.with_suffix("")

        png_path = str(base.with_name(f"{base.name}.png"))
        self.fig.savefig(png_path, dpi=dpi)

        vp: str | None = None
        if export_vector:
            vp = str(base.with_name(f"{base.name}.pdf"))
            self.fig.savefig(vp)

        sp: str | None = None
        if export_svg:
            sp = str(base.with_name(f"{base.name}.svg"))
            self.fig.savefig(sp)

        plt.close(self.fig)

        df = pd.DataFrame(self.log)
        csv_path = str(base.with_name(f"{base.name}_source.csv"))
        df.to_csv(csv_path, index=False)

        full_caption = f"{caption_title}. {notes}" if notes else caption_title
        return FigureResult(
            png_path=png_path,
            alt_text=alt_text,
            caption=full_caption,
            source_df=df,
            csv_path=csv_path,
            vector_path=vp,
            svg_path=sp,
            qa=q,
        )


__all__ = [
    "Scene",
    "SEM",
    "shoelace",
    "ellipse_pts",
    "shrink",
]

# Vector Medical & Scientific Illustration (Cloud Sandbox Edition)

This reference provides self-contained, standalone Python recipes for rendering anatomical cross-sections, cellular biology, blood vessels, and signaling pathways in cloud sandboxes using pure `matplotlib` and `numpy`.

---

## 1. Semantic Art Palette (`SEM`)

```python
SEM = {
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
```

---

## 2. Recipe: Standalone Skin Cross-Section & Ulcer Crater

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from pathlib import Path

def draw_standalone_skin_cross_section(
    out_path: str = "figures/skin_anatomy.png",
    show_crater: bool = True,
):
    """
    Renders an anatomical skin cross-section with epidermis, dermis, hypodermis, and muscle.
    When show_crater=True, carves out an ulcer depression with a granulation tissue bed.
    """
    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=300)
    try:
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 60)
        ax.axis("off")

        xs = np.linspace(5, 95, 200)

        # Epidermis & dermal-epidermal junction
        ytop = np.full_like(xs, 50.0)
        dej = ytop - 4.0 - 1.2 * np.abs(np.sin((xs - 5) * 0.8))
        y_dermis = dej - 14.0 + 0.4 * np.sin(xs * 0.3)
        y_hypo = y_dermis - 12.0 + 0.5 * np.sin(xs * 0.2 + 1)
        y_muscle = y_hypo - 12.0

        if show_crater:
            # Carve crater depression into epidermis and upper dermis
            crater_depth = 12.0 * np.exp(-((xs - 50) ** 2) / (2 * (6.5 ** 2)))
            ytop = ytop - crater_depth
            dej = np.minimum(dej, ytop)

        def draw_band(top, bot, color):
            poly_pts = np.r_[np.c_[xs, top], np.c_[xs[::-1], bot[::-1]]]
            ax.add_patch(Polygon(poly_pts, facecolor=color, edgecolor="none"))

        draw_band(ytop, dej, SEM["epidermis"])
        draw_band(dej, y_dermis, SEM["dermis"])
        draw_band(y_dermis, y_hypo, SEM["hypodermis"])
        draw_band(y_hypo, y_muscle, SEM["muscle"])

        if show_crater:
            # Render granulation bed at crater floor
            cr_mask = np.abs(xs - 50) < 10
            ax.plot(xs[cr_mask], ytop[cr_mask], color=SEM["granulation"], lw=2.5, label="Granulation bed")
            ax.text(50, 36.5, "Ulcer Crater / Granulation", fontsize=7.0, fontweight="bold", color="#78281F", ha="center")

        # Labels
        ax.text(2, 48, "Epidermis", fontsize=7.5, ha="right", va="center")
        ax.text(2, 38, "Dermis", fontsize=7.5, ha="right", va="center")
        ax.text(2, 24, "Subcutis", fontsize=7.5, ha="right", va="center")
        ax.text(2, 10, "Muscle", fontsize=7.5, ha="right", va="center")

        # Mandatory Schematic Disclaimer
        ax.text(
            50, 2.0,
            "Schematic illustration for conceptual mechanism only; not diagnostic patient imaging.",
            fontsize=6.5, fontstyle="italic", ha="center", color="#666666",
        )

        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=600, bbox_inches="tight")
        return str(p)
    finally:
        plt.close(fig)
```

---

## 3. Recipe: Standalone Cellular Elements

```python
def draw_standalone_cellular_elements(ax, cx: float, cy: float, cell_type: str = "neutrophil"):
    """
    Draws biological cell with membrane, nucleus, or granules.
    Supported types: 'rbc', 'neutrophil', 'macrophage', 'lymphocyte'.
    """
    valid_types = ("rbc", "neutrophil", "macrophage", "lymphocyte")
    if cell_type not in valid_types:
        raise ValueError(f"Unsupported cell_type '{cell_type}'. Supported: {valid_types}")

    th = np.linspace(0, 2 * np.pi, 60)
    if cell_type == "rbc":
        x = cx + 2.0 * np.cos(th)
        y = cy + 1.2 * np.sin(th)
        ax.fill(x, y, color=SEM["rbc"], ec="#8E2A25", lw=0.5)
        ax.fill(cx + 1.0 * np.cos(th), cy + 0.6 * np.sin(th), color=SEM["rbc_pallor"])
    elif cell_type == "neutrophil":
        x = cx + 3.0 * np.cos(th)
        y = cy + 3.0 * np.sin(th)
        ax.fill(x, y, color=SEM["neutrophil"], ec=SEM["outline"], lw=0.6, alpha=0.85)
        # 3 nuclear lobes
        for a in [-0.8, 0.0, 0.8]:
            lx = cx + 1.4 * np.cos(a)
            ly = cy + 1.4 * np.sin(a)
            ax.fill(lx + 0.8 * np.cos(th), ly + 0.8 * np.sin(th), color="#4A235A")
    elif cell_type == "macrophage":
        # Amoeboid shape with kidney-shaped nucleus
        r = 3.8 + 0.5 * np.sin(5 * th)
        x = cx + r * np.cos(th)
        y = cy + r * np.sin(th)
        ax.fill(x, y, color=SEM["macrophage"], ec=SEM["outline"], lw=0.6, alpha=0.9)
        kn_th = np.linspace(-np.pi / 2, np.pi / 2, 30)
        nx = cx - 0.5 + 1.5 * np.cos(kn_th)
        ny = cy + 2.0 * np.sin(kn_th)
        ax.fill(nx, ny, color=SEM["nucleus"], ec="#3A2E2E", lw=0.5)
    elif cell_type == "lymphocyte":
        # Large dense round nucleus occupying most of the cell volume
        x = cx + 2.6 * np.cos(th)
        y = cy + 2.6 * np.sin(th)
        ax.fill(x, y, color=SEM["lymphocyte"], ec=SEM["outline"], lw=0.6)
        nx = cx + 2.1 * np.cos(th)
        ny = cy + 2.1 * np.sin(th)
        ax.fill(nx, ny, color=SEM["nucleus"], ec="#3A2E2E", lw=0.5)
```

---

## 4. Recipe: Standalone Blood Vessel Cross-Section

```python
from matplotlib.patches import Circle

def draw_standalone_blood_vessel(ax, cx: float, cy: float, r_outer: float = 12.0, r_inner: float = 8.0):
    """
    Renders an arterial or venous cross-section with muscular wall, endothelium, and lumen.
    """
    th = np.linspace(0, 2 * np.pi, 80)
    # Tunica Media / Muscular Wall
    ax.fill(cx + r_outer * np.cos(th), cy + r_outer * np.sin(th), color=SEM["vessel_wall"], ec=SEM["outline"], lw=0.8)
    # Lumen
    ax.fill(cx + r_inner * np.cos(th), cy + r_inner * np.sin(th), color=SEM["lumen"], ec=SEM["endothelium"], lw=1.2)
    # Endothelial nuclei lining lumen border
    for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        ex = cx + (r_inner + 0.4) * np.cos(a)
        ey = cy + (r_inner + 0.4) * np.sin(a)
        ax.add_patch(Circle((ex, ey), 0.45, color=SEM["nucleus"]))
```

---

## 5. Recipe: Standalone Signaling Pathway Diagram

```python
from matplotlib.patches import FancyBboxPatch

def draw_standalone_signaling_pathway(
    ax, steps: list[str], start_x: float = 20.0, start_y: float = 50.0, step_dx: float = 20.0
):
    """
    Renders a clean conceptual signaling pathway / biological cascade with ligand, receptor, and effector nodes.
    """
    for idx, name in enumerate(steps):
        x = start_x + idx * step_dx
        rect = FancyBboxPatch((x - 7, start_y - 4), 14, 8, boxstyle="round,pad=0.3", fc="#EBF5FB", ec="#2980B9", lw=1.0)
        ax.add_patch(rect)
        ax.text(x, start_y, name, ha="center", va="center", fontsize=7.5, fontweight="bold", color="#1B4F72")
        if idx < len(steps) - 1:
            ax.annotate("", xy=(x + 12, start_y), xytext=(x + 7.5, start_y), arrowprops=dict(arrowstyle="->", color="#2980B9", lw=1.2))
```

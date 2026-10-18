# Vector Medical & Scientific Illustration (Cloud Sandbox Edition)

This reference provides self-contained, standalone Python recipes for rendering anatomical cross-sections, cellular biology, and signaling pathways in cloud sandboxes using pure `matplotlib` and `numpy`.

---

## 1. Semantic Art Palette (`SEM`)

```python
SEM = {
    "rbc": "#D9534F",
    "rbc_pallor": "#F9D5D3",
    "platelet": "#F0AD4E",
    "neutrophil": "#E74C3C",
    "macrophage": "#8E44AD",
    "epidermis": "#E8C4A2",
    "dermis": "#F4A261",
    "hypodermis": "#FCE8A6",
    "muscle": "#C0392B",
    "bone": "#E0E0E0",
    "granulation": "#E67E22",
    "slough": "#F4D03F",
    "eschar": "#1C1C1C",
    "outline": "#222222",
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
    """
    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    xs = np.linspace(5, 95, 200)

    # Epidermis & dermal-epidermal junction
    ytop = 50.0
    dej = ytop - 4.0 - 1.2 * np.abs(np.sin((xs - 5) * 0.8))
    y_dermis = dej - 14.0 + 0.4 * np.sin(xs * 0.3)
    y_hypo = y_dermis - 12.0 + 0.5 * np.sin(xs * 0.2 + 1)
    y_muscle = y_hypo - 12.0

    def draw_band(top, bot, color):
        poly_pts = np.r_[np.c_[xs, top], np.c_[xs[::-1], bot[::-1]]]
        ax.add_patch(Polygon(poly_pts, facecolor=color, edgecolor="none"))

    draw_band(np.full_like(xs, ytop), dej, SEM["epidermis"])
    draw_band(dej, y_dermis, SEM["dermis"])
    draw_band(y_dermis, y_hypo, SEM["hypodermis"])
    draw_band(y_hypo, y_muscle, SEM["muscle"])

    # Labels
    ax.text(2, (ytop + dej[0]) / 2, "Epidermis", fontsize=7.5, ha="right", va="center")
    ax.text(2, (dej[0] + y_dermis[0]) / 2, "Dermis", fontsize=7.5, ha="right", va="center")
    ax.text(2, (y_dermis[0] + y_hypo[0]) / 2, "Subcutis", fontsize=7.5, ha="right", va="center")
    ax.text(2, (y_hypo[0] + y_muscle[0]) / 2, "Muscle", fontsize=7.5, ha="right", va="center")

    # Mandatory Schematic Disclaimer
    ax.text(
        50, 2.0,
        "Schematic illustration for conceptual mechanism only; not diagnostic patient imaging.",
        fontsize=6.5, fontstyle="italic", ha="center", color="#666666",
    )

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return str(p)
```

---

## 3. Recipe: Standalone Cellular Elements

```python
def draw_standalone_cellular_elements(ax, cx: float, cy: float, cell_type: str = "neutrophil"):
    """Draws biological cell with membrane, nucleus, or granules."""
    if cell_type == "rbc":
        th = np.linspace(0, 2 * np.pi, 50)
        x = cx + 2.0 * np.cos(th)
        y = cy + 1.2 * np.sin(th)
        ax.fill(x, y, color=SEM["rbc"], ec="#8E2A25", lw=0.5)
        ax.fill(cx + 1.0 * np.cos(th), cy + 0.6 * np.sin(th), color=SEM["rbc_pallor"])
    elif cell_type == "neutrophil":
        th = np.linspace(0, 2 * np.pi, 60)
        x = cx + 3.0 * np.cos(th)
        y = cy + 3.0 * np.sin(th)
        ax.fill(x, y, color=SEM["neutrophil"], ec=SEM["outline"], lw=0.6, alpha=0.85)
        # 3 nuclear lobes
        for a in [-0.8, 0.0, 0.8]:
            lx = cx + 1.4 * np.cos(a)
            ly = cy + 1.4 * np.sin(a)
            ax.fill(lx + 0.9 * np.cos(th), ly + 0.9 * np.sin(th), color="#4A235A")
```

# Clinical Study Flow Diagrams (Cloud Sandbox Edition)

This reference provides self-contained, standalone Python recipes for generating publication flow diagrams (CONSORT 2025/2010, STROBE, PRISMA 2020) in cloud sandboxes using pure `matplotlib` and `numpy`.

---

## 1. Strict Mathematical Reconciliation Invariant

Every flow diagram must satisfy mathematical reconciliation:
$$\text{Assessed} - \sum \text{Excluded} = \text{Randomized} = \sum \text{Allocated}$$
$$\text{Allocated} = \text{Analyzed} + \text{Excluded from Analysis}$$

If values do not sum exactly, an exception must be raised to prevent reporting discrepancies.

---

## 2. Recipe: Standalone CONSORT 2025 Flow Diagram

```python
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

def draw_standalone_consort_rct(
    assessed: int,
    excluded: dict[str, int],
    arms: list[dict],
    version: str = "2025",
    journal: str = "NEJM",
    out_path: str = "figures/consort_flow.png",
):
    """
    Renders an audit-reconciled CONSORT 2025/2010 flow diagram using pure matplotlib.
    """
    n_ex = sum(excluded.values())
    n_rand = assessed - n_ex
    n_alloc = sum(a["allocated"] for a in arms)
    if n_rand != n_alloc:
        raise ValueError(f"CONSORT reconciliation mismatch: randomized ({n_rand}) != allocated ({n_alloc})")

    fig, ax = plt.subplots(figsize=(7.0, 8.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(cx, cy, w, h, title, subtitle=None, fc="#FFFFFF", ec="#222222"):
        rect = FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="square,pad=0",
            facecolor=fc, edgecolor=ec, linewidth=0.8,
        )
        ax.add_patch(rect)
        if subtitle:
            ax.text(cx, cy + h * 0.18, title, ha="center", va="center", fontsize=8.0, fontweight="bold", color="#111111")
            ax.text(cx, cy - h * 0.20, subtitle, ha="center", va="center", fontsize=7.5, color="#333333")
        else:
            ax.text(cx, cy, title, ha="center", va="center", fontsize=8.0, fontweight="bold", color="#111111")
        return cx, cy

    def arrow(x0, y0, x1, y1):
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color="#222222", lw=0.9),
        )

    # 1. Enrollment
    box(45, 94, 38, 6, "Assessed for eligibility", f"(N = {assessed:,})")
    arrow(45, 91, 45, 80)
    
    # Excluded box
    excl_lines = [f"\u2022 {k} (n = {v:,})" for k, v in excluded.items()]
    box(78, 85, 30, 8, f"Excluded (n = {n_ex:,})", "\n".join(excl_lines[:2]))
    arrow(45, 85, 63, 85)

    # 2. Randomized
    box(45, 77, 34, 6, "Randomized" if version == "2025" else "Randomised", f"(N = {n_rand:,})")

    # Split to arms
    arrow(45, 74, 25, 66)
    arrow(45, 74, 75, 66)

    # 3. Allocation
    for idx, (arm, cx) in enumerate(zip(arms, [25, 75])):
        box(cx, 63, 40, 6, f"Allocated to {arm['name']}", f"(n = {arm['allocated']:,})")
        arrow(cx, 60, cx, 48)
        
        # Follow-up
        lo = sum(arm.get("lost", {}).values()) if isinstance(arm.get("lost"), dict) else arm.get("lost", 0)
        di = sum(arm.get("discontinued", {}).values()) if isinstance(arm.get("discontinued"), dict) else arm.get("discontinued", 0)
        box(cx, 44, 40, 7, "Follow-Up", f"\u2022 Lost to follow-up (n = {lo})\n\u2022 Discontinued (n = {di})")
        arrow(cx, 40.5, cx, 28)

        # Analysis (2025: itemized primary outcome)
        po = " for primary outcome" if version == "2025" else ""
        box(cx, 24, 40, 7, f"Analysed{po}", f"(n = {arm['analyzed']:,})")

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return str(p)
```

---

## 3. Recipe: Standalone STROBE Observational Flow

```python
def draw_standalone_strobe_flow(
    stages: list[tuple[str, int, int, str]],  # (stage_name, retained_n, excluded_n, exclusion_reason)
    out_path: str = "figures/strobe_flow.png",
):
    """
    Renders an audit-reconciled sequential cohort exclusion flow diagram.
    """
    fig, ax = plt.subplots(figsize=(6.5, 2.0 + len(stages) * 1.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, len(stages) * 20 + 10)
    ax.axis("off")

    y_cursor = len(stages) * 20
    for idx, (name, n_retain, n_ex, reason) in enumerate(stages):
        # Center cohort box
        rect = FancyBboxPatch((20, y_cursor - 6), 40, 10, boxstyle="square,pad=0", fc="#FFFFFF", ec="#222222", lw=0.8)
        ax.add_patch(rect)
        ax.text(40, y_cursor - 0.5, name, ha="center", va="center", fontsize=8.0, fontweight="bold")
        ax.text(40, y_cursor - 4.0, f"(N = {n_retain:,})", ha="center", va="center", fontsize=7.5, color="#333333")

        # Excluded side box if n_ex > 0
        if n_ex > 0:
            ax.annotate("", xy=(68, y_cursor - 1), xytext=(60, y_cursor - 1), arrowprops=dict(arrowstyle="->", color="#222222", lw=0.9))
            ex_rect = FancyBboxPatch((68, y_cursor - 6), 28, 8, boxstyle="square,pad=0", fc="#F8F9FA", ec="#777777", lw=0.7)
            ax.add_patch(ex_rect)
            ax.text(82, y_cursor - 1.5, f"Excluded (n = {n_ex:,})", ha="center", va="center", fontsize=7.0, fontweight="bold")
            ax.text(82, y_cursor - 4.5, reason, ha="center", va="center", fontsize=6.5, color="#555555")

        if idx < len(stages) - 1:
            ax.annotate("", xy=(40, y_cursor - 16), xytext=(40, y_cursor - 6), arrowprops=dict(arrowstyle="->", color="#222222", lw=0.9))

        y_cursor -= 18

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return str(p)
```

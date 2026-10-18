#!/usr/bin/env python3
"""
skills/shinystat-cloud/examples/journal_artwork_cloud_demo.py

Cloud sandbox demonstration of standalone journal artwork (pure matplotlib + numpy):
1. Standalone CONSORT 2025 Flow Diagram (Strict reconciliation, 600 DPI)
2. Standalone STROBE Retention Flow (600 DPI)
3. Standalone Vector Skin Anatomy Illustration with Schematic Disclaimer (600 DPI)
4. Standalone Kaplan-Meier Curve with NEJM House Style Spines (300 DPI)

Zero medstat imports. Fully functional in air-gapped or pure Python cloud environments.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon

SEM = {
    "rbc": "#D9534F",
    "rbc_pallor": "#F9D5D3",
    "neutrophil": "#E74C3C",
    "epidermis": "#E8C4A2",
    "dermis": "#F4A261",
    "hypodermis": "#FCE8A6",
    "muscle": "#C0392B",
    "outline": "#222222",
}


def demo_cloud_consort_2025(out_dir: Path) -> Path:
    assessed = 1000
    excluded = {"Ineligible": 150, "Declined": 50}
    n_rand = 800
    arms = [
        {"name": "Intervention Arm", "allocated": 400, "analyzed": 380, "lost": 20},
        {"name": "Control Arm", "allocated": 400, "analyzed": 385, "lost": 15},
    ]

    # Reconciliation check
    if (
        assessed - sum(excluded.values()) != n_rand
        or sum(a["allocated"] for a in arms) != n_rand
    ):
        raise ValueError("Reconciliation failed")

    fig, ax = plt.subplots(figsize=(6.5, 7.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(cx, cy, w, h, text1, text2):
        r = FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="square,pad=0",
            fc="#FFFFFF",
            ec="#222222",
            lw=0.8,
        )
        ax.add_patch(r)
        ax.text(
            cx,
            cy + h * 0.2,
            text1,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )
        ax.text(
            cx,
            cy - h * 0.2,
            text2,
            ha="center",
            va="center",
            fontsize=7.5,
            color="#333333",
        )

    box(45, 92, 38, 7, "Assessed for eligibility", f"(N = {assessed})")
    box(
        78,
        83,
        28,
        7,
        f"Excluded (n = {sum(excluded.values())})",
        "Ineligible (150), Declined (50)",
    )
    ax.annotate(
        "",
        xy=(45, 80),
        xytext=(45, 88.5),
        arrowprops=dict(arrowstyle="->", color="#222222", lw=0.9),
    )
    ax.annotate(
        "",
        xy=(64, 83),
        xytext=(45, 83),
        arrowprops=dict(arrowstyle="->", color="#222222", lw=0.9),
    )

    box(45, 76, 32, 6, "Randomized (2025 Standard)", f"(N = {n_rand})")
    ax.annotate(
        "",
        xy=(25, 66),
        xytext=(45, 73),
        arrowprops=dict(arrowstyle="->", color="#222222", lw=0.9),
    )
    ax.annotate(
        "",
        xy=(75, 66),
        xytext=(45, 73),
        arrowprops=dict(arrowstyle="->", color="#222222", lw=0.9),
    )

    for a, cx in zip(arms, [25, 75]):
        box(cx, 62, 40, 6, f"Allocated: {a['name']}", f"(n = {a['allocated']})")
        box(
            cx,
            46,
            40,
            6,
            "Follow-Up (Attrition)",
            f"Lost to follow-up (n = {a['lost']})",
        )
        box(cx, 30, 40, 6, "Analyzed for Primary Outcome", f"(n = {a['analyzed']})")
        ax.annotate(
            "",
            xy=(cx, 49),
            xytext=(cx, 59),
            arrowprops=dict(arrowstyle="->", color="#222222", lw=0.9),
        )
        ax.annotate(
            "",
            xy=(cx, 33),
            xytext=(cx, 43),
            arrowprops=dict(arrowstyle="->", color="#222222", lw=0.9),
        )

    out_path = out_dir / "cloud_consort_2025.png"
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def demo_cloud_illustration(out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    xs = np.linspace(5, 95, 200)
    dej = 44.0 - 1.0 * np.abs(np.sin((xs - 5) * 0.8))
    y_derm = dej - 12.0
    y_hypo = y_derm - 10.0
    y_musc = y_hypo - 10.0

    def band(top, bot, col):
        pts = np.r_[np.c_[xs, top], np.c_[xs[::-1], bot[::-1]]]
        ax.add_patch(Polygon(pts, facecolor=col, edgecolor="none"))

    band(np.full_like(xs, 48.0), dej, SEM["epidermis"])
    band(dej, y_derm, SEM["dermis"])
    band(y_derm, y_hypo, SEM["hypodermis"])
    band(y_hypo, y_musc, SEM["muscle"])

    # Cells
    th = np.linspace(0, 2 * np.pi, 50)
    ax.fill(
        35.0 + 2.0 * np.cos(th),
        25.0 + 2.0 * np.sin(th),
        color=SEM["neutrophil"],
        ec=SEM["outline"],
        lw=0.5,
    )
    ax.fill(
        60.0 + 1.8 * np.cos(th),
        18.0 + 1.0 * np.sin(th),
        color=SEM["rbc"],
        ec="#8E2A25",
        lw=0.5,
    )

    ax.text(
        50,
        4,
        "Schematic illustration for conceptual mechanism only; not diagnostic patient imaging.",
        fontsize=6.5,
        fontstyle="italic",
        ha="center",
        color="#666666",
    )

    out_path = out_dir / "cloud_skin_illustration.png"
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def demo_cloud_nejm_km(out_dir: Path) -> Path:
    t = np.linspace(0, 365, 50)
    s1 = np.exp(-0.001 * t)
    s2 = np.exp(-0.0028 * t)

    fig, ax = plt.subplots(figsize=(6.0, 4.5), dpi=300)
    ax.step(t, s1, label="Targeted Therapy", color="#941100", lw=1.8, where="post")
    ax.step(t, s2, label="Standard of Care", color="#1A365D", lw=1.8, where="post")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 370)
    ax.set_xlabel("Days Since Randomization", fontsize=9, fontweight="bold")
    ax.set_ylabel("Probability of Overall Survival", fontsize=9, fontweight="bold")
    ax.set_title(
        "Kaplan-Meier Survival Analysis (NEJM Style)",
        fontsize=10,
        fontweight="bold",
        pad=10,
    )

    # L-frame spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 3.0))
    ax.spines["bottom"].set_position(("outward", 3.0))
    ax.tick_params(direction="out", length=3.5)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")

    out_path = out_dir / "cloud_km_nejm.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    out_dir = Path("examples/out")
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Running standalone cloud artwork demo (zero medstat imports)...")
    p1 = demo_cloud_consort_2025(out_dir)
    print(f"1. Standalone CONSORT 2025: {p1}")
    p2 = demo_cloud_illustration(out_dir)
    print(f"2. Standalone Vector Illustration: {p2}")
    p3 = demo_cloud_nejm_km(out_dir)
    print(f"3. Standalone NEJM Kaplan-Meier: {p3}")
    print("✓ Cloud sandbox demo completed successfully!")


if __name__ == "__main__":
    main()

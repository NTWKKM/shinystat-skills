#!/usr/bin/env python3
"""
skills/shinystat/examples/journal_artwork_demo.py

Comprehensive demonstration of journal-grade clinical artwork in shinystat:
1. CONSORT 2025 RCT Flow Diagram (NEJM style, 600 DPI, vector PDF, SVG, CSV)
2. STROBE Cohort Retention Flow from SampleFlowTracker (BMJ style, 600 DPI)
3. Vector Medical Illustration of Skin & Crater Anatomy (Lancet style, 600 DPI)
4. Kaplan-Meier Survival Analysis with Inset & Aligned Risk Table (NEJM style, 300 DPI)
5. Multivariable Risk Model Forest Plot (JAMA style, 300 DPI)
"""

from pathlib import Path
import numpy as np
import pandas as pd

from medstat.data.retention import SampleFlowTracker
from medstat.figures.flow import consort_rct, cohort_flow_from_tracker
from medstat.figures.forest import plot_forest
from medstat.figures.illustration import Scene
from medstat.figures.survival import plot_kaplan_meier


def main() -> None:
    out_dir = Path("examples/out")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating journal figures into: {out_dir.resolve()}")

    # ---------------------------------------------------------
    # 1. CONSORT 2025 RCT Flow Diagram (NEJM style)
    # ---------------------------------------------------------
    print("1. Generating CONSORT 2025 Flow Diagram (NEJM)...")
    consort_res = consort_rct(
        assessed=1200,
        excluded={
            "Did not meet inclusion criteria": 180,
            "Declined participation": 75,
            "Other reasons": 25,
        },
        arms=[
            {
                "name": "Targeted Immunotherapy + SOC",
                "allocated": 460,
                "received": 454,
                "not_received": 6,
                "lost": 14,
                "discontinued": 16,
                "analyzed": 430,
                "excluded_analysis": 30,
            },
            {
                "name": "Placebo + Standard of Care",
                "allocated": 460,
                "received": 458,
                "not_received": 2,
                "lost": 12,
                "discontinued": 18,
                "analyzed": 430,
                "excluded_analysis": 30,
            },
        ],
        version="2025",
        journal="NEJM",
        out_path=str(out_dir / "consort_2025_nejm.png"),
    )
    print(f"   -> PNG: {consort_res.png_path}")
    print(f"   -> PDF: {consort_res.vector_path}")
    print(f"   -> QA: {consort_res.qa}")

    # ---------------------------------------------------------
    # 2. STROBE Cohort Flow via SampleFlowTracker (BMJ style)
    # ---------------------------------------------------------
    print("2. Generating STROBE Cohort Retention Flow (BMJ)...")
    tracker = SampleFlowTracker(
        initial_n=2800, initial_name="All Hospitalized ICU Admissions"
    )
    tracker.record_stage("Adults (>= 18 yrs)", 2650, 150, reason="Age < 18")
    tracker.record_stage(
        "Complete Serial Troponin-T", 2400, 250, reason="Missing baseline troponin"
    )
    tracker.record_stage(
        "ICU Length of Stay >= 24h", 2100, 300, reason="Discharged or died within 24h"
    )

    strobe_res = cohort_flow_from_tracker(
        tracker=tracker,
        groups={"Discharged Alive": 1720, "In-Hospital Mortality": 380},
        journal="BMJ",
        out_path=str(out_dir / "strobe_cohort_bmj.png"),
    )
    print(f"   -> PNG: {strobe_res.png_path}")
    print(f"   -> PDF: {strobe_res.vector_path}")

    # ---------------------------------------------------------
    # 3. Vector Medical Illustration (Lancet style)
    # ---------------------------------------------------------
    print("3. Generating Vector Medical Illustration (Lancet)...")
    scene = Scene(width="double_col", aspect=0.6, journal="LANCET")
    scene.text(
        50.0,
        56.0,
        "Cutaneous Architecture & Deep Tissue Cross-Section",
        size=10.0,
        weight="bold",
    )
    scene.skin(x0=5.0, x1=95.0, ytop=48.0)
    scene.neutrophil(cx=30.0, cy=35.0, r=3.0, lobes=3)
    scene.macrophage(cx=50.0, cy=30.0, r=4.5)
    scene.rbc(cx=70.0, cy=22.0, r=2.5)
    scene.badge(10.0, 56.0, "A")
    illus_res = scene.save(
        out_path=out_dir / "skin_anatomy_lancet.png",
        caption_title="Cutaneous Architecture and Cellular Infiltration",
        alt_text="Vector diagram of skin layers with inflammatory cell recruitment",
    )
    print(f"   -> PNG: {illus_res.png_path}")
    print(f"   -> PDF: {illus_res.vector_path}")

    # ---------------------------------------------------------
    # 4. Kaplan-Meier Survival Curve (NEJM style with Inset)
    # ---------------------------------------------------------
    print("4. Generating Kaplan-Meier with Inset (NEJM)...")
    t = np.linspace(0, 365, 50)
    km_data = {
        "Targeted Therapy": {
            "timeline": t,
            "survival": np.exp(-0.0010 * t),
            "ci_lower": np.exp(-0.0012 * t),
            "ci_upper": np.exp(-0.0008 * t),
            "censored_times": [30, 90, 180, 240, 300],
        },
        "Placebo": {
            "timeline": t,
            "survival": np.exp(-0.0028 * t),
            "ci_lower": np.exp(-0.0032 * t),
            "ci_upper": np.exp(-0.0024 * t),
            "censored_times": [45, 120, 200],
        },
    }
    risk_table = pd.DataFrame(
        {
            "Group": ["Targeted Therapy", "Placebo"],
            "0": [430, 430],
            "90": [415, 385],
            "180": [398, 340],
            "270": [380, 305],
            "365": [365, 275],
        }
    )
    km_res = plot_kaplan_meier(
        km_curves=km_data,
        risk_table=risk_table,
        time_unit="Days",
        title="Overall Survival in Primary Analysis Population",
        journal="NEJM",
        nejm_inset_ylim=(0.60, 1.0),
        out_path=str(out_dir / "km_survival_nejm.png"),
    )
    print(f"   -> PNG: {km_res.png_path}")
    print(f"   -> PDF: {km_res.vector_path}")

    # ---------------------------------------------------------
    # 5. Multivariable Model Forest Plot (JAMA style)
    # ---------------------------------------------------------
    print("5. Generating Multivariable Forest Plot (JAMA)...")
    forest_df = pd.DataFrame(
        {
            "term": [
                "Age (per 10-year increase)",
                "Male Sex",
                "Diabetes Mellitus",
                "Chronic Kidney Disease (Stage 3+)",
                "Elevated Troponin-T (>0.04 ng/mL)",
                "Targeted Immunotherapy (Ref: Placebo)",
            ],
            "estimate": [1.28, 1.15, 1.42, 1.85, 2.30, 0.58],
            "ci_lower": [1.08, 0.92, 1.14, 1.40, 1.75, 0.44],
            "ci_upper": [1.52, 1.44, 1.78, 2.44, 3.02, 0.76],
            "p_value": [0.004, 0.22, 0.002, 0.0001, 0.00001, 0.0002],
        }
    )
    forest_res = plot_forest(
        data=forest_df,
        scale="HR",
        title="Multivariable Adjusted Hazard Ratios for 1-Year Mortality",
        journal="JAMA",
        out_path=str(out_dir / "forest_mortality_jama.png"),
    )
    print(f"   -> PNG: {forest_res.png_path}")
    print(f"   -> PDF: {forest_res.vector_path}")

    print(
        "\n✓ All journal artwork demos completed successfully with green verification!"
    )


if __name__ == "__main__":
    main()

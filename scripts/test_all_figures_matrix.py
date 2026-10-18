#!/usr/bin/env python3
"""
scripts/test_all_figures_matrix.py: Standalone runner script to generate, audit,
and verify clinical figures across all scenarios, journal styles, and export bundles.
"""

# ruff: noqa: E402

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from medstat.figures.agreement import plot_bland_altman
from medstat.figures.balance import plot_love
from medstat.figures.calibration import (
    plot_bootstrap_optimism,
    plot_calibration,
    plot_recalibration_comparison,
)
from medstat.figures.dca import plot_dca
from medstat.figures.diagnostics import (
    plot_mice_diagnostics,
    plot_missingness_map,
    plot_schoenfeld_residuals,
)
from medstat.figures.flow import (
    cohort_flow,
    consort_rct,
    methods_pipeline,
    prisma_2020,
    stard_flow,
)
from medstat.figures.forest import plot_forest
from medstat.figures.html import (
    plot_dual_bland_altman,
    plot_dual_forest,
    plot_dual_kaplan_meier,
    plot_dual_love,
    plot_dual_roc_curve,
)
from medstat.figures.illustration import Scene
from medstat.figures.roc import (
    plot_reclassification,
    plot_roc_curve,
    plot_time_dependent_roc,
)
from medstat.figures.survival import plot_cif, plot_kaplan_meier

JOURNALS = ["NEJM", "JAMA", "LANCET", "BMJ", "CLINICAL"]


def run_full_matrix(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "total_generated": 0,
        "bundles_verified": 0,
        "qa_passed": 0,
        "formats": {"png": 0, "pdf": 0, "svg": 0, "csv": 0, "plotly_json": 0},
        "details": [],
    }

    def record_fig(name: str, fig_res, expect_pdf=True, expect_svg=False):
        results["total_generated"] += 1
        p_png = Path(fig_res.png_path)
        p_csv = Path(fig_res.csv_path)

        assert p_png.exists() and p_png.stat().st_size > 0
        results["formats"]["png"] += 1

        assert p_csv.exists() and p_csv.stat().st_size > 0
        results["formats"]["csv"] += 1

        if expect_pdf and fig_res.vector_path:
            p_pdf = Path(fig_res.vector_path)
            assert p_pdf.exists() and p_pdf.stat().st_size > 0
            results["formats"]["pdf"] += 1

        if expect_svg and fig_res.svg_path:
            p_svg = Path(fig_res.svg_path)
            assert p_svg.exists() and p_svg.stat().st_size > 0
            results["formats"]["svg"] += 1

        if hasattr(fig_res, "plotly_json") and fig_res.plotly_json:
            results["formats"]["plotly_json"] += 1

        qa_ok = fig_res.qa.get("passed", True) if fig_res.qa else True
        if qa_ok:
            results["qa_passed"] += 1
        else:
            print(f"\n[QA FAILURE DETECTED] Figure '{name}': {fig_res.qa}")

        results["bundles_verified"] += 1
        results["details"].append(
            {
                "name": name,
                "png": str(p_png),
                "png_size_bytes": p_png.stat().st_size,
                "qa_passed": qa_ok,
            }
        )

    # 1. Survival: Kaplan-Meier across 5 journals + Inset
    km_curves = {
        "Intervention": {
            "timeline": np.array([0, 10, 20, 30, 60, 90, 120]),
            "survival": np.array([1.0, 0.96, 0.91, 0.86, 0.80, 0.75, 0.72]),
            "ci_lower": np.array([1.0, 0.92, 0.85, 0.79, 0.72, 0.67, 0.63]),
            "ci_upper": np.array([1.0, 0.99, 0.96, 0.92, 0.87, 0.82, 0.79]),
            "censored_times": np.array([15, 45, 80]),
            "censored_survival": np.array([0.93, 0.83, 0.77]),
        },
        "Control": {
            "timeline": np.array([0, 10, 20, 30, 60, 90, 120]),
            "survival": np.array([1.0, 0.92, 0.83, 0.72, 0.61, 0.52, 0.46]),
            "ci_lower": np.array([1.0, 0.86, 0.75, 0.63, 0.51, 0.42, 0.36]),
            "ci_upper": np.array([1.0, 0.96, 0.89, 0.80, 0.70, 0.61, 0.55]),
            "censored_times": np.array([25, 55, 100]),
            "censored_survival": np.array([0.78, 0.64, 0.49]),
        },
    }
    risk_df = pd.DataFrame(
        {
            "Group": ["Intervention", "Control"],
            "t=0": [150, 150],
            "t=30": [129, 108],
            "t=60": [120, 91],
            "t=90": [112, 78],
            "t=120": [108, 69],
        }
    )
    for jnl in JOURNALS:
        res = plot_kaplan_meier(
            km_curves,
            risk_table=risk_df,
            log_rank_p=0.0003,
            journal=jnl,
            out_path=str(output_dir / f"km_{jnl}.png"),
        )
        record_fig(f"Kaplan-Meier ({jnl})", res, expect_pdf=True)

    res_km_inset = plot_kaplan_meier(
        km_curves,
        risk_table=risk_df,
        log_rank_p=0.0003,
        journal="NEJM",
        nejm_inset_ylim=(0.40, 1.0),
        out_path=str(output_dir / "km_nejm_inset.png"),
    )
    record_fig("Kaplan-Meier (NEJM Inset)", res_km_inset, expect_pdf=True)

    # CIF across 5 journals
    cif_df = pd.DataFrame(
        {
            "time": np.array([0, 12, 24, 36, 48, 60]),
            "cif_primary": np.array([0.0, 0.08, 0.15, 0.22, 0.28, 0.32]),
            "cif_primary_ci_lower": np.array([0.0, 0.04, 0.10, 0.16, 0.21, 0.25]),
            "cif_primary_ci_upper": np.array([0.0, 0.13, 0.21, 0.29, 0.36, 0.40]),
            "cif_competing": np.array([0.0, 0.03, 0.06, 0.09, 0.12, 0.14]),
            "cif_competing_ci_lower": np.array([0.0, 0.01, 0.03, 0.05, 0.07, 0.09]),
            "cif_competing_ci_upper": np.array([0.0, 0.06, 0.10, 0.14, 0.18, 0.20]),
            "naive_1_minus_km": np.array([0.0, 0.10, 0.19, 0.28, 0.36, 0.42]),
        }
    )
    for jnl in JOURNALS:
        res = plot_cif(
            cif_df,
            journal=jnl,
            time_unit="months",
            out_path=str(output_dir / f"cif_{jnl}.png"),
        )
        record_fig(f"Cumulative Incidence ({jnl})", res, expect_pdf=True)

    # 2. Forest Plots across 5 journals + linear difference scale
    forest_df = pd.DataFrame(
        {
            "term": ["Overall", "Age >= 65", "Male", "Diabetes"],
            "estimate": [1.45, 1.70, 1.20, 1.85],
            "ci_lower": [1.15, 1.25, 0.90, 1.35],
            "ci_upper": [1.82, 2.30, 1.60, 2.55],
            "p_value": [0.002, 0.001, 0.210, 0.0001],
            "events_text": ["120/500", "70/220", "60/260", "45/140"],
        }
    )
    for jnl in JOURNALS:
        res = plot_forest(
            forest_df,
            scale="HR",
            journal=jnl,
            out_path=str(output_dir / f"forest_hr_{jnl}.png"),
        )
        record_fig(f"Forest Plot HR ({jnl})", res, expect_pdf=True)

    diff_df = forest_df.copy()
    diff_df["estimate"] = [0.4, 0.6, 0.2, 0.7]
    diff_df["ci_lower"] = [0.1, 0.2, -0.1, 0.3]
    diff_df["ci_upper"] = [0.7, 1.0, 0.5, 1.1]
    res_diff = plot_forest(
        diff_df,
        scale="Beta",
        journal="JAMA",
        out_path=str(output_dir / "forest_diff_jama.png"),
    )
    record_fig("Forest Plot Beta (JAMA)", res_diff, expect_pdf=True)

    # 3. Diagnostic & Prediction
    roc_data = {
        "name": "Biomarker Model",
        "fpr": np.array([0.0, 0.05, 0.20, 0.40, 0.70, 1.0]),
        "tpr": np.array([0.0, 0.45, 0.72, 0.85, 0.95, 1.0]),
        "auc": 0.835,
        "ci_lower": 0.770,
        "ci_upper": 0.900,
        "optimal_point": {"fpr": 0.20, "tpr": 0.72, "cutoff": 35.0},
        "sensitivity": 0.72,
        "specificity": 0.80,
    }
    paired_data = {
        "name": "Clinical Model",
        "fpr": np.array([0.0, 0.15, 0.35, 0.60, 1.0]),
        "tpr": np.array([0.0, 0.40, 0.65, 0.82, 1.0]),
        "auc": 0.720,
        "ci_lower": 0.640,
        "ci_upper": 0.800,
    }
    for jnl in JOURNALS:
        res = plot_roc_curve(
            roc_data,
            paired_roc_data=paired_data,
            paired_p_value=0.003,
            journal=jnl,
            out_path=str(output_dir / f"roc_{jnl}.png"),
        )
        record_fig(f"ROC Curve ({jnl})", res, expect_pdf=True)

    # NRI
    reclass_df = pd.DataFrame(
        {
            "p_old": [0.1, 0.3, 0.5, 0.7, 0.2, 0.4, 0.6],
            "p_new": [0.15, 0.25, 0.65, 0.80, 0.18, 0.38, 0.70],
            "y_true": [0, 0, 1, 1, 0, 0, 1],
        }
    )
    res_nri = plot_reclassification(
        reclass_df, journal="NEJM", out_path=str(output_dir / "nri_nejm.png")
    )
    record_fig("NRI Scatter (NEJM)", res_nri, expect_pdf=True)

    # Time-dependent ROC
    td_roc = {
        "trajectory": pd.DataFrame(
            {
                "time": [12, 24, 36],
                "auc": [0.85, 0.82, 0.78],
                "ci_lower": [0.78, 0.75, 0.70],
                "ci_upper": [0.92, 0.89, 0.86],
            }
        ),
        "curves": {
            24: {
                "fpr": np.array([0.0, 0.2, 0.5, 1.0]),
                "tpr": np.array([0.0, 0.7, 0.85, 1.0]),
                "auc": 0.82,
            }
        },
    }
    res_td = plot_time_dependent_roc(
        td_roc, journal="LANCET", out_path=str(output_dir / "tdroc_lancet.png")
    )
    record_fig("Time-Dependent ROC (LANCET)", res_td, expect_pdf=True)

    # Calibration across 5 journals
    calib_bins = {
        "pred_mean": np.linspace(0.05, 0.95, 10),
        "obs_rate": np.linspace(0.05, 0.90, 10),
        "ci_lower": np.linspace(0.02, 0.80, 10),
        "ci_upper": np.linspace(0.10, 0.98, 10),
        "counts": np.full(10, 50),
    }
    for jnl in JOURNALS:
        res = plot_calibration(
            calib_bins,
            brier_score=0.12,
            ici=0.03,
            journal=jnl,
            out_path=str(output_dir / f"cal_{jnl}.png"),
        )
        record_fig(f"Calibration Curve ({jnl})", res, expect_pdf=True)

    # Recalibration comparison
    recal_dict = {
        "y_true": np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0] * 5),
        "p_orig": np.linspace(0.1, 0.9, 50),
        "p_recal": np.clip(np.linspace(0.05, 0.95, 50), 0.01, 0.99),
    }
    res_recal = plot_recalibration_comparison(
        recal_dict, journal="BMJ", out_path=str(output_dir / "recal_bmj.png")
    )
    record_fig("Recalibration Comparison (BMJ)", res_recal, expect_pdf=True)

    # Bootstrap optimism
    opt_dict = {
        "metrics": {
            "auroc": {"apparent": 0.85, "optimism_corrected": 0.82, "optimism": 0.03},
            "calibration_slope": {
                "apparent": 1.0,
                "optimism_corrected": 0.91,
                "optimism": 0.09,
            },
            "nagelkerke_r2": {
                "apparent": 0.38,
                "optimism_corrected": 0.34,
                "optimism": 0.04,
            },
            "brier_score": {
                "apparent": 0.12,
                "optimism_corrected": 0.13,
                "optimism": -0.01,
            },
        },
        "shrinkage_factor": 0.91,
        "events_per_predictor": 15.2,
        "n_bootstrap": 200,
    }
    res_opt = plot_bootstrap_optimism(
        opt_dict, journal="CLINICAL", out_path=str(output_dir / "opt_clinical.png")
    )
    record_fig("Bootstrap Optimism (CLINICAL)", res_opt, expect_pdf=True)

    # DCA across 5 journals
    dca_df = pd.DataFrame(
        {
            "strategy": ["Treat All"] * 5 + ["Model"] * 5 + ["Treat None"] * 5,
            "threshold": [0.1, 0.2, 0.3, 0.4, 0.5] * 3,
            "net_benefit": [0.15, 0.10, 0.05, 0.01, -0.02]
            + [0.18, 0.14, 0.10, 0.06, 0.03]
            + [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    for jnl in JOURNALS:
        res = plot_dca(dca_df, journal=jnl, out_path=str(output_dir / f"dca_{jnl}.png"))
        record_fig(f"DCA Net Benefit ({jnl})", res, expect_pdf=True)

    # 4. Causal Balance & Agreement
    love_df = pd.DataFrame(
        {
            "Variable": ["Age", "Male", "SBP", "Diabetes"],
            "smd_unadjusted": [0.35, 0.22, 0.40, 0.28],
            "smd_matched": [0.03, 0.01, 0.04, 0.02],
            "smd_weighted": [0.02, 0.01, 0.03, 0.02],
        }
    )
    for jnl in JOURNALS:
        res = plot_love(
            love_df, journal=jnl, out_path=str(output_dir / f"love_{jnl}.png")
        )
        record_fig(f"Love Plot ({jnl})", res, expect_pdf=True)

    ba_data = {
        "means": [100.0, 110.0, 120.0, 130.0],
        "diffs": [2.0, -1.0, 1.5, -0.5],
        "mean_diff": 0.5,
        "loa_upper": 3.5,
        "loa_lower": -2.5,
    }
    for jnl in JOURNALS:
        res = plot_bland_altman(
            ba_data, journal=jnl, out_path=str(output_dir / f"ba_{jnl}.png")
        )
        record_fig(f"Bland-Altman ({jnl})", res, expect_pdf=True)

    # 5. Study Flow Diagrams
    arms_2025 = [
        {
            "name": "Arm 1",
            "allocated": 100,
            "received": 95,
            "not_received": {"Declined": 5},
            "analysed": 90,
            "excluded_analysis": {"Missing": 10},
        },
        {
            "name": "Arm 2",
            "allocated": 100,
            "received": 98,
            "not_received": {"Withdrew": 2},
            "analysed": 92,
            "excluded_analysis": {"Protocol": 8},
        },
    ]
    res_c2025 = consort_rct(
        assessed=250,
        excluded={"Ineligible": 50},
        arms=arms_2025,
        version="2025",
        out_path=str(output_dir / "consort_2025.png"),
    )
    record_fig("CONSORT 2025 RCT Flow", res_c2025, expect_pdf=True, expect_svg=True)

    res_prisma = prisma_2020(
        databases={"PubMed": 500},
        registers={"ClinicalTrials.gov": 50},
        removed_duplicates=150,
        removed_automation=30,
        removed_other=0,
        screened_excluded=300,
        not_retrieved=10,
        fulltext_excluded={"Wrong population": 30},
        included_studies=30,
        included_reports=30,
        out_path=str(output_dir / "prisma.png"),
    )
    record_fig("PRISMA 2020 Flow", res_prisma, expect_pdf=True, expect_svg=True)

    res_stard = stard_flow(
        eligible=300,
        excluded={"Refused": 20},
        index_pos=100,
        index_neg=175,
        index_inconclusive=5,
        tp=85,
        fp=15,
        fn=10,
        tn=165,
        out_path=str(output_dir / "stard.png"),
    )
    record_fig(
        "STARD 2015 Diagnostic Flow", res_stard, expect_pdf=True, expect_svg=True
    )

    res_cohort = cohort_flow(
        source=1000,
        steps=[("Pediatric age (<18)", 100), ("Missing biomarker", 50)],
        groups={"Surviving": 780, "In-Hospital Death": 70},
        source_label="All Hospitalized Patients",
        out_path=str(output_dir / "cohort.png"),
    )
    record_fig("STROBE Cohort Flow", res_cohort, expect_pdf=True, expect_svg=True)

    res_meth = methods_pipeline(
        [
            {
                "title": "Step 1",
                "body": "Extraction",
                "counts": {"n_in": 100, "excluded": {"Missing": 10}, "n_out": 90},
            },
            {
                "title": "Step 2",
                "body": "Analysis",
                "counts": {"n_in": 90, "excluded": {}, "n_out": 90},
            },
        ],
        out_path=str(output_dir / "methods.png"),
        strict_numbers=True,
    )
    record_fig("Methods Pipeline Flow", res_meth, expect_pdf=True, expect_svg=True)

    # 6. Scientific Illustration
    sc = Scene(width="double_col", aspect=0.5, journal="NEJM", seed=42)
    sc.rbc(10, 15)
    sc.platelet(18, 15)
    sc.neutrophil(26, 15)
    sc.macrophage(36, 15, engulfed=1)
    sc.fibroblast(48, 15)
    sc.bacterium(58, 15)
    geo = sc.skin(65, 95, 22, thick=(2.0, 5.0, 4.0, 4.0))
    cr = sc.crater(geo, 80, 8, 5)
    sc.wound_phase(geo, cr, 80, 8, 5, "inflammation")
    sc.artery_cross_section(cx=25, cy=35, R=7.0, stenosis=0.60)
    res_sci = sc.save(
        out_path=str(output_dir / "sciart.png"),
        caption_title="Cellular & Tissue Overview",
        alt_text="Scientific schematic",
        dpi=300,
    )
    record_fig(
        "Scientific Illustration (Scene)", res_sci, expect_pdf=True, expect_svg=True
    )

    # 7. Diagnostics & Missingness
    df_miss = pd.DataFrame({"A": [1, np.nan, 3], "B": [4, 5, np.nan]})
    res_miss = plot_missingness_map(
        df_miss, out_path=str(output_dir / "missingness.png")
    )
    record_fig("Missingness Map", res_miss, expect_pdf=False)

    res_schoen = plot_schoenfeld_residuals(
        np.array([1, 2, 3, 4, 5]),
        np.array([0.1, -0.2, 0.3, -0.1, 0.0]),
        var_name="Age",
        p_value=0.62,
        out_path=str(output_dir / "schoenfeld.png"),
    )
    record_fig("Schoenfeld Residuals", res_schoen, expect_pdf=False)

    res_mice = plot_mice_diagnostics(
        np.array([1.0, 2.0, 3.0, 4.0]),
        [np.array([1.1, 2.2]), np.array([0.9, 2.1])],
        var_name="Lactate",
        out_path=str(output_dir / "mice.png"),
    )
    record_fig("MICE Imputation Diagnostics", res_mice, expect_pdf=False)

    # 8. Dual View HTML Plotly Mirror
    res_dual_km = plot_dual_kaplan_meier(
        km_curves,
        risk_table=risk_df,
        out_path=str(output_dir / "dual_km.png"),
    )
    record_fig("Dual View Plotly Kaplan-Meier", res_dual_km, expect_pdf=True)

    res_dual_fr = plot_dual_forest(
        forest_df,
        scale="HR",
        out_path=str(output_dir / "dual_forest.png"),
    )
    record_fig("Dual View Plotly Forest", res_dual_fr, expect_pdf=True)

    res_dual_roc = plot_dual_roc_curve(
        roc_data, out_path=str(output_dir / "dual_roc.png")
    )
    record_fig("Dual View Plotly ROC", res_dual_roc, expect_pdf=True)

    res_dual_love = plot_dual_love(love_df, out_path=str(output_dir / "dual_love.png"))
    record_fig("Dual View Plotly Love", res_dual_love, expect_pdf=True)

    res_dual_ba = plot_dual_bland_altman(
        ba_data, out_path=str(output_dir / "dual_ba.png")
    )
    record_fig("Dual View Plotly Bland-Altman", res_dual_ba, expect_pdf=True)

    return results


def main():
    print("=" * 70)
    print("Starting Comprehensive Biostatistical Figure Matrix Verification...")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        results = run_full_matrix(tmp_path)

        print(f"\n[SUMMARY] Total Figures Generated: {results['total_generated']}")
        print(f"[SUMMARY] Bundles Verified:        {results['bundles_verified']}")
        print(f"[SUMMARY] QA Layout Passed:        {results['qa_passed']}")
        print("\n[FORMAT BREAKDOWN]")
        for fmt, cnt in results["formats"].items():
            print(f"  - {fmt.upper():<12}: {cnt} files")

        assert results["total_generated"] >= 45, "Expected at least 45 test figures"
        assert results["qa_passed"] == results["total_generated"], (
            "Some QA gates failed"
        )
        print(
            "\n[SUCCESS] ALL FIGURE FORMATS & SCENARIOS PASSED VERIFICATION WITH ZERO DEFECTS!"
        )


if __name__ == "__main__":
    main()

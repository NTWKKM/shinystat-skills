"""
tests/unit/test_all_figures_matrix.py: Comprehensive test matrix for all figure types,
scenarios, journal house styles (NEJM, JAMA, Lancet, BMJ, Clinical), canvas widths,
and multi-format export bundles (PNG, PDF fonttype=42, SVG, CSV, Dual View Plotly HTML).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from medstat.diagnostic.roc import compute_time_dependent_roc
from medstat.figures.agreement import plot_bland_altman
from medstat.figures.balance import plot_love
from medstat.figures.base import FigureResult
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
    DualFigureResult,
    plot_dual_bland_altman,
    plot_dual_forest,
    plot_dual_kaplan_meier,
    plot_dual_love,
    plot_dual_roc_curve,
)
from medstat.figures.illustration import Scene
from medstat.figures.qa import assert_zero_phi_text, cvd_check, qa_layout
from medstat.figures.roc import (
    plot_reclassification,
    plot_roc_curve,
    plot_time_dependent_roc,
)
from medstat.figures.survival import plot_cif, plot_kaplan_meier

JOURNALS = ["NEJM", "JAMA", "LANCET", "BMJ", "CLINICAL"]


def _assert_bundle_intact(
    res: FigureResult, expect_vector: bool = True, expect_svg: bool = False
):
    """Asserts that all expected files exist, are non-empty, and QA passed."""
    assert Path(res.png_path).exists(), f"PNG missing: {res.png_path}"
    assert Path(res.png_path).stat().st_size > 0, "PNG file is empty"

    assert Path(res.csv_path).exists(), f"CSV missing: {res.csv_path}"
    assert Path(res.csv_path).stat().st_size > 0, "CSV file is empty"
    assert res.source_df is not None
    assert len(res.source_df) > 0

    if expect_vector:
        assert res.vector_path is not None, "vector_path is None"
        assert Path(res.vector_path).exists(), f"PDF missing: {res.vector_path}"
        assert Path(res.vector_path).stat().st_size > 0, "PDF file is empty"

    if expect_svg:
        assert res.svg_path is not None, "svg_path is None"
        assert Path(res.svg_path).exists(), f"SVG missing: {res.svg_path}"
        assert Path(res.svg_path).stat().st_size > 0, "SVG file is empty"

    if res.qa is not None:
        assert res.qa["passed"] is True, f"QA layout failed: {res.qa}"


# ==============================================================================
# 1. Survival & Competing Risks
# ==============================================================================


def test_survival_matrix_all_journals_and_inset(tmp_path: Path):
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

    # Test across all 5 journals
    for jnl in JOURNALS:
        out_p = tmp_path / f"km_{jnl}.png"
        res = plot_kaplan_meier(
            km_curves,
            risk_table=risk_df,
            log_rank_p=0.0003,
            journal=jnl,
            target="single_col",
            out_path=str(out_p),
        )
        _assert_bundle_intact(res, expect_vector=True)
        assert (tmp_path / f"km_{jnl}_risktable.csv").exists()

    # Test NEJM low-event zoomed inset
    out_inset = tmp_path / "km_nejm_inset.png"
    res_inset = plot_kaplan_meier(
        km_curves,
        risk_table=risk_df,
        log_rank_p=0.0003,
        journal="NEJM",
        nejm_inset_ylim=(0.40, 1.0),
        target="double_col",
        out_path=str(out_inset),
    )
    _assert_bundle_intact(res_inset, expect_vector=True)


def test_competing_risks_cif_matrix(tmp_path: Path):
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
        out_p = tmp_path / f"cif_{jnl}.png"
        res = plot_cif(
            cif_df,
            journal=jnl,
            target="single_col",
            time_unit="months",
            out_path=str(out_p),
        )
        _assert_bundle_intact(res, expect_vector=True)


# ==============================================================================
# 2. Forest Plots
# ==============================================================================


def test_forest_plot_matrix_all_scales_and_journals(tmp_path: Path):
    forest_df = pd.DataFrame(
        {
            "term": [
                "Overall Cohort",
                "Age >= 65 years",
                "Age < 65 years",
                "Male Sex",
                "Female Sex",
                "Diabetes Mellitus",
                "Chronic Kidney Disease",
            ],
            "estimate": [1.48, 1.75, 1.22, 1.35, 1.62, 1.95, 2.10],
            "ci_lower": [1.18, 1.30, 0.91, 1.02, 1.21, 1.42, 1.50],
            "ci_upper": [1.86, 2.35, 1.64, 1.79, 2.17, 2.68, 2.94],
            "p_value": [0.001, 0.0002, 0.180, 0.035, 0.001, 0.0001, 0.0001],
            "events_text": [
                "210/850",
                "125/400",
                "85/450",
                "115/480",
                "95/370",
                "90/280",
                "75/190",
            ],
            "is_header": [False, False, False, False, False, False, False],
        }
    )

    # Ratio scale (HR) across journals
    for jnl in JOURNALS:
        out_p = tmp_path / f"forest_hr_{jnl}.png"
        res = plot_forest(
            forest_df,
            scale="HR",
            journal=jnl,
            target="double_col",
            out_path=str(out_p),
        )
        _assert_bundle_intact(res, expect_vector=True)

    # Difference scale (Beta / MD) on linear axis
    diff_df = forest_df.copy()
    diff_df["estimate"] = [0.45, 0.70, 0.20, 0.35, 0.55, 0.85, 1.10]
    diff_df["ci_lower"] = [0.15, 0.30, -0.10, 0.05, 0.20, 0.45, 0.65]
    diff_df["ci_upper"] = [0.75, 1.10, 0.50, 0.65, 0.90, 1.25, 1.55]

    res_diff = plot_forest(
        diff_df,
        scale="Beta",
        journal="JAMA",
        target="double_col",
        out_path=str(tmp_path / "forest_diff_jama.png"),
    )
    _assert_bundle_intact(res_diff, expect_vector=True)

    # Fine-Gray Subdistribution HR label test
    res_fg = plot_forest(
        forest_df,
        scale="Subdistribution HR",
        journal="LANCET",
        target="double_col",
        out_path=str(tmp_path / "forest_fg_lancet.png"),
    )
    _assert_bundle_intact(res_fg, expect_vector=True)


# ==============================================================================
# 3. Diagnostic & Prediction Validation
# ==============================================================================


def test_diagnostic_and_prediction_matrix(tmp_path: Path):
    # ROC Single and Paired
    roc_data = {
        "name": "Novel Biomarker Score",
        "fpr": np.array([0.0, 0.05, 0.15, 0.30, 0.55, 0.80, 1.0]),
        "tpr": np.array([0.0, 0.40, 0.68, 0.82, 0.92, 0.97, 1.0]),
        "auc": 0.845,
        "ci_lower": 0.790,
        "ci_upper": 0.900,
        "optimal_point": {"fpr": 0.15, "tpr": 0.68, "cutoff": 42.5},
        "sensitivity": 0.68,
        "specificity": 0.85,
    }
    paired_data = {
        "name": "Standard Clinical Model",
        "fpr": np.array([0.0, 0.10, 0.25, 0.45, 0.70, 1.0]),
        "tpr": np.array([0.0, 0.35, 0.58, 0.75, 0.88, 1.0]),
        "auc": 0.730,
        "ci_lower": 0.660,
        "ci_upper": 0.800,
    }

    for jnl in JOURNALS:
        out_roc = tmp_path / f"roc_{jnl}.png"
        res_roc = plot_roc_curve(
            roc_data,
            paired_roc_data=paired_data,
            paired_p_value=0.0018,
            journal=jnl,
            target="single_col",
            out_path=str(out_roc),
        )
        _assert_bundle_intact(res_roc, expect_vector=True)

    # Net Reclassification Improvement (NRI)
    np.random.seed(42)
    n = 200
    p_old = np.random.uniform(0.05, 0.85, n)
    p_new = np.clip(p_old + np.random.normal(0, 0.08, n), 0.01, 0.99)
    y_event = (np.random.uniform(0, 1, n) < p_new).astype(int)

    reclass_df = pd.DataFrame({"p_old": p_old, "p_new": p_new, "y_true": y_event})
    res_reclass = plot_reclassification(
        reclass_df,
        journal="NEJM",
        out_path=str(tmp_path / "reclass_nejm.png"),
    )
    _assert_bundle_intact(res_reclass, expect_vector=True)

    # Time-Dependent ROC
    dur = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0] * 5)
    ev = np.array([1, 0, 1, 0, 1, 0] * 5)
    sc_val = np.array([0.8, 0.2, 0.7, 0.3, 0.9, 0.1] * 5)
    td_roc_data = compute_time_dependent_roc(
        dur, ev, sc_val, time_horizons=[25.0, 45.0], n_bootstrap=0
    )
    res_tdroc = plot_time_dependent_roc(
        td_roc_data,
        journal="LANCET",
        out_path=str(tmp_path / "tdroc_lancet.png"),
    )
    _assert_bundle_intact(res_tdroc, expect_vector=True)

    # Decile Calibration
    calib_bins = {
        "pred_mean": np.linspace(0.05, 0.95, 10),
        "obs_rate": np.linspace(0.06, 0.91, 10),
        "ci_lower": np.linspace(0.02, 0.82, 10),
        "ci_upper": np.linspace(0.12, 0.98, 10),
        "counts": np.full(10, 60),
    }
    for jnl in JOURNALS:
        out_cal = tmp_path / f"cal_{jnl}.png"
        res_cal = plot_calibration(
            calib_bins,
            brier_score=0.118,
            ici=0.028,
            journal=jnl,
            out_path=str(out_cal),
        )
        _assert_bundle_intact(res_cal, expect_vector=True)

    # Recalibration Comparison
    # Recalibration Comparison
    recal_dict = {
        "y_true": np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0] * 5),
        "p_orig": np.linspace(0.1, 0.9, 50),
        "p_recal": np.clip(np.linspace(0.05, 0.95, 50), 0.01, 0.99),
    }
    res_recal = plot_recalibration_comparison(
        recal_dict,
        journal="BMJ",
        out_path=str(tmp_path / "recal_bmj.png"),
    )
    _assert_bundle_intact(res_recal, expect_vector=True)

    # Bootstrap Optimism
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
        opt_dict,
        journal="CLINICAL",
        out_path=str(tmp_path / "opt_clinical.png"),
    )
    _assert_bundle_intact(res_opt, expect_vector=True)

    # Decision Curve Analysis (DCA)
    dca_df = pd.DataFrame(
        {
            "strategy": ["Treat All"] * 6
            + ["Biomarker Model"] * 6
            + ["Treat None"] * 6,
            "threshold": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30] * 3,
            "net_benefit": [0.18, 0.14, 0.09, 0.05, 0.01, -0.03]
            + [0.20, 0.17, 0.13, 0.09, 0.06, 0.03]
            + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    for jnl in JOURNALS:
        out_dca = tmp_path / f"dca_{jnl}.png"
        res_dca = plot_dca(dca_df, journal=jnl, out_path=str(out_dca))
        _assert_bundle_intact(res_dca, expect_vector=True)


# ==============================================================================
# 4. Causal Balance & Agreement
# ==============================================================================


def test_causal_balance_and_agreement_matrix(tmp_path: Path):
    # Love Plot (Austin 2009)
    love_df = pd.DataFrame(
        {
            "Variable": [
                "Age (years)",
                "Male Sex (%)",
                "Baseline SBP (mmHg)",
                "Serum Creatinine (mg/dL)",
                "Diabetes History (%)",
                "Prior Myocardial Infarction (%)",
            ],
            "smd_unadjusted": [0.38, 0.25, 0.42, 0.31, 0.29, 0.22],
            "smd_matched": [0.04, 0.02, 0.05, 0.03, 0.01, 0.04],
            "smd_weighted": [0.03, 0.01, 0.04, 0.02, 0.02, 0.03],
        }
    )
    for jnl in JOURNALS:
        out_love = tmp_path / f"love_{jnl}.png"
        res_love = plot_love(
            love_df,
            journal=jnl,
            out_path=str(out_love),
        )
        _assert_bundle_intact(res_love, expect_vector=True)

    # Bland-Altman Agreement
    np.random.seed(42)
    m1 = np.random.normal(120, 15, 100)
    m2 = m1 + np.random.normal(2.5, 4.0, 100)
    ba_data = {
        "means": (m1 + m2) / 2.0,
        "diffs": m1 - m2,
        "mean_diff": float(np.mean(m1 - m2)),
        "loa_upper": float(np.mean(m1 - m2) + 1.96 * np.std(m1 - m2)),
        "loa_lower": float(np.mean(m1 - m2) - 1.96 * np.std(m1 - m2)),
        "ci_mean": (
            float(np.mean(m1 - m2) - 0.8),
            float(np.mean(m1 - m2) + 0.8),
        ),
        "ci_upper": (
            float(np.mean(m1 - m2) + 1.96 * np.std(m1 - m2) - 1.2),
            float(np.mean(m1 - m2) + 1.96 * np.std(m1 - m2) + 1.2),
        ),
        "ci_lower": (
            float(np.mean(m1 - m2) - 1.96 * np.std(m1 - m2) - 1.2),
            float(np.mean(m1 - m2) - 1.96 * np.std(m1 - m2) + 1.2),
        ),
    }
    for jnl in JOURNALS:
        out_ba = tmp_path / f"ba_{jnl}.png"
        res_ba = plot_bland_altman(
            ba_data,
            journal=jnl,
            units="mmHg",
            out_path=str(out_ba),
        )
        _assert_bundle_intact(res_ba, expect_vector=True)


# ==============================================================================
# 5. Programmatic Study Flow Diagrams
# ==============================================================================


def test_study_flow_diagrams_matrix(tmp_path: Path):
    # 5.1 CONSORT 2025
    arms_2025 = [
        {
            "name": "Intervention Arm",
            "allocated": 150,
            "received": 145,
            "not_received": {"Declined": 5},
            "lost": {"Relocated": 4},
            "discontinued": {"Adverse event": 3},
            "analysed": 138,
            "excluded_analysis": {"Missing primary endpoint": 12},
        },
        {
            "name": "Standard Care Arm",
            "allocated": 150,
            "received": 148,
            "not_received": {"Consent withdrawn": 2},
            "lost": {"Uncontactable": 6},
            "discontinued": {},
            "analysed": 142,
            "excluded_analysis": {"Protocol violation": 8},
        },
    ]
    out_consort_2025 = tmp_path / "consort_2025.png"
    res_c2025 = consort_rct(
        assessed=380,
        excluded={"Did not meet inclusion criteria": 55, "Declined participation": 25},
        arms=arms_2025,
        version="2025",
        out_path=str(out_consort_2025),
    )
    _assert_bundle_intact(res_c2025, expect_vector=True, expect_svg=True)

    # 5.2 CONSORT 2010
    arms_2010 = [
        {
            "name": "Arm A",
            "allocated": 100,
            "received": 100,
            "lost": {"Lost to follow-up": 5},
            "discontinued": {},
            "analysed": 95,
            "excluded_analysis": {"Missing endpoint": 5},
        },
        {
            "name": "Arm B",
            "allocated": 100,
            "received": 98,
            "not_received": {"Refused": 2},
            "lost": {"Lost to follow-up": 4},
            "discontinued": {},
            "analysed": 94,
            "excluded_analysis": {"Protocol deviation": 6},
        },
    ]
    out_consort_2010 = tmp_path / "consort_2010.png"
    res_c2010 = consort_rct(
        assessed=250,
        excluded={"Ineligible": 50},
        arms=arms_2010,
        version="2010",
        out_path=str(out_consort_2010),
    )
    _assert_bundle_intact(res_c2010, expect_vector=True, expect_svg=True)

    # 5.3 PRISMA 2020 (Single and Dual Stream)
    other_sources = {
        "sources": {"Websites": 40, "Citation searching": 25},
        "not_retrieved": 5,
        "fulltext_excluded": {"Ineligible design": 20},
    }
    out_prisma = tmp_path / "prisma_dual.png"
    res_prisma = prisma_2020(
        databases={"PubMed": 600, "Embase": 450, "Cochrane": 120},
        registers={"ClinicalTrials.gov": 65},
        removed_duplicates=235,
        removed_automation=40,
        removed_other=0,
        screened_excluded=650,
        not_retrieved=30,
        fulltext_excluded={"Wrong population": 140, "No relevant outcome": 80},
        included_studies=90,
        included_reports=100,
        other=other_sources,
        out_path=str(out_prisma),
    )
    _assert_bundle_intact(res_prisma, expect_vector=True, expect_svg=True)

    # 5.4 STARD 2015 Diagnostic Flow
    out_stard = tmp_path / "stard.png"
    res_stard = stard_flow(
        eligible=450,
        excluded={"Refused": 25, "Incomplete index test": 15},
        index_pos=160,
        index_neg=240,
        index_inconclusive=10,
        tp=140,
        fp=20,
        fn=15,
        tn=225,
        out_path=str(out_stard),
    )
    _assert_bundle_intact(res_stard, expect_vector=True, expect_svg=True)

    # 5.5 STROBE Cohort Flow
    out_cohort = tmp_path / "cohort.png"
    res_cohort = cohort_flow(
        source=1200,
        steps=[
            ("Age < 18", 150),
            ("Prior admission in 30d", 50),
            ("Missing core biomarker", 80),
        ],
        groups={"Surviving": 850, "Deceased": 70},
        source_label="Hospital Inpatient Admissions",
        out_path=str(out_cohort),
    )
    _assert_bundle_intact(res_cohort, expect_vector=True, expect_svg=True)

    # 5.6 Methods Pipeline
    steps = [
        {
            "title": "Data Extraction",
            "body": "Electronic Medical Records 2021-2024",
            "counts": {"n_in": 1500, "excluded": {"Non-consent": 100}, "n_out": 1400},
        },
        {
            "title": "Feature Engineering",
            "body": "Extraction of clinical biomarkers and vital signs",
            "counts": {
                "n_in": 1400,
                "excluded": {"Artifactual readings": 50},
                "n_out": 1350,
            },
        },
    ]
    out_methods = tmp_path / "methods.png"
    res_meth = methods_pipeline(steps, out_path=str(out_methods), strict_numbers=True)
    _assert_bundle_intact(res_meth, expect_vector=True, expect_svg=True)


# ==============================================================================
# 6. Scientific Schematic Illustrations
# ==============================================================================


def test_scientific_schematic_scene_matrix(tmp_path: Path):
    sc = Scene(width="double_col", aspect=0.55, journal="NEJM", seed=42)

    # Primitives: Cells
    sc.rbc(10, 15)
    sc.platelet(18, 15)
    sc.neutrophil(26, 15)
    sc.macrophage(36, 15, engulfed=2)
    sc.fibroblast(48, 15)
    sc.bacterium(58, 15, kind="bacillus")
    sc.bacterium(64, 15, kind="coccus")

    # Primitives: Skin Layers and Excavated Crater
    geo = sc.skin(70, 98, 25, thick=(2.5, 6.0, 5.0, 4.0))
    cr = sc.crater(geo, 84, 8, 5)
    sc.wound_phase(geo, cr, 84, 8, 5, "proliferation")

    # Arterial Stenosis Cross Section
    sc.artery_cross_section(cx=25, cy=40, R=8.0, stenosis=0.70)

    out_sci = tmp_path / "sciart_matrix.png"
    res_sci = sc.save(
        out_path=str(out_sci),
        caption_title="Pathophysiological Microenvironment Overview",
        alt_text="Comprehensive scientific schematic of cellular and vascular response",
        dpi=300,
    )
    _assert_bundle_intact(res_sci, expect_vector=True, expect_svg=True)
    assert "Schematic illustration; not to scale" in res_sci.caption


# ==============================================================================
# 7. Diagnostics & Missingness
# ==============================================================================


def test_model_diagnostics_and_missingness(tmp_path: Path):
    # Missingness Map
    df_missing = pd.DataFrame(
        {
            "Age": [50, 60, np.nan, 70, 80],
            "Sex": [1, 0, 1, 0, 1],
            "Troponin": [np.nan, 12.5, 8.4, np.nan, 25.1],
            "Creatinine": [1.1, 1.4, 0.9, np.nan, 2.2],
        }
    )
    res_miss = plot_missingness_map(
        df_missing,
        out_path=str(tmp_path / "missingness.png"),
    )
    _assert_bundle_intact(res_miss, expect_vector=False)

    # Schoenfeld Residuals
    times = np.array([5, 12, 18, 25, 34, 45, 60, 78, 90, 120])
    residuals = np.random.normal(0, 0.5, 10)
    res_schoen = plot_schoenfeld_residuals(
        times,
        residuals,
        var_name="Systolic BP",
        p_value=0.45,
        out_path=str(tmp_path / "schoenfeld.png"),
    )
    _assert_bundle_intact(res_schoen, expect_vector=False)

    # MICE Diagnostics
    observed = np.random.normal(10, 2, 80)
    imputed = [np.random.normal(10.2, 2.1, 20) for _ in range(5)]
    res_mice = plot_mice_diagnostics(
        observed,
        imputed,
        var_name="Serum Lactate (mmol/L)",
        out_path=str(tmp_path / "mice.png"),
    )
    _assert_bundle_intact(res_mice, expect_vector=False)


# ==============================================================================
# 8. Dual / Toggle HTML View Mirror (Plotly Interactive)
# ==============================================================================


def test_dual_view_plotly_interactive_matrix(tmp_path: Path):
    # 8.1 Dual Kaplan-Meier
    km_curves = {
        "Active": {
            "timeline": np.array([0, 30, 60, 90]),
            "survival": np.array([1.0, 0.9, 0.8, 0.75]),
            "ci_lower": np.array([1.0, 0.82, 0.71, 0.65]),
            "ci_upper": np.array([1.0, 0.96, 0.88, 0.83]),
        },
        "Placebo": {
            "timeline": np.array([0, 30, 60, 90]),
            "survival": np.array([1.0, 0.8, 0.65, 0.55]),
            "ci_lower": np.array([1.0, 0.7, 0.53, 0.42]),
            "ci_upper": np.array([1.0, 0.88, 0.75, 0.66]),
        },
    }
    risk_df = pd.DataFrame(
        {"Group": ["Active", "Placebo"], "t=0": [100, 100], "t=90": [75, 55]}
    )
    res_dual_km = plot_dual_kaplan_meier(
        km_curves,
        risk_table=risk_df,
        out_path=str(tmp_path / "dual_km.png"),
    )
    assert isinstance(res_dual_km, DualFigureResult)
    assert res_dual_km.plotly_json is not None
    assert "data" in json.loads(res_dual_km.plotly_json)

    # 8.2 Dual Forest
    forest_df = pd.DataFrame(
        {
            "term": ["Overall", "Subgroup A"],
            "estimate": [1.4, 1.6],
            "ci_lower": [1.1, 1.2],
            "ci_upper": [1.8, 2.1],
            "p_value": [0.01, 0.005],
        }
    )
    res_dual_fr = plot_dual_forest(
        forest_df,
        scale="HR",
        out_path=str(tmp_path / "dual_forest.png"),
    )
    assert isinstance(res_dual_fr, DualFigureResult)

    # 8.3 Dual ROC
    roc_data = {
        "name": "Model 1",
        "fpr": np.array([0.0, 0.2, 0.5, 1.0]),
        "tpr": np.array([0.0, 0.7, 0.85, 1.0]),
        "auc": 0.80,
        "ci_lower": 0.72,
        "ci_upper": 0.88,
    }
    res_dual_roc = plot_dual_roc_curve(
        roc_data,
        out_path=str(tmp_path / "dual_roc.png"),
    )
    assert isinstance(res_dual_roc, DualFigureResult)

    # 8.4 Dual Love
    love_df = pd.DataFrame(
        {
            "Variable": ["Age", "SBP"],
            "smd_unadjusted": [0.35, 0.28],
            "smd_matched": [0.04, 0.02],
        }
    )
    res_dual_love = plot_dual_love(
        love_df,
        out_path=str(tmp_path / "dual_love.png"),
    )
    assert isinstance(res_dual_love, DualFigureResult)

    # 8.5 Dual Bland-Altman
    ba_data = {
        "means": [100.0, 110.0, 120.0],
        "diffs": [2.0, -1.0, 1.5],
        "mean_diff": 0.83,
        "loa_upper": 4.0,
        "loa_lower": -2.3,
    }
    res_dual_ba = plot_dual_bland_altman(
        ba_data,
        out_path=str(tmp_path / "dual_ba.png"),
    )
    assert isinstance(res_dual_ba, DualFigureResult)


# ==============================================================================
# 9. QA Layout Collision & CVD & PHI Gates
# ==============================================================================


def test_qa_gates_comprehensive():
    # 9.1 Layout Collision Gate
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.text(0.5, 0.5, "Label A", fontsize=12)
    ax.text(0.5, 0.5, "Label B", fontsize=12)
    qa_overlap = qa_layout(fig)
    assert qa_overlap["passed"] is False
    assert len(qa_overlap["text_overlaps"]) > 0
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [0, 1])
    ax.set_title("Clean Title")
    fig.tight_layout()
    qa_clean = qa_layout(fig)
    assert qa_clean["passed"] is True
    plt.close(fig)

    # 9.2 CVD Contrast Check
    palette = {"c1": "#BC3C29", "c2": "#0072B5", "c3": "#E18727"}
    cvd_df = cvd_check(pairs=[("c1", "c2"), ("c1", "c3")], palette=palette, min_de=12.0)
    assert isinstance(cvd_df, pd.DataFrame)
    assert set(cvd_df["vision"].unique()) == {
        "normal",
        "deuteranopia",
        "protanopia",
        "tritanopia",
    }

    # 9.3 Zero-PHI Gate
    assert_zero_phi_text(["Kaplan-Meier Curve", "Hazard Ratio (95% CI)", "Risk Table"])
    with pytest.raises(ValueError, match="Zero-PHI Violation"):
        assert_zero_phi_text(["Patient HN 12345678"])
    with pytest.raises(ValueError, match="Zero-PHI Violation"):
        assert_zero_phi_text(["CID: 1234567890123"])
    with pytest.raises(ValueError, match="Zero-PHI Violation"):
        assert_zero_phi_text(["Dr. Smith"])
    with pytest.raises(ValueError, match="Zero-PHI Violation"):
        assert_zero_phi_text(["นายสมชาย"])

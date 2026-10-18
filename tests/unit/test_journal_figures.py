"""
tests/unit/test_journal_figures.py: Verification of journal styling, formatters,
QA layout gate, Zero-PHI scan, and FigureResult contract extensions.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from medstat.figures.base import FigureResult
from medstat.figures.journal import (
    JOURNAL_SPECS,
    format_estimate,
    format_p_value,
)
from medstat.figures.qa import assert_zero_phi_text, cvd_check, qa_layout
from medstat.figures.styles import save_figure_bundle


def test_journal_specs_contain_all_five_styles():
    for j in ["NEJM", "JAMA", "LANCET", "BMJ", "CLINICAL"]:
        assert j in JOURNAL_SPECS
        assert "palette" in JOURNAL_SPECS[j]
        assert len(JOURNAL_SPECS[j]["palette"]) >= 6
        assert "bg_manuscript" in JOURNAL_SPECS[j]


def test_format_p_value_rules():
    # NEJM: standard P=0.03, P<0.001
    assert format_p_value(0.031, "NEJM") == "P=0.03"
    assert format_p_value(0.0004, "NEJM") == "P<0.001"
    assert format_p_value(0.031, "NEJM", bare=True) == "0.03"

    # JAMA: P = .03, P < .001, P > .99 (no leading zero)
    assert format_p_value(0.031, "JAMA") == "P = .03"
    assert format_p_value(0.0004, "JAMA") == "P < .001"
    assert format_p_value(0.995, "JAMA") == "P > .99"
    assert format_p_value(0.031, "JAMA", bare=True) == ".03"

    # LANCET: mid-dot, lowercase p, 4 dp for 0.0001-0.001
    assert format_p_value(0.014, "LANCET") == "p=0·01"
    assert format_p_value(0.0004, "LANCET") == "p=0·0004"
    assert format_p_value(0.00005, "LANCET") == "p<0·0001"
    assert format_p_value(0.0004, "LANCET", bare=True) == "0·0004"

    # BMJ: P = 0.03, P < 0.001 (keeps leading zero, spaced)
    assert format_p_value(0.031, "BMJ") == "P = 0.03"
    assert format_p_value(0.0004, "BMJ") == "P < 0.001"
    assert format_p_value(0.031, "BMJ", bare=True) == "0.03"

    # CLINICAL: P=0.03, P<0.001
    assert format_p_value(0.031, "CLINICAL") == "P=0.03"

    # Boundaries 0.0496 (just below .05 -> 3 dp) and 0.996 (> .99)
    assert format_p_value(0.0496, "NEJM") == "P=0.050"
    assert format_p_value(0.996, "NEJM") == "P>0.99"
    assert format_p_value(0.0496, "JAMA") == "P = .050"
    assert format_p_value(0.996, "JAMA") == "P > .99"
    assert format_p_value(0.0496, "BMJ") == "P = 0.050"
    assert format_p_value(0.996, "BMJ") == "P > 0.99"
    assert format_p_value(0.0496, "LANCET") == "p=0·050"
    assert format_p_value(0.996, "LANCET") == "p>0·99"


def test_format_estimate_rules():
    # HR 1.45 (1.12 to 1.88)
    nejm_est = format_estimate(1.45, 1.12, 1.88, "HR", journal="NEJM")
    assert "HR, 1.45" in nejm_est
    assert "95% CI" in nejm_est

    jama_est = format_estimate(1.45, 1.12, 1.88, "HR", journal="JAMA")
    assert "HR, 1.45" in jama_est

    lancet_est = format_estimate(1.45, 1.12, 1.88, "HR", journal="LANCET")
    assert "1·45" in lancet_est
    assert "1·12" in lancet_est

    bmj_est = format_estimate(1.45, 1.12, 1.88, "HR", journal="BMJ")
    assert "HR 1.45 (95% CI 1.12 to 1.88)" in bmj_est


def test_assert_zero_phi_text_raises_on_violations():
    # Valid clinical terms & Thai false-positive safety checks
    assert_zero_phi_text(["Survival Probability", "Cohort (n = 120)", "Stage 2"])
    assert_zero_phi_text(["นายกรัฐมนตรีแถลงนโยบาย"])
    assert_zero_phi_text(["นางพยาบาลประจำวอร์ด"])

    # Violations
    with pytest.raises(ValueError, match="Zero-PHI Violation"):
        assert_zero_phi_text(["Patient HN 123456"])

    with pytest.raises(ValueError, match="Zero-PHI Violation"):
        assert_zero_phi_text(["ID: 1100500123456"])

    with pytest.raises(ValueError, match="Zero-PHI Violation"):
        assert_zero_phi_text(["Dr. Somchai report"])

    with pytest.raises(ValueError, match="Zero-PHI Violation"):
        assert_zero_phi_text(["นพ. สมชาย"])

    with pytest.raises(ValueError, match="Zero-PHI Violation"):
        assert_zero_phi_text(["นายสมชาย"])

    with pytest.raises(ValueError, match="Zero-PHI Violation"):
        assert_zero_phi_text(["นายก้อง"])

    with pytest.raises(ValueError, match="Zero-PHI Violation"):
        assert_zero_phi_text(["ผู้ป่วยนางสาวสมศรี"])


def test_qa_layout_gate():
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1])
    ax.set_title("Clean Title")
    fig.tight_layout()
    qa = qa_layout(fig)
    assert qa["passed"] is True, f"QA failed: {qa}"
    plt.close(fig)

    # Intentionally collide two texts at the exact same coordinates
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.text(0.5, 0.5, "Long Text Colliding A", fontsize=12)
    ax.text(0.5, 0.5, "Long Text Colliding B", fontsize=12)
    qa_col = qa_layout(fig)
    assert qa_col["passed"] is False
    assert len(qa_col["text_overlaps"]) > 0
    plt.close(fig)

    # Collision with left-aligned title
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_title("Left-Aligned Main Title", loc="left")
    ax.text(0.0, 1.0, "Overlapping Label Over Title", transform=ax.transAxes)
    fig.canvas.draw()
    qa_title_col = qa_layout(fig)
    assert qa_title_col["passed"] is False
    assert len(qa_title_col["text_overlaps"]) > 0
    plt.close(fig)


def test_cvd_check_computation():
    palette = {
        "c1": "#BC3C29",
        "c2": "#0072B5",
        "c3": "#E18727",
    }
    df = cvd_check(pairs=[("c1", "c2"), ("c1", "c3")], palette=palette, min_de=12.0)
    assert isinstance(df, pd.DataFrame)
    assert "deltaE" in df.columns
    assert "needs_cue" in df.columns
    assert len(df) == 2 * 4  # 2 pairs * 4 vision modes (normal, deutan, protan, tritan)


def test_figure_result_contract_extensions(tmp_path):
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    res = FigureResult(
        png_path=str(tmp_path / "plot.png"),
        alt_text="Test",
        caption="Caption",
        source_df=df,
        csv_path=str(tmp_path / "plot.csv"),
        vector_path=str(tmp_path / "plot.pdf"),
        svg_path=str(tmp_path / "plot.svg"),
        qa={"passed": True},
    )
    d = res.to_dict()
    assert d["vector_path"] == str(tmp_path / "plot.pdf")
    assert d["svg_path"] == str(tmp_path / "plot.svg")
    assert d["qa"] == {"passed": True}


def test_save_figure_bundle_generates_pdf_truetype(tmp_path):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [1, 0])
    png_p = tmp_path / "test_bundle.png"
    p_png, p_pdf, p_svg = save_figure_bundle(
        fig, png_p, dpi=300, export_vector=True, export_svg=True
    )
    assert p_png.endswith(".png")
    assert p_pdf is not None and p_pdf.endswith(".pdf")
    assert p_svg is not None and p_svg.endswith(".svg")
    assert (tmp_path / "test_bundle.png").exists()
    assert (tmp_path / "test_bundle.pdf").exists()
    assert (tmp_path / "test_bundle.svg").exists()


def test_consort_2025_flow_reconciled_and_unreconciled(tmp_path):
    from medstat.figures.flow import consort_rct

    # Valid CONSORT 2025
    arms = [
        {
            "name": "Intervention",
            "allocated": 100,
            "received": 95,
            "not_received": {"Declined": 5},
            "lost": {"Moved": 3},
            "discontinued": {"Adverse event": 2},
            "analysed": 90,
            "excluded_analysis": {"Missing primary outcome": 10},
        },
        {
            "name": "Control",
            "allocated": 100,
            "received": 98,
            "not_received": {"Withdrew": 2},
            "lost": {"Unreachable": 4},
            "discontinued": {},
            "analysed": 92,
            "excluded_analysis": {"Protocol deviation": 8},
        },
    ]
    out_p = tmp_path / "consort.png"
    res = consort_rct(
        assessed=250,
        excluded={"Ineligible": 30, "Declined": 20},
        arms=arms,
        out_path=str(out_p),
        version="2025",
    )
    assert res.qa["passed"] is True
    assert out_p.exists()
    assert (tmp_path / "consort.pdf").exists()
    assert (tmp_path / "consort.svg").exists()
    assert (tmp_path / "consort_source.csv").exists()

    # Unreconciled error: assessed (250) != excluded (50) + randomized (200), if excluded changed to 40
    with pytest.raises(ValueError, match="Unreconciled counts"):
        consort_rct(
            assessed=250,
            excluded={"Ineligible": 25, "Declined": 15},  # sum 40, rand = 210 != 200
            arms=arms,
            out_path=str(tmp_path / "bad_consort.png"),
        )


def test_prisma_2020_flow_single_and_dual_stream(tmp_path):
    from medstat.figures.flow import prisma_2020

    # Single stream
    res_single = prisma_2020(
        databases={"PubMed": 500, "Embase": 300},
        registers={"ClinicalTrials.gov": 50},
        removed_duplicates=150,
        removed_automation=50,
        removed_other=0,
        screened_excluded=400,
        not_retrieved=20,
        fulltext_excluded={"Wrong population": 100, "No outcome": 50},
        included_studies=70,
        included_reports=80,
        out_path=str(tmp_path / "prisma_single.png"),
    )
    assert res_single.qa["passed"] is True
    assert (tmp_path / "prisma_single.png").exists()

    # Dual stream with 'other methods'
    other = {
        "sources": {"Websites": 30, "Citation searching": 20},
        "not_retrieved": 5,
        "fulltext_excluded": {"Ineligible design": 15},
    }  # other included reports = 50 - 5 - 15 = 30; total included reports = 80 + 30 = 110
    res_dual = prisma_2020(
        databases={"PubMed": 500, "Embase": 300},
        registers={"ClinicalTrials.gov": 50},
        removed_duplicates=150,
        removed_automation=50,
        removed_other=0,
        screened_excluded=400,
        not_retrieved=20,
        fulltext_excluded={"Wrong population": 100, "No outcome": 50},
        included_studies=95,
        included_reports=110,
        other=other,
        out_path=str(tmp_path / "prisma_dual.png"),
    )
    assert res_dual.qa["passed"] is True
    assert (tmp_path / "prisma_dual.png").exists()


def test_cohort_flow_from_tracker(tmp_path):
    from medstat.data.retention import SampleFlowTracker
    from medstat.figures.flow import cohort_flow_from_tracker

    tracker = SampleFlowTracker(
        initial_n=1000, initial_name="All Hospitalized Patients"
    )
    tracker.record_stage("Age >= 18", 900, 100, reason="Pediatric age (<18)")
    tracker.record_stage("Complete Biomarkers", 820, 80, reason="Missing Troponin-T")

    res = cohort_flow_from_tracker(
        tracker,
        groups={"Surviving": 750, "In-Hospital Death": 70},
        out_path=str(tmp_path / "cohort_tracker.png"),
    )
    assert res.qa["passed"] is True
    assert (tmp_path / "cohort_tracker.png").exists()
    assert (tmp_path / "cohort_tracker_source.csv").exists()


def test_stard_and_methods_pipeline(tmp_path):
    from medstat.figures.flow import methods_pipeline, stard_flow

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
        out_path=str(tmp_path / "stard.png"),
    )
    assert res_stard.qa["passed"] is True
    assert (tmp_path / "stard.png").exists()

    # Methods pipeline
    steps = [
        {
            "title": "Data Extraction",
            "body": "Electronic health records from 2022 to 2024",
            "side": "Source: Tertiary Center HIS",
            "cite": "STROBE statement",
            "counts": {"n_in": 1000, "excluded": {"Missing consent": 50}, "n_out": 950},
        },
        {
            "title": "Laboratory Quality Filter",
            "body": "Valid admission creatinine",
            "counts": {"n_in": 950, "excluded": {"Hemolyzed sample": 30}, "n_out": 920},
        },
    ]
    res_meth = methods_pipeline(
        steps,
        out_path=str(tmp_path / "methods.png"),
        strict_numbers=True,
    )
    assert res_meth.qa["passed"] is True
    assert (tmp_path / "methods.png").exists()


def test_scientific_illustration_scene_primitives(tmp_path):
    from medstat.figures.illustration import Scene

    sc = Scene(width="double_col", aspect=0.5, journal="NEJM", seed=42)
    # Cells
    sc.rbc(10, 15)
    sc.platelet(18, 15)
    sc.neutrophil(25, 15)
    sc.macrophage(35, 15, engulfed=1)
    sc.fibroblast(48, 15)
    sc.bacterium(58, 15, kind="bacillus")

    # Tissue & wound
    geo = sc.skin(65, 95, 22, thick=(2.0, 5.0, 4.0, 4.0))
    cr = sc.crater(geo, 80, 8, 5)
    sc.wound_phase(geo, cr, 80, 8, 5, "inflammation")

    res = sc.save(
        out_path=str(tmp_path / "sciart_test.png"),
        caption_title="Cellular Mechanism Overview",
        alt_text="Schematic of inflammatory response",
        dpi=300,
    )
    assert res.qa["passed"] is True
    assert (tmp_path / "sciart_test.png").exists()
    assert (tmp_path / "sciart_test.pdf").exists()
    assert (tmp_path / "sciart_test.svg").exists()
    assert (tmp_path / "sciart_test_source.csv").exists()
    assert "Schematic illustration; not to scale" in res.caption


def test_statistical_plots_journal_styles_and_backward_compatibility(tmp_path):
    from medstat.figures.calibration import plot_calibration
    from medstat.figures.dca import plot_dca
    from medstat.figures.forest import plot_forest
    from medstat.figures.roc import plot_roc_curve
    from medstat.figures.survival import plot_cif, plot_kaplan_meier

    # 1. Calibration
    calib_bins = {
        "pred_mean": np.linspace(0.05, 0.95, 10),
        "obs_rate": np.linspace(0.04, 0.92, 10),
        "ci_lower": np.linspace(0.01, 0.85, 10),
        "ci_upper": np.linspace(0.08, 0.99, 10),
        "counts": np.full(10, 50),
    }
    for jnl in ["NEJM", "JAMA", "LANCET", "BMJ"]:
        res_cal = plot_calibration(
            calib_bins,
            brier_score=0.123,
            ici=0.035,
            journal=jnl,
            out_path=str(tmp_path / f"cal_{jnl}.png"),
        )
        assert res_cal.qa["passed"] is True
        assert res_cal.vector_path is not None and Path(res_cal.vector_path).exists()

    # 2. ROC
    roc_data = {
        "name": "Biomarker Model",
        "fpr": np.array([0.0, 0.1, 0.2, 0.4, 0.7, 1.0]),
        "tpr": np.array([0.0, 0.5, 0.7, 0.85, 0.95, 1.0]),
        "auc": 0.825,
        "ci_lower": 0.750,
        "ci_upper": 0.900,
        "optimal_point": {"fpr": 0.2, "tpr": 0.7, "cutoff": 0.35},
        "sensitivity": 0.70,
        "specificity": 0.80,
    }
    paired_data = {
        "name": "Clinical Model",
        "fpr": np.array([0.0, 0.2, 0.5, 0.8, 1.0]),
        "tpr": np.array([0.0, 0.4, 0.65, 0.85, 1.0]),
        "auc": 0.710,
        "ci_lower": 0.620,
        "ci_upper": 0.800,
    }
    for jnl in ["NEJM", "JAMA", "LANCET", "BMJ"]:
        res_roc = plot_roc_curve(
            roc_data,
            paired_roc_data=paired_data,
            paired_p_value=0.0042,
            journal=jnl,
            out_path=str(tmp_path / f"roc_{jnl}.png"),
        )
        assert res_roc.qa["passed"] is True
        assert res_roc.vector_path is not None and Path(res_roc.vector_path).exists()

    # 3. Kaplan-Meier & Inset break
    km_curves = {
        "Treatment A": {
            "timeline": np.array([0, 10, 20, 30, 60, 90]),
            "survival": np.array([1.0, 0.95, 0.88, 0.82, 0.75, 0.70]),
            "ci_lower": np.array([1.0, 0.90, 0.80, 0.74, 0.66, 0.60]),
            "ci_upper": np.array([1.0, 0.98, 0.94, 0.89, 0.83, 0.79]),
            "censored_times": np.array([15, 45]),
            "censored_survival": np.array([0.91, 0.78]),
        },
        "Placebo": {
            "timeline": np.array([0, 10, 20, 30, 60, 90]),
            "survival": np.array([1.0, 0.90, 0.78, 0.68, 0.55, 0.48]),
            "ci_lower": np.array([1.0, 0.82, 0.69, 0.58, 0.45, 0.38]),
            "ci_upper": np.array([1.0, 0.95, 0.85, 0.76, 0.64, 0.57]),
            "censored_times": np.array([25, 55]),
            "censored_survival": np.array([0.72, 0.51]),
        },
    }
    risk_df = pd.DataFrame(
        {
            "Group": ["Treatment A", "Placebo"],
            "t=0": [100, 100],
            "t=30": [82, 68],
            "t=60": [75, 55],
            "t=90": [70, 48],
        }
    )
    res_km = plot_kaplan_meier(
        km_curves,
        risk_table=risk_df,
        log_rank_p=0.002,
        journal="NEJM",
        nejm_inset_ylim=(0.40, 1.0),
        out_path=str(tmp_path / "km_nejm.png"),
    )
    assert res_km.qa["passed"] is True
    assert (tmp_path / "km_nejm.pdf").exists()
    assert (tmp_path / "km_nejm_risktable.csv").exists()

    # 4. CIF
    cif_df = pd.DataFrame(
        {
            "time": np.array([0, 10, 20, 30, 60]),
            "cif_primary": np.array([0.0, 0.05, 0.12, 0.18, 0.25]),
            "cif_primary_ci_lower": np.array([0.0, 0.02, 0.07, 0.12, 0.18]),
            "cif_primary_ci_upper": np.array([0.0, 0.09, 0.18, 0.25, 0.33]),
            "cif_competing": np.array([0.0, 0.02, 0.04, 0.06, 0.09]),
            "cif_competing_ci_lower": np.array([0.0, 0.01, 0.02, 0.03, 0.05]),
            "cif_competing_ci_upper": np.array([0.0, 0.04, 0.07, 0.10, 0.14]),
            "naive_1_minus_km": np.array([0.0, 0.06, 0.14, 0.22, 0.31]),
        }
    )
    res_cif = plot_cif(
        cif_df, journal="LANCET", out_path=str(tmp_path / "cif_lancet.png")
    )
    assert res_cif.qa["passed"] is True
    assert (tmp_path / "cif_lancet.pdf").exists()

    # 5. Forest 3-zone
    forest_df = pd.DataFrame(
        {
            "term": ["Overall", "Age >= 65", "Male Sex", "Diabetes", "Hypertension"],
            "estimate": [1.45, 1.62, 1.15, 1.88, 1.30],
            "ci_lower": [1.12, 1.20, 0.88, 1.35, 0.98],
            "ci_upper": [1.88, 2.18, 1.50, 2.62, 1.72],
            "p_value": [0.004, 0.001, 0.312, 0.0002, 0.068],
            "events_text": ["120/500", "65/220", "55/260", "48/150", "72/310"],
        }
    )
    for jnl in ["NEJM", "JAMA", "LANCET", "BMJ"]:
        res_fr = plot_forest(
            forest_df,
            scale="HR",
            journal=jnl,
            out_path=str(tmp_path / f"forest_{jnl}.png"),
        )
        assert res_fr.qa["passed"] is True
        assert res_fr.vector_path is not None and Path(res_fr.vector_path).exists()

    # 6. DCA
    dca_df = pd.DataFrame(
        {
            "strategy": ["Treat All"] * 5 + ["Model"] * 5 + ["Treat None"] * 5,
            "threshold": [0.1, 0.2, 0.3, 0.4, 0.5] * 3,
            "net_benefit": [0.12, 0.08, 0.04, 0.01, -0.02]
            + [0.14, 0.11, 0.08, 0.05, 0.02]
            + [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    res_dca = plot_dca(dca_df, journal="BMJ", out_path=str(tmp_path / "dca_bmj.png"))
    assert res_dca.qa["passed"] is True
    assert (tmp_path / "dca_bmj.pdf").exists()

    # 7. Backward compatibility check (journal=None)
    res_cal_none = plot_calibration(calib_bins, out_path=str(tmp_path / "cal_none.png"))
    assert res_cal_none.png_path.endswith(".png")
    res_roc_none = plot_roc_curve(roc_data, out_path=str(tmp_path / "roc_none.png"))
    assert res_roc_none.png_path.endswith(".png")
    res_km_none = plot_kaplan_meier(
        km_curves, risk_table=risk_df, out_path=str(tmp_path / "km_none.png")
    )
    assert res_km_none.png_path.endswith(".png")
    res_fr_none = plot_forest(forest_df, out_path=str(tmp_path / "fr_none.png"))
    assert res_fr_none.png_path.endswith(".png")
    res_dca_none = plot_dca(dca_df, out_path=str(tmp_path / "dca_none.png"))
    assert res_dca_none.png_path.endswith(".png")


def test_skill_reference_ssot_and_palette_parity():
    from medstat.figures.illustration import SEM
    from medstat.figures.journal import TARGET_WIDTHS

    repo_root = Path(__file__).resolve().parents[2]
    skill_main = (repo_root / "skills/shinystat/SKILL.md").read_text(encoding="utf-8")
    skill_cloud = (repo_root / "skills/shinystat-cloud/SKILL.md").read_text(
        encoding="utf-8"
    )
    hs_main = (repo_root / "skills/shinystat/references/house-styles.md").read_text(
        encoding="utf-8"
    )
    hs_cloud = (
        repo_root / "skills/shinystat-cloud/references/house-styles.md"
    ).read_text(encoding="utf-8")
    il_cloud = (
        repo_root / "skills/shinystat-cloud/references/illustration.md"
    ).read_text(encoding="utf-8")

    # 1. SKILL.md files must not inline hex palettes (disclosed in house-styles.md)
    for stale_or_inlined in ("#941100", "#1A365D", "#BC3C29", "#002D62", "#00468B"):
        assert stale_or_inlined not in skill_main
        assert stale_or_inlined not in skill_cloud

    # 2. Both house-styles.md files must match JOURNAL_SPECS and TARGET_WIDTHS
    for j_name, spec in JOURNAL_SPECS.items():
        assert j_name in hs_main
        assert j_name in hs_cloud
        for hex_col in spec["palette"]:
            assert hex_col in hs_main, (
                f"{j_name} color {hex_col} missing in shinystat/references/house-styles.md"
            )
            assert hex_col in hs_cloud, (
                f"{j_name} color {hex_col} missing in shinystat-cloud/references/house-styles.md"
            )

    for w_key, w_val in TARGET_WIDTHS.items():
        assert f'"{w_key}": {w_val}' in hs_main
        assert f'"{w_key}": {w_val}' in hs_cloud

    # 3. Cloud illustration.md SEM dict must match medstat.figures.illustration.SEM
    for sem_key, sem_hex in SEM.items():
        assert f'"{sem_key}": "{sem_hex}"' in il_cloud


def test_forest_non_contiguous_index_and_cif_validation(tmp_path):
    from medstat.figures.forest import plot_forest
    from medstat.figures.survival import plot_cif, plot_kaplan_meier

    # 1. Forest plot with non-contiguous index
    forest_df = pd.DataFrame(
        {
            "term": ["Overall", "Age >= 65", "Male Sex", "Diabetes", "Hypertension"],
            "estimate": [1.45, 1.62, 1.15, 1.88, 1.30],
            "ci_lower": [1.12, 1.20, 0.88, 1.35, 0.98],
            "ci_upper": [1.88, 2.18, 1.50, 2.62, 1.72],
            "p_value": [0.004, 0.001, 0.312, 0.0002, 0.068],
            "events_text": ["120/500", "65/220", "55/260", "48/150", "72/310"],
            "is_header": [False, np.nan, False, True, False],
        }
    )
    forest_non_contig = forest_df.iloc[[0, 2, 4]].copy()
    forest_non_contig.index = [10, 25, 42]
    res_fr = plot_forest(
        forest_non_contig,
        scale="HR",
        journal="NEJM",
        out_path=str(tmp_path / "forest_non_contig.png"),
    )
    assert res_fr.qa["passed"] is True

    # Ratio scale strictly positive validation raises ValueError on non-positive CI
    bad_forest = forest_df.copy()
    bad_forest.loc[2, "ci_lower"] = -0.20
    with pytest.raises(
        ValueError, match="Ratio scale 'HR' requires strictly positive ci_lower"
    ):
        plot_forest(
            bad_forest,
            scale="HR",
            journal="NEJM",
            out_path=str(tmp_path / "bad_forest.png"),
        )

    # 2. Inverted CIF lower CI check raises ValueError
    bad_cif_df = pd.DataFrame(
        {
            "time": [0, 10],
            "cif_primary": [0.10, 0.20],
            "cif_primary_ci_lower": [0.15, 0.25],  # ci_lower > cif
            "cif_primary_ci_upper": [0.20, 0.30],
            "cif_competing": [0.05, 0.10],
            "cif_competing_ci_lower": [0.02, 0.05],
            "cif_competing_ci_upper": [0.08, 0.15],
        }
    )
    with pytest.raises(
        ValueError, match="Primary CIF lower confidence bound exceeds point estimate"
    ):
        plot_cif(bad_cif_df, out_path=str(tmp_path / "bad_cif.png"))

    # Inverted CIF upper CI check raises ValueError
    bad_cif_upper = pd.DataFrame(
        {
            "time": [0, 10],
            "cif_primary": [0.10, 0.20],
            "cif_primary_ci_lower": [0.05, 0.10],
            "cif_primary_ci_upper": [0.08, 0.15],  # ci_upper < cif
            "cif_competing": [0.05, 0.10],
            "cif_competing_ci_lower": [0.02, 0.05],
            "cif_competing_ci_upper": [0.08, 0.15],
        }
    )
    with pytest.raises(
        ValueError, match="Primary CIF upper confidence bound is below point estimate"
    ):
        plot_cif(bad_cif_upper, out_path=str(tmp_path / "bad_cif_upper.png"))

    # 3. Kaplan-Meier time_unit PHI gate check
    km_phi_dict = {
        "Group A": {
            "timeline": np.array([0, 10, 20]),
            "survival": np.array([1.0, 0.9, 0.8]),
            "ci_lower": np.array([1.0, 0.85, 0.75]),
            "ci_upper": np.array([1.0, 0.95, 0.85]),
        }
    }
    with pytest.raises(ValueError, match="Zero-PHI Violation"):
        plot_kaplan_meier(
            km_phi_dict,
            time_unit="Patient HN 98765432",
            out_path=str(tmp_path / "km_phi.png"),
        )

    # 4. CIF time_unit PHI gate check
    valid_cif_df = pd.DataFrame(
        {
            "time": [0, 10],
            "cif_primary": [0.10, 0.20],
            "cif_primary_ci_lower": [0.05, 0.10],
            "cif_primary_ci_upper": [0.15, 0.25],
            "cif_competing": [0.05, 0.10],
            "cif_competing_ci_lower": [0.02, 0.05],
            "cif_competing_ci_upper": [0.08, 0.15],
        }
    )
    with pytest.raises(ValueError, match="Zero-PHI Violation"):
        plot_cif(
            valid_cif_df,
            time_unit="Patient HN 98765432",
            out_path=str(tmp_path / "cif_phi.png"),
        )

    # 5. Forest plot favor labels PHI gate check
    with pytest.raises(ValueError, match="Zero-PHI Violation"):
        plot_forest(
            forest_non_contig,
            journal="NEJM",
            favor_left_label="Favors นายสมชาย",
            out_path=str(tmp_path / "forest_phi.png"),
        )

    # 6. Forest plot events_text PHI gate check
    forest_phi_events = forest_non_contig.copy()
    forest_phi_events.loc[10, "events_text"] = "HN 12345678"
    with pytest.raises(ValueError, match="Zero-PHI Violation"):
        plot_forest(
            forest_phi_events,
            journal="NEJM",
            out_path=str(tmp_path / "forest_phi_events.png"),
        )


def test_bland_altman_and_love_phi_gates_and_fallback(tmp_path):
    from medstat.figures.agreement import plot_bland_altman
    from medstat.figures.balance import plot_love

    ba_data = {
        "means": [10.0, 12.0],
        "diffs": [0.5, -0.2],
        "mean_diff": 0.15,
        "loa_upper": 1.2,
        "loa_lower": -0.9,
    }
    with pytest.raises(ValueError, match="Zero-PHI Violation"):
        plot_bland_altman(
            ba_data,
            title="Patient HN 98765432 Agreement",
            out_path=str(tmp_path / "ba_phi.png"),
        )

    love_data = {"covariates": ["Age", "นายสมชาย"], "smd_raw": [0.2, 0.4]}
    with pytest.raises(ValueError, match="Zero-PHI Violation"):
        plot_love(love_data, title="Love Plot", out_path=str(tmp_path / "love_phi.png"))

    # Test dict-input handling without Variable alias (Var_i fallback)
    no_var_dict = {"smd_raw": [0.2, 0.4], "smd_matched": [0.05, 0.08]}
    res_love = plot_love(
        no_var_dict,
        title="Clean Love Plot",
        out_path=str(tmp_path / "love_novar.png"),
    )
    assert res_love.qa["passed"] is True
    assert "Var_1" in res_love.source_df["Variable"].values


def test_consort_rct_and_cohort_flow_validation(tmp_path):
    from medstat.data.retention import SampleFlowTracker
    from medstat.figures.flow import cohort_flow_from_tracker, consort_rct

    # Exactly 2 arms required
    with pytest.raises(
        ValueError, match="CONSORT layout requires exactly two trial arms"
    ):
        consort_rct(
            assessed=100,
            excluded={"refused": 10},
            arms=[{"name": "Arm 1", "allocated": 45, "received": 45}],
            out_path=str(tmp_path / "consort_1arm.png"),
        )

    # Invalid version
    with pytest.raises(ValueError, match="version must be '2025' or '2010'"):
        consort_rct(
            assessed=100,
            excluded={"refused": 10},
            arms=[
                {"name": "Arm 1", "allocated": 45, "received": 45},
                {"name": "Arm 2", "allocated": 45, "received": 45},
            ],
            version="2000",
            out_path=str(tmp_path / "consort_badver.png"),
        )

    # Caller arm dictionary is copied and preserved
    original_arm = {
        "name": "Arm 1",
        "allocated": 45,
        "received": 45,
        "analyzed": 45,
    }
    arms_input = [
        original_arm,
        {"name": "Arm 2", "allocated": 45, "received": 45, "analyzed": 45},
    ]
    consort_rct(
        assessed=100,
        excluded={"refused": 10},
        arms=arms_input,
        out_path=str(tmp_path / "consort_ok.png"),
    )
    assert "analysed" not in original_arm, "Caller input dict was mutated"

    # Cohort tracker reconciliation check
    tracker = SampleFlowTracker(initial_n=500, initial_name="All Enrolled")
    tracker.record_stage("Stage 1", 450, 50, reason="Ineligible")
    # Tamper stage to cause itemized reason mismatch
    tracker.stages[0].reasons[0].count = 40  # 40 != 50
    with pytest.raises(ValueError, match="Sum of itemized exclusion reasons"):
        cohort_flow_from_tracker(
            tracker, out_path=str(tmp_path / "tracker_mismatch.png")
        )


def test_wound_topdown_and_artery_validation(monkeypatch):
    from medstat.figures.illustration import Scene

    sc = Scene(width="single_col", seed=42)
    try:
        # Invalid stenosis
        with pytest.raises(ValueError, match="stenosis must be a finite float in"):
            sc.artery_cross_section(cx=50, cy=50, stenosis=1.2)
        with pytest.raises(ValueError, match="stenosis must be a finite float in"):
            sc.artery_cross_section(cx=50, cy=50, stenosis=-0.1)
        with pytest.raises(ValueError, match="stenosis must be a finite float in"):
            sc.artery_cross_section(cx=50, cy=50, stenosis=float("nan"))

        # Invalid wound fractions
        with pytest.raises(ValueError, match="Unknown tissue fraction keys"):
            sc.wound_topdown(cx=50, cy=50, r=10, fractions={"unknown_tissue": 0.5})
        with pytest.raises(ValueError, match="Invalid tissue fraction"):
            sc.wound_topdown(cx=50, cy=50, r=10, fractions={"granulation": -0.5})
        with pytest.raises(ValueError, match="Total fraction must be positive"):
            sc.wound_topdown(cx=50, cy=50, r=10, fractions={"granulation": 0.0})

        # Boundary crossing check
        with monkeypatch.context() as m:
            m.setattr(
                "medstat.figures.illustration.MplPath.contains_points",
                lambda self, pts: np.array([False]),
            )
            with pytest.raises(ValueError, match="crosses outer boundary"):
                sc.wound_topdown(
                    cx=50, cy=50, r=10, fractions={"granulation": 0.5, "slough": 0.5}
                )

        # Tolerance mismatch check
        with pytest.raises(ValueError, match="Wound tissue area mismatch"):
            sc.wound_topdown(
                cx=50,
                cy=50,
                r=10,
                fractions={"granulation": 0.5, "slough": 0.5},
                tol=0.0,
            )

        # Valid wound with zero-fraction tissues (epithelial and necrotic = 0.0)
        ach = sc.wound_topdown(
            cx=50, cy=50, r=10, fractions={"granulation": 0.7, "slough": 0.3}
        )
        assert abs(ach["epithelial"]) < 1e-4
        assert abs(ach["necrotic"]) < 1e-4
        assert abs(ach["granulation"] - 0.7) < 0.05
        assert abs(ach["slough"] - 0.3) < 0.05
    finally:
        plt.close(sc.fig)

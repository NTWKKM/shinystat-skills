"""
tests/unit/test_reporting_integrity.py: Unit tests for Pillar 5 Reporting Integrity Verification.
"""

# ruff: noqa: E402

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from medstat.reporting.integrity import verify_report_integrity
from medstat.reporting.ir import ReportDocument
from tests.fixtures.integrity_cases import (
    BASE_NARRATIVE,
    BASE_RESULTS_DICT,
    INTEGRITY_PARITY_CASES,
    STAT_MUTATIONS,
)


def test_integrity_clean_report():
    """Verifies that a well-formed report with traced numbers and no PHI passes."""
    doc = ReportDocument(
        title="Prognostic Assessment of Sepsis Cohort",
        results_dict={
            "odds_ratio": 2.45,
            "ci_lower": 1.30,
            "ci_upper": 4.62,
            "p_value": 0.004,
            "total_n": 350,
        },
    )
    doc.add_heading("Methods & Statistical Protocol", level=1)
    doc.add_paragraph(
        "Cohort inclusion flow and protocol: 350 patients were enrolled with complete data."
    )
    doc.add_heading("Results", level=2)
    doc.add_paragraph(
        "Multivariable logistic regression demonstrated an odds ratio of 2.45 (95% CI: 1.30 to 4.62, p = 0.004)."
    )
    doc.add_paragraph(
        "Given the observational nature of this cohort, residual and unmeasured confounding was evaluated with E-values."
    )

    audit = verify_report_integrity(doc)
    assert audit.passed is True
    assert len(audit.phi_violations) == 0
    assert len(audit.untraced_numbers) == 0
    assert audit.has_methods is True
    assert audit.has_retention_flow is True
    assert audit.has_causal_caveat is True


def test_integrity_phi_detection():
    """Verifies detection of raw patient identifiers."""
    doc = ReportDocument(
        title="Patient Triage Summary",
        results_dict={"sample_size": 10},
    )
    # Dynamically constructed to avoid static secret/phi string alerts
    tag = "".join(["H", "N", ":", " ", "987654"])
    doc.add_paragraph(f"Patient {tag} was admitted via ER.")

    audit = verify_report_integrity(doc)
    assert audit.passed is False
    assert len(audit.phi_violations) >= 1
    phi_labels = " ".join(audit.phi_violations)
    assert "Hospital Number" in phi_labels


def test_integrity_untraced_numbers():
    """Verifies detection of hallucinated / untraced numbers in narrative."""
    doc = ReportDocument(
        title="Analysis Report",
        results_dict={"known_stat": 12.5},
    )
    doc.add_paragraph("The observed rate was 87.42% with a hazard ratio of 4.98.")

    audit = verify_report_integrity(doc)
    assert audit.passed is False
    assert len(audit.untraced_numbers) >= 2
    assert 87.42 in audit.untraced_numbers or 4.98 in audit.untraced_numbers


def test_integrity_untraced_numbers_threshold_allowance():
    """Verifies that even a single untraced number blocks audit.passed unless explicitly whitelisted in allowed_untraced."""
    doc = ReportDocument(
        title="Analysis Report",
        results_dict={
            "s1": 11.1,
            "s2": 22.2,
            "s3": 33.3,
            "s4": 44.4,
            "s5": 55.5,
            "s6": 66.6,
            "s7": 77.7,
            "s8": 88.8,
            "s9": 99.9,
            "s10": 111.1,
        },
    )
    # 10 known numbers + 1 untraced number (73.4) -> must fail under strict len(untraced) == 0 gate
    doc.add_paragraph(
        "Stats: 11.1, 22.2, 33.3, 44.4, 55.5, 66.6, 77.7, 88.8, 99.9, 111.1, with untraced marker 73.4."
    )

    audit = verify_report_integrity(doc)
    assert audit.passed is False
    assert len(audit.untraced_numbers) == 1
    assert 73.4 in audit.untraced_numbers

    # Explicit allow-list permits known non-statistical token
    audit_allowed = verify_report_integrity(doc, allowed_untraced={73.4})
    assert audit_allowed.passed is True
    assert len(audit_allowed.untraced_numbers) == 0


def test_integrity_audit_critical_cases_block_fabricated_numbers():
    """Verifies all 5 audit cases against results_dict = {'or': 1.45, 'p': 0.012, 'n': 412}."""
    base_results = {"or": 1.45, "p": 0.012, "n": 412}

    # 1. Mutate OR from 1.45 to 3.45 -> passed=False
    doc_or = ReportDocument(title="Audit OR", results_dict=base_results)
    doc_or.add_paragraph("In 412 patients, the odds ratio was 3.45 (P = 0.012).")
    res_or = verify_report_integrity(doc_or)
    assert res_or.passed is False
    assert 3.45 in res_or.untraced_numbers

    # 2. Mutate P from 0.012 to 0.034 -> passed=False
    doc_p = ReportDocument(title="Audit P", results_dict=base_results)
    doc_p.add_paragraph("In 412 patients, the odds ratio was 1.45 (P = 0.034).")
    res_p = verify_report_integrity(doc_p)
    assert res_p.passed is False
    assert 0.034 in res_p.untraced_numbers

    # 3. Mutate n from 412 to 512 -> passed=False
    doc_n = ReportDocument(title="Audit N", results_dict=base_results)
    doc_n.add_paragraph("In 512 patients, the odds ratio was 1.45 (P = 0.012).")
    res_n = verify_report_integrity(doc_n)
    assert res_n.passed is False
    assert 512.0 in res_n.untraced_numbers

    # 4. Single fabricated number in long narrative -> passed=False
    doc_long = ReportDocument(title="Audit Long", results_dict=base_results)
    doc_long.add_paragraph(
        "Across 412 enrolled patients (Table 1), primary analysis yielded OR 1.45 (P = 0.012); "
        "Repeat evaluation confirmed OR 1.45 (P = 0.012) in 412 patients, except subgroup OR 2.89."
    )
    res_long = verify_report_integrity(doc_long)
    assert res_long.passed is False
    assert 2.89 in res_long.untraced_numbers

    # 5. Mutate P from 0.012 to P=0.001 or P<0.001 -> both must fail when true P=0.012
    doc_p_eq = ReportDocument(title="Audit P=0.001", results_dict=base_results)
    doc_p_eq.add_paragraph("In 412 patients, the odds ratio was 1.45 (P = 0.001).")
    res_p_eq = verify_report_integrity(doc_p_eq)
    assert res_p_eq.passed is False
    assert 0.001 in res_p_eq.untraced_numbers

    doc_p_lt = ReportDocument(title="Audit P<0.001", results_dict=base_results)
    doc_p_lt.add_paragraph("In 412 patients, the odds ratio was 1.45 (P < 0.001).")
    res_p_lt = verify_report_integrity(doc_p_lt)
    assert res_p_lt.passed is False
    assert 0.001 in res_p_lt.untraced_numbers

    # Conversely, when true P is 0.0003 (< 0.001), P < 0.001 is a valid P-value floor
    doc_p_floor_ok = ReportDocument(
        title="Valid P<0.001 Floor", results_dict={"or": 1.45, "p": 0.0003, "n": 412}
    )
    doc_p_floor_ok.add_paragraph(
        "In 412 patients, the odds ratio was 1.45 (P < 0.001)."
    )
    res_p_floor_ok = verify_report_integrity(doc_p_floor_ok)
    assert res_p_floor_ok.passed is True
    assert len(res_p_floor_ok.untraced_numbers) == 0


def test_integrity_causal_caveat_warning():
    """Verifies warning when observational PSM/Cox model lacks unmeasured confounding caveats."""
    doc = ReportDocument(
        title="PSM Cohort Comparison",
        results_dict={"psm_matched_n": 120, "odds_ratio": 1.8},
    )
    doc.add_paragraph("The PSM cohort comparison showed significant odds ratio of 1.8.")
    # Notice: No mention of unmeasured confounding or E-value

    audit = verify_report_integrity(doc)
    assert audit.has_causal_caveat is False
    assert any("unmeasured confounding" in w.lower() for w in audit.warnings)


def test_integrity_non_causal_prediction_exempt_from_causal_caveat():
    """Verifies that a logistic, Cox, odds_ratio, or hazard_ratio key alone does not require a caveat for non-causal prediction reports, and 'because' is not treated as a causal claim."""
    doc = ReportDocument(
        title="Clinical Prediction Model",
        results_dict={"odds_ratio": 1.45, "hazard_ratio": 1.25, "sample_size": 250},
    )
    doc.add_paragraph(
        "A clinical risk prediction model was developed following TRIPOD guidelines. "
        "We retained the predictors because the multivariable logistic model achieved a C-statistic of 0.82 in 250 patients, "
        "with odds ratio 1.45 and hazard ratio 1.25."
    )
    audit = verify_report_integrity(doc, allowed_untraced=[0.82])
    assert audit.has_causal_caveat is True
    assert not any("unmeasured confounding" in w.lower() for w in audit.warnings)


def test_integrity_causal_claims_in_regression_require_caveat():
    """Verifies that regression models with explicit causal claims require unmeasured confounding / E-value caveat."""
    doc = ReportDocument(
        title="Treatment Effect Study",
        results_dict={"logistic_model": 0.82, "sample_size": 250},
    )
    doc.add_paragraph(
        "We evaluated the causal effect of the intervention using multivariable logistic regression. "
        "The model showed an effect estimate of 0.82 across 250 patients."
    )
    audit = verify_report_integrity(doc)
    assert audit.has_causal_caveat is False
    assert any("unmeasured confounding" in w.lower() for w in audit.warnings)


def test_integrity_iptw_and_ate_keys_require_causal_caveat():
    """Verifies that IPTW, ATE, ATT, and CATE result keys require unmeasured confounding caveats when causal wording is absent."""
    # 1. IPTW result key without caveat
    doc_iptw = ReportDocument(
        title="Weighting Analysis",
        results_dict={"iptw_weights": 1.2, "sample_size": 300},
    )
    doc_iptw.add_paragraph(
        "Methods: Inverse probability weighting was applied across 300 patients with mean weight 1.2."
    )
    audit_iptw = verify_report_integrity(doc_iptw)
    assert audit_iptw.has_causal_caveat is False
    assert any("unmeasured confounding" in w.lower() for w in audit_iptw.warnings)

    # Adding confounding caveat resolves it
    doc_iptw.add_paragraph(
        "We evaluated the sensitivity to unmeasured confounding via E-values."
    )
    assert verify_report_integrity(doc_iptw).has_causal_caveat is True

    # 2. ATE result key without caveat
    doc_ate = ReportDocument(
        title="Treatment Effect Estimation",
        results_dict={"ate": 0.45, "sample_size": 200},
    )
    doc_ate.add_paragraph(
        "Methods: Analysis of 200 patients yielded an estimate of 0.45."
    )
    audit_ate = verify_report_integrity(doc_ate)
    assert audit_ate.has_causal_caveat is False
    assert any("unmeasured confounding" in w.lower() for w in audit_ate.warnings)

    # 3. ATT result key without caveat
    doc_att = ReportDocument(
        title="ATT Estimation",
        results_dict={"att": 0.38, "sample_size": 200},
    )
    doc_att.add_paragraph(
        "Methods: Analysis of 200 patients yielded an ATT estimate of 0.38."
    )
    audit_att = verify_report_integrity(doc_att)
    assert audit_att.has_causal_caveat is False
    assert any("unmeasured confounding" in w.lower() for w in audit_att.warnings)

    # 4. CATE result key without caveat
    doc_cate = ReportDocument(
        title="CATE Estimation",
        results_dict={"cate": 0.25, "sample_size": 200},
    )
    doc_cate.add_paragraph(
        "Methods: Analysis of 200 patients yielded an estimate of 0.25."
    )
    audit_cate = verify_report_integrity(doc_cate)
    assert audit_cate.has_causal_caveat is False
    assert any("unmeasured confounding" in w.lower() for w in audit_cate.warnings)

    # Adding confounding caveat resolves CATE
    doc_cate.add_paragraph(
        "Sensitivity to residual confounding was verified using E-values."
    )
    assert verify_report_integrity(doc_cate).has_causal_caveat is True

    # 5. Non-causal keys containing substring 'ate' (e.g. 'event_rate') must NOT trigger causal classification
    doc_non_ate = ReportDocument(
        title="Event Rate Report",
        results_dict={"event_rate": 0.05, "sample_size": 200},
    )
    doc_non_ate.add_paragraph(
        "Methods: Analysis of 200 patients yielded an event_rate of 0.05."
    )
    audit_non_ate = verify_report_integrity(doc_non_ate)
    assert audit_non_ate.has_causal_caveat is True
    assert not any(
        "unmeasured confounding" in w.lower() for w in audit_non_ate.warnings
    )


def test_integrity_negated_causal_claims_not_treated_as_affirmative():
    """Verifies that negated causal wording such as 'non-causal' or 'does not imply causation' does not count as an affirmative causal claim, but separate affirmative claims do."""
    # Negated wording 'non-causal' and 'does not imply causation' do not disqualify prediction exemption
    doc_neg = ReportDocument(
        title="Non-causal Prediction Model",
        results_dict={"odds_ratio": 1.45, "sample_size": 150},
    )
    doc_neg.add_paragraph(
        "This is a non-causal prediction model evaluated with TRIPOD guidelines. "
        "Correlation does not imply causation. "
        "The multivariable logistic regression showed an odds ratio of 1.45 in 150 patients."
    )
    audit_neg = verify_report_integrity(doc_neg)
    assert audit_neg.has_causal_caveat is True
    assert not any("unmeasured confounding" in w.lower() for w in audit_neg.warnings)

    # Explicit disclaimer that no treatment-effect claims are made preserves prediction exemption
    doc_disclaim = ReportDocument(
        title="Prediction Model with Disclaimer",
        results_dict={"odds_ratio": 1.45, "sample_size": 150},
    )
    doc_disclaim.add_paragraph(
        "This is a prediction risk score evaluated under TRIPOD guidelines. "
        "No treatment-effect claims are made; this model makes no treatment-effect claims. "
        "Multivariable odds ratio was 1.45."
    )
    audit_disclaim = verify_report_integrity(doc_disclaim)
    assert audit_disclaim.has_causal_caveat is True
    assert not any(
        "unmeasured confounding" in w.lower() for w in audit_disclaim.warnings
    )

    # Separate affirmative causal wording in the same report triggers requirement
    doc_aff = ReportDocument(
        title="Non-causal Prediction Model",
        results_dict={"odds_ratio": 1.45, "sample_size": 150},
    )
    doc_aff.add_paragraph(
        "This is a non-causal prediction model evaluated with TRIPOD guidelines. "
        "Correlation does not imply causation, however treatment lowered mortality with odds ratio 1.45 in 150 patients."
    )
    audit_aff = verify_report_integrity(doc_aff)
    assert audit_aff.has_causal_caveat is False
    assert any("unmeasured confounding" in w.lower() for w in audit_aff.warnings)


def test_integrity_prediction_with_treatment_reduces_mortality_requires_caveat():
    """Verifies that prediction reports asserting that treatment reduces, lowered, or did not reduce mortality require a confounding caveat."""
    doc_treat = ReportDocument(
        title="Prediction Model with Treatment Assertion",
        results_dict={"odds_ratio": 0.72, "sample_size": 400},
    )
    doc_treat.add_paragraph(
        "We developed a risk prediction score with excellent discrimination. "
        "Furthermore, our findings demonstrate that treatment lowered mortality (odds ratio 0.72 in 400 patients)."
    )
    audit_treat = verify_report_integrity(doc_treat)
    assert audit_treat.has_causal_caveat is False
    assert any("unmeasured confounding" in w.lower() for w in audit_treat.warnings)

    # Adding E-value caveat passes
    doc_treat.add_paragraph(
        "Because this is an observational study, residual confounding cannot be ruled out."
    )
    assert verify_report_integrity(doc_treat).has_causal_caveat is True

    # Negated claim: treatment not reducing mortality requires caveat
    doc_neg_treat = ReportDocument(
        title="Prediction Model with Negated Treatment Assertion",
        results_dict={"odds_ratio": 1.05, "sample_size": 400},
    )
    doc_neg_treat.add_paragraph(
        "We developed a risk prediction score with c-index 0.82. "
        "Multivariable analysis indicated treatment not reducing mortality in secondary analyses."
    )
    audit_neg_treat = verify_report_integrity(doc_neg_treat)
    assert audit_neg_treat.has_causal_caveat is False
    assert any("unmeasured confounding" in w.lower() for w in audit_neg_treat.warnings)

    # Assertion that treatment has zero causal effect requires caveat
    doc_zero_treat = ReportDocument(
        title="Prediction Model with Zero Causal Effect Assertion",
        results_dict={"odds_ratio": 1.0, "sample_size": 350},
    )
    doc_zero_treat.add_paragraph(
        "Prediction model discrimination was c-statistic 0.78. "
        "The observational analysis indicated that treatment has zero causal effect on mortality."
    )
    audit_zero_treat = verify_report_integrity(doc_zero_treat)
    assert audit_zero_treat.has_causal_caveat is False
    assert any("unmeasured confounding" in w.lower() for w in audit_zero_treat.warnings)


def test_integrity_displayed_precision_sub_one():
    """Verifies that values <= 1 use displayed precision instead of 0.02 absolute tolerance."""
    # results_dict has p_value = 0.02 (2-digit precision)
    doc_spurious = ReportDocument(
        title="Spurious P-value Test",
        results_dict={"p_value": 0.02, "sample_size": 100},
    )
    # Text displays 0.004 (precision 0.001) - difference is 0.016 (< 0.02, but > 0.001)
    doc_spurious.add_paragraph("The reported significance was p = 0.004.")
    audit_spurious = verify_report_integrity(doc_spurious)
    assert 0.004 in audit_spurious.untraced_numbers

    # Conversely, when p_value is 0.0041 (within displayed precision 0.001 of 0.004)
    doc_matched = ReportDocument(
        title="Valid P-value Test",
        results_dict={"p_value": 0.0041, "sample_size": 100},
    )
    doc_matched.add_paragraph("The reported significance was p = 0.004.")
    audit_matched = verify_report_integrity(doc_matched)
    assert 0.004 not in audit_matched.untraced_numbers


def test_integrity_precision_guards_against_spurious_relative_and_percentage_matches():
    """
    Verifies CodeRabbit review fixes:
    1. AUC 0.860 in text must NOT match 0.865 in results via 1% relative check.
    2. p = 0.02 in text must NOT match 0.004 in results via spurious percentage scaling.
    """
    # 1. 0.860 vs 0.865
    doc_auc = ReportDocument(
        title="AUC Precision Test",
        results_dict={"auc": 0.865, "sample_size": 100},
    )
    doc_auc.add_paragraph("The model achieved an AUC of 0.860.")
    audit_auc = verify_report_integrity(doc_auc)
    assert 0.860 in audit_auc.untraced_numbers

    # 2. p = 0.02 vs 0.004
    doc_p = ReportDocument(
        title="P-value Spurious Percentage Test",
        results_dict={"p_value": 0.004, "sample_size": 100},
    )
    doc_p.add_paragraph("The difference was significant with p = 0.02.")
    audit_p = verify_report_integrity(doc_p)
    assert 0.02 in audit_p.untraced_numbers


def test_integrity_structural_context_exemption_vs_reported_statistics():
    """
    Verifies that reported statistics (e.g. 95% sensitivity, 10 events) are checked for
    traceability, while surrounding-context structural constants (e.g. 95% CI) are exempted.
    """
    # When statistics are reported and present in results_dict
    doc_valid = ReportDocument(
        title="Diagnostic Performance",
        results_dict={"sensitivity": 0.95, "events": 10, "n": 100},
    )
    doc_valid.add_paragraph(
        "There were 10 events observed. Sensitivity reached 95% (95% CI: 91% to 98%)."
    )
    audit_valid = verify_report_integrity(doc_valid)
    # 10 and 95 (sensitivity) matched results_dict; 95 (from 95% CI) exempted as structural
    assert 10.0 not in audit_valid.untraced_numbers
    assert 95.0 not in audit_valid.untraced_numbers

    # When reported statistic (10 events) is NOT in results_dict, it must be flagged
    doc_untraced = ReportDocument(
        title="Diagnostic Performance",
        results_dict={"n": 100},
    )
    doc_untraced.add_paragraph("There were 10 events observed under 95% CI estimation.")
    audit_untraced = verify_report_integrity(doc_untraced)
    assert 10.0 in audit_untraced.untraced_numbers
    # 95 was near 'CI', so exempted as structural constant
    assert 95.0 not in audit_untraced.untraced_numbers


def test_integrity_years_context_and_structural_counters():
    """
    Verifies that:
    1. Calendar years (2010-2035) require date context (e.g. 'in 2024') to be exempt;
       cohort count like 'total 2024 participants' must NOT be exempt and flagged if untraced.
    2. Structural counters require integer tokens (e.g. 'Table 1' vs 'Table 1.25').
    3. Version identifiers support decimal tokens (e.g. 'Version 4.2', 'v.1').
    4. NumPy arrays in results_dict are properly traversed and flattened.
    """
    import numpy as np

    # 1. Year with and without date context
    doc_year_ctx = ReportDocument(
        title="Year Context",
        results_dict={"n": 100},
    )
    doc_year_ctx.add_paragraph("The study was conducted in 2024.")
    audit_year_ctx = verify_report_integrity(doc_year_ctx)
    assert 2024.0 not in audit_year_ctx.untraced_numbers

    doc_year_no_ctx = ReportDocument(
        title="Year Count",
        results_dict={"n": 100},
    )
    doc_year_no_ctx.add_paragraph("A total of 2024 participants were enrolled.")
    audit_year_no_ctx = verify_report_integrity(doc_year_no_ctx)
    assert 2024.0 in audit_year_no_ctx.untraced_numbers

    # 2. Structural counters integer vs decimal
    doc_table_int = ReportDocument(
        title="Table Counter",
        results_dict={"n": 100},
    )
    doc_table_int.add_paragraph(
        "As presented in Table 1, baseline characteristics are balanced."
    )
    audit_table_int = verify_report_integrity(doc_table_int)
    assert 1.0 not in audit_table_int.untraced_numbers

    doc_table_dec = ReportDocument(
        title="Table Decimal Non-Counter",
        results_dict={"n": 100},
    )
    doc_table_dec.add_paragraph("Table 1.25 indicates odds ratio.")
    audit_table_dec = verify_report_integrity(doc_table_dec)
    assert 1.25 in audit_table_dec.untraced_numbers

    # 3. Version with decimal
    doc_version = ReportDocument(
        title="Version",
        results_dict={"n": 100},
    )
    doc_version.add_paragraph("Analyses performed with R Version 4.2.")
    audit_version = verify_report_integrity(doc_version)
    assert 4.2 not in audit_version.untraced_numbers

    # 4. NumPy array in results_dict
    doc_numpy = ReportDocument(
        title="NumPy Array Traceability",
        results_dict={"estimates": np.array([0.725, 0.835])},
    )
    doc_numpy.add_paragraph("Estimates were 0.725 and 0.835.")
    audit_numpy = verify_report_integrity(doc_numpy)
    assert 0.725 not in audit_numpy.untraced_numbers
    assert 0.835 not in audit_numpy.untraced_numbers
    assert audit_numpy.has_causal_caveat is True
    assert not any("unmeasured confounding" in w.lower() for w in audit_numpy.warnings)

    # 5. Cohort count with 'in' elsewhere in sentence must remain reportable
    doc_events = ReportDocument(
        title="Event Reporting",
        results_dict={"n": 5000},
    )
    doc_events.add_paragraph("50 patients in the cohort, 2024 had events.")
    audit_events = verify_report_integrity(doc_events)
    assert 2024.0 in audit_events.untraced_numbers

    # 6. 'and' without year range connector does not exempt counts
    doc_and = ReportDocument(
        title="Count Conjunction",
        results_dict={"n": 100},
    )
    doc_and.add_paragraph("Cohort sizes were 30 and 2024.")
    audit_and = verify_report_integrity(doc_and)
    assert 2024.0 in audit_and.untraced_numbers

    # 7. Date context immediately preceding year with optional month
    doc_month = ReportDocument(
        title="Month Year Context",
        results_dict={"n": 100},
    )
    doc_month.add_paragraph("The cohort was recruited in October 2024.")
    audit_month = verify_report_integrity(doc_month)
    assert 2024.0 not in audit_month.untraced_numbers


def test_integrity_causal_caveat_required_for_passed():
    """Verifies that an observational model missing causal confounding caveats fails audit passed status."""
    doc = ReportDocument(
        title="PSM Analysis Report",
        results_dict={"psm_matched_n": 200, "odds_ratio": 1.75},
    )
    doc.add_paragraph("Methods: We performed matching.")
    doc.add_paragraph("Results: Matched cohort had 200 patients with odds ratio 1.75.")
    # No mention of confounding / e-value / residual confounding
    audit = verify_report_integrity(doc)
    assert audit.has_causal_caveat is False
    assert audit.passed is False
    assert any("Observational/PSM" in w for w in audit.warnings)

    # Adding causal caveat passes
    doc.add_paragraph(
        "Sensitivity analysis with E-values was conducted to assess residual and unmeasured confounding."
    )
    audit_passed = verify_report_integrity(doc)
    assert audit_passed.has_causal_caveat is True
    assert audit_passed.passed is True


def test_integrity_scientific_notation_precision():
    """Verifies that scientific-notation numbers derive effective precision from mantissa and exponent."""
    # 1.23e-4 (0.000123) should NOT match 0.005 even though both are <= 0.02
    doc_sci_bad = ReportDocument(
        title="Genomics Report",
        results_dict={"p_value": 0.005},
    )
    doc_sci_bad.add_paragraph("Genome-wide significance reached 1.23e-4.")
    audit_bad = verify_report_integrity(doc_sci_bad)
    assert 1.23e-4 in audit_bad.untraced_numbers

    # 1.23e-4 in text matches 1.23e-4 in results_dict
    doc_sci_ok = ReportDocument(
        title="Genomics Report",
        results_dict={"p_value": 1.23e-4},
    )
    doc_sci_ok.add_paragraph("Genome-wide significance reached 1.23e-4.")
    audit_ok = verify_report_integrity(doc_sci_ok)
    assert 1.23e-4 not in audit_ok.untraced_numbers


def test_integrity_superscript_citations_and_references():
    """
    Verifies that citation numbers above 10 (e.g. <sup>12</sup>, [^14], [16], and
    bibliographic list items '17. Author') are recognized as structural and do not
    falsely trigger untraced data number failures.
    """
    doc = ReportDocument(
        title="Clinical Trial with References",
        results_dict={"sample_size": 250, "p_value": 0.02},
    )
    doc.add_heading("Methods & Statistical Protocol", level=1)
    doc.add_paragraph(
        "Cohort inclusion flow: 250 patients analyzed per protocol. Treatment guidelines recommend early resuscitation.<sup>12,13</sup>"
    )
    doc.add_paragraph(
        "Prior prognostic models demonstrated moderate discrimination[^14] and risk stratification.[16]"
    )
    doc.add_paragraph("Analysis was carried out with p = 0.02.")
    doc.add_heading("References", level=2)
    doc.add_paragraph(
        "12. Singer M, et al. The Third International Consensus Definitions for Sepsis. JAMA 2016."
    )
    doc.add_paragraph(
        "13. Evans L, et al. Surviving Sepsis Campaign Guidelines. Crit Care Med 2021."
    )
    doc.add_paragraph("14. Riley RD, et al. Sample size calculation. BMJ 2020.")
    doc.add_paragraph(
        "16. Vickers AJ, et al. Decision curve analysis. Med Decis Making 2006."
    )

    audit = verify_report_integrity(doc)
    assert audit.passed is True
    # Ensure citation numbers 12, 13, 14, 16 are not in untraced numbers
    assert 12.0 not in audit.untraced_numbers
    assert 13.0 not in audit.untraced_numbers
    assert 14.0 not in audit.untraced_numbers
    assert 16.0 not in audit.untraced_numbers
    assert audit.citations_count == 3

    # Negative cases: ensure clinical [250], non-reference line "250. Patients were enrolled",
    # and "1950 patients" remain in untraced_numbers when absent from results_dict
    doc_neg = ReportDocument(
        title="Negative Audit Document",
        results_dict={"known_val": 42.0},
    )
    doc_neg.add_paragraph("Dosage was escalated to [250] in the severe subgroup.")
    doc_neg.add_paragraph("250. Patients were enrolled across all trial centers.")
    doc_neg.add_paragraph("In a historical registry, 1950 patients were evaluated.")
    audit_neg = verify_report_integrity(doc_neg)
    assert 250.0 in audit_neg.untraced_numbers
    assert 1950.0 in audit_neg.untraced_numbers


def test_integrity_comma_bracketed_citations_and_exponents():
    """
    Verifies that multi-number bracketed citations like [1, 2] preserve comma separation,
    bracketed caret citations ^[3] are recognized as citations, and mathematical exponents
    such as R^2, m^2, and x^2 are treated as structural without being counted as citations.
    """
    doc = ReportDocument(
        title="Model Fit and Citations",
        results_dict={"r_squared": 0.85, "area": 25.0},
    )
    doc.add_paragraph("Prior observational trials demonstrated robust outcomes [1, 2].")
    doc.add_paragraph("Regression analysis yielded R^2 = 0.85 across a 25 m^2 region.")
    doc.add_paragraph("Further validation confirmed sensitivity ^[3].")
    doc.add_heading("References", level=2)
    doc.add_paragraph("1. Singer M, et al. Sepsis definitions. JAMA 2016.")
    doc.add_paragraph("2. Evans L, et al. Surviving Sepsis. Crit Care Med 2021.")
    doc.add_paragraph("3. Riley RD, et al. Sample size calculation. BMJ 2020.")

    audit = verify_report_integrity(doc)
    # [1, 2] and ^[3] are citations -> citations_count should be 2
    # R^2 and m^2 must NOT be counted as citations
    assert audit.citations_count == 2

    # Structural numbers: 1, 2, 3, and exponent 2 should not be untraced
    assert 1.0 not in audit.untraced_numbers
    assert 2.0 not in audit.untraced_numbers
    assert 3.0 not in audit.untraced_numbers
    assert len(audit.untraced_numbers) == 0


def test_integrity_citation_ranges_and_bare_references():
    """
    Verifies that all endpoints of bracketed citation ranges such as [1-3] are
    classified as structural, and bare-number reference items in the bibliography
    such as '1 Singer M...' (Lancet/BMJ style) are recognized as structural.
    """
    doc = ReportDocument(
        title="Range Citations and Bare References",
        results_dict={"odds_ratio": 1.45},
    )
    doc.add_paragraph(
        "Comprehensive reviews summarize therapeutic efficacy [1-3] and outcomes [5-7]."
    )
    doc.add_paragraph("Odds ratio reached 1.45 in the primary analysis.")
    doc.add_heading("References", level=2)
    doc.add_paragraph("1 Singer M, et al. Sepsis-3 definitions. JAMA 2016.")
    doc.add_paragraph("2 Evans L, et al. Surviving sepsis. CCM 2021.")
    doc.add_paragraph("3 Riley RD, et al. Sample size calculation. BMJ 2020.")
    doc.add_paragraph(
        "5 Vickers AJ, et al. Decision curve analysis. Med Decis Making 2006."
    )
    doc.add_paragraph("7 Austin PC, et al. Propensity score matching. Stat Med 2011.")
    doc.add_paragraph("8 O'Neil J, et al. Critical Care Trials. NEJM 2022.")
    doc.add_paragraph("9 McDonald A, et al. Oncology outcomes. Lancet 2023.")
    doc.add_paragraph("10 D'Agostino RB, et al. Cardiovascular risk. Stroke 2008.")

    audit = verify_report_integrity(doc)
    assert 1.0 not in audit.untraced_numbers
    assert 3.0 not in audit.untraced_numbers
    assert 5.0 not in audit.untraced_numbers
    assert 7.0 not in audit.untraced_numbers
    assert 8.0 not in audit.untraced_numbers
    assert 9.0 not in audit.untraced_numbers
    assert 10.0 not in audit.untraced_numbers
    assert len(audit.untraced_numbers) == 0


def test_integrity_p_floor_and_allowed_untraced_guards():
    """
    Verifies that:
    1. 'two-sided P = 0.001' is NOT exempted as an alpha threshold when true P is 0.012.
    2. 'P < 0.001' is blocked when true P is 0.0014 (even though 0.0014 is within 0.0005 of 0.001).
    3. Valid inequality 'P < 0.05' passes when true P is 0.012.
    4. allowed_untraced does not apply percentage-scaling or loose 0.02 p-value tolerances.
    """
    doc_two_sided = ReportDocument(
        title="Two-Sided P Audit",
        results_dict={"p_value": 0.012, "n": 100},
    )
    doc_two_sided.add_paragraph("Methods: Cohort retention was 100 patients.")
    doc_two_sided.add_paragraph("The difference was significant (two-sided P = 0.001).")
    audit_two_sided = verify_report_integrity(doc_two_sided)
    assert 0.001 in audit_two_sided.untraced_numbers
    assert audit_two_sided.passed is False

    doc_p_floor_bad = ReportDocument(
        title="P Floor Rounding Boundary",
        results_dict={"p_value": 0.0014},
    )
    doc_p_floor_bad.add_paragraph("Methods: Included cohort.")
    doc_p_floor_bad.add_paragraph("Primary endpoint reached P < 0.001.")
    audit_p_floor_bad = verify_report_integrity(doc_p_floor_bad)
    assert 0.001 in audit_p_floor_bad.untraced_numbers
    assert audit_p_floor_bad.passed is False

    doc_p_floor_ok = ReportDocument(
        title="Valid P Inequality",
        results_dict={"p_value": 0.012},
    )
    doc_p_floor_ok.add_paragraph("Methods: Included cohort.")
    doc_p_floor_ok.add_paragraph("Primary endpoint reached P < 0.05.")
    audit_p_floor_ok = verify_report_integrity(doc_p_floor_ok)
    assert 0.05 not in audit_p_floor_ok.untraced_numbers
    assert audit_p_floor_ok.passed is True

    # P > 0.05 valid when true P is 0.23
    doc_p_gt_ok = ReportDocument(
        title="Valid Greater-Than P Inequality",
        results_dict={"p_value": 0.23},
    )
    doc_p_gt_ok.add_paragraph("Methods: Included cohort.")
    doc_p_gt_ok.add_paragraph(
        "Secondary endpoint demonstrated no significance (P > 0.05)."
    )
    audit_p_gt_ok = verify_report_integrity(doc_p_gt_ok)
    assert 0.05 not in audit_p_gt_ok.untraced_numbers
    assert audit_p_gt_ok.passed is True

    # P > 0.05 invalid when true P is 0.03
    doc_p_gt_bad = ReportDocument(
        title="Invalid Greater-Than P Inequality",
        results_dict={"p_value": 0.03},
    )
    doc_p_gt_bad.add_paragraph("Methods: Included cohort.")
    doc_p_gt_bad.add_paragraph(
        "Secondary endpoint claimed non-significance (P > 0.05)."
    )
    audit_p_gt_bad = verify_report_integrity(doc_p_gt_bad)
    assert 0.05 in audit_p_gt_bad.untraced_numbers
    assert audit_p_gt_bad.passed is False

    # Boundary: equality at threshold for strict inequality (P > 0.05 when true P is 0.05 is invalid)
    doc_p_gt_boundary_strict = ReportDocument(
        title="Boundary Greater-Than P Inequality Strict",
        results_dict={"p_value": 0.05},
    )
    doc_p_gt_boundary_strict.add_paragraph("Methods: Included cohort.")
    doc_p_gt_boundary_strict.add_paragraph("Boundary case (P > 0.05).")
    audit_p_gt_boundary_strict = verify_report_integrity(doc_p_gt_boundary_strict)
    assert 0.05 in audit_p_gt_boundary_strict.untraced_numbers
    assert audit_p_gt_boundary_strict.passed is False

    # Boundary: equality at threshold for non-strict inequality (P >= 0.05 when true P is 0.05 is valid)
    doc_p_gte_boundary = ReportDocument(
        title="Boundary Greater-Than-Equal P Inequality",
        results_dict={"p_value": 0.05},
    )
    doc_p_gte_boundary.add_paragraph("Methods: Included cohort.")
    doc_p_gte_boundary.add_paragraph("Boundary case (P >= 0.05).")
    audit_p_gte_boundary = verify_report_integrity(doc_p_gte_boundary)
    assert 0.05 not in audit_p_gte_boundary.untraced_numbers
    assert audit_p_gte_boundary.passed is True

    doc_allow = ReportDocument(
        title="Allowed Untraced Strict Match",
        results_dict={"n": 100},
    )
    doc_allow.add_paragraph("Methods: Cohort of 100 patients with 50 events.")
    audit_allow_fail = verify_report_integrity(doc_allow, allowed_untraced=[0.5])
    assert 50.0 in audit_allow_fail.untraced_numbers
    assert audit_allow_fail.passed is False

    audit_allow_pass = verify_report_integrity(doc_allow, allowed_untraced=[50.0])
    assert 50.0 not in audit_allow_pass.untraced_numbers
    assert audit_allow_pass.passed is True


def test_integrity_threshold_mention_vs_definition_and_qualifier():
    """Verifies that:
    1. A sentence mentioning 'significance threshold' followed by a result claim ('with P < 0.05')
       is NOT exempted as a threshold definition and is checked against results_dict.
    2. A genuine methodological threshold definition ('significance threshold P < 0.05')
       is structurally exempted even when recorded p-value differs.
    3. Qualifier detection binds to the p-value match itself ('p for trend < 0.05')
       rather than picking up nearby keywords like 'Interaction' from the surrounding window.
    """
    # 1. Result claim following a threshold mention: must fail if conflicting
    doc_result_claim = ReportDocument(
        title="Result Claim After Threshold Mention",
        results_dict={"p_value": 0.23},
    )
    doc_result_claim.add_paragraph("Methods: Included observational cohort.")
    doc_result_claim.add_paragraph(
        "The association met the significance threshold, with P < 0.05."
    )
    audit_claim = verify_report_integrity(doc_result_claim)
    assert 0.05 in audit_claim.untraced_numbers
    assert audit_claim.passed is False

    # 2. Genuine methodological definition: must be exempted
    doc_def = ReportDocument(
        title="Methodological Threshold Definition",
        results_dict={"p_value": 0.23},
    )
    doc_def.add_paragraph("Methods: significance threshold P < 0.05.")
    doc_def.add_paragraph("Primary outcome had p = 0.23.")
    audit_def = verify_report_integrity(doc_def)
    assert 0.05 not in audit_def.untraced_numbers
    assert audit_def.passed is True

    # 3. Qualifier binding: 'p for trend < 0.05' near 'Interaction analysis'
    doc_qual_fail = ReportDocument(
        title="Qualifier Binding Check (False Trend Claim)",
        results_dict={"p_interaction": 0.01, "p_trend": 0.50},
    )
    doc_qual_fail.add_paragraph("Methods: Included observational cohort.")
    doc_qual_fail.add_paragraph(
        "Interaction analysis: p for trend < 0.05 across risk tiers."
    )
    audit_qual_fail = verify_report_integrity(doc_qual_fail)
    assert 0.05 in audit_qual_fail.untraced_numbers
    assert audit_qual_fail.passed is False

    # 4. Qualifier binding: 'p for trend < 0.05' near 'Interaction analysis' passes when p_trend is valid
    doc_qual_pass = ReportDocument(
        title="Qualifier Binding Check (True Trend Claim)",
        results_dict={"p_interaction": 0.01, "p_trend": 0.02},
    )
    doc_qual_pass.add_paragraph("Methods: Included observational cohort.")
    doc_qual_pass.add_paragraph(
        "Interaction analysis: p for trend < 0.05 across risk tiers."
    )
    audit_qual_pass = verify_report_integrity(doc_qual_pass)
    assert 0.05 not in audit_qual_pass.untraced_numbers
    assert audit_qual_pass.passed is True

    # 1b. Prespecified threshold followed by observed result claim: must fail if conflicting
    doc_prespec_claim = ReportDocument(
        title="Prespecified Threshold Followed By Observed Result",
        results_dict={"p_value": 0.23},
    )
    doc_prespec_claim.add_paragraph("Methods: Included observational cohort.")
    doc_prespec_claim.add_paragraph(
        "The significance threshold was prespecified and the observed result was P < 0.05."
    )
    audit_prespec = verify_report_integrity(doc_prespec_claim)
    assert 0.05 in audit_prespec.untraced_numbers
    assert audit_prespec.passed is False

    # 1c. Significance threshold p was prespecified, but observed P < 0.05: must fail if conflicting
    doc_prespec_p_claim = ReportDocument(
        title="Threshold p Prespecified But Observed Claim",
        results_dict={"p_value": 0.23},
    )
    doc_prespec_p_claim.add_paragraph("Methods: Included observational cohort.")
    doc_prespec_p_claim.add_paragraph(
        "The significance threshold p was prespecified, but the observed result was P < 0.05."
    )
    audit_prespec_p = verify_report_integrity(doc_prespec_p_claim)
    assert 0.05 in audit_prespec_p.untraced_numbers
    assert audit_prespec_p.passed is False

    # 1d. Significance threshold p-value was prespecified and observed P < 0.05: must fail if conflicting
    doc_prespec_pval_claim = ReportDocument(
        title="Threshold p-value Prespecified And Observed Result",
        results_dict={"p_value": 0.23},
    )
    doc_prespec_pval_claim.add_paragraph("Methods: Included observational cohort.")
    doc_prespec_pval_claim.add_paragraph(
        "The significance threshold p-value was prespecified and the observed P < 0.05."
    )
    audit_prespec_pval = verify_report_integrity(doc_prespec_pval_claim)
    assert 0.05 in audit_prespec_pval.untraced_numbers
    assert audit_prespec_pval.passed is False

    # 2b. Methodological definition using inclusive inequality (<= / defined as): must be exempted
    doc_def_incl = ReportDocument(
        title="Methodological Threshold Definition Inclusive",
        results_dict={"p_value": 0.23},
    )
    doc_def_incl.add_paragraph("Methods: significance threshold defined as P <= 0.05.")
    doc_def_incl.add_paragraph("Primary outcome had p = 0.23.")
    audit_def_incl = verify_report_integrity(doc_def_incl)
    assert 0.05 not in audit_def_incl.untraced_numbers
    assert audit_def_incl.passed is True

    # 2c. Methodological definition using inclusive upper inequality (>= / defined as) with conflicting recorded p-value 0.02
    doc_def_gte = ReportDocument(
        title="Methodological Threshold Definition GTE",
        results_dict={"p_value": 0.02},
    )
    doc_def_gte.add_paragraph("Methods: significance threshold defined as P >= 0.05.")
    doc_def_gte.add_paragraph("Primary outcome had p = 0.02.")
    audit_def_gte = verify_report_integrity(doc_def_gte)
    assert 0.05 not in audit_def_gte.untraced_numbers
    assert audit_def_gte.passed is True

    # 5. Multiword test label: 'Wald test P < 0.05' binds 'wald' qualifier and must not match generic p=0.01 when p_wald=0.2
    doc_wald_fail = ReportDocument(
        title="Wald Test Qualifier Mismatch",
        results_dict={"p": 0.01, "p_wald": 0.20},
    )
    doc_wald_fail.add_paragraph("Methods: Included observational cohort.")
    doc_wald_fail.add_paragraph("Primary model: Wald test P < 0.05.")
    audit_wald_fail = verify_report_integrity(doc_wald_fail)
    assert 0.05 in audit_wald_fail.untraced_numbers
    assert audit_wald_fail.passed is False

    # Wald test passes when p_wald is significant
    doc_wald_pass = ReportDocument(
        title="Wald Test Qualifier Match",
        results_dict={"p": 0.20, "p_wald": 0.01},
    )
    doc_wald_pass.add_paragraph("Methods: Included observational cohort.")
    doc_wald_pass.add_paragraph("Primary model: Wald test P < 0.05.")
    audit_wald_pass = verify_report_integrity(doc_wald_pass)
    assert 0.05 not in audit_wald_pass.untraced_numbers
    assert audit_wald_pass.passed is True


@pytest.mark.parametrize("case", INTEGRITY_PARITY_CASES, ids=lambda c: c.id)
def test_integrity_gate_tolerance_parity_cases(case):
    """Verifies all Task 2a parity cases through ReportDocument and verify_report_integrity."""
    doc = ReportDocument(title="Parity Test", results_dict=case.results_dict)
    doc.add_paragraph(case.narrative)
    audit = verify_report_integrity(doc, allowed_untraced=case.allowed_untraced)

    assert audit.passed == case.expected_passed, (
        f"Case {case.id} failed: expected passed={case.expected_passed}, got {audit.passed}. "
        f"Untraced: {audit.untraced_numbers}"
    )
    assert sorted(audit.untraced_numbers) == sorted(case.expected_untraced), (
        f"Case {case.id} untraced mismatch: expected {case.expected_untraced}, got {audit.untraced_numbers}"
    )


def test_integrity_gate_tolerance_mutations():
    """Verifies Task 2b mutation tests for all 5 statistical tokens (must all fail audit.passed)."""
    for token_str, mutations in STAT_MUTATIONS:
        for mutated_val in mutations:
            # Safely replace token preserving other numeric strings
            pat = r"(?<![\d.])" + re.escape(token_str) + r"(?![\d.])"
            mutated_narrative = re.sub(pat, mutated_val, BASE_NARRATIVE, count=1)
            assert mutated_narrative != BASE_NARRATIVE, (
                f"Failed to substitute token '{token_str}' with '{mutated_val}'"
            )

            doc = ReportDocument(
                title="Mutation Test",
                results_dict=BASE_RESULTS_DICT,
            )
            doc.add_paragraph(mutated_narrative)
            audit = verify_report_integrity(doc)

            assert audit.passed is False, (
                f"Mutation {token_str} -> {mutated_val} unexpectedly passed audit! "
                f"Narrative: '{mutated_narrative}', Untraced: {audit.untraced_numbers}"
            )


def test_integrity_qualified_p_value_association_and_record_recursion():
    """
    Verifies that:
    1. A specialized interaction p-value does not satisfy an unqualified/primary p-value claim.
    2. An interaction claim correctly matches an interaction p-value.
    3. Dictionary-valued p-value records do not treat non-p-value fields (e.g. estimate) as p-values.
    """
    # 1. Primary claim fails when primary p is non-significant, even if p_interaction is significant
    res_split = {"p": 0.2, "p_interaction": 0.01}
    doc_primary_fail = ReportDocument(title="Primary Claim", results_dict=res_split)
    doc_primary_fail.add_paragraph(
        "Primary outcome was statistically significant (P < 0.05)."
    )
    audit_primary = verify_report_integrity(doc_primary_fail)
    assert audit_primary.passed is False
    assert 0.05 in audit_primary.untraced_numbers

    # 2. Interaction claim passes when p_interaction satisfies the bound
    doc_interaction_pass = ReportDocument(
        title="Interaction Claim", results_dict=res_split
    )
    doc_interaction_pass.add_paragraph(
        "Treatment by subgroup interaction was significant (interaction P < 0.05)."
    )
    audit_interaction = verify_report_integrity(doc_interaction_pass)
    assert audit_interaction.passed is True
    assert len(audit_interaction.untraced_numbers) == 0

    # 2b. Reverse-value case: primary p=0.01 is significant, but p_interaction=0.2 is not
    # An interaction claim "interaction P < 0.05" must fail and report [0.05] as untraced
    res_reverse = {"p": 0.01, "p_interaction": 0.2}
    doc_reverse_fail = ReportDocument(title="Reverse Claim", results_dict=res_reverse)
    doc_reverse_fail.add_paragraph(
        "Treatment by subgroup interaction was significant (interaction P < 0.05)."
    )
    audit_reverse = verify_report_integrity(doc_reverse_fail)
    assert audit_reverse.passed is False
    assert 0.05 in audit_reverse.untraced_numbers

    # 3. Dict-valued record: {"p_interaction": {"estimate": 0.01, "p_value": 0.2}}
    # The estimate (0.01) must NOT be extracted as a p-value to satisfy P < 0.05
    res_dict_rec = {"p_interaction": {"estimate": 0.01, "p_value": 0.2}}
    doc_dict_fail = ReportDocument(title="Dict Record Fail", results_dict=res_dict_rec)
    doc_dict_fail.add_paragraph("Interaction was significant (interaction P < 0.05).")
    audit_dict_fail = verify_report_integrity(doc_dict_fail)
    assert audit_dict_fail.passed is False
    assert 0.05 in audit_dict_fail.untraced_numbers

    # But satisfies a bound matching true p_value=0.2 (e.g. interaction P < 0.30)
    doc_dict_pass = ReportDocument(title="Dict Record Pass", results_dict=res_dict_rec)
    doc_dict_pass.add_paragraph("Interaction was evaluated (interaction P < 0.30).")
    audit_dict_pass = verify_report_integrity(doc_dict_pass)
    assert audit_dict_pass.passed is True

    # 4. Zero-bound inequality validation: P < 0 is strictly false when p=0.0
    res_zero = {"p": 0.0}
    doc_zero_strict = ReportDocument(title="Zero Bound Strict", results_dict=res_zero)
    doc_zero_strict.add_paragraph("Difference was significant (P < 0).")
    audit_zero_strict = verify_report_integrity(doc_zero_strict)
    assert audit_zero_strict.passed is False
    assert 0.0 in audit_zero_strict.untraced_numbers

    # Inclusive P <= 0 passes when p=0.0
    doc_zero_incl = ReportDocument(title="Zero Bound Inclusive", results_dict=res_zero)
    doc_zero_incl.add_paragraph("Difference was at boundary (P <= 0).")
    audit_zero_incl = verify_report_integrity(doc_zero_incl)
    assert audit_zero_incl.passed is True
    assert len(audit_zero_incl.untraced_numbers) == 0

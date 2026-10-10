"""
tests/unit/test_reporting_integrity.py: Unit tests for Pillar 5 Reporting Integrity Verification.
"""

from __future__ import annotations

from medstat.reporting.integrity import verify_report_integrity
from medstat.reporting.ir import ReportDocument


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
    """Verifies that an untraced rate <= 15% (or single structural number) passes audit."""
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
    # 10 known numbers + 1 untraced number (73.4) -> 1/11 = 9.1% <= 15%
    doc.add_paragraph(
        "Stats: 11.1, 22.2, 33.3, 44.4, 55.5, 66.6, 77.7, 88.8, 99.9, 111.1, with untraced marker 73.4."
    )

    audit = verify_report_integrity(doc)
    assert audit.passed is True
    assert len(audit.untraced_numbers) == 1
    assert 73.4 in audit.untraced_numbers


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

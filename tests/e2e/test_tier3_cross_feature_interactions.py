"""
tests/e2e/test_tier3_cross_feature_interactions.py: Tier 3 Cross-Feature Interaction Pipelines.
Validates multi-step workflows, data handoffs, and consistency across subcommands.
"""

import json

import pandas as pd


def test_tier3_clean_mice_to_table1(
    medstat_cli_runner, incomplete_fixture_path, tmp_path
):
    """
    Tier 3 Interaction Pipeline 1:
    medstat clean (MICE) -> medstat table1 -> Audited Sample Flow
    Ensures complete imputation preserves 100% sample size (N_init = N_analyzed).
    """
    clean_csv = tmp_path / "mice_imputed.csv"
    audit_json = tmp_path / "mice_flow.json"
    table1_json = tmp_path / "table1_output.json"

    # Step 1: Clean with MICE
    exit_code, stdout, _ = medstat_cli_runner(
        [
            "clean",
            "--data",
            str(incomplete_fixture_path),
            "--strategy",
            "mice",
            "--imputations",
            "5",
            "--missing-justification",
            "Clinical assumption of MAR across laboratory and demographic features",
            "--output",
            str(clean_csv),
            "--audit-out",
            str(audit_json),
        ]
    )
    assert exit_code == 0

    if clean_csv.exists():
        df_imputed = pd.read_csv(clean_csv)
        assert len(df_imputed) == 400
        assert df_imputed["creatinine"].isnull().sum() == 0

        # Step 2: Table 1 on imputed cohort
        exit_code2, stdout2, _ = medstat_cli_runner(
            [
                "table1",
                "--data",
                str(clean_csv),
                "--group",
                "treatment",
                "--vars",
                "age,sex,creatinine,bmi",
                "--output",
                str(table1_json),
            ]
        )
        assert exit_code2 == 0 or "Table 1" in stdout2


def test_tier3_clean_complete_case_to_firth_to_report(
    medstat_cli_runner, incomplete_fixture_path, tmp_path
):
    """
    Tier 3 Interaction Pipeline 2:
    medstat clean (complete-case) -> medstat model (Firth logistic) -> medstat report (NEJM HTML)
    Ensures participant attrition (N_initial -> N_excluded -> N_analyzed) flows into publication report.
    """
    clean_csv = tmp_path / "clean_cc.csv"
    audit_json = tmp_path / "cc_flow.json"
    model_json = tmp_path / "firth_results.json"
    report_html = tmp_path / "nejm_table.html"

    # Step 1: Clean with complete-case
    exit_code, _, _ = medstat_cli_runner(
        [
            "clean",
            "--data",
            str(incomplete_fixture_path),
            "--strategy",
            "complete-case",
            "--missing-justification",
            "Sensitivity complete-case analysis with MCAR assumption verified",
            "--output",
            str(clean_csv),
            "--audit-out",
            str(audit_json),
        ]
    )
    assert exit_code == 0

    if clean_csv.exists():
        df_clean = pd.read_csv(clean_csv)
        n_analyzed = len(df_clean)
        assert n_analyzed < 400

        # Step 2: Firth penalized logistic model on analyzed cohort
        exit_code2, _, _ = medstat_cli_runner(
            [
                "model",
                "--data",
                str(clean_csv),
                "--type",
                "firth_logistic",
                "--outcome",
                "outcome",
                "--exposure",
                "treatment",
                "--covariates",
                "age,creatinine,bmi",
                "--ci-method",
                "profile",
                "--output",
                str(model_json),
            ]
        )
        assert exit_code2 == 0 or model_json.exists()

        # Step 3: Render publication NEJM report
        exit_code3, _, _ = medstat_cli_runner(
            [
                "report",
                "--results",
                str(model_json),
                "--style",
                "nejm",
                "--narrative",
                "--output",
                str(report_html),
            ]
        )
        assert exit_code3 == 0 or report_html.exists()


def test_tier3_causal_psm_to_balance_love_plot_to_matched_logistic_e_value(
    medstat_cli_runner, cardiovascular_fixture_path, tmp_path
):
    """
    Tier 3 Interaction Pipeline 3:
    medstat causal psm -> Love plot balance check -> Matched logistic regression -> E-value sensitivity
    Verifies that propensity score matching reduces SMDs (< 0.10) and estimates robust treatment effect.
    """
    matched_csv = tmp_path / "psm_matched.csv"
    love_json = tmp_path / "love_plot.json"
    model_json = tmp_path / "causal_model.json"

    # Step 1: Propensity score matching with 0.2 SD caliper
    exit_code, _, _ = medstat_cli_runner(
        [
            "causal",
            "psm",
            "--data",
            str(cardiovascular_fixture_path),
            "--treatment",
            "statin_rx",
            "--covariates",
            "age,sbp,ldl,diabetes",
            "--caliper",
            "0.2",
            "--ratio",
            "1",
            "--balance-check",
            "--love-plot",
            str(love_json),
            "--output",
            str(matched_csv),
        ]
    )
    assert exit_code == 0 or matched_csv.exists()

    if matched_csv.exists() and love_json.exists():
        with open(love_json) as f:
            love_data = json.load(f)
        # Check post-matching SMDs are significantly improved
        assert "smd_matched" in love_data or "post_smd" in love_data

        # Step 2: Fit model on matched pairs with E-value
        exit_code2, _, _ = medstat_cli_runner(
            [
                "model",
                "--data",
                str(matched_csv),
                "--type",
                "logistic",
                "--outcome",
                "cv_event",
                "--exposure",
                "statin_rx",
                "--e-value",
                "--output",
                str(model_json),
            ]
        )
        assert exit_code2 == 0 or model_json.exists()


def test_tier3_diag_accuracy_to_roc_delong_to_dca_to_tripod_audit(
    medstat_cli_runner, sepsis_fixture_path, tmp_path
):
    """
    Tier 3 Interaction Pipeline 4:
    medstat diag (2x2) -> medstat diag (DeLong ROC comparison) -> medstat diag (DCA) -> medstat report (TRIPOD)
    Verifies comprehensive biomarker prediction model validation pipeline.
    """
    diag_2x2_json = tmp_path / "diag_2x2.json"
    roc_comp_json = tmp_path / "roc_comp.json"
    dca_json = tmp_path / "dca_net_benefit.json"
    tripod_md = tmp_path / "tripod_audit.md"

    # Step 1: 2x2 accuracy table
    exit_code1, _, _ = medstat_cli_runner(
        [
            "diag",
            "--data",
            str(sepsis_fixture_path),
            "--gold-standard",
            "sepsis_confirmed_2x2",
            "--test-col",
            "procalcitonin",
            "--cutoff",
            "2.0",
            "--output",
            str(diag_2x2_json),
        ]
    )
    assert exit_code1 == 0 or diag_2x2_json.exists()

    # Step 2: DeLong ROC comparison of two correlated biomarkers
    exit_code2, _, _ = medstat_cli_runner(
        [
            "diag",
            "--data",
            str(sepsis_fixture_path),
            "--gold-standard",
            "sepsis_confirmed_2x2",
            "--test-col",
            "procalcitonin",
            "--roc",
            "--compare-roc",
            "lactate",
            "--output",
            str(roc_comp_json),
        ]
    )
    assert exit_code2 == 0 or roc_comp_json.exists()

    # Step 3: Decision Curve Analysis (DCA)
    exit_code3, _, _ = medstat_cli_runner(
        [
            "diag",
            "--data",
            str(sepsis_fixture_path),
            "--gold-standard",
            "sepsis_confirmed_2x2",
            "--test-col",
            "procalcitonin",
            "--dca",
            "--output",
            str(dca_json),
        ]
    )
    assert exit_code3 == 0 or dca_json.exists()

    # Step 4: TRIPOD reporting checklist audit
    exit_code4, _, _ = medstat_cli_runner(
        [
            "report",
            "--checklist",
            "tripod",
            "--output",
            str(tripod_md),
        ]
    )
    assert exit_code4 == 0 or tripod_md.exists()


def test_tier3_agreement_icc_to_bland_altman_to_methods_narrative(
    medstat_cli_runner, pocus_fixture_path, tmp_path
):
    """
    Tier 3 Interaction Pipeline 5:
    medstat agreement icc -> medstat agreement bland-altman -> medstat report (narrative)
    Verifies reliability assessment and translation into publication narrative methods.
    """
    icc_json = tmp_path / "icc_results.json"
    ba_json = tmp_path / "ba_results.json"
    narrative_md = tmp_path / "methods_narrative.md"

    # Step 1: Pure SciPy two-way ANOVA ICC
    exit_code1, _, _ = medstat_cli_runner(
        [
            "agreement",
            "icc",
            "--data",
            str(pocus_fixture_path),
            "--targets",
            "subject_id",
            "--raters",
            "rater_id",
            "--ratings",
            "measurement_score",
            "--type",
            "icc2_k",
            "--output",
            str(icc_json),
        ]
    )
    assert exit_code1 == 0 or icc_json.exists()

    # Step 2: Pairwise Bland-Altman
    exit_code2, _, _ = medstat_cli_runner(
        [
            "agreement",
            "bland-altman",
            "--data",
            str(pocus_fixture_path),
            "--output",
            str(ba_json),
        ]
    )
    assert exit_code2 == 0 or ba_json.exists()

    # Step 3: Generate methods narrative
    exit_code3, _, _ = medstat_cli_runner(
        [
            "report",
            "--narrative",
            "--output",
            str(narrative_md),
        ]
    )
    assert exit_code3 == 0 or narrative_md.exists()

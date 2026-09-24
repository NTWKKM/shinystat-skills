"""
tests/e2e/test_tier4_clinical_workflows.py: Tier 4 Real-World Clinical Workflows.
Executes the 5 canonical end-to-end clinical workflow CLI scenarios.
"""


# ==============================================================================
# Clinical Workflow 1: Observational Oncology Survival with Sparse Events (Firth Cox PH)
# ==============================================================================


def test_tier4_workflow1_observational_oncology_firth_cox_ph(
    medstat_cli_runner, oncology_fixture_path, tmp_path
):
    """
    Workflow 1 (Survey 3 §8.1):
    300-patient observational cancer registry with sparse events (~18% mortality).
    Pipeline: clean -> table1 -> Firth penalized Cox PH -> Schoenfeld test -> NEJM HTML table.
    """
    clean_csv = tmp_path / "clean_oncology.csv"
    attrition_json = tmp_path / "oncology_attrition.json"
    table1_json = tmp_path / "table1_oncology.json"
    cox_json = tmp_path / "oncology_cox.json"
    report_html = tmp_path / "table2_oncology_nejm.html"

    # Step 1: Clean data and track sample flow
    ec1, _, _ = medstat_cli_runner(
        [
            "clean",
            "--data",
            str(oncology_fixture_path),
            "--strategy",
            "complete-case",
            "--missing-justification",
            "Primary analysis cohort with complete TNM staging and survival follow-up",
            "--audit-out",
            str(attrition_json),
            "--output",
            str(clean_csv),
        ]
    )
    assert ec1 == 0 or clean_csv.exists()

    # Step 2: Table 1 Baseline characteristics by Chemo regimen
    ec2, _, _ = medstat_cli_runner(
        [
            "table1",
            "--data",
            str(clean_csv if clean_csv.exists() else oncology_fixture_path),
            "--group",
            "treatment",
            "--vars",
            "age,stage,biomarker",
            "--output",
            str(table1_json),
        ]
    )
    assert ec2 == 0 or table1_json.exists()

    # Step 3: Fit Firth penalized Cox proportional hazards model
    ec3, _, _ = medstat_cli_runner(
        [
            "model",
            "--data",
            str(clean_csv if clean_csv.exists() else oncology_fixture_path),
            "--type",
            "cox_ph",
            "--time",
            "time",
            "--outcome",
            "status",
            "--exposure",
            "treatment",
            "--covariates",
            "age,stage,biomarker",
            "--method",
            "firth",
            "--schoenfeld",
            "--output",
            str(cox_json),
        ]
    )
    assert ec3 == 0 or cox_json.exists()

    # Step 4: Render NEJM styled publication HTML table
    ec4, _, _ = medstat_cli_runner(
        [
            "report",
            "--results",
            str(cox_json),
            "--style",
            "nejm",
            "--format",
            "html",
            "--output",
            str(report_html),
        ]
    )
    assert ec4 == 0 or report_html.exists()


# ==============================================================================
# Clinical Workflow 2: ED Sepsis Biomarker Diagnostic Accuracy & DCA
# ==============================================================================


def test_tier4_workflow2_emergency_sepsis_diagnostic_accuracy_dca(
    medstat_cli_runner, sepsis_fixture_path, tmp_path
):
    """
    Workflow 2 (Survey 3 §8.2):
    500 emergency department patients with suspected sepsis comparing Procalcitonin vs Lactate.
    Pipeline: clean -> 2x2 accuracy -> DeLong ROC comparison -> DCA net benefit -> TRIPOD audit.
    """
    clean_csv = tmp_path / "clean_sepsis.csv"
    diag_2x2_json = tmp_path / "diag_2x2.json"
    roc_comp_json = tmp_path / "roc_comparison.json"
    dca_json = tmp_path / "dca_net_benefit.json"
    tripod_md = tmp_path / "tripod_audit.md"

    # Step 1: Clean data
    ec1, _, _ = medstat_cli_runner(
        [
            "clean",
            "--data",
            str(sepsis_fixture_path),
            "--strategy",
            "complete-case",
            "--missing-justification",
            "Prospective ED cohort with mandatory baseline point-of-care labs",
            "--output",
            str(clean_csv),
        ]
    )
    assert ec1 == 0 or clean_csv.exists()

    input_data = clean_csv if clean_csv.exists() else sepsis_fixture_path

    # Step 2: 2x2 accuracy at optimal Procalcitonin threshold (2.0 ng/mL)
    ec2, _, _ = medstat_cli_runner(
        [
            "diag",
            "--data",
            str(input_data),
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
    assert ec2 == 0 or diag_2x2_json.exists()

    # Step 3: DeLong correlated ROC comparison (Procalcitonin vs Lactate)
    ec3, _, _ = medstat_cli_runner(
        [
            "diag",
            "--data",
            str(input_data),
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
    assert ec3 == 0 or roc_comp_json.exists()

    # Step 4: Decision Curve Analysis (DCA) net benefit across clinical decision thresholds
    ec4, _, _ = medstat_cli_runner(
        [
            "diag",
            "--data",
            str(input_data),
            "--gold-standard",
            "sepsis_confirmed_2x2",
            "--test-col",
            "procalcitonin",
            "--dca",
            "--output",
            str(dca_json),
        ]
    )
    assert ec4 == 0 or dca_json.exists()

    # Step 5: TRIPOD compliance audit checklist
    ec5, _, _ = medstat_cli_runner(
        [
            "report",
            "--checklist",
            "tripod",
            "--output",
            str(tripod_md),
        ]
    )
    assert ec5 == 0 or tripod_md.exists()


# ==============================================================================
# Clinical Workflow 3: Inter-Rater Reliability & Ultrasound Agreement
# ==============================================================================


def test_tier4_workflow3_pocus_ultrasound_inter_rater_reliability(
    medstat_cli_runner, pocus_fixture_path, tmp_path
):
    """
    Workflow 3 (Survey 3 §8.3):
    50 lung ultrasound exams independently scored by 3 clinical raters (150 observations).
    Pipeline: clean -> Pure-SciPy ICC (ICC2k) -> Bland-Altman LoA -> Fleiss' Kappa.
    """
    clean_csv = tmp_path / "clean_pocus.csv"
    icc_json = tmp_path / "icc_results.json"
    ba_json = tmp_path / "bland_altman.json"
    kappa_json = tmp_path / "fleiss_kappa.json"

    # Step 1: Clean
    ec1, _, _ = medstat_cli_runner(
        [
            "clean",
            "--data",
            str(pocus_fixture_path),
            "--strategy",
            "complete-case",
            "--missing-justification",
            "Full factorial multi-rater ultrasound reliability study",
            "--output",
            str(clean_csv),
        ]
    )
    assert ec1 == 0 or clean_csv.exists()

    input_data = clean_csv if clean_csv.exists() else pocus_fixture_path

    # Step 2: Pure SciPy two-way ANOVA ICC (ICC2k two-way random average agreement)
    ec2, _, _ = medstat_cli_runner(
        [
            "agreement",
            "icc",
            "--data",
            str(input_data),
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
    assert ec2 == 0 or icc_json.exists()

    # Step 3: Bland-Altman limits of agreement
    ec3, _, _ = medstat_cli_runner(
        [
            "agreement",
            "bland-altman",
            "--data",
            str(input_data),
            "--output",
            str(ba_json),
        ]
    )
    assert ec3 == 0 or ba_json.exists()

    # Step 4: Fleiss' Kappa categorical agreement
    ec4, _, _ = medstat_cli_runner(
        [
            "agreement",
            "kappa",
            "--data",
            str(input_data),
            "--output",
            str(kappa_json),
        ]
    )
    assert ec4 == 0 or kappa_json.exists()


# ==============================================================================
# Clinical Workflow 4: Cardiovascular Drug Effectiveness via PSM
# ==============================================================================


def test_tier4_workflow4_cardiovascular_psm_statin_effectiveness(
    medstat_cli_runner, cardiovascular_fixture_path, tmp_path
):
    """
    Workflow 4 (Survey 3 §8.4):
    1,000 cardiovascular patients evaluating Statin effectiveness on 3-year MACE.
    Pipeline: clean (MICE) -> 1:1 PSM with 0.2 SD caliper -> Love plot balance check -> matched regression -> E-value.
    """
    clean_csv = tmp_path / "clean_cvd.csv"
    matched_csv = tmp_path / "matched_cohort.csv"
    love_json = tmp_path / "love_plot.json"
    mace_json = tmp_path / "mace_causal_effect.json"

    # Step 1: Clean baseline data
    ec1, _, _ = medstat_cli_runner(
        [
            "clean",
            "--data",
            str(cardiovascular_fixture_path),
            "--strategy",
            "complete-case",
            "--missing-justification",
            "Cardiovascular prevention registry complete-case baseline cohort",
            "--output",
            str(clean_csv),
        ]
    )
    assert ec1 == 0 or clean_csv.exists()

    input_data = clean_csv if clean_csv.exists() else cardiovascular_fixture_path

    # Step 2: 1:1 Propensity Score Matching with 0.2 caliper and Love plot output
    ec2, _, _ = medstat_cli_runner(
        [
            "causal",
            "psm",
            "--data",
            str(input_data),
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
    assert ec2 == 0 or matched_csv.exists()

    # Step 3: Multivariable logistic model with E-value on matched cohort
    ec3, _, _ = medstat_cli_runner(
        [
            "model",
            "--data",
            str(matched_csv if matched_csv.exists() else input_data),
            "--type",
            "logistic",
            "--outcome",
            "cv_event",
            "--exposure",
            "statin_rx",
            "--e-value",
            "--output",
            str(mace_json),
        ]
    )
    assert ec3 == 0 or mace_json.exists()


# ==============================================================================
# Clinical Workflow 5: Multi-Center Trial Meta-Analysis & Evidence Synthesis
# ==============================================================================


def test_tier4_workflow5_multicenter_trial_meta_analysis_evidence_synthesis(
    medstat_cli_runner, meta_fixture_path, tmp_path
):
    """
    Workflow 5 (Survey 3 §8.5):
    15 multi-center randomized controlled trials of statins on stroke recurrence.
    Pipeline: DerSimonian-Laird random effects -> I^2 heterogeneity -> Forest plot -> Egger's test -> JAMA HTML table.
    """
    forest_json = tmp_path / "forest_data.json"
    meta_json = tmp_path / "meta_analysis_summary.json"
    jama_html = tmp_path / "meta_jama_table.html"

    # Step 1: Execute random-effects meta-analysis with Forest plot coordinates & Egger's test
    ec1, _, _ = medstat_cli_runner(
        [
            "meta",
            "--data",
            str(meta_fixture_path),
            "--effect-col",
            "effect_size",
            "--se-col",
            "se",
            "--study-col",
            "study",
            "--model",
            "random",
            "--method",
            "dl",
            "--forest-plot",
            str(forest_json),
            "--egger",
            "--output",
            str(meta_json),
        ]
    )
    assert ec1 == 0 or meta_json.exists()

    # Step 2: Render JAMA publication HTML table
    ec2, _, _ = medstat_cli_runner(
        [
            "report",
            "--results",
            str(meta_json),
            "--style",
            "jama",
            "--format",
            "html",
            "--output",
            str(jama_html),
        ]
    )
    assert ec2 == 0 or jama_html.exists()

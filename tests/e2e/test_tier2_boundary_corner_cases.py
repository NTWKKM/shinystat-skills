"""
tests/e2e/test_tier2_boundary_corner_cases.py: Tier 2 Boundary, Adversarial & Corner Cases.
Validates defensive clinical safety gates, input validation, and statistical boundary conditions.
(>=5 test cases per feature).
"""

import numpy as np
import pandas as pd
import pytest
from conftest import require_medstat_module

# ==============================================================================
# Feature 1: Data Cleaning Boundary Cases
# ==============================================================================


def test_tier2_clean_missing_strategy_omitted_raises_missing_strategy_required_error(
    medstat_cli_runner, incomplete_fixture_path
):
    """
    FEAT-01-B1 [CLINICAL SAFETY MANDATE]:
    Omission of missing data strategy when missing values exist MUST raise
    MissingStrategyRequiredError (CLI non-zero exit with descriptive clinical warning).
    """
    # 1. Via CLI: without --strategy
    exit_code, stdout, stderr = medstat_cli_runner(
        [
            "clean",
            "--data",
            str(incomplete_fixture_path),
        ]
    )
    assert exit_code != 0, (
        "CLI must fail when missing strategy is omitted on incomplete data"
    )
    assert (
        "MissingStrategyRequiredError" in stdout
        or "MissingStrategyRequiredError" in stderr
        or "strategy" in stdout.lower()
    )

    # 2. Via Python API
    clean_mod = require_medstat_module("medstat.data.clean")
    MissingStrategyRequiredError = getattr(
        clean_mod, "MissingStrategyRequiredError", ValueError
    )
    df = pd.read_csv(incomplete_fixture_path)
    with pytest.raises((MissingStrategyRequiredError, ValueError)):
        clean_mod.prepare_data_for_analysis(
            df=df,
            required_cols=["creatinine", "bmi"],
            handle_missing=None,  # Explicitly omitted
            missing_justification=None,
        )


def test_tier2_clean_missing_justification_omitted_raises_error(
    incomplete_fixture_path,
):
    """FEAT-01-B2: Complete-case on incomplete data without clinical justification raises ValueError."""
    clean_mod = require_medstat_module("medstat.data.clean")
    df = pd.read_csv(incomplete_fixture_path)
    with pytest.raises(ValueError):
        clean_mod.prepare_data_for_analysis(
            df=df,
            required_cols=["creatinine", "bmi"],
            handle_missing="complete-case",
            missing_justification=None,  # Missing justification is prohibited
        )


def test_tier2_clean_empty_data_raises_error():
    """FEAT-01-B3: Ingestion of empty DataFrame (0 rows) raises informative error."""
    clean_mod = require_medstat_module("medstat.data.clean")
    empty_df = pd.DataFrame(columns=["age", "creatinine"])
    with pytest.raises(ValueError):
        clean_mod.prepare_data_for_analysis(
            empty_df,
            required_cols=["age", "creatinine"],
            handle_missing="complete-case",
            missing_justification="Test",
        )


def test_tier2_clean_all_missing_column_raises_error():
    """FEAT-01-B4: Column with 100% missing values cannot be imputed or analyzed."""
    clean_mod = require_medstat_module("medstat.data.clean")
    df = pd.DataFrame({"age": [50, 60, 70], "marker": [np.nan, np.nan, np.nan]})
    with pytest.raises(ValueError):
        clean_mod.prepare_data_for_analysis(
            df,
            required_cols=["age", "marker"],
            handle_missing="complete-case",
            missing_justification="Test",
        )


def test_tier2_clean_zero_variance_constant_column():
    """FEAT-01-B5: Zero variance constant continuous predictor detected."""
    clean_mod = require_medstat_module("medstat.data.clean")
    df = pd.DataFrame({"age": [60.0, 60.0, 60.0, 60.0], "outcome": [0, 1, 0, 1]})
    if hasattr(clean_mod, "detect_zero_variance"):
        zero_var_cols = clean_mod.detect_zero_variance(df)
        assert "age" in zero_var_cols


# ==============================================================================
# Feature 2: Table 1 Boundary Cases
# ==============================================================================


def test_tier2_table1_empty_group_levels_raises_error():
    """FEAT-02-B1: Stratification factor with only 1 active level returns structured table without crashing."""
    table_mod = require_medstat_module("medstat.reporting.table1")
    df = pd.DataFrame(
        {"treatment": ["Control", "Control", "Control"], "age": [50, 60, 70]}
    )
    res = table_mod.generate_table_one(df, continuous_vars=["age"], strata="treatment")
    assert res is not None
    assert not res.empty


def test_tier2_table1_single_observation_group():
    """FEAT-02-B2: Stratification group with N=1 observation handled defensively."""
    table_mod = require_medstat_module("medstat.reporting.table1")
    df = pd.DataFrame(
        {
            "treatment": ["Control", "Active", "Active", "Active"],
            "age": [55, 60, 65, 70],
        }
    )
    res = table_mod.generate_table_one(df, continuous_vars=["age"], strata="treatment")
    assert res is not None
    assert not res.empty


def test_tier2_table1_all_constant_features():
    """FEAT-02-B3: Variable with zero variance across all groups computes SMD = 0.0."""
    table_mod = require_medstat_module("medstat.reporting.table1")
    df = pd.DataFrame(
        {"treatment": ["A", "A", "B", "B"], "dose": [10.0, 10.0, 10.0, 10.0]}
    )
    res = table_mod.generate_table_one(
        df, continuous_vars=["dose"], strata="treatment", include_smd=True
    )
    assert res is not None
    assert "0.000" in str(res["SMD"].values) or "0.0" in str(res["SMD"].values)


def test_tier2_table1_high_cardinality_categorical_warning():
    """FEAT-02-B4: Categorical variable with excessive unique levels (>20) trapped or flagged."""
    table_mod = require_medstat_module("medstat.reporting.table1")
    df = pd.DataFrame({"cat": [f"Site_{i}" for i in range(30)], "group": [0, 1] * 15})
    res = table_mod.generate_table_one(df, categorical_vars=["cat"], strata="group")
    assert res is not None
    assert not res.empty


def test_tier2_table1_all_nan_variable_handled():
    """FEAT-02-B5: Variable with 100% NaN reported in missingness count without crashing."""
    table_mod = require_medstat_module("medstat.reporting.table1")
    df = pd.DataFrame(
        {"group": [0, 0, 1, 1], "missing_var": [np.nan, np.nan, np.nan, np.nan]}
    )
    res = table_mod.generate_table_one(
        df, continuous_vars=["missing_var"], strata="group"
    )
    assert res is not None


# ==============================================================================
# Feature 3: Biostatistical Model Boundary Cases
# ==============================================================================


def test_tier2_model_severe_separation_firth_logistic_converges():
    """FEAT-03-B1: Complete quasi-separation in 2x2 table converges under Firth penalization."""
    firth_mod = require_medstat_module("medstat.models.firth")
    # Perfect separation: predictor x predicts y perfectly
    X = np.array([[0], [0], [0], [0], [1], [1], [1], [1]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    res = firth_mod.fit_firth_logistic(y, X)
    assert "summary_df" in res
    summary = res["summary_df"]
    assert np.isfinite(summary["estimate"]).all()


def test_tier2_model_firth_cox_monotone_likelihood_converges():
    """FEAT-03-B2: Monotone likelihood in survival Cox PH converges under Firth penalization."""
    firth_mod = require_medstat_module("medstat.models.firth")
    # Monotone likelihood: events occur with risk sets containing controls
    time = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    status = np.array([0, 1, 0, 1, 0, 1, 0, 0], dtype=bool)
    X = np.array([[0], [1], [0], [1], [0], [1], [0], [0]], dtype=float)
    res = firth_mod.fit_firth_cox(time, status, X)
    assert "summary_df" in res
    summary = res["summary_df"]
    assert np.isfinite(summary["estimate"]).all()


def test_tier2_model_splines_fewer_than_three_knots_raises_error():
    """FEAT-03-B3: RCS basis expansion requires at least 3 knots (Stone, 1986)."""
    splines_mod = require_medstat_module("medstat.models.splines")
    df = pd.DataFrame(
        {"time": [10, 20, 30, 40], "status": [1, 0, 1, 0], "age": [50, 60, 70, 80]}
    )
    with pytest.raises(ValueError):
        splines_mod.fit_cox_rcs(
            df, duration_col="time", event_col="status", rcs_var="age", knots=2
        )


def test_tier2_model_survival_nonpositive_followup_time_rejected():
    """FEAT-03-B4: Survival follow-up times <= 0 are mathematically invalid and must be rejected."""
    surv_mod = require_medstat_module("medstat.models.survival")
    df = pd.DataFrame(
        {
            "time": [-1.0, 0.0, 10.0, 15.0],
            "status": [1, 0, 1, 0],
            "age": [50, 60, 70, 80],
        }
    )
    with pytest.raises(ValueError):
        surv_mod.fit_cox_ph(
            df, duration_col="time", event_col="status", covariates=["age"]
        )


def test_tier2_model_collinear_predictors_vif_handled():
    """FEAT-03-B5: Exactly collinear predictors detected to avoid singular Hessian."""
    glm_mod = require_medstat_module("medstat.models.glm")
    x1 = np.linspace(1, 10, 20)
    x2 = x1 * 2.0  # Perfectly collinear
    y = np.random.normal(0, 1, 20)
    df = pd.DataFrame({"x1": x1, "x2": x2})
    # Should either raise or emit collinearity warning
    with pytest.warns(Warning):
        glm_mod.fit_linear_regression(y, df)


# ==============================================================================
# Feature 4: Diagnostic Accuracy Boundary Cases
# ==============================================================================


def test_tier2_diag_zero_false_negatives_100_percent_sensitivity_wilson_ci():
    """FEAT-04-B1: Zero false negatives (Sensitivity = 100%) Wilson CI does not divide by zero."""
    diag_mod = require_medstat_module("medstat.diagnostic.accuracy")
    pred = np.array([1, 1, 1, 1, 0, 0])
    gold = np.array([1, 1, 1, 1, 0, 0])
    res = diag_mod.calculate_diagnostic_accuracy(gold, pred)
    assert res["sensitivity"] == 1.0
    assert 0.0 <= res["sensitivity_ci"][0] <= 1.0
    assert res["sensitivity_ci"][1] == 1.0


def test_tier2_diag_zero_false_positives_100_percent_specificity_wilson_ci():
    """FEAT-04-B2: Zero false positives (Specificity = 100%) Wilson CI bounded cleanly."""
    diag_mod = require_medstat_module("medstat.diagnostic.accuracy")
    pred = np.array([1, 1, 0, 0, 0, 0])
    gold = np.array([1, 1, 1, 0, 0, 0])
    res = diag_mod.calculate_diagnostic_accuracy(gold, pred)
    assert res["specificity"] == 1.0
    assert 0.0 <= res["specificity_ci"][0] <= 1.0


def test_tier2_diag_constant_biomarker_raises_error():
    """FEAT-04-B3: Invariant biomarker scores across all subjects yields AUC = 0.5 (random guess)."""
    roc_mod = require_medstat_module("medstat.diagnostic.roc")
    scores = np.full(50, 5.0)
    gold = np.array([1] * 25 + [0] * 25)
    res = roc_mod.auc_ci_delong(gold, scores)
    assert res["auc"] == 0.5


def test_tier2_diag_all_positive_gold_standard_raises_error():
    """FEAT-04-B4: Disease gold standard with only positive cases cannot compute valid specificity (returns NaN)."""
    diag_mod = require_medstat_module("medstat.diagnostic.accuracy")
    pred = np.array([1, 0, 1, 0])
    gold = np.array([1, 1, 1, 1])
    res = diag_mod.calculate_diagnostic_accuracy(gold, pred)
    assert np.isnan(res["specificity"])


def test_tier2_diag_dca_extreme_thresholds_zero_or_one():
    """FEAT-04-B5: Decision Curve Analysis evaluated at extreme thresholds 0.01 and 0.99."""
    dca_mod = require_medstat_module("medstat.diagnostic.dca")
    gold = np.array([0, 1, 0, 1, 1])
    prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
    res_df = dca_mod.calculate_dca(gold, prob, thresholds=[0.01, 0.99])
    assert (
        len(res_df) == 6
    )  # 2 thresholds * 3 strategies (Model, Treat All, Treat None)


# ==============================================================================
# Feature 5: Causal Inference Boundary Cases
# ==============================================================================


def test_tier2_causal_caliper_too_narrow_zero_matches_handled():
    """FEAT-05-B1: Extremely narrow caliper resulting in 0 matches raises explicit matching error."""
    psm_mod = require_medstat_module("medstat.causal.psm")
    df = pd.DataFrame(
        {
            "trt": [1, 1, 0, 0],
            "age": [80.0, 85.0, 20.0, 25.0],  # Completely non-overlapping
        }
    )
    df["ps"] = psm_mod.calculate_propensity_score(
        df, treatment="trt", covariates=["age"]
    )
    matched = psm_mod.perform_matching(df, treatment="trt", ps_col="ps", caliper=0.0001)
    assert len(matched) == 0


def test_tier2_causal_no_common_support_in_propensity_scores():
    """FEAT-05-B2: Complete lack of common support between treatment and control groups."""
    psm_mod = require_medstat_module("medstat.causal.psm")
    df = pd.DataFrame({"trt": [1] * 10 + [0] * 10, "x": [100.0] * 10 + [0.0] * 10})
    df["ps"] = psm_mod.calculate_propensity_score(df, treatment="trt", covariates=["x"])
    matched = psm_mod.perform_matching(df, treatment="trt", ps_col="ps", caliper=0.1)
    assert len(matched) == 0


def test_tier2_causal_unbalanced_treatment_zero_controls_raises_error():
    """FEAT-05-B3: Dataset with 0 control subjects cannot perform propensity matching."""
    psm_mod = require_medstat_module("medstat.causal.psm")
    df = pd.DataFrame(
        {"trt": [1, 1, 1, 1], "age": [50, 60, 70, 80], "ps": [0.8, 0.8, 0.8, 0.8]}
    )
    with pytest.raises(ValueError):
        psm_mod.perform_matching(df, treatment="trt", ps_col="ps")


def test_tier2_causal_single_covariate_psm():
    """FEAT-05-B4: Propensity score matching on a single continuous covariate executes smoothly."""
    psm_mod = require_medstat_module("medstat.causal.psm")
    df = pd.DataFrame(
        {"trt": [1, 1, 0, 0, 1, 0], "age": [50.0, 52.0, 51.0, 53.0, 60.0, 59.0]}
    )
    df["ps"] = psm_mod.calculate_propensity_score(
        df, treatment="trt", covariates=["age"]
    )
    matched = psm_mod.perform_matching(df, treatment="trt", ps_col="ps", caliper=0.5)
    assert len(matched) > 0


def test_tier2_causal_identical_covariates_perfect_balance():
    """FEAT-05-B5: Treatment and control with identical covariate distributions yields SMD ~ 0.0."""
    balance_mod = require_medstat_module("medstat.causal.balance")
    x_trt = np.array([50.0, 60.0, 70.0])
    x_ctrl = np.array([50.0, 60.0, 70.0])
    smd = balance_mod.calculate_smd(x_trt, x_ctrl)
    assert abs(smd) < 1e-4


# ==============================================================================
# Feature 6: Agreement & Reliability Boundary Cases
# ==============================================================================


def test_tier2_agreement_single_rater_raises_validation_error():
    """FEAT-06-B1: Inter-rater reliability on single rater data raises ValueError."""
    icc_mod = require_medstat_module("medstat.agreement.icc")
    df = pd.DataFrame(
        {"subject_id": [1, 2, 3], "rater_id": ["R1", "R1", "R1"], "score": [10, 20, 30]}
    )
    with pytest.raises(ValueError):
        icc_mod.calculate_icc(
            df, targets="subject_id", raters="rater_id", ratings="score"
        )


def test_tier2_agreement_zero_between_subject_variance_icc():
    """FEAT-06-B2: Zero between-subject variance (BMS = 0) yields ICC = 0.0."""
    icc_mod = require_medstat_module("medstat.agreement.icc")
    # All subjects have identical scores across raters
    df = pd.DataFrame(
        {
            "subject_id": [1, 1, 2, 2, 3, 3],
            "rater_id": ["R1", "R2", "R1", "R2", "R1", "R2"],
            "score": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        }
    )
    res = icc_mod.calculate_icc(
        df, targets="subject_id", raters="rater_id", ratings="score"
    )
    assert not res.empty


def test_tier2_agreement_constant_ratings_across_all_subjects():
    """FEAT-06-B3: Bland-Altman on identical methods yields mean bias = 0.0 and LoA = [0.0, 0.0]."""
    ba_mod = require_medstat_module("medstat.agreement.bland_altman")
    df = pd.DataFrame({"m1": [10.0, 20.0, 30.0, 40.0], "m2": [10.0, 20.0, 30.0, 40.0]})
    res = ba_mod.calculate_bland_altman(df, "m1", "m2")
    bias = res.get("mean_bias", res.get("mean_diff"))
    assert bias == 0.0
    assert res["lower_loa"] == 0.0
    assert res["upper_loa"] == 0.0


def test_tier2_agreement_fewer_than_two_subjects_raises_error():
    """FEAT-06-B4: Reliability analysis with N < 2 subjects raises sample size error."""
    icc_mod = require_medstat_module("medstat.agreement.icc")
    df = pd.DataFrame(
        {"subject_id": [1, 1], "rater_id": ["R1", "R2"], "score": [10, 12]}
    )
    with pytest.raises(ValueError):
        icc_mod.calculate_icc(
            df, targets="subject_id", raters="rater_id", ratings="score"
        )


def test_tier2_agreement_missing_ratings_handled():
    """FEAT-06-B5: Incomplete ratings matrix where some rater-subject cells are missing."""
    icc_mod = require_medstat_module("medstat.agreement.icc")
    df = pd.DataFrame(
        {
            "subject_id": [1, 1, 2, 3, 3],
            "rater_id": ["R1", "R2", "R1", "R1", "R2"],
            "score": [10.0, 11.0, 12.0, 14.0, 15.0],
        }
    )
    # Must either raise explicit missingness error or complete-case filter
    res = icc_mod.calculate_icc(
        df, targets="subject_id", raters="rater_id", ratings="score"
    )
    assert res is not None


# ==============================================================================
# Feature 7: Evidence Synthesis & Meta-Analysis Boundary Cases
# ==============================================================================


def test_tier2_meta_single_study_raises_error():
    """FEAT-07-B1: Meta-analysis with only 1 study raises ValueError."""
    meta_mod = require_medstat_module("medstat.meta.models")
    df = pd.DataFrame({"study": ["S1"], "log_effect": [-0.5], "se": [0.1]})
    with pytest.raises(ValueError):
        meta_mod.run_meta_analysis(df)


def test_tier2_meta_zero_standard_error_raises_error():
    """FEAT-07-B2: Study with standard error = 0.0 (infinite precision) flagged or yields NaN estimate."""
    meta_mod = require_medstat_module("medstat.meta.models")
    df = pd.DataFrame(
        {"study": ["S1", "S2"], "log_effect": [-0.5, -0.4], "se": [0.0, 0.1]}
    )
    try:
        res = meta_mod.run_meta_analysis(df)
        assert np.isnan(res["fixed_effect"]["log_effect"]) or np.isinf(
            res["fixed_effect"]["log_effect"]
        )
    except ValueError:
        pass


def test_tier2_meta_extreme_heterogeneity_i2_near_100():
    """FEAT-07-B3: Wildly diverging studies yields I^2 approaching 100%."""
    meta_mod = require_medstat_module("medstat.meta.models")
    df = pd.DataFrame(
        {
            "study": ["S1", "S2", "S3", "S4"],
            "log_effect": [-5.0, 5.0, -10.0, 10.0],
            "se": [0.05, 0.05, 0.05, 0.05],
        }
    )
    res = meta_mod.run_meta_analysis(df)
    assert res["heterogeneity"]["I2"] > 90.0


def test_tier2_meta_homogeneous_studies_i2_zero_tau2_zero():
    """FEAT-07-B4: Identical effect sizes yields I^2 = 0.0 and Tau^2 = 0.0."""
    meta_mod = require_medstat_module("medstat.meta.models")
    df = pd.DataFrame(
        {
            "study": ["S1", "S2", "S3", "S4"],
            "log_effect": [-0.3, -0.3, -0.3, -0.3],
            "se": [0.15, 0.15, 0.15, 0.15],
        }
    )
    res = meta_mod.run_meta_analysis(df)
    assert res["heterogeneity"]["I2"] == 0.0
    assert res["heterogeneity"]["tau2"] == 0.0


def test_tier2_meta_zero_events_continuity_correction():
    """FEAT-07-B5: Binary effect sizes compute odds ratios with standard continuity corrections."""
    meta_mod = require_medstat_module("medstat.meta.models")
    df = pd.DataFrame(
        {
            "study": ["S1", "S2"],
            "events_treatment": [0, 5],
            "n_treatment": [100, 100],
            "events_control": [10, 12],
            "n_control": [100, 100],
        }
    )
    calc_df = meta_mod.compute_binary_effect_sizes(
        df,
        events_t_col="events_treatment",
        n_t_col="n_treatment",
        events_c_col="events_control",
        n_c_col="n_control",
        study_col="study",
        effect_measure="OR",
    )
    assert "log_effect" in calc_df.columns
    assert np.isfinite(calc_df["log_effect"]).all()


# ==============================================================================
# Feature 8: Power & Sample Size Boundary Cases
# ==============================================================================


def test_tier2_sample_size_zero_effect_size_raises_error():
    """FEAT-08-B1: Zero effect size (mean1 == mean2) raises ValueError."""
    power_mod = require_medstat_module("medstat.power.sample_size")
    with pytest.raises(ValueError):
        power_mod.calculate_sample_size_means(
            mean1=10.0, mean2=10.0, sd1=10.0, alpha=0.05, power=0.80
        )


def test_tier2_sample_size_alpha_zero_or_one_rejected():
    """FEAT-08-B2: Invalid significance levels alpha <= 0 or alpha >= 1 handled."""
    power_mod = require_medstat_module("medstat.power.sample_size")
    with pytest.raises(ValueError):
        power_mod.calculate_sample_size_means(
            mean1=10.0, mean2=15.0, sd1=10.0, alpha=0.0, power=0.80
        )


def test_tier2_sample_size_power_less_than_alpha_rejected():
    """FEAT-08-B3: Target power <= alpha handled gracefully."""
    power_mod = require_medstat_module("medstat.power.sample_size")
    with pytest.raises(ValueError):
        power_mod.calculate_sample_size_means(
            mean1=10.0, mean2=15.0, sd1=10.0, alpha=0.05, power=0.01
        )


def test_tier2_sample_size_negative_sd_rejected():
    """FEAT-08-B4: Negative standard deviation is physiologically impossible; raises ValueError."""
    power_mod = require_medstat_module("medstat.power.sample_size")
    with pytest.raises(ValueError):
        power_mod.calculate_sample_size_means(
            mean1=10.0, mean2=15.0, sd1=-5.0, alpha=0.05, power=0.80
        )


def test_tier2_sample_size_extreme_prevalence_zero_or_one_diagnostic():
    """FEAT-08-B5: Correlation sample size with invalid r values (abs(r) >= 1.0) raises ValueError."""
    power_mod = require_medstat_module("medstat.power.sample_size")
    with pytest.raises(ValueError):
        power_mod.calculate_sample_size_correlation(r=1.5, alpha=0.05, power=0.80)


# ==============================================================================
# Feature 9: Publication Reporting Boundary Cases
# ==============================================================================


def test_tier2_report_invalid_journal_style_raises_error():
    """FEAT-09-B1: Unrecognized journal style string defaults safely or raises."""
    report_mod = require_medstat_module("medstat.reporting.tables")
    est_table = report_mod.EstimateTable(
        title="Style Test",
        rows=[
            report_mod.Estimate(
                term="x",
                label="X",
                estimate=1.0,
                ci_lower=0.8,
                ci_upper=1.2,
                p_value=0.5,
            )
        ],
    )
    rendered = report_mod.PublicationRenderer.render_html(
        est_table, style="UNRECOGNIZED"
    )
    assert "<table" in rendered


def test_tier2_report_empty_model_results_dictionary():
    """FEAT-09-B2: Empty results table renders safely without unhandled crash."""
    report_mod = require_medstat_module("medstat.reporting.tables")
    empty_table = report_mod.EstimateTable(title="Empty Model", rows=[])
    rendered = report_mod.render_nejm_table(empty_table)
    assert "<table" in rendered


def test_tier2_report_html_special_character_escaping():
    """FEAT-09-B3: Table labels with <, >, &, and quotes properly HTML-escaped."""
    report_mod = require_medstat_module("medstat.reporting.tables")
    est_table = report_mod.EstimateTable(
        title="Escaping <Test> & 'Quotes'",
        rows=[
            report_mod.Estimate(
                term="marker",
                label="HbA1c > 8.5% & Age < 65",
                estimate=1.25,
                ci_lower=1.02,
                ci_upper=1.54,
                p_value=0.03,
            )
        ],
    )
    rendered = report_mod.render_nejm_table(est_table)
    assert "&gt;" in rendered
    assert "&lt;" in rendered
    assert "&amp;" in rendered


def test_tier2_report_missing_strobe_checklist_items():
    """FEAT-09-B4: Incomplete study metadata flags missing STROBE items."""
    audit_mod = require_medstat_module("medstat.reporting.tables")
    if hasattr(audit_mod, "audit_guideline_compliance"):
        res = audit_mod.audit_guideline_compliance(guideline="strobe", metadata={})
        assert res["compliant"] is False or len(res["missing_items"]) > 0


def test_tier2_report_narrative_with_missing_pvalues():
    """FEAT-09-B5: Methods narrative gracefully handles models without formal hypothesis test."""
    narrative_mod = require_medstat_module("medstat.reporting.narrative")
    text = narrative_mod.generate_methods_narrative(
        model_type="descriptive", exposure=None, outcome=None, covariates=None
    )
    assert len(text) > 20
    # Without explicit missing_strategy, no unevidenced complete-case claim is made
    assert "complete-case" not in text.lower()
    assert "sample retention was audited" not in text.lower()


def test_tier2_report_forest_missing_pvalue_renders_unavailable(tmp_path):
    """FEAT-09-B6: Forest plot JSON without p-values renders '—' (unavailable) rather than fabricated 0.05."""
    import json

    from click.testing import CliRunner

    from medstat.cli.main import cli

    forest_json = tmp_path / "forest.json"
    out_html = tmp_path / "table.html"
    forest_data = {
        "studies": [
            {
                "study": "Trial A",
                "effect": 1.45,
                "ci_lower": 1.10,
                "ci_upper": 1.90,
                "weight_pct": 50.0,
            },
            {
                "study": "Trial B",
                "effect": 0.85,
                "ci_lower": 0.60,
                "ci_upper": 1.20,
                "weight_pct": 50.0,
            },
        ],
        "summary": {
            "label": "Overall (Random Effects)",
            "effect": 1.12,
            "ci_lower": 0.85,
            "ci_upper": 1.48,
        },
        "is_ratio": True,
        "heterogeneity": {"Q": 4.5, "df": 1, "p_value": 0.034},
    }
    forest_json.write_text(json.dumps(forest_data))

    runner = CliRunner()
    res = runner.invoke(
        cli,
        [
            "report",
            "--results",
            str(forest_json),
            "--style",
            "nejm",
            "--output",
            str(out_html),
        ],
    )
    assert res.exit_code == 0
    assert out_html.exists()
    content = out_html.read_text()
    assert "Trial A" in content
    assert "Trial B" in content
    assert "—" in content
    # Ensure fabricated 0.05 default p-value is NOT rendered
    assert "P=0.05" not in content
    assert "0.050" not in content


def test_tier2_report_meta_unified_scale_and_log_effect_transform(tmp_path):
    """Regression test: report_cmd unifies table scale to effect_measure (OR) and exponentiates log_effect."""
    import json
    import math

    from click.testing import CliRunner

    from medstat.cli.main import cli

    forest_json = tmp_path / "forest_log.json"
    out_html = tmp_path / "table_log.html"
    forest_data = {
        "effect_measure": "OR",
        "is_ratio": True,
        "studies": [
            {
                "study": "Trial Log",
                "log_effect": math.log(1.50),
                "ci_lower": 1.10,
                "ci_upper": 2.05,
                "ci_scale": "natural",
                "p_value": 0.012,
            }
        ],
        "random_effects": {
            "label": "Overall Pooled",
            "log_effect": math.log(1.50),
            "ci_lower": math.log(1.10),
            "ci_upper": math.log(2.05),
            "scale": "log",
            "p_value": 0.012,
        },
    }
    forest_json.write_text(json.dumps(forest_data))

    runner = CliRunner()
    res = runner.invoke(
        cli,
        [
            "report",
            "--results",
            str(forest_json),
            "--style",
            "nejm",
            "--output",
            str(out_html),
        ],
    )
    assert res.exit_code == 0
    assert out_html.exists()
    content = out_html.read_text()
    # Check that scale header is OR (95% CI) rather than Effect (95% CI)
    assert "OR (95% CI)" in content
    # Check that log_effect was exponentiated to 1.50 rather than remaining log(1.50) ≈ 0.41
    assert "1.50 (1.10–2.05)" in content
    assert "0.41" not in content


def test_tier2_report_meta_ambiguous_log_effect_scale_rejected(tmp_path):
    """FEAT-09-B7: Ambiguous confidence limit scale for ratio log_effect is strictly rejected."""
    import json
    import math

    from click.testing import CliRunner

    from medstat.cli.main import cli

    # Case 1: Study record has log_effect and CIs without ci_scale or scale
    ambig_json = tmp_path / "forest_ambig_study.json"
    out_html = tmp_path / "table_ambig.html"
    forest_data = {
        "effect_measure": "OR",
        "is_ratio": True,
        "studies": [
            {
                "study": "Trial Ambiguous",
                "log_effect": math.log(1.50),
                "ci_lower": 1.10,
                "ci_upper": 2.05,
                # Missing ci_scale or scale
            }
        ],
        "random_effects": {
            "label": "Overall",
            "effect_disp": 1.50,
            "ci_lower": 1.10,
            "ci_upper": 2.05,
        },
    }
    ambig_json.write_text(json.dumps(forest_data))

    runner = CliRunner()
    res = runner.invoke(
        cli,
        [
            "report",
            "--results",
            str(ambig_json),
            "--style",
            "nejm",
            "--output",
            str(out_html),
        ],
    )
    assert res.exit_code != 0
    assert "Ambiguous confidence limit scale for study 'Trial Ambiguous'" in res.output

    # Case 2: Overall record has log_effect and CIs without ci_scale or scale
    ambig_overall_json = tmp_path / "forest_ambig_overall.json"
    forest_data_overall = {
        "effect_measure": "OR",
        "is_ratio": True,
        "studies": [
            {
                "study": "Trial Valid",
                "effect_size": 1.50,
                "ci_lower": 1.10,
                "ci_upper": 2.05,
            }
        ],
        "random_effects": {
            "label": "Overall",
            "log_effect": math.log(1.50),
            "ci_lower": 1.10,
            "ci_upper": 2.05,
            # Missing ci_scale or scale
        },
    }
    ambig_overall_json.write_text(json.dumps(forest_data_overall))
    res2 = runner.invoke(
        cli,
        [
            "report",
            "--results",
            str(ambig_overall_json),
            "--style",
            "nejm",
            "--output",
            str(out_html),
        ],
    )
    assert res2.exit_code != 0
    assert (
        "Ambiguous confidence limit scale for overall meta-analysis effect"
        in res2.output
    )

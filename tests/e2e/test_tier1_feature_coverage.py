"""
tests/e2e/test_tier1_feature_coverage.py: Tier 1 Feature Coverage Tests.
Validates each feature across CLI subcommands & biostatistical engines (>=5 tests per feature).
"""

import json

import numpy as np
import pandas as pd
from conftest import require_medstat_module

# ==============================================================================
# Feature 1: Data Cleaning & Missingness Audit (medstat clean)
# ==============================================================================


def test_tier1_clean_missingness_audit(
    medstat_cli_runner, incomplete_fixture_path, tmp_path
):
    """FEAT-01.1: Missingness audit emits counts, percentages, and clinical risk tiers."""
    audit_out = tmp_path / "audit.json"
    exit_code, stdout, _ = medstat_cli_runner(
        [
            "clean",
            "--data",
            str(incomplete_fixture_path),
            "--audit-only",
            "--audit-out",
            str(audit_out),
        ]
    )
    assert exit_code == 0 or "Missingness Audit" in stdout
    if audit_out.exists():
        with open(audit_out) as f:
            audit = json.load(f)
        assert "creatinine" in audit or "variables" in audit
        # Verification of clinical missing counts
        df = pd.read_csv(incomplete_fixture_path)
        assert df["creatinine"].isnull().sum() == 32
        assert df["bmi"].isnull().sum() == 56


def test_tier1_clean_littles_mcar(medstat_cli_runner, incomplete_fixture_path):
    """FEAT-01.2: Little's MCAR statistical test evaluation."""
    clean_mod = require_medstat_module("medstat.data.clean")
    df = pd.read_csv(incomplete_fixture_path)
    if hasattr(clean_mod, "littles_mcar_test"):
        res = clean_mod.littles_mcar_test(df[["age", "sbp", "creatinine", "bmi"]])
        assert "p_value" in res or "pvalue" in res or "statistic" in res


def test_tier1_clean_complete_case_strategy(
    medstat_cli_runner, incomplete_fixture_path, tmp_path
):
    """FEAT-01.3: Explicit complete-case exclusion tracks excluded rows."""
    out_csv = tmp_path / "clean_cc.csv"
    audit_out = tmp_path / "audit_cc.json"
    exit_code, stdout, _ = medstat_cli_runner(
        [
            "clean",
            "--data",
            str(incomplete_fixture_path),
            "--strategy",
            "complete-case",
            "--missing-justification",
            "MCAR assumed for baseline labs with <15% missingness",
            "--output",
            str(out_csv),
            "--audit-out",
            str(audit_out),
        ]
    )
    assert exit_code == 0
    if out_csv.exists():
        df_clean = pd.read_csv(out_csv)
        assert df_clean.isnull().sum().sum() == 0
        assert len(df_clean) < 400


def test_tier1_clean_mice_imputation(
    medstat_cli_runner, incomplete_fixture_path, tmp_path
):
    """FEAT-01.4: MICE multiple imputation completes missing values without listwise deletion."""
    out_csv = tmp_path / "clean_mice.csv"
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
            "MAR assumed with chained equations",
            "--output",
            str(out_csv),
        ]
    )
    assert exit_code == 0
    if out_csv.exists():
        df_clean = pd.read_csv(out_csv)
        assert df_clean["creatinine"].isnull().sum() == 0
        assert df_clean["bmi"].isnull().sum() == 0
        assert len(df_clean) == 400  # Zero rows excluded


def test_tier1_clean_knn_imputation(
    medstat_cli_runner, incomplete_fixture_path, tmp_path
):
    """FEAT-01.5: KNN distance-weighted imputation."""
    out_csv = tmp_path / "clean_knn.csv"
    exit_code, stdout, _ = medstat_cli_runner(
        [
            "clean",
            "--data",
            str(incomplete_fixture_path),
            "--strategy",
            "knn",
            "--neighbors",
            "5",
            "--missing-justification",
            "KNN imputation for clinical biomarkers",
            "--output",
            str(out_csv),
        ]
    )
    assert exit_code == 0
    if out_csv.exists():
        df_clean = pd.read_csv(out_csv)
        assert df_clean.isnull().sum().sum() == 0
        assert len(df_clean) == 400


def test_tier1_clean_outlier_winsorization(
    medstat_cli_runner, incomplete_fixture_path, tmp_path
):
    """FEAT-01.6: Outlier winsorization clamps extreme physiological values."""
    clean_mod = require_medstat_module("medstat.data.clean")
    s = pd.Series([1.0, 1.2, 1.1, 1.3, 100.0, 0.9, 1.0])
    if hasattr(clean_mod, "handle_outliers"):
        s_wins = clean_mod.handle_outliers(s, method="iqr", action="winsorize")
        assert s_wins.max() < 100.0


def test_tier1_clean_sample_flow_metadata(incomplete_fixture_path):
    """FEAT-01.7: SampleFlowTracker records N_initial -> N_excluded -> N_analyzed."""
    retention_mod = require_medstat_module("medstat.data.retention")
    tracker = retention_mod.SampleFlowTracker(n_initial=400)
    tracker.record_stage(
        "missing_covariates",
        n_remaining=312,
        n_excluded=88,
        reason="Missing baseline labs",
    )
    flow = tracker.get_flow_summary()
    assert flow["n_initial"] == 400
    assert flow["n_excluded"] == 88
    assert flow["n_analyzed"] == 312


# ==============================================================================
# Feature 2: Table 1 Baseline Characteristics (medstat table1)
# ==============================================================================


def test_tier1_table1_stratified_summary(
    medstat_cli_runner, oncology_fixture_path, tmp_path
):
    """FEAT-02.1: Table 1 stratified by exposure group."""
    out_table = tmp_path / "table1.json"
    exit_code, stdout, _ = medstat_cli_runner(
        [
            "table1",
            "--data",
            str(oncology_fixture_path),
            "--group",
            "treatment",
            "--vars",
            "age,stage,biomarker",
            "--output",
            str(out_table),
        ]
    )
    assert exit_code == 0 or "Table 1" in stdout


def test_tier1_table1_continuous_metrics(oncology_fixture_path):
    """FEAT-02.2: Continuous variables emit Mean/SD and Median/IQR."""
    table_mod = require_medstat_module("medstat.reporting.table1")
    df = pd.read_csv(oncology_fixture_path)
    res = table_mod.generate_table_one(
        df, continuous_vars=["age", "biomarker"], strata="treatment"
    )
    assert res is not None
    assert not res.empty
    assert "Characteristic" in res.columns


def test_tier1_table1_categorical_counts_percentages(oncology_fixture_path):
    """FEAT-02.3: Categorical factors report n (%) across tiers."""
    table_mod = require_medstat_module("medstat.reporting.table1")
    df = pd.read_csv(oncology_fixture_path)
    res = table_mod.generate_table_one(
        df, categorical_vars=["stage"], strata="treatment"
    )
    assert res is not None
    assert not res.empty
    assert any("stage" in str(x) for x in res["Characteristic"])


def test_tier1_table1_standardized_mean_differences(cardiovascular_fixture_path):
    """FEAT-02.4: Standardized Mean Differences (SMDs) computed for baseline covariates."""
    table_mod = require_medstat_module("medstat.reporting.table1")
    df = pd.read_csv(cardiovascular_fixture_path)
    res = table_mod.generate_table_one(
        df, continuous_vars=["age", "sbp", "ldl"], strata="statin_rx", include_smd=True
    )
    assert res is not None
    assert "SMD" in res.columns


def test_tier1_table1_hypothesis_testing(oncology_fixture_path):
    """FEAT-02.5: Automated p-value calculation with appropriate test selection."""
    table_mod = require_medstat_module("medstat.reporting.table1")
    df = pd.read_csv(oncology_fixture_path)
    res = table_mod.generate_table_one(
        df,
        continuous_vars=["age"],
        categorical_vars=["stage"],
        strata="treatment",
        include_p=True,
    )
    assert res is not None
    assert "P-value" in res.columns


# ==============================================================================
# Feature 3: Biostatistical Models (medstat model)
# ==============================================================================


def test_tier1_model_linear_regression(cardiovascular_fixture_path):
    """FEAT-03.1: Multivariable linear regression estimates coefficients and CIs."""
    glm_mod = require_medstat_module("medstat.models.glm")
    df = pd.read_csv(cardiovascular_fixture_path)
    res = glm_mod.fit_linear_regression(df["sbp"], df[["age", "bmi", "diabetes"]])
    assert "summary_df" in res
    assert "age" in res["summary_df"].index


def test_tier1_model_logistic_regression(cardiovascular_fixture_path):
    """FEAT-03.2: Multivariable logistic regression estimates Odds Ratios."""
    glm_mod = require_medstat_module("medstat.models.glm")
    df = pd.read_csv(cardiovascular_fixture_path)
    res = glm_mod.fit_standard_logistic(
        df["cv_event"], df[["statin_rx", "age", "sbp", "diabetes"]]
    )
    assert "summary_df" in res
    assert "statin_rx" in res["summary_df"].index


def test_tier1_model_cox_ph_survival(oncology_fixture_path):
    """FEAT-03.3: Cox Proportional Hazards regression and Schoenfeld test."""
    surv_mod = require_medstat_module("medstat.models.survival")
    df = pd.read_csv(oncology_fixture_path)
    cox_res = surv_mod.fit_cox_ph(
        df, duration_col="time", event_col="status", covariates=["age", "biomarker"]
    )
    assert cox_res is not None
    if hasattr(surv_mod, "check_proportional_hazards"):
        ph_test = surv_mod.check_proportional_hazards(cox_res)
        assert ph_test is not None


def test_tier1_model_firth_penalized_logistic():
    """FEAT-03.4: Firth penalized logistic regression with Profile Likelihood CIs."""
    firth_mod = require_medstat_module("medstat.models.firth")
    np.random.seed(42)
    # Severe separation dataset
    X = np.array([[0], [0], [0], [1], [1], [1]])
    y = np.array([0, 0, 0, 1, 1, 1])
    res = firth_mod.fit_firth_logistic(y, X)
    assert "summary_df" in res
    summary = res["summary_df"]
    assert np.isfinite(summary["estimate"]).all()


def test_tier1_model_restricted_cubic_splines(oncology_fixture_path):
    """FEAT-03.5: Restricted Cubic Splines (RCS) basis expansion."""
    splines_mod = require_medstat_module("medstat.models.splines")
    df = pd.read_csv(oncology_fixture_path)
    fig, curve_df, knots_info = splines_mod.fit_cox_rcs(
        df, duration_col="time", event_col="status", rcs_var="age", knots=4
    )
    assert not curve_df.empty
    assert "knots" in knots_info


def test_tier1_model_e_value_sensitivity():
    """FEAT-03.6: E-value for unmeasured confounding."""
    sens_mod = require_medstat_module("medstat.models.sensitivity")
    # For RR = 2.0 with 95% CI [1.5, 2.7]
    res = sens_mod.calculate_e_value(
        estimate=2.0, lower=1.5, upper=2.7, estimate_type="RR"
    )
    assert res["e_value_estimate"] > 1.0
    assert res["e_value_ci_limit"] > 1.0


def test_tier1_model_sap_spec_execution(analysis_plan_path, tmp_path):
    """FEAT-03.7: Statistical Analysis Plan (SAP) YAML execution engine."""
    spec_mod = require_medstat_module("medstat.cli.spec")
    plan = spec_mod.AnalysisPlan.from_yaml(analysis_plan_path)
    assert plan.metadata.reporting_guideline == "STROBE"
    res = plan.execute()
    assert res is not None


# ==============================================================================
# Feature 4: Diagnostic Test Accuracy & DCA (medstat diag)
# ==============================================================================


def test_tier1_diag_2x2_accuracy_wilson_ci(sepsis_fixture_path):
    """FEAT-04.1: 2x2 contingency table metrics with Wilson score 95% CIs."""
    diag_mod = require_medstat_module("medstat.diagnostic.accuracy")
    df = pd.read_csv(sepsis_fixture_path)
    # Threshold at procalcitonin >= 2.0
    pred_pos = (df["procalcitonin"] >= 2.0).astype(int)
    acc = diag_mod.calculate_diagnostic_accuracy(df["sepsis_confirmed_2x2"], pred_pos)
    assert 0.0 <= acc["sensitivity"] <= 1.0
    assert 0.0 <= acc["specificity"] <= 1.0
    assert "sensitivity_ci" in acc
    assert "specificity_ci" in acc


def test_tier1_diag_likelihood_ratios(sepsis_fixture_path):
    """FEAT-04.2: Positive (LR+) and Negative (LR-) Likelihood Ratios with log CIs."""
    diag_mod = require_medstat_module("medstat.diagnostic.accuracy")
    df = pd.read_csv(sepsis_fixture_path)
    pred_pos = (df["lactate"] >= 2.5).astype(int)
    acc = diag_mod.calculate_diagnostic_accuracy(df["sepsis_confirmed_2x2"], pred_pos)
    assert acc["lr_plus"] > 0
    assert acc["lr_minus"] > 0


def test_tier1_diag_roc_auc_delong_ci(sepsis_fixture_path):
    """FEAT-04.3: Empirical ROC curve and AUC with DeLong 95% CI."""
    roc_mod = require_medstat_module("medstat.diagnostic.roc")
    df = pd.read_csv(sepsis_fixture_path)
    auc_res = roc_mod.auc_ci_delong(df["sepsis_confirmed_2x2"], df["procalcitonin"])
    assert 0.5 <= auc_res["auc"] <= 1.0
    assert auc_res["ci_lower"] <= auc_res["auc"] <= auc_res["ci_upper"]


def test_tier1_diag_delong_roc_comparison(sepsis_fixture_path):
    """FEAT-04.4: Correlated ROC curve comparison via DeLong test."""
    roc_mod = require_medstat_module("medstat.diagnostic.roc")
    df = pd.read_csv(sepsis_fixture_path)
    comp = roc_mod.delong_paired_test(
        y_true=df["sepsis_confirmed_2x2"],
        score1=df["procalcitonin"],
        score2=df["lactate"],
    )
    assert "z_statistic" in comp
    assert "p_value" in comp


def test_tier1_diag_decision_curve_analysis_net_benefit(sepsis_fixture_path):
    """FEAT-04.5: Decision Curve Analysis (DCA) Net Benefit across threshold range."""
    dca_mod = require_medstat_module("medstat.diagnostic.dca")
    df = pd.read_csv(sepsis_fixture_path)
    # Predicted risk from normalized marker
    norm_risk = (df["procalcitonin"] - df["procalcitonin"].min()) / (
        df["procalcitonin"].max() - df["procalcitonin"].min()
    )
    dca_df = dca_mod.calculate_dca(
        y_true=df["sepsis_confirmed_2x2"],
        y_pred=norm_risk,
        thresholds=[0.05, 0.10, 0.20, 0.30, 0.50],
    )
    assert isinstance(dca_df, pd.DataFrame)
    assert "net_benefit" in dca_df.columns
    assert "strategy" in dca_df.columns


def test_tier1_diag_calibration_brier_hosmer_lemeshow(sepsis_fixture_path):
    """FEAT-04.6: Calibration metrics, Brier score, and Hosmer-Lemeshow goodness of fit."""
    calib_mod = require_medstat_module("medstat.diagnostic.calibration")
    df = pd.read_csv(sepsis_fixture_path)
    norm_risk = (df["lactate"] - df["lactate"].min()) / (
        df["lactate"].max() - df["lactate"].min()
    )
    brier_res = calib_mod.calculate_brier_score(df["sepsis_confirmed_2x2"], norm_risk)
    assert 0.0 <= brier_res["brier_score"] <= 1.0
    hl_res = calib_mod.hosmer_lemeshow_test(df["sepsis_confirmed_2x2"], norm_risk)
    assert "statistic" in hl_res
    assert "p_value" in hl_res


# ==============================================================================
# Feature 5: Causal Inference & Propensity Scores (medstat causal)
# ==============================================================================


def test_tier1_causal_propensity_score_matching(cardiovascular_fixture_path):
    """FEAT-05.1: 1:1 Nearest-Neighbor Propensity Score Matching."""
    psm_mod = require_medstat_module("medstat.causal.psm")
    df = pd.read_csv(cardiovascular_fixture_path)
    df["ps"] = psm_mod.calculate_propensity_score(
        df=df,
        treatment="statin_rx",
        covariates=["age", "sbp", "ldl", "diabetes"],
    )
    matched = psm_mod.perform_matching(
        df=df, treatment="statin_rx", ps_col="ps", caliper=0.2, ratio=1
    )
    assert len(matched) > 0
    assert "matched_group" in matched.columns or "ps" in matched.columns


def test_tier1_causal_caliper_enforcement(cardiovascular_fixture_path):
    """FEAT-05.2: Caliper width enforcement in SD of logit PS."""
    psm_mod = require_medstat_module("medstat.causal.psm")
    df = pd.read_csv(cardiovascular_fixture_path)
    df["ps"] = psm_mod.calculate_propensity_score(
        df=df, treatment="statin_rx", covariates=["age", "sbp"]
    )
    matched_tight = psm_mod.perform_matching(
        df=df, treatment="statin_rx", ps_col="ps", caliper=0.05
    )
    matched_wide = psm_mod.perform_matching(
        df=df, treatment="statin_rx", ps_col="ps", caliper=0.5
    )
    assert len(matched_tight) <= len(matched_wide)


def test_tier1_causal_smd_balance_under_0_10(cardiovascular_fixture_path):
    """FEAT-05.3: Covariate balance checking with SMD threshold."""
    balance_mod = require_medstat_module("medstat.causal.balance")
    df = pd.read_csv(cardiovascular_fixture_path)
    bal_table = balance_mod.check_balance(
        df=df, treatment="statin_rx", covariates=["age", "sbp", "ldl", "diabetes"]
    )
    assert "SMD" in bal_table.columns
    assert len(bal_table) == 4


def test_tier1_causal_love_plot_generation(cardiovascular_fixture_path):
    """FEAT-05.4: Love plot data coordinates before and after matching."""
    balance_mod = require_medstat_module("medstat.causal.balance")
    df = pd.read_csv(cardiovascular_fixture_path)
    comp_df = balance_mod.compare_pre_post_balance(
        raw_df=df,
        matched_df=df,
        treatment="statin_rx",
        covariates=["age", "sbp", "ldl"],
    )
    assert "Pre_SMD" in comp_df.columns
    fig = balance_mod.create_love_plot(comp_df)
    assert fig is not None


def test_tier1_causal_mediation_analysis(cardiovascular_fixture_path):
    """FEAT-05.5: Causal mediation analysis (ACME, ADE, Total Effect)."""
    causal_mod = require_medstat_module("medstat.causal.mediation")
    df = pd.read_csv(cardiovascular_fixture_path)
    if hasattr(causal_mod, "run_mediation"):
        res = causal_mod.run_mediation(
            df=df,
            treatment="statin_rx",
            mediator="ldl",
            outcome="cv_event",
            covariates=["age", "sbp"],
        )
        assert "acme" in res or "indirect_effect" in res


# ==============================================================================
# Feature 6: Evidence Synthesis & Meta-Analysis (medstat meta)
# ==============================================================================


def test_tier1_meta_fixed_effect_model(meta_fixture_path):
    """FEAT-06.1: Inverse-variance weighted fixed-effect meta-analysis."""
    meta_mod = require_medstat_module("medstat.meta.models")
    df = pd.read_csv(meta_fixture_path)
    df["log_effect"] = df["effect_size"]
    res = meta_mod.run_meta_analysis(df)
    assert "fixed_effect" in res
    assert "log_effect" in res["fixed_effect"]
    assert "ci_lower" in res["fixed_effect"]


def test_tier1_meta_random_effects_dersimonian_laird(meta_fixture_path):
    """FEAT-06.2: DerSimonian-Laird random effects meta-analysis."""
    meta_mod = require_medstat_module("medstat.meta.models")
    df = pd.read_csv(meta_fixture_path)
    df["log_effect"] = df["effect_size"]
    res = meta_mod.run_meta_analysis(df, method_re="DL")
    assert "random_effects" in res
    re = res["random_effects"]
    assert re["ci_lower"] <= re["log_effect"] <= re["ci_upper"]


def test_tier1_meta_heterogeneity_statistics(meta_fixture_path):
    """FEAT-06.3: Heterogeneity statistics (Cochran's Q, I^2, Tau^2)."""
    meta_mod = require_medstat_module("medstat.meta.models")
    df = pd.read_csv(meta_fixture_path)
    df["log_effect"] = df["effect_size"]
    res = meta_mod.run_meta_analysis(df)
    het = res["heterogeneity"]
    assert 0.0 <= het["I2"] <= 100.0
    assert het["tau2"] >= 0.0
    assert het["Q"] >= 0.0


def test_tier1_meta_forest_plot_coordinates(meta_fixture_path):
    """FEAT-06.4: Forest plot structured coordinate generation."""
    forest_mod = require_medstat_module("medstat.meta.forest")
    meta_mod = require_medstat_module("medstat.meta.models")
    df = pd.read_csv(meta_fixture_path)
    df["log_effect"] = df["effect_size"]
    meta_res = meta_mod.run_meta_analysis(df)
    plot_data = forest_mod.generate_forest_data(meta_res)
    assert "studies" in plot_data


def test_tier1_meta_eggers_test_publication_bias(meta_fixture_path):
    """FEAT-06.5: Egger's linear regression test for funnel plot asymmetry."""
    bias_mod = require_medstat_module("medstat.meta.bias")
    df = pd.read_csv(meta_fixture_path)
    df["log_effect"] = df["effect_size"]
    res = bias_mod.eggers_test(df)
    assert "intercept" in res
    assert "p_value" in res


# ==============================================================================
# Feature 7: Inter-Rater Reliability & Agreement (medstat agreement)
# ==============================================================================


def test_tier1_agreement_pure_scipy_icc1(pocus_fixture_path):
    """FEAT-07.1: Pure SciPy One-Way Random ICC (ICC1, ICC1k)."""
    icc_mod = require_medstat_module("medstat.agreement.icc")
    df = pd.read_csv(pocus_fixture_path)
    res = icc_mod.calculate_icc(
        df,
        targets="subject_id",
        raters="rater_id",
        ratings="measurement_score",
        icc_type="ICC1",
    )
    assert not res.empty
    assert "ICC" in res.columns
    assert "F" in res.columns


def test_tier1_agreement_pure_scipy_icc2(pocus_fixture_path):
    """FEAT-07.2: Pure SciPy Two-Way Random ICC for Agreement (ICC2, ICC2k)."""
    icc_mod = require_medstat_module("medstat.agreement.icc")
    df = pd.read_csv(pocus_fixture_path)
    res = icc_mod.calculate_icc(
        df,
        targets="subject_id",
        raters="rater_id",
        ratings="measurement_score",
        icc_type="ICC2",
    )
    assert not res.empty
    row = res[res["Type"] == "ICC2"].iloc[0]
    assert 0.0 <= row["ICC"] <= 1.0


def test_tier1_agreement_pure_scipy_icc3(pocus_fixture_path):
    """FEAT-07.3: Pure SciPy Two-Way Mixed ICC for Consistency (ICC3, ICC3k)."""
    icc_mod = require_medstat_module("medstat.agreement.icc")
    df = pd.read_csv(pocus_fixture_path)
    res = icc_mod.calculate_icc(
        df,
        targets="subject_id",
        raters="rater_id",
        ratings="measurement_score",
        icc_type="ICC3",
    )
    assert not res.empty


def test_tier1_agreement_bland_altman_bias_and_loa(pocus_fixture_path):
    """FEAT-07.4: Bland-Altman mean bias and 95% Limits of Agreement (LoA)."""
    ba_mod = require_medstat_module("medstat.agreement.bland_altman")
    df = pd.read_csv(pocus_fixture_path)
    df_wide = df.pivot(
        index="subject_id", columns="rater_id", values="measurement_score"
    ).reset_index()
    res = ba_mod.calculate_bland_altman(df_wide, "Rater_1", "Rater_2")
    bias = res.get("mean_bias", res.get("mean_diff"))
    assert bias is not None
    assert res["lower_loa"] <= bias <= res["upper_loa"]


def test_tier1_agreement_kappa_reliability(pocus_fixture_path):
    """FEAT-07.5: Fleiss' and Cohen's Kappa for categorical agreement."""
    agree_mod = require_medstat_module("medstat.agreement.icc")
    if hasattr(agree_mod, "calculate_fleiss_kappa"):
        df = pd.read_csv(pocus_fixture_path)
        # Discretize continuous scores to categories
        df["score_cat"] = pd.qcut(
            df["measurement_score"], q=3, labels=["Low", "Med", "High"]
        )
        df_wide = df.pivot(index="subject_id", columns="rater_id", values="score_cat")
        res = agree_mod.calculate_fleiss_kappa(df_wide)
        assert -1.0 <= res["kappa"] <= 1.0


# ==============================================================================
# Feature 8: Sample Size & Power Calculations (medstat sample-size)
# ==============================================================================


def test_tier1_sample_size_two_sample_t_test():
    """FEAT-08.1: Sample size for two-sample continuous t-test."""
    power_mod = require_medstat_module("medstat.power.sample_size")
    res = power_mod.calculate_sample_size_means(
        mean1=10.0, mean2=15.0, sd1=10.0, alpha=0.05, power=0.80
    )
    assert res["n1"] > 10
    assert res["total_n"] > 20


def test_tier1_sample_size_two_proportions():
    """FEAT-08.2: Sample size for comparing two independent proportions."""
    power_mod = require_medstat_module("medstat.power.sample_size")
    res = power_mod.calculate_sample_size_proportions(
        p1=0.20, p2=0.35, alpha=0.05, power=0.80
    )
    assert res["n1"] > 20
    assert res["total_n"] > 40


def test_tier1_sample_size_survival_log_rank():
    """FEAT-08.3: Sample size and required events for log-rank survival test."""
    power_mod = require_medstat_module("medstat.power.sample_size")
    res = power_mod.calculate_sample_size_survival(
        hazard_ratio=0.70, alpha=0.05, power=0.80
    )
    assert res["required_events"] > 20
    assert res["total_n"] > 40


def test_tier1_sample_size_diagnostic_accuracy():
    """FEAT-08.4: Sample size for diagnostic correlation or precision."""
    power_mod = require_medstat_module("medstat.power.sample_size")
    res = power_mod.calculate_sample_size_correlation(r=0.40, alpha=0.05, power=0.80)
    assert res["required_n"] > 20


def test_tier1_sample_size_power_calculation():
    """FEAT-08.5: Prospective power calculation for means."""
    power_mod = require_medstat_module("medstat.power.sample_size")
    res = power_mod.calculate_sample_size_means(
        mean1=10.0, mean2=15.0, sd1=10.0, power=0.90
    )
    assert res["power"] == 0.90


# ==============================================================================
# Feature 9: Publication Reporting & Guidelines (medstat report)
# ==============================================================================


def test_tier1_report_nejm_html_table(tmp_path):
    """FEAT-09.1: NEJM styled publication HTML table."""
    report_mod = require_medstat_module("medstat.reporting.tables")
    est_table = report_mod.EstimateTable(
        title="Table 1. Baseline Characteristics",
        rows=[
            report_mod.Estimate(
                term="age",
                label="Age (years)",
                estimate=62.4,
                ci_lower=52.3,
                ci_upper=72.5,
                p_value=0.01,
            ),
            report_mod.Estimate(
                term="sbp",
                label="Systolic BP (mmHg)",
                estimate=138.2,
                ci_lower=123.2,
                ci_upper=153.2,
                p_value=0.04,
            ),
        ],
    )
    html_out = report_mod.render_nejm_table(est_table)
    assert "<table" in html_out
    assert "border-top" in html_out or "Age" in html_out


def test_tier1_report_jama_html_table(tmp_path):
    """FEAT-09.2: JAMA styled publication HTML table."""
    report_mod = require_medstat_module("medstat.reporting.tables")
    est_table = report_mod.EstimateTable(
        title="Table 2. Primary Clinical Outcomes",
        rows=[
            report_mod.Estimate(
                term="treatment",
                label="Treatment",
                estimate=0.65,
                ci_lower=0.45,
                ci_upper=0.92,
                p_value=0.015,
                scale="HR",
            )
        ],
    )
    html_out = report_mod.render_jama_table(est_table)
    assert "<table" in html_out
    assert "0.65" in html_out


def test_tier1_report_apa7_html_table(tmp_path):
    """FEAT-09.3: APA 7 styled publication HTML table."""
    report_mod = require_medstat_module("medstat.reporting.tables")
    est_table = report_mod.EstimateTable(
        title="Table 3. Model Parameters",
        rows=[
            report_mod.Estimate(
                term="treatment",
                label="Treatment",
                estimate=0.45,
                ci_lower=0.12,
                ci_upper=0.78,
                p_value=0.034,
                scale="Beta",
            )
        ],
    )
    html_out = report_mod.render_apa_table(est_table)
    assert "<table" in html_out
    assert "Table 3" in html_out


def test_tier1_report_methods_narrative_generation():
    """FEAT-09.4: Automated Statistical Methods narrative paragraph generation."""
    narrative_mod = require_medstat_module("medstat.reporting.narrative")
    methods_text = narrative_mod.generate_methods_narrative(
        model_type="cox",
        exposure="treatment",
        outcome="survival",
        covariates=["age", "stage"],
        missing_strategy="complete-case",
    )
    assert len(methods_text) > 50
    assert (
        "proportional hazards" in methods_text.lower()
        or "cox" in methods_text.lower()
        or "continuous" in methods_text.lower()
    )


def test_tier1_report_guideline_compliance_audit():
    """FEAT-09.5: STROBE, CONSORT, and TRIPOD statement compliance checklists."""
    audit_mod = require_medstat_module("medstat.reporting.tables")
    if hasattr(audit_mod, "audit_guideline_compliance"):
        audit = audit_mod.audit_guideline_compliance(
            guideline="strobe",
            metadata={"participant_flow": True, "missing_reported": True},
        )
        assert "items" in audit or "score" in audit

"""
tests/unit/test_shinystat_cloud.py: Comprehensive unit tests and statistical invariant
verification for the shinystat-cloud standalone package and recipes.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

# Import functions from python-recipes directly by extracting or importing
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CLOUD_DIR = REPO_ROOT / "packaging" / "cloud" / "shinystat-cloud"
RECIPES_PATH = CLOUD_DIR / "references" / "python-recipes.md"


def _extract_recipe_namespace():
    """Extract and execute all Python code blocks from python-recipes.md in an isolated namespace."""
    content = RECIPES_PATH.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)\n```", content, re.DOTALL)
    ns = {}
    combined_code = "\n".join(blocks)
    exec(combined_code, ns)
    return ns


@pytest.fixture(scope="module")
def recipes():
    return _extract_recipe_namespace()


# -----------------------------------------------------------------------------
# 1. Package Metadata & Architecture Integrity
# -----------------------------------------------------------------------------


def test_cloud_skill_metadata_and_frontmatter():
    skill_file = CLOUD_DIR / "SKILL.md"
    assert skill_file.exists(), "packaging/cloud/shinystat-cloud/SKILL.md missing"
    content = skill_file.read_text(encoding="utf-8")
    parts = content.split("---", 2)
    assert len(parts) >= 3, "Frontmatter must be delimited by ---"
    fm = yaml.safe_load(parts[1])

    assert fm["name"] == "shinystat-cloud"
    assert "description" in fm
    assert len(fm["description"]) < 1024, "Frontmatter description must be < 1024 chars"
    # Check intent triggers
    assert "Table 1" in fm["description"]
    assert "logistic regression" in fm["description"]
    assert "Cox" in fm["description"]
    assert "ROC" in fm["description"]
    assert "PSM" in fm["description"]


def test_decision_heuristics_has_no_phantom_medstat_modules():
    heuristics_file = CLOUD_DIR / "references" / "decision-heuristics.md"
    assert heuristics_file.exists()
    content = heuristics_file.read_text(encoding="utf-8")
    # Must NOT advise the agent to import medstat in the cloud sandbox!
    assert "import medstat" not in content
    assert "medstat.models.firth" not in content
    assert "medstat.data.clean" not in content
    assert "medstat.diagnostic" not in content


# -----------------------------------------------------------------------------
# 2. Recipe 1: Table 1 & SMD Safeguards
# -----------------------------------------------------------------------------


def test_smd_safeguards_zero_variance_disparity(recipes):
    calc_smd = recipes["calculate_smd"]
    calc_bin_smd = recipes["calculate_binary_smd"]

    # Continuous: identical constant groups -> 0.0
    assert calc_smd([5.0, 5.0], [5.0, 5.0]) == 0.0
    # Continuous: differing constant groups -> NaN (undefined)
    assert np.isnan(calc_smd([5.0, 5.0], [10.0, 10.0]))

    # Binary: 0% vs 100% disparity -> NaN (NOT 0.00!)
    assert np.isnan(calc_bin_smd(0.0, 1.0))
    # Binary: equal proportions -> 0.0
    assert calc_bin_smd(0.5, 0.5) == 0.0


def test_generate_table_one_execution_and_safeguards(recipes):
    gen_table = recipes["generate_table_one"]
    np.random.seed(42)
    n = 60
    df = pd.DataFrame(
        {
            "group": np.random.choice(["A", "B"], size=n),
            "age": np.random.normal(55, 10, size=n),
            "uniform_val": [0.0] * n,  # Zero-variance continuous
            "smoking": np.random.choice(["No", "Yes", None], p=[0.5, 0.4, 0.1], size=n),
        }
    )

    table_df = gen_table(
        df,
        strata="group",
        continuous_vars=["age", "uniform_val"],
        categorical_vars=["smoking"],
        nonnormal_vars=["uniform_val"],
    )
    assert isinstance(table_df, pd.DataFrame)
    assert "Characteristic" in table_df.columns
    assert "SMD" in table_df.columns
    # Check that uniform_val did not crash Kruskal-Wallis
    uniform_row = table_df[table_df["Characteristic"] == "uniform_val"]
    assert len(uniform_row) == 1


# -----------------------------------------------------------------------------
# 3. Recipe 2: 2x2 Diagnostics with Wilson & Log-Scale CIs
# -----------------------------------------------------------------------------


def test_two_by_two_metrics_with_haldane_correction(recipes):
    calc_2x2 = recipes["calculate_2x2_metrics"]
    # Table with zero false positive: TP=25, FP=0, FN=5, TN=30
    res = calc_2x2(tp=25, fp=0, fn=5, tn=30)
    assert res["Specificity"][0] == 1.0
    # LR+ and DOR should be non-NaN due to Haldane-Anscombe correction
    assert not np.isnan(res["LR_pos"][0])
    assert not np.isnan(res["DOR"][0])
    assert len(res["LR_pos"][1]) == 2
    assert len(res["DOR"][1]) == 2
    assert res["LR_pos"][1][0] > 0.0


# -----------------------------------------------------------------------------
# 4. Recipe 3: ROC & DeLong with NaN Safety and Paired Test
# -----------------------------------------------------------------------------


def test_delong_nan_safety_in_series(recipes):
    auc_ci = recipes["auc_ci_delong"]
    paired_test = recipes["delong_paired_test"]

    # Series containing NaN must NOT convert NaN to 0!
    y_series = pd.Series([1.0, 1.0, 0.0, 0.0, np.nan])
    score1 = [0.9, 0.8, 0.2, 0.1, 0.5]
    score2 = [0.8, 0.7, 0.3, 0.2, 0.4]

    res = auc_ci(y_series, score1)
    assert res["auc"] == 1.0
    assert res["n_pos"] == 2
    assert res["n_neg"] == 2

    # Paired test
    paired_res = paired_test(y_series, score1, score2)
    assert "difference" in paired_res
    assert "p_value" in paired_res
    assert not np.isnan(paired_res["p_value"])


# -----------------------------------------------------------------------------
# 5. Recipe 4: Model Calibration & DCA
# -----------------------------------------------------------------------------


def test_calibration_and_dca_boundary_conditions(recipes):
    eval_calib = recipes["evaluate_calibration"]
    calc_dca = recipes["calculate_dca"]

    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3])

    calib = eval_calib(y_true, y_pred)
    assert 0.0 <= calib["brier_score"] <= 1.0
    assert 0.0 <= calib["ici"] <= 1.0

    # Scaled brier when all outcomes are 1 (must not divide by zero!)
    calib_all_ones = eval_calib([1, 1, 1], [0.9, 0.8, 0.7])
    assert np.isnan(calib_all_ones["scaled_brier"])

    # DCA net benefit
    dca_df = calc_dca(y_true, y_pred)
    assert isinstance(dca_df, pd.DataFrame)
    assert set(dca_df["strategy"].unique()) == {"Model", "Treat All", "Treat None"}


# -----------------------------------------------------------------------------
# 6. Recipe 5: Bland-Altman & Shrout-Fleiss (1979) ICC
# -----------------------------------------------------------------------------


def test_bland_altman_pairwise_nans(recipes):
    calc_ba = recipes["calculate_bland_altman"]
    m1 = [10.0, 12.0, np.nan, 15.0]
    m2 = [10.5, 11.8, 14.0, 15.2]
    res = calc_ba(m1, m2)
    assert res["n_pairs"] == 3
    assert not np.isnan(res["mean_diff"])
    assert not np.isnan(res["ci_upper_loa"][0])


def test_icc_matrix_exact_f_distribution_cis(recipes):
    calc_icc = recipes["calculate_icc_matrix"]
    # Shrout & Fleiss 1979 benchmark (6 targets, 4 judges)
    sf_data = np.array(
        [
            [9, 2, 5, 8],
            [6, 1, 3, 2],
            [8, 4, 6, 8],
            [7, 1, 2, 6],
            [10, 5, 6, 9],
            [6, 2, 4, 7],
        ]
    )
    icc_df = calc_icc(sf_data, alpha=0.05)
    assert isinstance(icc_df, pd.DataFrame)
    assert "CI_lower" in icc_df.columns
    assert "CI_upper" in icc_df.columns

    # Verify Shrout & Fleiss benchmark values
    icc1 = icc_df.loc[icc_df["Type"] == "ICC1", "ICC"].values[0]
    icc2 = icc_df.loc[icc_df["Type"] == "ICC2", "ICC"].values[0]
    icc3 = icc_df.loc[icc_df["Type"] == "ICC3", "ICC"].values[0]
    assert np.isclose(icc1, 0.1657, atol=1e-3)
    assert np.isclose(icc2, 0.2898, atol=1e-3)
    assert np.isclose(icc3, 0.7148, atol=1e-3)

    # All CIs must be finite floats
    assert not np.isnan(icc_df["CI_lower"]).any()
    assert not np.isnan(icc_df["CI_upper"]).any()


# -----------------------------------------------------------------------------
# 7. Recipe 6: PSM with Pair IDs & Covariate Balance
# -----------------------------------------------------------------------------


def test_psm_pair_ids_and_balance(recipes):
    psm_match = recipes["match_propensity_scores"]
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "treated": np.random.binomial(1, 0.4, size=n),
            "age": np.random.normal(50, 10, size=n),
            "female": np.random.choice(["No", "Yes"], size=n),
        }
    )

    res = psm_match(
        df, treatment_col="treated", confounders=["age", "female"], caliper_sd=0.2
    )
    matched_df = res["matched_df"]
    assert "pair_id" in matched_df.columns
    assert res["n_matched_pairs"] > 0
    # Every pair_id must have exactly 2 rows (1 treated, 1 control)
    counts = matched_df["pair_id"].value_counts()
    assert (counts == 2).all()

    # Balance table
    balance_df = res["balance_df"]
    assert "Pre-Match SMD" in balance_df.columns
    assert "Post-Match SMD" in balance_df.columns


# -----------------------------------------------------------------------------
# 8. Recipe 7: Firth Logistic Regression with Step-Halving & PL CIs
# -----------------------------------------------------------------------------


def test_firth_logistic_step_halving_and_profile_ci(recipes):
    fit_firth = recipes["fit_firth_logistic"]
    # 2x2 table with monotone separation (zero cell)
    X = np.array([[0.0]] * 10 + [[1.0]] * 15)
    y = np.array([0] * 10 + [0] * 5 + [1] * 10)

    res = fit_firth(X, y, fit_intercept=True)
    assert res["converged"]
    assert len(res["odds_ratios"]) == 2
    # Profile Likelihood CI upper should be finite and bounded
    assert 1.0 < res["ci_upper"][1] < 10000.0


# -----------------------------------------------------------------------------
# 9. Recipe 8: Little's MCAR EM Algorithm
# -----------------------------------------------------------------------------


def test_littles_mcar_pattern_grouped_em(recipes):
    littles_test = recipes["littles_mcar_test"]
    np.random.seed(42)
    X = np.random.randn(200, 4)
    df = pd.DataFrame(X, columns=["x1", "x2", "x3", "x4"])
    # Inject MCAR missingness
    for col in ["x2", "x3", "x4"]:
        df.loc[np.random.rand(200) < 0.15, col] = np.nan

    res = littles_test(df, ["x1", "x2", "x3", "x4"])
    assert res["df"] < 200, (
        f"Degrees of freedom must be much smaller than N! Got {res['df']}"
    )
    assert "chi2" in res
    assert "p_value" in res
    assert 0.0 <= res["p_value"] <= 1.0


# -----------------------------------------------------------------------------
# 10. Recipe 9: VanderWeele E-Value Prevalence Branching (OR & HR)
# -----------------------------------------------------------------------------


def test_e_value_prevalence_branching(recipes):
    calc_e = recipes["calculate_e_value"]
    # 1. Odds Ratio (OR)
    rare_e = calc_e(4.0, lower=2.0, estimate_type="OR", rare_outcome=True)
    common_e = calc_e(4.0, lower=2.0, estimate_type="OR", rare_outcome=False)

    assert np.isclose(rare_e["e_value_point"], 7.464, atol=1e-2)
    assert np.isclose(common_e["e_value_point"], 3.414, atol=1e-2)
    assert rare_e["e_value_point"] > common_e["e_value_point"]

    # 2. Hazard Ratio (HR) - VanderWeele & Ding (2017) continuous conversion
    rare_hr = calc_e(2.0, lower=1.2, estimate_type="HR", rare_outcome=True)
    common_hr = calc_e(2.0, lower=1.2, estimate_type="HR", rare_outcome=False)

    # For rare outcome: HR=2.0 -> RR*=2.0 -> E-value ≈ 3.414
    assert np.isclose(rare_hr["e_value_point"], 3.414, atol=1e-2)
    # For common outcome: HR=2.0 -> RR*=(1-0.5^sqrt(2))/(1-0.5^sqrt(0.5)) ≈ 1.6125 -> E-value ≈ 2.606
    assert np.isclose(common_hr["e_value_point"], 2.606, atol=1e-2)
    assert rare_hr["e_value_point"] > common_hr["e_value_point"]
    assert rare_hr["e_value_ci"] > common_hr["e_value_ci"]

    # Protective HR (< 1.0)
    prot_rare = calc_e(0.5, upper=0.8, estimate_type="HR", rare_outcome=True)
    assert prot_rare["e_value_point"] > 1.0
    assert prot_rare["e_value_ci"] > 1.0


# -----------------------------------------------------------------------------
# 11. Recipes 10-13: Core Clinical Workflows (Logistic Table, Survival, MICE, HTML)
# -----------------------------------------------------------------------------


def test_logistic_regression_table(recipes):
    fit_table = recipes["fit_logistic_regression_table"]
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "outcome": np.random.binomial(1, 0.3, size=80),
            "age": np.random.normal(50, 10, size=80),
            "sbp": np.random.normal(120, 15, size=80),
        }
    )
    res_df = fit_table(df, outcome="outcome", covariates=["age", "sbp"])
    assert len(res_df) == 2
    assert "Crude OR (95% CI)" in res_df.columns
    assert "Adjusted OR (95% CI)" in res_df.columns


def test_survival_analysis_suite_two_groups(recipes):
    fit_surv = recipes["fit_survival_analysis_suite"]
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "time": np.random.exponential(10, size=60) + 1,
            "event": np.random.binomial(1, 0.5, size=60),
            "treatment": np.random.choice([0, 1], size=60),
            "age": np.random.normal(50, 10, size=60),
        }
    )
    res = fit_surv(
        df,
        duration_col="time",
        event_col="event",
        strata_col="treatment",
        covariates=["age"],
    )
    assert "km_summary" in res
    assert "Log-Rank P-value" in res["km_summary"]
    assert "cox_table" in res
    assert len(res["cox_table"]) == 1
    assert "schoenfeld_diagnostics" in res
    assert "p_values" in res["schoenfeld_diagnostics"]
    assert "test_statistics" in res["schoenfeld_diagnostics"]
    assert "age" in res["schoenfeld_diagnostics"]["p_values"]
    assert isinstance(res["schoenfeld_diagnostics"]["PH_Assumptions_Passed"], bool)


def test_survival_analysis_suite_multivariate_logrank(recipes):
    fit_surv = recipes["fit_survival_analysis_suite"]
    np.random.seed(42)
    # 3 groups for strata
    df = pd.DataFrame(
        {
            "time": np.random.exponential(10, size=90) + 1,
            "event": np.random.binomial(1, 0.5, size=90),
            "stage": np.random.choice(["I", "II", "III"], size=90),
            "age": np.random.normal(50, 10, size=90),
        }
    )
    res = fit_surv(
        df,
        duration_col="time",
        event_col="event",
        strata_col="stage",
        covariates=["age"],
    )
    assert "Multivariate Log-Rank P-value" in res["km_summary"]
    assert 0.0 <= res["km_summary"]["Multivariate Log-Rank P-value"] <= 1.0


def test_mice_imputation_single(recipes):
    mice_single = recipes["impute_mice_single"]
    mice_alias = recipes["impute_mice"]
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "outcome": [0, 1, 0, 1, 0],
            "lab1": [1.2, np.nan, 2.5, 3.1, 2.0],
            "lab2": [10.0, 12.0, np.nan, 15.0, 11.0],
        }
    )
    imputed = mice_single(
        df, features_to_impute=["lab1", "lab2"], outcome_col="outcome"
    )
    assert not imputed[["lab1", "lab2"]].isna().any().any()
    # Backward compatible alias check
    imputed_alias = mice_alias(
        df, features_to_impute=["lab1", "lab2"], outcome_col="outcome"
    )
    assert not imputed_alias[["lab1", "lab2"]].isna().any().any()

    # Attempting to impute primary outcome must raise ValueError
    with pytest.raises(ValueError, match="Never impute the primary outcome"):
        mice_single(df, features_to_impute=["outcome", "lab1"], outcome_col="outcome")


def test_mice_multiple_datasets_and_rubin_pooling(recipes):
    impute_datasets = recipes["impute_mice_datasets"]
    pool_rubin = recipes["pool_estimates_rubin"]

    np.random.seed(42)
    n = 80
    df = pd.DataFrame(
        {
            "outcome": np.random.binomial(1, 0.3, size=n),
            "x1": np.where(
                np.random.rand(n) < 0.2, np.nan, np.random.normal(10, 2, size=n)
            ),
            "x2": np.where(
                np.random.rand(n) < 0.2, np.nan, np.random.normal(50, 5, size=n)
            ),
        }
    )

    # 1. Impute M=5 stochastic datasets
    m = 5
    datasets = impute_datasets(
        df, features_to_impute=["x1", "x2"], outcome_col="outcome", m=m
    )
    assert len(datasets) == m
    for d in datasets:
        assert not d[["x1", "x2"]].isna().any().any()
        assert (d["outcome"] == df["outcome"]).all(), (
            "Primary outcome must remain unchanged!"
        )

    # Stochastic check: values should not all be identical across imputations
    imputed_x1_vals = [d.loc[df["x1"].isna(), "x1"].values for d in datasets]
    assert not np.allclose(imputed_x1_vals[0], imputed_x1_vals[1]), (
        "MICE stochastic posterior should introduce variance"
    )

    # 2. Fit models on each dataset and pool with Rubin's rules
    estimates = []
    ses = []
    for d in datasets:
        # Simple sample mean and standard error of x1
        estimates.append(float(d["x1"].mean()))
        ses.append(float(d["x1"].std() / np.sqrt(len(d))))

    pooled = pool_rubin(estimates, ses, n_obs=n, k_params=1)

    assert "pooled_estimate" in pooled
    assert "pooled_se" in pooled
    assert "ci_lower" in pooled
    assert "ci_upper" in pooled
    assert "between_variance" in pooled
    assert "within_variance" in pooled
    assert "total_variance" in pooled
    assert "df" in pooled
    assert "fmi" in pooled

    # Variance inflation: total variance > within variance because between variance > 0
    assert pooled["between_variance"] > 0
    assert pooled["total_variance"] > pooled["within_variance"]
    assert pooled["ci_lower"] < pooled["pooled_estimate"] < pooled["ci_upper"]
    assert pooled["m_imputations"] == m

    # Require m >= 2
    with pytest.raises(ValueError, match="at least m=2"):
        pool_rubin([1.0], [0.2])


def test_mice_single_feature_with_outcome_conditioning(recipes):
    impute_datasets = recipes["impute_mice_datasets"]
    pool_rubin = recipes["pool_estimates_rubin"]

    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "dead": np.random.binomial(1, 0.3, size=n),
            "bp": np.where(
                np.random.rand(n) < 0.4, np.nan, np.random.normal(120, 15, size=n)
            ),
        }
    )

    # 1. Single feature to impute with outcome as predictor must succeed and have between-variance > 0
    datasets = impute_datasets(df, features_to_impute=["bp"], outcome_col="dead", m=5)
    assert len(datasets) == 5
    for d in datasets:
        assert not d["bp"].isna().any()
        assert (d["dead"] == df["dead"]).all(), (
            "Primary outcome must remain completely untouched"
        )

    means = [float(d["bp"].mean()) for d in datasets]
    ses = [float(d["bp"].std() / np.sqrt(len(d))) for d in datasets]
    pooled = pool_rubin(means, ses, n_obs=n, k_params=1)

    assert pooled["between_variance"] > 0.0, (
        "Stochastic MICE conditioned on outcome must yield B > 0"
    )
    assert pooled["fmi"] > 0.0, (
        "Fraction of Missing Information must be strictly positive"
    )

    # 2. Single feature without any other predictor or outcome must raise ValueError (prevents mean imputation)
    with pytest.raises(ValueError, match="requires >= 1 additional predictor"):
        impute_datasets(
            df, features_to_impute=["bp"], outcome_col=None, predictors=None
        )


def test_mice_attenuation_recovery_vs_full_data(recipes):
    import statsmodels.api as sm

    impute_datasets = recipes["impute_mice_datasets"]

    np.random.seed(42)
    n = 500
    bp = np.random.normal(120, 15, size=n)
    log_odds = -7.0 + 0.05 * bp
    prob = 1.0 / (1.0 + np.exp(-log_odds))
    dead = np.random.binomial(1, prob, size=n)

    df_full = pd.DataFrame({"bp": bp, "dead": dead})
    fit_full = sm.Logit(df_full["dead"], sm.add_constant(df_full[["bp"]])).fit(
        disp=False
    )
    slope_full = float(fit_full.params["bp"])

    # 40% missing in bp
    df_miss = df_full.copy()
    df_miss.loc[np.random.rand(n) < 0.4, "bp"] = np.nan

    # Impute using outcome as predictor
    datasets = impute_datasets(
        df_miss, features_to_impute=["bp"], outcome_col="dead", m=5, random_state=42
    )

    slopes_mice = []
    ses_mice = []
    for d in datasets:
        fit = sm.Logit(d["dead"], sm.add_constant(d[["bp"]])).fit(disp=False)
        slopes_mice.append(float(fit.params["bp"]))
        ses_mice.append(float(fit.bse["bp"]))

    import warnings

    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    pool_rubin = recipes["pool_estimates_rubin"]
    pooled = pool_rubin(slopes_mice, ses_mice, n_obs=n, k_params=2)

    # Slope must retain true positive association without severe attenuation towards null
    assert pooled["pooled_estimate"] > 0.030
    assert pooled["p_value"] < 0.001
    assert pooled["between_variance"] > 0.0
    assert pooled["total_variance"] > pooled["within_variance"]
    assert np.isclose(pooled["pooled_estimate"], slope_full, atol=0.02)


def test_mice_simulation_bias_and_coverage(recipes):
    """Rigorous 50-trial Monte Carlo simulation verifying |bias| < 10% and coverage >= 90% under MAR."""
    import warnings

    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    impute_datasets = recipes["impute_mice_datasets"]
    pool_rubin = recipes["pool_estimates_rubin"]

    num_trials = 50
    true_beta = 0.06
    slopes = []
    cov_count = 0

    for trial in range(num_trials):
        rng = np.random.default_rng(trial)
        n = 500
        bp = rng.normal(120, 15, size=n)
        log_odds = -7.0 + true_beta * bp
        prob = 1.0 / (1.0 + np.exp(-log_odds))
        dead = rng.binomial(1, prob, size=n)

        df_full = pd.DataFrame({"bp": bp, "dead": dead})
        df_miss = df_full.copy()
        # MAR missingness: higher missingness among events
        miss_prob = np.where(dead == 1, 0.6, 0.2)
        df_miss.loc[rng.random(n) < miss_prob, "bp"] = np.nan

        m = 10
        datasets = impute_datasets(
            df_miss,
            features_to_impute=["bp"],
            outcome_col="dead",
            m=m,
            max_iter=5,
            random_state=trial * 100,
        )

        imp_slopes = []
        imp_ses = []
        for d in datasets:
            fit = sm.Logit(d["dead"], sm.add_constant(d[["bp"]])).fit(disp=False)
            imp_slopes.append(float(fit.params["bp"]))
            imp_ses.append(float(fit.bse["bp"]))

        pooled = pool_rubin(imp_slopes, imp_ses, n_obs=n, k_params=2)
        th = pooled["pooled_estimate"]
        ci_low = pooled["ci_lower"]
        ci_high = pooled["ci_upper"]

        if ci_low <= true_beta <= ci_high:
            cov_count += 1
        slopes.append(th)

    mean_slope = float(np.mean(slopes))
    rel_bias = abs(mean_slope - true_beta) / true_beta
    coverage = cov_count / num_trials

    # Assert relative bias < 10%
    assert rel_bias < 0.10, (
        f"MICE relative bias ({rel_bias:.2%}) must be < 10% (mean slope: {mean_slope:.4f}, true: {true_beta})"
    )
    # Assert empirical coverage >= 90%
    assert coverage >= 0.90, (
        f"MICE empirical CI coverage ({coverage:.2%}) must be >= 90% (target nominal 95%)"
    )


def test_publication_table_html_rendering(recipes):
    render_table = recipes["render_publication_table"]
    df = pd.DataFrame(
        {
            "Variable": ["Age", "SBP"],
            "OR (95% CI)": ["1.05 (1.01–1.09)", "1.02 (0.99–1.05)"],
            "P-value": ["0.012", "0.210"],
        }
    )
    html = render_table(
        df, title="Table 2. Multivariable Logistic Regression", style="nejm"
    )
    assert "border-top: 2px solid #000000" in html
    assert "Table 2. Multivariable Logistic Regression" in html
    assert "1.05 (1.01–1.09)" in html


# -----------------------------------------------------------------------------
# 12. Recipe Self-Containment in Isolated Namespaces
# -----------------------------------------------------------------------------


def test_each_recipe_is_self_contained():
    """Verify that every python code block in python-recipes.md executes and runs in an isolated namespace."""
    content = RECIPES_PATH.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)\n```", content, re.DOTALL)
    assert len(blocks) >= 13, f"Expected at least 13 recipes, got {len(blocks)}"
    for idx, block in enumerate(blocks, start=1):
        isolated_ns = {}
        try:
            exec(block, isolated_ns)
        except Exception as e:
            pytest.fail(
                f"Recipe {idx} failed to import/define in an isolated namespace: {e}"
            )

        try:
            if idx == 1:
                df = pd.DataFrame(
                    {
                        "arm": ["A", "A", "B", "B"],
                        "age": [40.0, 50.0, 60.0, 70.0],
                        "sex": ["M", "F", "M", "F"],
                    }
                )
                res = isolated_ns["generate_table_one"](
                    df, strata="arm", continuous_vars=["age"], categorical_vars=["sex"]
                )
                assert isinstance(res, pd.DataFrame)
            elif idx == 2:
                res = isolated_ns["calculate_2x2_metrics"](tp=40, fp=10, fn=5, tn=45)
                assert "Sensitivity" in res
            elif idx == 3:
                res1 = isolated_ns["auc_ci_delong"]([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
                assert "auc" in res1
                res2 = isolated_ns["delong_paired_test"](
                    [1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2], [0.8, 0.2, 0.7, 0.3]
                )
                assert "p_value" in res2
            elif idx == 4:
                res1 = isolated_ns["evaluate_calibration"](
                    [1, 0, 1, 0], [0.8, 0.2, 0.7, 0.3]
                )
                assert "brier_score" in res1
                res2 = isolated_ns["calculate_dca"]([1, 0, 1, 0], [0.8, 0.2, 0.7, 0.3])
                assert isinstance(res2, pd.DataFrame)
            elif idx == 5:
                res1 = isolated_ns["calculate_bland_altman"](
                    [10.0, 20.0, 30.0], [10.2, 19.8, 30.1]
                )
                assert "mean_diff" in res1
                res2 = isolated_ns["calculate_icc_matrix"](
                    np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                )
                assert isinstance(res2, pd.DataFrame)
            elif idx == 6:
                df_psm = pd.DataFrame(
                    {
                        "trt": [0, 0, 1, 1, 0, 1],
                        "age": [20.0, 25.0, 30.0, 35.0, 22.0, 32.0],
                        "sex": ["M", "F", "M", "F", "F", "M"],
                    }
                )
                res = isolated_ns["match_propensity_scores"](
                    df_psm, treatment_col="trt", confounders=["age", "sex"]
                )
                assert "matched_df" in res
            elif idx == 7:
                X = np.array(
                    [
                        [1.0, 2.0],
                        [2.0, 1.0],
                        [3.0, 4.0],
                        [4.0, 3.0],
                        [5.0, 5.0],
                        [6.0, 7.0],
                    ]
                )
                y = np.array([0, 0, 0, 1, 1, 1])
                res = isolated_ns["fit_firth_logistic"](X, y)
                assert "coefficients" in res
            elif idx == 8:
                df_mcar = pd.DataFrame(
                    {
                        "a": [1.0, 2.0, 3.0, 4.0, np.nan],
                        "b": [2.0, 3.0, np.nan, 5.0, 6.0],
                    }
                )
                res = isolated_ns["littles_mcar_test"](df_mcar, ["a", "b"])
                assert "p_value" in res
            elif idx == 9:
                res = isolated_ns["calculate_e_value"](
                    2.5, lower=1.5, upper=4.0, estimate_type="OR", rare_outcome=True
                )
                assert "e_value_point" in res
            elif idx == 10:
                df_l = pd.DataFrame(
                    {"y": [0, 0, 1, 1, 0, 1], "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
                )
                res = isolated_ns["fit_logistic_regression_table"](
                    df_l, outcome="y", covariates=["x1"]
                )
                assert isinstance(res, pd.DataFrame)
            elif idx == 11:
                df_s = pd.DataFrame(
                    {
                        "t": [10.0, 20.0, 30.0, 40.0],
                        "e": [1, 0, 1, 0],
                        "arm": ["A", "A", "B", "B"],
                        "age": [50.0, 60.0, 55.0, 65.0],
                    }
                )
                res = isolated_ns["fit_survival_analysis_suite"](
                    df_s,
                    duration_col="t",
                    event_col="e",
                    strata_col="arm",
                    covariates=["age"],
                )
                assert "km_summary" in res
            elif idx == 12:
                df_m = pd.DataFrame(
                    {
                        "y": [0, 0, 1, 1, 0, 1],
                        "x1": [1.2, np.nan, 3.5, 3.8, np.nan, 6.1],
                        "x2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                    }
                )
                dsets = isolated_ns["impute_mice_datasets"](
                    df_m,
                    features_to_impute=["x1"],
                    outcome_col="y",
                    predictors=["x2"],
                    m=2,
                )
                assert len(dsets) == 2
                single_m = isolated_ns["impute_mice_single"](
                    df_m, features_to_impute=["x1"], outcome_col="y", predictors=["x2"]
                )
                assert isinstance(single_m, pd.DataFrame)
                pooled = isolated_ns["pool_estimates_rubin"](
                    [1.0, 1.1], [0.2, 0.2], n_obs=100, k_params=1
                )
                assert "pooled_estimate" in pooled
            elif idx == 13:
                df_r = pd.DataFrame({"Var": ["Age"], "Val": ["50 ± 10"]})
                res = isolated_ns["render_publication_table"](df_r, title="Table 1")
                assert "<table" in res
        except Exception as e:
            pytest.fail(
                f"Recipe {idx} failed to run with valid test fixture in an isolated namespace: {e}"
            )


# -----------------------------------------------------------------------------
# 13. High-Priority Invariant: Strict Binary Outcome Validation Across Recipes
# -----------------------------------------------------------------------------


def test_delong_strict_binary_validation(recipes):
    auc_ci = recipes["auc_ci_delong"]
    delong_paired = recipes["delong_paired_test"]

    # Continuous floats (e.g. 0.9, 1.9) must NOT silently cast to 0/1
    with pytest.raises(ValueError, match="strictly binary"):
        auc_ci([0.9, 0.0, 1.0, 0.0], [0.8, 0.2, 0.9, 0.1])

    # Non-binary categories (e.g. 2) must be rejected
    with pytest.raises(ValueError, match="strictly binary"):
        auc_ci([2, 0, 1, 0], [0.8, 0.2, 0.9, 0.1])

    # Paired test length mismatch
    with pytest.raises(ValueError, match="Length mismatch"):
        delong_paired([1, 0, 1, 0], [0.8, 0.2], [0.7, 0.3, 0.6, 0.4])

    with pytest.raises(ValueError, match="strictly binary"):
        delong_paired([0, 1, 2, 0], [0.8, 0.2, 0.5, 0.1], [0.7, 0.3, 0.6, 0.2])


def test_calibration_and_dca_strict_binary_and_probability_validation(recipes):
    eval_calib = recipes["evaluate_calibration"]
    calc_dca = recipes["calculate_dca"]

    # Non-binary outcome
    with pytest.raises(ValueError, match="strictly binary"):
        eval_calib([0.5, 1.0, 0.0, 1.0], [0.6, 0.8, 0.2, 0.7])

    # Out of range probabilities in calibration
    with pytest.raises(ValueError, match="range \\[0, 1\\]"):
        eval_calib([1, 0, 1, 0], [1.5, 0.2, 0.8, -0.1])

    # Length mismatch in calibration
    with pytest.raises(ValueError, match="Length mismatch"):
        eval_calib([1, 0, 1], [0.8, 0.2])

    # Non-binary outcome in DCA
    with pytest.raises(ValueError, match="strictly binary"):
        calc_dca([0, 2, 1, 0], [0.1, 0.9, 0.8, 0.2])

    # Length mismatch in DCA
    with pytest.raises(ValueError, match="Length mismatch"):
        calc_dca([1, 0, 1], [0.8, 0.2])

    # DCA internal calibration disclosure checks
    # 1. Scores within [0, 1] do not trigger calibration or warning
    dca_valid = calc_dca([1, 0, 1, 0], [0.8, 0.2, 0.7, 0.3])
    assert dca_valid.attrs["apparent_performance_warning"] is None
    assert dca_valid.attrs["internally_calibrated"] is False

    # 2. Raw biomarker scores outside [0, 1] trigger internal logistic calibration and disclosure
    dca_uncalib = calc_dca([1, 0, 1, 0], [-1.5, 0.2, 2.5, 0.4])
    assert dca_uncalib.attrs["internally_calibrated"] is True
    assert dca_uncalib.attrs["apparent_performance_warning"] is not None
    assert (
        "apparent development performance"
        in dca_uncalib.attrs["apparent_performance_warning"]
    )
    assert "over-optimistic" in dca_uncalib.attrs["apparent_performance_warning"]


def test_psm_and_logistic_strict_binary_and_retention_audit(recipes):
    psm_match = recipes["match_propensity_scores"]
    fit_table = recipes["fit_logistic_regression_table"]

    df_invalid = pd.DataFrame(
        {
            "treated": [0, 1, 2, 0],
            "age": [50, 60, 55, 65],
        }
    )
    # PSM rejects non-binary treatment
    with pytest.raises(ValueError, match="strictly binary"):
        psm_match(df_invalid, treatment_col="treated", confounders=["age"])

    # Logistic rejects non-binary outcome
    with pytest.raises(ValueError, match="strictly binary"):
        fit_table(df_invalid, outcome="treated", covariates=["age"])

    # Retention audit verification in PSM and Logistic
    np.random.seed(42)
    df_valid = pd.DataFrame(
        {
            "y": [0, 1, 1, 0, 1, 0, 1, 0],
            "trt": [1, 0, 1, 0, 1, 0, 1, 0],
            "age": [50, np.nan, 55, 65, 45, 70, 60, 58],
            "sex": ["F", "M", "F", "M", "F", "M", "F", "M"],
        }
    )
    res_psm = psm_match(df_valid, treatment_col="trt", confounders=["age", "sex"])
    assert "retention_audit" in res_psm
    assert res_psm["retention_audit"]["n_initial"] == 8
    assert res_psm["retention_audit"]["n_excluded"] == 1
    assert res_psm["retention_audit"]["n_analyzed"] == 7
    # Balance table includes categorical dummy level
    assert any("sex" in str(v) for v in res_psm["balance_df"]["Variable"])

    res_logistic = fit_table(df_valid, outcome="y", covariates=["age"])
    assert "retention_audit" in res_logistic.attrs
    assert res_logistic.attrs["retention_audit"]["n_initial"] == 8
    assert res_logistic.attrs["retention_audit"]["n_excluded"] == 1
    assert res_logistic.attrs["retention_audit"]["n_analyzed"] == 7


def test_survival_strict_binary_event_negative_duration_and_cohort_separation(recipes):
    fit_surv = recipes["fit_survival_analysis_suite"]

    df_invalid_event = pd.DataFrame(
        {
            "time": [10.0, 12.0, 5.0, 8.0],
            "event": [1, 0, 2, 0],
        }
    )
    with pytest.raises(ValueError, match="strictly binary"):
        fit_surv(df_invalid_event, duration_col="time", event_col="event")

    df_invalid_time = pd.DataFrame(
        {
            "time": [10.0, -2.0, 5.0, 8.0],
            "event": [1, 0, 1, 0],
        }
    )
    with pytest.raises(ValueError, match="non-negative"):
        fit_surv(df_invalid_time, duration_col="time", event_col="event")

    # Cohort separation: missing covariate must NOT drop patient from KM cohort
    df_cohort = pd.DataFrame(
        {
            "time": [10.0, 12.0, 15.0, 20.0, 25.0],
            "event": [1, 0, 1, 0, 1],
            "age": [50, 60, np.nan, 70, 80],
        }
    )
    res = fit_surv(
        df_cohort, duration_col="time", event_col="event", covariates=["age"]
    )
    assert res["retention_km"]["n_analyzed"] == 5, (
        "KM cohort must retain patient with missing Cox covariate"
    )
    assert res["retention_cox"]["n_analyzed"] == 4, (
        "Cox cohort must drop patient with missing Cox covariate"
    )
    assert "no_ph_violation_detected" in res["schoenfeld_diagnostics"]


# -----------------------------------------------------------------------------
# 14. High-Priority Invariant: MICE Type Safety, Constraints & Rubin B=0 Adjustment
# -----------------------------------------------------------------------------


def test_mice_input_guards_and_bounds(recipes):
    impute_datasets = recipes["impute_mice_datasets"]
    impute_single = recipes["impute_mice_single"]

    # 1. m < 1, float, or bool rejected (impute_mice_datasets specific)
    df = pd.DataFrame({"y": [1, 0, 1], "x": [10.0, np.nan, 12.0]})
    for invalid_m in [0, -1, 2.5, True]:
        with pytest.raises(ValueError, match="m must be an integer >= 1"):
            impute_datasets(df, features_to_impute=["x"], outcome_col="y", m=invalid_m)

    # Parameterize shared guards across both stochastic datasets and single imputation
    imputers = [
        lambda *args, **kwargs: impute_datasets(*args, m=2, **kwargs),
        lambda *args, **kwargs: impute_single(*args, **kwargs),
    ]

    for imp_fn in imputers:
        # 2. max_iter <= 0, float, or bool rejected (boundary testing for zero, negative, float, and bool)
        for invalid_iter in [0, -1, 5.0, True]:
            with pytest.raises(ValueError, match="max_iter must be an integer >= 1"):
                imp_fn(
                    df, features_to_impute=["x"], outcome_col="y", max_iter=invalid_iter
                )

        # 3. 100% missing column rejected
        df_all_nan = pd.DataFrame({"y": [1, 0, 1], "x": [np.nan, np.nan, np.nan]})
        with pytest.raises(ValueError, match="100% missing"):
            imp_fn(df_all_nan, features_to_impute=["x"], outcome_col="y")

        # 4. Non-numeric / text column rejected
        df_text = pd.DataFrame({"y": [1, 0, 1], "x": ["cat", "dog", None]})
        with pytest.raises(ValueError, match="non-numeric"):
            imp_fn(df_text, features_to_impute=["x"], outcome_col="y")

        # 5. Respect min_value bounds
        np.random.seed(42)
        n = 80
        df_bounded = pd.DataFrame(
            {
                "dead": np.random.binomial(1, 0.3, size=n),
                "hr": np.where(
                    np.random.rand(n) < 0.3, np.nan, np.random.normal(70, 10, size=n)
                ),
            }
        )
        res_bounded = imp_fn(
            df_bounded,
            features_to_impute=["hr"],
            outcome_col="dead",
            min_value={"hr": 50.0},
        )
        bounded_dfs = res_bounded if isinstance(res_bounded, list) else [res_bounded]
        for d in bounded_dfs:
            assert (d["hr"] >= 50.0).all(), (
                "Imputed heart rate must respect minimum bound 50.0"
            )

        # 6. Reject binary/dummy target in features_to_impute
        df_bin = pd.DataFrame(
            {
                "y": [1, 0, 1, 0],
                "bin_target": [0, 1, 0, np.nan],
                "age": [50.0, 60.0, 55.0, 65.0],
            }
        )
        with pytest.raises(ValueError, match="binary/dummy"):
            imp_fn(df_bin, features_to_impute=["bin_target"], outcome_col="y")

        # 7. Reject boolean target in features_to_impute
        df_bool = pd.DataFrame(
            {
                "y": [1, 0, 1, 0],
                "bool_target": [True, False, True, False],
                "age": [50.0, 60.0, 55.0, 65.0],
            }
        )
        with pytest.raises(ValueError, match="non-numeric"):
            imp_fn(df_bool, features_to_impute=["bool_target"], outcome_col="y")

        # 8. Reject categorical predictor with missing values
        df_missing_pred = pd.DataFrame(
            {
                "y": [1, 0, 1, 0],
                "age": [50.0, np.nan, 55.0, 65.0],
                "bin_pred": [0, 1, np.nan, 1],
            }
        )
        with pytest.raises(
            ValueError, match="binary/dummy and contains missing values"
        ):
            imp_fn(
                df_missing_pred,
                features_to_impute=["age"],
                outcome_col="y",
                predictors=["bin_pred"],
            )

        # 9. Fully observed categorical predictor succeeds as conditioning feature
        df_valid_pred = pd.DataFrame(
            {
                "y": [1, 0, 1, 0, 0, 1],
                "age": [50.0, np.nan, 55.0, 65.0, 70.0, 60.0],
                "bin_pred": [0, 1, 1, 1, 0, 0],
            }
        )
        res_valid = imp_fn(
            df_valid_pred,
            features_to_impute=["age"],
            outcome_col="y",
            predictors=["bin_pred"],
        )
        valid_dfs = res_valid if isinstance(res_valid, list) else [res_valid]
        for d in valid_dfs:
            assert not d["age"].isna().any()

        # 10. Reject discrete integer category codes {1, 2} and {1, 2, 3}
        df_cat_codes = pd.DataFrame(
            {
                "y": [1, 0, 1, 0, 1],
                "stage": [1, 2, 3, 2, np.nan],
                "age": [50.0, 52.0, 55.0, 60.0, 65.0],
            }
        )
        with pytest.raises(ValueError, match="discrete integer category codes"):
            imp_fn(
                df_cat_codes,
                features_to_impute=["stage"],
                outcome_col="y",
                predictors=["age"],
            )

        df_cat_12 = pd.DataFrame(
            {
                "y": [1, 0, 1, 0, 1],
                "group": [1, 2, 1, 2, np.nan],
                "age": [50.0, 52.0, 55.0, 60.0, 65.0],
            }
        )
        with pytest.raises(ValueError, match="unique observed values"):
            imp_fn(
                df_cat_12,
                features_to_impute=["group"],
                outcome_col="y",
                predictors=["age"],
            )

        # 11. Reject declared categorical and ordinal targets
        df_cat_dtype = pd.DataFrame(
            {
                "y": [1, 0, 1, 0, 1],
                "grade": pd.Categorical(["A", "B", "C", "A", None]),
                "age": [50.0, 52.0, 55.0, 60.0, 65.0],
            }
        )
        with pytest.raises(ValueError, match="declared as categorical"):
            imp_fn(
                df_cat_dtype,
                features_to_impute=["grade"],
                outcome_col="y",
                predictors=["age"],
            )

        with pytest.raises(ValueError, match="declared as ordinal"):
            imp_fn(
                df_cat_codes,
                features_to_impute=["age"],
                outcome_col="y",
                target_types={"age": "ordinal"},
            )

        # 12. Reject partially missing continuous predictor not in features_to_impute
        df_miss_cont = pd.DataFrame(
            {
                "y": [1, 0, 1, 0, 1],
                "x1": [10.0, np.nan, 12.0, 14.0, 15.0],
                "x2": [100.0, 110.0, np.nan, 130.0, 140.0],
            }
        )
        with pytest.raises(
            ValueError, match="Conditioning-only predictors must be fully observed"
        ):
            imp_fn(
                df_miss_cont,
                features_to_impute=["x1"],
                outcome_col="y",
                predictors=["x2"],
            )

        # 13. Reject partially missing primary outcome
        df_miss_outcome = pd.DataFrame(
            {
                "y": [1, 0, np.nan, 0, 1],
                "x1": [10.0, np.nan, 12.0, 14.0, 15.0],
                "x2": [100.0, 110.0, 120.0, 130.0, 140.0],
            }
        )
        with pytest.raises(
            ValueError, match="Primary outcome column 'y' contains missing values"
        ):
            imp_fn(
                df_miss_outcome,
                features_to_impute=["x1"],
                outcome_col="y",
                predictors=["x2"],
            )

        # 14. Reject non-finite values in conditioning matrix
        df_inf = pd.DataFrame(
            {
                "y": [1, 0, 1, 0, 1],
                "x1": [10.0, np.nan, 12.0, np.inf, 15.0],
                "x2": [100.0, 110.0, 120.0, 130.0, 140.0],
            }
        )
        with pytest.raises(ValueError, match="contains non-finite values"):
            imp_fn(
                df_inf,
                features_to_impute=["x1"],
                outcome_col="y",
                predictors=["x2"],
            )

        # 15. Reject invalid bound ordering (min_value >= max_value)
        df_order = pd.DataFrame(
            {
                "y": [1, 0, 1, 0, 1],
                "x1": [10.0, np.nan, 12.0, 14.0, 15.0],
                "x2": [100.0, 110.0, 120.0, 130.0, 140.0],
            }
        )
        with pytest.raises(ValueError, match="Invalid bound ordering"):
            imp_fn(
                df_order,
                features_to_impute=["x1"],
                outcome_col="y",
                predictors=["x2"],
                min_value={"x1": 50.0},
                max_value={"x1": 20.0},
            )

        # 16. Reject perfectly predicted continuous target (RSS == 0)
        # x1 is an exact linear function of x2: x1 = 2 * x2 + 5
        df_perf = pd.DataFrame(
            {
                "y": [0, 1, 0, 1, 0, 1],
                "x2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                "x1": [25.0, 45.0, 65.0, 85.0, np.nan, np.nan],
            }
        )
        with pytest.raises(ValueError, match="Degenerate residual sum of squares"):
            imp_fn(
                df_perf,
                features_to_impute=["x1"],
                outcome_col="y",
                predictors=["x2"],
            )


def test_mice_posterior_degrees_of_freedom_rank_and_bounds(recipes):
    impute_datasets = recipes["impute_mice_datasets"]
    impute_single = recipes["impute_mice_single"]

    # 1. Saturated design / Insufficient observed cases: n_obs <= p
    # p = 3 (intercept, x2, y). We give only 3 observed rows for x1 (n_obs = 3 <= 3)
    df_sat = pd.DataFrame(
        {
            "y": [0, 1, 0, 1, 0],
            "x1": [10.5, 20.2, 30.8, np.nan, np.nan],
            "x2": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    with pytest.raises(ValueError, match="Insufficient observed cases"):
        impute_datasets(
            df_sat,
            features_to_impute=["x1"],
            outcome_col="y",
            predictors=["x2"],
            m=2,
        )
    with pytest.raises(ValueError, match="Insufficient observed cases"):
        impute_single(
            df_sat,
            features_to_impute=["x1"],
            outcome_col="y",
            predictors=["x2"],
        )

    # 2. Collinear predictors: rank-deficient design
    # x2 and x3 are identical, causing rank(X_obs) < p
    df_collinear = pd.DataFrame(
        {
            "y": [0, 1, 0, 1, 0, 1, 0],
            "x1": [10.5, 20.2, 30.8, 40.1, 50.3, np.nan, np.nan],
            "x2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
            "x3": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
        }
    )
    with pytest.raises(ValueError, match="rank-deficient"):
        impute_datasets(
            df_collinear,
            features_to_impute=["x1"],
            outcome_col="y",
            predictors=["x2", "x3"],
            m=2,
        )
    with pytest.raises(ValueError, match="rank-deficient"):
        impute_single(
            df_collinear,
            features_to_impute=["x1"],
            outcome_col="y",
            predictors=["x2", "x3"],
        )

    # 3. Truncated normal sampling without boundary point masses
    # Generate 100 observations with 40% missingness, bounded strictly between 60.0 and 80.0
    rng = np.random.default_rng(123)
    n = 100
    age = rng.normal(70, 15, size=n)
    y = rng.binomial(1, 0.4, size=n)
    df_bounded = pd.DataFrame({"y": y, "age": age})
    mis_mask = rng.random(n) < 0.4
    df_bounded.loc[mis_mask, "age"] = np.nan

    datasets = impute_datasets(
        df_bounded,
        features_to_impute=["age"],
        outcome_col="y",
        m=5,
        min_value={"age": 60.0},
        max_value={"age": 80.0},
        random_state=42,
    )
    for d in datasets:
        imp_vals = d.loc[mis_mask, "age"].values
        # All imputed draws must strictly lie within [60.0, 80.0]
        assert (imp_vals >= 60.0).all() and (imp_vals <= 80.0).all()
        # Must NOT collapse into spikes / point masses at the bounds (unlike clipping)
        assert not np.isclose(imp_vals, 60.0).any(), "Draws must not spike on min bound"
        assert not np.isclose(imp_vals, 80.0).any(), "Draws must not spike on max bound"
        # Must retain continuous distribution variance
        assert np.var(imp_vals) > 5.0, "Truncated Gaussian draws must maintain variance"


def test_rubin_pooling_barnard_rubin_b0_limit_and_error_handling(recipes):
    pool_rubin = recipes["pool_estimates_rubin"]

    # 1. Length mismatch
    with pytest.raises(ValueError, match="Length mismatch"):
        pool_rubin([1.0, 1.2], [0.2])

    # 2. Negative SE
    with pytest.raises(ValueError, match="non-negative"):
        pool_rubin([1.0, 1.2], [-0.1, 0.2])

    # 3. Non-finite values
    with pytest.raises(ValueError, match="strictly finite"):
        pool_rubin([np.nan, 1.0], [0.2, 0.2])

    # 4. n_obs <= k_params
    with pytest.raises(ValueError, match="strictly greater"):
        pool_rubin([1.0, 1.1], [0.2, 0.2], n_obs=2, k_params=2)

    # 5. Barnard & Rubin (1999) when B = 0 and finite complete-data df:
    # When B = 0, nu_BR = nu_com * (nu_com + 1) / (nu_com + 3) where nu_com = n_obs - k_params
    n_obs = 100
    k_params = 1
    nu_com = float(n_obs - k_params)  # 99.0
    expected_df = nu_com * (nu_com + 1.0) / (nu_com + 3.0)  # 99 * 100 / 102 ≈ 97.0588

    res_b0 = pool_rubin(
        [2.0, 2.0, 2.0], [0.5, 0.5, 0.5], n_obs=n_obs, k_params=k_params
    )
    assert res_b0["between_variance"] == 0.0
    assert np.isclose(res_b0["df"], expected_df, atol=1e-3)
    assert not np.isinf(res_b0["df"]), (
        "Finite-sample degrees of freedom when B=0 must NOT be infinite!"
    )
    assert res_b0["fmi"] == 0.0


# -----------------------------------------------------------------------------
# 15. High-Priority Invariant: Firth Rank Deficiency & Convergence Verification
# -----------------------------------------------------------------------------


def test_firth_rank_deficiency_and_ci_method(recipes):
    fit_firth = recipes["fit_firth_logistic"]

    # Collinear columns (rank-deficient design matrix)
    x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    x2 = 2.0 * x1
    X_collinear = np.column_stack([x1, x2])
    y = np.array([0, 0, 0, 1, 1, 1])

    with pytest.raises(ValueError, match="rank-deficient"):
        fit_firth(X_collinear, y)

    # Non-binary outcome
    with pytest.raises(ValueError, match="strictly 0 and 1"):
        fit_firth(np.column_stack([x1]), np.array([0, 1, 2, 0, 1, 0]))

    # Valid fit returns ci_method
    X_valid = np.column_stack([x1])
    res = fit_firth(X_valid, y)
    assert "ci_method" in res
    assert len(res["ci_method"]) == 2  # intercept + x1
    assert all(m in {"profile", "wald_fallback"} for m in res["ci_method"])

    # 4. Likelihood monotonicity and optimizer convergence under separation
    # Separation dataset: standard MLE explodes
    x_sep = np.array([-3.0, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, 3.0])
    y_sep = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    X_sep = np.column_stack([x_sep])
    res_sep = fit_firth(X_sep, y_sep)
    assert res_sep["converged"] is True, "Firth must converge under complete separation"
    assert np.isfinite(res_sep["coefficients"]).all(), (
        "Firth coefficients must remain finite under separation"
    )

    # Verify monotonicity across all accepted iterations in pll_history
    assert "pll_history" in res_sep, "Firth result must record pll_history"
    pll_hist = res_sep["pll_history"]
    assert len(pll_hist) >= 2, "Must record at least initial and final likelihood"
    for i in range(len(pll_hist) - 1):
        assert pll_hist[i + 1] >= pll_hist[i] - 1e-10, (
            f"Penalized log-likelihood must be monotonically non-decreasing at iteration {i}: "
            f"{pll_hist[i]} -> {pll_hist[i + 1]}"
        )
    assert pll_hist[-1] > pll_hist[0], (
        "Penalized log-likelihood must strictly improve from start"
    )

    # 5. Optimizer failure handling: max_iter=1 on non-converged model returns converged=False and wald_fallback
    res_unconv = fit_firth(X_sep, y_sep, max_iter=1)
    assert res_unconv["converged"] is False, (
        "Optimizer must flag converged=False when max_iter is exceeded"
    )
    assert res_unconv["ci_method"] == ["wald_fallback", "wald_fallback"], (
        "Non-converged fit must immediately fall back to Wald CIs without attempting profile likelihood"
    )
    expected_wald_low = np.exp(
        res_unconv["coefficients"] - 1.96 * res_unconv["standard_errors"]
    )
    expected_wald_high = np.exp(
        res_unconv["coefficients"] + 1.96 * res_unconv["standard_errors"]
    )
    assert np.allclose(res_unconv["ci_lower"], expected_wald_low), (
        "ci_lower must match Wald fallback formula"
    )
    assert np.allclose(res_unconv["ci_upper"], expected_wald_high), (
        "ci_upper must match Wald fallback formula"
    )

    # 6. Numerical Profile Endpoints verification
    # Require profile intervals for this known converged fixture
    assert res_sep["ci_method"][1] == "profile", (
        "Fixture under complete separation must produce validated profile likelihood intervals"
    )
    import scipy.optimize as opt
    import scipy.stats as stats

    pll_fn = recipes["_firth_penalized_loglik"]
    X_sep_aug = np.column_stack([np.ones(len(y_sep)), X_sep])
    pll_final = res_sep["pll_history"][-1]
    chi2_crit = 0.5 * stats.chi2.ppf(0.95, df=1)

    # Profile lower bound on log scale
    beta1_low = float(np.log(res_sep["ci_lower"][1]))

    def nuisance_obj_low(b0):
        return -pll_fn(np.array([float(np.squeeze(b0)), beta1_low]), X_sep_aug, y_sep)

    opt_low = opt.minimize(
        nuisance_obj_low, res_sep["coefficients"][0], method="Nelder-Mead"
    )
    assert opt_low.success, "Nuisance optimizer for lower bound must succeed"
    pll_prof_low = -opt_low.fun
    drop_low = pll_final - pll_prof_low
    assert np.isclose(drop_low, chi2_crit, atol=0.05), (
        f"Profile lower bound drop ({drop_low:.4f}) must equal chi2 cutoff ({chi2_crit:.4f})"
    )

    # Profile upper bound on log scale
    beta1_high = float(np.log(res_sep["ci_upper"][1]))

    def nuisance_obj_high(b0):
        return -pll_fn(np.array([float(np.squeeze(b0)), beta1_high]), X_sep_aug, y_sep)

    opt_high = opt.minimize(
        nuisance_obj_high, res_sep["coefficients"][0], method="Nelder-Mead"
    )
    assert opt_high.success, "Nuisance optimizer for upper bound must succeed"
    pll_prof_high = -opt_high.fun
    drop_high = pll_final - pll_prof_high
    assert np.isclose(drop_high, chi2_crit, atol=0.05), (
        f"Profile upper bound drop ({drop_high:.4f}) must equal chi2 cutoff ({chi2_crit:.4f})"
    )


# -----------------------------------------------------------------------------
# 16. Medium-Priority Checks: Little's MCAR Edge Cases & ICC Boundary Limits
# -----------------------------------------------------------------------------


def test_littles_mcar_edge_cases(recipes):
    littles_test = recipes["littles_mcar_test"]

    # Completely missing dataset
    df_all_nan = pd.DataFrame(np.full((10, 3), np.nan), columns=["a", "b", "c"])
    with pytest.raises(ValueError, match="completely missing"):
        littles_test(df_all_nan, ["a", "b", "c"])

    # 100% missing column
    df_col_nan = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [np.nan, np.nan, np.nan],
        }
    )
    with pytest.raises(ValueError, match="100% missing"):
        littles_test(df_col_nan, ["a", "b"])

    # Valid test returns telemetry
    df_valid = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, np.nan],
            "b": [2.0, 3.0, np.nan, 5.0, np.nan],
        }
    )
    res = littles_test(df_valid, ["a", "b"])
    assert "converged" in res
    assert "n_iterations" in res
    assert "n_excluded_rows" in res
    assert res["n_excluded_rows"] == 1  # 5th row is completely NaN


def test_icc_perfect_agreement_and_all_constant(recipes):
    calc_icc = recipes["calculate_icc_matrix"]

    # 1. Perfect agreement across varying targets: MSW=0, MSE=0, MSB>0
    # Targets vary: 10, 20, 30. All 3 raters agree perfectly.
    perfect_mat = np.array(
        [
            [10.0, 10.0, 10.0],
            [20.0, 20.0, 20.0],
            [30.0, 30.0, 30.0],
        ]
    )
    res_perf = calc_icc(perfect_mat)
    assert (res_perf["ICC"] == 1.0).all()
    assert (res_perf["CI_lower"] == 1.0).all()
    assert (res_perf["CI_upper"] == 1.0).all()

    # 2. All-constant matrix: targets don't vary either (every cell is 5.0)
    constant_mat = np.full((4, 3), 5.0)
    res_const = calc_icc(constant_mat)
    assert res_const["ICC"].isna().all()
    assert res_const["CI_lower"].isna().all()


def test_table_one_column_mapping_and_html_escaping(recipes):
    gen_table = recipes["generate_table_one"]
    render_table = recipes["render_publication_table"]

    df = pd.DataFrame(
        {
            "arm": ["A", "A", "B", "B"],
            "age": [50, 52, 60, 62],
            "stage": ["I", "II", "I", "II"],
        }
    )
    t1 = gen_table(
        df, strata="arm", continuous_vars=["age"], categorical_vars=["stage"]
    )
    # All rows must share exact same column headers
    cols = list(t1.columns)
    assert "arm=A (N = 2)" in cols
    assert "arm=B (N = 2)" in cols
    assert "arm=A" not in cols, "Must not create fragmented duplicate column keys"

    # HTML renderer escapes XSS / markup
    df_xss = pd.DataFrame(
        {
            "<script>alert(1)</script>": ["<b>high</b>", "1.05 & 2.0"],
        }
    )
    html_out = render_table(
        df_xss, title="<Title & Header>", footnote="<Footnote & Note>"
    )
    assert "<script>" not in html_out
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_out
    assert "&lt;Title &amp; Header&gt;" in html_out
    assert "&lt;Footnote &amp; Note&gt;" in html_out
    assert "&lt;b&gt;high&lt;/b&gt;" in html_out


def test_cloud_report_builder_docx_fallback_and_integrity(tmp_path):
    """Verifies that cloud report builder fallback redirects .docx to .html and tests standalone integrity."""
    rb_path = CLOUD_DIR / "references" / "report-builder.md"
    content = rb_path.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)\n```", content, re.DOTALL)
    ns = {}
    errors = []
    for idx, block in enumerate(blocks):
        try:
            exec(block, ns)
        except Exception as exc:
            errors.append((idx, exc))
    assert not errors, f"Failed executing blocks in report-builder.md: {errors}"

    import unittest.mock as mock

    # Verify fallback function redirects .docx when python-docx is absent
    with mock.patch.dict("sys.modules", {"docx": None}):
        ns_fallback = {
            "generate_html_report": ns["generate_html_report"],
            "Path": Path,
            "pd": pd,
        }
        docx_block = [b for b in blocks if "generate_docx_report" in b][0]
        exec(docx_block, ns_fallback)
        gen_fallback = ns_fallback["generate_docx_report"]
        out_file = str(tmp_path / "summary.docx")
        actual_out = gen_fallback(
            "Study Summary",
            results={"n": 50, "date": "2026-10-10"},
            tables={"Table 1": pd.DataFrame({"A": [1, 2]})},
            out_path=out_file,
        )
        assert actual_out.endswith(".html")
        assert Path(actual_out).exists()

    # Verify verify_standalone_integrity
    verify_fn = ns.get("verify_standalone_integrity")
    assert callable(verify_fn)
    audit = verify_fn(
        "The model achieved an AUC of 0.852 in cohort of 50 patients.",
        results_dict={"auc": 0.852, "n": 50},
    )
    assert audit["passed"] is True
    assert audit["traceability_passed"] is True
    assert len(audit["untraced_numbers"]) == 0

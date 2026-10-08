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

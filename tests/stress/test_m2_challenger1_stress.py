"""
tests/stress/test_m2_challenger1_stress.py: Empirical Stress Harness for Milestone M2.

Author: Challenger 1 (Empirical Challenger)
Scope:
1. Little's MCAR Test:
   - Statistical sensitivity under true MCAR vs MAR vs MNAR.
   - Micro-datasets: N < P, collinear variables, all-complete data (df <= 0),
     single missingness pattern, massive missingness (>80%), ridge regularization.
2. Imputation & Rubin's Rules:
   - MICE convergence with extreme missingness, high collinearity, zero-variance columns.
   - Rubin's rules pooling with m=2, 5, 20, testing Barnard-Rubin small-sample adjusted df
     vs standard infinite-sample df.
   - Standardized KNN vs unstandardized KNN: empirical proof of numerical scale distortion.
3. Outlier Winsorization:
   - Heavy-tailed Cauchy distributions (masking effect under Z-score vs robust IQR/MAD).
   - Log-normal skewed distributions.
   - Discrete integers (zero-inflated count data and the MAD = 0 collapse phenomenon).
"""

import numpy as np
import pandas as pd
import pytest

from medstat.data.clean import (
    LittlesMCARResult,
    detect_outliers,
    handle_outliers,
    littles_mcar_test,
)
from medstat.data.missing import (
    MICEImputer,
    PooledEstimate,
    impute_knn,
    pool_estimates,
    pool_regression_results,
)

pytestmark = pytest.mark.unit


# =============================================================================
# 1. Little's MCAR Test: Sensitivity, Power & Micro-Datasets
# =============================================================================


class TestLittlesMCARStress:
    """Adversarial stress-testing of Roderick Little's (1988) MCAR test."""

    def test_littles_mcar_null_hypothesis_true_mcar(self):
        """Under true MCAR, Little's test must fail to reject H0 (p > 0.05)."""
        np.random.seed(42)
        N, P = 500, 4
        # Generate correlated multivariate Gaussian data
        mean = [10.0, 50.0, 100.0, 25.0]
        cov = [
            [4.0, 2.0, 1.0, 0.5],
            [2.0, 9.0, 3.0, 1.0],
            [1.0, 3.0, 16.0, 2.0],
            [0.5, 1.0, 2.0, 5.0],
        ]
        X = np.random.multivariate_normal(mean=mean, cov=cov, size=N)
        df_mcar = pd.DataFrame(X, columns=["x1", "x2", "x3", "x4"])

        # Completely Random Missingness (Bernoulli trials with p=0.15)
        mask = np.random.rand(N, P) < 0.15
        df_mcar[mask] = np.nan

        res = littles_mcar_test(df_mcar)

        assert isinstance(res, LittlesMCARResult)
        assert res.em_converged is True
        assert res.df > 0
        assert res.statistic >= 0.0
        # Under true MCAR, p-value should not reject at alpha=0.05
        assert res.p_value > 0.05, f"False positive under MCAR: p={res.p_value:.4f}"
        assert res.is_mcar is True

    def test_littles_mcar_alternative_hypothesis_mar(self):
        """Under MAR (missingness depends on observed covariates), test must reject H0 (p < 0.05)."""
        np.random.seed(42)
        N = 500
        X = np.random.multivariate_normal(
            mean=[0, 0, 0, 0],
            cov=[
                [1, 0.5, 0.3, 0.2],
                [0.5, 1, 0.4, 0.3],
                [0.3, 0.4, 1, 0.5],
                [0.2, 0.3, 0.5, 1],
            ],
            size=N,
        )
        df_mar = pd.DataFrame(X, columns=["x1", "x2", "x3", "x4"])

        # Missingness in x2 is driven by observed x1; missingness in x4 is driven by observed x3
        prob_x2_missing = 1.0 / (1.0 + np.exp(-2.5 * df_mar["x1"]))
        prob_x4_missing = 1.0 / (1.0 + np.exp(-2.5 * df_mar["x3"]))

        df_mar.loc[np.random.rand(N) < prob_x2_missing, "x2"] = np.nan
        df_mar.loc[np.random.rand(N) < prob_x4_missing, "x4"] = np.nan

        res = littles_mcar_test(df_mar)

        assert res.em_converged is True
        # Little's test should decisively detect the systematic mean departure between patterns
        assert res.p_value < 0.01, f"Failed to reject H0 under MAR: p={res.p_value:.4e}"
        assert res.is_mcar is False
        assert res.statistic > res.df  # High chi-square statistic

    def test_littles_mcar_alternative_hypothesis_mnar(self):
        """Under MNAR (missingness depends on unobserved values), test should reject H0 (p < 0.05)."""
        np.random.seed(42)
        N = 500
        X = np.random.multivariate_normal(
            mean=[0, 0, 0, 0],
            cov=[
                [1, 0.6, 0.4, 0.2],
                [0.6, 1, 0.5, 0.3],
                [0.4, 0.5, 1, 0.4],
                [0.2, 0.3, 0.4, 1],
            ],
            size=N,
        )
        df_mnar = pd.DataFrame(X, columns=["x1", "x2", "x3", "x4"])

        # Severe MNAR: high values of x2 are missing with high probability
        prob_x2_missing = 1.0 / (1.0 + np.exp(-3.0 * df_mnar["x2"]))
        df_mnar.loc[np.random.rand(N) < prob_x2_missing, "x2"] = np.nan

        res = littles_mcar_test(df_mnar)

        assert res.em_converged is True
        assert res.p_value < 0.01, f"Failed to detect MNAR: p={res.p_value:.4e}"
        assert res.is_mcar is False

    def test_littles_mcar_micro_dataset_n_less_than_p(self):
        """Micro-dataset with N < P: must not crash, ridge regularizer guarantees inversion."""
        np.random.seed(42)
        N, P = 5, 10
        X = np.random.randn(N, P)
        df_np = pd.DataFrame(X, columns=[f"v{i}" for i in range(P)])
        df_np.iloc[0, 1] = np.nan
        df_np.iloc[1, 3] = np.nan
        df_np.iloc[2, 5] = np.nan

        res = littles_mcar_test(df_np)
        assert isinstance(res, LittlesMCARResult)
        assert np.isfinite(res.statistic)
        assert np.isfinite(res.p_value)
        assert res.n_variables == 10
        assert res.n_observations == 5

    def test_littles_mcar_collinear_variables(self):
        """Perfect collinearity: covariance matrix is singular without ridge regularization."""
        x = np.linspace(1, 20, 30)
        df = pd.DataFrame(
            {
                "a": x,
                "b": 2.0 * x,
                "c": 3.0 * x - 1.0,
                "d": -1.5 * x + 4.0,
            }
        )
        # Introduce missing values across rows
        df.iloc[2:5, 0] = np.nan
        df.iloc[8:12, 1] = np.nan

        res = littles_mcar_test(df)
        assert np.isfinite(res.statistic)
        assert np.isfinite(res.p_value)
        assert res.em_converged is True

    def test_littles_mcar_all_complete_data_df_zero(self):
        """All-complete data has df=0; test must return p=1.0 and is_mcar=True without error."""
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "b": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )
        res = littles_mcar_test(df)
        assert res.df == 0
        assert res.statistic == 0.0
        assert res.p_value == 1.0
        assert res.is_mcar is True
        assert res.n_patterns == 1

    def test_littles_mcar_df_less_equal_zero_with_missingness(self):
        """Defensive handling when degrees of freedom df = sum(p_s) - P <= 0 with missing values."""
        # 3 variables: A, B, C.
        # Pattern 1: A observed, B & C missing -> p_1 = 1
        # Pattern 2: B & C observed, A missing -> p_2 = 2
        # Sum p_s = 1 + 2 = 3. Total P = 3. df = 3 - 3 = 0.
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, np.nan, np.nan],
                "b": [np.nan, np.nan, 3.0, 4.0],
                "c": [np.nan, np.nan, 5.0, 6.0],
            }
        )
        res = littles_mcar_test(df)
        assert res.df == 0
        assert res.statistic == 0.0
        assert res.p_value == 1.0
        assert res.is_mcar is True
        assert "Degrees of freedom" in res.message

    def test_littles_mcar_single_missingness_pattern(self):
        """Dataset with exactly 2 patterns (all-observed and 1 partial pattern)."""
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "b": [10.0, 20.0, 30.0, np.nan, np.nan, np.nan],
            }
        )
        res = littles_mcar_test(df)
        assert res.n_patterns == 2
        # p_1 = 2 (complete), p_2 = 1 (only a observed). Total p_s = 3. P = 2. df = 3 - 2 = 1.
        assert res.df == 1
        assert np.isfinite(res.statistic)
        assert 0.0 <= res.p_value <= 1.0

    def test_littles_mcar_massive_missingness(self):
        """Dataset with massive missingness (>80%) across variables."""
        np.random.seed(42)
        X = np.random.randn(100, 4)
        df_mass = pd.DataFrame(X, columns=["c1", "c2", "c3", "c4"])
        mask = np.random.rand(100, 4) < 0.85
        df_mass[mask] = np.nan

        # Ensure no column is 100% missing
        for col in df_mass.columns:
            if df_mass[col].isna().all():
                df_mass.loc[0, col] = 0.0

        res = littles_mcar_test(df_mass)
        assert np.isfinite(res.statistic)
        assert np.isfinite(res.p_value)
        # In Little's MCAR formulation, rows with p_s = 0 (all variables missing) carry no
        # information to compare against mu and Sigma and are dropped from pattern evaluation.
        assert 0 < res.n_observations <= 100


# =============================================================================
# 2. Imputation & Rubin's Rules Pooling
# =============================================================================


class TestImputationAndRubinsRulesStress:
    """Stress tests for MICE, KNN standardization, and Rubin's pooling."""

    def test_mice_convergence_extreme_missingness(self):
        """MICE must converge and eliminate all NaNs even with 80% missingness."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "x1": np.random.randn(n),
                "x2": np.random.randn(n) * 2.0,
                "x3": np.random.randn(n) + 1.0,
            }
        )
        df[np.random.rand(n, 3) < 0.80] = np.nan
        for col in df.columns:
            df.loc[0, col] = 1.0  # At least one observation per column

        imputer = MICEImputer(n_imputations=3, max_iter=10, random_state=42)
        res = imputer.fit_transform(df)

        assert res.n_imputations == 3
        for imp_df in res.imputed_datasets:
            assert imp_df.isna().sum().sum() == 0
            assert np.all(np.isfinite(imp_df.to_numpy()))

    def test_mice_high_collinearity(self):
        """MICE with collinear predictors must converge without LinAlgError."""
        np.random.seed(42)
        x = np.random.randn(80)
        df = pd.DataFrame(
            {
                "x1": x,
                "x2": 2.0 * x,
                "x3": 3.0 * x + 0.5,
            }
        )
        df.loc[0:15, "x1"] = np.nan
        df.loc[10:25, "x2"] = np.nan

        imputer = MICEImputer(n_imputations=3, max_iter=5, random_state=42)
        res = imputer.fit_transform(df)
        for imp_df in res.imputed_datasets:
            assert imp_df.isna().sum().sum() == 0

    def test_mice_zero_variance_columns(self):
        """MICE with constant (zero-variance) columns: both with and without missingness."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "const_clean": [5.0] * 80,
                "const_missing": [10.0] * 70 + [np.nan] * 10,
                "var_clean": np.random.randn(80),
            }
        )

        imputer = MICEImputer(n_imputations=2, max_iter=5, random_state=42)
        res = imputer.fit_transform(df)

        for imp_df in res.imputed_datasets:
            assert imp_df.isna().sum().sum() == 0
            # Const missing should be imputed near 10.0
            imputed_const = imp_df.loc[70:79, "const_missing"]
            assert np.allclose(imputed_const, 10.0, atol=0.1)

    @pytest.mark.parametrize("m", [2, 5, 20])
    def test_rubins_rules_barnard_rubin_vs_infinite_df(self, m: int):
        """
        Mathematical verification of Barnard-Rubin (1999) adjusted df:
        1. Small sample size (n_obs=30) yields nu_adj < nu_old (infinite-sample df).
        2. Adjusted df produces a strictly wider, more conservative 95% CI.
        """
        np.random.seed(42)
        true_theta = 2.0
        estimates = np.random.normal(true_theta, scale=0.3, size=m).tolist()
        variances = [0.25] * m

        res_inf = pool_estimates(estimates, variances, n_obs=None)
        res_adj = pool_estimates(estimates, variances, n_obs=30)

        assert isinstance(res_inf, PooledEstimate)
        assert isinstance(res_adj, PooledEstimate)

        # 1. Barnard-Rubin adjusted df must be strictly smaller than infinite-sample df
        assert res_adj.df < res_inf.df, (
            f"Expected nu_adj ({res_adj.df}) < nu_old ({res_inf.df})"
        )

        # 2. Confidence interval width must be strictly larger under small-sample adjustment
        width_inf = res_inf.ci_upper - res_inf.ci_lower
        width_adj = res_adj.ci_upper - res_adj.ci_lower
        assert width_adj > width_inf, (
            f"Expected wider CI under small sample df: {width_adj} vs {width_inf}"
        )

        # 3. Fraction of Missing Information (FMI) and lambda must be in [0, 1]
        assert 0.0 <= res_adj.fmi <= 1.0
        assert 0.0 <= res_adj.lambda_ <= 1.0

    def test_rubins_rules_identical_estimates_zero_between_variance(self):
        """When between-imputation variance B = 0, T = W_bar and df = inf."""
        estimates = [1.5, 1.5, 1.5]
        variances = [0.04, 0.04, 0.04]

        res = pool_estimates(estimates, variances)
        assert res.between_variance == 0.0
        assert res.total_variance == 0.04
        assert res.se == 0.2
        assert res.relative_increase_variance == 0.0
        assert res.fmi == 0.0

    def test_rubins_rules_zero_within_variance_edge_case(self):
        """Edge case: W_bar == 0, B > 0. Must not divide by zero or crash."""
        estimates = [1.0, 1.2, 1.4]
        variances = [0.0, 0.0, 0.0]

        res = pool_estimates(estimates, variances)
        assert res.within_variance == 0.0
        assert res.between_variance > 0.0
        assert res.total_variance > 0.0
        assert np.isfinite(res.se)

    def test_pool_regression_logistic_and_cox(self):
        """Verify pooling across multiple imputations for logistic (OR) and Cox (HR) models."""

        class MockRegressionModel:
            def __init__(self, coefs: list[float], ses: list[float]):
                self.params = pd.Series(coefs, index=["treatment", "age"])
                self.bse = pd.Series(ses, index=["treatment", "age"])
                self.nobs = 150

        # m = 3 imputed models
        models = [
            MockRegressionModel([0.6931, 0.05], [0.15, 0.02]),  # beta_trt ~ ln(2)
            MockRegressionModel([0.7200, 0.048], [0.16, 0.021]),
            MockRegressionModel([0.6500, 0.052], [0.14, 0.019]),
        ]

        # 1. Logistic pooling with Odds Ratios
        res_log = pool_regression_results(models, model_type="logistic")
        assert res_log.odds_ratios is not None
        assert "treatment" in res_log.odds_ratios
        or_val, or_low, or_high = res_log.odds_ratios["treatment"]
        # OR should be approx exp(0.6877) ~ 1.99
        assert 1.8 < or_val < 2.2
        assert or_low < or_val < or_high

        # 2. Cox pooling with Hazard Ratios
        res_cox = pool_regression_results(models, model_type="cox")
        assert res_cox.hazard_ratios is not None
        assert "treatment" in res_cox.hazard_ratios
        hr_val, hr_low, hr_high = res_cox.hazard_ratios["treatment"]
        assert 1.8 < hr_val < 2.2
        assert hr_low < hr_val < hr_high

    def test_knn_standardization_prevents_scale_distortion(self):
        """
        Empirical demonstration of KNN scale distortion:
        Feature 'Age' is clinical scale (std ~ 18), 'Income' is currency scale (std ~ 50,000).
        Target patient has Age 25, Income 50,000.
        Candidate 1 (True Peer): Age 26 (diff = 1), Income 52,000 (diff = 2,000). Outcome = 10.0
        Candidate 2 (Distorted): Age 75 (diff = 50!), Income 50,050 (diff = 50). Outcome = 90.0

        Unstandardized KNN chooses Candidate 2 (Outcome = 90.0) because 50 << 2000 in raw scale.
        Standardized KNN chooses Candidate 1 (Outcome = 10.0) because Age diff (0.05 SD) << 2.8 SD.
        """
        np.random.seed(42)
        n = 25
        ages = np.array([25.0, 26.0, 75.0] + list(np.random.uniform(20, 80, n - 3)))
        incomes = np.array(
            [50000.0, 52000.0, 50050.0] + list(np.random.uniform(20000, 200000, n - 3))
        )
        outcomes = np.array(
            [np.nan, 10.0, 90.0] + list(np.random.uniform(10, 90, n - 3))
        )

        df = pd.DataFrame({"Age": ages, "Income": incomes, "Outcome": outcomes})

        # Unstandardized KNN
        res_unscaled = impute_knn(df, n_neighbors=1, scale=False)
        # Standardized KNN
        res_scaled = impute_knn(df, n_neighbors=1, scale=True)

        # Unscaled is distorted by currency magnitude and picks Candidate 2
        assert res_unscaled.loc[0, "Outcome"] == 90.0
        # Scaled equalizes variance across dimensions and picks clinical peer Candidate 1
        assert res_scaled.loc[0, "Outcome"] == 10.0


# =============================================================================
# 3. Outlier Winsorization: Heavy Tails, Skew & Discrete Zero-Inflation
# =============================================================================


class TestOutlierWinsorizationStress:
    """Stress tests for outlier detection and winsorization across distribution shapes."""

    def test_cauchy_heavy_tailed_masking_effect(self):
        """
        Empirical proof of Z-score masking on heavy-tailed Cauchy distribution:
        Extreme Cauchy outliers inflate sample standard deviation, inflating Z-score bounds
        to extreme limits (>30), whereas IQR and MAD bounds remain tight (around +-4).
        """
        np.random.seed(42)
        cauchy_data = pd.Series(np.random.standard_cauchy(size=1000))

        _, stats_iqr = detect_outliers(cauchy_data, method="iqr")
        _, stats_z = detect_outliers(cauchy_data, method="zscore")
        _, stats_mad = detect_outliers(cauchy_data, method="modified_zscore")

        # IQR and MAD maintain robust finite bounds
        assert -5.0 < stats_iqr["lower_bound"] < -2.0
        assert 2.0 < stats_iqr["upper_bound"] < 5.0
        assert -6.0 < stats_mad["lower_bound"] < -3.0
        assert 3.0 < stats_mad["upper_bound"] < 6.0

        # Z-score standard deviation is blown up by extreme Cauchy draws
        assert stats_z["upper_bound"] > 25.0
        assert stats_z["lower_bound"] < -25.0

        # Winsorizing with IQR strictly clamps extreme variance
        w_iqr = handle_outliers(cauchy_data, method="iqr", action="winsorize")
        assert w_iqr.max() <= stats_iqr["upper_bound"]
        assert w_iqr.min() >= stats_iqr["lower_bound"]
        assert w_iqr.std() < 3.0  # Cauchy sample std was originally ~11

    def test_lognormal_skewed_distribution_bounds(self):
        """Log-normal distribution: strictly positive support, right-skewed tail."""
        np.random.seed(42)
        lognorm_data = pd.Series(np.random.lognormal(mean=0.0, sigma=1.5, size=1000))

        # Winsorize using IQR
        w_lognorm = handle_outliers(lognorm_data, method="iqr", action="winsorize")

        # Must not introduce negative values when original data is strictly positive
        assert w_lognorm.min() > 0.0
        # Right tail must be capped at upper bound
        _, stats_info = detect_outliers(lognorm_data, method="iqr")
        assert w_lognorm.max() <= stats_info["upper_bound"]

    def test_discrete_integers_zero_inflated_mad_degeneracy(self):
        """
        Critical edge case: Zero-inflated counts with >50% zeros causes MAD = 0.
        When MAD = 0, modified Z-score lower = upper = median = 0, causing EVERY non-zero
        value to be classified as an outlier and collapsing the entire distribution to 0!
        IQR and Z-score must be tested to demonstrate resilience against this collapse.
        """
        # Clinical fixture: ICU hospital-acquired infections (mostly 0, some 1, 2, 3, 10)
        counts = pd.Series([0] * 700 + [1] * 200 + [2] * 70 + [3] * 20 + [10] * 10)

        # 1. Modified Z-score (MAD) degeneracy verification
        mask_mad, stats_mad = detect_outliers(counts, method="modified_zscore")
        assert stats_mad["lower_bound"] == 0.0
        assert stats_mad["upper_bound"] == 0.0
        # Exactly 300 non-zero items flagged as outliers
        assert stats_mad["outlier_count"] == 300
        w_mad = handle_outliers(counts, method="modified_zscore", action="winsorize")
        # Collapses all data into a single constant!
        assert w_mad.nunique() == 1
        assert w_mad.unique()[0] == 0.0

        # 2. IQR resilience: IQR preserves discrete counts [0, 1, 2] and only flags true extreme tail (10)
        mask_iqr, stats_iqr = detect_outliers(counts, method="iqr")
        assert stats_iqr["upper_bound"] == 2.5
        assert stats_iqr["outlier_count"] == 30  # Flags counts >= 3
        w_iqr = handle_outliers(counts, method="iqr", action="winsorize")
        assert w_iqr.nunique() > 1
        assert set(w_iqr.unique()) == {0.0, 1.0, 2.0, 2.5}

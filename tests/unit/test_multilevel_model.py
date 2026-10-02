"""
Unit tests for medstat.models.multilevel (Design Effect, GEE, Random Intercept).
"""

import numpy as np
import pandas as pd
import pytest

from medstat.models.multilevel import (
    calculate_design_effect,
    fit_gee,
    fit_random_intercept,
)


@pytest.fixture
def clustered_binary_data():
    """Simulated multi-center cohort with binary outcome and cluster effect."""
    np.random.seed(42)
    n_clusters = 15
    cluster_size = 20
    N = n_clusters * cluster_size

    cluster_id = np.repeat(np.arange(n_clusters), cluster_size)
    u_i = np.repeat(np.random.normal(0, 0.8, n_clusters), cluster_size)
    x = np.random.normal(0, 1, N)

    logit_p = 0.2 + 0.9 * x + u_i
    p = 1.0 / (1.0 + np.exp(-logit_p))
    y = np.random.binomial(1, p)

    return pd.DataFrame({"y": y, "x": x, "hospital_id": cluster_id})


@pytest.fixture
def clustered_continuous_data():
    """Simulated multi-center cohort with continuous outcome and random intercept."""
    np.random.seed(42)
    n_clusters = 15
    cluster_size = 20
    N = n_clusters * cluster_size

    cluster_id = np.repeat(np.arange(n_clusters), cluster_size)
    u_i = np.repeat(np.random.normal(0, 1.2, n_clusters), cluster_size)
    x = np.random.normal(0, 1, N)
    y = 5.0 + 1.8 * x + u_i + np.random.normal(0, 1.0, N)

    return pd.DataFrame({"y": y, "x": x, "center_id": cluster_id})


class TestDesignEffectCalculation:
    def test_single_cluster_returns_deff_one(self):
        y = np.array([1, 2, 3, 4, 5])
        cluster = np.array([1, 1, 1, 1, 1])
        res = calculate_design_effect(y, cluster)

        assert res["n_clusters"] == 1
        assert res["design_effect"] == 1.0
        assert res["effective_sample_size"] == 5

    def test_clustered_data_computes_deff_and_icc(self, clustered_continuous_data):
        df = clustered_continuous_data
        res = calculate_design_effect(df["y"], df["center_id"])

        assert res["n_clusters"] == 15
        assert res["n_total"] == 300
        assert res["mean_cluster_size"] == 20.0
        assert res["icc_cluster"] > 0.05
        assert res["design_effect"] > 1.5
        assert res["effective_sample_size"] < 300


class TestGEEModel:
    def test_fit_gee_binomial_exchangeable(self, clustered_binary_data):
        df = clustered_binary_data
        res = fit_gee(
            y=df["y"],
            X=df[["x"]],
            cluster_ids=df["hospital_id"],
            family="binomial",
            cov_struct="exchangeable",
        )

        assert "summary_df" in res
        sum_df = res["summary_df"]
        assert "x" in sum_df.index
        assert "const" in sum_df.index

        # Odds ratio = exp(coef)
        x_coef = sum_df.loc["x", "coef"]
        x_or = sum_df.loc["x", "odds_ratio"]
        assert np.isclose(x_or, np.exp(x_coef), atol=1e-5)
        assert sum_df.loc["x", "or_ci_lower"] < x_or < sum_df.loc["x", "or_ci_upper"]
        assert res["n_clusters"] == 15
        assert res["nobs"] == 300

    def test_fit_gee_independence(self, clustered_binary_data):
        df = clustered_binary_data
        res = fit_gee(
            y=df["y"],
            X=df[["x"]],
            cluster_ids=df["hospital_id"],
            family="binomial",
            cov_struct="independence",
        )
        assert res["cov_struct"] == "independence"
        assert res["nobs"] == 300


class TestRandomInterceptModel:
    def test_fit_random_intercept_continuous(self, clustered_continuous_data):
        df = clustered_continuous_data
        res = fit_random_intercept(
            y=df["y"],
            X=df[["x"]],
            cluster_ids=df["center_id"],
        )

        assert "summary_df" in res
        sum_df = res["summary_df"]
        assert "x" in sum_df.index
        assert sum_df.loc["x", "p_value"] < 0.001

        # Check random effects variance components
        assert res["random_intercept_var"] > 0
        assert res["residual_var"] > 0
        assert 0.0 < res["cluster_icc"] < 1.0

    def test_fit_random_intercept_special_colnames_and_add_constant(self):
        np.random.seed(42)
        n = 100
        centers = np.random.choice([1, 2, 3, 4], size=n)
        df_special = pd.DataFrame(
            {
                "SOFA score": np.random.normal(5, 2, n),
                "C(sex)[T.M]": np.random.choice([0, 1], n),
            }
        )
        y = (
            2.0
            + 0.5 * df_special["SOFA score"]
            + centers * 0.3
            + np.random.normal(0, 1, n)
        )

        # Test with spaces and special syntax in column names
        res = fit_random_intercept(
            y=y,
            X=df_special,
            cluster_ids=centers,
            add_constant=True,
        )
        sum_df = res["summary_df"]
        assert "SOFA score" in sum_df.index
        assert "C(sex)[T.M]" in sum_df.index
        assert "const" in sum_df.index

    def test_multilevel_missing_covariates_rejected(self, clustered_continuous_data):
        df = clustered_continuous_data.copy()
        df.loc[0, "x"] = np.nan

        with pytest.raises(ValueError, match="Missing values detected in covariates X"):
            fit_gee(
                y=df["y"],
                X=df[["x"]],
                cluster_ids=df["center_id"],
                family="gaussian",
            )

        with pytest.raises(ValueError, match="Missing values detected in covariates X"):
            fit_random_intercept(
                y=df["y"],
                X=df[["x"]],
                cluster_ids=df["center_id"],
            )

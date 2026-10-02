"""
Unit tests for medstat.models.ordinal (Proportional Odds & Brant Test).
"""

import numpy as np
import pandas as pd
import pytest

from medstat.models.ordinal import (
    fit_multinomial_logistic,
    fit_proportional_odds,
    test_proportional_odds,
)


@pytest.fixture
def ordinal_parallel_data():
    """Simulated 3-level ordinal data satisfying proportional odds."""
    np.random.seed(42)
    n = 300
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.binomial(1, 0.5, n)
    latent = 0.8 * x1 - 0.5 * x2 + np.random.logistic(0, 1, n)
    y = np.zeros(n, dtype=int)
    y[latent > -0.5] = 1
    y[latent > 1.0] = 2
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


@pytest.fixture
def ordinal_non_parallel_data():
    """Simulated 3-level ordinal data violating proportional odds."""
    np.random.seed(123)
    n = 600
    x = np.random.normal(0, 1, n)
    # Cutpoint 1 has slope 2.0, Cutpoint 2 has slope -1.0
    p1 = 1.0 / (1.0 + np.exp(-(0.0 + 2.0 * x)))
    p2 = 1.0 / (1.0 + np.exp(-(1.5 - 1.5 * x)))
    y = np.zeros(n, dtype=int)
    for i in range(n):
        u = np.random.uniform(0, 1)
        if u < p2[i]:
            y[i] = 2
        elif u < p1[i]:
            y[i] = 1
        else:
            y[i] = 0
    return pd.DataFrame({"y": y, "x": x})


class TestOrdinalProportionalOdds:
    def test_fit_proportional_odds_success(self, ordinal_parallel_data):
        df = ordinal_parallel_data
        res = fit_proportional_odds(df["y"], df[["x1", "x2"]])

        assert "summary_df" in res
        assert "predictor_df" in res
        assert "threshold_df" in res
        assert res["k_categories"] == 3
        assert res["categories"] == [0, 1, 2]

        pred_df = res["predictor_df"]
        assert list(pred_df.index) == ["x1", "x2"]
        # Verify odds ratio = exp(coef)
        for var in ["x1", "x2"]:
            coef = pred_df.loc[var, "coef"]
            or_val = pred_df.loc[var, "odds_ratio"]
            assert np.isclose(or_val, np.exp(coef), atol=1e-5)
            assert (
                pred_df.loc[var, "or_ci_lower"]
                < or_val
                < pred_df.loc[var, "or_ci_upper"]
            )

        # Thresholds should have NaN odds ratio
        thresh_df = res["threshold_df"]
        assert len(thresh_df) == 2  # K - 1 cutpoints
        assert np.isnan(thresh_df["odds_ratio"]).all()

    def test_fit_insufficient_categories_raises_value_error(self):
        # Binary data (only 2 categories)
        y_binary = pd.Series([0, 1, 0, 1, 0])
        X = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        with pytest.raises(ValueError, match="at least 3 ordered categories"):
            fit_proportional_odds(y_binary, X)

    def test_fit_length_mismatch_raises_value_error(self):
        y = pd.Series([0, 1, 2, 1, 0])
        X = np.array([[1], [2], [3]])
        with pytest.raises(ValueError, match="Length mismatch"):
            fit_proportional_odds(y, X)


class TestBrantProportionalOddsTest:
    def test_brant_test_on_parallel_data(self, ordinal_parallel_data):
        df = ordinal_parallel_data
        brant_res = test_proportional_odds(df["y"], df[["x1", "x2"]])

        assert "omnibus" in brant_res
        assert "variables" in brant_res
        assert brant_res["omnibus"]["df"] == 2  # (J-1)*p = (2-1)*2 = 2
        # Data was generated under parallel lines, so p-value should comfortably not reject H0
        assert brant_res["omnibus"]["p_value"] > 0.05
        assert brant_res["omnibus"]["passed"] is True

        for var in ["x1", "x2"]:
            assert var in brant_res["variables"]
            assert brant_res["variables"][var]["df"] == 1
            assert "cutpoint_coefficients" in brant_res["variables"][var]

    def test_brant_test_detects_violation(self, ordinal_non_parallel_data):
        df = ordinal_non_parallel_data
        brant_res = test_proportional_odds(df["y"], df[["x"]])

        # In non-parallel data, slopes differ substantially (2.0 vs -1.5)
        assert brant_res["omnibus"]["p_value"] < 0.05
        assert brant_res["omnibus"]["passed"] is False
        assert "violated" in brant_res["interpretation"]

    def test_brant_insufficient_categories_raises(self):
        y = pd.Series([0, 1, 1, 0])
        X = pd.DataFrame({"x": [1, 2, 3, 4]})
        with pytest.raises(ValueError, match=">= 3 categories"):
            test_proportional_odds(y, X)


class TestMultinomialFallback:
    def test_fit_multinomial_logistic(self, ordinal_parallel_data):
        df = ordinal_parallel_data
        res = fit_multinomial_logistic(df["y"], df[["x1", "x2"]])

        assert "model" in res
        assert "params" in res
        assert res["model"].converged

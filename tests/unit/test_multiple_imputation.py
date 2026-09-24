"""
tests/unit/test_multiple_imputation.py: Unit tests for MICE imputation, Rubin's rules pooling, and diagnostics.

Ported from shiny-stat with updates for medstat-core headless architecture.
"""

import numpy as np
import pandas as pd
import pytest

from medstat.data.missing import (
    MICEImputer,
    MICEResult,
    PooledEstimate,
    create_imputation_diagnostics,
    get_imputation_summary,
    impute_indicator,
    impute_knn,
    pool_estimates,
    pool_regression_results,
)

pytestmark = pytest.mark.unit


class TestPoolEstimates:
    """Tests for Rubin's rules pooling."""

    def test_pool_estimates_basic(self):
        """Test basic pooling of estimates."""
        estimates = [1.5, 1.6, 1.4, 1.55, 1.45]
        variances = [0.1, 0.12, 0.09, 0.11, 0.1]

        result = pool_estimates(estimates, variances)

        assert isinstance(result, PooledEstimate)
        assert result.n_imputations == 5
        assert abs(result.estimate - np.mean(estimates)) < 0.001
        assert result.se > 0
        assert result.ci_lower < result.estimate < result.ci_upper
        assert 0 <= result.p_value <= 1

    def test_pool_estimates_single_imputation(self):
        """Test pooling with single imputation."""
        result = pool_estimates([2.0], [0.25])

        assert result.n_imputations == 1
        assert result.estimate == 2.0
        assert abs(result.se - 0.5) < 0.001
        assert result.between_variance == 0.0

    def test_pool_estimates_high_between_variance(self):
        """Test pooling when between-imputation variance is high."""
        estimates = [1.0, 3.0, 2.0, 4.0, 0.0]
        variances = [0.01, 0.01, 0.01, 0.01, 0.01]

        result = pool_estimates(estimates, variances)

        assert result.between_variance > result.within_variance
        assert result.fmi > 0
        assert result.lambda_ > 0.5

    def test_pool_estimates_with_sample_size(self):
        """Test pooling with sample size for df adjustment."""
        estimates = [1.0, 1.1, 0.9, 1.05, 0.95]
        variances = [0.1, 0.1, 0.1, 0.1, 0.1]

        result = pool_estimates(estimates, variances, n_obs=100)

        assert np.isfinite(result.df)
        assert result.df > 0

    def test_pool_estimates_zero_within_variance_positive_between_variance(self):
        """Test pooling when W_bar == 0 but B > 0 yields df = m - 1."""
        estimates = [1.0, 2.0, 3.0]
        variances = [0.0, 0.0, 0.0]

        result = pool_estimates(estimates, variances, n_obs=100, df_complete=98)

        assert result.df == 2.0
        assert result.fmi == 1.0
        assert result.lambda_ == 1.0


class TestMICEImputer:
    """Tests for MICE imputation."""

    @pytest.fixture
    def df_with_missing(self):
        """Create test DataFrame with missing values."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "x1": np.random.randn(n),
                "x2": np.random.randn(n) * 2 + 1,
                "x3": np.random.randn(n) + 0.5,
            }
        )
        for col in df.columns:
            mask = np.random.random(n) < 0.2
            df.loc[mask, col] = np.nan
        return df

    def test_mice_basic(self, df_with_missing):
        """Test basic MICE imputation."""
        imputer = MICEImputer(n_imputations=3, max_iter=5, random_state=42)
        result = imputer.fit_transform(df_with_missing)

        assert isinstance(result, MICEResult)
        assert result.n_imputations == 3
        assert len(result.imputed_datasets) == 3

        for df_imp in result.imputed_datasets:
            assert df_imp.isna().sum().sum() == 0

    def test_mice_no_missing(self):
        """Test MICE with complete data."""
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0],
                "b": [4.0, 5.0, 6.0],
            }
        )

        imputer = MICEImputer(n_imputations=3, random_state=42)
        result = imputer.fit_transform(df)

        assert len(result.columns_imputed) == 0
        assert len(result.imputed_datasets) == 1

    def test_mice_selected_columns(self, df_with_missing):
        """Test MICE on selected columns only."""
        imputer = MICEImputer(n_imputations=2, random_state=42)
        result = imputer.fit_transform(df_with_missing, columns=["x1", "x2"])

        for col in result.columns_imputed:
            assert col in ["x1", "x2"]


class TestKNNAndIndicatorImputation:
    """Tests for KNN and indicator imputation methods."""

    def test_knn_imputation(self):
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, np.nan, 4.0, 5.0],
                "b": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )
        df_imp = impute_knn(df, n_neighbors=2, scale=True)
        assert df_imp.isna().sum().sum() == 0
        assert 2.0 < df_imp.loc[2, "a"] < 4.0

    def test_indicator_imputation(self):
        df = pd.DataFrame(
            {
                "num": [10.0, np.nan, 30.0, 40.0],
                "cat": ["A", "B", np.nan, "D"],
            }
        )
        df_imp, added = impute_indicator(df)
        assert "num_missing" in added
        assert "cat_missing" in added
        assert df_imp["num_missing"].tolist() == [0, 1, 0, 0]
        assert df_imp["cat_missing"].tolist() == [0, 0, 1, 0]
        assert df_imp.isna().sum().sum() == 0
        assert df_imp.loc[1, "num"] == 30.0  # median of [10, 30, 40]
        assert df_imp.loc[2, "cat"] == "Missing"


class TestPoolRegressionResults:
    """Tests for pooling regression results."""

    def test_pool_regression_mock_models(self):
        """Test pooling with mock model objects."""

        class MockModel:
            def __init__(self, coefs, ses, n):
                self.params = pd.Series(coefs, index=["const", "x1", "x2"])
                self.bse = pd.Series(ses, index=["const", "x1", "x2"])
                self.nobs = n
                self.rsquared = 0.5

        models = [
            MockModel([1.0, 2.0, -0.5], [0.1, 0.2, 0.1], 100),
            MockModel([1.1, 1.9, -0.6], [0.12, 0.18, 0.11], 100),
            MockModel([0.9, 2.1, -0.4], [0.11, 0.22, 0.09], 100),
        ]

        pooled = pool_regression_results(models, model_type="linear")

        assert pooled.n_imputations == 3
        assert "const" in pooled.coefficients
        assert "x1" in pooled.coefficients

        x1_pooled = pooled.coefficients["x1"]
        assert 1.8 < x1_pooled.estimate < 2.2
        assert x1_pooled.se > 0

    def test_pool_regression_to_dataframe(self):
        """Test converting pooled results to DataFrame."""

        class MockModel:
            def __init__(self):
                self.params = pd.Series([1.0, 2.0], index=["a", "b"])
                self.bse = pd.Series([0.1, 0.2], index=["a", "b"])
                self.nobs = 50

        models = [MockModel(), MockModel()]
        pooled = pool_regression_results(models)

        df = pooled.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert "Variable" in df.columns
        assert "Coefficient" in df.columns
        assert "P-value" in df.columns
        assert len(df) == 2


class TestDiagnostics:
    """Tests for imputation diagnostics."""

    def test_imputation_summary(self):
        """Test imputation summary generation."""
        original_mask = pd.DataFrame(
            {
                "x": [False, True, False, True, False],
            }
        )

        imp1 = pd.DataFrame({"x": [1.0, 1.5, 2.0, 2.5, 3.0]})
        imp2 = pd.DataFrame({"x": [1.0, 1.6, 2.0, 2.4, 3.0]})

        result = MICEResult(
            imputed_datasets=[imp1, imp2],
            n_imputations=2,
            columns_imputed=["x"],
            original_missing_mask=original_mask,
        )

        summary = get_imputation_summary(result)

        assert isinstance(summary, pd.DataFrame)
        assert "Variable" in summary.columns
        assert "Mean" in summary.columns
        assert len(summary) == 1

    def test_create_diagnostic_plots(self):
        """Test diagnostic plot creation."""
        np.random.seed(42)
        df = pd.DataFrame({"x": np.random.randn(50)})
        df.loc[10:15, "x"] = np.nan

        imputer = MICEImputer(n_imputations=2, max_iter=3, random_state=42)
        mice_result = imputer.fit_transform(df)

        figures = create_imputation_diagnostics(mice_result, df)

        assert isinstance(figures, dict)

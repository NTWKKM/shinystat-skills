"""
tests/unit/test_data_cleaning.py: Unit tests for data cleaning utilities.

Ported from shiny-stat with updates for medstat-core headless architecture.
Tests:
- clean_numeric
- clean_numeric_vector
- detect_outliers
- handle_outliers
- clean_dataframe
- get_cleaning_summary
- check_missing_data_impact
- apply_missing_values_to_df
- detect_zero_variance
"""

import numpy as np
import pandas as pd
import pytest

from medstat.data.clean import (
    apply_missing_values_to_df,
    check_missing_data_impact,
    clean_dataframe,
    clean_numeric,
    clean_numeric_vector,
    detect_outliers,
    detect_zero_variance,
    get_cleaning_summary,
    handle_outliers,
    prepare_data_for_analysis,
)

pytestmark = pytest.mark.unit


class TestNumericCleaning:
    """Tests for numeric cleaning functions."""

    def test_clean_numeric_scalar(self):
        """Test clean_numeric with various scalar inputs."""
        assert clean_numeric(">100") == 100.0
        assert clean_numeric("1,234.56") == 1234.56
        assert np.isnan(clean_numeric(None))
        assert np.isnan(clean_numeric("abc"))
        assert clean_numeric("$500") == 500.0
        assert clean_numeric("10%") == 10.0
        assert clean_numeric("(100)") == -100.0
        # Currency and unicode extensions
        assert clean_numeric("฿1,500.50") == 1500.50
        assert clean_numeric("€250") == 250.0
        assert clean_numeric("−42.5") == -42.5
        assert clean_numeric("–10.0") == -10.0

    def test_clean_numeric_vector(self):
        """Test clean_numeric_vector with series input."""
        test_series = pd.Series([">100", "1,234.56", None, "abc", "$500"])
        result = clean_numeric_vector(test_series)

        expected = pd.Series([100.0, 1234.56, np.nan, np.nan, 500.0])
        pd.testing.assert_series_equal(
            result.reset_index(drop=True), expected, check_names=False
        )


class TestOutlierDetection:
    """Tests for outlier detection and handling."""

    def test_detect_outliers_iqr(self):
        """Test detect_outliers using IQR method."""
        test_series = pd.Series([1, 2, 3, 4, 5, 100])
        mask, stats = detect_outliers(test_series, method="iqr")

        assert mask.iloc[5]
        assert not mask.iloc[0:5].any()
        assert stats["method"] == "iqr"
        assert stats["outlier_count"] == 1

    def test_detect_outliers_zscore(self):
        """Test detect_outliers using Z-score method."""
        test_series = pd.Series([10, 10, 11, 10, 11, 10, 100])
        mask, stats = detect_outliers(test_series, method="zscore", threshold=2.0)

        assert mask.iloc[6]
        assert stats["method"] == "zscore"

    def test_detect_outliers_modified_zscore(self):
        """Test detect_outliers using Modified Z-score (MAD)."""
        test_series = pd.Series([10.0, 10.2, 9.8, 10.1, 10.0, 9.9, 100.0])
        mask, stats = detect_outliers(
            test_series, method="modified_zscore", threshold=3.5
        )

        assert mask.iloc[6]
        assert stats["method"] == "modified_zscore"
        assert stats["outlier_count"] == 1

    def test_handle_outliers_actions(self):
        """Test various outlier handling actions."""
        test_series = pd.Series([1, 2, 3, 4, 5, 100])

        # Test flag (NaN replacement)
        flagged = handle_outliers(test_series, action="flag")
        assert np.isnan(flagged.iloc[5])

        # Test winsorize
        winsorized = handle_outliers(test_series, action="winsorize")
        assert winsorized.iloc[5] < 100
        assert winsorized.iloc[5] > 5

        # Test cap
        capped = handle_outliers(test_series, action="cap")
        assert capped.iloc[5] < 100

        # Test winsorize with zscore (threshold=2.0 needed because max z-score on n=6 is (n-1)/sqrt(n) = 2.04)
        winsorized_z = handle_outliers(
            test_series, method="zscore", threshold=2.0, action="winsorize"
        )
        assert winsorized_z.iloc[5] < 100


class TestDataFrameCleaning:
    """Tests for DataFrame cleaning and reporting."""

    def test_prepare_data_duplicate_cols(self):
        """Test that prepare_data_for_analysis handles duplicate required columns gracefully."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        required_cols = ["A", "A", "B"]
        numeric_cols = ["A", "B"]

        df_clean, info = prepare_data_for_analysis(
            df,
            required_cols=required_cols,
            numeric_cols=numeric_cols,
            handle_missing="complete-case",
            missing_justification="Duplicate cols test",
        )

        assert not df_clean.empty
        assert len(df_clean.columns) == 2
        assert list(df_clean.columns) == ["A", "B"]

    def test_clean_dataframe_granular(self):
        """Test clean_dataframe with mixed dirty data."""
        test_df = pd.DataFrame(
            {
                "Dirty_Numeric": [">100", "1,234", "ERROR", "500"],
                "Strictly_Text": ["A", "B", "C", "D"],
                "Mixed": ["10", "20", "bad", "worse"],
            }
        )

        cleaned_df, _report = clean_dataframe(test_df)

        assert pd.api.types.is_numeric_dtype(cleaned_df["Dirty_Numeric"])
        assert cleaned_df["Dirty_Numeric"].isna().sum() == 1

        assert pd.api.types.is_numeric_dtype(cleaned_df["Mixed"])
        assert cleaned_df["Mixed"].isna().sum() == 2

        assert pd.api.types.is_string_dtype(
            cleaned_df["Strictly_Text"]
        ) or pd.api.types.is_object_dtype(cleaned_df["Strictly_Text"])

    def test_get_cleaning_summary(self):
        """Test summary generation."""
        test_df = pd.DataFrame({"a": [1, 2, 3]})
        _, report = clean_dataframe(test_df)
        summary = get_cleaning_summary(report)

        assert "DATA CLEANING SUMMARY" in summary
        assert "Original shape: (3, 1)" in summary

    def test_detect_zero_variance(self):
        """Test constant column detection."""
        df = pd.DataFrame({"const": [5.0, 5.0, 5.0], "var": [1.0, 2.0, 3.0]})
        zero_vars = detect_zero_variance(df)
        assert "const" in zero_vars
        assert "var" not in zero_vars


class TestMissingDataImpact:
    """Tests for check_missing_data_impact with missing_codes."""

    def test_check_missing_data_impact_with_coded_values(self):
        """Test that impact is accurately reported when coded values are used."""
        df_original = pd.DataFrame(
            {"age": [25, -99, 30, 999, 40], "score": [100, 85, -99, 90, 95]}
        )
        df_clean = pd.DataFrame({"age": [25.0, 40.0], "score": [100.0, 95.0]})

        var_meta = {
            "age": {"label": "Age", "missing_values": [-99, 999]},
            "score": {"label": "Score", "missing_values": [-99]},
        }

        missing_codes = {"age": [-99, 999], "score": [-99]}

        result = check_missing_data_impact(
            df_original, df_clean, var_meta, missing_codes=missing_codes
        )

        assert result["rows_removed"] == 3
        assert result["observations_lost"]["age"]["count"] == 2
        assert result["observations_lost"]["score"]["count"] == 1
        assert "age" in result["variables_affected"]
        assert "score" in result["variables_affected"]

    def test_apply_missing_values_dict_support(self):
        """Test that apply_missing_values_to_df supports dict-based missing_codes."""
        df = pd.DataFrame({"a": [1, -99, 3], "b": [10, -88, 30]})
        missing_codes = {"a": [-99], "b": [-88]}

        result = apply_missing_values_to_df(df, {}, missing_codes=missing_codes)

        assert result["a"].isna().sum() == 1
        assert result["b"].isna().sum() == 1
        assert np.isnan(result.loc[1, "a"])
        assert np.isnan(result.loc[1, "b"])

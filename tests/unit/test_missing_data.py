"""
tests/unit/test_missing_data.py: Unit tests for missing data management functionality.

Ported from shiny-stat with updates for medstat-core headless architecture.
Tests:
- apply_missing_values_to_df - Convert missing codes to NaN
- detect_missing_in_variable - Count missing values
- get_missing_summary_df - Generate summary table
- handle_missing_for_analysis - Apply handling strategy
- check_missing_data_impact - Report impact
"""

import numpy as np
import pandas as pd
import pytest

from medstat.data.clean import (
    apply_missing_values_to_df,
    check_missing_data_impact,
    detect_missing_in_variable,
    get_missing_summary_df,
)
from medstat.data.missing import handle_missing_for_analysis

pytestmark = pytest.mark.unit


class TestApplyMissingValuesToDf:
    """Tests for apply_missing_values_to_df function."""

    def test_replaces_missing_codes_with_nan(self):
        """Test that specified missing codes are replaced with NaN."""
        df = pd.DataFrame(
            {"age": [25, -99, 30, -999, 40], "score": [100, 85, -99, 90, 95]}
        )
        var_meta = {
            "age": {"type": "Continuous", "missing_values": [-99, -999]},
            "score": {"type": "Continuous", "missing_values": [-99]},
        }

        result = apply_missing_values_to_df(df, var_meta)

        assert result["age"].isna().sum() == 2
        assert np.isnan(result.loc[1, "age"])
        assert np.isnan(result.loc[3, "age"])

        assert result["score"].isna().sum() == 1
        assert np.isnan(result.loc[2, "score"])

    def test_uses_global_codes_as_fallback(self):
        """Test that global missing codes are used when not in var_meta."""
        df = pd.DataFrame({"value": [10, -99, 20, -99, 30]})
        var_meta = {}

        result = apply_missing_values_to_df(df, var_meta, missing_codes=[-99])

        assert result["value"].isna().sum() == 2

    def test_preserves_original_df(self):
        """Test that original DataFrame is not modified."""
        df = pd.DataFrame({"age": [25, -99, 30]})
        var_meta = {"age": {"missing_values": [-99]}}
        original_value = df.loc[1, "age"]

        apply_missing_values_to_df(df, var_meta)

        assert df.loc[1, "age"] == original_value


class TestDetectMissingInVariable:
    """Tests for detect_missing_in_variable function."""

    def test_counts_nan_values(self):
        """Test counting of NaN values."""
        series = pd.Series([1, np.nan, 3, np.nan, 5])

        result = detect_missing_in_variable(series)

        assert result["total_count"] == 5
        assert result["missing_nan_count"] == 2
        assert result["missing_count"] == 2
        assert result["valid_count"] == 3

    def test_counts_coded_missing_values(self):
        """Test counting of coded missing values."""
        series = pd.Series([1, -99, 3, -99, 5])

        result = detect_missing_in_variable(series, missing_codes=[-99])

        assert result["missing_coded_count"] == 2
        assert result["missing_count"] == 2

    def test_calculates_percentage(self):
        """Test percentage calculation."""
        series = pd.Series([1, np.nan, 3, np.nan, 5])  # 2 out of 5 = 40%

        result = detect_missing_in_variable(series)

        assert result["missing_pct"] == 40.0


class TestGetMissingSummaryDf:
    """Tests for get_missing_summary_df function."""

    def test_generates_summary_for_all_columns(self):
        """Test that summary is generated for all columns."""
        df = pd.DataFrame(
            {"age": [25, np.nan, 30], "weight": [70, 80, np.nan], "group": [1, 2, 1]}
        )
        var_meta = {
            "age": {"type": "Continuous"},
            "weight": {"type": "Continuous"},
            "group": {"type": "Categorical"},
        }

        result = get_missing_summary_df(df, var_meta)

        assert len(result) == 3
        assert "Variable" in result.columns
        assert "Type" in result.columns
        assert "N_Missing" in result.columns
        assert "Pct_Missing" in result.columns

    def test_formats_percentage_as_string(self):
        """Test that percentage is formatted as string with %."""
        df = pd.DataFrame({"value": [1, np.nan]})  # 50% missing
        var_meta = {"value": {"type": "Continuous"}}

        result = get_missing_summary_df(df, var_meta)

        assert "50.0%" in result["Pct_Missing"].values


class TestMissingValueBreakdownFix:
    """Tests for the fix of missing value breakdown (coded vs NaN)."""

    def test_detect_missing_breakdown_before_and_after_normalization(self):
        """Test that breakdown is correct both before and after normalization."""
        series = pd.Series([1, -99, 3, np.nan, 5])
        missing_codes = [-99]

        # 1. Before normalization (raw data)
        info_raw = detect_missing_in_variable(
            series, missing_codes, already_normalized=False
        )
        assert info_raw["missing_coded_count"] == 1
        assert info_raw["missing_nan_count"] == 1
        assert info_raw["missing_count"] == 2

        # 2. After normalization
        df = pd.DataFrame({"val": series})
        var_meta = {"val": {"missing_values": missing_codes}}
        df_normalized = apply_missing_values_to_df(df, var_meta)

        info_norm = detect_missing_in_variable(
            df_normalized["val"], missing_codes, already_normalized=True
        )
        assert info_norm["missing_coded_count"] == 0
        assert info_norm["missing_nan_count"] == 2
        assert info_norm["missing_count"] == 2

    def test_detect_missing_avoids_double_counting_with_nan_input(self):
        """Test that detect_missing_in_variable handles coded values that might be NaN."""
        series = pd.Series([1, np.nan, 3])
        missing_codes = [np.nan]

        info = detect_missing_in_variable(series, missing_codes)
        assert info["missing_nan_count"] == 1
        assert info["missing_coded_count"] == 0
        assert info["missing_count"] == 1


class TestHandleMissingForAnalysis:
    """Tests for handle_missing_for_analysis function."""

    def test_complete_case_drops_rows_with_nan(self):
        """Test that complete-case strategy drops rows with any NaN."""
        df = pd.DataFrame({"a": [1, np.nan, 3, 4], "b": [10, 20, np.nan, 40]})
        var_meta = {}

        result = handle_missing_for_analysis(df, var_meta, strategy="complete-case")

        assert len(result) == 2

    def test_returns_counts_when_requested(self):
        """Test that counts are returned when return_counts=True."""
        df = pd.DataFrame({"a": [1, np.nan, 3, 4], "b": [10, 20, np.nan, 40]})
        var_meta = {}

        _, counts = handle_missing_for_analysis(df, var_meta, return_counts=True)

        assert "original_rows" in counts
        assert "final_rows" in counts
        assert "rows_removed" in counts
        assert counts["rows_removed"] == 2

    def test_applies_missing_codes(self):
        """Test that missing codes are converted to NaN before dropping."""
        df = pd.DataFrame({"value": [1, -99, 3, 4]})
        var_meta = {"value": {"missing_values": [-99]}}

        result = handle_missing_for_analysis(df, var_meta)

        assert len(result) == 3


class TestCheckMissingDataImpact:
    """Tests for check_missing_data_impact function."""

    def test_calculates_rows_removed(self):
        """Test calculation of rows removed."""
        df_original = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        df_clean = pd.DataFrame({"a": [1, 2, 3]})
        var_meta = {}

        result = check_missing_data_impact(df_original, df_clean, var_meta)

        assert result["rows_removed"] == 2
        assert result["pct_removed"] == 40.0

    def test_identifies_affected_variables(self):
        """Test identification of variables with missing data."""
        df_original = pd.DataFrame({"a": [1, np.nan, 3], "b": [10, 20, 30]})
        df_clean = df_original.dropna()
        var_meta = {"a": {"label": "Variable A"}, "b": {"label": "Variable B"}}

        result = check_missing_data_impact(df_original, df_clean, var_meta)

        assert "a" in result["variables_affected"]
        assert "b" not in result["variables_affected"]

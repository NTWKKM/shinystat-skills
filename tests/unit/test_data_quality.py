"""
tests/unit/test_data_quality.py: Unit tests for check_data_quality warnings.

Ported from shiny-stat with updates for medstat-core headless architecture.
"""

import pandas as pd
import pytest

from medstat.data.quality import check_data_quality

pytestmark = pytest.mark.unit


def test_check_data_quality_numeric_with_non_standard():
    df = pd.DataFrame({"Age": [25, 30, "<35", 40, "45%", 50, 55, 60, 65, 70]})
    warnings = check_data_quality(df)
    assert len(warnings) > 0
    assert "Age" in warnings[0]
    assert "non-standard values" in warnings[0]


def test_check_data_quality_categorical_with_numeric():
    df = pd.DataFrame(
        {
            "Gender": [
                "Male",
                "Female",
                "Male",
                "1",
                "Female",
                "Male",
                "Male",
                "Female",
                "Male",
                "Female",
            ]
        }
    )
    warnings = check_data_quality(df)
    assert len(warnings) > 0
    assert "Gender" in warnings[0]
    assert "numeric values inside categorical" in warnings[0]


def test_check_data_quality_rare_categories():
    df = pd.DataFrame({"City": ["Bangkok"] * 90 + ["Phuket", "Chiang Mai", "Phuket"]})
    warnings = check_data_quality(df)
    assert len(warnings) > 0
    assert "City" in warnings[0]
    assert "rare categories" in warnings[0]


def test_check_data_quality_missing_data():
    df = pd.DataFrame({"Income": [1000, 2000, None, 4000, 5000]})
    warnings = check_data_quality(df)
    assert len(warnings) > 0
    assert "Income" in warnings[0]
    assert "Missing 1 values" in warnings[0]


def test_check_data_quality_empty_df():
    df = pd.DataFrame()
    warnings = check_data_quality(df)
    assert warnings == []


def test_check_data_quality_no_issues():
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "B": ["X", "X", "X", "X", "X", "Y", "Y", "Y", "Y", "Y"],
        }
    )
    warnings = check_data_quality(df)
    assert warnings == []

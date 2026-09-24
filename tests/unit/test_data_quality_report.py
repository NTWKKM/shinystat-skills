"""
tests/unit/test_data_quality_report.py: Unit tests for DataQualityReport framework.

Ported from shiny-stat with 5D extensions (Completeness, Validity, Consistency, Uniqueness, Plausibility).
"""

import numpy as np
import pandas as pd
import pytest

from medstat.data.quality import (
    ColumnRule,
    DataQualityReport,
    DataQualitySchema,
)

pytestmark = pytest.mark.unit


class TestDataQualityReport:
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": ["a", "b", "c", "d", "e"],
                "C": [1.1, 2.2, np.nan, 4.4, 5.5],
                "D": [1, 1, 2, 2, 3],  # Duplicates potential
            }
        )

    def test_completeness_score(self, sample_data):
        report = DataQualityReport(sample_data)
        # Total cells = 20. 1 missing. 19/20 = 0.95 -> 95.0
        assert report.completeness_score() == 95.0

    def test_consistency_score_clean(self, sample_data):
        report = DataQualityReport(sample_data)
        assert report.consistency_score() == 100.0

    def test_consistency_score_dirty(self):
        df = pd.DataFrame({"A": ["1", "2", "<5", "4", "10%"]})
        report = DataQualityReport(df)
        assert report.consistency_score() == 60.0

    def test_uniqueness_score(self):
        df = pd.DataFrame({"A": [1, 1, 2, 3, 3], "B": ["a", "a", "b", "c", "c"]})
        report = DataQualityReport(df)
        assert report.uniqueness_score() == 60.0

    def test_validity_score(self, sample_data):
        report = DataQualityReport(sample_data)
        assert report.validity_score() == 100.0

    def test_validity_score_with_schema(self):
        df = pd.DataFrame({"age": [25, 45, 150, -5, 30]})
        schema = DataQualitySchema(columns={"age": ColumnRule(min_val=0, max_val=125)})
        report = DataQualityReport(df, schema=schema)
        # 5 rows, 2 invalid (150, -5). Score = 100 * (1 - 2/5) = 60.0
        assert report.validity_score() == 60.0

    def test_plausibility_score(self):
        # 100 observations, 1 extreme outlier
        np.random.seed(42)
        vals = np.random.normal(50, 5, 100).tolist()
        vals.append(1000.0)  # extreme outlier >3*IQR
        df = pd.DataFrame({"biomarker": vals})
        report = DataQualityReport(df)
        score = report.plausibility_score()
        assert score < 100.0
        assert score > 90.0

    def test_generate_report(self, sample_data):
        report = DataQualityReport(sample_data)
        result = report.generate_report()

        assert "overall_score" in result
        assert "grade" in result
        assert "dimension_scores" in result
        assert "issues" in result
        assert "recommendations" in result

        scores = result["dimension_scores"]
        assert scores["completeness"] == 95.0
        assert scores["validity"] == 100.0
        assert scores["consistency"] == 100.0
        assert scores["uniqueness"] == 100.0
        assert scores["plausibility"] == 100.0

    def test_recommendations(self):
        df = pd.DataFrame({"A": [np.nan] * 10})  # 0% completeness
        report = DataQualityReport(df)
        result = report.generate_report()
        recs = result["recommendations"]
        assert any("significant missing values" in r for r in recs)

    def test_render_ascii_summary(self, sample_data):
        report = DataQualityReport(sample_data)
        ascii_out = report.render_ascii_summary()
        assert "DATA QUALITY AUDIT REPORT" in ascii_out
        assert "Completeness:" in ascii_out

"""
tests/unit/test_littles_mcar.py: Unit tests for Little's MCAR test & MissingnessAudit.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from medstat.data.clean import (
    LittlesMCARResult,
    MissingnessAudit,
    audit_missingness,
    littles_mcar_test,
)

pytestmark = pytest.mark.unit

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent / "fixtures" / "incomplete_clinical.csv"
)


class TestLittlesMCARTest:
    def test_littles_mcar_on_incomplete_fixture(self):
        assert FIXTURE_PATH.exists()
        df = pd.read_csv(FIXTURE_PATH)
        cols = ["age", "sbp", "creatinine", "bmi"]

        res = littles_mcar_test(df, cols=cols)

        assert isinstance(res, LittlesMCARResult)
        assert res.n_variables == 4
        assert res.n_observations == 400
        assert res.n_patterns >= 5
        assert res.df == 15  # sum(p_s) - P = 19 - 4 = 15
        assert res.em_converged
        assert 15.0 < res.statistic < 25.0  # Approx 20.79
        assert 0.05 < res.p_value < 0.30  # Approx 0.14
        assert res.is_mcar is True
        # Dict & property access
        assert res["statistic"] == res.statistic
        assert res["p_value"] == res.p_value
        assert res["pvalue"] == res.pvalue
        assert "p_value" in res

    def test_littles_mcar_fully_observed_data(self):
        df = pd.DataFrame(
            {
                "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "x2": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )
        res = littles_mcar_test(df)
        assert res.is_mcar is True
        assert res.statistic == 0.0
        assert res.p_value == 1.0

    def test_littles_mcar_empty_df_raises_error(self):
        with pytest.raises(ValueError, match="empty"):
            littles_mcar_test(pd.DataFrame())

    def test_littles_mcar_100_percent_missing_raises_error(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [np.nan, np.nan]})
        with pytest.raises(ValueError, match="100% missing"):
            littles_mcar_test(df)

    def test_littles_mcar_singular_covariance_ridge(self):
        # Collinear columns
        x = np.linspace(1, 10, 20)
        df = pd.DataFrame({"a": x, "b": 2 * x, "c": 3 * x})
        df.loc[2:4, "a"] = np.nan
        df.loc[5:7, "b"] = np.nan
        # Should converge cleanly without LinAlgError
        res = littles_mcar_test(df)
        assert np.isfinite(res.statistic)


class TestMissingnessAudit:
    def test_audit_missingness_structure(self):
        assert FIXTURE_PATH.exists()
        df = pd.read_csv(FIXTURE_PATH)

        audit = audit_missingness(df)
        assert isinstance(audit, MissingnessAudit)
        assert audit.total_rows == 400
        assert audit["creatinine"]["missing_count"] == 32
        assert audit["bmi"]["missing_count"] == 56
        assert audit["sbp"]["missing_count"] == 16
        assert "creatinine" in audit
        assert "variables" in audit
        assert len(audit.patterns) > 0
        assert "creatinine" in audit.co_occurrence

        # JSON export test
        audit_json = audit.to_json()
        assert "total_rows" in audit_json
        assert "creatinine" in audit_json

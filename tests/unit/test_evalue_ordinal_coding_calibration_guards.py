"""
tests/unit/test_evalue_ordinal_coding_calibration_guards.py: Input guards for E-value exposure matching,
ordinal outcome code standardization, and calibration metric ranges.

Covers:
1. CLI E-value: ambiguous exposure prefix matches rejected (logistic and ordinal).
2. standardize_categorical_outcome: missing values, unmapped labels, non-monotonic codes rejected.
3. validate_calibration_metrics: brier / ici outside [0, 1] rejected; slope/intercept unaffected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from medstat.cli.main import cli
from medstat.data.clean import standardize_categorical_outcome
from medstat.reporting.narrative import validate_calibration_metrics


@pytest.fixture
def multilevel_exposure_csv(tmp_path):
    rng = np.random.default_rng(7)
    n = 300
    arm = rng.choice(["A", "B", "C"], size=n)
    age = rng.normal(size=n)
    latent = 0.5 * age + 0.4 * (arm == "B") - 0.4 * (arm == "C") + rng.logistic(size=n)
    df = pd.DataFrame(
        {
            "y_bin": (latent > 0).astype(int),
            "y_ord": np.digitize(latent, [-0.5, 1.0]),
            "arm": arm,
            "age": age,
        }
    )
    path = tmp_path / "multi.csv"
    df.to_csv(path, index=False)
    return path


@pytest.mark.parametrize(
    ("mtype", "outcome"), [("logistic", "y_bin"), ("ordinal", "y_ord")]
)
def test_cli_e_value_rejects_ambiguous_exposure(
    multilevel_exposure_csv, tmp_path, mtype, outcome
):
    res = CliRunner().invoke(
        cli,
        [
            "model",
            "--data",
            str(multilevel_exposure_csv),
            "--type",
            mtype,
            "--outcome",
            outcome,
            "--exposure",
            "arm",
            "--covariates",
            "age",
            "--e-value",
            "--output",
            str(tmp_path / "out.json"),
        ],
    )
    assert res.exit_code != 0
    assert "ambiguous" in res.output.lower()


class TestStandardizeCategoricalOutcome:
    cats = ["Home", "Ward", "Death"]

    def _series(self, values):
        return pd.Series(pd.Categorical(values, categories=self.cats, ordered=True))

    def test_missing_values_rejected(self):
        with pytest.raises(ValueError, match="missing"):
            standardize_categorical_outcome(self._series(["Home", None, "Death"]))

    def test_missing_values_rejected_with_mapping(self):
        with pytest.raises(ValueError, match="missing"):
            standardize_categorical_outcome(
                self._series(["Home", None]), {"Home": 0, "Ward": 1, "Death": 2}
            )

    def test_unmapped_label_rejected(self):
        with pytest.raises(ValueError, match="not defined in outcome_ordinal_mapping"):
            standardize_categorical_outcome(
                self._series(["Home", "Ward"]), {"Home": 0, "Death": 2}
            )

    def test_non_monotonic_codes_rejected(self):
        with pytest.raises(ValueError, match="increase"):
            standardize_categorical_outcome(
                self._series(["Home", "Ward"]), {"Home": 0, "Ward": 2, "Death": 1}
            )

    def test_duplicate_codes_rejected(self):
        with pytest.raises(ValueError, match="unique"):
            standardize_categorical_outcome(
                self._series(["Home", "Ward"]), {"Home": 0, "Ward": 1, "Death": 1}
            )

    def test_valid_mapping_and_fallback_preserved(self):
        s = self._series(["Home", "Ward", "Death"])
        assert standardize_categorical_outcome(
            s, {0: "Home", 1: "Ward", 6: "Death"}
        ).tolist() == [0, 1, 6]
        assert standardize_categorical_outcome(s).tolist() == [0, 1, 2]
        ints = pd.Series(pd.Categorical([0, 2, 1], categories=[0, 1, 2], ordered=True))
        assert standardize_categorical_outcome(ints).tolist() == [0, 2, 1]


class TestCalibrationRange:
    @pytest.mark.parametrize("key", ["brier", "ici", "Brier", "ICI"])
    @pytest.mark.parametrize("val", [-0.01, 1.01])
    def test_out_of_range_rejected(self, key, val):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            validate_calibration_metrics({key: val}, provenance="apparent")

    def test_boundaries_and_slope_intercept_unaffected(self):
        out = validate_calibration_metrics(
            {"brier": 0.0, "ici": 1.0, "slope": 2.5, "intercept": -3.0},
            provenance="external",
        )
        assert out["slope"] == 2.5 and out["intercept"] == -3.0

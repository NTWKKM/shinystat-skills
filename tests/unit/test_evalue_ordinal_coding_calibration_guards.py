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


class TestNonOrderedMissingRejection:
    def test_non_ordered_categorical_or_series_with_missing_rejected(self):
        # Plain non-categorical series with NaNs
        s = pd.Series([1, 2, np.nan, 3])
        with pytest.raises(ValueError, match="missing"):
            standardize_categorical_outcome(s)

        # Unordered Categorical with NaNs
        s_cat = pd.Series(pd.Categorical(["A", "B", None], ordered=False))
        with pytest.raises(ValueError, match="missing"):
            standardize_categorical_outcome(s_cat)


class TestGeeCovarianceAndSmallClusterAdjustment:
    def test_gee_small_cluster_uses_bias_reduced(self):
        from medstat.models.multilevel import fit_gee

        rng = np.random.default_rng(42)
        n_clusters = 10  # < 40 threshold
        per_cluster = 5
        n = n_clusters * per_cluster
        df = pd.DataFrame(
            {
                "y": rng.normal(size=n),
                "x": rng.normal(size=n),
                "cluster": np.repeat(np.arange(n_clusters), per_cluster),
            }
        )
        res = fit_gee(
            y=df["y"],
            X=df[["x"]],
            cluster_ids=df["cluster"],
            family="gaussian",
            cov_struct="independence",
        )
        assert res["n_clusters"] == 10
        assert res["cov_type"] == "bias_reduced"
        assert res["small_cluster_adjustment"] is True

    def test_gee_large_cluster_uses_robust(self):
        from medstat.models.multilevel import fit_gee

        rng = np.random.default_rng(42)
        n_clusters = 50  # >= 40 threshold
        per_cluster = 2
        n = n_clusters * per_cluster
        df = pd.DataFrame(
            {
                "y": rng.normal(size=n),
                "x": rng.normal(size=n),
                "cluster": np.repeat(np.arange(n_clusters), per_cluster),
            }
        )
        res = fit_gee(
            y=df["y"],
            X=df[["x"]],
            cluster_ids=df["cluster"],
            family="gaussian",
            cov_struct="independence",
        )
        assert res["n_clusters"] == 50
        assert res["cov_type"] == "robust"
        assert res["small_cluster_adjustment"] is False


class TestOrdinalThresholdCutpointScale:
    def test_threshold_df_on_cutpoint_scale_with_delta_method_ci(self):
        from medstat.models.ordinal import fit_proportional_odds

        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(size=n)
        latent = 0.8 * x + rng.logistic(size=n)
        # 3 categories: cutpoints around -0.5 and 1.0
        y = np.digitize(latent, [-0.5, 1.0])
        res = fit_proportional_odds(pd.Series(y), pd.DataFrame({"x": x}))

        thresh_df = res["threshold_df"]
        assert len(thresh_df) == 2
        assert "scale" in thresh_df.columns
        assert (thresh_df["scale"] == "cutpoint").all()

        # Actual cutpoint values should be monotonic increasing
        c1, c2 = thresh_df["coef"].iloc[0], thresh_df["coef"].iloc[1]
        assert c1 < c2
        # Standard errors should be positive and finite
        assert (thresh_df["std_error"] > 0).all()
        assert np.isfinite(thresh_df["std_error"]).all()

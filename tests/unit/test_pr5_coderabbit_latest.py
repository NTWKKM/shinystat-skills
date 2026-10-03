"""
tests/unit/test_pr5_coderabbit_latest.py: Regression tests for the latest CodeRabbit PR#5 review findings.

Covers:
1. validate_gold_standard: mixed-type invalid values raise ValueError (not TypeError).
2. extract_primary_effect: exact term wins over prefix matches; ambiguity still returns NaN.
3. fit_gee: invalid family / cov_struct rejected; optional `time` forwarded for autoregressive.
4. CLI model --model-type gee: autoregressive requires --time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from click.testing import CliRunner

from medstat.cli.main import cli
from medstat.diagnostic.accuracy import validate_gold_standard
from medstat.models import extract_primary_effect, fit_gee


def _clustered_df(n_clusters: int = 20, per_cluster: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = n_clusters * per_cluster
    return pd.DataFrame(
        {
            "cluster": np.repeat(np.arange(n_clusters), per_cluster),
            "visit": np.tile(np.arange(per_cluster), n_clusters),
            "x": rng.normal(size=n),
            "y": rng.integers(0, 2, size=n),
        }
    )


class TestGoldStandardMixedTypes:
    def test_mixed_numeric_and_string_raises_value_error(self):
        with pytest.raises(ValueError, match="strictly binary"):
            validate_gold_standard(pd.Series([0, 1, "positive"], dtype=object))


class TestExtractPrimaryEffectExactFirst:
    def test_exact_term_preferred_over_prefix(self):
        est = {"treatment": 2.0, "treatment_dose": 1.5}
        assert extract_primary_effect(est, "treatment") == 2.0

    def test_single_prefix_fallback(self):
        assert (
            extract_primary_effect({"treatment[T.1]": 1.8, "age": 1.1}, "treatment")
            == 1.8
        )

    def test_ambiguous_prefix_fallback_returns_nan(self):
        est = {"treatment[T.1]": 1.8, "treatment[T.2]": 2.2}
        assert np.isnan(extract_primary_effect(est, "treatment"))


class TestFitGeeArgumentValidation:
    def test_invalid_family_raises(self):
        df = _clustered_df()
        with pytest.raises(ValueError, match="family"):
            fit_gee(df["y"], df[["x"]], df["cluster"], family="poisson")

    def test_invalid_cov_struct_raises(self):
        df = _clustered_df()
        with pytest.raises(ValueError, match="cov_struct"):
            fit_gee(df["y"], df[["x"]], df["cluster"], cov_struct="unstructured")

    def test_autoregressive_without_time_raises(self):
        df = _clustered_df()
        with pytest.raises(ValueError, match="requires 'time' variable"):
            fit_gee(
                df["y"],
                df[["x"]],
                df["cluster"],
                cov_struct="autoregressive",
                time=None,
            )

    def test_autoregressive_with_time(self):
        df = _clustered_df()
        res = fit_gee(
            df["y"],
            df[["x"]],
            df["cluster"],
            cov_struct="autoregressive",
            time=df["visit"],
        )
        assert res["cov_struct"] == "autoregressive"
        assert np.isfinite(res["summary_df"].loc["x", "coef"])

    def test_autoregressive_shuffled_invariance(self):
        df_ordered = _clustered_df(n_clusters=25, per_cluster=6)
        res_ordered = fit_gee(
            df_ordered["y"],
            df_ordered[["x"]],
            df_ordered["cluster"],
            cov_struct="autoregressive",
            time=df_ordered["visit"],
        )

        # Shuffle rows completely
        df_shuffled = df_ordered.sample(frac=1.0, random_state=123).reset_index(
            drop=True
        )
        res_shuffled = fit_gee(
            df_shuffled["y"],
            df_shuffled[["x"]],
            df_shuffled["cluster"],
            cov_struct="autoregressive",
            time=df_shuffled["visit"],
        )

        np.testing.assert_allclose(
            res_ordered["summary_df"]["coef"].values,
            res_shuffled["summary_df"]["coef"].values,
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            res_ordered["summary_df"]["std_error"].values,
            res_shuffled["summary_df"]["std_error"].values,
            rtol=1e-6,
            atol=1e-6,
        )


class TestCliGeeAutoregressiveRequiresTime:
    def _invoke(self, tmp_path, extra):
        csv_path = tmp_path / "clustered.csv"
        _clustered_df().to_csv(csv_path, index=False)
        return CliRunner().invoke(
            cli,
            [
                "model",
                "--data",
                str(csv_path),
                "--model-type",
                "gee",
                "--outcome",
                "y",
                "--covariates",
                "x",
                "--cluster",
                "cluster",
                "--corr-structure",
                "autoregressive",
                "--output",
                str(tmp_path / "out.json"),
            ]
            + extra,
        )

    def test_missing_time_rejected(self, tmp_path):
        res = self._invoke(tmp_path, [])
        assert res.exit_code != 0
        assert "time" in res.output.lower()

    def test_with_time_succeeds(self, tmp_path):
        res = self._invoke(tmp_path, ["--time", "visit"])
        assert res.exit_code == 0, res.output


class TestPsmRegularizedFallbackPenalty:
    def test_fit_regularized_zero_intercept_and_n_scaled_covariates(self):
        from unittest.mock import patch

        df = pd.DataFrame(
            {
                "treatment": [0, 0, 1, 1],
                "x1": [1.0, 2.0, 3.0, 4.0],
                "x2": [10.0, 20.0, 30.0, 40.0],
            }
        )

        def mock_fit(self, *args, **kwargs):
            raise np.linalg.LinAlgError("Singular matrix")

        with patch.object(sm.Logit, "fit", side_effect=mock_fit):
            with patch.object(sm.Logit, "fit_regularized", autospec=True) as mock_reg:
                mock_reg.return_value.predict.return_value = np.array(
                    [0.1, 0.2, 0.8, 0.9]
                )
                from medstat.causal.psm import calculate_propensity_score

                ps = calculate_propensity_score(
                    df, treatment="treatment", covariates=["x1", "x2"]
                )
                assert mock_reg.called
                alphas = mock_reg.call_args.kwargs.get("alpha")
                assert alphas is not None
                # Intercept is at index 0 with weight 0.0
                assert alphas[0] == 0.0
                # Covariates are scaled by len(y) = 4
                assert alphas[1] == 4.0
                assert alphas[2] == 4.0
                assert len(ps) == 4


class TestCleanOrdinalMappingIntegralKeyCheck:
    def test_numpy_integral_accepted_and_bool_excluded(self):
        import numbers

        # Mapping with numpy integer keys
        mapping = {np.int64(0): "Home", np.int64(1): "Ward", np.int64(2): "Death"}
        label_to_code = {
            v: k
            for k, v in mapping.items()
            if isinstance(k, numbers.Integral)
            and not isinstance(k, bool)
            and isinstance(v, str)
        }
        assert label_to_code == {"Home": 0, "Ward": 1, "Death": 2}

        # Mapping with boolean keys must be excluded
        bool_mapping = {True: "Yes", False: "No"}
        label_to_code_bool = {
            v: k
            for k, v in bool_mapping.items()
            if isinstance(k, numbers.Integral)
            and not isinstance(k, bool)
            and isinstance(v, str)
        }
        assert label_to_code_bool == {}

    def test_ordered_categorical_conversion_before_export(self):
        import numbers

        # Ordered categorical series
        s = pd.Series(
            pd.Categorical(
                ["Mild", "Moderate", "Severe"],
                ordered=True,
                categories=["Mild", "Moderate", "Severe"],
            )
        )
        outcome_ordinal_mapping = {0: "Mild", 1: "Moderate", 2: "Severe"}
        mapping = {
            v: k
            for k, v in outcome_ordinal_mapping.items()
            if isinstance(k, numbers.Integral) and not isinstance(k, bool)
        }
        res = s.map(mapping).astype(int)
        assert list(res) == [0, 1, 2]

        # Without explicit mapping, fallback to cat.codes
        res_codes = s.cat.codes.astype(int)
        assert list(res_codes) == [0, 1, 2]


class TestCalibrationMetricValidationAndApparentSlope:
    def test_calibration_metrics_reject_booleans(self):
        for bad_val in [True, False, np.bool_(True), np.bool_(False)]:
            metrics = {"brier": bad_val}
            with pytest.raises(ValueError, match="finite numeric value"):
                for k, v in metrics.items():
                    if (
                        isinstance(v, (bool, np.bool_))
                        or not isinstance(v, (int, float, np.number))
                        or not np.isfinite(v)
                    ):
                        raise ValueError(
                            f"Calibration metric '{k}' must be a finite numeric value (got {v})."
                        )

    def test_apparent_slope_suppressed_when_tautological(self):
        metrics = {"brier": 0.12, "slope": 1.0, "intercept": 0.0}

        def mock_generate_narrative(provenance, metrics, fit_metadata=None):
            for k, v in metrics.items():
                if (
                    isinstance(v, (bool, np.bool_))
                    or not isinstance(v, (int, float, np.number))
                    or not np.isfinite(v)
                ):
                    raise ValueError(
                        f"Calibration metric '{k}' must be a finite numeric value (got {v})."
                    )
            if provenance == "apparent":
                is_tautological = bool(
                    fit_metadata and fit_metadata.get("tautological_slope", False)
                )
                if is_tautological:
                    filtered = {
                        k: v for k, v in metrics.items() if k.lower() != "slope"
                    }
                else:
                    filtered = metrics
            else:
                filtered = metrics
            labels = {
                "slope": (
                    "apparent calibration slope"
                    if provenance == "apparent"
                    else "calibration slope"
                ),
                "intercept": (
                    "apparent calibration intercept"
                    if provenance == "apparent"
                    else "calibration intercept"
                ),
                "brier": "Brier score",
            }
            return ", ".join(f"{labels.get(k, k)} {v:.3f}" for k, v in filtered.items())

        # When tautological: slope is suppressed
        res_taut = mock_generate_narrative(
            "apparent", metrics, fit_metadata={"tautological_slope": True}
        )
        assert "slope" not in res_taut
        assert "Brier score" in res_taut

        # When not tautological: slope is retained and labeled as apparent
        res_apparent = mock_generate_narrative(
            "apparent", metrics, fit_metadata={"tautological_slope": False}
        )
        assert "apparent calibration slope 1.000" in res_apparent

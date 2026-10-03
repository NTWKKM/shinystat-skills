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
    def test_fit_regularized_zero_intercept_and_covariate_penalty(self):
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
                # Covariates have justified alpha=1.0 for statsmodels summed-likelihood objective
                assert alphas[1] == 1.0
                assert alphas[2] == 1.0
                assert len(ps) == 4

    def test_fit_regularized_with_binary_confounders_separation(self):
        from medstat.causal.psm import calculate_propensity_score

        # Dataset with binary confounders where bin1 causes separation
        df = pd.DataFrame(
            {
                "treatment": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                "bin1": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                "bin2": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            }
        )
        ps = calculate_propensity_score(
            df, treatment="treatment", covariates=["bin1", "bin2"]
        )
        assert len(ps) == 10
        assert ps.notna().all()
        assert ((ps > 0.0) & (ps < 1.0)).all()

    def test_varying_covariate_named_const_remains_penalized(self):
        from unittest.mock import patch

        df = pd.DataFrame(
            {
                "treatment": [0, 0, 1, 1],
                "const": [10.0, 20.0, 30.0, 40.0],  # varying user covariate named const
                "x1": [1.0, 2.0, 3.0, 4.0],
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

                calculate_propensity_score(
                    df, treatment="treatment", covariates=["const", "x1"]
                )
                assert mock_reg.called
                alphas = mock_reg.call_args.kwargs.get("alpha")
                # Intercept (added by add_constant) is at index 0 with weight 0.0
                assert alphas[0] == 0.0
                # Varying 'const' covariate (index 1) must remain penalized (alpha=1.0)
                assert alphas[1] == 1.0
                assert alphas[2] == 1.0


class TestCleanOrdinalMappingIntegralKeyCheck:
    def test_numpy_integral_accepted_and_bool_excluded(self):
        from medstat.data.clean import build_ordinal_mapping

        # Mapping with numpy integer keys
        mapping = {np.int64(0): "Home", np.int64(1): "Ward", np.int64(2): "Death"}
        label_to_code = build_ordinal_mapping(mapping)
        assert label_to_code == {"Home": 0, "Ward": 1, "Death": 2}

        # Mapping with boolean keys must be excluded
        bool_mapping = {True: "Yes", False: "No"}
        label_to_code_bool = build_ordinal_mapping(bool_mapping)
        assert label_to_code_bool == {}

    def test_ordered_categorical_conversion_before_export(self):
        from medstat.data.clean import standardize_categorical_outcome

        # 1. Label-valued categorical series converts via mapping
        s_label = pd.Series(
            pd.Categorical(
                ["Mild", "Moderate", "Severe"],
                ordered=True,
                categories=["Mild", "Moderate", "Severe"],
            )
        )
        mapping = {0: "Mild", 1: "Moderate", 2: "Severe"}
        res_label = standardize_categorical_outcome(s_label, mapping)
        assert list(res_label) == [0, 1, 2]

        # 2. Integer-coded categorical series is preserved and validated
        s_int = pd.Series(
            pd.Categorical(
                [0, 1, 2],
                ordered=True,
                categories=[0, 1, 2],
            )
        )
        res_int = standardize_categorical_outcome(s_int, mapping)
        assert list(res_int) == [0, 1, 2]

        # 3. Without mapping, fallback to cat.codes
        res_codes = standardize_categorical_outcome(s_label, None)
        assert list(res_codes) == [0, 1, 2]

        # 4. Unrecognized integer code raises ValueError
        s_bad = pd.Series(pd.Categorical([99], ordered=True, categories=[99]))
        with pytest.raises(ValueError, match="not recognized"):
            standardize_categorical_outcome(s_bad, mapping)


class TestCalibrationMetricValidationAndApparentSlope:
    def test_calibration_metrics_reject_booleans(self):
        from medstat.reporting.narrative import validate_calibration_metrics

        for bad_val in [True, False, np.bool_(True), np.bool_(False)]:
            with pytest.raises(ValueError, match="finite numeric value"):
                validate_calibration_metrics({"brier": bad_val}, provenance="apparent")

    def test_apparent_slope_excluded_for_apparent_provenance(self):
        from medstat.reporting.narrative import validate_calibration_metrics

        metrics = {"brier": 0.12, "slope": 1.0, "intercept": 0.0}
        # In apparent provenance, slope is excluded per TRIPOD
        filtered = validate_calibration_metrics(metrics, provenance="apparent")
        assert "slope" not in filtered
        assert "brier" in filtered
        assert "intercept" in filtered

    def test_external_validation_requires_full_quartet(self):
        from medstat.reporting.narrative import validate_calibration_metrics

        # Missing ici
        with pytest.raises(ValueError, match="External validation requires"):
            validate_calibration_metrics(
                {"brier": 0.12, "slope": 0.98, "intercept": 0.01},
                provenance="external",
            )

        # Full quartet succeeds
        full = {
            "brier": 0.12,
            "slope": 0.98,
            "intercept": 0.01,
            "ici": 0.02,
        }
        res = validate_calibration_metrics(full, provenance="external")
        assert len(res) == 4

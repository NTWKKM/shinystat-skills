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


class TestPsmPropensityModelFailsClosed:
    """ADR 82 replaces the ADR 24 fallback chain (logit, then Firth, then L1)."""

    def test_singular_fit_raises_without_a_regularized_refit(self):
        from unittest.mock import patch

        from medstat.causal.psm import calculate_propensity_score

        df = pd.DataFrame(
            {
                "treatment": [0, 1, 0, 1, 0, 1, 0, 1],
                "x1": [1.0, 2.0, 3.0, 4.0, 2.5, 1.5, 3.5, 0.5],
                "x2": [10.0, 5.0, 30.0, 20.0, 15.0, 25.0, 8.0, 12.0],
            }
        )

        def mock_fit(self, *args, **kwargs):
            raise np.linalg.LinAlgError("Singular matrix")

        with patch.object(sm.Logit, "fit", side_effect=mock_fit, autospec=True):
            with patch.object(sm.Logit, "fit_regularized", autospec=True) as mock_reg:
                with pytest.raises(ValueError, match="Singular matrix"):
                    calculate_propensity_score(
                        df, treatment="treatment", covariates=["x1", "x2"]
                    )
        assert not mock_reg.called

    def test_binary_confounder_separation_raises_then_firth_is_explicit(self):
        from medstat.causal.psm import calculate_propensity_score

        # Dataset with binary confounders where bin1 causes separation
        df = pd.DataFrame(
            {
                "treatment": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                "bin1": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                "bin2": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            }
        )
        with pytest.raises(ValueError, match="separation"):
            calculate_propensity_score(
                df, treatment="treatment", covariates=["bin1", "bin2"]
            )
        ps = calculate_propensity_score(
            df, treatment="treatment", covariates=["bin1", "bin2"], method="firth"
        )
        assert len(ps) == 10
        assert ps.notna().all()
        assert ((ps > 0.0) & (ps < 1.0)).all()

    def test_varying_covariate_named_const_is_kept_as_a_covariate(self):
        from medstat.causal.psm import calculate_propensity_score

        rng = np.random.default_rng(5)
        n = 200
        const = rng.normal(20.0, 5.0, n)  # varying user covariate named const
        x1 = rng.normal(size=n)
        trt = rng.binomial(1, 1 / (1 + np.exp(-(0.1 * (const - 20.0) + 0.5 * x1))))
        df = pd.DataFrame({"treatment": trt, "const": const, "x1": x1})
        ps = calculate_propensity_score(
            df, treatment="treatment", covariates=["const", "x1"]
        )
        design = sm.add_constant(df[["const", "x1"]], has_constant="add")
        expected = sm.Logit(trt, design).fit(disp=False).predict()
        np.testing.assert_allclose(ps.to_numpy(), expected, rtol=1e-10)


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

        # 5. Whole-number float categories recognized as integer-coded
        s_float_cat = pd.Series(
            pd.Categorical([1.0, 2.0, 0.0], ordered=True, categories=[0.0, 1.0, 2.0])
        )
        res_float = standardize_categorical_outcome(s_float_cat, mapping)
        assert list(res_float) == [1, 2, 0]
        assert res_float.dtype == int

        # 6. Fractional numeric values in plain Series are rejected
        s_frac = pd.Series([0.5, 1.0, 2.0])
        with pytest.raises(ValueError, match="non-integral"):
            standardize_categorical_outcome(s_frac)


class TestCalibrationMetricValidationAndApparentSlope:
    def test_calibration_metrics_reject_booleans(self):
        from medstat.reporting.narrative import validate_calibration_metrics

        for bad_val in [True, False, np.bool_(True), np.bool_(False)]:
            with pytest.raises(ValueError, match="finite numeric value"):
                validate_calibration_metrics({"brier": bad_val}, provenance="apparent")

    def test_apparent_slope_excluded_for_apparent_provenance(self):
        from medstat.reporting.narrative import validate_calibration_metrics

        metrics = {"brier": 0.12, "slope": 1.0, "intercept": 0.0}
        # In apparent provenance with tautological_slope=True, slope & intercept are excluded
        filtered_tauto = validate_calibration_metrics(
            metrics, provenance="apparent", tautological_slope=True
        )
        assert "slope" not in filtered_tauto
        assert "intercept" not in filtered_tauto
        assert "brier" in filtered_tauto

        # In apparent provenance without tautological_slope, all metrics are retained so they can be reported as apparent
        filtered_non_tauto = validate_calibration_metrics(
            metrics, provenance="apparent", tautological_slope=False
        )
        assert "slope" in filtered_non_tauto
        assert "intercept" in filtered_non_tauto
        assert "brier" in filtered_non_tauto

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

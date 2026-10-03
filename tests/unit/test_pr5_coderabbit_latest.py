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

"""
E2E integration tests for CLI model --type ordinal, gee, mixed.
"""

import json

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from medstat.cli.main import cli


@pytest.fixture
def sample_ordinal_csv(tmp_path):
    np.random.seed(42)
    n = 200
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.binomial(1, 0.5, n)
    latent = 0.8 * x1 - 0.5 * x2 + np.random.logistic(0, 1, n)
    y = np.zeros(n, dtype=int)
    y[latent > -0.5] = 1
    y[latent > 1.0] = 2

    df = pd.DataFrame({"mrs_score": y, "age": x1, "treatment": x2})
    csv_path = tmp_path / "ordinal_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_clustered_csv(tmp_path):
    np.random.seed(42)
    n_clusters = 10
    cluster_size = 20
    N = n_clusters * cluster_size

    hospital_id = np.repeat(np.arange(n_clusters), cluster_size)
    u_i = np.repeat(np.random.normal(0, 0.7, n_clusters), cluster_size)
    x = np.random.normal(0, 1, N)

    logit_p = 0.3 + 0.8 * x + u_i
    p = 1.0 / (1.0 + np.exp(-logit_p))
    y_bin = np.random.binomial(1, p)
    y_cont = 10.0 + 2.0 * x + u_i + np.random.normal(0, 1.0, N)

    df = pd.DataFrame(
        {
            "mortality": y_bin,
            "recovery_days": y_cont,
            "statin": x,
            "hospital_id": hospital_id,
        }
    )
    csv_path = tmp_path / "clustered_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_cli_model_ordinal_with_po_test(sample_ordinal_csv, tmp_path):
    runner = CliRunner()
    out_json = tmp_path / "ordinal_res.json"

    res = runner.invoke(
        cli,
        [
            "model",
            "--data",
            str(sample_ordinal_csv),
            "--type",
            "ordinal",
            "--outcome",
            "mrs_score",
            "--exposure",
            "treatment",
            "--covariates",
            "age",
            "--po-test",
            "--e-value",
            "--output",
            str(out_json),
        ],
    )

    assert res.exit_code == 0, res.output
    assert out_json.exists()

    with open(out_json) as f:
        data = json.load(f)

    assert data["model_type"] == "ordinal"
    assert "coefficients" in data
    assert "thresholds" in data
    assert "proportional_odds_test" in data
    assert "e_value" in data

    # Check that treatment is present in coefficients
    coef_terms = [c["term"] for c in data["coefficients"]]
    assert "treatment" in coef_terms
    assert "age" in coef_terms


def test_cli_model_gee(sample_clustered_csv, tmp_path):
    runner = CliRunner()
    out_json = tmp_path / "gee_res.json"

    res = runner.invoke(
        cli,
        [
            "model",
            "--data",
            str(sample_clustered_csv),
            "--type",
            "gee",
            "--outcome",
            "mortality",
            "--exposure",
            "statin",
            "--cluster",
            "hospital_id",
            "--corr-structure",
            "exchangeable",
            "--output",
            str(out_json),
        ],
    )

    assert res.exit_code == 0, res.output
    assert out_json.exists()

    with open(out_json) as f:
        data = json.load(f)

    assert data["model_type"] == "gee"
    assert data["family"] == "binomial"
    assert data["n_clusters"] == 10
    assert "clustering_diagnostics" in data
    assert data["clustering_diagnostics"]["design_effect"] >= 1.0


def test_cli_model_mixed(sample_clustered_csv, tmp_path):
    runner = CliRunner()
    out_json = tmp_path / "mixed_res.json"

    res = runner.invoke(
        cli,
        [
            "model",
            "--data",
            str(sample_clustered_csv),
            "--type",
            "mixed",
            "--outcome",
            "recovery_days",
            "--exposure",
            "statin",
            "--cluster",
            "hospital_id",
            "--output",
            str(out_json),
        ],
    )

    assert res.exit_code == 0, res.output
    assert out_json.exists()

    with open(out_json) as f:
        data = json.load(f)

    assert data["model_type"] == "mixed"
    assert data["n_clusters"] == 10
    assert "random_intercept_var" in data
    assert "residual_var" in data
    assert "cluster_icc" in data
    assert data["cluster_icc"] > 0

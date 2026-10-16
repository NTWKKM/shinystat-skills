"""
tests/e2e/test_validation_riley_cli.py: End-to-end CLI and AnalysisPlan YAML tests
for bootstrap internal validation, recalibration, and Riley / diagnostic AUROC sample size.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from click.testing import CliRunner
from scipy.special import expit

from medstat.cli.main import cli
from medstat.cli.spec import AnalysisPlan


def test_cli_diag_internal_validation(tmp_path: Path):
    """Test 'medstat diag --internal-validation' end-to-end."""
    runner = CliRunner()
    csv_file = tmp_path / "cohort.csv"
    out_json = tmp_path / "val_results.json"

    np.random.seed(42)
    n = 150
    x1 = np.random.normal(0, 1, size=n)
    x2 = np.random.normal(0, 1, size=n)
    prob = expit(-0.5 + 0.8 * x1 - 0.6 * x2)
    y = (np.random.uniform(size=n) < prob).astype(int)

    df = pd.DataFrame({"outcome": y, "x1": x1, "x2": x2})
    df.to_csv(csv_file, index=False)

    result = runner.invoke(
        cli,
        [
            "diag",
            "--data",
            str(csv_file),
            "--gold-standard",
            "outcome",
            "--internal-validation",
            "--predictors",
            "x1,x2",
            "--n-bootstrap",
            "20",
            "--output",
            str(out_json),
        ],
    )
    assert result.exit_code == 0, f"CLI error: {result.output}"
    assert out_json.exists()

    with open(out_json) as f:
        res = json.load(f)

    assert "internal_validation" in res
    val_data = res["internal_validation"]
    assert "apparent_metrics" in val_data
    assert "metrics" in val_data
    assert "shrinkage_factor" in val_data
    assert "shrunk_coefficients" in val_data
    assert 0.01 <= val_data["shrinkage_factor"] <= 1.0


def test_cli_diag_recalibration(tmp_path: Path):
    """Test 'medstat diag --recalibration' end-to-end."""
    runner = CliRunner()
    csv_file = tmp_path / "recal_cohort.csv"
    out_json = tmp_path / "recal_results.json"

    np.random.seed(42)
    n = 150
    x = np.random.normal(0, 1, size=n)
    y = (np.random.uniform(size=n) < expit(0.5 * x)).astype(int)
    p = np.clip(expit(-1.0 + 1.8 * x), 0.01, 0.99)

    df = pd.DataFrame({"y": y, "p_pred": p})
    df.to_csv(csv_file, index=False)

    result = runner.invoke(
        cli,
        [
            "diag",
            "--data",
            str(csv_file),
            "--gold-standard",
            "y",
            "--recalibration",
            "--p-recal",
            "p_pred",
            "--recalibration-method",
            "logistic",
            "--output",
            str(out_json),
        ],
    )
    assert result.exit_code == 0, f"CLI error: {result.output}"
    assert out_json.exists()

    with open(out_json) as f:
        res = json.load(f)

    assert "recalibration" in res
    recal_data = res["recalibration"]
    assert "pre" in recal_data
    assert "post" in recal_data
    assert "recalibration_parameters" in recal_data
    # Patient-level array must be stripped in output JSON
    assert "p_recalibrated" not in recal_data


def test_cli_sample_size_riley(tmp_path: Path):
    """Test 'medstat sample-size --type riley' end-to-end."""
    runner = CliRunner()
    out_json = tmp_path / "riley_out.json"

    result = runner.invoke(
        cli,
        [
            "sample-size",
            "--type",
            "riley",
            "--n-parameters",
            "10",
            "--prevalence",
            "0.20",
            "--cox-snell-r2",
            "0.15",
            "--shrinkage-target",
            "0.90",
            "--output",
            str(out_json),
        ],
    )
    assert result.exit_code == 0, f"CLI error: {result.output}"
    assert out_json.exists()

    with open(out_json) as f:
        res = json.load(f)

    assert res["test_type"] == "riley"
    assert res["total_sample_size"] >= 549
    assert res["details"]["criteria"]["criterion3_shrinkage"]["n"] == 549


def test_cli_sample_size_diagnostic_auroc(tmp_path: Path):
    """Test 'medstat sample-size --type diagnostic-auroc' end-to-end."""
    runner = CliRunner()
    out_json = tmp_path / "auroc_out.json"

    result = runner.invoke(
        cli,
        [
            "sample-size",
            "--type",
            "diagnostic-auroc",
            "--auroc-expected",
            "0.85",
            "--auroc-null",
            "0.50",
            "--prevalence",
            "0.25",
            "--power",
            "0.80",
            "--output",
            str(out_json),
        ],
    )
    assert result.exit_code == 0, f"CLI error: {result.output}"
    assert out_json.exists()

    with open(out_json) as f:
        res = json.load(f)

    assert res["test_type"] == "diagnostic-auroc"
    assert res["required_cases"] > 0
    assert res["required_controls"] > 0
    assert res["total_sample_size"] == res["required_cases"] + res["required_controls"]


def test_spec_yaml_analysis_plan_execution(tmp_path: Path):
    """Test AnalysisPlan executing internal_validation and recalibration via spec."""
    np.random.seed(42)
    n = 200
    x1 = np.random.normal(0, 1, size=n)
    x2 = np.random.normal(0, 1, size=n)
    p = np.clip(expit(-0.5 + 0.8 * x1 + 0.5 * x2), 0.01, 0.99)
    y = (np.random.uniform(size=n) < p).astype(int)

    df = pd.DataFrame({"outcome": y, "x1": x1, "x2": x2, "p_pred": p})
    csv_file = tmp_path / "spec_data.csv"
    df.to_csv(csv_file, index=False)

    plan_yaml = f"""
metadata:
  study_title: "Validation & Recalibration SAP"
  protocol_id: "SAP-VAL-001"
data:
  input_path: "{csv_file}"
  filters: []
models:
  - name: "validation_model"
    type: "internal_validation"
    outcome: "outcome"
    covariates: ["x1", "x2"]
    options:
      n_bootstrap: 15
  - name: "recalibration_model"
    type: "recalibration"
    outcome: "outcome"
    exposure: "p_pred"
    options:
      method: "logistic"
reporting:
  style: "nejm"
  format: "html"
"""
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(plan_yaml)

    plan = AnalysisPlan.from_yaml(str(plan_file))
    results = plan.execute()

    assert "models" in results
    assert "reports" in results

    # Validation model checks
    assert "validation_model" in results["models"]
    val_res = results["models"]["validation_model"]
    assert "shrinkage_factor" in val_res
    assert "reports" in results
    assert "validation_model" in results["reports"]
    assert "<table" in results["reports"]["validation_model"]["html_table"]
    assert "TRIPOD+AI" in results["reports"]["validation_model"]["methods_narrative"]

    # Recalibration model checks
    assert "recalibration_model" in results["models"]
    recal_res = results["models"]["recalibration_model"]
    assert "recalibration_parameters" in recal_res
    assert "p_recalibrated" not in recal_res
    assert "y_true" not in recal_res
    assert "p_orig" not in recal_res
    assert "recalibration_model" in results["reports"]
    assert "<table" in results["reports"]["recalibration_model"]["html_table"]

"""
tests/e2e/test_validation_riley_cli.py: End-to-end CLI and AnalysisPlan YAML tests
for bootstrap internal validation, recalibration, and Riley / diagnostic AUROC sample size.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from scipy.special import expit
from sklearn.metrics import roc_auc_score

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
    options:
      p_recal: "p_pred"
      method: "logistic"
  - name: "logistic_with_p_pred_recal"
    type: "logistic"
    outcome: "outcome"
    exposure: "x1"
    covariates: ["x2"]
    options:
      recalibration: true
      p_pred: "p_pred"
      recalibration_method: "logistic"
  - name: "recalibration_via_options_p_pred"
    type: "recalibration"
    outcome: "outcome"
    options:
      p_pred: "p_pred"
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

    expected_auc = float(roc_auc_score(y, p))
    expected_brier = float(np.mean((y - p) ** 2))

    # Recalibration model checks (options: {p_recal: "p_pred"})
    assert "recalibration_model" in results["models"]
    recal_res = results["models"]["recalibration_model"]
    assert "recalibration_parameters" in recal_res
    assert "p_recalibrated" not in recal_res
    assert "y_true" not in recal_res
    assert "p_orig" not in recal_res
    assert "recalibration_model" in results["reports"]
    assert "<table" in results["reports"]["recalibration_model"]["html_table"]
    # Verify pre-calibration AUROC and Brier score against fixture ground-truth y and p
    assert np.isclose(recal_res["pre"]["auroc"], expected_auc, atol=1e-5)
    assert np.isclose(recal_res["pre"]["brier_score"], expected_brier, atol=1e-5)

    # Logistic model with options: {recalibration: true, p_pred: "p_pred"} (without p_recal)
    assert "logistic_with_p_pred_recal" in results["models"]
    log_recal = results["models"]["logistic_with_p_pred_recal"]
    assert "recalibration" in log_recal
    assert "recalibration_parameters" in log_recal["recalibration"]
    assert "pre" in log_recal["recalibration"]
    assert "post" in log_recal["recalibration"]
    # Check that logistic_with_p_pred_recal preserves AUROC and has post-calibration slope near 1.0
    log_recal_pre = log_recal["recalibration"]["pre"]
    log_recal_post = log_recal["recalibration"]["post"]
    assert np.isclose(log_recal_post["auroc"], log_recal_pre["auroc"], atol=1e-5)
    assert np.isclose(log_recal_post["calibration_slope"], 1.0, atol=0.05)

    # Recalibration model via options: {p_pred: "p_pred"} (without p_recal, exposure, or covariates)
    assert "recalibration_via_options_p_pred" in results["models"]
    opt_recal = results["models"]["recalibration_via_options_p_pred"]
    assert "recalibration_parameters" in opt_recal
    assert "pre" in opt_recal
    assert "post" in opt_recal
    # Compare standalone recalibration_via_options_p_pred pre/post metrics with recalibration_model
    assert np.isclose(opt_recal["pre"]["auroc"], recal_res["pre"]["auroc"], atol=1e-6)
    assert np.isclose(
        opt_recal["pre"]["brier_score"], recal_res["pre"]["brier_score"], atol=1e-6
    )
    assert np.isclose(opt_recal["post"]["auroc"], recal_res["post"]["auroc"], atol=1e-6)
    assert np.isclose(
        opt_recal["post"]["brier_score"], recal_res["post"]["brier_score"], atol=1e-6
    )
    assert np.isclose(
        opt_recal["post"]["calibration_slope"],
        recal_res["post"]["calibration_slope"],
        atol=1e-6,
    )


def test_spec_yaml_recalibration_validation_errors(tmp_path: Path):
    """Verify ValueError with resolved column name when recalibration column is missing/invalid."""
    df = pd.DataFrame(
        {
            "outcome": [0, 1] * 20,
            "x1": np.linspace(-1.0, 1.0, 40),
            "score": np.linspace(0.1, 0.9, 40),
        }
    )
    csv_file = tmp_path / "err_data.csv"
    df.to_csv(csv_file, index=False)

    # 1. Logistic with recalibration enabled but unresolved p_recal/p_pred
    yaml_log_missing = f"""
metadata:
  study_title: "Err"
data:
  input_path: "{csv_file}"
models:
  - name: "log_err"
    type: "logistic"
    outcome: "outcome"
    covariates: ["x1"]
    options:
      recalibration: true
"""
    f1 = tmp_path / "p1.yaml"
    f1.write_text(yaml_log_missing)
    with pytest.raises(
        ValueError,
        match=r"Recalibration requires a valid predicted probability column.*Resolved 'None'",
    ):
        AnalysisPlan.from_yaml(str(f1)).execute()

    # 2. Logistic with recalibration enabled and nonexistent p_pred column
    yaml_log_nonexistent = f"""
metadata:
  study_title: "Err"
data:
  input_path: "{csv_file}"
models:
  - name: "log_nonexistent"
    type: "logistic"
    outcome: "outcome"
    covariates: ["x1"]
    options:
      recalibration: true
      p_pred: "nonexistent_col"
"""
    f2 = tmp_path / "p2.yaml"
    f2.write_text(yaml_log_nonexistent)
    with pytest.raises(
        ValueError,
        match=r"Recalibration requires a valid predicted probability column.*Resolved 'nonexistent_col'",
    ):
        AnalysisPlan.from_yaml(str(f2)).execute()

    # 3. Standalone recalibration with nonexistent column
    yaml_recal_absent = f"""
metadata:
  study_title: "Err"
data:
  input_path: "{csv_file}"
models:
  - name: "recal_err"
    type: "recalibration"
    outcome: "outcome"
    options:
      p_recal: "nonexistent_prob_col"
"""
    f3 = tmp_path / "p3.yaml"
    f3.write_text(yaml_recal_absent)
    with pytest.raises(
        ValueError,
        match=r"Recalibration requires a valid predicted probability column.*Resolved 'nonexistent_prob_col'",
    ):
        AnalysisPlan.from_yaml(str(f3)).execute()

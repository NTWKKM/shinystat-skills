"""tests/e2e/test_reclassification_tdroc_cli.py

E2E integration tests for medstat CLI diagnostic reclassification and
time-dependent ROC workflows, including YAML SAP execution.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from medstat.cli.main import cli


@pytest.fixture
def sample_reclass_csv(tmp_path: Path) -> Path:
    np.random.seed(42)
    n = 200
    y = np.random.binomial(1, 0.35, size=n)
    p_old = np.clip(0.1 + 0.3 * y + np.random.normal(0, 0.1, size=n), 0.02, 0.98)
    p_new = np.clip(0.05 + 0.6 * y + np.random.normal(0, 0.08, size=n), 0.02, 0.98)

    df = pd.DataFrame(
        {
            "patient_id": [f"PT_{i:04d}" for i in range(n)],
            "outcome": y,
            "pred_old": p_old,
            "pred_new": p_new,
        }
    )
    p = tmp_path / "reclass_data.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def sample_tdroc_csv(tmp_path: Path) -> Path:
    np.random.seed(42)
    n = 200
    marker1 = np.random.normal(0, 1, size=n)
    marker2 = marker1 * 0.8 + np.random.normal(0, 0.5, size=n)
    hazard = 0.05 * np.exp(0.8 * marker2)
    durations = np.random.exponential(1.0 / hazard)
    censoring = np.random.exponential(30.0, size=n)
    obs_time = np.minimum(durations, censoring)
    events = (durations <= censoring).astype(int)

    df = pd.DataFrame(
        {
            "patient_id": [f"PT_{i:04d}" for i in range(n)],
            "time_to_event": obs_time,
            "death_status": events,
            "biomarker_baseline": marker1,
            "biomarker_updated": marker2,
        }
    )
    p = tmp_path / "tdroc_data.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def sample_combined_csv(tmp_path: Path) -> Path:
    np.random.seed(42)
    n = 200
    y = np.random.binomial(1, 0.35, size=n)
    p_old = np.clip(0.1 + 0.3 * y + np.random.normal(0, 0.1, size=n), 0.02, 0.98)
    p_new = np.clip(0.05 + 0.6 * y + np.random.normal(0, 0.08, size=n), 0.02, 0.98)
    marker1 = np.random.normal(0, 1, size=n)
    marker2 = marker1 * 0.8 + np.random.normal(0, 0.5, size=n)
    hazard = 0.05 * np.exp(0.8 * marker2)
    durations = np.random.exponential(1.0 / hazard)
    censoring = np.random.exponential(30.0, size=n)
    obs_time = np.minimum(durations, censoring)
    events = (durations <= censoring).astype(int)

    df = pd.DataFrame(
        {
            "patient_id": [f"PT_{i:04d}" for i in range(n)],
            "outcome": y,
            "pred_old": p_old,
            "pred_new": p_new,
            "time_to_event": obs_time,
            "death_status": events,
            "biomarker_baseline": marker1,
            "biomarker_updated": marker2,
        }
    )
    p = tmp_path / "combined_data.csv"
    df.to_csv(p, index=False)
    return p


def test_cli_diag_reclassification(sample_reclass_csv: Path, tmp_path: Path):
    """Test medstat diag --reclassification via CLI."""
    runner = CliRunner()
    out_json = tmp_path / "reclass_out.json"

    res = runner.invoke(
        cli,
        [
            "diag",
            "--data",
            str(sample_reclass_csv),
            "--gold",
            "outcome",
            "--p-old",
            "pred_old",
            "--p-new",
            "pred_new",
            "--thresholds",
            "0.2,0.5",
            "--reclassification",
            "--output",
            str(out_json),
        ],
    )
    assert res.exit_code == 0, f"CLI error: {res.output}"
    assert out_json.exists()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert "reclassification" in data
    reclass = data["reclassification"]
    assert "nri_continuous" in reclass
    assert "nri_categorical" in reclass
    assert "idi" in reclass
    assert reclass["nri_continuous"]["total"]["estimate"] > 0
    assert reclass["idi"]["estimate"] > 0
    # Patient-level arrays must be stripped from serialized JSON output
    assert "p_old" not in reclass
    assert "p_new" not in reclass
    assert "y_true" not in reclass


def test_cli_diag_reclassification_requires_gold_standard(sample_reclass_csv: Path):
    """Verify that reclassification strictly requires --gold-standard and rejects --event-col."""
    runner = CliRunner()
    res = runner.invoke(
        cli,
        [
            "diag",
            "--data",
            str(sample_reclass_csv),
            "--event-col",
            "outcome",
            "--p-old",
            "pred_old",
            "--p-new",
            "pred_new",
            "--reclassification",
        ],
    )
    assert res.exit_code != 0
    assert (
        "Binary reclassification (NRI/IDI) requires a fully ascertained binary outcome via '--gold-standard'"
        in res.output
    )
    assert (
        "Censored survival event indicators ('--event') are invalid for binary reclassification"
        in res.output
    )


def test_cli_diag_time_dependent_roc(sample_tdroc_csv: Path, tmp_path: Path):
    """Test medstat diag --time-dependent-roc via CLI."""
    runner = CliRunner()
    out_json = tmp_path / "tdroc_out.json"

    res = runner.invoke(
        cli,
        [
            "diag",
            "--data",
            str(sample_tdroc_csv),
            "--gold",
            "death_status",
            "--test",
            "biomarker_updated",
            "--time-dependent-roc",
            "--time-col",
            "time_to_event",
            "--event-col",
            "death_status",
            "--time-horizons",
            "10,25",
            "--output",
            str(out_json),
        ],
    )
    assert res.exit_code == 0, f"CLI error: {res.output}"
    assert out_json.exists()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert "time_dependent_roc" in data
    td = data["time_dependent_roc"]
    assert "results_by_horizon" in td
    assert "trajectory" in td
    assert len(td["trajectory"]["auroc"]) == 2


def test_cli_diag_dual_model_comparison(sample_reclass_csv: Path, tmp_path: Path):
    """Test medstat diag with dual models (--p-old and --p-new) running paired DeLong and NRI/IDI."""
    runner = CliRunner()
    out_json = tmp_path / "comp_out.json"

    res = runner.invoke(
        cli,
        [
            "diag",
            "--data",
            str(sample_reclass_csv),
            "--gold",
            "outcome",
            "--p-old",
            "pred_old",
            "--p-new",
            "pred_new",
            "--thresholds",
            "0.3",
            "--output",
            str(out_json),
        ],
    )
    assert res.exit_code == 0, f"CLI error: {res.output}"
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert "reclassification" in data


def test_yaml_spec_reclassification_and_tdroc(
    sample_combined_csv: Path, tmp_path: Path
):
    """Test YAML SAP execution via medstat model --spec for reclassification and time-dependent ROC."""
    sap_yaml = tmp_path / "analysis_plan.yaml"
    out_json = tmp_path / "spec_results.json"

    sap_content = f"""
version: "1.0"
metadata:
  study_title: "Diagnostic Reclassification & TD-ROC Study"
  reporting_guideline: "STARD"
data:
  input_path: "{sample_combined_csv}"
  id_column: "patient_id"
variables:
  - name: "outcome"
    role: "outcome"
  - name: "death_status"
    role: "outcome"
  - name: "pred_old"
    role: "covariate"
  - name: "pred_new"
    role: "covariate"
  - name: "time_to_event"
    role: "covariate"
  - name: "biomarker_updated"
    role: "exposure"
models:
  - name: "reclass_analysis"
    type: "reclassification"
    outcome: "outcome"
    options:
      p_old: "pred_old"
      p_new: "pred_new"
      thresholds: [0.2, 0.5]

  - name: "tdroc_analysis"
    type: "time_dependent_roc"
    outcome: "death_status"
    time: "time_to_event"
    exposure: "biomarker_updated"
    options:
      time_horizons: [10.0, 25.0]
reporting:
  style: "nejm"
  format: "html"
"""
    sap_yaml.write_text(sap_content, encoding="utf-8")

    runner = CliRunner()
    res = runner.invoke(
        cli,
        [
            "model",
            "--spec",
            str(sap_yaml),
            "--output",
            str(out_json),
        ],
    )
    assert res.exit_code == 0, f"CLI spec error: {res.output}"
    assert out_json.exists()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert "models" in data
    assert "reclass_analysis" in data["models"]
    assert "tdroc_analysis" in data["models"]

    rec_res = data["models"]["reclass_analysis"]
    assert "reclassification" in rec_res
    assert "nri_continuous" in rec_res["reclassification"]

    td_res = data["models"]["tdroc_analysis"]
    assert "results_by_horizon" in td_res
    assert "trajectory" in td_res

    assert "reports" in data
    assert "reclass_analysis" in data["reports"]
    assert "tdroc_analysis" in data["reports"]

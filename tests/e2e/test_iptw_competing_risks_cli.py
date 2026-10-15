"""
E2E integration tests for medstat CLI IPTW and Competing Risks workflows.
Verifies command line execution for causal iptw, model competing_risks, and YAML SAP execution.
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
def sample_iptw_csv(tmp_path: Path) -> Path:
    np.random.seed(42)
    n = 200
    age = np.random.normal(55, 10, n)
    bmi = np.random.normal(25, 4, n)

    # Propensity score
    z = -1.5 + 0.02 * age + 0.03 * bmi
    ps = 1.0 / (1.0 + np.exp(-z))
    t = np.random.binomial(1, ps)

    # Outcomes
    y_cont = 100.0 - 5.0 * t + 0.5 * age + np.random.normal(0, 5, n)
    y_bin = np.random.binomial(1, 1.0 / (1.0 + np.exp(-(0.5 - 0.8 * t + 0.01 * age))))
    surv_time = np.random.exponential(20.0, n)
    surv_event = np.random.binomial(1, 0.7, n)

    df = pd.DataFrame(
        {
            "patient_id": [f"P_{i:04d}" for i in range(n)],
            "treatment": t,
            "age": age,
            "bmi": bmi,
            "outcome_cont": y_cont,
            "outcome_bin": y_bin,
            "time": surv_time,
            "event": surv_event,
        }
    )
    csv_path = tmp_path / "iptw_sample.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_competing_risks_csv(tmp_path: Path) -> Path:
    np.random.seed(42)
    n = 250
    treatment = np.random.binomial(1, 0.5, n)
    age = np.random.normal(60, 8, n)

    # Latent times for primary event (1) and competing risk (2)
    t1 = np.random.exponential(scale=30.0 / np.exp(0.4 * treatment), size=n)
    t2 = np.random.exponential(scale=40.0 / np.exp(-0.2 * treatment), size=n)
    censor_time = np.random.uniform(10.0, 50.0, size=n)

    t_obs = np.minimum(np.minimum(t1, t2), censor_time)
    event_obs = np.zeros(n, dtype=int)
    for i in range(n):
        if t1[i] <= t2[i] and t1[i] <= censor_time[i]:
            event_obs[i] = 1
        elif t2[i] < t1[i] and t2[i] <= censor_time[i]:
            event_obs[i] = 2
        else:
            event_obs[i] = 0

    df = pd.DataFrame(
        {
            "patient_id": [f"CR_{i:04d}" for i in range(n)],
            "time": t_obs,
            "event": event_obs,
            "treatment": treatment,
            "age": age,
        }
    )
    csv_path = tmp_path / "competing_risks_sample.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_cli_causal_iptw_basic_and_balance(sample_iptw_csv: Path, tmp_path: Path):
    runner = CliRunner()
    out_json = tmp_path / "iptw_res.json"
    love_json = tmp_path / "love.json"

    result = runner.invoke(
        cli,
        [
            "causal",
            "iptw",
            "--data",
            str(sample_iptw_csv),
            "--treatment",
            "treatment",
            "--covariates",
            "age,bmi",
            "--estimand",
            "ATE",
            "--balance-check",
            "--love-plot",
            str(love_json),
            "--output",
            str(out_json),
        ],
    )
    assert result.exit_code == 0, f"Error: {result.output}"
    assert "Austin (2009) IPTW Covariate Balance" in result.output
    assert out_json.exists()
    assert love_json.exists()

    data = json.loads(out_json.read_text())
    assert data["estimand"] == "ATE"
    assert "diagnostics" in data
    assert "effective_sample_size" in data
    assert "overall" in data["diagnostics"]
    assert "balance_table" in data
    assert len(data["balance_table"]) == 2


def test_cli_causal_iptw_with_outcome_models(sample_iptw_csv: Path, tmp_path: Path):
    runner = CliRunner()

    # 1. Continuous outcome with ATT
    res_cont = runner.invoke(
        cli,
        [
            "causal",
            "iptw",
            "--data",
            str(sample_iptw_csv),
            "--treatment",
            "treatment",
            "--covariates",
            "age,bmi",
            "--estimand",
            "ATT",
            "--outcome",
            "outcome_cont",
            "--outcome-type",
            "continuous",
        ],
    )
    assert res_cont.exit_code == 0
    cont_json = json.loads(res_cont.output)
    assert cont_json["estimand"] == "ATT"
    assert "outcome_model" in cont_json
    assert cont_json["outcome_model"]["model_type"] == "continuous"
    assert "coef" in cont_json["outcome_model"]

    # 2. Binary outcome with ATO
    res_bin = runner.invoke(
        cli,
        [
            "causal",
            "iptw",
            "--data",
            str(sample_iptw_csv),
            "--treatment",
            "treatment",
            "--covariates",
            "age,bmi",
            "--estimand",
            "ATO",
            "--outcome",
            "outcome_bin",
            "--outcome-type",
            "binary",
        ],
    )
    assert res_bin.exit_code == 0
    bin_json = json.loads(res_bin.output)
    assert bin_json["estimand"] == "ATO"
    assert bin_json["outcome_model"]["model_type"] == "binary"
    assert "odds_ratio" in bin_json["outcome_model"]


def test_cli_competing_risks_model_and_cif_plot(
    sample_competing_risks_csv: Path, tmp_path: Path
):
    runner = CliRunner()
    cif_png = tmp_path / "cif_plot.png"
    out_json = tmp_path / "cr_results.json"

    result = runner.invoke(
        cli,
        [
            "model",
            "--data",
            str(sample_competing_risks_csv),
            "--time",
            "time",
            "--outcome",
            "event",
            "--type",
            "competing_risks",
            "--covariates",
            "treatment,age",
            "--competing-event",
            "2",
            "--event-of-interest",
            "1",
            "--cif-plot",
            str(cif_png),
            "--output",
            str(out_json),
        ],
    )
    assert result.exit_code == 0, f"Error: {result.output}"
    assert out_json.exists()
    assert cif_png.exists()

    res_data = json.loads(out_json.read_text())
    assert res_data["model_type"] == "competing_risks"
    assert "comparison_table" in res_data
    assert "interpretation" in res_data
    assert len(res_data["comparison_table"]) == 2  # treatment and age


def test_cli_spec_yaml_iptw_execution(sample_iptw_csv: Path, tmp_path: Path):
    spec_content = f"""
version: "1.0"
metadata:
  study_title: "IPTW Causal Analysis Plan"
  reporting_guideline: "STROBE"
data:
  input_path: "{sample_iptw_csv}"
  id_column: "patient_id"
variables:
  - name: "treatment"
    role: "exposure"
  - name: "outcome_cont"
    role: "outcome"
  - name: "age"
    role: "covariate"
  - name: "bmi"
    role: "covariate"
models:
  - name: "iptw_analysis"
    type: "iptw"
    outcome: "outcome_cont"
    exposure: "treatment"
    covariates: ["age", "bmi"]
    options:
      estimand: "ATE"
      stabilized: true
      truncate_quantiles: [0.01, 0.99]
reporting:
  style: "nejm"
  format: "html"
"""
    spec_path = tmp_path / "iptw_sap.yaml"
    spec_path.write_text(spec_content)
    out_json = tmp_path / "sap_output.json"

    runner = CliRunner()
    result = runner.invoke(
        cli, ["model", "--spec", str(spec_path), "--output", str(out_json)]
    )
    assert result.exit_code == 0, f"Error: {result.output}"
    assert out_json.exists()

    res = json.loads(out_json.read_text())
    assert "iptw_analysis" in res["models"]
    iptw_mod = res["models"]["iptw_analysis"]
    assert "diagnostics" in iptw_mod
    assert "balance_table" in iptw_mod
    assert "iptw_analysis" in res["reports"]
    assert "methods_narrative" in res["reports"]["iptw_analysis"]
    assert (
        "Inverse Probability of Treatment Weighting"
        in res["reports"]["iptw_analysis"]["methods_narrative"]
    )


def test_cli_spec_yaml_competing_risks_execution(
    sample_competing_risks_csv: Path, tmp_path: Path
):
    spec_content = f"""
version: "1.0"
metadata:
  study_title: "Competing Risks Analysis Plan"
  reporting_guideline: "STROBE"
data:
  input_path: "{sample_competing_risks_csv}"
  id_column: "patient_id"
variables:
  - name: "time"
    role: "time"
  - name: "event"
    role: "outcome"
  - name: "treatment"
    role: "exposure"
  - name: "age"
    role: "covariate"
models:
  - name: "cr_analysis"
    type: "competing_risks"
    outcome: "event"
    exposure: "treatment"
    time: "time"
    covariates: ["treatment", "age"]
    options:
      event_of_interest: 1
      competing_event: 2
reporting:
  style: "nejm"
  format: "html"
"""
    spec_path = tmp_path / "cr_sap.yaml"
    spec_path.write_text(spec_content)
    out_json = tmp_path / "cr_sap_output.json"

    runner = CliRunner()
    result = runner.invoke(
        cli, ["model", "--spec", str(spec_path), "--output", str(out_json)]
    )
    assert result.exit_code == 0, f"Error: {result.output}"
    assert out_json.exists()

    res = json.loads(out_json.read_text())
    assert "cr_analysis" in res["models"]
    cr_mod = res["models"]["cr_analysis"]
    assert "comparison_table" in cr_mod
    assert "cr_analysis" in res["reports"]
    assert "methods_narrative" in res["reports"]["cr_analysis"]
    assert "Aalen-Johansen" in res["reports"]["cr_analysis"]["methods_narrative"]

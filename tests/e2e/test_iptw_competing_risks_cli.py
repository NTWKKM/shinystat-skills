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


def test_cli_competing_risks_categorical_covariates(tmp_path: Path):
    """Regression test for Finding 1: verify categorical covariates encoded in X_df work in regression."""
    np.random.seed(42)
    n = 150
    stages = np.random.choice(["Stage_I", "Stage_II", "Stage_III"], size=n)
    time = np.random.exponential(15.0, size=n)
    event = np.random.choice([0, 1, 2], size=n, p=[0.4, 0.4, 0.2])

    df = pd.DataFrame(
        {
            "id": range(n),
            "time": time,
            "event": event,
            "stage": stages,
        }
    )
    csv_path = tmp_path / "cr_cat.csv"
    df.to_csv(csv_path, index=False)

    runner = CliRunner()
    out_json = tmp_path / "cr_cat_out.json"
    result = runner.invoke(
        cli,
        [
            "model",
            "--data",
            str(csv_path),
            "--time",
            "time",
            "--outcome",
            "event",
            "--type",
            "competing_risks",
            "--covariates",
            "stage",
            "--competing-event",
            "2",
            "--event-of-interest",
            "1",
            "--output",
            str(out_json),
        ],
    )
    assert result.exit_code == 0, f"Error: {result.output}"
    assert out_json.exists()
    res_data = json.loads(out_json.read_text())
    assert res_data["model_type"] == "competing_risks"
    assert "comparison_table" in res_data
    # 3-level categorical stage yields 2 dummy columns
    assert len(res_data["comparison_table"]) == 2


def test_cli_iptw_invalid_binary_and_survival_validation(
    sample_iptw_csv: Path, tmp_path: Path
):
    """Regression test for Finding 2: non-0/1 outcome for binary/survival, missing --time for survival."""
    runner = CliRunner()

    # 1. Non-0/1 for binary outcome
    res_bad_bin = runner.invoke(
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
            "--outcome",
            "outcome_cont",  # Continuous values, not 0/1
            "--outcome-type",
            "binary",
        ],
    )
    assert res_bad_bin.exit_code != 0
    assert "must contain only numeric 0 and 1 values" in str(res_bad_bin.output)

    # 2. Survival outcome without --time
    res_no_time = runner.invoke(
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
            "--outcome",
            "event",
            "--outcome-type",
            "survival",
        ],
    )
    assert res_no_time.exit_code != 0
    assert "--time is required when outcome-type is survival" in str(res_no_time.output)


def test_cli_iptw_json_output_serialization_strips_model(
    sample_iptw_csv: Path, tmp_path: Path
):
    """Regression test for Finding 3: JSON output does not crash on non-serializable 'model'."""
    runner = CliRunner()
    out_json = tmp_path / "iptw_out.json"

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
            "--outcome",
            "outcome_bin",
            "--outcome-type",
            "binary",
            "--output",
            str(out_json),
        ],
    )
    assert result.exit_code == 0, f"Error: {result.output}"
    assert out_json.exists()
    data = json.loads(out_json.read_text())
    assert "outcome_model" in data
    # Verify 'model' object is stripped in file output
    assert "model" not in data["outcome_model"]
    assert "odds_ratio" in data["outcome_model"]

    # 2. Stdout JSON branch (without --output)
    res_stdout = runner.invoke(
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
            "--outcome",
            "outcome_bin",
            "--outcome-type",
            "binary",
        ],
    )
    assert res_stdout.exit_code == 0, f"Error: {res_stdout.output}"
    stdout_data = json.loads(res_stdout.output)
    assert "outcome_model" in stdout_data
    # Verify 'model' object is stripped in stdout
    assert "model" not in stdout_data["outcome_model"]
    assert "odds_ratio" in stdout_data["outcome_model"]
    assert stdout_data["outcome_model"]["odds_ratio"] > 0


def test_cli_spec_yaml_iptw_binary_and_survival_scales(
    sample_iptw_csv: Path, tmp_path: Path
):
    """Regression test for Findings 4, 5, 6: verify OR and HR scales and non-zero estimates."""
    spec_content = f"""
version: "1.0"
metadata:
  study_title: "IPTW Binary & Survival Spec"
data:
  input_path: "{sample_iptw_csv}"
  id_column: "patient_id"
variables:
  - name: "treatment"
    role: "exposure"
  - name: "outcome_bin"
    role: "outcome"
  - name: "time"
    role: "time"
  - name: "event"
    role: "outcome"
  - name: "age"
    role: "covariate"
models:
  - name: "iptw_bin"
    type: "iptw"
    outcome: "outcome_bin"
    exposure: "treatment"
    covariates: ["age"]
    options:
      estimand: "ATE"
      outcome_type: "binary"
      stabilized: true
      truncate_quantiles: null
  - name: "iptw_surv"
    type: "iptw"
    outcome: "event"
    exposure: "treatment"
    time: "time"
    covariates: ["age"]
    options:
      estimand: "ATE"
      outcome_type: "survival"
reporting:
  style: "nejm"
  format: "html"
"""
    spec_path = tmp_path / "iptw_scales_sap.yaml"
    spec_path.write_text(spec_content)
    out_json = tmp_path / "scales_output.json"

    runner = CliRunner()
    result = runner.invoke(
        cli, ["model", "--spec", str(spec_path), "--output", str(out_json)]
    )
    assert result.exit_code == 0, f"Error: {result.output}"

    res = json.loads(out_json.read_text())
    assert "iptw_bin" in res["models"]
    assert "iptw_surv" in res["models"]

    # Explicit assertions verifying odds_ratio and hazard_ratio are positive
    assert res["models"]["iptw_bin"]["odds_ratio"] > 0
    assert res["models"]["iptw_surv"]["hazard_ratio"] > 0

    # If estimates list is present, verify OR and HR entries
    if "estimates" in res:
        scales = [e.get("scale") for e in res["estimates"]]
        assert "OR" in scales
        assert "HR" in scales

    # Verify narrative for iptw_bin mentions "Weights were not truncated."
    bin_narrative = res["reports"]["iptw_bin"]["methods_narrative"]
    assert "Weights were not truncated." in bin_narrative

    # Verify narrative for iptw_surv mentions truncation
    surv_narrative = res["reports"]["iptw_surv"]["methods_narrative"]
    assert "Weights were truncated at pre-specified quantiles" in surv_narrative


def test_cli_iptw_survival_invalid_duration_rejects(tmp_path: Path):
    """Verify that causal iptw with survival outcome rejects non-numeric or negative duration."""
    df_neg = pd.DataFrame(
        {
            "treatment": [0, 1, 0, 1, 0, 1],
            "age": [50, 60, 55, 65, 52, 58],
            "event": [0, 1, 0, 1, 0, 1],
            "time": [10.0, -5.0, 12.0, 8.0, 15.0, 9.0],  # negative duration
        }
    )
    csv_neg = tmp_path / "neg_dur.csv"
    df_neg.to_csv(csv_neg, index=False)

    runner = CliRunner()
    res_neg = runner.invoke(
        cli,
        [
            "causal",
            "iptw",
            "--data",
            str(csv_neg),
            "--treatment",
            "treatment",
            "--covariates",
            "age",
            "--outcome",
            "event",
            "--time",
            "time",
        ],
    )
    assert res_neg.exit_code != 0
    assert "must contain non-negative numeric values" in res_neg.output

    df_str = pd.DataFrame(
        {
            "treatment": [0, 1, 0, 1, 0, 1],
            "age": [50, 60, 55, 65, 52, 58],
            "event": [0, 1, 0, 1, 0, 1],
            "time": ["10", "bad_time", "12", "8", "15", "9"],  # non-numeric duration
        }
    )
    csv_str = tmp_path / "str_dur.csv"
    df_str.to_csv(csv_str, index=False)

    res_str = runner.invoke(
        cli,
        [
            "causal",
            "iptw",
            "--data",
            str(csv_str),
            "--treatment",
            "treatment",
            "--covariates",
            "age",
            "--outcome",
            "event",
            "--time",
            "time",
        ],
    )
    assert res_str.exit_code != 0
    assert "must contain non-negative numeric values" in res_str.output


def test_cli_spec_yaml_iptw_survival_invalid_event_rejects(tmp_path: Path):
    """Verify that YAML spec IPTW survival model rejects event columns with >2 levels or non-0/1 values."""
    df_multi = pd.DataFrame(
        {
            "treatment": [0, 1, 0, 1, 0, 1, 0, 1],
            "age": [50, 60, 55, 65, 52, 58, 62, 54],
            "event_code": [0, 1, 2, 1, 0, 2, 1, 0],  # 3 levels: 0, 1, 2
            "time": [10.0, 5.0, 12.0, 8.0, 15.0, 9.0, 11.0, 7.0],
        }
    )
    csv_multi = tmp_path / "multi_event.csv"
    df_multi.to_csv(csv_multi, index=False)

    spec_content = f"""
version: "1.0"
metadata:
  study_title: "Invalid Event Test"
  reporting_guideline: "STROBE"
data:
  input_path: "{csv_multi}"
variables:
  - name: "treatment"
    role: "exposure"
  - name: "event_code"
    role: "outcome"
  - name: "age"
    role: "covariate"
models:
  - name: "iptw_surv_bad"
    type: "iptw"
    outcome: "event_code"
    exposure: "treatment"
    time: "time"
    covariates: ["age"]
    options:
      outcome_type: "survival"
reporting:
  style: "nejm"
  format: "html"
"""
    spec_path = tmp_path / "invalid_event_sap.yaml"
    spec_path.write_text(spec_content)

    runner = CliRunner()
    result = runner.invoke(cli, ["model", "--spec", str(spec_path)])
    assert result.exit_code != 0
    assert "must contain only numeric 0 and 1 values" in (
        result.output or str(result.exception)
    )

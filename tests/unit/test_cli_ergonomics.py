import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from medstat.cli.main import cli, emit_agent_actionable_error
from medstat.cli.spec import (
    AnalysisPlan,
    get_cli_schema_template,
    preflight_validate_spec,
)
from medstat.data.loader import get_clinical_columns, load_clinical_data

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_cli_schema_template():
    template = get_cli_schema_template()
    assert "MedStat Declarative Analysis Plan Schema" in template
    assert "metadata:" in template
    assert "data:" in template
    assert "models:" in template


def test_cli_schema_flag():
    runner = CliRunner()
    result = runner.invoke(cli, ["--schema"])
    assert result.exit_code == 0
    assert "MedStat Declarative Analysis Plan Schema" in result.output
    assert "metadata:" in result.output
    assert "models:" in result.output


def test_preflight_validate_valid_spec(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    fixture_path = REPO_ROOT / "tests/fixtures/analysis_plan_example.yaml"
    assert fixture_path.exists()
    report = preflight_validate_spec(fixture_path)
    assert report["valid"] is True
    assert len(report["errors"]) == 0


def test_preflight_validate_nonexistent_spec():
    report = preflight_validate_spec("nonexistent_path_to_spec.yaml")
    assert report["valid"] is False
    assert any("not found" in err.lower() for err in report["errors"])
    assert len(report["agent_remediation"]) > 0


def test_preflight_validate_malformed_spec(tmp_path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("invalid_root: [1, 2, 3]\n", encoding="utf-8")
    report = preflight_validate_spec(bad_yaml)
    assert report["valid"] is False
    assert any("must contain at least one" in err.lower() for err in report["errors"])
    assert len(report["agent_remediation"]) > 0


def test_preflight_validate_missing_dataset(tmp_path):
    spec = tmp_path / "plan.yaml"
    spec.write_text(
        """
metadata:
  study_title: "Test"
data:
  input_path: "nonexistent_cohort.csv"
models:
  - name: "m1"
    type: "logistic"
    outcome: "y"
""",
        encoding="utf-8",
    )
    report = preflight_validate_spec(spec)
    assert report["valid"] is False
    assert any("dataset file not found" in err.lower() for err in report["errors"])
    assert any("data.input_path" in rem for rem in report["agent_remediation"])


def test_preflight_validate_missing_columns(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("patient_id,age,death\n1,60,0\n2,70,1\n", encoding="utf-8")

    spec = tmp_path / "plan.yaml"
    spec.write_text(
        f"""
metadata:
  study_title: "Test"
data:
  input_path: "{csv_file}"
models:
  - name: "m1"
    type: "logistic"
    outcome: "death"
    covariates: ["age", "missing_biomarker"]
""",
        encoding="utf-8",
    )
    report = preflight_validate_spec(spec)
    assert report["valid"] is False
    assert any("missing_biomarker" in err for err in report["errors"])
    assert any("missing_biomarker" in rem for rem in report["agent_remediation"])


def test_preflight_validate_analyses_and_data_path_invalid(tmp_path):
    spec = tmp_path / "legacy_plan.yaml"
    spec.write_text(
        """
project:
  title: "Legacy"
data:
  path: "some_data.csv"
analyses:
  - id: "m1"
    outcome: "death"
""",
        encoding="utf-8",
    )
    report = preflight_validate_spec(spec)
    assert report["valid"] is False
    assert any("input_path" in err.lower() for err in report["errors"])


def test_cli_validate_only_flag(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--validate-only",
            str(REPO_ROOT / "tests/fixtures/analysis_plan_example.yaml"),
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["valid"] is True


def test_cli_validate_only_flag_failure(tmp_path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("not_a_valid_spec: true\n", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(cli, ["--validate-only", str(bad_yaml)])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["valid"] is False
    assert len(payload["agent_remediation"]) > 0


def test_emit_agent_actionable_error(capsys):
    exc = ValueError("Perfect separation detected in logistic model")
    emit_agent_actionable_error(exc)
    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert payload["status"] == "error"
    assert payload["error_type"] == "ValueError"
    assert any("options.method: firth" in rem for rem in payload["agent_remediation"])


def test_preflight_validate_duplicate_keys(tmp_path):
    dup_yaml = tmp_path / "dup_plan.yaml"
    dup_yaml.write_text(
        """
metadata:
  study_title: "Test"
data:
  input_path: "test.csv"
models:
  - name: "m1"
    type: "logistic"
    outcome: "y"
models:
  - name: "m2"
    type: "logistic"
    outcome: "y"
""",
        encoding="utf-8",
    )
    report = preflight_validate_spec(dup_yaml)
    assert report["valid"] is False
    assert any("duplicate key" in err.lower() for err in report["errors"])


def test_preflight_validate_merge_key_override(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("patient_id,x1,x2,y\n1,0,1,0\n2,1,0,1\n", encoding="utf-8")
    merge_yaml = tmp_path / "merge_plan.yaml"
    merge_yaml.write_text(
        f"""
metadata:
  study_title: "Merge Test"
data:
  input_path: "{csv_file}"
.base_model: &base_model
  type: "logistic"
  outcome: "y"
  covariates: ["x1"]
models:
  - <<: *base_model
    name: "overridden_model"
    covariates: ["x1", "x2"]
""",
        encoding="utf-8",
    )
    report = preflight_validate_spec(merge_yaml)
    assert report["valid"] is True


def test_preflight_validate_unsupported_model_type(tmp_path):
    spec = tmp_path / "unsupported.yaml"
    spec.write_text(
        """
metadata:
  study_title: "Test"
data:
  input_path: "data.csv"
models:
  - name: "m1"
    type: "deep_learning"
    outcome: "y"
""",
        encoding="utf-8",
    )
    report = preflight_validate_spec(spec)
    assert report["valid"] is False
    assert any(
        "unsupported or missing model type" in err.lower() for err in report["errors"]
    )


def test_preflight_validate_non_mapping_model_item(tmp_path):
    spec = tmp_path / "non_map.yaml"
    spec.write_text(
        """
metadata:
  study_title: "Test"
data:
  input_path: "data.csv"
models:
  - "not_a_mapping"
""",
        encoding="utf-8",
    )
    report = preflight_validate_spec(spec)
    assert report["valid"] is False
    assert any(
        "must be a dictionary/mapping" in err.lower() for err in report["errors"]
    )


def test_preflight_validate_survival_missing_time(tmp_path):
    spec = tmp_path / "surv_no_time.yaml"
    spec.write_text(
        """
metadata:
  study_title: "Test"
data:
  input_path: "nonexistent.csv"
models:
  - name: "surv1"
    type: "cox"
    outcome: "death"
""",
        encoding="utf-8",
    )
    report = preflight_validate_spec(spec)
    assert report["valid"] is False
    assert any(
        "requires a 'time' or 'time_col' specification" in err.lower()
        for err in report["errors"]
    )


def test_preflight_validate_options_referenced_columns(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("patient_id,death\n1,0\n2,1\n", encoding="utf-8")
    spec = tmp_path / "opt_spec.yaml"
    spec.write_text(
        f"""
metadata:
  study_title: "Test"
data:
  input_path: "{csv_file}"
models:
  - name: "recal"
    type: "recalibration"
    outcome: "death"
    options:
      p_recal: "missing_score_column"
""",
        encoding="utf-8",
    )
    report = preflight_validate_spec(spec)
    assert report["valid"] is False
    assert any("missing_score_column" in err for err in report["errors"])


def test_cli_model_spec_failure_emits_json_stderr(tmp_path):
    bad_spec = tmp_path / "bad_plan.yaml"
    bad_spec.write_text(
        """
metadata:
  study_title: "Test"
data:
  input_path: "nonexistent_file.csv"
models:
  - name: "m1"
    type: "logistic"
    outcome: "y"
""",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["model", "--spec", str(bad_spec)])
    assert result.exit_code == 1
    assert result.stderr != ""
    payload = json.loads(result.stderr)
    assert payload["status"] == "error"
    assert len(payload["agent_remediation"]) > 0


def test_get_clinical_columns_formats_and_duplicates(tmp_path):
    # 1. Parquet
    df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
    parquet_path = tmp_path / "data.parquet"
    df.to_parquet(parquet_path)
    cols = get_clinical_columns(parquet_path)
    assert cols == ["col_a", "col_b"]

    # 1b. Parquet with named index: index must not appear in columns
    df_idx = pd.DataFrame(
        {"col_a": [1, 2], "col_b": [3, 4]},
        index=pd.Index(["p1", "p2"], name="patient_id"),
    )
    parquet_idx_path = tmp_path / "data_idx.parquet"
    df_idx.to_parquet(parquet_idx_path)
    cols_idx = get_clinical_columns(parquet_idx_path)
    assert cols_idx == ["col_a", "col_b"]
    loaded_df = load_clinical_data(parquet_idx_path)
    assert list(loaded_df.columns) == cols_idx

    # 1c. Parquet with named RangeIndex: range index entry must not filter columns
    df_range = pd.DataFrame(
        {"col_a": [1, 2], "col_b": [3, 4]},
        index=pd.RangeIndex(0, 2, name="col_a"),
    )
    parquet_range_path = tmp_path / "data_range.parquet"
    df_range.to_parquet(parquet_range_path)
    cols_range = get_clinical_columns(parquet_range_path)
    assert cols_range == ["col_a", "col_b"]

    # 2. Excel .xlsx
    xlsx_path = tmp_path / "data.xlsx"
    df.to_excel(xlsx_path, index=False)
    cols_xlsx = get_clinical_columns(xlsx_path)
    assert cols_xlsx == ["col_a", "col_b"]

    # 3. Legacy Excel .xls raises ClickException
    xls_path = tmp_path / "legacy.xls"
    xls_path.touch()
    with pytest.raises(click.ClickException, match="Legacy Excel format"):
        get_clinical_columns(xls_path)

    # 4. Headers duplicated after stripping whitespace
    dup_csv = tmp_path / "dup.csv"
    dup_csv.write_text("age , age,gender\n1,2,M\n", encoding="utf-8")
    with pytest.raises(
        click.ClickException, match="duplicate column names after normalization"
    ):
        get_clinical_columns(dup_csv)


def test_preflight_validate_non_string_input_path(tmp_path):
    for bad_input in [123, True, ["some/path.csv"]]:
        spec_file = tmp_path / "spec_bad_path.yaml"
        spec_content = f"""
metadata:
  study_title: "Non-string input_path test"
data:
  input_path: {bad_input}
models:
  - name: "m1"
    type: "logistic"
    outcome: "y"
"""
        spec_file.write_text(spec_content, encoding="utf-8")
        report = preflight_validate_spec(spec_file)
        assert report["valid"] is False
        assert any("data" in err and "input_path" in err for err in report["errors"])


def test_analysis_plan_from_yaml_time_col_precedence(tmp_path):
    # 1. time_col populates ModelSpec.time when time is unset
    spec_time_col = tmp_path / "spec_time_col.yaml"
    spec_time_col.write_text(
        """
metadata:
  study_title: "Time Col Test"
data:
  input_path: "dummy.csv"
models:
  - name: "cox_model"
    type: "cox"
    time_col: "follow_up_days"
    outcome: "dead"
""",
        encoding="utf-8",
    )
    plan1 = AnalysisPlan.from_yaml(spec_time_col)
    assert plan1.models[0].time == "follow_up_days"

    # 2. time takes precedence over time_col and options.time_col when specified
    spec_precedence = tmp_path / "spec_precedence.yaml"
    spec_precedence.write_text(
        """
metadata:
  study_title: "Precedence Test"
data:
  input_path: "dummy.csv"
models:
  - name: "cox_model"
    type: "cox"
    time: "primary_time"
    time_col: "secondary_time"
    options:
      time_col: "tertiary_time"
    outcome: "dead"
  - name: "cox_from_options"
    type: "cox"
    outcome: "dead"
    options:
      time_col: "options_time"
  - name: "logistic_ignore_options"
    type: "logistic"
    outcome: "dead"
    options:
      time_col: "ignored_time"
  - name: "logistic_ignore_top_time_col"
    type: "logistic"
    time_col: "ignored_top_time"
    outcome: "dead"
  - name: "iptw_survival_top_time_col"
    type: "iptw"
    time_col: "iptw_surv_time"
    outcome: "dead"
  - name: "iptw_survival_options_time_col"
    type: "causal_iptw"
    options:
      time_col: "iptw_opt_time"
    outcome: "dead"
  - name: "td_roc_top_time_col"
    type: "time_dependent_roc"
    time_col: "td_time"
    outcome: "dead"
  - name: "iptw_binary_no_time"
    type: "iptw"
    outcome: "dead"
  - name: "iptw_binary_explicit_outcome_type"
    type: "iptw"
    time_col: "iptw_bin_time"
    outcome: "dead"
    options:
      outcome_type: "binary"
  - name: "iptw_continuous_explicit_outcome_type"
    type: "causal_iptw"
    outcome: "dead"
    options:
      outcome_type: "continuous"
      time_col: "iptw_cont_time"
  - name: "iptw_survival_explicit_outcome_type"
    type: "iptw"
    time_col: "iptw_surv_time_explicit"
    outcome: "dead"
    options:
      outcome_type: "survival"
  - name: "logistic_method_iptw_time_fallback"
    type: "logistic"
    time_col: "log_iptw_time"
    outcome: "dead"
    options:
      method: "iptw"
  - name: "logistic_method_iptw_binary_no_time"
    type: "logistic"
    time_col: "log_iptw_bin_time"
    outcome: "dead"
    options:
      method: "iptw"
      outcome_type: "binary"
  - name: "diagnostic_td_roc_top_time_col"
    type: "diagnostic"
    time_col: "diag_td_time"
    outcome: "dead"
    options:
      time_dependent_roc: true
  - name: "diagnostic_ordinary_no_time"
    type: "diagnostic"
    time_col: "diag_ignored_time"
    outcome: "dead"
  - name: "diagnostic_reclass_with_td_roc_and_time_col"
    type: "diagnostic"
    time_col: "diag_reclass_time"
    outcome: "dead"
    options:
      p_old_col: "pred_old"
      p_new_col: "pred_new"
      time_dependent_roc: true
  - name: "td_roc_with_p_old_col"
    type: "time_dependent_roc"
    time_col: "td_roc_time"
    outcome: "dead"
    options:
      p_old_col: "pred_old"
  - name: "diagnostic_iptw_survival_with_p_old_col"
    type: "diagnostic"
    time_col: "diag_iptw_time"
    outcome: "dead"
    options:
      method: "iptw"
      p_old_col: "pred_old"
      outcome_type: "survival"
  - name: "diagnostic_iptw_binary_with_p_old_col"
    type: "diagnostic"
    time_col: "diag_iptw_bin_time"
    outcome: "dead"
    options:
      method: "iptw"
      p_old_col: "pred_old"
      outcome_type: "binary"
""",
        encoding="utf-8",
    )
    plan2 = AnalysisPlan.from_yaml(spec_precedence)
    assert plan2.models[0].time == "primary_time"
    assert plan2.models[1].time == "options_time"
    assert plan2.models[2].time is None
    assert plan2.models[3].time is None
    assert plan2.models[4].time == "iptw_surv_time"
    assert plan2.models[5].time == "iptw_opt_time"
    assert plan2.models[6].time == "td_time"
    assert plan2.models[7].time is None
    assert plan2.models[8].time is None  # iptw binary explicit
    assert plan2.models[9].time is None  # iptw continuous explicit
    assert plan2.models[10].time == "iptw_surv_time_explicit"
    assert plan2.models[11].time == "log_iptw_time"  # options.method: iptw inherits
    assert plan2.models[12].time is None  # options.method: iptw + binary outcome
    assert plan2.models[13].time == "diag_td_time"  # diagnostic with time_dependent_roc
    assert plan2.models[14].time is None  # ordinary diagnostic without td roc
    assert plan2.models[15].time is None  # diagnostic reclassification ignores duration
    assert (
        plan2.models[16].time == "td_roc_time"
    )  # td_roc model retains duration even with p_old_col
    assert (
        plan2.models[17].time == "diag_iptw_time"
    )  # diagnostic with options.method: iptw + survival outcome inherits time
    assert (
        plan2.models[18].time is None
    )  # diagnostic with options.method: iptw + binary outcome ignores time


def test_execute_required_cols_duration_dedup(tmp_path, monkeypatch):
    """AnalysisPlan.execute includes resolved duration from model.time and omits overridden options.time_col."""
    import medstat.cli.spec as spec_mod

    captured_req_cols = []
    orig_prepare = spec_mod.prepare_data_for_analysis

    def mock_prepare(*args, **kwargs):
        captured_req_cols.append(kwargs.get("required_cols"))
        return orig_prepare(*args, **kwargs)

    monkeypatch.setattr(spec_mod, "prepare_data_for_analysis", mock_prepare)

    csv_path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "patient_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "dead": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "age": [50, 60, 70, 80, 90, 55, 65, 75, 85, 95],
            "time_top": [10, 20, 30, 40, 50, 15, 25, 35, 45, 55],
            "time_stale": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    ).to_csv(csv_path, index=False)

    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.write_text(
        f"""
metadata:
  study_title: "Duration Col Dedup"
data:
  input_path: "{csv_path}"
models:
  - name: "cox_model"
    type: "cox"
    time: "time_top"
    outcome: "dead"
    covariates: ["age"]
    options:
      time_col: "time_stale"
""",
        encoding="utf-8",
    )

    plan = AnalysisPlan.from_yaml(plan_yaml)
    plan.execute()
    assert len(captured_req_cols) == 1
    assert "time_top" in captured_req_cols[0]
    assert "time_stale" not in captured_req_cols[0]


def test_diagnostic_reclassification_does_not_require_duration(tmp_path, monkeypatch):
    """Diagnostic reclassification models using p_old_col omit duration columns during cohort preparation."""
    import medstat.cli.spec as spec_mod

    captured_req_cols = []
    orig_prepare = spec_mod.prepare_data_for_analysis

    def mock_prepare(*args, **kwargs):
        captured_req_cols.append(kwargs.get("required_cols"))
        return orig_prepare(*args, **kwargs)

    monkeypatch.setattr(spec_mod, "prepare_data_for_analysis", mock_prepare)

    csv_path = tmp_path / "reclass_data.csv"
    pd.DataFrame(
        {
            "patient_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "dead": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "p_old": [0.2, 0.4, 0.3, 0.7, 0.1, 0.8, 0.2, 0.6, 0.3, 0.9],
            "p_new": [0.1, 0.5, 0.2, 0.8, 0.2, 0.9, 0.1, 0.7, 0.4, 0.95],
            "time_duration": [
                10.0,
                20.0,
                np.nan,
                40.0,
                np.nan,
                15.0,
                25.0,
                np.nan,
                45.0,
                55.0,
            ],
        }
    ).to_csv(csv_path, index=False)

    plan_yaml = tmp_path / "reclass_plan.yaml"
    plan_yaml.write_text(
        f"""
metadata:
  study_title: "Reclassification Spec Test"
data:
  input_path: "{csv_path}"
models:
  - name: "reclass_model"
    type: "diagnostic"
    time_col: "time_duration"
    outcome: "dead"
    options:
      p_old_col: "p_old"
      p_new_col: "p_new"
      time_dependent_roc: true
""",
        encoding="utf-8",
    )

    plan = AnalysisPlan.from_yaml(plan_yaml)
    assert plan.models[0].time is None
    res = plan.execute()
    assert len(captured_req_cols) == 1
    assert "time_duration" not in captured_req_cols[0]
    assert "p_old" in captured_req_cols[0]
    assert "p_new" in captured_req_cols[0]
    assert "reclass_model" in res["models"]
    assert "reclassification" in res["models"]["reclass_model"]
    assert res["model_flow"]["reclass_model"]["final_n"] == 10


def test_diagnostic_method_iptw_with_p_old_col_executes_iptw(tmp_path, monkeypatch):
    """When options.method selects IPTW, diagnostic type is not treated as reclassification and executes IPTW."""
    import medstat.cli.spec as spec_mod

    captured_req_cols = []
    orig_prepare = spec_mod.prepare_data_for_analysis

    def mock_prepare(*args, **kwargs):
        captured_req_cols.append(kwargs.get("required_cols"))
        return orig_prepare(*args, **kwargs)

    monkeypatch.setattr(spec_mod, "prepare_data_for_analysis", mock_prepare)

    csv_path = tmp_path / "iptw_diag_data.csv"
    pd.DataFrame(
        {
            "patient_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "dead": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "tx": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            "age": [50, 60, 55, 65, 70, 45, 52, 63, 68, 72],
            "p_old": [0.2, 0.4, 0.3, 0.7, 0.1, 0.8, 0.2, 0.6, 0.3, 0.9],
            "time_duration": [
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
                15.0,
                25.0,
                35.0,
                45.0,
                55.0,
            ],
        }
    ).to_csv(csv_path, index=False)

    plan_yaml = tmp_path / "iptw_diag_plan.yaml"
    plan_yaml.write_text(
        f"""
metadata:
  study_title: "IPTW Diagnostic Spec Test"
data:
  input_path: "{csv_path}"
models:
  - name: "iptw_model"
    type: "diagnostic"
    time_col: "time_duration"
    outcome: "dead"
    exposure: "tx"
    covariates: ["age"]
    options:
      method: "iptw"
      p_old_col: "p_old"
      outcome_type: "survival"
""",
        encoding="utf-8",
    )

    plan = AnalysisPlan.from_yaml(plan_yaml)
    assert plan.models[0].time == "time_duration"
    res = plan.execute()
    assert len(captured_req_cols) == 1
    assert "time_duration" in captured_req_cols[0]
    assert "tx" in captured_req_cols[0]
    assert "iptw_model" in res["models"]
    assert "balance_table" in res["models"]["iptw_model"]


def test_actionable_error_ph_and_competing_risk_remediation(capsys):
    # Cox error about competing events matches competing risk remediation
    exc_competing = ValueError("Competing events detected in Cox survival analysis")
    emit_agent_actionable_error(exc_competing)
    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert any("competing_risk" in rem for rem in payload["agent_remediation"])

    # Cox error about too few observations must NOT suggest competing risk
    exc_obs = ValueError("Cox PH requires at least 5 observations.")
    emit_agent_actionable_error(exc_obs)
    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert not any("competing_risk" in rem for rem in payload["agent_remediation"])

    # Competing risks error about too few observations must NOT suggest competing risk
    exc_comp_obs = ValueError("Too few observations in competing risks cohort")
    emit_agent_actionable_error(exc_comp_obs)
    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert not any("competing_risk" in rem for rem in payload["agent_remediation"])

    # Proportional hazards error directs agents to halt unless using standalone script
    exc_ph = ValueError("Proportional hazard assumption violated")
    emit_agent_actionable_error(exc_ph)
    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert any(
        "Declarative plan cannot apply time-varying covariates or strata; halt" in rem
        for rem in payload["agent_remediation"]
    )

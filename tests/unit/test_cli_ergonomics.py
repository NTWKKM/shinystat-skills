import json
from pathlib import Path

from click.testing import CliRunner

from medstat.cli.main import cli, emit_agent_actionable_error
from medstat.cli.spec import get_cli_schema_template, preflight_validate_spec


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


def test_preflight_validate_valid_spec():
    fixture_path = Path("tests/fixtures/analysis_plan_example.yaml")
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
project:
  title: "Test"
data:
  path: "nonexistent_cohort.csv"
analyses:
  - id: "m1"
    outcome: "y"
""",
        encoding="utf-8",
    )
    report = preflight_validate_spec(spec)
    assert report["valid"] is False
    assert any("dataset file not found" in err.lower() for err in report["errors"])
    assert any("data.path" in rem for rem in report["agent_remediation"])


def test_preflight_validate_missing_columns(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("patient_id,age,death\n1,60,0\n2,70,1\n", encoding="utf-8")

    spec = tmp_path / "plan.yaml"
    spec.write_text(
        f"""
project:
  title: "Test"
data:
  path: "{csv_file}"
analyses:
  - id: "m1"
    outcome: "death"
    covariates: ["age", "missing_biomarker"]
""",
        encoding="utf-8",
    )
    report = preflight_validate_spec(spec)
    assert report["valid"] is False
    assert any("missing_biomarker" in err for err in report["errors"])
    assert any("missing_biomarker" in rem for rem in report["agent_remediation"])


def test_cli_validate_only_flag():
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--validate-only", "tests/fixtures/analysis_plan_example.yaml"]
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
    assert any("firth" in rem.lower() for rem in payload["agent_remediation"])


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

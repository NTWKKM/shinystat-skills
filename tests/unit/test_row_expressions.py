"""Row expressions from a plan or a quality rule are data, not code (ADR 83)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from medstat.cli.spec import AnalysisPlan, preflight_validate_spec
from medstat.data.expressions import (
    eval_rows,
    query_rows,
    validate_row_expression,
)
from medstat.data.quality import (
    CrossVariableRule,
    DataQualityReport,
    DataQualitySchema,
)


def _cohort() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [30, 50, 70, 65],
            "sbp": [85.0, 120.0, 140.0, np.nan],
            "sex": ["F", "M", "F", "M"],
            "stage": ["I", "III", "IV", "II"],
            "BMI (kg/m2)": [22.0, 31.0, 27.5, 35.0],
        }
    )


@pytest.mark.parametrize(
    "expression",
    [
        "age >= 45",
        "sbp > 90 and age < 80",
        "(age >= 45) & (sbp > 90)",
        "not (sex == 'F')",
        "~(age > 60) | (sbp <= 100)",
        "stage in ['III', 'IV']",
        "stage not in ('I', 'II')",
        "18 <= age <= 70",
        "sbp / age > 1.5",
        "`BMI (kg/m2)` > 30",
        "age.between(40, 70)",
        "stage.isin(['III', 'IV'])",
        "sbp.notna()",
        "sex.str.lower() == 'f'",
        "stage.str.startswith('I')",
        "sex.str.contains('F', case=False, regex=False)",
    ],
)
def test_predicates_pass_and_match_plain_query(expression):
    df = _cohort()
    assert validate_row_expression(expression) == expression
    pd.testing.assert_frame_equal(query_rows(df, expression), df.query(expression))
    pd.testing.assert_series_equal(eval_rows(df, expression), df.eval(expression))


@pytest.mark.parametrize(
    ("expression", "reason"),
    [
        ("age > @threshold", "@"),
        ("age > @pd.io.common.os.getpid()", "@"),
        ("age.__class__.__name__ == 'Series'", "attribute"),
        ("__import__('os') == age", "call"),
        ("__builtins__ == age", "name"),
        ("age @ sbp > 1", "@"),
        ("age.to_csv('leak.csv') == 1", "attribute"),
        ("age.to_pickle('leak.pkl') == 1", "attribute"),
        ("age.map(print) > 1", "attribute"),
        ("age.apply(len) > 1", "attribute"),
        ("abs(age) > 1", "call"),
        ("(lambda: 1)() < age", "call"),
        ("(lambda: age) == age", "Lambda"),
        ("[a for a in age] == age", "ListComp"),
        ("age[0] > 1", "Subscript"),
        ("age if sbp else age", "IfExp"),
        ("total = age + sbp", "valid"),
        ("age > 1; age < 9", "valid"),
        ("age ==", "valid"),
        ("sex.str.contains('(a+)+$')", "literal"),
        ("sex.str.contains('F', regex=True)", "literal"),
        ("sex.str.contains('F', False, 0, None, True)", "literal"),
        ("", "empty"),
        ("   ", "empty"),
    ],
)
def test_non_predicates_are_rejected_with_a_reason(expression, reason):
    with pytest.raises(ValueError, match=reason):
        validate_row_expression(expression)
    with pytest.raises(ValueError, match=reason):
        query_rows(_cohort(), expression)
    with pytest.raises(ValueError, match=reason):
        eval_rows(_cohort(), expression)


@pytest.mark.parametrize("expression", ["age", "age + 1", "age / age", "1 > 0"])
def test_query_rows_requires_a_boolean_mask(expression):
    """A numeric result is never read as row labels (it would select rows by index)."""
    with pytest.raises(ValueError, match="boolean"):
        query_rows(_cohort(), expression)


def test_non_string_expression_raises():
    with pytest.raises(ValueError, match="must be a string"):
        validate_row_expression(["age > 1"])  # type: ignore[arg-type]


def _plan_with_filter(tmp_path: Path, expression: str) -> Path:
    csv = tmp_path / "cohort.csv"
    pd.DataFrame(
        {"patient_id": ["PATIENT_MOCK_001", "PATIENT_MOCK_002"], "age": [40, 60]}
    ).to_csv(csv, index=False)
    spec = tmp_path / "plan.yaml"
    spec.write_text(
        "metadata:\n"
        '  study_title: "Filter expression test"\n'
        "data:\n"
        f'  input_path: "{csv}"\n'
        "  filters:\n"
        f"    - expression: '{expression}'\n"
        '      reason: "test"\n'
        "models:\n"
        '  - name: "m1"\n'
        '    type: "logistic"\n'
        '    outcome: "age"\n',
        encoding="utf-8",
    )
    return spec


def test_plan_filter_cannot_run_code(tmp_path):
    """A filter that reaches a module through '@' is rejected before pandas sees it."""
    marker = tmp_path / "created_by_filter"
    spec = _plan_with_filter(tmp_path, f'age > @pd.io.common.os.mkdir("{marker}")')
    with pytest.raises(ValueError, match="Failed to apply cohort filter"):
        AnalysisPlan.from_yaml(spec).execute()
    assert not marker.exists()


def test_preflight_reports_an_unsafe_filter_with_remediation(tmp_path):
    spec = _plan_with_filter(tmp_path, "age.to_csv(1) == 1")
    report = preflight_validate_spec(spec)
    assert report["valid"] is False
    assert any("data.filters[0]" in e for e in report["errors"])
    assert any("comparison" in h for h in report["agent_remediation"])

    ok = preflight_validate_spec(_plan_with_filter(tmp_path, "age >= 45"))
    assert not any("data.filters" in e for e in ok["errors"])


def test_preflight_rejects_a_bare_string_filter(tmp_path):
    spec = _plan_with_filter(tmp_path, "age >= 45")
    spec.write_text(
        spec.read_text(encoding="utf-8").replace(
            "    - expression: 'age >= 45'\n      reason: \"test\"\n",
            "    - 'age >= 45'\n",
        ),
        encoding="utf-8",
    )
    report = preflight_validate_spec(spec)
    assert report["valid"] is False
    assert any("data.filters[0]" in e and "mapping" in e for e in report["errors"])


def test_quality_rule_string_is_validated_and_recorded_not_evaluated(tmp_path):
    marker = tmp_path / "created_by_rule"
    rules = [
        CrossVariableRule(
            name="unsafe",
            condition=f'age > @pd.io.common.os.mkdir("{marker}")',
            description="reaches a module",
        ),
        CrossVariableRule(
            name="bp", condition="sbp > age", description="plain comparison"
        ),
    ]
    report = DataQualityReport(
        _cohort(), schema=DataQualitySchema(cross_variable_rules=rules)
    )
    score = report.consistency_score()
    assert not marker.exists()
    assert [e["rule"] for e in report.rule_errors] == ["unsafe"]
    assert "@" in report.rule_errors[0]["error"]
    # An unevaluated rule leaves consistency undefined, never a passing score.
    assert np.isnan(score)
    summary = report.generate_report()
    assert np.isnan(summary["overall_score"])
    assert summary["grade"] == "Unavailable"
    assert any("unsafe" in r for r in summary["recommendations"])


def test_unsafe_rule_is_recorded_when_its_columns_are_all_null():
    df = pd.DataFrame({"age": [np.nan, np.nan], "sbp": [1.0, 2.0]})
    rule = CrossVariableRule(
        name="unsafe", condition="age > @threshold", description="reference"
    )
    report = DataQualityReport(
        df, schema=DataQualitySchema(cross_variable_rules=[rule])
    )
    assert np.isnan(report.consistency_score())
    assert [e["rule"] for e in report.rule_errors] == ["unsafe"]


def test_permitted_rule_violations_lower_the_consistency_score():
    """`sbp < age` fails on all three rows with a systolic value: 100 - 25 (warning) = 75."""
    rule = CrossVariableRule(name="bp", condition="sbp < age", description="violated")
    report = DataQualityReport(
        _cohort(), schema=DataQualitySchema(cross_variable_rules=[rule])
    )
    assert report.consistency_score() == pytest.approx(75.0)
    assert report.rule_errors == []

"""
tests/fixtures/integrity_cases.py: Shared parity test cases and mutations for reporting integrity.

Used by both tests/unit/test_reporting_integrity.py (ReportDocument IR) and
tests/unit/test_shinystat_cloud.py (standalone pure-Python implementation extracted from report-builder.md).
"""

from dataclasses import dataclass
from typing import Any

BASE_RESULTS_DICT: dict[str, Any] = {
    "or_age": 1.45,
    "ci_low": 1.10,
    "ci_high": 1.91,
    "p": 0.012,
    "n": 412,
}

BASE_NARRATIVE = "Age was associated with the outcome (OR 1.45, 95% CI 1.10 to 1.91, P=0.012; n=412)."


@dataclass(frozen=True)
class IntegrityParityCase:
    id: str
    narrative: str
    results_dict: dict[str, Any]
    expected_passed: bool
    expected_untraced: list[float]
    allowed_untraced: list[float] | None = None


INTEGRITY_PARITY_CASES: list[IntegrityParityCase] = [
    # --- MUST PASS CASES ---
    IntegrityParityCase(
        id="pass_base",
        narrative=BASE_NARRATIVE,
        results_dict=BASE_RESULTS_DICT,
        expected_passed=True,
        expected_untraced=[],
    ),
    IntegrityParityCase(
        id="pass_alpha_0_05",
        narrative=f"Alpha was 0.05. {BASE_NARRATIVE}",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=True,
        expected_untraced=[],
    ),
    IntegrityParityCase(
        id="pass_p_rounding_0_01",
        narrative="Age was associated with the outcome (OR 1.45, 95% CI 1.10 to 1.91, P=0.01; n=412).",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=True,
        expected_untraced=[],
    ),
    IntegrityParityCase(
        id="pass_mort_pct_and_dec",
        narrative="Mortality was 12.5% (observed 0.125). Age was associated with the outcome (OR 1.45, 95% CI 1.10 to 1.91, P=0.012; n=412).",
        results_dict={**BASE_RESULTS_DICT, "mort": 0.125},
        expected_passed=True,
        expected_untraced=[],
    ),
    IntegrityParityCase(
        id="pass_allowed_untraced_73_4",
        narrative="Age was associated with the outcome (OR 1.45, 95% CI 1.10 to 1.91, P=0.012; n=412), marker 73.4.",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=True,
        expected_untraced=[],
        allowed_untraced=[73.4],
    ),
    IntegrityParityCase(
        id="pass_p_lt_5_pct",
        narrative="Age was associated with the outcome (OR 1.45, 95% CI 1.10 to 1.91, P<5%; n=412).",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=True,
        expected_untraced=[],
    ),
    IntegrityParityCase(
        id="pass_p_lt_0_05",
        narrative="Age was associated with the outcome (OR 1.45, 95% CI 1.10 to 1.91, P<0.05; n=412).",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=True,
        expected_untraced=[],
    ),
    IntegrityParityCase(
        id="pass_p_le_unicode_0_05",
        narrative="Age was associated with the outcome (OR 1.45, 95% CI 1.10 to 1.91, P ≤ 0.05; n=412).",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=True,
        expected_untraced=[],
    ),
    IntegrityParityCase(
        id="pass_p_ge_unicode_0_01",
        narrative="Age was associated with the outcome (OR 1.45, 95% CI 1.10 to 1.91, P ≥ 0.01; n=412).",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=True,
        expected_untraced=[],
    ),
    IntegrityParityCase(
        id="pass_p_value_whitespace_lt_0_05",
        narrative="Age was associated with the outcome (OR 1.45, 95% CI 1.10 to 1.91, P value < 0.05; n=412).",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=True,
        expected_untraced=[],
    ),
    # --- MUST FAIL CASES ---
    IntegrityParityCase(
        id="fail_or_3_45",
        narrative="Age was associated with the outcome (OR 3.45, 95% CI 1.10 to 1.91, P=0.012; n=412).",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=False,
        expected_untraced=[3.45],
    ),
    IntegrityParityCase(
        id="fail_or_1_46",
        narrative="Age was associated with the outcome (OR 1.46, 95% CI 1.10 to 1.91, P=0.012; n=412).",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=False,
        expected_untraced=[1.46],
    ),
    IntegrityParityCase(
        id="fail_or_1_47",
        narrative="Age was associated with the outcome (OR 1.47, 95% CI 1.10 to 1.91, P=0.012; n=412).",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=False,
        expected_untraced=[1.47],
    ),
    IntegrityParityCase(
        id="fail_ci_low_1_15",
        narrative="Age was associated with the outcome (OR 1.45, 95% CI 1.15 to 1.91, P=0.012; n=412).",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=False,
        expected_untraced=[1.15],
    ),
    IntegrityParityCase(
        id="fail_ci_low_1_20",
        narrative="Age was associated with the outcome (OR 1.45, 95% CI 1.20 to 1.91, P=0.012; n=412).",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=False,
        expected_untraced=[1.20],
    ),
    IntegrityParityCase(
        id="fail_p_0_034",
        narrative="Age was associated with the outcome (OR 1.45, 95% CI 1.10 to 1.91, P=0.034; n=412).",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=False,
        expected_untraced=[0.034],
    ),
    IntegrityParityCase(
        id="fail_p_0_02",
        narrative="Age was associated with the outcome (OR 1.45, 95% CI 1.10 to 1.91, P=0.02; n=412).",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=False,
        expected_untraced=[0.02],
    ),
    IntegrityParityCase(
        id="fail_p_0_013",
        narrative="Age was associated with the outcome (OR 1.45, 95% CI 1.10 to 1.91, P=0.013; n=412).",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=False,
        expected_untraced=[0.013],
    ),
    IntegrityParityCase(
        id="fail_p_0_001",
        narrative="Age was associated with the outcome (OR 1.45, 95% CI 1.10 to 1.91, P=0.001; n=412).",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=False,
        expected_untraced=[0.001],
    ),
    IntegrityParityCase(
        id="fail_p_lt_0_001",
        narrative="Age was associated with the outcome (OR 1.45, 95% CI 1.10 to 1.91, P<0.001; n=412).",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=False,
        expected_untraced=[0.001],
    ),
    IntegrityParityCase(
        id="fail_p_le_unicode_0_01",
        narrative="Age was associated with the outcome (OR 1.45, 95% CI 1.10 to 1.91, P ≤ 0.01; n=412).",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=False,
        expected_untraced=[0.01],
    ),
    IntegrityParityCase(
        id="fail_p_lt_0_05_pct",
        narrative="Age was associated with the outcome (OR 1.45, 95% CI 1.10 to 1.91, P<0.05%; n=412).",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=False,
        expected_untraced=[0.05],
    ),
    IntegrityParityCase(
        id="fail_p_0_05",
        narrative="Age was associated with the outcome (OR 1.45, 95% CI 1.10 to 1.91, P=0.05; n=412).",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=False,
        expected_untraced=[0.05],
    ),
    IntegrityParityCase(
        id="fail_n_415",
        narrative="Age was associated with the outcome (OR 1.45, 95% CI 1.10 to 1.91, P=0.012; n=415).",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=False,
        expected_untraced=[415.0],
    ),
    IntegrityParityCase(
        id="fail_n_408",
        narrative="Age was associated with the outcome (OR 1.45, 95% CI 1.10 to 1.91, P=0.012; n=408).",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=False,
        expected_untraced=[408.0],
    ),
    IntegrityParityCase(
        id="fail_n_512",
        narrative="Age was associated with the outcome (OR 1.45, 95% CI 1.10 to 1.91, P=0.012; n=512).",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=False,
        expected_untraced=[512.0],
    ),
    IntegrityParityCase(
        id="fail_mort_without_result",
        narrative="Mortality was 12.5%. Age was associated with the outcome (OR 1.45, 95% CI 1.10 to 1.91, P=0.012; n=412).",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=False,
        expected_untraced=[12.5],
    ),
    IntegrityParityCase(
        id="fail_bare_year_1950",
        narrative="Age was associated with the outcome (OR 1.45, 95% CI 1.10 to 1.91, P=0.012; n=412) (1950).",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=False,
        expected_untraced=[1950.0],
    ),
    IntegrityParityCase(
        id="fail_bracketed_250",
        narrative="Age was associated with the outcome (OR 1.45, 95% CI 1.10 to 1.91, P=0.012; n=412) [250].",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=False,
        expected_untraced=[250.0],
    ),
    IntegrityParityCase(
        id="fail_412_of_500",
        narrative="Age was associated with the outcome in 412 of 500 patients (OR 1.45, 95% CI 1.10 to 1.91, P=0.012).",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=False,
        expected_untraced=[500.0],
    ),
    IntegrityParityCase(
        id="fail_allowed_untraced_73_9",
        narrative="Age was associated with the outcome (OR 1.45, 95% CI 1.10 to 1.91, P=0.012; n=412), marker 73.9.",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=False,
        expected_untraced=[73.9],
        allowed_untraced=[73.4],
    ),
]

# Task 2b: Statistical token mutations (+-1 last decimal digit, *10, *100, /100, and n+-3)
# Format: (token_str, list of mutated strings)
STAT_MUTATIONS: list[tuple[str, list[str]]] = [
    # 1. OR 1.45 (d=2, unit=0.01)
    ("1.45", ["1.46", "1.44", "14.5", "145", "0.0145"]),
    # 2. CI low 1.10 (d=2, unit=0.01)
    ("1.10", ["1.11", "1.09", "11.0", "110", "0.011"]),
    # 3. CI high 1.91 (d=2, unit=0.01)
    ("1.91", ["1.92", "1.90", "19.1", "191", "0.0191"]),
    # 4. P 0.012 (d=3, unit=0.001)
    ("0.012", ["0.013", "0.011", "0.12", "1.2", "0.00012"]),
    # 5. n 412 (d=0, unit=1, plus n+-3)
    ("412", ["413", "411", "415", "409", "4120", "41200", "4.12"]),
]

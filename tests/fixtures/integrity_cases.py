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
    # --- ADR 79: author-year citations and section numbers are structural ---
    IntegrityParityCase(
        id="pass_author_year_citations",
        narrative=f"{BASE_NARRATIVE} Matching followed Austin (2011), VanderWeele & Ding 2017 and Kerr and Pepe (2014a); see also (Rubin, 1987).",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=True,
        expected_untraced=[],
    ),
    IntegrityParityCase(
        id="fail_capitalised_count_is_not_a_citation",
        narrative=f"Total 1987 patients were screened. {BASE_NARRATIVE}",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=False,
        expected_untraced=[1987.0],
    ),
    IntegrityParityCase(
        id="fail_sentence_final_capitalised_count_is_not_a_citation",
        narrative=f"{BASE_NARRATIVE} Patients screened: Total 1987.",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=False,
        expected_untraced=[1987.0],
    ),
    IntegrityParityCase(
        id="pass_sentence_final_citation_with_author_context",
        narrative=f"{BASE_NARRATIVE} Matching followed VanderWeele & Ding 2017. Pooling followed Rubin, 1987.",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=True,
        expected_untraced=[],
    ),
    IntegrityParityCase(
        id="fail_line_initial_percentage_is_not_a_section_number",
        narrative=f"{BASE_NARRATIVE}\n45%. Mortality was lower after matching.",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=False,
        expected_untraced=[45.0],
    ),
    IntegrityParityCase(
        id="pass_section_numbers",
        narrative=f"## 4. Results\n{BASE_NARRATIVE}\n7. Discussion\n### 6.1 Sensitivity Analysis\nNo further estimates are reported.",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=True,
        expected_untraced=[],
    ),
    # --- ADR 81: analysis scale constants ---
    IntegrityParityCase(
        id="pass_per_unit_and_caliper_constants",
        narrative=f"Matching used a caliper of 0.2 SD; age was scaled per 10 years and creatinine per 0.5 mg/dL. {BASE_NARRATIVE}",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=True,
        expected_untraced=[],
    ),
    IntegrityParityCase(
        id="fail_number_after_per_unit_still_traced",
        narrative=f"The OR per 10 years was 1.37. {BASE_NARRATIVE}",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=False,
        expected_untraced=[1.37],
    ),
    # --- ADR 85: live-agent replicates (compound P keys, significance statement, z constants) ---
    IntegrityParityCase(
        id="pass_compound_p_key_traces_inequality",
        narrative="Sex was not associated with the outcome (P = 0.68). Age showed P < 0.001 (OR 1.45, n=412).",
        results_dict={"or_age": 1.45, "n": 412, "sex_p_value": 0.68, "age_p_value": 0.0004},
        expected_passed=True,
        expected_untraced=[],
    ),
    IntegrityParityCase(
        id="fail_compound_p_key_above_threshold_stays_untraced",
        narrative="Age showed P < 0.001 (OR 1.45, n=412).",
        results_dict={"or_age": 1.45, "n": 412, "age_p_value": 0.04},
        expected_passed=False,
        expected_untraced=[0.001],
    ),
    IntegrityParityCase(
        id="pass_two_sided_significance_statement",
        narrative="Two-sided P < .05 was the threshold. Treatment showed OR 1.10 (P = 0.68; n=412).",
        results_dict={"or_trt": 1.10, "n": 412, "p": 0.68},
        expected_passed=True,
        expected_untraced=[],
    ),
    IntegrityParityCase(
        id="pass_normal_critical_value_in_method",
        narrative=f"Limits of agreement were the mean difference ± 1.96 SD. {BASE_NARRATIVE}",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=True,
        expected_untraced=[],
    ),
    IntegrityParityCase(
        id="fail_critical_value_beside_sd_is_still_a_result",
        narrative=f"The mean difference was 1.96 (SD 2.10). {BASE_NARRATIVE}",
        results_dict={**BASE_RESULTS_DICT, "sd": 2.10},
        expected_passed=False,
        expected_untraced=[1.96],
    ),
    IntegrityParityCase(
        id="fail_bare_two_sided_result_p_stays_traced",
        narrative="Age had two-sided P < 0.001 (OR 1.45, n=412).",
        results_dict={"or_age": 1.45, "n": 412, "age_p_value": 0.04},
        expected_passed=False,
        expected_untraced=[0.001],
    ),
    IntegrityParityCase(
        id="pass_critical_value_named",
        narrative=f"The one-sided test used a critical value of 1.645. {BASE_NARRATIVE}",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=True,
        expected_untraced=[],
    ),
    IntegrityParityCase(
        id="fail_observed_sd_after_plus_minus",
        narrative=f"The mean ± SD was 0.4 ± 1.96. {BASE_NARRATIVE}",
        results_dict={**BASE_RESULTS_DICT, "mean": 0.4, "sd": 0.5},
        expected_passed=False,
        expected_untraced=[1.96],
    ),
    IntegrityParityCase(
        id="fail_observed_z_statistic",
        narrative=f"The Wald test gave z = 1.96. {BASE_NARRATIVE}",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=False,
        expected_untraced=[1.96],
    ),
    IntegrityParityCase(
        id="fail_result_in_sd_units_stays_traced",
        narrative=f"The standardized mean difference was 1.96 SDs; the mean difference was 1.96 standard deviations. {BASE_NARRATIVE}",
        results_dict={**BASE_RESULTS_DICT, "smd": 0.4},
        expected_passed=False,
        expected_untraced=[1.96, 1.96],
    ),
    IntegrityParityCase(
        id="pass_critical_value_multiplier_of_se",
        narrative=f"The 95% CI was the estimate ± 1.96 SE, i.e. 1.96 × SE either side. {BASE_NARRATIVE}",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=True,
        expected_untraced=[],
    ),
    IntegrityParityCase(
        id="fail_p_value_ci_bound_key_is_not_a_p_value",
        narrative="Age showed P < 0.05 (OR 1.45, n=412).",
        results_dict={"or_age": 1.45, "n": 412, "p_value_ci_lower": 0.01},
        expected_passed=False,
        expected_untraced=[0.05],
    ),
    IntegrityParityCase(
        id="fail_plural_p_values_ci_bound_key_is_not_a_p_value",
        narrative="Age showed P < 0.05 (OR 1.45, n=412).",
        results_dict={"or_age": 1.45, "n": 412, "p_values_ci_lower": 0.01},
        expected_passed=False,
        expected_untraced=[0.05],
    ),
    IntegrityParityCase(
        id="pass_plural_p_values_key_traces_inequality",
        narrative="Age showed P < 0.05 (OR 1.45, n=412).",
        results_dict={"or_age": 1.45, "n": 412, "p_values": [0.01, 0.30]},
        expected_passed=True,
        expected_untraced=[],
    ),
    IntegrityParityCase(
        id="fail_p_prefixed_keys_are_not_p_values",
        narrative="Age showed P < 0.05 (OR 1.45, n=412).",
        results_dict={"or_age": 1.45, "n": 412, "p_orig": 0.01, "p_score": 0.02},
        expected_passed=False,
        expected_untraced=[0.05],
    ),
    IntegrityParityCase(
        id="fail_critical_value_as_a_result_stays_untraced",
        narrative=f"There were 1.96 events per patient. {BASE_NARRATIVE}",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=False,
        expected_untraced=[1.96],
    ),
    IntegrityParityCase(
        id="fail_line_start_count_is_not_a_section",
        narrative=f"12 patients died.\n{BASE_NARRATIVE}",
        results_dict=BASE_RESULTS_DICT,
        expected_passed=False,
        expected_untraced=[12.0],
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


@dataclass(frozen=True)
class StudyDesignIntegrityCase:
    id: str
    study_design: str
    narrative: str
    results_dict: dict[str, Any]
    expected_caveat: bool
    allowed_untraced: set[float] | None = None
    expected_passed: bool | None = None


STUDY_DESIGN_INTEGRITY_CASES: list[StudyDesignIntegrityCase] = [
    StudyDesignIntegrityCase(
        id="non_randomized_diagnostic_test_requires_caveat",
        study_design="non-randomized diagnostic test",
        narrative="Methods: Non-randomized diagnostic test was conducted across 400 patients; adjusted odds ratio was 1.45.",
        results_dict={"sample_size": 400, "odds_ratio": 1.45},
        expected_caveat=False,
        expected_passed=False,
    ),
    StudyDesignIntegrityCase(
        id="non_randomized_diagnostic_test_with_caveat_passes",
        study_design="non-randomized diagnostic test",
        narrative="Methods: Cohort retention flow was analyzed. Non-randomized diagnostic test across 400 patients; adjusted odds ratio was 1.45. Residual confounding cannot be excluded.",
        results_dict={"sample_size": 400, "odds_ratio": 1.45},
        expected_caveat=True,
        expected_passed=True,
    ),
    StudyDesignIntegrityCase(
        id="etiologic_research_claim_smoking",
        study_design="risk prediction model",
        narrative="Methods: Clinical risk prediction model. An etiologic investigation of smoking as a lung-cancer risk factor was evaluated across 500 patients; odds ratio was 1.25.",
        results_dict={"sample_size": 500, "odds_ratio": 1.25},
        expected_caveat=False,
        expected_passed=False,
    ),
    StudyDesignIntegrityCase(
        id="etiology_workup_narrative_diagnostic",
        study_design="diagnostic accuracy",
        narrative="Methods: Cohort retention flow was evaluated. Etiology workup of acute abdominal pain across 300 patients; odds ratio was 1.25.",
        results_dict={"sample_size": 300, "odds_ratio": 1.25},
        expected_caveat=True,
        expected_passed=True,
    ),
    StudyDesignIntegrityCase(
        id="type_2_etiology_workup",
        study_design="type 2 etiology workup",
        narrative="Methods: Type 2 etiology workup was performed across 500 patients; odds ratio was 1.35.",
        results_dict={"sample_size": 500, "odds_ratio": 1.35},
        expected_caveat=False,
        expected_passed=False,
    ),
    StudyDesignIntegrityCase(
        id="type_2_etiologic_workup",
        study_design="type 2 etiologic workup",
        narrative="Methods: Type 2 etiologic workup was performed across 500 patients; odds ratio was 1.35.",
        results_dict={"sample_size": 500, "odds_ratio": 1.35},
        expected_caveat=False,
        expected_passed=False,
    ),
    StudyDesignIntegrityCase(
        id="etiologic_workup_type_2_diabetes",
        study_design="etiologic workup of type 2 diabetes",
        narrative="Methods: Cohort retention flow was analyzed. Diagnostic evaluation of etiologic workup of type 2 diabetes in 450 patients; odds ratio was 1.20.",
        results_dict={"sample_size": 450, "odds_ratio": 1.20},
        expected_caveat=True,
        allowed_untraced={2},
        expected_passed=True,
    ),
    StudyDesignIntegrityCase(
        id="diagnostic_accuracy_type_2_workup",
        study_design="diagnostic accuracy of type 2 etiologic work-up",
        narrative="Methods: Cohort retention flow was analyzed. Diagnostic accuracy of type 2 etiologic work-up across 350 patients; odds ratio was 1.25.",
        results_dict={"sample_size": 350, "odds_ratio": 1.25},
        expected_caveat=True,
        allowed_untraced={2},
        expected_passed=True,
    ),
    StudyDesignIntegrityCase(
        id="prediction_etiologic_stroke_subtype",
        study_design="prediction model for etiologic stroke subtype",
        narrative="Methods: Cohort retention flow was analyzed. A clinical risk prediction model for etiologic stroke subtype following TRIPOD guidelines in 600 patients; odds ratio was 1.30.",
        results_dict={"sample_size": 600, "odds_ratio": 1.30},
        expected_caveat=True,
        expected_passed=True,
    ),
    StudyDesignIntegrityCase(
        id="non_observational_prediction_simulation",
        study_design="non-observational prediction simulation",
        narrative="Methods: Cohort retention flow was analyzed. Non-observational prediction simulation study of 1000 synthetic patient cohorts; odds ratio was 1.15.",
        results_dict={"sample_size": 1000, "odds_ratio": 1.15},
        expected_caveat=True,
        expected_passed=True,
    ),
    StudyDesignIntegrityCase(
        id="disease_risk_score_matched_study",
        study_design="disease risk score matched study",
        narrative="Methods: Disease risk score matched study evaluating treatment across 800 patients; odds ratio was 1.40.",
        results_dict={"sample_size": 800, "odds_ratio": 1.40},
        expected_caveat=False,
        expected_passed=False,
    ),
    StudyDesignIntegrityCase(
        id="framingham_risk_score_validation",
        study_design="framingham risk score validation",
        narrative="Methods: Cohort retention flow was analyzed. Framingham risk score validation study following TRIPOD guidelines across 800 patients; odds ratio was 1.40.",
        results_dict={"sample_size": 800, "odds_ratio": 1.40},
        expected_caveat=True,
        expected_passed=True,
    ),
    StudyDesignIntegrityCase(
        id="matched_pair_randomized_trial",
        study_design="matched-pair randomized trial",
        narrative="Methods: Cohort retention flow was analyzed. A matched-pair randomized trial was conducted across 200 patients; mean difference was 4.2.",
        results_dict={"sample_size": 200, "mean_diff": 4.2},
        expected_caveat=True,
        expected_passed=True,
    ),
    StudyDesignIntegrityCase(
        id="etiologic_study_predictive_risk_factors",
        study_design="etiologic study of predictive risk factors",
        narrative="Methods: Etiologic study of predictive risk factors across 500 patients; odds ratio was 1.35.",
        results_dict={"sample_size": 500, "odds_ratio": 1.35},
        expected_caveat=False,
        expected_passed=False,
    ),
    StudyDesignIntegrityCase(
        id="natural_experiment",
        study_design="natural experiment",
        narrative="Methods: Natural experiment evaluating policy change across 600 patients; sample size 600.",
        results_dict={"sample_size": 600},
        expected_caveat=False,
        expected_passed=False,
    ),
    StudyDesignIntegrityCase(
        id="non_experimental_study",
        study_design="non-experimental study",
        narrative="Methods: Non-experimental study evaluating outcome across 500 patients; sample size 500.",
        results_dict={"sample_size": 500},
        expected_caveat=False,
        expected_passed=False,
    ),
    StudyDesignIntegrityCase(
        id="single_arm_study",
        study_design="single-arm study",
        narrative="Methods: Single-arm study of treatment across 300 patients; sample size 300.",
        results_dict={"sample_size": 300},
        expected_caveat=False,
        expected_passed=False,
    ),
    StudyDesignIntegrityCase(
        id="age_matched_myocardial_infarction_study",
        study_design="age-matched myocardial infarction study",
        narrative="Methods: Age-matched myocardial infarction study was conducted across 400 patients; sample size 400.",
        results_dict={"sample_size": 400},
        expected_caveat=False,
        expected_passed=False,
    ),
    StudyDesignIntegrityCase(
        id="quasi_randomized_trial",
        study_design="quasi-randomized trial",
        narrative="Methods: Cohort retention flow was analyzed. Quasi-randomized trial evaluating intervention across 300 patients; odds ratio was 1.25.",
        results_dict={"sample_size": 300, "odds_ratio": 1.25},
        expected_caveat=False,
        expected_passed=False,
    ),
    StudyDesignIntegrityCase(
        id="pseudo_randomized_study",
        study_design="pseudo-randomized study",
        narrative="Methods: Cohort retention flow was analyzed. Pseudo-randomized study across 350 patients; odds ratio was 1.20.",
        results_dict={"sample_size": 350, "odds_ratio": 1.20},
        expected_caveat=False,
        expected_passed=False,
    ),
    StudyDesignIntegrityCase(
        id="un_randomized_trial",
        study_design="un-randomized trial",
        narrative="Methods: Un-randomized trial of therapy was conducted across 300 patients; sample size 300.",
        results_dict={"sample_size": 300},
        expected_caveat=False,
        expected_passed=False,
    ),
    StudyDesignIntegrityCase(
        id="mendelian_randomization_study",
        study_design="mendelian randomization study",
        narrative="Methods: Mendelian randomization study of genetic variants was conducted across 500 patients; sample size 500.",
        results_dict={"sample_size": 500},
        expected_caveat=False,
        expected_passed=False,
    ),
    StudyDesignIntegrityCase(
        id="quasi_rct_study",
        study_design="quasi-rct study",
        narrative="Methods: Quasi-rct study of intervention was conducted across 300 patients; sample size 300.",
        results_dict={"sample_size": 300},
        expected_caveat=False,
        expected_passed=False,
    ),
    StudyDesignIntegrityCase(
        id="not_randomized_trial",
        study_design="not randomized trial",
        narrative="Methods: Not randomized trial of therapy was conducted across 350 patients; sample size 350.",
        results_dict={"sample_size": 350},
        expected_caveat=False,
        expected_passed=False,
    ),
    StudyDesignIntegrityCase(
        id="pre_experimental_study",
        study_design="pre-experimental study",
        narrative="Methods: Pre-experimental study evaluating intervention across 250 patients; sample size 250.",
        results_dict={"sample_size": 250},
        expected_caveat=False,
        expected_passed=False,
    ),
    StudyDesignIntegrityCase(
        id="randomized_trial_in_mendelian_disease",
        study_design="randomized controlled trial in mendelian disease",
        narrative="Methods: Cohort retention flow was analyzed. A randomized controlled trial in mendelian disease was conducted across 120 patients; sample size 120.",
        results_dict={"sample_size": 120},
        expected_caveat=True,
        expected_passed=True,
    ),
    StudyDesignIntegrityCase(
        id="randomized_trial_with_pre_experiment_baseline_measurements",
        study_design="randomized controlled trial with pre-experiment baseline measurements",
        narrative="Methods: Cohort retention flow was analyzed. A randomized controlled trial with pre-experiment baseline measurements was conducted across 300 patients; sample size 300.",
        results_dict={"sample_size": 300},
        expected_caveat=True,
        expected_passed=True,
    ),
    StudyDesignIntegrityCase(
        id="randomized_trial_with_pre_experimental_period",
        study_design="randomized controlled trial with pre-experimental period",
        narrative="Methods: Cohort retention flow was analyzed. A randomized controlled trial with pre-experimental period was conducted across 280 patients; sample size 280.",
        results_dict={"sample_size": 280},
        expected_caveat=True,
        expected_passed=True,
    ),
    StudyDesignIntegrityCase(
        id="pre_experimental_one_group_pretest_posttest_design",
        study_design="pre-experimental one-group pretest-posttest design",
        narrative="Methods: Pre-experimental one-group pretest-posttest design evaluating intervention across 250 patients; sample size 250.",
        results_dict={"sample_size": 250},
        expected_caveat=False,
        expected_passed=False,
    ),
]

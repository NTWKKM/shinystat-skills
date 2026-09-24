"""
tests/stress/test_m2_challenger2_stress.py: Adversarial Empirical Stress Harness for Milestone M2.

Author: Challenger 2 (Empirical Challenger)
Scope:
1. SampleFlowTracker Invariants & CONSORT Trial Branching:
   - Strict conservation invariants: mismatched sums (n_start != n_remaining + n_excluded),
     negative remaining, negative excluded.
   - Constructor gap: SampleFlowTracker(initial_n < 0) bypasses validation.
   - Branching gap: branch_arms permits negative arm allocations when net sum matches.
   - Multi-level trial designs: 3-arm, 4-arm (2x2 factorial), and sequential cross-over trials.
   - Total attrition to N=0 and uninitialized flow safety.
   - Flow diagram rendering resilience: extreme box widths (0, 1, 1000), long text (300+ chars),
     empty reasons, multi-reason tuples/strings, Thai/Arabic/Greek/Emoji Unicode characters.
   - Mermaid diagram export: unescaped quotes in labels and arm omission.
2. Data Quality Scoring & Parity Stress:
   - Extreme boundary DataFrames: 0 rows, 1 row, 1 column, all-null, all-duplicate.
   - Fatal Boolean Column Crash: numpy boolean subtraction TypeError in plausibility_score().
   - Duplicate column names crash in check_data_quality and DataQualityReport.
   - Unhashable nested types crash (list/dict in cells).
   - Mixed dirty tokens: currencies ($ , €, £, ฿, ¥), scientific notation, percentages.
   - Unicode negative signs (- vs - vs -): misclassification and inversion of warnings.
   - Infinite values (+inf, -inf): stealth bypass of quality dimensions.
   - Cross-variable logical rules: syntax errors, missing columns, lambda exceptions, and
     the 10-point penalty cap allowing impossible vitals to receive Grade A (98/100).
   - Plausibility blindspot: zero-inflated / zero-IQR data causes extreme outliers to be completely missed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from medstat.data.quality import (
    CrossVariableRule,
    DataQualityReport,
    DataQualitySchema,
    _is_numeric_column,
    check_data_quality,
)
from medstat.data.retention import FlowDesign, SampleFlowTracker

pytestmark = pytest.mark.unit


# =============================================================================
# 1. SampleFlowTracker Invariants & Multi-Arm CONSORT Branching
# =============================================================================


class TestSampleFlowTrackerInvariantsStress:
    """Adversarial stress-testing of participant conservation and invariant checks."""

    def test_invariant_conservation_mismatch_raises_error(self):
        """Mismatched sum (n_start != n_remaining + n_excluded) must raise ValueError."""
        tracker = SampleFlowTracker(initial_n=500)
        with pytest.raises(ValueError, match="Conservation invariant violated"):
            tracker.record_stage(
                "Faulty Math", n_remaining=400, n_excluded=50
            )  # 450 != 500

    def test_invariant_negative_remaining_raises_error(self):
        """Negative n_remaining must raise ValueError."""
        tracker = SampleFlowTracker(initial_n=100)
        with pytest.raises(
            ValueError, match="Remaining sample size cannot be negative"
        ):
            tracker.record_stage("Negative Remaining", n_remaining=-10, n_excluded=110)

    def test_invariant_negative_excluded_raises_error(self):
        """Negative n_excluded (cohort expansion attempt) must raise ValueError."""
        tracker = SampleFlowTracker(initial_n=100)
        # Explicit negative excluded
        with pytest.raises(ValueError, match="Excluded sample size cannot be negative"):
            tracker.record_stage("Expansion", n_remaining=120, n_excluded=-20)

        # Implicit negative excluded (n_remaining > n_start with n_excluded=None)
        with pytest.raises(ValueError, match="Excluded sample size cannot be negative"):
            tracker.record_stage("Expansion Implicit", n_remaining=120)

    def test_invariant_negative_initial_n_constructor_gap(self):
        """Constructor must raise ValueError on negative initial_n."""
        tracker_valid = SampleFlowTracker()
        with pytest.raises(ValueError, match="Initial sample size cannot be negative"):
            tracker_valid.set_initial(-50)

        with pytest.raises(ValueError, match="Initial sample size cannot be negative"):
            SampleFlowTracker(initial_n=-50)

    def test_branch_arms_negative_allocation_gap(self):
        """branch_arms must raise ValueError if any individual arm allocation is negative."""
        tracker = SampleFlowTracker(initial_n=100)
        with pytest.raises(ValueError, match="Arm allocation .* cannot be negative"):
            tracker.branch_arms({"Arm_Valid": 120, "Arm_Negative": -20})

    def test_consort_3arm_sequential_attrition(self):
        """
        Multi-arm CONSORT trial: 3 arms (High Dose, Low Dose, Placebo) with
        sequential follow-up attrition (Month 1, Month 6, Complete Analysis).
        """
        tracker = SampleFlowTracker(initial_n=300, design=FlowDesign.CONSORT)
        tracker.record_stage("Eligibility Screening", n_remaining=300, n_excluded=0)

        arms = tracker.branch_arms({"High_Dose": 100, "Low_Dose": 100, "Placebo": 100})

        # Arm-specific independent attrition
        arms["High_Dose"].record_stage(
            "Month 1 Follow-up", n_remaining=92, n_excluded=8, reason="Severe nausea"
        )
        arms["High_Dose"].record_stage(
            "Month 6 Follow-up",
            n_remaining=85,
            n_excluded=7,
            reason="Lost to follow-up",
        )

        arms["Low_Dose"].record_stage(
            "Month 1 Follow-up", n_remaining=96, n_excluded=4, reason="Rash"
        )
        arms["Low_Dose"].record_stage(
            "Month 6 Follow-up", n_remaining=91, n_excluded=5, reason="Relocated"
        )

        arms["Placebo"].record_stage(
            "Month 1 Follow-up",
            n_remaining=98,
            n_excluded=2,
            reason="Consent withdrawn",
        )
        arms["Placebo"].record_stage(
            "Month 6 Follow-up", n_remaining=94, n_excluded=4, reason="Lack of efficacy"
        )

        summary = tracker.get_flow_summary()
        assert len(summary["arms"]) == 3
        assert summary["arms"]["High_Dose"]["final_n"] == 85
        assert summary["arms"]["Low_Dose"]["final_n"] == 91
        assert summary["arms"]["Placebo"]["final_n"] == 94

        # Verify diagram contains all 3 arms
        ascii_flow = tracker.render_ascii_flow()
        assert "Allocated into 3 trial arms:" in ascii_flow
        assert "Arm [High_Dose]: N = 100 -> Analyzed: N = 85" in ascii_flow
        assert "Arm [Low_Dose]: N = 100 -> Analyzed: N = 91" in ascii_flow
        assert "Arm [Placebo]: N = 100 -> Analyzed: N = 94" in ascii_flow

    def test_consort_4arm_factorial_branching(self):
        """2x2 Factorial trial: 4 arms tracking independent multi-stage retention."""
        tracker = SampleFlowTracker(initial_n=400, design=FlowDesign.CONSORT)
        arms = tracker.branch_arms(
            {
                "DrugA_DrugB": 100,
                "DrugA_PlaceboB": 100,
                "PlaceboA_DrugB": 100,
                "PlaceboA_PlaceboB": 100,
            }
        )
        assert len(arms) == 4
        for name, arm in arms.items():
            arm.record_stage(
                "Protocol Attrition",
                n_remaining=90,
                n_excluded=10,
                reason=f"{name} dropout",
            )
            assert arm.final_n == 90
            assert arm.total_excluded == 10

        summary = tracker.get_flow_summary()
        assert len(summary["arms"]) == 4

    def test_consort_crossover_sequential_attrition(self):
        """
        Two-period cross-over trial (AB and BA sequence):
        Period 1 -> Washout Attrition -> Period 2 (Crossover).
        """
        crossover = SampleFlowTracker(
            initial_n=100, initial_name="Crossover Cohort", design=FlowDesign.CONSORT
        )
        arms = crossover.branch_arms({"Seq_AB": 50, "Seq_BA": 50})

        # Seq AB
        arms["Seq_AB"].record_stage(
            "Period 1 (Drug A)", n_remaining=48, n_excluded=2, reason="Adverse event"
        )
        arms["Seq_AB"].record_stage(
            "Washout Period", n_remaining=45, n_excluded=3, reason="Consent withdrawn"
        )
        arms["Seq_AB"].record_stage(
            "Period 2 (Drug B)",
            n_remaining=44,
            n_excluded=1,
            reason="Lost to follow-up",
        )

        # Seq BA
        arms["Seq_BA"].record_stage(
            "Period 1 (Drug B)", n_remaining=47, n_excluded=3, reason="Lack of efficacy"
        )
        arms["Seq_BA"].record_stage(
            "Washout Period", n_remaining=46, n_excluded=1, reason="Relocated"
        )
        arms["Seq_BA"].record_stage(
            "Period 2 (Drug A)", n_remaining=43, n_excluded=3, reason="Adverse event"
        )

        summary = crossover.get_flow_summary()
        assert summary["arms"]["Seq_AB"]["final_n"] == 44
        assert summary["arms"]["Seq_BA"]["final_n"] == 43
        total_analyzed = (
            summary["arms"]["Seq_AB"]["final_n"] + summary["arms"]["Seq_BA"]["final_n"]
        )
        assert total_analyzed == 87

    def test_total_cohort_attrition_to_zero(self):
        """Total attrition (100% loss): must not divide by zero or crash."""
        tracker = SampleFlowTracker(initial_n=50)
        s1 = tracker.record_stage(
            "Terminated", n_remaining=0, n_excluded=50, reason="Study terminated"
        )
        assert s1.retention_rate_stage == 0.0
        assert s1.retention_rate_overall == 0.0
        assert tracker.final_n == 0
        assert tracker.overall_retention_rate == 0.0

        # Post-closure stage
        s2 = tracker.record_stage(
            "Post-Closure", n_remaining=0, n_excluded=0, reason="No remaining subjects"
        )
        assert s2.n_start == 0
        assert s2.n_remaining == 0
        assert s2.retention_rate_stage == 0.0

    def test_uninitialized_sample_flow_tracker(self):
        """Uninitialized tracker (initial_n=None): getters and diagrams must be zero-safe."""
        tracker = SampleFlowTracker()
        assert tracker.current_n == 0
        assert tracker.final_n == 0
        assert tracker.total_excluded == 0
        assert tracker.overall_retention_rate == 0.0

        summary = tracker.get_flow_summary()
        assert summary["n_initial"] == 0
        assert summary["n_analyzed"] == 0

        ascii_out = tracker.render_ascii_flow()
        assert "N = 0" in ascii_out

        mermaid_out = tracker.to_mermaid()
        assert "N = 0" in mermaid_out

    def test_diagram_rendering_stress_unicode_and_ascii(self):
        """
        Stress test ASCII and Unicode rendering:
        - Box widths: 0, 1, 10, 1000.
        - Long text: 300+ character stage names and reasons.
        - Empty reasons: '' and [].
        - Multi-reason lists: mixing tuples and strings.
        - Special characters: Thai (ภาษาไทย), Arabic (مريض), Greek (α, β, γ), Emojis (🏥, 💊), quotes, tabs, newlines.
        """
        tracker = SampleFlowTracker(
            initial_n=1000, initial_name="Screened (รพช. & รพท. 🏥)"
        )

        long_stage_name = (
            "Stage with an exceptionally verbose clinical description that far exceeds standard display "
            * 3
        )
        long_reasons = [
            (
                "Patient excluded due to anaphylaxis to drug formulation A in combination with elevated ALT > 500 U/L",
                40,
            ),
            ("ผู้ป่วยปฏิเสธการรักษาและขอถอนตัวจากการวิจัย (Withdrawal of consent)", 30),
            (
                "Complex symbols: <script>alert('xss')</script> & \"quotes\" and \t tabs",
                20,
            ),
            "Uncounted reason string without tuple count",
        ]

        tracker.record_stage(
            stage_name=long_stage_name,
            n_remaining=900,
            n_excluded=100,
            reason=long_reasons,
        )

        tracker.record_stage(
            stage_name="Empty string reason", n_remaining=850, n_excluded=50, reason=""
        )
        tracker.record_stage(
            stage_name="Empty list reason", n_remaining=800, n_excluded=50, reason=[]
        )

        # Verify no crash across diverse styles and box widths
        for width in [0, 1, 10, 58, 200]:
            u_flow = tracker.render_ascii_flow(style="unicode", box_width=width)
            a_flow = tracker.render_ascii_flow(style="ascii", box_width=width)
            assert len(u_flow) > 0
            assert len(a_flow) > 0
            assert "ผู้ป่วยปฏิเสธการรักษา" in u_flow
            assert "ALT > 500 U/L" in a_flow

    def test_to_mermaid_quotes_and_arms_omission(self):
        """
        EMPIRICAL OBSERVATION: to_mermaid() syntax and arm handling.
        1. Quotes in stage names or reasons are emitted unescaped in Mermaid node labels.
        2. to_mermaid() omits trial arms created via branch_arms.
        """
        tracker = SampleFlowTracker(initial_n=100, initial_name='Screened "Cohort A"')
        tracker.record_stage(
            'Stage 1: "Eligible"',
            n_remaining=90,
            n_excluded=10,
            reason='Reason "Ineligible"',
        )
        tracker.branch_arms({"Arm_1": 45, "Arm_2": 45})

        mermaid_str = tracker.to_mermaid()
        assert (
            "Screened 'Cohort A'" in mermaid_str
        )  # Double quotes escaped to single quotes
        assert "Arm_1" in mermaid_str  # Arms are now rendered in Mermaid


# =============================================================================
# 2. Data Quality Scoring & Parity Stress
# =============================================================================


class TestDataQualityScoringStress:
    """Adversarial stress-testing of check_data_quality and DataQualityReport."""

    def test_boundary_empty_dataframe(self):
        """0-row DataFrames: must not raise ZeroDivisionError or crash."""
        df_empty = pd.DataFrame()
        assert check_data_quality(df_empty) == []

        r_empty = DataQualityReport(df_empty)
        rep = r_empty.generate_report()
        assert rep["total_rows"] == 0
        assert rep["total_columns"] == 0
        assert rep["overall_score"] == 100.0  # Empty dataset receives Grade A

        df_empty_cols = pd.DataFrame(columns=["age", "bmi"])
        r_cols = DataQualityReport(df_empty_cols)
        rep_cols = r_cols.generate_report()
        assert rep_cols["total_rows"] == 0
        assert rep_cols["total_columns"] == 2
        assert rep_cols["overall_score"] == 100.0

    def test_boundary_single_row_and_single_column(self):
        """1-row and 1-column DataFrames."""
        df_1x1 = pd.DataFrame({"val": [42.0]})
        r_1x1 = DataQualityReport(df_1x1)
        res_1x1 = r_1x1.generate_report()
        assert res_1x1["overall_score"] == 100.0

        df_1x5 = pd.DataFrame(
            {"a": [1], "b": [2.5], "c": ["x"], "d": [None], "e": [100]}
        )
        r_1x5 = DataQualityReport(df_1x5)
        res_1x5 = r_1x5.generate_report()
        assert res_1x5["dimension_scores"]["completeness"] == 80.0  # 4 of 5 non-missing

    def test_boundary_all_null_dataframe(self):
        """All-null DataFrame: completeness must be 0.0, uniqueness 50% (duplicates)."""
        df_null = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
        r_null = DataQualityReport(df_null)
        res = r_null.generate_report()
        assert res["dimension_scores"]["completeness"] == 0.0
        assert any(
            "significant missing values" in rec for rec in res["recommendations"]
        )

    def test_boundary_all_duplicate_dataframe(self):
        """All-duplicate rows: uniqueness penalized."""
        df_dup = pd.DataFrame({"a": [1, 1, 1, 1], "b": ["x", "x", "x", "x"]})
        r_dup = DataQualityReport(df_dup)
        res = r_dup.generate_report()
        assert res["dimension_scores"]["uniqueness"] == 25.0  # 3 of 4 are duplicates
        assert any("Duplicate rows detected" in rec for rec in res["recommendations"])

    def test_boolean_column_plausibility_crash(self):
        """Boolean columns must be handled gracefully without boolean subtraction TypeError."""
        df_bool = pd.DataFrame({"is_smoker": [True, False, True, False, True, True]})
        report = DataQualityReport(df_bool)
        assert report.plausibility_score() == 100.0

    def test_duplicate_column_names_crash(self):
        """Verify duplicate column headers are handled gracefully without unhandled crashes."""
        df_dup_cols = pd.DataFrame([[1, 2], [3, 4]], columns=["score", "score"])
        warnings = check_data_quality(df_dup_cols)
        assert isinstance(warnings, list)

        report = DataQualityReport(df_dup_cols)
        rep = report.generate_report()
        assert isinstance(rep, dict)

    def test_unhashable_nested_objects_crash(self):
        """Verify unhashable objects (lists/dicts) are handled gracefully without crashes."""
        df_nested = pd.DataFrame({"payload": [[1, 2], [3, 4], None, [5, 6]]})
        warnings = check_data_quality(df_nested)
        assert isinstance(warnings, list)

    def test_dirty_tokens_multi_country_currencies(self):
        """Currencies from multiple countries ($ , €, £, ฿, ¥) flagged as non-standard."""
        df = pd.DataFrame(
            {
                "usd": ["$ 100", "$ 200", "$ 300", "$ 400"],
                "thb": ["฿ 500", "฿ 1,000", "฿ 1,500", "฿ 2,000"],
                "eur": ["€ 50", "€ 60", "€ 70", "€ 80"],
                "gbp": ["£ 10", "£ 20", "£ 30", "£ 40"],
                "jpy": ["¥ 1,000", "¥ 2,000", "¥ 3,000", "¥ 4,000"],
            }
        )
        warnings = check_data_quality(df)
        assert len(warnings) == 5
        for col in ["usd", "thb", "eur", "gbp", "jpy"]:
            assert any(
                f"Column '{col}'" in w and "non-standard values" in w for w in warnings
            )

        report = DataQualityReport(df)
        # All columns have non-standard tokens -> consistency score = 0.0
        assert report.consistency_score() == 0.0

    def test_dirty_tokens_unicode_negative_signs_misclassification(self):
        """
        ALGORITHMIC VULNERABILITY EMPIRICAL PROOF:
        Unicode negative symbols ('−' U+2212, '–' U+2013) are omitted from _CLEAN_SYMBOL_RE.
        As a result, a numeric column with unicode negatives is classified as CATEGORICAL,
        and valid positive numbers are erroneously flagged as 'numeric values inside categorical column'.
        """
        # 5 unicode negative numbers, 5 positive numbers
        df = pd.DataFrame(
            {
                "diff": [
                    "−1.2",
                    "−3.4",
                    "−5.6",
                    "−7.8",
                    "−9.0",
                    "1.1",
                    "2.2",
                    "3.3",
                    "4.4",
                    "5.5",
                ]
            }
        )
        warnings = check_data_quality(df)

        assert len(warnings) == 1
        assert "Column 'diff'" in warnings[0]
        assert any(
            term in warnings[0]
            for term in [
                "non-standard values",
                "numeric values inside categorical column",
            ]
        )

    def test_dirty_tokens_scientific_notation_and_percentages(self):
        """Scientific notation is parsed natively; percentage strings are flagged."""
        # Scientific notation: 1e-4, 3E+2
        df_sci = pd.DataFrame(
            {"sci": ["1.23e-4", "4.56E+2", "7.89e3", "0.01", "100.0"]}
        )
        is_num, _, _, _, nan_count = _is_numeric_column(df_sci["sci"], 5)
        assert bool(is_num) is True
        assert nan_count == 0  # Scientific notation parsed cleanly

        # Percentages: 10%, 20%
        df_pct = pd.DataFrame({"rate": ["10%", "20%", "30%", "40%", "50%"]})
        warnings = check_data_quality(df_pct)
        assert len(warnings) == 1
        assert "non-standard values" in warnings[0]

    @pytest.mark.xfail(
        strict=True,
        reason="Infinite values should trigger quality warnings and penalize plausibility/validity",
    )
    def test_infinite_values_handling(self):
        """
        Infinite values (np.inf, -np.inf) should trigger warnings and reduce plausibility or validity scores.
        """
        df_inf = pd.DataFrame({"lab_ratio": [1.0, 2.0, np.inf, -np.inf, 5.0]})
        report = DataQualityReport(df_inf)

        # Expect reduced plausibility or validity score
        assert report.plausibility_score() < 100.0 or report.validity_score() < 100.0
        # Check data quality reports warnings
        issues = check_data_quality(df_inf)
        assert len(issues) > 0

    def test_cross_variable_rules_exception_resilience(self):
        """
        CrossVariableRule exceptions (syntax errors, missing columns, division by zero,
        non-boolean return values) are safely caught and ignored without crashing.
        """
        df = pd.DataFrame({"age": [20, 30, 40], "score": [5, 10, 15]})
        rules = [
            CrossVariableRule(
                name="syntax_err", condition="age ==", description="bad syntax"
            ),
            CrossVariableRule(
                name="missing_col",
                condition="age > unknown_col",
                description="missing col",
            ),
            CrossVariableRule(
                name="div_zero", condition=lambda d: 1 / 0, description="div zero"
            ),
            CrossVariableRule(
                name="scalar_bool",
                condition=lambda d: True,
                description="returns scalar bool",
            ),
            CrossVariableRule(
                name="returns_none",
                condition=lambda d: None,
                description="returns None",
            ),
        ]
        schema = DataQualitySchema(cross_variable_rules=rules)
        report = DataQualityReport(df, schema=schema)
        # Does not crash; exceptions are swallowed
        score = report.consistency_score()
        assert score == 100.0

    def test_cross_variable_impossible_vitals_penalty_cap(self):
        """
        CLINICAL AUDITING GAP EMPIRICAL PROOF:
        In consistency_score(), rule penalty is hard-capped at (violations / total_rows) * 10.0.
        Even if 100% of rows violate a critical physiological rule (diastolic > systolic),
        consistency only drops by 10 points (to 90.0), allowing the impossible dataset to score 98.0 (Grade A).
        """
        # 100 unique patients, all with impossible vitals (diastolic > systolic)
        df = pd.DataFrame(
            {
                "systolic": [80.0 + i for i in range(100)],
                "diastolic": [140.0 + i for i in range(100)],
            }
        )
        rule = CrossVariableRule(
            name="bp_sanity",
            condition="systolic > diastolic",
            description="Systolic must exceed diastolic",
            severity="critical",
        )
        schema = DataQualitySchema(cross_variable_rules=[rule])
        report = DataQualityReport(df, schema=schema)
        res = report.generate_report()

        # Critical violations properly penalize consistency score
        assert res["dimension_scores"]["consistency"] <= 50.0

    def test_plausibility_zero_iqr_extreme_outlier_blindspot(self):
        """Zero-inflated medical data with extreme outlier is penalized via MAD fallback."""
        vals = [0.0] * 99 + [1_000_000.0]
        df = pd.DataFrame({"icu_events": vals})
        report = DataQualityReport(df)

        score = report.plausibility_score()
        assert score <= 99.0, (
            f"Expected <= 99.0 due to MAD outlier detection, got {score}"
        )

    def test_plausibility_constant_column_step_change(self):
        """
        EMPIRICAL OBSERVATION: Arbitrary threshold n > 20 for constant column penalty.
        A constant column with N=20 receives plausibility score 100.0,
        while N=21 receives plausibility score 90.5%.
        """
        df_20 = pd.DataFrame({"const": [5.0] * 20})
        report_20 = DataQualityReport(df_20)
        assert report_20.plausibility_score() == 100.0

        df_21 = pd.DataFrame({"const": [5.0] * 21})
        report_21 = DataQualityReport(df_21)
        assert report_21.plausibility_score() < 95.0

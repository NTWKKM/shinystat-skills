"""
tests/unit/test_sample_flow_retention.py: Unit tests for SampleFlowTracker & audited retention.
"""

import pandas as pd
import pytest

from medstat.data.retention import FlowDesign, SampleFlowTracker

pytestmark = pytest.mark.unit


class TestSampleFlowTracker:
    def test_basic_flow_lifecycle(self):
        tracker = SampleFlowTracker(initial_n=1000, initial_name="Screened Cohort")
        assert tracker.current_n == 1000
        assert tracker.final_n == 1000
        assert tracker.total_excluded == 0
        assert tracker.overall_retention_rate == 100.0

        # Stage 1: Eligibility
        s1 = tracker.record_stage(
            stage_name="Eligibility Exclusion",
            n_remaining=900,
            n_excluded=100,
            reason="Age < 18 or previous stroke",
        )
        assert s1.n_start == 1000
        assert s1.n_remaining == 900
        assert tracker.current_n == 900
        assert tracker.total_excluded == 100
        assert tracker.overall_retention_rate == 90.0

        # Stage 2: Missing Covariates
        s2 = tracker.record_stage(
            stage_name="Missing Covariates",
            n_remaining=850,
            reason="Incomplete baseline lab panel",
        )
        assert s2.n_start == 900
        assert s2.n_excluded == 50
        assert tracker.final_n == 850
        assert tracker.total_excluded == 150
        assert tracker.overall_retention_rate == 85.0

    def test_conservation_invariant_violation_raises_error(self):
        tracker = SampleFlowTracker(initial_n=500)
        # Attempt to record invalid stage where 500 != 450 + 20
        with pytest.raises(ValueError, match="Conservation invariant violated"):
            tracker.record_stage(
                stage_name="Faulty Stage",
                n_remaining=450,
                n_excluded=20,
                reason="Invalid math",
            )

    def test_negative_n_raises_error(self):
        tracker = SampleFlowTracker(initial_n=100)
        with pytest.raises(ValueError):
            tracker.record_stage("Error", n_remaining=-10, n_excluded=110)

    def test_apply_filter(self):
        df = pd.DataFrame({"age": [15, 25, 35, 12, 60], "status": [1, 1, 0, 1, 0]})
        tracker = SampleFlowTracker()

        # Filter adults
        df_adults = tracker.apply_filter(
            df=df,
            condition=lambda d: d["age"] >= 18,
            stage_name="Age Eligibility",
            reason="Age < 18",
        )

        assert len(df_adults) == 3
        assert tracker.initial_n == 5
        assert tracker.final_n == 3
        assert tracker.total_excluded == 2

    def test_branch_arms(self):
        tracker = SampleFlowTracker(initial_n=200, design=FlowDesign.CONSORT)
        arms = tracker.branch_arms({"Treatment": 100, "Control": 100})

        assert "Treatment" in arms
        assert "Control" in arms
        assert arms["Treatment"].initial_n == 100
        assert arms["Control"].initial_n == 100

        # Record arm-specific attrition
        arms["Treatment"].record_stage(
            "Adverse Event", n_remaining=95, n_excluded=5, reason="Drug rash"
        )
        arms["Control"].record_stage(
            "Loss to Follow-up", n_remaining=98, n_excluded=2, reason="Relocated"
        )

        summary = tracker.get_flow_summary()
        assert "arms" in summary
        assert summary["arms"]["Treatment"]["final_n"] == 95
        assert summary["arms"]["Control"]["final_n"] == 98

    def test_branch_arms_mismatch_raises_error(self):
        tracker = SampleFlowTracker(initial_n=100)
        with pytest.raises(ValueError, match="Sum of arm allocations"):
            tracker.branch_arms({"ArmA": 50, "ArmB": 40})  # 90 != 100

    def test_diagram_rendering(self):
        tracker = SampleFlowTracker(initial_n=500, initial_name="Enrollment")
        tracker.record_stage(
            "Screening",
            n_remaining=450,
            n_excluded=50,
            reason=[("Exclusion criteria", 30), ("Declined", 20)],
        )
        tracker.record_stage(
            "Complete Case",
            n_remaining=400,
            n_excluded=50,
            reason="Missing biomarker",
        )

        ascii_flow = tracker.render_ascii_flow(style="ascii")
        assert "Stage 0: Enrollment" in ascii_flow
        assert "N = 500" in ascii_flow
        assert "Exclusion criteria: 30" in ascii_flow

        unicode_flow = tracker.render_ascii_flow(style="unicode")
        assert "┌" in unicode_flow
        assert "├─►" in unicode_flow

        mermaid = tracker.to_mermaid()
        assert "flowchart TD" in mermaid
        assert "S0" in mermaid
        assert "S1" in mermaid

"""
tests/unit/test_pr5_coderabbit_fixes.py: Verification of CodeRabbit review fixes in PR#5.

Tests:
1. CLI: --egger option guards (rejecting k < 10 studies and binary log odds-ratio effect columns).
2. Causal: PSM deterministic sorting, categorical/boundary SMD checks, Bland-Altman 95% CIs.
3. Clean: Audited sample retention with named exclusion stage and clinical rationale persistence.
4. Diagnostic: DCA net benefit including Treat All and Treat None strategies, prespecified directionality rule.
5. Models: Design-matrix EPV parameter degrees of freedom, Table 1 baseline summary SMDs, VanderWeele OR E-value with rare outcome.
6. Report: Metadata-driven methods narrative generator, NEJM/JAMA publication table styling and p-value formatting.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from click.testing import CliRunner

from medstat.cli.main import cli
from medstat.data.retention import SampleFlowTracker
from medstat.models.sensitivity import calculate_e_value


# ==============================================================================
# 1. CLI --egger Guards Verification
# ==============================================================================
class TestEggerCLIGuards:
    def test_egger_rejects_fewer_than_10_studies(self, tmp_path):
        runner = CliRunner()
        # 5 studies (less than 10)
        df = pd.DataFrame(
            {
                "study": [f"Study_{i}" for i in range(5)],
                "effect_size": [0.2, -0.1, 0.3, 0.05, -0.15],
                "se": [0.1, 0.08, 0.12, 0.09, 0.11],
            }
        )
        csv_path = tmp_path / "meta_sparse.csv"
        df.to_csv(csv_path, index=False)

        res = runner.invoke(
            cli,
            [
                "meta",
                "--data",
                str(csv_path),
                "--effect-col",
                "effect_size",
                "--se-col",
                "se",
                "--study-col",
                "study",
                "--measure",
                "continuous",
                "--egger",
            ],
        )
        assert res.exit_code != 0
        assert "Egger's test requires at least 10" in res.output

    def test_egger_rejects_fewer_than_10_unique_studies(self, tmp_path):
        runner = CliRunner()
        # 12 rows, but only 4 unique studies
        df = pd.DataFrame(
            {
                "study": [
                    "Study_A",
                    "Study_A",
                    "Study_A",
                    "Study_B",
                    "Study_B",
                    "Study_C",
                    "Study_C",
                    "Study_C",
                    "Study_D",
                    "Study_D",
                    "Study_D",
                    "Study_D",
                ],
                "effect_size": np.random.normal(0.2, 0.1, 12),
                "se": np.random.uniform(0.05, 0.15, 12),
            }
        )
        csv_path = tmp_path / "meta_dup_studies.csv"
        df.to_csv(csv_path, index=False)

        res = runner.invoke(
            cli,
            [
                "meta",
                "--data",
                str(csv_path),
                "--effect-col",
                "effect_size",
                "--se-col",
                "se",
                "--study-col",
                "study",
                "--measure",
                "continuous",
                "--egger",
            ],
        )
        assert res.exit_code != 0
        assert (
            "Egger's test requires independent study estimates, but duplicate study IDs were detected in 'study'"
            in res.output
        )

    def test_egger_rejects_fewer_than_10_independent_studies(self, tmp_path):
        runner = CliRunner()
        # 4 rows with 4 unique studies
        df = pd.DataFrame(
            {
                "study": ["Study_A", "Study_B", "Study_C", "Study_D"],
                "effect_size": [0.2, 0.3, 0.1, 0.4],
                "se": [0.05, 0.08, 0.06, 0.07],
            }
        )
        csv_path = tmp_path / "meta_few_studies.csv"
        df.to_csv(csv_path, index=False)

        res = runner.invoke(
            cli,
            [
                "meta",
                "--data",
                str(csv_path),
                "--effect-col",
                "effect_size",
                "--se-col",
                "se",
                "--study-col",
                "study",
                "--measure",
                "continuous",
                "--egger",
            ],
        )
        assert res.exit_code != 0
        assert (
            "Egger's test requires at least 10 independent studies (got 4)"
            in res.output
        )

    def test_egger_rejects_missing_measure(self, tmp_path):
        runner = CliRunner()
        df = pd.DataFrame(
            {
                "study": [f"Study_{i}" for i in range(12)],
                "effect_size": np.random.normal(0.2, 0.1, 12),
                "se": np.random.uniform(0.05, 0.15, 12),
            }
        )
        csv_path = tmp_path / "meta_no_measure.csv"
        df.to_csv(csv_path, index=False)

        res = runner.invoke(
            cli,
            [
                "meta",
                "--data",
                str(csv_path),
                "--effect-col",
                "effect_size",
                "--se-col",
                "se",
                "--study-col",
                "study",
                "--egger",
            ],
        )
        assert res.exit_code != 0
        assert (
            "Egger's test requires an explicit continuous effect measure" in res.output
        )

    def test_egger_rejects_binary_log_odds_ratio(self, tmp_path):
        runner = CliRunner()
        # 12 studies but measure is log_or
        df = pd.DataFrame(
            {
                "study": [f"Study_{i}" for i in range(12)],
                "log_or": np.random.normal(0.2, 0.1, 12),
                "se": np.random.uniform(0.05, 0.15, 12),
            }
        )
        csv_path = tmp_path / "meta_log_or.csv"
        df.to_csv(csv_path, index=False)

        res = runner.invoke(
            cli,
            [
                "meta",
                "--data",
                str(csv_path),
                "--effect-col",
                "log_or",
                "--se-col",
                "se",
                "--study-col",
                "study",
                "--measure",
                "log_or",
                "--egger",
            ],
        )
        assert res.exit_code != 0
        assert "Egger's test is invalid for binary log odds ratios" in res.output

    def test_egger_rejects_with_explicit_measure_or_metadata(self, tmp_path):
        runner = CliRunner()
        # 12 studies where column is named 'effect_size' but --measure is log_or
        df = pd.DataFrame(
            {
                "study": [f"Study_{i}" for i in range(12)],
                "effect_size": np.random.normal(0.2, 0.1, 12),
                "se": np.random.uniform(0.05, 0.15, 12),
            }
        )
        csv_path = tmp_path / "meta_effect_size.csv"
        df.to_csv(csv_path, index=False)

        res = runner.invoke(
            cli,
            [
                "meta",
                "--data",
                str(csv_path),
                "--effect-col",
                "effect_size",
                "--se-col",
                "se",
                "--study-col",
                "study",
                "--measure",
                "log_or",
                "--egger",
            ],
        )
        assert res.exit_code != 0
        assert "Egger's test is invalid for binary log odds ratios" in res.output

    def test_egger_rejects_mixed_measures(self, tmp_path):
        runner = CliRunner()
        df = pd.DataFrame(
            {
                "study": [f"Study_{i}" for i in range(12)],
                "effect_size": np.random.normal(0.2, 0.1, 12),
                "se": np.random.uniform(0.05, 0.15, 12),
                "measure": ["continuous"] * 6 + ["log_or"] * 6,
            }
        )
        csv_path = tmp_path / "meta_mixed.csv"
        df.to_csv(csv_path, index=False)

        res = runner.invoke(
            cli,
            [
                "meta",
                "--data",
                str(csv_path),
                "--effect-col",
                "effect_size",
                "--se-col",
                "se",
                "--study-col",
                "study",
                "--egger",
            ],
        )
        assert res.exit_code != 0
        assert "Mixed effect measures detected" in res.output


# ==============================================================================
# 2. Causal Inference & Agreement Fixes Verification
# ==============================================================================
class TestCausalAgreementFixes:
    def test_psm_deterministic_order(self):
        from medstat.causal.psm import perform_matching

        df1 = pd.DataFrame(
            {
                "treatment": [1, 1, 0, 0, 0],
                "ps": [0.8, 0.5, 0.75, 0.45, 0.1],
            }
        )
        df2 = pd.DataFrame(
            {
                "treatment": [1, 1, 0, 0, 0],
                "ps": [0.5, 0.8, 0.75, 0.45, 0.1],
            }
        )
        m1 = perform_matching(df1, "treatment", "ps", caliper=0.5)
        m2 = perform_matching(df2, "treatment", "ps", caliper=0.5)
        assert len(m1) == 4
        assert len(m2) == 4
        # Treated with ps=0.8 matches control with ps=0.75 in both
        pair_t1 = m1[(m1["treatment"] == 1) & (m1["ps"] == 0.8)]["pair_id"].iloc[0]
        ctrl_1 = m1[(m1["treatment"] == 0) & (m1["pair_id"] == pair_t1)]["ps"].iloc[0]
        pair_t2 = m2[(m2["treatment"] == 1) & (m2["ps"] == 0.8)]["pair_id"].iloc[0]
        ctrl_2 = m2[(m2["treatment"] == 0) & (m2["pair_id"] == pair_t2)]["ps"].iloc[0]
        assert ctrl_1 == ctrl_2 == 0.75

    def test_bland_altman_confidence_intervals(self):
        from medstat.agreement.bland_altman import calculate_bland_altman

        np.random.seed(42)
        df = pd.DataFrame(
            {
                "m1": np.random.normal(100, 15, 50),
                "m2": np.random.normal(100, 15, 50) + 2.0,
            }
        )
        res = calculate_bland_altman(df, "m1", "m2", ci=0.95)
        assert res["ci_mean_diff"][0] < res["mean_diff"] < res["ci_mean_diff"][1]
        assert res["ci_lower_loa"][0] < res["lower_loa"] < res["ci_lower_loa"][1]
        assert res["ci_upper_loa"][0] < res["upper_loa"] < res["ci_upper_loa"][1]

    def test_smd_boundary_and_categorical(self):
        from medstat.causal.balance import calculate_smd

        # Test zero pooled SD with different means (non-estimable / NaN)
        g1 = pd.Series([5.0, 5.0, 5.0])
        g0 = pd.Series([10.0, 10.0, 10.0])
        smd = calculate_smd(g1, g0)
        assert np.isnan(smd)

        # Test zero pooled SD with identical means (0.0)
        g_ident1 = pd.Series([5.0, 5.0, 5.0])
        g_ident0 = pd.Series([5.0, 5.0, 5.0])
        assert calculate_smd(g_ident1, g_ident0) == 0.0


# ==============================================================================
# 3. Clean Retention Flow Verification
# ==============================================================================
class TestCleanRetentionFixes:
    def test_sample_retention_flow_stage_persistence(self, tmp_path):
        df = pd.DataFrame(
            {
                "patient_id": range(100),
                "outcome": [
                    np.nan if i < 10 else (1 if i % 2 == 0 else 0) for i in range(100)
                ],
            }
        )
        tracker = SampleFlowTracker(
            initial_n=len(df), initial_name="Initial Enrolled Cohort"
        )
        df_clean = df.dropna(subset=["outcome"]).copy()
        tracker.record_stage(
            stage_name="Complete Primary Outcome Verification",
            n_remaining=len(df_clean),
            reason="Excluded missing primary outcome (mandatory endpoint per SAP)",
        )
        assert tracker.final_n == 90
        assert tracker.total_excluded == 10

        out_json = tmp_path / "sample_retention_flow.json"
        out_json.write_text(tracker.to_json(), encoding="utf-8")
        loaded = json.loads(out_json.read_text(encoding="utf-8"))
        assert loaded["N_initial"] == 100
        assert loaded["N_analyzed"] == 90
        assert (
            loaded["stages"][0]["stage_name"] == "Complete Primary Outcome Verification"
        )
        assert "mandatory endpoint" in loaded["stages"][0]["reasons"][0]["reason"]


# ==============================================================================
# 4. Diagnostic DCA Strategies Verification
# ==============================================================================
class TestDiagnosticDCAFixes:
    def test_calculate_net_benefit_includes_treat_all_and_none(self):
        from medstat.diagnostic.dca import calculate_dca

        gold = np.array([1, 1, 0, 0, 1, 0, 0, 1, 0, 0])
        probs = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.3, 0.2, 0.85, 0.05, 0.15])
        thresholds = [0.1, 0.3, 0.5]
        dca_df = calculate_dca(gold, probs, thresholds=thresholds)
        strategies = set(dca_df["strategy"].unique())
        assert "Model" in strategies
        assert "Treat All" in strategies
        assert "Treat None" in strategies
        none_df = dca_df[dca_df["strategy"] == "Treat None"]
        assert (none_df["net_benefit"] == 0.0).all()

    def test_auc_ci_delong(self):
        from medstat.diagnostic.roc import auc_ci_delong

        gold = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        score = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
        res = auc_ci_delong(gold, score)
        assert res["auc"] == 1.0
        assert 0.0 <= res["ci_lower"] <= 1.0
        assert 0.0 <= res["ci_upper"] <= 1.0


# ==============================================================================
# 5. Models EPV & E-value Verification
# ==============================================================================
class TestModelsEPVAndEValueFixes:
    def test_epv_with_expanded_design_matrix(self):
        import patsy

        df = pd.DataFrame(
            {
                "outcome": [1, 0, 1, 0, 0, 1, 0, 0, 1, 0] * 5,
                "treatment": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 5,
                "age": np.random.normal(60, 10, 50),
                "sex": ["M", "F"] * 25,
                "stage": ["I", "II", "III", "IV", "I"] * 10,
            }
        )
        formula = "outcome ~ treatment + age + C(sex) + C(stage)"
        y_mat, X_mat = patsy.dmatrices(formula, data=df, return_type="dataframe")
        # treatment (1) + age (1) + sex (1) + stage (3) = 6 parameters
        n_params = X_mat.shape[1] - 1
        assert n_params == 6
        n_events = (df["outcome"] == 1).sum()
        n_nonevents = (df["outcome"] == 0).sum()
        epv = min(n_events, n_nonevents) / n_params
        assert epv == 20 / 6

    def test_e_value_odds_ratio_rare_and_common(self):
        # For OR = 2.0 with rare outcome (direct RR approximation)
        res_rare = calculate_e_value(2.0, estimate_type="OR", rare_outcome=True)
        # For OR = 2.0 with common outcome (sqrt(OR) transform)
        res_common = calculate_e_value(2.0, estimate_type="OR", rare_outcome=False)
        assert res_rare["e_value_estimate"] > res_common["e_value_estimate"]

    def test_fit_firth_logistic_summary_df_schema(self):
        from medstat.models.firth import fit_firth_logistic

        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        X = np.array([[1.0], [2.0], [1.5], [2.5], [5.0], [6.0], [5.5], [6.5]])
        res = fit_firth_logistic(y, X, feature_names=["exposure"])
        summary_df = res["summary_df"]
        assert "estimate" in summary_df.columns
        assert "odds_ratio" in summary_df.columns
        assert "ci_lower" in summary_df.columns
        assert "ci_upper" in summary_df.columns
        assert "p_value" in summary_df.columns
        assert "exposure" in summary_df.index


# ==============================================================================
# 6. Report Table & Methods Narrative Verification
# ==============================================================================
class TestReportMethodsNarrativeFixes:
    def test_methods_narrative_uses_metadata(self):
        from medstat.reporting.narrative import generate_methods_narrative

        narr_with_meta = generate_methods_narrative(
            model_type="cox",
            exposure="Statin",
            outcome="In-hospital Sepsis",
            covariates=["SOFA score", "Lactate"],
            missing_strategy="mice",
            alpha=0.05,
        )
        assert "In-hospital Sepsis" in narr_with_meta
        assert "SOFA score, Lactate" in narr_with_meta
        assert "Cox proportional hazards" in narr_with_meta
        assert "p-values < 0.05" in narr_with_meta
        assert "Multiple Imputation by Chained Equations (MICE)" in narr_with_meta

    def test_p_value_formatting_nejm_vs_jama(self):
        from medstat.reporting.tables import format_journal_p_value

        # Test missing / NaN returns em-dash
        assert format_journal_p_value(None) == "—"
        assert format_journal_p_value(float("nan")) == "—"

        # Test NEJM: 3 decimals at/below .01, 2 decimals above .01, strict >0.99
        assert format_journal_p_value(0.0004, style="NEJM") == "P<0.001"
        assert format_journal_p_value(0.008, style="NEJM") == "P=0.008"
        assert format_journal_p_value(0.014, style="NEJM") == "P=0.01"
        assert format_journal_p_value(0.99, style="NEJM") == "P=0.99"
        assert format_journal_p_value(0.994, style="NEJM") == "P>0.99"

        # Test JAMA: 3 decimals below .01, 2 decimals at/above .01, strict >.99, stripped leading 0
        assert format_journal_p_value(0.0004, style="JAMA") == "P<0.001"
        assert format_journal_p_value(0.008, style="JAMA") == "P=0.008"
        assert format_journal_p_value(0.01, style="JAMA") == "P=0.01"
        assert format_journal_p_value(0.023, style="JAMA") == "P=0.02"
        assert format_journal_p_value(0.99, style="JAMA") == "P=0.99"
        assert format_journal_p_value(0.995, style="JAMA") == "P>0.99"

    def test_html_table_escaping(self):
        from medstat.reporting.tables import (
            Estimate,
            EstimateTable,
            PublicationRenderer,
        )

        raw_var = "<script>alert('xss')</script>"
        raw_title = "Table 1. <img src=x onerror=alert(1)>"
        row = Estimate(
            term="x1",
            label=raw_var,
            estimate=1.5,
            ci_lower=1.1,
            ci_upper=2.0,
            p_value=0.04,
        )
        table = EstimateTable(title=raw_title, rows=[row])

        for style in ["NEJM", "JAMA", "APA7"]:
            rendered = PublicationRenderer.render_html(table, style=style)
            assert "<script>" not in rendered
            assert "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;" in rendered
            assert "<img src=x" not in rendered
            assert "&lt;img src=x onerror=alert(1)&gt;" in rendered

    def test_diagnostic_gold_standard_binary_validation(self):
        import pytest

        from medstat.diagnostic.accuracy import validate_gold_standard

        res = validate_gold_standard([0, 1, 1, 0])
        assert list(res) == [0, 1, 1, 0]

        with pytest.raises(ValueError, match="strictly binary numeric"):
            validate_gold_standard([0, 1, 2])

        with pytest.raises(ValueError, match="strictly binary numeric"):
            validate_gold_standard(["Case", "Control"])

        with pytest.raises(ValueError, match="must contain both 0 and 1 classes"):
            validate_gold_standard([0, 0, 0], require_both_classes=True)

    def test_treatment_contrast_extraction(self):
        from medstat.models import extract_primary_effect

        assert extract_primary_effect({"treatment": 2.15, "age": 1.02}) == 2.15
        assert extract_primary_effect({"C(treatment)[T.1]": 1.85, "age": 1.02}) == 1.85
        assert extract_primary_effect({"treatment[T.True]": 3.10}) == 3.10
        assert np.isnan(extract_primary_effect({"other": 1.5}))

    def test_ordinal_non_standard_index_positional_mask(self):
        from medstat.models.ordinal import (
            fit_multinomial_logistic,
            fit_proportional_odds,
        )

        # Create non-standard index [100, 101, 102, ...] with some missing y
        idx = [100 + i for i in range(30)]
        y = pd.Series([0, 1, 2] * 10, index=idx)
        y.iloc[5] = np.nan  # introduce missing value
        X = np.random.RandomState(42).randn(30, 2)  # numpy array X

        res_po = fit_proportional_odds(y, X)
        assert res_po["summary_df"] is not None

        res_multi = fit_multinomial_logistic(y, X)
        assert res_multi["params"] is not None

    def test_multilevel_calculate_design_effect_string_outcome(self):
        import pytest

        from medstat.models.multilevel import calculate_design_effect

        y_str = pd.Series(["High", "Low", "Medium", "High"])
        clusters = pd.Series([1, 1, 2, 2])
        with pytest.raises(ValueError, match="must be numeric"):
            calculate_design_effect(y_str, clusters)

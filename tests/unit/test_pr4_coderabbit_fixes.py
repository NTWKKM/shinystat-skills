"""
tests/unit/test_pr4_coderabbit_fixes.py: Verification of CodeRabbit review fixes in PR#4.

Tests:
1. Kappa: non-null SE vs null SE, equal rater count validation in Fleiss, continuous rating rejection, no explicit rater ValueError.
2. Mediation: binary outcome log-odds scale, nonbinary linear scale, no silent OLS fallback.
3. Splines: logistic RCS contrast_df at ref_value, masked NaN OR for basis and intercept terms.
4. Tables: non-finite SMD handling in render_balance_table.
5. Loader: CSV/TSV encoding fallback order (utf-8, utf-8-sig, latin1, cp1252).
6. CLI:
   - clean: outlier-action flag, exclude binary/low-cardinality from outliers.
   - diag: DCA/calibration orientation under direction low, probability scale enforcement.
   - sample-size: proportions requiring p1/p2, survival requiring HR, correlation requiring r.
"""

from __future__ import annotations

import json
import math

import click
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from medstat.agreement.kappa import (
    calculate_kappa,
    cohens_kappa,
    fleiss_kappa,
    validate_categorical_ratings,
)
from medstat.causal.mediation import run_mediation
from medstat.cli.main import cli
from medstat.data.clean import _is_id_column, audit_missingness
from medstat.data.loader import load_clinical_data
from medstat.models.splines import fit_logistic_rcs
from medstat.reporting.checklists import get_prisma_checklist, get_stard_checklist
from medstat.reporting.narrative import generate_methods_narrative
from medstat.reporting.tables import render_balance_table, render_diagnostic_table


# ==============================================================================
# 1. Kappa Verification
# ==============================================================================
class TestKappaFixes:
    def test_cohens_kappa_null_and_non_null_se(self):
        # 50 subjects, 2 raters, 3 categories
        np.random.seed(42)
        r1 = np.random.choice([0, 1, 2], size=100, p=[0.5, 0.3, 0.2])
        r2 = r1.copy()
        # introduce slight disagreement
        r2[:15] = (r2[:15] + 1) % 3

        # Unweighted
        res_unweighted = cohens_kappa(r1, r2)
        assert "se" in res_unweighted
        assert "se_null" in res_unweighted
        assert res_unweighted["se"] > 0
        assert res_unweighted["se_null"] > 0
        assert (
            res_unweighted["ci_lower"]
            <= res_unweighted["kappa"]
            <= res_unweighted["ci_upper"]
        )

        # Weighted (linear and quadratic)
        res_linear = cohens_kappa(r1, r2, weights="linear")
        assert res_linear["se"] > 0
        assert res_linear["se_null"] > 0

        res_quad = cohens_kappa(r1, r2, weights="quadratic")
        assert res_quad["se"] > 0
        assert res_quad["se_null"] > 0

    def test_fleiss_kappa_unequal_ratings_per_subject_rejected(self):
        # Subject 0 has 3 ratings, Subject 1 has 2 ratings
        mat = np.array(
            [
                [2, 1, 0],
                [1, 1, 0],  # only 2 ratings
            ]
        )
        with pytest.raises(ValueError, match="same number of ratings"):
            fleiss_kappa(mat)

    def test_calculate_kappa_no_explicit_raters_raises_value_error(self):
        df = pd.DataFrame(
            {
                "patient_id": [1, 2, 3],
                "age": [50, 60, 70],
                "gender": ["M", "F", "M"],
            }
        )
        with pytest.raises(ValueError, match="No explicit rater columns resolved"):
            calculate_kappa(df)

    def test_calculate_kappa_continuous_ratings_rejected(self):
        # Long format with continuous ratings
        df_long = pd.DataFrame(
            {
                "subject_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "rater_id": ["A", "B", "C", "A", "B", "C", "A", "B", "C"],
                "score": [12.4, 15.6, 11.2, 25.1, 24.8, 26.2, 8.5, 9.1, 8.9],
            }
        )
        with pytest.raises(ValueError, match="Continuous ratings are not supported"):
            calculate_kappa(df_long)

    def test_calculate_kappa_two_raters_continuous_rejected(self):
        # Two explicit columns with continuous ratings
        df = pd.DataFrame(
            {
                "r1": [12.4, 15.6, 11.2, 25.1, 24.8, 26.2, 8.5, 9.1, 8.9],
                "r2": [12.1, 15.8, 11.0, 24.9, 24.5, 26.0, 8.7, 9.0, 9.1],
            }
        )
        with pytest.raises(ValueError, match="Continuous ratings are not supported"):
            calculate_kappa(df, rater1="r1", rater2="r2")

        # Two-rater long format with continuous ratings
        df_2rater_long = pd.DataFrame(
            {
                "subject_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                "rater_id": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
                "score": [12.4, 12.1, 15.6, 15.8, 11.2, 11.0, 25.1, 24.9, 8.5, 8.7],
            }
        )
        with pytest.raises(ValueError, match="Continuous ratings are not supported"):
            calculate_kappa(df_2rater_long)

        # Direct call to cohens_kappa with continuous ratings
        with pytest.raises(ValueError, match="Continuous ratings are not supported"):
            cohens_kappa(df["r1"], df["r2"])

    def test_calculate_kappa_valid_discrete_ratings_accepted(self):
        df_long = pd.DataFrame(
            {
                "subject_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "rater_id": ["A", "B", "C", "A", "B", "C", "A", "B", "C"],
                "score": [1, 1, 1, 2, 2, 3, 1, 1, 2],
            }
        )
        res = calculate_kappa(df_long)
        assert res["type"] == "fleiss"
        assert "kappa" in res

    def test_calculate_kappa_fractional_discrete_ratings_accepted(self):
        # Clinical Dementia Rating (CDR) scale: 0, 0.5, 1.0, 2.0, 3.0
        np.random.seed(42)
        n = 60
        r1 = np.random.choice([0.0, 0.5, 1.0, 2.0, 3.0], size=n)
        r2 = r1.copy()
        r2[:10] = np.random.choice([0.0, 0.5, 1.0, 2.0, 3.0], size=10)
        df = pd.DataFrame({"r1": r1, "r2": r2})
        res = calculate_kappa(df, rater1="r1", rater2="r2")
        assert res["type"] == "cohen"
        assert "kappa" in res

    def test_calculate_kappa_many_categories_discrete_accepted(self):
        # 12 discrete integer categories across 100 subjects
        np.random.seed(42)
        n = 100
        r1 = np.random.choice(range(1, 13), size=n)
        r2 = r1.copy()
        r2[:15] = np.random.choice(range(1, 13), size=15)
        df = pd.DataFrame({"r1": r1, "r2": r2})
        res = calculate_kappa(df, rater1="r1", rater2="r2")
        assert res["type"] == "cohen"
        assert "kappa" in res

    def test_calculate_kappa_fractional_scale_more_than_15_levels_accepted(self):
        # 18 valid fractional levels (0.5 to 9.0 in steps of 0.5) across 150 subjects
        np.random.seed(42)
        n = 150
        levels = [0.5 * i for i in range(1, 19)]  # 18 levels
        r1 = np.random.choice(levels, size=n)
        r2 = r1.copy()
        r2[:25] = np.random.choice(levels, size=25)
        df = pd.DataFrame({"r1": r1, "r2": r2})

        # Accepted via repeating observations contract
        res = calculate_kappa(df, rater1="r1", rater2="r2")
        assert res["type"] == "cohen"
        assert len(res["categories"]) == 18
        assert "kappa" in res

        # Accepted via declared categories contract
        res_cat = calculate_kappa(df, rater1="r1", rater2="r2", categories=levels)
        assert res_cat["type"] == "cohen"
        assert len(res_cat["categories"]) == 18

    def test_calculate_kappa_declared_categories_contract_rejects_unrecognized(self):
        df = pd.DataFrame({"r1": [1, 2, 3], "r2": [1, 2, 99]})
        with pytest.raises(ValueError, match="not in declared categories"):
            calculate_kappa(df, rater1="r1", rater2="r2", categories=[1, 2, 3])


# ==============================================================================
# 2. Mediation Verification
# ==============================================================================
class TestMediationFixes:
    def test_binary_outcome_mediation_scale_and_no_ols_fallback(self):
        np.random.seed(42)
        n = 150
        trt = np.random.binomial(1, 0.5, size=n)
        cov = np.random.normal(0, 1, size=n)
        # Continuous mediator
        med = 1.2 * trt + 0.5 * cov + np.random.normal(0, 1, size=n)
        # Binary outcome (0/1)
        logits = 0.8 * trt + 1.0 * med - 0.5 * cov
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = np.random.binomial(1, probs)

        df = pd.DataFrame({"trt": trt, "med": med, "y": y, "cov": cov})
        res = run_mediation(
            df, treatment="trt", mediator="med", outcome="y", covariates=["cov"]
        )

        assert res["scale"] == "log_odds"
        assert res["effect_scale"] == "log_odds"
        assert "Log-Odds" in res["method"]
        assert "acme" in res
        assert "ade" in res

    def test_binary_outcome_mediation_logit_failure_raises(self, monkeypatch):
        # Force Logit fit to fail and verify that it raises ValueError without falling back to OLS
        from statsmodels.discrete.discrete_model import Logit

        def mock_fit(*args, **kwargs):
            raise ValueError("Forced singular matrix in Logit fit")

        monkeypatch.setattr(Logit, "fit", mock_fit)

        df = pd.DataFrame(
            {
                "trt": [0, 1] * 10,
                "med": [1.0, 2.0, 1.5, 2.5] * 5,
                "y": [0, 1] * 10,
            }
        )
        with pytest.raises(ValueError, match="Failed to fit logistic outcome model"):
            run_mediation(df, treatment="trt", mediator="med", outcome="y")

    def test_continuous_outcome_mediation_scale(self):
        np.random.seed(42)
        n = 100
        trt = np.random.binomial(1, 0.5, size=n)
        med = 0.8 * trt + np.random.normal(0, 1, size=n)
        y = 0.5 * trt + 1.2 * med + np.random.normal(0, 1, size=n)

        df = pd.DataFrame({"trt": trt, "med": med, "y": y})
        res = run_mediation(df, treatment="trt", mediator="med", outcome="y")

        assert res["scale"] == "linear"
        assert "Linear" in res["method"]


# ==============================================================================
# 3. Splines Verification
# ==============================================================================
class TestSplinesFixes:
    def test_fit_logistic_rcs_basis_terms_masked_and_contrast_df(self):
        np.random.seed(42)
        n = 200
        age = np.linspace(30, 80, n)
        prob = 1.0 / (
            1.0 + np.exp(-(-3.0 + 0.08 * (age - 50) + 0.001 * (age - 50) ** 2))
        )
        outcome = np.random.binomial(1, prob)
        covar = np.random.normal(0, 1, n)

        df = pd.DataFrame({"outcome": outcome, "age": age, "covar": covar})
        ref_val = 55.0
        res = fit_logistic_rcs(
            df,
            outcome="outcome",
            spline_var="age",
            covariates=["covar"],
            ref_value=ref_val,
        )

        sum_df = res["summary_df"]
        # Basis terms and Intercept must have NaN for OR and CIs
        for idx in sum_df.index:
            if "cr(" in str(idx) or "Intercept" in str(idx):
                assert math.isnan(sum_df.loc[idx, "odds_ratio"])
                assert math.isnan(sum_df.loc[idx, "ci_lower"])
                assert math.isnan(sum_df.loc[idx, "ci_upper"])
            else:
                # Covar should have valid OR
                assert not math.isnan(sum_df.loc[idx, "odds_ratio"])

        # Contrast df must be present
        assert "contrast_df" in res
        cdf = res["contrast_df"]
        assert "OR" in cdf.columns
        assert "OR_lower" in cdf.columns
        assert "OR_upper" in cdf.columns

        # At ref_value, OR should be 1.0
        closest_row = cdf.iloc[(cdf["age"] - ref_val).abs().argsort()[:1]].iloc[0]
        assert np.isclose(closest_row["OR"], 1.0, atol=0.05)


# ==============================================================================
# 4. Tables Non-Finite SMD Verification
# ==============================================================================
class TestTablesFixes:
    def test_render_balance_table_non_finite_smd(self):
        bal_data = {
            "covariates": ["age", "bmi", "ldl"],
            "smd_pre": [0.35, float("nan"), 0.15],
            "smd_post": [0.04, float("nan"), float("inf")],
        }
        html_table = render_balance_table("Covariate Balance", bal_data)
        # NaN and inf must be formatted as "—" and not classified as Imbalanced when missing
        assert "—" in html_table
        assert "nan" not in html_table.lower()


# ==============================================================================
# 5. Loader Encoding Fallback Verification
# ==============================================================================
class TestLoaderEncodings:
    def test_csv_tsv_encoding_fallbacks(self, tmp_path):
        # Create CSV files in different encodings
        content = "patient_id,score\nPAT_001,10.5\nPAT_002,15.2\n"

        # UTF-8 with BOM (utf-8-sig)
        p_sig = tmp_path / "test_sig.csv"
        p_sig.write_bytes(content.encode("utf-8-sig"))
        df_sig = load_clinical_data(str(p_sig))
        assert len(df_sig) == 2

        # Latin1 / ISO-8859-1
        p_latin = tmp_path / "test_latin1.csv"
        p_latin.write_bytes("patient_id,médecine\nPAT_001,10\n".encode("latin1"))
        df_latin = load_clinical_data(str(p_latin))
        assert len(df_latin) == 1

        # TSV with Latin1
        p_tsv = tmp_path / "test_latin1.tsv"
        p_tsv.write_bytes("patient_id\tvaleur\nPAT_001\t20\n".encode("latin1"))
        df_tsv = load_clinical_data(str(p_tsv))
        assert len(df_tsv) == 1

        # Windows-1252 with smart quotes and em-dash (0x93, 0x94, 0x97)
        # These are C1 control codes in Latin-1, but valid punctuation in CP1252
        p_cp1252 = tmp_path / "test_cp1252.csv"
        p_cp1252.write_bytes(
            "patient_id,note\nPAT_001,“mild” syndrome — stable\n".encode("cp1252")
        )
        df_cp = load_clinical_data(str(p_cp1252))
        assert len(df_cp) == 1
        assert "“mild” syndrome — stable" in df_cp["note"].values[0]

        # Explicit encoding parameter
        df_cp_exp = load_clinical_data(str(p_cp1252), encoding="cp1252")
        assert len(df_cp_exp) == 1


# ==============================================================================
# 6. CLI Verification
# ==============================================================================
class TestCLIFixes:
    def test_cli_clean_outlier_flag_and_binary_exclusion(self, tmp_path):
        runner = CliRunner()
        # Dataset with binary column and continuous column with an outlier
        df = pd.DataFrame(
            {
                "patient_id": [f"P_{i}" for i in range(50)],
                "is_dead": [0] * 48 + [1, 1],  # binary column with 2 unique values
                "sbp": [115.0 + (i % 10) for i in range(48)]
                + [260.0, 270.0],  # continuous column with 2 outliers
            }
        )
        csv_path = tmp_path / "test_outliers.csv"
        df.to_csv(csv_path, index=False)
        out_csv = tmp_path / "clean_flag.csv"
        audit_json = tmp_path / "audit_out.json"

        res = runner.invoke(
            cli,
            [
                "clean",
                "--data",
                str(csv_path),
                "--strategy",
                "complete-case",
                "--outlier-action",
                "flag",
                "--output",
                str(out_csv),
                "--audit-out",
                str(audit_json),
            ],
        )
        assert res.exit_code == 0
        assert audit_json.exists()
        audit_data = json.loads(audit_json.read_text())
        assert "outlier_counts" in audit_data
        # Binary column should NOT be in outlier_counts
        assert "is_dead" not in audit_data["outlier_counts"]
        # SBP should have recorded outliers
        assert "sbp" in audit_data["outlier_counts"]
        assert audit_data["outlier_counts"]["sbp"] == 2

    def test_cli_diag_direction_and_probabilities(self, tmp_path):
        runner = CliRunner()
        np.random.seed(42)
        n = 100
        outcome = np.random.binomial(1, 0.4, size=n)
        # Probabilities in [0, 1]
        probs = np.clip(0.3 * outcome + np.random.uniform(0.1, 0.6, size=n), 0.05, 0.95)
        # Continuous biomarker outside [0, 1] (e.g. lactate 1.0 to 7.0)
        lactate = 1.5 + 2.5 * outcome + np.random.normal(0, 0.8, size=n)
        df = pd.DataFrame({"outcome": outcome, "probs": probs, "lactate": lactate})
        csv_path = tmp_path / "diag_data.csv"
        df.to_csv(csv_path, index=False)

        # 1. Probabilities in [0, 1] with direction low
        out_json = tmp_path / "diag_prob_out.json"
        res_prob = runner.invoke(
            cli,
            [
                "diag",
                "--data",
                str(csv_path),
                "--gold-standard",
                "outcome",
                "--test",
                "probs",
                "--direction",
                "low",
                "--dca",
                "--calibration",
                "--output",
                str(out_json),
            ],
        )
        assert res_prob.exit_code == 0
        assert out_json.exists()
        res_data = json.loads(out_json.read_text())
        assert "calibration" in res_data
        brier_low = res_data["calibration"]["brier"]["brier_score"]
        # Inverted risk probabilities under direction low must be (1.0 - probs)
        expected_brier = float(np.mean((outcome - (1.0 - probs)) ** 2))
        counterfactual_brier = float(np.mean((outcome - probs) ** 2))
        assert np.isclose(brier_low, expected_brier, atol=1e-5)
        assert not np.isclose(brier_low, counterfactual_brier, atol=1e-3)

        # 2. Continuous biomarker score requiring logistic risk calibration
        out_bio_json = tmp_path / "diag_bio_out.json"
        res_bio = runner.invoke(
            cli,
            [
                "diag",
                "--data",
                str(csv_path),
                "--gold-standard",
                "outcome",
                "--test",
                "lactate",
                "--direction",
                "high",
                "--dca",
                "--calibration",
                "--output",
                str(out_bio_json),
            ],
        )
        assert res_bio.exit_code == 0
        assert out_bio_json.exists()
        bio_data = json.loads(out_bio_json.read_text())
        assert "dca" in bio_data
        assert "calibration" in bio_data

    def test_cli_diag_uncalibratable_score_raises_click_exception(self, tmp_path):
        runner = CliRunner()
        # Dataset with constant score outside [0, 1], causing Logit fit to fail
        df = pd.DataFrame(
            {
                "outcome": [0, 1, 0, 1, 0, 1, 0, 1],
                "bad_score": [5.0] * 8,  # Constant score > 1.0 cannot be calibrated
            }
        )
        csv_path = tmp_path / "bad_diag.csv"
        df.to_csv(csv_path, index=False)

        res = runner.invoke(
            cli,
            [
                "diag",
                "--data",
                str(csv_path),
                "--gold-standard",
                "outcome",
                "--test",
                "bad_score",
                "--dca",
                "--calibration",
            ],
        )
        assert res.exit_code != 0
        assert (
            "Cannot calibrate continuous score" in res.output
            or "fitting a logistic risk calibration model failed" in res.output
        )

    def test_cli_diag_logistic_fit_failure_on_varying_score_raises_click_exception(
        self, tmp_path, monkeypatch
    ):
        runner = CliRunner()
        from statsmodels.discrete.discrete_model import Logit

        def mock_fit(*args, **kwargs):
            raise ValueError("Forced Hessian inversion failure in Logit")

        monkeypatch.setattr(Logit, "fit", mock_fit)

        # Dataset with varying score (passes variance check, but Logit fitting fails)
        df = pd.DataFrame(
            {
                "outcome": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                "lactate": [1.5, 2.5, 3.2, 4.1, 5.0, 1.8, 2.9, 3.8, 4.5, 5.2],
            }
        )
        csv_path = tmp_path / "varying_diag.csv"
        df.to_csv(csv_path, index=False)

        res = runner.invoke(
            cli,
            [
                "diag",
                "--data",
                str(csv_path),
                "--gold-standard",
                "outcome",
                "--test",
                "lactate",
                "--dca",
                "--calibration",
            ],
        )
        assert res.exit_code != 0
        assert "fitting a logistic risk calibration model failed" in res.output
        assert "Forced Hessian inversion failure in Logit" in res.output

    def test_cli_sample_size_proportions_requires_p1_p2(self):
        runner = CliRunner()
        res = runner.invoke(cli, ["sample-size", "--type", "proportions"])
        assert res.exit_code != 0
        assert "requires explicitly supplied --p1 and --p2" in res.output

        res_ok = runner.invoke(
            cli,
            ["sample-size", "--type", "proportions", "--p1", "0.20", "--p2", "0.35"],
        )
        assert res_ok.exit_code == 0
        assert "sample_size_per_group" in res_ok.output

    def test_cli_sample_size_survival_requires_hr(self):
        runner = CliRunner()
        res = runner.invoke(cli, ["sample-size", "--type", "survival"])
        assert res.exit_code != 0
        assert "requires explicitly supplied --hazard-ratio" in res.output

        res_no_p = runner.invoke(
            cli, ["sample-size", "--type", "survival", "--hr", "0.70"]
        )
        assert res_no_p.exit_code != 0
        assert "event probability" in res_no_p.output.lower()

        res_ok = runner.invoke(
            cli,
            [
                "sample-size",
                "--type",
                "survival",
                "--hr",
                "0.70",
                "--event-probability",
                "0.50",
            ],
        )
        assert res_ok.exit_code == 0
        assert "required_events" in res_ok.output

    def test_cli_sample_size_correlation_requires_r(self):
        runner = CliRunner()
        res = runner.invoke(cli, ["sample-size", "--type", "correlation"])
        assert res.exit_code != 0
        assert "requires explicitly supplied --r" in res.output

        res_ok = runner.invoke(
            cli, ["sample-size", "--type", "correlation", "--r", "0.35"]
        )
        assert res_ok.exit_code == 0
        assert "correlation_r" in res_ok.output

    def test_cli_sample_size_ttest_requires_positive_effect_size(self):
        runner = CliRunner()
        # Missing effect size
        res_missing = runner.invoke(cli, ["sample-size", "--type", "t-test"])
        assert res_missing.exit_code != 0
        assert (
            "requires an explicitly supplied positive --effect-size"
            in res_missing.output
        )

        # Non-positive effect size
        res_neg = runner.invoke(
            cli, ["sample-size", "--type", "t-test", "--effect-size", "-0.5"]
        )
        assert res_neg.exit_code != 0
        assert (
            "requires an explicitly supplied positive --effect-size" in res_neg.output
        )

        # Valid positive effect size
        res_ok = runner.invoke(
            cli, ["sample-size", "--type", "t-test", "--effect-size", "0.6"]
        )
        assert res_ok.exit_code == 0
        assert "sample_size_per_group" in res_ok.output


# ==============================================================================
# 7. Additional Rigorous Verification for PR#4 Follow-up Fixes
# ==============================================================================
class TestPR4FollowupFixes:
    def test_validate_categorical_ratings_declared_categories_on_categorical_dtype(
        self,
    ):
        # Categorical series containing values outside declared categories
        s = pd.Series(["A", "B", "C"], dtype="category")
        with pytest.raises(ValueError, match="not in declared categories"):
            validate_categorical_ratings(s, categories=["A", "B"])

        # Categorical series within declared categories
        s_ok = pd.Series(["A", "B", "A"], dtype="category")
        validate_categorical_ratings(s_ok, categories=["A", "B"])

    def test_cohens_kappa_category_union_and_error_on_unmapped(self):
        # Two categorical series with disjoint observed categories
        s1 = pd.Series(pd.Categorical(["A", "B", "A"], categories=["A", "B"]))
        s2 = pd.Series(pd.Categorical(["B", "C", "B"], categories=["B", "C"]))
        res = cohens_kappa(s1, s2)
        assert "A" in res["categories"]
        assert "B" in res["categories"]
        assert "C" in res["categories"]

        # Raising error if rating pair excluded
        with pytest.raises(
            ValueError, match="not in declared categories|excluded from confusion count"
        ):
            cohens_kappa(
                pd.Series(["A", "B", "UNKNOWN"]),
                pd.Series(["A", "B", "A"]),
                categories=["A", "B"],
            )

    def test_fleiss_kappa_returns_se_null_and_none_ci(self):
        mat = np.array(
            [
                [3, 0, 0],
                [2, 1, 0],
                [0, 2, 1],
                [0, 0, 3],
            ]
        )
        res = fleiss_kappa(mat)
        assert "se_null" in res
        assert res["se_null"] > 0
        assert res["ci_lower"] is None
        assert res["ci_upper"] is None
        assert "ci_note" in res
        assert (
            "null-hypothesis standard error is valid for hypothesis testing"
            in res["ci_note"]
        )

    def test_cli_profile_mcar_interpretation_wording(self, tmp_path):
        runner = CliRunner()
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "patient_id": list(range(1, 101)),
                "age": np.random.normal(55, 10, 100),
                "sbp": np.random.normal(120, 15, 100),
            }
        )
        df.loc[1:10, "sbp"] = np.nan
        csv_path = tmp_path / "mcar_cohort.csv"
        df.to_csv(csv_path, index=False)
        out_json = tmp_path / "profile.json"

        res = runner.invoke(
            cli, ["profile", "--data", str(csv_path), "--output", str(out_json)]
        )
        assert res.exit_code == 0
        assert out_json.exists()
        prof_data = json.loads(out_json.read_text())
        mcar_interp = prof_data["littles_mcar"]["interpretation"]
        assert (
            "No evidence against MCAR (P > 0.05)" in mcar_interp
            or "Evidence against MCAR (P <= 0.05)" in mcar_interp
        )
        assert "MICE recommended" not in mcar_interp
        assert mcar_interp in res.output
        assert "(MCAR)" not in res.output
        assert "(Non-MCAR)" not in res.output

    def test_cli_clean_imputed_datasets_rejects_outlier_modifications(self, tmp_path):
        runner = CliRunner()
        df = pd.DataFrame(
            {
                "patient_id": list(range(1, 61)),
                "age": [50.0] * 50 + [np.nan] * 10,
                "sbp": [120.0] * 50 + [130.0] * 10,
            }
        )
        csv_path = tmp_path / "mice_outlier.csv"
        df.to_csv(csv_path, index=False)

        out_csv = tmp_path / "clean_mice.csv"
        # remove should fail when imputed_datasets present
        res_remove = runner.invoke(
            cli,
            [
                "clean",
                "--data",
                str(csv_path),
                "--strategy",
                "mice",
                "--imputations",
                "3",
                "--missing-justification",
                "Clinical test for MICE",
                "--outlier-action",
                "remove",
                "--output",
                str(out_csv),
            ],
        )
        assert res_remove.exit_code != 0
        assert (
            "cannot be applied post-imputation when multiple imputed datasets exist"
            in res_remove.output
        )

        # flag should succeed
        res_flag = runner.invoke(
            cli,
            [
                "clean",
                "--data",
                str(csv_path),
                "--strategy",
                "mice",
                "--imputations",
                "3",
                "--missing-justification",
                "Clinical test for MICE",
                "--outlier-action",
                "flag",
                "--output",
                str(out_csv),
            ],
        )
        assert res_flag.exit_code == 0

    def test_cli_clean_outlier_removal_uses_sample_flow_tracker(self, tmp_path):
        runner = CliRunner()
        df = pd.DataFrame(
            {
                "patient_id": list(range(1, 21)),
                "age": [45.0 + i for i in range(18)] + [150.0, 160.0],
            }
        )
        csv_path = tmp_path / "outlier_removal.csv"
        df.to_csv(csv_path, index=False)
        out_csv = tmp_path / "cleaned.csv"
        audit_json = tmp_path / "audit.json"

        res = runner.invoke(
            cli,
            [
                "clean",
                "--data",
                str(csv_path),
                "--strategy",
                "complete-case",
                "--missing-justification",
                "No missing values",
                "--outlier-action",
                "remove",
                "--outlier-cols",
                "age",
                "--output",
                str(out_csv),
                "--audit-out",
                str(audit_json),
            ],
        )
        assert res.exit_code == 0
        assert out_csv.exists()
        cleaned_data = pd.read_csv(out_csv)
        assert len(cleaned_data) == 18

    def test_cli_diag_insample_logistic_recalibration_labels(self, tmp_path):
        runner = CliRunner()
        df = pd.DataFrame(
            {
                "outcome": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                "lactate": [1.5, 2.5, 3.2, 4.1, 5.0, 1.8, 2.9, 3.8, 4.5, 5.2],
            }
        )
        csv_path = tmp_path / "diag_recal.csv"
        df.to_csv(csv_path, index=False)
        out_json = tmp_path / "diag_out.json"

        res = runner.invoke(
            cli,
            [
                "diag",
                "--data",
                str(csv_path),
                "--gold-standard",
                "outcome",
                "--test",
                "lactate",
                "--calibration",
                "--output",
                str(out_json),
            ],
        )
        assert res.exit_code == 0
        diag_res = json.loads(out_json.read_text())
        assert diag_res["probability_source"] == "in-sample logistic recalibration"
        assert "apparent estimates" in diag_res["apparent_estimates_note"].lower()
        cal = diag_res["calibration"]
        assert cal["slope_and_intercept"]["slope"] is None
        assert cal["slope_and_intercept"]["intercept"] is None
        assert "not informative" in cal["slope_and_intercept"]["note"].lower()

    def test_audit_missingness_excludes_binary_and_id_columns(self):
        df = pd.DataFrame(
            {
                "patient_id": list(range(1, 51)),
                "is_female": [0, 1] * 25,
                "age": [50.0 + i for i in range(50)],
                "sbp": [120.0 + i for i in range(40)] + [np.nan] * 10,
            }
        )
        audit = audit_missingness(df)
        assert audit.littles_mcar is not None
        # Should only evaluate continuous non-ID columns: age, sbp (2 columns)
        assert audit.littles_mcar.n_variables == 2

    def test_loader_rejects_legacy_xls(self, tmp_path):
        fake_xls = tmp_path / "legacy.xls"
        fake_xls.write_bytes(b"\xd0\xcf\x11\xe0")  # OLE CF header
        with pytest.raises(click.ClickException, match="Legacy Excel format"):
            load_clinical_data(fake_xls)

    def test_stard_and_prisma_checklists_guideline_alignment(self):
        stard = get_stard_checklist()
        distinct_stard_numbers = {it.number.rstrip("abcdef") for it in stard.items}
        assert len(distinct_stard_numbers) == 30
        assert len(stard.items) == 34
        item_names = [it.item for it in stard.items]
        assert "Decision curve analysis" not in item_names

        prisma = get_prisma_checklist()
        distinct_prisma_numbers = {it.number.rstrip("abcdef") for it in prisma.items}
        assert len(distinct_prisma_numbers) == 27
        assert len(prisma.items) == 42
        item_names_prisma = [it.item for it in prisma.items]
        assert "Results of individual studies" in item_names_prisma
        assert "Results of syntheses - statistical results" in item_names_prisma
        prisma_dict = {it.number: it.item for it in prisma.items}
        assert prisma_dict["16a"] == "Study selection - results"
        assert prisma_dict["16b"] == "Study selection - excluded studies"

    def test_render_diagnostic_table_nested_and_scalar_brier(self):
        # 1. Nested brier dict
        nested_diag = {
            "calibration": {
                "brier": {"brier_score": 0.1234, "null_brier": 0.25},
            }
        }
        html_nested = render_diagnostic_table("Diagnostic Report", nested_diag)
        assert "0.1234" in html_nested

        # 2. Scalar brier
        scalar_diag = {
            "calibration": {
                "brier": 0.0567,
            }
        }
        html_scalar = render_diagnostic_table("Diagnostic Report", scalar_diag)
        assert "0.0567" in html_scalar

        # 3. Missing brier_score in dict should be skipped gracefully
        missing_diag = {
            "calibration": {
                "brier": {"other_key": 123},
            }
        }
        html_missing = render_diagnostic_table("Diagnostic Report", missing_diag)
        assert "Brier Score" not in html_missing


class TestCoderabbitReviewRound2Fixes:
    def test_loader_rejects_unsupported_extensions(self, tmp_path):
        json_file = tmp_path / "cohort.json"
        json_file.write_text('{"patient_id": [1, 2], "age": [50, 60]}')
        with pytest.raises(
            click.ClickException, match="Unsupported file format '.json'"
        ):
            load_clinical_data(json_file)

        xyz_file = tmp_path / "cohort.xyz"
        xyz_file.write_text("a,b\n1,2")
        with pytest.raises(
            click.ClickException, match="Unsupported file format '.xyz'"
        ):
            load_clinical_data(xyz_file)

    def test_cohens_kappa_degenerate_single_category(self):
        r1 = [1, 1, 1, 1]
        r2 = [1, 1, 1, 1]
        res = cohens_kappa(r1, r2)
        assert res["kappa"] is None
        assert res["p_value"] is None
        assert res["se"] is None
        assert res["n_subjects"] == 4
        assert res["weighting"] == "unweighted"
        assert "note" in res
        assert "Degenerate" in res["note"]

    def test_cohens_kappa_denominator_zero_degenerate(self):
        s1 = pd.Series([1, 1, 1, 1])
        s2 = pd.Series([1, 1, 1, 1])
        res = cohens_kappa(s1, s2, categories=[1, 2])
        assert res["kappa"] is None
        assert res["p_value"] is None
        assert res["se"] is None
        assert res["n_subjects"] == 4
        assert res["weighting"] == "unweighted"
        assert "note" in res
        assert "Degenerate agreement" in res["note"]

    def test_fleiss_kappa_degenerate_single_category(self):
        mat = np.array([[3], [3], [3]])
        res = fleiss_kappa(mat)
        assert res["kappa"] is None
        assert res["p_value"] is None
        assert res["se"] is None
        assert res["n_subjects"] == 3
        assert res["n_raters"] == 3
        assert "note" in res
        assert "Degenerate" in res["note"]

    def test_fleiss_kappa_denominator_zero_degenerate(self):
        mat = np.array([[3, 0], [3, 0], [3, 0]])
        res = fleiss_kappa(mat)
        assert res["kappa"] is None
        assert res["p_value"] is None
        assert res["se"] is None
        assert res["n_subjects"] == 3
        assert res["n_raters"] == 3
        assert "note" in res
        assert "Degenerate agreement" in res["note"]

    def test_has_clusters_does_not_match_embedded_delivery_method(self, tmp_path):
        runner = CliRunner()
        df = pd.DataFrame(
            {
                "patient_id": list(range(1, 21)),
                "delivery_method": [0, 1] * 10,
                "age": [30.0 + i for i in range(20)],
            }
        )
        csv_path = tmp_path / "delivery.csv"
        df.to_csv(csv_path, index=False)
        res = runner.invoke(cli, ["profile", "--data", str(csv_path)])
        assert res.exit_code == 0
        assert "Type 6: Inter-Rater Reliability / Agreement Study" not in res.output

        # Verify rater_1 and rater_2 match
        df_cluster = pd.DataFrame(
            {
                "patient_id": list(range(1, 21)),
                "rater_1": [1, 2] * 10,
                "rater_2": [1, 2] * 10,
            }
        )
        csv_cluster = tmp_path / "raters.csv"
        df_cluster.to_csv(csv_cluster, index=False)
        res_cluster = runner.invoke(cli, ["profile", "--data", str(csv_cluster)])
        assert res_cluster.exit_code == 0
        assert "Type 6: Inter-Rater Reliability / Agreement Study" in res_cluster.output

        # Verify rater_id_1, observer_id_2, method_id_1 match
        for col_name in ("rater_id_1", "observer_id_2", "method_id_1", "rater_id1"):
            df_comp = pd.DataFrame(
                {
                    "patient_id": list(range(1, 21)),
                    col_name: [1, 2] * 10,
                }
            )
            p_comp = tmp_path / f"{col_name}.csv"
            df_comp.to_csv(p_comp, index=False)
            res_comp = runner.invoke(cli, ["profile", "--data", str(p_comp)])
            assert res_comp.exit_code == 0
            assert (
                "Type 6: Inter-Rater Reliability / Agreement Study" in res_comp.output
            )

    def test_diag_cmd_gold_standard_validation(self, tmp_path):
        runner = CliRunner()
        # 1. Non-numeric strings
        df_str = pd.DataFrame(
            {
                "outcome": ["pos", "neg", "pos", "neg"],
                "biomarker": [1.5, 0.8, 2.1, 0.4],
            }
        )
        p_str = tmp_path / "str_diag.csv"
        df_str.to_csv(p_str, index=False)
        res1 = runner.invoke(
            cli,
            [
                "diag",
                "--data",
                str(p_str),
                "--gold-standard",
                "outcome",
                "--test",
                "biomarker",
                "--cutoff",
                "1.0",
            ],
        )
        assert res1.exit_code != 0
        assert "must be numeric containing only 0 and 1" in res1.output

        # 1b. Boolean series (non-numeric for diagnostic modeling)
        df_bool = pd.DataFrame(
            {
                "outcome": [True, False, True, False],
                "biomarker": [1.5, 0.8, 2.1, 0.4],
            }
        )
        p_bool = tmp_path / "bool_diag.csv"
        df_bool.to_csv(p_bool, index=False)
        res_bool = runner.invoke(
            cli,
            [
                "diag",
                "--data",
                str(p_bool),
                "--gold-standard",
                "outcome",
                "--test",
                "biomarker",
                "--cutoff",
                "1.0",
            ],
        )
        assert res_bool.exit_code != 0
        assert "must be numeric containing only 0 and 1" in res_bool.output

        # 2. Non-0/1 integers
        df_int = pd.DataFrame(
            {
                "outcome": [1, 2, 1, 2],
                "biomarker": [1.5, 0.8, 2.1, 0.4],
            }
        )
        p_int = tmp_path / "int_diag.csv"
        df_int.to_csv(p_int, index=False)
        res2 = runner.invoke(
            cli,
            [
                "diag",
                "--data",
                str(p_int),
                "--gold-standard",
                "outcome",
                "--test",
                "biomarker",
                "--cutoff",
                "1.0",
            ],
        )
        assert res2.exit_code != 0
        assert "must contain only 0 and 1" in res2.output

        # 3. Continuous float values
        df_float = pd.DataFrame(
            {
                "outcome": [0.5, 0.2, 0.8, 0.1],
                "biomarker": [1.5, 0.8, 2.1, 0.4],
            }
        )
        p_float = tmp_path / "float_diag.csv"
        df_float.to_csv(p_float, index=False)
        res3 = runner.invoke(
            cli,
            [
                "diag",
                "--data",
                str(p_float),
                "--gold-standard",
                "outcome",
                "--test",
                "biomarker",
                "--cutoff",
                "1.0",
            ],
        )
        assert res3.exit_code != 0
        assert "must contain only 0 and 1" in res3.output

        # 4. Valid 0/1 labels
        df_valid = pd.DataFrame(
            {
                "outcome": [0, 1, 0, 1],
                "biomarker": [0.2, 0.8, 0.3, 0.9],
            }
        )
        p_valid = tmp_path / "valid_diag.csv"
        df_valid.to_csv(p_valid, index=False)
        res4 = runner.invoke(
            cli,
            [
                "diag",
                "--data",
                str(p_valid),
                "--gold-standard",
                "outcome",
                "--test",
                "biomarker",
                "--cutoff",
                "0.5",
            ],
        )
        assert res4.exit_code == 0

    def test_is_id_column_token_boundary(self):
        """Verify _is_id_column requires whole-token 'id' naming and no integer dtype shortcut."""
        n = 10
        unique_int_series = pd.Series(list(range(1, n + 1)))
        unique_float_series = pd.Series([float(x) for x in range(1, n + 1)])

        # Words ending with 'id' that are not whole tokens
        for col in [
            "lipid",
            "fluid",
            "acid",
            "opioid",
            "steroid",
            "carotid",
            "pyramid",
            "solid",
            "rapid",
        ]:
            assert not _is_id_column(col, unique_float_series), (
                f"{col} should not be an ID column"
            )
            assert not _is_id_column(col, unique_int_series), (
                f"{col} should not be an ID column"
            )

        # Words starting with 'id' that are not whole tokens
        for col in ["idiopathic", "idea", "identical", "identity"]:
            assert not _is_id_column(col, unique_int_series), (
                f"{col} should not be an ID column"
            )

        # Unique integer columns with non-ID names (integer-dtype shortcut removed)
        for col in ["heart_rate", "sbp", "age", "creatinine_int", "rank", "score"]:
            assert not _is_id_column(col, unique_int_series), (
                f"{col} should not be an ID column"
            )

        # Whole token 'id' at start or end matches when unique and len > 5
        for col in [
            "id",
            "id_patient",
            "id.patient",
            "id-patient",
            "id patient",
            "patient_id",
            "patient.id",
            "patient-id",
            "patient id",
        ]:
            assert _is_id_column(col, unique_int_series), (
                f"{col} should be an ID column"
            )

        # Non-ASCII and Unicode token boundary tests
        for col in ["id_ผู้ป่วย", "id.ผู้ป่วย", "id-ผู้ป่วย", "ผู้ป่วย_id", "ผู้ป่วย.id", "ผู้ป่วย-id"]:
            assert _is_id_column(col, unique_int_series), (
                f"{col} should be an ID column"
            )
        for col in ["idผู้ป่วย", "ผู้ป่วยid"]:
            assert not _is_id_column(col, unique_int_series), (
                f"{col} should not be an ID column"
            )

        # Preserves existing uniqueness threshold (len <= 5 does not trigger uniqueness fallback)
        short_series = pd.Series([1, 2, 3, 4, 5])
        assert not _is_id_column("id-patient", short_series)
        assert not _is_id_column("patient id", short_series)
        # Exact matches and .endswith(('_id', '.id')) still return True regardless of length
        assert _is_id_column("patient_id", short_series)
        assert _is_id_column("id", short_series)

    def test_cli_profile_study_design_type_mapping(self, tmp_path):
        """Verify profile_cmd maps survival to Type 3, causal PSM to Type 5, binary to Type 2, and retains Type 1 and Type 6."""
        runner = CliRunner()

        # 1. Survival cohort -> Type 3
        df_surv = pd.DataFrame(
            {
                "patient_id": list(range(1, 21)),
                "followup_time": [10.5 + i for i in range(20)],
                "event_status": [0, 1] * 10,
                "age": [50 + i for i in range(20)],
            }
        )
        p_surv = tmp_path / "surv.csv"
        df_surv.to_csv(p_surv, index=False)
        out_surv = tmp_path / "surv_prof.json"
        res_surv = runner.invoke(
            cli, ["profile", "--data", str(p_surv), "--output", str(out_surv)]
        )
        assert res_surv.exit_code == 0
        data_surv = json.loads(out_surv.read_text())
        assert (
            data_surv["inferred_study_design"]
            == "Type 3: Time-to-Event / Survival Cohort"
        )
        assert "Type 3: Time-to-Event / Survival Cohort" in res_surv.output

        # 2. Causal PSM (>5 cols, treatment column, no survival/clusters) -> Type 5
        df_causal = pd.DataFrame(
            {
                "patient_id": list(range(1, 25)),
                "treatment": [0, 1] * 12,
                "age": [45 + i for i in range(24)],
                "bmi": [22.0 + (i * 0.5) for i in range(24)],
                "sbp": [120 + i for i in range(24)],
                "hr": [70 + (i % 10) for i in range(24)],
            }
        )
        p_causal = tmp_path / "causal.csv"
        df_causal.to_csv(p_causal, index=False)
        out_causal = tmp_path / "causal_prof.json"
        res_causal = runner.invoke(
            cli, ["profile", "--data", str(p_causal), "--output", str(out_causal)]
        )
        assert res_causal.exit_code == 0
        data_causal = json.loads(out_causal.read_text())
        assert (
            data_causal["inferred_study_design"]
            == "Type 5: Observational Comparative Effectiveness (Causal PSM)"
        )
        assert (
            "Type 5: Observational Comparative Effectiveness (Causal PSM)"
            in res_causal.output
        )

        # 3. Binary risk prediction (binary endpoint, <=5 cols or no treatment, no survival) -> Type 2
        df_binary = pd.DataFrame(
            {
                "patient_id": list(range(1, 21)),
                "mortality": [0, 1] * 10,
                "age": [50 + i for i in range(20)],
                "score": [1.5 * i for i in range(20)],
            }
        )
        p_binary = tmp_path / "binary.csv"
        df_binary.to_csv(p_binary, index=False)
        out_binary = tmp_path / "binary_prof.json"
        res_binary = runner.invoke(
            cli, ["profile", "--data", str(p_binary), "--output", str(out_binary)]
        )
        assert res_binary.exit_code == 0
        data_binary = json.loads(out_binary.read_text())
        assert (
            data_binary["inferred_study_design"]
            == "Type 2: Multivariable Risk Prediction / Binary Outcome"
        )
        assert (
            "Type 2: Multivariable Risk Prediction / Binary Outcome"
            in res_binary.output
        )

        # 4. Cross-sectional / observational (continuous outcome, no survival/clusters/treatment) -> Type 1
        df_cross = pd.DataFrame(
            {
                "patient_id": list(range(1, 21)),
                "sbp": [120.0 + i for i in range(20)],
                "age": [50 + i for i in range(20)],
            }
        )
        p_cross = tmp_path / "cross.csv"
        df_cross.to_csv(p_cross, index=False)
        out_cross = tmp_path / "cross_prof.json"
        res_cross = runner.invoke(
            cli, ["profile", "--data", str(p_cross), "--output", str(out_cross)]
        )
        assert res_cross.exit_code == 0
        data_cross = json.loads(out_cross.read_text())
        assert (
            data_cross["inferred_study_design"]
            == "Type 1: Cross-Sectional / Observational Study"
        )
        assert "Type 1: Cross-Sectional / Observational Study" in res_cross.output

        # 5. Agreement / clusters (cluster/rater columns, no survival) -> Type 6
        df_clusters = pd.DataFrame(
            {
                "rater_id": [1, 2] * 10,
                "score": [3.5 + i for i in range(20)],
            }
        )
        p_clusters = tmp_path / "clusters.csv"
        df_clusters.to_csv(p_clusters, index=False)
        out_clusters = tmp_path / "clusters_prof.json"
        res_clusters = runner.invoke(
            cli, ["profile", "--data", str(p_clusters), "--output", str(out_clusters)]
        )
        assert res_clusters.exit_code == 0
        data_clusters = json.loads(out_clusters.read_text())
        assert (
            data_clusters["inferred_study_design"]
            == "Type 6: Inter-Rater Reliability / Agreement Study"
        )
        assert (
            "Type 6: Inter-Rater Reliability / Agreement Study" in res_clusters.output
        )


# ==============================================================================
# 11. Follow-up PR#4 Review Verification Tests
# ==============================================================================
class TestPR4ReviewFollowupFixes:
    def test_mediation_joint_covariance_sampling(self):
        np.random.seed(42)
        n = 120
        trt = np.random.binomial(1, 0.5, size=n)
        med = 0.7 * trt + np.random.normal(0, 1, size=n)
        y = 0.5 * trt + 0.8 * med + np.random.normal(0, 1, size=n)
        df = pd.DataFrame({"trt": trt, "med": med, "y": y})

        res = run_mediation(df, treatment="trt", mediator="med", outcome="y", seed=42)
        assert "total_ci" in res
        assert "acme_ci" in res
        assert "ade_ci" in res
        assert res["total_ci"][0] < res["total_effect"] < res["total_ci"][1]

    def test_fit_logistic_rcs_rejects_knots_below_3(self):
        np.random.seed(42)
        n = 50
        df = pd.DataFrame(
            {
                "y": np.random.binomial(1, 0.5, size=n),
                "x": np.linspace(10, 50, n),
            }
        )
        with pytest.raises(ValueError, match="n_knots must be >= 3"):
            fit_logistic_rcs(df, outcome="y", spline_var="x", n_knots=2)

        # n_knots=3 should succeed
        res = fit_logistic_rcs(df, outcome="y", spline_var="x", n_knots=3)
        assert res["knots"] == 3

    def test_loader_preserves_literal_na_and_treats_empty_as_nan(self, tmp_path):
        csv_file = tmp_path / "literal_na.csv"
        csv_file.write_text(
            "id,code,val\n1,NA,\n2,N/A,10.5\n3,Normal,20.0\n", encoding="utf-8"
        )
        df = load_clinical_data(csv_file)
        # Empty cell should be NaN
        assert pd.isna(df.loc[0, "val"])
        # Literal "NA" should remain string "NA" for explicit clinical missingness rules
        assert df.loc[0, "code"] == "NA"
        assert df.loc[1, "code"] == "N/A"

    def test_loader_rejects_duplicate_column_headers_after_normalization(
        self, tmp_path
    ):
        csv_file = tmp_path / "dup_cols.csv"
        csv_file.write_text("age ,age,score\n50,51,100\n", encoding="utf-8")
        with pytest.raises(
            click.ClickException,
            match="duplicate column names after normalization: \\['age'\\]",
        ):
            load_clinical_data(csv_file)

    def test_profile_survival_endpoint_token_matching(self, tmp_path):
        runner = CliRunner()
        # Columns that should NOT trigger survival_time
        df_non_surv = pd.DataFrame(
            {
                "patient_id": list(range(1, 21)),
                "survey_score": [1.0] * 20,
                "estimator_val": [2.0] * 20,
                "outcome": [0, 1] * 10,
            }
        )
        p1 = tmp_path / "non_surv.csv"
        df_non_surv.to_csv(p1, index=False)
        out1 = tmp_path / "prof1.json"
        res1 = runner.invoke(cli, ["profile", "--data", str(p1), "--output", str(out1)])
        assert res1.exit_code == 0
        data1 = json.loads(out1.read_text())
        surv_cols1 = [
            e["column"]
            for e in data1.get("candidate_endpoints", [])
            if e.get("type") == "survival_time"
        ]
        assert len(surv_cols1) == 0

        # Columns that SHOULD trigger survival_time
        df_surv = pd.DataFrame(
            {
                "patient_id": list(range(1, 21)),
                "time_to_event": [10.0 + i for i in range(20)],
                "overall_survival": [20.0 + i for i in range(20)],
                "status": [0, 1] * 10,
            }
        )
        p2 = tmp_path / "surv.csv"
        df_surv.to_csv(p2, index=False)
        out2 = tmp_path / "prof2.json"
        res2 = runner.invoke(cli, ["profile", "--data", str(p2), "--output", str(out2)])
        assert res2.exit_code == 0
        data2 = json.loads(out2.read_text())
        surv_cols2 = [
            e["column"]
            for e in data2.get("candidate_endpoints", [])
            if e.get("type") == "survival_time"
        ]
        assert "time_to_event" in surv_cols2
        assert "overall_survival" in surv_cols2

    def test_clean_outlier_excludes_id_and_zero_iqr(self, tmp_path):
        runner = CliRunner()
        # id column, zero-IQR column, and normal numeric column with an extreme outlier
        df = pd.DataFrame(
            {
                "patient_id": list(range(1, 21)),
                "zero_iqr": [0.0] * 18 + [5.0, 10.0],  # 18 zeros, q25=0, q75=0 -> IQR=0
                "biomarker": [10.0 + (i % 5) for i in range(19)]
                + [1000.0],  # normal numeric column with outlier
            }
        )
        p = tmp_path / "outlier_test.csv"
        df.to_csv(p, index=False)
        out_clean = tmp_path / "clean.csv"
        audit_out = tmp_path / "retention.json"
        res = runner.invoke(
            cli,
            [
                "clean",
                "--data",
                str(p),
                "--strategy",
                "complete-case",
                "--missing-justification",
                "Clinical audit complete",
                "--outlier-action",
                "remove",
                "--outlier-cols",
                "biomarker",
                "--output",
                str(out_clean),
                "--audit-out",
                str(audit_out),
            ],
        )
        assert res.exit_code == 0
        audit_data = json.loads(audit_out.read_text())
        outlier_counts = audit_data.get("outlier_counts", {})
        assert "patient_id" not in outlier_counts
        assert "zero_iqr" not in outlier_counts
        assert "biomarker" in outlier_counts

    def test_kappa_cmd_validates_columns_and_rejects_missing(self, tmp_path):
        runner = CliRunner()
        # Missing values in ratings should be rejected
        df_missing = pd.DataFrame(
            {
                "rater1": [0, 1, np.nan, 2],
                "rater2": [0, 1, 1, 2],
            }
        )
        p = tmp_path / "kappa_miss.csv"
        df_missing.to_csv(p, index=False)
        res = runner.invoke(
            cli,
            [
                "agreement",
                "kappa",
                "--data",
                str(p),
                "--rater1",
                "rater1",
                "--rater2",
                "rater2",
            ],
        )
        assert res.exit_code != 0
        assert "missing" in res.output.lower()

        # Missing column name should raise validation error
        res_missing_col = runner.invoke(
            cli,
            [
                "agreement",
                "kappa",
                "--data",
                str(p),
                "--rater1",
                "rater1",
                "--rater2",
                "nonexistent",
            ],
        )
        assert res_missing_col.exit_code != 0

    def test_sample_size_survival_requires_and_validates_event_probability(self):
        runner = CliRunner()
        # Missing event probability must fail
        res_no_pevent = runner.invoke(
            cli, ["sample-size", "--type", "survival", "--hr", "1.5"]
        )
        assert res_no_pevent.exit_code != 0
        assert "event probability" in res_no_pevent.output.lower()

        # Invalid event probability <= 0 or > 1 must fail
        res_bad_pevent = runner.invoke(
            cli,
            [
                "sample-size",
                "--type",
                "survival",
                "--hr",
                "1.5",
                "--event-probability",
                "1.5",
            ],
        )
        assert res_bad_pevent.exit_code != 0

        # Valid event probability should succeed and report event_probability
        res_ok = runner.invoke(
            cli,
            [
                "sample-size",
                "--type",
                "survival",
                "--hr",
                "1.5",
                "--event-probability",
                "0.4",
            ],
        )
        assert res_ok.exit_code == 0
        out_dict = json.loads(res_ok.output)
        assert out_dict["event_probability"] == 0.4
        assert out_dict["hazard_ratio"] == 1.5
        assert out_dict["required_events"] > 0
        assert out_dict["total_sample_size"] > 0

    def test_report_diagnostic_narrative_and_table_apparent_estimates(self, tmp_path):
        runner = CliRunner()
        diag_data = {
            "accuracy_at_cutoff": {
                "cutoff": 2.5,
                "sensitivity": 0.85,
                "sensitivity_ci": [0.70, 0.93],
                "specificity": 0.80,
                "specificity_ci": [0.65, 0.90],
            },
            "roc": {
                "auc": 0.88,
                "direction": "high",
            },
            "dca": [{"threshold": 0.1, "net_benefit": 0.05}],
            "calibration": {
                "brier": {"brier_score": 0.15},
            },
            "probability_source": "in-sample logistic recalibration",
            "apparent_estimates_note": "Brier and net benefit are apparent estimates on derivation cohort.",
        }
        res_file = tmp_path / "diag.json"
        res_file.write_text(json.dumps(diag_data), encoding="utf-8")
        out_html = tmp_path / "diag_report.html"

        res = runner.invoke(
            cli,
            [
                "report",
                "--results",
                str(res_file),
                "--narrative",
                "--output",
                str(out_html),
            ],
        )
        assert res.exit_code == 0
        content = out_html.read_text(encoding="utf-8")
        # Check Brier apparent estimate label and note
        assert "Brier Score (Calibration - in-sample apparent estimate)" in content
        assert (
            "Brier and net benefit are apparent estimates on derivation cohort."
            in content
        )
        # Check diagnostic narrative
        assert "Wilson score method" in content
        assert "DeLong's non-parametric method" in content
        assert "Decision Curve Analysis" in content
        # Check that generic regression narrative is NOT present
        assert "Multivariable logistic regression was conducted" not in content
        assert (
            "Normally distributed continuous variables were expressed as mean"
            not in content
        )

    def test_fleiss_rejects_unmapped_rating(self):
        """Verify that calculate_kappa raises ValueError when a rating is absent from categories."""
        df = pd.DataFrame(
            {
                "subject": [1, 2, 3, 1, 2, 3, 1, 2, 3],
                "rater": ["r1", "r1", "r1", "r2", "r2", "r2", "r3", "r3", "r3"],
                "score": [
                    "A",
                    "B",
                    "C",
                    "A",
                    "B",
                    "D",
                    "A",
                    "B",
                    "C",
                ],  # "D" not in categories
            }
        )
        with pytest.raises(
            ValueError, match="not in (declared|the specified) categories"
        ):
            calculate_kappa(
                df,
                targets="subject",
                raters="rater",
                ratings="score",
                categories=["A", "B", "C"],
            )

    def test_sample_size_survival_rejects_p1_without_event_probability(self):
        """Verify that survival sample size calculation rejects --p1 and strictly requires --event-probability."""
        runner = CliRunner()
        res = runner.invoke(
            cli,
            [
                "sample-size",
                "--type",
                "survival",
                "--hr",
                "1.5",
                "--p1",
                "0.3",
            ],
        )
        assert res.exit_code != 0
        assert "event probability" in res.output.lower()

    def test_calibration_narrative_apparent_estimates(self):
        """Verify that methods narrative distinguishes in-sample apparent calibration from standard slope/intercept."""
        # Standard calibration
        narr_std = generate_methods_narrative(
            model_type="diagnostic",
            has_calibration=True,
            diagnostic_data={"calibration": {"brier": 0.1}},
        )
        assert "Brier score, calibration slope and intercept" in narr_std

        # Apparent calibration with probability_source
        narr_app = generate_methods_narrative(
            model_type="diagnostic",
            has_calibration=True,
            diagnostic_data={
                "calibration": {"brier": 0.1},
                "probability_source": "in-sample logistic recalibration",
            },
        )
        assert "apparent estimates" in narr_app
        assert "calibration slope and intercept were not reported" in narr_app

    def test_calculate_kappa_two_raters_incomplete_pairs_rejected(self):
        """Verify that calculate_kappa in two-rater branch rejects incomplete pairs with ValueError."""
        # Long format with 3 subjects: subject 1 and 2 rated by both, subject 3 only rated by r1
        df_incomplete = pd.DataFrame(
            {
                "subject": ["s1", "s1", "s2", "s2", "s3"],
                "rater": ["r1", "r2", "r1", "r2", "r1"],
                "score": [1, 1, 2, 2, 1],
            }
        )
        with pytest.raises(
            ValueError,
            match="Two-rater agreement requires complete pairs for all subjects",
        ):
            calculate_kappa(
                df_incomplete, targets="subject", raters="rater", ratings="score"
            )

        # Complete pairs should succeed
        df_complete = pd.DataFrame(
            {
                "subject": ["s1", "s1", "s2", "s2", "s3", "s3"],
                "rater": ["r1", "r2", "r1", "r2", "r1", "r2"],
                "score": [1, 1, 2, 2, 1, 2],
            }
        )
        res = calculate_kappa(
            df_complete, targets="subject", raters="rater", ratings="score"
        )
        assert res["type"] == "cohen"
        assert res["n_subjects"] == 3

    def test_diag_cmd_score_numeric_validation_and_cutoff_metadata(self, tmp_path):
        """Verify diag_cmd validates test_col and compare_roc numeric types, and preserves cutoff/direction metadata."""
        runner = CliRunner()

        # 1. Boolean test_col rejected
        df_bool_score = pd.DataFrame(
            {
                "gold": [0, 1, 0, 1],
                "score": [True, False, False, True],
            }
        )
        p_bool = tmp_path / "bool_score.csv"
        df_bool_score.to_csv(p_bool, index=False)
        res_bool = runner.invoke(
            cli,
            [
                "diag",
                "--data",
                str(p_bool),
                "--gold-standard",
                "gold",
                "--test",
                "score",
                "--cutoff",
                "0.5",
            ],
        )
        assert res_bool.exit_code != 0
        assert "must be numeric" in res_bool.output

        # 2. String compare_roc rejected
        df_comp_str = pd.DataFrame(
            {
                "gold": [0, 1, 0, 1],
                "score": [1.0, 2.0, 1.5, 3.0],
                "comp": ["low", "high", "low", "high"],
            }
        )
        p_comp = tmp_path / "comp_str.csv"
        df_comp_str.to_csv(p_comp, index=False)
        res_comp = runner.invoke(
            cli,
            [
                "diag",
                "--data",
                str(p_comp),
                "--gold-standard",
                "gold",
                "--test",
                "score",
                "--compare-roc",
                "comp",
            ],
        )
        assert res_comp.exit_code != 0
        assert "Comparison ROC column 'comp' must be numeric" in res_comp.output

        # 3. Valid cutoff preserves cutoff and direction in diag_res and accuracy_at_cutoff
        df_valid = pd.DataFrame(
            {
                "gold": [0, 1, 0, 1, 0, 1],
                "score": [1.0, 2.5, 1.2, 3.1, 0.9, 4.0],
            }
        )
        p_valid = tmp_path / "valid_diag.csv"
        df_valid.to_csv(p_valid, index=False)
        out_json = tmp_path / "diag_meta.json"
        res_valid = runner.invoke(
            cli,
            [
                "diag",
                "--data",
                str(p_valid),
                "--gold-standard",
                "gold",
                "--test",
                "score",
                "--cutoff",
                "2.0",
                "--direction",
                "high",
                "--output",
                str(out_json),
            ],
        )
        assert res_valid.exit_code == 0
        diag_data = json.loads(out_json.read_text())
        assert diag_data["cutoff"] == 2.0
        assert diag_data["direction"] == "high"
        assert diag_data["accuracy_at_cutoff"]["cutoff"] == 2.0
        assert diag_data["accuracy_at_cutoff"]["direction"] == "high"

    def test_sample_size_alpha_power_validation_across_types(self):
        """Verify that sample-size command validates alpha in (0,1), power in (0,1), and power > alpha."""
        runner = CliRunner()

        # Alpha <= 0
        res1 = runner.invoke(
            cli,
            [
                "sample-size",
                "--type",
                "survival",
                "--hr",
                "1.5",
                "--p-event",
                "0.4",
                "--alpha",
                "0.0",
            ],
        )
        assert res1.exit_code != 0
        assert "Alpha must be strictly between 0 and 1" in res1.output

        # Power >= 1
        res2 = runner.invoke(
            cli,
            [
                "sample-size",
                "--type",
                "correlation",
                "--r",
                "0.3",
                "--power",
                "1.0",
            ],
        )
        assert res2.exit_code != 0
        assert "Power must be strictly between 0 and 1" in res2.output

        # Power <= Alpha
        res3 = runner.invoke(
            cli,
            [
                "sample-size",
                "--type",
                "survival",
                "--hr",
                "1.5",
                "--p-event",
                "0.4",
                "--alpha",
                "0.10",
                "--power",
                "0.08",
            ],
        )
        assert res3.exit_code != 0
        assert "Power must exceed alpha" in res3.output

    def test_clean_outlier_cols_selection(self, tmp_path):
        """Verify that medstat clean --outlier-cols limits outlier transformations to specified columns."""
        runner = CliRunner()
        df = pd.DataFrame(
            {
                "patient_id": list(range(1, 21)),
                "biomarker_a": [10.0 + (i % 3) for i in range(19)] + [1000.0],
                "biomarker_b": [20.0 + (i % 3) for i in range(19)] + [2000.0],
            }
        )
        csv_in = tmp_path / "outliers_multi.csv"
        df.to_csv(csv_in, index=False)
        out_clean = tmp_path / "cleaned_multi.csv"
        audit_out = tmp_path / "audit_multi.json"

        # Scope winsorize only to biomarker_a
        res = runner.invoke(
            cli,
            [
                "clean",
                "--data",
                str(csv_in),
                "--strategy",
                "complete-case",
                "--missing-justification",
                "Audit complete",
                "--outlier-action",
                "winsorize",
                "--outlier-cols",
                "biomarker_a",
                "--output",
                str(out_clean),
                "--audit-out",
                str(audit_out),
            ],
        )
        assert res.exit_code == 0
        df_out = pd.read_csv(out_clean)
        # biomarker_a was winsorized (no longer 1000.0)
        assert df_out.loc[19, "biomarker_a"] < 1000.0
        # biomarker_b was untouched (still 2000.0)
        assert df_out.loc[19, "biomarker_b"] == 2000.0

        # Reject invalid column in --outlier-cols
        res_bad_col = runner.invoke(
            cli,
            [
                "clean",
                "--data",
                str(csv_in),
                "--strategy",
                "complete-case",
                "--missing-justification",
                "Audit complete",
                "--outlier-action",
                "winsorize",
                "--outlier-cols",
                "nonexistent_column",
            ],
        )
        assert res_bad_col.exit_code != 0
        assert (
            "Specified outlier column 'nonexistent_column' not found"
            in res_bad_col.output
        )

    def test_prisma_checklist_item_16b_official_wording(self):
        """Verify PRISMA 2020 item 16b description matches official reporting wording."""
        prisma = get_prisma_checklist()
        item_16b = next(it for it in prisma.items if it.number == "16b")
        assert (
            "might appear to meet the inclusion criteria, but which were excluded"
            in item_16b.description
        )

    def test_diagnostic_narrative_wilson_score_proportions_only(self):
        """Verify diagnostic methods narrative restricts Wilson score method strictly to proportion CIs."""
        narr = generate_methods_narrative(
            model_type="diagnostic",
            cutoff=2.0,
            direction="high",
            diagnostic_data={
                "accuracy_at_cutoff": {"sensitivity": 0.85, "specificity": 0.90}
            },
        )
        assert (
            "Confidence intervals (95%) for proportions (sensitivity, specificity, PPV, and NPV) were calculated using the Wilson score method."
            in narr
        )
        assert (
            "Point estimates and 95% confidence intervals were calculated using the Wilson score method"
            not in narr
        )

    def test_clean_destructive_outlier_action_without_selector_raises_error(
        self, tmp_path
    ):
        """Verify that remove, winsorize, and cap require explicit --outlier-cols, while flag does not."""
        runner = CliRunner()
        df = pd.DataFrame(
            {
                "patient_id": list(range(1, 21)),
                "biomarker": [10.0 + (i % 3) for i in range(19)] + [1000.0],
            }
        )
        csv_in = tmp_path / "outliers_destr.csv"
        df.to_csv(csv_in, index=False)

        for act in ("remove", "winsorize", "cap"):
            res = runner.invoke(
                cli,
                [
                    "clean",
                    "--data",
                    str(csv_in),
                    "--strategy",
                    "complete-case",
                    "--missing-justification",
                    "Audit complete",
                    "--outlier-action",
                    act,
                ],
            )
            assert res.exit_code != 0
            assert (
                f"Destructive outlier action '{act}' requires explicit variable selection via --outlier-cols"
                in res.output
            )

        # Non-destructive flag without --outlier-cols must succeed
        res_flag = runner.invoke(
            cli,
            [
                "clean",
                "--data",
                str(csv_in),
                "--strategy",
                "complete-case",
                "--missing-justification",
                "Audit complete",
                "--outlier-action",
                "flag",
            ],
        )
        assert res_flag.exit_code == 0

    def test_clean_rejects_non_positive_iqr_multiplier(self, tmp_path):
        """Verify that --iqr-multiplier rejects 0 and negative values."""
        runner = CliRunner()
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        csv_in = tmp_path / "iqr_test.csv"
        df.to_csv(csv_in, index=False)

        for invalid_k in ("0", "0.0", "-1.5"):
            res = runner.invoke(
                cli,
                [
                    "clean",
                    "--data",
                    str(csv_in),
                    "--strategy",
                    "complete-case",
                    "--missing-justification",
                    "Audit complete",
                    "--iqr-multiplier",
                    invalid_k,
                ],
            )
            assert res.exit_code != 0
            assert (
                "is not in the range x > 0.0" in res.output
                or "invalid" in res.output.lower()
            )

    def test_model_cmd_spline_var_column_and_missingness_validation(self, tmp_path):
        """Verify that --spline-var is included in column validation and missingness check."""
        runner = CliRunner()
        df = pd.DataFrame(
            {
                "y": [0, 1, 0, 1, 0, 1, 0, 1],
                "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "age_missing": [50.0, np.nan, 60.0, 70.0, 45.0, 55.0, 65.0, 75.0],
            }
        )
        csv_in = tmp_path / "spline_val_test.csv"
        df.to_csv(csv_in, index=False)

        # 1. Typo in --spline-var triggers validate_columns
        res_typo = runner.invoke(
            cli,
            [
                "model",
                "--data",
                str(csv_in),
                "--type",
                "logistic",
                "--outcome",
                "y",
                "--covariates",
                "x1",
                "--spline-var",
                "age_typo",
            ],
        )
        assert res_typo.exit_code != 0
        assert (
            "not found in dataset" in res_typo.output and "age_typo" in res_typo.output
        )

        # 2. Missing data in --spline-var triggers check_data_missingness
        res_nan = runner.invoke(
            cli,
            [
                "model",
                "--data",
                str(csv_in),
                "--type",
                "logistic",
                "--outcome",
                "y",
                "--covariates",
                "x1",
                "--spline-var",
                "age_missing",
            ],
        )
        assert res_nan.exit_code != 0
        assert (
            "Unhandled missing data detected" in res_nan.output
            and "age_missing" in res_nan.output
        )

    def test_sample_size_event_probability_validation(self):
        """Verify event probability error message states (0, 1] and accepts 1.0."""
        runner = CliRunner()
        # Invalid <= 0
        res_zero = runner.invoke(
            cli,
            [
                "sample-size",
                "--type",
                "survival",
                "--hazard-ratio",
                "1.5",
                "--p-event",
                "0.0",
            ],
        )
        assert res_zero.exit_code != 0
        assert "Event probability must be in (0, 1] (found: 0.0)." in res_zero.output

        # Invalid > 1
        res_high = runner.invoke(
            cli,
            [
                "sample-size",
                "--type",
                "survival",
                "--hazard-ratio",
                "1.5",
                "--p-event",
                "1.5",
            ],
        )
        assert res_high.exit_code != 0
        assert "Event probability must be in (0, 1] (found: 1.5)." in res_high.output

        # Valid 1.0 succeeds
        res_one = runner.invoke(
            cli,
            [
                "sample-size",
                "--type",
                "survival",
                "--hazard-ratio",
                "1.5",
                "--p-event",
                "1.0",
            ],
        )
        assert res_one.exit_code == 0
        assert "total_sample_size" in res_one.output

    def test_model_cmd_ci_method_forwarding_to_firth(self, tmp_path, monkeypatch):
        """Verify --ci-method profile and wald are forwarded to Firth logistic and Cox."""
        import sys

        import medstat.cli.main  # noqa: F401

        runner = CliRunner()
        recorded_calls = []

        cli_mod = sys.modules["medstat.cli.main"]
        orig_fit_log = cli_mod.fit_firth_logistic
        orig_fit_cox = cli_mod.fit_firth_cox

        def spy_fit_log(*args, **kwargs):
            recorded_calls.append(("logistic", kwargs.get("ci_method")))
            return orig_fit_log(*args, **kwargs)

        def spy_fit_cox(*args, **kwargs):
            recorded_calls.append(("cox", kwargs.get("ci_method")))
            return orig_fit_cox(*args, **kwargs)

        monkeypatch.setattr(cli_mod, "fit_firth_logistic", spy_fit_log)
        monkeypatch.setattr(cli_mod, "fit_firth_cox", spy_fit_cox)

        # Logistic dataset with separation
        df_log = pd.DataFrame(
            {
                "y": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                "x": [1.0, 2.0, 1.5, 2.5, 8.0, 9.0, 8.5, 9.5, 10.0, 10.5],
            }
        )
        csv_log = tmp_path / "firth_ci_log.csv"
        df_log.to_csv(csv_log, index=False)

        # Profile
        res_prof = runner.invoke(
            cli,
            [
                "model",
                "--data",
                str(csv_log),
                "--type",
                "logistic",
                "--outcome",
                "y",
                "--covariates",
                "x",
                "--method",
                "firth",
                "--ci-method",
                "profile",
            ],
        )
        assert res_prof.exit_code == 0

        # Wald
        res_wald = runner.invoke(
            cli,
            [
                "model",
                "--data",
                str(csv_log),
                "--type",
                "logistic",
                "--outcome",
                "y",
                "--covariates",
                "x",
                "--method",
                "firth",
                "--ci-method",
                "wald",
            ],
        )
        assert res_wald.exit_code == 0

        # Cox dataset
        df_cox = pd.DataFrame(
            {
                "time": [5.0, 10.0, 12.0, 15.0, 20.0, 25.0],
                "event": [1, 1, 0, 1, 0, 1],
                "x": [1.0, 2.0, 1.5, 3.0, 2.5, 4.0],
            }
        )
        csv_cox = tmp_path / "firth_ci_cox.csv"
        df_cox.to_csv(csv_cox, index=False)

        res_cox_prof = runner.invoke(
            cli,
            [
                "model",
                "--data",
                str(csv_cox),
                "--type",
                "cox",
                "--time",
                "time",
                "--outcome",
                "event",
                "--covariates",
                "x",
                "--method",
                "firth",
                "--ci-method",
                "profile",
            ],
        )
        assert res_cox_prof.exit_code == 0

        res_cox_wald = runner.invoke(
            cli,
            [
                "model",
                "--data",
                str(csv_cox),
                "--type",
                "cox",
                "--time",
                "time",
                "--outcome",
                "event",
                "--covariates",
                "x",
                "--method",
                "firth",
                "--ci-method",
                "wald",
            ],
        )
        assert res_cox_wald.exit_code == 0

        # Verify exact ci_method arguments passed to each fitter
        assert recorded_calls == [
            ("logistic", "pl"),
            ("logistic", "wald"),
            ("cox", "pl"),
            ("cox", "wald"),
        ]

    def test_diagnostic_narrative_omits_wilson_when_no_cutoff(self):
        """Verify diagnostic narrative omits Wilson score claim when no cutoff was evaluated."""
        narr = generate_methods_narrative(
            model_type="diagnostic",
            diagnostic_data={
                "roc": {"auc": 0.88, "ci_lower": 0.80, "ci_upper": 0.95},
                "dca": {"thresholds": [0.1, 0.2]},
            },
        )
        assert "Wilson score method" not in narr
        assert "Receiver Operating Characteristic (ROC)" in narr
        assert "Decision Curve Analysis (DCA)" in narr

    def test_run_mediation_missing_data_gate(self):
        """Verify that run_mediation routes data through prepare_data_for_analysis."""
        from medstat.data.missing import MissingStrategyRequiredError

        np.random.seed(42)
        n = 50
        trt = np.random.binomial(1, 0.5, size=n)
        med = 0.8 * trt + np.random.normal(0, 1, size=n)
        y = 0.5 * trt + 1.2 * med + np.random.normal(0, 1, size=n)

        # Introduce missing data in mediator
        med_missing = med.copy()
        med_missing[0] = np.nan
        med_missing[1] = np.nan

        df_missing = pd.DataFrame({"trt": trt, "med": med_missing, "y": y})

        # 1. Unspecified strategy raises MissingStrategyRequiredError
        with pytest.raises(MissingStrategyRequiredError):
            run_mediation(df_missing, treatment="trt", mediator="med", outcome="y")

        # 2. Explicit complete-case strategy succeeds and returns retention counts
        res = run_mediation(
            df_missing,
            treatment="trt",
            mediator="med",
            outcome="y",
            missing_strategy="complete-case",
            missing_justification="MCAR confirmed by clinical audit",
        )
        assert res["n_input"] == 50
        assert res["n_observations"] == 48
        assert res["n_excluded"] == 2
        assert res["missing_counts"]["med"] == 2

        # 3. Reject MICE strategy in run_mediation
        with pytest.raises(
            NotImplementedError, match="MICE multiple imputation is not supported"
        ):
            run_mediation(
                df_missing,
                treatment="trt",
                mediator="med",
                outcome="y",
                missing_strategy="mice",
                missing_justification="MAR assumption",
            )

        # 4. Reject disallowed strategies in prepare_data_for_analysis
        from medstat.data.missing import prepare_data_for_analysis

        with pytest.raises(
            NotImplementedError, match="not supported for this analysis workflow"
        ):
            prepare_data_for_analysis(
                df_missing,
                required_cols=["trt", "med", "y"],
                handle_missing="mice",
                missing_justification="MAR assumption",
                disallowed_strategies={"mice"},
            )

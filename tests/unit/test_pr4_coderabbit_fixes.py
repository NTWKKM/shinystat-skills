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

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from medstat.agreement.kappa import calculate_kappa, cohens_kappa, fleiss_kappa
from medstat.causal.mediation import run_mediation
from medstat.cli.main import cli
from medstat.data.loader import load_clinical_data
from medstat.models.splines import fit_logistic_rcs
from medstat.reporting.tables import render_balance_table


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
                "sbp": [120.0] * 48
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

        res_ok = runner.invoke(
            cli, ["sample-size", "--type", "survival", "--hr", "0.70"]
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

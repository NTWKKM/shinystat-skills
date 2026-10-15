"""
tests/unit/test_iptw.py: Unit tests for Inverse Probability of Treatment Weighting (IPTW).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from medstat.causal.balance import calculate_smd, compute_balance_table
from medstat.causal.iptw import compute_iptw_weights, fit_iptw_outcome_model
from medstat.figures.balance import plot_love
from medstat.figures.html.balance import plot_dual_love
from medstat.reporting.narrative import generate_iptw_narrative
from medstat.reporting.tables import render_balance_table


@pytest.fixture
def synthetic_cohort() -> pd.DataFrame:
    """Creates a reproducible synthetic cohort with confounding."""
    np.random.seed(42)
    n = 300
    age = np.random.normal(60, 10, n)
    biomarker = np.random.normal(15, 4, n)

    # Treatment assignment depends on age and biomarker
    logit_p = -3.5 + 0.04 * age + 0.10 * biomarker
    prob = 1.0 / (1.0 + np.exp(-logit_p))
    treatment = np.random.binomial(1, prob)

    # Continuous outcome
    y_cont = (
        100.0
        - 5.0 * treatment
        + 0.5 * age
        + 1.2 * biomarker
        + np.random.normal(0, 3, n)
    )

    # Binary outcome
    logit_y = -2.0 + 0.8 * treatment + 0.02 * age
    prob_y = 1.0 / (1.0 + np.exp(-logit_y))
    y_bin = np.random.binomial(1, prob_y)

    # Survival outcome
    baseline_hazard = 0.01
    hr = np.exp(0.5 * treatment + 0.01 * age)
    durations = np.random.exponential(1.0 / (baseline_hazard * hr))
    event_observed = np.random.binomial(1, 0.7, n)

    return pd.DataFrame(
        {
            "treatment": treatment,
            "age": age,
            "biomarker": biomarker,
            "y_cont": y_cont,
            "y_bin": y_bin,
            "durations": durations,
            "event_observed": event_observed,
        }
    )


def test_compute_iptw_weights_all_estimands(synthetic_cohort: pd.DataFrame):
    covariates = ["age", "biomarker"]
    for estimand in ["ATE", "ATT", "ATC", "ATO"]:
        res = compute_iptw_weights(
            synthetic_cohort,
            treatment="treatment",
            covariates=covariates,
            estimand=estimand,
        )
        assert res.weights is not None
        assert len(res.weights) == len(synthetic_cohort)
        assert (res.weights >= 0).all()
        assert res.effective_sample_size["treated"] > 0
        assert res.effective_sample_size["control"] > 0
        assert res.effective_sample_size["total"] > 0
        assert 0.0 <= res.propensity_scores.min() <= res.propensity_scores.max() <= 1.0

        # Assert estimand-specific raw weight formulas (before truncation)
        t = res.df["treatment"].to_numpy()
        ps = res.propensity_scores.to_numpy()
        w_raw = res.df["iptw_weight_raw"].to_numpy()

        if estimand == "ATT":
            np.testing.assert_allclose(w_raw[t == 1], 1.0)
            np.testing.assert_allclose(w_raw[t == 0], (ps / (1.0 - ps))[t == 0])
        elif estimand == "ATC":
            np.testing.assert_allclose(w_raw[t == 0], 1.0)
            np.testing.assert_allclose(w_raw[t == 1], ((1.0 - ps) / ps)[t == 1])
        elif estimand == "ATO":
            np.testing.assert_allclose(w_raw[t == 1], (1.0 - ps)[t == 1])
            np.testing.assert_allclose(w_raw[t == 0], ps[t == 0])
        elif estimand == "ATE":
            p_t = float(np.mean(t))
            np.testing.assert_allclose(w_raw[t == 1], (p_t / ps)[t == 1])
            np.testing.assert_allclose(
                w_raw[t == 0], ((1.0 - p_t) / (1.0 - ps))[t == 0]
            )


def test_stabilized_weights_mean_approx_one(synthetic_cohort: pd.DataFrame):
    res = compute_iptw_weights(
        synthetic_cohort,
        treatment="treatment",
        covariates=["age", "biomarker"],
        estimand="ATE",
        stabilized=True,
    )
    # Stabilized weights for ATE should average near 1.0
    mean_sw = float(res.weights.mean())
    assert abs(mean_sw - 1.0) < 0.15


def test_weight_truncation_reduces_extreme(synthetic_cohort: pd.DataFrame):
    res_raw = compute_iptw_weights(
        synthetic_cohort,
        treatment="treatment",
        covariates=["age", "biomarker"],
        estimand="ATE",
        stabilized=False,
        truncation=None,
    )
    res_trunc = compute_iptw_weights(
        synthetic_cohort,
        treatment="treatment",
        covariates=["age", "biomarker"],
        estimand="ATE",
        stabilized=False,
        truncation=(0.05, 0.95),
    )
    assert res_trunc.weights.max() <= res_raw.weights.max()
    assert res_trunc.weights.min() >= res_raw.weights.min()


def test_bessel_weighted_smd(synthetic_cohort: pd.DataFrame):
    res = compute_iptw_weights(
        synthetic_cohort,
        treatment="treatment",
        covariates=["age", "biomarker"],
        estimand="ATE",
    )
    df = synthetic_cohort.copy()
    df["w"] = res.weights

    smd_raw = calculate_smd(df, "treatment", "age")
    smd_weighted = calculate_smd(
        treated=df.loc[df["treatment"] == 1, "age"],
        control=df.loc[df["treatment"] == 0, "age"],
        weights_treated=df.loc[df["treatment"] == 1, "w"],
        weights_control=df.loc[df["treatment"] == 0, "w"],
    )

    assert pd.notna(smd_raw)
    assert pd.notna(smd_weighted)
    # Weighted SMD should achieve better balance (closer to 0)
    assert abs(smd_weighted) < abs(smd_raw)


def test_compute_balance_table(synthetic_cohort: pd.DataFrame):
    res = compute_iptw_weights(
        synthetic_cohort,
        treatment="treatment",
        covariates=["age", "biomarker"],
    )
    df = synthetic_cohort.copy()
    df["iptw_w"] = res.weights

    bt = compute_balance_table(
        df,
        treatment="treatment",
        covariates=["age", "biomarker"],
        weights_col="iptw_w",
    )
    assert "Covariate" in bt.columns
    assert "Unadjusted_SMD" in bt.columns
    assert "Weighted_SMD" in bt.columns
    assert "Status" in bt.columns
    assert len(bt) == 2


def test_fit_iptw_outcome_models(synthetic_cohort: pd.DataFrame):
    res = compute_iptw_weights(
        synthetic_cohort,
        treatment="treatment",
        covariates=["age", "biomarker"],
        stabilized=True,
    )
    df = synthetic_cohort.copy()
    df["weights"] = res.weights

    # Continuous WLS
    fit_cont = fit_iptw_outcome_model(
        df,
        outcome_col="y_cont",
        treatment_col="treatment",
        weights_col="weights",
        model_type="continuous",
        robust_cov_type="HC1",
    )
    assert fit_cont["model_type"] == "continuous"
    assert "coef" in fit_cont
    assert "se" in fit_cont
    assert "ci_lower" in fit_cont
    assert "ci_upper" in fit_cont
    assert "p_value" in fit_cont

    # Assert continuous-effect recovery: seeded fixture treatment effect is -5.0
    assert fit_cont["coef"] == pytest.approx(-5.0, abs=1.5)
    assert fit_cont["ci_lower"] < fit_cont["ci_upper"]

    # Binary Logistic GLM
    fit_bin = fit_iptw_outcome_model(
        df,
        outcome_col="y_bin",
        treatment_col="treatment",
        weights_col="weights",
        model_type="binary",
    )
    assert fit_bin["model_type"] == "binary"
    assert "odds_ratio" in fit_bin
    assert fit_bin["odds_ratio"] > 0

    # Survival CoxPH
    fit_surv = fit_iptw_outcome_model(
        df,
        outcome_col="durations",
        treatment_col="treatment",
        weights_col="weights",
        model_type="survival",
        event_col="event_observed",
    )
    assert fit_surv["model_type"] == "survival"
    assert "hazard_ratio" in fit_surv
    assert fit_surv["hazard_ratio"] > 0


def test_plot_love_with_weighted_smd(tmp_path, synthetic_cohort: pd.DataFrame):
    res = compute_iptw_weights(
        synthetic_cohort,
        treatment="treatment",
        covariates=["age", "biomarker"],
    )
    df = synthetic_cohort.copy()
    df["w"] = res.weights
    bt = compute_balance_table(df, "treatment", ["age", "biomarker"], weights_col="w")

    out_file = tmp_path / "love_test.png"
    fig_res = plot_love(bt, out_path=out_file)
    assert out_file.exists()
    assert Path(fig_res.png_path) == out_file

    dual_res = plot_dual_love(bt, out_path=tmp_path / "love_dual.png")
    assert dual_res.plotly_fig is not None


def test_att_atc_truncation_preserves_reference_group(
    synthetic_cohort: pd.DataFrame,
):
    # ATT: treated (T=1) must stay exactly 1.0, only controls truncated
    res_att = compute_iptw_weights(
        synthetic_cohort,
        treatment="treatment",
        covariates=["age", "biomarker"],
        estimand="ATT",
        truncation=(0.05, 0.95),
    )
    t = res_att.df["treatment"].to_numpy()
    np.testing.assert_allclose(res_att.weights[t == 1], 1.0)

    # Compare non-reference control group's final weights with raw weights clipped at quantiles
    w_ctrl_raw = res_att.df.loc[t == 0, "iptw_weight_raw"].to_numpy()
    lo_att, hi_att = np.quantile(w_ctrl_raw, (0.05, 0.95))
    expected_ctrl = np.clip(w_ctrl_raw, lo_att, hi_att)
    np.testing.assert_allclose(res_att.weights[t == 0], expected_ctrl)
    assert np.any(res_att.weights[t == 0] != w_ctrl_raw)

    # ATC: controls (T=0) must stay exactly 1.0, only treated truncated
    res_atc = compute_iptw_weights(
        synthetic_cohort,
        treatment="treatment",
        covariates=["age", "biomarker"],
        estimand="ATC",
        truncation=(0.05, 0.95),
    )
    np.testing.assert_allclose(res_atc.weights[t == 0], 1.0)

    # Compare non-reference treated group's final weights with raw weights clipped at quantiles
    w_trt_raw = res_atc.df.loc[t == 1, "iptw_weight_raw"].to_numpy()
    lo_atc, hi_atc = np.quantile(w_trt_raw, (0.05, 0.95))
    expected_trt = np.clip(w_trt_raw, lo_atc, hi_atc)
    np.testing.assert_allclose(res_atc.weights[t == 1], expected_trt)
    assert np.any(res_atc.weights[t == 1] != w_trt_raw)


def test_fit_iptw_outcome_model_invalid_weights_validation(
    synthetic_cohort: pd.DataFrame,
):
    df = synthetic_cohort.copy()
    # Non-positive weights
    df["bad_weights"] = 1.0
    df.loc[0, "bad_weights"] = -0.5
    with pytest.raises(ValueError, match="bad_weights"):
        fit_iptw_outcome_model(
            df,
            outcome_col="y_cont",
            treatment_col="treatment",
            weights_col="bad_weights",
            model_type="continuous",
        )

    # Zero weights
    df.loc[0, "bad_weights"] = 0.0
    with pytest.raises(ValueError, match="bad_weights"):
        fit_iptw_outcome_model(
            df,
            outcome_col="y_cont",
            treatment_col="treatment",
            weights_col="bad_weights",
            model_type="continuous",
        )


def test_balance_table_status_exact_zero_smd():
    df = pd.DataFrame(
        {
            "treatment": [1, 1, 0, 0],
            "x": [10.0, 10.0, 10.0, 10.0],  # Perfectly balanced
            "w": [1.0, 1.0, 1.0, 1.0],
        }
    )
    bt = compute_balance_table(df, "treatment", ["x"], weights_col="w")
    assert bt.loc[0, "Weighted_Abs_SMD"] == pytest.approx(0.0)
    assert bt.loc[0, "Status"] == "Balanced"


def test_render_balance_table_consistent_columns():
    bal_data = {
        "covariates": ["age", "bmi"],
        "smd_pre": [0.25, 0.15],
        "smd_post": [np.nan, 0.04],  # Row 0 has NaN post SMD, Row 1 is finite
        "smd_weighted": [0.03, 0.02],
    }
    html = render_balance_table("Balance Table", bal_data)
    assert "Post-Matching SMD" in html
    assert "IPTW Weighted SMD" in html


def test_generate_iptw_narrative_truncated_and_diagnostics_schema():
    # 1. Truncated True with extreme weights and positivity violations
    diag = {
        "overall": {"n_extreme_gt_10": 4, "n_eff": 180.5},
        "extreme_ps_count": 6,
    }
    nar_trunc = generate_iptw_narrative(
        estimand="ATE",
        exposure="treatment",
        outcome="mortality",
        covariates=["age", "bmi"],
        n_eff={"total": 180.5, "treated": 90.0, "control": 90.5},
        max_smd=0.04,
        stabilized=True,
        truncated=True,
        diagnostics=diag,
    )
    assert "Weights were truncated at pre-specified quantiles" in nar_trunc
    assert "A total of 4 observations exhibited extreme weights (> 10.0)." in nar_trunc
    assert "Positivity inspection identified 6 subjects" in nar_trunc
    assert "N_eff = 180.5" in nar_trunc

    # 2. Truncated False
    nar_notrunc = generate_iptw_narrative(
        estimand="ATE",
        exposure="treatment",
        outcome="mortality",
        stabilized=True,
        truncated=False,
    )
    assert "Weights were not truncated." in nar_notrunc

    # 3. ATO (Overlap Weights) should not mention truncation
    nar_ato = generate_iptw_narrative(
        estimand="ATO",
        exposure="treatment",
        outcome="mortality",
        stabilized=False,
        truncated=False,
    )
    assert "Weights were not truncated" not in nar_ato
    assert "Weights were truncated" not in nar_ato

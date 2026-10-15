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

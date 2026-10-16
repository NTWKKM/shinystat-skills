"""
tests/unit/test_validation_and_riley_power.py: Comprehensive unit tests and mathematical
invariant verification for TRIPOD+AI bootstrap internal validation, uniform shrinkage,
model recalibration, and Riley / Obuchowski sample size determinations.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from medstat.diagnostic.validation import (
    _compute_calibration_slope,
    bootstrap_internal_validation,
    recalibrate_prediction_model,
)
from medstat.figures.calibration import (
    plot_bootstrap_optimism,
    plot_recalibration_comparison,
)
from medstat.figures.html.calibration import (
    plot_dual_bootstrap_optimism,
    plot_dual_recalibration_comparison,
)
from medstat.power.sample_size import (
    _approximate_cox_snell_r2_from_c_stat,
    calculate_diagnostic_auroc_sample_size,
    calculate_riley_sample_size,
)
from medstat.reporting.narrative import (
    generate_diagnostic_auroc_sample_size_narrative,
    generate_internal_validation_narrative,
    generate_recalibration_narrative,
    generate_riley_sample_size_narrative,
)
from medstat.reporting.tables import (
    format_internal_validation_table,
    format_recalibration_table,
    format_riley_sample_size_table,
)

# ==============================================================================
# 1. TRIPOD+AI Bootstrap Internal Validation & Uniform Shrinkage
# ==============================================================================


def test_bootstrap_internal_validation_mathematical_invariants():
    """Verify optimism calculation, shrinkage factor, and intercept re-estimation."""
    np.random.seed(42)
    n = 250
    x1 = np.random.normal(0, 1, size=n)
    x2 = np.random.normal(0, 1, size=n)
    x3 = np.random.binomial(1, 0.4, size=n)
    lp = -0.8 + 0.9 * x1 - 0.7 * x2 + 0.5 * x3
    prob = expit(lp)
    y = (np.random.uniform(size=n) < prob).astype(int)

    df = pd.DataFrame({"outcome": y, "x1": x1, "x2": x2, "x3": x3})
    res = bootstrap_internal_validation(
        df=df,
        outcome="outcome",
        predictors=["x1", "x2", "x3"],
        n_bootstrap=40,
        random_state=42,
    )

    # Invariant: Output dictionary keys
    assert "apparent_metrics" in res
    assert "metrics" in res
    assert "shrinkage_factor" in res
    assert "shrunk_coefficients" in res
    assert "warnings" in res
    assert res["n_total"] == n
    assert res["n_predictors"] == 3
    assert res["n_successful_replicates"] > 0

    # Invariant: Optimism = boot_mean - test_mean; Corrected = apparent - optimism
    for m_name in (
        "auroc",
        "brier_score",
        "nagelkerke_r2",
        "calibration_intercept",
        "calibration_slope",
    ):
        info = res["metrics"][m_name]
        assert "apparent" in info
        assert "boot_mean" in info
        assert "test_mean" in info
        assert "optimism" in info
        assert "optimism_corrected" in info

        expected_optimism = info["boot_mean"] - info["test_mean"]
        assert np.isclose(info["optimism"], expected_optimism, atol=1e-5)
        # Corrected metric formula: apparent - optimism (bounded appropriately)
        if m_name in ("auroc", "nagelkerke_r2"):
            expected_corr = max(0.0, min(1.0, info["apparent"] - info["optimism"]))
        elif m_name == "brier_score":
            expected_corr = max(0.0, info["apparent"] - info["optimism"])
        else:
            expected_corr = info["apparent"] - info["optimism"]
        assert np.isclose(info["optimism_corrected"], expected_corr, atol=1e-5)

    # Invariant: Uniform shrinkage factor bounded in [0.01, 1.0]
    s_hat = res["shrinkage_factor"]
    assert 0.01 <= s_hat <= 1.0

    # Invariant: Shrunk coefficients = s_hat * orig
    shrunk = res["shrunk_coefficients"]
    for p in ["x1", "x2", "x3"]:
        orig_coef = shrunk["original"][p]
        shrunk_coef = shrunk["shrunk"][p]
        assert np.isclose(shrunk_coef, orig_coef * s_hat, atol=1e-6)

    # Invariant: Re-estimated intercept preserves mean predicted risk equal to sample event rate
    re_inter = shrunk["shrunk"]["Intercept"]
    lp_shrunk = (
        re_inter
        + shrunk["shrunk"]["x1"] * x1
        + shrunk["shrunk"]["x2"] * x2
        + shrunk["shrunk"]["x3"] * x3
    )
    p_shrunk = expit(lp_shrunk)
    observed_prevalence = np.mean(y)
    mean_shrunk_risk = np.mean(p_shrunk)
    assert np.isclose(mean_shrunk_risk, observed_prevalence, atol=1e-3)


def test_bootstrap_internal_validation_edge_cases():
    """Verify input validation, degenerate handling, and clinical warnings."""
    rng = np.random.default_rng(0)

    # 1. Missing outcome or predictor column
    df = pd.DataFrame({"y": [0, 1] * 10, "x1": rng.standard_normal(20)})
    with pytest.raises(ValueError, match="not found"):
        bootstrap_internal_validation(df, "missing_y", ["x1"])
    with pytest.raises(ValueError, match="not found"):
        bootstrap_internal_validation(df, "y", ["missing_x"])
    with pytest.raises(ValueError, match="At least one predictor"):
        bootstrap_internal_validation(df, "y", [])

    # 2. Explicit Missing-Value Rejection with column details
    df_missing = pd.DataFrame(
        {
            "y": [0, 1, 0, 1, 0, 1, 0, 1, 0, np.nan],
            "x1": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "x2": np.arange(10, dtype=float),
        }
    )
    with pytest.raises(
        ValueError,
        match="Missing values detected in input columns: y \\(1\\), x1 \\(1\\)|Missing values detected in input columns:",
    ):
        bootstrap_internal_validation(df_missing, "y", ["x1", "x2"])

    # 3. Non-binary outcome
    df_multiclass = pd.DataFrame({"y": [0, 1, 2] * 10, "x": rng.standard_normal(30)})
    with pytest.raises(ValueError, match="strictly binary"):
        bootstrap_internal_validation(df_multiclass, "y", ["x"])

    # 4. Low minority-class EPP warning
    df_low_epp = pd.DataFrame(
        {
            "y": [0] * 95 + [1] * 5,  # Only 5 minority events
            "x1": rng.standard_normal(100),
            "x2": rng.standard_normal(100),
        }
    )
    res_low = bootstrap_internal_validation(
        df_low_epp, "y", ["x1", "x2"], n_bootstrap=10, random_state=42
    )
    assert any("EPP" in w for w in res_low["warnings"])
    assert res_low["events_per_predictor"] == pytest.approx(2.5)

    # 5. Penalty validation & Firth support
    with pytest.raises(ValueError, match="Unsupported penalty"):
        bootstrap_internal_validation(df, "y", ["x1"], penalty="unsupported")

    res_firth = bootstrap_internal_validation(
        df_low_epp, "y", ["x1", "x2"], n_bootstrap=10, penalty="firth", random_state=42
    )
    assert "metrics" in res_firth
    assert res_firth["n_successful_replicates"] > 0
    assert 0.01 <= res_firth["shrinkage_factor"] <= 1.0

    # 6. Degenerate / zero-variance linear predictor returns NaN slope
    y_bin = np.array([0, 1] * 10)
    lp_const = np.ones(20) * 0.5
    slope_nan = _compute_calibration_slope(y_bin, lp_const)
    assert np.isnan(slope_nan)


def test_bootstrap_internal_validation_categorical_predictors():
    """Verify categorical and boolean predictor expansion with pd.get_dummies."""
    np.random.seed(42)
    n = 150
    df = pd.DataFrame(
        {
            "y": np.random.binomial(1, 0.35, size=n),
            "age": np.random.normal(55, 10, size=n),
            "sex": np.random.choice(["male", "female"], size=n),
            "stage": pd.Categorical(
                np.random.choice(["I", "II", "III"], size=n),
                categories=["I", "II", "III"],
                ordered=True,
            ),
            "smoker": np.random.choice([True, False], size=n),
        }
    )

    res = bootstrap_internal_validation(
        df=df,
        outcome="y",
        predictors=["age", "sex", "stage", "smoker"],
        n_bootstrap=25,
        random_state=42,
    )

    # 1 continuous (age) + 1 sex dummy + 2 stage dummies + 1 smoker dummy = 5 parameters
    assert res["n_predictors"] == 5
    assert res["n_successful_replicates"] > 0
    assert 0.01 <= res["shrinkage_factor"] <= 1.0

    shrunk_orig = res["shrunk_coefficients"]["original"]
    shrunk_shrunk = res["shrunk_coefficients"]["shrunk"]
    for expected_feat in ["age", "stage_II", "stage_III"]:
        assert any(expected_feat in k for k in shrunk_orig)
        assert any(expected_feat in k for k in shrunk_shrunk)


def test_bootstrap_internal_validation_singular_matrix_skipped():
    """Verify bootstrap samples with zero-variance dummy columns are cleanly skipped."""
    np.random.seed(42)
    n = 100
    cat_vals = ["common"] * 95 + ["rare"] * 5
    y = np.array([0, 1] * 50)
    x1 = np.random.normal(0, 1, size=n)
    df = pd.DataFrame({"y": y, "x1": x1, "rare_cat": cat_vals})

    res = bootstrap_internal_validation(
        df=df,
        outcome="y",
        predictors=["x1", "rare_cat"],
        n_bootstrap=30,
        random_state=42,
    )

    # Replicates with zero variance in rare_cat are cleanly skipped
    assert res["n_successful_replicates"] > 0
    assert res["n_successful_replicates"] <= res["n_bootstrap"]


# ==============================================================================
# 2. Prediction Model Recalibration
# ==============================================================================


def test_recalibrate_prediction_model_logistic():
    """Verify logistic recalibration achieves slope ~ 1 and intercept ~ 0 and preserves AUROC."""
    np.random.seed(42)
    n = 300
    x = np.random.normal(0, 1, size=n)
    true_lp = 0.2 + 1.1 * x
    y = (np.random.uniform(size=n) < expit(true_lp)).astype(int)

    # Poorly calibrated predictions (under-estimates risk and over-confident)
    miscal_lp = -1.2 + 2.2 * x
    p_miscal = np.clip(expit(miscal_lp), 0.001, 0.999)

    res = recalibrate_prediction_model(y, p_miscal, method="logistic")

    # Invariant: Pre-recalibration slope != 1
    assert not np.isclose(res["pre"]["calibration_slope"], 1.0, atol=0.2)

    # Invariant: Post-recalibration slope is 1.0 and intercept is 0.0
    assert np.isclose(res["post"]["calibration_slope"], 1.0, atol=0.05)
    assert np.isclose(res["post"]["calibration_intercept"], 0.0, atol=0.05)

    # Invariant: Discrimination AUROC is strictly preserved
    assert np.isclose(res["post"]["auroc"], res["pre"]["auroc"], atol=1e-5)

    # Invariant: Brier score improves
    assert res["post"]["brier_score"] <= res["pre"]["brier_score"]


def test_recalibrate_prediction_model_intercept_only():
    """Verify intercept-only recalibration preserves slope and matches mean risk."""
    np.random.seed(42)
    n = 200
    x = np.random.normal(0, 1, size=n)
    y = (np.random.uniform(size=n) < 0.40).astype(int)

    # Model developed on population with 10% incidence applied to 40% cohort
    p_orig = np.clip(expit(-2.2 + 1.0 * x), 0.001, 0.999)

    res = recalibrate_prediction_model(y, p_orig, method="calibration_in_the_large")

    # Beta fixed at 1.0
    assert res["recalibration_parameters"]["beta"] == 1.0
    # Mean predicted probability matches event rate
    p_recal = res["p_recalibrated"]
    assert np.isclose(np.mean(p_recal), np.mean(y), atol=1e-3)
    # Discrimination AUROC is strictly preserved
    assert np.isclose(res["post"]["auroc"], res["pre"]["auroc"], atol=1e-5)


def test_recalibrate_prediction_model_both():
    """Verify method='both' returns both intercept-only and logistic updates."""
    np.random.seed(42)
    n = 100
    p = np.clip(np.random.uniform(0.1, 0.9, size=n), 0.01, 0.99)
    y = (np.random.uniform(size=n) < p).astype(int)

    res = recalibrate_prediction_model(y, p, method="both")
    assert "post_intercept_only" in res
    assert "post_logistic" in res
    assert "parameters_intercept_only" in res
    assert "parameters_logistic" in res
    assert "p_recalibrated_intercept_only" in res
    assert "p_recalibrated_logistic" in res


def test_recalibrate_prediction_model_method_selection():
    """Verify p_recalibrated respects method selection while providing both arrays."""
    np.random.seed(42)
    n = 200
    x = np.random.normal(0, 1, size=n)
    y = (np.random.uniform(size=n) < expit(-0.5 + 1.2 * x)).astype(int)
    # Miscalibrated probabilities
    p = np.clip(expit(-1.5 + 2.0 * x), 0.01, 0.99)

    # 1. method='intercept_only'
    res_int = recalibrate_prediction_model(y, p, method="intercept_only")
    assert "p_recalibrated_intercept_only" in res_int
    assert "p_recalibrated_logistic" in res_int
    assert np.array_equal(
        res_int["p_recalibrated"], res_int["p_recalibrated_intercept_only"]
    )
    assert not np.allclose(
        res_int["p_recalibrated"], res_int["p_recalibrated_logistic"]
    )

    # 2. method='calibration_in_the_large'
    res_citl = recalibrate_prediction_model(y, p, method="calibration_in_the_large")
    assert np.array_equal(
        res_citl["p_recalibrated"], res_citl["p_recalibrated_intercept_only"]
    )

    # 3. method='logistic'
    res_log = recalibrate_prediction_model(y, p, method="logistic")
    assert "p_recalibrated_intercept_only" in res_log
    assert "p_recalibrated_logistic" in res_log
    assert np.array_equal(res_log["p_recalibrated"], res_log["p_recalibrated_logistic"])
    assert not np.allclose(
        res_log["p_recalibrated"], res_log["p_recalibrated_intercept_only"]
    )

    # 4. method='both'
    res_both = recalibrate_prediction_model(y, p, method="both")
    assert np.array_equal(
        res_both["p_recalibrated"], res_both["p_recalibrated_logistic"]
    )
    assert "p_recalibrated_intercept_only" in res_both
    assert "p_recalibrated_logistic" in res_both


def test_recalibrate_prediction_model_errors():
    """Verify bounds and shape validation for recalibration."""
    # Length mismatch
    with pytest.raises(ValueError, match="Length mismatch"):
        recalibrate_prediction_model([0, 1, 0], [0.1, 0.2])

    # Out of bounds probabilities
    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        recalibrate_prediction_model([0, 1, 0] * 5, [-0.1, 0.5, 0.6] * 5)
    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        recalibrate_prediction_model([0, 1, 0] * 5, [0.1, 1.5, 0.6] * 5)

    # Non-finite values in y_true or p_pred
    y_nan = np.array([0, 1, 0, 1, np.nan, 0, 1, 0, 1, 0])
    p_valid = np.linspace(0.1, 0.9, 10)
    with pytest.raises(
        ValueError,
        match="Non-finite values detected in input arrays: y_true has 1 non-finite value",
    ):
        recalibrate_prediction_model(y_nan, p_valid)

    y_valid = np.array([0, 1] * 5)
    p_inf = np.array([0.1, 0.2, np.inf, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    with pytest.raises(
        ValueError,
        match="Non-finite values detected in input arrays: .* p_pred has 1 non-finite value",
    ):
        recalibrate_prediction_model(y_valid, p_inf)

    # Invalid clip_eps
    with pytest.raises(
        ValueError,
        match="clip_eps .* must be a finite number strictly between 0 and 0.5",
    ):
        recalibrate_prediction_model(y_valid, p_valid, clip_eps=0.0)
    with pytest.raises(
        ValueError,
        match="clip_eps .* must be a finite number strictly between 0 and 0.5",
    ):
        recalibrate_prediction_model(y_valid, p_valid, clip_eps=-0.01)
    with pytest.raises(
        ValueError,
        match="clip_eps .* must be a finite number strictly between 0 and 0.5",
    ):
        recalibrate_prediction_model(y_valid, p_valid, clip_eps=0.5)
    with pytest.raises(
        ValueError,
        match="clip_eps .* must be a finite number strictly between 0 and 0.5",
    ):
        recalibrate_prediction_model(y_valid, p_valid, clip_eps=0.7)
    with pytest.raises(
        ValueError,
        match="clip_eps .* must be a finite number strictly between 0 and 0.5",
    ):
        recalibrate_prediction_model(y_valid, p_valid, clip_eps=float("nan"))

    # Test recalibration narrative with method == 'both'
    recal_res = recalibrate_prediction_model(y_valid, p_valid, method="both")
    narr_both = generate_recalibration_narrative(recal_res)
    assert (
        "both intercept-only (calibration-in-the-large) and logistic recalibration updating"
        in narr_both
    )
    assert "apparent in-sample estimates" in narr_both


# ==============================================================================
# 3. Riley et al. (2020) Minimum Sample Size for Clinical Prediction Models
# ==============================================================================


def test_riley_sample_size_benchmark_criterion_3():
    """
    Verify Riley et al. (BMJ 2020; 368:m441) textbook benchmark:
    P = 10 candidate parameters, prevalence phi = 0.20, Cox-Snell R2 = 0.15,
    shrinkage target S = 0.90 -> Criterion 3 requires N = 549.
    """
    res = calculate_riley_sample_size(
        n_parameters=10,
        prevalence=0.20,
        cox_snell_r2=0.15,
        shrinkage_target=0.90,
    )

    n_crit3 = res["criteria"]["criterion3_shrinkage"]["n"]
    assert n_crit3 == 549, f"Expected N3 = 549 for Riley benchmark, got {n_crit3}"
    assert res["n_total"] >= 549
    assert res["n_events"] == int(math.ceil(res["n_total"] * 0.20))
    assert res["events_per_parameter"] == pytest.approx(res["n_events"] / 10.0)


def test_riley_sample_size_from_c_statistic():
    """Verify Riley derivation of Cox-Snell R2 from C-statistic using numerical quadrature."""
    res = calculate_riley_sample_size(
        n_parameters=8,
        prevalence=0.10,
        c_statistic=0.80,
        shrinkage_target=0.90,
    )
    assert res["cox_snell_r2"] > 0
    assert res["nagelkerke_r2"] > 0
    assert res["n_total"] > 0
    assert "binding_criterion" in res
    assert res["binding_criterion"] in [
        "Criterion 1 (Intercept)",
        "Criterion 2 (MAPE)",
        "Criterion 3 (Shrinkage)",
        "Criterion 4 (Optimism)",
    ]


def test_approximate_cox_snell_r2_from_c_stat_accuracy():
    """Verify Riley et al. (2021) Gauss-Hermite quadrature c-statistic to R2_CS conversion."""
    # Benchmark tests matching Riley et al. (Stat Med 2021)
    # c=0.80, phi=0.10 -> R2_CS ~ 0.11149
    r2_1 = _approximate_cox_snell_r2_from_c_stat(0.80, 0.10)
    assert np.isclose(r2_1, 0.11149, atol=1e-4)

    # c=0.70, phi=0.20 -> R2_CS ~ 0.08070
    r2_2 = _approximate_cox_snell_r2_from_c_stat(0.70, 0.20)
    assert np.isclose(r2_2, 0.08070, atol=1e-4)

    # c=0.80, phi=0.20 -> R2_CS ~ 0.18272
    r2_3 = _approximate_cox_snell_r2_from_c_stat(0.80, 0.20)
    assert np.isclose(r2_3, 0.18272, atol=1e-4)

    # Monotonicity: higher c-statistic with same prevalence yields higher R2_CS
    r2_high_c = _approximate_cox_snell_r2_from_c_stat(0.85, 0.20)
    assert r2_high_c > r2_3

    # Error handling on invalid bounds
    with pytest.raises(ValueError, match="c_statistic"):
        _approximate_cox_snell_r2_from_c_stat(0.50, 0.20)
    with pytest.raises(ValueError, match="c_statistic"):
        _approximate_cox_snell_r2_from_c_stat(1.00, 0.20)
    with pytest.raises(ValueError, match="prevalence"):
        _approximate_cox_snell_r2_from_c_stat(0.80, 0.00)
    with pytest.raises(ValueError, match="prevalence"):
        _approximate_cox_snell_r2_from_c_stat(0.80, 1.00)


def test_riley_sample_size_from_nagelkerke():
    """Verify Riley derivation of Cox-Snell R2 from Nagelkerke R2."""
    res = calculate_riley_sample_size(
        n_parameters=5,
        prevalence=0.25,
        nagelkerke_r2=0.30,
        shrinkage_target=0.90,
    )
    assert res["cox_snell_r2"] == pytest.approx(0.30 * res["r2_max"], rel=1e-5)
    assert res["nagelkerke_r2"] == pytest.approx(0.30, rel=1e-5)


def test_riley_sample_size_input_validation():
    """Verify parameter bounds and errors."""
    with pytest.raises(ValueError, match="n_parameters must be >= 1"):
        calculate_riley_sample_size(n_parameters=0, prevalence=0.2, c_statistic=0.7)
    with pytest.raises(ValueError, match="prevalence must be strictly in"):
        calculate_riley_sample_size(n_parameters=5, prevalence=1.2, c_statistic=0.7)
    with pytest.raises(ValueError, match="at least one model strength measure"):
        calculate_riley_sample_size(n_parameters=5, prevalence=0.2)
    # When Cox-Snell R2 exceeds or equals shrinkage target
    with pytest.raises(ValueError, match="must be strictly greater than"):
        calculate_riley_sample_size(
            n_parameters=10, prevalence=0.5, cox_snell_r2=0.70, shrinkage_target=0.65
        )
    # Extreme C-statistic should evaluate without OverflowError or math domain error
    res_extreme = calculate_riley_sample_size(
        n_parameters=5, prevalence=0.20, c_statistic=0.98, shrinkage_target=0.95
    )
    assert res_extreme["n_total"] > 0
    # Prevalence > 0.5 with MAPE target (symmetric van Smeden formula)
    res_high_prev = calculate_riley_sample_size(
        n_parameters=8, prevalence=0.80, cox_snell_r2=0.20, mape_target=0.05
    )
    assert res_high_prev["criteria"]["criterion2_mape"]["n"] is not None


# ==============================================================================
# 4. Obuchowski Diagnostic AUROC Sample Size
# ==============================================================================


def test_diagnostic_auroc_sample_size_obuchowski():
    """Verify Obuchowski / Hanley-McNeil AUROC sample size calculations."""
    res = calculate_diagnostic_auroc_sample_size(
        auroc_expected=0.80,
        auroc_null=0.50,
        prevalence=0.30,
        alpha=0.05,
        power=0.80,
        two_sided=True,
    )

    assert res["n_cases"] > 0
    assert res["n_controls"] > 0
    assert res["total_n"] == res["n_cases"] + res["n_controls"]
    assert res["allocation_ratio_controls_to_cases"] == pytest.approx(
        (1.0 - 0.30) / 0.30
    )

    # Monotonicity tests:
    # 1. Higher power requires larger sample size
    res_power90 = calculate_diagnostic_auroc_sample_size(
        auroc_expected=0.80, auroc_null=0.50, prevalence=0.30, power=0.90
    )
    assert res_power90["total_n"] > res["total_n"]

    # 2. Smaller effect size (closer to null) requires larger sample size
    res_smaller_effect = calculate_diagnostic_auroc_sample_size(
        auroc_expected=0.65, auroc_null=0.50, prevalence=0.30, power=0.80
    )
    assert res_smaller_effect["total_n"] > res["total_n"]


def test_diagnostic_auroc_sample_size_obuchowski_benchmark():
    """Verify published Obuchowski (1997) / Hanley-McNeil benchmark in calculate_diagnostic_auroc_sample_size."""
    res = calculate_diagnostic_auroc_sample_size(
        auroc_expected=0.70,
        auroc_null=0.50,
        prevalence=0.50,
        alpha=0.05,
        power=0.80,
        two_sided=True,
    )
    # Under Obuchowski (1997) / Hanley-McNeil asymptotic variance per case:
    # V0 = (Q2_0 - 0.25) + (Q1_0 - 0.25) = 1/6 ~ 0.1667
    # V1 = (Q2_1 - 0.49) + (Q1_1 - 0.49) ~ 0.1349
    # (z_alpha * sqrt(V0) + z_beta * sqrt(V1))^2 / (0.7 - 0.5)^2 = 30.76 -> 31 cases
    assert res["n_cases"] == 31
    assert res["n_controls"] == 31
    assert res["total_n"] == 62

    # Asymmetric prevalence benchmark (phi = 0.25, R = 3.0, theta_0 = 0.50, theta_1 = 0.70, alpha = 0.05 two-sided, power = 0.80)
    # Independently compute Q1, Q2:
    # Q1(0.50) = 1/3, Q2(0.50) = 1/3 -> V(theta_0) = (1/12) + (1/12)/3.0 = 1/9
    # Q1(0.70) = 7/13, Q2(0.70) = 49/85 -> V(theta_1) = (49/85 - 0.49) + (7/13 - 0.49)/3.0 ~= 0.102624434
    # (z_alpha * sqrt(V0) + z_beta * sqrt(V1))^2 / (0.7 - 0.5)^2 ~= 21.295 -> n_cases = 22, n_controls = 66, total = 88.
    res_asym = calculate_diagnostic_auroc_sample_size(
        auroc_expected=0.70,
        auroc_null=0.50,
        prevalence=0.25,
        alpha=0.05,
        power=0.80,
        two_sided=True,
    )
    assert res_asym["n_cases"] == 22
    assert res_asym["n_controls"] == 66
    assert res_asym["total_n"] == 88

    # Verify one-sided vs two-sided narrative generation
    narr_two = generate_diagnostic_auroc_sample_size_narrative(res)
    assert "two-sided" in narr_two

    res_one = calculate_diagnostic_auroc_sample_size(
        auroc_expected=0.70,
        auroc_null=0.50,
        prevalence=0.50,
        alpha=0.05,
        power=0.80,
        two_sided=False,
    )
    narr_one = generate_diagnostic_auroc_sample_size_narrative(res_one)
    assert "one-sided" in narr_one


# ==============================================================================
# 5. Dual Figures: Recalibration and Bootstrap Optimism
# ==============================================================================


def test_recalibration_and_optimism_figure_generators(tmp_path: Path):
    """Verify static and interactive HTML figure generation for validation & recalibration."""
    np.random.seed(42)
    n = 150
    x = np.random.normal(0, 1, size=n)
    y = (np.random.uniform(size=n) < expit(0.4 * x)).astype(int)
    p = np.clip(expit(-0.5 + 1.2 * x), 0.01, 0.99)

    # Recalibration figures
    recal_res = recalibrate_prediction_model(y, p, method="both")
    out_png_recal = tmp_path / "recalibration.png"
    fig_recal = plot_recalibration_comparison(recal_res, out_path=str(out_png_recal))
    assert Path(fig_recal.png_path).exists()
    assert fig_recal.source_df is not None

    dual_recal = plot_dual_recalibration_comparison(
        recal_res, out_path=str(out_png_recal)
    )
    assert dual_recal.plotly_spec is not None
    assert "data" in dual_recal.plotly_spec
    assert "layout" in dual_recal.plotly_spec

    # Bootstrap optimism figures
    df_val = pd.DataFrame({"y": y, "x": x})
    val_res = bootstrap_internal_validation(
        df_val, "y", ["x"], n_bootstrap=15, random_state=42
    )
    out_png_opt = tmp_path / "optimism.png"
    fig_opt = plot_bootstrap_optimism(val_res, out_path=str(out_png_opt))
    assert Path(fig_opt.png_path).exists()
    assert fig_opt.source_df is not None

    dual_opt = plot_dual_bootstrap_optimism(val_res, out_path=str(out_png_opt))
    assert dual_opt.plotly_spec is not None
    assert "data" in dual_opt.plotly_spec


# ==============================================================================
# 6. Reporting Tables & Academic Narratives
# ==============================================================================


def test_reporting_tables_and_narratives():
    """Verify publication table formatting and academic methods/results narratives."""
    np.random.seed(42)
    n = 120
    x = np.random.normal(0, 1, size=n)
    y = (np.random.uniform(size=n) < expit(0.5 * x)).astype(int)
    p = np.clip(expit(0.5 * x), 0.01, 0.99)
    df_cohort = pd.DataFrame({"outcome": y, "x": x})

    # 1. Internal validation table & narrative
    val_res = bootstrap_internal_validation(
        df_cohort, "outcome", ["x"], n_bootstrap=15, random_state=42
    )
    tbl_val = format_internal_validation_table(val_res, style="NEJM")
    assert "<table" in tbl_val
    assert "C-statistic" in tbl_val
    assert "Shrunk (ŝ =" in tbl_val

    narr_val = generate_internal_validation_narrative(val_res)
    assert "TRIPOD+AI" in narr_val
    assert "optimism-corrected" in narr_val

    # 2. Recalibration table & narrative
    recal_res = recalibrate_prediction_model(y, p, method="both")
    tbl_recal = format_recalibration_table(recal_res, style="JAMA")
    assert "<table" in tbl_recal
    assert "Pre-Recalibration" in tbl_recal
    assert "Post-Recalibration" in tbl_recal

    narr_recal = generate_recalibration_narrative(recal_res)
    assert "recalibration" in narr_recal.lower()

    # 3. Riley sample size table & narrative
    riley_res = calculate_riley_sample_size(
        n_parameters=10, prevalence=0.20, cox_snell_r2=0.15
    )
    tbl_riley = format_riley_sample_size_table(riley_res, style="NEJM")
    assert "<table" in tbl_riley
    assert "Criterion 3" in tbl_riley
    assert "Binding" in tbl_riley

    narr_riley = generate_riley_sample_size_narrative(riley_res)
    assert "Riley et al." in narr_riley
    assert "minimum sample size" in narr_riley

    # 4. Diagnostic AUROC narrative
    auroc_res = calculate_diagnostic_auroc_sample_size(0.85, 0.50, prevalence=0.30)
    narr_auroc = generate_diagnostic_auroc_sample_size_narrative(auroc_res)
    assert "Obuchowski" in narr_auroc
    assert "AUROC" in narr_auroc

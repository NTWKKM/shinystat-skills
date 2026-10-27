"""
tests/unit/test_recipes_snapshot.py: Baseline snapshot capture and regression test
for all 13 biostatistical recipes in python-recipes.md.
"""

# ruff: noqa: E402

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.synthetic_clinical import generate_synthetic_clinical_cohort

RECIPES_PATH = (
    REPO_ROOT
    / "packaging"
    / "cloud"
    / "shinystat-cloud"
    / "references"
    / "python-recipes.md"
)
SNAPSHOT_PATH = REPO_ROOT / "tests" / "fixtures" / "recipes_baseline_snapshot.json"


def extract_recipes_namespace() -> dict[str, Any]:
    """Execute code blocks in modular references/recipes/*.md (or python-recipes.md) in an isolated namespace."""
    recipes_dir = RECIPES_PATH.parent / "recipes"
    if recipes_dir.exists() and list(recipes_dir.glob("*.md")):
        parts = [
            p.read_text(encoding="utf-8") for p in sorted(recipes_dir.glob("*.md"))
        ]
        content = "\n\n".join(parts)
    else:
        content = RECIPES_PATH.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)\n```", content, re.DOTALL)
    ns = {}
    combined_code = "\n".join(blocks)
    exec(combined_code, ns)
    return ns


def _serialize_val(val: Any) -> Any:
    """Helper to convert NumPy/Pandas objects into JSON-serializable structures."""
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        return None if np.isnan(val) else round(float(val), 6)
    if isinstance(val, (np.ndarray, list, tuple)):
        return [_serialize_val(v) for v in val]
    if isinstance(val, dict):
        return {k: _serialize_val(v) for k, v in val.items()}
    if isinstance(val, pd.DataFrame):
        return val.fillna("").to_dict(orient="records")
    return str(val)


def run_all_recipes(ns: dict[str, Any]) -> dict[str, Any]:
    """Run all 13 recipes deterministically and return their baseline output dict."""
    df = generate_synthetic_clinical_cohort(n=500, seed=42)
    results = {}

    # Recipe 1: Table 1 & SMD
    t1_df = ns["generate_table_one"](
        df=df,
        strata="treatment_arm",
        continuous_vars=["age", "bmi", "egfr"],
        categorical_vars=["sex", "hypertension", "diabetes"],
    )
    results["recipe_1_table_one"] = _serialize_val(t1_df)

    # Recipe 2: 2x2 Diagnostics
    # AMI vs high troponin (e.g. > 10.0)
    pred_pos = (df["troponin_level"] >= 10.0).astype(int)
    gold = df["gold_standard_ami"].astype(int)
    tp = int(np.sum((pred_pos == 1) & (gold == 1)))
    fp = int(np.sum((pred_pos == 1) & (gold == 0)))
    fn = int(np.sum((pred_pos == 0) & (gold == 1)))
    tn = int(np.sum((pred_pos == 0) & (gold == 0)))
    r2_res = ns["calculate_2x2_metrics"](tp, fp, fn, tn)
    results["recipe_2_diagnostic_2x2"] = _serialize_val(r2_res)

    # Recipe 3: Empirical ROC & DeLong
    r3_res = ns["auc_ci_delong"](gold.to_numpy(), df["troponin_level"].to_numpy())
    results["recipe_3_roc_delong"] = _serialize_val(
        {k: v for k, v in r3_res.items() if not isinstance(v, np.ndarray)}
    )

    # Recipe 4: Calibration & DCA
    pred_probs = 1.0 / (
        1.0 + np.exp(-(-2.0 + 0.03 * df["age"] + 0.05 * df["troponin_level"]))
    )
    r4_calib = ns["evaluate_calibration"](gold.to_numpy(), pred_probs.to_numpy())
    r4_dca = ns["calculate_dca"](gold.to_numpy(), pred_probs.to_numpy())
    results["recipe_4_calibration"] = _serialize_val(r4_calib)
    results["recipe_4_dca_head"] = _serialize_val(r4_dca.head(5))

    # Recipe 5: Bland-Altman & ICC
    r5_ba = ns["calculate_bland_altman"](
        df["sbp_device_a"].to_numpy(), df["sbp_device_b"].to_numpy()
    )
    rater_matrix = df[["rater_1", "rater_2"]].to_numpy()
    r5_icc = ns["calculate_icc_matrix"](rater_matrix)
    results["recipe_5_bland_altman"] = _serialize_val(r5_ba)
    results["recipe_5_icc"] = _serialize_val(r5_icc)

    # Recipe 6: PSM & Covariate Balance
    r6_psm = ns["match_propensity_scores"](
        df=df,
        treatment_col="treatment_arm",
        confounders=["age", "bmi", "egfr", "hypertension", "diabetes"],
        caliper_sd=0.2,
    )
    results["recipe_6_psm_summary"] = {
        "n_matched": len(r6_psm["matched_df"]),
        "balance_df": _serialize_val(r6_psm["balance_df"]),
    }

    # Recipe 7: Firth Logistic
    X_firth = df[["age", "diabetes", "treatment_arm"]].to_numpy()
    y_firth = df["mortality_30d"].to_numpy()
    r7_firth = ns["fit_firth_logistic"](X_firth, y_firth)
    results["recipe_7_firth"] = _serialize_val(
        {k: v for k, v in r7_firth.items() if k != "cov_matrix"}
    )

    # Recipe 8: Little's MCAR
    r8_mcar = ns["littles_mcar_test"](df, ["age", "bmi", "lab_crp", "lab_ldl"])
    results["recipe_8_mcar"] = _serialize_val(r8_mcar)

    # Recipe 9: E-Value
    r9_eval = ns["calculate_e_value"](
        estimate=1.85, lower=1.20, upper=2.85, estimate_type="RR"
    )
    results["recipe_9_e_value"] = _serialize_val(r9_eval)

    # Recipe 10: Multivariable Logistic Regression Table
    r10_log = ns["fit_logistic_regression_table"](
        df=df,
        outcome="mortality_30d",
        covariates=["age", "sex", "diabetes", "treatment_arm"],
    )
    results["recipe_10_logistic_table"] = _serialize_val(r10_log)

    # Recipe 11: Survival Analysis Suite
    r11_surv = ns["fit_survival_analysis_suite"](
        df=df,
        duration_col="time_to_event",
        event_col="event_death",
        strata_col="treatment_arm",
        covariates=["age", "diabetes"],
    )
    results["recipe_11_survival"] = {
        "km_summary": _serialize_val(r11_surv["km_summary"]),
        "cox_table": _serialize_val(r11_surv["cox_table"]),
        "schoenfeld_no_violation": r11_surv["schoenfeld_diagnostics"][
            "no_ph_violation_detected"
        ],
    }

    # Recipe 12: MICE Imputation
    r12_mice = ns["impute_mice_single"](
        df=df[["age", "bmi", "lab_crp", "lab_ldl"]],
        features_to_impute=["lab_crp", "lab_ldl"],
        predictors=["age", "bmi"],
        max_iter=5,
        random_state=42,
    )
    results["recipe_12_mice_n_missing_after"] = int(r12_mice.isna().sum().sum())

    # Recipe 13: HTML Table Renderer
    sample_df = pd.DataFrame(
        {"Metric": ["Sensitivity", "Specificity"], "Value": ["88.5%", "92.1%"]}
    )
    r13_html = ns["render_publication_table"](sample_df, title="Table. Diagnostics")
    results["recipe_13_html_rendered"] = "<table" in r13_html

    # Recipe 14: IPTW & Weighted Balance
    r14_df, r14_diag = ns["compute_iptw"](
        df=df,
        treatment_col="treatment_arm",
        covariates=["age", "bmi", "egfr"],
        estimand="ATE",
    )
    r14_smd = ns["calculate_weighted_smd"](
        r14_df.loc[r14_df["treatment_arm"] == 1, "age"].to_numpy(),
        r14_df.loc[r14_df["treatment_arm"] == 0, "age"].to_numpy(),
        r14_df.loc[r14_df["treatment_arm"] == 1, "iptw_weight"].to_numpy(),
        r14_df.loc[r14_df["treatment_arm"] == 0, "iptw_weight"].to_numpy(),
    )
    r14_fit = ns["fit_weighted_outcome_model"](
        r14_df,
        outcome_col="mortality_30d",
        treatment_col="treatment_arm",
        weights_col="iptw_weight",
        model_type="binary",
    )
    results["recipe_14_iptw"] = {
        "n_eff": r14_diag["n_eff"],
        "age_weighted_smd": round(float(r14_smd), 4),
        "odds_ratio": round(float(r14_fit["odds_ratio"]), 4),
    }

    # Recipe 15: Competing Risks
    cr_status = np.where(df["event_death"] == 1, 1, np.where(df["age"] > 70, 2, 0))
    r15_cif = ns["fit_competing_risks_cif"](
        durations=df["time_to_event"].to_numpy(),
        event_observed=cr_status,
        event_of_interest=1,
        competing_event=2,
        horizons=[30.0, 90.0, 180.0],
    )
    df_cr_reg = df.copy()
    df_cr_reg["cr_status"] = cr_status
    r15_reg = ns["fit_competing_risks_regression"](
        df=df_cr_reg,
        duration_col="time_to_event",
        event_col="cr_status",
        covariates=["treatment_arm", "diabetes"],
        event_of_interest=1,
        competing_event=2,
    )
    results["recipe_15_competing_risks"] = {
        "cif_table_len": len(r15_cif["cif_table"]),
        "regression_results": _serialize_val(r15_reg),
    }

    # Recipe 16: Diagnostic Reclassification (Continuous & Categorical NRI + IDI)
    p_old = 1.0 / (1.0 + np.exp(-(-2.0 + 0.03 * df["age"])))
    p_new = pred_probs
    r16_res = ns["compute_reclassification_metrics"](
        y_true=gold.to_numpy(),
        p_old=p_old.to_numpy(),
        p_new=p_new.to_numpy(),
        thresholds=[0.1, 0.3],
        n_bootstrap=0,
    )
    results["recipe_16_reclassification"] = {
        "nri_continuous": _serialize_val(r16_res["nri_continuous"]),
        "nri_categorical": _serialize_val(r16_res["nri_categorical"]),
        "idi": _serialize_val(r16_res["idi"]),
    }

    # Recipe 17: Time-Dependent ROC (Cumulative/Dynamic IPCW AUROC(t))
    r17_res = ns["compute_time_dependent_roc"](
        durations=df["time_to_event"].to_numpy(),
        events=df["event_death"].to_numpy(),
        scores=df["age"].to_numpy(),
        time_horizons=[30.0, 90.0, 180.0],
        n_bootstrap=0,
    )
    results["recipe_17_time_dependent_roc"] = {
        "horizons": _serialize_val(r17_res["time_horizons"]),
        "trajectory": _serialize_val(r17_res["trajectory"]),
        "auroc_30": _serialize_val(r17_res["results_by_horizon"][30.0]["auroc"]),
        "auroc_90": _serialize_val(r17_res["results_by_horizon"][90.0]["auroc"]),
        "auroc_180": _serialize_val(r17_res["results_by_horizon"][180.0]["auroc"]),
    }

    # Recipe 18: TRIPOD+AI Bootstrap Internal Validation & Uniform Shrinkage
    r18_res = ns["bootstrap_internal_validation"](
        df=df,
        outcome="gold_standard_ami",
        predictors=["age", "bmi", "troponin_level"],
        n_bootstrap=10,
        random_state=42,
    )
    results["recipe_18_validation"] = {
        "shrinkage_factor": _serialize_val(r18_res["shrinkage_factor"]),
        "apparent_auroc": _serialize_val(r18_res["apparent_metrics"]["auroc"]),
        "corrected_auroc": _serialize_val(
            r18_res["metrics"]["auroc"]["optimism_corrected"]
        ),
    }

    # Recipe 19: Prediction Model Recalibration
    r19_res = ns["recalibrate_prediction_model"](
        y_true=gold.to_numpy(),
        p_pred=pred_probs.to_numpy(),
        method="logistic",
    )
    results["recipe_19_recalibration"] = {
        "alpha": _serialize_val(r19_res["recalibration_parameters"]["alpha"]),
        "beta": _serialize_val(r19_res["recalibration_parameters"]["beta"]),
        "post_brier": _serialize_val(r19_res["post"]["brier_score"]),
    }

    # Recipe 20: Riley Prediction & Diagnostic AUROC Sample Size
    r20_riley = ns["calculate_riley_sample_size"](
        n_parameters=10,
        prevalence=0.20,
        cox_snell_r2=0.15,
        shrinkage_target=0.90,
    )
    r20_auroc = ns["calculate_diagnostic_auroc_sample_size"](
        auroc_expected=0.85,
        auroc_null=0.50,
        prevalence=0.30,
        alpha=0.05,
        power=0.80,
    )
    results["recipe_20_sample_size"] = {
        "riley_n_total": _serialize_val(r20_riley["n_total"]),
        "riley_criterion3_n": _serialize_val(
            r20_riley["criteria"]["criterion3_shrinkage"]["n"]
        ),
        "auroc_total_n": _serialize_val(r20_auroc["total_n"]),
        "auroc_n_cases": _serialize_val(r20_auroc["n_cases"]),
    }

    return results


def test_capture_or_verify_baseline_snapshot():
    """Verify that current recipe executions match baseline snapshot exactly."""
    ns = extract_recipes_namespace()
    current_results = run_all_recipes(ns)

    if not SNAPSHOT_PATH.exists() or os.environ.get("UPDATE_RECIPES_SNAPSHOT") == "1":
        if os.environ.get("UPDATE_RECIPES_SNAPSHOT") == "1":
            SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
            SNAPSHOT_PATH.write_text(
                json.dumps(current_results, indent=2), encoding="utf-8"
            )
            assert SNAPSHOT_PATH.exists()
            print(f"Captured initial baseline snapshot at {SNAPSHOT_PATH}")
            return
        pytest.fail(
            f"Baseline snapshot missing at {SNAPSHOT_PATH}. "
            "To capture or update baseline snapshot, run with UPDATE_RECIPES_SNAPSHOT=1"
        )

    # Subsequent runs: assert zero regression on all baseline keys
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    for recipe_key, baseline_val in snapshot.items():
        assert recipe_key in current_results, f"Missing {recipe_key} in current results"
        curr_val = current_results[recipe_key]
        if isinstance(baseline_val, dict):
            for k, expected_v in baseline_val.items():
                assert k in curr_val, f"Baseline key '{k}' dropped in {recipe_key}!"
                assert curr_val[k] == expected_v, (
                    f"Regression in {recipe_key}['{k}']: {curr_val[k]} != {expected_v}"
                )
        else:
            assert curr_val == baseline_val, f"Regression detected in {recipe_key}!"

    # Assert new additive plot-ready keys are present
    r2_raw = ns["calculate_2x2_metrics"](10, 5, 2, 80)
    assert "contingency_matrix" in r2_raw
    assert "confusion_table" in r2_raw

    r3_raw = ns["auc_ci_delong"]([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
    assert "roc_curve" in r3_raw
    assert "optimal_point" in r3_raw

    r4_raw = ns["evaluate_calibration"]([1, 1, 0, 0] * 5, [0.8, 0.7, 0.3, 0.2] * 5)
    assert "calibration_bins" in r4_raw
    assert "calibration_curve" in r4_raw

    r5_raw = ns["calculate_bland_altman"]([10, 20, 30], [11, 22, 29])
    assert "ba_points" in r5_raw

    r6_raw = ns["match_propensity_scores"](
        pd.DataFrame({"t": [1, 1, 0, 0], "x": [10, 20, 11, 19]}),
        treatment_col="t",
        confounders=["x"],
        caliper_sd=0.5,
    )
    assert "love_plot_data" in r6_raw

    r7_raw = ns["fit_firth_logistic"](
        np.array([[1.0], [2.0], [3.0], [4.0]]), np.array([0, 0, 1, 1])
    )
    assert "forest_data" in r7_raw

    r8_raw = ns["littles_mcar_test"](
        pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [np.nan, 2.0, 3.0]}), ["a", "b"]
    )
    assert "missingness_summary" in r8_raw

    r16_raw = ns["compute_reclassification_metrics"](
        [1, 1, 0, 0], [0.8, 0.7, 0.3, 0.2], [0.9, 0.8, 0.2, 0.1], thresholds=[0.5]
    )
    assert "nri_continuous" in r16_raw
    assert "nri_categorical" in r16_raw
    assert "idi" in r16_raw

    r17_raw = ns["compute_time_dependent_roc"](
        [10.0, 20.0, 30.0, 40.0] * 3,
        [1, 0, 1, 0] * 3,
        [0.8, 0.2, 0.7, 0.3] * 3,
        time_horizons=[25.0],
    )
    assert "trajectory" in r17_raw
    assert "results_by_horizon" in r17_raw

    r18_raw = ns["bootstrap_internal_validation"](
        pd.DataFrame({"y": [0, 1, 0, 1] * 10, "x1": np.linspace(0, 1, 40)}),
        outcome="y",
        predictors=["x1"],
        n_bootstrap=10,
        random_state=42,
    )
    assert "apparent_metrics" in r18_raw
    assert "shrinkage_factor" in r18_raw
    assert "shrunk_coefficients" in r18_raw

    r19_raw = ns["recalibrate_prediction_model"](
        [0, 1, 0, 1] * 10,
        np.clip(np.linspace(0.1, 0.9, 40), 0.01, 0.99),
        method="both",
    )
    assert "post_intercept_only" in r19_raw
    assert "post_logistic" in r19_raw

    r20_riley = ns["calculate_riley_sample_size"](
        n_parameters=5, prevalence=0.20, cox_snell_r2=0.10
    )
    assert "binding_criterion" in r20_riley
    assert "criterion3_shrinkage" in r20_riley["criteria"]

    r20_auroc = ns["calculate_diagnostic_auroc_sample_size"](
        auroc_expected=0.80, auroc_null=0.50, prevalence=0.30
    )
    assert "n_cases" in r20_auroc
    assert "n_controls" in r20_auroc

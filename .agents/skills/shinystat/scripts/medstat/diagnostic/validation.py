"""
TRIPOD+AI Model Validation, Uniform Shrinkage, and Recalibration.

Provides:
- TRIPOD+AI optimism-corrected bootstrap internal validation (Efron-Gong optimism).
- Uniform shrinkage factor calculation (s_hat) and re-estimated intercept.
- Model recalibration (intercept-only / calibration-in-the-large, logistic recalibration, both).
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy.optimize as opt
import statsmodels.api as sm
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score


def _compute_cox_snell_and_nagelkerke(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    eps: float = 1e-15,
) -> tuple[float, float]:
    """
    Compute Cox-Snell R2 and Nagelkerke R2 for binary predictions against true outcomes.
    """
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0

    p_clipped = np.clip(p_pred, eps, 1.0 - eps)
    prev = float(np.mean(y_true))
    prev_clipped = np.clip(prev, eps, 1.0 - eps)

    # Log-likelihood of null model (intercept only)
    ll_null = float(
        np.sum(
            y_true * math.log(prev_clipped)
            + (1.0 - y_true) * math.log(1.0 - prev_clipped)
        )
    )

    # Log-likelihood of model
    ll_model = float(
        np.sum(y_true * np.log(p_clipped) + (1.0 - y_true) * np.log(1.0 - p_clipped))
    )

    # R2_CS = 1 - exp(-2/N * (ll_model - ll_null))
    lr = 2.0 * (ll_model - ll_null)
    r2_cs = max(0.0, 1.0 - math.exp(-lr / n))

    # R2_max = 1 - exp(2 * ll_null / n)
    r2_max = max(1e-12, 1.0 - math.exp(2.0 * ll_null / n))
    r2_nag = min(1.0, max(0.0, r2_cs / r2_max))

    return float(r2_cs), float(r2_nag)


def _compute_calibration_intercept_offset(
    y_true: np.ndarray,
    lp: np.ndarray,
) -> float:
    """
    Calibration-in-the-large intercept: logit(Y) = alpha + 1 * LP.
    Solves sum(y_i - expit(alpha + LP_i)) = 0 via 1D monotone root finding.
    """
    y_arr = np.asarray(y_true, dtype=float)
    lp_arr = np.asarray(lp, dtype=float)
    n = len(y_arr)
    sum_y = float(np.sum(y_arr))

    if sum_y <= 0:
        return -50.0
    if sum_y >= n:
        return 50.0

    def obj(alpha: float) -> float:
        return float(np.sum(expit(alpha + lp_arr)) - sum_y)

    try:
        sol = opt.brentq(obj, -50.0, 50.0)
        return float(sol)
    except Exception:
        # Fallback to statsmodels Logit with offset
        try:
            mod = sm.Logit(y_arr, np.ones(n), offset=lp_arr).fit(disp=False)
            return float(mod.params[0])
        except Exception:
            return 0.0


def _compute_calibration_slope(
    y_true: np.ndarray,
    lp: np.ndarray,
) -> float:
    """
    Calibration slope: logit(Y) = alpha + beta * LP.
    """
    y_arr = np.asarray(y_true, dtype=float)
    lp_arr = np.asarray(lp, dtype=float)

    # Guard against zero-variance LP
    if np.var(lp_arr) <= 1e-12 or np.allclose(lp_arr, lp_arr[0]):
        return float("nan")

    X = sm.add_constant(lp_arr)
    try:
        mod = sm.Logit(y_arr, X).fit(disp=False)
        if not getattr(mod, "mle_retvals", {}).get("converged", True):
            return float("nan")
        val = float(mod.params[1])
        if not np.isfinite(val):
            return float("nan")
        return val
    except Exception:
        return float("nan")


def _find_reestimated_intercept(
    y_true: np.ndarray,
    lp_shrunk_no_intercept: np.ndarray,
) -> float:
    """
    Find intercept alpha_0 such that mean(expit(alpha_0 + lp_shrunk_no_intercept)) == mean(y_true).
    """
    y_arr = np.asarray(y_true, dtype=float)
    lp_arr = np.asarray(lp_shrunk_no_intercept, dtype=float)
    n = len(y_arr)
    target = float(np.sum(y_arr))

    if target <= 0:
        return -50.0
    if target >= n:
        return 50.0

    def obj(alpha: float) -> float:
        return float(np.sum(expit(alpha + lp_arr)) - target)

    try:
        sol = opt.brentq(obj, -50.0, 50.0)
        return float(sol)
    except Exception:
        prev = np.clip(target / n, 1e-6, 1.0 - 1e-6)
        return float(logit(prev) - np.mean(lp_arr))


def bootstrap_internal_validation(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    n_bootstrap: int = 200,
    penalty: str = "none",
    random_state: int = 42,
) -> dict[str, Any]:
    """
    TRIPOD+AI Optimism-Corrected Bootstrap Internal Validation, Uniform Shrinkage,
    and Shrunk Model Coefficients.

    Parameters:
        df: Input DataFrame containing outcome and predictor columns.
        outcome: Binary outcome column name (must be coded 0/1).
        predictors: List of predictor column names.
        n_bootstrap: Number of bootstrap replicates (default: 200).
        penalty: Regularization penalty ('none' or 'firth').
        random_state: Random number seed for reproducible sampling.

    Returns:
        dict: Comprehensive internal validation dictionary containing:
            - n_total, n_events, n_predictors, events_per_predictor (EPP)
            - n_bootstrap, n_successful_replicates
            - metrics: apparent, optimism, optimism_corrected, boot_mean, test_mean
              for auroc, brier_score, nagelkerke_r2, calibration_intercept, calibration_slope
            - shrinkage_factor: uniform shrinkage factor s_hat
            - shrunk_coefficients: dictionary comparing original vs shrunk coefficients
            - warnings: clinical guardrail alerts (EPP < 10, severe overfitting, convergence)
    """
    if outcome not in df.columns:
        raise ValueError(f"Outcome column '{outcome}' not found in DataFrame.")
    for p in predictors:
        if p not in df.columns:
            raise ValueError(f"Predictor column '{p}' not found in DataFrame.")
    if len(predictors) == 0:
        raise ValueError("At least one predictor must be provided.")

    req_cols = [outcome] + list(predictors)
    missing_counts = {
        c: int(df[c].isna().sum()) for c in req_cols if df[c].isna().sum() > 0
    }
    if missing_counts:
        counts_str = ", ".join(f"{c} ({cnt})" for c, cnt in missing_counts.items())
        raise ValueError(
            f"Missing values detected in input columns: {counts_str}. "
            "Please perform complete-case filtering or multiple imputation before running internal validation."
        )

    sub_df = df[req_cols]
    if len(sub_df) < 10:
        raise ValueError(f"Insufficient non-missing observations (N = {len(sub_df)}).")

    y_orig = sub_df[outcome].to_numpy(dtype=float)
    unique_y = np.unique(y_orig)
    if not np.array_equal(unique_y, [0.0, 1.0]):
        raise ValueError(
            f"Outcome '{outcome}' must be strictly binary with both classes 0 and 1, got {unique_y}."
        )

    norm_penalty = str(penalty).lower().strip()
    if norm_penalty not in ("none", "firth"):
        raise ValueError(f"Unsupported penalty '{penalty}'. Must be 'none' or 'firth'.")

    # Check for categorical / non-float predictor dtypes (object, string, category, bool)
    pred_df = sub_df[predictors].copy()
    has_categorical = any(
        col_dtype.name in ("object", "category", "string", "bool")
        or isinstance(col_dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(col_dtype)
        or pd.api.types.is_string_dtype(col_dtype)
        or pd.api.types.is_object_dtype(col_dtype)
        for col_dtype in [pred_df[c].dtype for c in predictors]
    )

    if has_categorical:
        X_expanded = pd.get_dummies(pred_df, drop_first=True, dtype=float)
        model_predictors = list(X_expanded.columns)
        X_orig = X_expanded.to_numpy(dtype=float)
    else:
        model_predictors = list(predictors)
        X_orig = pred_df.to_numpy(dtype=float)

    n_total = len(y_orig)
    n_events = int(np.sum(y_orig))
    n_non_events = n_total - n_events
    n_minority = min(n_events, n_non_events)
    n_predictors = len(model_predictors)
    epp = float(n_minority / n_predictors) if n_predictors > 0 else 0.0

    # 1. Fit original model on full dataset
    X_orig_const = sm.add_constant(X_orig, has_constant="add")
    if norm_penalty == "firth":
        from medstat.models.firth import fit_firth_logistic

        try:
            firth_res = fit_firth_logistic(
                y=y_orig, X=X_orig, feature_names=model_predictors, fit_intercept=True
            )
            orig_params = np.asarray(firth_res["model"].coef_).ravel()
        except Exception as e:
            raise RuntimeError(
                f"Failed to fit initial Firth logistic model: {e}"
            ) from e
    else:
        try:
            orig_model = sm.Logit(y_orig, X_orig_const).fit(disp=False)
            orig_params = orig_model.params
        except Exception as e:
            raise RuntimeError(f"Failed to fit initial logistic model: {e}") from e

    lp_orig = np.dot(X_orig_const, orig_params)
    p_orig = expit(lp_orig)

    # Apparent metrics on original derivation data
    apparent_auroc = float(roc_auc_score(y_orig, p_orig))
    apparent_brier = float(np.mean((y_orig - p_orig) ** 2))
    apparent_cs_r2, apparent_nag_r2 = _compute_cox_snell_and_nagelkerke(y_orig, p_orig)
    apparent_intercept = _compute_calibration_intercept_offset(y_orig, lp_orig)
    apparent_slope = _compute_calibration_slope(y_orig, lp_orig)

    apparent_metrics = {
        "auroc": apparent_auroc,
        "brier_score": apparent_brier,
        "cox_snell_r2": apparent_cs_r2,
        "nagelkerke_r2": apparent_nag_r2,
        "calibration_intercept": apparent_intercept,
        "calibration_slope": apparent_slope,
    }

    # 2. Bootstrap Loop
    rng = np.random.default_rng(random_state)
    metric_keys = [
        "auroc",
        "brier_score",
        "nagelkerke_r2",
        "calibration_intercept",
        "calibration_slope",
    ]
    boot_records: dict[str, list[float]] = {k: [] for k in metric_keys}
    test_records: dict[str, list[float]] = {k: [] for k in metric_keys}
    optimism_records: dict[str, list[float]] = {k: [] for k in metric_keys}
    test_slopes: list[float] = []

    n_successful = 0

    for _ in range(n_bootstrap):
        boot_idx = rng.choice(n_total, size=n_total, replace=True)
        y_boot = y_orig[boot_idx]
        X_boot = X_orig[boot_idx]

        # Skip degenerate samples
        sum_boot = np.sum(y_boot)
        if sum_boot == 0 or sum_boot == n_total:
            continue

        # Skip samples with zero variance in any dummy/predictor column (singular design matrix)
        if np.any(np.var(X_boot, axis=0) <= 1e-12):
            continue

        X_boot_const = sm.add_constant(X_boot, has_constant="add")
        if norm_penalty == "firth":
            from medstat.models.firth import fit_firth_logistic

            try:
                b_fit = fit_firth_logistic(
                    y=y_boot,
                    X=X_boot,
                    feature_names=model_predictors,
                    fit_intercept=True,
                )
                b_params = np.asarray(b_fit["model"].coef_).ravel()
                if np.any(np.isnan(b_params)) or np.any(np.abs(b_params) > 50.0):
                    continue
            except Exception:
                continue
        else:
            try:
                boot_model = sm.Logit(y_boot, X_boot_const).fit(disp=False)
                if not boot_model.mle_retvals.get("converged", True):
                    continue
                b_params = boot_model.params
                if np.any(np.isnan(b_params)) or np.any(np.abs(b_params) > 50.0):
                    continue
            except Exception:
                continue

        # Evaluate on bootstrap sample (boot_metrics)
        lp_boot = np.dot(X_boot_const, b_params)
        p_boot = expit(lp_boot)

        b_auc = float(roc_auc_score(y_boot, p_boot))
        b_brier = float(np.mean((y_boot - p_boot) ** 2))
        _, b_nag = _compute_cox_snell_and_nagelkerke(y_boot, p_boot)
        b_inter = _compute_calibration_intercept_offset(y_boot, lp_boot)
        b_slope = _compute_calibration_slope(y_boot, lp_boot)

        # Evaluate on original dataset (test_metrics)
        lp_test = np.dot(X_orig_const, b_params)
        p_test = expit(lp_test)

        t_auc = float(roc_auc_score(y_orig, p_test))
        t_brier = float(np.mean((y_orig - p_test) ** 2))
        _, t_nag = _compute_cox_snell_and_nagelkerke(y_orig, p_test)
        t_inter = _compute_calibration_intercept_offset(y_orig, lp_test)
        t_slope = _compute_calibration_slope(y_orig, lp_test)

        if not np.isfinite(t_slope) or not np.isfinite(t_inter):
            continue

        test_slopes.append(t_slope)

        # Record metrics
        b_vals = {
            "auroc": b_auc,
            "brier_score": b_brier,
            "nagelkerke_r2": b_nag,
            "calibration_intercept": b_inter,
            "calibration_slope": b_slope,
        }
        t_vals = {
            "auroc": t_auc,
            "brier_score": t_brier,
            "nagelkerke_r2": t_nag,
            "calibration_intercept": t_inter,
            "calibration_slope": t_slope,
        }

        for k in metric_keys:
            boot_records[k].append(b_vals[k])
            test_records[k].append(t_vals[k])
            # Consistent definition: optimism = boot - test
            optimism_records[k].append(b_vals[k] - t_vals[k])

        n_successful += 1

    if n_successful < 5:
        raise RuntimeError(
            f"Bootstrap internal validation failed: only {n_successful}/{n_bootstrap} replicates converged."
        )

    # 3. Compile Metric Summaries
    metrics_summary: dict[str, dict[str, float]] = {}
    for k in metric_keys:
        app = apparent_metrics[k]
        b_mean = float(np.mean(boot_records[k]))
        t_mean = float(np.mean(test_records[k]))
        opt_mean = float(np.mean(optimism_records[k]))
        # Universal identity: corrected = apparent - optimism
        corrected = float(app - opt_mean)
        if k in ("auroc", "nagelkerke_r2"):
            corrected = float(np.clip(corrected, 0.0, 1.0))
        elif k == "brier_score":
            corrected = float(max(0.0, corrected))

        metrics_summary[k] = {
            "apparent": app,
            "optimism": opt_mean,
            "optimism_corrected": corrected,
            "boot_mean": b_mean,
            "test_mean": t_mean,
        }

    # 4. Uniform Shrinkage Factor
    mean_test_slope = float(np.mean(test_slopes))
    shrinkage_factor = float(np.clip(mean_test_slope, 0.01, 1.0))

    # 5. Shrunk Model Coefficients
    # beta_shrunk_j = s_hat * beta_orig_j for j >= 1
    # Intercept beta_shrunk_0 re-estimated to preserve mean predicted risk
    orig_intercept = float(orig_params[0])
    orig_slopes = [float(c) for c in orig_params[1:]]
    shrunk_slopes = [shrinkage_factor * c for c in orig_slopes]

    lp_shrunk_no_intercept = np.dot(X_orig, shrunk_slopes)
    reestimated_intercept = _find_reestimated_intercept(y_orig, lp_shrunk_no_intercept)

    orig_coef_dict = {"Intercept": orig_intercept}
    shrunk_coef_dict = {"Intercept": reestimated_intercept}
    for p_name, o_c, s_c in zip(model_predictors, orig_slopes, shrunk_slopes):
        orig_coef_dict[p_name] = o_c
        shrunk_coef_dict[p_name] = s_c

    # 6. Clinical Guardrails & Warnings
    warnings_list = []
    if epp < 10.0:
        warnings_list.append(
            f"Low events per predictor (EPP = {epp:.2f} < 10), indicating elevated risk of model overfitting."
        )
    if shrinkage_factor < 0.80:
        warnings_list.append(
            f"Uniform Shrinkage Factor (ŝ = {shrinkage_factor:.3f}) is below 0.80, indicating severe empirical overfitting."
        )
    if n_successful < 0.5 * n_bootstrap:
        warnings_list.append(
            f"Low bootstrap convergence rate: only {n_successful}/{n_bootstrap} ({n_successful / n_bootstrap * 100:.1f}%) replicates converged."
        )

    return {
        "n_total": n_total,
        "n_events": n_events,
        "n_predictors": n_predictors,
        "events_per_predictor": epp,
        "n_bootstrap": n_bootstrap,
        "n_successful_replicates": n_successful,
        "apparent_metrics": apparent_metrics,
        "metrics": metrics_summary,
        "shrinkage_factor": shrinkage_factor,
        "shrunk_coefficients": {
            "original": orig_coef_dict,
            "shrunk": shrunk_coef_dict,
        },
        "warnings": warnings_list,
    }


def recalibrate_prediction_model(
    y_true: pd.Series | np.ndarray,
    p_pred: pd.Series | np.ndarray,
    method: Literal[
        "intercept_only", "calibration_in_the_large", "logistic", "both"
    ] = "logistic",
    clip_eps: float = 1e-6,
) -> dict[str, Any]:
    """
    Prediction Model Recalibration and Updating.

    Supports:
    - Intercept-only / Calibration-in-the-large updating (preserves slope=1, shifts mean risk to event rate).
    - Logistic recalibration (updates both calibration intercept and slope).
    - Combined evaluation ('both').

    Parameters:
        y_true: Binary ground truth outcomes (0 and 1).
        p_pred: Baseline model predicted probabilities in [0, 1].
        method: Recalibration strategy ('intercept_only', 'calibration_in_the_large', 'logistic', or 'both').
        clip_eps: Numerical stability threshold for probability logit transformation.

    Returns:
        dict: Recalibration summary containing:
            - pre: pre-recalibration calibration metrics (intercept, slope, Brier, Emax, ICI, AUROC)
            - post: post-recalibration metrics
            - recalibration_parameters: estimated alpha and beta coefficients with CIs
            - p_recalibrated: updated patient risk predictions (locally available, stripped before CLI JSON)
    """
    y_arr = np.asarray(y_true, dtype=float).ravel()
    p_arr = np.asarray(p_pred, dtype=float).ravel()

    if len(y_arr) != len(p_arr):
        raise ValueError(
            f"Length mismatch between y_true (N={len(y_arr)}) and p_pred (N={len(p_arr)})."
        )

    n_invalid_y = int(np.sum(~np.isfinite(y_arr)))
    n_invalid_p = int(np.sum(~np.isfinite(p_arr)))
    if n_invalid_y > 0 or n_invalid_p > 0:
        raise ValueError(
            f"Non-finite values detected in input arrays: y_true has {n_invalid_y} non-finite value(s), "
            f"p_pred has {n_invalid_p} non-finite value(s). "
            "Please perform complete-case filtering or multiple imputation before running recalibration."
        )

    n = len(y_arr)
    if n < 10:
        raise ValueError(f"Insufficient observations for recalibration (N = {n}).")

    unique_y = np.unique(y_arr)
    if not np.array_equal(unique_y, [0.0, 1.0]):
        raise ValueError(
            f"y_true must contain exactly two distinct binary classes {{0, 1}}, got {unique_y}."
        )

    if np.any(p_arr < 0.0) or np.any(p_arr > 1.0):
        raise ValueError(
            "Predicted probabilities p_pred must lie strictly within [0, 1]."
        )

    p_clipped = np.clip(p_arr, clip_eps, 1.0 - clip_eps)
    lp_orig = logit(p_clipped)

    # Pre-recalibration metrics helper
    def _calc_suite(y: np.ndarray, p: np.ndarray, lp: np.ndarray) -> dict[str, float]:
        auc = float(roc_auc_score(y, p))
        brier = float(np.mean((y - p) ** 2))
        inter = _compute_calibration_intercept_offset(y, lp)
        slope = _compute_calibration_slope(y, lp)

        # ICI and Emax via LOWESS or sliding window
        order = np.argsort(p)
        p_s = p[order]
        y_s = y[order]
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess

            smooth_obs = lowess(y_s, p_s, frac=2.0 / 3.0, it=0, return_sorted=False)
        except Exception:
            w = max(len(y) // 10, 5)
            smooth_obs = (
                pd.Series(y_s)
                .rolling(window=w, center=True, min_periods=3)
                .mean()
                .bfill()
                .ffill()
                .to_numpy()
            )
        abs_err = np.abs(p_s - smooth_obs)
        ici = float(np.mean(abs_err))
        emax = float(np.max(abs_err))

        return {
            "calibration_intercept": inter,
            "calibration_slope": slope,
            "brier_score": brier,
            "emax": emax,
            "ici": ici,
            "eavg": ici,
            "auroc": auc,
        }

    pre_metrics = _calc_suite(y_arr, p_clipped, lp_orig)

    norm_method = method.lower().strip()

    # 1. Intercept-only recalibration
    alpha_inter = _compute_calibration_intercept_offset(y_arr, lp_orig)
    beta_inter = 1.0
    lp_inter = alpha_inter + lp_orig
    p_recal_int = expit(lp_inter)

    # 2. Logistic recalibration
    X = sm.add_constant(lp_orig)
    try:
        fit_mod = sm.Logit(y_arr, X).fit(disp=False)
        alpha_log = float(fit_mod.params[0])
        beta_log = float(fit_mod.params[1])
        alpha_ci = [float(c) for c in fit_mod.conf_int()[0]]
        beta_ci = [float(c) for c in fit_mod.conf_int()[1]]
        alpha_p = float(fit_mod.pvalues[0])
        beta_p = float(fit_mod.pvalues[1])
        lp_log = alpha_log + beta_log * lp_orig
        p_recal_log = expit(lp_log)
    except Exception as e:
        if norm_method in ("logistic", "both"):
            raise RuntimeError(
                f"Logistic recalibration model failed to converge: {e}"
            ) from e
        alpha_log = alpha_inter
        beta_log = 1.0
        alpha_ci = [alpha_inter, alpha_inter]
        beta_ci = [1.0, 1.0]
        alpha_p = 1.0
        beta_p = 1.0
        lp_log = lp_inter
        p_recal_log = p_recal_int

    if norm_method in ("calibration_in_the_large", "intercept_only"):
        post_metrics = _calc_suite(y_arr, p_recal_int, lp_inter)
        params_info = {
            "method": "intercept_only",
            "alpha": alpha_inter,
            "beta": beta_inter,
        }
        return {
            "method": "intercept_only",
            "n": n,
            "y_true": y_arr,
            "p_orig": p_clipped,
            "pre": pre_metrics,
            "post": post_metrics,
            "recalibration_parameters": params_info,
            "p_recalibrated": p_recal_int,
            "p_recalibrated_intercept_only": p_recal_int,
            "p_recalibrated_logistic": p_recal_log,
        }
    elif norm_method == "logistic":
        post_metrics = _calc_suite(y_arr, p_recal_log, lp_log)
        params_info = {
            "method": "logistic",
            "alpha": alpha_log,
            "alpha_ci": alpha_ci,
            "alpha_pvalue": alpha_p,
            "beta": beta_log,
            "beta_ci": beta_ci,
            "beta_pvalue": beta_p,
        }
        return {
            "method": "logistic",
            "n": n,
            "y_true": y_arr,
            "p_orig": p_clipped,
            "pre": pre_metrics,
            "post": post_metrics,
            "recalibration_parameters": params_info,
            "p_recalibrated": p_recal_log,
            "p_recalibrated_intercept_only": p_recal_int,
            "p_recalibrated_logistic": p_recal_log,
        }
    elif norm_method == "both":
        post_inter = _calc_suite(y_arr, p_recal_int, lp_inter)
        post_log = _calc_suite(y_arr, p_recal_log, lp_log)
        return {
            "method": "both",
            "n": n,
            "y_true": y_arr,
            "p_orig": p_clipped,
            "pre": pre_metrics,
            "post_intercept_only": post_inter,
            "post_logistic": post_log,
            "post": post_log,  # Primary post
            "parameters_intercept_only": {
                "alpha": alpha_inter,
                "beta": 1.0,
            },
            "parameters_logistic": {
                "alpha": alpha_log,
                "alpha_ci": alpha_ci,
                "beta": beta_log,
                "beta_ci": beta_ci,
            },
            "p_recalibrated": p_recal_log,
            "p_recalibrated_intercept_only": p_recal_int,
            "p_recalibrated_logistic": p_recal_log,
        }
    else:
        raise ValueError(
            f"Unsupported recalibration method '{method}'. Must be 'intercept_only', 'calibration_in_the_large', 'logistic', or 'both'."
        )

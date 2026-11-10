# Prediction Model Validation, Sample Size & Meta-Analysis Recipes (Cloud Sandbox)

> **Scope:** Recipes 18–20 and 24: TRIPOD+AI bootstrap internal validation with uniform shrinkage, model recalibration, Riley / Obuchowski sample size, and fixed / DerSimonian-Laird random-effects meta-analysis.

---

## 18. TRIPOD+AI Optimism-Corrected Bootstrap Internal Validation & Uniform Shrinkage

Computes Efron-Gong optimism-corrected performance metrics (AUROC, Brier score, Cox-Snell $R^2$, Nagelkerke $R^2$, calibration-in-the-large, calibration slope) across $B$ bootstrap resamples. Derives the uniform shrinkage factor $\hat{s}$ and re-estimates the intercept $\alpha_0$ to preserve sample prevalence on future predictions.

```python
import math
from typing import Any

import numpy as np
import pandas as pd
import scipy.optimize as opt
import statsmodels.api as sm
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score


def bootstrap_internal_validation(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    n_bootstrap: int = 200,
    random_state: int = 42,
    missing_strategy: str = "error",
) -> dict[str, Any]:
    """TRIPOD+AI Optimism-Corrected Bootstrap Internal Validation and Uniform Shrinkage."""
    norm_strat = str(missing_strategy).lower().strip()
    if norm_strat not in ("error", "complete_case"):
        raise ValueError(
            f"Unsupported missing_strategy '{missing_strategy}'. Must be 'error' or 'complete_case'."
        )

    cols = [outcome] + list(predictors)
    n_initial = len(df)
    missing_counts = {
        c: int(df[c].isna().sum()) for c in cols if df[c].isna().sum() > 0
    }
    valid_mask = ~df[cols].isna().any(axis=1)
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            inf_mask = np.isinf(df[c].to_numpy(dtype=float))
            valid_mask &= ~inf_mask
            if inf_mask.sum() > 0:
                missing_counts[f"{c}_inf"] = int(inf_mask.sum())
    sub_df = df.loc[valid_mask, cols].copy()
    n_excluded = n_initial - len(sub_df)

    if n_excluded > 0 and norm_strat == "error":
        counts_str = ", ".join(f"{c} ({cnt})" for c, cnt in missing_counts.items())
        raise ValueError(
            f"Missing or non-finite values detected in input columns: {counts_str}. "
            "Please perform complete-case filtering or multiple imputation before running internal validation."
        )

    exclusion_summary = {
        "n_initial": n_initial,
        "n_excluded": n_excluded,
        "n_retained": len(sub_df),
        "missing_counts": missing_counts,
    }

    if len(sub_df) < 10:
        raise ValueError(f"Insufficient non-missing observations (N = {len(sub_df)}).")

    y_orig_raw = sub_df[outcome].to_numpy(dtype=float)
    if not np.all(np.isin(y_orig_raw, [0.0, 1.0])):
        raise ValueError(
            f"Binary outcome '{outcome}' must contain strictly binary values (0 and 1)."
        )
    if len(np.unique(y_orig_raw)) < 2:
        raise ValueError(
            f"Binary outcome '{outcome}' must contain both event classes (0 and 1)."
        )
    y_orig = y_orig_raw.astype(int)

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

    # 1. Fit initial model with separation/singular fallback
    X_orig_const = sm.add_constant(X_orig, has_constant="add")
    warnings = []
    try:
        orig_model = sm.Logit(y_orig, X_orig_const).fit(disp=False)
        converged = getattr(orig_model, "mle_retvals", {}).get("converged", True)
        if not converged or np.any(np.abs(orig_model.params) > 50.0):
            raise RuntimeError("Model separation or non-convergence detected.")
        orig_params = orig_model.params
    except Exception as e:
        warnings.append(
            f"Standard MLE logit failed or encountered quasi-complete separation ({e}); falling back to regularized logit."
        )
        reg_model = sm.Logit(y_orig, X_orig_const).fit_regularized(
            method="l1", alpha=1e-4, disp=False
        )
        orig_params = reg_model.params

    lp_orig = np.dot(X_orig_const, orig_params)
    p_orig = expit(lp_orig)

    def _calc_metrics(
        y_true: np.ndarray, p_pred: np.ndarray, lp: np.ndarray
    ) -> dict[str, float]:
        n = len(y_true)
        p_c = np.clip(p_pred, 1e-15, 1.0 - 1e-15)
        prev = np.clip(float(np.mean(y_true)), 1e-15, 1.0 - 1e-15)
        ll_null = float(
            np.sum(y_true * np.log(prev) + (1.0 - y_true) * np.log(1.0 - prev))
        )
        ll_model = float(
            np.sum(y_true * np.log(p_c) + (1.0 - y_true) * np.log(1.0 - p_c))
        )
        cs_r2 = max(0.0, 1.0 - math.exp(-2.0 * (ll_model - ll_null) / n))
        r2_max = max(1e-12, 1.0 - math.exp(2.0 * ll_null / n))
        nag_r2 = min(1.0, max(0.0, cs_r2 / r2_max))
        auc = (
            float(roc_auc_score(y_true, p_pred))
            if len(np.unique(y_true)) == 2
            else 0.5
        )
        brier = float(np.mean((y_true - p_pred) ** 2))

        # Intercept offset
        sum_y = float(np.sum(y_true))
        if sum_y <= 0:
            inter = -50.0
        elif sum_y >= n:
            inter = 50.0
        else:
            try:
                inter = float(
                    opt.brentq(
                        lambda a: float(np.sum(expit(a + lp)) - sum_y),
                        -50.0,
                        50.0,
                    )
                )
            except Exception:
                inter = 0.0

        # Slope
        if np.allclose(lp, lp[0]):
            slope = 1.0
        else:
            try:
                mod_slope = sm.Logit(y_true, sm.add_constant(lp)).fit(disp=False)
                slope = float(mod_slope.params[1])
            except Exception:
                slope = 1.0

        return {
            "auroc": auc,
            "brier_score": brier,
            "cox_snell_r2": cs_r2,
            "nagelkerke_r2": nag_r2,
            "calibration_intercept": inter,
            "calibration_slope": slope,
        }

    apparent = _calc_metrics(y_orig, p_orig, lp_orig)

    # 2. Bootstrap loop
    rng = np.random.default_rng(random_state)
    boot_records = {k: [] for k in apparent}
    test_records = {k: [] for k in apparent}
    optimism_records = {k: [] for k in apparent}
    test_slopes = []
    n_successful = 0

    for _ in range(n_bootstrap):
        boot_idx = rng.choice(n_total, size=n_total, replace=True)
        y_b = y_orig[boot_idx]
        X_b = X_orig[boot_idx]
        if len(np.unique(y_b)) < 2:
            continue
        # Skip samples with zero variance in any dummy/predictor column
        if np.any(np.var(X_b, axis=0) <= 1e-12):
            continue
        X_b_const = sm.add_constant(X_b, has_constant="add")
        try:
            boot_mod = sm.Logit(y_b, X_b_const).fit(disp=False)
            b_params = boot_mod.params
            if np.isnan(b_params).any() or np.isinf(b_params).any():
                continue
        except Exception:
            continue

        lp_b = np.dot(X_b_const, b_params)
        p_b = expit(lp_b)
        m_b = _calc_metrics(y_b, p_b, lp_b)

        lp_test = np.dot(X_orig_const, b_params)
        p_test = expit(lp_test)
        m_test = _calc_metrics(y_orig, p_test, lp_test)

        n_successful += 1
        for k in apparent:
            boot_records[k].append(m_b[k])
            test_records[k].append(m_test[k])
            optimism_records[k].append(m_b[k] - m_test[k])
        test_slopes.append(m_test["calibration_slope"])

    # Summary
    metrics_summary = {}
    for k in apparent:
        app_val = apparent[k]
        if n_successful == 0:
            opt_mean = np.nan
            b_mean = np.nan
            t_mean = np.nan
            corrected = np.nan
        else:
            opt_mean = float(np.mean(optimism_records[k])) if optimism_records[k] else 0.0
            b_mean = float(np.mean(boot_records[k])) if boot_records[k] else app_val
            t_mean = float(np.mean(test_records[k])) if test_records[k] else app_val
            corrected = app_val - opt_mean
            if k in ("auroc", "nagelkerke_r2", "cox_snell_r2"):
                corrected = max(0.0, min(1.0, corrected))
            elif k == "brier_score":
                corrected = max(0.0, corrected)
        metrics_summary[k] = {
            "apparent": float(app_val),
            "boot_mean": float(b_mean) if not np.isnan(b_mean) else np.nan,
            "test_mean": float(t_mean) if not np.isnan(t_mean) else np.nan,
            "optimism": float(opt_mean) if not np.isnan(opt_mean) else np.nan,
            "optimism_corrected": float(corrected) if not np.isnan(corrected) else np.nan,
        }

    orig_intercept = float(orig_params[0])
    orig_betas = [float(b) for b in orig_params[1:]]

    if n_successful == 0:
        shrinkage_factor = np.nan
        shrunk_intercept = np.nan
        shrunk_betas = [np.nan for _ in orig_betas]
    else:
        raw_s = float(np.mean(test_slopes)) if test_slopes else 1.0
        shrinkage_factor = float(np.clip(raw_s, 0.01, 1.0))
        shrunk_betas = [shrinkage_factor * b for b in orig_betas]
        lp_shrunk_no_inter = np.dot(X_orig, shrunk_betas)

        # Re-estimate intercept
        target_events = float(np.sum(y_orig))
        try:
            shrunk_intercept = float(
                opt.brentq(
                    lambda a: float(np.sum(expit(a + lp_shrunk_no_inter)) - target_events),
                    -50.0,
                    50.0,
                )
            )
        except Exception:
            prev_orig = np.clip(target_events / n_total, 1e-6, 1.0 - 1e-6)
            shrunk_intercept = float(logit(prev_orig) - np.mean(lp_shrunk_no_inter))

    orig_coef_dict = {"Intercept": orig_intercept}
    shrunk_coef_dict = {"Intercept": shrunk_intercept}
    for name, ob, sb in zip(model_predictors, orig_betas, shrunk_betas):
        orig_coef_dict[name] = ob
        shrunk_coef_dict[name] = sb

    if epp < 10.0:
        warnings.append(
            f"Low events per predictor (EPP = {epp:.2f} < 10), indicating elevated risk of model overfitting."
        )
    if not np.isnan(shrinkage_factor) and shrinkage_factor < 0.80:
        warnings.append(
            f"Uniform shrinkage factor s_hat = {shrinkage_factor:.3f} (< 0.80) indicates substantial overfitting."
        )
    if n_successful == 0:
        warnings.append(
            "Zero bootstrap resamples converged successfully; optimism and shrinkage could not be estimated."
        )
    elif n_successful < n_bootstrap * 0.5:
        warnings.append(
            f"Fewer than 50% of bootstrap resamples converged ({n_successful}/{n_bootstrap})."
        )

    return {
        "n_total": n_total,
        "n_events": n_events,
        "n_predictors": n_predictors,
        "events_per_predictor": epp,
        "n_bootstrap": n_bootstrap,
        "n_successful_replicates": n_successful,
        "missing_data_summary": exclusion_summary,
        "apparent_metrics": apparent,
        "metrics": metrics_summary,
        "shrinkage_factor": shrinkage_factor,
        "shrunk_coefficients": {
            "original": orig_coef_dict,
            "shrunk": shrunk_coef_dict,
        },
        "warnings": warnings,
    }
```

---

---

## 19. Prediction Model Recalibration and Updating (Logistic & Intercept-Only)

Adjusts existing prediction models to a new target population or cohort. Supports Intercept-Only updating (calibration-in-the-large, shifting mean predicted risk to the observed outcome incidence while fixing slope=1) and Logistic Recalibration (updating both calibration intercept $\alpha$ and calibration slope $\beta$).

```python
from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy.optimize as opt
import statsmodels.api as sm
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score


def recalibrate_prediction_model(
    y_true: pd.Series | np.ndarray,
    p_pred: pd.Series | np.ndarray,
    method: Literal[
        "intercept_only", "calibration_in_the_large", "logistic", "both"
    ] = "logistic",
    clip_eps: float = 1e-6,
) -> dict[str, Any]:
    if (
        not isinstance(clip_eps, (int, float))
        or not np.isfinite(clip_eps)
        or not (0.0 < float(clip_eps) < 0.5)
    ):
        raise ValueError(
            f"clip_eps ({clip_eps}) must be a finite number strictly between 0 and 0.5."
        )
    clip_eps = float(clip_eps)

    norm_m = (
        "intercept_only"
        if method in ("intercept_only", "calibration_in_the_large")
        else method
    )
    if norm_m not in ("intercept_only", "logistic", "both"):
        raise ValueError(f"Unsupported recalibration method: {method}")

    y_arr = np.asarray(y_true, dtype=float).ravel()
    p_arr = np.asarray(p_pred, dtype=float).ravel()

    if len(y_arr) != len(p_arr):
        raise ValueError(
            f"Length mismatch between y_true (N={len(y_arr)}) and p_pred (N={len(p_arr)})."
        )

    n_invalid_y = int(np.sum(~np.isfinite(y_arr)))
    n_invalid_p = int(np.sum(~np.isfinite(p_arr)))
    n_excluded = n_invalid_y + n_invalid_p
    if n_excluded > 0:
        raise ValueError(
            f"Non-finite values detected in input arrays: y_true has {n_invalid_y} non-finite value(s), "
            f"p_pred has {n_invalid_p} non-finite value(s) ({n_excluded} non-finite values total). "
            "Please perform complete-case filtering or multiple imputation before running recalibration."
        )

    n = len(y_arr)
    if n == 0:
        raise ValueError("Cannot perform recalibration on empty arrays (N = 0).")
    if n < 10:
        raise ValueError(f"Insufficient observations for recalibration (N = {n}).")

    unique_y = np.unique(y_arr)
    if not np.array_equal(unique_y, [0.0, 1.0]):
        raise ValueError(
            f"y_true must contain strictly binary classes {{0, 1}} with both classes present, got {unique_y}."
        )

    if np.any(p_arr < 0.0) or np.any(p_arr > 1.0):
        raise ValueError(
            "Predicted probabilities p_pred must lie strictly within [0, 1]."
        )

    p_clipped = np.clip(p_arr, clip_eps, 1.0 - clip_eps)
    lp_orig = logit(p_clipped)

    def _calc_suite(y: np.ndarray, p: np.ndarray, lp: np.ndarray) -> dict[str, float]:
        sum_y = float(np.sum(y))
        try:
            inter = float(
                opt.brentq(
                    lambda a: float(np.sum(expit(a + lp)) - sum_y), -50.0, 50.0
                )
            )
        except Exception:
            inter = 0.0

        if np.var(lp) <= 1e-12 or np.allclose(lp, lp[0]):
            slope = float("nan")
        else:
            try:
                mod_s = sm.Logit(y, sm.add_constant(lp)).fit(disp=False)
                if not getattr(mod_s, "mle_retvals", {}).get("converged", True):
                    slope = float("nan")
                else:
                    slope = float(mod_s.params[1])
                    if not np.isfinite(slope):
                        slope = float("nan")
            except Exception:
                slope = float("nan")

        brier = float(np.mean((y - p) ** 2))
        auc = (
            float(roc_auc_score(y, p))
            if len(np.unique(y)) == 2
            else 0.5
        )

        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess

            smoothed = lowess(y, p, frac=0.67, return_sorted=False)
            cal_err = np.abs(smoothed - p)
            emax = float(np.max(cal_err))
            ici = float(np.mean(cal_err))
        except Exception:
            emax = float("nan")
            ici = float("nan")

        return {
            "calibration_intercept": inter,
            "calibration_slope": slope,
            "brier_score": brier,
            "emax": emax,
            "ici": ici,
            "auroc": auc,
        }

    pre_metrics = _calc_suite(y_arr, p_clipped, lp_orig)

    sum_y = float(np.sum(y_arr))
    try:
        alpha_inter = float(
            opt.brentq(
                lambda a: float(np.sum(expit(a + lp_orig)) - sum_y), -50.0, 50.0
            )
        )
    except Exception:
        alpha_inter = 0.0
    p_inter = expit(alpha_inter + lp_orig)

    X = sm.add_constant(lp_orig)
    logistic_fallback = False
    try:
        fit_mod = sm.Logit(y_arr, X).fit(disp=False)
        converged = getattr(fit_mod, "mle_retvals", {}).get("converged", True)
        if not converged:
            raise RuntimeError("Logistic recalibration failed to converge.")
        alpha_log = float(fit_mod.params[0])
        beta_log = float(fit_mod.params[1])
        alpha_ci = [float(c) for c in fit_mod.conf_int()[0]]
        beta_ci = [float(c) for c in fit_mod.conf_int()[1]]
        p_log = expit(alpha_log + beta_log * lp_orig)
    except Exception:
        logistic_fallback = True
        alpha_log = alpha_inter
        beta_log = 1.0
        alpha_ci = [alpha_inter, alpha_inter]
        beta_ci = [1.0, 1.0]
        p_log = p_inter

    if norm_m == "intercept_only":
        lp_recal = alpha_inter + lp_orig
        post_metrics = _calc_suite(y_arr, p_inter, lp_recal)
        return {
            "method": "intercept_only",
            "n": n,
            "pre": pre_metrics,
            "post": post_metrics,
            "recalibration_parameters": {
                "method": "intercept_only",
                "alpha": alpha_inter,
                "beta": 1.0,
            },
            "p_recalibrated": p_inter,
            "p_recalibrated_intercept_only": p_inter,
            "p_recalibrated_logistic": p_log,
        }
    elif norm_m == "logistic":
        reported_method = "intercept_only_fallback" if logistic_fallback else "logistic"
        lp_recal = alpha_log + beta_log * lp_orig
        post_metrics = _calc_suite(y_arr, p_log, lp_recal)
        return {
            "method": reported_method,
            "n": n,
            "pre": pre_metrics,
            "post": post_metrics,
            "recalibration_parameters": {
                "method": reported_method,
                "alpha": alpha_log,
                "alpha_ci": alpha_ci,
                "beta": beta_log,
                "beta_ci": beta_ci,
            },
            "p_recalibrated": p_log,
            "p_recalibrated_intercept_only": p_inter,
            "p_recalibrated_logistic": p_log,
        }
    else:  # both
        post_inter = _calc_suite(y_arr, p_inter, alpha_inter + lp_orig)
        post_log = _calc_suite(y_arr, p_log, alpha_log + beta_log * lp_orig)

        return {
            "method": "both",
            "n": n,
            "pre": pre_metrics,
            "post_intercept_only": post_inter,
            "post_logistic": post_log,
            "post": post_log,
            "logistic_fallback": logistic_fallback,
            "parameters_intercept_only": {"alpha": alpha_inter, "beta": 1.0},
            "parameters_logistic": {
                "alpha": alpha_log,
                "beta": beta_log,
                "fallback": logistic_fallback,
            },
            "p_recalibrated": p_log,
            "p_recalibrated_intercept_only": p_inter,
            "p_recalibrated_logistic": p_log,
        }
```

---

---

## 20. Sample Size Determination for Clinical Prediction Models (Riley et al. 2020) & Diagnostic AUROC (Obuchowski 1998)

Calculates the minimum required sample size for multivariable prediction model development per the 4 criteria of Riley et al. (BMJ 2020;368:m441), including numerical derivation of $R^2_{CS}$ from C-statistic or Nagelkerke $R^2$. Also determines required case and control sample sizes for diagnostic ROC test evaluation against a null benchmark per Obuchowski (1998) and Hanley & McNeil (1982).

```python
import math
from typing import Any

import numpy as np
import scipy.optimize as opt
import scipy.stats as stats
from scipy.special import expit, log_expit


def _approximate_cox_snell_r2_from_c_stat(
    c_stat: float,
    prevalence: float,
    deg: int = 64,
) -> float:
    """Exact deterministic Gauss-Hermite numerical integration of Riley et al. (2021)."""
    c = float(c_stat)
    phi = float(prevalence)
    if not (0.5 < c < 1.0):
        raise ValueError(f"c_statistic ({c}) must be strictly between 0.5 and 1.0.")
    if not (0.0 < phi < 1.0):
        raise ValueError(f"prevalence ({phi}) must be strictly between 0 and 1.")

    mu1 = math.sqrt(2.0) * float(stats.norm.ppf(c))
    nodes, weights = np.polynomial.hermite.hermgauss(deg)
    z = np.sqrt(2.0) * nodes
    inv_sqrt_pi = 1.0 / math.sqrt(math.pi)

    def f_alpha(alpha0: float) -> float:
        e0 = inv_sqrt_pi * float(np.sum(weights * expit(alpha0 + mu1 * z)))
        e1 = inv_sqrt_pi * float(np.sum(weights * expit(alpha0 + mu1 * (z + mu1))))
        return (1.0 - phi) * e0 + phi * e1 - phi

    alpha0 = float(opt.brentq(f_alpha, -50.0, 50.0))

    u0 = alpha0 + mu1 * z
    u1 = alpha0 + mu1 * (z + mu1)
    ll_0 = inv_sqrt_pi * float(np.sum(weights * log_expit(-u0)))
    ll_1 = inv_sqrt_pi * float(np.sum(weights * log_expit(u1)))
    ll_model = (1.0 - phi) * ll_0 + phi * ll_1

    ll_null = phi * math.log(phi) + (1.0 - phi) * math.log(1.0 - phi)
    r2_cs = -math.expm1(-2.0 * (ll_model - ll_null))
    r2_max = -math.expm1(2.0 * ll_null)

    return float(min(r2_max, max(0.0, r2_cs)))


def calculate_riley_sample_size(
    n_parameters: int,
    prevalence: float,
    c_statistic: float | None = None,
    cox_snell_r2: float | None = None,
    nagelkerke_r2: float | None = None,
    shrinkage_target: float = 0.90,
    intercept_margin: float = 0.05,
    optimism_margin: float = 0.05,
    mape_target: float | None = None,
) -> dict[str, Any]:
    """Riley et al. (2020) sample size for developing clinical prediction models."""
    if n_parameters < 1:
        raise ValueError(f"n_parameters must be >= 1, got {n_parameters}")
    if not (0.0 < prevalence < 1.0):
        raise ValueError(f"prevalence must be strictly in (0, 1), got {prevalence}")
    if not (0.5 < shrinkage_target < 1.0):
        raise ValueError(
            f"shrinkage_target must be strictly in (0.5, 1.0), got {shrinkage_target}"
        )
    if not (math.isfinite(intercept_margin) and intercept_margin > 0.0):
        raise ValueError(
            f"intercept_margin must be finite and > 0, got {intercept_margin}"
        )
    if not (math.isfinite(optimism_margin) and optimism_margin > 0.0):
        raise ValueError(
            f"optimism_margin must be finite and > 0, got {optimism_margin}"
        )
    if mape_target is not None and not (math.isfinite(mape_target) and mape_target > 0.0):
        raise ValueError(f"mape_target must be finite and > 0, got {mape_target}")

    phi = float(prevalence)
    p_params = int(n_parameters)
    s_target = float(shrinkage_target)

    ll_null = phi * math.log(phi) + (1.0 - phi) * math.log(1.0 - phi)
    r2_max = -math.expm1(2.0 * ll_null)

    if cox_snell_r2 is not None:
        cs_r2 = float(cox_snell_r2)
        if not (0.0 < cs_r2 < r2_max):
            raise ValueError(
                f"cox_snell_r2 ({cs_r2}) must be strictly between 0 and R2_max ({r2_max:.4f})."
            )
    elif nagelkerke_r2 is not None:
        nag_r2 = float(nagelkerke_r2)
        if not (0.0 < nag_r2 < 1.0):
            raise ValueError(
                f"nagelkerke_r2 ({nag_r2}) must be strictly between 0 and 1."
            )
        cs_r2 = nag_r2 * r2_max
    elif c_statistic is not None:
        c_stat = float(c_statistic)
        cs_r2 = _approximate_cox_snell_r2_from_c_stat(c_stat, phi)
    else:
        raise ValueError("Must provide c_statistic, cox_snell_r2, or nagelkerke_r2.")

    # Criterion 1: Overall risk/intercept
    n1 = int(math.ceil(((1.96 / intercept_margin) ** 2) * phi * (1.0 - phi)))

    # Criterion 2: MAPE (van Smeden 2019)
    # MAPE is invariant to swapping event / non-event labels, so min(phi, 1 - phi) keeps the
    # van Smeden (2019) formula inside its validated range (<= 0.5) for any prevalence.
    target_m = mape_target if mape_target is not None else 0.05
    if target_m is not None:
        min_phi = min(phi, 1.0 - phi)
        exp_val = (
            -0.508
            + 0.259 * math.log(min_phi)
            + 0.504 * math.log(p_params)
            - math.log(target_m)
        ) / 0.544
        n2 = int(math.ceil(math.exp(exp_val)))
    else:
        n2 = None

    # Criterion 3: Uniform shrinkage
    arg_s = 1.0 - cs_r2 / s_target
    if arg_s <= 0.0:
        raise ValueError(
            f"Target shrinkage factor ({s_target:.2f}) must be strictly greater than anticipated Cox-Snell R2 ({cs_r2:.4f})."
        )
    denom_s = (s_target - 1.0) * math.log(arg_s)
    n3 = 1000000 if denom_s <= 0 else int(math.ceil(p_params / denom_s))

    # Criterion 4: Small optimism in R2
    s_req = cs_r2 / (cs_r2 + optimism_margin * r2_max)
    arg_opt = 1.0 - cs_r2 / s_req
    if s_req >= 1.0 or arg_opt <= 0.0:
        n4 = n3
    else:
        denom_opt = (s_req - 1.0) * math.log(arg_opt)
        n4 = n3 if denom_opt <= 0 else int(math.ceil(p_params / denom_opt))

    crit_dict: dict[str, Any] = {
        "Criterion 1 (Intercept)": n1,
        "Criterion 3 (Shrinkage)": n3,
        "Criterion 4 (Optimism)": n4,
    }
    if n2 is not None:
        crit_dict["Criterion 2 (MAPE)"] = n2
    else:
        crit_dict["Criterion 2 (MAPE)"] = None

    valid_crits = {k: v for k, v in crit_dict.items() if v is not None}
    n_total = max(valid_crits.values())
    binding = max(valid_crits, key=valid_crits.get)  # type: ignore[arg-type]
    n_events = int(math.ceil(n_total * phi))
    epp = float(n_events / p_params)

    return {
        "n_total": n_total,
        "n_events": n_events,
        "n_parameters": p_params,
        "prevalence": phi,
        "events_per_parameter": epp,
        "cox_snell_r2": cs_r2,
        "nagelkerke_r2": float(cs_r2 / r2_max),
        "r2_max": float(r2_max),
        "shrinkage_target": s_target,
        "binding_criterion": binding,
        "criteria": {
            "criterion1_intercept": {"n": n1, "margin": intercept_margin},
            "criterion2_mape": {
                "n": n2,
                "target_mape": target_m,
                "status": "evaluated"
                if n2 is not None
                else "N/A (MAPE criterion not evaluated)",
            },
            "criterion3_shrinkage": {"n": n3, "target_shrinkage": s_target},
            "criterion4_optimism": {"n": n4, "optimism_margin": optimism_margin},
        },
    }


def calculate_diagnostic_auroc_sample_size(
    auroc_expected: float,
    auroc_null: float = 0.50,
    prevalence: float = 0.50,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> dict[str, Any]:
    """Obuchowski (1998) / Hanley-McNeil (1982) diagnostic AUROC sample size."""
    theta1 = float(auroc_expected)
    theta0 = float(auroc_null)
    phi = float(prevalence)

    if not (0.0 <= theta0 < theta1 <= 1.0):
        raise ValueError(
            f"auroc_null ({theta0}) must be strictly less than auroc_expected ({theta1}) within [0, 1]."
        )
    if not (0.0 < phi < 1.0):
        raise ValueError(f"prevalence ({phi}) must be strictly in (0, 1).")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha ({alpha}) must be strictly in (0, 1).")
    if not (0.0 < power < 1.0):
        raise ValueError(f"power ({power}) must be strictly in (0, 1).")

    z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0 if two_sided else 1.0 - alpha))
    z_beta = float(stats.norm.ppf(power))

    R = (1.0 - phi) / phi

    def _v(t: float) -> float:
        q1 = t / (2.0 - t)
        q2 = 2.0 * (t**2) / (1.0 + t)
        return float((q2 - t**2) + (q1 - t**2) / R)

    var_theta1 = _v(theta1)
    var_theta0 = _v(theta0)

    numerator = (z_alpha * math.sqrt(var_theta0) + z_beta * math.sqrt(var_theta1)) ** 2
    delta = (theta1 - theta0) ** 2
    n_cases = int(math.ceil(numerator / delta))
    n_controls = int(math.ceil(n_cases * R))

    return {
        "n_cases": n_cases,
        "n_controls": n_controls,
        "total_n": n_cases + n_controls,
        "n_total": n_cases + n_controls,
        "auroc_expected": theta1,
        "auroc_null": theta0,
        "prevalence": phi,
        "allocation_ratio_controls_to_cases": float(R),
        "alpha": alpha,
        "power": power,
        "two_sided": two_sided,
    }
```

---

## 24. Fixed-Effects & DerSimonian-Laird Random-Effects Meta-Analysis (PRISMA 2020)

Computes study-level binary ($\text{OR}$, $\text{RR}$, $\text{RD}$ with $0.5$ Haldane-Anscombe continuity correction for zero cells) or continuous ($\text{MD}$, Hedges' $g$ $\text{SMD}$) effect sizes and runs **Inverse-Variance Fixed-Effect** and **DerSimonian-Laird Random-Effects Meta-Analysis** with optional **Hartung-Knapp-Sidik-Jonkman (HKSJ)** small-sample adjustment, **Cochran's $Q$**, **Higgins' $I^2$**, between-study variance **$\hat{\tau}^2$**, **95% Prediction Interval**, and **Egger's regression asymmetry test** (when $k \ge 10$).

```python
import math
from typing import Any, Literal
import numpy as np
import pandas as pd
import scipy.stats as stats


def compute_binary_effect_sizes(
    df: pd.DataFrame,
    events_t_col: str,
    n_t_col: str,
    events_c_col: str,
    n_c_col: str,
    study_col: str,
    effect_measure: Literal["OR", "RR", "RD"] = "OR",
) -> pd.DataFrame:
    """Compute log effect sizes and standard errors for 2x2 binary endpoints."""
    cols = [study_col, events_t_col, n_t_col, events_c_col, n_c_col]
    clean = df[cols].dropna().copy()
    records = []
    for _, row in clean.iterrows():
        study = str(row[study_col])
        a_raw, n1_raw = float(row[events_t_col]), float(row[n_t_col])
        c_raw, n0_raw = float(row[events_c_col]), float(row[n_c_col])
        if not (
            math.isfinite(a_raw)
            and math.isfinite(n1_raw)
            and math.isfinite(c_raw)
            and math.isfinite(n0_raw)
            and a_raw.is_integer()
            and n1_raw.is_integer()
            and c_raw.is_integer()
            and n0_raw.is_integer()
            and n1_raw >= 1
            and n0_raw >= 1
            and 0 <= a_raw <= n1_raw
            and 0 <= c_raw <= n0_raw
        ):
            raise ValueError(
                f"Invalid 2x2 counts in study '{study}': require integer 0 <= events <= n and n >= 1."
            )
        a, n1, c, n0 = a_raw, n1_raw, c_raw, n0_raw
        b, d = n1 - a, n0 - c
        if a == 0 or b == 0 or c == 0 or d == 0:
            a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
            n1, n0 = n1 + 1.0, n0 + 1.0

        if effect_measure == "OR":
            val = (a * d) / (b * c)
            log_eff = math.log(val)
            se = math.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
            is_ratio = True
        elif effect_measure == "RR":
            val = (a / n1) / (c / n0)
            log_eff = math.log(val)
            se = math.sqrt(1.0 / a - 1.0 / n1 + 1.0 / c - 1.0 / n0)
            is_ratio = True
        else:
            val = (a / n1) - (c / n0)
            log_eff = val
            se = math.sqrt((a * b) / (n1**3) + (c * d) / (n0**3))
            is_ratio = False

        ci_l, ci_u = log_eff - 1.96 * se, log_eff + 1.96 * se
        records.append(
            {
                "study": study,
                "effect_size": val,
                "log_effect": log_eff,
                "se": se,
                "ci_lower": math.exp(ci_l) if is_ratio else ci_l,
                "ci_upper": math.exp(ci_u) if is_ratio else ci_u,
                "is_ratio": is_ratio,
                "effect_measure": effect_measure,
            }
        )
    return pd.DataFrame(records)


def compute_continuous_effect_sizes(
    df: pd.DataFrame,
    mean_t_col: str,
    sd_t_col: str,
    n_t_col: str,
    mean_c_col: str,
    sd_c_col: str,
    n_c_col: str,
    study_col: str,
    effect_measure: Literal["MD", "SMD"] = "MD",
) -> pd.DataFrame:
    """Compute Mean Difference (MD) or Hedges' g (SMD) and standard errors for continuous endpoints."""
    if effect_measure not in ("MD", "SMD"):
        raise ValueError(
            f"Unsupported effect_measure '{effect_measure}'. Expected 'MD' or 'SMD'."
        )
    cols = [study_col, mean_t_col, sd_t_col, n_t_col, mean_c_col, sd_c_col, n_c_col]
    clean = df[cols].dropna().copy()
    records = []
    for _, row in clean.iterrows():
        study = str(row[study_col])
        m1, sd1, n1 = float(row[mean_t_col]), float(row[sd_t_col]), float(row[n_t_col])
        m0, sd0, n0 = float(row[mean_c_col]), float(row[sd_c_col]), float(row[n_c_col])
        if not (
            all(math.isfinite(x) for x in (m1, sd1, n1, m0, sd0, n0))
            and n1.is_integer()
            and n0.is_integer()
            and n1 >= 2
            and n0 >= 2
            and sd1 > 0
            and sd0 > 0
        ):
            raise ValueError(f"Invalid continuous summary inputs in study '{study}'.")
        if effect_measure == "SMD":
            dof = n1 + n0 - 2.0
            s_pool = math.sqrt(((n1 - 1.0) * (sd1**2) + (n0 - 1.0) * (sd0**2)) / dof)
            j_corr = 1.0 - (3.0 / (4.0 * dof - 1.0)) if dof > 1 else 1.0
            eff = j_corr * ((m1 - m0) / s_pool)
            se = math.sqrt((n1 + n0) / (n1 * n0) + (eff**2) / (2.0 * (n1 + n0)))
        else:
            eff = m1 - m0
            se = math.sqrt((sd1**2) / n1 + (sd0**2) / n0)
        records.append(
            {
                "study": study,
                "effect_size": eff,
                "log_effect": eff,
                "se": se,
                "ci_lower": eff - 1.96 * se,
                "ci_upper": eff + 1.96 * se,
                "is_ratio": False,
                "effect_measure": effect_measure,
            }
        )
    return pd.DataFrame(records)


def run_meta_analysis(
    df: pd.DataFrame,
    use_hksj: bool = False,
    alpha: float = 0.05,
    effect_col: str = "log_effect",
    se_col: str = "se",
    study_col: str = "study",
    exponentiate: bool | None = None,
) -> dict[str, Any]:
    """Perform Fixed-Effect and DerSimonian-Laird Random-Effects Meta-Analysis.

    Parameters:
        df: DataFrame containing study estimates and standard errors.
        use_hksj: If True, uses Hartung-Knapp-Sidik-Jonkman adjustment for random effects.
        alpha: Significance level (default 0.05).
        effect_col: Column name containing effect sizes (default 'log_effect').
        se_col: Column name containing standard errors (default 'se').
        study_col: Column name containing study identifiers (default 'study').
        exponentiate: Whether to exponentiate log effects for display (effect_disp, CIs).
            When None, infers True if 'is_ratio' column is True or effect_col starts with 'log_',
            otherwise False.
    """
    if not (isinstance(alpha, (int, float)) and math.isfinite(alpha) and 0.0 < alpha < 1.0):
        raise ValueError("alpha must be a finite number strictly between 0 and 1.")

    df_work = df.copy()
    k = len(df_work)
    if k < 2:
        raise ValueError(f"Meta-analysis requires at least 2 studies (got {k}).")

    if exponentiate is not None:
        is_ratio = bool(exponentiate)
    elif "is_ratio" in df_work.columns:
        col = df_work["is_ratio"]
        if col.isna().any() or not all(isinstance(v, (bool, np.bool_)) for v in col):
            raise ValueError(
                "Column 'is_ratio' contains missing or non-boolean values; pass 'exponentiate' explicitly."
            )
        first_val = bool(col.iloc[0])
        if not all(bool(v) == first_val for v in col):
            raise ValueError(
                "Column 'is_ratio' contains conflicting values across studies; pass 'exponentiate' explicitly."
            )
        is_ratio = first_val
    elif isinstance(effect_col, str) and effect_col.startswith("log_"):
        is_ratio = True
    else:
        is_ratio = False
    if effect_col in df_work.columns:
        theta = df_work[effect_col].to_numpy(dtype=float)
    elif "effect_size" in df_work.columns:
        raw_eff = df_work["effect_size"].to_numpy(dtype=float)
        if not np.all(np.isfinite(raw_eff)):
            raise ValueError("Non-finite values detected in effect_size.")
        if is_ratio:
            if np.any(raw_eff <= 0):
                raise ValueError("Ratio effect_size values must be > 0 to log-transform.")
            theta = np.log(raw_eff)
        else:
            theta = raw_eff
    else:
        raise ValueError(f"Column '{effect_col}' not found in DataFrame.")

    se = df_work[se_col].to_numpy(dtype=float)
    if not np.all(np.isfinite(theta)):
        raise ValueError("Effects array contains non-finite values (NaN or inf).")
    if not np.all(np.isfinite(se)) or np.any(se <= 0):
        raise ValueError("Standard errors must be strictly positive and finite.")

    w_fe = 1.0 / (se**2)
    sum_w_fe = float(np.sum(w_fe))
    theta_fe = float(np.sum(w_fe * theta) / sum_w_fe)
    se_fe = math.sqrt(1.0 / sum_w_fe)
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    ci_fe_low, ci_fe_high = theta_fe - z_crit * se_fe, theta_fe + z_crit * se_fe
    p_fe = float(2.0 * stats.norm.sf(abs(theta_fe / se_fe)))

    q_stat = float(np.sum(w_fe * ((theta - theta_fe) ** 2)))
    df_q = k - 1
    p_q = float(stats.chi2.sf(q_stat, df_q))
    c_const = sum_w_fe - float(np.sum(w_fe**2) / sum_w_fe)
    tau2 = max(0.0, (q_stat - df_q) / c_const) if c_const > 0 else 0.0
    tau = math.sqrt(tau2)
    i2 = max(0.0, ((q_stat - df_q) / q_stat) * 100.0) if q_stat > 0 else 0.0
    h2 = max(1.0, q_stat / df_q) if df_q > 0 else 1.0

    w_re = 1.0 / (se**2 + tau2)
    sum_w_re = float(np.sum(w_re))
    theta_re = float(np.sum(w_re * theta) / sum_w_re)

    if use_hksj and k >= 3:
        q_hksj = float((1.0 / (k - 1)) * np.sum(w_re * ((theta - theta_re) ** 2)))
        se_re = math.sqrt(max(1.0, q_hksj) / sum_w_re)
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=k - 1))
        ci_re_low, ci_re_high = theta_re - t_crit * se_re, theta_re + t_crit * se_re
        p_re = float(2.0 * stats.t.sf(abs(theta_re / se_re), df=k - 1))
    else:
        se_re = math.sqrt(1.0 / sum_w_re)
        ci_re_low, ci_re_high = theta_re - z_crit * se_re, theta_re + z_crit * se_re
        p_re = float(2.0 * stats.norm.sf(abs(theta_re / se_re)))

    pi_low, pi_high = np.nan, np.nan
    if k >= 3:
        t_crit_pi = float(stats.t.ppf(1.0 - alpha / 2.0, df=max(1, k - 2)))
        se_pi = math.sqrt(tau2 + se_re**2)
        pi_low, pi_high = theta_re - t_crit_pi * se_pi, theta_re + t_crit_pi * se_pi

    def _disp(val: float) -> float:
        return float(math.exp(val)) if is_ratio else float(val)

    egger_p = None
    is_smd = (
        "effect_measure" in df_work.columns
        and str(df_work["effect_measure"].iloc[0]).upper() == "SMD"
    )
    if (
        study_col in df_work.columns
        and not is_ratio
        and not is_smd
        and df_work[study_col].dropna().nunique() >= 10
    ):
        if not df_work[study_col].is_unique:
            raise ValueError("Duplicate study IDs detected; Egger's test requires unique studies.")
        prec = 1.0 / se
        if len(np.unique(prec)) >= 2:
            snd = theta / se
            # Egger asymmetry = INTERCEPT of SND ~ precision (linregress.pvalue tests the slope).
            res_egger = stats.linregress(prec, snd)
            t_inter = res_egger.intercept / res_egger.intercept_stderr
            egger_p = float(2.0 * stats.t.sf(abs(t_inter), df=k - 2))

    leave_one_out = []
    if k >= 3 and study_col in df_work.columns:
        for idx in range(k):
            th_sub = np.delete(theta, idx)
            se_sub = np.delete(se, idx)
            w_sub = 1.0 / (se_sub**2)
            sw_sub = float(np.sum(w_sub))
            th_fe_sub = float(np.sum(w_sub * th_sub) / sw_sub)
            q_sub = float(np.sum(w_sub * ((th_sub - th_fe_sub) ** 2)))
            c_sub = sw_sub - float(np.sum(w_sub**2) / sw_sub)
            t2_sub = max(0.0, (q_sub - (k - 2)) / c_sub) if c_sub > 0 else 0.0
            wre_sub = 1.0 / (se_sub**2 + t2_sub)
            swre_sub = float(np.sum(wre_sub))
            th_re_sub = float(np.sum(wre_sub * th_sub) / swre_sub)
            sere_sub = math.sqrt(1.0 / swre_sub)
            leave_one_out.append(
                {
                    "omitted_study": str(df_work[study_col].iloc[idx]),
                    "effect_disp": _disp(th_re_sub),
                    "ci_lower": _disp(th_re_sub - z_crit * sere_sub),
                    "ci_upper": _disp(th_re_sub + z_crit * sere_sub),
                }
            )

    studies_out = df_work.copy()
    studies_out["weight_fe_pct"] = (w_fe / sum_w_fe) * 100.0
    studies_out["weight_re_pct"] = (w_re / sum_w_re) * 100.0

    return {
        "k": k,
        "is_ratio": is_ratio,
        "studies": studies_out,
        "fixed_effect": {
            "log_effect": theta_fe,
            "effect_disp": _disp(theta_fe),
            "se": se_fe,
            "ci_lower": _disp(ci_fe_low),
            "ci_upper": _disp(ci_fe_high),
            "p_value": p_fe,
        },
        "random_effects": {
            "log_effect": theta_re,
            "effect_disp": _disp(theta_re),
            "se": se_re,
            "ci_lower": _disp(ci_re_low),
            "ci_upper": _disp(ci_re_high),
            "p_value": p_re,
            "prediction_interval": [_disp(pi_low), _disp(pi_high)]
            if not np.isnan(pi_low)
            else None,
        },
        "heterogeneity": {
            "estimator": "DerSimonian-Laird",
            "Q": q_stat,
            "df": df_q,
            "p_value": p_q,
            "I2": i2,
            "H2": h2,
            "tau2": tau2,
            "tau": tau,
            "egger_p_value": egger_p,
        },
        "leave_one_out": leave_one_out,
    }
```

"""
Firth's Penalized Likelihood Regression Models.

Implements Firth's penalized logistic regression and Firth's penalized Cox PH
using firthmodels (>= 0.8.2) to resolve monotone likelihood and complete/quasi-complete
separation in sparse event and small sample clinical studies.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy.stats as stats
from firthmodels import FirthCoxPH, FirthLogisticRegression, detect_separation

from medstat.logging import get_logger

logger = get_logger(__name__)


def check_separation(
    y: pd.Series | np.ndarray,
    X: pd.DataFrame | np.ndarray,
) -> dict[str, Any]:
    """
    Check for quasi-complete or complete linear separation in binary data.
    """
    y_arr = np.asarray(y, dtype=int).ravel()
    X_arr = np.asarray(X, dtype=float)

    is_separated = detect_separation(X_arr, y_arr)
    return {
        "is_separated": bool(is_separated),
        "recommendation": "Use Firth penalized regression"
        if is_separated
        else "Standard MLE acceptable",
    }


def fit_firth_logistic(
    y: pd.Series | np.ndarray,
    X: pd.DataFrame | np.ndarray,
    feature_names: list[str] | None = None,
    fit_intercept: bool = True,
    penalty_weight: float = 1.0,
    ci_method: Literal["pl", "wald"] = "pl",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Fit Firth's penalized logistic regression model.

    Parameters:
        y: Binary outcome vector (0/1).
        X: Covariate matrix or DataFrame.
        feature_names: Optional names for covariates.
        fit_intercept: Whether to fit an intercept term.
        penalty_weight: Firth penalty scale parameter (default 1.0 = standard Jeffreys prior).
        ci_method: 'pl' for Profile Likelihood (recommended) or 'wald'.
        alpha: Significance level for confidence intervals (default 0.05 for 95% CI).

    Returns:
        dict containing:
            - 'model': fitted FirthLogisticRegression instance
            - 'summary_df': pd.DataFrame with estimate, odds_ratio, ci_lower, ci_upper, p_value
            - 'loglik': penalized log-likelihood
            - 'aic', 'bic': model fit information criteria
            - 'ci_fallback': boolean indicating if PL fell back to Wald
    """
    y_arr = np.asarray(y, dtype=float).ravel()
    if isinstance(X, pd.DataFrame):
        names = list(X.columns)
        X_arr = X.to_numpy(dtype=float)
    else:
        X_arr = np.asarray(X, dtype=float)
        names = (
            feature_names
            if feature_names is not None
            else [f"x{i}" for i in range(X_arr.shape[1])]
        )

    if fit_intercept:
        # Check if intercept is already in X
        has_const = np.allclose(X_arr[:, 0], 1.0) if X_arr.shape[1] > 0 else False
        if not has_const:
            X_arr = np.column_stack([np.ones(len(y_arr)), X_arr])
            names = ["(Intercept)"] + names

    model = FirthLogisticRegression(fit_intercept=False, penalty_weight=penalty_weight)
    model.fit(X_arr, y_arr)

    coefs = np.asarray(model.coef_).ravel()
    se = (
        np.asarray(model.bse_).ravel()
        if hasattr(model, "bse_")
        else np.zeros_like(coefs)
    )

    # 1. P-values via penalized Likelihood Ratio Test (LRT) with Wald fallback
    lrt_fallback_vars: list[str] = []
    try:
        model.lrt()
        lrt_p = getattr(model, "lrt_pvalues_", None)
        if isinstance(lrt_p, (np.ndarray, list, pd.Series)) and not np.all(
            np.isnan(lrt_p)
        ):
            lrt_arr = np.asarray(lrt_p, dtype=float).ravel()
            if np.isnan(lrt_arr).any() and hasattr(model, "pvalues_"):
                pvals = np.where(np.isnan(lrt_arr), model.pvalues_, lrt_arr)
                for idx, is_nan in enumerate(np.isnan(lrt_arr)):
                    if is_nan and idx < len(names):
                        lrt_fallback_vars.append(names[idx])
            else:
                pvals = lrt_arr
        else:
            pvals = (
                np.asarray(model.pvalues_, dtype=float).ravel()
                if hasattr(model, "pvalues_")
                else np.full_like(coefs, np.nan)
            )
            lrt_fallback_vars = list(names)
    except Exception:
        pvals = (
            np.asarray(model.pvalues_, dtype=float).ravel()
            if hasattr(model, "pvalues_")
            else np.full_like(coefs, np.nan)
        )
        lrt_fallback_vars = list(names)

    # 2. Confidence intervals (Profile Likelihood first, fallback to Wald)
    ci_fallback = False
    wald_low = coefs - stats.norm.ppf(1.0 - alpha / 2.0) * se
    wald_high = coefs + stats.norm.ppf(1.0 - alpha / 2.0) * se

    if ci_method == "pl":
        try:
            pl_ci = np.asarray(model.conf_int(method="pl", alpha=alpha), dtype=float)
            if pl_ci.ndim == 2 and pl_ci.shape == (len(coefs), 2):
                low = pl_ci[:, 0]
                high = pl_ci[:, 1]
                non_finite = ~np.isfinite(low) | ~np.isfinite(high)
                if non_finite.any():
                    low = np.where(non_finite, wald_low, low)
                    high = np.where(non_finite, wald_high, high)
                    ci_fallback = True
            else:
                low, high = wald_low, wald_high
                ci_fallback = True
        except Exception:
            low, high = wald_low, wald_high
            ci_fallback = True
    else:
        low, high = wald_low, wald_high

    odds_ratios = np.exp(coefs)
    or_ci_low = np.exp(low)
    or_ci_high = np.exp(high)

    summary_df = pd.DataFrame(
        {
            "estimate": coefs,
            "std_error": se,
            "ci_lower": low,
            "ci_upper": high,
            "odds_ratio": odds_ratios,
            "or_ci_lower": or_ci_low,
            "or_ci_upper": or_ci_high,
            "p_value": pvals,
        },
        index=names,
    )
    summary_df.index.name = "term"

    # Fit metrics
    k_params = len(coefs)
    n_samples = len(y_arr)
    llf = getattr(model, "loglik_", np.nan)
    aic = 2 * k_params - 2 * llf if np.isfinite(llf) else np.nan
    bic = k_params * np.log(n_samples) - 2 * llf if np.isfinite(llf) else np.nan

    return {
        "model": model,
        "summary_df": summary_df,
        "loglik": float(llf) if np.isfinite(llf) else np.nan,
        "aic": float(aic) if np.isfinite(aic) else np.nan,
        "bic": float(bic) if np.isfinite(bic) else np.nan,
        "ci_fallback": ci_fallback,
        "lrt_fallback_vars": lrt_fallback_vars,
    }


def fit_firth_cox(
    time: pd.Series | np.ndarray,
    event: pd.Series | np.ndarray,
    X: pd.DataFrame | np.ndarray,
    covariate_names: list[str] | None = None,
    feature_names: list[str] | None = None,
    penalty_weight: float = 1.0,
    ci_method: Literal["pl", "wald"] = "pl",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Fit Firth's penalized Cox proportional hazards regression model.

    Parameters:
        time: Event or censoring time vector.
        event: Binary event indicator (True/1=event, False/0=censored).
        X: Covariate matrix or DataFrame.
        covariate_names: Optional names for covariates.
        feature_names: Alias for covariate_names.
        penalty_weight: Penalty scale parameter (default 1.0).
        ci_method: 'pl' for Profile Likelihood or 'wald'.
        alpha: Significance level for confidence intervals (default 0.05).

    Returns:
        dict containing:
            - 'model': fitted FirthCoxPH instance
            - 'summary_df': pd.DataFrame with estimate, hazard_ratio, ci_lower, ci_upper, p_value
    """
    time_arr = np.asarray(time, dtype=float).ravel()
    event_arr = np.asarray(event, dtype=bool).ravel()

    eff_names = covariate_names if covariate_names is not None else feature_names

    if isinstance(X, pd.DataFrame):
        names = list(X.columns)
        X_arr = X.to_numpy(dtype=float)
    else:
        X_arr = np.asarray(X, dtype=float)
        names = (
            eff_names
            if eff_names is not None
            else [f"x{i}" for i in range(X_arr.shape[1])]
        )

    model = FirthCoxPH(penalty_weight=penalty_weight)
    # FirthCoxPH expects y as tuple (event, time)
    model.fit(X_arr, (event_arr, time_arr))

    coefs = np.asarray(model.coef_).ravel()
    se = (
        np.asarray(model.bse_).ravel()
        if hasattr(model, "bse_")
        else np.zeros_like(coefs)
    )

    # Likelihood Ratio Test for p-values
    lrt_fallback_vars: list[str] = []
    try:
        model.lrt()
        lrt_p = getattr(model, "lrt_pvalues_", None)
        if isinstance(lrt_p, (np.ndarray, list, pd.Series)) and not np.all(
            np.isnan(lrt_p)
        ):
            lrt_arr = np.asarray(lrt_p, dtype=float).ravel()
            if np.isnan(lrt_arr).any() and hasattr(model, "pvalues_"):
                pvals = np.where(np.isnan(lrt_arr), model.pvalues_, lrt_arr)
            else:
                pvals = lrt_arr
        else:
            pvals = (
                np.asarray(model.pvalues_, dtype=float).ravel()
                if hasattr(model, "pvalues_")
                else np.full_like(coefs, np.nan)
            )
            lrt_fallback_vars = list(names)
    except Exception:
        pvals = (
            np.asarray(model.pvalues_, dtype=float).ravel()
            if hasattr(model, "pvalues_")
            else np.full_like(coefs, np.nan)
        )
        lrt_fallback_vars = list(names)

    # Confidence Intervals
    ci_fallback = False
    wald_low = coefs - stats.norm.ppf(1.0 - alpha / 2.0) * se
    wald_high = coefs + stats.norm.ppf(1.0 - alpha / 2.0) * se

    if ci_method == "pl":
        try:
            pl_ci = np.asarray(model.conf_int(method="pl", alpha=alpha), dtype=float)
            if pl_ci.ndim == 2 and pl_ci.shape == (len(coefs), 2):
                low = pl_ci[:, 0]
                high = pl_ci[:, 1]
                non_finite = ~np.isfinite(low) | ~np.isfinite(high)
                if non_finite.any():
                    low = np.where(non_finite, wald_low, low)
                    high = np.where(non_finite, wald_high, high)
                    ci_fallback = True
            else:
                low, high = wald_low, wald_high
                ci_fallback = True
        except Exception:
            low, high = wald_low, wald_high
            ci_fallback = True
    else:
        low, high = wald_low, wald_high

    hazard_ratios = np.exp(coefs)
    hr_ci_low = np.exp(low)
    hr_ci_high = np.exp(high)

    summary_df = pd.DataFrame(
        {
            "estimate": coefs,
            "std_error": se,
            "ci_lower": low,
            "ci_upper": high,
            "hazard_ratio": hazard_ratios,
            "hr_ci_lower": hr_ci_low,
            "hr_ci_upper": hr_ci_high,
            "p_value": pvals,
        },
        index=names,
    )
    summary_df.index.name = "term"

    return {
        "model": model,
        "summary_df": summary_df,
        "ci_fallback": ci_fallback,
        "lrt_fallback_vars": lrt_fallback_vars,
    }

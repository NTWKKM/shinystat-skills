"""
Generalized Linear Models (GLM) and Regression Core.

Provides OLS linear regression, standard logistic regression (MLE),
Poisson regression, and Negative Binomial regression using statsmodels.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm

from medstat.logging import get_logger

logger = get_logger(__name__)


def fit_linear_regression(
    y: pd.Series | np.ndarray,
    X: pd.DataFrame | np.ndarray,
    cov_type: Literal["nonrobust", "HC1", "HC3"] = "HC3",
    add_constant: bool = True,
) -> dict[str, Any]:
    """
    Fit Ordinary Least Squares (OLS) linear regression with robust standard errors.
    """
    y_arr = np.asarray(y, dtype=float).ravel()
    if isinstance(X, pd.DataFrame):
        names = list(X.columns)
        X_mat = X.to_numpy(dtype=float)
    else:
        X_mat = np.asarray(X, dtype=float)
        names = [f"x{i}" for i in range(X_mat.shape[1])]

    if add_constant:
        X_mat = sm.add_constant(X_mat)
        names = ["const"] + names

    model = sm.OLS(y_arr, X_mat)
    result = model.fit(cov_type=cov_type)

    summary_df = pd.DataFrame(
        {
            "coef": result.params,
            "std_error": result.bse,
            "t_stat": result.tvalues,
            "p_value": result.pvalues,
            "ci_lower": result.conf_int()[:, 0],
            "ci_upper": result.conf_int()[:, 1],
        },
        index=names,
    )

    return {
        "model": result,
        "summary_df": summary_df,
        "r_squared": float(result.rsquared),
        "r_squared_adj": float(result.rsquared_adj),
        "f_statistic": float(result.fvalue) if result.fvalue is not None else np.nan,
        "f_pvalue": float(result.f_pvalue) if result.f_pvalue is not None else np.nan,
        "aic": float(result.aic),
        "bic": float(result.bic),
    }


def fit_standard_logistic(
    y: pd.Series | np.ndarray,
    X: pd.DataFrame | np.ndarray,
    add_constant: bool = True,
    maxiter: int = 100,
) -> dict[str, Any]:
    """
    Fit standard Maximum Likelihood logistic regression.
    """
    y_arr = np.asarray(y, dtype=float).ravel()
    if isinstance(X, pd.DataFrame):
        names = list(X.columns)
        X_mat = X.to_numpy(dtype=float)
    else:
        X_mat = np.asarray(X, dtype=float)
        names = [f"x{i}" for i in range(X_mat.shape[1])]

    if add_constant:
        X_mat = sm.add_constant(X_mat)
        names = ["const"] + names

    model = sm.Logit(y_arr, X_mat)
    result = model.fit(maxiter=maxiter, disp=False)

    coefs = result.params
    conf = result.conf_int()
    odds_ratios = np.exp(coefs)
    or_low = np.exp(conf[:, 0])
    or_high = np.exp(conf[:, 1])

    summary_df = pd.DataFrame(
        {
            "coef": coefs,
            "std_error": result.bse,
            "z_stat": result.tvalues,
            "p_value": result.pvalues,
            "odds_ratio": odds_ratios,
            "or_ci_lower": or_low,
            "or_ci_upper": or_high,
        },
        index=names,
    )

    # McFadden's pseudo R-squared
    prsquared = getattr(result, "prsquared", np.nan)

    return {
        "model": result,
        "summary_df": summary_df,
        "pseudo_r2": float(prsquared) if np.isfinite(prsquared) else np.nan,
        "log_likelihood": float(result.llf),
        "aic": float(result.aic),
        "bic": float(result.bic),
    }


def fit_poisson_regression(
    y: pd.Series | np.ndarray,
    X: pd.DataFrame | np.ndarray,
    exposure: pd.Series | np.ndarray | None = None,
    add_constant: bool = True,
) -> dict[str, Any]:
    """
    Fit Poisson regression model for count data.
    """
    y_arr = np.asarray(y, dtype=float).ravel()
    if isinstance(X, pd.DataFrame):
        names = list(X.columns)
        X_mat = X.to_numpy(dtype=float)
    else:
        X_mat = np.asarray(X, dtype=float)
        names = [f"x{i}" for i in range(X_mat.shape[1])]

    if add_constant:
        X_mat = sm.add_constant(X_mat)
        names = ["const"] + names

    exposure_arr = (
        np.asarray(exposure, dtype=float).ravel() if exposure is not None else None
    )

    model = sm.Poisson(y_arr, X_mat, exposure=exposure_arr)
    result = model.fit(disp=False)

    coefs = result.params
    conf = result.conf_int()
    irr = np.exp(coefs)
    irr_low = np.exp(conf[:, 0])
    irr_high = np.exp(conf[:, 1])

    summary_df = pd.DataFrame(
        {
            "coef": coefs,
            "std_error": result.bse,
            "z_stat": result.tvalues,
            "p_value": result.pvalues,
            "irr": irr,
            "irr_ci_lower": irr_low,
            "irr_ci_upper": irr_high,
        },
        index=names,
    )

    return {
        "model": result,
        "summary_df": summary_df,
        "log_likelihood": float(result.llf),
        "aic": float(result.aic),
        "bic": float(result.bic),
    }


def fit_negative_binomial(
    y: pd.Series | np.ndarray,
    X: pd.DataFrame | np.ndarray,
    exposure: pd.Series | np.ndarray | None = None,
    add_constant: bool = True,
) -> dict[str, Any]:
    """
    Fit Negative Binomial regression for overdispersed count data.
    """
    y_arr = np.asarray(y, dtype=float).ravel()
    if isinstance(X, pd.DataFrame):
        names = list(X.columns)
        X_mat = X.to_numpy(dtype=float)
    else:
        X_mat = np.asarray(X, dtype=float)
        names = [f"x{i}" for i in range(X_mat.shape[1])]

    if add_constant:
        X_mat = sm.add_constant(X_mat)
        names = ["const"] + names

    exposure_arr = (
        np.asarray(exposure, dtype=float).ravel() if exposure is not None else None
    )

    model = sm.NegativeBinomial(y_arr, X_mat, exposure=exposure_arr)
    result = model.fit(disp=False)

    coefs = result.params
    conf = result.conf_int()
    irr = np.exp(coefs)
    irr_low = np.exp(conf[:, 0])
    irr_high = np.exp(conf[:, 1])

    summary_df = pd.DataFrame(
        {
            "coef": coefs,
            "std_error": result.bse,
            "z_stat": result.tvalues,
            "p_value": result.pvalues,
            "irr": irr,
            "irr_ci_lower": irr_low,
            "irr_ci_upper": irr_high,
        },
        index=names,
    )

    return {
        "model": result,
        "summary_df": summary_df,
        "alpha_dispersion": float(result.params.get("alpha", np.nan)),
        "log_likelihood": float(result.llf),
        "aic": float(result.aic),
        "bic": float(result.bic),
    }

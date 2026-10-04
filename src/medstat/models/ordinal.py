"""
Ordinal Regression Models and Proportional Odds Diagnostics.

Provides cumulative logit (proportional odds) regression using statsmodels OrderedModel,
Brant test (Brant 1990) for parallel slopes assumption, and multinomial logistic fallback.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel

from medstat.logging import get_logger

logger = get_logger(__name__)


def _extract_ordered_categories(
    y_series: pd.Series,
) -> tuple[pd.Series, list[Any] | np.ndarray]:
    """
    Extract observed categories in declared order.

    For pandas Categorical series, preserves the declared category hierarchy
    (y_series.cat.categories) filtered to observed levels.
    For other types, sorts unique values.
    """
    if isinstance(y_series.dtype, pd.CategoricalDtype):
        y_cleaned = y_series.cat.remove_unused_categories()
        observed = set(y_cleaned.dropna().unique())
        categories = [c for c in y_cleaned.cat.categories if c in observed]
        return y_cleaned, categories
    return y_series, np.sort(y_series.unique())


def fit_proportional_odds(
    y: pd.Series | np.ndarray,
    X: pd.DataFrame | np.ndarray,
    distr: str = "logit",
    method: str = "bfgs",
    maxiter: int = 200,
) -> dict[str, Any]:
    """
    Fit cumulative link proportional odds model for ordinal outcomes.

    Parameters
    ----------
    y : pd.Series | np.ndarray
        Ordinal dependent variable with 3+ ordered levels.
    X : pd.DataFrame | np.ndarray
        Covariates design matrix (without constant/intercept).
    distr : str
        Link distribution ('logit' for proportional odds).
    method : str
        Optimization method ('bfgs', 'nm', etc.).
    maxiter : int
        Maximum iterations for optimizer.

    Returns
    -------
    dict[str, Any]
        Dictionary containing fitted model, summary_df with cumulative ORs,
        threshold parameters, log-likelihood, AIC, and BIC.
    """
    y_raw = pd.Series(y)
    valid_mask = y_raw.notna().to_numpy()
    y_series = y_raw[valid_mask]
    y_series, categories = _extract_ordered_categories(y_series)
    k_categories = len(categories)

    if k_categories < 3:
        raise ValueError(
            f"Ordinal model requires at least 3 ordered categories (found {k_categories}: {categories}). "
            "For binary outcomes (2 categories), use fit_standard_logistic or fit_firth_logistic."
        )

    # Ensure index alignment
    if isinstance(X, pd.DataFrame):
        X_df = X.loc[y_series.index].copy()
        var_names = list(X_df.columns)
    else:
        X_mat = np.asarray(X)
        if len(X_mat) != len(y_raw):
            raise ValueError(
                f"Length mismatch: y has {len(y_raw)} rows, X has {len(X_mat)} rows."
            )
        if len(y_series) < len(y_raw):
            X_mat = X_mat[valid_mask]
        var_names = [f"x{i}" for i in range(X_mat.shape[1])]
        X_df = pd.DataFrame(X_mat, index=y_series.index, columns=var_names)

    # OrderedModel requires numeric or categorical series
    model = OrderedModel(y_series, X_df, distr=distr)
    result = model.fit(method=method, maxiter=maxiter, disp=False)

    k_vars = model.k_vars

    coefs = result.params
    se = result.bse
    z_stat = result.tvalues
    p_values = result.pvalues
    conf = result.conf_int()

    # Distinguish slope parameters from cutpoint / threshold parameters
    odds_ratios = []
    or_lower = []
    or_upper = []

    for i in range(len(coefs)):
        if i < k_vars:
            # Predictor slope: cumulative Odds Ratio = exp(beta)
            b = coefs.iloc[i] if hasattr(coefs, "iloc") else coefs[i]
            ci_l = conf.iloc[i, 0] if hasattr(conf, "iloc") else conf[i, 0]
            ci_u = conf.iloc[i, 1] if hasattr(conf, "iloc") else conf[i, 1]
            odds_ratios.append(float(np.exp(b)))
            or_lower.append(float(np.exp(ci_l)))
            or_upper.append(float(np.exp(ci_u)))
        else:
            # Threshold / cutpoint parameter
            odds_ratios.append(np.nan)
            or_lower.append(np.nan)
            or_upper.append(np.nan)

    param_names = list(result.params.index)
    summary_df = pd.DataFrame(
        {
            "coef": coefs.values if hasattr(coefs, "values") else coefs,
            "std_error": se.values if hasattr(se, "values") else se,
            "z_stat": z_stat.values if hasattr(z_stat, "values") else z_stat,
            "p_value": p_values.values if hasattr(p_values, "values") else p_values,
            "odds_ratio": odds_ratios,
            "or_ci_lower": or_lower,
            "or_ci_upper": or_upper,
        },
        index=param_names,
    )

    # Construct threshold_df on the actual cutpoint scale using transform_threshold_params
    # statsmodels OrderedModel estimates cutpoint 0 and subsequent log-increments (gamma_j = log(cutpoint_j - cutpoint_{j-1})).
    # We transform these to actual cutpoint values excluding infinite endpoints [-inf, c_1, ..., c_{K-1}, inf]
    # and compute cutpoint standard errors using the delta method on the threshold covariance matrix.
    raw_thresh = model.transform_threshold_params(result.params)
    cutpoints = np.asarray(raw_thresh[1:-1], dtype=float)
    thresh_sub = summary_df.iloc[k_vars:].copy()
    k_thresh = len(thresh_sub)

    try:
        cov_params_df = result.cov_params()
        cov_thresh = cov_params_df.iloc[k_vars:, k_vars:].values
        # Delta method Jacobian J: d(c_i) / d(theta_j)
        # c_0 = theta_0 => dc_0/dtheta_0 = 1
        # c_i = theta_0 + sum_{j=1}^i exp(theta_j) => dc_i/dtheta_0 = 1, dc_i/dtheta_j = exp(theta_j) for j <= i
        J = np.zeros((k_thresh, k_thresh), dtype=float)
        J[:, 0] = 1.0
        param_vals = result.params.iloc[k_vars:].values
        for i in range(1, k_thresh):
            for j in range(1, i + 1):
                J[i, j] = np.exp(param_vals[j])
        cov_cutpoints = J @ cov_thresh @ J.T
        se_cutpoints = np.sqrt(np.maximum(0.0, np.diag(cov_cutpoints)))
        z_crit = 1.959963984540054
        ci_lower = cutpoints - z_crit * se_cutpoints
        ci_upper = cutpoints + z_crit * se_cutpoints
        p_cutpoints = 2.0 * (
            1.0
            - stats.norm.cdf(
                np.abs(cutpoints / np.where(se_cutpoints > 0, se_cutpoints, np.nan))
            )
        )
    except Exception:
        se_cutpoints = np.full(k_thresh, np.nan)
        ci_lower = np.full(k_thresh, np.nan)
        ci_upper = np.full(k_thresh, np.nan)
        p_cutpoints = np.full(k_thresh, np.nan)

    threshold_df = pd.DataFrame(
        {
            "coef": cutpoints,
            "std_error": se_cutpoints,
            "z_stat": cutpoints / np.where(se_cutpoints > 0, se_cutpoints, np.nan),
            "p_value": p_cutpoints,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "odds_ratio": np.full(k_thresh, np.nan),
            "or_ci_lower": np.full(k_thresh, np.nan),
            "or_ci_upper": np.full(k_thresh, np.nan),
            "scale": "cutpoint",
        },
        index=thresh_sub.index,
    )
    predictor_df = summary_df.iloc[:k_vars].copy()

    # Null model log-likelihood for pseudo R2
    try:
        null_mod = OrderedModel(y_series, np.zeros((len(y_series), 0)), distr=distr)
        null_res = null_mod.fit(method=method, maxiter=maxiter, disp=False)
        ll_null = float(null_res.llf)
        pseudo_r2 = float(1.0 - (result.llf / ll_null)) if ll_null < 0 else np.nan
    except Exception:
        pseudo_r2 = np.nan

    return {
        "model": result,
        "summary_df": summary_df,
        "predictor_df": predictor_df,
        "threshold_df": threshold_df,
        "categories": [
            int(c) if isinstance(c, (int, np.integer)) else str(c) for c in categories
        ],
        "k_categories": int(k_categories),
        "log_likelihood": float(result.llf),
        "pseudo_r2": pseudo_r2,
        "aic": float(result.aic),
        "bic": float(result.bic),
        "nobs": int(result.nobs),
    }


def test_proportional_odds(
    y: pd.Series | np.ndarray,
    X: pd.DataFrame | np.ndarray,
) -> dict[str, Any]:
    """
    Perform Brant's Wald test (Brant 1990) for the proportional odds assumption.

    Under H0 (parallel lines), the slope vector across binary cutpoints is equal:
    beta_1 = beta_2 = ... = beta_{K-1}.

    Returns omnibus test statistic and p-value, plus per-variable diagnostics.
    """
    y_raw = pd.Series(y)
    valid_mask = y_raw.notna().to_numpy()
    y_series = y_raw[valid_mask]
    y_series, categories = _extract_ordered_categories(y_series)
    K = len(categories)
    J = K - 1

    if K < 3:
        raise ValueError("Brant test requires an ordinal outcome with >= 3 categories.")

    if isinstance(X, pd.DataFrame):
        X_df = X.loc[y_series.index].copy()
    else:
        X_mat = np.asarray(X)
        if len(X_mat) != len(y_raw):
            raise ValueError(
                f"Length mismatch: y has {len(y_raw)} rows, X has {len(X_mat)} rows."
            )
        if len(y_series) < len(y_raw):
            X_mat = X_mat[valid_mask]
        X_df = pd.DataFrame(
            X_mat,
            index=y_series.index,
            columns=[f"x{i}" for i in range(X_mat.shape[1])],
        )

    p_covars = X_df.shape[1]
    var_names = list(X_df.columns)
    X_const = sm.add_constant(X_df).values
    n = len(y_series)

    betas = []
    fitted_probs = []
    inv_infos = []

    for j in range(J):
        # Cutpoint: Y > categories[j]
        # In declared category order, Y > categories[j] corresponds to Y in categories[j+1:]
        z_j = y_series.isin(categories[j + 1 :]).to_numpy(dtype=int)
        # Verify both classes exist at cutpoint
        if z_j.sum() == 0 or z_j.sum() == n:
            raise ValueError(
                f"Cutpoint {categories[j]} has zero cases in one class; cannot compute Brant test."
            )

        res_j = sm.Logit(z_j, X_const).fit(disp=False)
        betas.append(res_j.params[1:])  # Exclude intercept
        p_j = res_j.predict(X_const)
        fitted_probs.append(p_j)

        w_j = p_j * (1.0 - p_j)
        info_j = X_const.T @ (w_j[:, None] * X_const)
        inv_infos.append(np.linalg.pinv(info_j))

    # Construct block covariance matrix V for (beta_1, ..., beta_J)
    V = np.zeros((J * p_covars, J * p_covars))

    for j in range(J):
        for m in range(J):
            if j == m:
                V_jm = inv_infos[j]
            else:
                pj = fitted_probs[j]
                pm = fitted_probs[m]
                if j < m:
                    w_jm = pm * (1.0 - pj)
                else:
                    w_jm = pj * (1.0 - pm)
                info_jm = X_const.T @ (w_jm[:, None] * X_const)
                V_jm = inv_infos[j] @ info_jm @ inv_infos[m]

            # Store only submatrix corresponding to slope parameters (index 1:, 1:)
            V[j * p_covars : (j + 1) * p_covars, m * p_covars : (m + 1) * p_covars] = (
                V_jm[1:, 1:]
            )

    # Contrast matrix C comparing each cutpoint to the first: beta_{j+1} - beta_1 = 0
    C_omnibus = np.zeros(((J - 1) * p_covars, J * p_covars))
    for k in range(J - 1):
        C_omnibus[k * p_covars : (k + 1) * p_covars, 0:p_covars] = -np.eye(p_covars)
        C_omnibus[
            k * p_covars : (k + 1) * p_covars, (k + 1) * p_covars : (k + 2) * p_covars
        ] = np.eye(p_covars)

    beta_vec = np.concatenate(betas)
    diff_omnibus = C_omnibus @ beta_vec
    cov_diff_omnibus = C_omnibus @ V @ C_omnibus.T

    wald_omnibus = float(
        diff_omnibus.T @ np.linalg.pinv(cov_diff_omnibus) @ diff_omnibus
    )
    df_omnibus = (J - 1) * p_covars
    p_omnibus = float(stats.chi2.sf(wald_omnibus, df=df_omnibus))

    # Per-variable Brant test
    var_results = {}
    for v_idx, v_name in enumerate(var_names):
        # Contrast for variable v_idx across all cutpoints relative to cutpoint 1
        C_v = np.zeros((J - 1, J * p_covars))
        for k in range(J - 1):
            C_v[k, v_idx] = -1.0
            C_v[k, (k + 1) * p_covars + v_idx] = 1.0

        diff_v = C_v @ beta_vec
        cov_v = C_v @ V @ C_v.T
        stat_v = float(diff_v.T @ np.linalg.pinv(cov_v) @ diff_v)
        df_v = J - 1
        p_v = float(stats.chi2.sf(stat_v, df=df_v))

        cutpoint_betas = {
            f"cutpoint_{categories[j]}": float(betas[j][v_idx]) for j in range(J)
        }

        var_results[v_name] = {
            "statistic": stat_v,
            "df": df_v,
            "p_value": p_v,
            "passed": bool(p_v >= 0.05),
            "cutpoint_coefficients": cutpoint_betas,
        }

    return {
        "test": "Brant Wald Test",
        "omnibus": {
            "statistic": wald_omnibus,
            "df": df_omnibus,
            "p_value": p_omnibus,
            "passed": bool(p_omnibus >= 0.05),
        },
        "variables": var_results,
        "interpretation": (
            "No evidence against proportional odds assumption (omnibus p >= 0.05); however, "
            "this does not prove parallel slopes across all cutpoints. Inspect per-variable "
            "cutpoint coefficients for subtle non-proportionality."
            if p_omnibus >= 0.05
            else "Proportional odds assumption violated (p < 0.05); consider partial proportional odds or multinomial logistic regression."
        ),
    }


# Prevent pytest from collecting test_proportional_odds as a unit test
test_proportional_odds.__test__ = False
assess_proportional_odds = test_proportional_odds


def fit_multinomial_logistic(
    y: pd.Series | np.ndarray,
    X: pd.DataFrame | np.ndarray,
    add_constant: bool = True,
    maxiter: int = 100,
) -> dict[str, Any]:
    """
    Fit multinomial logit model as an unconstrained fallback when
    the proportional odds assumption is rejected.
    """
    y_raw = pd.Series(y)
    valid_mask = y_raw.notna().to_numpy()
    y_series = y_raw[valid_mask]
    if isinstance(X, pd.DataFrame):
        X_df = X.loc[y_series.index].copy()
    else:
        X_mat = np.asarray(X)
        if len(X_mat) != len(y_raw):
            raise ValueError(
                f"Length mismatch: y has {len(y_raw)} rows, X has {len(X_mat)} rows."
            )
        if len(y_series) < len(y_raw):
            X_mat = X_mat[valid_mask]
        X_df = pd.DataFrame(X_mat, index=y_series.index)

    if add_constant:
        X_mat = sm.add_constant(X_df).values
    else:
        X_mat = X_df.values

    model = sm.MNLogit(y_series.values, X_mat)
    result = model.fit(maxiter=maxiter, disp=False)

    return {
        "model": result,
        "log_likelihood": float(result.llf),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "params": result.params,
        "pvalues": result.pvalues,
    }

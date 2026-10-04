"""
Multilevel Models, Clustered Data, and Generalized Estimating Equations (GEE).

Provides cluster design effect calculation, population-averaged GEE with robust sandwich standard errors,
and linear mixed-effects random-intercept models using statsmodels.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.cov_struct import Autoregressive, Exchangeable, Independence
from statsmodels.genmod.families import Binomial, Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE

from medstat.logging import get_logger

logger = get_logger(__name__)


def calculate_design_effect(
    y: pd.Series | np.ndarray,
    cluster_ids: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """
    Calculate cluster-level Intraclass Correlation Coefficient (ICC_cluster),
    Design Effect (DEFF), and Effective Sample Size (N_eff).

    DEFF = 1 + (m_bar - 1) * ICC_cluster
    N_eff = N_total / DEFF
    """
    try:
        y_arr = np.asarray(y, dtype=float).ravel()
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Outcome 'y' for cluster design effect must be numeric (continuous). Failed conversion: {e}"
        ) from e
    c_arr = np.asarray(cluster_ids).ravel()

    valid = ~(np.isnan(y_arr) | pd.isna(c_arr))
    y_clean = y_arr[valid]
    c_clean = c_arr[valid]

    n_total = len(y_clean)
    unique_clusters, cluster_counts = np.unique(c_clean, return_counts=True)
    k_clusters = len(unique_clusters)

    if k_clusters < 2:
        return {
            "n_clusters": int(k_clusters),
            "n_total": int(n_total),
            "mean_cluster_size": float(n_total),
            "icc_cluster": 0.0,
            "design_effect": 1.0,
            "effective_sample_size": float(n_total),
            "interpretation": "Single cluster present; clustering cannot be evaluated.",
        }

    # One-way ANOVA decomposition for ICC
    grand_mean = np.mean(y_clean)
    ss_total = np.sum((y_clean - grand_mean) ** 2)

    cluster_means = np.array([np.mean(y_clean[c_clean == c]) for c in unique_clusters])
    ss_between = np.sum(cluster_counts * (cluster_means - grand_mean) ** 2)
    ss_within = ss_total - ss_between

    df_between = k_clusters - 1
    df_within = n_total - k_clusters

    ms_between = ss_between / df_between if df_between > 0 else 0.0
    ms_within = ss_within / df_within if df_within > 0 else 0.0

    # Effective cluster size for unequal cluster sizes (Fleiss 1986 / Donner 1980)
    m_bar = float(np.mean(cluster_counts))
    m_0 = (
        float(n_total - np.sum(cluster_counts**2) / n_total) / float(df_between)
        if df_between > 0
        else m_bar
    )

    # Variance components
    s2_w = ms_within
    s2_a = max(0.0, (ms_between - ms_within) / m_0) if m_0 > 0 else 0.0
    total_var = s2_a + s2_w

    icc = float(s2_a / total_var) if total_var > 0 else 0.0
    deff = float(1.0 + (m_bar - 1.0) * icc)
    n_eff = float(n_total / deff) if deff > 0 else float(n_total)

    return {
        "n_clusters": int(k_clusters),
        "n_total": int(n_total),
        "mean_cluster_size": float(m_bar),
        "cluster_size_min": int(np.min(cluster_counts)),
        "cluster_size_max": int(np.max(cluster_counts)),
        "icc_cluster": float(icc),
        "design_effect": float(deff),
        "effective_sample_size": float(n_eff),
        "interpretation": (
            f"DEFF = {deff:.2f} (ICC = {icc:.3f}). Substantial clustering present (DEFF > 1.5); "
            "standard regression CIs are artificially narrow. Use GEE or mixed-effects models."
            if deff > 1.5
            else f"DEFF = {deff:.2f} (ICC = {icc:.3f}). Clustering impact is minimal (DEFF <= 1.5)."
        ),
    }


def fit_gee(
    y: pd.Series | np.ndarray,
    X: pd.DataFrame | np.ndarray,
    cluster_ids: pd.Series | np.ndarray,
    family: Literal["binomial", "gaussian"] = "binomial",
    cov_struct: Literal[
        "exchangeable", "independence", "autoregressive"
    ] = "exchangeable",
    add_constant: bool = True,
    time: pd.Series | np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Fit population-averaged Generalized Estimating Equations (GEE)
    with robust (sandwich) standard errors.

    `time` (within-cluster ordering) is forwarded to GEE, e.g. for autoregressive structures.
    """
    family_key = str(family).lower()
    struct_key = str(cov_struct).lower()
    if family_key not in ("binomial", "gaussian"):
        raise ValueError(
            f"Unsupported GEE family '{family}'. Supported: 'binomial', 'gaussian'."
        )
    if struct_key not in ("exchangeable", "independence", "autoregressive"):
        raise ValueError(
            f"Unsupported GEE cov_struct '{cov_struct}'. "
            "Supported: 'exchangeable', 'independence', 'autoregressive'."
        )
    if struct_key == "autoregressive" and time is None:
        raise ValueError(
            "GEE with autoregressive correlation structure requires 'time' variable for ordering."
        )

    y_series = pd.Series(y)
    c_series = pd.Series(cluster_ids)

    valid = ~(y_series.isna() | c_series.isna())
    y_clean = y_series[valid].values
    c_clean = c_series[valid].values
    t_clean = None
    if time is not None:
        t_arr = np.asarray(time, dtype=float)[valid.values]
        if not np.isfinite(t_arr).all():
            raise ValueError(
                "Missing or non-finite values detected in GEE time variable."
            )
        t_clean = t_arr

    if isinstance(X, pd.DataFrame):
        X_df = X.loc[y_series[valid].index].copy()
        if X_df.isna().any().any():
            raise ValueError(
                "Missing values detected in covariates X. Covariates must be imputed or cleaned before fitting GEE."
            )
        var_names = list(X_df.columns)
        X_mat = X_df.values.astype(float)
    else:
        X_mat_raw = np.asarray(X)[valid]
        if np.isnan(X_mat_raw).any():
            raise ValueError(
                "Missing values detected in covariates X. Covariates must be imputed or cleaned before fitting GEE."
            )
        X_mat = X_mat_raw.astype(float)
        var_names = [f"x{i}" for i in range(X_mat.shape[1])]

    if add_constant:
        X_mat = sm.add_constant(X_mat)
        var_names = ["const"] + var_names

    # If time is provided, or for autoregressive structures, sort observations by (cluster_ids, time)
    # so Autoregressive(grid=True) evaluates lags on sequentially ordered rows within each cluster.
    if t_clean is not None:
        order = np.lexsort((t_clean, c_clean))
        y_clean = y_clean[order]
        X_mat = X_mat[order]
        c_clean = c_clean[order]
        t_clean = t_clean[order]
    else:
        # Ensure contiguous clusters if cluster_ids are not already grouped
        order = np.argsort(c_clean, kind="stable")
        y_clean = y_clean[order]
        X_mat = X_mat[order]
        c_clean = c_clean[order]

    # Select correlation structure
    struct_map = {
        "exchangeable": Exchangeable,
        "independence": Independence,
        "autoregressive": lambda: Autoregressive(grid=True),
    }
    struct = struct_map[struct_key]()

    # Select family
    if family_key == "binomial":
        fam = Binomial()
        is_binary = True
    else:
        fam = Gaussian()
        is_binary = False

    n_clusters = len(np.unique(c_clean))
    small_cluster_threshold = 40
    small_cluster_flag = bool(n_clusters < small_cluster_threshold)
    cov_type = "bias_reduced" if small_cluster_flag else "robust"

    model = GEE(
        y_clean, X_mat, groups=c_clean, time=t_clean, family=fam, cov_struct=struct
    )
    result = model.fit(cov_type=cov_type)

    coefs = result.params
    se = result.bse  # Standard errors from chosen covariance structure
    z_stat = result.tvalues
    p_values = result.pvalues
    conf = result.conf_int()

    if is_binary:
        odds_ratios = np.exp(coefs)
        or_low = np.exp(conf[:, 0])
        or_high = np.exp(conf[:, 1])
    else:
        odds_ratios = np.full_like(coefs, np.nan)
        or_low = np.full_like(coefs, np.nan)
        or_high = np.full_like(coefs, np.nan)

    summary_df = pd.DataFrame(
        {
            "coef": coefs,
            "std_error": se,
            "z_stat": z_stat,
            "p_value": p_values,
            "ci_lower": conf[:, 0],
            "ci_upper": conf[:, 1],
            "odds_ratio": odds_ratios,
            "or_ci_lower": or_low,
            "or_ci_upper": or_high,
        },
        index=var_names,
    )

    # QIC (Quasi-likelihood under the Independence model Criterion)
    qic_val = getattr(result, "qic", np.nan)
    if callable(qic_val):
        try:
            qic_res = qic_val()
            qic = float(qic_res[0]) if isinstance(qic_res, tuple) else float(qic_res)
        except Exception:
            qic = np.nan
    else:
        qic = float(qic_val) if np.isfinite(qic_val) else np.nan

    return {
        "model": result,
        "summary_df": summary_df,
        "cov_struct": cov_struct,
        "family": family,
        "nobs": int(result.nobs),
        "n_clusters": int(n_clusters),
        "cov_type": cov_type,
        "small_cluster_adjustment": small_cluster_flag,
        "small_cluster_threshold": small_cluster_threshold,
        "qic": qic,
        "scale": float(getattr(result, "scale", 1.0)),
    }


def fit_random_intercept(
    y: pd.Series | np.ndarray,
    X: pd.DataFrame | np.ndarray,
    cluster_ids: pd.Series | np.ndarray,
    add_constant: bool = True,
) -> dict[str, Any]:
    """
    Fit linear mixed-effects model with a random cluster intercept.
    """
    y_series = pd.Series(y)
    c_series = pd.Series(cluster_ids)

    valid = ~(y_series.isna() | c_series.isna())
    y_clean = y_series[valid].values
    c_clean = c_series[valid].values

    if isinstance(X, pd.DataFrame):
        X_df = X.loc[y_series[valid].index].copy()
        if X_df.isna().any().any():
            raise ValueError(
                "Missing values detected in covariates X. Covariates must be imputed or cleaned before fitting random intercept models."
            )
        var_names = list(X_df.columns)
    else:
        X_mat = np.asarray(X)[valid].astype(float)
        if np.isnan(X_mat).any():
            raise ValueError(
                "Missing values detected in covariates X. Covariates must be imputed or cleaned before fitting random intercept models."
            )
        var_names = [f"x{i}" for i in range(X_mat.shape[1])]
        X_df = pd.DataFrame(X_mat, columns=var_names)

    exog = sm.add_constant(X_df, has_constant="add") if add_constant else X_df
    model = sm.MixedLM(y_clean, exog.astype(float), groups=c_clean)
    result = model.fit()

    # Extract fixed-effect estimates (excluding variance components like 'Group Var')
    fe_names = list(result.fe_params.index)
    coefs = result.fe_params
    se = result.bse.loc[fe_names]
    z_stat = result.tvalues.loc[fe_names]
    p_values = result.pvalues.loc[fe_names]
    conf = result.conf_int().loc[fe_names]

    summary_df = pd.DataFrame(
        {
            "coef": coefs.values if hasattr(coefs, "values") else coefs,
            "std_error": se.values if hasattr(se, "values") else se,
            "z_stat": z_stat.values if hasattr(z_stat, "values") else z_stat,
            "p_value": p_values.values if hasattr(p_values, "values") else p_values,
            "ci_lower": conf.iloc[:, 0].values if hasattr(conf, "iloc") else conf[:, 0],
            "ci_upper": conf.iloc[:, 1].values if hasattr(conf, "iloc") else conf[:, 1],
        },
        index=fe_names,
    )

    # Calculate cluster ICC from variance components
    re_var = (
        float(result.cov_re.iloc[0, 0])
        if hasattr(result.cov_re, "iloc")
        else float(result.cov_re)
    )
    resid_var = float(result.scale)
    total_var = re_var + resid_var
    icc_re = float(re_var / total_var) if total_var > 0 else 0.0

    # Compute AIC and BIC from Maximum Likelihood fit (reml=False); report as unavailable if unable to compute
    aic: float | None = None
    bic: float | None = None
    try:
        res_ml = model.fit(reml=False)
        if hasattr(res_ml, "aic") and np.isfinite(res_ml.aic):
            aic = float(res_ml.aic)
        if hasattr(res_ml, "bic") and np.isfinite(res_ml.bic):
            bic = float(res_ml.bic)
    except Exception:
        aic = None
        bic = None

    return {
        "model": result,
        "summary_df": summary_df,
        "random_intercept_var": re_var,
        "residual_var": resid_var,
        "cluster_icc": icc_re,
        "aic": aic,
        "bic": bic,
        "nobs": int(result.nobs),
        "n_clusters": len(np.unique(c_clean)),
    }

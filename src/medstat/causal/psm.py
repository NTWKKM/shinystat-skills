"""
Propensity Score Matching (PSM) and Inverse Probability Weighting (IPW).

Provides logistic propensity score estimation, greedy nearest-neighbor matching
with caliper constraints, and inverse probability of treatment weighting.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial.distance import cdist
from scipy.special import expit
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    PerfectSeparationError,
    PerfectSeparationWarning,
)

from medstat.logging import get_logger

logger = get_logger(__name__)


# Propensity model estimators. The caller names one; none is substituted for another.
PS_METHODS: tuple[str, ...] = ("logistic", "firth")

# Unit-free separation bound: |coefficient| per SD of its covariate (the bound Recipe 18 and
# diagnostic.validation._fit_logit_shared use for the same purpose).
_PS_SEPARATION_LIMIT = 10.0


def normalize_ps_method(method: str) -> str:
    """Return the canonical propensity model name, or raise for an unsupported one."""
    key = str(method).strip().lower()
    if key not in PS_METHODS:
        raise ValueError(
            f"Unsupported propensity score method '{method}'. "
            f"Choose from: {', '.join(repr(m) for m in PS_METHODS)}."
        )
    return key


def calculate_propensity_score(
    df: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    method: str = "logistic",
) -> pd.Series:
    """
    Estimate propensity scores with the named model. No estimator is substituted for another.

    Parameters:
        df: Input DataFrame.
        treatment: Name of binary treatment/exposure column (0/1).
        covariates: List of baseline confounder column names.
        method: 'logistic' (maximum likelihood, default) or 'firth' (Firth's penalized
            likelihood, finite under separation).

    Returns:
        pd.Series containing estimated propensity scores aligned with df.index.

    Raises:
        ValueError: unsupported method; no complete observations; treatment not 0/1 or
            missing a group; rank-deficient design; or, for 'logistic', a fit that shows
            separation or did not converge.
    """
    method_key = normalize_ps_method(method)
    cols = [treatment] + covariates
    clean_sub = df[cols].dropna()

    if clean_sub.empty:
        raise ValueError(
            "No complete observations available to estimate propensity scores."
        )

    t_vals = clean_sub[treatment].to_numpy(dtype=float)
    if not np.all(np.isin(t_vals, [0.0, 1.0])) or np.unique(t_vals).size < 2:
        raise ValueError(
            f"Treatment column '{treatment}' must be binary 0/1 with both groups present "
            "to estimate propensity scores."
        )
    y = t_vals.astype(int)
    X = clean_sub[covariates].astype(float)
    X_arr = X.to_numpy(dtype=float)

    # Rank check on standardized columns, so it does not depend on covariate units.
    sd = X_arr.std(axis=0)
    constant = [c for c, s in zip(covariates, sd) if s == 0.0]
    z_design = np.column_stack(
        [np.ones(len(y)), (X_arr - X_arr.mean(axis=0)) / np.where(sd > 0, sd, 1.0)]
    )
    if constant or np.linalg.matrix_rank(z_design) < z_design.shape[1]:
        detail = f" Constant covariates: {constant}." if constant else ""
        raise ValueError(
            "Propensity score design matrix is rank deficient (constant or collinear "
            f"covariates).{detail} Remove or combine the redundant covariates."
        )

    if method_key == "firth":
        from medstat.models.firth import fit_firth_logistic

        firth_fit = fit_firth_logistic(y, X, fit_intercept=True, ci_method="wald")
        coef = np.asarray(firth_fit["model"].coef_, dtype=float).ravel()
        if not firth_fit["converged"] or not np.all(np.isfinite(coef)):
            raise ValueError(
                "Firth propensity score model did not converge to finite estimates: "
                "the covariates determine treatment. Review confounding and positivity, "
                "and state the target estimand, before changing the adjustment set "
                "(Grilling Gate #3)."
            )
        ps_clean = expit(coef[0] + X_arr @ coef[1:])
    else:
        X_const = sm.add_constant(X, has_constant="add")
        reason = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", PerfectSeparationWarning)
                # Convergence and finiteness are read from the fit below.
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", RuntimeWarning)
                model = sm.Logit(y, X_const).fit(disp=False)
        except (
            PerfectSeparationWarning,
            PerfectSeparationError,
            np.linalg.LinAlgError,
        ) as e:
            reason = f"{type(e).__name__}: {e}"
        else:
            params = np.asarray(model.params, dtype=float)
            per_sd = np.abs(params[1:]) * sd
            if not np.all(np.isfinite(params)):
                reason = "non-finite coefficients"
            elif not bool(getattr(model, "mle_retvals", {}).get("converged", True)):
                reason = "the optimizer did not converge"
            elif np.any(per_sd > _PS_SEPARATION_LIMIT):
                flagged = [
                    c for c, g in zip(covariates, per_sd) if g > _PS_SEPARATION_LIMIT
                ]
                reason = f"|coefficient| per SD exceeds {_PS_SEPARATION_LIMIT:g} for {flagged}"
        if reason is not None:
            raise ValueError(
                f"Propensity score model shows separation or did not converge ({reason}); "
                "its fitted scores approach 0 or 1 and are not usable. Review confounding "
                "and positivity, and state the target estimand, before changing the "
                "adjustment set (Grilling Gate #3), or pass method='firth' "
                "(ps_method='firth') for Firth's penalized likelihood."
            )
        ps_clean = model.predict(X_const)

    ps_full = pd.Series(index=df.index, dtype=float)
    ps_full.loc[clean_sub.index] = np.asarray(ps_clean, dtype=float)
    return ps_full


def perform_matching(
    df: pd.DataFrame,
    treatment: str,
    ps_col: str,
    caliper: float = 0.2,
    ratio: int = 1,
) -> pd.DataFrame:
    """
    Perform greedy 1:1 or 1:k nearest-neighbor propensity score matching within caliper,
    processing treated subjects in descending propensity order (largest first).

    Parameters:
        df: Input DataFrame containing treatment and propensity scores.
        treatment: Column name for treatment indicator (1=treated, 0=control).
        ps_col: Column name containing propensity scores.
        caliper: Caliper width in standard deviations of logit propensity score (default 0.2).
        ratio: Number of controls matched to each treated subject (default 1).

    Returns:
        pd.DataFrame of matched subjects with columns 'pair_id' and 'matching_weight'.
    """
    sub = df.dropna(subset=[treatment, ps_col]).copy()
    treated = sub[sub[treatment] == 1]
    control = sub[sub[treatment] == 0]

    if treated.empty or control.empty:
        raise ValueError("Matching requires both treated and control subjects.")

    ps_vals = np.clip(sub[ps_col].to_numpy(dtype=float), 1e-7, 1.0 - 1e-7)
    sub["_logit_ps"] = np.log(ps_vals / (1.0 - ps_vals))
    sd_logit = float(np.std(sub["_logit_ps"], ddof=1)) if len(sub) > 1 else 0.0
    caliper_threshold = caliper * sd_logit if sd_logit > 0 else caliper

    treated_logit = sub.loc[treated.index, ["_logit_ps"]].values
    control_logit = sub.loc[control.index, ["_logit_ps"]].values

    # Pairwise absolute distance matrix on logit propensity score scale
    distances = cdist(treated_logit, control_logit, metric="euclidean")

    matched_treated_idx = []
    matched_control_idx = []
    treated_weights = []
    control_weights = []
    treated_pair_ids = []
    control_pair_ids = []
    used_controls = set()
    current_pair = 0

    # Greedy matching in descending propensity order ("largest first", the MatchIt default and the order
    # used by cloud Recipe 6): the hardest-to-match treated subjects choose first, independent of row order.
    match_order = np.argsort(-treated_logit[:, 0], kind="stable")
    for i in match_order:
        t_idx = treated.index[i]
        # Find closest available controls within caliper
        sorted_ctrl_indices = np.argsort(distances[i, :])
        curr_matched_controls = []

        for c_pos in sorted_ctrl_indices:
            c_idx = control.index[c_pos]
            if c_idx in used_controls:
                continue

            dist = distances[i, c_pos]
            if dist <= caliper_threshold:
                curr_matched_controls.append(c_idx)
                used_controls.add(c_idx)
                if len(curr_matched_controls) >= ratio:
                    break

        if curr_matched_controls:
            matched_treated_idx.append(t_idx)
            treated_pair_ids.append(current_pair)
            treated_weights.append(1.0)

            ctrl_weight = 1.0 / len(curr_matched_controls)
            for c_idx in curr_matched_controls:
                matched_control_idx.append(c_idx)
                control_pair_ids.append(current_pair)
                control_weights.append(ctrl_weight)

            current_pair += 1

    if not matched_treated_idx:
        return pd.DataFrame()

    matched_treated_df = sub.loc[matched_treated_idx].copy()
    matched_treated_df["pair_id"] = treated_pair_ids
    matched_treated_df["matching_weight"] = treated_weights

    matched_control_df = sub.loc[matched_control_idx].copy()
    matched_control_df["pair_id"] = control_pair_ids
    matched_control_df["matching_weight"] = control_weights

    matched_df = pd.concat([matched_treated_df, matched_control_df])
    if "_logit_ps" in matched_df.columns:
        matched_df = matched_df.drop(columns=["_logit_ps"])

    return matched_df


def propensity_score_match(
    df: pd.DataFrame,
    treatment_col: str,
    covariates: list[str],
    caliper: float = 0.2,
    ratio: int = 1,
    ps_method: str = "logistic",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Convenience wrapper: calculate propensity scores and perform matching.

    Parameters:
        df: Input DataFrame.
        treatment_col: Binary treatment indicator column name.
        covariates: List of confounder column names.
        caliper: Caliper width in SDs of the logit propensity score (default 0.2; Austin 2011).
        ratio: Matching ratio (default 1).
        ps_method: Propensity model, 'logistic' (default) or 'firth'; reported in info['ps_method'].

    Returns:
        tuple of (matched_df, info_dict).
    """
    df_ps = df.copy()
    ps_method = normalize_ps_method(ps_method)
    ps_series = calculate_propensity_score(
        df, treatment=treatment_col, covariates=covariates, method=ps_method
    )
    df_ps["ps"] = ps_series
    matched_df = perform_matching(
        df_ps,
        treatment=treatment_col,
        ps_col="ps",
        caliper=caliper,
        ratio=ratio,
    )
    if not matched_df.empty:
        matched_df["matched_group"] = matched_df["pair_id"]
        n_treated = int((matched_df[treatment_col] == 1).sum())
        n_ctrl = int((matched_df[treatment_col] == 0).sum())
    else:
        n_treated = 0
        n_ctrl = 0

    info = {
        "n_initial": len(df),
        "n_matched": len(matched_df),
        "n_treated_matched": n_treated,
        "n_control_matched": n_ctrl,
        "caliper": caliper,
        "ratio": ratio,
        "ps_method": ps_method,
    }
    return matched_df, info


def calculate_ipw(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    ps_col: str,
    estimand: Literal["ATE", "ATT"] = "ATE",
) -> dict[str, Any]:
    """
    Estimate treatment effect via Inverse Probability Weighting (IPW) with Weighted Least Squares.
    """
    cols = [treatment, outcome, ps_col]
    clean_df = df[cols].dropna()

    if clean_df.empty:
        raise ValueError("Insufficient data for IPW calculation.")

    T = clean_df[treatment].to_numpy(dtype=float)
    Y = clean_df[outcome].to_numpy(dtype=float)
    ps = np.clip(clean_df[ps_col].to_numpy(dtype=float), 0.01, 0.99)

    if estimand == "ATE":
        weights = np.where(T == 1, 1.0 / ps, 1.0 / (1.0 - ps))
    else:  # ATT
        weights = np.where(T == 1, 1.0, ps / (1.0 - ps))

    X = sm.add_constant(T)
    wls_model = sm.WLS(Y, X, weights=weights).fit(cov_type="HC1")

    effect = float(wls_model.params[1])
    se = float(wls_model.bse[1])
    ci = wls_model.conf_int()[1]
    p_val = float(wls_model.pvalues[1])

    return {
        "estimand": estimand,
        "effect": effect,
        "se": se,
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
        "p_value": p_val,
        "n_samples": len(clean_df),
    }

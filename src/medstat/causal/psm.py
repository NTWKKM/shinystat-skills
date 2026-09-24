"""
Propensity Score Matching (PSM) and Inverse Probability Weighting (IPW).

Provides logistic propensity score estimation, greedy nearest-neighbor matching
with caliper constraints, and inverse probability of treatment weighting.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial.distance import cdist

from medstat.logging import get_logger

logger = get_logger(__name__)


def calculate_propensity_score(
    df: pd.DataFrame,
    treatment: str,
    covariates: list[str],
) -> pd.Series:
    """
    Estimate propensity scores using logistic regression.

    Parameters:
        df: Input DataFrame.
        treatment: Name of binary treatment/exposure column (0/1).
        covariates: List of baseline confounder column names.

    Returns:
        pd.Series containing estimated propensity scores aligned with df.index.
    """
    cols = [treatment] + covariates
    clean_sub = df[cols].dropna()

    if clean_sub.empty:
        raise ValueError(
            "No complete observations available to estimate propensity scores."
        )

    y = clean_sub[treatment].astype(int)
    X = clean_sub[covariates].astype(float)
    X = sm.add_constant(X)

    try:
        model = sm.Logit(y, X).fit(disp=False)
        ps_clean = model.predict(X)
    except Exception:
        try:
            model = sm.Logit(y, X).fit_regularized(disp=False)
            ps_clean = model.predict(X)
        except Exception:
            from medstat.models.firth import fit_firth_logistic

            firth_fit = fit_firth_logistic(y, clean_sub[covariates], fit_intercept=True)
            fl_model = firth_fit["model"]
            ps_clean = fl_model.predict_proba(X)[:, 1]

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
    Perform greedy 1:1 or 1:k nearest-neighbor propensity score matching within caliper.

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

    for i, t_idx in enumerate(treated.index):
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
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Convenience wrapper: calculate propensity scores and perform matching.

    Parameters:
        df: Input DataFrame.
        treatment_col: Binary treatment indicator column name.
        covariates: List of confounder column names.
        caliper: Caliper width on PS scale (default 0.2).
        ratio: Matching ratio (default 1).

    Returns:
        tuple of (matched_df, info_dict).
    """
    df_ps = df.copy()
    ps_series = calculate_propensity_score(
        df, treatment=treatment_col, covariates=covariates
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

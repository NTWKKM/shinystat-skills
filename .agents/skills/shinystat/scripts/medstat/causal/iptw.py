"""
Inverse Probability of Treatment Weighting (IPTW) & Causal Inference.

Provides robust propensity score weighting (ATE, ATT, ATC, ATO), weight diagnostics
(Kish's Effective Sample Size, extreme weight detection, stabilized weights),
and downstream robust outcome modeling (GLM with Huber-White HC0/HC1, Cox with robust variance).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxPHFitter

from medstat.causal.psm import calculate_propensity_score
from medstat.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IPTWResult:
    """Container for IPTW results supporting both attribute and tuple unpacking."""

    df: pd.DataFrame
    diagnostics: dict[str, Any]

    @property
    def weights(self) -> pd.Series:
        return self.df["iptw_weight"]

    @property
    def propensity_scores(self) -> pd.Series:
        return self.df["propensity_score"]

    @property
    def effective_sample_size(self) -> dict[str, float]:
        return {
            "treated": self.diagnostics["treated"]["n_eff"],
            "control": self.diagnostics["control"]["n_eff"],
            "total": self.diagnostics["overall"]["n_eff"],
        }

    def __iter__(self):
        return iter((self.df, self.diagnostics))

    def __getitem__(self, item: int | str) -> Any:
        if item == 0 or item == "df":
            return self.df
        elif item == 1 or item == "diagnostics":
            return self.diagnostics
        raise KeyError(f"Invalid key or index {item} for IPTWResult")


def compute_iptw_weights(
    df: pd.DataFrame,
    treatment_col: str | None = None,
    covariates: list[str] | None = None,
    estimand: Literal["ATE", "ATT", "ATC", "ATO"] = "ATE",
    stabilized: bool = True,
    truncate_quantiles: tuple[float, float] | None = (0.01, 0.99),
    ps_clip: tuple[float, float] = (1e-4, 1.0 - 1e-4),
    treatment: str | None = None,
    truncation: tuple[float, float] | None = None,
) -> IPTWResult:
    """
    Compute Inverse Probability of Treatment Weights (IPTW) with comprehensive diagnostics.

    Parameters:
        df: Input DataFrame containing treatment and covariates.
        treatment_col: Name of binary treatment column (0/1). Alias: treatment.
        covariates: List of baseline confounder column names.
        estimand: Target causal estimand:
            - 'ATE': Average Treatment Effect (whole population)
            - 'ATT': Average Treatment Effect on the Treated
            - 'ATC': Average Treatment Effect on the Controls
            - 'ATO': Overlap Weights (Li et al., 2018; bounded in [0, 1])
        stabilized: If True, computes stabilized weights (ignored for ATO).
        truncate_quantiles: Quantiles for weight truncation (e.g. (0.01, 0.99)), or None. Alias: truncation.
        ps_clip: Minimum and maximum bounds to clip propensity scores before division.

    Returns:
        IPTWResult with attributes .df, .weights, .propensity_scores, .diagnostics,
        and iterable unpacking as (df_weighted, diagnostics).
    """
    t_col = treatment_col if treatment_col is not None else treatment
    if t_col is None:
        raise ValueError("Either 'treatment_col' or 'treatment' must be specified.")
    if covariates is None:
        raise ValueError("covariates list must be specified.")

    trunc = truncation if truncation is not None else truncate_quantiles

    clean_cols = [t_col] + covariates
    sub_df = df.dropna(subset=clean_cols).copy()
    if sub_df.empty:
        raise ValueError("No complete observations available to compute IPTW weights.")

    t_vals = sub_df[t_col].unique()
    if not set(t_vals).issubset({0, 1}):
        raise ValueError(
            f"Treatment column '{t_col}' must be binary 0/1. Observed: {t_vals}"
        )

    # 1. Estimate Propensity Scores
    ps_series = calculate_propensity_score(sub_df, t_col, covariates)
    sub_df["propensity_score"] = ps_series

    T = sub_df[t_col].to_numpy(dtype=float)
    ps = np.clip(
        sub_df["propensity_score"].to_numpy(dtype=float), ps_clip[0], ps_clip[1]
    )

    # Positivity violation check: observations near 0 or 1
    extreme_ps_count = int(np.sum((ps < 0.05) | (ps > 0.95)))
    extreme_ps_pct = float(extreme_ps_count / len(ps) * 100.0)

    p_t = float(np.mean(T))

    # 2. Compute Unstabilized / Target Weights
    estimand_upper = estimand.upper()
    if estimand_upper == "ATE":
        if stabilized:
            raw_weights = np.where(T == 1, p_t / ps, (1.0 - p_t) / (1.0 - ps))
        else:
            raw_weights = np.where(T == 1, 1.0 / ps, 1.0 / (1.0 - ps))
    elif estimand_upper == "ATT":
        raw_weights = np.where(T == 1, 1.0, ps / (1.0 - ps))
    elif estimand_upper == "ATC":
        raw_weights = np.where(T == 1, (1.0 - ps) / ps, 1.0)
    elif estimand_upper == "ATO":
        # Overlap Weights: w_i = e_i(1 - e_i) / e_i for treated, etc. -> T(1-e) + (1-T)e
        raw_weights = np.where(T == 1, 1.0 - ps, ps)
        stabilized = False  # Overlap weights are naturally bounded in (0, 1)
    else:
        raise ValueError(
            f"Unsupported estimand '{estimand}'. Choose from: 'ATE', 'ATT', 'ATC', 'ATO'."
        )

    sub_df["iptw_weight_raw"] = raw_weights

    # 3. Truncation / Trimming
    if trunc is not None and estimand_upper != "ATO":
        low_q, high_q = trunc
        w_low = float(np.quantile(raw_weights, low_q))
        w_high = float(np.quantile(raw_weights, high_q))
        final_weights = np.clip(raw_weights, w_low, w_high)
    else:
        final_weights = raw_weights.copy()
        w_low, w_high = float(np.min(raw_weights)), float(np.max(raw_weights))

    sub_df["iptw_weight"] = final_weights

    # 4. Diagnostics & Kish's Effective Sample Size (Neff)
    def _compute_group_diagnostics(w: np.ndarray, n_orig: int) -> dict[str, Any]:
        if len(w) == 0 or np.sum(w**2) == 0:
            return {
                "n_original": n_orig,
                "n_eff": 0.0,
                "mean": np.nan,
                "sd": np.nan,
                "min": np.nan,
                "max": np.nan,
                "median": np.nan,
                "iqr": np.nan,
                "n_extreme_gt_10": 0,
            }
        sum_w = float(np.sum(w))
        sum_w2 = float(np.sum(w**2))
        n_eff = float((sum_w**2) / sum_w2)
        q25, q50, q75 = np.percentile(w, [25, 50, 75])
        return {
            "n_original": n_orig,
            "n_eff": round(n_eff, 2),
            "mean": round(float(np.mean(w)), 4),
            "sd": round(float(np.std(w, ddof=1)), 4) if len(w) > 1 else 0.0,
            "min": round(float(np.min(w)), 4),
            "max": round(float(np.max(w)), 4),
            "median": round(float(q50), 4),
            "iqr": round(float(q75 - q25), 4),
            "n_extreme_gt_10": int(np.sum(w > 10.0)),
        }

    treated_mask = T == 1
    control_mask = T == 0

    overall_diag = _compute_group_diagnostics(final_weights, len(final_weights))
    treated_diag = _compute_group_diagnostics(
        final_weights[treated_mask], int(np.sum(treated_mask))
    )
    control_diag = _compute_group_diagnostics(
        final_weights[control_mask], int(np.sum(control_mask))
    )

    clinical_warnings = []
    if extreme_ps_pct > 5.0:
        clinical_warnings.append(
            f"Positivity alert: {extreme_ps_pct:.1f}% of observations have extreme propensity scores "
            f"(<0.05 or >0.95). Consider Overlap Weights ('ATO') or covariate trimming."
        )

    if overall_diag["n_extreme_gt_10"] > 0:
        clinical_warnings.append(
            f"Extreme weights alert: {overall_diag['n_extreme_gt_10']} observations have weight > 10.0. "
            "Examine truncation or Overlap Weights ('ATO') to control variance inflation."
        )

    if overall_diag["n_eff"] < 0.5 * len(final_weights):
        clinical_warnings.append(
            f"High weight variance: Kish's effective sample size (Neff={overall_diag['n_eff']:.1f}) is less than "
            f"50% of the original sample size (N={len(final_weights)})."
        )

    if estimand_upper == "ATE" and stabilized and trunc is None:
        mean_sw = float(np.mean(raw_weights))
        if abs(mean_sw - 1.0) > 0.05:
            clinical_warnings.append(
                f"Stabilized weights mean check: mean(sw)={mean_sw:.4f} deviates from expected 1.00 by >0.05. "
                "Verify propensity score model calibration."
            )

    diagnostics = {
        "estimand": estimand_upper,
        "stabilized": stabilized,
        "truncate_quantiles": trunc,
        "truncation_bounds": (round(w_low, 4), round(w_high, 4)),
        "extreme_ps_count": extreme_ps_count,
        "extreme_ps_percent": round(extreme_ps_pct, 2),
        "overall": overall_diag,
        "treated": treated_diag,
        "control": control_diag,
        "clinical_warnings": clinical_warnings,
    }

    # Emit logger warnings
    for warn_msg in clinical_warnings:
        logger.warning(warn_msg)

    # Re-align with original DataFrame index
    out_df = df.copy()
    out_df["propensity_score"] = sub_df["propensity_score"]
    out_df["iptw_weight_raw"] = sub_df["iptw_weight_raw"]
    out_df["iptw_weight"] = sub_df["iptw_weight"]

    return IPTWResult(df=out_df, diagnostics=diagnostics)


def fit_iptw_outcome_model(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str | None = None,
    weights_col: str = "iptw_weight",
    covariates: list[str] | None = None,
    outcome_type: Literal["continuous", "binary", "survival"] = "continuous",
    duration_col: str | None = None,
    robust_cov_type: str = "HC1",
    treatment: str | None = None,
    model_type: str | None = None,
    event_col: str | None = None,
) -> dict[str, Any]:
    """
    Fit outcome model on IPTW-weighted cohort using robust sandwich covariance estimation.

    Parameters:
        df: DataFrame containing outcome, treatment, and weights.
        outcome_col: Name of outcome column (or duration/event column for survival).
        treatment_col: Name of binary treatment column (0/1). Alias: treatment.
        weights_col: Column containing IPTW weights.
        covariates: Optional additional covariates to fit doubly robust models.
        outcome_type: 'continuous' (linear regression via WLS), 'binary' (logistic GLM),
                      or 'survival' (Cox proportional hazards). Alias: model_type.
        duration_col: Time-to-event column for survival.
        robust_cov_type: Robust covariance type for statsmodels ('HC0', 'HC1').
        event_col: Optional event indicator column when outcome_col is duration.

    Returns:
        dict containing treatment effect estimate, robust SE, 95% CIs, p-value, and model summary.
    """
    t_col = treatment_col if treatment_col is not None else treatment
    if t_col is None:
        raise ValueError("Either 'treatment_col' or 'treatment' must be specified.")

    m_type = (model_type or outcome_type).lower()
    if "binary" in m_type:
        m_type = "binary"
    elif "surv" in m_type:
        m_type = "survival"
    else:
        m_type = "continuous"

    dur_col = duration_col
    evt_col = outcome_col
    if m_type == "survival":
        if event_col is not None:
            evt_col = event_col
            dur_col = duration_col or outcome_col
        elif dur_col is None:
            raise ValueError(
                "duration_col or event_col must be specified when outcome_type='survival'."
            )

    required_cols = [t_col, weights_col]
    if m_type == "survival":
        required_cols.extend([dur_col, evt_col])
    else:
        required_cols.append(outcome_col)

    if covariates:
        required_cols.extend(covariates)

    clean_df = df[list(dict.fromkeys(required_cols))].dropna().copy()
    if clean_df.empty:
        raise ValueError("Insufficient data to fit IPTW outcome model.")

    weights = clean_df[weights_col].to_numpy(dtype=float)

    if m_type == "continuous":
        Y = clean_df[outcome_col].to_numpy(dtype=float)
        X_vars = [clean_df[t_col]]
        if covariates:
            for cov in covariates:
                X_vars.append(clean_df[cov])
        X = pd.concat(X_vars, axis=1)
        X = sm.add_constant(X)

        wls_model = sm.WLS(Y, X, weights=weights).fit(cov_type=robust_cov_type)
        t_idx = 1
        effect = float(wls_model.params.iloc[t_idx])
        se = float(wls_model.bse.iloc[t_idx])
        ci = wls_model.conf_int().iloc[t_idx].tolist()
        p_val = float(wls_model.pvalues.iloc[t_idx])

        return {
            "model_type": "continuous",
            "outcome": outcome_col,
            "treatment": t_col,
            "coef": effect,
            "coefficient": effect,
            "effect": effect,
            "se": se,
            "robust_se": se,
            "ci_lower": float(ci[0]),
            "ci_upper": float(ci[1]),
            "p_value": p_val,
            "robust_cov_type": robust_cov_type,
            "n_obs": len(clean_df),
            "model": wls_model,
        }

    elif m_type == "binary":
        Y = clean_df[outcome_col].to_numpy(dtype=float)
        X_vars = [clean_df[t_col]]
        if covariates:
            for cov in covariates:
                X_vars.append(clean_df[cov])
        X = pd.concat(X_vars, axis=1)
        X = sm.add_constant(X)

        glm_model = sm.GLM(
            Y, X, family=sm.families.Binomial(), freq_weights=weights
        ).fit(cov_type=robust_cov_type)

        t_idx = 1
        coef = float(glm_model.params.iloc[t_idx])
        se = float(glm_model.bse.iloc[t_idx])
        ci = glm_model.conf_int().iloc[t_idx].tolist()
        p_val = float(glm_model.pvalues.iloc[t_idx])

        odds_ratio = float(np.exp(coef))
        or_ci_lower = float(np.exp(ci[0]))
        or_ci_upper = float(np.exp(ci[1]))

        return {
            "model_type": "binary",
            "outcome": outcome_col,
            "treatment": t_col,
            "coef": coef,
            "coefficient": coef,
            "odds_ratio": odds_ratio,
            "or_ci_lower": or_ci_lower,
            "or_ci_upper": or_ci_upper,
            "se": se,
            "robust_se": se,
            "ci_lower": float(ci[0]),
            "ci_upper": float(ci[1]),
            "p_value": p_val,
            "robust_cov_type": robust_cov_type,
            "n_obs": len(clean_df),
            "model": glm_model,
        }

    elif m_type == "survival":
        cox_cols = [dur_col, evt_col, t_col, weights_col]
        if covariates:
            cox_cols.extend(covariates)
        cox_data = clean_df[list(dict.fromkeys(cox_cols))].copy()

        cph = CoxPHFitter()
        # weights_col with robust=True ensures Huber-White sandwich variance
        cph.fit(
            cox_data,
            duration_col=dur_col,
            event_col=evt_col,
            weights_col=weights_col,
            robust=True,
        )

        hr = float(cph.hazard_ratios_[t_col])
        coef = float(cph.params_[t_col])
        se = float(cph.standard_errors_[t_col])
        ci_row = cph.confidence_intervals_.loc[t_col]
        ci_lower = float(np.exp(ci_row.iloc[0]))
        ci_upper = float(np.exp(ci_row.iloc[1]))
        p_val = float(cph.summary.loc[t_col, "p"])

        return {
            "model_type": "survival",
            "outcome": evt_col,
            "duration": dur_col,
            "treatment": t_col,
            "hazard_ratio": hr,
            "hr_ci_lower": ci_lower,
            "hr_ci_upper": ci_upper,
            "coef": coef,
            "coefficient": coef,
            "se": se,
            "robust_se": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_val,
            "n_obs": len(clean_df),
            "model": cph,
        }

    else:
        raise ValueError(
            f"Unsupported outcome_type '{outcome_type}'. Choose from: 'continuous', 'binary', 'survival'."
        )

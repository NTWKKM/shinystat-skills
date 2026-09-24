"""
Clinical Meta-Analysis Models & Heterogeneity Statistics.

Implements Inverse-Variance Fixed-Effect, DerSimonian-Laird Random-Effects,
Cochran's Q, Higgins' I^2, Tau^2, and 95% prediction intervals (PRISMA 2020 compliant).
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy.stats as stats

from medstat.logging import get_logger

logger = get_logger(__name__)


def compute_binary_effect_sizes(
    df: pd.DataFrame,
    events_t_col: str,
    n_t_col: str,
    events_c_col: str,
    n_c_col: str,
    study_col: str,
    effect_measure: Literal["OR", "RR", "RD"] = "OR",
) -> pd.DataFrame:
    """
    Compute log effect sizes and standard errors for 2x2 binary endpoints.
    """
    cols = [study_col, events_t_col, n_t_col, events_c_col, n_c_col]
    clean = df[cols].dropna().copy()

    records = []
    for _, row in clean.iterrows():
        study = str(row[study_col])
        a = float(row[events_t_col])
        n1 = float(row[n_t_col])
        c = float(row[events_c_col])
        n0 = float(row[n_c_col])

        b = n1 - a
        d = n0 - c

        # Continuity correction of 0.5 if any cell is 0
        if a == 0 or b == 0 or c == 0 or d == 0:
            a += 0.5
            b += 0.5
            c += 0.5
            d += 0.5
            n1 += 1.0
            n0 += 1.0

        if effect_measure == "OR":
            or_val = (a * d) / (b * c)
            log_or = math.log(or_val)
            se = math.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
            eff_disp = or_val
            log_eff = log_or
            is_ratio = True
        elif effect_measure == "RR":
            rr_val = (a / n1) / (c / n0)
            log_rr = math.log(rr_val)
            se = math.sqrt(1.0 / a - 1.0 / n1 + 1.0 / c - 1.0 / n0)
            eff_disp = rr_val
            log_eff = log_rr
            is_ratio = True
        else:  # "RD"
            rd_val = (a / n1) - (c / n0)
            se = math.sqrt((a * b) / (n1**3) + (c * d) / (n0**3))
            eff_disp = rd_val
            log_eff = rd_val
            is_ratio = False

        ci_low = log_eff - 1.96 * se
        ci_high = log_eff + 1.96 * se

        records.append(
            {
                "study": study,
                "effect_size": eff_disp,
                "log_effect": log_eff,
                "se": se,
                "ci_lower": math.exp(ci_low) if is_ratio else ci_low,
                "ci_upper": math.exp(ci_high) if is_ratio else ci_high,
                "is_ratio": is_ratio,
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
    """
    Compute Mean Difference (MD) or Standardized Mean Difference (SMD / Hedges' g).
    """
    cols = [study_col, mean_t_col, sd_t_col, n_t_col, mean_c_col, sd_c_col, n_c_col]
    clean = df[cols].dropna().copy()

    records = []
    for _, row in clean.iterrows():
        study = str(row[study_col])
        m1 = float(row[mean_t_col])
        s1 = float(row[sd_t_col])
        n1 = float(row[n_t_col])
        m0 = float(row[mean_c_col])
        s0 = float(row[sd_c_col])
        n0 = float(row[n_c_col])

        if effect_measure == "MD":
            diff = m1 - m0
            se = math.sqrt((s1**2 / n1) + (s0**2 / n0))
            eff_disp = diff
            log_eff = diff
        else:  # SMD (Hedges' g)
            df_deg = n1 + n0 - 2
            s_pooled = (
                math.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / df_deg)
                if df_deg > 0
                else 1.0
            )
            d = (m1 - m0) / s_pooled if s_pooled > 0 else 0.0
            # Small sample correction J
            j = 1.0 - (3.0 / (4.0 * (n1 + n0) - 9.0)) if (n1 + n0) > 3 else 1.0
            g = d * j
            se = math.sqrt((n1 + n0) / (n1 * n0) + (g**2) / (2.0 * (n1 + n0)))
            eff_disp = g
            log_eff = g

        ci_low = log_eff - 1.96 * se
        ci_high = log_eff + 1.96 * se

        records.append(
            {
                "study": study,
                "effect_size": eff_disp,
                "log_effect": log_eff,
                "se": se,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "is_ratio": False,
            }
        )

    return pd.DataFrame(records)


def run_meta_analysis(
    df: pd.DataFrame,
    method_re: str = "DL",
    use_hksj: bool = False,
    alpha: float = 0.05,
    effect_col: str | None = None,
    se_col: str | None = None,
    study_col: str | None = None,
    model: str | None = None,
    method: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Perform fixed-effect and random-effects meta-analysis with heterogeneity diagnostics.

    Parameters:
        df: DataFrame containing columns 'study', 'log_effect', 'se', and optionally 'is_ratio'.
        method_re: Random-effects tau^2 estimator ('DL' = DerSimonian-Laird).
        use_hksj: If True, uses Hartung-Knapp-Sidik-Jonkman adjustment.
        alpha: Significance level (default 0.05 for 95% CIs).
    """
    df_work = df.copy()
    if effect_col and effect_col in df_work.columns:
        df_work["log_effect"] = df_work[effect_col]
    elif "log_effect" not in df_work.columns:
        for c in ["effect", "effect_size", "es"]:
            if c in df_work.columns:
                df_work["log_effect"] = df_work[c]
                break

    if se_col and se_col in df_work.columns:
        df_work["se"] = df_work[se_col]

    if study_col and study_col in df_work.columns:
        df_work["study"] = df_work[study_col]
    elif "study" not in df_work.columns:
        df_work["study"] = [f"Study {i + 1}" for i in range(len(df_work))]

    k = len(df_work)
    if k < 2:
        raise ValueError(f"Meta-analysis requires at least 2 studies (got {k}).")

    theta = df_work["log_effect"].to_numpy(dtype=float)
    se_arr = np.asarray(df_work["se"], dtype=float)
    if np.any(se_arr <= 0) or np.any(np.isnan(se_arr)):
        raise ValueError("Standard error must be strictly positive.")
    se = se_arr
    is_ratio = (
        bool(df_work["is_ratio"].iloc[0]) if "is_ratio" in df_work.columns else False
    )

    # 1. Fixed-Effect Model (Inverse Variance)
    w_fe = 1.0 / (se**2)
    sum_w_fe = np.sum(w_fe)
    theta_fe = float(np.sum(w_fe * theta) / sum_w_fe)
    se_fe = math.sqrt(1.0 / sum_w_fe)

    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    ci_fe_low = theta_fe - z_crit * se_fe
    ci_fe_high = theta_fe + z_crit * se_fe
    z_fe = theta_fe / se_fe if se_fe > 0 else 0.0
    p_fe = float(2.0 * (1.0 - stats.norm.cdf(abs(z_fe))))

    # 2. Heterogeneity (Cochran's Q, I^2, Tau^2)
    q_stat = float(np.sum(w_fe * ((theta - theta_fe) ** 2)))
    df_q = k - 1
    p_q = float(stats.chi2.sf(q_stat, df_q))

    # DerSimonian-Laird tau^2
    c_const = sum_w_fe - (np.sum(w_fe**2) / sum_w_fe)
    tau2 = max(0.0, (q_stat - df_q) / c_const) if c_const > 0 else 0.0
    tau = math.sqrt(tau2)

    # Higgins I^2 (%)
    i2 = max(0.0, ((q_stat - df_q) / q_stat) * 100.0) if q_stat > 0 else 0.0

    # 3. Random-Effects Model
    w_re = 1.0 / (se**2 + tau2)
    sum_w_re = np.sum(w_re)
    theta_re = float(np.sum(w_re * theta) / sum_w_re)

    if use_hksj and k >= 3:
        q_hksj = float((1.0 / (k - 1)) * np.sum(w_re * ((theta - theta_re) ** 2)))
        hksj_scale = max(1.0, q_hksj)
        se_re = math.sqrt(hksj_scale / sum_w_re)
        t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=k - 1)
        ci_re_low = theta_re - t_crit * se_re
        ci_re_high = theta_re + t_crit * se_re
        t_val = theta_re / se_re if se_re > 0 else 0.0
        p_re = float(2.0 * (1.0 - stats.t.cdf(abs(t_val), df=k - 1)))
    else:
        se_re = math.sqrt(1.0 / sum_w_re)
        ci_re_low = theta_re - z_crit * se_re
        ci_re_high = theta_re + z_crit * se_re
        z_re = theta_re / se_re if se_re > 0 else 0.0
        p_re = float(2.0 * (1.0 - stats.norm.cdf(abs(z_re))))

    # 4. 95% Prediction Interval
    pi_low, pi_high = np.nan, np.nan
    if k >= 3:
        t_crit_pi = stats.t.ppf(1.0 - alpha / 2.0, df=max(1, k - 2))
        se_pi = math.sqrt(tau2 + se_re**2)
        pi_low = theta_re - t_crit_pi * se_pi
        pi_high = theta_re + t_crit_pi * se_pi

    def disp(val: float) -> float:
        return float(math.exp(val)) if is_ratio else float(val)

    w_fe_pct = (w_fe / sum_w_fe) * 100.0
    w_re_pct = (w_re / sum_w_re) * 100.0

    studies_with_weights = df.copy()
    studies_with_weights["weight_fe_pct"] = w_fe_pct
    studies_with_weights["weight_re_pct"] = w_re_pct
    studies_with_weights["effect_size"] = np.exp(theta) if is_ratio else theta
    studies_with_weights["ci_lower"] = (
        np.exp(theta - 1.96 * se_arr) if is_ratio else (theta - 1.96 * se_arr)
    )
    studies_with_weights["ci_upper"] = (
        np.exp(theta + 1.96 * se_arr) if is_ratio else (theta + 1.96 * se_arr)
    )

    return {
        "k": k,
        "is_ratio": is_ratio,
        "studies": studies_with_weights,
        "fixed_effect": {
            "log_effect": theta_fe,
            "effect_disp": disp(theta_fe),
            "se": se_fe,
            "ci_lower": disp(ci_fe_low),
            "ci_upper": disp(ci_fe_high),
            "p_value": p_fe,
        },
        "random_effects": {
            "log_effect": theta_re,
            "effect_disp": disp(theta_re),
            "se": se_re,
            "ci_lower": disp(ci_re_low),
            "ci_upper": disp(ci_re_high),
            "p_value": p_re,
            "prediction_interval": [disp(pi_low), disp(pi_high)]
            if not np.isnan(pi_low)
            else None,
        },
        "heterogeneity": {
            "Q": q_stat,
            "df": df_q,
            "p_value": p_q,
            "I2": i2,
            "tau2": tau2,
            "tau": tau,
        },
    }

"""
Survival Analysis Module.

Provides Kaplan-Meier estimation, log-rank testing, Cox Proportional Hazards
regression, and Schoenfeld residual proportional hazard assumption tests via lifelines.
Pure headless with zero UI dependencies.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import (
    logrank_test,
    multivariate_logrank_test,
    proportional_hazard_test,
)

from medstat.logging import get_logger

logger = get_logger(__name__)


def fit_kaplan_meier(
    durations: pd.Series | np.ndarray,
    event_observed: pd.Series | np.ndarray,
    label: str = "KM_estimate",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Fit Kaplan-Meier survival curve.

    Returns:
        dict with kmf (fitted KaplanMeierFitter), median_survival_time,
        survival_table (survival probability with 95% CIs).
    """
    kmf = KaplanMeierFitter(alpha=alpha)
    kmf.fit(durations=durations, event_observed=event_observed, label=label)

    return {
        "kmf": kmf,
        "median_survival_time": float(kmf.median_survival_time_),
        "survival_table": kmf.survival_function_,
        "confidence_interval": kmf.confidence_interval_,
        "timeline": kmf.timeline.tolist(),
        "survival_probabilities": kmf.survival_function_[label].tolist(),
    }


def compare_survival_curves(
    durations: pd.Series | np.ndarray,
    groups: pd.Series | np.ndarray,
    event_observed: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """
    Perform Log-Rank test to compare survival curves across groups.
    """
    df = pd.DataFrame(
        {"time": durations, "group": groups, "event": event_observed}
    ).dropna()
    unique_groups = df["group"].unique()

    if len(unique_groups) < 2:
        raise ValueError("Comparison requires at least 2 distinct groups.")

    if len(unique_groups) == 2:
        g1 = df[df["group"] == unique_groups[0]]
        g2 = df[df["group"] == unique_groups[1]]
        res = logrank_test(
            g1["time"],
            g2["time"],
            event_observed_A=g1["event"],
            event_observed_B=g2["event"],
        )
    else:
        res = multivariate_logrank_test(df["time"], df["group"], df["event"])

    return {
        "test": "Log-rank test",
        "test_statistic": float(res.test_statistic),
        "p_value": float(res.p_value),
        "degrees_of_freedom": int(
            getattr(res, "degrees_of_freedom", len(unique_groups) - 1)
        ),
        "groups": [str(g) for g in unique_groups],
    }


def fit_cox_ph(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    covariates: list[str],
    penalizer: float = 0.0,
    check_assumptions: bool = True,
) -> dict[str, Any]:
    """
    Fit Cox Proportional Hazards regression model using lifelines.

    Parameters:
        df: Input DataFrame.
        duration_col: Column name representing follow-up duration.
        event_col: Column name representing binary event indicator (1=event, 0=censor).
        covariates: List of covariate column names.
        penalizer: L2 penalty parameter (default 0.0).
        check_assumptions: If True, runs Schoenfeld residual proportional hazard test.

    Returns:
        dict with cph model, summary_df (HR, 95% CIs, p-values), concordance_index,
        and assumption_test_summary.
    """
    cols = [duration_col, event_col] + covariates
    clean_df = df[cols].dropna()

    if len(clean_df) < 5:
        raise ValueError("Cox PH requires at least 5 observations.")

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(clean_df, duration_col=duration_col, event_col=event_col)

    summary = cph.summary.copy()
    # Format standard hazard ratio summary
    res_df = pd.DataFrame(
        {
            "coef": summary["coef"],
            "hazard_ratio": summary["exp(coef)"],
            "hr_ci_lower": summary["exp(coef) lower 95%"],
            "hr_ci_upper": summary["exp(coef) upper 95%"],
            "se": summary["se(coef)"],
            "z": summary["z"],
            "p_value": summary["p"],
        },
        index=summary.index,
    )

    ph_results = None
    if check_assumptions:
        try:
            ph_test = proportional_hazard_test(cph, clean_df, time_transform="rank")
            ph_results = ph_test.summary.to_dict()
        except Exception as e:
            logger.warning(f"Schoenfeld PH assumption test failed: {e}")

    return {
        "cph": cph,
        "model": cph,
        "summary_df": res_df,
        "concordance_index": float(cph.concordance_index_),
        "log_likelihood": float(cph.log_likelihood_),
        "aic": float(cph.AIC_partial_),
        "ph_test": ph_results,
    }


def check_proportional_hazards(
    cph_or_res: Any,
    df: pd.DataFrame | None = None,
    time_transform: str = "rank",
) -> dict[str, Any]:
    """
    Test proportional hazards assumption via Schoenfeld residuals.

    Parameters:
        cph_or_res: Either a CoxPHFitter instance or the dict returned by fit_cox_ph.
        df: Training DataFrame if not accessible via model._train_data.
        time_transform: Time transformation for Schoenfeld residuals ('rank', 'km', etc.).

    Returns:
        dict containing Schoenfeld test summary.
    """
    if isinstance(cph_or_res, dict):
        if "ph_test" in cph_or_res and cph_or_res["ph_test"] is not None:
            return cph_or_res["ph_test"]
        cph = cph_or_res.get("cph") or cph_or_res.get("model")
    else:
        cph = cph_or_res

    if cph is None:
        raise ValueError("Valid CoxPHFitter instance or result dict required.")

    data = df if df is not None else getattr(cph, "_train_data", None)
    if data is None:
        raise ValueError("Dataset required to run Schoenfeld test.")

    try:
        ph_test = proportional_hazard_test(cph, data, time_transform=time_transform)
        return ph_test.summary.to_dict()
    except Exception as e:
        logger.warning(f"Schoenfeld PH assumption test failed: {e}")
        return {"error": str(e)}

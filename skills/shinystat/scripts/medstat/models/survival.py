"""
Survival Analysis Module.

Provides Kaplan-Meier estimation, log-rank testing, Cox Proportional Hazards
regression, and Schoenfeld residual proportional hazard assumption tests via lifelines.
Pure headless with zero UI dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lifelines import AalenJohansenFitter, CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import (
    logrank_test,
    multivariate_logrank_test,
    proportional_hazard_test,
)
from scipy import stats

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

    if penalizer < 0:
        raise ValueError("penalizer must be >= 0")

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


@dataclass
class CompetingRisksCIFResult:
    """Container for Competing Risks Cumulative Incidence Function (CIF) results."""

    ajf_primary: AalenJohansenFitter
    ajf_competing: AalenJohansenFitter
    km_naive: KaplanMeierFitter
    cif_table: pd.DataFrame
    horizon_summary: pd.DataFrame | None
    event_of_interest: int
    competing_event: int
    probability_conservation_max_error: float
    clinical_warnings: list[str]

    def __getitem__(self, item: str) -> Any:
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(f"Invalid key '{item}' for CompetingRisksCIFResult")


def fit_competing_risks_cif(
    durations: pd.Series | np.ndarray,
    event_observed: pd.Series | np.ndarray,
    event_of_interest: int = 1,
    competing_event: int = 2,
    horizons: list[float] | None = None,
    alpha: float = 0.05,
) -> CompetingRisksCIFResult:
    """
    Fit non-parametric Cumulative Incidence Functions (CIF) using Aalen-Johansen estimator
    under competing risks, comparing against Naive 1 - Kaplan-Meier overestimation bias.

    Parameters:
        durations: Follow-up duration times.
        event_observed: Categorical event indicator (0 = censored, 1 = event of interest,
                        2 = competing risk).
        event_of_interest: Integer code identifying the primary event (default 1).
        competing_event: Integer code identifying the competing risk (default 2).
        horizons: Optional list of specific follow-up time horizons (e.g. [12.0, 36.0, 60.0]).
        alpha: Significance level for 100*(1-alpha)% confidence intervals (default 0.05).

    Returns:
        CompetingRisksCIFResult containing fitted models, step CIF table, horizon evaluations,
        probability conservation audit, and clinical alerts.
    """
    if event_of_interest == competing_event:
        raise ValueError(
            f"event_of_interest ({event_of_interest}) and competing_event ({competing_event}) must be distinct."
        )

    dur = np.asarray(durations, dtype=float)
    evt_float = np.asarray(event_observed, dtype=float)

    valid_mask = (
        np.isfinite(dur)
        & (dur >= 0)
        & np.isfinite(evt_float)
        & np.isin(evt_float, [0, event_of_interest, competing_event])
    )
    dur = dur[valid_mask]
    evt = evt_float[valid_mask].astype(int)

    if len(dur) < 5:
        raise ValueError(
            "Competing risks analysis requires at least 5 valid observations."
        )

    n_primary = int(np.sum(evt == event_of_interest))
    n_competing = int(np.sum(evt == competing_event))
    if n_primary == 0:
        raise ValueError(f"No events of interest (code={event_of_interest}) observed.")
    if n_competing == 0:
        logger.warning(
            f"No competing events (code={competing_event}) observed. Aalen-Johansen converges to standard KM."
        )

    # 1. Fit Aalen-Johansen for Primary Event
    ajf_primary = AalenJohansenFitter(calculate_variance=True)
    ajf_primary.fit(dur, evt, event_of_interest=event_of_interest)

    # 2. Fit Aalen-Johansen for Competing Event
    ajf_competing = AalenJohansenFitter(calculate_variance=True)
    ajf_competing.fit(dur, evt, event_of_interest=competing_event)

    # 3. Fit Naive Kaplan-Meier (treating competing risks as uninformative censoring)
    km_naive = KaplanMeierFitter(alpha=alpha)
    km_naive.fit(dur, event_observed=(evt == event_of_interest).astype(int))

    # 4. Fit Overall Event-Free Survival (any failure: 1 or 2)
    km_overall = KaplanMeierFitter(alpha=alpha)
    km_overall.fit(dur, event_observed=(evt != 0).astype(int))

    # Helper for step evaluation using binary search
    def _step_lookup(
        series_or_df: pd.Series | pd.DataFrame, eval_times: np.ndarray
    ) -> np.ndarray:
        s = (
            series_or_df.iloc[:, 0]
            if isinstance(series_or_df, pd.DataFrame)
            else series_or_df
        )
        idx = s.index.searchsorted(eval_times, side="right") - 1
        return np.where(idx >= 0, s.iloc[np.maximum(idx, 0)].to_numpy(dtype=float), 0.0)

    # Timeline of all failure times
    timeline = np.sort(np.unique(np.concatenate([[0.0], dur[evt > 0]])))

    cif1 = _step_lookup(ajf_primary.cumulative_density_, timeline)
    var1 = np.maximum(_step_lookup(ajf_primary.variance_, timeline), 0.0)

    cif2 = _step_lookup(ajf_competing.cumulative_density_, timeline)
    var2 = np.maximum(_step_lookup(ajf_competing.variance_, timeline), 0.0)

    s_naive = _step_lookup(km_naive.survival_function_, timeline)
    naive_cif = np.clip(1.0 - s_naive, 0.0, 1.0)

    s_overall = _step_lookup(km_overall.survival_function_, timeline)

    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    cif1_l = np.clip(cif1 - z * np.sqrt(var1), 0.0, 1.0)
    cif1_u = np.clip(cif1 + z * np.sqrt(var1), 0.0, 1.0)
    cif2_l = np.clip(cif2 - z * np.sqrt(var2), 0.0, 1.0)
    cif2_u = np.clip(cif2 + z * np.sqrt(var2), 0.0, 1.0)

    conservation_sum = cif1 + cif2 + s_overall
    max_error = float(np.max(np.abs(conservation_sum - 1.0)))

    cif_df = pd.DataFrame(
        {
            "time": timeline,
            "cif_primary": cif1,
            "cif_primary_ci_lower": cif1_l,
            "cif_primary_ci_upper": cif1_u,
            "cif_competing": cif2,
            "cif_competing_ci_lower": cif2_l,
            "cif_competing_ci_upper": cif2_u,
            "naive_1_minus_km": naive_cif,
            "bias_overestimation": np.maximum(naive_cif - cif1, 0.0),
            "event_free_survival": s_overall,
            "conservation_sum": conservation_sum,
        }
    )

    # Horizon evaluation table
    h_df = None
    clinical_warnings = []
    if horizons is not None and len(horizons) > 0:
        h_arr = np.asarray(horizons, dtype=float)
        h_cif1 = _step_lookup(ajf_primary.cumulative_density_, h_arr)
        h_var1 = np.maximum(_step_lookup(ajf_primary.variance_, h_arr), 0.0)
        h_cif2 = _step_lookup(ajf_competing.cumulative_density_, h_arr)
        h_var2 = np.maximum(_step_lookup(ajf_competing.variance_, h_arr), 0.0)
        h_s_naive = _step_lookup(km_naive.survival_function_, h_arr)
        h_naive = np.clip(1.0 - h_s_naive, 0.0, 1.0)

        h_cif1_l = np.clip(h_cif1 - z * np.sqrt(h_var1), 0.0, 1.0)
        h_cif1_u = np.clip(h_cif1 + z * np.sqrt(h_var1), 0.0, 1.0)
        h_cif2_l = np.clip(h_cif2 - z * np.sqrt(h_var2), 0.0, 1.0)
        h_cif2_u = np.clip(h_cif2 + z * np.sqrt(h_var2), 0.0, 1.0)

        h_rows = []
        for i, h in enumerate(h_arr):
            bias_abs = float(h_naive[i] - h_cif1[i])
            bias_rel = float(
                (bias_abs / h_cif1[i] * 100.0) if h_cif1[i] > 1e-4 else 0.0
            )
            h_rows.append(
                {
                    "horizon": h,
                    "cif_primary": round(float(h_cif1[i]), 4),
                    "cif_primary_ci": f"{h_cif1[i]:.3f} ({h_cif1_l[i]:.3f} to {h_cif1_u[i]:.3f})",
                    "cif_competing": round(float(h_cif2[i]), 4),
                    "cif_competing_ci": f"{h_cif2[i]:.3f} ({h_cif2_l[i]:.3f} to {h_cif2_u[i]:.3f})",
                    "naive_1_minus_km": round(float(h_naive[i]), 4),
                    "overestimation_abs": round(bias_abs, 4),
                    "overestimation_relative_pct": round(bias_rel, 2),
                }
            )
            if bias_abs > 0.03:
                clinical_warnings.append(
                    f"Horizon t={h}: Naive 1 - KM overestimates primary event risk by {bias_abs * 100:.1f}% "
                    f"({bias_rel:.1f}% relative) due to ignoring competing risks."
                )
        h_df = pd.DataFrame(h_rows)

    # Global bias check
    final_bias = float(cif_df["bias_overestimation"].iloc[-1])
    if final_bias > 0.05 and not clinical_warnings:
        clinical_warnings.append(
            f"Substantial competing risk overestimation: Naive 1 - KM exceeds true Aalen-Johansen CIF "
            f"by {final_bias * 100:.1f}% at maximum follow-up."
        )

    for msg in clinical_warnings:
        logger.warning(msg)

    return CompetingRisksCIFResult(
        ajf_primary=ajf_primary,
        ajf_competing=ajf_competing,
        km_naive=km_naive,
        cif_table=cif_df,
        horizon_summary=h_df,
        event_of_interest=event_of_interest,
        competing_event=competing_event,
        probability_conservation_max_error=round(max_error, 4),
        clinical_warnings=clinical_warnings,
    )


def fit_competing_risks_regression(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    covariates: list[str],
    event_of_interest: int = 1,
    competing_event: int = 2,
    id_col: str | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Fits dual Cause-Specific Cox Proportional Hazards and Fine-Gray Subdistribution Hazard models.

    Cause-Specific Cox Model (csHR):
        Censors competing events at the time of failure. Directly measures biological/etiological
        association among individuals currently alive and free of competing risks.

    Fine-Gray Subdistribution Model (SHR):
        Employs Inverse Probability of Censoring Weighting (IPCW) counting process dataset expansion.
        Subjects experiencing competing risks remain in the risk set beyond failure time with weight
        G_hat(t) / G_hat(T_i). Directly measures cumulative incidence effect (clinical prognosis).

    Parameters:
        df: Input DataFrame.
        duration_col: Follow-up duration column name.
        event_col: Categorical event column (0 = censor, 1 = primary, 2 = competing).
        covariates: List of baseline covariate column names.
        event_of_interest: Code for primary event (default 1).
        competing_event: Code for competing risk (default 2).
        id_col: Subject ID column name. If None, uses DataFrame index.
        alpha: Significance level (default 0.05).

    Returns:
        dict containing model_cs, model_fg, side-by-side comparison_table, and interpretations.
    """
    if event_of_interest == competing_event:
        raise ValueError(
            f"event_of_interest ({event_of_interest}) and competing_event ({competing_event}) must be distinct."
        )

    required_cols = [duration_col, event_col] + covariates
    clean_df = df[required_cols].dropna().copy()
    if len(clean_df) < 5:
        raise ValueError("Competing risks regression requires at least 5 observations.")

    clean_df["_subject_id"] = (
        clean_df[id_col].values
        if id_col and id_col in clean_df.columns
        else np.arange(len(clean_df))
    )

    # 1. Cause-Specific Cox Regression
    df_cs = clean_df.copy()
    df_cs["_cs_event"] = (df_cs[event_col] == event_of_interest).astype(int)
    cph_cs = CoxPHFitter(alpha=alpha)
    cph_cs.fit(
        df_cs[[duration_col, "_cs_event"] + covariates],
        duration_col=duration_col,
        event_col="_cs_event",
        robust=True,
    )

    # 2. Fine-Gray Model via IPCW Dataset Expansion
    # Fit censoring survival G(t) = P(C > t) where censoring is C = (event == 0)
    km_censor = KaplanMeierFitter()
    km_censor.fit(
        clean_df[duration_col], event_observed=(clean_df[event_col] == 0).astype(int)
    )
    g_sf = km_censor.survival_function_

    fail_times = np.sort(
        clean_df.loc[clean_df[event_col] == event_of_interest, duration_col].unique()
    )

    def _get_G(t: float) -> float:
        idx = g_sf.index.searchsorted(t, side="left") - 1
        val = float(g_sf.iloc[idx, 0]) if idx >= 0 else 1.0
        return max(val, 1e-4)

    expanded_rows = []
    for _, row in clean_df.iterrows():
        s_id = row["_subject_id"]
        t_i = float(row[duration_col])
        e_i = int(row[event_col])
        cov_vals = {cov: row[cov] for cov in covariates}

        if e_i == event_of_interest:
            r = {"_id": s_id, "_start": 0.0, "_stop": t_i, "_event": 1, "_weight": 1.0}
            r.update(cov_vals)
            expanded_rows.append(r)
        elif e_i == 0:
            r = {"_id": s_id, "_start": 0.0, "_stop": t_i, "_event": 0, "_weight": 1.0}
            r.update(cov_vals)
            expanded_rows.append(r)
        elif e_i == competing_event:
            # Baseline interval up to competing event time
            r0 = {"_id": s_id, "_start": 0.0, "_stop": t_i, "_event": 0, "_weight": 1.0}
            r0.update(cov_vals)
            expanded_rows.append(r0)

            # Continued presence in risk set for subsequent failure times
            subseq_times = fail_times[fail_times > t_i]
            g_ti = _get_G(t_i)
            prev_t = t_i
            for tj in subseq_times:
                g_tj = _get_G(tj)
                w_ij = max(g_tj / g_ti, 1e-4)
                rj = {
                    "_id": s_id,
                    "_start": prev_t,
                    "_stop": tj,
                    "_event": 0,
                    "_weight": w_ij,
                }
                rj.update(cov_vals)
                expanded_rows.append(rj)
                prev_t = tj

    exp_df = pd.DataFrame(expanded_rows)

    cph_fg = CoxPHFitter(alpha=alpha)
    cph_fg.fit(
        exp_df,
        duration_col="_stop",
        entry_col="_start",
        event_col="_event",
        weights_col="_weight",
        cluster_col="_id",
        robust=True,
    )

    # 3. Compile Side-by-Side Comparison Table
    rows = []
    for cov in covariates:
        cs_hr = float(cph_cs.hazard_ratios_[cov])
        cs_ci_row = cph_cs.confidence_intervals_.loc[cov]
        cs_ci_l = float(np.exp(cs_ci_row.iloc[0]))
        cs_ci_u = float(np.exp(cs_ci_row.iloc[1]))
        cs_se = float(cph_cs.standard_errors_[cov])
        cs_p = float(cph_cs.summary.loc[cov, "p"])

        fg_shr = float(cph_fg.hazard_ratios_[cov])
        fg_ci_row = cph_fg.confidence_intervals_.loc[cov]
        fg_ci_l = float(np.exp(fg_ci_row.iloc[0]))
        fg_ci_u = float(np.exp(fg_ci_row.iloc[1]))
        fg_se = float(cph_fg.standard_errors_[cov])
        fg_p = float(cph_fg.summary.loc[cov, "p"])

        # Discrepancy indicator
        cs_sig = cs_p < 0.05
        fg_sig = fg_p < 0.05
        divergent = (cs_sig != fg_sig) or ((cs_hr > 1.0) != (fg_shr > 1.0))

        rows.append(
            {
                "covariate": cov,
                "cause_specific_hr": round(cs_hr, 4),
                "cs_ci_lower": round(cs_ci_l, 4),
                "cs_ci_upper": round(cs_ci_u, 4),
                "cs_95_ci": f"{cs_hr:.3f} ({cs_ci_l:.3f} to {cs_ci_u:.3f})",
                "cs_se": round(cs_se, 4),
                "cs_p_value": cs_p,
                "fine_gray_shr": round(fg_shr, 4),
                "fg_ci_lower": round(fg_ci_l, 4),
                "fg_ci_upper": round(fg_ci_u, 4),
                "fg_95_ci": f"{fg_shr:.3f} ({fg_ci_l:.3f} to {fg_ci_u:.3f})",
                "fg_se": round(fg_se, 4),
                "fg_p_value": fg_p,
                "divergent": divergent,
            }
        )

    comp_df = pd.DataFrame(rows)

    return {
        "model_cause_specific": cph_cs,
        "model_fine_gray": cph_fg,
        "comparison_table": comp_df,
        "n_original": len(clean_df),
        "n_expanded": len(exp_df),
        "event_distribution": {
            "censored": int(np.sum(clean_df[event_col] == 0)),
            "event_of_interest": int(np.sum(clean_df[event_col] == event_of_interest)),
            "competing_risk": int(np.sum(clean_df[event_col] == competing_event)),
        },
        "interpretation": {
            "cause_specific": "Cause-specific HR measures relative event rate among subjects currently event-free (Etiology).",
            "fine_gray": "Subdistribution HR measures relative effect on cumulative incidence function over time (Prognosis).",
        },
    }

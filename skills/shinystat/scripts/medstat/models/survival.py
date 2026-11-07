"""
Survival Analysis Module.

Provides Kaplan-Meier estimation, log-rank testing, Cox Proportional Hazards
regression, and Schoenfeld residual proportional hazard assumption tests via lifelines.
Pure headless with zero UI dependencies.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lifelines import (
    AalenJohansenFitter,
    CoxPHFitter,
    CoxTimeVaryingFitter,
    KaplanMeierFitter,
)
from lifelines.exceptions import StatisticalWarning
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


def check_lifelines_entry_cluster_guard(
    entry_col: Any = None, cluster_col: Any = None
) -> None:
    """
    Guard against upstream lifelines defect where combining entry_col and cluster_col
    severely inflates standard errors due to SemiParametricPHFitter._compute_score_within_strata
    ignoring entry times (counting-process start intervals).
    """
    if entry_col is not None and cluster_col is not None:
        warnings.warn(
            "Upstream lifelines defect detected: combining entry_col and cluster_col in CoxPHFitter "
            "causes severe variance inflation because SemiParametricPHFitter._compute_score_within_strata "
            "ignores entry times. Use medstat.models.survival.entry_aware_cluster_sandwich instead.",
            UserWarning,
            stacklevel=2,
        )


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


def schoenfeld_diagnostics(
    cph: CoxPHFitter,
    df: pd.DataFrame,
    time_transform: str = "rank",
) -> dict[str, Any]:
    """
    Evaluate proportional hazards assumption via Schoenfeld residuals with Bonferroni multiplicity control.

    Parameters:
        cph: Fitted CoxPHFitter model.
        df: DataFrame containing duration, event, and covariates used for fitting.
        time_transform: Time transformation for Schoenfeld residuals ('rank', 'km', etc.).

    Returns:
        dict containing p_values, test_statistics, bonferroni_alpha,
        strong_violations, weak_violations, indeterminate_tests,
        no_ph_violation_detected, PH_Assumptions_Passed, and PH_Note.
    """
    prop_test = proportional_hazard_test(cph, df, time_transform=time_transform)
    p_values = {k: float(v) for k, v in prop_test.summary["p"].items()}
    test_statistics = {
        k: float(v) for k, v in prop_test.summary["test_statistic"].items()
    }
    k = len(p_values)
    bonferroni_alpha = 0.05 / max(k, 1)

    indeterminate_tests = [var for var, p in p_values.items() if not np.isfinite(p)]
    strong_violations = [
        var for var, p in p_values.items() if np.isfinite(p) and p <= bonferroni_alpha
    ]
    weak_violations = [
        var
        for var, p in p_values.items()
        if np.isfinite(p) and bonferroni_alpha < p <= 0.05
    ]
    passed = (
        len(strong_violations) == 0
        and len(weak_violations) == 0
        and len(indeterminate_tests) == 0
    )

    if passed:
        note = "No proportional hazards violation detected (all p > 0.05)."
    else:
        notes = []
        if strong_violations or weak_violations:
            notes.append(
                f"PH violation detected: {len(strong_violations)} strong (p <= {bonferroni_alpha:.4f}), "
                f"{len(weak_violations)} weak (0.05 >= p > {bonferroni_alpha:.4f})."
            )
        if indeterminate_tests:
            notes.append(
                f"Indeterminate PH test result: non-finite p-value for {indeterminate_tests}."
            )
        note = " ".join(notes)

    remediation = None
    if strong_violations or weak_violations:
        remediation = (
            "Proportional hazards assumption violated (Schoenfeld p < 0.05). "
            "Action: Add stratification (refit_cox_with_strata / strata_cols) for nuisance covariates, "
            "or period-specific time-varying hazard ratios (piecewise_hr) for exposure."
        )
    elif indeterminate_tests:
        remediation = (
            f"Indeterminate proportional hazards test result: non-finite p-value for {indeterminate_tests}. "
            "Action: Inspect Schoenfeld residual plots (plot_schoenfeld_residuals) and verify event density across follow-up."
        )

    return {
        "p_values": p_values,
        "test_statistics": test_statistics,
        "bonferroni_alpha": bonferroni_alpha,
        "strong_violations": strong_violations,
        "weak_violations": weak_violations,
        "indeterminate_tests": indeterminate_tests,
        "no_ph_violation_detected": passed,
        "PH_Assumptions_Passed": passed,
        "PH_Note": note,
        "agent_remediation": remediation,
    }


def refit_cox_with_strata(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    covariates: list[str],
    strata_cols: list[str],
    exposure: str,
    penalizer: float = 0.0,
) -> pd.DataFrame:
    """
    Refit Cox model with stratified baseline hazards for nuisance covariates violating PH.

    Parameters:
        df: Input DataFrame.
        duration_col: Follow-up duration column.
        event_col: Event indicator column (0/1).
        covariates: Covariates to include in the model.
        strata_cols: Covariates to stratify on (baseline hazard varies by stratum).
        exposure: Primary exposure variable name. Cannot be in strata_cols.
        penalizer: L2 penalty parameter.

    Returns:
        pd.DataFrame containing HR table for non-strata covariates with retention metadata.
    """
    if exposure in strata_cols:
        raise ValueError(
            f"Exposure variable '{exposure}' cannot be in strata_cols; "
            "stratification eliminates the exposure effect estimate."
        )

    all_needed = list(
        dict.fromkeys([duration_col, event_col] + covariates + strata_cols)
    )
    n_initial = len(df)
    missing_by_col = {
        col: int(df[col].isna().sum()) for col in all_needed if df[col].isna().sum() > 0
    }
    clean_df = df[all_needed].dropna().copy()
    n_analyzed = len(clean_df)
    n_excluded = n_initial - n_analyzed

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(
        clean_df, duration_col=duration_col, event_col=event_col, strata=strata_cols
    )

    summary = cph.summary.copy()
    res_df = pd.DataFrame(
        {
            "coef": summary["coef"],
            "exp(coef)": summary["exp(coef)"],
            "hazard_ratio": summary["exp(coef)"],
            "se(coef)": summary["se(coef)"],
            "se": summary["se(coef)"],
            "z": summary["z"],
            "p": summary["p"],
            "p_value": summary["p"],
            "exp(coef) lower 95%": summary["exp(coef) lower 95%"],
            "hr_ci_lower": summary["exp(coef) lower 95%"],
            "exp(coef) upper 95%": summary["exp(coef) upper 95%"],
            "hr_ci_upper": summary["exp(coef) upper 95%"],
        },
        index=summary.index,
    )
    res_df.attrs["model"] = cph
    res_df.attrs["strata_cols"] = list(strata_cols)
    res_df.attrs["exposure"] = exposure
    res_df.attrs["retention"] = {
        "n_initial": n_initial,
        "n_analyzed": n_analyzed,
        "n_excluded": n_excluded,
        "exclusion_reasons": {
            f"Missing {col}": count for col, count in missing_by_col.items()
        },
    }
    return res_df


def piecewise_hr(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    exposure: str,
    covariates: list[str] | None = None,
    cut_points: float | list[float] = 2.0,
    penalizer: float = 0.0,
) -> pd.DataFrame:
    """
    Estimate period-specific hazard ratios for an exposure violating proportional hazards.

    Builds episode-split counting-process data and fits via lifelines CoxTimeVaryingFitter.
    SEs are taken from model-based variance (cluster_col and robust=True omitted to avoid
    upstream lifelines entry-time variance inflation defect).

    Parameters:
        df: Input DataFrame.
        duration_col: Follow-up duration column.
        event_col: Event indicator column (0/1).
        exposure: Primary exposure variable name.
        covariates: Optional additional baseline confounders to adjust for.
        cut_points: Single cut time or list of cut times defining time periods.
        penalizer: L2 penalty parameter.

    Returns:
        pd.DataFrame containing period-specific HRs for the exposure.
    """
    if isinstance(cut_points, (int, float, np.integer, np.floating)):
        cuts = [float(cut_points)]
    else:
        cuts = [float(c) for c in cut_points]

    if len(cuts) == 0:
        raise ValueError("cut_points must contain at least one cut time.")
    if not np.all(np.isfinite(cuts)):
        raise ValueError("cut_points must be finite numbers.")
    if any(c <= 0 for c in cuts):
        raise ValueError("cut_points must be strictly positive (> 0).")
    cuts = sorted(list(set(cuts)))

    other_covs = [c for c in (covariates or []) if c != exposure]
    cols_needed = [duration_col, event_col, exposure] + other_covs
    clean_df = df[cols_needed].dropna().reset_index(drop=True)

    n_initial = len(clean_df)
    clean_df = clean_df[clean_df[duration_col] > 0].reset_index(drop=True)
    n_zero_duration_excluded = n_initial - len(clean_df)

    durations = clean_df[duration_col].to_numpy()
    if len(durations) == 0:
        raise ValueError(
            "No positive duration observations remaining for piecewise Cox model."
        )

    max_dur = float(durations.max())
    if any(c >= max_dur for c in cuts):
        raise ValueError(
            f"All cut_points must be strictly less than maximum follow-up duration ({max_dur})."
        )

    boundaries = [0.0] + cuts + [np.inf]
    period_names = []
    for m in range(len(boundaries) - 1):
        lo = boundaries[m]
        hi = boundaries[m + 1]
        if hi == np.inf:
            period_names.append(f"t > {lo}")
        elif lo == 0.0:
            period_names.append(f"t <= {hi}")
        else:
            period_names.append(f"{lo} < t <= {hi}")

    events = clean_df[event_col].to_numpy()
    exposures = clean_df[exposure].to_numpy()
    other_vals = {c: clean_df[c].to_numpy() for c in other_covs}

    period_term_cols = [f"{exposure}_p{m + 1}" for m in range(len(period_names))]

    dfs = []
    for m in range(len(period_names)):
        lo = boundaries[m]
        hi = boundaries[m + 1]
        mask = durations > lo
        if not np.any(mask):
            continue
        sub_df = pd.DataFrame(
            {
                "_id": clean_df.index[mask],
                "_start": lo,
                "_stop": np.minimum(
                    durations[mask], hi if hi != np.inf else durations[mask]
                ),
                "_event": np.where((durations[mask] <= hi) & (events[mask] == 1), 1, 0),
            }
        )
        for k_idx, term in enumerate(period_term_cols):
            sub_df[term] = np.where(k_idx == m, exposures[mask], 0.0)
        for c in other_covs:
            sub_df[c] = other_vals[c][mask]
        dfs.append(sub_df)

    split_df = pd.concat(dfs, ignore_index=True)

    ctv = CoxTimeVaryingFitter(penalizer=penalizer)
    ctv.fit(
        split_df,
        id_col="_id",
        start_col="_start",
        stop_col="_stop",
        event_col="_event",
    )

    summary = ctv.summary.copy()
    rows = []
    for m, term in enumerate(period_term_cols):
        if term in summary.index:
            coef = float(summary.loc[term, "coef"])
            exp_coef = float(summary.loc[term, "exp(coef)"])
            se = float(summary.loc[term, "se(coef)"])
            z = float(summary.loc[term, "z"])
            p = float(summary.loc[term, "p"])
            ci_lower = float(summary.loc[term, "exp(coef) lower 95%"])
            ci_upper = float(summary.loc[term, "exp(coef) upper 95%"])
        else:
            coef = exp_coef = se = z = p = ci_lower = ci_upper = np.nan
        rows.append(
            {
                "period": period_names[m],
                "term": term,
                "coef": coef,
                "exp(coef)": exp_coef,
                "hazard_ratio": exp_coef,
                "se(coef)": se,
                "se": se,
                "z": z,
                "p": p,
                "p_value": p,
                "exp(coef) lower 95%": ci_lower,
                "hr_ci_lower": ci_lower,
                "exp(coef) upper 95%": ci_upper,
                "hr_ci_upper": ci_upper,
            }
        )
    res_df = pd.DataFrame(rows, index=period_term_cols)
    res_df.attrs["model"] = ctv
    res_df.attrs["full_summary"] = summary
    res_df.attrs["cut_points"] = cuts
    res_df.attrs["exposure"] = exposure
    res_df.attrs["n_zero_duration_excluded"] = n_zero_duration_excluded
    return res_df


def fit_survival_analysis_suite(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    strata_col: str | None = None,
    covariates: list[str] | None = None,
    penalizer: float = 0.0,
    standardize_continuous: bool = False,
    continuous_covariates: list[str] | None = None,
) -> dict[str, Any]:
    """Run Kaplan-Meier, Log-Rank, and Cox PH modeling with per-covariate Schoenfeld test and optional scale-aware standardization."""
    n_initial = len(df)

    # 1. Unadjusted Kaplan-Meier cohort: subset duration, event, and strata (preserves KM cohort)
    km_cols = [duration_col, event_col] + ([strata_col] if strata_col else [])
    clean_km = df.dropna(subset=km_cols).copy()

    # Validate event indicator strictly binary and duration non-negative
    ev_km = clean_km[event_col].to_numpy()
    if not np.all(np.isin(ev_km, [0, 1])):
        raise ValueError(
            f"Event indicator '{event_col}' must contain strictly binary values (0 = Censored, 1 = Event)."
        )
    dur_km = clean_km[duration_col].to_numpy()
    if (dur_km < 0).any():
        raise ValueError(
            f"Duration column '{duration_col}' must contain non-negative values (>= 0)."
        )

    # Kaplan-Meier curve & median survival
    km_results: dict[str, Any] = {}
    km_curves: dict[str, Any] = {}
    risk_table = pd.DataFrame()
    if strata_col:
        groups = clean_km[strata_col].unique()
        for g in groups:
            sub = clean_km[clean_km[strata_col] == g]
            kmf = KaplanMeierFitter()
            kmf.fit(sub[duration_col], sub[event_col], label=str(g))
            km_results[f"Median Survival ({g})"] = float(kmf.median_survival_time_)
            ci_df = kmf.confidence_interval_
            cens_t = sub.loc[sub[event_col] == 0, duration_col].to_numpy()
            km_curves[str(g)] = {
                "timeline": np.asarray(kmf.timeline),
                "survival": np.asarray(kmf.survival_function_[str(g)]),
                "ci_lower": np.asarray(ci_df.iloc[:, 0]),
                "ci_upper": np.asarray(ci_df.iloc[:, 1]),
                "censored_times": cens_t,
                "censored_survival": (
                    np.asarray(kmf.survival_function_at_times(cens_t))
                    if len(cens_t) > 0
                    else np.array([])
                ),
            }

        # Risk table at milestone intervals
        max_t = float(clean_km[duration_col].max())
        milestones = np.linspace(0, max_t, 5).round(1)
        if len(milestones) > 0 and milestones[-1] > max_t:
            milestones[-1] = float(np.floor(max_t * 10) / 10)
        r_rows = []
        for g in groups:
            sub = clean_km[clean_km[strata_col] == g]
            row_dict = {"Group": str(g)}
            for m in milestones:
                row_dict[f"t={m}"] = int(np.sum(sub[duration_col] >= m))
            r_rows.append(row_dict)
        risk_table = pd.DataFrame(r_rows)

        # Log-rank test (2 groups or multivariate for >2 groups)
        if len(groups) == 2:
            lr_res = logrank_test(
                clean_km.loc[clean_km[strata_col] == groups[0], duration_col],
                clean_km.loc[clean_km[strata_col] == groups[1], duration_col],
                clean_km.loc[clean_km[strata_col] == groups[0], event_col],
                clean_km.loc[clean_km[strata_col] == groups[1], event_col],
            )
            km_results["Log-Rank P-value"] = float(lr_res.p_value)
        elif len(groups) > 2:
            lr_res = multivariate_logrank_test(
                clean_km[duration_col], clean_km[strata_col], clean_km[event_col]
            )
            km_results["Multivariate Log-Rank P-value"] = float(lr_res.p_value)
    else:
        kmf = KaplanMeierFitter()
        kmf.fit(clean_km[duration_col], clean_km[event_col], label="Overall")
        km_results["Overall Median Survival"] = float(kmf.median_survival_time_)
        ci_df = kmf.confidence_interval_
        cens_t = clean_km.loc[clean_km[event_col] == 0, duration_col].to_numpy()
        km_curves["Overall"] = {
            "timeline": np.asarray(kmf.timeline),
            "survival": np.asarray(kmf.survival_function_["Overall"]),
            "ci_lower": np.asarray(ci_df.iloc[:, 0]),
            "ci_upper": np.asarray(ci_df.iloc[:, 1]),
            "censored_times": cens_t,
            "censored_survival": (
                np.asarray(kmf.survival_function_at_times(cens_t))
                if len(cens_t) > 0
                else np.array([])
            ),
        }

    # 2. Cox Proportional Hazards model on complete-covariate cohort
    cox_table = pd.DataFrame()
    schoenfeld_summary: dict[str, Any] = {}
    forest_items = []
    cox_standardization = {}
    n_analyzed_cox = 0
    if covariates:
        cox_cols = [duration_col, event_col] + covariates
        clean_cox = df.dropna(subset=cox_cols).copy()
        n_analyzed_cox = len(clean_cox)
        ev_cox = clean_cox[event_col].to_numpy()
        if not np.all(np.isin(ev_cox, [0, 1])):
            raise ValueError(
                f"Event indicator '{event_col}' must contain strictly binary values."
            )

        if penalizer < 0:
            raise ValueError("penalizer must be >= 0")
        cox_data = clean_cox[cox_cols].copy()
        for var in covariates:
            col_s = cox_data[var]
            if not pd.api.types.is_numeric_dtype(col_s) or pd.api.types.is_bool_dtype(
                col_s
            ):
                raise ValueError(
                    f"Covariate '{var}' must be numeric; encode categorical covariates as binary 0/1 indicators before Cox fitting."
                )
        if standardize_continuous:
            if continuous_covariates is None:
                raise ValueError(
                    "continuous_covariates must be specified explicitly when standardize_continuous=True."
                )
            cont_set = set(continuous_covariates)
            unknown_cont = cont_set - set(covariates)
            if unknown_cont:
                raise ValueError(
                    f"continuous_covariates {sorted(unknown_cont)} must be present in covariates."
                )
            for var in covariates:
                col_s = cox_data[var]
                if var in cont_set:
                    mean_v = float(col_s.mean())
                    sd_v = float(col_s.std(ddof=1))
                    if sd_v <= 0:
                        raise ValueError(
                            f"Continuous covariate '{var}' has zero variance and cannot be standardized."
                        )
                    cox_data[var] = (col_s - mean_v) / sd_v
                    cox_standardization[var] = {
                        "mean": mean_v,
                        "sd": sd_v,
                        "scale": "per 1 SD increase",
                    }
                else:
                    if not np.all(np.isin(col_s.to_numpy(), [0, 1])):
                        raise ValueError(
                            f"Non-continuous covariate '{var}' must be pre-encoded as binary 0/1 indicator(s) before Cox fitting."
                        )
        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(cox_data, duration_col=duration_col, event_col=event_col)

        summary = cph.summary
        rows = []
        for var in covariates:
            hr = float(summary.loc[var, "exp(coef)"])
            ci_l = float(summary.loc[var, "exp(coef) lower 95%"])
            ci_u = float(summary.loc[var, "exp(coef) upper 95%"])
            p_val = float(summary.loc[var, "p"])
            rows.append(
                {
                    "Covariate": var,
                    "Hazard Ratio (95% CI)": f"{hr:.2f} ({ci_l:.2f}–{ci_u:.2f})",
                    "P-value": f"{p_val:.3f}" if p_val >= 0.001 else "<0.001",
                }
            )
            forest_items.append(
                {
                    "term": var,
                    "estimate": hr,
                    "ci_lower": ci_l,
                    "ci_upper": ci_u,
                    "p_value": p_val,
                    "scale": "HR",
                }
            )
        cox_table = pd.DataFrame(rows)

        # Proportional hazards test per covariate with Bonferroni multiplicity control
        try:
            schoenfeld_summary = schoenfeld_diagnostics(
                cph, cox_data, time_transform="rank"
            )
        except Exception as e:
            schoenfeld_summary = {"PH_Note": str(e)}

    return {
        "km_summary": km_results,
        "km_curves": km_curves,
        "risk_table": risk_table,
        "cox_table": cox_table,
        "cox_penalizer": penalizer if covariates else None,
        "cox_standardization": cox_standardization if covariates else None,
        "forest_data": forest_items,
        "schoenfeld_diagnostics": schoenfeld_summary,
        "retention_km": {
            "n_initial": n_initial,
            "n_analyzed": len(clean_km),
            "n_excluded": n_initial - len(clean_km),
        },
        "retention_cox": {
            "n_initial": n_initial,
            "n_analyzed": n_analyzed_cox,
            "n_excluded": n_initial - n_analyzed_cox,
        }
        if covariates
        else None,
    }


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
    if event_of_interest <= 0 or competing_event <= 0:
        raise ValueError(
            f"event_of_interest ({event_of_interest}) and competing_event ({competing_event}) must be positive integers (> 0); 0 is reserved for censoring."
        )
    if event_of_interest == competing_event:
        raise ValueError(
            f"event_of_interest ({event_of_interest}) and competing_event ({competing_event}) must be distinct."
        )

    dur = np.asarray(durations, dtype=float)
    if isinstance(event_observed, pd.Series):
        evt_series = event_observed
    else:
        evt_series = pd.Series(event_observed)
    evt_float = pd.to_numeric(evt_series, errors="coerce").to_numpy(dtype=float)

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


def expand_fine_gray(
    clean: pd.DataFrame,
    duration_col: str,
    event_col: str,
    covariates: list[str],
    event_of_interest: int = 1,
    competing_event: int = 2,
    id_col: str | None = None,
) -> pd.DataFrame:
    """IPCW counting-process expansion: a patient with the competing event stays in the risk set, weighted G(t)/G(t_i)."""
    clean = clean.reset_index(drop=True)
    km = KaplanMeierFitter().fit(
        clean[duration_col], event_observed=(clean[event_col] == 0).astype(int)
    )
    g_t = km.survival_function_.index.to_numpy(dtype=float)
    g_v = km.survival_function_.iloc[:, 0].to_numpy(dtype=float)

    def _G(t: np.ndarray | float) -> np.ndarray:
        idx = np.searchsorted(g_t, t, side="left") - 1
        return np.maximum(np.where(idx >= 0, g_v[np.maximum(idx, 0)], 1.0), 1e-4)

    fail = np.sort(
        clean.loc[clean[event_col] == event_of_interest, duration_col].unique()
    )
    main = clean[covariates].astype(float).copy()
    main["_id"] = (
        clean[id_col].to_numpy()
        if id_col and id_col in clean.columns
        else np.arange(len(clean))
    )
    main["_start"] = 0.0
    main["_stop"] = clean[duration_col].astype(float).to_numpy()
    main["_event"] = (clean[event_col] == event_of_interest).astype(int).to_numpy()
    main["_weight"] = 1.0
    parts = [main]
    for i in np.flatnonzero((clean[event_col] == competing_event).to_numpy()):
        t_i = float(clean.at[i, duration_col])
        later = fail[fail > t_i]
        if later.size == 0:
            continue
        blk = pd.DataFrame(
            np.repeat(clean.loc[[i], covariates].to_numpy(float), later.size, axis=0),
            columns=covariates,
        )
        blk["_id"] = clean.at[i, id_col] if id_col and id_col in clean.columns else i
        blk["_start"] = np.r_[t_i, later[:-1]]
        blk["_stop"] = later
        blk["_event"] = 0
        blk["_weight"] = np.maximum(_G(later) / float(_G(t_i)), 1e-4)
        parts.append(blk)
    return pd.concat(parts, ignore_index=True)


def entry_aware_cluster_sandwich(
    expanded: pd.DataFrame,
    covariates: list[str],
    beta: np.ndarray,
    bread: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Cluster-robust SE for counting-process Cox data: V = bread @ (sum_c U_c U_c') @ bread.

    Score residuals U_i = w_i d_i (x_i - xbar(t_i)) - w_i e^{b'x_i} * integral over (start_i, stop_i] of (x_i - xbar(s)) dLambda0(s),
    with risk set {start < t <= stop} (Breslow form). `bread` is the model-based covariance of the fitted Cox model.
    Returns (robust SE, score_gap); score_gap = max_j |sum_c U_cj| / sqrt(sum_c U_cj^2) over clusters is 0 when the fit and the
    score share the same tie convention, so a large value flags a tie mismatch.
    """
    X = expanded[covariates].to_numpy(float)
    st, sp = expanded["_start"].to_numpy(float), expanded["_stop"].to_numpy(float)
    ev, w = expanded["_event"].to_numpy(int), expanded["_weight"].to_numpy(float)
    r = w * np.exp(X @ beta)
    base = np.column_stack([r, r[:, None] * X])  # [r | r*x], summed over each risk set

    def _cum(key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(key, kind="stable")
        return key[order], np.vstack(
            [np.zeros((1, base.shape[1])), np.cumsum(base[order], axis=0)]
        )

    k_in, c_in = _cum(st)
    k_out, c_out = _cum(sp)
    t_ev = np.unique(sp[ev == 1])
    risk = (
        c_in[np.searchsorted(k_in, t_ev, side="left")]
        - c_out[np.searchsorted(k_out, t_ev, side="left")]
    )
    s0, s1 = risk[:, 0], risk[:, 1:]
    xbar = s1 / s0[:, None]
    d_k = np.bincount(
        np.searchsorted(t_ev, sp[ev == 1]),
        weights=w[ev == 1],
        minlength=len(t_ev),
    )
    d_lambda = d_k / s0
    cum_l = np.r_[0.0, np.cumsum(d_lambda)]
    cum_xl = np.vstack(
        [np.zeros(X.shape[1]), np.cumsum(xbar * d_lambda[:, None], axis=0)]
    )
    hi, lo = (
        np.searchsorted(t_ev, sp, side="right"),
        np.searchsorted(t_ev, st, side="right"),
    )
    compensator = X * (cum_l[hi] - cum_l[lo])[:, None] - (cum_xl[hi] - cum_xl[lo])
    score = -(r[:, None] * compensator)
    hit = np.flatnonzero(ev == 1)
    score[hit] += w[hit, None] * (X[hit] - xbar[np.searchsorted(t_ev, sp[hit])])

    _, inv = np.unique(expanded["_id"].to_numpy(), return_inverse=True)
    by_cluster = np.zeros((inv.max() + 1, X.shape[1]))
    np.add.at(by_cluster, inv, score)
    se = np.sqrt(np.diag(bread @ (by_cluster.T @ by_cluster) @ bread))
    score_gap = float(
        np.max(np.abs(by_cluster.sum(axis=0)) / np.sqrt((by_cluster**2).sum(axis=0)))
    )
    return se, score_gap


def bootstrap_fine_gray_se(
    clean: pd.DataFrame,
    duration_col: str,
    event_col: str,
    covariates: list[str],
    event_of_interest: int = 1,
    competing_event: int = 2,
    id_col: str | None = None,
    n_boot: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Patient-level bootstrap SE of the Fine-Gray coefficients (censoring distribution re-estimated per resample)."""
    if n_boot < 2:
        raise ValueError(f"n_boot must be at least 2, got {n_boot}")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        resample = clean.iloc[rng.integers(0, len(clean), len(clean))]
        try:
            exp_b = expand_fine_gray(
                resample,
                duration_col,
                event_col,
                covariates,
                event_of_interest,
                competing_event,
                id_col=id_col,
            )
            fit_b = _fit_fine_gray(exp_b, covariates, alpha=0.05)
            draws.append(fit_b.params_.reindex(covariates).to_numpy())
        except (
            Exception
        ):  # a resample can be degenerate (no events in a stratum); skip it
            continue
    if len(draws) < 2 or len(draws) < 0.9 * n_boot:
        raise RuntimeError(
            f"bootstrap converged in only {len(draws)}/{n_boot} resamples"
        )
    return np.std(np.asarray(draws), axis=0, ddof=1)


def _fit_fine_gray(
    expanded: pd.DataFrame, covariates: list[str], alpha: float = 0.05
) -> CoxPHFitter:
    # No cluster_col: lifelines' cluster sandwich ignores entry times and inflates SE (see entry_aware_cluster_sandwich).
    cph = CoxPHFitter(alpha=alpha)
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", StatisticalWarning
        )  # non-integer IPCW weights: variance comes from the sandwich/bootstrap
        cph.fit(
            expanded[covariates + ["_start", "_stop", "_event", "_weight"]],
            duration_col="_stop",
            entry_col="_start",
            event_col="_event",
            weights_col="_weight",
            robust=False,
        )
    return cph


class CompetingRisksRegressionResult(dict):
    """Container for dual Cause-Specific and Fine-Gray regression results."""

    @property
    def attrs(self) -> dict[str, Any]:
        return self["comparison_table"].attrs

    def set_index(self, *args, **kwargs) -> pd.DataFrame:
        return self["comparison_table"].set_index(*args, **kwargs)


def fit_competing_risks_regression(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    covariates: list[str],
    event_of_interest: int = 1,
    competing_event: int = 2,
    id_col: str | None = None,
    alpha: float = 0.05,
    se_method: str = "sandwich",
    n_boot: int = 200,
    seed: int = 0,
    score_gap_limit: float = 0.20,
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
        Standard errors are computed via entry-aware cluster sandwich (or patient-level bootstrap),
        avoiding lifelines' upstream defect when cluster_col is combined with entry_col.

    Parameters:
        df: Input DataFrame.
        duration_col: Follow-up duration column name.
        event_col: Categorical event column (0 = censor, 1 = primary, 2 = competing).
        covariates: List of baseline covariate column names.
        event_of_interest: Code for primary event (default 1).
        competing_event: Code for competing risk (default 2).
        id_col: Subject ID column name. If None, uses DataFrame index.
        alpha: Significance level (default 0.05).
        se_method: 'sandwich' (fast entry-aware cluster sandwich) or 'bootstrap' (patient resampling).
        n_boot: Number of bootstrap resamples if se_method='bootstrap' (default 200).
        seed: Random seed for bootstrap (default 0).
        score_gap_limit: Maximum permissible score_gap before warning for heavy ties (default 0.20).

    Returns:
        dict containing model_cause_specific, model_fine_gray, comparison_table, interpretations,
        fg_se_method, and fg_score_gap.
    """
    if event_of_interest <= 0 or competing_event <= 0:
        raise ValueError(
            f"event_of_interest ({event_of_interest}) and competing_event ({competing_event}) must be positive integers (> 0); 0 is reserved for censoring."
        )
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

    # 1. Cause-Specific Cox Regression (one row per patient, so lifelines' robust SE is exact)
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
    exp_df = expand_fine_gray(
        clean=clean_df,
        duration_col=duration_col,
        event_col=event_col,
        covariates=covariates,
        event_of_interest=event_of_interest,
        competing_event=competing_event,
        id_col="_subject_id",
    )

    cph_fg = _fit_fine_gray(exp_df, covariates, alpha=alpha)
    beta = cph_fg.params_.reindex(covariates).to_numpy()
    score_gap = float("nan")

    if se_method == "sandwich":
        bread = cph_fg.variance_matrix_.loc[covariates, covariates].to_numpy()
        fg_se_arr, score_gap = entry_aware_cluster_sandwich(
            exp_df, covariates, beta, bread
        )
        if score_gap > score_gap_limit:
            warnings.warn(
                f"score_gap={score_gap:.3f} > {score_gap_limit}: heavy ties; rerun with se_method='bootstrap'",
                UserWarning,
                stacklevel=2,
            )
    elif se_method == "bootstrap":
        fg_se_arr = bootstrap_fine_gray_se(
            clean_df,
            duration_col,
            event_col,
            covariates,
            event_of_interest,
            competing_event,
            id_col="_subject_id",
            n_boot=n_boot,
            seed=seed,
        )
    else:
        raise ValueError("se_method must be 'sandwich' or 'bootstrap'")

    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))

    # 3. Compile Side-by-Side Comparison Table
    rows = []
    for j, cov in enumerate(covariates):
        cs_hr = float(cph_cs.hazard_ratios_[cov])
        cs_ci_row = cph_cs.confidence_intervals_.loc[cov]
        cs_ci_l = float(np.exp(cs_ci_row.iloc[0]))
        cs_ci_u = float(np.exp(cs_ci_row.iloc[1]))
        cs_se = float(cph_cs.standard_errors_[cov])
        cs_p = float(cph_cs.summary.loc[cov, "p"])

        fg_shr = float(np.exp(beta[j]))
        fg_ci_l = float(np.exp(beta[j] - z_crit * fg_se_arr[j]))
        fg_ci_u = float(np.exp(beta[j] + z_crit * fg_se_arr[j]))
        fg_se = float(fg_se_arr[j])
        fg_p = float(2.0 * stats.norm.sf(abs(beta[j] / fg_se_arr[j])))

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
    comp_df.attrs.update({"fg_se_method": se_method, "fg_score_gap": score_gap})

    return CompetingRisksRegressionResult(
        {
            "model_cause_specific": cph_cs,
            "model_fine_gray": cph_fg,
            "comparison_table": comp_df,
            "n_original": len(clean_df),
            "n_expanded": len(exp_df),
            "fg_se_method": se_method,
            "fg_score_gap": score_gap,
            "event_distribution": {
                "censored": int(np.sum(clean_df[event_col] == 0)),
                "event_of_interest": int(
                    np.sum(clean_df[event_col] == event_of_interest)
                ),
                "competing_risk": int(np.sum(clean_df[event_col] == competing_event)),
            },
            "interpretation": {
                "cause_specific": "Cause-specific HR measures relative event rate among subjects currently event-free (Etiology).",
                "fine_gray": "Subdistribution HR measures relative effect on cumulative incidence function over time (Prognosis).",
            },
        }
    )


__all__ = [
    "fit_kaplan_meier",
    "compare_survival_curves",
    "check_lifelines_entry_cluster_guard",
    "fit_cox_ph",
    "check_proportional_hazards",
    "schoenfeld_diagnostics",
    "refit_cox_with_strata",
    "piecewise_hr",
    "fit_survival_analysis_suite",
    "CompetingRisksCIFResult",
    "fit_competing_risks_cif",
    "expand_fine_gray",
    "entry_aware_cluster_sandwich",
    "bootstrap_fine_gray_se",
    "CompetingRisksRegressionResult",
    "fit_competing_risks_regression",
]

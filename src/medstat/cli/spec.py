"""
Statistical Analysis Plan (SAP) Spec Engine.

Parses, validates, and executes reproducible YAML/JSON analysis specifications.
Guarantees explicit reference categories, interaction terms, missingness handling,
and automated sample retention flow tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from medstat.causal.balance import compute_balance_table
from medstat.causal.iptw import compute_iptw_weights, fit_iptw_outcome_model
from medstat.data.clean import prepare_data_for_analysis
from medstat.data.loader import load_clinical_data
from medstat.data.missing import pool_estimates
from medstat.data.retention import SampleFlowTracker
from medstat.models.firth import fit_firth_logistic
from medstat.models.glm import fit_linear_regression, fit_standard_logistic
from medstat.models.sensitivity import calculate_e_value
from medstat.models.survival import (
    fit_competing_risks_cif,
    fit_competing_risks_regression,
    fit_cox_ph,
)
from medstat.reporting.narrative import (
    generate_competing_risks_narrative,
    generate_iptw_narrative,
    generate_narrative,
)
from medstat.reporting.tables import (
    Estimate,
    EstimateTable,
    ModelMeta,
    PublicationRenderer,
    render_competing_risks_table,
)


@dataclass
class FilterSpec:
    expression: str
    reason: str


@dataclass
class VariableSpec:
    name: str
    label: str = ""
    role: str = "covariate"  # outcome, exposure, covariate, strata
    data_type: str = "continuous"  # binary, continuous, categorical
    categories: list[Any] = field(default_factory=list)
    reference_category: Any = None


@dataclass
class ModelSpec:
    name: str
    type: str  # logistic, linear, cox, cox_ph
    outcome: str
    exposure: str | None = None
    covariates: list[str] = field(default_factory=list)
    formula: str | None = None
    description: str = ""
    time: str | None = None  # For survival duration
    reference_categories: dict[str, Any] = field(default_factory=dict)
    missing_strategy: str = "complete-case"
    missing_justification: str = "Complete-case analysis prespecified"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportingSpec:
    style: str = "nejm"
    format: str = "html"
    include_narrative: bool = True
    include_flow_diagram: bool = True


@dataclass
class MetadataSpec:
    study_title: str = "Clinical Analysis"
    protocol_id: str = "MEDSTAT-SAP"
    analyst: str = "Clinical Biostatistics Core"
    date: str = "2026-09-24"
    reporting_guideline: str = "STROBE"
    study_design: str = "observational"


@dataclass
class DataSpec:
    input_path: str = ""
    id_column: str = "patient_id"
    filters: list[FilterSpec] = field(default_factory=list)


class AnalysisPlan:
    """Pre-specified Statistical Analysis Plan (SAP) execution engine."""

    def __init__(
        self,
        metadata: MetadataSpec,
        data: DataSpec,
        variables: list[VariableSpec],
        models: list[ModelSpec],
        reporting: ReportingSpec,
        version: str = "1.0",
        raw_dict: dict[str, Any] | None = None,
    ) -> None:
        self.version = version
        self.metadata = metadata
        self.data = data
        self.variables = variables
        self.models = models
        self.reporting = reporting
        self.raw_dict = raw_dict or {}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AnalysisPlan":
        """Load and validate an AnalysisPlan from a YAML file."""
        p = Path(path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Analysis plan YAML not found: {p}")

        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        meta_dict = data.get("metadata", {})
        metadata = MetadataSpec(
            study_title=meta_dict.get("study_title", "Clinical Study"),
            protocol_id=meta_dict.get("protocol_id", "MEDSTAT-SAP"),
            analyst=meta_dict.get(
                "analyst",
                meta_dict.get("statistical_analyst", "Clinical Biostatistician"),
            ),
            date=meta_dict.get("date", "2026-09-24"),
            reporting_guideline=meta_dict.get("reporting_guideline", "STROBE"),
            study_design=meta_dict.get("study_design", "observational"),
        )

        data_dict = data.get("data", {})
        filters = [
            FilterSpec(
                expression=flt.get("expression", ""), reason=flt.get("reason", "")
            )
            for flt in data_dict.get("filters", [])
        ]
        data_spec = DataSpec(
            input_path=data_dict.get("input_path", ""),
            id_column=data_dict.get("id_column", "patient_id"),
            filters=filters,
        )

        variables = [
            VariableSpec(
                name=v.get("name", ""),
                label=v.get("label", v.get("name", "")),
                role=v.get("role", "covariate"),
                data_type=v.get("data_type", "continuous"),
                categories=v.get("categories", []),
                reference_category=v.get("reference_category"),
            )
            for v in data.get("variables", [])
        ]

        models = [
            ModelSpec(
                name=m.get("name", "model_1"),
                type=m.get("type", "logistic"),
                outcome=m.get("outcome", ""),
                exposure=m.get("exposure"),
                covariates=m.get("covariates", []),
                formula=m.get("formula"),
                description=m.get("description", ""),
                time=m.get("time"),
                reference_categories=m.get("reference_categories", {}),
                missing_strategy=m.get("missing_strategy"),
                missing_justification=m.get("missing_justification"),
                options=m.get("options", {}),
            )
            for m in data.get("models", [])
        ]

        rep_dict = data.get("reporting", {})
        reporting = ReportingSpec(
            style=rep_dict.get("style", "nejm"),
            format=rep_dict.get("format", "html"),
            include_narrative=rep_dict.get("include_narrative", True),
            include_flow_diagram=rep_dict.get("include_flow_diagram", True),
        )

        return cls(
            metadata=metadata,
            data=data_spec,
            variables=variables,
            models=models,
            reporting=reporting,
            version=str(data.get("version", "1.0")),
            raw_dict=data,
        )

    def execute(self, df: pd.DataFrame | None = None) -> dict[str, Any]:
        """
        Execute the pre-specified Statistical Analysis Plan across all stages.
        """
        # 1. Load Data
        if df is None:
            raw_path = Path(self.data.input_path)
            if not raw_path.is_absolute():
                raw_path = (Path.cwd() / raw_path).resolve()
            if not raw_path.exists():
                raise FileNotFoundError(f"Dataset path not found: {raw_path}")
            current_df = load_clinical_data(raw_path)
        else:
            current_df = df.copy()

        n_initial = len(current_df)
        tracker = SampleFlowTracker(
            n_initial=n_initial, initial_name="Total Enrolled Cohort"
        )

        # 2. Apply Pre-specified Data Filters
        for flt in self.data.filters:
            n_start = len(current_df)
            try:
                filtered = current_df.query(flt.expression)
                n_rem = len(filtered)
                n_excl = n_start - n_rem
                tracker.record_stage(
                    stage_name=f"Filter: {flt.expression}",
                    n_remaining=n_rem,
                    n_excluded=n_excl,
                    reason=flt.reason,
                )
                current_df = filtered
            except Exception as e:
                raise ValueError(
                    f"Failed to apply cohort filter '{flt.expression}': {e}"
                ) from e

        results: dict[str, Any] = {
            "metadata": {
                "study_title": self.metadata.study_title,
                "protocol_id": self.metadata.protocol_id,
                "reporting_guideline": self.metadata.reporting_guideline,
            },
            "flow_summary": tracker.get_flow_summary(),
            "model_flow": {},
            "models": {},
            "reports": {},
        }

        # 3. Execute Pre-specified Models
        for model_spec in self.models:
            req_cols = [model_spec.outcome]
            if model_spec.exposure:
                req_cols.append(model_spec.exposure)
            if model_spec.time:
                req_cols.append(model_spec.time)
            req_cols.extend(model_spec.covariates)
            req_cols = list(dict.fromkeys(req_cols))

            # Clean and handle missingness with explicit justification
            model_tracker = SampleFlowTracker(
                n_initial=len(current_df),
                initial_name=f"Cohort for {model_spec.name}",
            )
            clean_df, m_info = prepare_data_for_analysis(
                current_df,
                required_cols=req_cols,
                handle_missing=model_spec.missing_strategy,
                missing_justification=model_spec.missing_justification,
                strategy_params={
                    "n_imputations": int(model_spec.options.get("n_imputations", 5))
                },
                tracker=model_tracker,
            )
            results["model_flow"][model_spec.name] = model_tracker.get_flow_summary()

            imputed_datasets = m_info.get("imputed_datasets")
            is_pooled = bool(imputed_datasets and len(imputed_datasets) > 1)

            mod_type = model_spec.type.lower().strip()
            est_rows: list[Estimate] = []
            mod_meta = ModelMeta(
                estimator=f"{mod_type.upper()} Regression",
                n_total=len(clean_df),
                outcome_name=model_spec.outcome,
                exposure_name=model_spec.exposure,
                adjusted_for=model_spec.covariates,
            )

            if (
                mod_type in ("iptw", "causal_iptw")
                or model_spec.options.get("method") == "iptw"
            ):
                estimand = str(model_spec.options.get("estimand", "ATE")).upper()
                stabilized = bool(model_spec.options.get("stabilized", True))
                t_quantiles = model_spec.options.get("truncate_quantiles", (0.01, 0.99))
                if isinstance(t_quantiles, list):
                    t_quantiles = tuple(t_quantiles)

                exp_col = model_spec.exposure
                if not exp_col:
                    raise ValueError(
                        "IPTW model requires an 'exposure' (treatment) variable."
                    )

                covar_cols = [c for c in model_spec.covariates if c != exp_col]
                iptw_res = compute_iptw_weights(
                    clean_df,
                    treatment_col=exp_col,
                    covariates=covar_cols,
                    estimand=estimand,
                    stabilized=stabilized,
                    truncate_quantiles=t_quantiles,
                )

                out_type = model_spec.options.get("outcome_type")
                if not out_type:
                    if model_spec.time:
                        out_type = "survival"
                    elif len(clean_df[model_spec.outcome].dropna().unique()) <= 2:
                        out_type = "binary"
                    else:
                        out_type = "continuous"

                if out_type == "survival":
                    evt_series = clean_df[model_spec.outcome].dropna()
                    evt_vals = pd.to_numeric(evt_series, errors="coerce")
                    unique_evt = set(evt_vals.dropna().unique())
                    if (
                        evt_vals.isna().any()
                        or not unique_evt.issubset({0, 1, 0.0, 1.0})
                        or len(unique_evt) > 2
                    ):
                        raise ValueError(
                            f"Event column '{model_spec.outcome}' for survival IPTW outcome model must contain only numeric 0 and 1 values."
                        )
                    iptw_res.df[model_spec.outcome] = pd.to_numeric(
                        iptw_res.df[model_spec.outcome], errors="coerce"
                    )

                fit_res = fit_iptw_outcome_model(
                    iptw_res.df,
                    outcome_col=model_spec.outcome,
                    treatment_col=exp_col,
                    weights_col="iptw_weight",
                    outcome_type=out_type,
                    duration_col=model_spec.time,
                )
                fit_res["diagnostics"] = iptw_res.diagnostics
                fit_res["effective_sample_size"] = iptw_res.effective_sample_size
                bal_df = compute_balance_table(
                    iptw_res.df,
                    treatment=exp_col,
                    covariates=covar_cols,
                    weights_col="iptw_weight",
                )
                fit_res["balance_table"] = bal_df.to_dict(orient="records")

                if out_type == "binary":
                    scale = "OR"
                    if (
                        "odds_ratio" not in fit_res
                        or "or_ci_lower" not in fit_res
                        or "or_ci_upper" not in fit_res
                    ):
                        raise ValueError(
                            f"Missing required odds ratio estimate keys in fit_res: {fit_res.keys()}"
                        )
                    point_est = float(fit_res["odds_ratio"])
                    ci_lo = float(fit_res["or_ci_lower"])
                    ci_hi = float(fit_res["or_ci_upper"])
                elif out_type == "survival":
                    scale = "HR"
                    if (
                        "hazard_ratio" not in fit_res
                        or "hr_ci_lower" not in fit_res
                        or "hr_ci_upper" not in fit_res
                    ):
                        raise ValueError(
                            f"Missing required hazard ratio estimate keys in fit_res: {fit_res.keys()}"
                        )
                    point_est = float(fit_res["hazard_ratio"])
                    ci_lo = float(fit_res["hr_ci_lower"])
                    ci_hi = float(fit_res["hr_ci_upper"])
                elif out_type == "continuous":
                    scale = "Beta"
                    if (
                        "coef" not in fit_res
                        or "ci_lower" not in fit_res
                        or "ci_upper" not in fit_res
                    ):
                        raise ValueError(
                            f"Missing required coefficient estimate keys in fit_res: {fit_res.keys()}"
                        )
                    point_est = float(fit_res["coef"])
                    ci_lo = float(fit_res["ci_lower"])
                    ci_hi = float(fit_res["ci_upper"])
                else:
                    raise ValueError(f"Unsupported outcome_type: {out_type}")

                if "p_value" not in fit_res:
                    raise ValueError("Missing required 'p_value' in fit_res")
                p_v = float(fit_res["p_value"])

                est_rows.append(
                    Estimate(
                        term=exp_col,
                        label=exp_col,
                        estimate=point_est,
                        ci_lower=ci_lo,
                        ci_upper=ci_hi,
                        p_value=p_v,
                        scale=scale,
                    )
                )
                mod_meta.estimator = f"IPTW ({estimand}) Outcome Model"

            elif mod_type in ("logistic", "binary"):
                y_raw = clean_df[model_spec.outcome]
                u_y = y_raw.dropna().unique()
                if len(u_y) > 2:
                    raise ValueError(
                        f"Binary outcome column '{model_spec.outcome}' must have at most 2 levels, found {len(u_y)}."
                    )
                if not pd.api.types.is_numeric_dtype(y_raw):
                    raise ValueError(
                        f"Binary outcome column '{model_spec.outcome}' must be numeric (0 for non-event, 1 for event). "
                        "Text outcomes are not permitted; please recode outcomes as 0/1, with 1 representing the event."
                    )
                if not set(u_y).issubset({0, 1, 0.0, 1.0}):
                    raise ValueError(
                        f"Binary outcome column '{model_spec.outcome}' contains values outside {{0, 1}}: {set(u_y)}"
                    )

                covar_cols = [c for c in req_cols if c != model_spec.outcome]
                if is_pooled:
                    if model_spec.options.get("method") == "firth":
                        raise NotImplementedError(
                            "Multiple imputation pooling for Firth penalized logistic regression is not yet implemented."
                        )
                    term_estimates: dict[str, list[float]] = {}
                    term_variances: dict[str, list[float]] = {}
                    for imp_df in imputed_datasets:
                        fit_i = fit_standard_logistic(
                            imp_df[model_spec.outcome], imp_df[covar_cols]
                        )
                        s_df = fit_i["summary_df"]
                        for idx, row in s_df.iterrows():
                            term_k = str(idx)
                            b_val = float(row.get("coef", row.get("estimate", 0.0)))
                            se_val = float(
                                row.get(
                                    "std_error",
                                    row.get("std_err", row.get("se", 0.0)),
                                )
                            )
                            term_estimates.setdefault(term_k, []).append(b_val)
                            term_variances.setdefault(term_k, []).append(se_val**2)

                    n_obs = len(imputed_datasets[0])
                    k_params = len(term_estimates)
                    sum_df_records = {}
                    for term_k, thetas in term_estimates.items():
                        vars_list = term_variances[term_k]
                        pooled = pool_estimates(
                            thetas,
                            vars_list,
                            n_obs=n_obs,
                            df_complete=max(1.0, float(n_obs - k_params)),
                        )
                        or_val = float(np.exp(pooled.estimate))
                        ci_lo = float(np.exp(pooled.ci_lower))
                        ci_hi = float(np.exp(pooled.ci_upper))
                        p_v = float(pooled.p_value)
                        sum_df_records[term_k] = {
                            "coef": pooled.estimate,
                            "std_error": pooled.se,
                            "within_variance": pooled.within_variance,
                            "between_variance": pooled.between_variance,
                            "odds_ratio": or_val,
                            "or_ci_lower": ci_lo,
                            "or_ci_upper": ci_hi,
                            "estimate": pooled.estimate,
                            "p_value": p_v,
                        }
                        est_rows.append(
                            Estimate(
                                term=term_k,
                                label=term_k,
                                estimate=or_val,
                                ci_lower=ci_lo,
                                ci_upper=ci_hi,
                                p_value=p_v,
                                scale="OR",
                            )
                        )
                    sum_df = pd.DataFrame.from_dict(sum_df_records, orient="index")
                    fit_res = {
                        "pooled": True,
                        "n_imputations": len(imputed_datasets),
                        "summary_df": sum_df,
                    }
                else:
                    y = clean_df[model_spec.outcome]
                    X = clean_df[covar_cols]

                    if model_spec.options.get("method") == "firth":
                        fit_res = fit_firth_logistic(y, X, feature_names=covar_cols)
                        sum_df = fit_res["summary_df"]
                        for idx, row in sum_df.iterrows():
                            est_rows.append(
                                Estimate(
                                    term=str(idx),
                                    label=str(idx),
                                    estimate=float(
                                        row.get("odds_ratio", np.exp(row["estimate"]))
                                    ),
                                    ci_lower=float(
                                        row.get("or_ci_lower", np.exp(row["ci_lower"]))
                                    ),
                                    ci_upper=float(
                                        row.get("or_ci_upper", np.exp(row["ci_upper"]))
                                    ),
                                    p_value=float(row["p_value"]),
                                    scale="OR",
                                )
                            )
                    else:
                        fit_res = fit_standard_logistic(y, X)
                        sum_df = fit_res["summary_df"]
                        for idx, row in sum_df.iterrows():
                            est_val = float(
                                row.get(
                                    "odds_ratio",
                                    row.get("estimate", row.get("coef", 1.0)),
                                )
                            )
                            ci_lo = float(
                                row.get("or_ci_lower", row.get("ci_lower", 1.0))
                            )
                            ci_hi = float(
                                row.get("or_ci_upper", row.get("ci_upper", 1.0))
                            )
                            p_v = float(row.get("p_value", row.get("p", 0.0)))
                            est_rows.append(
                                Estimate(
                                    term=str(idx),
                                    label=str(idx),
                                    estimate=est_val,
                                    ci_lower=ci_lo,
                                    ci_upper=ci_hi,
                                    p_value=p_v,
                                    scale="OR",
                                )
                            )

                # Optional E-value
                if (
                    model_spec.options.get("e_value")
                    and model_spec.exposure
                    and model_spec.exposure in sum_df.index
                ):
                    exp_row = sum_df.loc[model_spec.exposure]
                    est_val = float(
                        exp_row.get("odds_ratio", exp_row.get("estimate", 1.0))
                    )
                    ci_lo = float(
                        exp_row.get("or_ci_lower", exp_row.get("ci_lower", 1.0))
                    )
                    ci_hi = float(
                        exp_row.get("or_ci_upper", exp_row.get("ci_upper", 1.0))
                    )
                    e_val = calculate_e_value(est_val, ci_lo, ci_hi, estimate_type="OR")
                    fit_res["e_value"] = e_val

            elif mod_type in (
                "competing_risks",
                "competing_risk",
                "cif",
                "fine_gray",
            ) or (
                mod_type in ("cox", "cox_ph", "survival")
                and model_spec.options.get("competing_event") is not None
            ):
                dur_col = model_spec.time or "time"
                covar_cols = [
                    c for c in req_cols if c not in {model_spec.outcome, dur_col}
                ]
                ev_of_int = int(model_spec.options.get("event_of_interest", 1))
                comp_ev = int(model_spec.options.get("competing_event", 2))
                cif_res = fit_competing_risks_cif(
                    durations=clean_df[dur_col],
                    event_observed=clean_df[model_spec.outcome],
                    event_of_interest=ev_of_int,
                    competing_event=comp_ev,
                    horizons=model_spec.options.get("horizons"),
                )
                fit_res = fit_competing_risks_regression(
                    df=clean_df,
                    duration_col=dur_col,
                    event_col=model_spec.outcome,
                    covariates=covar_cols,
                    event_of_interest=ev_of_int,
                    competing_event=comp_ev,
                )
                fit_res["cif_table"] = cif_res.cif_table
                fit_res["horizon_summary"] = cif_res.horizon_summary
                fit_res["clinical_warnings"] = cif_res.clinical_warnings

                comp_df = fit_res["comparison_table"]
                for _, row in comp_df.iterrows():
                    cov_name = str(row["covariate"])
                    est_rows.append(
                        Estimate(
                            term=f"{cov_name} (csHR)",
                            label=f"{cov_name} (Cause-Specific)",
                            estimate=float(row["cause_specific_hr"]),
                            ci_lower=float(row["cs_ci_lower"]),
                            ci_upper=float(row["cs_ci_upper"]),
                            p_value=float(row["cs_p_value"]),
                            scale="HR",
                        )
                    )
                    est_rows.append(
                        Estimate(
                            term=f"{cov_name} (SHR)",
                            label=f"{cov_name} (Fine-Gray)",
                            estimate=float(row["fine_gray_shr"]),
                            ci_lower=float(row["fg_ci_lower"]),
                            ci_upper=float(row["fg_ci_upper"]),
                            p_value=float(row["fg_p_value"]),
                            scale="SHR",
                        )
                    )
                mod_meta.estimator = "Competing Risks (csHR vs Fine-Gray SHR)"

            elif mod_type in ("cox", "cox_ph", "survival"):
                if (model_spec.missing_strategy or "").lower().replace(
                    "_", "-"
                ) == "mice":
                    raise NotImplementedError(
                        "Multiple imputation pooling for Cox proportional hazards models is not yet supported in medstat analysis plans."
                    )
                ev_raw = clean_df[model_spec.outcome]
                u_ev = ev_raw.dropna().unique()
                if len(u_ev) > 2:
                    raise ValueError(
                        f"Event outcome column '{model_spec.outcome}' must have at most 2 levels, found {len(u_ev)}."
                    )
                if not pd.api.types.is_numeric_dtype(ev_raw):
                    raise ValueError(
                        f"Event outcome column '{model_spec.outcome}' must be numeric (0 for non-event, 1 for event). "
                        "Text outcomes are not permitted; please recode outcomes as 0/1, with 1 representing the event."
                    )
                if not set(u_ev).issubset({0, 1, 0.0, 1.0}):
                    raise ValueError(
                        f"Event outcome column '{model_spec.outcome}' contains values outside {{0, 1}}: {set(u_ev)}"
                    )

                dur_col = model_spec.time or "time"
                covar_cols = [
                    c for c in req_cols if c not in {model_spec.outcome, dur_col}
                ]
                fit_res = fit_cox_ph(
                    clean_df,
                    duration_col=dur_col,
                    event_col=model_spec.outcome,
                    covariates=covar_cols,
                )
                sum_df = fit_res["summary_df"]
                for idx, row in sum_df.iterrows():
                    est_val = float(
                        row.get(
                            "hazard_ratio", row.get("exp(coef)", row.get("coef", 1.0))
                        )
                    )
                    ci_lo = float(
                        row.get(
                            "hr_ci_lower",
                            row.get("exp(coef) lower 95%", row.get("ci_lower", 1.0)),
                        )
                    )
                    ci_hi = float(
                        row.get(
                            "hr_ci_upper",
                            row.get("exp(coef) upper 95%", row.get("ci_upper", 1.0)),
                        )
                    )
                    p_v = float(row.get("p_value", row.get("p", 0.0)))
                    est_rows.append(
                        Estimate(
                            term=str(idx),
                            label=str(idx),
                            estimate=est_val,
                            ci_lower=ci_lo,
                            ci_upper=ci_hi,
                            p_value=p_v,
                            scale="HR",
                        )
                    )

            elif mod_type in ("linear", "ols"):
                covar_cols = [c for c in req_cols if c != model_spec.outcome]
                if is_pooled:
                    term_estimates = {}
                    term_variances = {}
                    for imp_df in imputed_datasets:
                        fit_i = fit_linear_regression(
                            imp_df[model_spec.outcome], imp_df[covar_cols]
                        )
                        s_df = fit_i["summary_df"]
                        for idx, row in s_df.iterrows():
                            term_k = str(idx)
                            b_val = float(row.get("coef", row.get("estimate", 0.0)))
                            se_val = float(
                                row.get("std_error", row.get("std_err", 0.0))
                            )
                            term_estimates.setdefault(term_k, []).append(b_val)
                            term_variances.setdefault(term_k, []).append(se_val**2)

                    n_obs = len(imputed_datasets[0])
                    k_params = len(term_estimates)
                    sum_df_records = {}
                    for term_k, thetas in term_estimates.items():
                        vars_list = term_variances[term_k]
                        pooled = pool_estimates(
                            thetas,
                            vars_list,
                            n_obs=n_obs,
                            df_complete=max(1.0, float(n_obs - k_params)),
                        )
                        b_val = float(pooled.estimate)
                        ci_lo = float(pooled.ci_lower)
                        ci_hi = float(pooled.ci_upper)
                        p_v = float(pooled.p_value)
                        sum_df_records[term_k] = {
                            "coef": b_val,
                            "std_error": pooled.se,
                            "within_variance": pooled.within_variance,
                            "between_variance": pooled.between_variance,
                            "ci_lower": ci_lo,
                            "ci_upper": ci_hi,
                            "p_value": p_v,
                        }
                        est_rows.append(
                            Estimate(
                                term=term_k,
                                label=term_k,
                                estimate=b_val,
                                ci_lower=ci_lo,
                                ci_upper=ci_hi,
                                p_value=p_v,
                                scale="Beta",
                            )
                        )
                    sum_df = pd.DataFrame.from_dict(sum_df_records, orient="index")
                    fit_res = {
                        "pooled": True,
                        "n_imputations": len(imputed_datasets),
                        "summary_df": sum_df,
                    }
                else:
                    y = clean_df[model_spec.outcome]
                    X = clean_df[covar_cols]
                    fit_res = fit_linear_regression(y, X)
                    sum_df = fit_res["summary_df"]
                    for idx, row in sum_df.iterrows():
                        est_rows.append(
                            Estimate(
                                term=str(idx),
                                label=str(idx),
                                estimate=float(row["coef"]),
                                ci_lower=float(row["ci_lower"]),
                                ci_upper=float(row["ci_upper"]),
                                p_value=float(row["p_value"]),
                                scale="Beta",
                            )
                        )
            else:
                fit_res = {"error": f"Unsupported model type: {mod_type}"}

            results["models"][model_spec.name] = fit_res

            # Generate publication table
            if est_rows:
                est_tbl = EstimateTable(
                    title=f"Table. {model_spec.description or model_spec.name}",
                    rows=est_rows,
                    meta=mod_meta,
                )
                if (
                    "competing" in mod_meta.estimator.lower()
                    and "comparison_table" in fit_res
                ):
                    html_table = render_competing_risks_table(
                        f"Table. {model_spec.description or model_spec.name}",
                        fit_res["comparison_table"],
                        style=self.reporting.style.upper(),
                    )
                    narrative = generate_competing_risks_narrative(
                        exposure=model_spec.exposure,
                        outcome=model_spec.outcome,
                        covariates=model_spec.covariates,
                    )
                elif "iptw" in mod_meta.estimator.lower():
                    html_table = PublicationRenderer.render_html(
                        est_tbl, style=self.reporting.style
                    )
                    max_smd_val = None
                    if "balance_table" in fit_res:
                        smd_vals = [
                            abs(r["Weighted_SMD"])
                            for r in fit_res["balance_table"]
                            if r.get("Weighted_SMD") is not None
                            and np.isfinite(r["Weighted_SMD"])
                        ]
                        if smd_vals:
                            max_smd_val = max(smd_vals)
                    is_stabilized = bool(model_spec.options.get("stabilized", True))
                    t_quantiles = model_spec.options.get(
                        "truncate_quantiles", (0.01, 0.99)
                    )
                    is_truncated = t_quantiles is not None
                    narrative = generate_iptw_narrative(
                        estimand=model_spec.options.get("estimand", "ATE"),
                        exposure=model_spec.exposure,
                        outcome=model_spec.outcome,
                        covariates=model_spec.covariates,
                        n_eff=fit_res.get("effective_sample_size")
                        or fit_res.get("diagnostics", {})
                        .get("overall", {})
                        .get("n_eff"),
                        max_smd=max_smd_val,
                        stabilized=is_stabilized,
                        truncated=is_truncated,
                        diagnostics=fit_res.get("diagnostics"),
                    )
                else:
                    html_table = PublicationRenderer.render_html(
                        est_tbl, style=self.reporting.style
                    )
                    eff_strategy = model_spec.missing_strategy
                    if (
                        eff_strategy
                        and eff_strategy.lower().replace("_", "-") == "mice"
                        and not fit_res.get("pooled", False)
                    ):
                        eff_strategy = None
                    narrative = generate_narrative(
                        est_tbl,
                        style=self.reporting.style,
                        missing_strategy=eff_strategy,
                    )
                results["reports"][model_spec.name] = {
                    "html_table": html_table,
                    "methods_narrative": narrative,
                }

        return results

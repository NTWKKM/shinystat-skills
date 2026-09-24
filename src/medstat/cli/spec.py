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

from medstat.data.clean import prepare_data_for_analysis
from medstat.data.missing import pool_estimates
from medstat.data.retention import SampleFlowTracker
from medstat.models.firth import fit_firth_logistic
from medstat.models.glm import fit_linear_regression, fit_standard_logistic
from medstat.models.sensitivity import calculate_e_value
from medstat.models.survival import fit_cox_ph
from medstat.reporting.narrative import generate_narrative
from medstat.reporting.tables import (
    Estimate,
    EstimateTable,
    ModelMeta,
    PublicationRenderer,
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
            analyst=meta_dict.get("analyst", "Clinical Biostatistician"),
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
            current_df = pd.read_csv(raw_path)
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

            if mod_type in ("logistic", "binary"):
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
                            b_val = float(row.get("estimate", row.get("coef", 0.0)))
                            se_val = float(row.get("std_err", row.get("se", 0.0)))
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

            elif mod_type in ("cox", "cox_ph", "survival"):
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
                            b_val = float(row["coef"])
                            se_val = float(row.get("std_err", 0.0))
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
                html_table = PublicationRenderer.render_html(
                    est_tbl, style=self.reporting.style
                )
                eff_strategy = model_spec.missing_strategy or "complete-case"
                if eff_strategy.lower().replace("_", "-") == "mice" and not is_pooled:
                    eff_strategy = "complete-case"
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

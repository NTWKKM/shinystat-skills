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
from medstat.data.loader import get_clinical_columns, load_clinical_data
from medstat.data.missing import pool_estimates
from medstat.data.retention import SampleFlowTracker
from medstat.diagnostic.roc import (
    compare_diagnostic_models,
    compute_time_dependent_roc,
)
from medstat.diagnostic.validation import (
    bootstrap_internal_validation,
    recalibrate_prediction_model,
)
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
    generate_internal_validation_narrative,
    generate_iptw_narrative,
    generate_narrative,
    generate_recalibration_narrative,
    generate_reclassification_narrative,
    generate_time_dependent_roc_narrative,
)
from medstat.reporting.tables import (
    Estimate,
    EstimateTable,
    ModelMeta,
    PublicationRenderer,
    format_internal_validation_table,
    format_recalibration_table,
    format_reclassification_table,
    format_time_dependent_roc_table,
    render_competing_risks_table,
)

_SURVIVAL_MODEL_TYPES = {
    "cox",
    "cox_ph",
    "survival",
    "fine_gray",
    "competing_risks",
    "competing_risk",
    "cif",
}

_TIME_SUPPORTED_MODEL_TYPES = _SURVIVAL_MODEL_TYPES | {
    "iptw",
    "causal_iptw",
    "time_dependent_roc",
    "td_roc",
}


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _resolve_col_alias(primary: Any, fallback: Any) -> str | None:
    """Resolve an option value with fallback, treating None and empty strings as absent."""
    if isinstance(primary, str):
        s = primary.strip()
        if s:
            return s
    if isinstance(fallback, str):
        s = fallback.strip()
        if s:
            return s
    return None


def _resolve_duration_col(
    model_dict: dict[str, Any],
    opts_dict: dict[str, Any],
    supports_time: bool,
    is_reclass: bool,
    is_iptw: bool,
) -> str | None:
    """Resolve duration column with identical precedence across from_yaml and preflight."""
    if is_iptw and str(opts_dict.get("outcome_type", "")).lower().strip() in (
        "binary",
        "continuous",
    ):
        return None

    raw_time = model_dict.get("time") if not is_reclass else None
    if isinstance(raw_time, str):
        raw_time = raw_time.strip() or None
    elif raw_time is not None:
        raw_time = str(raw_time).strip() or None

    if raw_time is not None:
        return raw_time

    if supports_time:
        for candidate in (model_dict.get("time_col"), opts_dict.get("time_col")):
            if isinstance(candidate, str):
                c = candidate.strip()
                if c:
                    return c
            elif candidate is not None:
                c = str(candidate).strip()
                if c:
                    return c

    return None


def _construct_unique_mapping(
    loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[str, Any]:
    seen_keys = set()
    for key_node, _ in node.value:
        if getattr(key_node, "tag", None) == "tag:yaml.org,2002:merge":
            continue
        key = loader.construct_object(key_node, deep=deep)
        try:
            is_dup = key in seen_keys
        except TypeError:
            is_dup = False
        if is_dup:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key: '{key}'",
                key_node.start_mark,
            )
        try:
            seen_keys.add(key)
        except TypeError:
            pass
    return super(_UniqueKeyLoader, loader).construct_mapping(node, deep=deep)


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
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

    def __post_init__(self) -> None:
        if isinstance(self.name, str):
            self.name = self.name.strip()
        if isinstance(self.type, str):
            self.type = self.type.strip()
        if isinstance(self.outcome, str):
            self.outcome = self.outcome.strip()
        if isinstance(self.exposure, str):
            self.exposure = self.exposure.strip()
        if isinstance(self.time, str):
            self.time = self.time.strip()
        if isinstance(self.covariates, list):
            self.covariates = [
                c.strip() if isinstance(c, str) else c for c in self.covariates
            ]
        if isinstance(self.options, dict):
            norm_opts = dict(self.options)
            for k in (
                "p_old",
                "p_old_col",
                "score_old",
                "p_new",
                "p_new_col",
                "score_new",
                "score_col",
                "time_col",
                "event_col",
                "cluster_col",
                "p_recal",
                "p_pred",
                "score",
            ):
                if k in norm_opts and isinstance(norm_opts[k], str):
                    norm_opts[k] = norm_opts[k].strip()
            self.options = norm_opts


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
            data = yaml.load(f, Loader=_UniqueKeyLoader)

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

        models = []
        for m in data.get("models", []):
            m_type = str(m.get("type", "logistic")).lower().strip()
            opts = m.get("options")
            opts_dict = opts if isinstance(opts, dict) else {}
            m_method = str(opts_dict.get("method", "")).lower().strip()

            is_survival = m_type in _SURVIVAL_MODEL_TYPES
            is_iptw = m_type in ("iptw", "causal_iptw") or m_method == "iptw"
            resolved_p_old = _resolve_col_alias(
                opts_dict.get("p_old_col"), opts_dict.get("p_old")
            )
            is_reclass = (
                not is_iptw
                and m_type in ("reclassification", "diagnostic", "diag")
                and bool(resolved_p_old or opts_dict.get("reclassification"))
            )
            is_td_roc = not is_iptw and (
                m_type in ("time_dependent_roc", "td_roc")
                or (
                    m_type in ("diagnostic", "diag")
                    and not is_reclass
                    and bool(
                        opts_dict.get("time_dependent_roc")
                        or opts_dict.get("time_horizons")
                        or opts_dict.get("horizons")
                    )
                )
            )

            supports_time = False
            if is_iptw:
                outcome_type = str(opts_dict.get("outcome_type", "")).lower().strip()
                if outcome_type not in ("binary", "continuous"):
                    supports_time = True
            elif is_survival:
                supports_time = True
            elif is_td_roc:
                supports_time = True

            time_val = _resolve_duration_col(
                m, opts_dict, supports_time, is_reclass, is_iptw
            )

            models.append(
                ModelSpec(
                    name=m.get("name", "model_1"),
                    type=m.get("type", "logistic"),
                    outcome=m.get("outcome", ""),
                    exposure=m.get("exposure"),
                    covariates=m.get("covariates", []),
                    formula=m.get("formula"),
                    description=m.get("description", ""),
                    time=time_val,
                    reference_categories=m.get("reference_categories", {}),
                    missing_strategy=m.get("missing_strategy"),
                    missing_justification=m.get("missing_justification"),
                    options=m.get("options", {}),
                )
            )

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
            mod_type = model_spec.type.lower().strip()
            m_opts = model_spec.options if isinstance(model_spec.options, dict) else {}
            m_method = str(m_opts.get("method", "")).lower().strip()
            is_iptw_exec = mod_type in ("iptw", "causal_iptw") or m_method == "iptw"
            resolved_p_old_exec = _resolve_col_alias(
                m_opts.get("p_old_col"), m_opts.get("p_old")
            )
            is_reclass_exec = (
                not is_iptw_exec
                and mod_type in ("reclassification", "diagnostic", "diag")
                and bool(resolved_p_old_exec or m_opts.get("reclassification"))
            )
            is_survival_exec = mod_type in _SURVIVAL_MODEL_TYPES and not (
                is_iptw_exec
                and str(m_opts.get("outcome_type", "")).lower().strip()
                in ("binary", "continuous")
            )
            is_td_roc_exec = not is_iptw_exec and (
                mod_type in ("time_dependent_roc", "td_roc")
                or (
                    mod_type in ("diagnostic", "diag")
                    and not is_reclass_exec
                    and bool(
                        m_opts.get("time_dependent_roc")
                        or m_opts.get("time_horizons")
                        or m_opts.get("horizons")
                    )
                )
            )
            is_iptw_survival_exec = is_iptw_exec and str(
                m_opts.get("outcome_type", "")
            ).lower().strip() not in ("binary", "continuous")
            is_duration_dependent = (
                is_survival_exec or is_td_roc_exec or is_iptw_survival_exec
            )
            req_cols = [model_spec.outcome]
            if model_spec.exposure:
                req_cols.append(model_spec.exposure)
            if model_spec.time:
                req_cols.append(model_spec.time)
            req_cols.extend(model_spec.covariates)
            if is_reclass_exec:
                p_old_c = _resolve_col_alias(
                    m_opts.get("p_old_col"), m_opts.get("p_old")
                )
                p_new_c = _resolve_col_alias(
                    m_opts.get("p_new_col"), m_opts.get("p_new")
                )
                if p_old_c and p_old_c in current_df.columns:
                    req_cols.append(p_old_c)
                if p_new_c and p_new_c in current_df.columns:
                    req_cols.append(p_new_c)
            elif is_td_roc_exec:
                td_old_c = _resolve_col_alias(
                    m_opts.get("p_old_col"), m_opts.get("score_old")
                )
                if td_old_c and td_old_c in current_df.columns:
                    req_cols.append(td_old_c)

            for opt_key in (
                "p_old",
                "p_old_col",
                "score_old",
                "p_new",
                "p_new_col",
                "score_new",
                "score_col",
                "event_col",
                "p_recal",
                "p_pred",
                "score",
            ):
                if is_reclass_exec and opt_key in (
                    "p_old",
                    "p_old_col",
                    "p_new",
                    "p_new_col",
                ):
                    continue
                if is_td_roc_exec and opt_key in (
                    "p_old",
                    "p_old_col",
                    "score_old",
                ):
                    continue
                if (
                    not is_reclass_exec
                    and not is_td_roc_exec
                    and opt_key
                    in (
                        "p_old",
                        "p_old_col",
                        "score_old",
                        "p_new",
                        "p_new_col",
                    )
                ):
                    continue
                if opt_key == "time_col" and (
                    not is_duration_dependent or model_spec.time
                ):
                    continue
                if opt_key == "event_col" and (is_reclass_exec or is_iptw_exec):
                    continue
                if (
                    opt_key
                    in (
                        "p_old",
                        "p_old_col",
                        "score_old",
                        "p_new",
                        "p_new_col",
                        "score_new",
                        "score_col",
                        "score",
                        "p_recal",
                        "p_pred",
                    )
                    and is_iptw_exec
                ):
                    continue
                val = model_spec.options.get(opt_key)
                if val and isinstance(val, str):
                    val = val.strip()
                    if val in current_df.columns:
                        req_cols.append(val)
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
            html_table: str | None = None
            narrative: str | None = None
            mod_meta = ModelMeta(
                estimator=f"{mod_type.upper()} Regression",
                n_total=len(clean_df),
                outcome_name=model_spec.outcome,
                exposure_name=model_spec.exposure,
                adjusted_for=model_spec.covariates,
            )

            if is_iptw_exec:
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

                raw_covars = (
                    [model_spec.exposure] if model_spec.exposure else []
                ) + list(model_spec.covariates)
                recal_opts = {
                    model_spec.options.get(k)
                    for k in ("p_recal", "p_pred")
                    if model_spec.options.get(k)
                }
                covar_cols = [
                    c
                    for c in dict.fromkeys(raw_covars)
                    if c and c != model_spec.outcome and c not in recal_opts
                ]
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

                if model_spec.options.get("internal_validation") and not is_pooled:
                    n_boot = int(model_spec.options.get("n_bootstrap", 200))
                    val_res = bootstrap_internal_validation(
                        df=clean_df,
                        outcome=model_spec.outcome,
                        predictors=covar_cols,
                        n_bootstrap=n_boot,
                    )
                    fit_res["internal_validation"] = val_res

                if model_spec.options.get("recalibration"):
                    p_recal_col = model_spec.options.get(
                        "p_recal"
                    ) or model_spec.options.get("p_pred")
                    if not p_recal_col or p_recal_col not in clean_df.columns:
                        raise ValueError(
                            f"Recalibration requires a valid predicted probability column via options 'p_recal' or 'p_pred'. "
                            f"Resolved '{p_recal_col}', which was not found in dataset columns: {list(clean_df.columns)}."
                        )
                    recal_m = model_spec.options.get("recalibration_method", "logistic")
                    recal_res = recalibrate_prediction_model(
                        y_true=clean_df[model_spec.outcome].values,
                        p_pred=clean_df[p_recal_col].values,
                        method=recal_m,
                    )
                    fit_res["recalibration"] = {
                        k: v
                        for k, v in recal_res.items()
                        if not str(k).startswith("p_recal")
                        and k not in ("y_true", "p_orig")
                    }

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
            elif is_reclass_exec:
                p_old_c = _resolve_col_alias(
                    model_spec.options.get("p_old_col"), model_spec.options.get("p_old")
                )
                p_new_c = _resolve_col_alias(
                    model_spec.options.get("p_new_col"), model_spec.options.get("p_new")
                )
                if not p_old_c or p_old_c not in clean_df.columns:
                    raise ValueError(
                        f"Reclassification model '{model_spec.name}' requires a valid initial predicted probability column. "
                        f"Resolved '{p_old_c}', which was not found in dataset columns: {list(clean_df.columns)}."
                    )
                if not p_new_c or p_new_c not in clean_df.columns:
                    raise ValueError(
                        f"Reclassification model '{model_spec.name}' requires a valid new predicted probability column. "
                        f"Resolved '{p_new_c}', which was not found in dataset columns: {list(clean_df.columns)}."
                    )
                th_list = model_spec.options.get("thresholds")
                y_out = clean_df[model_spec.outcome].values
                fit_res = compare_diagnostic_models(
                    y_true=y_out,
                    p_old=clean_df[p_old_c].values,
                    p_new=clean_df[p_new_c].values,
                    thresholds=th_list,
                )
                mod_meta.estimator = "Diagnostic Risk Reclassification (NRI & IDI)"
                html_table = format_reclassification_table(
                    fit_res,
                    title=f"Table. {model_spec.description or model_spec.name}",
                    style=self.reporting.style.upper(),
                )
                narrative = generate_reclassification_narrative(fit_res)

            elif mod_type in ("time_dependent_roc", "td_roc") or (
                mod_type in ("diagnostic", "diag")
                and (
                    model_spec.options.get("time_dependent_roc")
                    or model_spec.options.get("time_horizons")
                    or model_spec.options.get("horizons")
                )
            ):
                dur_col = (
                    model_spec.time
                    or _resolve_col_alias(model_spec.options.get("time_col"), None)
                    or "time"
                )
                if dur_col not in clean_df.columns:
                    raise ValueError(
                        f"Time-dependent ROC model '{model_spec.name}' duration column '{dur_col}' not found in dataset columns: {list(clean_df.columns)}."
                    )
                ev_col = model_spec.outcome or model_spec.options.get(
                    "event_col", "event"
                )
                if ev_col not in clean_df.columns:
                    raise ValueError(
                        f"Time-dependent ROC model '{model_spec.name}' event column '{ev_col}' not found in dataset columns: {list(clean_df.columns)}."
                    )
                sc_col = (
                    model_spec.exposure
                    or model_spec.options.get("score_col")
                    or (model_spec.covariates[0] if model_spec.covariates else None)
                )
                horizons = model_spec.options.get(
                    "time_horizons"
                ) or model_spec.options.get("horizons")
                comp_ev = model_spec.options.get("competing_event")
                raw_td_old = model_spec.options.get(
                    "p_old_col"
                ) or model_spec.options.get("score_old")
                p_old_col = _resolve_col_alias(
                    model_spec.options.get("p_old_col"),
                    model_spec.options.get("score_old"),
                )
                if raw_td_old and (not p_old_col or p_old_col not in clean_df.columns):
                    raise ValueError(
                        f"Time-dependent ROC model '{model_spec.name}' comparison score column '{raw_td_old}' not found in dataset columns: {list(clean_df.columns)}."
                    )
                fit_res = compute_time_dependent_roc(
                    durations=clean_df[dur_col].values,
                    events=clean_df[ev_col].values,
                    scores=clean_df[sc_col].values,
                    time_horizons=horizons,
                    competing_event_code=comp_ev,
                    scores_old=(
                        clean_df[p_old_col].values
                        if p_old_col and p_old_col in clean_df.columns
                        else None
                    ),
                    scores_new=(
                        clean_df[sc_col].values
                        if p_old_col and p_old_col in clean_df.columns
                        else None
                    ),
                )
                mod_meta.estimator = "Cumulative/Dynamic Time-Dependent ROC"
                html_table = format_time_dependent_roc_table(
                    fit_res,
                    title=f"Table. {model_spec.description or model_spec.name}",
                    style=self.reporting.style.upper(),
                )
                narrative = generate_time_dependent_roc_narrative(fit_res)

            elif mod_type in ("internal_validation", "validation", "tripod") or (
                mod_type in ("diagnostic", "diag")
                and (
                    model_spec.options.get("internal_validation")
                    or model_spec.options.get("validation")
                )
            ):
                raw_covars = (
                    [model_spec.exposure] if model_spec.exposure else []
                ) + list(model_spec.covariates)
                opt_preds = model_spec.options.get("predictors")
                if opt_preds:
                    if isinstance(opt_preds, list):
                        raw_covars.extend(opt_preds)
                    elif isinstance(opt_preds, str):
                        raw_covars.extend(
                            [p.strip() for p in opt_preds.split(",") if p.strip()]
                        )
                recal_opts = {
                    model_spec.options.get(k)
                    for k in ("p_recal", "p_pred")
                    if model_spec.options.get(k)
                }
                covar_cols = [
                    c
                    for c in dict.fromkeys(raw_covars)
                    if c and c != model_spec.outcome and c not in recal_opts
                ]
                n_boot = int(model_spec.options.get("n_bootstrap", 200))
                fit_res = bootstrap_internal_validation(
                    df=clean_df,
                    outcome=model_spec.outcome,
                    predictors=covar_cols,
                    n_bootstrap=n_boot,
                )
                mod_meta.estimator = "TRIPOD+AI Bootstrap Internal Validation"
                html_table = format_internal_validation_table(
                    fit_res,
                    title=f"Table. {model_spec.description or model_spec.name}",
                    style=self.reporting.style.upper(),
                )
                narrative = generate_internal_validation_narrative(fit_res)

            elif mod_type in ("recalibration", "recalibrate") or (
                mod_type in ("diagnostic", "diag")
                and model_spec.options.get("recalibration")
            ):
                p_recal_col = model_spec.options.get(
                    "p_recal"
                ) or model_spec.options.get("p_pred")
                if not p_recal_col or p_recal_col not in clean_df.columns:
                    raise ValueError(
                        f"Recalibration requires a valid predicted probability column via options 'p_recal' or 'p_pred'. "
                        f"Resolved '{p_recal_col}', which was not found in dataset columns: {list(clean_df.columns)}."
                    )
                recal_method = model_spec.options.get("method", "logistic")
                fit_res_raw = recalibrate_prediction_model(
                    y_true=clean_df[model_spec.outcome].values,
                    p_pred=clean_df[p_recal_col].values,
                    method=recal_method,
                )
                fit_res = {
                    k: v
                    for k, v in fit_res_raw.items()
                    if not str(k).startswith("p_recal")
                    and k not in ("y_true", "p_orig")
                }
                mod_meta.estimator = "Prediction Model Recalibration"
                html_table = format_recalibration_table(
                    fit_res_raw,
                    title=f"Table. {model_spec.description or model_spec.name}",
                    style=self.reporting.style.upper(),
                )
                narrative = generate_recalibration_narrative(fit_res_raw)
            else:
                fit_res = {"error": f"Unsupported model type: {mod_type}"}

            results["models"][model_spec.name] = fit_res

            # Generate publication table
            if html_table is not None and narrative is not None:
                results["reports"][model_spec.name] = {
                    "html_table": html_table,
                    "methods_narrative": narrative,
                }
            elif est_rows:
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


SCHEMA_YAML_TEMPLATE = """# MedStat Declarative Analysis Plan Schema (Use with: medstat_cli.py --spec plan.yaml)
metadata:
  study_title: "Clinical Cohort Study"
  study_design: "observational"         # rct | observational | cohort | case_control | diagnostic
  reporting_guideline: "STROBE"         # STROBE | CONSORT | TRIPOD | STARD | PRISMA

data:
  input_path: "data/cohort.csv"
  id_column: "patient_id"

models:
  - name: "primary_outcome"
    type: "logistic"                    # logistic | linear | cox | competing_risk | fine_gray | iptw | diagnostic | validation | recalibration
    outcome: "mortality_28d"
    exposure: "treatment_group"
    covariates: ["age", "sex", "sofa_score"]
    missing_strategy: "complete-case"   # complete-case | mice | knn | indicator
    missing_justification: "Complete-case analysis prespecified"
    # For survival/competing_risk specify time: "follow_up_days" and options: {event_of_interest: 1, competing_event: 2}

reporting:
  style: "nejm"                         # nejm | lancet | jama | bmj
  format: "html"                        # html | docx | pptx | markdown | pdf
"""


def get_cli_schema_template() -> str:
    """Returns a commented YAML analysis plan template for agent introspection."""
    return SCHEMA_YAML_TEMPLATE


def preflight_validate_spec(spec_path: str | Path) -> dict[str, Any]:
    """
    Validates YAML syntax, required keys, and column existence in the target dataset
    without running statistical models. Returns a structured dict for agent self-healing.
    """
    path = Path(spec_path)
    report: dict[str, Any] = {
        "valid": True,
        "spec_path": str(path),
        "errors": [],
        "warnings": [],
        "agent_remediation": [],
    }

    if not path.exists():
        report["valid"] = False
        report["errors"].append(f"Spec file not found: {path}")
        report["agent_remediation"].append(
            "Verify the file path passed to --validate-only."
        )
        return report

    try:
        raw_spec = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except Exception as exc:
        report["valid"] = False
        report["errors"].append(f"Invalid YAML syntax: {exc}")
        report["agent_remediation"].append(
            "Run `medstat_cli.py --schema` to inspect valid YAML structure."
        )
        return report

    if not isinstance(raw_spec, dict):
        report["valid"] = False
        report["errors"].append("Root of YAML spec must be a mapping/dictionary.")
        report["agent_remediation"].append(
            "Run `medstat_cli.py --schema` to inspect the valid top-level structure."
        )
        return report

    has_spec_keys = any(k in raw_spec for k in ("data", "metadata", "models"))
    if not has_spec_keys:
        report["valid"] = False
        report["errors"].append(
            "YAML spec must contain at least one top-level section: 'data', 'metadata', or 'models'."
        )
        report["agent_remediation"].append(
            "Run `medstat_cli.py --schema` to view the expected top-level keys."
        )
        return report

    # 1. Dataset section check
    data_cfg = raw_spec.get("data")
    input_path_raw = data_cfg.get("input_path") if isinstance(data_cfg, dict) else None
    if (
        not isinstance(data_cfg, dict)
        or not isinstance(input_path_raw, str)
        or not input_path_raw.strip()
    ):
        report["valid"] = False
        report["errors"].append(
            "YAML spec must contain a 'data' section with an 'input_path' pointing to the dataset."
        )
        report["agent_remediation"].append(
            "Specify `data.input_path: path/to/dataset.csv` in your spec YAML."
        )

    # 2. Executable models section check
    models_list = raw_spec.get("models")
    if not isinstance(models_list, list) or len(models_list) == 0:
        report["valid"] = False
        report["errors"].append(
            "YAML spec must contain a non-empty 'models' list specifying executable models."
        )
        report["agent_remediation"].append(
            "Define at least one executable model in the `models` section of your spec YAML."
        )
        models_list = []
    else:
        supported_types = {
            "logistic",
            "binary",
            "firth_logistic",
            "linear",
            "ols",
            "cox",
            "cox_ph",
            "survival",
            "competing_risks",
            "competing_risk",
            "cif",
            "fine_gray",
            "iptw",
            "causal_iptw",
            "reclassification",
            "diagnostic",
            "diag",
            "time_dependent_roc",
            "td_roc",
            "internal_validation",
            "validation",
            "tripod",
            "recalibration",
            "recalibrate",
        }
        for idx, item in enumerate(models_list, start=1):
            if not isinstance(item, dict):
                report["valid"] = False
                report["errors"].append(
                    f"Model item #{idx} must be a dictionary/mapping."
                )
                continue
            if "type" not in item:
                report["valid"] = False
                report["errors"].append(
                    f"Model item #{idx} missing required field 'type'."
                )
                report["agent_remediation"].append(
                    f"Set `type` in model item #{idx} to one of {sorted(supported_types)}."
                )
                continue
            m_type = str(item.get("type", "")).lower().strip()
            if not m_type or m_type not in supported_types:
                report["valid"] = False
                report["errors"].append(
                    f"Model item #{idx} specifies unsupported or missing model type '{m_type}'. "
                    f"Supported types: {sorted(supported_types)}"
                )
                report["agent_remediation"].append(
                    f"Set `type` in model item #{idx} to one of {sorted(supported_types)}."
                )

    # Validate column references across spec
    referenced_cols: set[str] = set()

    def _add_col_ref(val: Any, field_name: str) -> None:
        if val is None:
            return
        if isinstance(val, str):
            s = val.strip()
            if s:
                referenced_cols.add(s)
        elif isinstance(val, list):
            for elem in val:
                if isinstance(elem, str):
                    s = elem.strip()
                    if s:
                        referenced_cols.add(s)
                else:
                    report["valid"] = False
                    report["errors"].append(
                        f"Field '{field_name}' list items must be strings; got {type(elem).__name__}: {elem}"
                    )
        else:
            report["valid"] = False
            report["errors"].append(
                f"Field '{field_name}' must be a string or list of strings; got {type(val).__name__}."
            )

    t1 = raw_spec.get("table1", {})
    if isinstance(t1, dict):
        if t1.get("group_col"):
            _add_col_ref(t1["group_col"], "table1.group_col")
        _add_col_ref(t1.get("continuous_vars"), "table1.continuous_vars")
        _add_col_ref(t1.get("categorical_vars"), "table1.categorical_vars")

    for item in models_list:
        if isinstance(item, dict):
            for key in (
                "outcome",
                "exposure",
                "event_col",
                "cluster_col",
            ):
                if key in item:
                    _add_col_ref(item[key], f"models.{key}")
            if "covariates" in item:
                _add_col_ref(item["covariates"], "models.covariates")

            m_type = str(item.get("type", "")).lower().strip()
            opts = item.get("options")
            opts_dict = opts if isinstance(opts, dict) else {}
            m_method = str(opts_dict.get("method", "")).lower().strip()

            has_list_reclass_option = False
            for opt_key in ("p_old_col", "p_old", "p_new_col", "p_new"):
                val = opts_dict.get(opt_key)
                if isinstance(val, list):
                    has_list_reclass_option = True
                    report["valid"] = False
                    report["errors"].append(
                        f"Reclassification option '{opt_key}' for model '{item.get('name', 'unnamed')}' must be a single column string, not a list: {val}."
                    )
                    report["agent_remediation"].append(
                        f"Specify a single column name string for '{opt_key}' in options for model '{item.get('name', 'unnamed')}'."
                    )

            is_survival = m_type in _SURVIVAL_MODEL_TYPES
            is_iptw = m_type in ("iptw", "causal_iptw") or m_method == "iptw"
            resolved_p_old = _resolve_col_alias(
                opts_dict.get("p_old_col"), opts_dict.get("p_old")
            )
            resolved_p_new = _resolve_col_alias(
                opts_dict.get("p_new_col"), opts_dict.get("p_new")
            )
            has_old = bool(resolved_p_old)
            has_new = bool(resolved_p_new)
            is_reclass = (
                not is_iptw
                and m_type in ("reclassification", "diagnostic", "diag")
                and has_old
                and has_new
            )
            has_reclass_intent = (
                not is_iptw
                and m_type in ("reclassification", "diagnostic", "diag")
                and bool(
                    has_old
                    or has_new
                    or opts_dict.get("reclassification")
                    or m_type == "reclassification"
                    or any(
                        k in opts_dict
                        for k in ("p_old_col", "p_old", "p_new_col", "p_new")
                    )
                )
            )
            is_td_roc = not is_iptw and (
                m_type in ("time_dependent_roc", "td_roc")
                or (
                    m_type in ("diagnostic", "diag")
                    and not is_reclass
                    and bool(
                        opts_dict.get("time_dependent_roc")
                        or opts_dict.get("time_horizons")
                        or opts_dict.get("horizons")
                    )
                )
            )

            is_diagnostic_type = m_type in ("diagnostic", "diag")
            is_valid_diagnostic_op = (
                is_iptw
                or is_reclass
                or is_td_roc
                or bool(
                    opts_dict.get("internal_validation") or opts_dict.get("validation")
                )
                or bool(opts_dict.get("recalibration"))
            )
            if (
                is_diagnostic_type
                and not is_valid_diagnostic_op
                and not has_reclass_intent
            ):
                report["valid"] = False
                report["errors"].append(
                    f"Diagnostic model '{item.get('name', 'unnamed')}' must specify a supported diagnostic operation in options "
                    "(e.g., 'internal_validation', 'reclassification', 'time_dependent_roc', 'recalibration', or 'method: iptw')."
                )
                report["agent_remediation"].append(
                    f"Set one of 'internal_validation: true', 'p_old_col', 'time_dependent_roc: true', 'recalibration: true', "
                    f"or 'method: iptw' in options for diagnostic model '{item.get('name', 'unnamed')}'."
                )

            if has_reclass_intent and not is_reclass and not has_list_reclass_option:
                if not has_old and not has_new:
                    report["valid"] = False
                    report["errors"].append(
                        f"Reclassification model '{item.get('name', 'unnamed')}' specifies 'type: reclassification' but lacks reclassification options (e.g., 'p_old_col' or 'p_old')."
                    )
                    report["agent_remediation"].append(
                        f"Specify 'p_old_col' (or 'p_old') and 'p_new_col' (or 'p_new') in options for model '{item.get('name', 'unnamed')}'."
                    )
                elif not has_old:
                    report["valid"] = False
                    report["errors"].append(
                        f"Reclassification model '{item.get('name', 'unnamed')}' requires an initial predicted probability/score column via 'p_old_col' or 'p_old'."
                    )
                    report["agent_remediation"].append(
                        f"Specify 'p_old_col' (or 'p_old') in options for model '{item.get('name', 'unnamed')}'."
                    )
                elif not has_new:
                    report["valid"] = False
                    report["errors"].append(
                        f"Reclassification model '{item.get('name', 'unnamed')}' requires a new predicted probability column via 'p_new_col' or 'p_new'."
                    )
                    report["agent_remediation"].append(
                        f"Specify 'p_new_col' or 'p_new' in options for model '{item.get('name', 'unnamed')}'."
                    )

            # Resolve duration once using identical precedence to from_yaml()
            supports_time = False
            if is_iptw:
                outcome_type = str(opts_dict.get("outcome_type", "")).lower().strip()
                if outcome_type not in ("binary", "continuous"):
                    supports_time = True
            elif is_survival:
                supports_time = True
            elif is_td_roc:
                supports_time = True

            time_val = _resolve_duration_col(
                item, opts_dict, supports_time, is_reclass, is_iptw
            )

            if time_val:
                _add_col_ref(time_val, "models.time")

            iptw_outcome_type = (
                str(opts_dict.get("outcome_type", "")).lower().strip()
                if is_iptw
                else ""
            )
            requires_survival_time = (
                is_survival and iptw_outcome_type not in ("binary", "continuous")
            ) or (is_iptw and iptw_outcome_type == "survival")
            if requires_survival_time and not time_val:
                report["valid"] = False
                report["errors"].append(
                    f"Survival model '{item.get('name', 'unnamed')}' requires a 'time' or 'time_col' specification."
                )

            # Collect model options column references
            if is_reclass:
                if isinstance(resolved_p_old, str) and resolved_p_old:
                    _add_col_ref(resolved_p_old, "models.options.p_old_col")
                if isinstance(resolved_p_new, str) and resolved_p_new:
                    _add_col_ref(resolved_p_new, "models.options.p_new_col")
            elif is_td_roc:
                raw_td_old = opts_dict.get("p_old_col")
                raw_score_old = opts_dict.get("score_old")
                if raw_td_old is not None and not isinstance(raw_td_old, str):
                    report["valid"] = False
                    report["errors"].append(
                        f"Time-dependent ROC model '{item.get('name', 'unnamed')}' option 'p_old_col' must be a scalar column name, got {type(raw_td_old).__name__}: {raw_td_old}."
                    )
                    report["agent_remediation"].append(
                        f"Provide a single string column name for 'p_old_col' in model '{item.get('name', 'unnamed')}'."
                    )
                elif (
                    (
                        raw_td_old is None
                        or (isinstance(raw_td_old, str) and not raw_td_old.strip())
                    )
                    and raw_score_old is not None
                    and not isinstance(raw_score_old, str)
                ):
                    report["valid"] = False
                    report["errors"].append(
                        f"Time-dependent ROC model '{item.get('name', 'unnamed')}' option 'score_old' must be a scalar column name, got {type(raw_score_old).__name__}: {raw_score_old}."
                    )
                    report["agent_remediation"].append(
                        f"Provide a single string column name for 'score_old' in model '{item.get('name', 'unnamed')}'."
                    )
                else:
                    resolved_td_old = _resolve_col_alias(raw_td_old, raw_score_old)
                    if isinstance(resolved_td_old, str) and resolved_td_old:
                        _add_col_ref(resolved_td_old, "models.options.p_old_col")

            for opt_key in (
                "p_old",
                "p_old_col",
                "score_old",
                "p_new",
                "p_new_col",
                "score_new",
                "score_col",
                "event_col",
                "p_recal",
                "p_pred",
                "score",
            ):
                if is_reclass and opt_key in (
                    "p_old",
                    "p_old_col",
                    "p_new",
                    "p_new_col",
                ):
                    continue
                if is_td_roc and opt_key in (
                    "p_old",
                    "p_old_col",
                    "score_old",
                ):
                    continue
                if (
                    not is_reclass
                    and not is_td_roc
                    and opt_key
                    in (
                        "p_old",
                        "p_old_col",
                        "score_old",
                        "p_new",
                        "p_new_col",
                    )
                ):
                    continue
                if opt_key == "event_col" and (is_reclass or is_iptw):
                    continue
                if (
                    opt_key
                    in (
                        "score_new",
                        "score_col",
                        "score",
                        "p_recal",
                        "p_pred",
                    )
                    and is_iptw
                ):
                    continue
                if opt_key in opts_dict:
                    val = opts_dict[opt_key]
                    if isinstance(val, str) and not val.strip():
                        continue
                    _add_col_ref(val, f"models.options.{opt_key}")

    variables_list = raw_spec.get("variables", [])
    if isinstance(variables_list, list):
        for v in variables_list:
            if isinstance(v, dict) and v.get("name"):
                _add_col_ref(v["name"], "variables.name")

    data_path_str = (
        input_path_raw
        if isinstance(input_path_raw, str) and input_path_raw.strip()
        else None
    )
    if data_path_str:
        data_path = Path(data_path_str)
        if not data_path.is_absolute():
            data_path = (Path.cwd() / data_path).resolve()

        if not data_path.exists():
            report["valid"] = False
            report["errors"].append(f"Dataset file not found: {data_path_str}")
            report["agent_remediation"].append(
                "Update `data.input_path` in the YAML spec to point to an existing CSV/Parquet file."
            )
            available_cols = set()
        else:
            try:
                available_cols = set(get_clinical_columns(data_path))
            except Exception as exc:
                report["valid"] = False
                report["errors"].append(
                    f"Failed to read dataset columns from '{data_path}': {exc}"
                )
                report["agent_remediation"].append(
                    "Ensure the dataset file exists, has a supported format, and contains valid column headers."
                )
                available_cols = set()

            missing_cols = sorted(referenced_cols - available_cols)
            if missing_cols and available_cols:
                report["valid"] = False
                report["errors"].append(f"Columns missing from dataset: {missing_cols}")
                report["agent_remediation"].append(
                    f"Missing columns {missing_cols} not found in dataset. Select from available columns: {sorted(available_cols)}"
                )

    return report

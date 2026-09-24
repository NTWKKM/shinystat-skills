"""
Unified Command Line Interface for medstat-core.

Provides 9 subcommands corresponding to the clinical biostatistical workflow:
clean, table1, model, diag, agreement, causal, meta, sample-size, and report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd

from medstat.agreement.bland_altman import bland_altman_analysis
from medstat.agreement.icc import calculate_icc
from medstat.causal.balance import calculate_smd
from medstat.causal.psm import propensity_score_match
from medstat.cli.spec import AnalysisPlan
from medstat.data.clean import (
    MissingStrategyRequiredError,
    audit_missingness,
    prepare_data_for_analysis,
)
from medstat.data.retention import SampleFlowTracker
from medstat.diagnostic.accuracy import calculate_diagnostic_accuracy
from medstat.diagnostic.calibration import (
    calculate_brier_score,
    calculate_calibration_slope_and_intercept,
    calculate_ici,
)
from medstat.diagnostic.dca import calculate_dca
from medstat.diagnostic.roc import (
    auc_ci_delong,
    calculate_roc_curve,
    delong_paired_test,
)
from medstat.meta.bias import eggers_test
from medstat.meta.forest import generate_forest_data
from medstat.meta.models import run_meta_analysis
from medstat.models.firth import fit_firth_cox, fit_firth_logistic
from medstat.models.glm import fit_linear_regression, fit_standard_logistic
from medstat.models.sensitivity import calculate_e_value
from medstat.models.splines import fit_cox_rcs
from medstat.models.survival import check_proportional_hazards, fit_cox_ph
from medstat.power.sample_size import calculate_sample_size_t_test
from medstat.reporting.checklists import get_checklist
from medstat.reporting.narrative import generate_methods_narrative
from medstat.reporting.table1 import generate_table_one
from medstat.reporting.tables import (
    Estimate,
    EstimateTable,
    PublicationRenderer,
)


def check_data_missingness(
    df: pd.DataFrame, required_cols: list[str], subcommand: str
) -> None:
    """Enforce clinical safety directive: block unhandled missing data in analysis subcommands."""
    cols_to_check = [c for c in required_cols if c in df.columns]
    if not cols_to_check:
        return
    missing_counts = df[cols_to_check].isna().sum()
    cols_with_nan = missing_counts[missing_counts > 0]
    if not cols_with_nan.empty:
        details = "\n".join(
            f"  - '{col}': {count} missing ({count / len(df) * 100:.1f}%)"
            for col, count in cols_with_nan.items()
        )
        err_msg = (
            f"[CLINICAL SAFETY ERROR: MissingStrategyRequiredError] Unhandled missing data detected in '{subcommand}':\n"
            f"{details}\n\n"
            f"Silent listwise deletion is prohibited under clinical safety directives.\n"
            f"Run 'medstat clean' to produce a validated cohort first:\n"
            f"  medstat clean --data <raw.csv> --strategy <complete-case|mice|knn> "
            f'--missing-justification "<rationale>" --output <clean.csv>'
        )
        raise click.ClickException(err_msg)


def _serialize_summary_df(sum_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize model summary DataFrame to records, preserving the index as 'term'."""
    df_copy = sum_df.copy()
    if "term" not in df_copy.columns:
        df_copy.insert(0, "term", df_copy.index.astype(str))
    return df_copy.to_dict(orient="records")


@click.group()
@click.version_option(version="0.1.0", prog_name="medstat")
def cli() -> None:
    """medstat: Clinical Biostatistics & Health Data Science CLI."""
    pass


# ==============================================================================
# 1. clean
# ==============================================================================
@cli.command("clean")
@click.option(
    "--data", required=True, type=click.Path(exists=True), help="Input CSV dataset."
)
@click.option(
    "--strategy",
    type=click.Choice(["complete-case", "mice", "knn", "indicator"]),
    default=None,
    help="Missing data strategy.",
)
@click.option(
    "--mechanism",
    type=click.Choice(["mcar", "mar", "mnar"], case_sensitive=False),
    default=None,
    help="Clinically assumed missingness mechanism (mcar, mar, mnar).",
)
@click.option(
    "--missing-justification",
    default=None,
    help="Mandatory clinical justification for missing data strategy.",
)
@click.option(
    "--audit-only",
    is_flag=True,
    help="Only run missingness audit without modifying data.",
)
@click.option(
    "--audit-out", type=click.Path(), help="Output path for missingness audit JSON."
)
@click.option(
    "--imputations",
    default=5,
    type=int,
    help="Number of multiple imputations for MICE.",
)
@click.option(
    "--neighbors", default=5, type=int, help="Number of nearest neighbors for KNN."
)
@click.option("--output", type=click.Path(), help="Output cleaned CSV file path.")
def clean_cmd(
    data: str,
    strategy: str | None,
    mechanism: str | None,
    missing_justification: str | None,
    audit_only: bool,
    audit_out: str | None,
    imputations: int,
    neighbors: int,
    output: str | None,
) -> None:
    """Audit data missingness and apply clinically justified cleaning strategies."""
    df = pd.read_csv(data)
    audit = audit_missingness(df)

    if audit_out and audit_only:
        Path(audit_out).parent.mkdir(parents=True, exist_ok=True)
        with open(audit_out, "w") as f:
            audit_dict = audit.to_dict() if hasattr(audit, "to_dict") else audit
            json.dump(audit_dict, f, indent=2, default=str)

    if audit_only:
        click.echo(
            f"Missingness Audit completed for {len(df)} rows across {len(df.columns)} columns."
        )
        return

    tracker = SampleFlowTracker(n_initial=len(df), initial_name="Total Enrolled Cohort")
    try:
        cleaned_df, info = prepare_data_for_analysis(
            df,
            required_cols=list(df.columns),
            handle_missing=strategy,
            missing_justification=missing_justification,
            tracker=tracker,
            n_imputations=imputations,
            n_neighbors=neighbors,
        )
    except MissingStrategyRequiredError as e:
        raise click.ClickException(str(e))

    assumed_mech = (
        mechanism or ("mcar" if strategy == "complete-case" else "mar")
    ).upper()
    info["assumed_mechanism"] = assumed_mech

    if audit_out:
        Path(audit_out).parent.mkdir(parents=True, exist_ok=True)
        with open(audit_out, "w") as f:
            audit_dict = audit.to_dict() if hasattr(audit, "to_dict") else audit
            audit_dict["assumed_mechanism"] = assumed_mech
            audit_dict["sample_flow"] = tracker.to_dict()
            json.dump(audit_dict, f, indent=2, default=str)

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        out_p = Path(output)
        if info.get("imputed_datasets") is not None:
            cleaned_df.to_csv(output, index=False)
            stem = out_p.stem
            suffix = out_p.suffix
            num_imps = len(info["imputed_datasets"])
            for m_idx, imp_df in enumerate(info["imputed_datasets"]):
                imp_path = out_p.parent / f"{stem}_imp{m_idx + 1}{suffix}"
                imp_df.to_csv(imp_path, index=False)
            click.echo(
                f"Cleaned dataset saved to: {output} (Rows: {len(cleaned_df)}, Mechanism: {assumed_mech})\n"
                f"Preserved {num_imps} imputed datasets ({stem}_imp1{suffix} to {stem}_imp{num_imps}{suffix}). "
                f"Downstream analyses must pool results across imputations using Rubin's rules."
            )
        else:
            cleaned_df.to_csv(output, index=False)
            click.echo(
                f"Cleaned dataset saved to: {output} (Rows: {len(cleaned_df)}, Mechanism: {assumed_mech})"
            )


# ==============================================================================
# 2. table1
# ==============================================================================
@cli.command("table1")
@click.option(
    "--data", required=True, type=click.Path(exists=True), help="Input CSV dataset."
)
@click.option(
    "--group", default=None, help="Grouping/strata variable (e.g. treatment)."
)
@click.option(
    "--vars", default=None, help="Comma-separated list of variables to include."
)
@click.option("--digits", default=1, type=int, help="Decimal precision.")
@click.option(
    "--include-smd/--no-include-smd",
    default=True,
    help="Include Standardized Mean Differences.",
)
@click.option(
    "--include-p/--no-include-p",
    default=True,
    help="Include p-values.",
)
@click.option("--output", type=click.Path(), help="Output path (JSON, HTML, or CSV).")
def table1_cmd(
    data: str,
    group: str | None,
    vars: str | None,
    digits: int,
    include_smd: bool,
    include_p: bool,
    output: str | None,
) -> None:
    df = pd.read_csv(data)
    var_list = [v.strip() for v in vars.split(",")] if vars else None
    cols_to_check = ([group] if group else []) + (
        var_list if var_list else list(df.columns)
    )
    check_data_missingness(df, cols_to_check, "table1")
    cont_vars = None
    cat_vars = None
    if var_list:
        cont_vars = [
            v
            for v in var_list
            if pd.api.types.is_numeric_dtype(df[v]) and df[v].nunique() > 10
        ]
        cat_vars = [v for v in var_list if v not in cont_vars]

    t1_df = generate_table_one(
        df,
        strata=group,
        continuous_vars=cont_vars,
        categorical_vars=cat_vars,
        include_smd=include_smd,
        include_p=include_p,
        digits=digits,
    )

    if output:
        out_p = Path(output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if out_p.suffix == ".json":
            t1_df.to_json(output, orient="records", indent=2)
        elif out_p.suffix == ".html":
            t1_df.to_html(output, index=False)
        else:
            t1_df.to_csv(output, index=False)
        click.echo(f"Table 1 generated successfully: {output}")
    else:
        click.echo(t1_df.to_string())


# ==============================================================================
# 3. model
# ==============================================================================
@cli.command("model")
@click.option("--data", type=click.Path(exists=True), help="Input CSV dataset.")
@click.option(
    "--spec", type=click.Path(exists=True), help="Path to SAP analysis_plan.yaml."
)
@click.option(
    "--type",
    "model_type",
    default="logistic",
    help="Model type (logistic, linear, cox, cox_ph).",
)
@click.option("--outcome", default=None, help="Outcome variable name.")
@click.option(
    "--time", "time_col", default=None, help="Survival duration variable name."
)
@click.option(
    "--exposure", default=None, help="Primary exposure/treatment variable name."
)
@click.option("--covariates", default=None, help="Comma-separated covariates.")
@click.option(
    "--method", default="standard", help="Estimation method (standard, firth)."
)
@click.option(
    "--ci-method", default="profile", help="Confidence interval method (profile, wald)."
)
@click.option(
    "--schoenfeld",
    is_flag=True,
    help="Run Schoenfeld proportional hazards assumption test.",
)
@click.option(
    "--e-value",
    is_flag=True,
    help="Compute VanderWeele E-value for unmeasured confounding.",
)
@click.option(
    "--spline-var",
    default=None,
    help="Continuous variable to model with restricted cubic splines (RCS).",
)
@click.option(
    "--knots",
    default=4,
    type=int,
    help="Number of knots for restricted cubic splines (default: 4).",
)
@click.option("--output", type=click.Path(), help="Output results JSON file path.")
def model_cmd(
    data: str | None,
    spec: str | None,
    model_type: str,
    outcome: str | None,
    time_col: str | None,
    exposure: str | None,
    covariates: str | None,
    method: str,
    ci_method: str,
    schoenfeld: bool,
    e_value: bool,
    spline_var: str | None,
    knots: int,
    output: str | None,
) -> None:
    """Fit regression or survival models or execute a pre-specified SAP plan."""
    if spec:
        plan = AnalysisPlan.from_yaml(spec)
        res = plan.execute()
        if output:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                json.dump(res, f, indent=2, default=str)
            click.echo(f"Analysis Plan executed successfully: {output}")
        return

    if not data or not outcome:
        raise click.UsageError(
            "Must specify either --spec <plan.yaml> or both --data and --outcome."
        )

    df = pd.read_csv(data)
    covar_list = [c.strip() for c in covariates.split(",")] if covariates else []
    if exposure and exposure not in covar_list:
        covar_list.insert(0, exposure)

    cols_to_check = [outcome] + ([time_col] if time_col else []) + covar_list
    check_data_missingness(df, cols_to_check, "model")

    mtype = model_type.lower().strip()
    if "firth" in mtype:
        method = "firth"
        if "logistic" in mtype or "binary" in mtype:
            mtype = "logistic"
        elif "cox" in mtype or "survival" in mtype:
            mtype = "cox"

    result_data: dict[str, Any] = {"model_type": mtype, "outcome": outcome}

    # Dummy-encode any categorical covariates
    X_raw = df[covar_list].copy()
    cat_cols = [c for c in covar_list if not pd.api.types.is_numeric_dtype(X_raw[c])]
    if cat_cols:
        X_df = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True, dtype=float)
    else:
        X_df = X_raw.astype(float)
    feature_names = list(X_df.columns)

    if mtype in ("cox", "cox_ph"):
        t_col = time_col or "time"
        t_vals = pd.to_numeric(df[t_col], errors="coerce").values
        ev_raw = df[outcome]
        u_ev = ev_raw.dropna().unique()
        if len(u_ev) > 2:
            raise click.ClickException(
                f"Event outcome column '{outcome}' must have at most 2 levels, found {len(u_ev)}."
            )
        if not pd.api.types.is_numeric_dtype(ev_raw):
            raise click.ClickException(
                f"Event outcome column '{outcome}' must be numeric (0 for non-event, 1 for event). "
                "Text outcomes are not permitted; please recode outcomes as 0/1, with 1 representing the event."
            )
        if not set(u_ev).issubset({0, 1, 0.0, 1.0}):
            raise click.ClickException(
                f"Event outcome column '{outcome}' contains values outside {{0, 1}}: {set(u_ev)}"
            )
        ev_vals = ev_raw.astype(int).values

        if method == "firth":
            fit_res = fit_firth_cox(
                t_vals,
                ev_vals,
                X_df.values,
                feature_names=feature_names,
            )
            sum_df = fit_res["summary_df"]
            result_data["coefficients"] = _serialize_summary_df(sum_df)
            if schoenfeld:
                result_data["schoenfeld_test"] = {
                    "status": "not_performed",
                    "reason": "Schoenfeld residuals test is not performed for Firth penalized Cox models",
                }
        else:
            df_cox = X_df.copy()
            df_cox[t_col] = t_vals
            df_cox[outcome] = ev_vals
            fit_res = fit_cox_ph(
                df_cox, duration_col=t_col, event_col=outcome, covariates=feature_names
            )
            sum_df = fit_res["summary_df"]
            result_data["coefficients"] = _serialize_summary_df(sum_df)
            if schoenfeld:
                ph_res = check_proportional_hazards(fit_res["model"], df=df_cox)
                result_data["schoenfeld_test"] = str(ph_res)

    elif mtype in ("logistic", "binary"):
        y_raw = df[outcome]
        u_y = y_raw.dropna().unique()
        if len(u_y) > 2:
            raise click.ClickException(
                f"Binary outcome column '{outcome}' must have at most 2 levels, found {len(u_y)}."
            )
        if not pd.api.types.is_numeric_dtype(y_raw):
            raise click.ClickException(
                f"Binary outcome column '{outcome}' must be numeric (0 for non-event, 1 for event). "
                "Text outcomes are not permitted; please recode outcomes as 0/1, with 1 representing the event."
            )
        if not set(u_y).issubset({0, 1, 0.0, 1.0}):
            raise click.ClickException(
                f"Binary outcome column '{outcome}' contains values outside {{0, 1}}: {set(u_y)}"
            )
        y = y_raw.astype(int).values

        if method == "firth":
            fit_res = fit_firth_logistic(y, X_df, feature_names=feature_names)
            sum_df = fit_res["summary_df"]
            result_data["coefficients"] = _serialize_summary_df(sum_df)
        else:
            fit_res = fit_standard_logistic(y, X_df)
            sum_df = fit_res["summary_df"]
            result_data["coefficients"] = _serialize_summary_df(sum_df)

        if e_value and exposure:
            matching = [idx for idx in sum_df.index if str(idx) == exposure]
            if not matching:
                matching = [
                    idx for idx in sum_df.index if str(idx).startswith(f"{exposure}_")
                ]
            if not matching:
                raise click.ClickException(
                    f"Exposure term '{exposure}' not found in model results for E-value calculation."
                )
            exp_term = sum_df.loc[matching[0]]
            or_val = exp_term.get("odds_ratio", exp_term.get("estimate"))
            ci_lo = exp_term.get("or_ci_lower", exp_term.get("ci_lower"))
            ci_hi = exp_term.get("or_ci_upper", exp_term.get("ci_upper"))
            if (
                or_val is None
                or ci_lo is None
                or ci_hi is None
                or np.isnan(float(or_val))
            ):
                raise click.ClickException(
                    f"Model output for exposure '{exposure}' missing required OR or CI values for E-value calculation."
                )
            ev = calculate_e_value(
                float(or_val), float(ci_lo), float(ci_hi), estimate_type="OR"
            )
            result_data["e_value"] = ev

    elif mtype in ("linear", "ols"):
        fit_res = fit_linear_regression(df[outcome], X_df)
        result_data["coefficients"] = _serialize_summary_df(fit_res["summary_df"])
    else:
        raise click.UsageError(f"Unsupported model type: {mtype}")

    if spline_var:
        if mtype in ("cox", "cox_ph"):
            t_col = time_col or "time"
            other_covars = [c for c in covar_list if c != spline_var]
            rcs_res = fit_cox_rcs(
                df,
                duration_col=t_col,
                event_col=outcome,
                spline_var=spline_var,
                covariates=other_covars,
                n_knots=knots,
            )
            rcs_summary = rcs_res.get("summary_df")
            if isinstance(rcs_summary, pd.DataFrame):
                result_data["spline_estimates"] = _serialize_summary_df(rcs_summary)
            result_data["knots"] = rcs_res.get("knots", [])
            result_data["non_linear_pvalue"] = rcs_res.get("non_linear_pvalue")

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(result_data, f, indent=2, default=str)
        click.echo(f"Model results saved: {output}")


# ==============================================================================
# 4. diag
# ==============================================================================
@cli.command("diag")
@click.option(
    "--data", required=True, type=click.Path(exists=True), help="Input CSV dataset."
)
@click.option(
    "--gold-standard", required=True, help="Binary reference standard column."
)
@click.option("--test-col", required=True, help="Biomarker/diagnostic score column.")
@click.option(
    "--cutoff", type=float, default=None, help="Diagnostic classification cutoff."
)
@click.option("--roc", is_flag=True, help="Compute empirical ROC curve and AUC.")
@click.option(
    "--compare-roc",
    default=None,
    help="Second biomarker column for paired DeLong ROC comparison.",
)
@click.option(
    "--dca", is_flag=True, help="Perform Decision Curve Analysis (DCA net benefit)."
)
@click.option(
    "--calibration", is_flag=True, help="Compute calibration curve and Brier score."
)
@click.option("--output", type=click.Path(), help="Output path for diagnostic JSON.")
def diag_cmd(
    data: str,
    gold_standard: str,
    test_col: str,
    cutoff: float | None,
    roc: bool,
    compare_roc: str | None,
    dca: bool,
    calibration: bool,
    output: str | None,
) -> None:
    """Evaluate diagnostic test accuracy, ROC curves, DeLong comparisons, and DCA."""
    df = pd.read_csv(data)
    cols_to_check = [gold_standard, test_col] + ([compare_roc] if compare_roc else [])
    check_data_missingness(df, cols_to_check, "diag")
    y_true = df[gold_standard].values
    y_score = df[test_col].values
    diag_res: dict[str, Any] = {}

    if cutoff is not None:
        y_pred = (y_score >= cutoff).astype(int)
        acc = calculate_diagnostic_accuracy(y_true, y_pred)
        diag_res["accuracy_at_cutoff"] = acc

    if roc or compare_roc:
        roc_res = calculate_roc_curve(y_true, y_score)
        delong_res = auc_ci_delong(y_true, y_score)
        diag_res["roc"] = {
            "auc": roc_res["auc"],
            "auc_ci": [delong_res["ci_lower"], delong_res["ci_upper"]],
            "ci_lower": delong_res["ci_lower"],
            "ci_upper": delong_res["ci_upper"],
            "se": delong_res["se"],
            "youden_index": roc_res.get("youden_index"),
        }
        if compare_roc:
            y_comp = df[compare_roc].values
            comp_res = delong_paired_test(y_true, y_score, y_comp)
            diag_res["delong_comparison"] = comp_res

    if dca:
        dca_res = calculate_dca(y_true, y_score)
        if isinstance(dca_res, pd.DataFrame):
            diag_res["dca"] = dca_res.to_dict(orient="records")
        else:
            diag_res["dca"] = dca_res

    if calibration:
        brier = calculate_brier_score(y_true, y_score)
        slope_inter = calculate_calibration_slope_and_intercept(y_true, y_score)
        ici = calculate_ici(y_true, y_score)
        diag_res["calibration"] = {
            "brier": brier,
            "slope_and_intercept": slope_inter,
            "ici": ici,
        }

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(diag_res, f, indent=2, default=str)
        click.echo(f"Diagnostic results saved to: {output}")


# ==============================================================================
# 5. agreement
# ==============================================================================
@cli.group("agreement")
def agreement_grp() -> None:
    """Evaluate rater agreement, Bland-Altman LoA, and Pure-SciPy ICC."""
    pass


@agreement_grp.command("icc")
@click.option(
    "--data", required=True, type=click.Path(exists=True), help="Input CSV dataset."
)
@click.option("--targets", required=True, help="Target/Subject identifier column.")
@click.option("--raters", required=True, help="Rater identifier column.")
@click.option("--ratings", required=True, help="Measurement score column.")
@click.option(
    "--type",
    "icc_type",
    default="icc2_k",
    help="ICC variant (icc1, icc2, icc3, icc1_k, icc2_k, icc3_k).",
)
@click.option("--output", type=click.Path(), help="Output path for ICC JSON.")
def icc_cmd(
    data: str,
    targets: str,
    raters: str,
    ratings: str,
    icc_type: str,
    output: str | None,
) -> None:
    """Compute pure-Python/SciPy Intraclass Correlation Coefficient (ICC)."""
    df = pd.read_csv(data)
    check_data_missingness(df, [targets, raters, ratings], "agreement icc")
    icc_df = calculate_icc(
        df, targets=targets, raters=raters, ratings=ratings, icc_type=icc_type
    )
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        icc_df.to_json(output, orient="records", indent=2)
        click.echo(f"ICC results saved to: {output}")


@agreement_grp.command("bland-altman")
@click.option(
    "--data", required=True, type=click.Path(exists=True), help="Input CSV dataset."
)
@click.option("--m1", default=None, help="First measurement column name.")
@click.option("--m2", default=None, help="Second measurement column name.")
@click.option("--output", type=click.Path(), help="Output path for Bland-Altman JSON.")
def bland_altman_cmd(
    data: str, m1: str | None, m2: str | None, output: str | None
) -> None:
    """Compute Bland-Altman mean bias and limits of agreement (LoA)."""
    df = pd.read_csv(data)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2 and (not m1 or not m2):
        cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        if len(cat_cols) >= 2 and len(num_cols) == 1:
            df = df.pivot(
                index=cat_cols[0], columns=cat_cols[1], values=num_cols[0]
            ).reset_index()

    cols = (
        [m1, m2]
        if m1 and m2
        else [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:2]
    )
    check_data_missingness(df, cols, "agreement bland-altman")
    res = bland_altman_analysis(df[cols[0]], df[cols[1]])
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(res, f, indent=2, default=str)
        click.echo(f"Bland-Altman analysis saved to: {output}")


@agreement_grp.command("kappa")
@click.option(
    "--data", required=True, type=click.Path(exists=True), help="Input CSV dataset."
)
@click.option("--rater1", default=None, help="First rater column name.")
@click.option("--rater2", default=None, help="Second rater column name.")
@click.option("--output", type=click.Path(), help="Output path for Kappa JSON.")
def kappa_cmd(
    data: str, rater1: str | None, rater2: str | None, output: str | None
) -> None:
    """Compute Fleiss' or Cohen's Kappa for categorical agreement."""
    raise click.ClickException(
        "Cohen's/Fleiss' kappa calculation is not yet implemented in the agreement engine."
    )


# ==============================================================================
# 6. causal
# ==============================================================================
@cli.group("causal")
def causal_grp() -> None:
    """Propensity score matching, weighting, and balance diagnostics."""
    pass


@causal_grp.command("psm")
@click.option(
    "--data", required=True, type=click.Path(exists=True), help="Input CSV dataset."
)
@click.option("--treatment", required=True, help="Binary treatment column.")
@click.option(
    "--covariates", required=True, help="Comma-separated baseline covariates."
)
@click.option(
    "--caliper", default=0.2, type=float, help="Caliper width in standard deviations."
)
@click.option("--ratio", default=1, type=int, help="Matching ratio.")
@click.option(
    "--balance-check",
    is_flag=True,
    help="Evaluate post-match Standardized Mean Differences.",
)
@click.option("--love-plot", type=click.Path(), help="Output path for Love plot JSON.")
@click.option("--output", type=click.Path(), help="Output path for matched cohort CSV.")
def psm_cmd(
    data: str,
    treatment: str,
    covariates: str,
    caliper: float,
    ratio: int,
    balance_check: bool,
    love_plot: str | None,
    output: str | None,
) -> None:
    """Execute Propensity Score Matching (PSM) with caliper and balance diagnostics."""
    df = pd.read_csv(data)
    covar_list = [c.strip() for c in covariates.split(",")]
    check_data_missingness(df, [treatment] + covar_list, "causal psm")
    matched_df, info = propensity_score_match(
        df,
        treatment_col=treatment,
        covariates=covar_list,
        caliper=caliper,
        ratio=ratio,
    )

    if love_plot:
        Path(love_plot).parent.mkdir(parents=True, exist_ok=True)
        love_data = {
            "covariates": covar_list,
            "smd_raw": [float(calculate_smd(df, treatment, c)) for c in covar_list],
            "smd_pre": [float(calculate_smd(df, treatment, c)) for c in covar_list],
            "smd_matched": [
                float(calculate_smd(matched_df, treatment, c)) for c in covar_list
            ],
            "smd_post": [
                float(calculate_smd(matched_df, treatment, c)) for c in covar_list
            ],
            "post_smd": [
                float(calculate_smd(matched_df, treatment, c)) for c in covar_list
            ],
        }
        with open(love_plot, "w") as f:
            json.dump(love_data, f, indent=2)

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        matched_df.to_csv(output, index=False)
        click.echo(
            f"Matched cohort saved to: {output} (Matched pairs: {len(matched_df) // 2})"
        )


# ==============================================================================
# 7. meta
# ==============================================================================
@cli.command("meta")
@click.option(
    "--data", required=True, type=click.Path(exists=True), help="Input CSV dataset."
)
@click.option("--effect-col", required=True, help="Effect size column name.")
@click.option("--se-col", required=True, help="Standard error column name.")
@click.option("--study-col", required=True, help="Study name column.")
@click.option(
    "--model",
    default="random",
    type=click.Choice(["fixed", "random"]),
    help="Fixed or random effects.",
)
@click.option(
    "--method",
    default="dl",
    help="Estimation method (dl=DerSimonian-Laird, iv=Inverse Variance).",
)
@click.option(
    "--forest-plot", type=click.Path(), help="Output path for Forest plot JSON."
)
@click.option(
    "--egger", is_flag=True, help="Run Egger's regression test for publication bias."
)
@click.option(
    "--output", type=click.Path(), help="Output path for meta-analysis summary JSON."
)
def meta_cmd(
    data: str,
    effect_col: str,
    se_col: str,
    study_col: str,
    model: str,
    method: str,
    forest_plot: str | None,
    egger: bool,
    output: str | None,
) -> None:
    """Perform fixed and random effects meta-analysis with Forest plot data."""
    df = pd.read_csv(data)
    check_data_missingness(df, [effect_col, se_col, study_col], "meta")
    res = run_meta_analysis(
        df,
        effect_col=effect_col,
        se_col=se_col,
        study_col=study_col,
        model=model,
        method=method,
    )

    if egger:
        res["egger_test"] = eggers_test(df[effect_col], df[se_col])

    if forest_plot:
        Path(forest_plot).parent.mkdir(parents=True, exist_ok=True)
        f_data = generate_forest_data(df, effect_col, se_col, study_col, res)
        with open(forest_plot, "w") as f:
            json.dump(f_data, f, indent=2, default=str)

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(res, f, indent=2, default=str)
        click.echo(f"Meta-analysis results saved: {output}")


# ==============================================================================
# 8. sample-size
# ==============================================================================
@cli.command("sample-size")
@click.option(
    "--type",
    "test_type",
    default="t-test",
    help="Test type (t-test, proportions, survival).",
)
@click.option("--alpha", default=0.05, type=float, help="Type I error rate.")
@click.option("--power", default=0.80, type=float, help="Statistical power (1 - beta).")
@click.option("--effect-size", default=0.5, type=float, help="Target effect size.")
@click.option("--output", type=click.Path(), help="Output path for sample size JSON.")
def sample_size_cmd(
    test_type: str,
    alpha: float,
    power: float,
    effect_size: float,
    output: str | None,
) -> None:
    """Calculate statistical power and required sample size."""
    valid_types = ("t-test", "t_test", "ttest")
    if test_type.lower() not in valid_types:
        raise click.ClickException(
            f"Sample size calculation for test type '{test_type}' is not supported. Supported: {list(valid_types)}"
        )

    n_per_group = calculate_sample_size_t_test(
        effect_size=effect_size, alpha=alpha, power=power
    )
    res = {
        "test_type": test_type,
        "alpha": alpha,
        "power": power,
        "effect_size": effect_size,
        "sample_size_per_group": n_per_group,
        "total_sample_size": n_per_group * 2,
    }
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(res, f, indent=2)
        click.echo(f"Sample size calculation saved: {output}")
    else:
        click.echo(json.dumps(res, indent=2))


# ==============================================================================
# 9. report
# ==============================================================================
@cli.command("report")
@click.option(
    "--results",
    type=click.Path(exists=True),
    help="Input model/meta analysis results JSON.",
)
@click.option(
    "--style",
    default="nejm",
    type=click.Choice(["nejm", "jama", "apa7"]),
    help="Journal styling.",
)
@click.option(
    "--format", "out_format", default="html", help="Report format (html, markdown)."
)
@click.option(
    "--checklist", default=None, help="Checklist audit name (strobe, consort, tripod)."
)
@click.option("--narrative", is_flag=True, help="Generate methods narrative.")
@click.option(
    "--output", required=True, type=click.Path(), help="Output report file path."
)
def report_cmd(
    results: str | None,
    style: str,
    out_format: str,
    checklist: str | None,
    narrative: bool,
    output: str,
) -> None:
    """Render publication-grade HTML tables and reporting guideline checklists."""
    out_p = Path(output)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if checklist:
        chk = get_checklist(checklist)
        out_p.write_text(chk.to_markdown(), encoding="utf-8")
        click.echo(f"Reporting checklist saved to: {output}")
        return

    if narrative and not results:
        text = generate_methods_narrative()
        out_p.write_text(text, encoding="utf-8")
        click.echo(f"Methods narrative saved to: {output}")
        return

    if results:
        with open(results, "r") as f:
            res_data = json.load(f)

        est_rows: list[Estimate] = []
        coefs = res_data.get("coefficients", [])
        if isinstance(coefs, list) and len(coefs) > 0:
            for c in coefs:
                term = str(c.get("term", c.get("feature", c.get("variable", "var"))))
                est_candidates = [
                    c.get("odds_ratio"),
                    c.get("hazard_ratio"),
                    c.get("estimate"),
                    c.get("exp(coef)"),
                    c.get("coef"),
                ]
                est_val = next((v for v in est_candidates if v is not None), None)
                if est_val is None:
                    raise click.ClickException(
                        f"Missing estimate/coef for term '{term}'"
                    )
                est = float(est_val)

                ci_lo_candidates = [
                    c.get("hr_ci_lower"),
                    c.get("or_ci_lower"),
                    c.get("ci_lower"),
                    c.get("exp(coef) lower 95%"),
                    c.get("lower_ci"),
                ]
                ci_lo_val = next((v for v in ci_lo_candidates if v is not None), None)

                ci_hi_candidates = [
                    c.get("hr_ci_upper"),
                    c.get("or_ci_upper"),
                    c.get("ci_upper"),
                    c.get("exp(coef) upper 95%"),
                    c.get("upper_ci"),
                ]
                ci_hi_val = next((v for v in ci_hi_candidates if v is not None), None)

                pval_val = c.get("p_value", c.get("p", c.get("p-val", c.get("p_val"))))

                missing_keys = []
                if ci_lo_val is None:
                    missing_keys.append("ci_lower")
                if ci_hi_val is None:
                    missing_keys.append("ci_upper")
                if pval_val is None:
                    missing_keys.append("p_value")

                if missing_keys:
                    raise click.ClickException(
                        f"Term '{term}' missing required keys: {', '.join(missing_keys)}"
                    )

                ci_lo = float(ci_lo_val)
                ci_hi = float(ci_hi_val)
                pval = float(pval_val)
                est_rows.append(
                    Estimate(
                        term=term,
                        label=term,
                        estimate=est,
                        ci_lower=ci_lo,
                        ci_upper=ci_hi,
                        p_value=pval,
                    )
                )
        elif "studies" in res_data:
            studies = res_data["studies"]
            if isinstance(studies, dict):
                if all(isinstance(v, dict) for v in studies.values()):
                    studies_records = pd.DataFrame.from_dict(studies).to_dict(
                        orient="records"
                    )
                else:
                    studies_records = [studies]
            elif isinstance(studies, list):
                studies_records = studies
            else:
                studies_records = []

            for s in studies_records:
                if isinstance(s, dict):
                    pval_raw = s.get(
                        "p_value", s.get("p", s.get("p_val", s.get("p-val")))
                    )
                    pval = (
                        float(pval_raw)
                        if (
                            pval_raw is not None
                            and not (isinstance(pval_raw, float) and np.isnan(pval_raw))
                        )
                        else float("nan")
                    )
                    ci_lo = s.get("ci_lower")
                    ci_hi = s.get("ci_upper")
                    study_name = str(s.get("study", "Study"))
                    raw_eff = s.get("effect_size", s.get("effect"))
                    if raw_eff is None:
                        is_log_scale = (
                            "log" in str(s.get("scale", "")).lower()
                            or "log" in str(res_data.get("effect_measure", "")).lower()
                        )
                        if is_log_scale:
                            raw_eff = s.get("log_effect")
                    if raw_eff is None or (
                        isinstance(raw_eff, float) and np.isnan(raw_eff)
                    ):
                        raise click.ClickException(
                            f"Missing required effect value for study '{study_name}' in meta-analysis results."
                        )
                    eff_val = float(raw_eff)
                    ci_lo = s.get("ci_lower")
                    ci_hi = s.get("ci_upper")
                    est_rows.append(
                        Estimate(
                            term=study_name,
                            label=study_name,
                            estimate=eff_val,
                            ci_lower=float(ci_lo)
                            if ci_lo is not None
                            else float("nan"),
                            ci_upper=float(ci_hi)
                            if ci_hi is not None
                            else float("nan"),
                            p_value=pval,
                            scale="Effect",
                        )
                    )
            re = res_data.get(
                "random_effects",
                res_data.get("fixed_effect", res_data.get("summary", {})),
            )
            if re and isinstance(re, dict):
                re_p_raw = re.get("p_value", re.get("p", re.get("p_val")))
                re_pval = (
                    float(re_p_raw)
                    if (
                        re_p_raw is not None
                        and not (isinstance(re_p_raw, float) and np.isnan(re_p_raw))
                    )
                    else float("nan")
                )
                raw_overall = re.get("effect_disp", re.get("effect"))
                if raw_overall is None or (
                    isinstance(raw_overall, float) and np.isnan(raw_overall)
                ):
                    raise click.ClickException(
                        "Missing required overall effect value in meta-analysis summary."
                    )
                re_ci_lo = re.get("ci_lower")
                re_ci_hi = re.get("ci_upper")
                est_rows.append(
                    Estimate(
                        term="Overall Effect",
                        label=str(re.get("label", "Overall Effect")),
                        estimate=float(raw_overall),
                        ci_lower=float(re_ci_lo)
                        if re_ci_lo is not None
                        else float("nan"),
                        ci_upper=float(re_ci_hi)
                        if re_ci_hi is not None
                        else float("nan"),
                        p_value=re_pval,
                        scale="Overall",
                    )
                )

        narr_kwargs = {
            "model_type": res_data.get("model_type", "logistic"),
            "outcome": res_data.get("outcome", "primary outcome"),
            "exposure": res_data.get("exposure"),
            "covariates": res_data.get("covariates"),
            "missing_strategy": res_data.get(
                "missing_strategy",
                res_data.get("missing_data", {}).get("strategy", "complete-case"),
            ),
            "is_firth": res_data.get("method") == "firth",
            "is_rcs": bool(
                res_data.get("spline_var") or res_data.get("spline_estimates")
            ),
            "check_schoenfeld": bool(
                res_data.get("schoenfeld_test")
                and res_data.get("schoenfeld_test") != "not_performed"
                and not (
                    isinstance(res_data.get("schoenfeld_test"), dict)
                    and res_data.get("schoenfeld_test", {}).get("status")
                    == "not_performed"
                )
            ),
        }

        if out_format.lower() in ("markdown", "md") or str(output).endswith(".md"):
            narr_text = generate_methods_narrative(**narr_kwargs)
            out_p.write_text(narr_text, encoding="utf-8")
        else:
            est_tbl = EstimateTable(
                title="Table. Multivariable Clinical Outcomes", rows=est_rows
            )
            html_out = PublicationRenderer.render_html(est_tbl, style=style)
            if narrative:
                narr_text = generate_methods_narrative(**narr_kwargs)
                html_out += f"\n<!-- Methods Narrative -->\n<div class='methods-narrative'><p>{narr_text}</p></div>"
            out_p.write_text(html_out, encoding="utf-8")
        click.echo(f"Publication report saved to: {output}")


def main() -> None:
    """CLI Entry point."""
    cli()


if __name__ == "__main__":
    main()

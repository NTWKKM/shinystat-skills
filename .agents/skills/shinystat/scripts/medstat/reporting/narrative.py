"""
Automated Statistical Methods Narrative Generator.

Drafts publication-grade Statistical Methods section narratives conforming
to STROBE and CONSORT reporting guidelines.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def validate_calibration_metrics(
    metrics: dict[str, Any],
    provenance: str | None = None,
    tautological_slope: bool = False,
) -> dict[str, float]:
    """
    Validate calibration metrics: reject booleans and non-finite values.
    For apparent provenance with tautological_slope=True, omit calibration slope and intercept
    (apparent slope is 1.0 and intercept is 0.0 by construction in derivation data).
    Otherwise, retain all valid finite metrics so they can be reported as apparent.
    For external provenance, require full quartet: brier, slope, intercept, ici.
    """
    if provenance not in ("apparent", "external", None):
        raise ValueError(
            "calibration_provenance must be 'apparent', 'external', or None."
        )
    if not metrics:
        raise ValueError(
            "calibration_metrics must be provided when calibration_provenance is set."
        )

    for k, v in metrics.items():
        if (
            isinstance(v, (bool, np.bool_))
            or not isinstance(v, (int, float, np.number))
            or not np.isfinite(v)
        ):
            raise ValueError(
                f"Calibration metric '{k}' must be a finite numeric value (got {v})."
            )
        if k.lower() in ("brier", "ici") and not 0.0 <= float(v) <= 1.0:
            raise ValueError(
                f"Calibration metric '{k}' must lie within [0, 1] (got {v})."
            )

    if provenance == "apparent":
        if tautological_slope:
            filtered = {
                k: float(v)
                for k, v in metrics.items()
                if k.lower() not in ("slope", "intercept")
            }
            if not filtered:
                raise ValueError(
                    "Apparent validation requires at least one non-slope/intercept calibration metric (e.g., Brier score)."
                )
            return filtered
        return {k: float(v) for k, v in metrics.items()}
    elif provenance == "external":
        required = {"brier", "slope", "intercept", "ici"}
        missing = required - {k.lower() for k in metrics}
        if missing:
            raise ValueError(
                f"External validation requires calibration metrics {sorted(required)}; missing: {sorted(missing)}."
            )
        return {k: float(v) for k, v in metrics.items()}
    return {k: float(v) for k, v in metrics.items()}


def generate_methods_narrative(
    model_type: str = "logistic",
    exposure: str | None = None,
    outcome: str | None = None,
    covariates: list[str] | None = None,
    missing_strategy: str | None = None,
    is_firth: bool = False,
    is_rcs: bool = False,
    alpha: float = 0.05,
    software_name: str = "medstat-core",
    include_descriptive: bool = True,
    check_schoenfeld: bool = False,
    normality_test: bool = False,
    fisher_exact: bool = False,
    cutoff: float | None = None,
    direction: str | None = None,
    has_roc: bool = False,
    has_delong: bool = False,
    has_dca: bool = False,
    has_calibration: bool = False,
    **kwargs: Any,
) -> str:
    """
    Generate an academic Statistical Methods section paragraph.

    Parameters:
        model_type: 'logistic', 'cox', 'linear', 'poisson', 'diagnostic', etc.
        exposure: Primary exposure/treatment variable.
        outcome: Primary clinical outcome.
        covariates: List of confounders adjusted for.
        missing_strategy: Strategy used to handle missing data ('complete-case', 'mice', 'knn', 'indicator').
        is_firth: Whether Firth penalized likelihood was used.
        is_rcs: Whether restricted cubic splines were fitted for non-linear terms.
        alpha: Significance threshold (default 0.05).
        software_name: Software package citation.
        include_descriptive: Whether to include baseline descriptive statistics description.
        check_schoenfeld: Whether Schoenfeld residuals testing was performed for Cox models.
        normality_test: Whether formal normality tests (Shapiro-Wilk) were performed.
        fisher_exact: Whether Fisher's exact test was used for small cell counts.
        cutoff: Prespecified diagnostic cutoff value.
        direction: Biomarker directionality ('high' or 'low').
        has_roc: Whether ROC analysis was conducted.
        has_delong: Whether DeLong paired ROC comparison was conducted.
        has_dca: Whether Decision Curve Analysis was conducted.
        has_calibration: Whether calibration analysis was conducted.

    Returns:
        Formatted English narrative paragraph.
    """
    norm_mtype = model_type.lower()
    if norm_mtype in ("diagnostic", "diag", "dta"):
        diag_data = kwargs.get("diagnostic_data", {})
        cutoff_val = cutoff
        if cutoff_val is None and isinstance(diag_data, dict):
            cutoff_val = diag_data.get("cutoff")
            if cutoff_val is None:
                acc = diag_data.get("accuracy_at_cutoff", {})
                if isinstance(acc, dict):
                    cutoff_val = acc.get("cutoff")

        dir_val = direction
        if dir_val is None and isinstance(diag_data, dict):
            dir_val = diag_data.get("direction")
            if dir_val is None:
                roc_dict = diag_data.get("roc", {})
                if isinstance(roc_dict, dict):
                    dir_val = roc_dict.get("direction")

        has_roc_eval = has_roc or (isinstance(diag_data, dict) and "roc" in diag_data)
        has_delong_eval = has_delong or (
            isinstance(diag_data, dict) and "delong_comparison" in diag_data
        )
        has_dca_eval = has_dca or (isinstance(diag_data, dict) and "dca" in diag_data)
        has_cal_eval = has_calibration or (
            isinstance(diag_data, dict) and "calibration" in diag_data
        )

        diag_parts = []
        if cutoff_val is not None or (
            isinstance(diag_data, dict) and "accuracy_at_cutoff" in diag_data
        ):
            dir_clause = ""
            if dir_val == "low":
                dir_clause = " (with test scores less than or equal to the cutoff threshold considered positive)"
            elif dir_val == "high":
                dir_clause = " (with test scores greater than or equal to the cutoff threshold considered positive)"
            thresh_str = (
                f"at a prespecified cutoff of {cutoff_val}"
                if cutoff_val is not None
                else "at the evaluated cutoff"
            )
            diag_parts.append(
                f"Diagnostic test performance (sensitivity, specificity, positive predictive value, negative predictive value, and likelihood ratios) was evaluated {thresh_str}{dir_clause}. "
                "Confidence intervals (95%) for proportions (sensitivity, specificity, PPV, and NPV) were calculated using the Wilson score method."
            )

        if has_roc_eval:
            roc_sentence = (
                "Discriminatory performance was evaluated by constructing the Receiver Operating Characteristic (ROC) curve, "
                "with the Area Under the Curve (AUC) and 95% confidence intervals estimated using DeLong's non-parametric method."
            )
            if has_delong_eval:
                roc_sentence += " Differences between paired ROC curves were tested using the paired DeLong test."
            diag_parts.append(roc_sentence)

        if has_dca_eval:
            diag_parts.append(
                "Clinical utility and net benefit across decision threshold probabilities were assessed using Decision Curve Analysis (DCA)."
            )

        if has_cal_eval:
            cal_dict = (
                diag_data.get("calibration", {}) if isinstance(diag_data, dict) else {}
            )
            prob_source = (
                diag_data.get("probability_source")
                if isinstance(diag_data, dict)
                else None
            ) or (
                cal_dict.get("probability_source")
                if isinstance(cal_dict, dict)
                else None
            )
            if prob_source:
                diag_parts.append(
                    "Model calibration was evaluated using apparent estimates (Brier score and Integrated Calibration Index [ICI]); calibration slope and intercept were not reported for in-sample apparent probability estimates."
                )
            else:
                diag_parts.append(
                    "Model calibration was evaluated using the Brier score, calibration slope and intercept, and the Integrated Calibration Index (ICI)."
                )

        diag_parts.append(
            f"All statistical tests were two-tailed (alpha = {alpha}), and analyses were conducted using {software_name}."
        )
        return "\n\n".join(diag_parts)

    paragraphs = []

    # 1. Descriptive stats sentence
    if include_descriptive:
        desc_parts = []
        if normality_test:
            desc_parts.append(
                "Continuous variables were assessed for distributional normality using the Shapiro-Wilk test and visual inspection of Q-Q plots."
            )
        desc_parts.append(
            "Normally distributed continuous variables were expressed as mean ± standard deviation (SD) and compared using Student's or Welch's t-test, "
            "whereas skewed variables were reported as median with interquartile range (IQR) and compared using the Mann-Whitney U test."
        )
        if fisher_exact:
            desc_parts.append(
                "Categorical variables were summarized as counts and percentages [n (%)] and compared across groups using the Pearson chi-square test or Fisher's exact test when expected cell counts were below 5."
            )
        else:
            desc_parts.append(
                "Categorical variables were summarized as counts and percentages [n (%)] and compared across groups using the Pearson chi-square test."
            )
        paragraphs.append(" ".join(desc_parts))

    # 2. Primary modeling sentence
    cov_str = (
        f"adjusted for {', '.join(covariates)}"
        if covariates
        else "in unadjusted models"
    )
    exp_str = f"the association between {exposure} and " if exposure else ""
    out_str = f"{outcome}" if outcome else "the primary outcome"

    norm_mtype = model_type.lower()
    if "cox" in norm_mtype or "survival" in norm_mtype:
        norm_mtype = "cox"
    elif "logistic" in norm_mtype or "logit" in norm_mtype or norm_mtype == "binary":
        norm_mtype = "logistic"
    elif "linear" in norm_mtype or "ols" in norm_mtype:
        norm_mtype = "linear"
    elif "poisson" in norm_mtype:
        norm_mtype = "poisson"

    # Check for specialized causal/competing risk models
    if "iptw" in norm_mtype:
        return generate_iptw_narrative(
            exposure=exposure,
            outcome=outcome,
            covariates=covariates,
            alpha=alpha,
            software_name=software_name,
            **kwargs,
        )
    if any(k in norm_mtype for k in ("competing", "cif", "fine_gray", "fine-gray")):
        return generate_competing_risks_narrative(
            exposure=exposure,
            outcome=outcome,
            covariates=covariates,
            alpha=alpha,
            software_name=software_name,
            **kwargs,
        )

    if norm_mtype == "cox":
        if is_firth:
            ph_sentence = (
                " Proportional hazards assumptions were evaluated using Schoenfeld residual tests."
                if check_schoenfeld
                else ""
            )
            paragraphs.append(
                f"To evaluate {exp_str}{out_str}, multivariable Cox proportional hazards regression with Firth's penalized partial likelihood "
                f"was performed to reduce small-sample bias and address potential monotone likelihood or separation ({cov_str}). "
                f"Parameter estimates are presented as hazard ratios (HR) with 95% profile likelihood confidence intervals (CIs).{ph_sentence}"
            )
        else:
            ph_sentence = (
                " The proportional hazards assumption was assessed across all covariates via Schoenfeld residual correlation tests."
                if check_schoenfeld
                else ""
            )
            paragraphs.append(
                f"Multivariable Cox proportional hazards regression was used to evaluate {exp_str}{out_str} ({cov_str}). "
                f"Results are reported as hazard ratios (HR) and 95% confidence intervals (CIs).{ph_sentence}"
            )
    elif norm_mtype == "logistic":
        if is_firth:
            paragraphs.append(
                f"Multivariable logistic regression with Firth's penalized likelihood was fitted to estimate {exp_str}{out_str} ({cov_str}). "
                f"Firth's penalization was selected to control for finite-sample bias and potential quasi-complete separation. "
                f"Results are presented as odds ratios (OR) with 95% profile likelihood confidence intervals and penalized likelihood ratio test p-values."
            )
        else:
            paragraphs.append(
                f"Multivariable logistic regression was conducted to assess {exp_str}{out_str} ({cov_str}). "
                f"Effect estimates are presented as odds ratios (OR) with corresponding 95% confidence intervals."
            )
    else:
        paragraphs.append(
            f"Multivariable {model_type} regression was conducted ({cov_str}). "
            f"Parameter estimates are reported with 95% confidence intervals."
        )

    # 3. Splines sentence if applicable
    if is_rcs:
        paragraphs.append(
            "Potential non-linear dose-response relationships for continuous predictors were modeled using restricted cubic splines "
            "with natural boundary knots. Contrast curves relative to the median reference value were plotted with 95% confidence bands."
        )

    # 4. Missing data sentence
    norm_strat = (missing_strategy or "").lower().replace("_", "-")
    if norm_strat == "complete-case":
        paragraphs.append(
            "Missing data were managed via complete-case analysis, and sample retention was audited from initial enrollment to final analytic cohort."
        )
    elif norm_strat == "mice":
        paragraphs.append(
            "Missing covariate values were imputed using Multiple Imputation by Chained Equations (MICE) under the missing-at-random (MAR) assumption. "
            "Estimates and standard errors across imputed datasets were pooled using Rubin's rules."
        )
    elif norm_strat == "knn":
        paragraphs.append(
            "Missing covariate values were imputed using k-nearest neighbors (KNN) imputation based on Euclidean distance across normalized observed features."
        )
    elif norm_strat == "indicator":
        paragraphs.append(
            "Missing categorical covariates were encoded using missing-indicator categories to retain observations with incomplete data in multivariable models."
        )

    # 5. Significance & Software sentence
    paragraphs.append(
        f"All statistical tests were two-tailed, and p-values < {alpha} were considered statistically significant. "
        f"Statistical calculations were executed using {software_name} (Python biostatistical computing environment)."
    )

    return "\n\n".join(paragraphs)


def generate_iptw_narrative(
    estimand: str = "ATE",
    exposure: str | None = None,
    outcome: str | None = None,
    covariates: list[str] | None = None,
    n_eff: dict[str, float] | float | None = None,
    max_smd: float | None = None,
    stabilized: bool = True,
    diagnostics: dict[str, Any] | None = None,
    alpha: float = 0.05,
    software_name: str = "medstat-core",
    **kwargs: Any,
) -> str:
    """
    Generate automated clinical Methods and Results narrative for IPTW causal inference analysis.
    """
    paragraphs: list[str] = []
    cov_str = (
        f"adjusting for {', '.join(covariates)}"
        if covariates
        else "adjusting for baseline covariates"
    )
    exp_str = exposure or "treatment assignment"
    out_str = outcome or "primary clinical outcome"

    estimand_u = estimand.upper()
    if estimand_u == "ATT":
        est_desc = "Average Treatment Effect on the Treated (ATT), reweighting control participants to reflect baseline characteristics of the treated cohort"
    elif estimand_u == "ATC":
        est_desc = "Average Treatment Effect on the Controls (ATC), reweighting treated participants to match the control cohort profile"
    elif estimand_u == "ATO":
        est_desc = "Overlap Weights (ATO; Li et al., 2018), evaluating the causal effect among patients at clinical equipoise and achieving exact mean balance"
    else:
        est_desc = "Average Treatment Effect (ATE), reweighting the entire study cohort to a pseudo-population where treatment assignment is independent of measured baseline confounders"

    p1 = (
        f"To evaluate the causal effect of {exp_str} on {out_str}, Inverse Probability of Treatment Weighting (IPTW) was performed "
        f"based on propensity scores estimated via multivariable logistic regression ({cov_str}). "
        f"The target estimand was the {est_desc}."
    )
    if stabilized and estimand_u != "ATO":
        p1 += " Stabilized weights were computed to preserve the nominal sample size and prevent variance inflation."

    if diagnostics:
        ext_n = diagnostics.get("extreme_weights_gt_10", 0)
        if ext_n > 0:
            p1 += f" A total of {ext_n} observations exhibited extreme weights (> 10.0), which were truncated at pre-specified quantiles."
        pos_n = diagnostics.get("positivity_violations", 0)
        if pos_n > 0:
            p1 += f" Positivity inspection identified {pos_n} subjects with near-deterministic propensity scores (< 0.05 or > 0.95)."

    paragraphs.append(p1)

    # Effective Sample Size and SMD Balance
    p2_parts = []
    if n_eff is not None:
        if isinstance(n_eff, dict):
            tot = n_eff.get("total", "N/A")
            tx = n_eff.get("treated", "N/A")
            ctrl = n_eff.get("control", "N/A")
            p2_parts.append(
                f"Statistical precision following weighting was quantified using Kish's Effective Sample Size "
                f"(total N_eff = {tot}; treated N_eff = {tx}; control N_eff = {ctrl})."
            )
        else:
            p2_parts.append(
                f"Statistical precision following weighting was evaluated using Kish's Effective Sample Size (N_eff = {n_eff:.1f})."
            )

    if max_smd is not None:
        bal_status = (
            "indicating acceptable covariate balance across all confounders (maximum SMD < 0.10; Austin 2009)"
            if max_smd < 0.10
            else f"indicating residual baseline imbalance (maximum SMD = {max_smd:.3f} >= 0.10), warranting doubly robust multivariable adjustment"
        )
        p2_parts.append(
            f"Post-weighting baseline balance was verified using Standardized Mean Differences (SMDs); "
            f"the maximum post-weighting SMD was {max_smd:.3f}, {bal_status}."
        )

    p2_parts.append(
        "Weighted outcome models were fitted using robust Huber-White sandwich standard errors "
        "to ensure nominal confidence interval coverage in the weighted pseudo-population."
    )
    paragraphs.append(" ".join(p2_parts))

    paragraphs.append(
        f"All statistical tests were two-tailed, and p-values < {alpha} were considered statistically significant. "
        f"Calculations were executed using {software_name}."
    )
    return "\n\n".join(paragraphs)


def generate_competing_risks_narrative(
    exposure: str | None = None,
    outcome: str | None = None,
    competing_name: str | None = None,
    covariates: list[str] | None = None,
    competing_pct: float | None = None,
    overestimation_km: float | None = None,
    cshr_dict: dict[str, Any] | None = None,
    fg_dict: dict[str, Any] | None = None,
    alpha: float = 0.05,
    software_name: str = "medstat-core",
    **kwargs: Any,
) -> str:
    """
    Generate automated clinical Methods and Results narrative for Competing Risks Survival Analysis.
    """
    paragraphs: list[str] = []
    out_str = outcome or "the primary clinical event"
    comp_str = competing_name or "competing events"
    exp_str = f"the association between {exposure} and " if exposure else ""
    cov_str = (
        f"adjusting for {', '.join(covariates)}"
        if covariates
        else "adjusting for baseline covariates"
    )

    p1 = (
        f"Under competing risks between {out_str} and {comp_str}, cumulative incidence functions (CIF) "
        f"were estimated non-parametrically using the Aalen-Johansen estimator with Greenwood 95% confidence intervals, "
        f"strictly satisfying probability conservation across all horizons."
    )

    if (competing_pct is not None and competing_pct > 10.0) or (
        overestimation_km is not None and overestimation_km > 0.05
    ):
        bias_desc = (
            f"exhibited substantial overestimation bias ({overestimation_km * 100:.1f} percentage points higher than Aalen-Johansen CIF)"
            if overestimation_km
            else "substantially overestimated cumulative incidence"
        )
        pct_desc = (
            f"Competing events occurred in {competing_pct:.1f}% of patients. "
            if competing_pct
            else ""
        )
        p1 += (
            f" {pct_desc}The naive 1 - Kaplan-Meier estimator {bias_desc} by treating competing events as uninformative right-censoring, "
            f"which estimates event risk in a hypothetical world where competing events do not exist rather than real-world cumulative incidence."
        )

    paragraphs.append(p1)

    # Dual regression framework: Cause-Specific vs Fine-Gray
    p2_parts = [
        f"To evaluate {exp_str}{out_str} ({cov_str}), multivariable competing risks regression was performed "
        f"using both Cause-Specific Cox Proportional Hazards regression (csHR) and Fine-Gray Subdistribution Hazard regression (SHR).",
        "The Cause-Specific HR treats competing events as censored at failure time, measuring the instantaneous relative rate "
        "among individuals currently event-free to answer biological and etiological questions.",
        "In contrast, the Fine-Gray SHR maintains competing-event subjects in the risk set with decaying Inverse Probability of Censoring Weights (IPCW), "
        "directly quantifying overall cumulative incidence to guide clinical prognosis and risk prediction.",
    ]
    if cshr_dict and fg_dict:
        cs_val = cshr_dict.get("hr", cshr_dict.get("estimate", "N/A"))
        cs_ci = cshr_dict.get("ci", "N/A")
        fg_val = fg_dict.get("shr", fg_dict.get("estimate", "N/A"))
        fg_ci = fg_dict.get("ci", "N/A")
        p2_parts.append(
            f"Estimated Cause-Specific HR was {cs_val} (95% CI {cs_ci}), compared to Fine-Gray SHR of {fg_val} (95% CI {fg_ci})."
        )

    paragraphs.append(" ".join(p2_parts))

    paragraphs.append(
        f"All statistical tests were two-tailed with significance defined at p < {alpha}. "
        f"Calculations were performed using {software_name}."
    )
    return "\n\n".join(paragraphs)


def generate_narrative(
    table: Any = None,
    style: str = "nejm",
    model_type: str = "logistic",
    **kwargs: Any,
) -> str:
    """
    Generate an academic Methods / Results narrative for an EstimateTable or model parameters.
    """
    if table is not None and hasattr(table, "meta") and table.meta is not None:
        meta = table.meta
        mtype_raw = str(
            getattr(meta, "estimator", getattr(meta, "model_type", "regression"))
        ).lower()
        if "iptw" in mtype_raw:
            mtype = "iptw"
        elif any(
            k in mtype_raw for k in ("competing", "cif", "fine_gray", "fine-gray")
        ):
            mtype = "competing_risks"
        elif "cox" in mtype_raw or "survival" in mtype_raw:
            mtype = "cox"
        elif "logistic" in mtype_raw or "logit" in mtype_raw or mtype_raw == "binary":
            mtype = "logistic"
        elif "linear" in mtype_raw or "ols" in mtype_raw:
            mtype = "linear"
        elif "poisson" in mtype_raw:
            mtype = "poisson"
        else:
            mtype = mtype_raw

        dep_var = getattr(
            meta, "outcome_name", getattr(meta, "dependent_var", "primary outcome")
        )
        exp_var = getattr(meta, "exposure_name", getattr(meta, "exposure", None))
        covars = getattr(meta, "adjusted_for", None)
        if not covars and hasattr(table, "rows"):
            covars = [
                getattr(
                    row, "term", getattr(row, "label", getattr(row, "variable", ""))
                )
                for row in table.rows
                if getattr(
                    row, "term", getattr(row, "label", getattr(row, "variable", ""))
                )
                not in ("(Intercept)", "const", "Intercept")
            ]
        return generate_methods_narrative(
            model_type=mtype,
            exposure=exp_var,
            outcome=dep_var,
            covariates=covars,
            **kwargs,
        )
    return generate_methods_narrative(model_type=model_type, **kwargs)

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
) -> dict[str, float]:
    """
    Validate calibration metrics: reject booleans and non-finite values.
    For apparent provenance, omit calibration slope (apparent slope is 1.0 / tautological).
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

    if provenance == "apparent":
        filtered = {k: float(v) for k, v in metrics.items() if k.lower() != "slope"}
        if not filtered:
            raise ValueError(
                "Apparent validation requires at least one non-slope calibration metric (e.g., Brier score)."
            )
        return filtered
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
        if "cox" in mtype_raw or "survival" in mtype_raw:
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

"""
Automated Statistical Methods Narrative Generator.

Drafts publication-grade Statistical Methods section narratives conforming
to STROBE and CONSORT reporting guidelines.
"""

from __future__ import annotations

from typing import Any


def generate_methods_narrative(
    model_type: str = "logistic",
    exposure: str | None = None,
    outcome: str | None = None,
    covariates: list[str] | None = None,
    missing_strategy: str = "complete-case",
    is_firth: bool = False,
    is_rcs: bool = False,
    alpha: float = 0.05,
    software_name: str = "medstat-core",
) -> str:
    """
    Generate an academic Statistical Methods section paragraph.

    Parameters:
        model_type: 'logistic', 'cox', 'linear', 'poisson', etc.
        exposure: Primary exposure/treatment variable.
        outcome: Primary clinical outcome.
        covariates: List of confounders adjusted for.
        missing_strategy: Strategy used to handle missing data ('complete-case', 'mice', etc.).
        is_firth: Whether Firth penalized likelihood was used.
        is_rcs: Whether restricted cubic splines were fitted for non-linear terms.
        alpha: Significance threshold (default 0.05).
        software_name: Software package citation.

    Returns:
        Formatted English narrative paragraph.
    """
    paragraphs = []

    # 1. Descriptive stats sentence
    paragraphs.append(
        "Continuous variables were assessed for distributional normality using the Shapiro-Wilk test and visual inspection of Q-Q plots. "
        "Normally distributed continuous variables were expressed as mean ± standard deviation (SD) and compared using Student's or Welch's t-test, "
        "whereas skewed variables were reported as median with interquartile range (IQR) and compared using the Mann-Whitney U test. "
        "Categorical variables were summarized as counts and percentages [n (%)] and compared across groups using the Pearson chi-square test or Fisher's exact test when expected cell counts were below 5."
    )

    # 2. Primary modeling sentence
    cov_str = (
        f"adjusted for {', '.join(covariates)}"
        if covariates
        else "in unadjusted models"
    )
    exp_str = f"the association between {exposure} and " if exposure else ""
    out_str = f"{outcome}" if outcome else "the primary outcome"

    if model_type.lower() == "cox":
        if is_firth:
            paragraphs.append(
                f"To evaluate {exp_str}{out_str}, multivariable Cox proportional hazards regression with Firth's penalized partial likelihood "
                f"was performed to reduce small-sample bias and address potential monotone likelihood or separation ({cov_str}). "
                f"Parameter estimates are presented as hazard ratios (HR) with 95% profile likelihood confidence intervals (CIs). "
                f"Proportional hazards assumptions were evaluated using Schoenfeld residual tests."
            )
        else:
            paragraphs.append(
                f"Multivariable Cox proportional hazards regression was used to evaluate {exp_str}{out_str} ({cov_str}). "
                f"Results are reported as hazard ratios (HR) and 95% confidence intervals (CIs). "
                f"The proportional hazards assumption was verified across all covariates via Schoenfeld residual correlation tests."
            )
    elif model_type.lower() == "logistic":
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
    if missing_strategy == "complete-case":
        paragraphs.append(
            "Missing data were managed via complete-case analysis, and sample retention was audited from initial enrollment to final analytic cohort."
        )
    elif missing_strategy == "mice":
        paragraphs.append(
            "Missing covariate values were imputed using Multiple Imputation by Chained Equations (MICE) under the missing-at-random (MAR) assumption. "
            "Estimates and standard errors across imputed datasets were pooled using Rubin's rules."
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
        mtype = getattr(meta, "estimator", getattr(meta, "model_type", "regression"))
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

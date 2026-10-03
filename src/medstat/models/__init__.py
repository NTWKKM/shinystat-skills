"""
Statistical Modeling and Regression Module.
"""

from typing import Any

import numpy as np

from medstat.models.firth import check_separation, fit_firth_cox, fit_firth_logistic
from medstat.models.glm import (
    fit_linear_regression,
    fit_negative_binomial,
    fit_poisson_regression,
    fit_standard_logistic,
)
from medstat.models.multilevel import (
    calculate_design_effect,
    fit_gee,
    fit_random_intercept,
)
from medstat.models.ordinal import (
    fit_multinomial_logistic,
    fit_proportional_odds,
    test_proportional_odds,
)
from medstat.models.sensitivity import bootstrap_confidence_interval, calculate_e_value
from medstat.models.splines import fit_cox_rcs
from medstat.models.survival import (
    compare_survival_curves,
    fit_cox_ph,
    fit_kaplan_meier,
)


def extract_primary_effect(
    estimates_map: dict[str, Any],
    primary_var: str = "treatment",
    scale: str = "ratio",
    estimate_type: str | None = None,
) -> float:
    """
    Extract primary exposure/treatment effect estimate (e.g. OR, HR, coef) from estimates dict,
    handling exact matches, dummy encodings ('treatment[T.1]'), CLI dummies ('treatment_1'),
    or Patsy syntax ('C(treatment)[T.1]').

    Preserves negative values for coefficient-scale estimates, while requiring positive values
    for ratio-scale estimates. Detects multiple matching terms and returns np.nan for ambiguous matches.
    """
    effective_scale = (estimate_type or scale).lower()

    exact_terms = [term for term in estimates_map.keys() if str(term) == primary_var]
    matching_terms = exact_terms or [
        term
        for term in estimates_map.keys()
        if (
            str(term).startswith(f"{primary_var}[")
            or str(term).startswith(f"C({primary_var})[")
            or str(term).startswith(f"{primary_var}_")
        )
    ]

    # Ambiguous matches: multiple candidate terms detected
    if len(matching_terms) != 1:
        return np.nan

    val = estimates_map[matching_terms[0]]
    if val is None or not np.isfinite(val):
        return np.nan

    val_float = float(val)
    if effective_scale in ("ratio", "or", "hr", "rr"):
        return val_float if val_float > 0 else np.nan
    return val_float


__all__ = [
    "fit_linear_regression",
    "fit_standard_logistic",
    "fit_poisson_regression",
    "fit_negative_binomial",
    "fit_firth_logistic",
    "fit_firth_cox",
    "check_separation",
    "fit_kaplan_meier",
    "compare_survival_curves",
    "fit_cox_ph",
    "fit_cox_rcs",
    "fit_proportional_odds",
    "test_proportional_odds",
    "fit_multinomial_logistic",
    "calculate_design_effect",
    "fit_gee",
    "fit_random_intercept",
    "calculate_e_value",
    "bootstrap_confidence_interval",
    "extract_primary_effect",
]

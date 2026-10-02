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
) -> float:
    """
    Extract primary exposure/treatment effect estimate (e.g. OR, HR, coef) from estimates dict,
    handling exact matches, dummy encodings ('treatment[T.1]'), or Patsy syntax ('C(treatment)[T.1]').
    """
    for term, val in estimates_map.items():
        if (
            term == primary_var
            or term.startswith(f"{primary_var}[")
            or term.startswith(f"C({primary_var})[")
        ):
            if val is not None and np.isfinite(val) and val > 0:
                return float(val)
    return np.nan


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

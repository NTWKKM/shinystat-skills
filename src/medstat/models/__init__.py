"""
Statistical Modeling and Regression Module.
"""

from medstat.models.firth import check_separation, fit_firth_cox, fit_firth_logistic
from medstat.models.glm import (
    fit_linear_regression,
    fit_negative_binomial,
    fit_poisson_regression,
    fit_standard_logistic,
)
from medstat.models.sensitivity import bootstrap_confidence_interval, calculate_e_value
from medstat.models.splines import fit_cox_rcs
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
from medstat.models.survival import (
    compare_survival_curves,
    fit_cox_ph,
    fit_kaplan_meier,
)

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
]

"""
Causal Inference, Propensity Score Matching, and Balance Diagnostics.
"""

from medstat.causal.balance import (
    calculate_smd,
    check_balance,
    compare_pre_post_balance,
    compute_balance_table,
    create_love_plot,
)
from medstat.causal.iptw import (
    compute_iptw_weights,
    fit_iptw_outcome_model,
)
from medstat.causal.psm import (
    calculate_ipw,
    calculate_propensity_score,
    perform_matching,
)

__all__ = [
    "calculate_propensity_score",
    "perform_matching",
    "calculate_ipw",
    "compute_iptw_weights",
    "fit_iptw_outcome_model",
    "calculate_smd",
    "check_balance",
    "compute_balance_table",
    "compare_pre_post_balance",
    "create_love_plot",
]

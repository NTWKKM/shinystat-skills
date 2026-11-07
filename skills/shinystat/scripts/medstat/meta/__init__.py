"""
Meta-Analysis, Heterogeneity, and Evidence Synthesis Module.
"""

from medstat.meta.bias import (
    beggs_test,
    create_funnel_plot,
    eggers_test,
    run_publication_bias_tests,
)
from medstat.meta.forest import create_forest_plot, generate_forest_data
from medstat.meta.models import (
    compute_binary_effect_sizes,
    compute_continuous_effect_sizes,
    run_meta_analysis,
)

__all__ = [
    "compute_binary_effect_sizes",
    "compute_continuous_effect_sizes",
    "run_meta_analysis",
    "generate_forest_data",
    "create_forest_plot",
    "eggers_test",
    "beggs_test",
    "run_publication_bias_tests",
    "create_funnel_plot",
]

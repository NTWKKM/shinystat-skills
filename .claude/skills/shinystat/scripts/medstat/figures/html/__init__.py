"""
src/medstat/figures/html: HTML Interactive and Dual-View Figure generators.
Provides companion Plotly interactive figures alongside 300 DPI Matplotlib figures.
"""

from __future__ import annotations

from medstat.figures.html.agreement import plot_dual_bland_altman
from medstat.figures.html.balance import plot_dual_love
from medstat.figures.html.base import DualFigureResult
from medstat.figures.html.calibration import (
    plot_bootstrap_optimism,
    plot_dual_bootstrap_optimism,
    plot_dual_calibration,
    plot_dual_recalibration_comparison,
    plot_recalibration_comparison,
)
from medstat.figures.html.dca import plot_dual_dca
from medstat.figures.html.forest import plot_dual_forest
from medstat.figures.html.roc import (
    plot_dual_reclassification,
    plot_dual_roc_curve,
    plot_dual_time_dependent_roc,
)
from medstat.figures.html.survival import plot_dual_cif, plot_dual_kaplan_meier

__all__ = [
    "DualFigureResult",
    "plot_dual_roc_curve",
    "plot_dual_time_dependent_roc",
    "plot_dual_reclassification",
    "plot_dual_kaplan_meier",
    "plot_dual_cif",
    "plot_dual_calibration",
    "plot_dual_recalibration_comparison",
    "plot_dual_bootstrap_optimism",
    "plot_recalibration_comparison",
    "plot_bootstrap_optimism",
    "plot_dual_dca",
    "plot_dual_bland_altman",
    "plot_dual_forest",
    "plot_dual_love",
]

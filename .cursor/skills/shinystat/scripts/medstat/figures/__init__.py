"""
src/medstat/figures/__init__.py: Standardized figure generation library
for medical and biostatistical publications.
"""

from __future__ import annotations

from medstat.figures.agreement import plot_bland_altman
from medstat.figures.balance import plot_love
from medstat.figures.base import FigureResult, export_source_data
from medstat.figures.calibration import plot_calibration
from medstat.figures.dca import plot_dca
from medstat.figures.diagnostics import (
    plot_mice_diagnostics,
    plot_missingness_map,
    plot_schoenfeld_residuals,
)
from medstat.figures.forest import plot_forest
from medstat.figures.retention import plot_retention_flow
from medstat.figures.roc import plot_roc_curve
from medstat.figures.styles import (
    CLINICAL_PALETTE,
    GROUP_COLORS,
    save_publication_figure,
    set_clinical_figure_style,
)
from medstat.figures.survival import plot_kaplan_meier

__all__ = [
    "FigureResult",
    "export_source_data",
    "CLINICAL_PALETTE",
    "GROUP_COLORS",
    "set_clinical_figure_style",
    "save_publication_figure",
    "plot_forest",
    "plot_kaplan_meier",
    "plot_roc_curve",
    "plot_calibration",
    "plot_dca",
    "plot_bland_altman",
    "plot_love",
    "plot_retention_flow",
    "plot_missingness_map",
    "plot_schoenfeld_residuals",
    "plot_mice_diagnostics",
]

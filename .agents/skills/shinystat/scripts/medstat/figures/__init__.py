"""
src/medstat/figures/__init__.py: Standardized figure generation library
for medical and biostatistical publications.
"""

from __future__ import annotations

from medstat.figures.agreement import plot_bland_altman
from medstat.figures.balance import plot_love
from medstat.figures.base import FigureResult, export_source_data
from medstat.figures.calibration import (
    plot_bootstrap_optimism,
    plot_calibration,
    plot_recalibration_comparison,
)
from medstat.figures.dca import plot_dca
from medstat.figures.diagnostics import (
    plot_mice_diagnostics,
    plot_missingness_map,
    plot_schoenfeld_residuals,
)
from medstat.figures.forest import plot_forest
from medstat.figures.journal import (
    JOURNAL_SPECS,
    TARGET_WIDTHS,
    apply_journal_theme,
    format_estimate,
    format_num,
    format_p_value,
    style_axis,
)
from medstat.figures.qa import (
    TISSUE_MUST_DIFFER,
    assert_zero_phi_text,
    cvd_check,
    qa_layout,
)
from medstat.figures.retention import plot_retention_flow
from medstat.figures.roc import (
    plot_reclassification,
    plot_roc_curve,
    plot_time_dependent_roc,
)
from medstat.figures.styles import (
    CLINICAL_PALETTE,
    GROUP_COLORS,
    save_figure_bundle,
    save_publication_figure,
    set_clinical_figure_style,
)
from medstat.figures.survival import plot_cif, plot_kaplan_meier

__all__ = [
    "FigureResult",
    "export_source_data",
    "CLINICAL_PALETTE",
    "GROUP_COLORS",
    "JOURNAL_SPECS",
    "TARGET_WIDTHS",
    "apply_journal_theme",
    "format_p_value",
    "format_estimate",
    "format_num",
    "style_axis",
    "qa_layout",
    "cvd_check",
    "assert_zero_phi_text",
    "TISSUE_MUST_DIFFER",
    "set_clinical_figure_style",
    "save_publication_figure",
    "save_figure_bundle",
    "plot_forest",
    "plot_kaplan_meier",
    "plot_cif",
    "plot_roc_curve",
    "plot_time_dependent_roc",
    "plot_reclassification",
    "plot_calibration",
    "plot_recalibration_comparison",
    "plot_bootstrap_optimism",
    "plot_dca",
    "plot_bland_altman",
    "plot_love",
    "plot_retention_flow",
    "plot_missingness_map",
    "plot_schoenfeld_residuals",
    "plot_mice_diagnostics",
]

"""
src/medstat/figures/html/base.py: DualFigureResult contract for HTML interactive figures.
Inherits from standard FigureResult to ensure 100% compatibility with ReportDocument and IR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from medstat.figures.base import FigureResult


@dataclass
class DualFigureResult(FigureResult):
    """
    Subclasses FigureResult with optional Plotly interactive specification.
    Preserves all existing attributes (png_path, alt_text, caption, source_df, csv_path).
    """

    plotly_json: str | None = None
    plotly_fig: Any | None = None

    def __post_init__(self) -> None:
        if self.plotly_fig is not None and self.plotly_json is None:
            if hasattr(self.plotly_fig, "to_json"):
                self.plotly_json = self.plotly_fig.to_json()
            elif isinstance(self.plotly_fig, str):
                self.plotly_json = self.plotly_fig

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["has_interactive"] = bool(self.plotly_json)
        return d

    @property
    def plotly_spec(self) -> dict[str, Any] | None:
        """Returns the dictionary specification of the Plotly figure."""
        if self.plotly_fig is not None and hasattr(self.plotly_fig, "to_dict"):
            return self.plotly_fig.to_dict()
        if self.plotly_json is not None:
            import json

            try:
                return json.loads(self.plotly_json)
            except Exception:
                pass
        return None

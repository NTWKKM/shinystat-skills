"""
src/medstat/figures/base.py: FigureResult contract and source data exporter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class FigureResult:
    """Standard contract returned by every figure generator."""

    png_path: str
    alt_text: str
    caption: str
    source_df: pd.DataFrame
    csv_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "png_path": self.png_path,
            "alt_text": self.alt_text,
            "caption": self.caption,
            "csv_path": self.csv_path,
            "rows": len(self.source_df),
        }


def export_source_data(df: pd.DataFrame, out_png_path: str | Path) -> str:
    """Exports source dataframe to CSV with matching basename for reproducibility."""
    png_p = Path(out_png_path).resolve()
    csv_p = png_p.with_suffix(".csv")
    csv_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_p, index=False)
    return str(csv_p)

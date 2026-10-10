"""
src/medstat/reporting/ir.py: Report Intermediate Representation (IR).

Format-agnostic document schema acting as the Single Source of Truth (SSOT).
Every number in paragraph blocks must trace back to the results dictionary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from medstat.figures import FigureResult


@dataclass
class Block:
    """Base class for all document blocks."""

    pass


@dataclass
class HeadingBlock(Block):
    text: str
    level: int = 1  # 1 to 4


@dataclass
class ParagraphBlock(Block):
    text: str


@dataclass
class TableBlock(Block):
    df: pd.DataFrame
    caption: str
    footnote: str | None = None
    style: str = "nejm"  # "nejm", "jama", "apa"


@dataclass
class FigureBlock(Block):
    figure: FigureResult


@dataclass
class CalloutBlock(Block):
    text: str
    level: Literal["info", "warning", "note"] = "info"


@dataclass
class ReportDocument:
    """
    Format-agnostic document model containing structured blocks
    and the source results dictionary for verification.
    """

    title: str
    authors: list[str] = field(default_factory=list)
    date: str = ""
    blocks: list[Block] = field(default_factory=list)
    results_dict: dict[str, Any] = field(default_factory=dict)
    institution: str = "Biostatistics & Clinical Research Unit"

    def add_heading(self, text: str, level: int = 1) -> ReportDocument:
        self.blocks.append(HeadingBlock(text=text, level=level))
        return self

    def add_paragraph(self, text: str) -> ReportDocument:
        self.blocks.append(ParagraphBlock(text=text))
        return self

    def add_table(
        self,
        df: pd.DataFrame,
        caption: str,
        footnote: str | None = None,
        style: str = "nejm",
    ) -> ReportDocument:
        self.blocks.append(
            TableBlock(df=df, caption=caption, footnote=footnote, style=style)
        )
        return self

    def add_figure(self, figure: FigureResult) -> ReportDocument:
        self.blocks.append(FigureBlock(figure=figure))
        return self

    def add_callout(
        self, text: str, level: Literal["info", "warning", "note"] = "info"
    ) -> ReportDocument:
        self.blocks.append(CalloutBlock(text=text, level=level))
        return self

    def extract_all_text(self) -> str:
        """Extracts all narrative text and headings into a single plain text string."""
        lines = [self.title]
        for b in self.blocks:
            if isinstance(b, HeadingBlock):
                lines.append(b.text)
            elif isinstance(b, ParagraphBlock):
                lines.append(b.text)
            elif isinstance(b, CalloutBlock):
                lines.append(b.text)
            elif isinstance(b, TableBlock):
                lines.append(b.caption)
                if b.footnote:
                    lines.append(b.footnote)
            elif isinstance(b, FigureBlock):
                lines.append(b.figure.caption)
                if b.figure.alt_text:
                    lines.append(b.figure.alt_text)
        return "\n\n".join(lines)

"""
Publication Table Renderers (NEJM, JAMA, APA 7).

Formats biostatistical results and regression estimates into publication-standard
HTML tables and clean ASCII text.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Estimate:
    term: str
    label: str
    estimate: float
    ci_lower: float
    ci_upper: float
    p_value: float
    scale: str = "OR"  # "OR", "HR", "RR", "Beta", "IRR", "MD"
    reference: bool = False
    ref_label: str = "Reference"
    std_error: float | None = None
    n: int | None = None
    events: int | None = None


@dataclass
class ModelMeta:
    estimator: str
    n_total: int
    n_events: int | None = None
    outcome_name: str | None = None
    exposure_name: str | None = None
    adjusted_for: list[str] = field(default_factory=list)
    software_version: str = (
        "medstat-core (Python 3.12+ statsmodels/lifelines/firthmodels)"
    )
    seed: int | None = None
    notes: str | None = None


@dataclass
class EstimateTable:
    title: str
    rows: list[Estimate]
    meta: ModelMeta | None = None
    confidence_level: float = 0.95


def format_journal_p_value(p: float, style: str = "NEJM") -> str:
    """
    Format p-value per journal conventions.
    """
    if p is None or math.isnan(p):
        return "—"

    if style.upper() in ("NEJM", "JAMA"):
        if p < 0.001:
            return "P<0.001"
        elif p < 0.01:
            return f"P={p:.3f}"
        elif p >= 0.99:
            return "P>0.99"
        else:
            return f"P={p:.2f}"
    else:  # APA 7
        if p < 0.001:
            return "< .001"
        elif p < 0.01:
            return f"{p:.3f}".lstrip("0")
        elif p >= 0.99:
            return "> .99"
        else:
            return f"{p:.2f}".lstrip("0")


class PublicationRenderer:
    """
    Renders EstimateTable instances into publication-grade HTML tables.
    """

    @classmethod
    def render_html(
        cls, table: EstimateTable, style: Literal["NEJM", "JAMA", "APA7"] = "NEJM"
    ) -> str:
        s = style.upper()
        if s == "APA7" or s == "APA":
            return cls._render_apa7(table)
        elif s == "JAMA":
            return cls._render_jama(table)
        else:
            return cls._render_nejm(table)

    @classmethod
    def _render_nejm(cls, table: EstimateTable) -> str:
        pct_ci = f"{int(table.confidence_level * 100)}%"
        scale_label = table.rows[0].scale if table.rows else "Estimate"

        html_lines = [
            "<table style='border-collapse: collapse; width: 100%; font-family: -apple-system, sans-serif; font-size: 14px;'>",
            f"  <caption style='caption-side: top; text-align: left; font-weight: bold; margin-bottom: 8px;'>{html.escape(table.title)}</caption>",
            "  <thead>",
            "    <tr style='border-top: 2px solid #000; border-bottom: 1px solid #000;'>",
            "      <th style='text-align: left; padding: 6px 12px;'>Variable</th>",
            f"      <th style='text-align: right; padding: 6px 12px;'>{scale_label} ({pct_ci} CI)</th>",
            "      <th style='text-align: right; padding: 6px 12px;'>P Value</th>",
            "    </tr>",
            "  </thead>",
            "  <tbody>",
        ]

        for r in table.rows:
            if r.reference:
                est_str = r.ref_label
                p_str = "—"
            else:
                est_str = f"{r.estimate:.2f} ({r.ci_lower:.2f}–{r.ci_upper:.2f})"
                p_str = format_journal_p_value(r.p_value, style="NEJM")

            html_lines.append(
                f"    <tr>\n"
                f"      <td style='padding: 6px 12px;'>{html.escape(r.label)}</td>\n"
                f"      <td style='text-align: right; padding: 6px 12px;'>{est_str}</td>\n"
                f"      <td style='text-align: right; padding: 6px 12px;'>{p_str}</td>\n"
                f"    </tr>"
            )

        html_lines.extend(
            [
                "  </tbody>",
                "  <tfoot>",
                "    <tr style='border-top: 2px solid #000;'>",
                "      <td colspan='3' style='font-size: 12px; color: #555; padding-top: 6px;'>",
            ]
        )

        if table.meta:
            m = table.meta
            foot = f"Model: {m.estimator}, Total N = {m.n_total}"
            if m.n_events is not None:
                foot += f", Events = {m.n_events}"
            if m.adjusted_for:
                foot += f". Adjusted for: {', '.join(m.adjusted_for)}"
            html_lines.append(f"        {html.escape(foot)}")

        html_lines.extend(["      </td>", "    </tr>", "  </tfoot>", "</table>"])
        return "\n".join(html_lines)

    @classmethod
    def _render_jama(cls, table: EstimateTable) -> str:
        # JAMA uses clean borders and slightly compact typography
        return cls._render_nejm(table).replace(
            "caption-side: top", "caption-side: top; color: #111;"
        )

    @classmethod
    def _render_apa7(cls, table: EstimateTable) -> str:
        pct_ci = f"{int(table.confidence_level * 100)}%"
        scale_label = table.rows[0].scale if table.rows else "Estimate"

        html_lines = [
            "<table style='border-collapse: collapse; width: 100%; font-family: Times New Roman, serif; font-size: 14px;'>",
            f"  <caption style='caption-side: top; text-align: left; font-style: italic; margin-bottom: 8px;'>{html.escape(table.title)}</caption>",
            "  <thead>",
            "    <tr style='border-top: 1px solid #000; border-bottom: 1px solid #000;'>",
            "      <th style='text-align: left; padding: 6px 12px; font-weight: normal;'>Predictor</th>",
            f"      <th style='text-align: right; padding: 6px 12px; font-weight: normal;'><em>{scale_label}</em></th>",
            f"      <th style='text-align: right; padding: 6px 12px; font-weight: normal;'>{pct_ci} CI</th>",
            "      <th style='text-align: right; padding: 6px 12px; font-weight: normal;'><em>p</em></th>",
            "    </tr>",
            "  </thead>",
            "  <tbody>",
        ]

        for r in table.rows:
            if r.reference:
                est_str = r.ref_label
                ci_str = "—"
                p_str = "—"
            else:
                est_str = f"{r.estimate:.2f}"
                ci_str = f"[{r.ci_lower:.2f}, {r.ci_upper:.2f}]"
                p_str = format_journal_p_value(r.p_value, style="APA7")

            html_lines.append(
                f"    <tr>\n"
                f"      <td style='padding: 6px 12px;'>{html.escape(r.label)}</td>\n"
                f"      <td style='text-align: right; padding: 6px 12px;'>{est_str}</td>\n"
                f"      <td style='text-align: right; padding: 6px 12px;'>{ci_str}</td>\n"
                f"      <td style='text-align: right; padding: 6px 12px;'>{p_str}</td>\n"
                f"    </tr>"
            )

        html_lines.extend(
            [
                "  </tbody>",
                "  <tfoot>",
                "    <tr style='border-top: 1px solid #000;'>",
                "      <td colspan='4' style='font-size: 12px; font-style: italic; padding-top: 6px;'>",
                "        <em>Note</em>. CI = confidence interval.",
                "      </td>",
                "    </tr>",
                "  </tfoot>",
                "</table>",
            ]
        )
        return "\n".join(html_lines)


def render_nejm_table(table: EstimateTable) -> str:
    """Render table in NEJM publication style HTML."""
    return PublicationRenderer.render_html(table, style="NEJM")


def render_jama_table(table: EstimateTable) -> str:
    """Render table in JAMA publication style HTML."""
    return PublicationRenderer.render_html(table, style="JAMA")


def render_apa_table(table: EstimateTable) -> str:
    """Render table in APA 7 publication style HTML."""
    return PublicationRenderer.render_html(table, style="APA7")


def render_ascii_table(table: EstimateTable) -> str:
    """Render table as plain text ASCII table."""
    headers = [
        "Variable",
        f"{table.rows[0].scale if table.rows else 'Est'} (95% CI)",
        "P-Value",
    ]
    rows = []
    for r in table.rows:
        if r.reference:
            rows.append([r.label, r.ref_label, "—"])
        else:
            rows.append(
                [
                    r.label,
                    f"{r.estimate:.2f} ({r.ci_lower:.2f}, {r.ci_upper:.2f})",
                    format_journal_p_value(r.p_value),
                ]
            )

    col_widths = [max(len(str(x)) for x in col) for col in zip(*([headers] + rows))]
    line = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"

    output = [table.title, line]
    hdr_str = "| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths)) + " |"
    output.extend([hdr_str, line])

    for row in rows:
        row_str = (
            "| " + " | ".join(f"{val:<{w}}" for val, w in zip(row, col_widths)) + " |"
        )
        output.append(row_str)

    output.append(line)
    return "\n".join(output)

"""
src/medstat/reporting/renderers/html.py: Self-contained single-file HTML document renderer.
Embeds figures as inline base64 data and styles publication tables with ICMJE 3-rule borders.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path

import pandas as pd

from medstat.reporting.ir import (
    CalloutBlock,
    FigureBlock,
    HeadingBlock,
    ParagraphBlock,
    ReportDocument,
    TableBlock,
)


def _encode_image_base64(img_path: str | Path) -> str:
    """Reads PNG file and returns data:image/png;base64 string."""
    p = Path(img_path).resolve()
    if not p.exists():
        return ""
    data = p.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def render_html(
    doc: ReportDocument,
    out_path: str | Path,
) -> str:
    """
    Renders ReportDocument into a self-contained, publication-grade HTML document.
    """
    out_p = Path(out_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    escaped_title = html.escape(doc.title)

    html_parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8">',
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        f"  <title>{escaped_title}</title>",
        "  <style>",
        "    body {",
        "      font-family: -apple-system, BlinkMacSystemFont, 'Sarabun', 'Thonburi', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;",
        "      color: #0F172A;",
        "      background-color: #FAFAFA;",
        "      margin: 0;",
        "      padding: 40px 20px;",
        "      line-height: 1.6;",
        "    }",
        "    .container {",
        "      max-width: 900px;",
        "      margin: 0 auto;",
        "      background: #FFFFFF;",
        "      padding: 48px;",
        "      border-radius: 8px;",
        "      box-shadow: 0 1px 3px rgba(0,0,0,0.06);",
        "    }",
        "    h1 { font-size: 26px; font-weight: 700; margin-top: 0; color: #0F172A; border-bottom: 2px solid #0F172A; padding-bottom: 12px; }",
        "    h2 { font-size: 20px; font-weight: 600; margin-top: 32px; margin-bottom: 12px; color: #1E293B; border-bottom: 1px solid #E2E8F0; padding-bottom: 6px; }",
        "    h3 { font-size: 16px; font-weight: 600; margin-top: 24px; color: #334155; }",
        "    p { margin: 12px 0; font-size: 15px; }",
        "    .metadata { color: #64748B; font-size: 13.5px; margin-bottom: 24px; padding-bottom: 12px; }",
        "    .callout {",
        "      background: #F8FAFC;",
        "      border-left: 4px solid #2563EB;",
        "      padding: 12px 18px;",
        "      margin: 20px 0;",
        "      border-radius: 0 6px 6px 0;",
        "      font-size: 14px;",
        "    }",
        "    .pub-table-container { margin: 28px 0; overflow-x: auto; }",
        "    .pub-table-caption { font-weight: 600; font-size: 14.5px; margin-bottom: 8px; color: #0F172A; }",
        "    table.pub-table {",
        "      width: 100%;",
        "      border-collapse: collapse;",
        "      border-top: 2px solid #000000;",
        "      border-bottom: 2px solid #000000;",
        "      font-size: 13.5px;",
        "    }",
        "    table.pub-table th {",
        "      border-bottom: 1px solid #000000;",
        "      padding: 8px 10px;",
        "      text-align: left;",
        "      font-weight: 600;",
        "    }",
        "    table.pub-table td {",
        "      padding: 6px 10px;",
        "      border: none;",
        "    }",
        "    table.pub-table tr:hover { background-color: #F8FAFC; }",
        "    .pub-table-footnote { font-size: 12px; color: #64748B; margin-top: 6px; font-style: italic; }",
        "    .figure-container { margin: 32px 0; text-align: center; }",
        "    .figure-container img { max-width: 100%; height: auto; border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }",
        "    .figure-caption { font-size: 13.5px; color: #334155; margin-top: 10px; text-align: left; }",
        "    @media print {",
        "      body { background: #FFFFFF; padding: 0; }",
        "      .container { box-shadow: none; padding: 0; max-width: 100%; }",
        "    }",
        "  </style>",
        "</head>",
        "<body>",
        '  <div class="container">',
        f"    <h1>{escaped_title}</h1>",
    ]

    # Metadata block
    meta_parts = []
    if doc.authors:
        meta_parts.append(
            f"<strong>Authors:</strong> {html.escape(', '.join(doc.authors))}"
        )
    if doc.date:
        meta_parts.append(f"<strong>Date:</strong> {html.escape(doc.date)}")
    if doc.institution:
        meta_parts.append(
            f"<strong>Affiliation:</strong> {html.escape(doc.institution)}"
        )

    if meta_parts:
        html_parts.append(
            f'    <div class="metadata">{" &bull; ".join(meta_parts)}</div>'
        )

    for block in doc.blocks:
        if isinstance(block, HeadingBlock):
            tag = f"h{max(2, min(6, block.level + 1))}"
            html_parts.append(f"    <{tag}>{html.escape(block.text)}</{tag}>")
        elif isinstance(block, ParagraphBlock):
            html_parts.append(f"    <p>{html.escape(block.text)}</p>")
        elif isinstance(block, CalloutBlock):
            html_parts.append(
                f'    <div class="callout">{html.escape(block.text)}</div>'
            )
        elif isinstance(block, TableBlock):
            caption_esc = html.escape(block.caption)
            html_parts.append('    <div class="pub-table-container">')
            html_parts.append(
                f'      <div class="pub-table-caption">{caption_esc}</div>'
            )
            html_parts.append('      <table class="pub-table">')
            html_parts.append("        <thead>\n          <tr>")
            for col in block.df.columns:
                html_parts.append(f"            <th>{html.escape(str(col))}</th>")
            html_parts.append("          </tr>\n        </thead>\n        <tbody>")
            for _, r in block.df.iterrows():
                html_parts.append("          <tr>")
                for col in block.df.columns:
                    val_str = str(r[col]) if pd.notna(r[col]) else ""
                    html_parts.append(f"            <td>{html.escape(val_str)}</td>")
                html_parts.append("          </tr>")
            html_parts.append("        </tbody>\n      </table>")
            if block.footnote:
                html_parts.append(
                    f'      <div class="pub-table-footnote">{html.escape(block.footnote)}</div>'
                )
            html_parts.append("    </div>")
        elif isinstance(block, FigureBlock):
            b64_src = _encode_image_base64(block.figure.png_path)
            alt_esc = html.escape(block.figure.alt_text)
            caption_esc = html.escape(block.figure.caption)
            html_parts.append('    <div class="figure-container">')
            html_parts.append(f'      <img src="{b64_src}" alt="{alt_esc}">')
            html_parts.append(f'      <div class="figure-caption">{caption_esc}</div>')
            html_parts.append("    </div>")

    html_parts.extend(["  </div>", "</body>", "</html>"])

    full_html = "\n".join(html_parts)
    out_p.write_text(full_html, encoding="utf-8")
    return str(out_p)

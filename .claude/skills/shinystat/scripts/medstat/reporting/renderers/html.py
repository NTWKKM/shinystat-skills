"""
src/medstat/reporting/renderers/html.py: Self-contained single-file HTML document renderer.
Embeds figures as inline base64 data, styles publication tables with ICMJE 3-rule borders,
and provides Dual / Toggle View (Plotly interactive + 300 DPI publication static) with offline fallback.
"""

from __future__ import annotations

import base64
import html
import urllib.parse
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
        raise FileNotFoundError(f"Figure file not found: {p}")
    data = p.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _encode_csv_data_uri(csv_path: str | Path | None) -> str | None:
    """Reads CSV file and returns data:text/csv;charset=utf-8 data URI."""
    if not csv_path:
        return None
    p = Path(csv_path).resolve()
    if not p.exists():
        return None
    try:
        content = p.read_text(encoding="utf-8")
        escaped = urllib.parse.quote(content)
        return f"data:text/csv;charset=utf-8,{escaped}"
    except Exception:
        return None


def render_html(
    doc: ReportDocument,
    out_path: str | Path,
) -> str:
    """
    Renders ReportDocument into a self-contained, publication-grade HTML document
    with responsive tables and dual interactive/publication figures.
    """
    out_p = Path(out_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    escaped_title = html.escape(doc.title)

    has_interactive_figs = any(
        isinstance(b, FigureBlock) and getattr(b.figure, "plotly_json", None)
        for b in doc.blocks
    )

    html_parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8">',
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        f"  <title>{escaped_title}</title>",
    ]

    if has_interactive_figs:
        html_parts.append(
            '  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
        )

    html_parts.extend(
        [
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
            "      max-width: 920px;",
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
            "",
            "    /* Dual-View Figure Card Styling */",
            "    .figure-card {",
            "      margin: 32px 0;",
            "      border: 1px solid #E2E8F0;",
            "      border-radius: 8px;",
            "      background: #FFFFFF;",
            "      box-shadow: 0 1px 3px rgba(0,0,0,0.04);",
            "      overflow: hidden;",
            "    }",
            "    .figure-toolbar {",
            "      display: flex;",
            "      justify-content: space-between;",
            "      align-items: center;",
            "      padding: 10px 16px;",
            "      background: #F8FAFC;",
            "      border-bottom: 1px solid #E2E8F0;",
            "      flex-wrap: wrap;",
            "      gap: 8px;",
            "    }",
            "    .figure-tabs {",
            "      display: inline-flex;",
            "      background: #E2E8F0;",
            "      border-radius: 6px;",
            "      padding: 2px;",
            "    }",
            "    .fig-tab-btn {",
            "      border: none;",
            "      background: transparent;",
            "      padding: 5px 12px;",
            "      border-radius: 5px;",
            "      font-size: 12.5px;",
            "      font-weight: 500;",
            "      color: #475569;",
            "      cursor: pointer;",
            "      display: inline-flex;",
            "      align-items: center;",
            "      gap: 6px;",
            "      transition: background-color 0.15s, color 0.15s;",
            "    }",
            "    .fig-tab-btn:hover { color: #0F172A; }",
            "    .fig-tab-btn.active {",
            "      background: #FFFFFF;",
            "      color: #0F172A;",
            "      font-weight: 600;",
            "      box-shadow: 0 1px 2px rgba(0,0,0,0.06);",
            "    }",
            "    .figure-badge-static {",
            "      font-size: 12.5px;",
            "      font-weight: 600;",
            "      color: #475569;",
            "      display: inline-flex;",
            "      align-items: center;",
            "      gap: 6px;",
            "    }",
            "    .figure-actions {",
            "      display: inline-flex;",
            "      gap: 8px;",
            "      align-items: center;",
            "    }",
            "    .fig-action-btn {",
            "      display: inline-flex;",
            "      align-items: center;",
            "      gap: 5px;",
            "      padding: 5px 10px;",
            "      background: #FFFFFF;",
            "      border: 1px solid #CBD5E1;",
            "      border-radius: 5px;",
            "      color: #334155;",
            "      text-decoration: none;",
            "      font-size: 12px;",
            "      font-weight: 500;",
            "      cursor: pointer;",
            "      transition: background-color 0.15s, border-color 0.15s;",
            "    }",
            "    .fig-action-btn:hover {",
            "      background: #F1F5F9;",
            "      border-color: #94A3B8;",
            "      color: #0F172A;",
            "    }",
            "    .figure-view-interactive, .figure-view-static {",
            "      padding: 16px;",
            "      text-align: center;",
            "    }",
            "    .pub-figure-img {",
            "      max-width: 100%;",
            "      height: auto;",
            "      border-radius: 4px;",
            "      box-shadow: 0 1px 2px rgba(0,0,0,0.05);",
            "    }",
            "    .figure-meta-badge {",
            "      font-size: 12px;",
            "      color: #64748B;",
            "      margin-top: 8px;",
            "      font-style: italic;",
            "    }",
            "    .offline-banner {",
            "      margin: 16px;",
            "      padding: 14px 18px;",
            "      background: #FFFBEB;",
            "      border: 1px solid #FDE68A;",
            "      border-radius: 6px;",
            "      text-align: left;",
            "      display: flex;",
            "      flex-direction: column;",
            "      gap: 10px;",
            "    }",
            "    .offline-banner-content {",
            "      display: flex;",
            "      align-items: flex-start;",
            "      gap: 10px;",
            "    }",
            "    .offline-icon { font-size: 18px; line-height: 1.2; }",
            "    .offline-text {",
            "      font-size: 13px;",
            "      color: #92400E;",
            "      line-height: 1.5;",
            "    }",
            "    .offline-switch-btn {",
            "      align-self: flex-start;",
            "      padding: 6px 12px;",
            "      background: #F59E0B;",
            "      color: #FFFFFF;",
            "      border: none;",
            "      border-radius: 4px;",
            "      font-size: 12.5px;",
            "      font-weight: 600;",
            "      cursor: pointer;",
            "      transition: background 0.15s;",
            "    }",
            "    .offline-switch-btn:hover { background: #D97706; }",
            "    .figure-caption {",
            "      padding: 10px 16px 14px;",
            "      font-size: 13px;",
            "      color: #334155;",
            "      border-top: 1px solid #F1F5F9;",
            "      background: #FAFAFA;",
            "      text-align: left;",
            "    }",
            "",
            "    @media print {",
            "      body { background: #FFFFFF; padding: 0; }",
            "      .container { box-shadow: none; padding: 0; max-width: 100%; }",
            "      .figure-toolbar { display: none !important; }",
            "      .figure-view-interactive { display: none !important; }",
            "      .figure-view-static { display: block !important; padding: 0 !important; }",
            "      .figure-meta-badge { display: none !important; }",
            "      .offline-banner { display: none !important; }",
            "      .figure-card { border: none !important; box-shadow: none !important; margin: 20px 0 !important; }",
            "      .figure-caption { border: none !important; background: transparent !important; padding: 8px 0 !important; }",
            "    }",
            "  </style>",
            "</head>",
            "<body>",
            '  <div class="container">',
            f"    <h1>{escaped_title}</h1>",
        ]
    )

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

    fig_idx = 0
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
            fig_idx += 1
            idx_str = str(fig_idx)
            b64_src = _encode_image_base64(block.figure.png_path)
            alt_esc = html.escape(block.figure.alt_text)
            caption_esc = html.escape(block.figure.caption)
            csv_uri = _encode_csv_data_uri(getattr(block.figure, "csv_path", None))
            plotly_json = getattr(block.figure, "plotly_json", None)

            filename_base = Path(block.figure.png_path).stem

            html_parts.append(f'    <div class="figure-card" id="fig-card-{idx_str}">')
            html_parts.append('      <div class="figure-toolbar">')

            if plotly_json:
                html_parts.extend(
                    [
                        '        <div class="figure-tabs" role="tablist">',
                        f'          <button type="button" class="fig-tab-btn active" role="tab" id="tab-interactive-{idx_str}" aria-selected="true" data-tab="interactive" onclick="switchFigView(\'{idx_str}\', \'interactive\')">',
                        '            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3v18h18"/><path d="M18 9l-5 5-4-4-6 6"/></svg>',
                        "            Interactive (Plotly)",
                        "          </button>",
                        f'          <button type="button" class="fig-tab-btn" role="tab" id="tab-static-{idx_str}" aria-selected="false" data-tab="static" onclick="switchFigView(\'{idx_str}\', \'static\')">',
                        '            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>',
                        "            Publication 300 DPI (Static)",
                        "          </button>",
                        "        </div>",
                    ]
                )
            else:
                html_parts.extend(
                    [
                        '        <div class="figure-badge-static">',
                        '          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>',
                        "          Publication Figure (300 DPI)",
                        "        </div>",
                    ]
                )

            # Action download buttons
            html_parts.append('        <div class="figure-actions">')
            html_parts.append(
                f'          <a class="fig-action-btn" href="{b64_src}" download="{filename_base}.png" title="Download 300 DPI Publication Figure">'
            )
            html_parts.append(
                '            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>'
            )
            html_parts.append("            Download 300 DPI PNG")
            html_parts.append("          </a>")

            if csv_uri:
                html_parts.append(
                    f'          <a class="fig-action-btn" href="{csv_uri}" download="{filename_base}.csv" title="Download Underlying Data (CSV)">'
                )
                html_parts.append(
                    '            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>'
                )
                html_parts.append("            CSV Data")
                html_parts.append("          </a>")

            html_parts.append("        </div>")
            html_parts.append("      </div>")  # end toolbar

            # Panels
            if plotly_json:
                html_parts.extend(
                    [
                        f'      <div class="figure-view-interactive" id="fig-interactive-{idx_str}">',
                        f'        <div id="plotly-graph-{idx_str}" class="plotly-container" style="min-height: 480px; width: 100%;"></div>',
                        f'        <div id="plotly-offline-banner-{idx_str}" class="offline-banner" style="display: none;">',
                        '          <div class="offline-banner-content">',
                        '            <span class="offline-icon">📡</span>',
                        '            <div class="offline-text">',
                        "              <strong>โหมดออฟไลน์ (Offline Mode):</strong> ระบบไม่สามารถเชื่อมต่อ Plotly CDN ได้ในเครือข่ายนี้ ",
                        "              คุณสามารถกดปุ่มด้านล่างหรือคลิกแท็บเพื่อสลับไปดูรูปคุณภาพสูง 300 DPI ได้ทันที",
                        "            </div>",
                        "          </div>",
                        f'          <button type="button" class="offline-switch-btn" onclick="switchFigView(\'{idx_str}\', \'static\')">',
                        "            🖼️ สลับไปดู Publication Figure (300 DPI)",
                        "          </button>",
                        "        </div>",
                        "      </div>",
                        f'      <div class="figure-view-static" id="fig-static-{idx_str}" style="display: none;">',
                        f'        <img src="{b64_src}" alt="{alt_esc}" class="pub-figure-img">',
                        '        <div class="figure-meta-badge">Journal Publication Standard &bull; 300 DPI Matplotlib</div>',
                        "      </div>",
                        f'      <script type="application/json" id="plotly-spec-{idx_str}">',
                        plotly_json,
                        "      </script>",
                    ]
                )
            else:
                html_parts.extend(
                    [
                        f'      <div class="figure-view-static" id="fig-static-{idx_str}">',
                        f'        <img src="{b64_src}" alt="{alt_esc}" class="pub-figure-img">',
                        '        <div class="figure-meta-badge">Journal Publication Standard &bull; 300 DPI Matplotlib</div>',
                        "      </div>",
                    ]
                )

            html_parts.append(f'      <div class="figure-caption">{caption_esc}</div>')
            html_parts.append("    </div>")  # end figure-card

    if has_interactive_figs:
        html_parts.extend(
            [
                "  <script>",
                "    function switchFigView(id, view) {",
                "      var interactiveEl = document.getElementById('fig-interactive-' + id);",
                "      var staticEl = document.getElementById('fig-static-' + id);",
                "      var tabInteractive = document.getElementById('tab-interactive-' + id);",
                "      var tabStatic = document.getElementById('tab-static-' + id);",
                "",
                "      if (view === 'interactive') {",
                "        if (window._plotlyOffline || typeof Plotly === 'undefined') {",
                "          var banner = document.getElementById('plotly-offline-banner-' + id);",
                "          if (banner) banner.style.display = 'block';",
                "          return;",
                "        }",
                "        if (interactiveEl) interactiveEl.style.display = 'block';",
                "        if (staticEl) staticEl.style.display = 'none';",
                "        if (tabInteractive) {",
                "          tabInteractive.classList.add('active');",
                "          tabInteractive.setAttribute('aria-selected', 'true');",
                "        }",
                "        if (tabStatic) {",
                "          tabStatic.classList.remove('active');",
                "          tabStatic.setAttribute('aria-selected', 'false');",
                "        }",
                "        var graphDiv = document.getElementById('plotly-graph-' + id);",
                "        if (graphDiv && window.Plotly) {",
                "          Plotly.Plots.resize(graphDiv);",
                "        }",
                "      } else {",
                "        if (interactiveEl) interactiveEl.style.display = 'none';",
                "        if (staticEl) staticEl.style.display = 'block';",
                "        if (tabInteractive) {",
                "          tabInteractive.classList.remove('active');",
                "          tabInteractive.setAttribute('aria-selected', 'false');",
                "        }",
                "        if (tabStatic) {",
                "          tabStatic.classList.add('active');",
                "          tabStatic.setAttribute('aria-selected', 'true');",
                "        }",
                "      }",
                "    }",
                "",
                "    document.addEventListener('DOMContentLoaded', function() {",
                "      var isPlotlyAvailable = typeof Plotly !== 'undefined';",
                "      if (!isPlotlyAvailable) {",
                "        window._plotlyOffline = true;",
                "        document.querySelectorAll('.figure-card').forEach(function(card) {",
                "          var id = card.id.replace('fig-card-', '');",
                "          var banner = document.getElementById('plotly-offline-banner-' + id);",
                "          if (banner) banner.style.display = 'block';",
                "        });",
                "        return;",
                "      }",
                "",
                '      document.querySelectorAll(\'script[type="application/json"][id^="plotly-spec-"]\').forEach(function(scriptEl) {',
                "        var id = scriptEl.id.replace('plotly-spec-', '');",
                "        var graphDiv = document.getElementById('plotly-graph-' + id);",
                "        if (graphDiv) {",
                "          try {",
                "            var spec = JSON.parse(scriptEl.textContent);",
                "            var data = spec.data || [];",
                "            var layout = spec.layout || {};",
                "            layout.autosize = true;",
                "            layout.responsive = true;",
                "            Plotly.newPlot(graphDiv, data, layout, {responsive: true, displayModeBar: true, displaylogo: false});",
                "          } catch (e) {",
                "            console.error('Failed to render Plotly chart for ' + id, e);",
                "            switchFigView(id, 'static');",
                "          }",
                "        }",
                "      });",
                "",
                "      window.addEventListener('resize', function() {",
                "        document.querySelectorAll('.figure-view-interactive .plotly-container').forEach(function(el) {",
                "          if (el.offsetParent !== null && window.Plotly) {",
                "            Plotly.Plots.resize(el);",
                "          }",
                "        });",
                "      });",
                "    });",
                "  </script>",
            ]
        )

    html_parts.extend(["  </div>", "</body>", "</html>"])

    full_html = "\n".join(html_parts)
    out_p.write_text(full_html, encoding="utf-8")
    return str(out_p)

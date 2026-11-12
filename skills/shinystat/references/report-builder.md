# Cloud Multi-Format Report Builder & Pillar 5 Integrity Reference

In cloud sandboxes (e.g. Claude Web Artifacts), `medstat` is not installed as an external package. This reference provides self-contained, pure-Python reporting recipes that assemble biostatistical findings into publication-grade documents (Markdown, self-contained HTML, Word `.docx`, and PowerPoint `.pptx`) with zero custom dependencies.

**Load conditions** (read only the section your `#5a` format needs):
- Markdown or HTML: [§1–§2.B](#1-cloud-report-generation-architecture); confirm the format first via Gate #5a (`SKILL.md` §2).
- Interactive `plotly_figs` inside the HTML report: [report-plotly-specs.md](report-plotly-specs.md) (`build_plotly_*_spec`).
- `.docx`: [§2.C (`generate_docx_report`)](#c-standalone-word-docx-report-when-python-docx-is-installed); also load §2.B (`generate_html_report`) when `python-docx` is unavailable and the HTML fallback is used.
- `.pptx`: [§2.D (`generate_pptx_report`)](#2d-standalone-powerpoint-deck-generator-pptx); also load §2.B (`generate_html_report`) when `python-pptx` is unavailable and the HTML fallback is used.
- `.pdf`: [§2.E (`generate_pdf_report`)](#2e-standalone-pdf-report-generator-pdf).
- Every format, before delivery: [§3 (`verify_standalone_integrity`)](#3-standalone-reporting-integrity-verification).

---

## 1. Cloud Report Generation Architecture

1. **Default & Primary Formats (Zero Extra Dependencies)**:
   - **Self-Contained HTML (`.html`)**: **Recommended format at alignment gate**. When the user prompt omits a deliverable format, the agent must wait for the user's format selection at the Output Format Alignment Gate rather than silently falling back. Always available; embeds plots via base64 data URIs and applies ICMJE 3-rule CSS styling.
   - **Markdown (`.md`)**: Always available; formats tables via `pandas.DataFrame.to_markdown()` and links figure assets.
2. **Office Document Fallbacks (`python-docx` / `python-pptx`)**:
   - If `docx` or `pptx` packages are installed in the sandbox, use the standalone snippets below upon explicit user request.
   - If either package is missing (`ImportError`), automatically fall back to generating **Self-Contained HTML**.

---

## 2. Standalone Document Generation Recipes

### A. Markdown Report (`generate_markdown_report`)

```python
from pathlib import Path
import pandas as pd


def generate_markdown_report(
    title: str,
    results: dict,
    tables: dict[str, pd.DataFrame],
    out_path: str = "report.md",
    figure_paths: list[str] = None,
) -> str:
    lines = [
        f"# {title}\n",
        f"**Date:** {results.get('date', 'N/A')} | **Analysis Cohort:** N = {results.get('n', 'N/A')}\n\n---\n",
    ]

    # Executive Summary / Narrative
    lines.append("## Executive Summary\n")
    if "narrative" in results:
        lines.append(f"{results['narrative']}\n")

    # Tables
    for caption, df in tables.items():
        lines.append(f"\n### {caption}\n")
        try:
            lines.append(df.to_markdown(index=False) + "\n")
        except Exception:
            # Table formatting fallback preserving Markdown pipe structure without tabulate
            hdr = "| " + " | ".join(str(c) for c in df.columns) + " |"
            sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
            rows = ["| " + " | ".join(str(val) for val in row) + " |" for _, row in df.iterrows()]
            lines.append("\n".join([hdr, sep] + rows) + "\n")

    out_p = Path(out_path)
    out_dir = out_p.parent.resolve()

    # Figures
    if figure_paths:
        lines.append("\n## Figures\n")
        for fp in figure_paths:
            p = Path(fp)
            try:
                rel_path = p.resolve().relative_to(out_dir).as_posix()
            except ValueError:
                rel_path = p.name
            escaped_link = rel_path.replace(" ", "%20")
            lines.append(f"![{p.stem}]" + f"({escaped_link})\n")

    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("\n".join(lines), encoding="utf-8")
    return str(out_p)
```

### B. Self-Contained HTML Report with Dual / Toggle View

Generates a standalone single-file HTML report with ICMJE 3-rule tables and **Dual / Toggle View when a Plotly specification is supplied** (interactive Plotly charts with instant toggle to 300 DPI publication figures, download buttons, and offline resilience; without a Plotly specification, the report contains only the static publication view):

```python
import base64
import html
import json
from pathlib import Path
import re
import pandas as pd


def generate_html_report(
    title: str,
    results: dict,
    tables: dict[str, pd.DataFrame],
    figure_paths: list[str] = None,
    plotly_figs: list[any] = None,  # optional list of go.Figure matching figure_paths
    out_path: str = "report.html",
) -> str:
    has_plotly = bool(plotly_figs)
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
    ]
    if has_plotly:
        parts.append("<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>")

    parts.extend([
        "<style>",
        "  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #1e293b; max-width: 920px; margin: 40px auto; padding: 0 20px; background: #fafafa; }",
        "  .container { background: #fff; padding: 40px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }",
        "  h1 { font-size: 26px; border-bottom: 2px solid #0f172a; padding-bottom: 10px; margin-top: 0; }",
        "  h2 { font-size: 20px; color: #0284c7; margin-top: 30px; }",
        "  table.pub-table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 14px; }",
        "  table.pub-table thead tr { border-top: 2px solid #0f172a; border-bottom: 1px solid #0f172a; }",
        "  table.pub-table tbody tr:last-child { border-bottom: 2px solid #0f172a; }",
        "  table.pub-table th, table.pub-table td { padding: 8px 12px; text-align: left; }",
        "  sup.citation { font-size: 11px; vertical-align: super; line-height: 0; }",
        "  sup.citation a { text-decoration: none; color: #0284c7; font-weight: 600; }",
        "  ol.references-list { padding-left: 20px; font-size: 13.5px; color: #334155; }",
        "  ol.references-list li { margin-bottom: 6px; }",
        "  .figure-card { margin: 30px 0; border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; background: #fff; }",
        "  .figure-toolbar { display: flex; justify-content: space-between; align-items: center; padding: 8px 14px; background: #f8fafc; border-bottom: 1px solid #e2e8f0; flex-wrap: wrap; gap: 8px; }",
        "  .fig-tabs { display: inline-flex; background: #e2e8f0; border-radius: 6px; padding: 2px; }",
        "  .fig-btn { border: none; background: transparent; padding: 5px 12px; border-radius: 5px; font-size: 12.5px; cursor: pointer; color: #475569; }",
        "  .fig-btn.active { background: #fff; color: #0f172a; font-weight: 600; box-shadow: 0 1px 2px rgba(0,0,0,0.06); }",
        "  .fig-actions a { padding: 4px 10px; border: 1px solid #cbd5e1; border-radius: 4px; font-size: 12px; text-decoration: none; color: #334155; background: #fff; }",
        "  .figure-view { padding: 16px; text-align: center; }",
        "  .figure-view img { max-width: 100%; height: auto; border-radius: 4px; }",
        "  .offline-banner { margin: 12px; padding: 12px; background: #fffbeb; border: 1px solid #fde68a; border-radius: 6px; text-align: left; font-size: 13px; color: #92400e; }",
        "  .caption { font-size: 13px; color: #64748b; font-style: italic; margin-top: 6px; }",
        "  @media print { .figure-toolbar { display: none !important; } .view-interactive { display: none !important; } .view-static { display: block !important; } }",
        "</style></head><body><div class='container'>",
        f"<h1>{html.escape(title)}</h1>",
    ])

    # Narrative with Citation Links
    referenced_ids = set()
    if "narrative" in results:
        raw_narr = str(results["narrative"])
        for m in re.finditer(r"\[(\d+(?:,\s*\d+)*)\]", raw_narr):
            for n_str in m.group(1).split(","):
                referenced_ids.add(int(n_str.strip()))
        for m in re.finditer(r"<sup[^>]*>.*?(\d+).*?</sup>", raw_narr):
            for n_str in re.findall(r"\d+", m.group(0)):
                referenced_ids.add(int(n_str))

        narr = html.escape(raw_narr)
        # Convert in-text citation markers [1] or [1, 2] to anchor links
        narr = re.sub(
            r"\[(\d+(?:,\s*\d+)*)\]",
            lambda m: '<sup class="citation">'
            + ", ".join(f'<a href="#ref-{n.strip()}">{n.strip()}</a>' for n in m.group(1).split(","))
            + "</sup>",
            narr,
        )
        # Restore permitted citation superscript tags with allowlisted attributes
        def _restore_sup(match: re.Match) -> str:
            attrs = match.group(1).strip()
            content = match.group(2)
            if not attrs:
                return f"<sup>{content}</sup>"
            attrs_clean = html.unescape(attrs).strip()
            m_cls = re.fullmatch(r'class=(["\'])(.*?)\1', attrs_clean)
            if m_cls:
                cls_tokens = m_cls.group(2).split()
                allowed = {"citation", "footnote", "ref"}
                if cls_tokens and all(c in allowed for c in cls_tokens):
                    return f'<sup {attrs_clean}>{content}</sup>'
            return f"<sup>{content}</sup>"

        narr = re.sub(r"&lt;sup(.*?)&gt;(.*?)&lt;/sup&gt;", _restore_sup, narr)
        parts.append(f"<p>{narr}</p>")

    # Tables
    for caption, df in tables.items():
        parts.append(f"<div class='caption'><strong>{html.escape(caption)}</strong></div>")
        parts.append("<table class='pub-table'><thead><tr>")
        for col in df.columns:
            parts.append(f"<th>{html.escape(str(col))}</th>")
        parts.append("</tr></thead><tbody>")
        for _, row in df.iterrows():
            parts.append("<tr>")
            for col in df.columns:
                val = str(row[col]) if pd.notna(row[col]) else ""
                parts.append(f"<td>{html.escape(val)}</td>")
            parts.append("</tr>")
        parts.append("</tbody></table>")

    # Figures (Dual / Toggle View)
    if figure_paths:
        for idx, fig_p_str in enumerate(figure_paths):
            fp = Path(fig_p_str)
            if not fp.exists():
                raise FileNotFoundError(f"Figure file not found: {fig_p_str}")
            suffix = fp.suffix.lower()
            if suffix == ".png":
                mime = "image/png"
                dl_label = "Download 300 DPI PNG"
                dl_ext = "png"
            elif suffix in {".jpg", ".jpeg"}:
                mime = "image/jpeg"
                dl_label = "Download JPEG"
                dl_ext = "jpg"
            elif suffix == ".svg":
                mime = "image/svg+xml"
                dl_label = "Download SVG"
                dl_ext = "svg"
            else:
                raise ValueError(
                    f"Unsupported figure format for HTML embedding: {fp.suffix}. Figures must be PNG, JPEG, or SVG."
                )

            b64 = base64.b64encode(fp.read_bytes()).decode("utf-8")
            plt_fig = plotly_figs[idx] if plotly_figs and idx < len(plotly_figs) else None
            idx_str = str(idx + 1)
            escaped_stem = html.escape(fp.stem, quote=True)

            parts.append(f"<div class='figure-card' id='fig-card-{idx_str}'>")
            parts.append("<div class='figure-toolbar'>")
            if plt_fig:
                parts.append("<div class='fig-tabs'>")
                parts.append(f"<button type='button' class='fig-btn active' id='tab-i-{idx_str}' onclick=\"switchFig('{idx_str}', 'i')\">Interactive (Plotly)</button>")
                parts.append(f"<button type='button' class='fig-btn' id='tab-s-{idx_str}' onclick=\"switchFig('{idx_str}', 's')\">Publication 300 DPI Figure (Static)</button>")
                parts.append("</div>")
            else:
                parts.append("<span style='font-size:12.5px; font-weight:600;'>Publication 300 DPI Figure (Static)</span>")
            parts.append(f"<div class='fig-actions'><a href='data:{mime};base64,{b64}' download='{escaped_stem}.{dl_ext}'>{dl_label}</a></div>")
            parts.append("</div>")

            if plt_fig:
                raw_json = (
                    plt_fig.to_json()
                    if hasattr(plt_fig, "to_json")
                    else json.dumps(plt_fig)
                )
                plotly_json = raw_json.replace("<", "\\u003c")
                parts.append(f"<div class='figure-view view-interactive' id='view-i-{idx_str}'>")
                parts.append(f"<div id='plotly-div-{idx_str}' style='min-height:480px; width:100%;'></div>")
                parts.append(f"<div id='offline-msg-{idx_str}' class='offline-banner' style='display:none;'>📡 โหมดออฟไลน์: ไม่สามารถโหลด Plotly CDN ได้ กรุณาสลับไปดูรูป 300 DPI <button onclick=\"switchFig('{idx_str}', 's')\" style='margin-left:8px; cursor:pointer;'>สลับไปดูรูป 300 DPI</button></div>")
                parts.append("</div>")
                parts.append(f"<div class='figure-view view-static' id='view-s-{idx_str}' style='display:none;'>")
                parts.append(f"<img src='data:{mime};base64,{b64}' alt='{escaped_stem}'>")
                parts.append("</div>")
                parts.append(f"<script type='application/json' id='spec-{idx_str}'>{plotly_json}</script>")
            else:
                parts.append(f"<div class='figure-view view-static' id='view-s-{idx_str}'>")
                parts.append(f"<img src='data:{mime};base64,{b64}' alt='{escaped_stem}'>")
                parts.append("</div>")
            parts.append("</div>")

    # References
    citations = results.get("references") or results.get("citations")
    if citations or referenced_ids:
        parts.append("<h2>References</h2><ol class='references-list'>")
        if citations:
            for idx, ref in enumerate(citations, 1):
                ref_str = (
                    ref.get("formatted") or ref.get("citation") or ref.get("text") or str(ref)
                    if isinstance(ref, dict)
                    else str(ref)
                )
                parts.append(f"<li id='ref-{idx}'>{html.escape(ref_str)}</li>")
        else:
            for idx in sorted(referenced_ids):
                parts.append(f"<li id='ref-{idx}'>Reference {idx}</li>")
        parts.append("</ol>")

    if has_plotly:
        parts.append("""
<script>
function switchFig(id, view) {
  var vi = document.getElementById('view-i-' + id);
  var vs = document.getElementById('view-s-' + id);
  var ti = document.getElementById('tab-i-' + id);
  var ts = document.getElementById('tab-s-' + id);
  if (view === 'i') {
    if (typeof Plotly === 'undefined') {
      var msg = document.getElementById('offline-msg-' + id);
      if (msg) msg.style.display = 'block';
      return;
    }
    if (vi) vi.style.display = 'block';
    if (vs) vs.style.display = 'none';
    if (ti) ti.classList.add('active');
    if (ts) ts.classList.remove('active');
    var div = document.getElementById('plotly-div-' + id);
    if (div && window.Plotly) Plotly.Plots.resize(div);
  } else {
    if (vi) vi.style.display = 'none';
    if (vs) vs.style.display = 'block';
    if (ti) ti.classList.remove('active');
    if (ts) ts.classList.add('active');
  }
}
document.addEventListener('DOMContentLoaded', function() {
  if (typeof Plotly === 'undefined') {
    document.querySelectorAll('.figure-card').forEach(function(c) {
      var id = c.id.replace('fig-card-', '');
      var msg = document.getElementById('offline-msg-' + id);
      if (msg) msg.style.display = 'block';
      switchFig(id, 's');
    });
    return;
  }
  document.querySelectorAll('script[type=\"application/json\"][id^=\"spec-\"]').forEach(function(s) {
    var id = s.id.replace('spec-', '');
    var div = document.getElementById('plotly-div-' + id);
    if (div) {
      try {
        var spec = JSON.parse(s.textContent);
        Plotly.newPlot(div, spec.data || [], spec.layout || {}, {responsive: true, displaylogo: false});
      } catch(e) { switchFig(id, 's'); }
    }
  });
});
</script>
""")

    parts.append("</div></body></html>")
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("\n".join(parts), encoding="utf-8")
    return str(out_p)
```

### C. Standalone Word (`.docx`) Report (when `python-docx` is installed)

```python
try:
    from pathlib import Path
    import pandas as pd
    import docx
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import parse_xml
    from docx.oxml.ns import nsdecls

    def generate_docx_report(
        title: str,
        results: dict,
        tables: dict[str, pd.DataFrame],
        figure_paths: list[str] = None,
        out_path: str = "report.docx",
    ) -> str:
        doc = docx.Document()
        doc.add_heading(title, level=0)

        if "narrative" in results:
            doc.add_paragraph(results["narrative"])

        for caption, df in tables.items():
            p = doc.add_paragraph()
            p.add_run(caption).bold = True
            tbl = doc.add_table(rows=len(df) + 1, cols=len(df.columns))
            # Format header
            for j, col in enumerate(df.columns):
                tbl.cell(0, j).text = str(col)
                tbl.cell(0, j).paragraphs[0].runs[0].bold = True
            # Populate data
            for i, (_, row) in enumerate(df.iterrows()):
                for j, col in enumerate(df.columns):
                    val = str(row[col]) if pd.notna(row[col]) else ""
                    tbl.cell(i + 1, j).text = val
            # Apply 3-rule borders
            tblPr = tbl._tbl.tblPr
            borders = parse_xml(
                f"<w:tblBorders {nsdecls('w')}>\n"
                f'  <w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>\n'
                f'  <w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/>\n'
                f'  <w:insideH w:val="none"/>\n'
                f'  <w:left w:val="none"/>\n'
                f'  <w:right w:val="none"/>\n'
                f'  <w:insideV w:val="none"/>\n'
                f"</w:tblBorders>"
            )
            tblPr.append(borders)

        if figure_paths:
            for fp in figure_paths:
                p = Path(fp)
                if p.exists():
                    doc.add_picture(str(p), width=Inches(6.5))

        doc.save(out_path)
        return out_path

except ImportError:
    # Automatic fallback when python-docx is unavailable
    def generate_docx_report(
        title: str,
        results: dict,
        tables: dict[str, pd.DataFrame],
        figure_paths: list[str] = None,
        out_path: str = "report.docx",
    ) -> str:
        if Path(out_path).suffix.lower() == ".docx":
            out_path = str(Path(out_path).with_suffix(".html"))
        return generate_html_report(
            title, results, tables, figure_paths, out_path=out_path
        )
```

### 2D. Standalone PowerPoint Deck Generator (`.pptx`)

```python
import re
from pathlib import Path
import pandas as pd

try:
    from pptx import Presentation
    from pptx.util import Inches, Pt

    def generate_pptx_report(
        title: str,
        results: dict,
        tables: dict[str, pd.DataFrame],
        figure_paths: list[str] = None,
        kpi_cards: list[dict] = None,
        out_path: str = "report.pptx",
    ) -> str:
        prs = Presentation()
        prs.slide_width = Inches(13.333)
        prs.slide_height = Inches(7.5)

        # 1. Title & Executive Narrative Slide
        slide_layout = prs.slide_layouts[1] if len(prs.slide_layouts) > 1 else prs.slide_layouts[0]
        slide = prs.slides.add_slide(slide_layout)
        if slide.shapes.title:
            slide.shapes.title.text = str(title)
        if len(slide.placeholders) > 1 and "narrative" in results:
            clean_narr = re.sub(r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s+[^>]*)?>", "", str(results["narrative"]))
            slide.placeholders[1].text = clean_narr

        blank_layout = prs.slide_layouts[6] if len(prs.slide_layouts) > 6 else prs.slide_layouts[0]

        # 2. Key Findings & KPI Cards Slides (paginated up to 4 per slide)
        cards = kpi_cards if kpi_cards is not None else results.get("kpi_cards", [])
        if cards:
            for chunk_idx in range(0, len(cards), 4):
                chunk = cards[chunk_idx : chunk_idx + 4]
                kslide = prs.slides.add_slide(blank_layout)
                hdr = kslide.shapes.add_textbox(Inches(0.6), Inches(0.4), Inches(12.0), Inches(0.6))
                slide_num = f" (Part {chunk_idx // 4 + 1})" if len(cards) > 4 else ""
                hdr.text_frame.text = f"Key Clinical Findings & KPI Summary{slide_num}"
                n_cards = len(chunk)
                card_w = 11.6 / max(1, n_cards)
                for idx, card in enumerate(chunk):
                    left = Inches(0.6 + idx * (card_w + 0.15))
                    box = kslide.shapes.add_textbox(left, Inches(1.5), Inches(card_w), Inches(2.5))
                    tf = box.text_frame
                    tf.word_wrap = True
                    p_lbl = tf.paragraphs[0]
                    p_lbl.text = str(card.get("label", f"Metric {chunk_idx + idx + 1}"))
                    for r in p_lbl.runs:
                        r.font.size = Pt(12)
                        r.font.bold = True
                    p_val = tf.add_paragraph()
                    p_val.text = str(card.get("value", ""))
                    for r in p_val.runs:
                        r.font.size = Pt(22)
                        r.font.bold = True
                    if card.get("subtitle"):
                        p_sub = tf.add_paragraph()
                        p_sub.text = str(card["subtitle"])
                        for r in p_sub.runs:
                            r.font.size = Pt(10)

        # 3. Table Slides
        for caption, df in (tables or {}).items():
            tslide = prs.slides.add_slide(blank_layout)
            tx = tslide.shapes.add_textbox(Inches(0.6), Inches(0.4), Inches(12.0), Inches(0.6))
            tx.text_frame.text = str(caption)
            rows, cols = len(df) + 1, len(df.columns)
            tbl_shape = tslide.shapes.add_table(
                rows, cols, Inches(0.6), Inches(1.2), Inches(12.0), Inches(min(5.5, 0.45 * rows))
            )
            tbl = tbl_shape.table
            for j, col in enumerate(df.columns):
                cell = tbl.cell(0, j)
                cell.text = str(col)
                for p in cell.text_frame.paragraphs:
                    for r in p.runs:
                        r.font.bold = True
                        r.font.size = Pt(11)
            for i, (_, row) in enumerate(df.iterrows()):
                for j, col in enumerate(df.columns):
                    val = "" if pd.isna(row[col]) else str(row[col])
                    cell = tbl.cell(i + 1, j)
                    cell.text = val
                    for p in cell.text_frame.paragraphs:
                        for r in p.runs:
                            r.font.size = Pt(10)

        # 3. Figure Slides
        for fp in figure_paths or []:
            p = Path(fp)
            if p.exists():
                fslide = prs.slides.add_slide(blank_layout)
                fslide.shapes.add_picture(str(p), Inches(1.5), Inches(0.8), width=Inches(10.0))

        prs.save(out_path)
        return out_path

except ImportError:
    def generate_pptx_report(
        title: str,
        results: dict,
        tables: dict[str, pd.DataFrame],
        figure_paths: list[str] = None,
        kpi_cards: list[dict] = None,
        out_path: str = "report.pptx",
    ) -> str:
        if Path(out_path).suffix.lower() == ".pptx":
            out_path = str(Path(out_path).with_suffix(".html"))
        merged_results = dict(results) if results else {}
        if kpi_cards is not None and "kpi_cards" not in merged_results:
            merged_results["kpi_cards"] = kpi_cards
        return generate_html_report(
            title, merged_results, tables, figure_paths, out_path=out_path
        )
```

### 2E. Standalone PDF Report Generator (`.pdf`)

```python
import re
import textwrap
from pathlib import Path
import pandas as pd


def generate_pdf_report(
    title: str,
    results: dict,
    tables: dict[str, pd.DataFrame],
    figure_paths: list[str] = None,
    out_path: str = "report.pdf",
) -> str:
    """Renders a publication-grade multi-page PDF via reportlab Platypus (primary) or Matplotlib PdfPages (Type 42 fallback)."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Image as RLImage,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table as RLTable,
            TableStyle,
        )

        doc = SimpleDocTemplate(str(out_path), pagesize=A4)
        styles = getSampleStyleSheet()
        story = [Paragraph(str(title), styles["Title"]), Spacer(1, 0.2 * inch)]
        if "narrative" in results:
            clean_narr = re.sub(
                r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s+[^>]*)?>", "", str(results["narrative"])
            )
            for para in clean_narr.split("\n\n"):
                if para.strip():
                    story.append(Paragraph(para.strip(), styles["BodyText"]))
                    story.append(Spacer(1, 0.1 * inch))

        for caption, df in (tables or {}).items():
            if df is None or df.empty:
                continue
            story.append(Paragraph(f"<b>{caption}</b>", styles["Heading3"]))
            data = [[str(c) for c in df.columns]] + [
                ["" if pd.isna(v) else str(v) for v in row]
                for _, row in df.iterrows()
            ]
            t = RLTable(data, repeatRows=1)
            t.setStyle(
                TableStyle(
                    [
                        ("LINEABOVE", (0, 0), (-1, 0), 1.2, colors.black),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black),
                        ("LINEBELOW", (0, -1), (-1, -1), 1.2, colors.black),
                        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ]
                )
            )
            story.append(t)
            story.append(Spacer(1, 0.2 * inch))

        for fp in figure_paths or []:
            p = Path(fp)
            if p.exists() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                img = RLImage(str(p))
                target_w = 5.8 * inch
                if getattr(img, "imageWidth", 0) > 0:
                    aspect = img.imageHeight / img.imageWidth
                    img.drawWidth = target_w
                    img.drawHeight = target_w * aspect
                else:
                    img.drawWidth = target_w
                    img.drawHeight = 3.8 * inch
                story.append(img)
                story.append(Spacer(1, 0.2 * inch))

        doc.build(story)
        return out_path
    except Exception:
        pass

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    with PdfPages(out_path) as pdf:
        # Page 1: Title & Narrative Summary
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.92, str(title), fontsize=16, fontweight="bold", ha="left", va="top")
        if "narrative" in results:
            clean_narr = re.sub(
                r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s+[^>]*)?>", "", str(results["narrative"])
            )
            wrapped = "\n".join(
                textwrap.fill(line, width=88) if line.strip() else ""
                for line in clean_narr.splitlines()
            )
            fig.text(0.08, 0.85, wrapped, fontsize=10, ha="left", va="top", linespacing=1.4)
        plt.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Table Pages
        for caption, df in (tables or {}).items():
            if df is None or df.empty:
                continue
            tfig, ax = plt.subplots(figsize=(8.27, max(4.0, 1.2 + 0.35 * len(df))))
            ax.axis("off")
            ax.set_title(str(caption), fontsize=12, fontweight="bold", loc="left", pad=12)
            cell_text = [
                ["" if pd.isna(v) else str(v) for v in row]
                for _, row in df.iterrows()
            ]
            mpl_tbl = ax.table(
                cellText=cell_text,
                colLabels=[str(c) for c in df.columns],
                loc="upper center",
                cellLoc="left",
            )
            mpl_tbl.auto_set_font_size(False)
            mpl_tbl.set_fontsize(8.5)
            mpl_tbl.scale(1.0, 1.3)
            pdf.savefig(tfig, bbox_inches="tight")
            plt.close(tfig)

        # Figure Pages
        for fp in figure_paths or []:
            p = Path(fp)
            if p.exists() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                img = plt.imread(str(p))
                ffig, fax = plt.subplots(figsize=(8.27, 6.0))
                fax.imshow(img)
                fax.axis("off")
                pdf.savefig(ffig, bbox_inches="tight", dpi=300)
                plt.close(ffig)

    return out_path
```

---

## 3. Standalone Reporting Integrity Verification

Before releasing any report, run the standalone verification routine:

```python
import csv
import io
import math
import re
from pathlib import Path
import pandas as pd


def verify_standalone_integrity(
    narrative_text: str,
    results_dict: dict,
    table_dfs: list[pd.DataFrame] = None,
    allowed_untraced: list[float] | set[float] | tuple[float, ...] = None,
    study_design: str = None,
    exported_files: list = None,
) -> dict:
    """Verifies numerical traceability (len(untraced_numbers) == 0) and screens for PHI without importing medstat.

    Pass the exact text delivered to the reader (for In-Chat Summary, the chat reply) as `narrative_text`,
    and every data file written beside the report as `exported_files`: a missing file raises, and an
    identifier column (patient_id, hn, mrn, ...) or PHI pattern in one is a violation (engine parity, ADR 81).
    """
    violations = []

    # 1. PHI Scan
    phi_patterns = [
        (
            re.compile(
                r"\b(?:HN|hn|H\.N\.)\s*[:#/-]?\s*\d{2,}(?:[-/]\d{2,})+\b|\b(?:HN|hn|H\.N\.)\s*[:#/-]?\s*\d{4,}\b"
            ),
            "Hospital Number (HN)",
        ),
        (
            re.compile(r"\b\d{1}[-\s]?\d{4}[-\s]?\d{5}[-\s]?\d{2}[-\s]?\d{1}\b"),
            "Thai Citizen ID (13 digits)",
        ),
        (
            re.compile(
                r"\b(?:0[689]\d{1}[-\s]?\d{3}[-\s]?\d{4}|0[2-57]\d{1}[-\s]?\d{3}[-\s]?\d{3})\b"
            ),
            "Phone Number",
        ),
        (
            re.compile(
                r"\b(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?|นาย|นาง|นางสาว|นพ\.?|พญ\.?|ดร\.?)\s+[A-Za-zก-๙]{2,}\b"
            ),
            "Personal Name Marker",
        ),
        (
            re.compile(
                r"\b(?:DOB|dob|Date of Birth|วันเกิด)\s*[:#-]?\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b",
                re.I,
            ),
            "Date of Birth",
        ),
    ]

    for pat, label in phi_patterns:
        for m in pat.findall(narrative_text):
            violations.append(f"{label}: '{m}'")

    id_col_pattern = re.compile(
        r"^(?:patient|subject|participant|person|record)[_ ]?(?:id|no|number)$"
        r"|^(?:mrn|hn|hospital[_ ]?(?:id|no|number)|national[_ ]?id|citizen[_ ]?id|cid"
        r"|ssn|patient[_ ]?name|full[_ ]?name|name)$",
        re.IGNORECASE,
    )
    for path in exported_files or []:
        fp = Path(path)
        if not fp.is_file():
            raise FileNotFoundError(f"Exported file not found for integrity audit: {fp}")
        # Read by format; utf-8-sig drops a BOM that would hide a patient_id header (ADR 81).
        suffix = fp.suffix.lower()
        try:
            if suffix in (".csv", ".tsv", ".tab"):
                content = fp.read_text(encoding="utf-8-sig", errors="replace")
                delim = "," if suffix == ".csv" else "\t"
                cols = [c.strip() for c in next(csv.reader(io.StringIO(content), delimiter=delim), [])]
            elif suffix in (".xlsx", ".parquet"):
                frames = (
                    pd.read_excel(fp, sheet_name=None, dtype=str)
                    if suffix == ".xlsx"
                    else {"": pd.read_parquet(fp)}
                )
                cols = [str(c) for f in frames.values() for c in f.columns]
                content = "\n".join(f.astype(str).to_csv(index=False) for f in frames.values())
            else:
                raise ValueError(f"unsupported format '{suffix or fp.name}' (use .csv, .tsv, .xlsx or .parquet)")
        except (ValueError, ImportError) as exc:
            # Fail closed: a file the audit cannot read is never passed unchecked.
            violations.append(f"Exported file '{fp.name}' could not be audited: {exc}")
            continue
        for col in cols:
            if id_col_pattern.match(col):
                violations.append(f"Identifier column '{col}' in exported file '{fp.name}'")
        for pat, label in phi_patterns:
            for m in pat.findall(content):
                violations.append(f"{label} in exported file '{fp.name}': '{m}'")

    if table_dfs:
        for df in table_dfs:
            if df is not None and isinstance(df, pd.DataFrame):
                for col in df.columns:
                    col_str = str(col)
                    for pat, label in phi_patterns:
                        for m in pat.findall(col_str):
                            violations.append(f"{label}: '{m}'")
                    for val in df[col].dropna():
                        val_str = str(val)
                        for pat, label in phi_patterns:
                            for m in pat.findall(val_str):
                                violations.append(f"{label}: '{m}'")

    # 2. Observational Causal / E-value Caveat Check
    resolved_design = (
        study_design
        or (results_dict.get("study_design") if isinstance(results_dict, dict) else None)
    )

    def _normalize_key(k) -> str:
        s = re.sub(r"[\s\-]+", "_", str(k))
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
        s = re.sub(r"([a-zA-Z])([0-9])", r"\1_\2", s)
        s = re.sub(r"([0-9])([a-zA-Z])", r"\1_\2", s)
        return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s).lower().strip("_")

    results_keys_normalized = [_normalize_key(k) for k in results_dict]
    results_keys_str = " ".join(results_keys_normalized)
    text_lower = narrative_text.lower()
    has_causal_key = (
        any(
            k in results_keys_str
            for k in [
                "psm",
                "propensity",
                "love_plot",
                "iptw",
                "average_treatment_effect",
            ]
        )
        or any(
            bool(re.search(r"(?:^|_)(?:ate|att|cate)(?:_|$)", k))
            and not re.search(
                r"(?:^|_)(?:n_dropped|dropped_n|dropped_count|count_dropped)(?:_|$)", k
            )
            for k in results_keys_normalized
        )
    )
    has_regression_key = any(
        k in results_keys_str
        for k in [
            "cox",
            "logistic",
            "regression",
            "hazard_ratio",
            "or_table",
            "odds_ratio",
        ]
    )
    text_clean_claims = re.sub(
        r"\b(?:does not|do not|did not|cannot|could not|not|no|without)\s+(?:necessarily\s+)?(?:imply|mean|prove|establish|infer|show|indicate|be|have)?\s*(?:causal\w*|causes?|causation)\b",
        " ",
        text_lower,
    )
    text_clean_claims = re.sub(r"\bnon[- ]?causal\b", " ", text_clean_claims)
    text_clean_claims = re.sub(
        r"\b(?:no|without(?:\s+making)?|makes?\s+no)\s+(?:any\s+)?(?:effect\s+of\s+treatment|treatment[- ]effects?|treatment\s+effects?)\s+claims?(?:\s+(?:are|is|were|was)\s+made)?\b",
        " ",
        text_clean_claims,
    )
    text_clean_claims = re.sub(
        r"\bno\s+claims?\s+(?:of|regarding)\s+(?:effect\s+of\s+treatment|treatment[- ]effects?|treatment\s+effects?)(?:\s+(?:are|is|were|was)\s+made)?\b",
        " ",
        text_clean_claims,
    )
    text_clean_claims = re.sub(
        r"\bno\s+(?:effect\s+of\s+treatment|treatment[- ]effects?|treatment\s+effects?)\s+(?:is|are|was|were)\s+(?:made|claimed)\b",
        " ",
        text_clean_claims,
    )
    # Diagnostic / prediction targets (e.g. etiologic work-up, etiologic stroke subtype) are evaluation targets, not causal claims.
    # Restrict substitution to diagnostic work-up and subtype classification targets; retain etiologic research / risk-factor investigations.
    text_clean_claims = re.sub(
        r"\betiolog(?:ic(?:al)?|y)\s+(?:work[- ]?up|workup|evaluation|testing|tests?|(?:[a-zA-Z_-]+\s+)?subtypes?|classification|categories|category|diagnosis)\b",
        "diagnostic workup",
        text_clean_claims,
    )
    # Model-performance statements (e.g. effect of sample size on model discrimination or prediction interval width)
    # are descriptive of model evaluation, not clinical treatment-effect claims.
    # The negative lookahead prevents consuming clinical claims (such as aspirin on mortality) earlier in the sentence.
    text_clean_claims = re.sub(
        r"\b(?:effect|impact)\s+of\s+"
        r"(?:(?!\b(?:on|mortality|death|morbidity|survival|stroke|infarction|mi\b|bleeding|recurrence|blood\s+pressure|bp\b|glucose|hba1c)\b)[^.;\n])+?\s+on\s+"
        r"(?:model\s+)?(?:discrimination|calibration|performance|accuracy|c[- ]statistic|c[- ]index|auroc|roc|brier|fit|prediction\s+intervals?(?:\s+width)?|interval\s+width)\b",
        " ",
        text_clean_claims,
    )
    # Coordinated model-performance clauses (e.g. "and on model discrimination was negligible")
    text_clean_claims = re.sub(
        r"\b(?:and\s+)?(?:also\s+)?on\s+(?:model\s+)?(?:discrimination|calibration|performance|accuracy|c[- ]statistic|c[- ]index|auroc|roc|brier|fit)\s+(?:was|is|were)\s+[^.;\n]+",
        " ",
        text_clean_claims,
    )
    # Statements about improving model performance (e.g. "aspirin use improves model discrimination")
    text_clean_claims = re.sub(
        r"\b[a-zA-Z_-]+(?:\s+[a-zA-Z_-]+){0,2}\s+(?:(?:does|did|do|is|was|were)\s+not\s+|(?:not|failed\s+to)\s+)?(?:improves?|improved|improving|enhances?|enhanced|enhancing)\s+(?:model\s+)?(?:discrimination|calibration|performance|accuracy|c[- ]statistic|c[- ]index|auroc|roc|brier|fit)\b",
        " ",
        text_clean_claims,
    )

    _clinical_outcomes = (
        r"(?:all-cause\s+)?"
        r"(?:mortality|death|morbidity|survival|outcome|stroke|infarction|mi\b|"
        r"bleeding|recurrence|blood\s+pressure|bp\b|glucose|hba1c)"
    )
    _verbs = (
        r"(?:reduces?|reduced|reducing|increases?|increased|increasing|"
        r"decreases?|decreased|decreasing|lowers?|lowered|lowering|"
        r"prevents?|prevented|preventing|improves?|improved|improving|"
        r"cuts?|cut|cutting|slashes?|slashed|slashing|"
        r"halves?|halved|halving|doubles?|doubled|doubling)"
    )
    _aux = r"(?:(?:does|did|do|is|was|were)\s+not\s+|(?:not|failed\s+to)\s+)?"
    _patient_terms = (
        r"patients?|participants?|subjects?|individuals?|persons?|"
        r"cohort|group|subgroup|population|those|men|women|children|adults?"
    )
    _model_terms = (
        r"model|score|index|statistic|c[- ]statistic|c[- ]index|auroc|roc|brier|fit|"
        r"variable|predictor|covariate|feature|sample|parameter|estimate|ratio|"
        r"coefficient|prevalence|incidence|rate|frequency|proportion"
    )
    _non_interventions = (
        rf"{_patient_terms}|{_model_terms}|with|experiencing|showing|having|exhibiting"
    )
    _model_performance_terms = (
        r"model|discrimination|calibration|performance|accuracy|c[- ]statistic|"
        r"c[- ]index|auroc|roc|brier|fit|prediction\s+intervals?|interval\s+width|"
        r"loss|residuals?|overfitting|concordance|sample\s+size|predicted\s+probability"
    )
    _prepositions = (
        r"from|by|to|with|in|at|for|on|during|between|into|through|over|under|after|before"
    )

    has_causal_claims = (
        bool(
            re.search(
                r"\b(?:causal|causes?|causation|treatment[- ]effects?|treatment\s+effects?|effect of|impact of|etiolog\w*)\b",
                text_clean_claims,
            )
        )
        or bool(
            re.search(
                rf"\b(?!(?:{_non_interventions})\b)([a-zA-Z_-]+(?:\s+[a-zA-Z_-]+)?)\s+{_aux}{_verbs}\s+"
                rf"(?:(?:the|a|an)\s+)?(?:(?:risk|rate|incidence|odds|hazard|probability)\s+of\s+)?"
                rf"(?!(?:{_model_performance_terms}|{_prepositions})\b)[a-zA-Z0-9_-]+",
                text_clean_claims,
            )
        )
        or bool(
            re.search(
                rf"\b(?!(?:{_non_interventions})\b)([a-zA-Z_-]+(?:\s+[a-zA-Z_-]+)?)\s+"
                rf"(?:(?:has|had|have|shows?|showed|demonstrates?|demonstrated)\s+)?"
                rf"no\s+(?:statistically\s+significant\s+)?(?:treatment\s+)?effect\s+on\s+"
                rf"(?:(?:the|a|an)\s+)?(?:(?:risk|rate|incidence|odds|hazard|probability)\s+of\s+)?"
                rf"(?!(?:{_model_performance_terms}|{_prepositions})\b)[a-zA-Z0-9_-]+",
                text_clean_claims,
            )
        )
        or bool(
            re.search(
                r"\bno\s+treatment[- ]effect\s+(?:on\s+[^.;\n]+?\s+)?(?:was|is|were)\s+observed\b",
                text_clean_claims,
            )
        )
    )
    clean_design = (
        re.sub(r"[\s\/\-\u2010-\u2015\u2212]+", "_", str(resolved_design).lower())
        if resolved_design
        else ""
    )
    # Explicit observational keywords in clean_design
    # Negated observational labels (e.g. "non_observational", "nonobservational") do not trigger confounding caveats.
    # Distinguish matched randomized trials from observational matching.
    # Token-bounded true randomization test excludes quasi/pseudo/unrandomized/not-randomized/Mendelian-randomization designs and avoids matching "rct" in words like "infarction".
    is_negated_observational = bool(
        re.search(r"(?:^|_)non_?observational(?:_|$)", clean_design)
    )
    is_true_randomized = bool(
        clean_design
        and (
            bool(re.search(r"(?:^|_)rct(?:_|$)", clean_design))
            or bool(
                re.search(
                    r"(?:^|_)(?:randomi[zs]ed|randomi[zs]ation)(?:_|$)",
                    clean_design,
                )
            )
        )
        and not bool(
            re.search(
                r"(?:^|_)(?:quasi|pseudo|un|non|not)_?(?:randomi|rct)",
                clean_design,
            )
        )
        and not bool(re.search(r"mendelian_?randomi", clean_design))
    )
    is_matched_observational = bool(
        ("matched" in clean_design or "matching" in clean_design)
        and not is_true_randomized
    )
    is_pre_experimental = bool(
        clean_design
        and not is_true_randomized
        and bool(
            re.search(
                r"(?:^|_)pre_?experiment(?:al)?(?:$|(?!_(?:period|phase|stage|epoch|interval|baseline|measure|data|time|washout|run_in)(?:_|$))(?:_[a-z0-9]+)*?_(?:study|design|trial|evaluation|investigation|research)(?:_|$))",
                clean_design,
            )
        )
    )
    has_explicit_observational = bool(
        clean_design
        and (
            ("observational" in clean_design and not is_negated_observational)
            or is_matched_observational
            or is_pre_experimental
            or any(
                obs in clean_design
                for obs in [
                    "cohort",
                    "case_control",
                    "cross_sectional",
                    "case_referent",
                    "non_randomized",
                    "nonrandomized",
                    "non_randomised",
                    "nonrandomised",
                    "not_randomi",
                    "notrandomi",
                    "quasi_randomi",
                    "quasirandomi",
                    "pseudo_randomi",
                    "pseudorandomi",
                    "un_randomi",
                    "unrandomi",
                    "quasi_rct",
                    "quasirct",
                    "quasi_experimental",
                    "quasiexperimental",
                    "quasi_experiment",
                    "quasiexperiment",
                    "non_experimental",
                    "nonexperimental",
                    "natural_experiment",
                    "single_arm",
                    "target_trial",
                    "trial_emulation",
                    "emulated_trial",
                    "mendelian_randomi",
                    "mendelianrandomi",
                ]
            )
        )
    )

    # Type 2 study archetype matching:
    # Exclude disease entities (e.g. "type 2 diabetes", "type 2 MI") and diagnostic accuracy / evaluation targets
    # from the Type 2 etiologic effect path, preserving genuine Type 2 etiologic studies.
    is_type_2_archetype = bool(
        clean_design
        and "type_2" in clean_design
        and not bool(
            re.search(
                r"(?:^|_)type_2_(?:diabetes|dm|diabetic|myocardial|mi|bipolar)(?:_|$)",
                clean_design,
            )
        )
    )
    is_type_2_etiologic = bool(
        is_type_2_archetype
        and any(et in clean_design for et in ["etiologic", "etiological", "etiology"])
        and not any(
            diag in clean_design
            for diag in [
                "diagnostic_accuracy",
                "diagnostic_performance",
                "diagnostic_evaluation",
                "diagnostic_study",
                "diagnostic_test",
            ]
        )
    )

    # Diagnostic target / workup exemption:
    # Explicit observational keywords take precedence over the exemption (e.g. "non-randomized diagnostic test").
    # Type 2 etiology workups consistently require confounding caveats.
    is_etiologic_diagnostic_target = bool(
        clean_design
        and (
            any(term in clean_design for term in ["work_up", "workup"])
            or any(
                diag in clean_design
                for diag in [
                    "diagnostic_accuracy",
                    "diagnostic_performance",
                    "diagnostic_evaluation",
                    "diagnostic_study",
                    "diagnostic_test",
                ]
            )
        )
        and not has_explicit_observational
        and not is_type_2_etiologic
    )

    # Etiologic prediction targets (e.g. "prediction model for etiologic stroke subtype")
    # predict an etiologic category/subtype rather than evaluating etiologic effects.
    # Restrict to designs predicting an etiologic category or subtype; do not exempt
    # etiologic studies merely because they mention predictive risk factors.
    is_etiologic_prediction_target = bool(
        clean_design
        and any(et in clean_design for et in ["etiologic", "etiological", "etiology"])
        and any(
            pred in clean_design
            for pred in [
                "prediction",
                "predictive",
                "prognostic",
                "risk_model",
                "risk_prediction",
                "tripod",
            ]
        )
        and any(
            target in clean_design
            for target in [
                "subtype",
                "subtypes",
                "classification",
                "category",
                "categories",
                "diagnosis",
            ]
        )
        and not has_explicit_observational
    )

    has_observational_keyword = bool(
        clean_design
        and (
            is_type_2_etiologic
            or (
                any(
                    et in clean_design
                    for et in ["etiologic", "etiological", "etiology"]
                )
                and not is_etiologic_diagnostic_target
                and not is_etiologic_prediction_target
            )
            or has_explicit_observational
        )
    )
    is_trial = bool(
        clean_design
        and (
            is_true_randomized
            or any(t in clean_design for t in ["trial", "experiment"])
        )
        and not has_observational_keyword
    )
    is_observational = bool(
        clean_design
        and has_observational_keyword
        and not is_trial
        and not is_etiologic_diagnostic_target
    )
    is_confirmed_prediction = bool(
        clean_design
        and any(
            pred_type in clean_design
            for pred_type in [
                "prediction",
                "predictive",
                "prognostic",
                "diagnostic",
                "validation",
                "tripod",
                "risk_model",
                "risk_prediction",
                "risk_score",
                "work_up",
                "workup",
            ]
        )
        and not (
            "risk_score" in clean_design
            and any(m in clean_design for m in ["match", "matched", "matching"])
        )
    ) or (
        not clean_design
        and bool(
            re.search(
                r"\b(?:clinical\s+risk\s+prediction\s+model|prediction\s+model|tripod(?:\s+guidelines|\s+statement)?)\b",
                text_lower,
            )
        )
    )
    is_non_causal_prediction = (
        is_confirmed_prediction
        and not has_causal_claims
        and not is_observational
    )

    is_causal = (
        is_observational
        or has_causal_key
        or (has_regression_key and not is_non_causal_prediction)
        or has_causal_claims
    )
    has_caveat = True
    if is_causal:
        has_caveat = any(
            phrase in text_lower
            for phrase in [
                "unmeasured confounding",
                "residual confounding",
                "confound",
                "e-value",
            ]
        )

    # 3. Numerical Traceability Check
    def _flatten_nums(d):
        nums = []
        if isinstance(d, dict):
            for v in d.values():
                nums.extend(_flatten_nums(v))
        elif hasattr(d, "tolist") and callable(d.tolist):
            nums.extend(_flatten_nums(d.tolist()))
        elif isinstance(d, (list, tuple, set)):
            for v in d:
                nums.extend(_flatten_nums(v))
        elif isinstance(d, (int, float)) and not isinstance(d, bool):
            if pd.notna(d):
                nums.append(float(d))
        elif isinstance(d, pd.DataFrame):
            for col in d.columns:
                for v in d[col]:
                    if pd.notna(v) and isinstance(v, (int, float)) and not isinstance(v, bool):
                        nums.append(float(v))
        return nums

    p_key_re = re.compile(
        r"^(?:p|p[-_]?(?:val|value|adj|overall|non[-_]?linear|omnibus|q|lrt|wald|trend|interaction))s?$",
        re.IGNORECASE,
    )

    def _detect_p_qual(key_str):
        k_lower = key_str.lower()
        if "interaction" in k_lower:
            return "interaction"
        if "trend" in k_lower:
            return "trend"
        if "non" in k_lower and "linear" in k_lower:
            return "nonlinear"
        if "omnibus" in k_lower:
            return "omnibus"
        if "lrt" in k_lower or "likelihood" in k_lower:
            return "lrt"
        if "wald" in k_lower:
            return "wald"
        return None

    def _extract_qualified_p_vals(d, parent_qual=None):
        pvals = []
        if isinstance(d, dict):
            for k, v in d.items():
                k_str = str(k)
                k_qual = _detect_p_qual(k_str) or parent_qual
                if p_key_re.match(k_str):
                    if isinstance(v, dict):
                        pvals.extend(_extract_qualified_p_vals(v, parent_qual=k_qual))
                    elif isinstance(v, pd.DataFrame):
                        pvals.extend(_extract_qualified_p_vals(v, parent_qual=k_qual))
                    else:
                        pvals.extend((k_qual, n) for n in _flatten_nums(v) if 0.0 <= n <= 1.0)
                else:
                    pvals.extend(_extract_qualified_p_vals(v, parent_qual=k_qual))
        elif isinstance(d, (list, tuple, set)):
            for item in d:
                pvals.extend(_extract_qualified_p_vals(item, parent_qual=parent_qual))
        elif isinstance(d, pd.DataFrame):
            for col in d.columns:
                col_str = str(col)
                col_qual = _detect_p_qual(col_str) or parent_qual
                if p_key_re.match(col_str):
                    pvals.extend((col_qual, n) for n in _flatten_nums(d[col]) if 0.0 <= n <= 1.0)
        return pvals

    known_nums = _flatten_nums(results_dict)
    qualified_p_vals = _extract_qualified_p_vals(results_dict)
    # Require an actual identified p-value in results_dict to support p-value inequalities
    allowed_list = (
        [float(x) for x in allowed_untraced] if allowed_untraced is not None else []
    )
    clean_text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", narrative_text)
    doc_has_references = bool(
        re.search(
            r"(?:^|\n)\s*(?:#{1,4}\s*)?(?:References|Bibliography)\b",
            narrative_text,
            re.IGNORECASE,
        )
    )
    raw_matches = list(
        re.finditer(
            r"(?<!\w)(?<![a-zA-Z]-)(?P<num>(?:[-+]?\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?)(?P<pct>\s*%)?",
            clean_text,
        )
    )
    untraced_nums = []
    for m in raw_matches:
        raw_str = m.group("num")
        pct_group = m.group("pct")
        has_pct = bool(pct_group and pct_group.strip() == "%")
        try:
            val = float(raw_str)
        except ValueError:
            continue
        if not (-1e7 < val < 1e7):
            continue

        start = m.start()
        window_before = clean_text[max(0, start - 80) : start].split("\n")[-1]
        window_after = clean_text[m.end() : min(len(clean_text), m.end() + 80)].split("\n")[0]
        window = f"{window_before}{raw_str}{window_after}"

        p_ineq_match = re.search(
            r"(?:((?:[a-zA-Z_-]+(?:\s+(?:ratio\s+)?(?:test|score|statistic))?|[a-zA-Z_-]+))\s*,?\s+)?\bp(?=[-_\s(\<>=≤≥])\s*(?:[-_\s]*(?:value|val|\(?for\s+([a-zA-Z_-]+)\)?|\(?([a-zA-Z_-]+)\)?))?\s*([<>]=?|[≤≥])\s*$",
            window_before,
            re.IGNORECASE,
        )
        is_threshold_def = bool(
            re.search(
                r"\b(?:alpha|significance\s+(?:level|threshold|cutoff)|threshold|cutoff|prespecified(?:\s+(?:level|threshold|cutoff))?|defined\s+as)"
                r"(?:\s*(?:[:=]|\b(?:was|were|is)?\s*(?:set\s+at|defined\s+as|of|at)?\b))?\s*"
                r"\bp(?=[-_\s(\<>=≤≥])\s*(?:[-_\s]*(?:value|val|\(?for\s+[a-zA-Z_-]+\)?|\(?[a-zA-Z_-]+\)?))?\s*([<>]=?|[≤≥])\s*$",
                window_before,
                re.IGNORECASE,
            )
        )
        is_alpha_cutoff = (
            (not p_ineq_match or is_threshold_def)
            and val in (0.05, 0.01, 0.001, 0.005, 0.0001)
            and bool(
                re.search(
                    r"\b(?:alpha|significance\s+(?:level|threshold|cutoff)|threshold|cutoff)\b",
                    window,
                    re.IGNORECASE,
                )
                and not re.search(r"\bp\s*(?:-\s*value\s*)?=\s*$", window_before, re.IGNORECASE)
            )
        )
        p_val = val / 100.0 if has_pct else val
        p_inequality = None
        p_qualifier = None
        if 0.0 <= p_val <= 1.0 and not is_alpha_cutoff and p_ineq_match:
            qual_raw = (
                p_ineq_match.group(2)
                or p_ineq_match.group(3)
                or p_ineq_match.group(1)
            )
            raw_op = p_ineq_match.group(4)
            p_inequality = "<=" if raw_op == "≤" else (">=" if raw_op == "≥" else raw_op)
            p_qualifier = _detect_p_qual(qual_raw) if qual_raw else None
        if p_inequality is not None:
            if p_qualifier is not None:
                candidate_pvals = [
                    pv for qual, pv in qualified_p_vals if qual == p_qualifier
                ]
            else:
                candidate_pvals = [
                    pv for qual, pv in qualified_p_vals if qual is None
                ]

            if not candidate_pvals:
                matched = False
            elif p_inequality == "<":
                matched = any(0.0 <= pv < p_val for pv in candidate_pvals)
            elif p_inequality == "<=":
                matched = any(0.0 <= pv <= p_val for pv in candidate_pvals)
            elif p_inequality == ">":
                matched = any(p_val < pv <= 1.0 for pv in candidate_pvals)
            elif p_inequality == ">=":
                matched = any(p_val <= pv <= 1.0 for pv in candidate_pvals)
            else:
                matched = False
            if not matched:
                untraced_nums.append(val)
            continue

        # Determine decimal precision d from raw token string:
        # For scientific notation (e.g. 1.2e-3), derive d from mantissa decimals minus exponent.
        # For standard decimals (e.g. 1.45), d is the count of fractional digits.
        # For integers (e.g. 412), d = 0.
        if re.search(r"[eE]", raw_str):
            parts = re.split(r"[eE]", raw_str)
            mantissa = parts[0]
            try:
                exp_str = re.split(r"[%]", parts[1])[0]
                exp = int(exp_str)
            except (IndexError, ValueError):
                exp = 0
            mantissa_dec = len(mantissa.split(".")[1]) if "." in mantissa else 0
            d = max(0, mantissa_dec - exp)
        elif "." in raw_str:
            dec_part = raw_str.split(".")[1]
            d = len(re.split(r"[%]", dec_part)[0])
        else:
            d = 0

        # Half-unit tolerance based on claimed precision: abs(val - kn) <= 0.5 * 10^(-d) + 1e-9.
        # Intended behavior: P=0.01 passes when true P=0.012 (correct 2-decimal rounding),
        # whereas 0.02 and 0.013 are rejected.
        tol = 0.5 * (10.0 ** (-d)) + 1e-9

        matched = False
        for kn in known_nums:
            # 1. Direct match:
            if abs(val - kn) <= tol:
                matched = True
                break

            # 2. Percentage scale match: only permitted when token explicitly has '%' attached
            if has_pct:
                if abs(val - kn * 100.0) <= tol:
                    matched = True
                    break

        # 3. Allowed untraced whitelist: uses the exact same half-unit precision tolerance
        if not matched and allowed_list:
            if any(abs(val - aw) <= tol for aw in allowed_list):
                matched = True

        if not matched:
            is_ci_level = val in (90.0, 95.0, 99.0) and bool(
                re.search(r"\b(?:CI|confidence|credible)\b", window, re.IGNORECASE)
            )
            is_scale_const = val in (0.0, 100.0) and bool(
                re.search(r"\b(?:scale|normalized|range|total)\b", window, re.IGNORECASE)
            )
            is_version = bool(
                re.search(r"\b(?:Version|v\.)\s*$", window_before, re.IGNORECASE)
            )
            is_counter = "." not in raw_str and bool(
                re.search(
                    r"\b(?:Table|Figure|Fig\.?|Tier|Phase|Stage|Grade|Step|Level|Item|Section)\s*$",
                    window_before,
                    re.IGNORECASE,
                )
            )
            is_exponent = "." not in raw_str and bool(
                re.search(r"[a-zA-Z\)]\^$", window_before)
            )
            # Author-year citation without "et al." (Austin (2011), VanderWeele & Ding 2017); engine parity (ADR 79)
            is_author_year = (
                1900.0 <= val <= 2035.0
                and "." not in raw_str
                and bool(re.search(r"\b[A-Z][a-zA-Z'’\-]+(?:\s+(?:&|and)\s+[A-Z][a-zA-Z'’\-]+)?,?\s*\(?$", window_before))
                and (
                    # Bracket or next author closes a citation; ; , . : or line end only after
                    # "(", "Surname," or a second author, so "Total 1987." stays untraced (ADR 79)
                    bool(re.match(r"^[a-z]?(?:\s*[)\]]|\s+(?:and|&)\s+[A-Z])", window_after))
                    or (
                        bool(re.match(r"^[a-z]?\s*(?:[;,.:]|$)", window_after))
                        and bool(re.search(r"(?:\(|,|(?:&|\band)\s+[A-Z][a-zA-Z'’\-]+)\s*\(?$", window_before))
                    )
                )
            )
            # Analysis scale constants: "per 10 years", "caliper 0.2 SD" (engine parity, ADR 81)
            is_scale_unit = bool(re.search(r"\b(?:per|caliper(?:\s+(?:of|width))?)\s*$", window_before, re.IGNORECASE))
            # Section number opening a line or Markdown heading (## 4. Results, 6.1 Sensitivity Analysis)
            is_section = not has_pct and bool(re.match(r"^\s*(?:#{1,6}\s*)?$", window_before)) and (
                (bool(re.fullmatch(r"\d{1,2}", raw_str)) and 1.0 <= val <= 50.0 and bool(re.match(r"^[.)]\s+[A-Z]", window_after)))
                or (bool(re.fullmatch(r"\d{1,2}\.\d{1,2}", raw_str)) and bool(re.match(r"^\.?\s+[A-Z]", window_after)))
            )
            date_prefix_pattern = (
                r"(?:"
                r"\b(?:in|year|years|during|period|from|dated|since|until|between)(?:\s+(?:the\s+)?(?:calendar|fiscal|academic|study)?(?:\s*year)?)?(?:\s+"
                r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?))?"
                r"|\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
                r"|\b(?:19\d{2}|20\d{2})\s*(?:-|–|/|\b(?:to|through|until|and)\b)"
                r")\s*$"
            )
            is_year = (
                1900.0 <= val <= 2035.0
                and "." not in raw_str
                and (
                    bool(re.search(date_prefix_pattern, window_before, re.IGNORECASE))
                    or bool(
                        re.search(
                            r"(?:\b(?:et\s+al\.?|JAMA|NEJM|Lancet|BMJ|Med|Crit\s+Care|J\b|Ann\b|Clin\b|Sci\b|Press|Univ|doi|vol\.?)\b|\bpp?\.)",
                            window_before,
                            re.IGNORECASE,
                        )
                    )
                )
            )
            is_citation = (
                "." not in raw_str
                and (
                    ("<sup" in window_before and "</sup>" in window_after and window_before.rfind("<sup") > window_before.rfind("</sup>"))
                    or bool(re.search(r"\[\^\s*$", window_before) and window_after.startswith("]"))
                    or bool(re.search(r"\^\[(?:[\d\s,]+|[\d\s,]*[-–]\s*)*$", window_before) and re.match(r"^(?:[\s,\-–]+\d+)*\s*\](?![a-zA-Z])", window_after))
                    or bool(
                        1.0 <= val <= 100.0
                        and re.search(r"\[(?:[\d\s,]+|[\d\s,]*[-–]\s*)*$", window_before)
                        and re.match(r"^(?:[\s,\-–]+\d+)*\s*\](?![a-zA-Z])", window_after)
                    )
                    or bool(re.search(r"\b(?:Ref\.?|Refs\.?|Reference|References|Citation|Citations)\s*#?$", window_before, re.IGNORECASE))
                    or bool(
                        doc_has_references
                        and 1.0 <= val <= 100.0
                        and re.match(r"^\s*(?:\[\s*)?$", window_before)
                        and re.match(r"^\s*(?:\]\s*)?[\.\)]?\s+[A-Z][a-zA-Z'’\-]+(?:\s+[A-Z]{1,3}\b|\s*,\s*[A-Z]{1,3}\b|\s+et\s+al)", window_after)
                    )
                )
            )
            if not (
                is_ci_level
                or is_alpha_cutoff
                or is_scale_const
                or is_version
                or is_counter
                or is_exponent
                or is_year
                or is_author_year
                or is_scale_unit
                or is_section
                or is_citation
            ):
                untraced_nums.append(val)

    traceability_passed = len(untraced_nums) == 0

    return {
        "passed": len(violations) == 0 and has_caveat and traceability_passed,
        "phi_violations": violations,
        "has_causal_caveat": has_caveat,
        "untraced_numbers": untraced_nums,
        "traceability_passed": traceability_passed,
    }
```

---

## 4. Evidence-Base Verification & Superscript Citations Pipeline

Disclosed to [evidence-pipeline.md](evidence-pipeline.md). Read [evidence-pipeline.md](evidence-pipeline.md) whenever **Polish Grilling Gate (#6)** or **Output Alignment Gate (#5c)** citation verification is engaged.
```

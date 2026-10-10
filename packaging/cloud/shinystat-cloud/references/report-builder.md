# Cloud Multi-Format Report Builder & Pillar 5 Integrity Reference

In cloud sandboxes (e.g. Claude Web Artifacts), `medstat` is not installed as an external package. This reference provides self-contained, pure-Python reporting recipes that assemble biostatistical findings into publication-grade documents (Markdown, self-contained HTML, Word `.docx`, and PowerPoint `.pptx`) with zero custom dependencies.

> [!IMPORTANT]
> **Default Output Format**: If the user does not specify an output format, **always default to Self-Contained HTML (`.html`)** with embedded base64 figures and ICMJE 3-rule CSS styling (`generate_html_report`).

---

## 1. Cloud Report Generation Architecture

1. **Default & Primary Formats (Zero Extra Dependencies)**:
   - **Self-Contained HTML (`.html`)**: **Default deliverable format** when unspecified. Always available; embeds plots via base64 data URIs and applies ICMJE 3-rule CSS styling.
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
            lines.append(df.to_string(index=False) + "\n")

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("\n".join(lines), encoding="utf-8")
    return str(out_p)
```

### B. Self-Contained HTML Report with Embedded Figures

```python
import base64
import html
from pathlib import Path
import pandas as pd


def generate_html_report(
    title: str,
    results: dict,
    tables: dict[str, pd.DataFrame],
    figure_paths: list[str] = None,
    out_path: str = "report.html",
) -> str:
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #1e293b; max-width: 900px; margin: 40px auto; padding: 0 20px; }",
        "  h1 { font-size: 26px; border-bottom: 2px solid #0f172a; padding-bottom: 10px; }",
        "  h2 { font-size: 20px; color: #0284c7; margin-top: 30px; }",
        "  table.pub-table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 14px; }",
        "  table.pub-table thead tr { border-top: 2px solid #0f172a; border-bottom: 1px solid #0f172a; }",
        "  table.pub-table tbody tr:last-child { border-bottom: 2px solid #0f172a; }",
        "  table.pub-table th, table.pub-table td { padding: 8px 12px; text-align: left; }",
        "  .figure-container { text-align: center; margin: 25px 0; }",
        "  .figure-container img { max-width: 100%; height: auto; border: 1px solid #e2e8f0; }",
        "  .caption { font-size: 13px; color: #64748b; font-style: italic; margin-top: 6px; }",
        "</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
    ]

    # Narrative
    if "narrative" in results:
        parts.append(f"<p>{html.escape(results['narrative'])}</p>")

    # Tables
    for caption, df in tables.items():
        parts.append(
            f"<div class='caption'><strong>{html.escape(caption)}</strong></div>"
        )
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

    # Figures (Base64 embedded)
    if figure_paths:
        for fig_p_str in figure_paths:
            fp = Path(fig_p_str)
            if fp.exists():
                b64 = base64.b64encode(fp.read_bytes()).decode("utf-8")
                parts.append("<div class='figure-container'>")
                parts.append(f"<img src='data:image/png;base64,{b64}'>")
                parts.append("</div>")

    parts.append("</body></html>")
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("\n".join(parts), encoding="utf-8")
    return str(out_p)
```

### C. Standalone Word (`.docx`) Report (when `python-docx` is installed)

```python
try:
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
            for i, row in df.iterrows():
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

---

## 3. Standalone Reporting Integrity Verification

Before releasing any report, run the standalone verification routine:

```python
import re
from pathlib import Path
import pandas as pd


def verify_standalone_integrity(
    narrative_text: str, results_dict: dict, table_dfs: list[pd.DataFrame] = None
) -> dict:
    """Verifies numerical traceability and screens for PHI without importing medstat."""
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
    keys_str = " ".join(results_dict.keys()).lower()
    is_causal = any(
        k in keys_str
        for k in ["psm", "propensity", "cox", "hazard_ratio", "odds_ratio", "logistic"]
    )
    has_caveat = True
    if is_causal:
        has_caveat = any(
            phrase in narrative_text.lower()
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
            nums.append(float(d))
        return nums

    known_nums = _flatten_nums(results_dict)
    clean_text = narrative_text.replace(",", "")
    raw_matches = list(
        re.finditer(
            r"(?<!\w)(?<![a-zA-Z]-)(?:[-+]?\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?",
            clean_text,
        )
    )
    ignored_structural = {
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        95.0,
        99.0,
        100.0,
        0.05,
        0.01,
        0.001,
    }
    untraced_nums = []
    for m in raw_matches:
        raw_str = m.group(0)
        try:
            val = float(raw_str)
        except ValueError:
            continue
        if not (-1e7 < val < 1e7):
            continue

        if abs(val) <= 1.0:
            if re.search(r"[eE]", raw_str):
                parts = re.split(r"[eE]", raw_str)
                mantissa = parts[0]
                try:
                    exp = int(parts[1])
                except (IndexError, ValueError):
                    exp = 0
                mantissa_dec = len(mantissa.split(".")[1]) if "." in mantissa else 0
                dec_digits = max(0, mantissa_dec - exp)
            else:
                dec_digits = len(raw_str.split(".")[1]) if "." in raw_str else 2
            abs_tol = min(0.02, 10.0 ** (-dec_digits))
        else:
            abs_tol = 0.02

        matched = False
        for kn in known_nums:
            if abs(val - kn) <= abs_tol:
                matched = True
                break
            if abs(val) > 1.0 and abs(kn) > 1e-9 and abs(val - kn) / abs(kn) <= 0.01:
                matched = True
                break
            if abs(val) > 1.0 and abs(kn) <= 1.0 and abs(val - kn * 100.0) <= 0.05:
                matched = True
                break

        if not matched:
            start = m.start()
            window_before = clean_text[max(0, start - 30) : start].split("\n")[-1]
            is_counter = bool(
                re.search(
                    r"\b(?:Table|Figure|Fig\.?|Tier|Phase|Stage|Grade|Step|Level|Item|Section)\s*$",
                    window_before,
                    re.IGNORECASE,
                )
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
                2010.0 <= val <= 2035.0
                and "." not in raw_str
                and bool(re.search(date_prefix_pattern, window_before, re.IGNORECASE))
            )
            is_structural_10 = val == 10.0 and is_counter
            if not (val in ignored_structural or is_structural_10 or is_year):
                untraced_nums.append(val)

    untraced_rate = len(untraced_nums) / max(len(raw_matches), 1)
    traceability_passed = len(untraced_nums) <= 1 or untraced_rate <= 0.15

    return {
        "passed": len(violations) == 0 and has_caveat and traceability_passed,
        "phi_violations": violations,
        "has_causal_caveat": has_caveat,
        "untraced_numbers": untraced_nums,
        "traceability_passed": traceability_passed,
    }
```

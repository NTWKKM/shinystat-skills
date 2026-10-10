# Multi-Format Report Builder & Pillar 5 Integrity Reference

`medstat.reporting` provides an end-to-end publishing pipeline that transforms raw biostatistical outputs into publication-grade documents (Markdown, self-contained HTML, Word `.docx`, PowerPoint `.pptx`, and PDF).

> [!IMPORTANT]
> **Default Output Format**: When the user does not specify an output format, **always default to self-contained HTML (`render_html`, `.html`)** with embedded base64 figures and responsive ICMJE/NEJM/JAMA table styling.

---

## 1. Document Architecture (Report IR)

The **Report Intermediate Representation (IR)** acts as the Single Source of Truth (SSOT). Format-agnostic blocks are assembled into a `ReportDocument`:

```python
from medstat.reporting.ir import (
    ReportDocument,
    HeadingBlock,
    ParagraphBlock,
    TableBlock,
    FigureBlock,
    CalloutBlock,
)

doc = ReportDocument(
    title="Clinical Validation of Troponin-T for Acute Myocardial Infarction",
    authors=["Jane Doe, MD", "John Smith, PhD"],
    date="2026-10-09",
    institution="Biostatistics & Emergency Care Research Unit",
    results_dict=results,  # Single source of truth dictionary
)

# Chained block builder
doc.add_heading("Executive Summary", level=1)
doc.add_paragraph(
    f"In this cohort of {results['sample_size']} patients, the biomarker achieved an AUC of {results['auc']:.3f}."
)
doc.add_callout(
    "Observational study design requires cautious interpretation regarding unmeasured confounding.",
    level="warning",
)
doc.add_table(
    table_df,
    caption="Table 1. Baseline Characteristics",
    footnote="Values expressed as Mean (SD) or N (%).",
)
doc.add_figure(fig_result)
```

---

## 2. Multi-Format Renderers

### Markdown (`render_markdown`)
Exports clean GitHub Flavored Markdown and copies linked high-resolution figures into an assets folder:
```python
from medstat.reporting.renderers import render_markdown

md_path = render_markdown(
    doc, out_path="reports/ami_study.md", assets_dir_name="figures"
)
```

### Self-Contained HTML (`render_html`)
Produces a single, fully offline HTML file with embedded base64 images and ICMJE 3-rule table styling:
```python
from medstat.reporting.renderers import render_html

html_path = render_html(doc, out_path="reports/ami_study.html")
```

### Word Document (`render_docx`)
Builds a native Word `.docx` file using `python-docx` with:
- Standard 1.0 inch margins.
- ICMJE 3-rule publication table borders via OpenXML (`<w:tblBorders>`).
- Centered figures at exact 6.5 inch printable width.
```python
from medstat.reporting.renderers import render_docx

docx_path = render_docx(doc, out_path="reports/ami_study.docx")
```

### PowerPoint Presentation (`render_pptx`)
Builds an executive 16:9 widescreen presentation (`13.333" x 7.5"`) using `python-pptx` with:
- Title slide with metadata and affiliation styling.
- **1 figure per slide layout**: High-resolution image on the left, structured callout card with key takeaways on the right.
- Publication-formatted table slides.
```python
from medstat.reporting.renderers import render_pptx

pptx_path = render_pptx(doc, out_path="reports/ami_presentation.pptx")
```

### Publication PDF (`render_pdf`)
Generates high-fidelity PDF documents via headless Chromium (`Playwright`), with automatic fallback to LibreOffice (`soffice`):
```python
from medstat.reporting.renderers import render_pdf

pdf_path = render_pdf(doc, out_path="reports/ami_study.pdf")
```

---

## 3. Pillar 5: Reporting Integrity Invariants

Before finalizing any document, run `verify_report_integrity`:

```python
from medstat.reporting.integrity import verify_report_integrity

audit = verify_report_integrity(doc)
if not audit.passed:
    print(audit.summary())
```

### Enforced Invariants
1. **Numerical Traceability**: Narrative numbers are verified against `doc.results_dict` within $\pm 0.02$ or 1% relative error (including percentage scale $\pm 0.05$). Common statistical and index constants (`IGNORED_NUMBERS`) are bypassed. An untraced rate of up to 15% (minimum 1 untraced value) is tolerated for non-statistical headings and structural annotations.
2. **Zero-PHI Compliance**: Automated regex screening checks text, table cells, headers, and figure metadata for Hospital Numbers (`HN`), Thai 13-digit National IDs, phone numbers, dates of birth, and honorific-prefixed names (`Mr.`, `Mrs.`, `Ms.`, `Dr.`, `นาย`, `นาง`, `นพ.`, `พญ.`). A passing automated audit does not prove absolute zero PHI (names without honorifics are not detected); manual clinical review remains mandatory.
3. **Observational Causal Inference & E-Value Caveats**: If observational regression, Cox survival, or PSM models are reported, narrative must explicitly address unmeasured confounding and E-values.
4. **Retention Flow & Methodology**: Enforces presence of cohort participant flow details and statistical software/version citations.

# Multi-Format Report Builder & Pillar 5 Integrity Reference

`medstat.reporting` provides an end-to-end publishing pipeline that transforms raw biostatistical outputs into publication-grade documents (Markdown, self-contained HTML, Word `.docx`, PowerPoint `.pptx`, and PDF).

> [!IMPORTANT]
> **Output Format Alignment Gate**: When the user prompt does not explicitly specify a deliverable format, **proactively ask the user upfront** (via `ask_question` in Antigravity or direct prompt) to align on the deliverable format, recommending **(Dual / Toggle View) Interactive HTML Report (`render_html`, `.html`)** where interactive Plotly.js charts serve as the primary view, paired with a tab/button to switch to "Publication 300 DPI Figure (Static)" for journal submission and downloads.

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
Produces a single, fully offline-capable HTML file with embedded base64 images, ICMJE 3-rule table styling, and **Dual / Toggle View when a Plotly specification is supplied**:
- **Primary Interactive View**: Plotly.js interactive charts (`Interactive (Plotly)`) as the default active view (hover over ROC cutpoints, zoom Kaplan-Meier timelines, inspect calibration deciles) when using `medstat.figures.html.*` with a supplied Plotly specification.
- **Publication Static View**: Seamless segmented switch to `"Publication 300 DPI Figure (Static)"` Matplotlib figures formatted for journal submission (NEJM/JAMA). Without a Plotly specification, the report contains only this static view.
- **Direct Downloads**: One-click download buttons for high-resolution 300 DPI PNG and underlying CSV dataset.
- **Offline & Sandbox Resilience**: Automatic fallback notice with manual toggle button if Plotly CDN is unreachable.
- **Print Optimization**: `@media print` cleanly suppresses toggle chrome and prints high-resolution static images.

```python
from medstat.figures.html import plot_dual_roc_curve
from medstat.reporting.renderers import render_html

# Generate dual figure (Matplotlib 300 DPI PNG + Plotly interactive)
fig_res = plot_dual_roc_curve(roc_data, out_path="figures/roc.png")
doc.add_figure(fig_res)

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

---

## 4. Evidence-Base Verification & Superscript Citations Pipeline

When the **Polish Grilling Gate (#6)** is engaged (via Gate #5 Full Polish selection or triggered on-demand), execute this 4-step evidence verification and citation workflow (if Skip Polish is chosen at Gate #5, standard analysis deliverables proceed directly through the normal flow):

### Step 1: Clinical Claims vs Analyzed Data Audit
- Cross-audit every narrative statement in Discussion and Results against `results_dict` and study design.
- Prohibit causal assertions (e.g., "Drug X reduces mortality") when analyzing unadjusted or observational cohorts; require associative framing ("was associated with lower mortality") and E-value sensitivity discussion.
- Verify that non-significant results ($P > 0.05$) are strictly stated as "insufficient evidence to reject the null hypothesis" rather than claiming equivalence or "no effect".
- Ensure point estimates, 95% CIs, and $P$-values match tabular outputs exactly.

### Step 2: Empirical Literature Grounding via WebSearch & MCP Databases
- Audit every cited study, trial, or guideline against real biomedical literature:
  - **`search_web`**: Verify publication metadata via query: `"<First Author> <Journal> <Year> <Title keywords>"` or DOI/PMID.
  - **`medical-terminology` MCP**: Validate medical terminology, standardized ontology codes (SNOMED CT, LOINC, RxNorm, ICD-10), and laboratory reference ranges.
- Confirm that the cited paper's primary findings directly support the specific sentence in the manuscript.
- **Anti-Phantom Citation Invariant**: Zero fabricated papers, hallucinated author lists, or synthetic DOIs. If a citation cannot be verified or its actual conclusions contradict the narrative statement, halt at Grilling Gate #6 to reconcile or replace the citation.

### Step 3: Sequential In-Text Superscript Citations
- Number citations sequentially in order of first appearance using superscript notation (`<sup>1</sup>`, `<sup>2,3</sup>`, `<sup>4–6</sup>`).
- Multi-renderer implementation:
  - **HTML (`render_html`)**: `Early resuscitation improves survival.<sup class="citation"><a href="#ref-1">1</a></sup>` (styled via CSS `sup.citation`, un-escaped safely).
  - **Word (`render_docx`)**: Converted into native OpenXML text runs with `run.font.superscript = True` and `8.5 pt` font size.
  - **PowerPoint (`render_pptx`)**: Cleaned to Unicode superscripts (`¹`, `²`) or bracketed numbers `[1]`, eliminating raw HTML tag leaks.
  - **Markdown (`render_markdown`)**: Rendered cleanly as `<sup>1</sup>` or `[^1]`.

### Step 4: Bibliographic Reference List Assembly by Journal House Style
Format the concluding **References** section strictly in the locked Journal House Style:

| Journal Style | In-Text Citation | Reference List Format Example |
| :--- | :--- | :--- |
| **NEJM** | `...guidelines.<sup>1</sup>` | `1. Singer M, Deutschman CS, Seymour CW, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA 2016;315:801-10.` |
| **JAMA** | `...guidelines.<sup>1</sup>` | `1. Singer M, Deutschman CS, Seymour CW, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA. 2016;315(8):801-810. doi:10.1001/jama.2016.0287` |
| **Lancet** | `...guidelines<sup>1</sup>` | `1 Singer M, Deutschman CS, Seymour CW, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA 2016; 315: 801–10.` |
| **BMJ** | `...guidelines.<sup>1</sup>` | `1 Singer M, Deutschman CS, Seymour CW, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA 2016;315:801-10. doi:10.1001/jama.2016.0287` |
| **Clinical** | `...guidelines<sup>1</sup>` | `1. Singer, M. et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA 315, 801–810 (2016).` |

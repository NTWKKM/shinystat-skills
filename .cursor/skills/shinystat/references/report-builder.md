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

When the **Polish Grilling Gate (#6)** is engaged (via explicit prompt request e.g. "หาหลักฐานมาประกอบ" / "find evidence", Gate #5 Full Polish selection, or triggered on-demand), execute this 4-step evidence verification and citation workflow (if Standard Skip Polish is active, standard deliverables proceed directly using strictly user-provided sources, resolving incomplete citations and comparing real empirical findings by routing citation lookups through available literature MCPs first, with search_web as fallback, without exploratory literature reviews; if user provided no references, omit the References section entirely):

### Step 1: Clinical Claims vs Analyzed Data Audit
- Cross-audit every narrative statement in Discussion and Results against `results_dict` and study design.
- Base permitted causal language on the prespecified estimand, study design, confounder control, and identification assumptions (do not treat weighting alone as sufficient or unweighted observational data as automatically prohibiting causal claims; report associative framing where identification is unverified, and require E-values only where defined and appropriate for the supported effect measure and sensitivity question).
- Derive the significance threshold ($\alpha$) and confidence-interval level ($1-\alpha$) from analysis metadata rather than assuming fixed $0.05$ and $95\%$; verify that non-significant results in superiority or null-hypothesis tests ($P > \alpha$) are strictly stated as "insufficient evidence to reject the null hypothesis" rather than claiming equivalence or "no effect", while preserving separate margin- and CI-based interpretation for prespecified non-inferiority or equivalence analyses.
- Compare normalized point estimates, confidence intervals, and $P$-values against tabular outputs using the magnitude- and percentage-specific tolerance rule (`abs_tol = min(0.02, 10**(-dec_digits))` for $|val| \le 1.0$, $0.02$ absolute or $0.01$ relative for $|val| > 1.0$, or $\min(0.05, abs\_tol \times 100.0)$ for percentages).

### Step 2: Empirical Research Grounding & Primary Source Verification via MCPs & Bibliographic Engines

- **Dynamic Bibliographic Source & MCP Adaptation**:
  - Dynamically discover and adapt to any literature or medical database MCP servers connected in the environment (e.g. PubMed MCP, PRISMA MCP, NEJM, Cochrane, Semantic Scholar, ScienceDirect, `medical-terminology`).
  - Prioritize specialized MCP tools when available for fast, authoritative metadata and abstract retrieval; seamlessly fallback to `search_web` / PubMed web queries when dedicated MCP database servers are absent.
- **Step 2A: User-Scoped Source Resolution & Online Empirical Comparison (Standard & Full Polish)**:
  - When user-provided materials, clinical notes, or prompts cite or mention external studies (e.g., "Singer 2016", "Sepsis-3", "KDIGO 2012") with or without full text or complete bibliographic metadata:
    - Do NOT leave it as an informal or truncated citation.
    - Query available MCPs or search engines to retrieve and verify the authentic primary publication and claim support using bibliographic fields appropriate to the source type: for journal articles, Title, full Author list, Journal, Year, Volume, Pages (or article number / e-locator when pages are not assigned), and DOI (when assigned); for clinical practice guidelines, consensus statements, government reports, or standards, Title, issuing body/organization, Year, version or report identifier, and official URL when no DOI exists; and for books or older publications, authors/editors, Title, publisher, and Year.
    - **Discussion Empirical Grounding (Conditional on Source Content)**: Even when the user supplies a local document or full text, query online sources (MCPs/search) if searchable. Extract and present real empirical research figures (sample sizes, benchmark effect sizes, control group event rates, 95% CIs) as comparable ONLY when the source actually reports them, matches the target estimand, population, outcome, and time horizon, and directly supports the specific **Discussion narrative** claim. Sources that do not report relevant empirical figures (e.g., clinical guidelines, consensus criteria, classification frameworks) or differ in clinical context/estimand must be cited for the conceptual claims, recommendations, or criteria they support without inventing, importing, or presenting non-comparable figures.
    - **Offline / Unavailable Literature Services**: When online literature services (MCPs, `search_web`) are unavailable or unreachable, authenticated local full text with source-appropriate bibliographic metadata satisfies verification ONLY when carrying an accepted provenance or validation signal (user-provided file in workspace/prompt, publisher/journal masthead or frontmatter, DOI/PMID header, or official organization publication cover; accepting issuing body and official URL/identifier when no DOI exists); otherwise halt explicitly (anti-phantom citation invariant).
    - Insert the complete, verified study into the References list and assign sequential superscript citations (`<sup>1</sup>`, `<sup>2,3</sup>`) formatted in the chosen Journal House Style.
  - **Zero Unsolicited References in Standard Mode**: If the user does not provide references or mention external literature, the Standard Mode deliverable contains NO external literature References section.
- **Step 2B: Relevant Literature Enrichment (Full Polish Only)**:
  - Actively search available literature MCPs / `search_web` for additional authentic, relevant published studies, clinical trials, or guidelines that directly relate to the study's clinical topic, target population, and analyzed outcomes.
  - Extract and present real empirical findings and comparison benchmarks as comparable ONLY when reported by the source, matching the target estimand, population, outcome, and time horizon, and directly supporting a specific Discussion claim (otherwise cite the source for its conceptual claims without figures or omit it), assigning sequential superscript citations and expanding the Reference list.
- **Verification & Safety Rules (All Modes)**:
  - **Bibliographic Verification**: Verify bibliographic metadata via query: `"<First Author> <Journal> <Year> <Title keywords>"` or DOI/PMID (or `"<Issuing Body> <Title> <Year>"` for guidelines/reports) across available MCPs or `search_web`. Require bibliographic fields appropriate to the source type (accepting issuing body and official URL/identifier when no DOI exists). When literature services are unavailable, authenticated local full text satisfies verification only when carrying an accepted provenance signal with source-appropriate bibliographic metadata; explicitly halt if verification is unavailable.
  - **`medical-terminology` MCP**: Validate medical terminology, standardized ontology codes (SNOMED CT, LOINC, RxNorm, ICD-10), and laboratory reference ranges.
  - Confirm that the cited paper's primary findings directly support the specific sentence in the manuscript.
  - **Anti-Phantom Citation Invariant**: Zero fabricated papers, hallucinated author lists, or synthetic DOIs. If a citation cannot be verified or its actual conclusions contradict the narrative statement, halt at Grilling Gate #6 to reconcile or replace the citation.

### Step 3: Sequential In-Text Superscript Citations
- Number citations sequentially in order of first appearance using superscript notation (`<sup>1</sup>`, `<sup>2,3</sup>`, `<sup>4–6</sup>`). Track citation-to-reference assignments so repeated citations reuse the same reference number and anchor instead of creating duplicate entries.
- Multi-renderer implementation:
  - **HTML (`render_html`)**: `Early resuscitation improves survival.<sup class="citation"><a href="#ref-1">1</a></sup>` (styled via CSS `sup.citation`, un-escaped safely; emits `id="ref-N"` on reference paragraphs matching each citation's `href="#ref-N"` target).
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

# Evidence-Base Verification & Superscript Citations Pipeline (Gate #6)

Reached from `SKILL.md` Pillar 5, **Output Alignment Gate** (`#5c`), and **Polish Grilling Gate** (`#6`). Defines the 4-step evidence verification, Zero-PHI query boundary, and sequential superscript citation workflow for both `shinystat` and `shinystat-cloud`.

When the **Polish Grilling Gate (#6)** is engaged (via explicit prompt request e.g. "หาหลักฐานมาประกอบ" / "find evidence", Gate #5c Full Polish selection, or triggered on-demand), execute this 4-step pipeline. When **Standard (Skip Polish)** is active, scope citations strictly to user-provided sources—resolving incomplete citations and comparing real empirical findings by routing citation lookups through available literature MCPs first, with any available web search tool (e.g., `web_search`, `search_web`) as fallback, without exploratory literature searches; if the user provided no references, omit the References section entirely.

---

## Step 1: Clinical Claims vs Analyzed Data Audit

- Cross-audit every narrative statement in Discussion and Results against `results_dict` and study design.
- Base permitted causal language on the prespecified estimand, study design, confounder control, and identification assumptions (do not treat weighting alone as sufficient or unweighted observational data as automatically prohibiting causal claims; report associative framing where identification is unverified, and require E-values only where defined and appropriate for the supported effect measure and sensitivity question).
- Derive the significance threshold ($\alpha$) and confidence-interval level ($1-\alpha$) from analysis metadata rather than assuming fixed $0.05$ and $95\%$; verify that non-significant results in superiority or null-hypothesis tests ($P > \alpha$) are strictly stated as "insufficient evidence to reject the null hypothesis" rather than claiming equivalence or "no effect", while preserving separate margin- and CI-based interpretation for prespecified non-inferiority or equivalence analyses.
- Compare normalized point estimates, confidence intervals, and $P$-values against tabular outputs using the half-unit decimal precision tolerance rule (`abs(val - kn) <= 0.5 * 10**(-d) + 1e-9`, where `d` is the number of decimal digits displayed in the narrative text; $d=0$ for integers), strictly permitting percentage-scale matching (`kn * 100.0`) only when the token explicitly carries a `%` sign, and enforcing the same half-unit precision tolerance on `allowed_untraced` constants.

---

## Step 2: Empirical Research Grounding & Primary Source Verification via MCPs & Bibliographic Engines

- **Dynamic Bibliographic Source & MCP Adaptation**:
  - Dynamically discover and adapt to any literature or medical database MCP servers connected in the environment (e.g. PubMed MCP, PRISMA MCP, NEJM, Cochrane, Semantic Scholar, ScienceDirect, or optional `medical-terminology` MCP when available).
  - Prioritize specialized MCP tools when available for fast, authoritative metadata and abstract retrieval; seamlessly fallback to any available web search tool (e.g., `web_search`, `search_web`) or PubMed web queries when dedicated MCP database servers are absent.
- **Step 2A: User-Scoped Source Resolution & Online Empirical Comparison (Standard & Full Polish)**:
  - When user-provided materials, clinical notes, or prompts cite or mention external studies (e.g., "Singer 2016", "Sepsis-3", "KDIGO 2012") with or without full text or complete bibliographic metadata:
    - Always resolve it into a complete, verified citation.
    - **Zero-PHI Literature Query Boundary**: Keep raw user materials, clinical-note text, and patient excerpts out of external MCP or web search queries. Build external requests strictly from bibliographic metadata and, when needed, de-identified, non-patient-specific topic or claim descriptors (e.g., disease entity, estimand, intervention); redact any Protected Health Information (HN, patient name, DOB, Thai National ID, phone) before dispatching queries.
    - Query available MCPs or any available web search tool to retrieve and verify the authentic primary publication and claim support using bibliographic fields appropriate to the source type: for journal articles, Title, full Author list, Journal, Year, Volume, Pages (or article number / e-locator when pages are not assigned), and DOI (when assigned); for clinical practice guidelines, consensus statements, government reports, or standards, Title, issuing body/organization, Year, version or report identifier, and official URL when no DOI exists; and for books or older publications, authors/editors, Title, publisher, and Year.
    - **Discussion Empirical Grounding (Conditional on Source Content)**: Even when the user supplies a local document or full text, query online sources (MCPs or any available web search tool) if searchable. Extract and present real empirical research figures (sample sizes, benchmark effect sizes, control group event rates, 95% CIs) as comparable ONLY when the source actually reports them, matches the target estimand, population, outcome, and time horizon, and directly supports the specific **Discussion narrative** claim. Sources that do not report relevant empirical figures (e.g., clinical guidelines, consensus criteria, classification frameworks) or differ in clinical context/estimand must be cited for the conceptual claims, recommendations, or criteria they support without inventing, importing, or presenting non-comparable figures.
    - **Offline / Unavailable Literature Services**: When online literature services (MCPs and web search tools) are unavailable or unreachable, authenticated local full text with source-appropriate bibliographic metadata satisfies verification ONLY when carrying an accepted provenance or validation signal (user-provided file in workspace/prompt, publisher/journal masthead or frontmatter, DOI/PMID header, or official organization publication cover; accepting issuing body and official URL/identifier when no DOI exists); otherwise halt explicitly (anti-phantom citation invariant).
    - Insert the complete, verified study into the References list and assign sequential superscript citations (`<sup>1</sup>`, `<sup>2,3</sup>`) formatted in the chosen Journal House Style.
  - **Zero Unsolicited References in Standard Mode**: If the user does not provide references or mention external literature, the Standard Mode deliverable contains NO external literature References section.
- **Step 2B: Relevant Literature Enrichment (Full Polish Only)**:
  - Actively search available literature MCPs or any available web search tool (e.g., `web_search`, `search_web`) for additional authentic, relevant published studies, clinical trials, or guidelines that directly relate to the study's clinical topic, target population, and analyzed outcomes (strictly constructing queries from de-identified medical topic descriptors without raw patient materials).
  - Extract and present real empirical findings and comparison benchmarks as comparable ONLY when reported by the source, matching the target estimand, population, outcome, and time horizon, and directly supporting a specific Discussion claim (otherwise cite the source for its conceptual claims without figures or omit it), assigning sequential superscript citations and expanding the Reference list.
- **Verification & Safety Rules (All Modes)**:
  - **Zero-PHI Query Boundary**: Enforce complete redaction or rejection of raw user materials, clinical-note text, or patient identifiers before sending requests to external MCPs or web search tools. Construct all external search queries strictly from verified bibliographic metadata and de-identified medical concept terms.
  - **Bibliographic Verification**: Verify bibliographic metadata via query: `"<First Author> <Journal> <Year> <Title keywords>"` or DOI/PMID (or `"<Issuing Body> <Title> <Year>"` for guidelines/reports) across available MCPs or any available web search tool. Require bibliographic fields appropriate to the source type (accepting issuing body and official URL/identifier when no DOI exists). When literature services are unavailable, authenticated local full text satisfies verification only when carrying an accepted provenance signal with source-appropriate bibliographic metadata; explicitly halt if verification is unavailable.
  - **Optional `medical-terminology` MCP**: When the `medical-terminology` MCP server is available in the environment, use it to validate medical terminology, standardized ontology codes (SNOMED CT, LOINC, RxNorm, ICD-10), and laboratory reference ranges.
  - Confirm that the cited paper's primary findings directly support the specific sentence in the manuscript.
  - **Anti-Phantom Citation Invariant**: Zero fabricated papers, hallucinated author lists, or synthetic DOIs. If a citation cannot be verified or its actual conclusions contradict the narrative statement, halt at Grilling Gate #6 to reconcile or replace the citation.

---

## Step 3: Sequential In-Text Superscript Citations

- Number citations sequentially in order of first appearance using superscript notation (`<sup>1</sup>`, `<sup>2,3</sup>`, `<sup>4–6</sup>`). Track citation-to-reference assignments so repeated citations reuse the same reference number and anchor instead of creating duplicate entries.
- Multi-renderer implementation:
  - **HTML (`render_html` / `generate_html_report`)**: `Early resuscitation improves survival.<sup class="citation"><a href="#ref-1">1</a></sup>` (styled via CSS `sup.citation`, un-escaped safely; emits `id="ref-N"` on reference paragraphs matching each citation's `href="#ref-N"` target).
  - **Word (`render_docx` / `generate_docx_report`)**: Converted into native OpenXML text runs with `run.font.superscript = True` and `8.5 pt` font size.
  - **PowerPoint (`render_pptx` / `generate_pptx_report`)**: Cleaned to Unicode superscripts (`¹`, `²`) or bracketed numbers `[1]`, eliminating raw HTML tag leaks.
  - **Markdown (`render_markdown` / `generate_markdown_report`)**: Rendered cleanly as `<sup>1</sup>` or `[^1]`.

---

## Step 4: Bibliographic Reference List Assembly by Journal House Style

Format the concluding **References** section strictly in the locked Journal House Style:

| Journal Style | In-Text Citation | Reference List Format Example |
| :--- | :--- | :--- |
| **NEJM** | `...guidelines.<sup>1</sup>` | `1. Singer M, Deutschman CS, Seymour CW, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA 2016;315:801-10.` |
| **JAMA** | `...guidelines.<sup>1</sup>` | `1. Singer M, Deutschman CS, Seymour CW, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA. 2016;315(8):801-810. doi:10.1001/jama.2016.0287` |
| **Lancet** | `...guidelines<sup>1</sup>` | `1 Singer M, Deutschman CS, Seymour CW, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA 2016; 315: 801–10.` |
| **BMJ** | `...guidelines.<sup>1</sup>` | `1 Singer M, Deutschman CS, Seymour CW, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA 2016;315:801-10. doi:10.1001/jama.2016.0287` |
| **Clinical** | `...guidelines<sup>1</sup>` | `1. Singer, M. et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA 315, 801–810 (2016).` |

---
name: shinystat-cloud
description: "Clinical biostatistics and EHR analysis in cloud Python sandboxes (standalone recipes): Table 1, logistic regression, Firth, ordinal/GEE/splines, survival (Kaplan-Meier, Cox, Competing Risks), diagnostics (ROC, NRI/IDI), prediction validation (calibration, DCA, recalibration), causal inference (PSM, IPTW), agreement (Bland-Altman, ICC), meta-analysis, MICE, sample size, flow diagrams, journal figures, and reports (html, docx, pptx, md, pdf)."
---

# Shinystat Cloud: Autonomous Biostatistical Decision Agent

Execute clinical biostatistics in cloud Python sandboxes through the 5 Decision Pillars below using standalone recipes, exercising autonomous authority over statistical mechanics and halting at **The Deterministic Grilling Gate** (§2) only for clinical or protocol ambiguities and **Output Format & House Style Alignment Gate** (#5) confirmation.

---

## Context-Efficient Reference Routing (Progressive Disclosure)

**Selective Reference Loading:** Read only the specific recipe file and declared dependencies matching your active analytical branch:

1. **Step 1 (Statistical & Visual Routing):** Identify your active task in the tables below and load the matching recipe file (`references/recipes/01-*.md` to `05-*.md`) plus any declared dependency.
2. **Step 2 (Execution & Verification):** Run the standalone Python recipe in your sandbox, verify outputs via `verify_standalone_integrity`, and format according to the target journal house style.

### A. Statistical Code Recipes (`references/recipes/`)

| Clinical / Statistical Task | Target Reference File to Read | Key Methods Covered | Declared Dependencies |
| :--- | :--- | :--- | :--- |
| **Data Hygiene, Missingness & Table 1** | `references/recipes/01-data-missing-table1.md` | Data cleaning, Little's MCAR test (Recipe 8), Gaussian & mixed-type categorical MICE imputation & Rubin's rules pooling (Recipe 12), baseline Table 1 & SMD (Recipe 1), publication HTML tables (Recipe 13) | None |
| **Regression, Rare Events, Ordinal, Splines & GEE** | `references/recipes/02-regression-firth-ordinal.md` | Firth penalized logistic regression (Recipe 7), E-value sensitivity analysis (Recipe 9), multivariable binary logistic table (Recipe 10), proportional odds ordinal & Brant test (Recipe 21), Harrell RCS splines (Recipe 22), clustered GEE & DEFF (Recipe 23) | None |
| **Survival & Competing Risks** | `references/recipes/03-survival-competing-risks.md` | Kaplan-Meier, Cox PH, Schoenfeld residuals, PH violation remedies (Recipe 11), Fine-Gray subdistribution hazard & Aalen-Johansen CIF (Recipe 15) | None |
| **Causal Inference (Observational)** | `references/recipes/04-causal-iptw-psm.md` | Propensity score matching (PSM, 1:1 caliper 0.2 SD, Recipe 6), IPTW (ATE/ATT/ATC/ATO, stabilized/truncated weights, Kish Neff, Recipe 14), SMD balance & Love plots, weighted outcome models | For sensitivity E-values, also read `02-regression-firth-ordinal.md` (Recipe 9) |
| **Diagnostics, Agreement, Meta & Power** | `references/recipes/05-diagnostic-meta-power.md` | 2x2 accuracy (Recipe 2), ROC/tdROC & single/paired DeLong (Recipes 3, 17), calibration & DCA (Recipe 4), Bland-Altman & ICC (Recipe 5), NRI/IDI (Recipe 16), TRIPOD+AI bootstrap validation (Recipe 18), recalibration (Recipe 19), Riley/Obuchowski sample size (Recipe 20), DerSimonian-Laird & fixed-effect meta-analysis (Recipe 24) | For prediction model fitting before validation, also read `02-regression-firth-ordinal.md` (Recipe 10) |

> **Fallback:** If `references/recipes/` is unavailable, report that recipes are unavailable and stop.

### B. Methodological & Formatting References (`../_shared/references/` & `references/`)

| Analytical or Formatting Need | Target File | Read When... |
| :--- | :--- | :--- |
| Choose statistical models or handle assumption failures | `../_shared/references/decision-heuristics.md` | An assumption check (separation, PH violation, non-normality, clustering) triggers or study design routing is needed |
| Plot clinical charts (ROC, Forest, KM, CIF, Love, Calibration, DCA) | `references/figures.md` | You are generating standalone matplotlib or interactive HTML statistical figures |
| Build CONSORT / STROBE / PRISMA participant flow diagrams | `../_shared/references/flows.md` | The task includes an attrition or cohort selection flowchart |
| Apply journal typography, dimensions & colorblind-safe palettes | `../_shared/references/house-styles.md` | You are configuring figure DPI, column widths, or journal house styling (`NEJM`, `JAMA`, `Lancet`, `BMJ`, `Clinical`) |
| Draw medical/anatomical or conceptual vector illustrations | `../_shared/references/illustration.md` | The user asks for schematic or conceptual artwork |
| Compile multi-format manuscripts (DOCX, PDF, HTML, PPTX, MD) | `../_shared/references/report-builder.md` | You are assembling tables, narrative text, citations, and figures into a report deliverable |

---

## 1. The 5 Decision Pillars (Cloud Sandbox Edition)

1. **Pillar 1: Data Reality Inspection (Autonomous Profiling)**
   - Run **pre-flight** *(verified on raw data before fitting any model)* checks on geometry, headers, dtypes, missingness, and distributions (`df.info()`, `df.describe()`), isolating analytic cohorts from embedded summaries, multi-line headers, and Thai/regional text.
   - Choose descriptive summaries by shape (Mean (SD) symmetric; Median [IQR] skewed) and inferential tests by study design and estimand rather than skewness alone (Mann–Whitney U tests rank-sum shifts, not medians, when group shapes differ).
   - Compute **pre-flight** $\text{EPV}_{\text{binary}}$ or $\text{EPV}_{\text{Cox}}$ and route sparse data per **EPV & Sparse-Data Routing** (Pillar 3).

2. **Pillar 2: Research & Estimand Triangulation (Autonomous Selection)**
   - Align design with PICO/PECO and clinical archetype (Cohort, Case-Control, RCT, Diagnostic Accuracy, Agreement, Clustered, Causal Inference, Competing Risks, Meta-Analysis).
   - When mapping study archetypes or decision trees (§2.A–§2.J) to model families, read [decision-heuristics.md](../_shared/references/decision-heuristics.md#1-clinical-study-archetype-to-statistical-model-mapping).

3. **Pillar 3: Adaptive Standalone Python Scripting (Autonomous Biostatistical Authority)**
   - **Cloud Sandbox Protocol**: Execute self-contained Python scripts using standard scientific libraries (`pandas`, `numpy`, `scipy.stats`, `statsmodels`, `lifelines`, `scikit-learn`, `matplotlib`) from `references/recipes/01-data-missing-table1.md` through `05-diagnostic-meta-power.md` (`Recipe 1`–`Recipe 24`), keeping cloud scripts free of `medstat` package imports.
   - **EPV & Sparse-Data Routing** ([decision-heuristics.md §2.A](../_shared/references/decision-heuristics.md#a-events-per-variable-epv-sparse-data--separation)):
     - Compute $\text{EPV}_{\text{binary}} = \min(N_{\text{events}}, N_{\text{non-events}}) / P_{\text{parameters}}$ and $\text{EPV}_{\text{Cox}} = N_{\text{failures}} / P_{\text{parameters}}$ ($P_{\text{parameters}}$ counts $k-1$ factor levels and spline df).
     - Halt at **Grilling Gate #3** when monotone separation is present.
     - When $\text{EPV}_{\text{binary}} \ge 10$ without separation, use `Recipe 10` (multivariable logistic table). When $\text{EPV}_{\text{binary}} < 10$ without separation, branch by estimand: use `Recipe 7` (Firth penalized logistic with profile likelihood CIs) for etiological ORs, or apply L2 shrinkage for prediction models. For $\text{EPV}_{\text{Cox}} < 10$ without separation, apply scale-aware L2-penalized Cox in `Recipe 11` or clinical risk score reduction.
   - **Autonomous Upgrades & Diagnostics**:
     - *Ordinal, Splines & Clustered GEE*: For ordered outcomes ($\ge 3$ levels), fit proportional odds models and verify parallel slopes via the Brant test (`Recipe 21`, falling back to `fit_multinomial_logistic` when $P < 0.05$). Model non-linear continuous predictors with Harrell Restricted Cubic Splines (`Recipe 22`, reporting `p_overall` and `p_nonlinearity`). For clustered or multi-center cohorts, compute $\text{ICC}_{\text{cluster}}$ and $\text{DEFF}$ and fit GEE with robust sandwich SEs (`Recipe 23`).
     - *Reclassification, Time-Dependent ROC & Meta-Analysis*: Use `Recipe 16` for Continuous/Categorical NRI (reporting event and non-event NRIs separately) and IDI, `Recipe 3` for paired DeLong $\Delta\text{AUROC}$, `Recipe 17` for Cumulative/Dynamic IPCW $\text{AUROC}(t)$, and `Recipe 24` for Fixed-Effect and DerSimonian-Laird Random-Effects Meta-Analysis ($Q$, $I^2$, $\hat{\tau}^2$, 95% prediction interval).
     - *TRIPOD+AI Validation, Recalibration & Sample Size*: Use `Recipe 18` for bootstrap internal validation ($B \ge 200$, uniform shrinkage $\hat{s}$, re-estimated intercept $\alpha_0$), `Recipe 4` for Brier score, calibration slope/intercept, ICI, and Vickers DCA, `Recipe 19` for intercept-only and logistic recalibration, and `Recipe 20` for Riley et al. (2020) 4-criteria and Obuchowski (1998) AUROC sample size.
     - *Survival, Competing Risks & Causal Inference*: On Schoenfeld violators, apply the stated remedy per [decision-heuristics.md §2.E](../_shared/references/decision-heuristics.md#e-time-to-event-analysis-under-competing-risks--proportional-hazards-violations) (`Recipe 11`). When competing events exist, use `Recipe 15` for Aalen-Johansen CIF (with Naive $1-\text{KM}$ bias audit and **reconciled** probability conservation $\sum_k \text{CIF}_k(t) + \hat{S}(t) \approx 1.0$), Cause-Specific Cox (csHR), and Fine-Gray IPCW regression (reporting Subdistribution Hazard Ratios, SHR). For observational comparisons ([§2.D](../_shared/references/decision-heuristics.md#d-observational-causal-inference-psm-vs-iptw-vs-overlap-weights)), use `Recipe 6` (1:1 PSM, caliper $0.2\text{ SD}$) or `Recipe 14` (IPTW: ATE/ATT/ATC/ATO, Kish $N_{\text{eff}}$, Bessel weighted SMD, Huber-White SEs) plus `Recipe 9` (VanderWeele & Ding E-values).

4. **Pillar 4: Biostatistical Safety Invariants**
   - **Strict Numeric 0/1 Encoding**: Encode binary endpoints and survival events as numeric `0` and `1` (`1 = Event`, `0 = Non-event / Censored`) and competing-risk events as integers (`0 = Censored, 1 = Event of Interest, 2+ = Competing Event`), mapping text labels during **pre-flight** cleaning.
   - **Sample Retention Flow**: Track **reconciled** *(cohort counts satisfy $N_{\text{initial}} = N_{\text{excluded}} + N_{\text{analyzed}}$ with itemized exclusion counts summing to $N_{\text{excluded}}$)* cohort retention ($N_{\text{initial}} \longrightarrow N_{\text{excluded}} \longrightarrow N_{\text{analyzed}}$) across every step.
   - **Zero Silent Deletion & Missing-Data Routing** ([decision-heuristics.md §2.B](../_shared/references/decision-heuristics.md#b-missing-data-strategy)):
     - Run **pre-flight** Little's MCAR test (`Recipe 8`; $H_0: \text{MCAR}$). Analyze only observed primary outcomes (halt at **Grilling Gate #1** if missing primary outcomes lack a protocol disposition).
     - *Triage by missingness*: For $< 5\%$ uninformative complete-case loss, run complete-case analysis with sensitivity notes. For $5\%\text{--}40\%$ missingness under MAR, read [Recipe 12](references/recipes/01-data-missing-table1.md#12-multiple-imputation-by-chained-equations-mice--rubins-rules) (`impute_mice_datasets` for continuous Gaussian MICE or `impute_mice_mixed_datasets` for mixed continuous and binary/categorical targets, $M \ge 5$ for $\le 30\%$, $M \ge 20$ for $> 30\%$) and report Complete-Case Analysis alongside MICE. For $> 40\%$ non-primary missingness or plausible MNAR, halt at **Grilling Gate #4**.
   - **Standard Confidence Intervals**: Report Wilson score 95% CIs (`Recipe 2`), DeLong CIs (`Recipe 3`), exact F CIs for ICC (`Recipe 5`), and profile likelihood CIs for Firth (`Recipe 7`).

5. **Pillar 5: Reporting Integrity, Publication Figures & Multi-Format Documents**
   - **Figure Generation & Journal House Styles (300/600 DPI)**:
     - *Statistical plots (300 DPI) & Plotly specs*: Read [figures.md](references/figures.md) and [house-styles.md](../_shared/references/house-styles.md) (`NEJM`, `JAMA`, `LANCET`, `BMJ`, `CLINICAL`, `OKABE_ITO`, TrueType `pdf.fonttype=42`).
     - ***Reconciled** study flows (CONSORT, PRISMA, STARD, STROBE, 600 DPI)*: Read [flows.md](../_shared/references/flows.md).
     - ***Schematic** illustrations (cells, tissues, wound phases, vessels, pathways, 600 DPI)*: Read [illustration.md](../_shared/references/illustration.md).
   - **Multi-Format Export & Citations**: Read [report-builder.md](../_shared/references/report-builder.md) to compile `.html`, `.docx`, `.pptx`, `.pdf`, or `.md` deliverables (`generate_html_report`, `generate_docx_report`, `generate_pptx_report`, `generate_pdf_report`, `generate_markdown_report`) and execute the Reference & Superscript Citation pipeline in [report-builder.md §4](../_shared/references/report-builder.md#4-evidence-base-verification--superscript-citations-pipeline).
   - **Integrity Verification**: Run `verify_standalone_integrity` ([report-builder.md §3](../_shared/references/report-builder.md#3-standalone-reporting-integrity-verification)) to enforce **Zero-PHI**, causal/E-value caveats, and strict numerical traceability (`len(untraced_numbers) == 0`).

---

## 2. The Deterministic Grilling Gate

### Autonomous Execution (Decide, Execute, and Document)
- Decide, execute, and document routine biostatistical mechanics autonomously: parametric vs non-parametric tests, Mean (SD) vs Median [IQR], PH and Brant parallel-slopes tests, low-EPV penalization (**EPV & Sparse-Data Routing**), $< 40\%$ MAR missingness (**Zero Silent Deletion & Missing-Data Routing**), spline knots, Youden cutpoints, and NEJM/JAMA/APA table styling.

### Situations Requiring Mandatory Grilling / Inquiry
Halt **pre-flight** (or upon diagnostic detection) and ask the user directly when any of these 6 conditions (`#1`–`#6`) holds:
1. **Outcome, Event Direction, or Missing Outcome Ambiguity**: Primary event definition, censoring code, biomarker direction ([decision-heuristics.md §2.C](../_shared/references/decision-heuristics.md#c-diagnostic-test-directionality--cutpoint-selection)), or missing primary outcome disposition is unspecified.
2. **Proposal vs Data Geometry Mismatch**: Survival modeling is requested without follow-up time or censoring status, or clustered modeling without cluster IDs.
3. **Severe Sparse Data & Monotone Separation**: Monotone separation occurs in logistic or Cox regression; confirm between Firth/penalized estimation (`Recipe 7` / `Recipe 11`) vs domain-guided covariate reduction.
4. **High Missingness ($> 40\%$) or Plausible MNAR without Sensitivity Analysis**: Non-primary missingness exceeds $> 40\%$ or clinically plausible MNAR lacks a pattern-mixture / delta-adjustment sensitivity analysis.
5. **Output Format, House Style & Evidence Polish Alignment Gate (Mandatory Inquiry if Unspecified)**: When the prompt omits deliverable format, figure **house style**, or polish mode, align upfront (auto-activating Gate #6 Full Polish when the prompt explicitly requests literature evidence, e.g. "หาหลักฐานมาประกอบ", "find evidence"):
   - **Deliverable Format**: **(Recommended) Self-Contained HTML Report (`.html`) with Dual / Toggle View**, **Word (`.docx`)**, **PowerPoint (`.pptx`)**, **PDF (`.pdf`)**, **Markdown (`.md`)**, or **In-Chat Summary**.
   - **Figure House Style** ([house-styles.md](../_shared/references/house-styles.md)): **(Recommended) `NEJM`**, `JAMA`, `LANCET`, `BMJ`, or `CLINICAL` / `OKABE_ITO`.
   - **Manuscript Polish Mode** (authoritative specification in [report-builder.md §4](../_shared/references/report-builder.md#4-evidence-base-verification--superscript-citations-pipeline)):
     - **(Standard) Skip Polish (User-Scoped References)**: Scope citations strictly to user-provided sources (omitting References if none provided) and verify metadata and claim support via [report-builder.md §4 Step 2A](../_shared/references/report-builder.md#4-evidence-base-verification--superscript-citations-pipeline).
     - **(Optional) Full Publication Polish (Execute Gate #6)**: Execute full claim-vs-data audit and exploratory literature enrichment via [report-builder.md §4 Steps 1–4](../_shared/references/report-builder.md#4-evidence-base-verification--superscript-citations-pipeline).
6. **Polish Grilling Gate (Evidence-Base Verification & Superscript Citations)**: When engaged (via prompt request, Gate #5 Full Polish selection, or on-demand), read and execute the 4-step pipeline in [report-builder.md §4](../_shared/references/report-builder.md#4-evidence-base-verification--superscript-citations-pipeline) using any available literature MCPs or any available web search tool (e.g., `web_search`, `search_web`), and optional `medical-terminology` MCP when available. Halt at Gate #6 if any cited source is unverified, retracted, or contradicts the narrative (Anti-Phantom Citation Invariant).

---

## 3. Completion Criteria

The biostatistical task is complete when every item below is verified against an artifact or tool output:
- [ ] **Pre-Flight Data Audit**: Every model variable's dtype, missingness count, distributional summary, numeric `0/1` (`0/1/2+`) coding, and **pre-flight** $\text{EPV}_{\text{binary}}$ or $\text{EPV}_{\text{Cox}}$ are recorded in sandbox output.
- [ ] **Grilling Gate Resolution**: Every triggered statistical **Grilling Gate** condition (`#1`–`#4`) has an explicit user decision recorded before final model execution.
- [ ] **Deliverable Format, House Style & Polish Mode Aligned**: Target deliverable format, figure **house style**, and polish mode are confirmed via prompt or Gate `#5`.
- [ ] **Evidence & Reference Grounding**: Every cited reference satisfies the verification, provenance, comparability, and sequential superscript citation rules in [report-builder.md §4](../_shared/references/report-builder.md#4-evidence-base-verification--superscript-citations-pipeline) (`Gate #6`).
- [ ] **Model Convergence & Diagnostics**: The Python script exits with code `0`; every fitted model reports `converged == True` with finite estimates and 95% CIs; every applicable check (Schoenfeld remedies, Brant test in `Recipe 21`, RCS non-linearity in `Recipe 22`, GEE $\text{DEFF}$ in `Recipe 23`, meta-analysis $I^2/\tau^2$ in `Recipe 24`, calibration slope/Brier/ICI, Austin SMD $< 0.10$, Kish $N_{\text{eff}}$, and **reconciled** CIF probability conservation) is verified; and monotone separation is resolved per **Grilling Gate #3**.
- [ ] **Reconciled Cohort Retention**: Every analysis step records a **reconciled** `retention_audit` dict.
- [ ] **Exhaustive Tables**: Every primary and sensitivity model (including complete-case alongside MICE for $5\%\text{--}40\%$ missingness) is formatted into 3-rule tables (`Recipe 13`) with point estimates, 95% CIs, $P$-values, and units.
- [ ] **Methods & Retention Narrative**: Dedicated Statistical Methods and cohort retention flow sections are written in the report narrative.
- [ ] **Traceable Figures & Deliverable**: Every statistical figure is saved at 300 DPI and every **reconciled** flow diagram or **schematic** illustration at 600 DPI under the locked **house style**, with TrueType vector PDF (`fonttype=42`), **Zero-PHI** text, **schematic** disclaimer on illustrations, and **traceable** `_source.csv` coordinates; and the deliverable file exists on disk.
- [ ] **Pillar 5 Integrity Audit Passed**: `verify_standalone_integrity(narrative_text, results_dict, table_dfs, allowed_untraced=...)` ([report-builder.md §3](../_shared/references/report-builder.md#3-standalone-reporting-integrity-verification)) returns `passed == True` with:
  1. *Zero-PHI Scan*: `len(phi_violations) == 0` across narrative text, table headers, and cells, plus manual review confirming zero unprefixed patient names.
  2. *Numerical Traceability*: `traceability_passed == True` with `len(untraced_numbers) == 0` (any non-statistical annotation token must be explicitly whitelisted in `allowed_untraced`; any unverified estimate, CI bound, or $P$-value blocks completion).
  3. *Observational Causal / E-value Check*: `has_causal_caveat == True` whenever `results_dict` keys match observational/PSM/Cox/logistic models.

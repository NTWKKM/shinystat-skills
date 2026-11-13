---
name: shinystat-cloud
description: "Cloud-sandbox clinical biostatistics via standalone recipes, zero medstat install. Use in hosted sandboxes (ChatGPT, Claude.ai, Gemini) or when medstat is unavailable: Table 1, logistic regression, Firth, ordinal/GEE/splines, survival (Kaplan-Meier, Cox, Competing Risks), diagnostics (ROC, NRI/IDI), prediction validation (calibration, DCA, recalibration), causal inference (PSM, IPTW), agreement (Bland-Altman, ICC), meta-analysis, MICE, sample size, flow diagrams, journal figures, and reports (html, docx, pptx, md, pdf)."
---

# Shinystat Cloud: Autonomous Biostatistical Decision Agent

Execute clinical biostatistics in cloud Python sandboxes through the **Run Sequence** (§0) and **The 5 Decision Pillars** (§1) using standalone recipes, halting at **The Deterministic Grilling Gate** (§2) only for clinical/protocol ambiguities and **Output Alignment Gate** (#5) confirmation.

---

## 0. Run Sequence

1. **Inspect**: Run **Pillar 1** **pre-flight** profile (`df.info()`, `df.describe()`, missingness, numeric `0/1` coding; `Recipe 8` when any model variable is missing). *Done when* every model variable's dtype, missingness count, and coding are printed in sandbox output.
2. **Ask**: Present one consolidated turn holding every triggered pre-flight gate (§2: `#1`–`#4`) plus every unspecified **Output Alignment Gate** axis (`#5a`–`#5d`) via `ask_question` when available. *Done when* the user has answered every triggered question.
3. **Plan**: Map study archetype to model family (**Pillar 2**, [decision-heuristics.md §1](references/decision-heuristics.md#1-clinical-study-archetype-to-statistical-model-mapping)), load matching recipe (`references/recipes/`), and compute **pre-flight** $\text{EPV}_{\text{binary}}$ or $\text{EPV}_{\text{Cox}}$ (**Pillar 3**; record "N/A" when the task has no logistic or Cox model, e.g. ordinal-only, agreement, diagnostic 2×2, or meta-analysis). *Done when* target `Recipe #`, missing-data route (**Pillar 4**), and EPV route are recorded.
4. **Execute**: Run **Pillar 3** standalone Python recipes (or dispatch subagents per [subagent-protocol.md](references/subagent-protocol.md) when `#5d` selected **Subagent Offload**). *Done when* every fitted model satisfies convergence (`converged == True`, or for Recipe 11 Cox models, fits without convergence warnings) and each applicable diagnostic passes or has its documented remedy applied and reported; halt at `#3` or `#4` (§2) if triggered mid-run.
5. **Render**: Generate **Pillar 5** 300/600 DPI figures in the `#5b` **house style** and compile the `#5a` deliverable via [report-builder.md](references/report-builder.md) (running [evidence-pipeline.md](references/evidence-pipeline.md) when citations or `#5c` Full Polish are active). *Done when* each primary estimate is plotted (figure bundle on disk), and deliverable exists on disk (when #5a requests a file deliverable; for In-Chat Summary, completion permitted without an on-disk report file).
6. **Audit**: Run `verify_standalone_integrity` ([report-builder.md §3](references/report-builder.md#3-standalone-reporting-integrity-verification)) on the exact narrative and tables delivered (In-Chat Summary reply) with every data file written as `exported_files` (a file deliverable's chat message is audited too), and check **§3 Completion Criteria**. *Done when* `passed == True` and every §3 box is verified against sandbox output.

---

## Context-Efficient Reference Routing (Progressive Disclosure)

Load only the recipe file and reference matching your active analytical branch:

### A. Statistical Code Recipes (`references/recipes/`): Recipe Index

| Recipe # | Recipe Title | Recipe File |
| :--- | :--- | :--- |
| **Recipe 1** | Table 1 Baseline Characteristics & SMD | `01-data-missing-table1.md` |
| **Recipe 2** | 2x2 Diagnostic Accuracy with Wilson Score & Log-Scale 95% CIs | `05-diagnostic-accuracy-agreement.md` |
| **Recipe 3** | Empirical ROC, Optimal Cutoff & DeLong 95% CIs | `05-diagnostic-accuracy-agreement.md` |
| **Recipe 4** | Model Calibration & Vickers Decision Curve Analysis (DCA) | `05-diagnostic-accuracy-agreement.md` |
| **Recipe 5** | Observer Agreement: Bland-Altman & ICC | `05-diagnostic-accuracy-agreement.md` |
| **Recipe 6** | Propensity Score Matching (PSM) & Balance | `04-causal-iptw-psm.md` |
| **Recipe 7** | Firth's Penalized Likelihood Logistic Regression | `02-regression-firth-ordinal.md` |
| **Recipe 8** | Little's MCAR Multivariate Missingness Test | `01-data-missing-table1.md` |
| **Recipe 9** | VanderWeele Sensitivity E-Value | `02-regression-firth-ordinal.md` |
| **Recipe 10** | Multivariable Logistic Regression Table | `02-regression-firth-ordinal.md` |
| **Recipe 11** | Survival Analysis: KM, Log-Rank & Cox PH | `03-survival-competing-risks.md` |
| **Recipe 12** | Multiple Imputation (MICE) & Rubin's Rules | `01-data-missing-table1.md` |
| **Recipe 13** | Publication HTML Table Renderer (NEJM / JAMA) | `01-data-missing-table1.md` |
| **Recipe 14** | IPTW Weighting & Weighted Balance | `04-causal-iptw-psm.md` |
| **Recipe 15** | Competing Risks (Aalen-Johansen CIF & Fine-Gray) | `03-survival-competing-risks.md` |
| **Recipe 16** | Diagnostic Risk Reclassification (NRI & IDI) | `06-reclassification-tdroc.md` |
| **Recipe 17** | Time-Dependent ROC: IPCW Estimator (AUROC(t)) | `06-reclassification-tdroc.md` |
| **Recipe 18** | TRIPOD+AI Bootstrap Internal Validation & Uniform Shrinkage | `07-validation-samplesize-meta.md` |
| **Recipe 19** | Model Recalibration & Updating (Logistic) | `07-validation-samplesize-meta.md` |
| **Recipe 20** | Prediction Sample Size (Riley 2020) & Diagnostic AUROC (Obuchowski 1998) | `07-validation-samplesize-meta.md` |
| **Recipe 21** | Proportional Odds Ordinal Logistic & Brant | `02-regression-firth-ordinal.md` |
| **Recipe 22** | Restricted Cubic Splines (Harrell RCS) | `02-regression-firth-ordinal.md` |
| **Recipe 23** | Clustered GEE with Robust Sandwich SEs, ICC & Design Effect ($\text{DEFF}$) | `02-regression-firth-ordinal.md` |
| **Recipe 24** | Fixed & Random-Effects Meta-Analysis (PRISMA) | `07-validation-samplesize-meta.md` |

Each file is self-contained except: E-values after `Recipe 6` / `Recipe 14` need `Recipe 9`; validation (`Recipe 18`–`19`) fits its model with `Recipe 10`.

> **Fallback:** If `references/recipes/` is unavailable, report that recipes are unavailable and stop.

### B. Methodological & Formatting References (`references/`)

| Need | Target File |
| :--- | :--- |
| Study design routing & assumption trees (§2.A–§2.K) | `references/decision-heuristics.md` |
| Statistical charts (ROC, Forest, KM, CIF, Love, Calibration, DCA) | `references/figures.md` |
| CONSORT / STROBE / PRISMA flow diagrams | `references/flows.md` |
| Journal typography, column widths & CVD palettes | `references/house-styles.md` |
| Schematic anatomical / conceptual vector illustrations | `references/illustration.md` |
| Multi-format report generators & integrity verifier | `references/report-builder.md` |
| Interactive Plotly specs for the HTML report (`build_plotly_*_spec`) | `references/report-plotly-specs.md` |
| Literature citation verification & superscripts (Gate #6) | `references/evidence-pipeline.md` |
| Subagent execution & citation offload (Gate #5d) | `references/subagent-protocol.md` |

---

## 1. The 5 Decision Pillars (Cloud Sandbox Edition)

1. **Pillar 1: Data Reality Inspection (Autonomous Profiling)**
   - Run **pre-flight** *(verified on raw data before fitting any model)* checks on geometry, headers, dtypes, missingness, and distributions (`df.info()`, `df.describe()`), isolating analytic cohorts from embedded summaries, multi-line headers, and Thai/regional text.
   - Choose descriptive summaries by shape (Mean (SD) symmetric; Median [IQR] skewed) and inferential tests by study design and estimand rather than skewness alone (Mann–Whitney U tests rank-sum shifts, not medians, when group shapes differ).
   - Compute **pre-flight** $\text{EPV}_{\text{binary}}$ or $\text{EPV}_{\text{Cox}}$ once outcome and event coding are unambiguous (or after resolving **Gate #1 / #2**).

2. **Pillar 2: Research & Estimand Triangulation (Autonomous Selection)**
   - Align PICO/PECO and clinical archetype (Cohort, Case-Control, RCT, Diagnostic, Agreement, Clustered, Causal, Competing Risks, Meta-Analysis) via [decision-heuristics.md §1](references/decision-heuristics.md#1-clinical-study-archetype-to-statistical-model-mapping).
   - **Effect Scale**: One headline measure per estimand (requested, else OR logistic / HR Cox); one stated unit per continuous predictor (e.g. per 10 years) across tables, text and figures.

3. **Pillar 3: Adaptive Standalone Python Scripting (Autonomous Biostatistical Authority)**
   - **Cloud Sandbox Protocol**: Execute self-contained Python scripts using standard scientific libraries (`pandas`, `numpy`, `scipy.stats`, `statsmodels`, `lifelines`, `scikit-learn`, `matplotlib`) from `references/recipes/` (`Recipe 1`–`Recipe 24`), keeping cloud scripts free of `medstat` package imports.
   - **Subagent Context-Offload**: When **Subagent Offload** is selected at **Gate #5d**, execute [subagent-protocol.md](references/subagent-protocol.md).
   - **EPV & Sparse-Data Routing** ([decision-heuristics.md §2.A](references/decision-heuristics.md#a-events-per-variable-epv-sparse-data--separation)):
     - $\text{EPV}_{\text{binary}} = \min(N_{\text{events}}, N_{\text{non-events}}) / P_{\text{parameters}}$; $\text{EPV}_{\text{Cox}} = N_{\text{failures}} / P_{\text{parameters}}$ ($k-1$ factor levels and spline df; no intercept).
     - Monotone separation $\longrightarrow$ **Grilling Gate #3** (§2). For $\text{EPV}_{\text{binary}} \ge 10$ without separation: `Recipe 10`. For $\text{EPV} < 10$ without separation: `Recipe 7` (Firth logistic) for etiological ORs, L2 shrinkage for prediction, or scale-aware L2 Cox (`Recipe 11`) / risk score reduction for survival.
   - **Autonomous Upgrades & Diagnostics** (load only the active branch in `decision-heuristics.md`):
     - *Observational causal (`Recipe 6` PSM, `Recipe 14` IPTW, `Recipe 9` E-values)* $\longrightarrow$ [§2.D](references/decision-heuristics.md#d-observational-causal-inference-psm-vs-iptw-vs-overlap-weights).
     - *Survival PH remedies (`Recipe 11`) & competing risks (`Recipe 15` CIF, csHR, Fine-Gray SHR)* $\longrightarrow$ [§2.E](references/decision-heuristics.md#e-time-to-event-analysis-under-competing-risks--proportional-hazards-violations).
     - *Reclassification (`Recipe 16` NRI & IDI, `Recipe 3` paired DeLong) & IPCW tdROC (`Recipe 17`)* $\longrightarrow$ [§2.F](references/decision-heuristics.md#f-diagnostic-risk-reclassification--incremental-value), [§2.G](references/decision-heuristics.md#g-time-dependent-roc-in-longitudinal-follow-up).
     - *TRIPOD+AI bootstrap (`Recipe 18`), calibration/DCA (`Recipe 4`), recalibration (`Recipe 19`) & sample size (`Recipe 20`)* $\longrightarrow$ [§2.H](references/decision-heuristics.md#h-tripodai-prediction-model-internal-validation--uniform-shrinkage), [§2.I](references/decision-heuristics.md#i-minimum-sample-size-for-clinical-prediction-models-riley-et-al-2020).
     - *Ordinal & Brant (`Recipe 21`), Harrell RCS (`Recipe 22`), clustered GEE $\text{DEFF}$ (`Recipe 23`) & meta-analysis (`Recipe 24`)* $\longrightarrow$ [§2.K](references/decision-heuristics.md#k-ordinal-regression-restricted-cubic-splines-clustered-gee--meta-analysis).

4. **Pillar 4: Biostatistical Safety Invariants**
   - **Numeric 0/1 Encoding**: Map binary/survival events to `0/1` (`1 = Event`, `0 = Non-event / Censored`) and competing risks to `0 = Censored, 1 = Event of Interest, 2+ = Competing Event` during **pre-flight** cleaning.
   - **Sample Retention Flow**: Track **reconciled** *($N_{\text{initial}} = N_{\text{excluded}} + N_{\text{analyzed}}$)* cohort retention across every step.
   - **Identifier-Free Exports**: Exported data files carry no patient identifier column.
   - **Zero Silent Deletion & Missing-Data Routing** ([decision-heuristics.md §2.B](references/decision-heuristics.md#b-missing-data-strategy)):
     - Run **pre-flight** Little's MCAR test (`Recipe 8`; $H_0: \text{MCAR}$). Missing primary outcomes $\longrightarrow$ **Grilling Gate #1**.
     - *Triage*: $< 5\%$ complete-case with sensitivity notes; $5\%\text{--}40\%$ MAR $\longrightarrow$ [Recipe 12](references/recipes/01-data-missing-table1.md#12-multiple-imputation-by-chained-equations-mice--rubins-rules) ($M \ge 5$ for $\le 30\%$, $M \ge 20$ for $> 30\%$) primary, Complete-Case sensitivity; $> 40\%$ exposure or covariate missingness or plausible MNAR $\longrightarrow$ **Grilling Gate #4** (§2).
   - **Standard CIs**: Wilson score 95% CIs (`Recipe 2`), DeLong CIs (`Recipe 3`), exact F CIs (`Recipe 5`), profile likelihood CIs (`Recipe 7`).

5. **Pillar 5: Reporting Integrity, Publication Figures & Multi-Format Documents**
   - **Figures & House Styles (300/600 DPI)**: Statistical plots (300 DPI) in [figures.md](references/figures.md); **house styles** (`NEJM`, `JAMA`, `LANCET`, `BMJ`, `CLINICAL`, `OKABE_ITO`) and TrueType PDF (`fonttype=42`) in [house-styles.md](references/house-styles.md); **reconciled** study flows (600 DPI) in [flows.md](references/flows.md); **schematic** illustrations (600 DPI) in [illustration.md](references/illustration.md).
   - **Multi-Format Export & Citations**: Read [report-builder.md](references/report-builder.md) to compile `.html`, `.docx`, `.pptx`, `.pdf`, or `.md` deliverables (`generate_html_report`, `generate_docx_report`, `generate_pptx_report`, `generate_pdf_report`, `generate_markdown_report`) and run [evidence-pipeline.md](references/evidence-pipeline.md) for citations.
   - **Integrity Verification**: Run `verify_standalone_integrity` ([report-builder.md §3](references/report-builder.md#3-standalone-reporting-integrity-verification)) for **Zero-PHI**, causal/E-value caveats, and traceability (`len(untraced_numbers) == 0`).

---

## 2. The Deterministic Grilling Gate

### Autonomous Execution (Decide, Execute, and Document)
- Decide, execute, and document routine biostatistical mechanics autonomously: parametric vs non-parametric tests, Mean (SD) vs Median [IQR], PH and Brant tests, low-EPV penalization (**EPV & Sparse-Data Routing**), $< 40\%$ MAR missingness (**Zero Silent Deletion & Missing-Data Routing**), spline knots, Youden cutpoints, and NEJM/JAMA/APA table styling.

### Situations Requiring Mandatory Grilling / Inquiry
After **Pillar 1** pre-flight inspection, ask all triggered pre-flight questions (`#1`–`#4`; `#4` when pre-flight confirms $> 40\%$ non-primary missingness or clinically plausible MNAR without sensitivity analysis, and every unspecified `#5` axis) in a **single consolidated turn**, written as one question per triggered gate and per axis (computing EPV after `#1`/`#2` resolution when outcome or event geometry was unspecified). When your tools include `ask_question`, send the consolidated turn through it—one question per gate and per Gate #5 axis, recommended option first. When the question tool caps a call (e.g. `AskUserQuestion`: 4 questions × 4 options), split into consecutive calls and name options beyond the cap in the question text. If a gate (`#2`–`#4` first found mid-run, or `#6`) triggers during modeling or citation verification, halt directly in **Direct in Main Agent** mode, or via [subagent-protocol.md](references/subagent-protocol.md) when **Subagent Offload** is active:
1. **Outcome, Event Direction, or Missing Outcome Ambiguity**: Primary event definition, censoring code, pooled effect scale (log ratio vs difference), biomarker direction not fixed by [decision-heuristics.md §2.C](references/decision-heuristics.md#c-diagnostic-test-directionality--cutpoint-selection), or missing primary outcome disposition is unspecified.
2. **Proposal vs Data Geometry Mismatch**: Survival modeling without follow-up time/censoring status, clustered modeling without cluster IDs, or a flow diagram arm without observed receipt/analysis counts.
3. **Severe Sparse Data & Monotone Separation**: Monotone separation in logistic or Cox regression; confirm between Firth/penalized estimation (`Recipe 7` / `Recipe 11`) vs covariate reduction.
4. **High Missingness ($> 40\%$) or Plausible MNAR without Sensitivity Analysis**: Missingness $> 40\%$ in the exposure or any covariate (Little's test cannot lift it) or clinically plausible MNAR lacks a pattern-mixture / delta-adjustment sensitivity analysis.
5. **Output Alignment Gate** (Mandatory Inquiry per Unspecified Axis): Gate #5 has four independent **axes**. Ask one separate question per axis the prompt leaves unspecified, listing every option (auto-activating Gate #6 Full Polish when the prompt explicitly requests literature evidence, e.g. "หาหลักฐานมาประกอบ", "find evidence"):
   - **5a. Deliverable Format**: **(Recommended) Self-Contained HTML Report (`.html`) with Dual / Toggle View**, **Word (`.docx`)**, **PowerPoint (`.pptx`)**, **PDF (`.pdf`)**, **Markdown (`.md`)**, or **In-Chat Summary**.
   - **5b. Figure House Style** ([house-styles.md](references/house-styles.md)): **(Recommended) `NEJM`**, `JAMA`, `LANCET`, `BMJ`, `CLINICAL`, or `OKABE_ITO` (CVD-safe); offer each as its own option.
   - **5c. Manuscript Polish Mode** ([evidence-pipeline.md](references/evidence-pipeline.md)): **(Standard) Skip Polish (User-Scoped References)** (Step 2A; omit References if none provided) vs **(Optional) Full Publication Polish (Execute Gate #6)** (Steps 1–4).
   - **5d. Execution Context Mode** *(ask when unspecified, tools include a subagent tool, e.g. `Agent` or `Task`, and the run fits ≥2 distinct estimands (a primary model and its sensitivity re-fits count as one) or uses Full Polish, even when every other axis is specified; otherwise record **Direct in Main Agent**)*: **(Recommended for multi-model/polish runs) Subagent Offload** ([subagent-protocol.md](references/subagent-protocol.md)) vs. **Direct in Main Agent**.
6. **Polish Grilling Gate (Evidence-Base Verification & Superscript Citations)**: When engaged, execute the 4-step pipeline in [evidence-pipeline.md](references/evidence-pipeline.md) via literature MCPs, web search (`web_search`, `search_web`), and optional `medical-terminology` MCP. Halt at Gate #6 if any cited source is unverified, retracted, or contradicts the narrative (Anti-Phantom Citation Invariant).

---

## 3. Completion Criteria

The biostatistical task is complete when every item below is verified against an artifact or tool output:
- [ ] **Pre-Flight Data Audit**: Every model variable's dtype, missingness, summary, numeric `0/1` (`0/1/2+`) coding, and **pre-flight** $\text{EPV}_{\text{binary}}$ or $\text{EPV}_{\text{Cox}}$ (or its "N/A") are in sandbox output.
- [ ] **Grilling Gate Resolution**: Every triggered **Grilling Gate** condition (`#1`–`#4`) has an explicit user decision recorded before final model execution.
- [ ] **Deliverable Format, House Style & Polish Mode Aligned**: Target format, figure **house style**, polish mode, and execution context mode (when subagent tools exist) are each confirmed via prompt or their own Gate `#5` axis question; when **Subagent Offload** is used, independent Main-Agent on-disk verification and external row-level / `.scratch_checkpoint/` cleanup per [subagent-protocol.md](references/subagent-protocol.md) are completed.
- [ ] **Evidence & Reference Grounding**: Every cited reference satisfies the verification, provenance, comparability, and superscript citation rules in [evidence-pipeline.md](references/evidence-pipeline.md) (`Gate #6`).
- [ ] **Model Convergence & Diagnostics**: Script exits `0`; every model satisfies convergence (`converged == True` for other models, or successful fitting without convergence warnings for Recipe 11 Cox models) with finite estimates/95% CIs; applicable diagnostics pass or carry their documented remedy; and separation is resolved per **Gate #3**.
- [ ] **Reconciled Cohort Retention**: Every analysis step records a **reconciled** `retention_audit` dict.
- [ ] **Exhaustive Tables**: Every model (MICE primary, complete-case sensitivity for $5\%\text{--}40\%$ missingness) is formatted into 3-rule tables (`Recipe 13`) with point estimates, 95% CIs, $P$-values, and units.
- [ ] **Methods & Retention Narrative**: Dedicated Statistical Methods and cohort retention flow sections are written in the report narrative.
- [ ] **Traceable Figures & Deliverable**: Every statistical figure (300 DPI) and **reconciled** flow / **schematic** illustration (600 DPI) is saved in the locked **house style** with TrueType PDF (`fonttype=42`), **Zero-PHI**, **schematic** disclaimer, and **traceable** `_source.csv`; the deliverable exists on disk when `#5a` requests a file-based deliverable (for In-Chat Summary, completion is permitted without creating or verifying a report file on disk).
- [ ] **Pillar 5 Integrity Audit Passed**: `verify_standalone_integrity(narrative_text, results_dict, table_dfs, allowed_untraced=..., study_design=study_design)` ([report-builder.md §3](references/report-builder.md#3-standalone-reporting-integrity-verification)) returns `passed == True`:
  1. *Zero-PHI*: `len(phi_violations) == 0` across narrative and tables, plus a read of narrative and table cells confirming every patient identifier is synthetic.
  2. *Numerical Traceability*: `traceability_passed == True` (`len(untraced_numbers) == 0`, whitelisted constants in `allowed_untraced`) plus independent table-to-results reconciliation against `results_dict`.
  3. *Causal & E-value Check*: `has_causal_caveat == True` for observational/causal analyses or when causal claims are made; a logistic or Cox key in results_dict alone does not require a caveat for a non-causal prediction report lacking causal claims.

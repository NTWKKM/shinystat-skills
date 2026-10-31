---
name: shinystat
description: "Clinical biostatistics and EHR analysis with medstat: Table 1, logistic/ordinal/GEE regression, Firth, survival (Kaplan-Meier, Cox, Competing Risks), diagnostics (ROC/DeLong, NRI/IDI), prediction validation (calibration, DCA, recalibration), causal inference (PSM, IPTW), agreement (Bland-Altman, ICC), meta-analysis, MICE, sample size, flow diagrams, vector illustrations, journal figures, and clinical reports (html, docx, pptx, md, pdf)."
---

# Shinystat: Autonomous Biostatistical Decision Agent

Execute clinical biostatistics through the 5 Decision Pillars below, exercising autonomous authority over statistical mechanics and halting at **The Deterministic Grilling Gate** (§2) only for clinical or protocol ambiguities and **Output Format & House Style Alignment Gate** (#5) confirmation.

---

## 1. The 5 Decision Pillars

1. **Pillar 1: Data Reality Inspection (Autonomous Profiling)**
   - Run **pre-flight** *(verified on raw data before fitting any model)* checks on data geometry, headers, dtypes, missingness, and distributions (`df.info()`, `df.describe()`), isolating analytic cohorts from embedded summaries, multi-line headers, and Thai/regional text.
   - Select descriptive summaries from distributional shape (Mean (SD) for symmetric vs Median [IQR] for skewed data); select inferential tests from the study design and target estimand rather than skewness alone (Mann–Whitney U tests rank-sum shifts rather than medians when group distribution shapes differ).
   - Compute **pre-flight** Events Per Variable ($\text{EPV}_{\text{binary}}$ or $\text{EPV}_{\text{Cox}}$) and route sparse data per **EPV & Sparse-Data Routing** (Pillar 3).

2. **Pillar 2: Research & Estimand Triangulation (Autonomous Selection)**
   - Align statistical design with PICO/PECO and clinical archetype (Cohort, Case-Control, RCT, Diagnostic Accuracy, Agreement, Clustered, Causal Inference, Competing Risks, Meta-Analysis).
   - When mapping study archetypes or sparse-data decision trees to model families, read [decision-heuristics.md](references/decision-heuristics.md#1-clinical-study-archetype-to-statistical-model-mapping).

3. **Pillar 3: Adaptive Python Scripting (Autonomous Biostatistical Authority)**
   - Write standalone Python scripts using the bundled `medstat` engine in `scripts/medstat/` (CLI: `python <skill-dir>/scripts/medstat_cli.py`) or standard scientific libraries (`pandas`, `scipy.stats`, `statsmodels`, `lifelines`, `scikit-learn`).
   - Run every script with the pinned stack: `uv run --with-requirements <skill-dir>/requirements.txt python <script.py>` (append `--with <pkg>` only for packages outside that file).
   - **Pre-Flight Spec Validation & Self-Healing Ergonomics**: When executing declarative Statistical Analysis Plans (`--spec <plan.yaml>`), validate the YAML specification before running analysis:
     `python <skill-dir>/scripts/medstat_cli.py --validate-only <plan.yaml>`
     To inspect the reference schema template or generate starter specs, run:
     `python <skill-dir>/scripts/medstat_cli.py --schema`
     On non-zero exit or runtime failure, inspect the structured `agent_remediation` JSON field emitted by the CLI or exception to self-heal dataset columns, missingness strategies, or model configurations before retrying. For proportional-hazards violations requiring strata, piecewise/time-varying HR, or RMST, run a Python script invoking `medstat` directly.
   - **EPV & Sparse-Data Routing** ([decision-heuristics.md §2.A](references/decision-heuristics.md#a-events-per-variable-epv-sparse-data--separation)):
     - Compute $\text{EPV}_{\text{binary}} = \min(N_{\text{events}}, N_{\text{non-events}}) / P_{\text{parameters}}$ (binary logistic) and $\text{EPV}_{\text{Cox}} = N_{\text{failures}} / P_{\text{parameters}}$ (Cox PH), where $P_{\text{parameters}}$ counts $k-1$ factor levels and spline df.
     - Halt at **Grilling Gate #3** when monotone separation is present.
     - When $\text{EPV}_{\text{binary}} < 10$ without separation, branch by estimand: apply Firth penalized logistic regression (`medstat.models.firth.fit_firth_logistic`) for etiological ORs, or L2 penalization/shrinkage for prediction models. For $\text{EPV}_{\text{Cox}} < 10$ without separation, apply Firth penalized Cox (`medstat.models.firth.fit_firth_cox`) or clinical risk score reduction.
   - **Autonomous Upgrades & Diagnostics**:
     - *Reclassification & Time-Dependent ROC*: When comparing nested/updated risk models ([§2.F](references/decision-heuristics.md#f-diagnostic-risk-reclassification--incremental-value)), compute Continuous and Categorical NRI ($K \times K$ matrices, reporting event and non-event NRIs separately), IDI, and paired DeLong $\Delta\text{AUROC}$ (`compute_reclassification_metrics`, `compare_diagnostic_models`). For censored longitudinal outcomes ([§2.G](references/decision-heuristics.md#g-time-dependent-roc-in-longitudinal-follow-up)), compute Cumulative/Dynamic IPCW $\text{AUROC}(t)$ and Youden cutpoints (`compute_time_dependent_roc`).
     - *TRIPOD+AI Validation, Recalibration & Sample Size*: For prediction model development ([§2.H](references/decision-heuristics.md#h-tripodai-prediction-model-internal-validation--uniform-shrinkage)), run bootstrap internal validation ($B \ge 200$, `bootstrap_internal_validation`) for optimism ($\text{boot} - \text{test}$), corrected metrics ($\text{apparent} - \text{optimism}$), uniform shrinkage $\hat{s}$, and re-estimated intercept $\alpha_0$. For external validation/updating, run intercept-only and logistic recalibration (`recalibrate_prediction_model`). Size prediction models via all 4 criteria of Riley et al. (2020; [§2.I](references/decision-heuristics.md#i-minimum-sample-size-for-clinical-prediction-models-riley-et-al-2020)) and diagnostic AUROC studies via Obuchowski (`calculate_riley_sample_size`, `calculate_diagnostic_auroc_sample_size`).
     - *Survival, Competing Risks & Non-Linearity*: On Schoenfeld violators, apply the stated remedy per [decision-heuristics.md §2.E](references/decision-heuristics.md#e-time-to-event-analysis-under-competing-risks--proportional-hazards-violations) (strata for nuisance, piecewise/time-varying HR or RMST for exposure); fit Restricted Cubic Splines (`medstat.models.splines`) for non-linear continuous predictors. When competing events preclude the primary event ([§2.E](references/decision-heuristics.md#e-time-to-event-analysis-under-competing-risks--proportional-hazards-violations)), compute Aalen-Johansen CIF (`fit_competing_risks_cif`) with a Naive $1-\text{KM}$ bias audit and **reconciled** probability conservation ($\sum_k \text{CIF}_k(t) + \hat{S}(t) \approx 1.0$); fit Cause-Specific Cox (csHR) for etiological questions and Fine-Gray IPCW regression (`fit_competing_risks_regression`) for prognostic questions, reporting Fine-Gray estimates as Subdistribution Hazard Ratios (SHR).
     - *Causal Inference, Clustering, Ordinal & Meta-Analysis*: For observational comparisons ([§2.D](references/decision-heuristics.md#d-observational-causal-inference-psm-vs-iptw-vs-overlap-weights)), execute 1:1 PSM (caliper $0.2\text{ SD}$, `medstat.causal.psm`) or IPTW (`medstat.causal.iptw`: ATE, ATT, ATC, ATO overlap weights, stabilized/truncated weights, Kish $N_{\text{eff}}$, Bessel weighted SMD, Huber-White SEs) plus VanderWeele & Ding E-values (`medstat.models.sensitivity`). For ordinal outcomes, fit proportional odds and run the Brant test (`medstat.models.ordinal`). For clustered data, compute Design Effect ($\text{DEFF}$) and fit GEE or mixed models (`medstat.models.multilevel`). For systematic synthesis, run fixed-effect and DerSimonian-Laird random-effects meta-analysis (`medstat.meta.models`).

4. **Pillar 4: Biostatistical Safety Invariants**
   - **Strict Numeric 0/1 Encoding**: Encode binary endpoints and survival events as numeric `0` and `1` (`1 = Event`, `0 = Non-event / Censored`) and competing-risk events as integers (`0 = Censored, 1 = Event of Interest, 2+ = Competing Event`), mapping text labels during **pre-flight** cleaning.
   - **Sample Retention Flow**: Track **reconciled** *(cohort counts satisfy $N_{\text{initial}} = N_{\text{excluded}} + N_{\text{analyzed}}$ with itemized exclusion counts summing to $N_{\text{excluded}}$)* cohort retention ($N_{\text{initial}} \longrightarrow N_{\text{excluded}} \longrightarrow N_{\text{analyzed}}$) via `SampleFlowTracker` (`medstat.data.retention`).
   - **Zero Silent Deletion & Missing-Data Routing** ([decision-heuristics.md §2.B](references/decision-heuristics.md#b-missing-data-strategy)):
     - Run **pre-flight** Little's MCAR evaluation (`medstat.data.clean`). Analyze only observed primary outcomes (halt at **Grilling Gate #1** if missing primary outcomes lack a protocol disposition).
     - *Triage by missingness*: For $< 5\%$ uninformative complete-case loss, run complete-case analysis with sensitivity notes. For $5\%\text{--}40\%$ missingness under MAR, run MICE (`medstat.data.missing`) and report Complete-Case Analysis alongside MICE. If non-primary missingness exceeds $> 40\%$ or MNAR is plausible, halt at **Grilling Gate #4**.
   - **Standard Confidence Intervals**: Report Wilson score 95% CIs for proportions/2x2 diagnostics, DeLong variance CIs for AUROC, exact F CIs for ICC, and profile likelihood CIs for Firth models.

5. **Pillar 5: Reporting Integrity & Multi-Format Documents**
   - **Report Intermediate Representation (IR)**: Assemble narrative blocks, 3-rule tables, and figures into `ReportDocument` (`medstat.reporting.ir`) so every reported value is **traceable** *(every narrative and table number matches `results_dict` within programmatic tolerance and is backed by exported source data)*.
   - **Publication Figures & Journal House Styles (300/600 DPI)**:
     - *Statistical plots (300 DPI)*: Read [figures.md](references/figures.md).
     - ***House style** (`NEJM`, `JAMA`, `LANCET`, `BMJ`, `CLINICAL`, `OKABE_ITO`), TrueType vector exports (`pdf.fonttype=42`), & layout/CVD/Zero-PHI gates*: Read [house-styles.md](references/house-styles.md).
     - ***Reconciled** study flows (CONSORT 2025/2010, PRISMA 2020, STARD 2015, STROBE cohort flow from `SampleFlowTracker`, 600 DPI)*: Read [flows.md](references/flows.md).
     - ***Schematic** illustrations (cells, skin/crater cross-sections, wound phases, pressure injuries, vessel stenosis, signaling pathways, 600 DPI)*: Read [illustration.md](references/illustration.md).
   - **Multi-Format Renderers & Citations**: Read [report-builder.md](references/report-builder.md) to compile `.html`, `.docx`, `.pptx`, `.md`, or `.pdf` deliverables and execute the Reference & Superscript Citation pipeline in [report-builder.md §4](references/report-builder.md#4-evidence-base-verification--superscript-citations-pipeline).
   - **Reporting Integrity Verification**: Audit every `ReportDocument` via `verify_report_integrity` (`medstat.reporting.integrity`) to enforce **Zero-PHI**, causal/E-value caveats, methods/retention documentation, and strict numerical traceability (`len(untraced_numbers) == 0`).

---

## 2. The Deterministic Grilling Gate

### Autonomous Execution (Decide, Execute, and Document)
- Decide, execute, and document routine biostatistical mechanics autonomously: parametric vs non-parametric tests, Mean (SD) vs Median [IQR], PH and Brant parallel-slopes tests, low-EPV penalization (**EPV & Sparse-Data Routing**), $< 40\%$ MAR missingness (**Zero Silent Deletion & Missing-Data Routing**), spline knots, Youden cutpoints, and NEJM/JAMA/APA table styling.

### Situations Requiring Mandatory Grilling / Inquiry
Halt **pre-flight** (or upon diagnostic detection) and ask the user directly when any of these 6 conditions (`#1`–`#6`) holds:
1. **Outcome, Event Direction, or Missing Outcome Ambiguity**: Primary event definition, censoring code, biomarker direction ([decision-heuristics.md §2.C](references/decision-heuristics.md#c-diagnostic-test-directionality--cutpoint-selection)), or missing primary outcome disposition is unspecified.
2. **Proposal vs Data Geometry Mismatch**: Survival modeling is requested without follow-up time or censoring status, or clustered modeling without cluster IDs.
3. **Severe Sparse Data & Monotone Separation**: Monotone separation occurs in logistic or Cox regression; confirm between Firth penalized estimation (`medstat.models.firth`) vs domain-guided covariate reduction.
4. **High Missingness ($> 40\%$) or Plausible MNAR without Sensitivity Analysis**: Non-primary missingness exceeds $> 40\%$ or clinically plausible MNAR lacks a pattern-mixture / delta-adjustment sensitivity analysis.
5. **Output Format, House Style & Evidence Polish Alignment Gate (Mandatory Inquiry if Unspecified)**: When the prompt omits deliverable format, figure **house style**, or polish mode, align upfront (auto-activating Gate #6 Full Polish when the prompt explicitly requests literature evidence, e.g. "หาหลักฐานมาประกอบ", "find evidence"):
   - **Deliverable Format**: **(Recommended) Interactive HTML Report (`.html`) with Dual / Toggle View** (`render_html`), **Word (`.docx`)**, **PowerPoint (`.pptx`)**, **PDF (`.pdf`)**, **Markdown (`.md`)**, or **In-Chat Summary**.
   - **Figure House Style** ([house-styles.md](references/house-styles.md)): **(Recommended) `NEJM`**, `JAMA`, `LANCET`, `BMJ`, or `CLINICAL` / `OKABE_ITO`.
   - **Manuscript Polish Mode** (authoritative specification in [report-builder.md §4](references/report-builder.md#4-evidence-base-verification--superscript-citations-pipeline)):
     - **(Standard) Skip Polish (User-Scoped References)**: Scope citations strictly to user-provided sources (omitting References if none provided) and verify metadata and claim support via [report-builder.md §4 Step 2A](references/report-builder.md#4-evidence-base-verification--superscript-citations-pipeline).
     - **(Optional) Full Publication Polish (Execute Gate #6)**: Execute full claim-vs-data audit and exploratory literature enrichment via [report-builder.md §4 Steps 1–4](references/report-builder.md#4-evidence-base-verification--superscript-citations-pipeline).
6. **Polish Grilling Gate (Evidence-Base Verification & Superscript Citations)**: When engaged (via prompt request, Gate #5 Full Polish selection, or on-demand), read and execute the 4-step pipeline in [report-builder.md §4](references/report-builder.md#4-evidence-base-verification--superscript-citations-pipeline) using any available literature MCPs or any available web search tool (e.g., `web_search`, `search_web`), and optional `medical-terminology` MCP when available. Halt at Gate #6 if any cited source is unverified, retracted, or contradicts the narrative (Anti-Phantom Citation Invariant).

---

## 3. Completion Criteria

The biostatistical task is complete when every item below is verified against an artifact or tool output:
- [ ] **Pre-Flight Data Audit & Spec Validation**: Every model variable's dtype, missingness count, distributional summary, numeric `0/1` (`0/1/2+`) coding, and **pre-flight** $\text{EPV}_{\text{binary}}$ or $\text{EPV}_{\text{Cox}}$ are recorded in script output; when declarative YAML plans are used, `medstat_cli.py --validate-only <plan.yaml>` returns `"valid": true` before model execution.
- [ ] **Grilling Gate Resolution**: Every triggered statistical **Grilling Gate** condition (`#1`–`#4`) has an explicit user decision recorded before final model execution.
- [ ] **Deliverable Format, House Style & Polish Mode Aligned**: Target deliverable format, figure **house style**, and polish mode are confirmed via prompt or Gate `#5`.
- [ ] **Evidence & Reference Grounding**: Every cited reference satisfies the verification, provenance, comparability, and sequential superscript citation rules in [report-builder.md §4](references/report-builder.md#4-evidence-base-verification--superscript-citations-pipeline) (`Gate #6`).
- [ ] **Model Convergence & Diagnostics**: The Python script exits with code `0`; every fitted model satisfies its model-specific convergence check (`converged == True` where exposed by GLM/Firth/GEE models, clean fit without `ConvergenceError` and finite standard errors for `fit_cox_ph`, and `model.mle_retvals["converged"] == True` plus Brant test for ordinal models) with finite estimates and 95% CIs; every applicable check (Schoenfeld remedies, Brant test, RCS non-linearity, GEE $\text{DEFF}$, meta-analysis $I^2/\tau^2$, calibration slope/Brier/ICI, Austin SMD $< 0.10$, Kish $N_{\text{eff}}$, and **reconciled** CIF probability conservation) is verified; and monotone separation is resolved per **Grilling Gate #3**.
- [ ] **Reconciled Cohort Retention**: `SampleFlowTracker` (`medstat.data.retention`) records **reconciled** cohort counts across every step.
- [ ] **Exhaustive Tables**: Every primary and sensitivity model (including complete-case alongside MICE for $5\%\text{--}40\%$ missingness) is formatted into 3-rule tables (NEJM/JAMA/APA) with point estimates, 95% CIs, $P$-values, and units.
- [ ] **Traceable Figures & Deliverable**: Every statistical figure is exported at 300 DPI and every **reconciled** flow diagram or **schematic** illustration at 600 DPI under the locked **house style** as a `FigureResult` with existing `png_path`, TrueType `vector_path` (`.pdf`, `fonttype=42`) / `svg_path`, `qa["passed"] == True`, **Zero-PHI** text, **schematic** disclaimer on illustrations, and **traceable** `csv_path` source data; and the deliverable file exists on disk.
- [ ] **Pillar 5 Integrity Audit Passed**: `verify_report_integrity(doc, allowed_untraced=...)` (`medstat.reporting.integrity`) returns `IntegrityReport.passed == True` with:
  1. *Zero-PHI Scan (gates `passed`)*: `len(phi_violations) == 0` across narrative text and `TableBlock` headers/cells, plus manual review confirming zero unprefixed patient names.
  2. *Numerical Traceability (gates `passed`)*: `len(untraced_numbers) == 0` for narrative text (any non-statistical annotation token must be explicitly whitelisted in `allowed_untraced`; any unverified estimate, CI bound, or $P$-value blocks completion), plus an independent table-to-results audit confirming every `TableBlock` value traces directly to `results_dict`.
  3. *Observational Causal Inference & E-value Check (gates `passed`)*: `has_causal_caveat == True` whenever `results_dict` keys match observational/regression/PSM/Cox models.
  4. *Methods & Retention Check (warning-only)*: `has_methods == True` and `has_retention_flow == True` (`warnings == []`).

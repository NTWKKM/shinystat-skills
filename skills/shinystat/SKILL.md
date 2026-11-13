---
name: shinystat
description: "Local clinical biostatistics with bundled medstat engine and uv shell: Table 1, regression (logistic, ordinal, GEE, Firth), survival (KM, Cox, competing risks), diagnostics (ROC, NRI/IDI), validation, causal (PSM, IPTW), agreement, meta-analysis, MICE, sample size, flow diagrams, figures, and clinical reports (html, docx, pptx, md, pdf)."
---

# Shinystat: Autonomous Biostatistical Decision Agent

Execute clinical biostatistics via **Run Sequence** (§0) and **The 5 Decision Pillars** (§1), halting at **The Deterministic Grilling Gate** (§2).

---

## 0. Run Sequence

1. **Inspect**: Run **Pillar 1** **pre-flight** profile (`df.info()`, `df.describe()`, missingness, numeric `0/1` coding; `check_separation` on each dummy-coded logistic design (rows complete on its variables) with rows, `littles_mcar_test` when any model variable is missing). *Done when* every model variable's dtype, missingness count, and coding are printed.
2. **Ask**: Present one consolidated turn holding every triggered pre-flight gate (§2: `#1`–`#4`) plus every unspecified **Output Alignment Gate** axis (`#5a`–`#5d`) via `ask_question` when available. *Done when* the user has answered every triggered question.
3. **Plan**: Map study archetype to model family (**Pillar 2**, [decision-heuristics.md §1](references/decision-heuristics.md#1-clinical-study-archetype-to-statistical-model-mapping)) and compute **pre-flight** $\text{EPV}_{\text{binary}}$ or $\text{EPV}_{\text{Cox}}$ (**Pillar 3**; "N/A" without a logistic or Cox model). *Done when* model family, missing-data route (**Pillar 4**), and EPV route are recorded.
4. **Execute**: Run **Pillar 3** scripts (or dispatch subagents per [subagent-protocol.md](references/subagent-protocol.md) when `#5d` selected **Subagent Offload**). *Done when* every fitted model converges and each applicable diagnostic passes or carries its documented remedy; halt at `#3` or `#4` (§2) if triggered mid-run.
5. **Render**: Generate **Pillar 5** 300/600 DPI figures in `#5b` **house style** and compile `#5a` deliverable via `medstat.reporting.renderers` (PDF needs Playwright/soffice or the Pillar 5 fallback; run [evidence-pipeline.md](references/evidence-pipeline.md) when citations/#5c active). *Done when* each primary estimate is plotted, every `FigureResult.qa["passed"] == True`, and the deliverable is one rendered `ReportDocument` (In-Chat Summary: the chat reply).
6. **Audit**: Run `verify_report_integrity` on that same `ReportDocument` with `exported_files` (data files written) and `delivered_text` (In-Chat Summary reply) or the chat message beside a file, and check **§3 Completion Criteria**. *Done when* `IntegrityReport.passed == True` and every §3 box is verified against tool output.

---

## 1. The 5 Decision Pillars

1. **Pillar 1: Data Reality Inspection (Autonomous Profiling)**
   - Run **pre-flight** checks on geometry, dtypes, missingness, and distributions (`df.info()`, `df.describe()`), isolating cohorts from non-data annotations (including Thai), preserving valid Thai analytic values.
   - Select summaries by shape (Mean (SD) symmetric; Median [IQR] skewed) and tests by design/estimand.

2. **Pillar 2: Research & Estimand Triangulation (Autonomous Selection)**
   - Align PICO/PECO and clinical archetype via [decision-heuristics.md §1](references/decision-heuristics.md#1-clinical-study-archetype-to-statistical-model-mapping).
   - **Effect Scale**: One headline measure per estimand (requested, else OR logistic / HR Cox); one stated unit per continuous predictor (e.g. per 10 years) across tables, text and figures.

3. **Pillar 3: Adaptive Python Scripting (Autonomous Biostatistical Authority)**
   - Write scripts using the bundled `medstat` engine in `scripts/medstat/` (signatures: [api-quickref.md](references/api-quickref.md)) and run with `uv run --with-requirements <skill-dir>/requirements.txt python <script.py>` (`--with <pkg>` for external packages).
   - **Spec Validation**: Validate YAML plans via `python <skill-dir>/scripts/medstat_cli.py --validate-only <plan.yaml>` (schema: `--schema`); self-heal from `agent_remediation` JSON on failure.
   - **EPV & Sparse-Data Routing** ([decision-heuristics.md §2.A](references/decision-heuristics.md#a-events-per-variable-epv-sparse-data--separation)):
     - $\text{EPV}_{\text{binary}} = \min(N_{\text{events}}, N_{\text{non-events}}) / P_{\text{parameters}}$; $\text{EPV}_{\text{Cox}} = N_{\text{failures}} / P_{\text{parameters}}$ ($k-1$ factor levels and spline df; no intercept).
     - Run `medstat.models.firth.check_separation` on every logistic design (outcome and propensity models); separated $\longrightarrow$ **Grilling Gate #3** (§2). For $\text{EPV} < 10$ without separation: Firth logistic (`medstat.models.firth.fit_firth_logistic`) for etiological ORs, L2 shrinkage for prediction, or Firth Cox (`fit_firth_cox`) / risk score reduction for survival.
   - **Autonomous Upgrades & Diagnostics** (load active branch in `decision-heuristics.md`):
     - *Observational causal (PSM, IPTW, Kish $N_{\text{eff}}$, E-values)* $\longrightarrow$ [§2.D](references/decision-heuristics.md#d-observational-causal-inference-psm-vs-iptw-vs-overlap-weights) (`medstat.causal`, `sensitivity`).
     - *Survival PH remedies & competing risks (CIF, csHR, Fine-Gray SHR)* $\longrightarrow$ [§2.E](references/decision-heuristics.md#e-time-to-event-analysis-under-competing-risks--proportional-hazards-violations) (`medstat.models.survival`).
     - *Reclassification (NRI, IDI, DeLong) & IPCW tdROC* $\longrightarrow$ [§2.F](references/decision-heuristics.md#f-diagnostic-risk-reclassification--incremental-value), [§2.G](references/decision-heuristics.md#g-time-dependent-roc-in-longitudinal-follow-up) (`medstat.diagnostic`).
     - *TRIPOD+AI bootstrap ($B \ge 200$, $\hat{s}$, $\alpha_0$), recalibration & sample size* $\longrightarrow$ [§2.H](references/decision-heuristics.md#h-tripodai-prediction-model-internal-validation--uniform-shrinkage), [§2.I](references/decision-heuristics.md#i-minimum-sample-size-for-clinical-prediction-models-riley-et-al-2020) (`validation`, `power`).
     - *Ordinal (Brant), RCS, clustered GEE ($\text{DEFF}$) & meta-analysis ($I^2, \hat{\tau}^2$)* $\longrightarrow$ [§2.K](references/decision-heuristics.md#k-ordinal-regression-restricted-cubic-splines-clustered-gee--meta-analysis) (`ordinal`, `splines`, `multilevel`, `meta`).

4. **Pillar 4: Biostatistical Safety Invariants**
   - **Numeric 0/1 Encoding**: Map binary/survival events to `0/1` (`1 = Event`, `0 = Non-event`) and competing risks to `0 = Censored, 1 = Event, 2+ = Competing Event` during **pre-flight** cleaning.
   - **Sample Retention Flow**: Track **reconciled** *($N_{\text{initial}} = N_{\text{excluded}} + N_{\text{analyzed}}$)* retention via `SampleFlowTracker` (`medstat.data.retention`).
   - **Zero Silent Deletion & Missing-Data Routing** ([decision-heuristics.md §2.B](references/decision-heuristics.md#b-missing-data-strategy)):
     - Run **pre-flight** Little's MCAR check (`medstat.data.clean`); name tested variables (untested or df 0 ≠ MCAR). Missing primary outcomes $\longrightarrow$ **Grilling Gate #1**.
     - *Triage*: $< 5\%$ complete-case with sensitivity notes; $5\%\text{--}40\%$ MAR $\longrightarrow$ MICE (`medstat.data.missing`) primary, Complete-Case sensitivity; $> 40\%$ exposure or covariate missingness or plausible MNAR $\longrightarrow$ **Grilling Gate #4** (§2).
     - **Standard CIs**: Wilson 95% CIs (proportions), DeLong (AUROC), exact F (ICC), profile likelihood (Firth).

5. **Pillar 5: Reporting Integrity & Multi-Format Documents**
   - **Report IR**: Assemble narrative, 3-rule tables, and figures into `ReportDocument` (`medstat.reporting.ir`); values are **traceable** *(match `results_dict`, backed by exported source data)*.
   - **Publication Figures & House Styles (300/600 DPI)**: Statistical plots in [figures.md](references/figures.md); **house styles** (`NEJM`, `JAMA`, `LANCET`, `BMJ`, `CLINICAL`, `OKABE_ITO`), TrueType PDF (`fonttype=42`), QA/CVD/Zero-PHI gates in [house-styles.md](references/house-styles.md); **reconciled** flows (600 DPI) in [flows.md](references/flows.md); **schematic** illustrations (600 DPI) in [illustration.md](references/illustration.md).
   - **Multi-Format Renderers & Citations**: Compile `.html`, `.docx`, `.pptx`, `.md`, or `.pdf` deliverables from `ReportDocument` via `medstat.reporting.renderers` (`render_html`, `render_docx`, `render_pptx`, `render_markdown`, `render_pdf`; run `PYTHONPATH=<skill-dir>/scripts uv run --with-requirements <skill-dir>/requirements.txt python <report.py>`; minimal build: [api-quickref.md](references/api-quickref.md); [report-builder.md](references/report-builder.md) holds cloud code, read only for the PDF fallback) and [evidence-pipeline.md](references/evidence-pipeline.md).

---

## 2. The Deterministic Grilling Gate

### Autonomous Execution (Decide, Execute, and Document)
- Decide, execute, and document routine mechanics autonomously: parametric vs non-parametric tests, Mean (SD) vs Median [IQR], PH/Brant tests, low-EPV penalization, $< 40\%$ MAR missingness, spline knots, Youden cutpoints, and table styling.

### Situations Requiring Mandatory Grilling / Inquiry
After **Pillar 1** pre-flight inspection, ask all triggered pre-flight questions (`#1`–`#4`; `#4` when pre-flight confirms $> 40\%$ exposure or covariate missingness or clinically plausible MNAR without sensitivity analysis, and every unspecified `#5` axis) in a **single consolidated turn**, written as one question per gate and axis (computing EPV after `#1`/`#2` resolution). When your tools include `ask_question`, send the consolidated turn through it—one question per gate and per Gate #5 axis, recommended option first. When the question tool caps a call (e.g. `AskUserQuestion`: 4 questions × 4 options), split into consecutive calls and name options beyond the cap in the question text. If a gate (`#2`–`#4` first found mid-run, `#6`) triggers during execution, halt directly in **Direct in Main Agent** mode, or via [subagent-protocol.md](references/subagent-protocol.md) when **Subagent Offload** is active:
1. **Outcome, Event Direction, or Missing Outcome Ambiguity**: Primary event definition, censoring code, pooled effect scale (log ratio vs difference), biomarker direction not fixed by [decision-heuristics.md §2.C](references/decision-heuristics.md#c-diagnostic-test-directionality--cutpoint-selection), or missing outcome disposition is unspecified.
2. **Proposal vs Data Geometry Mismatch**: Survival modeling without follow-up time/censoring, clustered modeling without cluster IDs, or a flow diagram arm without observed receipt/analysis counts.
3. **Severe Sparse Data & Monotone Separation**: Monotone separation in logistic or Cox regression; confirm Firth estimation (`medstat.models.firth`) vs covariate reduction.
4. **High Missingness ($> 40\%$) or Plausible MNAR without Sensitivity Analysis**: Missingness $> 40\%$ in the exposure or any covariate (Little's test cannot lift it) or clinically plausible MNAR lacks pattern-mixture / delta-adjustment sensitivity analysis.
5. **Output Alignment Gate** (Mandatory Inquiry per Unspecified Axis): Gate #5 has four independent **axes**. Ask one separate question per axis left unspecified, listing every option (auto-activating Gate #6 Full Polish when prompt explicitly requests literature evidence, e.g. "หาหลักฐานมาประกอบ", "find evidence"):
   - **5a. Deliverable Format**: **(Recommended) Interactive HTML (`.html`) with Dual/Toggle View** (`render_html`, `plot_dual_*` figures), **Word (`.docx`)**, **PowerPoint (`.pptx`)**, **PDF (`.pdf`)**, **Markdown (`.md`)**, or **In-Chat Summary**.
   - **5b. Figure House Style** ([house-styles.md](references/house-styles.md)): **(Recommended) `NEJM`**, `JAMA`, `LANCET`, `BMJ`, `CLINICAL`, or `OKABE_ITO` (CVD-safe); offer each separately.
   - **5c. Manuscript Polish Mode** ([evidence-pipeline.md](references/evidence-pipeline.md)): **(Standard) Skip Polish (User-Scoped References)** (Step 2A; omit if none provided) vs **(Optional) Full Publication Polish (Execute Gate #6)** (Steps 1–4).
   - **5d. Execution Context Mode** *(ask when unspecified, tools include a subagent tool, e.g. `Agent` or `Task`, and the run fits ≥2 distinct estimands (a primary model and its sensitivity re-fits count as one) or uses Full Polish, even when every other axis is specified; otherwise record **Direct in Main Agent**)*: **(Recommended for multi-model/polish runs) Subagent Offload** ([subagent-protocol.md](references/subagent-protocol.md)) vs. **Direct in Main Agent**.
6. **Polish Grilling Gate (Evidence-Base Verification & Superscript Citations)**: When engaged, run [evidence-pipeline.md](references/evidence-pipeline.md) via literature MCPs, web search (`web_search`, `search_web`), and optional `medical-terminology` MCP. Halt at Gate #6 if any cited source is unverified, retracted, or contradicts narrative.

---

## 3. Completion Criteria

The biostatistical task is complete when every item below is verified against an artifact or tool output:
- [ ] **Pre-Flight Data Audit & Spec Validation**: Every model variable's dtype, missingness count, summary, numeric `0/1` (`0/1/2+`) coding, and **pre-flight** $\text{EPV}_{\text{binary}}$ or $\text{EPV}_{\text{Cox}}$ (or its "N/A") are recorded in output; YAML plans pass `medstat_cli.py --validate-only <plan.yaml>` (`"valid": true`).
- [ ] **Grilling Gate Resolution**: Every triggered **Grilling Gate** condition (`#1`–`#4`) has an explicit user decision before final model execution.
- [ ] **Deliverable Format, House Style & Polish Mode Aligned**: Target format, figure **house style**, polish mode, and execution context mode (when subagent tools exist) are confirmed via prompt or Gate `#5` axis questions; when **Subagent Offload** is used, independent Main-Agent on-disk verification and external row-level / `.scratch_checkpoint/` cleanup per [subagent-protocol.md](references/subagent-protocol.md) are completed.
- [ ] **Evidence & Reference Grounding**: Every cited reference satisfies the verification, provenance, comparability, and superscript citation rules in [evidence-pipeline.md](references/evidence-pipeline.md) (`Gate #6`).
- [ ] **Model Convergence & Diagnostics**: Script exits `0`; every model satisfies convergence (`converged == True` for GLM/Firth/GEE, finite SEs for `fit_cox_ph`, and `mle_retvals["converged"] == True` plus Brant test for ordinal) with finite estimates/95% CIs; applicable diagnostics pass or carry their documented remedy; and separation is resolved per **Gate #3**.
- [ ] **Reconciled Cohort Retention**: `SampleFlowTracker` (`medstat.data.retention`) records **reconciled** cohort counts across every step.
- [ ] **Exhaustive Tables**: Every primary and sensitivity model (MICE primary, complete-case sensitivity for $5\%\text{--}40\%$ missingness) is formatted into 3-rule tables (NEJM/JAMA/APA) with estimates, 95% CIs, $P$-values, and units.
- [ ] **Traceable Figures & Deliverable**: Every statistical figure (300 DPI) and **reconciled** flow / **schematic** illustration is exported in the locked **house style** as a `FigureResult` (`png_path`, TrueType `vector_path` `fonttype=42`, `svg_path`, `qa["passed"] == True`, **Zero-PHI**, **schematic** disclaimer, **traceable** `csv_path`), and deliverable exists on disk when `#5a` requests a file deliverable (for In-Chat Summary, presented in chat without an on-disk report file).
- [ ] **Pillar 5 Integrity Audit Passed**: `verify_report_integrity(doc, allowed_untraced=..., study_design=study_design)` (`medstat.reporting.integrity`) returns `IntegrityReport.passed == True`:
  1. *Zero-PHI*: `len(phi_violations) == 0` across narrative and `TableBlock`s, plus a read of narrative and table cells confirming every patient identifier is synthetic.
  2. *Numerical Traceability*: `len(untraced_numbers) == 0` (whitelisted constants in `allowed_untraced`) plus independent table-to-results reconciliation against `results_dict`.
  3. *Causal & E-value Check*: `has_causal_caveat == True` for observational/causal analyses or when causal claims are made; a logistic or Cox key in results_dict alone does not require a caveat for non-causal prediction reports lacking causal claims.
  4. *Methods & Retention*: `has_methods == True` and `has_retention_flow == True`.

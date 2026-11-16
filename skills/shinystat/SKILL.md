---
name: shinystat
description: "Local clinical biostatistics with bundled medstat engine and uv shell: Table 1, regression (logistic, ordinal, GEE, Firth), survival (KM, Cox, competing risks), diagnostics (ROC, NRI/IDI), validation, causal (PSM, IPTW), agreement, meta-analysis, MICE, sample size, flow diagrams, figures, and clinical reports (html, docx, pptx, md, pdf)."
---

# Shinystat: Autonomous Biostatistical Decision Agent

Execute clinical biostatistics via **Run Sequence** (§0) and **The 5 Decision Pillars** (§1), halting at **The Deterministic Grilling Gate** (§2).

---

## 0. Run Sequence

1. **Inspect**: Run **Pillar 1** **pre-flight** profile (`df.info()`, `df.describe()`, missingness, numeric `0/1` coding; `check_separation` on each dummy-coded logistic design (rows complete on its variables) per §2.B, `littles_mcar_test` when any model variable is missing). *Done when* every model variable's dtype, missingness count, and coding are printed.
2. **Ask**: Present one consolidated turn holding every triggered pre-flight gate (§2: `#1`–`#4`) plus every unspecified **Output Alignment Gate** axis (`#5a`–`#5d`) via `ask_question` when available. *Done when* triggered `#1`–`#4` have explicit user decisions and `#5` is answered (or Gate `#5` defaults are logged in an explicitly non-interactive run).
3. **Plan**: Map archetype to model (**Pillar 2**, [decision-heuristics.md §1](references/decision-heuristics.md#1-clinical-study-archetype-to-statistical-model-mapping)), compute **pre-flight** $\text{EPV}_{\text{binary}}$ / $\text{EPV}_{\text{Cox}}$ (**Pillar 3**; "N/A" if non-logistic/Cox), and build `RunManifest` + `primary` node ([analysis-lineage.md](references/analysis-lineage.md)). *Done when* model, missingness (**Pillar 4**), EPV, manifest, and primary node are recorded.
4. **Execute**: Run **Pillar 3** scripts (or subagents per [subagent-protocol.md](references/subagent-protocol.md) on `#5d` **Subagent Offload**); repair provisional nodes (cap 2), freeze answered nodes, and branch sensitivity children (`plan_delta`). *Done when* models converge and diagnostics pass or carry remedies; halt at `#3`, `#4`, or `#7` (§2) mid-run.
5. **Render**: Generate **Pillar 5** 300/600 DPI figures in `#5b` **house style** and compile `#5a` deliverable via `medstat.reporting.renderers` (PDF needs Playwright/soffice or Pillar 5 fallback; [evidence-pipeline.md](references/evidence-pipeline.md) if citations/#5c active; follow [analysis-lineage.md §8–§9](references/analysis-lineage.md#8-output-artifact-folder-layout)). *Done when* each primary estimate is plotted, `FigureResult.qa["passed"] == True`, and deliverable is one `ReportDocument` (In-Chat Summary: chat reply).
6. **Audit**: Run `verify_report_integrity` on that same `ReportDocument` with `exported_files`, `data=`, and `delivered_text` (In-Chat Summary reply) or the chat message beside a file; re-run after any post-audit edit to an exported file, and check **§3**. *Done when* `IntegrityReport.passed == True` and every §3 box is verified.

---

## 1. The 5 Decision Pillars

1. **Pillar 1: Data Reality Inspection (Autonomous Profiling)**
   - Run **pre-flight** checks on geometry, dtypes, missingness, and distributions (`df.info()`, `df.describe()`), isolating cohorts from non-data annotations (including Thai), preserving valid Thai analytic values.
   - Select summaries by shape (Mean (SD) symmetric; Median [IQR] when skewed per `pd.Series.skew()`) and tests by design/estimand.

2. **Pillar 2: Research & Estimand Triangulation (Autonomous Selection)**
   - Align PICO/PECO and clinical archetype via [decision-heuristics.md §1](references/decision-heuristics.md#1-clinical-study-archetype-to-statistical-model-mapping).
   - **Effect Scale**: One headline measure per estimand (requested, else OR logistic / HR Cox); one stated unit per continuous predictor—per 10 years (age in years), per 1 SD (other continuous), or `"unit not stated"` if data omits units (never invent units)—across tables, text and figures.

3. **Pillar 3: Adaptive Python Scripting (Autonomous Biostatistical Authority)**
   - Write scripts using bundled `medstat` in `scripts/medstat/` ([api-quickref.md](references/api-quickref.md) — check signatures/`Return shapes` first), adding `<skill-dir>/scripts` to `sys.path` (never `PYTHONPATH=... uv run python`), via `uv run --with-requirements <skill-dir>/requirements.txt python <script.py>` (`--with <pkg>` for extras).
   - **Spec Validation**: Validate YAML plans via `scripts/medstat_cli.py --validate-only <plan.yaml>` (`--schema`; `icc`/`meta` use CLI subcommands); self-heal from `agent_remediation` JSON.
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
   - **Report IR & Lineage**: Assemble narrative, 3-rule tables, figures, `manifest`, and `lineage` into `ReportDocument` (`medstat.reporting.ir`); values are **traceable** *(match `results_dict`, backed by source data)*.
   - **Publication Figures & House Styles (300/600 DPI)**: Statistical plots ([figures.md](references/figures.md)), **house styles** (`NEJM`, `JAMA`, `LANCET`, `BMJ`, `CLINICAL`, `OKABE_ITO`), TrueType PDF (`fonttype=42`), QA/CVD/Zero-PHI ([house-styles.md](references/house-styles.md)), **reconciled** flows 600 DPI ([flows.md](references/flows.md)), **schematic** art 600 DPI ([illustration.md](references/illustration.md)).
   - **Multi-Format Renderers & Citations**: Compile `.html`, `.docx`, `.pptx`, `.md`, `.pdf` from `ReportDocument` via `medstat.reporting.renderers` (`render_html`, `render_docx`, `render_pptx`, `render_markdown`, `render_pdf`; [api-quickref.md](references/api-quickref.md); [report-builder.md](references/report-builder.md) is cloud code, read only for the PDF fallback) and [evidence-pipeline.md](references/evidence-pipeline.md).

---

## 2. The Deterministic Grilling Gate

### Autonomous Execution (Decide, Execute, and Document)
- Decide routine mechanics autonomously: test choice, Mean (SD)/Median [IQR], PH/Brant, low-EPV penalization, $< 40\%$ MAR, splines, Youden cutpoints, table styling.

### Situations Requiring Mandatory Grilling / Inquiry
After **Pillar 1** pre-flight, ask all triggered gates (`#1`–`#4`; `#4` for $> 40\%$ missingness or clinically plausible MNAR without sensitivity analysis, plus unspecified `#5` axes) in a **single consolidated turn**, one question per gate/axis (computing EPV after `#1`/`#2` resolution). Under `ask_question` or `invoke_subagent` (`send_message`), when your tools include `ask_question`, send the consolidated turn through it (recommended option first; split if capped, naming extra options in text); never silently skip Gate #5 or assume defaults—wait for the user/caller, or (only if explicitly told to run non-interactively) log chosen defaults. If `#2`–`#4` first found mid-run, `#6`, or `#7` triggers, halt directly (**Direct in Main Agent**) or per [subagent-protocol.md](references/subagent-protocol.md) (**Subagent Offload**):
1. **Outcome, Event Direction, or Missing Outcome Ambiguity**: Primary event definition, censoring code, pooled effect scale (log ratio vs difference), biomarker direction not fixed by [decision-heuristics.md §2.C](references/decision-heuristics.md#c-diagnostic-test-directionality--cutpoint-selection), or missing outcome disposition is unspecified.
2. **Proposal vs Data Geometry Mismatch**: Survival modeling without follow-up time/censoring, clustered modeling without cluster IDs, or a flow diagram arm without observed receipt/analysis counts.
3. **Severe Sparse Data & Monotone Separation**: Monotone separation in logistic or Cox regression; confirm Firth estimation (`medstat.models.firth`) vs covariate reduction.
4. **High Missingness ($> 40\%$) or Plausible MNAR without Sensitivity Analysis**: Missingness $> 40\%$ in the exposure or any covariate (Little's test cannot lift it) or clinically plausible MNAR lacks pattern-mixture / delta-adjustment sensitivity analysis.
5. **Output Alignment Gate** (Mandatory Inquiry per Unspecified Axis): Gate #5 has four independent **axes**. Ask one separate question per axis left unspecified, listing every option (auto-activating Gate #6 Full Polish when prompt explicitly requests literature evidence, e.g. "หาหลักฐานมาประกอบ", "find evidence"):
   - **5a. Deliverable Format**: **(Recommended) Interactive HTML (`.html`) with Dual/Toggle View** (`render_html`, `plot_dual_*` figures), **Word (`.docx`)**, **PowerPoint (`.pptx`)**, **PDF (`.pdf`)**, **Markdown (`.md`)**, or **In-Chat Summary**.
   - **5b. Figure House Style** ([house-styles.md](references/house-styles.md)): **(Recommended) `NEJM`**, `JAMA`, `LANCET`, `BMJ`, `CLINICAL`, or `OKABE_ITO` (CVD-safe); offer each separately.
   - **5c. Manuscript Polish Mode** ([evidence-pipeline.md](references/evidence-pipeline.md)): **(Standard) Skip Polish (User-Scoped References)** (Step 2A; omit if none provided) vs **(Optional) Full Publication Polish (Execute Gate #6)** (Steps 1–4).
   - **5d. Execution Context Mode** *(ask when unspecified, tools include a subagent tool, e.g. `invoke_subagent`, `Agent`, or `Task`, and the run fits ≥2 distinct estimands (a primary model and its sensitivity re-fits count as one) or uses Full Polish, even when every other axis is specified; otherwise record **Direct in Main Agent**)*: **(Recommended for multi-model/polish runs) Subagent Offload** ([subagent-protocol.md](references/subagent-protocol.md)) vs. **Direct in Main Agent**.
6. **Polish Grilling Gate (Evidence-Base Verification & Superscript Citations)**: Run [evidence-pipeline.md](references/evidence-pipeline.md) via literature MCPs, `search_web`/`web_search`, and optional `medical-terminology` MCP; halt if any source is unverified, retracted, or contradicts narrative.
7. **Repair Cap (2 Provisional Failures)**: Halt and ask user after 2 consecutive un-answered repairs on a node (`RepairCapExceededError`) or a repeated setup error ([analysis-lineage.md §4](references/analysis-lineage.md#4-repair-cap-2-attempts--gate-7)).

---

## 3. Completion Criteria

Complete when every box is verified against an artifact or tool output:
- [ ] **Pre-Flight Data Audit & Spec Validation**: Model dtypes, missingness, `0/1` (`0/1/2+`) coding, and **pre-flight** $\text{EPV}_{\text{binary}}$ / $\text{EPV}_{\text{Cox}}$ ("N/A" if unused) recorded; YAML plans pass `medstat_cli.py --validate-only <plan.yaml>` (`"valid": true`).
- [ ] **Grilling Gate Resolution**: Triggered **Grilling Gate** (`#1`–`#4`) conditions have explicit user decisions before final modeling.
- [ ] **Deliverable, House Style & Polish Aligned**: Format, **house style**, polish, and execution mode confirmed via prompt, Gate `#5`, or logged defaults (explicitly non-interactive run); under **Subagent Offload**, Main-Agent verification and external / `.scratch_checkpoint/` cleanup ([subagent-protocol.md](references/subagent-protocol.md)) done.
- [ ] **Evidence & Reference Grounding**: Citations satisfy [evidence-pipeline.md](references/evidence-pipeline.md) (`Gate #6`).
- [ ] **Convergence & Diagnostics**: Exits `0`; models converge (`converged == True` GLM/Firth/GEE, finite SEs `fit_cox_ph`, `mle_retvals["converged"] == True` + Brant ordinal); separation resolved (**Gate #3**).
- [ ] **Reconciled Cohort Retention**: `SampleFlowTracker` (`medstat.data.retention`) records **reconciled** counts at every step.
- [ ] **Exhaustive Tables**: Primary and sensitivity models (MICE primary, complete-case for $5\%\text{--}40\%$ missingness) use 3-rule tables with estimates, 95% CIs, $P$-values, and units.
- [ ] **Traceable Figures & Deliverable**: Statistical (300 DPI) and **reconciled** flow / **schematic** figures in **house style** (`FigureResult`: `png_path`, TrueType `vector_path`, `svg_path`, `qa["passed"] == True`, **Zero-PHI**, **schematic** disclaimer, **traceable** `csv_path`); file on disk if `#5a` requested one.
- [ ] **Lineage & Manifest**: `has_manifest == True`, `lineage_violations == []`, primary `answered` and frozen, sensitivity `plan_delta` recorded, `data=` hash matches.
- [ ] **Pillar 5 Integrity Audit Passed**: `verify_report_integrity(doc, allowed_untraced=..., study_design=study_design, data=...)` (`medstat.reporting.integrity`) returns `IntegrityReport.passed == True`:
  1. *Zero-PHI*: `len(phi_violations) == 0` across narrative and `TableBlock`s; synthetic IDs only.
  2. *Numerical Traceability*: `len(untraced_numbers) == 0` (`allowed_untraced` whitelisted) plus table-to-results reconciliation against `results_dict`.
  3. *Causal & E-value Check*: `has_causal_caveat == True` for observational/causal runs or causal claims.
  4. *Methods & Retention*: `has_methods == True` and `has_retention_flow == True`.

---
name: shinystat-cloud
description: "Use when analyzing clinical or EHR data in cloud sandboxes (standalone Python, no medstat import): Table 1, logistic regression, Firth sparse-data models, survival (Kaplan-Meier, Cox, Competing Risks), diagnostics (2x2, ROC, NRI/IDI, time-dependent ROC), prediction validation (TRIPOD+AI bootstrap shrinkage, calibration, DCA, recalibration), causal inference (PSM, IPTW), agreement (Bland-Altman, ICC), MICE, sample size (Riley, Obuchowski), 300 DPI figures, or reports (html, docx, pptx, md, pdf)."
---

# Shinystat Cloud: Autonomous Biostatistical Decision Agent

Execute clinical biostatistics in cloud sandboxes through the 5 Decision Pillars below using standalone Python recipes, exercising autonomous authority over statistical mechanics and halting at **The Deterministic Grilling Gate** (§2) only for clinical or protocol ambiguities and **Output Format Alignment Gate** (#5) confirmation.

---

## 1. The 5 Decision Pillars (Cloud Sandbox Edition)

1. **Pillar 1: Data Reality Inspection (Autonomous Profiling)**
   - Run **pre-flight** *(verified on raw data before fitting any model)* checks on geometry, headers, dtypes, missingness, and distributions (`df.info()`, `df.describe()`), isolating analytic cohorts from embedded summaries, multi-line headers, and Thai/regional text.
   - Choose descriptive summaries by shape (Mean (SD) symmetric; Median [IQR] skewed) and inferential tests by study design and estimand rather than skewness alone (Mann–Whitney U tests rank-sum shifts, not medians, when group shapes differ).
   - Compute **pre-flight** $\text{EPV}_{\text{binary}}$ or $\text{EPV}_{\text{Cox}}$ and route sparse data per **EPV & Sparse-Data Routing** (Pillar 3).

2. **Pillar 2: Research & Estimand Triangulation (Autonomous Selection)**
   - Align design with PICO/PECO and clinical archetype (Cohort, Case-Control, RCT, Diagnostic Accuracy, Agreement, Clustered, Causal Inference, Competing Risks).
   - When mapping study archetypes or decision trees (§2.A–§2.I) to model families, read [decision-heuristics.md](references/decision-heuristics.md#1-clinical-study-archetype-to-statistical-model-mapping).

3. **Pillar 3: Adaptive Standalone Python Scripting (Autonomous Biostatistical Authority)**
   - **Cloud Sandbox Protocol**: Write self-contained Python scripts using standard libraries (`pandas`, `numpy`, `scipy.stats`, `statsmodels`, `lifelines`, `scikit-learn`). When selecting standalone implementations (`Recipe 1`–`Recipe 20`), read the Recipe Index in [python-recipes.md](references/python-recipes.md#recipe-index) to jump to the target recipe heading anchor.
   - **EPV & Sparse-Data Routing** ([decision-heuristics.md §2.A](references/decision-heuristics.md#a-events-per-variable-epv-sparse-data--separation)):
     - Compute $\text{EPV}_{\text{binary}} = \min(N_{\text{events}}, N_{\text{non-events}}) / P_{\text{parameters}}$ and $\text{EPV}_{\text{Cox}} = N_{\text{failures}} / P_{\text{parameters}}$ ($P_{\text{parameters}}$ counts $k-1$ factor levels and spline df).
     - Halt at **Grilling Gate #3** when monotone separation is present.
     - When $\text{EPV}_{\text{binary}} \ge 10$ without separation, use `Recipe 10` (multivariable logistic table). If $\text{EPV}_{\text{binary}} < 10$ without separation, branch by estimand: use `Recipe 7` (Firth penalized logistic with profile likelihood CIs) for etiological ORs, or apply L2 shrinkage for prediction models. For $\text{EPV}_{\text{Cox}} < 10$ without separation, apply a scale-aware, data-validated L2-penalized Cox model in `Recipe 11` (standardize continuous predictors and report the transformation, or select and validate `penalizer` on the supported data), or use clinical risk score reduction.
   - **Autonomous Upgrades & Diagnostics**:
     - *Reclassification & Time-Dependent ROC*: When evaluating incremental biomarker value ([§2.F](references/decision-heuristics.md#f-diagnostic-risk-reclassification--incremental-value)), use `Recipe 16` for Continuous/Categorical NRI (reporting event and non-event NRIs separately) and IDI, and `Recipe 3` for paired DeLong $\Delta\text{AUROC}$; for censored longitudinal discrimination across horizons ([§2.G](references/decision-heuristics.md#g-time-dependent-roc-in-longitudinal-follow-up)), use `Recipe 17` for Cumulative/Dynamic IPCW $\text{AUROC}(t)$.
     - *TRIPOD+AI Validation, Recalibration & Sample Size*: For prediction model development ([§2.H](references/decision-heuristics.md#h-tripodai-prediction-model-internal-validation--uniform-shrinkage)), use `Recipe 18` for bootstrap internal validation ($B \ge 200$, uniform shrinkage $\hat{s}$, re-estimated intercept $\alpha_0$) and `Recipe 4` for Brier score, calibration slope/intercept, Austin–Steyerberg ICI, and Vickers DCA; for external model updating, use `Recipe 19` (intercept-only and logistic recalibration); for study sizing ([§2.I](references/decision-heuristics.md#i-minimum-sample-size-for-clinical-prediction-models-riley-et-al-2020)), use `Recipe 20` (Riley et al. 2020 4 criteria & Obuchowski 1998 diagnostic AUROC).
     - *Survival & Competing Risks*: Test Schoenfeld residuals in Cox PH (`Recipe 11`) and stratify or add time-varying covariates on PH violation. When competing events exist ([§2.E](references/decision-heuristics.md#e-time-to-event-analysis-under-competing-risks)), use `Recipe 15` for Aalen-Johansen CIF (with Naive $1-\text{KM}$ bias audit and **reconciled** probability conservation $\sum_k \text{CIF}_k(t) + \hat{S}(t) \approx 1.0$), Cause-Specific Cox (csHR, etiological), and Fine-Gray IPCW regression (prognostic); report Fine-Gray estimates as Subdistribution Hazard Ratios (SHR), never as relative risks (RR) or risk ratios.
     - *Causal Inference, Clustering & Non-Linearity*: For observational comparisons ([§2.D](references/decision-heuristics.md#d-observational-causal-inference-psm-vs-iptw-vs-overlap-weights)), use `Recipe 6` for 1:1 PSM (caliper $0.2\text{ SD}$) or `Recipe 14` for IPTW (ATE, ATT, ATC, ATO overlap weights, stabilized/truncated weights, Kish $N_{\text{eff}}$, Bessel weighted SMD, Huber-White robust SEs), plus `Recipe 9` for VanderWeele & Ding E-values. Fit splines for non-linear continuous predictors and `statsmodels` GEE (robust sandwich SEs, $\text{DEFF}$) for clustered cohorts.

4. **Pillar 4: Biostatistical Safety Invariants**
   - **Strict Numeric 0/1 Encoding**: Encode binary endpoints and survival events as numeric `0` and `1` (`1 = Event`, `0 = Non-event / Censored`) and competing-risk events as integers (`0 = Censored, 1 = Event of Interest, 2+ = Competing Event`), mapping text labels (`"Yes"/"No"`, `"Dead"/"Alive"`) during **pre-flight** cleaning.
   - **Sample Retention Flow**: Track **reconciled** *(cohort counts satisfy $N_{\text{initial}} = N_{\text{excluded}} + N_{\text{analyzed}}$ with itemized exclusion counts summing to $N_{\text{excluded}}$)* cohort retention ($N_{\text{initial}} \longrightarrow N_{\text{excluded}} \longrightarrow N_{\text{analyzed}}$) with documented exclusion reasons across every step.
   - **Zero Silent Deletion & Missing-Data Routing** ([decision-heuristics.md §2.B](references/decision-heuristics.md#b-missing-data-strategy)):
     - Run **pre-flight** Little's MCAR test (`Recipe 8`; $H_0: \text{MCAR}$, where $p \ge 0.05$ does not prove MCAR or establish MAR). Analyze only observed primary outcomes and never impute primary outcomes; halt at **Grilling Gate #1** if missing primary outcomes lack a protocol disposition.
     - *Triage by missingness*: When complete-case loss is $< 5\%$ and non-primary missingness is clinically uninformative, run complete-case analysis with sensitivity notes. For $5\%\text{--}40\%$ missingness under documented MAR, read [Recipe 12](references/python-recipes.md#12-multiple-imputation-by-chained-equations-mice--rubins-rules) for proper Bayesian MICE ($M \ge 5$ for $\le 30\%$ missingness, $M \ge 20$ for $> 30\%$, continuous targets vs type-specific categorical models) and always report Complete-Case Analysis alongside MICE as a sensitivity benchmark (agreement does not establish MAR or exclude MNAR bias). If non-primary missingness exceeds $> 40\%$ (not MNAR), halt at **Grilling Gate #4** to choose between (a) dropping the auxiliary covariate, (b) high-$M$ MICE ($M \ge 40$, tracking FMI) under plausible MAR, or (c) pattern-mixture modeling (or missing-indicator adjustment restricted to randomized-trial baseline covariates or explicitly labeled sensitivity analyses). Whenever MNAR is clinically plausible, run a pattern-mixture or delta-adjustment sensitivity analysis, or halt at **Grilling Gate #4** before reporting pooled estimates.
   - **Standard Confidence Intervals**: Report Wilson score 95% CIs for proportions/2x2 metrics (`Recipe 2`), DeLong CIs for AUROC (`Recipe 3`), exact F CIs for ICC (`Recipe 5`), and profile likelihood CIs for Firth models (`Recipe 7`).

5. **Pillar 5: Reporting Integrity, Publication Figures & Multi-Format Documents**
   - **Figure Generation**: When generating 300 DPI colorblind-safe Matplotlib figures (forest, KM, CIF, ROC, TD-ROC, reclassification, calibration, recalibration, bootstrap optimism, DCA, Bland-Altman, 3-way love plot, STROBE flow) or zero-dependency Plotly specs, read [figures.md](references/figures.md).
   - **Multi-Format Export**: When assembling `.html`, `.docx`, `.pptx`, `.md`, or `.pdf` deliverables after **Output Format Alignment Gate** (#5) confirmation, read [report-builder.md](references/report-builder.md) so every reported value is **traceable** *(every narrative and table number matches `results_dict` within programmatic tolerance and is backed by exported source data)*.
   - **Integrity Verification**: Run `verify_standalone_integrity` ([report-builder.md §3](references/report-builder.md#3-standalone-reporting-integrity-verification)) to confirm **traceable** narrative numbers, causal/E-value caveats, and **Zero-PHI** compliance (automated regex scan for `HN`, 13-digit Thai IDs, phone numbers, DOB, and honorific-prefixed names `Mr.`, `Mrs.`, `Ms.`, `Dr.`, `นาย`, `นาง`, `นางสาว`, `นพ.`, `พญ.`, `ดร.` across narrative text, table headers, and cells, plus manual review for unprefixed names before external release; never release unscreened patient identifiers).

---

## 2. The Deterministic Grilling Gate

### Autonomous Execution (Decide, Execute, and Document)
- Decide, execute, and document routine biostatistical mechanics autonomously: parametric vs non-parametric tests, Mean (SD) vs Median [IQR], PH and parallel-slopes tests, low-EPV penalization (**EPV & Sparse-Data Routing**), $< 40\%$ MAR missingness (**Zero Silent Deletion & Missing-Data Routing**), spline knots, Youden cutpoints, and NEJM/JAMA/APA table styling.

### Situations Requiring Mandatory Grilling / Inquiry
Halt **pre-flight** (or upon diagnostic detection) and ask the user directly when any of these 5 conditions holds:
1. **Outcome, Event Direction, or Missing Outcome Ambiguity**: Primary event definition, censoring code, biomarker direction (low-is-abnormal eGFR/Platelets vs high-is-abnormal Troponin/Lactate; [decision-heuristics.md §2.C](references/decision-heuristics.md#c-diagnostic-test-directionality--cutpoint-selection)), or missing primary outcome disposition is unspecified.
2. **Proposal vs Data Geometry Mismatch**: Survival modeling is requested without follow-up time or censoring status, or clustered modeling without cluster IDs.
3. **Severe Sparse Data & Monotone Separation**: Monotone separation occurs in logistic or Cox regression; confirm between Firth/penalized estimation (`Recipe 7` or scale-aware, data-validated L2-penalized Cox in `Recipe 11`) vs domain-guided covariate reduction.
4. **High Missingness ($> 40\%$) or Plausible MNAR without Sensitivity Analysis**: Non-primary missingness exceeds $> 40\%$ or clinically plausible MNAR lacks a pattern-mixture / delta-adjustment sensitivity analysis per **Zero Silent Deletion & Missing-Data Routing**.
5. **Output Format Alignment Gate (Ask Immediately if Unspecified)**: When the user prompt omits a deliverable format, ask upfront before finalizing the report:
   - **(Recommended) Self-Contained HTML Report (`generate_html_report`, `.html`) with Dual / Toggle View**: When `plotly_figs` are supplied, interactive Plotly.js charts (`Interactive (Plotly)`) serve as the primary view with a toggle to **"Publication 300 DPI Figure (Static)"** for download and journal submission (offline CDN fallback and `@media print` parity); when `plotly_figs` are omitted, renders the static figure alongside ICMJE 3-rule tables and Methods/Results narratives.
   - **Manuscript Word Document (`.docx`)**: ICMJE 3-rule tables and 6.5 in 300 DPI figures (falls back to `.html` if `python-docx` is unavailable).
   - **Executive PowerPoint Presentation (`.pptx`)**: 16:9 widescreen slides (1 figure per slide with takeaway cards, when `python-pptx` is available).
   - **Publication PDF (`.pdf`)** / **Markdown (`.md`)** / **Quick In-Chat Summary Only**.

---

## 3. Completion Criteria

The biostatistical task is complete when every item below is verified against an artifact or tool output:
- [ ] **Pre-Flight Data Audit**: Every model variable's dtype, missingness count, distributional summary (Mean (SD) or Median [IQR]), numeric `0/1` (`0/1/2+`) coding, and **pre-flight** $\text{EPV}_{\text{binary}}$ or $\text{EPV}_{\text{Cox}}$ are recorded in sandbox output.
- [ ] **Grilling Gate Resolution**: Every triggered **Grilling Gate** condition (#1–#4) has an explicit user decision recorded before final model execution.
- [ ] **Deliverable Format Aligned**: Target deliverable format is specified in the prompt or confirmed via **Output Format Alignment Gate** (#5).
- [ ] **Model Convergence & Diagnostics**: The Python script exits with code `0`; every fitted model reports `converged == True` with finite estimates and 95% CIs; every applicable check (Schoenfeld residuals, Brant test, calibration slope/Brier/ICI, Austin SMD $< 0.10$, Kish $N_{\text{eff}}$, and **reconciled** CIF probability conservation) is computed; and monotone separation is resolved per **Grilling Gate #3**.
- [ ] **Reconciled Cohort Retention**: Every analysis step records a **reconciled** `retention_audit` dict.
- [ ] **Exhaustive Tables**: Every primary and sensitivity model (including complete-case alongside MICE for $5\%\text{--}40\%$ missingness) is formatted into 3-rule tables (NEJM/JAMA/APA via `Recipe 13`) with point estimates, 95% CIs, $P$-values, and units.
- [ ] **Methods & Retention Narrative**: Dedicated Statistical Methods and cohort retention flow sections are written in the report narrative.
- [ ] **Traceable Figures & Deliverable**: Every figure is saved at 300 DPI with **traceable** plot coordinates, and the deliverable file exists on disk (with Dual / Toggle View when `.html` with `plotly_figs` is selected).
- [ ] **Pillar 5 Integrity Audit Passed**: `verify_standalone_integrity(narrative_text, results_dict, table_dfs)` (`report-builder.md §3`) returns `passed == True` (`len(phi_violations) == 0 and has_causal_caveat and traceability_passed`) with every non-structural narrative number verified (`len(untraced_numbers) == 0`, or every remaining value in `untraced_numbers` manually verified as a non-statistical annotation so any unverified estimate, CI bound, or $P$-value blocks completion), gated by its 3 programmatic checks:
  1. *Zero-PHI Scan*: `len(phi_violations) == 0` across narrative text, table headers, and cells, plus manual review confirming zero unprefixed patient names before release.
  2. *Numerical Traceability*: `traceability_passed == True` with `len(untraced_numbers) == 0` (or every remaining unmatched value in `untraced_numbers` manually verified as a non-statistical token; any unverified estimate, CI bound, or $P$-value blocks completion), where every non-structural narrative number is **traceable** to `results_dict` within `abs_tol = min(0.02, 10**(-dec_digits))` for `|val| <= 1.0`, `0.02` absolute or `0.01` (1%) relative tolerance for `|val| > 1.0`, or `0.05` percentage-scale tolerance (`|val - kn * 100.0| <= 0.05`).
  3. *Observational Causal / E-value Check*: `has_causal_caveat == True` whenever `results_dict` keys match observational/PSM/Cox/logistic models.



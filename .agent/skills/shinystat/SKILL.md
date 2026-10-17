---
name: shinystat
description: "Use when analyzing clinical or EHR data: Table 1, logistic/ordinal/GEE regression, Firth sparse-data models, survival (Kaplan-Meier, Cox, Competing Risks CIF/Fine-Gray), diagnostics (2x2, ROC/DeLong, NRI/IDI, time-dependent ROC), prediction validation (TRIPOD+AI bootstrap shrinkage, calibration, DCA, recalibration), causal inference (PSM, IPTW), agreement (Bland-Altman, ICC), meta-analysis, sample size (Riley, Obuchowski), 300 DPI figures, or clinical reports (html, docx, pptx, md, pdf)."
---

# Shinystat: Autonomous Biostatistical Decision Agent

Execute clinical biostatistics through the 5 Decision Pillars below, exercising autonomous authority over statistical mechanics and halting at **The Deterministic Grilling Gate** (§2) only for clinical or protocol ambiguities and **Output Format Alignment Gate** (#5) confirmation.

---

## 1. The 5 Decision Pillars

1. **Pillar 1: Data Reality Inspection (Autonomous Profiling)**
   - Run **pre-flight** *(verified on raw data before fitting any model)* checks on data geometry, headers, dtypes, missingness, and distributions (`df.info()`, `df.describe()`), isolating analytic cohorts from embedded summaries, multi-line headers, and Thai/regional text.
   - Select descriptive summaries from distributional shape (Mean (SD) for symmetric vs Median [IQR] for skewed data); select inferential tests from the study design and target estimand rather than skewness alone (Mann–Whitney U tests rank-sum shifts rather than medians when group distribution shapes differ).
   - Compute **pre-flight** Events Per Variable ($\text{EPV}_{\text{binary}}$ or $\text{EPV}_{\text{Cox}}$) and route sparse data per **EPV & Sparse-Data Routing** (Pillar 3).

2. **Pillar 2: Research & Estimand Triangulation (Autonomous Selection)**
   - Align statistical design with PICO/PECO and clinical archetype (Cohort, Case-Control, RCT, Diagnostic Accuracy, Agreement, Clustered, Causal Inference, Competing Risks).
   - When mapping study archetypes or sparse-data decision trees to model families, read [decision-heuristics.md](references/decision-heuristics.md#1-clinical-study-archetype-to-statistical-model-mapping).

3. **Pillar 3: Adaptive Python Scripting (Autonomous Biostatistical Authority)**
   - Write standalone Python scripts using the bundled `medstat` engine in `scripts/medstat/` (CLI entry: `python <skill-dir>/scripts/medstat_cli.py`) or standard scientific libraries (`pandas`, `scipy.stats`, `statsmodels`, `lifelines`, `scikit-learn`).
   - **EPV & Sparse-Data Routing**:
     - Calculate $\text{EPV}_{\text{binary}} = \frac{\min(N_{\text{events}}, N_{\text{non-events}})}{P_{\text{parameters}}}$ for binary logistic and $\text{EPV}_{\text{Cox}} = \frac{N_{\text{failures}}}{P_{\text{parameters}}}$ for Cox PH, where $P_{\text{parameters}}$ counts fitted predictor parameters (multi-level factors as $k-1$, splines by degrees of freedom).
     - When monotone separation is present, halt at **Grilling Gate #3**.
     - When $\text{EPV}_{\text{binary}} < 10$ without separation, branch by estimand: for etiological effects, apply Firth penalized logistic regression (`medstat.models.firth.fit_firth_logistic`); for prediction models, apply L2 penalization/shrinkage per [decision-heuristics.md §2.A](references/decision-heuristics.md#a-events-per-variable-epv--sparse-data).
     - When $\text{EPV}_{\text{Cox}} < 10$ without separation, apply Firth penalized Cox (`medstat.models.firth.fit_firth_cox`) or clinical risk score reduction per [decision-heuristics.md §2.A](references/decision-heuristics.md#a-events-per-variable-epv--sparse-data).
   - **Autonomous Upgrades & Diagnostics**:
     - *Diagnostic Reclassification*: When comparing updated vs baseline risk models, compute Continuous and Categorical NRI ($K \times K$ contingency matrices, reporting event and non-event NRIs separately), IDI, and paired DeLong $\Delta\text{AUROC}$ (`medstat.diagnostic.roc.compute_reclassification_metrics`, `compare_diagnostic_models`).
     - *Time-Dependent ROC*: For censored longitudinal outcomes, evaluate discrimination trajectories across milestone horizons via Cumulative/Dynamic IPCW $\text{AUROC}(t)$ and Youden cutpoints (`medstat.diagnostic.roc.compute_time_dependent_roc`).
     - *TRIPOD+AI Internal Validation & Shrinkage*: For prediction model development, run optimism-corrected bootstrap validation across $B \ge 200$ resamples (`medstat.diagnostic.validation.bootstrap_internal_validation`), computing optimism ($\text{boot} - \text{test}$), corrected metrics ($\text{apparent} - \text{optimism}$), uniform shrinkage factor $\hat{s}$, and re-estimated intercept $\alpha_0$ preserving sample event prevalence.
     - *Prediction Model Recalibration*: When validating or updating a prediction model in a new cohort, perform intercept-only updating (calibration-in-the-large) and logistic recalibration (`medstat.diagnostic.validation.recalibrate_prediction_model`).
     - *Prediction & Diagnostic Sample Size*: Size prediction models across all 4 criteria of Riley et al. (BMJ 2020) and prospective diagnostic accuracy studies via Obuchowski / Hanley-McNeil (`medstat.power.sample_size.calculate_riley_sample_size`, `calculate_diagnostic_auroc_sample_size`).
     - *Proportional Hazards & Non-Linearity*: Test Grambsch-Therneau Schoenfeld residuals (`cox_ph.check_assumptions()`) and apply stratification or time-varying covariates if violated; test continuous predictor linearity and fit Restricted Cubic Splines (`medstat.models.splines`) when non-linear.
     - *Competing Risks*: When competing events preclude the primary event, compute non-parametric Aalen-Johansen Cumulative Incidence Functions (`medstat.models.survival.fit_competing_risks_cif`) with a Naive $1 - \text{Kaplan-Meier}$ overestimation audit and **reconciled** *(cohort counts satisfy $N_{\text{initial}} = N_{\text{excluded}} + N_{\text{analyzed}}$ with itemized exclusion counts summing to $N_{\text{excluded}}$, and competing-risk probabilities satisfy $\sum_k \text{CIF}_k(t) + \hat{S}(t) \approx 1.0$)* probability conservation. Fit Cause-Specific Cox (csHR) for etiological questions and Fine-Gray IPCW regression (`fit_competing_risks_regression`) for prognostic questions; report Fine-Gray estimates as Subdistribution Hazard Ratios (SHR), never as relative risks (RR) or risk ratios.
     - *Causal Inference, Clustering & Sensitivity*: For observational comparisons, execute 1:1 PSM (caliper $0.2\text{ SD}$, `medstat.causal.psm`) or IPTW (`medstat.causal.iptw`) supporting ATE, ATT, ATC, and Overlap Weights (ATO; Li et al., 2018), stabilized weights, quantile truncation, Kish's Effective Sample Size ($N_{\text{eff}}$), Bessel-corrected weighted SMD, and Huber-White sandwich SEs, plus VanderWeele & Ding E-values (`medstat.models.sensitivity`). When multi-center clustering is present, compute Design Effect ($\text{DEFF}$) and fit population-averaged GEE with robust sandwich SEs or mixed models (`medstat.models.multilevel`).

4. **Pillar 4: Biostatistical Safety Invariants**
   - **Strict Numeric 0/1 Encoding**: Encode binary endpoints and survival events as numeric `0` and `1` (`1 = Event`, `0 = Non-event / Censored`) and competing-risk events as integers (`0 = Censored, 1 = Event of Interest, 2+ = Competing Event`), mapping text labels (`"Yes"/"No"`, `"Dead"/"Alive"`) explicitly during **pre-flight** cleaning.
   - **Sample Retention Flow**: Track **reconciled** cohort retention ($N_{\text{initial}} \longrightarrow N_{\text{excluded}} \longrightarrow N_{\text{analyzed}}$) via `SampleFlowTracker` (`medstat.data.retention`).
   - **Zero Silent Deletion & Missing-Data Routing**:
     - Run **pre-flight** missingness evaluation (Little's MCAR test via `medstat.data.clean`). Analyze only observed primary outcomes and never impute primary outcomes; if primary outcome missingness lacks a prespecified protocol disposition, halt at **Grilling Gate #1**.
     - *$< 5\%$ cohort loss*: When complete-case cohort loss is $< 5\%$ across the planned model and non-primary covariate missingness is verified low and clinically uninformative, execute complete-case analysis with documented sensitivity notes.
     - *$5\% - 40\%$ missingness*: Document the clinical rationale supporting Missing at Random (MAR) and execute MICE (`medstat.data.missing`) with stochastic parameter draws to preserve between-imputation variance. Report Complete-Case Analysis alongside MICE as a benchmark sensitivity analysis, noting that agreement between MICE and complete-case results does not establish MAR or exclude MNAR bias.
     - *$> 40\%$ non-primary missingness (not established as MNAR)*: Halt at **Grilling Gate #4** to select between (a) excluding the covariate if auxiliary to protect cohort sample size, (b) high-$M$ MICE ($M \ge 40$, monitoring Fraction of Missing Information [FMI]) if clinically essential under plausible MAR, or (c) missing-indicator / pattern-mixture sensitivity modeling.
     - *Clinically plausible MNAR*: Perform an MNAR sensitivity analysis (pattern-mixture or delta-adjustment); if no MNAR sensitivity analysis is performed, halt at **Grilling Gate #4** before presenting pooled estimates (stating the limitation alone is insufficient).
   - **Standard Confidence Intervals**: Report Wilson score 95% CIs for proportions and 2x2 diagnostics, DeLong variance CIs for AUROC, non-null large-sample CIs for agreement metrics (exact F-distribution CIs for ICC), and profile likelihood CIs for Firth models.

5. **Pillar 5: Reporting Integrity & Multi-Format Documents**
   - **Report Intermediate Representation (IR)**: Assemble narrative blocks, 3-rule tables, and figures into `ReportDocument` (`medstat.reporting.ir`) so every reported value is **traceable** *(every narrative and table number matches `results_dict` within programmatic tolerance and is backed by exported source data)*.
   - **Publication Figures (300 DPI)**: When generating 300 DPI colorblind-safe plots (Okabe-Ito/Tol palettes, Thai font fallbacks `Sarabun`, `Thonburi`, `Sukhumvit Set`) or interactive HTML dual plots (`medstat.figures.html`), read [figures.md](references/figures.md).
   - **Multi-Format Renderers**: When rendering `.html`, `.docx`, `.pptx`, `.md`, or `.pdf` deliverables after **Output Format Alignment Gate** (#5) confirmation, read [report-builder.md](references/report-builder.md#2-multi-format-renderers).
   - **Reporting Integrity Verification**: Audit every `ReportDocument` via `verify_report_integrity` (`medstat.reporting.integrity`) to confirm **traceable** narrative numbers, causal/E-value caveats, methods/retention documentation, and **Zero-PHI** compliance (automated regex scan for `HN`, 13-digit Thai IDs, phone numbers, DOB, and honorific-prefixed names `Mr.`, `Mrs.`, `Ms.`, `Dr.`, `นาย`, `นาง`, `นางสาว`, `นพ.`, `พญ.`, `ดร.`, plus mandatory manual review for unprefixed names before external release; never release unscreened patient identifiers).

---

## 2. The Deterministic Grilling Gate

### Autonomous Execution (Decide, Execute, and Document)
- Decide, execute, and document routine biostatistical mechanics autonomously: parametric vs non-parametric tests, Mean (SD) vs Median [IQR] reporting, proportional hazards and parallel-slopes tests, low-EPV penalization (**EPV & Sparse-Data Routing**), $< 40\%$ MAR missing-data handling (**Zero Silent Deletion & Missing-Data Routing**), spline knot counts, Youden cutpoints, and NEJM/JAMA/APA table styling.

### Situations Requiring Mandatory Grilling / Inquiry
Halt **pre-flight** (or upon diagnostic detection) and ask the user directly when any of these 5 conditions holds:
1. **Outcome, Event Direction, or Missing Outcome Ambiguity**: Primary event definition, censoring code, biomarker abnormality direction (low-is-abnormal like eGFR/Platelets vs high-is-abnormal like Troponin/Lactate), or missing primary outcome protocol disposition is unspecified.
2. **Proposal vs Data Geometry Mismatch**: Time-to-event survival is requested when follow-up duration or censoring status is absent in the dataset, or clustered modeling is requested without cluster identifiers.
3. **Severe Sparse Data & Monotone Separation**: Monotone separation occurs in multivariable logistic or Cox regression; confirm between Firth penalized estimation (`medstat.models.firth`) versus domain-guided covariate reduction.
4. **High Missingness ($> 40\%$) or Plausible MNAR without Sensitivity Analysis**: Non-primary missingness exceeds $> 40\%$ or clinically plausible MNAR lacks a pattern-mixture / delta-adjustment sensitivity analysis per **Zero Silent Deletion & Missing-Data Routing**.
5. **Output Format Alignment Gate (Ask Immediately if Unspecified)**: When the user prompt does not specify a deliverable format, ask the user upfront (via `ask_question` or interactive prompt) to select their deliverable:
   - **(Recommended) Interactive HTML Report (`.html`) with Dual / Toggle View** (`render_html`): When Plotly specs (`plotly_figs`) are supplied, interactive Plotly.js charts (`Interactive (Plotly)`) serve as the primary view (hover cutpoints, zoom timelines, inspect calibration deciles) with a tab/toggle button to switch to **"Publication 300 DPI Figure (Static)"** for downloading and journal submission (with offline CDN fallback and `@media print` parity); when `plotly_figs` are omitted, renders the static publication figure only, alongside ICMJE 3-rule tables and Methods & Results narratives.
   - **Manuscript Word Document (`.docx`)**: ICMJE 3-rule OpenXML table borders (`<w:tblBorders>`) and 6.5 in 300 DPI figures.
   - **Executive PowerPoint Presentation (`.pptx`)**: 16:9 widescreen slides (`13.333" x 7.5"`), 1 figure per slide with takeaway callout cards.
   - **Publication PDF (`.pdf`)** / **Markdown (`.md`)**.
   - **Quick In-Chat Summary Only**: Concise tables and narrative directly in chat.

---

## 3. Completion Criteria

The biostatistical task is complete when every item below is verified against an artifact or tool output:
- [ ] **Pre-Flight Data Audit**: Every model variable's dtype, missingness count, distributional summary (Mean (SD) or Median [IQR]), numeric `0/1` (or competing-risk `0/1/2+`) coding, and **pre-flight** $\text{EPV}_{\text{binary}}$ or $\text{EPV}_{\text{Cox}}$ are recorded in script output.
- [ ] **Grilling Gate Resolution**: Every triggered **Grilling Gate** condition (#1–#4) has an explicit user decision recorded before final model execution.
- [ ] **Deliverable Format Aligned**: Target deliverable format is specified in the user prompt or confirmed via **Output Format Alignment Gate** (#5).
- [ ] **Model Convergence & Diagnostics**: The Python analysis script exits with code `0`; every fitted model reports `converged == True` with finite point estimates and 95% CIs; every applicable assumption check (Schoenfeld residuals, Brant test, calibration slope/Brier/ICI, Austin SMD $< 0.10$, Kish $N_{\text{eff}}$, and **reconciled** CIF probability conservation $\sum_k \text{CIF}_k(t) + \hat{S}(t) \approx 1.0$) is computed; and any monotone separation is resolved per **Grilling Gate #3**.
- [ ] **Reconciled Cohort Retention**: `SampleFlowTracker` (`medstat.data.retention`) records **reconciled** counts where `N_initial = N_excluded + N_analyzed` holds numerically and itemized exclusion reasons sum exactly to `N_excluded`.
- [ ] **Exhaustive Tables**: Every primary and sensitivity model (including complete-case alongside MICE when missingness is $5\%\text{--}40\%$) is formatted into 3-rule tables (NEJM/JAMA/APA) with point estimates, 95% CIs, $P$-values, and clinical units.
- [ ] **Traceable Figures & Deliverable**: Every figure is exported at 300 DPI as a `FigureResult` with non-empty `png_path` and **traceable** `csv_path` source data, and the requested deliverable file exists on disk (rendering Dual / Toggle View when `.html` with `plotly_figs` is selected).
- [ ] **Pillar 5 Integrity Audit Passed**: `verify_report_integrity(doc)` (`medstat.reporting.integrity`) returns `IntegrityReport.passed == True` (`len(phi_violations) == 0 and has_causal_caveat and (len(untraced) == 0 or len(untraced) <= max(1, int(len(text_numbers) * 0.15)))`) across all 4 checks:
  1. *Zero-PHI Scan*: `len(phi_violations) == 0` across narrative text and `TableBlock` headers/cells, plus manual review confirming zero unprefixed patient names before external release.
  2. *Numerical Traceability*: Every non-structural narrative number is **traceable** to `doc.results_dict` within `abs_tol = min(0.02, 10**(-dec_digits))` for `|val| <= 1.0`, `0.02` absolute or `0.01` (1%) relative tolerance for `|val| > 1.0`, or percentage-scale tolerance `min(0.05, abs_tol * 100.0)` when `%` is present (`0.05` otherwise).
  3. *Observational Causal Inference & E-value Check*: `has_causal_caveat == True` whenever `results_dict` keys match observational/regression/PSM/Cox models.
  4. *Methods & Retention Check*: `has_methods == True` and `has_retention_flow == True` (`warnings == []`).


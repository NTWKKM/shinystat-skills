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
   - **Cloud Sandbox Protocol**: Write self-contained Python scripts using standard libraries (`pandas`, `numpy`, `scipy.stats`, `statsmodels`, `lifelines`, `scikit-learn`); when selecting standalone implementations, read [python-recipes.md](references/python-recipes.md#recipe-index).
   - **EPV & Sparse-Data Routing**:
     - Compute $\text{EPV}_{\text{binary}} = \min(N_{\text{events}}, N_{\text{non-events}}) / P_{\text{parameters}}$ and $\text{EPV}_{\text{Cox}} = N_{\text{failures}} / P_{\text{parameters}}$ ($P_{\text{parameters}}$ counts $k-1$ factor levels and spline df).
     - When monotone separation is present, halt at **Grilling Gate #3**.
     - When $\text{EPV}_{\text{binary}} \ge 10$ without separation, read [Recipe 10](references/python-recipes.md#10-multivariable-binary-logistic-regression-table) for multivariable logistic tables. When $\text{EPV}_{\text{binary}} < 10$ without separation, branch by estimand: read [Recipe 7](references/python-recipes.md#7-firths-penalized-likelihood-logistic-regression) (Firth logistic with profile likelihood CIs) for etiological ORs, or apply L2 shrinkage for prediction models.
     - When $\text{EPV}_{\text{Cox}} < 10$ without separation, apply L2-penalized Cox (`penalizer=0.1` in `Recipe 11`) or clinical risk score reduction.
   - **Branch-to-Recipe Routing**:
     - *Tables*: When building stratified Table 1 (Austin SMD, zero-variance guard), read [Recipe 1](references/python-recipes.md#1-table-1-baseline-characteristics--standardized-mean-difference-smd); when rendering NEJM/JAMA 3-rule HTML tables, read [Recipe 13](references/python-recipes.md#13-publication-grade-html-table-renderer-nejm--jama-standards).
     - *Diagnostics & ROC*: When computing 2x2 accuracy (Wilson & log-scale CIs, Haldane-Anscombe correction), read [Recipe 2](references/python-recipes.md#2-2x2-diagnostic-accuracy-with-wilson-score--log-scale-95-cis); when computing empirical ROC, Youden cutpoints, or single/paired DeLong CIs, read [Recipe 3](references/python-recipes.md#3-empirical-roc-optimal-cutoff--single--paired-delong-95-cis); when computing Continuous/Categorical NRI (separate event and non-event NRIs) and IDI, read [Recipe 16](references/python-recipes.md#16-diagnostic-risk-reclassification-continuous--categorical-nri-and-idi); when computing longitudinal Cumulative/Dynamic IPCW $\text{AUROC}(t)$, read [Recipe 17](references/python-recipes.md#17-time-dependent-roc-cumulativedynamic-ipcw-estimator-auroct).
     - *Calibration, DCA, Validation & Recalibration*: When computing Brier score, calibration slope/intercept, Austin–Steyerberg ICI, or Vickers DCA net benefit, read [Recipe 4](references/python-recipes.md#4-model-calibration--vickers-decision-curve-analysis-dca); when running TRIPOD+AI bootstrap internal validation ($B \ge 200$, uniform shrinkage $\hat{s}$, re-estimated intercept $\alpha_0$), read [Recipe 18](references/python-recipes.md#18-tripodai-optimism-corrected-bootstrap-internal-validation--uniform-shrinkage); when recalibrating a prediction model in a new cohort (intercept-only or logistic updating), read [Recipe 19](references/python-recipes.md#19-prediction-model-recalibration-and-updating-logistic--intercept-only).
     - *Survival & Competing Risks*: When fitting KM, log-rank, or Cox PH with Schoenfeld residual checks (stratifying or adding time-varying covariates on PH violation), read [Recipe 11](references/python-recipes.md#11-survival-analysis-kaplan-meier-log-rank--cox-proportional-hazards). When competing events exist, read [Recipe 15](references/python-recipes.md#15-competing-risks-survival-analysis-aalen-johansen-cif--fine-gray-model) for Aalen-Johansen CIF (with Naive $1-\text{KM}$ bias audit and **reconciled** *(cohort counts satisfy $N_{\text{initial}} = N_{\text{excluded}} + N_{\text{analyzed}}$ with itemized exclusion counts summing to $N_{\text{excluded}}$, and competing-risk probabilities satisfy $\sum_k \text{CIF}_k(t) + \hat{S}(t) \approx 1.0$)* probability conservation), Cause-Specific Cox (csHR, etiological), and Fine-Gray IPCW regression (prognostic); report Fine-Gray estimates as Subdistribution Hazard Ratios (SHR), never as relative risks (RR) or risk ratios.
     - *Causal Inference, Clustering & Non-Linearity*: When running 1:1 PSM (caliper $0.2\text{ SD}$, pair IDs, balance table), read [Recipe 6](references/python-recipes.md#6-propensity-score-matching-psm-with-pair-ids--covariate-balance); when running IPTW (ATE, ATT, ATC, ATO overlap weights, stabilized/truncated weights, Kish $N_{\text{eff}}$, Bessel-corrected weighted SMD, Huber-White robust SEs), read [Recipe 14](references/python-recipes.md#14-inverse-probability-of-treatment-weighting-iptw--weighted-balance); when computing VanderWeele & Ding E-values, read [Recipe 9](references/python-recipes.md#9-vanderweele--ding-2017-sensitivity-e-value). Fit splines for non-linear continuous predictors and `statsmodels` GEE (robust sandwich SEs, $\text{DEFF}$) for clustered cohorts.
     - *Agreement & Sample Size*: When computing Bland-Altman LoA (1999 SEs) or Shrout & Fleiss (1979) ANOVA ICC (exact F CIs), read [Recipe 5](references/python-recipes.md#5-observer-agreement-bland-altman--shrout-fleiss-1979-icc); when sizing prediction models (Riley et al. 2020 4 criteria) or diagnostic AUROC studies (Obuchowski 1998), read [Recipe 20](references/python-recipes.md#20-sample-size-determination-for-clinical-prediction-models-riley-et-al-2020--diagnostic-auroc-obuchowski-1998).

4. **Pillar 4: Biostatistical Safety Invariants**
   - **Strict Numeric 0/1 Encoding**: Encode binary endpoints and survival events as numeric `0` and `1` (`1 = Event`, `0 = Non-event / Censored`) and competing-risk events as integers (`0 = Censored, 1 = Event of Interest, 2+ = Competing Event`), mapping text labels (`"Yes"/"No"`, `"Dead"/"Alive"`) during **pre-flight** cleaning.
   - **Sample Retention Flow**: Track **reconciled** cohort retention ($N_{\text{initial}} \longrightarrow N_{\text{excluded}} \longrightarrow N_{\text{analyzed}}$) with documented exclusion reasons across every step.
   - **Zero Silent Deletion & Missing-Data Routing**:
     - Run **pre-flight** Little's MCAR test via [Recipe 8](references/python-recipes.md#8-littles-1988-mcar-multivariate-missingness-test) ($H_0: \text{MCAR}$; $p \ge 0.05$ does not prove MCAR or establish MAR). Analyze only observed primary outcomes and never impute primary outcomes; if missing primary outcomes lack a protocol disposition, halt at **Grilling Gate #1**.
     - *$< 5\%$ cohort loss*: When complete-case loss is $< 5\%$ across the planned model and non-primary missingness is clinically uninformative, run complete-case analysis with sensitivity notes.
     - *$5\% - 40\%$ missingness*: Document MAR rationale and run MICE via [Recipe 12](references/python-recipes.md#12-multiple-imputation-by-chained-equations-mice--rubins-rules) (Bayesian posterior draws $\sigma^{*2} \sim \text{Inv-}\chi^2, \beta^* \sim N(\hat{\beta}, \sigma^{*2}(X^TX)^{-1})$ with Rubin's rules; $M \ge 5$ for $\le 30\%$ missingness, $M \ge 20$ for $> 30\%$; Gaussian MICE applies to continuous targets with observed dummy predictors, while categorical targets require type-specific imputation). Always report Complete-Case Analysis alongside MICE as a sensitivity benchmark (agreement between MICE and complete-case does not establish MAR or exclude MNAR bias).
     - *$> 40\%$ non-primary missingness (not MNAR)*: Halt at **Grilling Gate #4** to choose (a) dropping the auxiliary covariate, (b) high-$M$ MICE ($M \ge 40$, tracking FMI) under plausible MAR, or (c) missing-indicator / pattern-mixture modeling.
     - *Clinically plausible MNAR*: Run an MNAR sensitivity analysis (pattern-mixture or delta-adjustment); if none is performed, halt at **Grilling Gate #4** before reporting pooled estimates (stating the limitation alone is insufficient).
   - **Standard Confidence Intervals**: Report Wilson score 95% CIs for proportions/2x2 metrics, DeLong CIs for AUROC, exact F CIs for ICC, and profile likelihood CIs for Firth models.

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
1. **Outcome, Event Direction, or Missing Outcome Ambiguity**: Primary event definition, censoring code, biomarker direction (low-is-abnormal eGFR/Platelets vs high-is-abnormal Troponin/Lactate), or missing primary outcome disposition is unspecified.
2. **Proposal vs Data Geometry Mismatch**: Survival modeling is requested without follow-up time or censoring status, or clustered modeling without cluster IDs.
3. **Severe Sparse Data & Monotone Separation**: Monotone separation occurs in logistic or Cox regression; confirm between Firth/penalized estimation (`Recipe 7` or `penalizer=0.1` in `Recipe 11`) vs domain-guided covariate reduction.
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
- [ ] **Model Convergence & Diagnostics**: The Python script exits with code `0`; every fitted model reports `converged == True` with finite estimates and 95% CIs; every applicable check (Schoenfeld residuals, Brant test, calibration slope/Brier/ICI, Austin SMD $< 0.10$, Kish $N_{\text{eff}}$, and **reconciled** CIF conservation $\sum_k \text{CIF}_k(t) + \hat{S}(t) \approx 1.0$) is computed; and monotone separation is resolved per **Grilling Gate #3**.
- [ ] **Reconciled Cohort Retention**: Every analysis step records a **reconciled** `retention_audit` dict where `N_initial = N_excluded + N_analyzed` holds numerically and itemized exclusion reasons sum to `N_excluded`.
- [ ] **Exhaustive Tables**: Every primary and sensitivity model (including complete-case alongside MICE for $5\%\text{--}40\%$ missingness) is formatted into 3-rule tables (NEJM/JAMA/APA via `Recipe 13`) with point estimates, 95% CIs, $P$-values, and units.
- [ ] **Traceable Figures & Deliverable**: Every figure is saved at 300 DPI with **traceable** plot coordinates, and the deliverable file exists on disk (with Dual / Toggle View when `.html` with `plotly_figs` is selected).
- [ ] **Pillar 5 Integrity Audit Passed**: `verify_standalone_integrity(narrative_text, results_dict, table_dfs)` (`report-builder.md §3`) returns `passed == True` (`len(phi_violations) == 0 and has_causal_caveat and traceability_passed`), alongside Methods & Retention narrative sections:
  1. *Zero-PHI Scan*: `len(phi_violations) == 0` across narrative text, table headers, and cells, plus manual review confirming zero unprefixed patient names before release.
  2. *Numerical Traceability*: `traceability_passed == True` (`len(untraced_numbers) <= 1 or len(untraced_numbers) / max(len(raw_matches), 1) <= 0.15`), where every non-structural narrative number is **traceable** to `results_dict` within `abs_tol = min(0.02, 10**(-dec_digits))` for `|val| <= 1.0`, `0.02` absolute or `0.01` (1%) relative tolerance for `|val| > 1.0`, or `0.05` percentage-scale tolerance (`|val - kn * 100.0| <= 0.05`).
  3. *Observational Causal / E-value Check*: `has_causal_caveat == True` whenever `results_dict` keys match observational/PSM/Cox/logistic models.
  4. *Methods & Retention Documentation*: Dedicated Methods/Statistical Analysis and cohort retention flow sections are present in the report narrative (mirroring `verify_report_integrity` checks 1–4 in `src/medstat/reporting/integrity.py`).



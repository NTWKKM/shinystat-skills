# Biostatistical Decision Heuristics & Reference Manual (Cloud Sandbox Edition)

Reference guide for the `shinystat-cloud` autonomous agent in cloud code-execution environments (e.g. Claude Web Analysis Tool). Consult this manual during Pillar 2 (Research & Estimand Triangulation) to map clinical research questions to optimal biostatistical models and standalone recipes in [python-recipes.md](python-recipes.md).

---

## 1. Clinical Study Archetype to Statistical Model Mapping

| Study Archetype | Target Estimand / Goal | Primary Model / Method | Key Assumption / Diagnostics | Recommended Recipe / Library |
| :--- | :--- | :--- | :--- | :--- |
| **Type 1: Baseline Cohort** | Cohort characteristics | Table 1 (Mean/SD, Median/IQR, Count/%, SMD) | Normality inspection; Fisher exact fallback for small cells | [Recipe 1: Table 1 & SMD](python-recipes.md#1-table-1-baseline-characteristics--standardized-mean-difference-smd) |
| **Type 2: Binary Outcome** | Odds Ratio (OR), Risk factors | Multivariable Binary Logistic Regression | Linearity of log-odds; Crude & Adjusted OR table | [Recipe 10: Logistic Regression Table](python-recipes.md#10-multivariable-binary-logistic-regression-table) |
| **Type 2-Sparse: Separation** | Unbiased OR in rare events | Firth Penalized Logistic Regression | Profile likelihood CIs; resolves monotone separation | [Recipe 7: Firth Logistic Regression](python-recipes.md#7-firths-penalized-likelihood-logistic-regression) |
| **Type 2b: Ordinal Scale** | Cumulative Odds Ratio | Cumulative Logit Proportional Odds | Brant test for parallel lines; OrderedModel fallback | `statsmodels.miscmodels.ordinal_model.OrderedModel` |
| **Type 3: Survival / Time-to-Event** | Hazard Ratio (HR), Median survival | Kaplan-Meier, Log-Rank & Cox Proportional Hazards | Proportional hazards test (Schoenfeld residuals); L2 shrinkage | [Recipe 11: Survival Analysis Suite](python-recipes.md#11-survival-analysis-kaplan-meier-log-rank--cox-proportional-hazards) |
| **Type 3b: Competing Risks Survival** | Subdistribution Hazard (SHR) & Cause-Specific HR (csHR) | Aalen-Johansen CIF & Fine-Gray Model (IPCW) | Competing risk probability conservation; Naive KM overestimation audit | [Recipe 15: Competing Risks Survival](python-recipes.md#15-competing-risks-survival-analysis-aalen-johansen-cif--fine-gray-model) |
| **Type 4: Diagnostic Test Accuracy** | Sens, Spec, PPV, NPV, AUC, LRs | 2x2 Contingency Matrix & Empirical ROC | Wilson score CIs; Haldane correction; Single/Paired DeLong | [Recipe 2: 2x2 Diagnostics](python-recipes.md#2-2x2-diagnostic-accuracy-with-wilson-score--log-scale-95-cis) & [Recipe 3: ROC & DeLong](python-recipes.md#3-empirical-roc-optimal-cutoff--single--paired-delong-95-cis) |
| **Type 4b: Clinical Utility & Calib** | Net Benefit, Model Calibration | Decision Curve Analysis (DCA) & Recalibration | Brier score; Calibration slope & intercept; Austin-Steyerberg ICI | [Recipe 4: Calibration & DCA](python-recipes.md#4-model-calibration--vickers-decision-curve-analysis-dca) |
| **Type 4c: Internal Validation & Shrinkage** | Optimism-corrected C-statistic, Brier, R2, Slope | TRIPOD+AI Bootstrap Internal Validation | B >= 200 resamples; uniform shrinkage factor s_hat, re-estimated intercept | [Recipe 18: Internal Validation & Shrinkage](python-recipes.md#18-tripodai-optimism-corrected-bootstrap-internal-validation--uniform-shrinkage) |
| **Type 4d: Model Recalibration & Updating** | Target Population Calibration, Intercept, Slope | Intercept-only & Logistic Recalibration | Alpha and Beta recalibration parameters, ICI, Emax, AUROC preserved | [Recipe 19: Model Recalibration](python-recipes.md#19-prediction-model-recalibration-and-updating-logistic--intercept-only) |
| **Type 5: Observational Causal (PSM)** | Average Treatment Effect in Treated (ATT) | Propensity Score Matching (PSM, 1:1, Caliper 0.2 SD) | Austin (2009) balance (SMD < 0.10); Pair ID tracking; E-value | [Recipe 6: PSM & Balance](python-recipes.md#6-propensity-score-matching-psm-with-pair-ids--covariate-balance) & [Recipe 9: E-Value](python-recipes.md#9-vanderweele--ding-2017-sensitivity-e-value) |
| **Type 5b: Causal Weighting (IPTW)** | Population Effect (ATE, ATT, ATC, Overlap ATO) | Inverse Probability Treatment Weighting (IPTW) | Kish's Effective Sample Size ($N_{\text{eff}} \ge 0.5 \times N$), Positivity (PS 0.05–0.95), Robust SE | [Recipe 14: IPTW & Balance](python-recipes.md#14-inverse-probability-of-treatment-weighting-iptw--weighted-balance) |
| **Type 6: Device / Rater Agreement** | Limits of Agreement, Reliability | Bland-Altman Difference Analysis & ICC (1–3k) | Bland-Altman 1999 large-sample CIs; Shrout-Fleiss exact F CIs | [Recipe 5: Bland-Altman & ICC](python-recipes.md#5-observer-agreement-bland-altman--shrout-fleiss-1979-icc) |
| **Type 7: Systematic Meta-Analysis** | Pooled Effect (RR/OR/MD) | Inverse-Variance Fixed-Effects or DerSimonian-Laird | Condition on clinical variation; small $k < 5$ caution | Pure SciPy meta-analysis or `statsmodels` |
| **Type 8: Clustered / Multi-Center** | Population-averaged effect | Generalized Estimating Equations (GEE) | Cluster Design Effect ($\text{DEFF}$); Robust sandwich SEs | `statsmodels.genmod.generalized_estimating_equations.GEE` |
| **Type 9: Sample Size Determination** | Required N, Events, EPP, Statistical Power | Riley et al. (BMJ 2020) & Obuchowski (1998) | 4 criteria (intercept, MAPE, shrinkage, optimism); Obuchowski AUROC variance | [Recipe 20: Sample Size Determination](python-recipes.md#20-sample-size-determination-for-clinical-prediction-models-riley-et-al-2020--diagnostic-auroc-obuchowski-1998) |

---

## 2. Decision Trees for Common Statistical Challenges

### A. Events Per Variable (EPV), Sparse Data & Separation
```
Calculate diagnostic EPV:
- Binary Logistic: EPV = min(Events, Non-Events) / P_parameters (counting multi-level factors as k-1)
- Cox Proportional Hazards: EPV = Observed Failures / P_parameters

├── EPV ≥ 10 & No Separation ──▶ Standard Multivariable Logistic (Recipe 10) / Standard Unpenalized Cox PH (Recipe 11 with default `penalizer=0.0`)
├── Monotone Separation Detected ──▶ Halt at Grilling Gate for clinical confirmation:
│   ├── Binary Logistic: (A) Firth penalized likelihood with profile likelihood CIs (Recipe 7), or (B) Remove/collapse separating covariate
│   └── Cox Proportional Hazards (Monotone Likelihood): (A) Scale-aware, data-validated L2-penalized Cox regression via lifelines (Recipe 11: standardize continuous predictors and report the transformation, or select and validate `penalizer > 0` on the supported data), or (B) Remove/collapse separating covariate
└── Low EPV (< 10) without Separation
    ├── Binary Logistic:
    │   ├── Strategy 1 (Etiological Effect): Firth penalized logistic regression with profile likelihood CIs (Recipe 7)
    │   ├── Strategy 2 (Prediction Model): L2 Ridge penalization / shrinkage to prevent overfitting (Riley et al., 2019)
    │   └── Strategy 3: Domain-guided clinical composite score or univariable reporting with sparse-data disclaimer
    └── Cox Proportional Hazards:
        ├── Strategy 1: Scale-aware, data-validated L2-penalized Cox regression via lifelines (Recipe 11: standardize continuous predictors and report the transformation, or select and validate `penalizer > 0` on the supported data)
        ├── Strategy 2: Pre-specified clinical risk index / score reduction
        └── Strategy 3: Univariable survival analysis (Kaplan-Meier / univariable Cox) with sparse-data disclaimer
```

*Note on Modern EPV Evidence:* As demonstrated by Vittinghoff & McCulloch (2007) and van Smeden et al. (2016), EPV < 10 is an exploratory diagnostic indicator rather than an unconditional hard stop. Firth penalization resolves monotone separation and reduces first-order finite-sample parameter estimation bias (Heinze & Schemper, 2002); however, for prediction modeling, shrinkage/penalization is required to control overfitting.

### B. Missing Data Strategy
```
Audit Missingness Pattern & Mechanism (Recipe 8: Little's MCAR Test; note H0: MCAR; failing to reject does not prove MCAR)

├── Per-variable missingness < 5% in non-primary covariates ──▶ Complete-case analysis with documented justification
├── Missing 5%–40% under Missing at Random (MAR) ──▶ Multiple Imputation by Chained Equations (Recipe 12: MICE; proper Bayesian draws, M ≥ 5 for ≤ 30% missing, M ≥ 20 for > 30% missing; always run Complete-Case Analysis as sensitivity analysis, noting agreement does not prove MAR or exclude MNAR bias)
├── Non-Primary Missing > 40% (Not MNAR) ──▶ Halt at Grilling Gate: (a) drop auxiliary covariate, (b) high-M MICE (M ≥ 40, tracking FMI) under MAR, or (c) pattern-mixture modeling (or missing-indicator adjustment restricted to randomized-trial baseline covariates or explicitly labeled sensitivity analyses)
├── Primary Outcome Missing ──▶ Never impute outcome; preserve true sample retention flow (N_initial ➔ N_analyzed)
└── Missingness Informative / MNAR ──▶ Halt for clinical domain review or sensitivity bounds
```

### C. Diagnostic Test Directionality & Cutpoint Selection
```
Identify Clinical Biomarker Mechanism

├── High biomarker indicates disease (e.g. Troponin, Lactate) ──▶ Direction: 'high' (Score ≥ Cutoff)
├── Low biomarker indicates disease (e.g. eGFR, Platelets) ──▶ Direction: 'low' (Score ≤ Cutoff; negate score for ROC)
└── Prespecified vs Exploratory Cutpoint
    ├── Prespecified clinical threshold ──▶ Report nominal 2x2 metrics (Recipe 2)
    └── Data-driven cutpoint (Youden's J) ──▶ Report with anti-p-hacking disclosure (Recipe 3)
```

### D. Observational Causal Inference: PSM vs. IPTW vs. Overlap Weights
```
Evaluate Treatment Assignment & Causal Target
├── Target Population
│   ├── Target = Treated Population (ATT) ──▶ 1:1 Nearest-Neighbor PSM with caliper 0.2 SD (Recipe 6) OR IPTW (estimand='ATT', Recipe 14)
│   ├── Target = Whole Target Population (ATE) ──▶ IPTW (estimand='ATE') with stabilized weights (Recipe 14)
│   └── Significant Propensity Truncation / Lack of Overlap ──▶ Overlap Weights (estimand='ATO', Li et al. 2018, Recipe 14)
├── Weight Stability & Positivity Audit
│   ├── Positivity Alert (PS < 0.05 or > 0.95) ──▶ Trim extreme tails OR switch to Overlap Weights (ATO)
│   ├── Extreme Weights (> 10.0) ──▶ Quantile Truncation (0.01, 0.99) OR Overlap Weights
│   └── Kish's Effective Sample Size (Neff < 0.5 * N) ──▶ Warning for variance inflation; use robust SE (HC1)
└── Balance Diagnostics: Austin (2009) Love Plot (absolute SMD < 0.10)
```

### E. Time-to-Event Analysis Under Competing Risks & Proportional Hazards Violations
```
Audit Event Types: Event of Interest vs. Competing Risks (e.g., CV death vs. non-CV death)
├── Single Failure Type (No Competing Risks) ──▶ Standard Kaplan-Meier & Cox Proportional Hazards (Recipe 11)
│   └── Proportional Hazards Violation Remedy Rule:
│       On Schoenfeld P <= 0.05, inspect plot_schoenfeld_residuals (figures.md §9), then:
│       ├── P <= 0.05/k (k = covariates tested) is a strong violation
│       ├── 0.05/k < P <= 0.05 is a weak violation (plot decides)
│       ├── Nuisance covariate ──▶ refit with strata=[...] for meaningful categorical or defensibly grouped continuous variables (`refit_cox_with_strata`; its HR is no longer reported); otherwise retain and model non-linearity (RCS)
│       ├── Exposure ──▶ piecewise/time-varying HR (`piecewise_hr`), or RMST
│       └── Reporting ──▶ report both models and name the remedy beside each violator
└── Multiple / Mutually Exclusive Failures
    ├── Cumulative Event Incidence / Absolute Risk Prediction:
    │   ├── DO NOT use Naive 1 - KM (treats competing events as censoring, estimating risk in a hypothetical world without competing events rather than real-world cumulative incidence; overestimates true risk)
    │   └── Use Aalen-Johansen Cumulative Incidence Function (CIF) (Recipe 15)
    └── Multivariable Regression:
        ├── Etiological / Biological Question ──▶ Cause-Specific Cox Model (csHR)
        │   └── Censors competing events at failure time (measures rate in event-free patients)
        └── Prognostic / Clinical Risk Stratification ──▶ Fine-Gray Subdistribution Hazard Model (SHR)
            └── Uses IPCW counting process expansion (measures cumulative incidence effect)
        └── Best Practice: Report both csHR and SHR side-by-side with clinical takeaway (Recipe 15)
```

#### Proportional Hazards Violation Decision Rule
On Schoenfeld P <= 0.05, inspect plot_schoenfeld_residuals (figures.md §9):
- P <= 0.05/k (k = covariates tested) is a strong violation; between that and 0.05 is weak (plot decides).
- nuisance covariate -> refit with strata=[...] for meaningful categorical or defensibly grouped continuous variables (`refit_cox_with_strata`; its HR is no longer reported); otherwise retain and model non-linearity (e.g. RCS);
- exposure -> piecewise/time-varying HR (`piecewise_hr`), or RMST;
- report both models and name the remedy beside each violator.


### F. Diagnostic Risk Reclassification & Incremental Value
```
Evaluating Added Value of Updated Biomarker/Model over Baseline
├── Discrimination Change ──▶ DeLong paired test for ΔAUROC (Recipe 3)
├── Net Reclassification Improvement (NRI) (Recipe 16)
│   ├── Continuous / Category-Free NRI ──▶ Primary metric when no established clinical thresholds exist (Pencina 2011)
│   ├── Categorical NRI ──▶ Use strictly when established clinical risk tiers exist (e.g. low <10%, intermediate 10-20%, high >20%)
│   ├── Separate Event & Non-Event Reporting ──▶ Mandatory per Kerr & Pepe (2014) to detect asymmetric reclassification
│   └── Calibration Guard ──▶ Enforce probability range validity (p_hat in [0, 1]); true model calibration requires separate evaluation (calibration plot, slope/intercept, Brier score), as poorly calibrated models produce spurious positive NRI/IDI values
└── Integrated Discrimination Improvement (IDI) (Recipe 16)
    └── Difference in discrimination slopes (mean predicted risk difference between events and non-events)
```

### G. Time-Dependent ROC in Longitudinal Follow-up
```
Evaluating Diagnostic / Prognostic Biomarkers Over Follow-up Time
├── Censoring Handling ──▶ Inverse Probability of Censoring Weighting (IPCW) via Kaplan-Meier on censoring distribution G(t) (Uno et al. 2007, Heagerty et al. 2000) (Recipe 17)
├── Definition ──▶ Cumulative/Dynamic ROC: Cases = primary events by t (T_i <= t, delta_i = 1); Controls = event-free beyond t (T_i > t)
├── Under Competing Risks ──▶ When competing_event_code is provided, patients experiencing competing events by t (T_i <= t, delta_i = k_comp) are retained in the control set weighted by IPCW (Fine-Gray/Uno subdistribution definition); default mode (competing_event_code=None) requires strictly binary event coding {0, 1}
├── Horizon Selection ──▶ Clinically meaningful milestones (e.g., 30d, 90d, 1y, 3y, 5y); avoid horizons where remaining cases or controls < 5
└── Reporting ──▶ AUROC(t) trajectory curve with 95% bootstrap CI band, optimal Youden cutpoint, sensitivity, specificity per milestone (Recipe 17)
```

### H. TRIPOD+AI Prediction Model Internal Validation & Uniform Shrinkage

```
Developing / Validating Clinical Prediction Models
├── Apparent Performance ──▶ Optimistically biased due to fitting on derivation data
├── Internal Validation Strategy:
│   ├── K-fold Cross-Validation ──▶ Acceptable, but bootstrap is preferred for smaller sample sizes
│   └── Efron-Gong Bootstrap (B ≥ 200) ──▶ Recommended by TRIPOD+AI; samples N with replacement, computes Optimism = Boot_mean - Test_mean, Corrected = Apparent - Optimism (Recipe 18)
├── Overfitting Adjustment:
│   ├── Uniform Shrinkage Factor s_hat = clip(mean(beta_cal,test), 0.01, 1.0)
│   ├── Shrunk Predictor Coefficients: beta_shrunk = s_hat * beta_orig
│   └── Re-estimated Intercept: solve sum(expit(alpha_0 + LP_shrunk)) = sum(Y) to maintain calibration-in-the-large
└── Clinical Safeguards ──▶ EPP < 10 warning; s_hat < 0.80 severe overfitting warning; <50% bootstrap convergence alert (Recipe 18)
```

### I. Minimum Sample Size for Clinical Prediction Models (Riley et al. 2020)

```
Determining Required Sample Size for Multivariable Prediction Model Development
├── Do NOT rely on simplistic 10 EPP rule-of-thumb
└── Evaluate All 4 Criteria of Riley et al. (BMJ 2020;368:m441) (Recipe 20):
    ├── Criterion 1: Precise estimation of overall intercept / risk (margin <= 0.05)
    ├── Criterion 2: Mean Absolute Prediction Error (MAPE <= 0.05, van Smeden 2019)
    ├── Criterion 3: Uniform shrinkage factor to minimize overfitting (S >= 0.90)
    └── Criterion 4: Small difference in apparent vs adjusted R2 (optimism <= 0.05)
├── Model Strength Input: C-statistic, Cox-Snell R2, or Nagelkerke R2
└── Required Sample Size = max(N1, N2, N3, N4) with binding criterion explicitly reported (Recipe 20)
```

### J. Polish Grilling Gate & Evidence Grounding Decision Tree
```
Polish Grilling Gate #6 Activation & Routing
├── Triggering Conditions:
│   ├── User prompt explicitly requests evidence/literature (e.g. "หาหลักฐานมาประกอบ", "find evidence") ──▶ Auto-activate Gate #6 Full Polish
│   └── Polish preference omitted in prompt ──▶ Gate #5 Alignment Inquiry:
│       ├── (Standard) Skip Polish (Direct Fast-Path / User-Scoped References) ──▶ Reference list strictly scoped to user-provided sources (omitted if none provided; zero unsolicited external citations); dynamically queries available literature MCPs (e.g. PubMed, PRISMA, NEJM, Cochrane) first, with search_web as fallback, for user sources to verify source-appropriate metadata (accepting issuing body and official URL/identifier when no DOI exists) and claim support; extracts reported empirical benchmarks and presents them as comparable only when matching target estimand, population, outcome, and time horizon and directly supporting Discussion claims (citing non-empirical sources for supported claims without inventing figures); local full text with accepted provenance signal satisfies verification if literature services are unavailable (fails closed with an explicit halt if unverified or conflicting; on-demand full polish available later)
│       └── (Optional) Full Publication Polish (Execute Gate #6 / Relevant Literature Enrichment) ──▶ Full empirical Discussion grounding on user sources + comprehensive exploratory retrieval across available literature MCPs / search engines to enrich Discussion and References
├── Polish Execution (Full Polish or On-Demand Trigger):
│   ├── Step 1: Claim vs Data Reconciliation
│   │   ├── Assess causal language by identification assumptions, prespecified estimand, and confounder control (do not treat weighting alone as sufficient or unweighted observational data as automatically prohibiting causal claims; report associative framing where identification is unverified, and require E-values only where defined and appropriate for the supported effect measure and sensitivity question)
│   │   ├── Superiority P > alpha (from analysis metadata, default 0.05) must be stated as "insufficient evidence to reject the null" (never "treatments are equivalent"); preserve separate interpretation for prespecified non-inferiority or equivalence analyses based on their margin and confidence interval
│   │   └── Numerical prose numbers must match results_dict within abs_tol = min(0.02, 10**(-dec_digits)) for |val| <= 1.0, 0.02 absolute or 0.01 (1%) relative tolerance for |val| > 1.0, or min(0.05, abs_tol * 100.0) for percentages
│   ├── Step 2: Empirical Research Grounding & Dynamic MCP / Literature Verification
│   │   ├── Dynamic Source Adaptation: Discover and prioritize any available bibliographic MCP servers (PubMed, PRISMA, NEJM, Cochrane, Semantic Scholar, medical-terminology); fallback to search_web / web queries when dedicated MCP tools are absent
│   │   ├── Step 2A (User-Scoped Sources): For every user-provided reference (even when full-text is available), query available MCPs/search to retrieve source-appropriate metadata (full author list, title, journal, year, volume, and pages—or article number/e-locator when pages are not assigned—and DOI when assigned for journal articles; issuing body, title, year, version/report identifier, and official URL when no DOI exists for guidelines/reports; authors or editors, title, publisher, and year for books or older publications) and verify claim support (constructing external queries strictly from sanitized metadata and de-identified descriptors without sending raw clinical text or PHI); extract and present reported empirical findings (benchmarks, effect sizes, control rates) as comparable only when matching target estimand, population, outcome, and time horizon and directly supporting Discussion claims (citing non-empirical sources for supported claims without inventing figures); local full text with accepted provenance signal satisfies verification when literature services are unavailable (halt explicitly if unverified)
│   │   ├── Step 2B (Literature Enrichment, Full Polish only): Actively search available MCPs / search_web for additional authentic, relevant published studies matching the study topic; extract and present real empirical benchmark figures as comparable only when matching target estimand, population, outcome, and time horizon and directly supporting a specific Discussion claim (otherwise cite without figures or omit) to enrich Discussion and Reference list
│   │   ├── Query available literature MCPs or bibliographic services for authentic details; restrict medical-terminology MCP to terminology, ontology codes, and laboratory reference ranges
│   │   ├── Anti-Phantom Citation Invariant: All cited sources must be verified real-world publications or authentic guidelines/reports with source-appropriate metadata; local full text with accepted provenance signal satisfies verification if literature services are unavailable; halt if unverified, retracted, or contradicting narrative
│   │   └── If reference is unverified, retracted, or contradicts narrative ──▶ Halt to reconcile or replace
│   └── Step 3: Citation & Reference Formatting
│       ├── Sequential numerical superscripts (<sup>1</sup>, <sup>2,3</sup>) in order of appearance
│       └── Format Reference List in locked Journal House Style (NEJM, JAMA, Lancet, BMJ, Clinical; strictly user-scoped in Standard Mode, enriched in Full Polish)
```

---

## 3. Reporting & Publication Standards

- **NEJM Style**: 3 horizontal rules (top border 2px, header border 1px, bottom border 2px), zero vertical dividers, $P$-values formatted as $P = 0.04$ or $P < 0.001$. Render via [Recipe 13: Publication Table Formatter](python-recipes.md#13-publication-grade-html-table-renderer-nejm--jama-standards).
- **JAMA Style**: Clean minimal dividers, $P$-values formatted without leading zero: $P = .04$ or $P < .001$.
- **Precision Invariants**: Unrounded intermediate calculations (`float64`); OR/HR reported to 2 decimal places with 95% CIs; percentages reported to 1 decimal place; explicit units mandatory (`mg/dL`, `mL/min/1.73m²`).
- **Reporting Checklists**: Consult EQUATOR standards: STROBE (observational cohorts), CONSORT (trials), TRIPOD (prediction models), STARD (diagnostic accuracy), or PRISMA (systematic reviews).
- **Figures & Visualizations**: When generating 300 DPI publication plots, read [figures.md](figures.md#2-available-plot-recipes); when rendering HTML Dual / Toggle View or multi-format reports, read [report-builder.md](report-builder.md#2-standalone-document-generation-recipes).

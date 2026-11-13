# Biostatistical Decision Heuristics & Reference Manual

Shared by `shinystat` (local) and `shinystat-cloud`. Consult during Pillar 2 (Research & Estimand Triangulation) to map a clinical research question to its model. Each method names both routes: the **local API** function in the bundled engine (`scripts/medstat/`; module map in the local SKILL.md Pillar 3) and the **cloud Recipe #** (file map in the cloud SKILL.md Recipe Index). Use the route your skill provides.

---

## 1. Clinical Study Archetype to Statistical Model Mapping

| Study Archetype | Target Estimand / Goal | Primary Model / Method | Key Assumption / Diagnostics | Local API · Cloud Recipe |
| :--- | :--- | :--- | :--- | :--- |
| **Type 1: Baseline Cohort** | Cohort characteristics | Table 1 (Mean/SD, Median/IQR, Count/%, SMD) | Normality inspection; Fisher exact fallback for small cells | `generate_table_one`, `calculate_smd` · Recipe 1 |
| **Type 2: Binary Outcome** | Odds Ratio (OR), Risk factors | Multivariable Binary Logistic Regression | Linearity of log-odds; Crude & Adjusted OR table | `fit_standard_logistic`, `fit_logistic_rcs` · Recipes 10, 22 |
| **Type 2-Sparse: Separation** | OR estimate in rare events | Firth Penalized Logistic Regression | Profile likelihood CIs; resolves monotone separation | `fit_firth_logistic` · Recipe 7 |
| **Type 2b: Ordinal Scale** | Cumulative Odds Ratio | Cumulative Logit Proportional Odds | Brant test for parallel lines; multinomial logit fallback | `fit_proportional_odds`, `test_proportional_odds` · Recipe 21 |
| **Type 3: Survival / Time-to-Event** | Hazard Ratio (HR), Median survival | Kaplan-Meier, Log-Rank & Cox Proportional Hazards | Proportional hazards test (Schoenfeld residuals); L2 shrinkage | `fit_survival_analysis_suite`, `fit_cox_ph` · Recipe 11 |
| **Type 3b: Competing Risks Survival** | Subdistribution Hazard (SHR) & Cause-Specific HR (csHR) | Aalen-Johansen CIF & Fine-Gray Model (IPCW) | Competing risk probability conservation; Naive KM overestimation audit | `fit_competing_risks_cif`, `fit_competing_risks_regression` · Recipe 15 |
| **Type 4: Diagnostic Test Accuracy** | Sens, Spec, PPV, NPV, AUC, LRs | 2x2 Contingency Matrix & Empirical ROC | Wilson score CIs; Haldane correction; Single/Paired DeLong | `calculate_2x2_metrics`, `auc_ci_delong`, `delong_paired_test` · Recipes 2, 3 |
| **Type 4b: Clinical Utility & Calib** | Net Benefit, Model Calibration | Decision Curve Analysis (DCA) & Recalibration | Brier score; Calibration slope & intercept; Austin-Steyerberg ICI | `calculate_calibration_slope_and_intercept`, `calculate_ici`, `calculate_dca` · Recipe 4 |
| **Type 4c: Internal Validation & Shrinkage** | Optimism-corrected C-statistic, Brier, R2, Slope | TRIPOD+AI Bootstrap Internal Validation | B >= 200 resamples; uniform shrinkage factor s_hat, re-estimated intercept | `bootstrap_internal_validation` · Recipe 18 |
| **Type 4d: Model Recalibration & Updating** | Target Population Calibration, Intercept, Slope | Intercept-only & Logistic Recalibration | Alpha and Beta recalibration parameters, ICI, Emax, AUROC preserved | `recalibrate_prediction_model` · Recipe 19 |
| **Type 5: Observational Causal (PSM)** | Average Treatment Effect in Treated (ATT) | Propensity Score Matching (PSM, 1:1, Caliper 0.2 SD) | Austin (2009) balance (SMD < 0.10); Pair ID tracking; E-value | `propensity_score_match`, `compute_balance_table`, `calculate_e_value` · Recipes 6, 9 |
| **Type 5b: Causal Weighting (IPTW)** | Population Effect (ATE, ATT, ATC, Overlap ATO) | Inverse Probability Treatment Weighting (IPTW) | Kish's Effective Sample Size ($N_{\text{eff}} \ge 0.5 \times N$), Positivity (PS 0.05–0.95), Robust SE | `compute_iptw_weights`, `fit_iptw_outcome_model` · Recipe 14 |
| **Type 6: Device / Rater Agreement** | Limits of Agreement, Reliability | Bland-Altman Difference Analysis & ICC (1–3k) | Bland-Altman 1999 large-sample CIs; Shrout-Fleiss exact F CIs | `calculate_bland_altman`, `calculate_icc` · Recipe 5 |
| **Type 7: Systematic Meta-Analysis** | Pooled Effect (RR/OR/MD/SMD) | Inverse-Variance Fixed-Effects & DerSimonian-Laird | Cochran's $Q$, Higgins' $I^2$, $\hat{\tau}^2$, 95% Prediction Interval | `run_meta_analysis` · Recipe 24 |
| **Type 8: Clustered / Multi-Center** | Population-averaged effect | Generalized Estimating Equations (GEE) | Cluster Design Effect ($\text{DEFF}$), $\text{ICC}_{\text{cluster}}$; Robust sandwich SEs | `fit_gee`, `calculate_design_effect` · Recipe 23 |
| **Type 9: Sample Size Determination** | Required N, Events, EPP, Statistical Power | Riley et al. (BMJ 2020) & Obuchowski (1998) | 4 criteria (intercept, MAPE, shrinkage, optimism); Obuchowski AUROC variance | `calculate_riley_sample_size`, `calculate_diagnostic_auroc_sample_size` · Recipe 20 |

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
│   └── Cox Proportional Hazards (Monotone Likelihood): (A) Penalized Cox — local: Firth penalized Cox with profile likelihood CIs (`fit_firth_cox`; `ci_fallback=True` means at least one term fell back to Wald: label each interval by its `ci_methods[term]` entry, or refit with `ci_method="wald"` and label all intervals Wald); cloud: scale-aware L2-penalized Cox (Recipe 11: standardize continuous predictors when scales differ, select and validate `penalizer > 0` on the supported data, report the transformation), or (B) Remove/collapse separating covariate
└── Low EPV (< 10) without Separation
    ├── Binary Logistic:
    │   ├── Strategy 1 (Etiological Effect): Firth penalized logistic regression with profile likelihood CIs (Recipe 7)
    │   ├── Strategy 2 (Prediction Model): L2 Ridge penalization / shrinkage to prevent overfitting (Riley et al., 2019)
    │   └── Strategy 3: Domain-guided clinical composite score or univariable reporting with sparse-data disclaimer
    └── Cox Proportional Hazards:
        ├── Strategy 1: Penalized Cox — local: Firth penalized Cox (`fit_firth_cox`); cloud: scale-aware L2-penalized Cox (Recipe 11, as above)
        ├── Strategy 2: Pre-specified clinical risk index / score reduction
        └── Strategy 3: Univariable survival analysis (Kaplan-Meier / univariable Cox) with sparse-data disclaimer
```

*Note on Modern EPV Evidence:* As demonstrated by Vittinghoff & McCulloch (2007) and van Smeden et al. (2016), EPV < 10 is an exploratory diagnostic indicator rather than an unconditional hard stop. Firth penalization resolves monotone separation and reduces first-order finite-sample parameter estimation bias (Heinze & Schemper, 2002); however, for prediction modeling, shrinkage/penalization is required to control overfitting.

### B. Missing Data Strategy
```
Audit Missingness Pattern & Mechanism (`littles_mcar_test` / Recipe 8; note H0: MCAR; failing to reject does not prove MCAR)
    Test scope: run it on the numeric model variables with at least one observed value and name them; a 100%-missing variable is reported as unavailable,
    and a categorical or otherwise excluded variable as untested, never as MCAR. A result with `df` ≤ 0 (too few missingness patterns; `p_value` NaN)
    is reported as unavailable, never as MCAR. `check_separation` takes the numeric model matrix (categorical predictors dummy-coded, rows complete on
    the model variables); a logistic design with no complete rows marks separation unevaluable until the approved missing-data strategy yields a
    usable design; check separation there.

├── Per-variable missingness < 5% in non-primary covariates ──▶ Complete-case analysis with documented justification
├── Missing 5%–40% under Missing at Random (MAR) ──▶ Multiple Imputation by Chained Equations for numeric (continuous) targets (`MICEImputer` / Recipe 12 `impute_mice_datasets`; categorical or binary targets need a type-appropriate imputation model such as logistic / polytomous regression, e.g. Recipe 12 `impute_mice_mixed_datasets`; include categorical predictors, fully observed ones too, in every continuous-target imputation model so the imputation model is congenial with the analysis model: fully observed categorical predictors are dummy-encoded before fitting (local: listed in `MICEImputer.fit_transform(columns=...)`; cloud: Recipe 12 `predictors`), while categorical predictors with missing values must be imputed jointly with type-appropriate categorical models (cloud: Recipe 12 `impute_mice_mixed_datasets`), never as dummy columns in `MICEImputer`, which imputes them as continuous; proper Bayesian draws, M ≥ 5 for ≤ 30% missing, M ≥ 20 for > 30% missing; MICE is the primary analysis and Complete-Case Analysis the sensitivity analysis, noting agreement does not prove MAR or exclude MNAR bias; the retention flow ends at the primary analytic N and the complete-case n is reported beside the sensitivity estimate)
├── Primary Exposure Missing = 100% in the analytic cohort (after applying the approved outcome disposition) ──▶ Stop: no exposure value is observed, so MICE cannot fit its imputation model and complete-case analysis has no rows; ask for usable exposure data or report the analysis as not estimable
├── Primary Exposure Missing > 40% and < 100% (Not MNAR) ──▶ Halt at Grilling Gate #4: the primary analysis is high-M MICE (M ≥ 40, tracking FMI) that imputes the exposure under MAR (imputation model includes the outcome and every analysis covariate); the user chooses its sensitivity analyses: (a) complete-case, and/or (b) pattern-mixture / delta-adjustment for departures from MAR; the exposure is never dropped. A categorical or binary exposure needs a type-appropriate imputation model (cloud: Recipe 12 `impute_mice_mixed_datasets`), never `MICEImputer`, which imputes numeric columns as continuous; the local engine has no categorical imputer, so the local run halts and reports the exposure as not locally imputable
├── Non-Primary Missing > 40% (Not MNAR; any model covariate counts as non-primary, only the outcome and the exposure of interest are primary) ──▶ Halt at Grilling Gate #4: (a) drop auxiliary covariate, (b) high-M MICE (M ≥ 40, tracking FMI) under MAR, or (c) pattern-mixture modeling (or missing-indicator adjustment restricted to randomized-trial baseline covariates or explicitly labeled sensitivity analyses)
├── Primary Outcome Missing (any row; ask at Gate #1 unless the user gave a disposition) ──▶ Recommended option: avoid outcome imputation in observational cohorts and preserve retention flow (N_initial ➔ N_analyzed); for trial/longitudinal dropout under MAR, allow protocol-justified outcome imputation (MICE) or MMRM with mandatory complete-case and tipping-point sensitivity analyses
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
│   ├── No estimand named by the user ──▶ ATT for both PSM and IPTW so the two analyses target the same population; state it, and use ATE only when asked
│   ├── Target = Treated Population (ATT) ──▶ IPTW (estimand='ATT', Recipe 14; if propensity-score trimming removes treated units, report excluded treated count and revise estimand to retained treated population, distinguishing from full ATT) OR 1:1 Nearest-Neighbor PSM with caliper 0.2 SD (Recipe 6; if caliper drops unmatched treated units, report unmatched-treated count and label estimand as matched-population effect, distinguishing from full ATT)
│   ├── Target = Whole Target Population (ATE) ──▶ IPTW (estimand='ATE') with stabilized weights (Recipe 14)
│   └── Significant Propensity Truncation / Lack of Overlap ──▶ Overlap Weights (estimand='ATO', Li et al. 2018, Recipe 14)
├── Weight Stability & Positivity Audit
│   ├── Positivity Alert (PS < 0.05 or > 0.95) ──▶ Trim extreme tails (if trimming removes treated patients under ATT, report excluded treated count and revised estimand) OR switch to Overlap Weights (ATO)
│   ├── Extreme Weights (> 10.0) ──▶ Quantile Truncation (0.01, 0.99) OR Overlap Weights
│   └── Kish's Effective Sample Size (Neff < 0.5 * N, overall weighted N) ──▶ Warning for variance inflation; use robust SE (HC1). When the user fixed the estimand (ATT) and positivity or Neff fails, report the requested estimand as primary with the diagnostics and add ATO as a labelled sensitivity analysis; never switch silently
├── Balance Diagnostics: Austin (2009) Love Plot (absolute SMD < 0.10); when any SMD stays at or above 0.10 after matching or weighting, report it as a limitation and add a covariate-adjusted outcome model as a labelled sensitivity analysis (do not change the primary model silently)
├── Matched-Pair Outcome ──▶ Headline OR from GEE logistic clustered on pair ID (exchangeable); RR and RD from the same pairs as secondary; E-value from the RR when the outcome is common (> 15%)
└── Weighted Outcome Model ──▶ IPTW weights as probability weights with sandwich variance (local: `medstat.causal.iptw.fit_iptw_outcome_model`; cloud: Recipe 14 `fit_weighted_outcome_model`); frequency weights inflate N and understate the variance
```

### E. Time-to-Event Analysis Under Competing Risks & Proportional Hazards Violations
```
Audit Event Types: Event of Interest vs. Competing Risks (e.g., CV death vs. non-CV death)
├── Single Failure Type (No Competing Risks) ──▶ Standard Kaplan-Meier & Cox Proportional Hazards (Recipe 11)
│   └── Schoenfeld P <= 0.05 ──▶ apply the Proportional Hazards Violation Decision Rule below
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
On Schoenfeld P <= 0.05, inspect `plot_schoenfeld_residuals` (figures.md §9), then:
- **Strength**: P <= 0.05/k (k = covariates tested) is a strong violation; 0.05/k < P <= 0.05 is weak (the plot decides).
- **Nuisance covariate**: refit with `strata=[...]` for a meaningful categorical or defensibly grouped continuous variable (`refit_cox_with_strata`; its HR is no longer reported). When continuous non-linear misspecification is suspected, refit with RCS and reassess PH; if the violation persists, model the time-varying effect or report it as a limitation.
- **Exposure**: piecewise / time-varying HR (`piecewise_hr`), or RMST.
- **Reporting**: report both models and name the remedy beside each violator.

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
│   ├── Uniform Shrinkage Factor s_hat = clip(optimism-corrected calibration slope = apparent slope - mean(boot slope - test slope), 0.01, 1.0)
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

For full verification rules, source-appropriate metadata requirements, provenance signals, and superscript citation formatting, read the single source of truth at [evidence-pipeline.md](evidence-pipeline.md):

```
Polish Grilling Gate #6 Activation & Routing
├── User prompt explicitly requests literature evidence (e.g. "หาหลักฐานมาประกอบ", "find evidence")
│   └── Auto-activate Gate #6 Full Polish (evidence-pipeline.md Steps 1–4, including Step 2B exploratory enrichment)
└── Polish preference omitted in prompt ──▶ Gate #5c Alignment Inquiry:
    ├── (Standard) Skip Polish (User-Scoped References) ──▶ Verify only user-provided sources via evidence-pipeline.md Step 2A (omit References section if none provided)
    └── (Optional) Full Publication Polish ──▶ Execute full evidence-pipeline.md Steps 1–4 including Step 2B exploratory literature enrichment
```

### K. Ordinal Regression, Restricted Cubic Splines, Clustered GEE & Meta-Analysis

- **Ordinal Outcomes ($\ge 3$ ordered levels)**: Fit Cumulative Logit Proportional Odds models (`fit_proportional_odds` [local API & Recipe 21 cloud]) and verify parallel slopes via the Brant test (`assess_proportional_odds` / `test_proportional_odds` in local API and Recipe 21 [cloud]); compare the omnibus $P$-value with the prespecified diagnostic $\alpha$ (using $0.05$ when unspecified), and fall back to multinomial logistic regression (`fit_multinomial_logistic`) when the omnibus $P$-value is below that $\alpha$ ($P < \alpha$).
- **Non-Linear Continuous Predictors**: Fit Harrell Restricted Cubic Splines (local API: `fit_logistic_rcs`, `fit_cox_rcs`; cloud recipe: `fit_rcs_spline_model` in Recipe 22) and report both `p_overall` and `p_nonlinearity`.
- **Clustered / Multi-Center Cohorts**: Compute cluster intraclass correlation ($\text{ICC}_{\text{cluster}}$) and Design Effect ($\text{DEFF} = 1 + (\bar{m} - 1)\text{ICC}$); fit GEE with robust sandwich standard errors or mixed-effects models (local API and cloud recipe: `fit_gee` in Recipe 23).
- **Systematic Synthesis & Meta-Analysis**: An undeclared effect scale (log ratio vs raw difference) is a Gate #1 question; `run_meta_analysis` raises `EffectScaleError` without it. Run inverse-variance Fixed-Effects and DerSimonian-Laird Random-Effects meta-analysis (`run_meta_analysis` / Recipe 24), reporting Cochran's $Q$, Higgins' $I^2$, $\hat{\tau}^2$, and the 95% prediction interval.

---

## 3. Reporting & Publication Standards

- **NEJM Style**: 3 horizontal rules (top border 2px, header border 1px, bottom border 2px), zero vertical dividers, $P$-values formatted as $P = 0.04$ or $P < 0.001$. Render via `render_nejm_table` (local) or Recipe 13 (cloud).
- **JAMA Style**: Clean minimal dividers, $P$-values formatted without leading zero: $P = .04$ or $P < .001$.
- **Predictor Units & Coding**: A continuous predictor without a user-named unit is scaled per 10 years (age, when the data record it in years) or per 1 SD (others, and age in an unknown unit; SD stated in the data's unit) in every table, text and figure. Under multiple imputation, compute that SD once from the observed values and apply it to every imputed dataset before pooling. Categorical covariates enter Cox and logistic suites as 0/1 dummies against a stated reference level. A unit absent from the data is written "unit not stated", never assumed.
- **Precision Invariants**: Unrounded intermediate calculations (`float64`); OR/HR reported to 2 decimal places with 95% CIs; percentages reported to 1 decimal place; explicit units mandatory (`mg/dL`, `mL/min/1.73m²`).
- **Reporting Checklists**: Consult EQUATOR standards: STROBE (observational cohorts), CONSORT (trials), TRIPOD (prediction models), STARD (diagnostic accuracy), or PRISMA (systematic reviews).
- **Figures & Visualizations**: When generating 300 DPI publication plots, read [figures.md](figures.md#2-available-plot-recipes); when rendering HTML Dual / Toggle View or multi-format reports, read [report-builder.md](report-builder.md#2-standalone-document-generation-recipes).

# TEST_INFRA.md: Biostatistical Engine & CLI E2E Test Infrastructure

## 1. Architectural & Testing Philosophy

The `medstat-core` end-to-end testing suite is designed around an **opaque-box, requirement-driven, clinical safety testing philosophy** derived authoritatively from `ORIGINAL_REQUEST.md` and `PROJECT.md`.

### 1.1 Opaque-Box & Requirement-Driven Architecture
- **Specification over Implementation**: Tests treat the statistical calculation engine and CLI as a black box. Assertions are formed against mathematical properties, published statistical methods, and documented CLI contracts, not against internal implementation minutiae.
- **Contract Adherence**: Public interfaces (`medstat` CLI commands, SAP `analysis_plan.yaml` schema, and public API returns) are verified against explicit behavioral and numerical specifications.
- **Deterministic Oracles**: Numerical results are compared against established statistical ground truths, including R package benchmarks (`logistf`, `coxphf`, `meta`, `irr`) with documented tolerances ($10^{-4}$ for point estimates).

### 1.2 Clinical Safety Directives
Clinical research applications require rigorous defensive safety guarantees:
1. **Zero-PHI Guarantee (HIPAA / PDPA Compliance)**:
   - All test fixtures and temporary execution files use strictly synthetic data.
   - Patient identifiers adhere to anonymized mock patterns (e.g. `PATIENT_MOCK_ONC_001`, `PATIENT_MOCK_SEP_001`, `POCUS_SUBJ_001`).
   - Any sensitive identifiers (names, dates of birth, national IDs, medical record numbers) are barred from fixtures, CLI outputs, and logs.
2. **Anti-Silent-Deletion & Explicit Missingness Strategy Gate**:
   - Silent listwise deletion (dropping incomplete rows without explicit instruction) is clinically dangerous and mathematically invalid under non-MCAR mechanisms.
   - When missing values exist in required analysis variables, the engine MUST raise `MissingStrategyRequiredError` unless an explicit strategy (`complete-case`, `mice`, `knn`, `indicator`) and documented clinical justification are provided.
3. **Audited Participant Retention Flow (CONSORT / STROBE Compliance)**:
   - Every analytical execution must track and emit the participant flow:
     $$N_{\text{initial}} \longrightarrow N_{\text{excluded}} \longrightarrow N_{\text{analyzed}}$$
   - Exclusion counts and specific clinical exclusion reasons must be recorded in structured metadata and verifiable in report outputs.
4. **Apache-2.0 License & Pure-Python Integrity**:
   - The statistical core maintains permissive Apache-2.0 licensing by implementing algorithms (such as two-way ANOVA for Intraclass Correlation Coefficient) purely via NumPy and SciPy, with zero GPL `pingouin` dependencies.

---

## 2. Feature Inventory & Verification Matrix

The test infrastructure covers all capabilities defined across the project lifecycle:

| Feature ID | Feature Domain | CLI Subcommand / Engine | Primary Requirements | Verification Tier |
|:---|:---|:---|:---|:---:|
| **FEAT-01** | Data Cleaning & Missing Audit | `medstat clean` | Missingness per-variable audit, Little's MCAR test, winsorization | Tier 1, 2, 3, 4 |
| **FEAT-02** | Explicit Missing Strategy Gate | `medstat clean`, `medstat model` | Mandatory `--missing` & `--missing-justification`, `MissingStrategyRequiredError` | Tier 1, 2 |
| **FEAT-03** | Audited Sample Retention Flow | `SampleFlowTracker` in CLI | $N_{initial} \to N_{excluded} \to N_{analyzed}$, CONSORT/STROBE flow | Tier 1, 3, 4 |
| **FEAT-04** | Imputation & MICE Pooling | `medstat clean --strategy [mice\|knn\|...]` | Rubin's rules pooling ($m=5$), FMI, between/within variance | Tier 1, 3, 4 |
| **FEAT-05** | Baseline Characteristics | `medstat table1` | Continuous/categorical summaries, SMDs, automated hypothesis testing | Tier 1, 3, 4 |
| **FEAT-06** | Generalized Linear Models | `medstat model --type [linear\|logistic]` | Parameter estimates, Wald/profile CIs, link functions | Tier 1, 3, 4 |
| **FEAT-07** | Survival Analysis & Schoenfeld | `medstat model --type cox_ph` | Kaplan-Meier, Cox PH hazard ratios, Schoenfeld PH assumption test | Tier 1, 4 |
| **FEAT-08** | Firth Penalized Regression | `medstat model --type firth_logistic\|firth_cox` | Separation handling, Profile Likelihood CIs, Likelihood Ratio Tests | Tier 1, 2, 3, 4 |
| **FEAT-09** | Restricted Cubic Splines (RCS) | `medstat model --rcs-var` | Non-linear hazard/odds ratios, knot placement, reference values | Tier 1, 2 |
| **FEAT-10** | E-Value Sensitivity Analysis | `medstat model --e-value` | Unmeasured confounding bounds for point estimate & CI limit | Tier 1, 3, 4 |
| **FEAT-11** | SAP Spec Engine Execution | `medstat model --spec <yaml>` | Declarative YAML/JSON analysis plan execution & validation | Tier 1, 4 |
| **FEAT-12** | 2x2 Contingency & Accuracy | `medstat diag` (2x2) | Sensitivity, Specificity, PPV, NPV, LR+, LR-, Wilson score 95% CIs | Tier 1, 2, 4 |
| **FEAT-13** | ROC Analysis & DeLong Tests | `medstat diag --roc` | Empirical ROC curves, AUC 95% CIs, correlated ROC comparison | Tier 1, 4 |
| **FEAT-14** | Decision Curve Analysis (DCA) | `medstat diag --dca` | Net benefit across decision thresholds, treat-all/treat-none comparisons | Tier 1, 4 |
| **FEAT-15** | Calibration & Risk Scoring | `medstat diag --calibration` | Hosmer-Lemeshow goodness-of-fit, Brier score, calibration slope/intercept | Tier 1 |
| **FEAT-16** | Propensity Score Matching (PSM) | `medstat causal psm` | Caliper matching (0.2 SD of logit), nearest neighbor, 1:1/1:k ratio | Tier 1, 3, 4 |
| **FEAT-17** | Covariate Balance & Love Plots | `medstat causal --balance-check` | Pre/post-matching SMDs (< 0.10 threshold), Love plot data generation | Tier 1, 3, 4 |
| **FEAT-18** | Pure-SciPy ICC Agreement | `medstat agreement icc` | Two-way ANOVA ICC (ICC1, ICC2, ICC3 single & average), F-tests, 95% CIs | Tier 1, 2, 4 |
| **FEAT-19** | Bland-Altman & Kappa Agreement | `medstat agreement [bland-altman\|kappa]` | Mean bias, Upper/Lower 95% LoA, Cohen's & Fleiss' Kappa | Tier 1, 4 |
| **FEAT-20** | Meta-Analysis & Heterogeneity | `medstat meta` | DerSimonian-Laird random effects, Fixed effects, $I^2, \tau^2, Q$ | Tier 1, 2, 4 |
| **FEAT-21** | Publication Bias & Egger's Test | `medstat meta --egger` | Funnel plot asymmetry, Egger's linear regression test | Tier 1, 4 |
| **FEAT-22** | Power & Sample Size | `medstat sample-size` | Two-sample t-test, Proportions, Survival log-rank, Diagnostic accuracy | Tier 1 |
| **FEAT-23** | Publication HTML Tables | `medstat report --style [nejm\|jama\|apa7]` | NEJM, JAMA, APA 7 styled tables with standard clinical typography | Tier 1, 3, 4 |
| **FEAT-24** | Automated Methods Narrative | `medstat report --narrative` | Synthesis of Methods and Results narrative sections | Tier 1, 3 |
| **FEAT-25** | Reporting Guideline Audits | `medstat report --audit [strobe\|consort\|tripod]` | Compliance checklist verification for clinical publication guidelines | Tier 1, 4 |

---

## 3. Four-Tier Testing Methodology

The test suite is structured into four distinct, progressive tiers designed to validate functionality from individual capabilities up to comprehensive multi-step clinical workflows:

```
┌─────────────────────────────────────────────────────────────┐
│ Tier 4: Real-World Clinical Workflows                       │
│ (5 Multi-Command Clinical Scenarios with End-to-End Audits) │
├─────────────────────────────────────────────────────────────┤
│ Tier 3: Cross-Feature Interactions                          │
│ (Pipelines: Clean -> Impute -> Model -> Balance -> Report)  │
├─────────────────────────────────────────────────────────────┤
│ Tier 2: Boundary, Adversarial & Corner Cases                │
│ (MissingStrategyRequiredError, Separation, Zero Variance)    │
├─────────────────────────────────────────────────────────────┤
│ Tier 1: Core Feature Coverage                               │
│ (>=5 Test Cases per Feature Across CLI Subcommands)         │
└─────────────────────────────────────────────────────────────┘
```

### 3.1 Tier 1: Feature Coverage (`test_tier1_feature_coverage.py`)
- **Objective**: Ensure every CLI subcommand and underlying statistical module executes correctly on valid inputs.
- **Coverage Standard**: At least 5 explicit test cases for each core feature/subcommand.
- **Focus Areas**:
  - `clean`: Missingness audit summary, Little's MCAR calculation, MICE imputation execution, KNN imputation, outlier winsorization.
  - `table1`: Group stratification, categorical variable frequency and percentage, continuous variable median/IQR or mean/SD, standardized mean differences (SMDs), auto-selection of parametric/non-parametric tests.
  - `model`: Linear regression OLS, logistic regression GLM, Cox proportional hazards regression with Kaplan-Meier, Firth penalized logistic regression with profile likelihood CIs, SAP spec engine YAML execution.
  - `diag`: 2x2 contingency table metrics with Wilson score CIs, ROC AUC with DeLong 95% CIs, DeLong curve comparison test, Decision Curve Analysis net benefit array, Hosmer-Lemeshow calibration.
  - `causal`: Propensity score matching execution, caliper enforcement, 1:1 nearest neighbor pairing, SMD balance calculation, Love plot data generation.
  - `meta`: Fixed-effect pooling, DerSimonian-Laird random-effects pooling, Cochran's $Q$ and $I^2$ heterogeneity, Forest plot coordinates, Egger's regression test.
  - `agreement`: Pure-SciPy ICC1, ICC2, ICC3 single and average measures, Bland-Altman mean difference and 95% LoA, Fleiss' multi-rater Kappa.
  - `sample-size`: Two-sample continuous t-test sample size, two-proportion power, survival log-rank hazard ratio sample size, diagnostic sensitivity sample size.
  - `report`: NEJM HTML table rendering, JAMA HTML table rendering, APA 7 formatting, statistical methods narrative generation, STROBE compliance audit.

### 3.2 Tier 2: Boundary & Corner Cases (`test_tier2_boundary_corner_cases.py`)
- **Objective**: Stress-test defensive programming, input validation, and statistical edge cases.
- **Coverage Standard**: At least 5 test cases per feature targeting adversarial inputs.
- **Critical Edge Conditions**:
  1. *Missing Strategy Omission*: Calling `clean` or `model` on data with missing values without `--missing` MUST raise `MissingStrategyRequiredError` (exit code non-zero with explanatory message).
  2. *Missing Justification Omission*: Specifying `--missing complete-case` on incomplete data without `--missing-justification` fails validation.
  3. *Empty or Degenerate Datasets*: Empty CSV (0 rows) or fewer rows than predictors ($N < P$) produces a clean clinical error, never an unhandled trace.
  4. *Zero Variance / Constant Columns*: Predictors with identical values across all observations are caught and reported defensively.
  5. *Severe Separation in Logistic Regression*: Complete or quasi-complete separation (e.g. zero events in one exposure group) successfully converges under Firth penalization where standard Newton-Raphson diverges.
  6. *Extreme Spline Knots*: Specifying $< 3$ knots, knots exceeding observed quantile boundaries, or non-monotonic knots raises explicit configuration errors.
  7. *Zero Cell Counts in Diagnostic 2x2*: Zero false negatives or zero false positives ($100\%$ Sensitivity or Specificity) handled with Wilson score continuity corrections without division-by-zero crashes.
  8. *Single Rater or Invariant Rater in Agreement*: Reliability data with 1 rater or zero between-subject variance raises an informative validation error.
  9. *Non-Positive Survival Times*: Survival data containing $t \le 0$ or negative durations are trapped prior to Cox model fitting.
  10. *Overly Restrictive Caliper in PSM*: Caliper setting so tight that zero treated units find matches raises a clear matching failure rather than returning an invalid empty cohort.

### 3.3 Tier 3: Cross-Feature Interactions (`test_tier3_cross_feature_interactions.py`)
- **Objective**: Validate the interoperability and sequential integrity of multi-stage analytical pipelines.
- **Integration Pipelines Tested**:
  1. *Pipeline A (MICE Imputation to Baseline Table)*:
     `incomplete_clinical.csv` $\to$ `medstat clean --strategy mice` $\to$ `medstat table1` verifying sample retention tracking ($N_{initial} = N_{analyzed}, N_{excluded} = 0$) and Rubin-pooled summary statistics.
  2. *Pipeline B (Complete-Case to Firth Model to Publication Table)*:
     `incomplete_clinical.csv` $\to$ `medstat clean --strategy complete-case` $\to$ `medstat model --type firth_logistic` $\to$ `medstat report --style nejm` verifying that excluded cases are reflected in the final NEJM table footer.
  3. *Pipeline C (Propensity Score Matching to Balance to Causal Effect)*:
     `cardiovascular_cohort.csv` $\to$ `medstat causal psm --caliper 0.2` $\to$ Love plot balance check (post-match SMD $< 0.10$) $\to$ `medstat model` logistic regression on matched cohort $\to$ E-value calculation for unmeasured confounding.
  4. *Pipeline D (Diagnostic Accuracy to ROC Comparison to Decision Curve Analysis)*:
     `sepsis_diagnostics.csv` $\to$ `medstat diag` 2x2 contingency $\to$ `medstat diag --roc --compare delong` (Procalcitonin vs Lactate) $\to$ `medstat diag --dca` net benefit calculation $\to$ TRIPOD audit report.
  5. *Pipeline E (Multi-Rater Agreement to Method Comparison to Reporting Narrative)*:
     `pocus_reliability.csv` $\to$ `medstat agreement icc` (pure-SciPy ICC2k) $\to$ `medstat agreement bland-altman` $\to$ `medstat report --narrative` drafting the inter-rater reliability methodology section.

### 3.4 Tier 4: Real-World Clinical Workflows (`test_tier4_clinical_workflows.py`)
- **Objective**: Execute end-to-end clinical workflow scenarios mimicking complete investigator workflows from raw clinical registries to publication-ready deliverables.
- **The 5 Clinical Scenarios**:
  1. **Workflow 1: Observational Oncology Survival with Sparse Events (Firth Cox PH)**:
     - Input: `tests/fixtures/oncology_survival.csv` (300 patients, sparse death events, TNM staging).
     - Execution: Data cleaning with complete-case $\to$ Table 1 stratified by treatment $\to$ Firth penalized Cox proportional hazards model $\to$ Schoenfeld proportional hazards test $\to$ NEJM publication HTML table.
  2. **Workflow 2: Emergency Department Sepsis Biomarker Diagnostic Accuracy & Decision Curve Analysis**:
     - Input: `tests/fixtures/sepsis_diagnostics.csv` (500 ED patients, suspected sepsis, Procalcitonin vs Lactate).
     - Execution: Data cleaning $\to$ 2x2 accuracy table at optimal clinical cutoff $\to$ DeLong correlated ROC comparison $\to$ DCA net benefit curves $\to$ TRIPOD compliance checklist.
  3. **Workflow 3: Point-of-Care Ultrasound (POCUS) Multi-Rater Reliability**:
     - Input: `tests/fixtures/pocus_reliability.csv` (80 lung ultrasound examinations evaluated by 3 independent clinicians).
     - Execution: Data cleaning $\to$ Pure-SciPy two-way ANOVA ICC (ICC2k agreement) $\to$ Pairwise Bland-Altman limits of agreement $\to$ Fleiss' multi-rater Kappa $\to$ Methods narrative.
  4. **Workflow 4: Cardiovascular Drug Effectiveness via Propensity Score Matching (PSM)**:
     - Input: `tests/fixtures/cardiovascular_cohort.csv` (1,000 cardiovascular patients evaluating Statin therapy on 3-year MACE).
     - Execution: MICE imputation of baseline confounders $\to$ 1:1 nearest-neighbor PSM with 0.2 SD caliper $\to$ Love plot SMD balance verification $\to$ Matched logistic regression $\to$ E-value calculation.
  5. **Workflow 5: Multi-Center Trial Meta-Analysis & Evidence Synthesis**:
     - Input: `tests/fixtures/multicenter_meta.csv` (15 multicenter randomized controlled trials).
     - Execution: DerSimonian-Laird random effects pooling $\to$ Heterogeneity assessment ($I^2, \tau^2, Q$) $\to$ Forest plot generation $\to$ Egger's regression test for publication bias $\to$ JAMA publication table.

---

## 4. Test Runner Environment & Execution Commands

### 4.1 Python Environment & Forwarding Interception
The project requires Python 3.12+ (PEP 695 type parameter syntax). Tests must be run using the virtual environment:
```bash
# Recommended direct test runner
.venv/bin/pytest tests/e2e/ -v
```

If invoked in an environment with legacy Python ($< 3.12$), the forwarding wrapper interceptor in `tests/conftest.py` captures execution before AST compilation and re-executes via `.venv/bin/pytest` with `MEDSTAT_PYTEST_FORWARDED=1` recursion guards.

### 4.2 Standard Test Invocation Commands

```bash
# Run all E2E tests
.venv/bin/pytest tests/e2e/ -v

# Run by Tier
.venv/bin/pytest tests/e2e/test_tier1_feature_coverage.py -v
.venv/bin/pytest tests/e2e/test_tier2_boundary_corner_cases.py -v
.venv/bin/pytest tests/e2e/test_tier3_cross_feature_interactions.py -v
.venv/bin/pytest tests/e2e/test_tier4_clinical_workflows.py -v

# Run by Subsystem / Feature Keyword
.venv/bin/pytest tests/e2e/ -k "clean or missing" -v
.venv/bin/pytest tests/e2e/ -k "firth or cox" -v
.venv/bin/pytest tests/e2e/ -k "diag or roc" -v
.venv/bin/pytest tests/e2e/ -k "psm or love_plot" -v
.venv/bin/pytest tests/e2e/ -k "icc or agreement" -v

# Run with Detailed Console Output and Durations
.venv/bin/pytest tests/e2e/ -v --durations=10 -s
```

---

## 5. Synthetic Clinical Fixtures Inventory

All fixtures are stored in `tests/fixtures/` and meet zero-PHI synthetic data rules:

| Fixture File | Clinical Domain | Key Variables | Observations | Target Use Cases |
|:---|:---|:---|:---:|:---|
| `oncology_survival.csv` | Oncology Survival Registry | `patient_id`, `time`, `status`, `age`, `treatment`, `stage`, `biomarker` | 300 | Firth Cox PH, survival, Schoenfeld tests, NEJM tables |
| `sepsis_diagnostics.csv` | ED Sepsis Biomarker Registry | `patient_id`, `procalcitonin`, `lactate`, `sepsis_confirmed_2x2`, `severe_sepsis_outcome`, `sofa_score` | 500 | 2x2 accuracy, DeLong ROC comparisons, DCA net benefit, TRIPOD |
| `pocus_reliability.csv` | Lung Ultrasound Agreement | `subject_id`, `rater_id`, `measurement_score`, `measurement_b_lines` | 150 (50x3) | Pure-SciPy ICC (ICC1, ICC2, ICC3), Bland-Altman, Kappa |
| `cardiovascular_cohort.csv` | CVD Preventive Cardiology | `patient_id`, `statin_rx`, `age`, `sbp`, `ldl`, `cv_event`, `diabetes`, `bmi` | 1,000 | Propensity score matching, SMD Love plots, E-values |
| `multicenter_meta.csv` | Multi-Center Stroke Trials | `study`, `year`, `effect_size`, `se`, `n_treatment`, `n_control`, `events_treatment`, `events_control` | 15 | DerSimonian-Laird random effects, $I^2$, Forest plots, Egger's test |
| `incomplete_clinical.csv` | General Medical Cohort | `patient_id`, `age`, `sex`, `creatinine`, `sbp`, `bmi`, `outcome`, `treatment` | 400 | Missingness audit, `MissingStrategyRequiredError`, MICE, KNN, Little's MCAR |
| `analysis_plan_example.yaml` | Declarative SAP Specification | Full YAML specification (filters, variables, reference categories, model settings) | N/A | SAP spec engine validation, CLI execution reproducibility |

---

## 6. Verification and Reporting Architecture

Upon completion of test execution, results are aggregated and published into `TEST_READY.md` at the project root, providing:
1. Exact execution command for one-touch verification.
2. Complete test count breakdown across Tiers 1 through 4.
3. Feature-by-feature coverage status with pass/fail tracking.
4. Any escalated implementation discrepancies or architectural notices.

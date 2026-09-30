# Study Design Decision Tree & Routing Heuristics

This reference details the clinical and structural heuristics used by `medstat-master` to automatically classify raw medical datasets and select the optimal statistical workflow.

---

## 1. Automated Column Classifier

The agent screens column names and value distributions against clinical patterns:

```
                          ┌───────────────────────────┐
                          │   Raw Clinical Dataset    │
                          └─────────────┬─────────────┘
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           ▼                            ▼                            ▼
  [Outcome Candidate]         [Exposure / Arm]             [Time-to-Event]
  - Binary 0/1, labels        - 2 or more groups           - time, duration, days
  - Mortality, sepsis         - Drug A vs B, surgery       - paired with event flag
           │                            │                            │
           ▼                            ▼                            ▼
  [Index Test / Marker]       [Rater / Agreement]          [Meta-Analysis]
  - continuous score, lab     - Subject ID, Rater ID       - study_id, effect_size
  - Gold standard diagnosis   - repeated numeric score     - se, variance, ci_limits
```

---

## 2. Study Design Decision Matrix

| IF Dataset Has... | AND Clinical Goal Is... | THEN Study Design Is... | Primary Model Command | Secondary Diagnostics |
| :--- | :--- | :--- | :--- | :--- |
| Single binary outcome ($Y \in \{0, 1\}$), multiple baseline covariates | Risk factor association, prognosis, or multivariable prediction | **Prognostic / Multivariable Cohort** | `medstat model --data <data.csv> --type logistic --outcome <Y> --covariates "<X>"` | - Method Firth (`--method firth --ci-method profile`) for separation or sparse events / low EPV<br>- RCS splines for continuous markers (`--spline-var`)<br>- E-value (`--e-value`) when assessing a specific causal exposure |
| Duration column ($T$) and event indicator ($\delta \in \{0, 1\}$; $1=\text{Event}, 0=\text{Censored}$) | Time-to-death, recurrence, or event-free survival | **Survival / Time-to-Event Cohort** | `medstat model --data <data.csv> --type cox --time <T> --outcome <delta> --covariates "<X>"` | - Kaplan-Meier curves<br>- Schoenfeld residuals correlation test (`--schoenfeld`)<br>- Firth Cox (`--method firth`) if zero events in subgroup |
| Continuous/ordinal test score ($S$) and binary gold standard ($D$) | Diagnostic biomarker accuracy, screening cutpoint validation | **Diagnostic Test Accuracy (DTA)** | `medstat diag --data <data.csv> --gold-standard <D> --test-col <S> --roc --dca` | - Wilson score 95% CIs for Sens/Spec<br>- DeLong AUC 95% CI & Youden index<br>- Vickers Decision Curve Analysis (DCA) |
| Non-randomized exposure indicator ($A \in \{0, 1\}$) and potential confounders | Comparative effectiveness, causal treatment effect estimation | **Observational Comparative Cohort** | `medstat causal psm --data <data.csv> --treatment <A> --covariates "<X>" --caliper 0.2` | - Austin (2009) Love plot<br>- Standardized Mean Differences ($\text{SMD} < 0.10$)<br>- Caliper width: $0.2 \times \text{SD}(\text{logit}(PS))$ (not raw PS distance)<br>- Outcome model on matched cohort |
| Multiple raters/devices assessing same subjects or wide paired columns | Inter-rater agreement, device equivalence, scoring reliability | **Reliability & Agreement Study** | `medstat agreement icc` OR `medstat agreement bland-altman` | - Pure-SciPy ICC (all 6 forms: ICC1, ICC2, ICC3)<br>- Bland-Altman mean bias & 95% LoA with Carkeet CIs |
| Effect sizes (log OR, HR, MD) with SEs or CIs across multiple studies | Evidence synthesis, pooled effect across literature | **Systematic Review & Meta-Analysis** | `medstat meta --data <data.csv> --effect-col <TE> --se-col <seTE> --study-col <study> --model random --method dl` | - DerSimonian-Laird random effects ($\tau^2, I^2$)<br>- Forest plot structured data (`--forest-plot`) |
| Baseline covariates across 2+ treatment or cohort groups | Characterize patient cohort, clinical trial randomization check | **Descriptive / Baseline Cohort** | `medstat table1 --data <data.csv> --group <group> --vars "<vars>"` | - SMDs across all continuous/binary variables<br>- Automatic parametric vs non-parametric tests |

---

## 3. Ambiguity Resolution & Edge-Case Handling

### Case 1: Sparse Events / Complete Monotone Separation
- **Condition**: Low events per parameter (EPV diagnostic evaluated against fitted model parameter count / degrees of freedom rather than a universal 10 EPV rule), sparse event counts across risk subgroups, or complete / quasi-complete separation where a predictor perfectly predicts the binary outcome.
- **Action**: Evaluate sparse-data evidence, total sample size, and the analysis estimand. Apply a prespecified criterion appropriate to the study goal and employ **Firth penalized regression** (`--method firth --ci-method profile`) as a bias-reduction approach for separation or related estimation issues, yielding finite profile likelihood estimates and mitigating small-sample bias in odds ratios without treating EPV < 10 as an automatic universal trigger.

### Case 2: Non-Linear Dose-Response
- **Condition**: Continuous laboratory values or vital signs (e.g., Blood Glucose, Lactate, Age, Systolic BP) entered into multivariable models.
- **Action**: Check if linear assumption holds; fit **Restricted Cubic Splines (RCS)** with 3 to 5 knots (`--spline-var <var> --knots 4`) to model U-shaped or non-linear hazard/odds curves.

### Case 3: Missingness Patterns & Mechanism Assessment
- **Condition**: Missing data detected in analytic variables.
- **Action**:
  - Do NOT rely on rigid percentage thresholds or assume Little's test $p > 0.05$ proves MCAR ($p > 0.05$ indicates only lack of evidence against MCAR; clinical mechanism must be considered).
  - Clinically assess the plausibility of MCAR, MAR, or MNAR based on data collection protocols and clinical workflows (see `references/missing-data-mechanisms.md`).
  - Choose strategy with documented clinical rationale:
    - **Complete-case**: Defensible only when MCAR is clinically plausible or missingness is trivial (<5%) and uninformative.
    - **MICE**: Applicable under plausible MAR, scaling imputations $m$ to Fraction of Missing Information (FMI).
    - **KNN / Indicator**: As clinically justified for point-of-care or structured patterns (note: indicator imputation is not an appropriate default for missing confounders in observational analyses due to the risk of residual confounding bias).
  - Always perform sensitivity analyses across mechanisms and track audited sample retention flow ($N_{\text{initial}} \to N_{\text{excluded}} \to N_{\text{analyzed}}$). Never perform silent listwise deletion.

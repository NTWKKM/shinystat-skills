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
| Single binary outcome ($Y \in \{0, 1\}$), multiple baseline covariates | Risk factor association, prognosis, or multivariable prediction | **Prognostic / Multivariable Cohort** | `medstat model fit --type logistic --y <Y> --x "<X>"` | - Firth penalized if $<10$ events per variable<br>- RCS splines for continuous markers<br>- E-value for unmeasured confounding |
| Duration column ($T$) and event indicator ($\delta \in \{0, 1\}$) | Time-to-death, recurrence, or event-free survival | **Survival / Time-to-Event Cohort** | `medstat model fit --type cox --time <T> --event <delta> --x "<X>"` | - Kaplan-Meier curves<br>- Schoenfeld residuals correlation test (`--schoenfeld`)<br>- Firth Cox if zero events in subgroup |
| Continuous/ordinal test score ($S$) and binary gold standard ($D$) | Diagnostic biomarker accuracy, screening cutpoint validation | **Diagnostic Test Accuracy (DTA)** | `medstat diag --gold-standard <D> --test-col <S> --roc --dca` | - Wilson score 95% CIs for Sens/Spec<br>- DeLong AUC 95% CI & Youden index<br>- Vickers Decision Curve Analysis (DCA) |
| Non-randomized exposure indicator ($A \in \{0, 1\}$) and potential confounders | Comparative effectiveness, causal treatment effect estimation | **Observational Comparative Cohort** | `medstat causal psm --treatment <A> --covariates "<X>" --caliper 0.2` | - Austin (2009) Love plot<br>- Standardized Mean Differences ($\text{SMD} < 0.10$)<br>- Outcome model on matched cohort |
| Multiple raters/devices assessing same subjects or wide paired columns | Inter-rater agreement, device equivalence, scoring reliability | **Reliability & Agreement Study** | `medstat agreement icc` OR `medstat agreement bland-altman` | - Pure-SciPy ICC (all 6 forms: ICC1, ICC2, ICC3)<br>- Bland-Altman mean bias & 95% LoA with Carkeet CIs |
| Effect sizes (log OR, HR, MD) with SEs or CIs across multiple studies | Evidence synthesis, pooled effect across literature | **Systematic Review & Meta-Analysis** | `medstat meta dl --effect <TE> --se <seTE>` | - DerSimonian-Laird random effects ($\tau^2, I^2$)<br>- Forest plot structured data<br>- Egger's test for funnel asymmetry |
| Baseline covariates across 2+ treatment or cohort groups | Characterize patient cohort, clinical trial randomization check | **Descriptive / Baseline Cohort** | `medstat table1 --group <group> --vars "<vars>"` | - SMDs across all continuous/binary variables<br>- Automatic parametric vs non-parametric tests |

---

## 3. Ambiguity Resolution & Edge-Case Handling

### Case 1: Sparse Events / Complete Monotone Separation
- **Condition**: Event count $< 10 \times (\text{number of covariates})$ OR complete separation where a predictor perfectly predicts the binary outcome.
- **Action**: Automatically activate **Firth penalized regression** (`--firth`) to compute finite profile likelihood estimates and avoid infinite odds ratios.

### Case 2: Non-Linear Dose-Response
- **Condition**: Continuous laboratory values or vital signs (e.g., Blood Glucose, Lactate, Age, Systolic BP) entered into multivariable models.
- **Action**: Check if linear assumption holds; fit **Restricted Cubic Splines (RCS)** with 3 to 5 knots (`medstat.models.splines`) to model U-shaped or non-linear hazard/odds curves.

### Case 3: Missingness Patterns
- **Condition**:
  - $< 5\%$ missing and Little's MCAR $p > 0.05 \implies$ Complete-Case acceptable.
  - $5\% - 40\%$ missing $\implies$ MICE (Chained Equations) with $m \ge 5$.
  - $> 40\%$ missing $\implies$ Flag high risk of bias; advise clinician on feasibility.
- **Action**: Never perform silent listwise deletion; always emit audited sample flow ($N_{\text{initial}} \to N_{\text{excluded}} \to N_{\text{analyzed}}$).

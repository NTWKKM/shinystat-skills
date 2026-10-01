# Autonomous Statistical Analysis Plan (SAP) Templates

This document provides standardized templates for presenting clinical proposals (Mode B) or executing automated analysis plans via YAML (`medstat model --spec`).

---

## 1. Markdown Proposal Template (Interactive User Alignment)

Use this format when presenting a proposal to the user before running heavy computations:

```markdown
# 📋 Proposed Statistical Analysis Plan (SAP)

## 1. Study Overview & Primary Estimand
- **Dataset**: `[Filename.csv]` (N = [Sample Size], P = [Feature Count])
- **Inferred Design**: [e.g., Observational Retrospective Cohort / Diagnostic Accuracy Study]
- **Target Population**: [e.g., Adult ICU patients with suspected sepsis]
- **Primary Endpoint**: `[outcome_variable]` ([type: binary 0/1 / time-to-event / continuous])
- **Primary Exposure / Intervention**: `[exposure_variable]` ([levels / units])

## 2. Data Cleaning & Sample Flow Architecture
- **Missing Data Mechanism**:
  - Missingness detected in: `[variable_1]` (X%), `[variable_2]` (Y%)
  - Little's MCAR Test: p = [value] ([Consistent / Inconsistent with MCAR])
  - **Proposed Strategy**: `[complete-case | mice | knn | indicator]`
  - **Clinical Rationale**: "[Clinical justification based on specimen collection/reporting practices]"
- **Audited Sample Flow Tracking**:
  - Initial cohort: $N_{\text{initial}} = [N]$
  - Exclusions: Missing primary endpoint or unresolvable missingness ($N_{\text{excluded}}$)
  - Target analyzed cohort: $N_{\text{analyzed}}$

## 3. Planned Statistical Analyses
1. **Baseline Patient Characteristics (Table 1)**:
   - Stratified by: `[exposure_variable]`
   - Covariates: `[age, sex, comorbidities, labs]`
   - Balance Metric: Standardized Mean Differences (SMD), flagging $\text{SMD} \ge 0.10$
2. **Primary Association / Effect Model**:
   - Model Type: `[Multivariable Logistic / Cox Proportional Hazards / PSM / Diagnostic ROC]`
   - Adjustments: `[covariates]`
   - Sparse Event Handling: `[Standard Maximum Likelihood | Firth Penalized Likelihood]`
   - Non-Linear Modeling: `[Restricted Cubic Splines on continuous markers]`
3. **Sensitivity & Robustness Analyses**:
   - VanderWeele E-value for unmeasured confounding (evaluating primary exposure–outcome effect).
   - Proportional hazards validation via Schoenfeld residuals (for Cox PH analyses).
4. **Reporting & Publication Formatting**:
   - Format: `[NEJM / JAMA / APA 7]` HTML table with strictly 0 vertical borders.
   - Reporting Guideline: `[STROBE / CONSORT 2025 / TRIPOD+AI 2024]` audit.

---
*Would you like to proceed with this analysis plan, or modify any variables/assumptions?*
```

---

## 2. Automated YAML Analysis Plan Spec Template

For headless automated execution via `medstat model --spec analysis_plan.yaml`:

```yaml
version: "1.0"

metadata:
  study_title: "Automated Clinical Cohort Analysis"
  protocol_id: "MEDSTAT-SAP-001"
  analyst: "Clinical Biostatistics Core"
  date: "2026-09-30"
  reporting_guideline: "STROBE"
  study_design: "retrospective_cohort"

data:
  input_path: "data/raw_clinical_cohort.csv"
  id_column: "patient_id"
  filters: []

variables:
  - name: "mortality_30d"
    label: "30-Day All-Cause Mortality"
    role: "outcome"
    data_type: "binary"
    categories: [0, 1]
    reference_category: 0

  - name: "treatment_arm"
    label: "High-Intensity Intervention"
    role: "exposure"
    data_type: "binary"
    categories: [0, 1]
    reference_category: 0

  - name: "age"
    label: "Baseline Age (years)"
    role: "covariate"
    data_type: "continuous"

  - name: "sofa_score"
    label: "Baseline SOFA Score"
    role: "covariate"
    data_type: "continuous"

models:
  - name: "primary_logistic_regression"
    description: "Multivariable logistic regression of 30-day mortality on treatment arm"
    type: "logistic"
    outcome: "mortality_30d"
    exposure: "treatment_arm"
    covariates:
      - "age"
      - "sofa_score"
    formula: "mortality_30d ~ treatment_arm + age + sofa_score"
    reference_categories:
      treatment_arm: 0
    missing_strategy: "complete-case"
    missing_justification: "Complete-case analysis under plausible MCAR assumption; sensitivity analysis assesses robustness to plausible departures from MCAR"
    options:
      method: "standard"  # options: standard (statsmodels coefficient-based CIs, default), firth (select when separation, sparse data, or prespecified bias-reduction applies)
      e_value: true

reporting:
  style: "nejm"
  format: "html"
  include_narrative: true
```

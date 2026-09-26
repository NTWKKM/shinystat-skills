# Autonomous Statistical Analysis Plan (SAP) Templates

This document provides standardized templates for presenting clinical proposals (Mode B) or executing automated analysis plans via YAML (`medstat --spec`).

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
   - VanderWeele E-value for unmeasured confounding.
   - Proportional hazards validation via Schoenfeld residuals.
4. **Reporting & Publication Formatting**:
   - Format: `[NEJM / JAMA / APA 7]` HTML table with strictly 0 vertical borders.
   - Reporting Guideline: `[STROBE / CONSORT 2025 / TRIPOD+AI 2024]` audit.

---
*Would you like to proceed with this analysis plan, or modify any variables/assumptions?*
```

---

## 2. Automated YAML Analysis Plan Spec Template

For headless automated execution via `medstat --spec analysis_plan.yaml`:

```yaml
version: "1.0"
study_title: "Automated Clinical Cohort Analysis"
dataset:
  input_path: "data/raw_clinical_cohort.csv"
  cleaned_path: "data/cleaned_cohort.csv"
  sample_flow_output: "reports/sample_flow.json"

cleaning:
  missing_strategy: "mice"
  imputations: 5
  justification: "Missing physiological vitals and lab values assumed MAR conditional on baseline severity scores; Little's MCAR test p=0.24."
  binary_outcome_recoding:
    mortality_30d:
      "Dead": 1
      "Alive": 0

table1:
  stratify_by: "treatment_arm"
  variables:
    - "age"
    - "sex"
    - "bmi"
    - "sofa_score"
    - "comorbidity_charlson"
  smd_threshold: 0.10
  output_json: "reports/table1.json"

primary_model:
  type: "logistic"
  outcome: "mortality_30d"
  covariates:
    - "treatment_arm"
    - "age"
    - "sex"
    - "sofa_score"
  firth_penalization: true  # auto-activated if events < 10 per variable
  splines:
    - variable: "sofa_score"
      knots: 4
  e_value_sensitivity: true
  output_json: "reports/primary_model.json"

reporting:
  target_journal: "nejm"
  output_html: "reports/manuscript_table.html"
  guideline: "strobe"
  methods_narrative: true
```

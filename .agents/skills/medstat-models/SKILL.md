---
name: medstat-models
description: Multivariable regression, survival analysis, Firth penalized likelihood, restricted cubic splines (RCS), Table 1 generation with SMDs, and VanderWeele E-value sensitivity. Use when fitting multivariable GLMs, Cox proportional hazards, handling separation/sparse events with Firth penalization, testing non-linear dose-response curves, or executing YAML Statistical Analysis Plans (SAP).
---

# medstat-models: Multivariable Regression, Survival & Penalized Models

Biostatistical modeling engine supporting generalized linear models, Cox proportional hazards with Schoenfeld diagnostics, Firth penalized likelihood, restricted cubic splines, and unmeasured confounding sensitivity analysis.

## Core Rules

1. **Pre-Model Table 1**: Always characterize baseline covariates with Standardized Mean Differences (SMD) before multivariable modeling.
2. **Proportional Hazards Assumption**: Every Cox model must check Schoenfeld residual correlation across time; violations require stratified Cox or time-varying covariates.
3. **Sparse Events & Monotone Likelihood**: In sparse event survival (< 20 events) or quasi-complete logistic separation, use Firth's penalized likelihood with profile likelihood confidence intervals.
4. **Non-Linearity Verification**: Continuous exposures with potential non-linear biology must be modeled using restricted cubic splines (RCS) with centered contrast reference points.

## Execution Sequence

```
[1. TABLE 1] ──▶ [2. SPECIFY SAP / CLI] ──▶ [3. FIT MODEL] ──▶ [4. SENSITIVITY (E-VALUE)]
```

### Step 1: Generate Table 1 Baseline Characteristics

```bash
medstat table1 --data <dataset.csv> \
  --group <treatment_col> \
  --vars "age,sex,bmi,sbp,creatinine" \
  --output table1.json
```
- Continuous normal: Mean ± SD (Student's/Welch's t-test).
- Continuous skewed: Median [IQR] (Mann-Whitney U).
- Categorical: $n$ (%) (Pearson Chi-Square / Fisher's exact).
- Imbalance screening: SMD > 0.10 flags clinically meaningful imbalance.

### Step 2: Fit Multivariable Model or Execute SAP Spec

#### Option A: Direct CLI Model Fitting
```bash
# Multivariable Logistic Regression with E-value
medstat model --data <clean.csv> \
  --type logistic --outcome outcome_cured \
  --exposure treatment --covariates "age,sex,bmi,hypertension" \
  --e-value --output logistic_res.json

# Firth Penalized Logistic Regression (resolves separation)
medstat model --data <sparse.csv> \
  --type firth_logistic --outcome death \
  --exposure drug_arm --covariates "age,stage,ecog" \
  --output firth_logistic_res.json

# Cox Proportional Hazards with Schoenfeld PH Test
medstat model --data <survival.csv> \
  --type cox_ph --time time_months --outcome status_death \
  --exposure treatment --covariates "age,stage,biomarker" \
  --schoenfeld --output cox_res.json

# Firth Penalized Cox PH (sparse mortality events)
medstat model --data <survival.csv> \
  --type cox_ph --time time_months --outcome status_death \
  --exposure treatment --covariates "age,stage,biomarker" \
  --method firth --output firth_cox_res.json
```

#### Option B: Statistical Analysis Plan (SAP) YAML Spec
For pre-registered, audit-trailed analyses, define `analysis_plan.yaml` and execute:
```bash
medstat model --spec analysis_plan.yaml --output sap_report.json
```
See [references/model-spec-schema.md](references/model-spec-schema.md) for full YAML schema.

### Step 3: Assess Model Diagnostics

- **Cox Proportional Hazards**: Confirm Schoenfeld test $p > 0.05$ across all covariates.
- **Firth Convergence**: Verify profile likelihood confidence intervals converge.
- **E-Value Sensitivity**: If effect is statistically significant, compute the minimum unmeasured confounding strength required to explain away the observed estimate.

## Completion Criteria

- [ ] Baseline characteristics tabulated with explicit SMD imbalance checks.
- [ ] Model coefficients, 95% confidence intervals, and p-values generated.
- [ ] Proportional hazards or separation diagnostics completed.
- [ ] E-value calculated for primary exposure to quantify sensitivity to unmeasured confounding.

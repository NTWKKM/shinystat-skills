---
name: medstat-clean
description: Clinical data cleaning, missingness audit, Little's MCAR testing, explicit imputation (MICE, KNN, indicator), outlier winsorization, and audited sample flow tracking (N_initial -> N_excluded -> N_analyzed). Use when auditing missingness, cleaning EHR/health registries, preparing clinical cohort data, or resolving missing data without silent listwise deletion.
---

# medstat-clean: Clinical Data Cleaning & Audited Sample Retention

Clinical data preparation engine enforcing explicit missing data justification and participant flow tracking under CONSORT and STROBE standards.

## Core Rules

1. **No Silent Listwise Deletion**: Omitting `--strategy` when missing values exist raises `MissingStrategyRequiredError`. Never drop rows without clinical justification.
2. **Audited Sample Retention Flow**: Every cleaning run tracks and records participant retention:
   $$N_{\text{initial}} \longrightarrow N_{\text{excluded}} \longrightarrow N_{\text{analyzed}}$$
3. **Preserve Raw Values**: Keep original files untouched; write transformed cohorts to distinct output targets.
4. **Binary Endpoint Standardization**: Recode all categorical/text endpoints (e.g., `'Alive'/'Dead'`, `'Yes'/'No'`) to numeric `0/1` (`1 = Event`, `0 = Non-event`) during cleaning so downstream modeling engines receive unambiguous event indicators.

## Execution Sequence

```
[1. AUDIT] ──▶ [2. STRATEGIZE] ──▶ [3. SANITIZE] ──▶ [4. EMIT FLOW]
```

### Step 1: Run Missingness Audit

Audit missingness patterns and test the Missing Completely at Random (MCAR) assumption:

```bash
medstat clean --data <dataset.csv> --audit-only --audit-out <audit.json>
```

Evaluate the resulting audit:
- Check per-variable missingness percentage:
  - **< 5% (Low)**: Complete-case analysis acceptable under MCAR.
  - **5% – 20% (Moderate)**: Multiple Imputation by Chained Equations (MICE) recommended.
  - **20% – 40% (High)**: Imputation required with sensitivity analysis.
  - **> 40% (Critical)**: High risk of bias; consider indicator method or dropping variable.
- Check Little's MCAR test: $p > 0.05$ provides insufficient evidence to reject MCAR; $p \le 0.05$ provides evidence against MCAR (data depart from MCAR).
- Consult [references/missing-data-mechanisms.md](references/missing-data-mechanisms.md) for mechanism selection criteria.

### Step 2: Execute Clinically Justified Strategy

Run cleaning with an approved strategy (`complete-case`, `mice`, `knn`, `indicator`) and documented rationale:

```bash
# Complete-case analysis (MCAR justified)
medstat clean --data <dataset.csv> \
  --strategy complete-case \
  --missing-justification "MCAR verified by Little test (p=0.42); baseline labs missing <5%" \
  --output clean_cc.csv --audit-out retention.json

# Multiple Imputation by Chained Equations (MICE, MAR justified)
medstat clean --data <dataset.csv> \
  --strategy mice --imputations 5 \
  --missing-justification "MAR assumed; chained predictive mean matching for creatinine and BMI" \
  --output clean_mice.csv --audit-out retention.json

# K-Nearest Neighbors imputation
medstat clean --data <dataset.csv> \
  --strategy knn --neighbors 5 \
  --missing-justification "Normalized Euclidean distance KNN imputation for point-of-care vitals" \
  --output clean_knn.csv --audit-out retention.json
```

### Step 3: Sanitize & Winsorize Outliers

Detect and clamp non-physiological or extreme values:
- Use Tukey's IQR rule ($1.5 \times \text{IQR}$) or Median Absolute Deviation (MAD > 3.0).
- Extreme laboratory readings or physiological vitals (e.g. SBP > 260 or < 40) are winsorized to boundary percentiles (1st and 99th), never silently deleted.
- Standardize all binary and survival clinical endpoints to numeric `0/1` (`1 = Event`, `0 = Non-event`); never forward text outcomes to modeling.

### Step 4: Verify Sample Retention Flow

Verify that the output contains the audited sample retention tracker:
- $N_{\text{initial}}$: Total enrolled patients before exclusions.
- $N_{\text{excluded}}$: Rows removed with categorized rationale.
- $N_{\text{analyzed}}$: Final analytic cohort size matching downstream model inputs.

## Completion Criteria

- [ ] Missingness audit executed and reviewed across all clinical variables.
- [ ] Explicit missing data strategy chosen with documented clinical justification.
- [ ] Binary and survival event endpoints standardized to numeric 0/1 (1 = Event).
- [ ] Cleaned dataset written to `--output` path with zero unexpected `NaN` cells.
- [ ] Sample retention flow metadata recorded with explicit initial, excluded, and analyzed counts.

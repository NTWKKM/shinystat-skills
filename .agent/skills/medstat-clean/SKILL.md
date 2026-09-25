---
name: medstat-clean
description: Clinical data cleaning, missingness audit, Little's MCAR testing, explicit imputation (MICE, KNN, indicator), outlier winsorization, and audited sample flow tracking (N_initial -> N_excluded -> N_analyzed). Use when auditing missingness, cleaning EHR/health registries, preparing clinical cohort data, or resolving missing data without silent listwise deletion.
---

# medstat-clean: Clinical Data Cleaning & Audited Sample Retention

Clinical data preparation engine enforcing explicit missing data justification and participant flow tracking under CONSORT and STROBE standards.

## Core Rules

1. **No Silent Listwise Deletion**: Omitting `--strategy` when missing values exist raises `MissingStrategyRequiredError`. Never drop rows without clinical justification.
2. **Audited Sample Retention Flow**: Every cleaning run (with `--strategy`) tracks participant retention:
   $$N_{\text{initial}} \longrightarrow N_{\text{excluded}} \longrightarrow N_{\text{analyzed}}$$
   Specify `--audit-out <file>` to persist the audited flow artifact. Note: The `--audit-only` path produces missingness audit output but does not execute cleaning or generate `sample_flow` data.
3. **Preserve Raw Values**: Keep original files untouched; write transformed cohorts to distinct output targets.
4. **Binary Endpoint Standardization**: Explicitly recode binary event-status endpoints to numeric `0/1` (`1 = Event`, `0 = Non-event`) before model execution. Event direction must not be inferred from arbitrary text labels. Multicategory outcomes and survival follow-up time columns must be preserved without recoding.

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
- Review descriptive missingness tiers (used for exploratory assessment; method selection must be clinically justified by mechanism rather than rigid percentage cutoffs):
  - **< 5% (Low)**: Complete-case analysis often viable if missingness mechanism is consistent with MCAR.
  - **5% – 20% (Moderate)**: Multiple Imputation by Chained Equations (MICE) under MAR assumption.
  - **20% – 40% (High)**: Substantial missingness; multiple imputation may be appropriate if missingness mechanism, imputation model adequacy, and analysis objectives support it. Mandatory sensitivity analysis comparing imputed vs. complete-case results.
  - **> 40% (Critical)**: High risk of residual bias; evaluate whether variable can be reliably imputed or retained.
- Check Little's MCAR test: $p > 0.05$ fails to reject MCAR (insufficient evidence against MCAR); $p \le 0.05$ provides evidence against MCAR (departures from MCAR).
- Consult [references/missing-data-mechanisms.md](references/missing-data-mechanisms.md) for mechanism selection criteria.

### Step 2: Execute Clinically Justified Strategy

Run cleaning with an approved strategy (`complete-case`, `mice`, `knn`, `indicator`) and documented rationale:

```bash
# Complete-case analysis (MCAR justified)
medstat clean --data <dataset.csv> \
  --strategy complete-case \
  --missing-justification "Little's test did not reject MCAR null (p=0.42); complete-case analysis prespecified with <5% missingness" \
  --output clean_cc.csv --audit-out retention.json

# Multiple Imputation by Chained Equations (MICE, MAR justified; m=5 shown as illustrative baseline, scale m to FMI and target SE precision)
medstat clean --data <dataset.csv> \
  --strategy mice --imputations 5 \
  --missing-justification "MAR assumed; chained predictive mean matching for creatinine and BMI (m=5 illustrative)" \
  --output clean_mice.csv --audit-out retention.json

# K-Nearest Neighbors imputation
medstat clean --data <dataset.csv> \
  --strategy knn --neighbors 5 \
  --missing-justification "Normalized Euclidean distance KNN imputation for point-of-care vitals" \
  --output clean_knn.csv --audit-out retention.json
```

### Step 3: Sanitize & Winsorize Outliers

Detect and handle non-physiological or extreme values:
- Use Tukey's IQR rule ($1.5 \times \text{IQR}$) or robust median deviation (e.g., modified z-score $> 3.5$ or $|x_i - \text{median}| > 3 \times \text{MAD}$).
- Extreme laboratory readings or physiological vitals (e.g., SBP > 260 or < 40 mmHg) should be flagged for clinical chart review and verification against source records; apply prespecified variable-specific handling rules rather than blanket automatic winsorization.
- Standardize explicitly binary event-status fields to numeric `0/1` (`1 = Event`, `0 = Non-event`), ensuring survival follow-up duration and multicategory endpoints remain intact; never forward text outcomes to modeling.

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

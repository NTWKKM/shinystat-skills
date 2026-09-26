---
name: medstat-master
description: Master clinical biostatistics orchestrator. Ingests raw clinical data (CSV, XLSX, TSV), automatically inspects schema and distributions, identifies the clinical study design, formulates or executes a Statistical Analysis Plan (SAP), and dynamically orchestrates the medstat skills pipeline (medstat-clean -> medstat-models / medstat-diagnostic / medstat-causal-meta -> medstat-report) with zero manual skill selection required. Operates autonomously or with interactive clinical proposals.
---

# medstat-master: Autonomous Biostatistical Orchestrator

The master intelligence layer for `medstat`. Ingests clinical spreadsheets and cohorts, profiles data geometry, infers study design, selects appropriate statistical methods, and orchestrates the downstream atomic skills (`medstat-clean`, `medstat-models`, `medstat-diagnostic`, `medstat-causal-meta`, `medstat-report`) autonomously.

---

## 1. When to Use This Skill

Activate **medstat-master** whenever:
- The user provides or points to a dataset (`.csv`, `.xlsx`, `.tsv`, `.parquet`) without specifying individual skills.
- The user asks: *"Analyze this data"*, *"What can we learn from this patient cohort?"*, *"Run statistical tests on this spreadsheet"*, or drops a dataset into the chat.
- End-to-end automated clinical pipelines from raw spreadsheet to publication-grade manuscript tables are needed.
- The user is unsure which statistical tests or modeling strategies conform to clinical research standards.

---

## 2. Core Operational Workflow

```
┌────────────────────────────────────────────────────────┐
│  Phase 1: Ingestion & Schema Profiling (Inspect Data)   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Phase 2: Clinical Study Design Inference & Routing     │
└──────────────────────────┬─────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
   [Path A: Direct Execution]   [Path B: Proposal First]
   (User goal clear / auto)     (Ambiguous / multi-path)
             │                           │
             │                           ▼
             │                  Render SAP & Align
             │                           │
             └─────────────┬─────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  Phase 3: Pipeline Orchestration & Execution            │
│  [medstat-clean] ──▶ [Analysis Skills] ──▶ [Report]    │
└────────────────────────────────────────────────────────┘
```

---

## 3. Phase 1: Ingestion & Autonomous Data Profiling

When a dataset is presented, inspect it before proposing or executing any models:

1. **Load & Inspect Metadata**:
   - File format (`.csv`, `.xlsx`, `.tsv`, `.parquet`).
   - Dimensions: Sample size $N$ (rows) and feature count $P$ (columns).
   - Column types: numeric continuous, integer count, binary indicators, categorical factors, dates/times, free text.

2. **Screen Clinical Invariants & Data Health**:
   - **Missingness Audit**: Calculate missing count and % per column.
   - **Outcome Candidate Identification**: Identify potential primary endpoints (e.g., mortality, readmission, sepsis, diagnosis). Check for forbidden text labels (`"Alive"`/`"Dead"`, `"Yes"`/`"No"`) that must be recoded to numeric `0/1`.
   - **Survival Features**: Detect paired time-to-event columns (`time`, `duration`, `days_to_event`) and event status columns (`status`, `death`, `censored`).
   - **Treatment / Exposure**: Detect binary or categorical intervention columns (`treatment`, `arm`, `drug`, `exposed`).
   - **Biomarkers & Diagnostic Tests**: Detect continuous scores, lab values, or point-of-care index tests paired with reference gold standards.
   - **Repeated Measures / Raters**: Detect cluster/subject IDs with multiple observations or paired device/rater evaluations.

---

## 4. Phase 2: Clinical Study Design Inference & Skill Routing

Map the data geometry and clinical context to one of the canonical clinical designs:

| Clinical Design Pattern | Detected Data Signature | Orchestrated Pipeline | Target Skill Set |
| :--- | :--- | :--- | :--- |
| **Type 1: Baseline Cohort & Descriptive** | Patient demographics, comorbidities, labs, group/arm comparison | Clean $\to$ Table 1 $\to$ Bivariate $\to$ Report | `medstat-clean`<br>`medstat-models`<br>`medstat-report` |
| **Type 2: Prognostic & Multivariable Risk** | Exposure/predictors + binary clinical outcome ($0/1$) | Clean $\to$ Table 1 $\to$ GLM/Firth Logistic $\to$ RCS Splines $\to$ E-value $\to$ Report | `medstat-clean`<br>`medstat-models`<br>`medstat-report` |
| **Type 3: Time-to-Event / Survival Cohort** | Follow-up time column + binary event indicator ($0/1$) | Clean $\to$ KM Curves $\to$ Cox PH + Schoenfeld $\to$ Firth Cox (if sparse) $\to$ Report | `medstat-clean`<br>`medstat-models`<br>`medstat-report` |
| **Type 4: Diagnostic Accuracy & Biomarker** | Continuous/ordinal index test + binary gold standard | Clean $\to$ 2×2 Contingency (Wilson CI) $\to$ ROC + DeLong AUC $\to$ DCA Net Benefit $\to$ Report | `medstat-clean`<br>`medstat-diagnostic`<br>`medstat-report` |
| **Type 5: Observational Causal Inference** | Non-randomized treatment indicator + baseline confounders | Clean $\to$ PSM Matching (caliper 0.2×SD) $\to$ Love Plot (SMD < 0.10) $\to$ Outcome Model $\to$ Report | `medstat-clean`<br>`medstat-causal-meta`<br>`medstat-models`<br>`medstat-report` |
| **Type 6: Agreement & Reliability** | Paired device measurements OR subject ID + multiple raters | Clean $\to$ Bland-Altman LoA (Carkeet CIs) OR Pure-SciPy ICC (all 6 forms) $\to$ Report | `medstat-clean`<br>`medstat-causal-meta`<br>`medstat-report` |
| **Type 7: Multi-Study Meta-Analysis** | Effect sizes, SEs / variance, study labels, sample sizes | Fixed/Random Effects (DerSimonian-Laird) $\to$ Forest Plot $\to$ Egger's Test $\to$ Report | `medstat-causal-meta`<br>`medstat-report` |

*See [references/study-design-decision-tree.md](references/study-design-decision-tree.md) for detailed clinical heuristics and decision thresholds.*

---

## 5. Dual Execution Modes: Direct Execution vs Proposal First

The agent supports **both modes** seamlessly depending on user intent and context:

### Mode A: Direct Execution (No Proposal Required)
**When to use**:
- The user provides an explicit prompt (e.g., *"Fit a logistic model predicting 30-day mortality adjusting for age and sex, output NEJM table"*).
- The user says *"Analyze this dataset end-to-end"* or asks for quick, immediate results.
- The pipeline has a single obvious gold-standard path.

**Action**:
1. Execute data audit and clean automatically with defensible defaults (Little's MCAR check, MICE if MAR/moderate missingness or complete-case if MCAR justified).
2. Run baseline Table 1 and primary model.
3. Render publication-ready tables and narrative.
4. Provide the complete result along with a transparent summary of decisions made.

### Mode B: Statistical Analysis Proposal (SAP First)
**When to use**:
- The user simply uploads a dataset without specifying the clinical question.
- Multiple competing analytical paths exist (e.g., Propensity Score Matching vs Multivariable Regression adjustment; dichotomizing a continuous biomarker vs spline curve).
- Missingness exceeds 20% or non-trivial clinical assumptions must be aligned.

**Action**:
Present a concise, structured 1-page **Statistical Analysis Proposal (SAP)**:
```markdown
### 📋 Proposed Statistical Analysis Plan (SAP)
- **Primary Objective**: [Inferred clinical question]
- **Identified Variables**:
  - Outcome: `mortality_30d` (binary 0/1)
  - Primary Exposure: `tx_group` (Treatment A vs B)
  - Confounders: `age`, `sex`, `sofa_score`, `lactate`
- **Data Quality & Missingness**:
  - Missingness: 8.2% across creatinine and BMI; Little's MCAR p=0.18.
  - Recommended Strategy: MICE (5 imputations) with sample flow audit.
- **Recommended Analysis Pipeline**:
  1. Baseline Table 1 stratified by `tx_group` with Standardized Mean Differences (SMDs).
  2. Multivariable Logistic Regression with Firth penalization if event rate is sparse (<10 EPV).
  3. Non-linear dose-response spline for continuous `lactate`.
  4. VanderWeele E-value sensitivity analysis for unmeasured confounding.
  5. Publication-grade Table formatted to NEJM style.
```
*Prompt the user: "Would you like me to proceed with this plan, or would you like to adjust any variables or methods?"*

*See [references/autonomous-sap-template.md](references/autonomous-sap-template.md) for full YAML and Markdown SAP templates.*

---

## 6. Execution & Skill Chaining Protocol

When executing the pipeline, strictly enforce the following sequence across downstream tools:

### Step 1: Clean & Standardize (`medstat-clean`)
- Run missingness audit:
  ```bash
  uv run medstat clean --data <raw_file.csv> --audit-only --audit-out audit.json
  ```
- Recode any text outcome columns (`"Dead"` $\to$ `1`, `"Alive"` $\to$ `0`) so downstream tools never receive text labels.
- Execute cleaning with explicit strategy and documented clinical justification:
  ```bash
  uv run medstat clean --data <raw_file.csv> --strategy <mice|complete-case|knn> \
    --missing-justification "<rationale>" --output clean_cohort.csv --audit-out retention.json
  ```

### Step 2: Baseline Descriptive & Balance (`medstat-models` / `medstat-causal-meta`)
- Generate Table 1 with SMDs:
  ```bash
  uv run medstat table1 --data clean_cohort.csv --group <exposure_col> --vars "<covariates>" --output table1.json
  ```

### Step 3: Core Statistical & Causal Modeling
- **Multivariable Regression / Survival**:
  ```bash
  # Logistic / Firth
  uv run medstat model fit --data clean_cohort.csv --type logistic --y <outcome> --x "<covariates>" --firth --output model_results.json
  # Cox Survival
  uv run medstat model fit --data clean_cohort.csv --type cox --time <duration> --event <status> --x "<covariates>" --schoenfeld --output cox_results.json
  ```
- **Diagnostic Testing**:
  ```bash
  uv run medstat diag --data clean_cohort.csv --gold-standard <disease> --test-col <biomarker> --roc --dca --output diag_results.json
  ```
- **Causal PSM Matching**:
  ```bash
  uv run medstat causal psm --data clean_cohort.csv --treatment <tx> --covariates "<vars>" --caliper 0.2 --output psm_results.json
  ```
- **Inter-Rater Agreement & ICC**:
  ```bash
  uv run medstat agreement icc --data clean_cohort.csv --targets <subject_id> --raters <rater_id> --ratings <score> --output icc_results.json
  ```

### Step 4: Publication Reporting (`medstat-report`)
- Format model results into journal-styled HTML tables (NEJM, JAMA, APA 7) and compile automated methods narratives and checklist audits (CONSORT, STROBE, TRIPOD+AI):
  ```bash
  uv run medstat report --results model_results.json --style nejm --narrative --output publication_table.html
  ```

---

## 7. Mandatory Clinical Governance & Safety Rules

All operations coordinated by **medstat-master** must strictly follow these rules:

1. **Strict Numeric 0/1 Endpoints**: Outcomes must be numeric `0` and `1` (`1 = Event`, `0 = Non-event`). Text outcomes (`"Yes"/"No"`, `"Dead"/"Alive"`) must be converted in Step 1.
2. **Never Silent Deletion**: `MissingStrategyRequiredError` is fatal. Always pass `--strategy` and track sample attrition:
   $$N_{\text{initial}} \longrightarrow N_{\text{excluded}} \longrightarrow N_{\text{analyzed}}$$
3. **Wilson Score Confidence Intervals**: All binomial proportions (Sensitivity, Specificity, PPV, NPV) must use Wilson score intervals.
4. **DeLong Covariance**: ROC AUC standard errors and paired AUC comparisons must use DeLong variance.
5. **Austin (2009) PSM Standard**: Propensity score caliper must default to $0.2 \times \text{SD}(\text{logit } e)$; post-match balance requires $\text{SMD} < 0.10$.
6. **Zero-PHI Compliance**: Never output raw patient identifiers (HN, Citizen ID, Name, Phone). Anonymize all sample rows in transcripts.

---

## 8. Completion Checklist

- [ ] Dataset ingested, dimensions confirmed, and column data types verified.
- [ ] Clinical study design inferred and mapped to correct pipeline.
- [ ] If ambiguous, SAP proposal presented to user; if direct, executed without delay.
- [ ] Missingness audited; Little's MCAR assessed; imputation/clean strategy justified.
- [ ] Binary endpoints recoded strictly to numeric `0/1`.
- [ ] Sample retention flow ($N_{\text{initial}} \to N_{\text{excluded}} \to N_{\text{analyzed}}$) tracked.
- [ ] Primary statistical model executed conforming to clinical standards (Wilson CI, DeLong, Firth, or PSM).
- [ ] Results compiled into publication-grade table (NEJM/JAMA) and methods narrative.

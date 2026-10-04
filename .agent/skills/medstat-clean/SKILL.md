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
4. **Binary Endpoint Standardization**: Supported binary and survival workflows require numeric outcomes and strictly reject text outcomes. Explicitly recode binary event-status endpoints to numeric `0/1` (`1 = Event`, `0 = Non-event`) before model execution; event direction must not be inferred from arbitrary text labels. Multicategory labels and survival follow-up time columns must be preserved in cleaned data, requiring separate documented model-input encoding for compatible multicategory models without overwriting original labels. Ordinal clinical outcomes (e.g., mRS 0–6, GCS 3–15, NYHA I–IV) must be preserved as ordered integers in the cleaned dataset rather than forced into binary dichotomization without clinical justification, ensuring compatibility with Type 2b proportional odds workflows.
5. **Strict Ban on Imputing Primary Outcomes**: Missing values in primary clinical endpoints (e.g., mortality, disease event, relapse) must never be blindly imputed using MICE or KNN. Drop missing outcome cases under audited retention flow ($N_{\text{initial}} \to N_{\text{excluded}} \to N_{\text{analyzed}}$) with documented clinical rationale, or perform prespecified sensitivity analysis; never fabricate clinical outcomes.
6. **Clinical Extreme Value Protection**: Tukey IQR fences ($1.5 \times \text{IQR}$) are purely statistical heuristics. Never delete or winsorize extreme values that represent genuine physiological crises (e.g., Troponin in massive STEMI, Lactate in septic shock) without confirming physiological impossibility (e.g., SBP = 999, Age = 250).

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
  - **20% – 40% (High)**: Substantial missingness; multiple imputation may be appropriate if missingness mechanism, imputation model adequacy, and analysis objectives support it. Mandatory sensitivity analysis selected based on estimand and assessed mechanism.
  - **> 40% (Critical)**: High risk of residual bias; evaluate whether variable can be reliably imputed or retained.
- Check Little's MCAR test: $p > 0.05$ fails to reject MCAR (insufficient evidence against MCAR); $p \le 0.05$ provides evidence against MCAR (departures from MCAR).
- Consult [references/missing-data-mechanisms.md](references/missing-data-mechanisms.md) for mechanism selection criteria.

### Step 2: Execute Clinically Justified Strategy

Run cleaning with an approved strategy (`complete-case`, `mice`, `knn`, `indicator`) and documented rationale:

```bash
# Complete-case analysis (MCAR justified)
medstat clean --data <dataset.csv> \
  --strategy complete-case \
  --missing-justification "Missing lab values attributable to random specimen handling failures unrelated to patient severity or the target estimand; MCAR plausible by clinical mechanism. Supporting: Little's test non-significant (p=0.42) and overall missingness <5%" \
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

Detect and handle extreme values using `--outlier-action`, `--outlier-cols`, and `--iqr-multiplier`. **Note**: Commands below are templates; replace all placeholders (`<...>`) with valid values before execution. Because Tukey fences are purely statistical rather than domain-specific physiological plausibility boundaries, always explicitly select variables via `--outlier-cols` and perform clinical review before applying destructive actions (`winsorize`, `cap`, or `remove`), rather than running them across all numeric columns without selection:

```bash
# Winsorize extreme values to Tukey IQR fences (Q1 - 1.5*IQR, Q3 + 1.5*IQR) for explicitly selected variables
medstat clean --data <dataset.csv> \
  --strategy <strategy> \
  --missing-justification "[audit-based clinical justification selected after missingness audit]" \
  --outlier-action winsorize \
  --outlier-cols <variable_1> \
  --iqr-multiplier 1.5 \
  --output clean_winsorized.csv --audit-out retention.json

# Or remove statistical outliers on explicitly selected columns with audited sample flow tracking
medstat clean --data <dataset.csv> \
  --strategy <strategy> \
  --missing-justification "[audit-based clinical justification selected after missingness audit]" \
  --outlier-action remove \
  --outlier-cols <variable_1> \
  --iqr-multiplier 3.0 \
  --output clean_no_outliers.csv --audit-out retention.json
```

- `--outlier-action [flag|remove|winsorize|cap]`:
  - `winsorize` / `cap`: Clamps extreme values to the Tukey fences ($Q_1 - k \times \text{IQR}$, $Q_3 + k \times \text{IQR}$) across explicitly selected numeric columns while retaining the full cohort. Requires explicit variable selection (`--outlier-cols`) and clinical review before capping.
  - `remove`: Excludes rows outside Tukey fences in selected numeric columns and logs them into the sample retention tracker ($N_{\text{excluded}}$). This is a statistical rather than a physiological-plausibility check; explicit column selection (`--outlier-cols`) and clinical verification are required before removal.
  - `flag`: Records per-column counts in `outlier_counts` without marking rows or changing values.
- `--outlier-cols`: Explicit numeric column(s) to evaluate and transform (e.g. `--outlier-cols <biomarker>`).
- `--iqr-multiplier`: Tukey's multiplier $k$ (default: `1.5` for inner fences, `3.0` for extreme outer fences).
- Standardize explicitly binary event-status fields to numeric `0/1` (`1 = Event`, `0 = Non-event`), ensuring survival follow-up duration and multicategory endpoints remain intact; never forward text outcomes to modeling.

### Step 4: Verify Sample Retention Flow

Verify that the output contains the audited sample retention tracker:
- $N_{\text{initial}}$: Total enrolled patients before exclusions.
- $N_{\text{excluded}}$: Rows removed with categorized rationale.
- $N_{\text{analyzed}}$: Final analytic cohort size matching downstream model inputs.

---

## Adaptive Python Scripting Protocol (ปรับแต่งสคริปต์ทำความสะอาดข้อมูลตามข้อมูลจริง)

> **Core Philosophy**: Never execute rigid canned scripts that make naive assumptions about file structure. The agent is empowered with full autonomy to inspect, write, adapt, and run Python scripts (`scratch/clean.py`) tailored to the specific layout, encodings, and clinical requirements of the raw dataset (e.g. multi-row headers, notes, Thai locale strings, embedded dashboard summary cards, or side-by-side tables).
>
> 🔒 **Subprocess & Script Execution Safety**:
> When generating and running analysis scripts (`scratch/clean.py`), enforce execution controls: disable shell/subprocess access, limit file reads strictly to the designated dataset and referenced prototype/core modules, limit file writes strictly to scratch and designated output paths, and ensure no access to credentials or environment secrets. Require explicit user confirmation if the runtime cannot enforce these sandbox controls.
>
> ⚠️ **Mandatory Directive — ต้องดู Script ต้นแบบประกอบเสมอ (Review Prototype Scripts First)**:
> แม้จะให้อิสระ Agent ในการเขียนและปรับ Python Script เองตามสภาพข้อมูลจริง แต่ **Agent ต้องเปิดดูและอ้างอิงสคริปต์ต้นแบบ (Prototype Scripts)** หรือศึกษาการคำนวณในโมดูลแกนกลาง `src/medstat/clean/` เสมอ เพื่อยึดมาตรฐานความถูกต้องทางชีวสถิติการแพทย์:
> - **การตรวจสอบการสูญหายและการทดสอบ MCAR**: ดูการวิเคราะห์ missing patterns และ Little's MCAR test จาก `src/medstat/data/missing.py` และ `src/medstat/data/quality.py`
> - **การจัดการค่าสูญหายและการทำความสะอาด**: ดูการทำ Imputation (MICE, KNN, Complete-case) และการกรองข้อมูลจาก `src/medstat/data/clean.py` และ `src/medstat/data/missing.py`
> - **การจัดการค่าผิดปกติ (Outliers & Tukey Fences)**: ดูการคำนวณ Tukey IQR fences ($Q_1 - 1.5\text{IQR}, Q_3 + 1.5\text{IQR}$) สำหรับการ winsorize / cap จาก `src/medstat/data/clean.py`
> - **การติดตามการคัดเข้า-ออกกลุ่มตัวอย่าง**: ดูการบันทึก $N_{\text{initial}} \to N_{\text{excluded}} \to N_{\text{analyzed}}$ จาก `src/medstat/data/retention.py`
>
> **วงจรการทำงานของ Agent**:
> `[1. สำรวจโครงสร้างข้อมูลดิบ] ──▶ [2. ดูสคริปต์ต้นแบบเพื่อยึดหลักชีวสถิติ] ──▶ [3. ปรับโค้ดและขัดเกลาข้อมูล]`

### Master Prototype Script for Clinical Cleaning (สคริปต์ต้นแบบมาตรฐาน)

Agent ควรนำโครงสร้างของสคริปต์ต้นแบบนี้ไปปรับแต่งลงใน workspace (เช่น `scratch/clean.py`) ให้เข้ากับโครงสร้างไฟล์จริง:

```python
import numbers
import numpy as np
import pandas as pd
from scipy import stats

# 1. LOAD & INSPECT RAW STRUCTURE
# ตรวจสอบและปรับ skiprows / header / usecols ให้แยกเฉพาะ cohort ผู้ป่วยจริง
# ตัดแถวหัวตารางที่เป็นคำอธิบาย หรือคอลัมน์ Dashboard สรุปผลด้านข้างออก
df_raw = pd.read_excel("dataset.xlsx", skiprows=2)  # ปรับ skiprows ตามจริง
n_initial = len(df_raw)
print(f"Loaded raw dataset: N = {n_initial}")

# 2. STANDARDIZE ENDPOINTS & LABELS
# บังคับใช้ Numeric 0/1 สำหรับ Binary Outcome เสมอ (1 = Event, 0 = Non-event)
# ตัวอย่าง: df['outcome'] = df['raw_outcome'].map({'Positive': 1, 'Negative': 0})
df = df_raw.copy()
# Non-binary numeric outcomes are NEVER inferred as ordinal from their values alone.
# Declare the ordinal type and its category mapping explicitly (from the protocol / data dictionary):
outcome_is_ordinal = False  # Set True ONLY when the SAP defines the endpoint as ordinal
outcome_ordinal_mapping = None  # e.g. {0: 'Home', 1: 'Ward', ..., 6: 'Death'} (ordered codes -> labels)
is_ordered_cat = isinstance(df['outcome'].dtype, pd.CategoricalDtype) and df['outcome'].dtype.ordered
if not is_ordered_cat:
    outcome_vals = set(df['outcome'].dropna().unique())
    if not outcome_vals.issubset({0, 1}):
        if not (outcome_is_ordinal and outcome_ordinal_mapping):
            raise ValueError(
                f"Primary endpoint contains non-binary values {outcome_vals}. "
                "Must be strictly binary {0, 1}, an ordered Categorical, or an explicitly declared "
                "ordinal endpoint (outcome_is_ordinal=True with outcome_ordinal_mapping)."
            )
        # Support code->label ({0: 'Home'}) or label->code ({'Home': 0}) mapping
        label_to_code = {
            v: k for k, v in outcome_ordinal_mapping.items()
            if isinstance(k, numbers.Integral) and not isinstance(k, bool) and isinstance(v, str)
        }
        if not label_to_code:
            label_to_code = {
                k: v for k, v in outcome_ordinal_mapping.items()
                if isinstance(v, numbers.Integral) and not isinstance(v, bool)
            }
        valid_keys = (
            set(outcome_ordinal_mapping.keys()) | set(label_to_code.keys())
            | {c for c in label_to_code.values() if isinstance(c, numbers.Integral) and not isinstance(c, bool)}
        )
        unmapped = outcome_vals - valid_keys
        if unmapped:
            raise ValueError(f"Outcome values {unmapped} are not defined in outcome_ordinal_mapping.")
        # Recode mapped text labels to protocol-defined integer codes
        if label_to_code and any(isinstance(v, str) for v in outcome_vals):
            df['outcome'] = df['outcome'].replace(label_to_code)
        if not df['outcome'].isnull().any():
            # Validate that mapped ordinal codes are integral and reject fractional values
            non_int = [
                v for v in df['outcome'].dropna().unique()
                if not (isinstance(v, (numbers.Integral, np.integer)) and not isinstance(v, (bool, np.bool_)))
                and not (isinstance(v, (float, np.floating)) and float(v).is_integer())
            ]
            if non_int:
                raise ValueError(
                    f"Mapped ordinal outcome contains non-integral codes {non_int}. "
                    "Ordinal endpoints must be integral codes without fractional values."
                )
            df['outcome'] = df['outcome'].astype(int)


# 3. MISSINGNESS AUDIT & SAMPLE RETENTION FLOW (SCAFFOLD TEMPLATE)
# NOTE: This block is an illustrative scaffold template that must be adapted to the specific study protocol;
# do NOT run as an unverified blanket workflow without addressing missing covariates.
missing_audit = df.isnull().mean()
print("Missingness audit per variable:\n", missing_audit[missing_audit > 0])

from medstat.data.retention import SampleFlowTracker
tracker = SampleFlowTracker(initial_n=n_initial, initial_name="Initial Enrolled Cohort")

# Verify protocol justification before complete-case outcome exclusion:
# An explicit verified-protocol flag and rationale must be established before dropping rows with missing outcome:
protocol_permits_outcome_exclusion = False  # Default False: requires explicit study protocol / SAP justification
protocol_rationale = None  # Provide documented rationale before excluding missing outcomes (e.g. 'Prespecified complete-case analysis')

if df['outcome'].isnull().any():
    if not (protocol_permits_outcome_exclusion and protocol_rationale and str(protocol_rationale).strip()):
        raise ValueError(
            "Missing values detected in primary outcome, but study protocol does not verify exclusion criteria "
            "with a documented rationale (protocol_permits_outcome_exclusion=True and non-empty protocol_rationale required). "
            "Clarify with PI/SAP."
        )
    df_clean = df.dropna(subset=['outcome']).copy()
    tracker.record_stage(
        stage_name="Primary Outcome Ascertainment",
        n_remaining=len(df_clean),
        reason=f"Excluded missing primary outcome: {protocol_rationale}",
    )
else:
    df_clean = df.copy()

# Address remaining missing values in covariates per study-specific strategy (e.g. MICE, indicator, or documented complete-case)
# before declaring the cohort clean and persisting.
unresolved_missing = df_clean.isnull().sum()
if unresolved_missing.any():
    print(f"Warning: Covariates with unresolved missing values:\n{unresolved_missing[unresolved_missing > 0]}")
    # Apply explicit imputation (e.g. MICE) or documented complete-case per study design

n_analyzed = len(df_clean)
n_excluded = n_initial - n_analyzed
print(f"Sample Retention Flow: Initial={n_initial} -> Excluded={n_excluded} -> Analyzed={n_analyzed}")

# 4. OUTLIER HANDLING (Tukey IQR Fences for Explicitly Selected Variables)
def winsorize_tukey(series, k=1.5):
    """Winsorize extreme values to Tukey fences [Q1 - k*IQR, Q3 + k*IQR]"""
    s = series.dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - k * iqr
    upper_fence = q3 + k * iqr
    return series.clip(lower=lower_fence, upper=upper_fence)

# ตัวอย่าง: บังคับใช้เฉพาะคอลัมน์ที่ผ่านการประเมินทางคลินิกแล้ว
# df_clean['sbp_winsorized'] = winsorize_tukey(df_clean['sbp'], k=1.5)

# บันทึกข้อมูลที่พร้อมสำหรับการวิเคราะห์ และ persist retention flow ควบคู่กัน
# Standardize validated outcome codes to integer dtype before export
if isinstance(df_clean['outcome'].dtype, pd.CategoricalDtype) and df_clean['outcome'].dtype.ordered:
    cats = df_clean['outcome'].cat.categories
    is_int_coded = all(isinstance(c, numbers.Integral) and not isinstance(c, bool) for c in cats)
    if is_int_coded:
        # Categories are already integer codes; preserve after validating against mapping if provided
        if outcome_ordinal_mapping:
            expected_codes = {
                k if isinstance(k, numbers.Integral) and not isinstance(k, bool) else v
                for k, v in outcome_ordinal_mapping.items()
                if (isinstance(k, numbers.Integral) and not isinstance(k, bool))
                or (isinstance(v, numbers.Integral) and not isinstance(v, bool))
            }
            if not set(cats).issubset(expected_codes):
                raise ValueError(f"Categorical codes {set(cats) - expected_codes} not recognized in outcome_ordinal_mapping.")
        df_clean['outcome'] = df_clean['outcome'].astype(int)
    elif outcome_ordinal_mapping:
        # Categories are text labels; convert via inverse label->code mapping
        label_to_code = {
            v: k for k, v in outcome_ordinal_mapping.items()
            if isinstance(k, numbers.Integral) and not isinstance(k, bool) and isinstance(v, str)
        }
        if not label_to_code:
            label_to_code = {
                k: v for k, v in outcome_ordinal_mapping.items()
                if isinstance(v, numbers.Integral) and not isinstance(v, bool) and isinstance(k, str)
            }
        unmapped_cats = [c for c in cats if c not in label_to_code]
        if unmapped_cats:
            raise ValueError(f"Declared categories {unmapped_cats} are not defined in outcome_ordinal_mapping.")
        mapped_codes = [label_to_code[c] for c in cats]
        if len(set(mapped_codes)) != len(mapped_codes):
            raise ValueError(f"outcome_ordinal_mapping codes must be unique across categories (got {dict(zip(cats, mapped_codes))}).")
        if any(b <= a for a, b in zip(mapped_codes, mapped_codes[1:])):
            raise ValueError(
                "outcome_ordinal_mapping codes must strictly increase in declared category order "
                f"(got {dict(zip(cats, mapped_codes))}). Non-increasing mapping would invert ordinal direction."
            )
        df_clean['outcome'] = df_clean['outcome'].map(label_to_code).astype(int)
    else:
        df_clean['outcome'] = df_clean['outcome'].cat.codes.astype(int)
else:
    df_clean['outcome'] = df_clean['outcome'].astype(int)
df_clean.to_csv("clean_cohort.csv", index=False)
with open("sample_retention_flow.json", "w") as f:
    f.write(tracker.to_json())
```

The agent may freely incorporate `medstat` modules (e.g. `from medstat.clean.missing import audit_missingness`, `from medstat.clean.outliers import winsorize_outliers`) or standard libraries as appropriate.

---

## Completion Criteria

- [ ] Missingness audit executed and reviewed across all clinical variables.
- [ ] Explicit missing data strategy chosen with documented clinical justification.
- [ ] Binary and survival event endpoints standardized to numeric 0/1 (1 = Event).
- [ ] Cleaned dataset written to `--output` path with zero unexpected `NaN` cells.
- [ ] Sample retention flow metadata recorded with explicit initial, excluded, and analyzed counts.

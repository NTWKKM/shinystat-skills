---
name: medstat-master
description: Master clinical biostatistics orchestrator. Ingests raw clinical data (CSV, XLSX, TSV, .parquet), inspects file layout and schema, identifies clinical study design, and guides the agent to adapt Python scripts for data cleaning, statistical modeling, and publication-grade reporting (NEJM/JAMA) tailored to the specific dataset.
---

# medstat-master: Autonomous Biostatistical Orchestrator

The master intelligence layer for clinical biostatistics. Ingests clinical spreadsheets and cohorts, profiles data geometry, infers study design, selects appropriate statistical methods, and guides the agent to perform data cleaning, analysis, modeling, and reporting using **adaptive Python scripting tailored to the specific dataset** rather than rigid, brittle canned scripts.

---

## 1. When to Use This Skill

Activate **medstat-master** whenever:
- The user provides or points to a dataset (`.csv`, `.xlsx`, `.tsv`, `.parquet`).
- The user asks: *"Analyze this data"*, *"What can we learn from this patient cohort?"*, *"Run statistical tests on this spreadsheet"*, or uploads a clinical dataset.
- Real-world clinical spreadsheets with messy layouts (multiple title rows, embedded notes, side-by-side summary tables, Thai dates/categories) need flexible parsing and robust biostatistical analysis.
- End-to-end clinical pipelines from raw spreadsheet to publication-grade manuscript tables are needed.

---

## 2. Core Operational Workflow

```
┌────────────────────────────────────────────────────────┐
│  Phase 1: Ingestion & Adaptive Layout Inspection       │
│  - Inspect sheet structure, title lines, header rows   │
│  - Separate raw cohort from side-by-side tables        │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  Phase 2: Study Design & Variable Mapping (ระเบียนวิธี)  │
│  - Identify outcome, exposure, confounders, time/event │
│  - Map to clinical design (Types 1–7)                  │
└──────────────────────────┬─────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
   [Mode A: Direct Analysis]   [Mode B: SAP Proposal First]
   (Clear objective / auto)    (Ambiguous / competing paths)
             │                           │
             │                           ▼
             │                  Present SAP & Align
             │                           │
             └─────────────┬─────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  Phase 3: Review Prototype Scripts & Adapt Analysis    │
│  - ดูสคริปต์ต้นแบบ (Prototype Scripts) เพื่อยึดหลักสถิติ │
│  - ปรับเขียนโค้ด Python ให้เข้ากับข้อมูลจริงและรัน        │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  Phase 4: Publication-Grade Reporting & Clinical Story │
│  - NEJM / JAMA styled tables & actionable insights     │
└────────────────────────────────────────────────────────┘
```

---

## 3. Phase 1: Ingestion & Adaptive Layout Inspection

Real-world medical spreadsheets rarely arrive as clean, single-header tables. Before modeling, the agent must inspect the file layout and adapt the loading logic:

### 1. Identify Non-Standard Layouts
- **Metadata & Notes**: Check if rows 1–3 contain report titles, hospital department headers, or clinical diagnostic criteria (e.g. *"ข้อมูลผู้ป่วยทั้งหมด...", "หมายเหตุ : Diag ICD-10 = ..."*).
- **Header Detection**: Locate the actual column header row (e.g. `header=2` or `header=3` in `pd.read_excel` / `pd.read_csv`).
- **Side-by-Side Summary Tables**: Clinical exports often place pivot tables, descriptive summaries, or audit boxes on the right-hand columns (e.g., columns `BS` to `BW`). Isolate the patient-level microdata and avoid parsing summary tables as patient features.
- **Thai & Locale Encodings**: Handle Thai text categories (`ช`/`ญ`, `บัตรทอง`, `มาเอง`, `รับ Refer`), Thai Buddhist Era years (พ.ศ. 2566–2568 $\to$ CE 2023–2025), and encodings (`utf-8`, `utf-8-sig`, `cp874`, `tis-620`).

### 2. Inspection Script Pattern
Write and run a quick inspection script to ground the layout:
```python
import pandas as pd

# Inspect raw header lines to locate the real table
df_peek = pd.read_excel("data.xlsx", header=None, nrows=10) # or read_csv
for idx, row in df_peek.iterrows():
    print(f"Row {idx}: {row.dropna().tolist()[:6]}")
```

---

## 4. Phase 2: Biostatistical Methodology & Study Design Guidelines (ระเบียนวิธีทางสถิติ)

The agent should map the clinical question and data geometry to the appropriate study design:

| Clinical Design Pattern | Detected Data Signature | Statistical Methodology & Tests | Target Reporting |
| :--- | :--- | :--- | :--- |
| **Type 1: Baseline Cohort & Descriptive** | Patient demographics, comorbidities, labs, group comparison | Table 1: Mean ± SD (t-test) or Median [IQR] (Mann-Whitney U); n (%) (Chi-Square / Fisher); SMD | Baseline Table 1 with SMDs & p-values |
| **Type 2: Prognostic & Multivariable Risk** | Binary clinical outcome ($0/1$) + clinical predictors | Multivariable Logistic Regression: Odds Ratios (OR), 95% CI, p-values. Firth penalization if sparse events/separation. RCS splines for non-linear continuous markers. E-value for unmeasured confounding. | Multivariable Regression Table & Forest Plot |
| **Type 3: Time-to-Event / Survival Cohort** | Follow-up time column + binary event indicator ($1=\text{Event}, 0=\text{Censored}$) | Kaplan-Meier survival curves, Log-rank test, Cox Proportional Hazards (HR with 95% CI), Schoenfeld residual test for PH assumption. Firth Cox if zero events in subgroup. | KM Curves & Cox PH Table |
| **Type 4: Diagnostic Accuracy & Biomarker** | Continuous/ordinal test score + binary gold standard | 2×2 Contingency (Sensitivity, Specificity, PPV, NPV, LR+, LR- with Wilson score 95% CIs), ROC curve, AUC with DeLong 95% CI, Vickers Decision Curve Analysis (DCA). | Diagnostic Accuracy & ROC Table |
| **Type 5: Observational Causal Inference** | Non-randomized treatment indicator + baseline confounders | Propensity Score Matching (PSM, Austin 2009 caliper $0.2 \times \text{SD}(\text{logit } e)$), Love plot (post-match $\text{SMD} < 0.10$), outcome model on matched cohort. | Covariate Balance & Matched Effect |
| **Type 6: Agreement & Reliability** | Paired device measurements OR subject ID + multiple raters | Bland-Altman Limits of Agreement with large-sample approximate CIs, Intraclass Correlation Coefficient (ICC forms 1, 2, 3), Cohen's / Fleiss' Kappa. | Agreement Plot & Reliability Table |
| **Type 7: Multi-Study Meta-Analysis** | Effect sizes (log OR, HR, MD) with SEs across studies | DerSimonian-Laird random effects ($\tau^2, I^2$), Forest plot data, Egger's test for funnel asymmetry (if continuous $k \ge 10$). | Forest Plot & Meta-Analysis Table |

---

## 5. Dual Execution Modes: Direct Execution vs Proposal First

### Mode A: Direct Execution (Immediate Analysis)
**When to use**:
- The user provides an explicit instruction (e.g. *"Analyze drug prevalence and find risk factors for positive drug test in this cohort"*).
- The user wants immediate results or says *"Analyze this dataset"*.
- The clinical question has an obvious primary outcome and exposure.

**Action**:
1. Inspect file layout and extract clean patient-level data.
2. Review the prototype scripts below to anchor correct statistical formulas.
3. Adapt the Python script to execute data cleaning, Table 1, and multivariable regression for this dataset.
4. Output publication-grade tables and clinical summary.

### Mode B: Statistical Analysis Proposal (SAP First)
**When to use**:
- Dataset has multiple potential outcomes with competing research questions.
- User requests a plan first or complex method trade-offs exist (e.g. PSM vs multivariable adjustment).

**Action**:
Present a concise 1-page SAP covering:
- **Primary Objective & Estimand**
- **Identified Variables** (Outcome, Exposure, Covariates)
- **Data Cleaning Strategy & Sample Retention Flow** ($N_{\text{initial}} \to N_{\text{excluded}} \to N_{\text{analyzed}}$)
- **Proposed Statistical Models & Diagnostics**
- **Target Journal Style** (NEJM / JAMA)

---

## 6. Phase 3: Adaptive Python Scripting Protocol (ปรับแต่งสคริปต์ตามข้อมูลจริง)

> **Core Philosophy**: Never execute rigid canned scripts that make naive assumptions about file structure. The agent is empowered with full autonomy to write, adapt, and run Python scripts (`scratch/analyze.py`) tailored to the specific columns, encodings, and clinical objectives of the ingested dataset.
>
> ⚠️ **Mandatory Directive — ต้องดู Script ต้นแบบประกอบเสมอ (Review Prototype Scripts First)**:
> แม้จะให้อิสระ Agent ในการเขียนและปรับ Python Script เองตามสภาพข้อมูลจริง แต่ **Agent ต้องเปิดดูและอ้างอิงสคริปต์ต้นแบบ (Prototype Scripts)** ที่ระบุไว้ในส่วนนี้ หรือศึกษาการคำนวณในโมดูลแกนกลาง `src/medstat/` เสมอ เพื่อยึดมาตรฐานความถูกต้องทางชีวสถิติการแพทย์:
> - **สถิติเปรียบเทียบและการทดสอบสมมติฐาน**: ดูการจัดกลุ่มตัวแปรต่อเนื่อง (Normality test, Mean ± SD vs Median [IQR], t-test vs Mann-Whitney U) และตัวแปรกลุ่ม (Chi-Square vs Fisher's exact) จาก `src/medstat/reporting/table1.py` และ `src/medstat/stats/bivariate.py`
> - **การสร้างแบบจำลอง Multivariable**: ดูการคำนวณ Adjusted Odds Ratio ($\exp(\beta)$), 95% Profile/Wald Confidence Intervals, และ Firth penalized likelihood จาก `src/medstat/models/glm.py` และ `src/medstat/models/firth.py`
> - **การประเมินการวินิจฉัยและ ROC**: ดูสูตร Wilson Score CI สำหรับ Sens/Spec และ DeLong 95% CI สำหรับ AUC จาก `src/medstat/diagnostic/accuracy.py` และ `src/medstat/diagnostic/roc.py`
> - **การจัดรูปแบบตารางมาตรฐานวารสาร**: ดูแม่แบบตาราง 3 เส้น (Three-horizontal-rule style) ไม่มีเส้นแนวตั้ง จาก `src/medstat/reporting/tables.py`
>
> **วงจรการทำงานของ Agent**:
> `[1. สำรวจโครงสร้างข้อมูลจริง] ──▶ [2. ดูสคริปต์ต้นแบบเพื่อยึดหลักสถิติ] ──▶ [3. ปรับโค้ดให้เข้ากับข้อมูลและรัน]`

### Master Prototype Script (สคริปต์ต้นแบบมาตรฐาน)

Agent ควรนำโครงสร้างและฟังก์ชันของสคริปต์ต้นแบบนี้ไปปรับแต่งลงใน workspace (เช่น `scratch/analyze.py`) ให้ตรงกับชื่อคอลัมน์ เงื่อนไขคัดกรอง และเป้าหมายการวิจัยของข้อมูลจริง:

```python
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ==============================================================================
# สคริปต์ต้นแบบ: การนำเข้า ขัดเกลาข้อมูล วิเคราะห์ Table 1 และ Multivariable Model
# ==============================================================================

# 1. LOAD & ISOLATE PATIENT COHORT
# ตรวจสอบและปรับ skiprows / header ให้ตรงกับแถวข้อมูลจริง
# ตัดคอลัมน์ตารางสรุปด้านข้าง (เช่น คอลัมน์ BS เป็นต้นไป) ออก ให้เหลือเฉพาะข้อมูลคนไข้
df = pd.read_excel("dataset.xlsx", skiprows=2)  # ปรับ skiprows ตามจริง

# 2. STANDARDIZE ENDPOINTS & SAMPLE FLOW TRACKING
# บังคับใช้ Numeric 0/1 เสมอ (1 = Event / Case, 0 = Non-event / Control)
# ตัวอย่าง: df['outcome'] = df['raw_outcome'].map({'Positive': 1, 'Negative': 0})
n_initial = len(df)
df_clean = df.dropna(subset=['outcome']).copy()
n_analyzed = len(df_clean)
n_excluded = n_initial - n_analyzed
print(f"Sample Flow: Initial={n_initial} -> Excluded={n_excluded} -> Analyzed={n_analyzed}")

# 3. TABLE 1: BASELINE CHARACTERISTICS & BIVARIATE TESTING
def summarize_continuous(series, group):
    """คำนวณ Mean ± SD, t-test, และ Standardized Mean Difference (SMD)"""
    g0 = series[group == 0].dropna()
    g1 = series[group == 1].dropna()
    t_stat, p_val = stats.ttest_ind(g1, g0, equal_var=False)
    pooled_sd = np.sqrt((g1.var() + g0.var()) / 2.0)
    smd = abs(g1.mean() - g0.mean()) / pooled_sd if pooled_sd > 0 else 0.0
    return {
        "Group 0 (Control)": f"{g0.mean():.1f} ± {g0.std():.1f}",
        "Group 1 (Event)": f"{g1.mean():.1f} ± {g1.std():.1f}",
        "p_value": f"{p_val:.3f}" if p_val >= 0.001 else "< 0.001",
        "SMD": f"{smd:.3f}"
    }

def summarize_categorical(series, group):
    """คำนวณ n (%) และ Pearson Chi-Square test"""
    ct = pd.crosstab(series, group)
    chi2, p_val, _, _ = stats.chi2_contingency(ct)
    return {"crosstab": ct, "p_value": f"{p_val:.3f}" if p_val >= 0.001 else "< 0.001"}

# 4. MULTIVARIABLE MODELING (Logistic Regression with Adjusted Odds Ratios)
# ฟิตโมเดลแบบควบคุมตัวแปรและดึง Adjusted OR (95% CI)
formula = "outcome ~ age + C(sex) + C(admission_status)"
model = smf.logit(formula, data=df_clean).fit(disp=False)

results = []
for term in model.params.index:
    if term == "Intercept":
        continue
    coef = model.params[term]
    ci_low, ci_high = model.conf_int().loc[term]
    p_val = model.pvalues[term]
    results.append({
        "Variable / Predictor": term,
        "Adjusted OR": f"{np.exp(coef):.2f}",
        "95% CI": f"({np.exp(ci_low):.2f} - {np.exp(ci_high):.2f})",
        "p_value": f"{p_val:.3f}" if p_val >= 0.001 else "< 0.001"
    })

res_df = pd.DataFrame(results)
print(res_df.to_markdown(index=False))
```

The agent may freely incorporate `medstat` modules (e.g. `from medstat.reporting.tables import render_records_table`, `from medstat.models.firth import fit_firth_logistic`) or standard libraries (`lifelines` for Cox PH/KM, `sklearn.metrics` for ROC curves) as appropriate.

---

## 7. Mandatory Clinical Governance Rules

1. **Strict Numeric 0/1 Endpoints & Explicit Event Mapping**: Binary outcomes and survival endpoints must be numeric `0` and `1` (`1 = Event`, `0 = Non-event / Censored`). Before recoding text outcomes, establish an explicit, unambiguous mapping of which category represents the clinical event. In survival analysis, ensure `1 = Event` and `0 = Censored` (never invert).
2. **Never Silent Deletion**: Always track participant retention:
   $$N_{\text{initial}} \longrightarrow N_{\text{excluded}} \longrightarrow N_{\text{analyzed}}$$
   Document reasons for exclusion (e.g. missing primary outcome, outside inclusion criteria).
3. **Wilson Score Confidence Intervals**: All binomial proportions (Sensitivity, Specificity, PPV, NPV) must use Wilson score intervals.
4. **DeLong Covariance**: ROC AUC standard errors and paired AUC comparisons must use DeLong variance.
5. **Austin (2009) PSM Standard**: Propensity score caliper must default to $0.2 \times \text{SD}(\text{logit } e)$; post-match balance requires $\text{SMD} < 0.10$.
6. **Publication Table Styling**: Tables must follow journal conventions (NEJM/JAMA: 3 horizontal rules, no vertical dividers, standard decimal precision: OR/HR to 2 decimal places, percentages to 1 decimal place, p-values formatted to 3 decimal places with `< 0.001` cutoff).

---

## 8. Completion Checklist

- [ ] Sheet layout inspected: title rows, metadata, and side-by-side dashboard tables identified and handled.
- [ ] Patient cohort isolated and column types properly mapped.
- [ ] Binary endpoints recoded strictly to numeric `0/1` (`1 = Event`).
- [ ] Sample retention flow ($N_{\text{initial}} \to N_{\text{excluded}} \to N_{\text{analyzed}}$) tracked and documented.
- [ ] Baseline characteristics (Table 1) generated with appropriate tests (t-test / Mann-Whitney, Chi-Square / Fisher) and SMDs.
- [ ] Multivariable model executed conforming to clinical standards (Adjusted OR / HR with 95% CIs and p-values).
- [ ] Results compiled into publication-grade table (NEJM/JAMA style) with clinical interpretation.

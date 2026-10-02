---
name: medstat-master
description: Master clinical biostatistics orchestrator. Ingests raw clinical data (CSV, XLSX, TSV, .parquet), inspects file layout and schema, identifies clinical study design via 3-pillar triangulation (Proposal, Clinical Principles, Raw Data), prevents statistical hallucination, and guides the agent to adapt Python scripts for data cleaning, statistical modeling, and publication-grade reporting (NEJM/JAMA) tailored to the specific dataset.
---

# medstat-master: Autonomous Biostatistical Orchestrator

The master intelligence layer for clinical biostatistics. Ingests clinical spreadsheets and cohorts, profiles data geometry, infers study design, selects appropriate statistical methods, and guides the agent to perform data cleaning, analysis, modeling, and reporting using **adaptive Python scripting grounded in biostatistical methodology and anti-hallucination guardrails**.

---

## 1. When to Use This Skill

Activate **medstat-master** whenever:
- The user provides or points to a dataset (`.csv`, `.xlsx`, `.tsv`, `.parquet`).
- The user presents a research proposal, PICO question, or conceptual framework and asks for the appropriate statistical analysis.
- Real-world clinical spreadsheets with messy layouts (multiple title rows, embedded notes, side-by-side summary tables, Thai dates/categories) need flexible parsing and robust biostatistical analysis.
- The agent must decisively arbitrate between competing statistical methods, resolve contradictions between research proposals and raw data, and prevent statistical hallucination.

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
│  Phase 2: 3-Pillar Triangulation & Method Selection    │
│  - Pillar 1: Research Proposal (PICO / Estimand)       │
│  - Pillar 2: Clinical Principles & Mechanisms          │
│  - Pillar 3: Raw Data Reality (Geometry & Constraints) │
│  - Pass through 5 Anti-Hallucination Stop Gates        │
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

# Inspect raw header lines to locate the real table (print non-empty cell count only; protect Zero-PHI)
df_peek = pd.read_excel("data.xlsx", header=None, nrows=10) # or read_csv
for idx, row in df_peek.iterrows():
    print(f"Row {idx}: non-empty cells = {row.dropna().count()}")
```

---

## 4. Phase 2: The 3-Pillar Triangulation Decision Engine (การตัดสินใจ 3 เสาหลัก)

To select the mathematically and clinically valid statistical analysis, the agent must evaluate 3 distinct sources of truth:

```
                            ┌─────────────────────────────────────────┐
                            │      Pillar 1: Research Proposal        │
                            │  - คำถามวิจัย (PICO / PECO)             │
                            │  - เป้าหมาย: Causal vs Prognostic vs     │
                            │    Diagnostic vs Agreement              │
                            └───────────────────┬─────────────────────┘
                                                │
                                                ▼
     ┌──────────────────────────────────────────┴──────────────────────────────────────────┐
     │                      Arbitration & Triangulation Gate                               │
     │                      (ตรวจสอบความสอดคล้อง ป้องกัน Hallucination)                     │
     └──────────────────┬───────────────────────────────────────────────────┬──────────────┘
                        │                                                   │
                        ▼                                                   ▼
┌─────────────────────────────────────────┐       ┌─────────────────────────────────────────┐
│     Pillar 2: Clinical Principles       │       │        Pillar 3: Raw Data Reality       │
│  - Confounding by indication            │       │  - Data geometry & sample size (N)      │
│  - Non-linear biological thresholds     │       │  - Events Per Variable (EPV diagnostic) │
│  - Missingness mechanism (MCAR/MAR)     │       │  - Outcome type (Binary, Time, Ordinal) │
│  - Biomarker directionality (High/Low)  │       │  - Distribution skewness & sparseness   │
└─────────────────────────────────────────┘       └─────────────────────────────────────────┘
```

### Study Design Matrix & Method Routing

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

## 5. The 5 Anti-Hallucination Stop Gates (เกราะป้องกันภาวะสร้างข้อมูลเท็จ)

Before writing analysis scripts or fitting models, the agent must pass through 5 deterministic stop gates:

### Gate 1: Contradiction Resolution Gate (ความขัดแย้งระหว่าง Proposal กับ ข้อมูลจริง)
- **The Risk**: User requests Survival Analysis (Cox regression), but raw data only contains a binary discharge status without a follow-up time column.
- **Directive**: **HALT & PIVOT**. Never hallucinate a synthetic time column. The agent must declare the mismatch, pivot to Type 2 (Multivariable Logistic or Firth), and notify the user:
  *"Proposal requests survival analysis, but raw data lacks follow-up duration; transitioning primary model to Multivariable Logistic/Firth Regression to preserve statistical validity."*

### Gate 2: Silent Assumption Barrier (ห้ามทึกทักสมมติฐานทางสถิติไปเอง)
- **The Risk**: Silently executing `df.dropna()` without checking missingness mechanisms, or reporting Mean ± SD and t-tests on highly skewed clinical labs (Troponin, Length of Stay, Ferritin).
- **Directive**:
  1. Every dropped row must be tracked via Sample Retention Flow ($N_{\text{initial}} \to N_{\text{excluded}} \to N_{\text{analyzed}}$).
  2. Continuous variables must undergo distribution checking (Shapiro-Wilk test or skewness examination). Skewed data must be presented as **Median [IQR]** with **Mann-Whitney U / Wilcoxon rank-sum test**.

### Gate 3: Directionality & Clinical Event Inversion Gate (ทิศทางตัวแปรและจุดตัด)
- **The Risk**: Survival events encoded backwards ($0 = \text{Death}, 1 = \text{Alive}$), or low-is-abnormal biomarkers (eGFR, Platelets, PaO2/FiO2) classified with $Score \ge Cutoff$, yielding $\text{AUC} < 0.50$ and inverted sensitivity/specificity.
- **Directive**:
  1. Strict numeric `0/1` encoding: $1 = \text{Event / Disease / Case}$, $0 = \text{Censored / Healthy / Control}$.
  2. Automated AUC Sanity Check: If empirical $\text{AUC} < 0.50$, invert score ($Score_{\text{eff}} = -Score$) to preserve concordance.

### Gate 4: Sparse Data & Events-Per-Variable (EPV) Gate (สถิติตัวแปรพหุคูณบนข้อมูลเบาบาง)
- **The Risk**: Fitting multivariable models with 10 covariates when only 12 events occurred, producing quasi-complete separation and astronomical odds ratios ($\text{OR} > 1000$).
- **Directive**:
  Compute $\text{EPV} = \frac{\min(N_{\text{events}}, N_{\text{non-events}})}{K_{\text{covariates}}}$. If $\text{EPV} < 10$ or zero cells appear in crosstabs, decisively switch to **Firth Penalized Likelihood** (`fit_firth_logistic` / `firth_cox`).

### Gate 5: Clinical Interpretation Guardrails (การแปลผลทางคลินิกอย่างรัดกุม)
- **The Risk**: Claiming $P > 0.05$ proves "no effect" or "treatments are identical", or substituting Pearson correlation for rater agreement.
- **Directive**:
  1. $P > 0.05$ must be reported as "insufficient evidence to reject the null hypothesis", focusing on the 95% CI.
  2. For device/rater reliability, strictly reject Pearson correlation ($r$) and enforce **Bland-Altman 95% LoA** or **Intraclass Correlation (ICC)**.

---

## 6. Dual Execution Modes: Direct Execution vs Proposal First

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

## 7. Phase 3: Adaptive Python Scripting Protocol (ปรับแต่งสคริปต์ตามข้อมูลจริง)

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
df = pd.read_excel("dataset.xlsx", skiprows=2)  # ปรับ skiprows ตามจริง

# 2. STANDARDIZE ENDPOINTS & SAMPLE FLOW TRACKING
# บังคับใช้ Numeric 0/1 เสมอ (1 = Event / Case, 0 = Non-event / Control)
n_initial = len(df)
df_clean = df.dropna(subset=['outcome']).copy()
n_analyzed = len(df_clean)
n_excluded = n_initial - n_analyzed
print(f"Sample Flow: Initial={n_initial} -> Excluded={n_excluded} -> Analyzed={n_analyzed}")

# 3. TABLE 1: BASELINE CHARACTERISTICS WITH NORMALITY AUDIT
def summarize_continuous(series, group):
    """ทดสอบ Normality ก่อนเลือก Mean ± SD (t-test) หรือ Median [IQR] (Mann-Whitney U)"""
    g0 = series[group == 0].dropna()
    g1 = series[group == 1].dropna()
    
    # Check normality using Shapiro-Wilk (requires 3 <= n <= 5000)
    if len(g0) < 3 or len(g1) < 3:
        is_normal = False
    else:
        stat0, p0 = stats.shapiro(g0) if len(g0) <= 5000 else (0, 0.05)
        stat1, p1 = stats.shapiro(g1) if len(g1) <= 5000 else (0, 0.05)
        is_normal = (p0 > 0.05) and (p1 > 0.05)
    
    if len(g0) < 2 or len(g1) < 2:
        smd_str = "Not estimable"
    else:
        pooled_sd = np.sqrt((g1.var(ddof=1) + g0.var(ddof=1)) / 2.0)
        if pooled_sd == 0:
            smd_str = "0.000" if g1.mean() == g0.mean() else "Not estimable"
        else:
            smd_str = f"{abs(g1.mean() - g0.mean()) / pooled_sd:.3f}"
    
    if is_normal:
        t_stat, p_val = stats.ttest_ind(g1, g0, equal_var=False)
        return {
            "Summary": f"Mean ± SD",
            "Group 0 (Control)": f"{g0.mean():.1f} ± {g0.std():.1f}",
            "Group 1 (Event)": f"{g1.mean():.1f} ± {g1.std():.1f}",
            "p_value": f"{p_val:.3f}" if p_val >= 0.001 else "< 0.001",
            "SMD": smd_str
        }
    else:
        u_stat, p_val = stats.mannwhitneyu(g1, g0, alternative='two-sided')
        return {
            "Summary": f"Median [IQR]",
            "Group 0 (Control)": f"{g0.median():.1f} [{g0.quantile(0.25):.1f}, {g0.quantile(0.75):.1f}]",
            "Group 1 (Event)": f"{g1.median():.1f} [{g1.quantile(0.25):.1f}, {g1.quantile(0.75):.1f}]",
            "p_value": f"{p_val:.3f}" if p_val >= 0.001 else "< 0.001",
            "SMD": smd_str
        }

def summarize_categorical(series, group):
    """คำนวณ n (%) และ Chi-Square หรือ Fisher's exact test"""
    ct = pd.crosstab(series, group)
    if ct.shape == (2, 2) and (ct.values < 5).any():
        _, p_val = stats.fisher_exact(ct)
    else:
        _, p_val, _, _ = stats.chi2_contingency(ct)
    return {"crosstab": ct, "p_value": f"{p_val:.3f}" if p_val >= 0.001 else "< 0.001"}

# 4. EPV DIAGNOSTIC & MULTIVARIABLE MODELING
import patsy
from medstat.models.firth import fit_firth_logistic

# Define missing-data strategy and track complete-case exclusions before model fitting
model_cols = ["outcome", "age", "sex", "admission_status"]
df_model = df_clean.dropna(subset=model_cols).copy()
n_model_excluded = len(df_clean) - len(df_model)
if n_model_excluded > 0:
    print(f"Model complete-case exclusion: excluded {n_model_excluded} rows with missing model variables.")
    # Reconcile retention counts with sample retention flow if tracker is used:
    # tracker.record_stage("Model Complete Cases", n_remaining=len(df_model), reason="Missing model variables")

formula = "outcome ~ age + C(sex) + C(admission_status)"
y_mat, X_mat = patsy.dmatrices(formula, data=df_model, return_type='dataframe')
# Count expanded model parameter degrees of freedom (excluding intercept)
n_params = X_mat.shape[1] - 1
n_events = (df_model['outcome'] == 1).sum()
n_nonevents = (df_model['outcome'] == 0).sum()
epv = min(n_events, n_nonevents) / n_params if n_params > 0 else np.nan
print(f"Events Per Parameter (EPV): {epv:.1f} (effective events={min(n_events, n_nonevents)}, parameters={n_params})")

# Route model fit: If EPV < 10 or quasi-complete separation occurs, route to Firth penalized regression
if epv < 10:
    print(f"Warning: Low EPV ({epv:.1f} < 10); routing to Firth penalized logistic regression to prevent separation bias.")
    firth_res = fit_firth_logistic(y_mat.iloc[:, 0], X_mat.drop(columns=['Intercept']), fit_intercept=True, ci_method="pl")
    results = []
    for term, coef in firth_res["params"].items():
        ci_low, ci_high = firth_res["ci"][term]
        p_val = firth_res["pvalues"][term]
        results.append({
            "Variable / Predictor": term,
            "Adjusted OR": f"{np.exp(coef):.2f}",
            "95% CI": f"({np.exp(ci_low):.2f} - {np.exp(ci_high):.2f})",
            "p_value": f"{p_val:.3f}" if p_val >= 0.001 else "< 0.001"
        })
else:
    model = smf.logit(formula, data=df_model).fit(disp=False)
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

## 8. Mandatory Clinical Governance Rules

1. **Strict Numeric 0/1 Endpoints & Explicit Event Mapping**: Binary outcomes and survival endpoints must be numeric `0` and `1` (`1 = Event`, `0 = Non-event / Censored`). Before recoding text outcomes, establish an explicit, unambiguous mapping of which category represents the clinical event. In survival analysis, ensure `1 = Event` and `0 = Censored` (never invert).
2. **Never Silent Deletion & No Outcome Imputation**: Always track participant retention:
   $$N_{\text{initial}} \longrightarrow N_{\text{excluded}} \longrightarrow N_{\text{analyzed}}$$
   Document reasons for exclusion. **Strict Ban**: Never impute missing values in the primary clinical outcome variable using MICE or KNN; drop missing outcome cases under audited retention flow with documented rationale.
3. **Wilson Score Confidence Intervals**: All binomial proportions (Sensitivity, Specificity, PPV, NPV) must use Wilson score intervals.
4. **DeLong Covariance & Automated Directionality Check**: ROC AUC standard errors and paired AUC comparisons must use DeLong variance. If empirical $\text{AUC} < 0.50$, audit marker directionality and invert score ($Score_{\text{eff}} = -Score$) to maintain concordance.
5. **Austin (2009) PSM Standard & Confounder Selection**: Propensity score caliper must default to $0.2 \times \text{SD}(\text{logit } e)$; post-match balance requires $\text{SMD} < 0.10$. Include *only* baseline pre-treatment confounders; strictly exclude post-treatment variables, mediators, or colliders.
6. **Publication Table Styling**: Tables must follow journal conventions (NEJM/JAMA: 3 horizontal rules, no vertical dividers, standard decimal precision: OR/HR to 2 decimal places, percentages to 1 decimal place, p-values formatted to 3 decimal places with `< 0.001` cutoff).
7. **No Correlation as Agreement**: Never substitute Pearson/Spearman correlation for agreement; enforce Bland-Altman LoA with large-sample CIs and pure-SciPy ICC.

---

## 9. Completion Checklist

- [ ] Sheet layout inspected: title rows, metadata, and side-by-side dashboard tables identified and handled.
- [ ] 3-Pillar Triangulation completed: Proposal PICO matched against clinical principles and empirical data reality.
- [ ] 5 Anti-Hallucination Stop Gates passed (Contradiction checked, No silent assumptions, Directionality verified, EPV diagnosed, Interpretation guarded).
- [ ] Binary endpoints recoded strictly to numeric `0/1` (`1 = Event`).
- [ ] Sample retention flow ($N_{\text{initial}} \to N_{\text{excluded}} \to N_{\text{analyzed}}$) tracked and documented.
- [ ] Baseline characteristics (Table 1) generated with distribution-appropriate tests (t-test vs Mann-Whitney, Chi-Square vs Fisher) and SMDs.
- [ ] Multivariable model executed conforming to clinical standards (Adjusted OR / HR with 95% CIs and p-values; Firth penalization applied if EPV < 10 or sparse).
- [ ] Results compiled into publication-grade table (NEJM/JAMA style) with clinical interpretation.

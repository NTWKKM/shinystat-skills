---
name: medstat-models
description: Multivariable regression, survival analysis, Firth penalized likelihood, restricted cubic splines (RCS), Table 1 generation with SMDs, and VanderWeele E-value sensitivity. Use when fitting multivariable GLMs, Cox proportional hazards, handling separation/sparse events with Firth penalization, testing non-linear dose-response curves, or executing YAML Statistical Analysis Plans (SAP).
---

# medstat-models: Multivariable Regression, Survival & Penalized Models

Biostatistical modeling engine supporting generalized linear models, Cox proportional hazards with Schoenfeld diagnostics, Firth penalized likelihood, restricted cubic splines, and unmeasured confounding sensitivity analysis.

## Core Rules

1. **Pre-Model Table 1**: Always characterize baseline covariates with Standardized Mean Differences (SMD) before multivariable modeling.
2. **Proportional Hazards Assumption**: Standard Cox models evaluate the proportional hazards assumption via Schoenfeld residuals when `--schoenfeld` is specified in the CLI (penalized Firth Cox does not compute Schoenfeld tests via CLI); violations require stratified Cox or time-varying covariates.
3. **Sparse Events & Monotone Likelihood**: In sparse event survival (< 20 events) or quasi-complete logistic separation, use Firth's penalized likelihood with profile likelihood confidence intervals.
4. **Non-Linearity Verification**: Continuous exposures with potential non-linear biology can be modeled using restricted cubic splines (RCS, supported for Cox regression in the CLI via `--spline-var`) with centered contrast reference points.
5. **Binary & Event Outcome Encoding**: Binary outcomes (logistic regression) and event indicators (Cox proportional hazards) must be explicitly encoded as numeric `0` and `1` (`1 = Event`, `0 = Non-event`). Raw text outcomes (e.g., `"Dead"`, `"Alive"`, `"Yes"`, `"No"`) are rejected to prevent clinical event inversion.
6. **Events-Per-Variable (EPV) Diagnostic Rule**: Before fitting multivariable regression, calculate EPV according to model type: for Cox proportional hazards, calculate $\text{EPV}_{\text{Cox}} = \frac{E}{P}$ where $E$ is the total failure-event count and $P$ is the fitted predictor parameter count (degrees of freedom, excluding intercept); for logistic regression, calculate $\text{EPV}_{\text{Logistic}} = \frac{\min(N_{\text{events}}, N_{\text{non-events}})}{P}$ where $P$ is the fitted parameter count from the expanded design matrix. If $\text{EPV} < 10$ or quasi-complete separation occurs, standard maximum likelihood estimation (MLE) is biased and produces unstable/infinite estimates. The agent must decisively transition to **Firth penalized likelihood** (`fit_firth_logistic` / `firth_cox`) or perform variable selection.

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

- **Cox Proportional Hazards (Standard Cox with `--schoenfeld`)**: The Schoenfeld residual correlation test is one diagnostic for the proportional hazards assumption; $p > 0.05$ indicates insufficient evidence against PH but does not establish proportionality. Review scaled Schoenfeld residual plots and model context, as the test may miss non-monotone departures (penalized Firth Cox does not compute Schoenfeld tests via CLI).
- **Firth Convergence**: Verify profile likelihood confidence intervals converge.
- **E-Value Sensitivity**: If effect is statistically significant, compute the minimum unmeasured confounding strength required to explain away the observed estimate.

---

## Adaptive Python Scripting Protocol (ปรับแต่งสคริปต์แบบจำลองทางสถิติตามข้อมูลจริง)

> **Core Philosophy**: Never execute rigid canned scripts that make naive assumptions about file structure. The agent is empowered with full autonomy to write, adapt, and run Python scripts (`scratch/model.py`) tailored to the specific columns, encodings, and clinical objectives of the analyzed dataset.
>
> ⚠️ **Mandatory Directive — ต้องดู Script ต้นแบบประกอบเสมอ (Review Prototype Scripts First)**:
> แม้จะให้อิสระ Agent ในการเขียนและปรับ Python Script เองตามสภาพข้อมูลจริง แต่ **Agent ต้องเปิดดูและอ้างอิงสคริปต์ต้นแบบ (Prototype Scripts)** หรือศึกษาการคำนวณในโมดูลแกนกลาง `src/medstat/` เสมอ เพื่อยึดมาตรฐานความถูกต้องทางชีวสถิติการแพทย์:
> - **Table 1 และสถิติ Bivariate**: ดูการคำนวณ Mean ± SD vs Median [IQR], t-test vs Mann-Whitney U, Chi-Square vs Fisher's exact, และ SMD จาก `src/medstat/reporting/table1.py` และ `src/medstat/stats/bivariate.py`
> - **Multivariable Logistic & GLM**: ดูการคำนวณ Adjusted Odds Ratio ($\exp(\beta)$) และ 95% CI จาก `src/medstat/models/glm.py`
> - **Firth Penalized Likelihood**: ดูการแก้ปัญหา separation / sparse events จาก `src/medstat/models/firth.py`
> - **Cox Proportional Hazards**: ดูการฟิต survival model และการทดสอบ Schoenfeld residuals จาก `src/medstat/models/survival.py`
> - **Non-linear Splines (RCS)**: ดูการทำ restricted cubic splines จาก `src/medstat/models/splines.py`
> - **Sensitivity to Unmeasured Confounding**: ดูสูตร VanderWeele E-value จาก `src/medstat/models/sensitivity.py`
>
> **วงจรการทำงานของ Agent**:
> `[1. สำรวจตัวแปรและการแจกแจง] ──▶ [2. ดูสคริปต์ต้นแบบเพื่อยึดหลักชีวสถิติ] ──▶ [3. ปรับโค้ดและรันแบบจำลอง]`

### Master Prototype Script for Statistical Modeling (สคริปต์ต้นแบบมาตรฐาน)

Agent ควรนำโครงสร้างและฟังก์ชันของสคริปต์ต้นแบบนี้ไปปรับแต่งลงใน workspace (เช่น `scratch/model.py`) ให้เข้ากับตัวแปรและคำถามวิจัยของข้อมูลจริง:

```python
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 1. LOAD CLEANED COHORT
df = pd.read_csv("clean_cohort.csv")

# 2. TABLE 1: BASELINE CHARACTERISTICS WITH SMDs
def summarize_continuous(series, group):
    g0 = series[group == 0].dropna()
    g1 = series[group == 1].dropna()
    t_stat, p_val = stats.ttest_ind(g1, g0, equal_var=False)
    if len(g0) < 2 or len(g1) < 2:
        smd = "Not estimable"
    else:
        diff = abs(g1.mean() - g0.mean())
        pooled_sd = np.sqrt((g1.var(ddof=1) + g0.var(ddof=1)) / 2.0)
        if pooled_sd == 0:
            smd = "0.000" if diff == 0 else "Not estimable (zero SD with non-zero diff)"
        else:
            smd = f"{diff / pooled_sd:.3f}"
    return {
        "Group 0": f"{g0.mean():.1f} ± {g0.std():.1f}",
        "Group 1": f"{g1.mean():.1f} ± {g1.std():.1f}",
        "p_value": f"{p_val:.3f}" if p_val >= 0.001 else "< 0.001",
        "SMD": smd
    }

def summarize_categorical(series, group):
    ct = pd.crosstab(series, group)
    chi2, p_val, _, _ = stats.chi2_contingency(ct)
    g0 = series[group == 0].dropna()
    g1 = series[group == 1].dropna()
    dummies = pd.get_dummies(series, drop_first=(series.nunique() == 2))
    smds = {}
    for col in dummies.columns:
        d0 = dummies.loc[g0.index, col].astype(float)
        d1 = dummies.loc[g1.index, col].astype(float)
        diff = abs(d1.mean() - d0.mean())
        pooled_sd = np.sqrt((d1.var(ddof=1) + d0.var(ddof=1)) / 2.0) if len(d1) > 1 and len(d0) > 1 else 0.0
        if len(d1) < 2 or len(d0) < 2:
            smds[col] = "Not estimable"
        elif pooled_sd == 0:
            smds[col] = "0.000" if diff == 0 else "Not estimable"
        else:
            smds[col] = f"{diff / pooled_sd:.3f}"
    smd_str = smds[dummies.columns[0]] if len(smds) == 1 else str(smds)
    return {
        "crosstab": ct,
        "p_value": f"{p_val:.3f}" if p_val >= 0.001 else "< 0.001",
        "SMD": smd_str,
        "category_smds": smds,
    }

# Report Table 1 baseline summaries and SMDs before model fitting
print("--- Table 1: Baseline Characteristics & SMDs ---")
for num_var in ["age", "bmi"]:
    res_num = summarize_continuous(df[num_var], df["treatment"])
    print(f"{num_var}: Control={res_num['Group 0']}, Treated={res_num['Group 1']}, p={res_num['p_value']}, SMD={res_num['SMD']}")

res_cat = summarize_categorical(df["sex"], df["treatment"])
print(f"sex: p={res_cat['p_value']}, SMD={res_cat['SMD']}")

# 3. EPV DIAGNOSTIC & MULTIVARIABLE MODELING (Logistic Regression / GLM)
import patsy
from medstat.models.firth import fit_firth_logistic

# Define missing-data strategy and complete cases before fitting
model_cols = ["outcome", "treatment", "age", "sex", "bmi"]
df_model = df.dropna(subset=model_cols).copy()
n_model_excluded = len(df) - len(df_model)
if n_model_excluded > 0:
    print(f"Excluded {n_model_excluded} incomplete cases for model variables.")

formula = "outcome ~ treatment + age + C(sex) + bmi"
y_mat, X_mat = patsy.dmatrices(formula, data=df_model, return_type='dataframe')
# Count fitted predictor parameters from expanded design matrix (excluding intercept)
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
            "Predictor": term,
            "Adjusted OR": f"{np.exp(coef):.2f}",
            "95% CI": f"({np.exp(ci_low):.2f} - {np.exp(ci_high):.2f})",
            "p_value": f"{p_val:.3f}" if p_val >= 0.001 else "< 0.001"
        })
    primary_or = np.exp(firth_res["params"]["treatment"])
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
            "Predictor": term,
            "Adjusted OR": f"{np.exp(coef):.2f}",
            "95% CI": f"({np.exp(ci_low):.2f} - {np.exp(ci_high):.2f})",
            "p_value": f"{p_val:.3f}" if p_val >= 0.001 else "< 0.001"
        })
    primary_or = np.exp(model.params["treatment"])

res_df = pd.DataFrame(results)
print(res_df.to_markdown(index=False))

# 4. SENSITIVITY ANALYSIS (VanderWeele E-value)
from medstat.models.sensitivity import calculate_e_value

# Compute E-value using calculate_e_value with estimate_type="OR" and verify rare-outcome assumption
# (If outcome incidence is common, >= 15%, the square-root transformation is applied to approximate risk ratio)
outcome_incidence = df_model['outcome'].mean()
is_rare = outcome_incidence < 0.15
e_val_res = calculate_e_value(primary_or, estimate_type="OR", rare_outcome=is_rare)
print(f"E-value for treatment effect (OR = {primary_or:.2f}, rare_outcome={is_rare}): {e_val_res['e_value_estimate']:.2f}")
```

The agent may freely incorporate `medstat` modules (e.g. `from medstat.models.firth import fit_firth_logistic`, `from medstat.models.survival import fit_cox_ph`, `from medstat.models.sensitivity import calculate_e_value`) or standard libraries (`lifelines` for Cox PH/KM) as appropriate.

---

## Completion Criteria

- [ ] Baseline characteristics tabulated with explicit SMD imbalance checks.
- [ ] Outcome variable verified and encoded as numeric 0/1 (1 = Event).
- [ ] Model coefficients, 95% confidence intervals, and p-values generated.
- [ ] Proportional hazards or separation diagnostics completed.
- [ ] E-value calculated for primary exposure to quantify sensitivity to unmeasured confounding.

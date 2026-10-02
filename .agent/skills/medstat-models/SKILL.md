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
6. **Events-Per-Variable (EPV) Diagnostic Rule**: Before fitting multivariable regression, calculate EPV according to model type: for Cox proportional hazards, calculate $\text{EPV}_{\text{Cox}} = \frac{E}{P}$ where $E$ is the total failure-event count and $P$ is the fitted predictor parameter count (degrees of freedom, excluding intercept); for logistic regression, calculate $\text{EPV}_{\text{Logistic}} = \frac{\min(N_{\text{events}}, N_{\text{non-events}})}{P}$ where $P$ is the fitted parameter count from the expanded design matrix. Note that $\text{EPV} < 10$ serves as a pragmatic risk screen for small-sample bias and overfitting rather than an absolute diagnosis of separation. If quasi-complete separation occurs, or when prespecified sparse-data criteria or estimation instability arise, standard maximum likelihood estimation (MLE) is biased or fails to converge. The agent must decisively transition to **Firth penalized likelihood** (`fit_firth_logistic` / `firth_cox`) or perform dimension reduction.
7. **Ordinal Outcome Modeling**: Ordinal outcomes with 3+ ordered levels (mRS, GCS, NYHA) should be modeled using cumulative link proportional odds models (`fit_proportional_odds` / CLI `--type ordinal`) with Brant test verification (`--po-test`). Do not treat ordinal scores as continuous OLS linear regressions.
8. **Clustered Data & GEE**: Multi-center datasets with patient clustering within hospitals violate independence. Use GEE (`--type gee --cluster <col>`) with robust standard errors or random-intercept mixed models (`--type mixed --cluster <col>`), and compute Design Effect (DEFF).

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

# Ordinal Proportional Odds Model (e.g., mRS, GCS)
medstat model --data <cohort.csv> --type ordinal --outcome mrs_score \
  --exposure treatment --covariates "age,nihss" --po-test --output ordinal_res.json

# Clustered GEE (Population-Average Effects for multi-center data)
medstat model --data <multicenter.csv> --type gee --outcome mortality \
  --exposure statin --covariates "age,sex" --cluster hospital_id \
  --corr-structure exchangeable --output gee_res.json

# Random Intercept Mixed Model (Subject-Specific Effects for clustered data)
medstat model --data <multicenter.csv> --type mixed --outcome recovery_days \
  --exposure statin --covariates "age,sex" --cluster hospital_id --output mixed_res.json
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
- **Calibration for Prediction Models**: For models intended for clinical deployment or TRIPOD-compliant prediction validation, evaluate calibration (Brier score, calibration slope/intercept, ICI) via `medstat diag --calibration` on model-predicted probabilities. High AUC alone does not guarantee well-calibrated predictions.

---

## Adaptive Python Scripting Protocol (ปรับแต่งสคริปต์แบบจำลองทางสถิติตามข้อมูลจริง)

> **Core Philosophy**: Never execute rigid canned scripts that make naive assumptions about file structure. The agent is empowered with full autonomy to write, adapt, and run Python scripts (`scratch/model.py`) tailored to the specific columns, encodings, and clinical objectives of the analyzed dataset.
>
> 🔒 **Subprocess & Script Execution Safety**:
> When generating and running analysis scripts (`scratch/model.py`), enforce execution controls: disable shell/subprocess access, limit file reads strictly to the designated dataset and referenced prototype/core modules, limit file writes strictly to scratch and designated output paths, and ensure no access to credentials or environment secrets. Require explicit user confirmation if the runtime cannot enforce these sandbox controls.
>
> ⚠️ **Mandatory Directive — ต้องดู Script ต้นแบบประกอบเสมอ (Review Prototype Scripts First)**:
> แม้จะให้อิสระ Agent ในการเขียนและปรับ Python Script เองตามสภาพข้อมูลจริง แต่ **Agent ต้องเปิดดูและอ้างอิงสคริปต์ต้นแบบ (Prototype Scripts)** หรือศึกษาการคำนวณในโมดูลแกนกลาง `src/medstat/` เสมอ เพื่อยึดมาตรฐานความถูกต้องทางชีวสถิติการแพทย์:
> - **Table 1 และสถิติ Bivariate**: ดูการคำนวณ Mean ± SD vs Median [IQR], t-test vs Mann-Whitney U, Chi-Square vs Fisher's exact, และ SMD จาก `src/medstat/reporting/table1.py` และ `src/medstat/stats/bivariate.py`
> - **Multivariable Logistic & GLM**: ดูการคำนวณ Adjusted Odds Ratio ($\exp(\beta)$) และ 95% CI จาก `src/medstat/models/glm.py`
> - **Firth Penalized Likelihood**: ดูการแก้ปัญหา separation / sparse events จาก `src/medstat/models/firth.py`
> - **Cox Proportional Hazards**: ดูการฟิต survival model และการทดสอบ Schoenfeld residuals จาก `src/medstat/models/survival.py`
> - **Non-linear Splines (RCS)**: ดูการทำ restricted cubic splines จาก `src/medstat/models/splines.py`
> - **Sensitivity to Unmeasured Confounding**: ดูสูตร VanderWeele E-value จาก `src/medstat/models/sensitivity.py`
> - **Ordinal Proportional Odds**: ดูโมเดลสะสม cumulative link และ Brant test จาก `src/medstat/models/ordinal.py`
> - **Clustered Data & GEE**: ดูแบบจำลอง GEE และ Mixed Effects จาก `src/medstat/models/multilevel.py`
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
def format_p_value(p_val):
    if p_val is None or pd.isna(p_val) or not np.isfinite(p_val):
        return "NA"
    return "< 0.001" if p_val < 0.001 else f"{p_val:.3f}"

def summarize_continuous(series, group):
    g0 = series[group == 0].dropna()
    g1 = series[group == 1].dropna()
    if len(g0) < 2 or len(g1) < 2:
        return {"Group 0": "NA", "Group 1": "NA", "p_value": "NA", "SMD": "Not estimable"}

    # Assess normality via Shapiro-Wilk when sample size permits (n <= 5000)
    is_normal = True
    if len(g0) >= 3 and len(g1) >= 3:
        _, p_norm0 = stats.shapiro(g0) if len(g0) <= 5000 else (None, 0.05)
        _, p_norm1 = stats.shapiro(g1) if len(g1) <= 5000 else (None, 0.05)
        if (p_norm0 is not None and p_norm0 < 0.05) or (p_norm1 is not None and p_norm1 < 0.05):
            is_normal = False

    if is_normal:
        t_stat, p_val = stats.ttest_ind(g1, g0, equal_var=False)
        g0_summary = f"{g0.mean():.1f} ± {g0.std():.1f}"
        g1_summary = f"{g1.mean():.1f} ± {g1.std():.1f}"
    else:
        u_stat, p_val = stats.mannwhitneyu(g1, g0, alternative="two-sided")
        g0_summary = f"{g0.median():.1f} [{g0.quantile(0.25):.1f}, {g0.quantile(0.75):.1f}]"
        g1_summary = f"{g1.median():.1f} [{g1.quantile(0.25):.1f}, {g1.quantile(0.75):.1f}]"

    diff = abs(g1.mean() - g0.mean())
    pooled_sd = np.sqrt((g1.var(ddof=1) + g0.var(ddof=1)) / 2.0)
    if pooled_sd == 0:
        smd = "0.000" if diff == 0 else "Not estimable (zero SD with non-zero diff)"
    else:
        smd = f"{diff / pooled_sd:.3f}"
    return {
        "Group 0": g0_summary,
        "Group 1": g1_summary,
        "p_value": format_p_value(p_val),
        "SMD": smd
    }

def summarize_categorical(series, group):
    ct = pd.crosstab(series, group)
    chi2, p_val_asymp, _, expected = stats.chi2_contingency(ct)
    is_sparse = (expected < 5).any()
    if is_sparse:
        if ct.shape == (2, 2):
            _, p_val = stats.fisher_exact(ct)
            p_formatted = format_p_value(p_val)
        else:
            # Sparse table larger than 2x2: asymptotic chi-square is invalid
            # Report as not estimable without exact/Monte Carlo permutation test
            p_val = np.nan
            p_formatted = "Not estimable (sparse table > 2x2 requires exact/permutation test)"
    else:
        p_val = p_val_asymp
        p_formatted = format_p_value(p_val)
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
        "p_value": p_formatted,
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

# Validate outcome is strictly binary {0, 1}
unique_outcomes = set(df_model["outcome"].dropna().unique())
if not unique_outcomes.issubset({0, 1, 0.0, 1.0}):
    raise ValueError(f"Outcome must be strictly binary {{0, 1}}, got: {unique_outcomes}")

formula = "outcome ~ treatment + age + C(sex) + bmi"
y_mat, X_mat = patsy.dmatrices(formula, data=df_model, return_type='dataframe')
# Count fitted predictor parameters from expanded design matrix (excluding intercept)
n_params = X_mat.shape[1] - 1
n_events = (df_model['outcome'] == 1).sum()
n_nonevents = (df_model['outcome'] == 0).sum()
epv = min(n_events, n_nonevents) / n_params if n_params > 0 else np.nan
print(f"Events Per Parameter (EPV): {epv:.1f} (effective events={min(n_events, n_nonevents)}, parameters={n_params})")

# Check for quasi-complete separation / zero cells across categorical/discrete predictors
has_zero_cells = False
cat_cols = [c for c in ["treatment", "sex"] if c in df_model.columns] + [
    c for c in df_model.select_dtypes(include=['category', 'object', 'bool']).columns
    if c != "outcome" and c not in ["treatment", "sex"]
]
for col in cat_cols:
    ct = pd.crosstab(df_model[col], df_model["outcome"])
    if (ct == 0).any().any():
        has_zero_cells = True
        break

# Helper to resolve treatment OR across numeric and patsy contrast terms:
from medstat.models import extract_primary_effect

# Route model fit: If EPV < 10 or quasi-complete separation occurs, route to Firth penalized regression
if epv < 10 or has_zero_cells:
    reason = f"Low EPV ({epv:.1f} < 10)" if epv < 10 else "Quasi-complete separation / zero cells detected"
    print(f"Warning: {reason}; routing to Firth penalized logistic regression to prevent separation bias.")
    firth_res = fit_firth_logistic(y_mat.iloc[:, 0], X_mat.drop(columns=['Intercept']), fit_intercept=True, ci_method="pl")
    summary = firth_res["summary_df"]
    results = []
    for term, row in summary.iterrows():
        if term == "(Intercept)":
            continue
        results.append({
            "Predictor": term,
            "Adjusted OR": f"{row['odds_ratio']:.2f}",
            "95% CI": f"({row['or_ci_lower']:.2f} - {row['or_ci_upper']:.2f})",
            "p_value": format_p_value(row["p_value"])
        })
    primary_or = extract_primary_effect({term: row["odds_ratio"] for term, row in summary.iterrows()})
else:
    try:
        model = smf.logit(formula, data=df_model).fit(disp=False)
        if not model.mle_retvals.get("converged", True) or not np.all(np.isfinite(model.params)):
            raise ValueError("Standard MLE did not converge or yielded non-finite estimates.")
        results = []
        for term in model.params.index:
            if term == "Intercept":
                continue
            coef = model.params[term]
            ci_low, ci_high = model.conf_int().loc[term]
            results.append({
                "Predictor": term,
                "Adjusted OR": f"{np.exp(coef):.2f}",
                "95% CI": f"({np.exp(ci_low):.2f} - {np.exp(ci_high):.2f})",
                "p_value": format_p_value(model.pvalues[term])
            })
        primary_or = extract_primary_effect({term: np.exp(model.params[term]) for term in model.params.index})
    except Exception as e:
        print(f"Standard MLE estimation failed or unstable ({e}); falling back to Firth penalized likelihood.")
        firth_res = fit_firth_logistic(y_mat.iloc[:, 0], X_mat.drop(columns=['Intercept']), fit_intercept=True, ci_method="pl")
        summary = firth_res["summary_df"]
        results = []
        for term, row in summary.iterrows():
            if term == "(Intercept)":
                continue
            results.append({
                "Predictor": term,
                "Adjusted OR": f"{row['odds_ratio']:.2f}",
                "95% CI": f"({row['or_ci_lower']:.2f} - {row['or_ci_upper']:.2f})",
                "p_value": format_p_value(row["p_value"])
            })
        primary_or = extract_primary_effect({term: row["odds_ratio"] for term, row in summary.iterrows()})

res_df = pd.DataFrame(results)
print(res_df.to_markdown(index=False))

# 4. SENSITIVITY ANALYSIS (VanderWeele E-value)
from medstat.models.sensitivity import calculate_e_value

# Compute E-value using calculate_e_value with estimate_type="OR" and verify rare-outcome assumption
# (If outcome incidence is common, >= 15%, the square-root transformation is applied to approximate risk ratio)
if np.isfinite(primary_or) and primary_or > 0:
    outcome_incidence = df_model['outcome'].mean()
    is_rare = outcome_incidence < 0.15
    e_val_res = calculate_e_value(primary_or, estimate_type="OR", rare_outcome=is_rare)
    print(f"E-value for treatment effect (OR = {primary_or:.2f}, rare_outcome={is_rare}): {e_val_res['e_value_estimate']:.2f}")
else:
    print("Primary treatment effect is not estimable or non-finite; E-value calculation skipped.")

# 5. ADVANCED MODELING: ORDINAL & CLUSTERED (Optional/Contextual)
# For Ordinal Outcomes (e.g., mRS):
# from medstat.models.ordinal import fit_proportional_odds, test_proportional_odds
# ordinal_res = fit_proportional_odds(y=df_model["mrs_score"], X=df_model[["treatment", "age"]])
# brant_res = test_proportional_odds(y=df_model["mrs_score"], X=df_model[["treatment", "age"]])
# print(f"Brant Omnibus Test p-value: {brant_res['omnibus_p_value']:.4f}")

# For Clustered Data (e.g., multicenter):
# from medstat.models.multilevel import fit_gee, calculate_design_effect
# df_multi = df.dropna(subset=["outcome", "treatment", "hospital_id"])
# deff_res = calculate_design_effect(df_multi["outcome"], df_multi["hospital_id"])
# print(f"Design Effect (DEFF): {deff_res['design_effect']:.2f}")
# gee_res = fit_gee(y=df_multi["outcome"], X=df_multi[["treatment"]], cluster_ids=df_multi["hospital_id"], family="binomial")
```

The agent may freely incorporate `medstat` modules (e.g. `from medstat.models.firth import fit_firth_logistic`, `from medstat.models.survival import fit_cox_ph`, `from medstat.models.sensitivity import calculate_e_value`) or standard libraries (`lifelines` for Cox PH/KM) as appropriate.

---

## Completion Criteria

- [ ] Baseline characteristics tabulated with explicit SMD imbalance checks.
- [ ] Outcome variable verified and encoded as numeric 0/1 (1 = Event).
- [ ] Model coefficients, 95% confidence intervals, and p-values generated.
- [ ] Proportional hazards or separation diagnostics completed.
- [ ] Ordinal proportional odds assumption verified via Brant test (if applicable).
- [ ] Clustering design effect (DEFF) evaluated and GEE/mixed models applied for multi-center data (if applicable).
- [ ] E-value calculated for primary exposure to quantify sensitivity to unmeasured confounding.

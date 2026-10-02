---
name: medstat-causal-meta
description: Causal inference via Propensity Score Matching (PSM), Austin 2009 covariate balance diagnostics, Love plots, Bland-Altman limits of agreement with large-sample CIs, pure-SciPy Intraclass Correlation Coefficient (ICC), and random-effects meta-analysis with Egger's test. Use when conducting observational comparative effectiveness studies, propensity score matching, assessing rater agreement or reliability, pooling multi-study effect sizes, or evaluating publication bias.
---

# medstat-causal-meta: Causal Inference, Agreement & Meta-Analysis

Biostatistical engine for observational causal inference, rater reliability analysis, and multi-study evidence synthesis under STROBE and PRISMA standards.

## Core Rules

1. **Caliper Enforcement**: Propensity score matching requires a strict caliper ($0.20 \times \text{SD}(\text{logit } e_i)$) to prevent poor pairs. Unmatched subjects must be logged in the sample retention flow.
2. **Standardized Mean Difference Criterion**: Evaluate post-match balance across all baseline covariates; every covariate must achieve $|\text{SMD}| < 0.10$.
3. **Pure-SciPy ICC (GPL-Free)**: Compute intraclass correlation coefficients via pure two-way ANOVA decomposition without external GPL dependencies.
4. **Meta-Analysis Model Selection**: Model choice (fixed-effect vs. DerSimonian-Laird random-effects) must reflect the study design and clinical/methodological variation assumptions; Cochrane advises against selecting models solely by statistical heterogeneity tests ($I^2$).

## Execution Sequence

```
[1. PSM & BALANCE] ──▶ [2. LOVE PLOT] ──▶ [3. AGREEMENT (BA & ICC)] ──▶ [4. META-ANALYSIS & EGGER]
```

### Step 1: Propensity Score Matching & Balance Check

Perform 1:1 nearest-neighbor matching on logit propensity score with caliper:

```bash
medstat causal psm --data <observational_cohort.csv> \
  --treatment <treatment_col> \
  --covariates "age,sex,bmi,egfr,diabetes,hypertension" \
  --caliper 0.20 \
  --ratio 1 \
  --balance-check \
  --love-plot love_plot.json \
  --output matched_cohort.csv
```

- **Output Evaluation**:
  - Review pre-match vs. post-match Standardized Mean Differences (SMD).
  - Confirm all post-match absolute SMDs ($|\text{SMD}|$) are below the $0.10$ threshold (Austin 2009).
  - Unmatched control/treatment rows are tracked in the sample retention flow ($N_{\text{initial}} \to N_{\text{excluded}} \to N_{\text{matched}}$).
  - Consult [references/balance-diagnostics.md](references/balance-diagnostics.md) for balance criteria and formulas.

### Step 2: Rater Agreement & Measurement Reliability

#### Bland-Altman Analysis (Paired Continuous Measures)
Compute mean bias, limits of agreement, and large-sample approximate confidence intervals:

```bash
medstat agreement bland-altman --data <paired_device_trials.csv> \
  --m1 reference_method \
  --m2 investigational_device \
  --output bland_altman_res.json
```

- Mean bias ($\bar{d}$) evaluates systematic over/under-estimation.
- $95\%$ Limits of Agreement ($\bar{d} \pm 1.96 \cdot s_d$) estimate the interval containing approximately $95\%$ of paired differences, valid under the assumption of approximately normally distributed paired differences with no material trend or changing spread (heteroscedasticity); assess these assumptions before clinical interpretation.

#### Pure-SciPy Intraclass Correlation Coefficient (ICC)
Evaluate intra- or inter-rater reliability across targets and raters:

```bash
medstat agreement icc --data <rater_scores.csv> \
  --targets subject_id \
  --raters rater_id \
  --ratings pocus_score \
  --type icc2 \
  --output icc_res.json
```

- **ICC Variant Selection**:
  - `icc1`: One-way random effects (different raters per subject).
  - `icc2`: Two-way random effects, absolute agreement (generalizable raters; standard for clinical trials).
  - `icc3`: Two-way mixed effects, consistency (fixed panel of expert clinicians).
  - `icc2_k`: Average score of $k$ independent raters.
- **Interpretation**: Koo & Li (2016): apply reliability categories to the 95% confidence interval rather than point estimate alone ($\text{ICC} < 0.50$ Poor, $0.50 \le \text{ICC} < 0.75$ Moderate, $0.75 \le \text{ICC} \le 0.90$ Good, $\text{ICC} > 0.90$ Excellent). When the 95% CI spans multiple categories, report the interval and characterize reliability by the spanning range (e.g., 95% CI 0.68–0.82 indicates moderate-to-good reliability).

### Step 3: Meta-Analysis & Funnel Plot Publication Bias

Pool effect sizes (log odds ratios, log hazard ratios, or mean differences) across studies:

```bash
# Model selection: use --model random (e.g. DerSimonian-Laird via --method dl) when pre-specified by analysis plan
medstat meta --data <clinical_trials.csv> \
  --effect-col log_hr \
  --se-col se_log_hr \
  --study-col trial_name \
  --model random \
  --method dl \
  --forest-plot forest_plot.json \
  --egger \
  --output meta_analysis.json
```

- **Heterogeneity Assessment**:
  - Cochran's $Q$: Chi-square test of homogeneity ($p < 0.10$ indicates significant between-study variance).
  - Higgins $I^2$: Percentage of total variability due to between-study heterogeneity ($I^2 \ge 50\%$ indicates substantial heterogeneity).
  - $\tau^2$: Between-study variance estimate via DerSimonian-Laird method.
- **Publication Bias & Funnel Asymmetry**:
  - Egger's linear regression test checks if standardized effect sizes regress on precision ($p < 0.10$ signals asymmetry / small-study effects).

---

## Adaptive Python Scripting Protocol (ปรับแต่งสคริปต์ Causal Inference, Agreement และ Meta-Analysis ตามข้อมูลจริง)

> **Core Philosophy**: Never execute rigid canned scripts that make naive assumptions about file structure. The agent is empowered with full autonomy to write, adapt, and run Python scripts (`scratch/causal_meta.py`) tailored to observational cohorts, paired measurement trials, or multi-study systematic review datasets.
>
> ⚠️ **Mandatory Directive — ต้องดู Script ต้นแบบประกอบเสมอ (Review Prototype Scripts First)**:
> แม้จะให้อิสระ Agent ในการเขียนและปรับ Python Script เองตามสภาพข้อมูลจริง แต่ **Agent ต้องเปิดดูและอ้างอิงสคริปต์ต้นแบบ (Prototype Scripts)** หรือศึกษาการคำนวณในโมดูลแกนกลาง `src/medstat/` เสมอ เพื่อยึดมาตรฐานความถูกต้องทางชีวสถิติการแพทย์:
> - **Propensity Score Matching & Balance**: ดูการจับคู่ 1:1 nearest neighbor บน logit propensity score, Caliper $0.2 \times \text{SD}$, และการตรวจสอบ post-match $|\text{SMD}| < 0.10$ จาก `src/medstat/causal/psm.py` และ `src/medstat/causal/balance.py`
> - **Bland-Altman Limits of Agreement**: ดูการคำนวณ mean bias ($\bar{d}$), 95% LoA ($\bar{d} \pm 1.96 \cdot s_d$), และ large-sample CIs จาก `src/medstat/agreement/bland_altman.py`
> - **Pure-SciPy Intraclass Correlation (ICC)**: ดู two-way ANOVA decomposition (ICC1, ICC2, ICC3) โดยไม่พึ่งพา GPL library จาก `src/medstat/agreement/icc.py`
> - **Random-Effects Meta-Analysis**: ดูการคำนวณ DerSimonian-Laird pooled effect, Cochran's $Q$, $I^2$, $\tau^2$, และ Egger's test จาก `src/medstat/meta/models.py` และ `src/medstat/meta/bias.py`
>
> **วงจรการทำงานของ Agent**:
> `[1. สำรวจลักษณะข้อมูลทางคลินิก] ──▶ [2. ดูสคริปต์ต้นแบบเพื่อยึดหลักชีวสถิติ] ──▶ [3. ปรับโค้ดและดำเนินการวิเคราะห์]`

### Master Prototype Script for Causal & Agreement Analysis (สคริปต์ต้นแบบมาตรฐาน)

Agent ควรนำโครงสร้างและฟังก์ชันของสคริปต์ต้นแบบนี้ไปปรับแต่งลงใน workspace (เช่น `scratch/causal_meta.py`) ให้เข้ากับโครงสร้างข้อมูลจริง:

```python
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ------------------------------------------------------------------------------
# 1. PROPENSITY SCORE MATCHING (Austin 2009 Standard)
# ------------------------------------------------------------------------------
def run_psm_pipeline(df, treatment_col, covariate_cols, caliper_sd=0.20):
    # Fit propensity score model
    formula = f"{treatment_col} ~ " + " + ".join(covariate_cols)
    ps_model = smf.logit(formula, data=df).fit(disp=False)
    df = df.copy()
    df['ps'] = ps_model.predict(df)
    df['logit_ps'] = np.log(df['ps'] / (1.0 - df['ps']))
    
    caliper = caliper_sd * df['logit_ps'].std()
    
    treated = df[df[treatment_col] == 1].copy()
    control = df[df[treatment_col] == 0].copy()
    
    matched_pairs = []
    available_ctrl_idx = set(control.index)
    
    # 1:1 Nearest-Neighbor Matching within Caliper
    for t_idx, t_row in treated.iterrows():
        if not available_ctrl_idx:
            break
        ctrl_subset = control.loc[list(available_ctrl_idx)]
        diffs = (ctrl_subset['logit_ps'] - t_row['logit_ps']).abs()
        min_diff = diffs.min()
        if min_diff <= caliper:
            best_match_idx = diffs.idxmin()
            matched_pairs.append((t_idx, best_match_idx))
            available_ctrl_idx.remove(best_match_idx)
            
    print(f"Matched {len(matched_pairs)} / {len(treated)} treated subjects (Caliper = {caliper:.4f})")
    
    # Post-Match Balance Check (SMD < 0.10)
    matched_idx = [t for t, c in matched_pairs] + [c for t, c in matched_pairs]
    df_matched = df.loc[matched_idx]
    for cov in covariate_cols:
        g1 = df_matched[df_matched[treatment_col] == 1][cov].dropna()
        g0 = df_matched[df_matched[treatment_col] == 0][cov].dropna()
        pooled_sd = np.sqrt((g1.var() + g0.var()) / 2.0)
        smd = abs(g1.mean() - g0.mean()) / pooled_sd if pooled_sd > 0 else 0.0
        status = "PASSED (< 0.10)" if smd < 0.10 else "UNBALANCED (>= 0.10)"
        print(f"Post-match SMD for {cov}: {smd:.3f} -> {status}")
    return df_matched

# ------------------------------------------------------------------------------
# 2. BLAND-ALTMAN LIMITS OF AGREEMENT
# ------------------------------------------------------------------------------
def run_bland_altman(m1_series, m2_series):
    diff = m1_series - m2_series
    mean_bias = diff.mean()
    sd_diff = diff.std()
    loa_upper = mean_bias + 1.96 * sd_diff
    loa_lower = mean_bias - 1.96 * sd_diff
    print(f"Bland-Altman: Mean Bias = {mean_bias:.2f}, 95% LoA = [{loa_lower:.2f}, {loa_upper:.2f}]")
    return {"mean_bias": mean_bias, "loa_lower": loa_lower, "loa_upper": loa_upper}
```

The agent may freely incorporate `medstat` modules (e.g. `from medstat.causal.psm import match_propensity_scores`, `from medstat.agreement.icc import calculate_icc`, `from medstat.meta.models import run_meta_analysis`) or standard libraries as appropriate.

---

## Completion Criteria

- [ ] Propensity score matching completed with documented caliper and sample retention flow.
- [ ] Post-matching covariate balance confirmed with all $|\text{SMD}| < 0.10$ or residual imbalance noted.
- [ ] Bland-Altman or ICC reliability metrics evaluated with 95% confidence intervals.
- [ ] Meta-analytic pooled effect computed with justified model selection (fixed vs. random effects) based on study design and variation assumptions, reporting $I^2$ and Cochran's $Q$.
- [ ] Publication bias evaluated via Egger's test when $\ge 10$ studies are pooled.

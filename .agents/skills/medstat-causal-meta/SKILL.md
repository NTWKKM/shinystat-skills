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
5. **Confounder Selection Invariant for PSM**: Include *only* baseline pre-treatment confounders that are causally related to treatment choice and/or outcome. Strictly exclude post-treatment variables, mediators (variables on the causal pathway between treatment and outcome), or colliders, which introduce conditioning bias and artifactual confounding.
6. **Association vs Agreement Trap**: Never substitute Pearson ($r$) or Spearman ($\rho$) correlation for rater or device agreement. Correlation evaluates linear association, not agreement. Enforce **Bland-Altman 95% Limits of Agreement** (with large-sample CIs) or **Intraclass Correlation (ICC)**.
7. **Egger's Test Scope & Limitations**: Egger's linear regression test for funnel plot asymmetry requires at least $k \ge 10$ studies with continuous effect measures. The `--egger` command path requires explicit continuous effect measure options (e.g. `--measure continuous` or `--measure md`) and rejects fewer than 10 studies ($k < 10$), standardized mean differences (`smd`), and binary log odds-ratio effect measures (e.g. `log_or`, `or`, `odds_ratio`) where artifactual correlation between effect size and standard error induces false-positive asymmetry.

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
  --effect-col mean_diff \
  --se-col se_mean_diff \
  --study-col trial_name \
  --model random \
  --method dl \
  --measure continuous \
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
> 🔒 **Subprocess & Script Execution Safety**:
> When generating and running analysis scripts (`scratch/causal_meta.py`), enforce execution controls: disable shell/subprocess access, limit file reads strictly to the designated dataset and referenced prototype/core modules, limit file writes strictly to scratch and designated output paths, and ensure no access to credentials or environment secrets. Require explicit user confirmation if the runtime cannot enforce these sandbox controls.
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
    if len(covariate_cols) != len(set(covariate_cols)):
        raise ValueError("Duplicate covariate columns detected.")
    if treatment_col in covariate_cols:
        raise ValueError(f"Treatment column '{treatment_col}' must not also be listed as a covariate.")
    # Fit propensity score model with collision-free column identifiers
    prefix = "_psm_tmp_"
    col_map = {treatment_col: f"{prefix}trt"}
    for i, col in enumerate(covariate_cols):
        col_map[col] = f"{prefix}cov_{i}"
    df_safe = df.rename(columns=col_map)
    formula = f"{prefix}trt ~ " + " + ".join(col_map[col] for col in covariate_cols)
    # Mirror calculate_propensity_score (src/medstat/causal/psm.py): retry with a penalized fit
    # on exceptions or non-convergence; never match on unconverged estimates.
    try:
        ps_model = smf.logit(formula, data=df_safe).fit(disp=False)
        if not ps_model.mle_retvals.get("converged", False):
            raise RuntimeError("Propensity score MLE did not converge.")
    except Exception:
        try:
            ps_model = smf.logit(formula, data=df_safe).fit_regularized(disp=False)
            if not ps_model.mle_retvals.get("converged", False):
                raise RuntimeError("Regularized propensity score fit did not converge.")
        except Exception as exc:
            raise ValueError(
                "Propensity score model failed to converge; consider Firth penalization "
                "(medstat.models.firth.fit_firth_logistic) or covariate reduction before matching."
            ) from exc
    df = df.copy()
    # Clip propensity scores strictly within [1e-7, 1 - 1e-7] so logit_ps remains finite
    df['ps'] = ps_model.predict(df_safe).clip(1e-7, 1.0 - 1e-7)
    df['logit_ps'] = np.log(df['ps'] / (1.0 - df['ps']))
    
    valid_ps = np.isfinite(df['logit_ps']) & df[treatment_col].isin([0, 1])
    if not valid_ps.all():
        raise ValueError(f"Rejecting non-finite propensity scores or non-binary treatment in {(~valid_ps).sum()} rows.")

    caliper = caliper_sd * df['logit_ps'].std()
    
    # Assign unique internal row IDs to eliminate ambiguous selection and extraction from duplicate index labels
    df['_row_id'] = np.arange(len(df))
    treated = df[df[treatment_col] == 1].copy().set_index('_row_id', drop=False)
    control = df[df[treatment_col] == 0].copy().set_index('_row_id', drop=False)
    
    matched_pairs = []
    available_ctrl_ids = set(control.index)
    
    # 1:1 Nearest-Neighbor Matching within Caliper
    # Deterministic treated-subject order (sort by logit_ps descending to prevent input row order dependency)
    treated_sorted = treated.sort_values('logit_ps', ascending=False)
    for t_id, t_row in treated_sorted.iterrows():
        if not available_ctrl_ids:
            break
        # Deterministic control order and tie-breaking by unique row id
        ctrl_subset = control.loc[sorted(available_ctrl_ids)]
        diffs = (ctrl_subset['logit_ps'] - t_row['logit_ps']).abs()
        min_diff = diffs.min()
        if min_diff <= caliper:
            tied_candidates = diffs[diffs == min_diff].index
            best_match_id = sorted(tied_candidates)[0]
            matched_pairs.append((t_id, best_match_id))
            available_ctrl_ids.remove(best_match_id)
            
    n_treated_initial = len(treated)
    n_control_initial = len(control)
    n_matched = len(matched_pairs)
    print(f"Matching retention flow (Caliper = {caliper:.4f}):")
    print(f"  Treated: Initial = {n_treated_initial}, Matched = {n_matched}, Unmatched = {n_treated_initial - n_matched}")
    print(f"  Control: Initial = {n_control_initial}, Matched = {n_matched}, Unmatched = {n_control_initial - n_matched}")
    
    # Post-Match Balance Check (SMD < 0.10)
    matched_ids = [t for t, c in matched_pairs] + [c for t, c in matched_pairs]
    df_matched = df.iloc[matched_ids].copy()
    for cov in covariate_cols:
        # Encode or assess categorical covariates per category before numeric mean/var
        if not pd.api.types.is_numeric_dtype(df_matched[cov]):
            cov_nonmissing = df_matched[[treatment_col, cov]].dropna()
            cats = pd.get_dummies(cov_nonmissing[cov], drop_first=(cov_nonmissing[cov].nunique() == 2), prefix=cov)
            sub_covs = cats.columns.tolist()
            df_eval = pd.concat([cov_nonmissing[[treatment_col]], cats], axis=1)
        else:
            sub_covs = [cov]
            df_eval = df_matched[[treatment_col, cov]].dropna()

        for sc in sub_covs:
            g1 = df_eval[df_eval[treatment_col] == 1][sc].dropna()
            g0 = df_eval[df_eval[treatment_col] == 0][sc].dropna()
            if len(g1) < 2 or len(g0) < 2:
                print(f"Post-match SMD for {sc}: NOT ESTIMABLE (insufficient sample, n < 2)")
                continue
            diff = abs(g1.mean() - g0.mean())
            pooled_sd = np.sqrt((g1.var(ddof=1) + g0.var(ddof=1)) / 2.0)
            if pooled_sd == 0:
                smd = 0.0 if diff == 0 else np.nan
            else:
                smd = diff / pooled_sd

            if np.isnan(smd):
                status = "NOT ESTIMABLE / UNBALANCED (zero SD with non-zero mean difference)"
                print(f"Post-match SMD for {sc}: NaN -> {status}")
            else:
                status = "PASSED (< 0.10)" if smd < 0.10 else "UNBALANCED (>= 0.10)"
                print(f"Post-match SMD for {sc}: {smd:.3f} -> {status}")
    return df_matched

# ------------------------------------------------------------------------------
# 2. BLAND-ALTMAN LIMITS OF AGREEMENT (with Configurable Confidence Intervals)
# ------------------------------------------------------------------------------
def run_bland_altman(m1_series, m2_series, ci=0.95):
    # Verify equal lengths and subtract positional values to avoid pandas index alignment
    m1 = np.asarray(m1_series, dtype=float)
    m2 = np.asarray(m2_series, dtype=float)
    if len(m1) != len(m2):
        raise ValueError(f"Bland-Altman requires paired inputs of equal length (got {len(m1)} and {len(m2)}).")
    valid_mask = np.isfinite(m1) & np.isfinite(m2)
    m1_clean = m1[valid_mask]
    m2_clean = m2[valid_mask]
    diff = m1_clean - m2_clean
    n = len(diff)
    if n < 2:
        raise ValueError(f"Bland-Altman requires at least 2 non-missing pairs (got {n}).")
    if not (0.0 < ci < 1.0):
        raise ValueError(f"Confidence level ci must be strictly between 0 and 1 (got {ci}).")
    mean_bias = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))
    se_bias = sd_diff / np.sqrt(n)
    t_crit = float(stats.t.ppf(1.0 - (1.0 - ci) / 2.0, df=n - 1))
    ci_bias = (mean_bias - t_crit * se_bias, mean_bias + t_crit * se_bias)
    
    z_loa = float(stats.norm.ppf(1.0 - (1.0 - ci) / 2.0))
    loa_upper = mean_bias + z_loa * sd_diff
    loa_lower = mean_bias - z_loa * sd_diff
    se_loa = float(np.sqrt((1.0 / n + (z_loa**2) / (2.0 * (n - 1))) * (sd_diff**2)))
    ci_loa_upper = (loa_upper - t_crit * se_loa, loa_upper + t_crit * se_loa)
    ci_loa_lower = (loa_lower - t_crit * se_loa, loa_lower + t_crit * se_loa)
    ci_pct = int(round(ci * 100))
    print(f"Bland-Altman: Mean Bias = {mean_bias:.2f} ({ci_pct}% CI [{ci_bias[0]:.2f}, {ci_bias[1]:.2f}]), "
          f"{ci_pct}% LoA = [{loa_lower:.2f}, {loa_upper:.2f}] (LoA Lower {ci_pct}% CI [{ci_loa_lower[0]:.2f}, {ci_loa_lower[1]:.2f}], "
          f"LoA Upper {ci_pct}% CI [{ci_loa_upper[0]:.2f}, {ci_loa_upper[1]:.2f}])")
    return {
        "mean_bias": mean_bias,
        "ci_mean_bias": ci_bias,
        "loa_lower": loa_lower,
        "ci_loa_lower": ci_loa_lower,
        "loa_upper": loa_upper,
        "ci_loa_upper": ci_loa_upper,
    }
```

The agent may freely incorporate `medstat` modules (e.g. `from medstat.causal.psm import match_propensity_scores`, `from medstat.agreement.icc import calculate_icc`, `from medstat.meta.models import run_meta_analysis`) or standard libraries as appropriate.

---

## Completion Criteria

- [ ] Propensity score matching completed with documented caliper and sample retention flow.
- [ ] Post-matching covariate balance confirmed with all $|\text{SMD}| < 0.10$ or residual imbalance noted.
- [ ] Bland-Altman or ICC reliability metrics evaluated with 95% confidence intervals.
- [ ] Meta-analytic pooled effect computed with justified model selection (fixed vs. random effects) based on study design and variation assumptions, reporting $I^2$ and Cochran's $Q$.
- [ ] Publication bias evaluated via Egger's test when $\ge 10$ studies are pooled.

---
name: medstat-causal-meta
description: Causal inference via Propensity Score Matching (PSM), Austin 2009 covariate balance diagnostics, Love plots, Bland-Altman limits of agreement with Carkeet CIs, pure-SciPy Intraclass Correlation Coefficient (ICC), and random-effects meta-analysis with Egger's test. Use when conducting observational comparative effectiveness studies, propensity score matching, assessing rater agreement or reliability, pooling multi-study effect sizes, or evaluating publication bias.
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
Compute mean bias, limits of agreement, and Carkeet confidence intervals:

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
- **Interpretation**: Koo & Li (2016): $\text{ICC} < 0.50$ Poor, $0.50 \le \text{ICC} < 0.75$ Moderate, $0.75 \le \text{ICC} \le 0.90$ Good, $\text{ICC} > 0.90$ Excellent.

### Step 3: Meta-Analysis & Funnel Plot Publication Bias

Pool effect sizes (log odds ratios, log hazard ratios, or mean differences) across studies:

```bash
medstat meta --data <clinical_trials.csv> \
  --effect-col log_hr \
  --se-col se_log_hr \
  --study-col trial_name \
  --model random \         # Use when random-effects is pre-specified by the analysis plan
  --method dl \            # DerSimonian-Laird; select method per analysis plan
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

## Completion Criteria

- [ ] Propensity score matching completed with documented caliper and sample retention flow.
- [ ] Post-matching covariate balance confirmed with all $|\text{SMD}| < 0.10$ or residual imbalance noted.
- [ ] Bland-Altman or ICC reliability metrics evaluated with 95% confidence intervals.
- [ ] Meta-analytic pooled effect computed with justified model selection (fixed vs. random effects) based on study design and variation assumptions, reporting $I^2$ and Cochran's $Q$.
- [ ] Publication bias evaluated via Egger's test when $\ge 10$ studies are pooled.

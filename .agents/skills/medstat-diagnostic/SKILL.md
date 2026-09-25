---
name: medstat-diagnostic
description: Diagnostic test accuracy evaluation, 2x2 contingency matrices, empirical ROC curves with DeLong 95% CIs, paired DeLong biomarker comparisons, and Vickers Decision Curve Analysis (DCA) net benefit. Use when validating clinical biomarkers, point-of-care diagnostics, triage scores, ROC comparisons, or decision-analytic clinical utility.
---

# medstat-diagnostic: Diagnostic Accuracy, ROC & Decision Curve Analysis

Validation engine for clinical biomarkers, laboratory diagnostic assays, point-of-care ultrasound, and clinical risk scores under STARD and TRIPOD reporting standards.

## Core Rules

1. **Wilson Score Intervals for Proportions**: Never use normal Wald approximation intervals for sensitivity or specificity near 0 or 1; Wilson score intervals are mandatory.
2. **DeLong Variance for ROC**: Empirical AUC confidence intervals and paired comparisons must use non-parametric placement values via DeLong's method.
3. **Clinical Utility Beyond Accuracy**: High AUC does not guarantee clinical utility. Decision Curve Analysis (DCA) is required to establish positive net benefit over "Treat All" and "Treat None" across threshold probabilities.

## Execution Sequence

```
[1. 2x2 ACCURACY] ──▶ [2. ROC & DELONG] ──▶ [3. BIOMARKER COMPARISON] ──▶ [4. DCA NET BENEFIT]
```

### Step 1: Calculate 2x2 Contingency Matrix & Accuracy Metrics

Evaluate classification performance at a clinically defined cut-off:

```bash
medstat diag --data <cohort.csv> \
  --gold-standard <disease_col> \
  --test-col <biomarker_score> \
  --cutoff 2.0 \
  --output diag_accuracy.json
```

Output includes:
- **Sensitivity & Specificity**: True positive / true negative proportions with Wilson 95% CIs.
- **Positive & Negative Predictive Values (PPV, NPV)**: Disease prevalence-adjusted.
- **Positive & Negative Likelihood Ratios (LR+, LR-)**: Pre-test to post-test odds transitions.
- **Diagnostic Odds Ratio (DOR)**: $(\text{TP} \times \text{TN}) / (\text{FP} \times \text{FN})$. When any cell contains zero, a Haldane–Anscombe 0.5 correction is added before calculation.

### Step 2: Empirical ROC Analysis & Youden's Index

Compute empirical AUC with DeLong confidence intervals and calculate Youden's Index:

```bash
medstat diag --data <cohort.csv> \
  --gold-standard <disease_col> \
  --test-col <biomarker_score> \
  --roc \
  --output roc_results.json
```

- **AUC & DeLong 95% CIs**: Analytical standard errors without bootstrapping.
- **Youden's J**: Optimal index maximizing $J = \text{Sensitivity} + \text{Specificity} - 1$.
- Consult [references/diagnostic-cutpoints.md](references/diagnostic-cutpoints.md) for clinical cutpoint trade-offs.

### Step 3: Compare Correlated Diagnostic Biomarkers (Paired DeLong)

Test whether a new biomarker significantly outperforms an existing comparator biomarker (note: `--gold-standard` supplies the disease-status labels, while `--compare-roc` identifies the comparator):

```bash
medstat diag --data <cohort.csv> \
  --gold-standard <disease_col> \
  --test-col <new_biomarker> \
  --roc \
  --compare-roc <established_biomarker> \
  --output roc_comparison.json
```

- Returns difference in AUC ($\Delta\text{AUC}$), asymptotic z-statistic, and two-tailed p-value accounting for biomarker covariance.

### Step 4: Decision Curve Analysis (DCA) Net Benefit

Quantify clinical net benefit across decision threshold probabilities $p_t$:

```bash
medstat diag --data <cohort.csv> \
  --gold-standard <disease_col> \
  --test-col <predicted_risk_prob> \
  --dca \
  --output dca_net_benefit.json
```

- **Net Benefit Formula**:
  $$\text{Net Benefit}(p_t) = \frac{\text{TP}}{N} - \frac{\text{FP}}{N} \cdot \left(\frac{p_t}{1 - p_t}\right)$$
- Clinical Rule: A biomarker should only be deployed across threshold ranges where its net benefit curve exceeds both "Treat All" and "Treat None".

## Completion Criteria

- [ ] 2x2 contingency table evaluated with Wilson 95% confidence intervals.
- [ ] ROC AUC calculated with analytical DeLong 95% CIs.
- [ ] Paired DeLong test executed if comparing multiple diagnostic tests.
- [ ] DCA net benefit confirmed superior to default strategies across the target decision range.

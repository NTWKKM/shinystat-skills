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
4. **Prespecified Directionality Rule (Low-is-Abnormal)**: Before calculating ROC or cutpoint metrics, establish biomarker directionality based on clinical biological mechanisms (e.g. abnormal when high: Troponin, Lactate; abnormal when low: Platelet count, eGFR, PaO2/FiO2). Recode or invert low-is-abnormal scores prior to analysis as a prespecified transformation. Never automatically invert scores post-hoc based solely on observing empirical $\text{AUC} < 0.50$; an unexpected low AUC must be reported and investigated for assay miscalibration, labeling reversal, or data errors rather than reversed after observing the data.
5. **Cutpoint Selection & Anti-P-Hacking**: Cutpoints must either be pre-specified by clinical guidelines or derived via objective metrics (Youden's Index $J = \text{Sens} + \text{Spec} - 1$, or a prespecified minimum sensitivity tier like 95% for triage screening). Never data-dredge through arbitrary cutpoints to maximize statistical significance without multiplicity disclosure.

## Execution Sequence

```
[1. 2x2 ACCURACY] ──▶ [2. ROC & DELONG] ──▶ [3. BIOMARKER COMPARISON] ──▶ [4. DCA NET BENEFIT]
```

### Step 1: Calculate 2x2 Contingency Matrix & Accuracy Metrics

Orient each biomarker score so higher values indicate disease-positive before cutoff classification, ROC analysis, or paired DeLong analysis. Scores greater than or equal to `--cutoff` are classified as positive; for low-is-positive biomarkers (e.g. platelet count in thrombocytopenia, PaO2/FiO2 in ARDS), invert or recode scores first:

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

- **Encoding Requirement**: `--gold-standard` must strictly contain only `0` (control / non-disease) and `1` (disease / event), matching the encoding required by `calculate_dca` and clinical contingency calculations.
- **Net Benefit Formula**:
  $$\text{Net Benefit}(p_t) = \frac{\text{TP}}{N} - \frac{\text{FP}}{N} \cdot \left(\frac{p_t}{1 - p_t}\right)$$
- Clinical Rule: A biomarker should only be deployed across threshold ranges where its net benefit curve exceeds both "Treat All" and "Treat None".

---

## Adaptive Python Scripting Protocol (ปรับแต่งสคริปต์ความแม่นยำในการวินิจฉัยและ ROC ตามข้อมูลจริง)

> **Core Philosophy**: Never execute rigid canned scripts that make naive assumptions about file structure. The agent is empowered with full autonomy to write, adapt, and run Python scripts (`scratch/diagnostic.py`) tailored to specific clinical biomarkers, cutoff evaluations, and risk scoring tools.
>
> ⚠️ **Mandatory Directive — ต้องดู Script ต้นแบบประกอบเสมอ (Review Prototype Scripts First)**:
> แม้จะให้อิสระ Agent ในการเขียนและปรับ Python Script เองตามสภาพข้อมูลจริง แต่ **Agent ต้องเปิดดูและอ้างอิงสคริปต์ต้นแบบ (Prototype Scripts)** หรือศึกษาการคำนวณในโมดูลแกนกลาง `src/medstat/diagnostic/` เสมอ เพื่อยึดมาตรฐานความถูกต้องทางชีวสถิติการแพทย์:
> - **ตาราง 2x2 และ Wilson Score Interval**: ดูการคำนวณ Sensitivity, Specificity, PPV, NPV, LR+, LR- พร้อม Wilson Score 95% CIs จาก `src/medstat/diagnostic/accuracy.py`
> - **Empirical ROC & DeLong Test**: ดูการคำนวณ Area Under Curve (AUC), DeLong 95% CIs, และ Paired DeLong test จาก `src/medstat/diagnostic/roc.py`
> - **Decision Curve Analysis (DCA)**: ดูการคำนวณ Net Benefit ข้าม Decision Threshold Probabilities ($p_t$) และการเปรียบเทียบ Treat All / Treat None จาก `src/medstat/diagnostic/dca.py`
>
> **วงจรการทำงานของ Agent**:
> `[1. สำรวจ Gold Standard และ Biomarker] ──▶ [2. ดูสคริปต์ต้นแบบเพื่อยึดหลักชีวสถิติ] ──▶ [3. ปรับโค้ดและประเมินความแม่นยำ]`

### Master Prototype Script for Diagnostic Accuracy & ROC (สคริปต์ต้นแบบมาตรฐาน)

Agent ควรนำโครงสร้างและฟังก์ชันของสคริปต์ต้นแบบนี้ไปปรับแต่งลงใน workspace (เช่น `scratch/diagnostic.py`) ให้เข้ากับตัวชี้วัดและจุดตัด (cut-off) ของข้อมูลจริง:

```python
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_curve, auc

# 1. LOAD DATA & VERIFY ENDPOINTS (Strict Numeric 0/1)
df = pd.read_csv("clean_cohort.csv")
# gold_standard: 1 = Disease/Event, 0 = Non-disease
# test_score: continuous biomarker or predicted probability

# 2. 2x2 CONTINGENCY MATRIX WITH WILSON SCORE INTERVAL
def wilson_score_interval(k, n, confidence=0.95):
    """Compute Wilson score interval for binomial proportions"""
    if n == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)

def evaluate_cutoff(gold, score, cutoff):
    pred = (score >= cutoff).astype(int)
    tp = np.sum((gold == 1) & (pred == 1))
    fp = np.sum((gold == 0) & (pred == 1))
    tn = np.sum((gold == 0) & (pred == 0))
    fn = np.sum((gold == 1) & (pred == 0))
    
    sens, (sens_l, sens_u) = tp / (tp + fn) if (tp + fn) > 0 else 0, wilson_score_interval(tp, tp + fn)
    spec, (spec_l, spec_u) = tn / (tn + fp) if (tn + fp) > 0 else 0, wilson_score_interval(tn, tn + fp)
    ppv, (ppv_l, ppv_u) = tp / (tp + fp) if (tp + fp) > 0 else 0, wilson_score_interval(tp, tp + fp)
    npv, (npv_l, npv_u) = tn / (tn + fn) if (tn + fn) > 0 else 0, wilson_score_interval(tn, tn + fn)
    
    print(f"Cutoff >= {cutoff}:")
    print(f"  Sensitivity: {sens*100:.1f}% (95% CI: {sens_l*100:.1f}% - {sens_u*100:.1f}%)")
    print(f"  Specificity: {spec*100:.1f}% (95% CI: {spec_l*100:.1f}% - {spec_u*100:.1f}%)")
    print(f"  PPV: {ppv*100:.1f}% | NPV: {npv*100:.1f}%")
    return {"sens": sens, "spec": spec, "ppv": ppv, "npv": npv}

# 3. EMPIRICAL ROC, DIRECTIONALITY SANITY CHECK & YOUDEN'S INDEX
scores = df['test_score'].values
fpr, tpr, thresholds = roc_curve(df['gold_standard'], scores)
roc_auc = auc(fpr, tpr)

# Directionality: Verify prespecified clinical orientation (low-is-abnormal markers like eGFR or Platelets must be prespecified and recoded prior to analysis)
# If an empirical AUC < 0.50 occurs contrary to clinical expectation, report and investigate potential coding/assay error rather than post-hoc flipping
if roc_auc < 0.50:
    print(f"Warning: Empirical AUC is {roc_auc:.3f} (< 0.50). Verify biomarker clinical directionality or coding before reporting.")

youden_j = tpr - fpr
opt_idx = np.argmax(youden_j)
opt_cutoff = thresholds[opt_idx]
print(f"ROC AUC: {roc_auc:.3f} | Optimal Cutoff (Youden J): {opt_cutoff:.2f}")

# 4. DECISION CURVE ANALYSIS (DCA: Net Benefit with Treat All and Treat None Reference Strategies)
def calculate_net_benefit(gold, probs, thresholds_range):
    n = len(gold)
    tp_all = np.sum(gold == 1)
    fp_all = np.sum(gold == 0)
    net_benefits = []
    for pt in thresholds_range:
        pred = (probs >= pt).astype(int)
        tp = np.sum((gold == 1) & (pred == 1))
        fp = np.sum((gold == 0) & (pred == 1))
        w = pt / (1.0 - pt) if pt < 1.0 else 0.0
        nb_model = (tp / n) - (fp / n) * w
        nb_all = (tp_all / n) - (fp_all / n) * w
        net_benefits.append({
            "pt": pt,
            "net_benefit": nb_model,
            "treat_all": nb_all,
            "treat_none": 0.0,
        })
    return pd.DataFrame(net_benefits)
```

The agent may freely incorporate `medstat` modules (e.g. `from medstat.diagnostic.roc import calculate_delong_ci`, `from medstat.diagnostic.accuracy import compute_diagnostic_metrics`, `from medstat.diagnostic.dca import calculate_dca`) or standard libraries as appropriate.

---

## Completion Criteria

- [ ] 2x2 contingency table evaluated with Wilson 95% confidence intervals.
- [ ] ROC AUC calculated with analytical DeLong 95% CIs.
- [ ] Paired DeLong test executed if comparing multiple diagnostic tests.
- [ ] DCA net benefit confirmed superior to default strategies across the target decision range.

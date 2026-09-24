# Statistical Analysis Plan (SAP) YAML Spec Schema & Model Options

Reference schema for defining headless, executable statistical analysis plans in `medstat`.

## 1. Minimal SAP Spec Example

```yaml
version: "1.0"
metadata:
  study_title: "Comparative Effectiveness of Statin Therapy on 3-Year MACE"
  protocol_id: "SAP-CVD-2026-001"
  principal_investigator: "Clinical Research Team"
  statistical_analyst: "medstat automated engine"
  date: "2026-09-24"
  reporting_guideline: "STROBE"
  random_seed: 42

data:
  input_path: "cardiovascular_cohort.csv"
  id_column: "patient_id"

variables:
  - name: "cv_event"
    type: "binary"
    role: "outcome"
  - name: "statin_rx"
    type: "binary"
    role: "exposure"
    reference_category: "0"
  - name: "age"
    type: "continuous"
    role: "covariate"
  - name: "diabetes"
    type: "categorical"
    role: "covariate"
    reference_category: "No"

models:
  - name: "primary_adjusted_logistic"
    type: "logistic"
    outcome: "cv_event"
    exposure: "statin_rx"
    covariates: ["age", "diabetes", "ldl", "sbp"]
    method: "standard"
    missing_strategy: "mice"
    missing_justification: "Multiple imputation with m=5 chained equations under MAR"
    options:
      n_imputations: 5
      e_value: true
      confidence_level: 0.95
```

---

## 2. Supported Model Types & Options

| Model Type | CLI Option | Key Options | Diagnostic Requirement |
| :--- | :--- | :--- | :--- |
| `logistic` | `--type logistic` | `e_value: true` | Hosmer-Lemeshow / Brier score |
| `firth_logistic` | `--type firth_logistic` | `ci_method: "profile"`, `penalty_weight: 1.0` | Separation detection |
| `cox_ph` | `--type cox_ph --time <t>` | `schoenfeld: true`, `penalizer: 0.0` | Schoenfeld residual correlation |
| `firth_cox` | `--type cox_ph --method firth` | `ci_method: "profile"` | Profile likelihood convergence |
| `linear` | `--type linear` | `robust: "HC1"` | White's heteroskedasticity test |
| `rcs_cox` | Via Python API | `knots: 4`, `constraints: "center"` | Non-linearity Wald test ($p < 0.05$) |

---

## 3. Restricted Cubic Splines (RCS) Centering Constraints

When modeling non-linear continuous predictors using natural cubic splines:
- **Partition of Unity Problem**: Natural spline basis functions $\sum B_j(x) = 1.0$, which causes exact collinearity with the intercept in Cox regression.
- **Centering Solution**: Enforce centering constraint relative to the median reference value:
  $$f(x) - f(x_{\text{ref}}) = \beta_1 (s_1(x) - s_1(x_{\text{ref}})) + \dots + \beta_{k-1} (s_{k-1}(x) - s_{k-1}(x_{\text{ref}}))$$
- **Interpretation**: HR contrast curve plotted relative to $x_{\text{ref}}$ with 95% pointwise confidence intervals.

---

## 4. VanderWeele E-Value Math

The E-value is the minimum strength of association on the risk ratio scale that an unmeasured confounder must have with both the treatment and the outcome to explain away the observed treatment-outcome association.

### For Point Estimate ($RR \ge 1$):
$$\text{E-value} = RR + \sqrt{RR(RR - 1)}$$

### For Point Estimate ($RR < 1$):
First invert $RR^* = 1 / RR$, then compute:
$$\text{E-value} = RR^* + \sqrt{RR^*(RR^* - 1)}$$

### Odds Ratio Approximation:
When the outcome is rare (< 15%), $OR$ approximates $RR$ directly ($RR \approx OR$).
For common outcomes, if baseline risk $p_0$ is unavailable, the square-root approximation can be used:
$$RR \approx \sqrt{OR}$$
When baseline risk $p_0$ (unexposed outcome risk) is known, use the baseline-risk conversion formula:
$$RR = \frac{OR}{1 - p_0 + (p_0 \cdot OR)}$$

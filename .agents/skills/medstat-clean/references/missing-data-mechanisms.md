# Missing Data Mechanisms & Clinical Decision Framework

Reference guide for handling missing data in clinical registries, electronic health records (EHR), and epidemiological cohorts.

## 1. Rubin's Missing Data Taxonomy

| Mechanism | Definition | Clinical Example | Valid Approaches |
| :--- | :--- | :--- | :--- |
| **MCAR** (Missing Completely at Random) | Probability of missingness is independent of observed and unobserved data. | Lab tube dropped in transit; random equipment failure. | Complete-case analysis, Mean/Median imputation (exploratory only), MICE. |
| **MAR** (Missing at Random) | Probability of missingness depends on observed data but not on unobserved values. | Sicker/older patients have more missing ambulatory vitals, but age and disease severity are observed. | Multiple Imputation by Chained Equations (MICE), Full Information Maximum Likelihood (FIML). |
| **MNAR** (Missing Not at Random) | Probability of missingness depends directly on the unobserved value itself. | Patients with severe depression skip completing the depression questionnaire. | Pattern-mixture models, Heckman selection models, Sensitivity analyses. |

---

## 2. Little's MCAR Test ($d^2$)

Little's test evaluates whether missing data patterns across continuous variables share a common mean vector:

$$d^2 = \sum_{s=1}^S n_s (\bar{y}_{s,\text{obs}} - \hat{\mu}_{s,\text{obs}})^T \hat{\Sigma}_{s,\text{obs}}^{-1} (\bar{y}_{s,\text{obs}} - \hat{\mu}_{s,\text{obs}})$$

- **Null Hypothesis ($H_0$)**: Missingness is MCAR.
- **Degrees of Freedom**: $\text{df} = \sum_{s=1}^S p_s - P$, where $p_s$ is observed variables in pattern $s$ and $P$ is total variables.
- **Interpretation**:
  - $p > 0.05$: Insufficient evidence against MCAR. A nonsignificant result does not prove MCAR or automatically justify complete-case analysis; evaluate clinical context and missingness proportion.
  - $p \le 0.05$: Evidence against MCAR (departures from MCAR). The test cannot distinguish between MAR and MNAR; require a clinically justified mechanism assumption and sensitivity analysis.

---

## 3. Clinical Risk Tiers

| Tier | Missingness % | Clinical Risk Assessment | Action Mandate |
| :--- | :--- | :--- | :--- |
| **Low** | $< 5\%$ | Negligible impact on effect estimates if MCAR holds. | Complete-case exclusion allowed with documented rationale. |
| **Moderate** | $5\% - 20\%$ | Potential loss of statistical power and mild bias. | MICE with $m \ge 5$ imputations. |
| **High** | $20\% - 40\%$ | Severe risk of distortion and attenuation of effects. | MICE with $m \ge 20$ imputations; mandatory sensitivity analysis comparing CC vs MICE. |
| **Critical** | $> 40\%$ | High risk of residual confounding or structural non-response. | Missing indicator method, separate reporting, or drop variable from primary multivariable model. |

---

## 4. Rubin's Rules for Pooling Parameter Estimates

When analyzing $m$ multiply imputed datasets:

### Pooled Estimate
$$\bar{\theta} = \frac{1}{m} \sum_{i=1}^m \hat{\theta}_i$$

### Within-Imputation Variance
$$\bar{U} = \frac{1}{m} \sum_{i=1}^m \widehat{\text{Var}}(\hat{\theta}_i)$$

### Between-Imputation Variance
$$B = \frac{1}{m - 1} \sum_{i=1}^m (\hat{\theta}_i - \bar{\theta})^2$$

### Total Variance
$$T = \bar{U} + \left(1 + \frac{1}{m}\right) B$$

### Barnard-Rubin (1999) Degrees of Freedom
For small samples ($n_{\text{obs}} < 500$):
$$\nu_{\text{obs}} = \frac{n - p + 1}{n - p + 3} (n - p) (1 - \gamma)$$
$$\nu_{\text{adj}} = \frac{\nu_{\text{old}} \cdot \nu_{\text{obs}}}{\nu_{\text{old}} + \nu_{\text{obs}}}$$
where $\gamma = \frac{(1 + 1/m) B}{T}$ is the Fraction of Missing Information (FMI).

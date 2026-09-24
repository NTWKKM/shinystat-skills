# Covariate Balance, Agreement & Meta-Analytic Diagnostics

Reference guide for evaluating covariate balance in propensity score matched cohorts, rater agreement metrics, and meta-analytic heterogeneity.

---

## 1. Standardized Mean Difference (SMD)

Standardized Mean Difference is the gold standard metric for assessing balance between treatment ($T$) and control ($C$) groups because it is independent of sample size and measurement scale.

### Continuous Covariates
$$\text{SMD} = \frac{\bar{X}_T - \bar{X}_C}{\sqrt{\frac{s_T^2 + s_C^2}{2}}}$$

where $\bar{X}_T, \bar{X}_C$ are sample means, and $s_T^2, s_C^2$ are sample variances.

### Binary / Categorical Proportions
$$\text{SMD} = \frac{p_T - p_C}{\sqrt{\frac{p_T(1 - p_T) + p_C(1 - p_C)}{2}}}$$

where $p_T, p_C$ are proportions in each group.

### Clinical Balance Thresholds (Austin 2009)

| SMD Range | Balance Quality | Clinical Implication | Action Required |
| :--- | :--- | :--- | :--- |
| **$\lvert\text{SMD}\rvert < 0.10$** | **Well-balanced** | Negligible difference; groups are clinically comparable. | Proceed to outcome analysis. |
| **$0.10 \le \lvert\text{SMD}\rvert \le 0.20$** | **Moderate Imbalance** | Potential residual confounding across this covariate. | Re-estimate propensity score with interaction/polynomial terms, or include as covariate in outcome model (Doubly Robust). |
| **$\lvert\text{SMD}\rvert > 0.20$** | **Severe Imbalance** | Substantial difference; confounding bias likely. | Tighten caliper, trim non-overlapping support, or switch to propensity score weighting (IPTW). |

---

## 2. Caliper Matching on the Logit Propensity Score

Propensity score matching matches treated patients to control patients with similar estimated probabilities of treatment:
$$e_i = P(Z_i = 1 \mid \mathbf{X}_i)$$

### Logit Transformation
Matching directly on the propensity score $e_i$ can yield poor matches near the boundaries (0 and 1). Matching on the logit-transformed propensity score stabilizes variance:
$$L_i = \text{logit}(e_i) = \ln\left(\frac{e_i}{1 - e_i}\right)$$

### Optimal Caliper Width (Austin 2011)
$$\text{Caliper Width} = 0.20 \times \text{SD}(L)$$

- Austin demonstrated that a caliper of $0.2 \times \text{SD}(L)$ removed approximately 98% to 99% of the bias of the crude estimator in simulated mean-difference and risk-difference settings with at least some continuous covariates; caliper choice had much less impact when all covariates were binary. This applies strictly to measured baseline covariates and does not eliminate unmeasured confounding; residual unmeasured confounding still requires assessment.
- When matching ratio is 1:1 nearest neighbor without replacement, unmatched subjects are excluded and recorded in the sample retention flow.

---

## 3. Bland-Altman Limits of Agreement (LoA)

Bland-Altman analysis evaluates the agreement between two clinical measurement methods ($M_1$ and $M_2$) on the same continuous scale.

### Mean Bias and Difference
For paired measurements $(x_{i1}, x_{i2})$ on $n$ patients:
$$d_i = x_{i1} - x_{i2}, \quad \bar{d} = \frac{1}{n}\sum_{i=1}^n d_i, \quad s_d = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (d_i - \bar{d})^2}$$

### Limits of Agreement (95% LoA)
$$\text{Lower LoA} = \bar{d} - 1.96 \cdot s_d, \quad \text{Upper LoA} = \bar{d} + 1.96 \cdot s_d$$

### Confidence Intervals for LoA (Carkeet 2015)
The variance of the limits of agreement accounts for sampling error in both $\bar{d}$ and $s_d$:
$$\widehat{\text{Var}}(\text{LoA}) = s_d^2 \left(\frac{1}{n} + \frac{z_{1 - \alpha/2}^2}{2(n - 1)}\right)$$

$$\text{95% CI of LoA} = \text{LoA} \pm t_{n-1, 1 - \alpha/2} \cdot \sqrt{\widehat{\text{Var}}(\text{LoA})}$$

---

## 4. Intraclass Correlation Coefficient (ICC)

ICC assesses reliability and consistency among multiple clinical raters or repeated measures based on two-way analysis of variance (ANOVA) mean squares:
- **BMS**: Between-subjects Mean Square ($\text{df} = n - 1$)
- **WMS**: Within-subjects Mean Square ($\text{df} = n(k - 1)$)
- **JMS**: Between-judges/raters Mean Square ($\text{df} = k - 1$)
- **EMS**: Residual/Error Mean Square ($\text{df} = (n - 1)(k - 1)$)

### Shrout & Fleiss (1979) Taxonomy

| ICC Variant | Model Type | Random/Fixed | Definition & Formula | Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **ICC(1,1)** | One-way random | Raters nested | $\frac{\text{BMS} - \text{WMS}}{\text{BMS} + (k - 1)\text{WMS}}$ | Different raters assess each subject. |
| **ICC(2,1)** | Two-way random | Raters random | $\frac{\text{BMS} - \text{EMS}}{\text{BMS} + (k - 1)\text{EMS} + \frac{k}{n}(\text{JMS} - \text{EMS})}$ | Same raters assess all subjects; generalization to rater population intended (absolute agreement). |
| **ICC(3,1)** | Two-way mixed | Raters fixed | $\frac{\text{BMS} - \text{EMS}}{\text{BMS} + (k - 1)\text{EMS}}$ | Specific fixed panel of raters; consistency across raters is paramount. |
| **ICC(2,k)** | Two-way random (average) | Raters random | $\frac{\text{BMS} - \text{EMS}}{\text{BMS} + \frac{\text{JMS} - \text{EMS}}{n}}$ | Clinical score is the mean of $k$ independent raters. |

### Koo & Li (2016) Clinical Interpretation
- **$< 0.50$**: Poor reliability.
- **$0.50 - 0.75$**: Moderate reliability.
- **$0.75 - 0.90$**: Good reliability.
- **$> 0.90$**: Excellent clinical reliability.

---

## 5. Meta-Analytic Heterogeneity & Publication Bias

### Cochran's Q Test
$$Q = \sum_{i=1}^k w_i (\hat{\theta}_i - \hat{\theta}_{\text{FE}})^2$$
where $w_i = \frac{1}{\text{SE}_i^2}$ and $\hat{\theta}_{\text{FE}} = \frac{\sum w_i \hat{\theta}_i}{\sum w_i}$.

### Higgins & Thompson $I^2$
$$I^2 = \max\left(0, \frac{Q - (k - 1)}{Q}\right) \times 100\%$$
- **$I^2 < 25\%$**: Low heterogeneity (fixed-effects inverse variance model valid).
- **$25\% - 50\%$**: Moderate heterogeneity.
- **$\ge 50\%$**: Substantial heterogeneity (DerSimonian-Laird random effects mandatory).

### DerSimonian-Laird Random Effects ($\tau^2$)
$$\tau^2 = \max\left(0, \frac{Q - (k - 1)}{\sum w_i - \frac{\sum w_i^2}{\sum w_i}}\right)$$
Random effects study weights: $w_i^* = \frac{1}{\text{SE}_i^2 + \tau^2}$.

### Egger's Linear Regression Test for Funnel Asymmetry
$$\frac{\hat{\theta}_i}{\text{SE}_i} = \alpha + \beta \left(\frac{1}{\text{SE}_i}\right) + \epsilon_i$$
- **Null Hypothesis**: Funnel plot is symmetric ($\alpha = 0$).
- **$p < 0.10$**: Signals significant funnel plot asymmetry, suggesting potential publication bias, small-study effects, or methodological heterogeneity.

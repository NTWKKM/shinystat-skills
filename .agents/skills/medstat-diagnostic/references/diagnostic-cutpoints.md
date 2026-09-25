# Diagnostic Cutpoint Criteria, Wilson Intervals & DCA Math

Reference guide for diagnostic test evaluation, threshold selection, and decision-analytic statistics under STARD and TRIPOD guidelines.

## 1. Wilson Score Confidence Interval

The Wald interval $\hat{p} \pm 1.96 \sqrt{\hat{p}(1-\hat{p})/n}$ breaks down near 0 or 1, producing intervals exceeding $[0, 1]$ or zero width. The **Wilson score interval** is used across all sensitivity and specificity bounds:

$$\text{CI} = \frac{2n\hat{p} + z^2 \pm z \sqrt{z^2 + 4n\hat{p}(1 - \hat{p})}}{2(n + z^2)}$$

where $z = 1.96$ for a 95% confidence level.

---

## 2. Likelihood Ratios & Clinical Significance

Likelihood ratios indicate how much a diagnostic test result shifts the pre-test odds to post-test odds:

$$\text{Post-test Odds} = \text{Pre-test Odds} \times \text{LR}$$

| LR+ | LR- | Clinical Impact | Action |
| :--- | :--- | :--- | :--- |
| $> 10$ | $< 0.10$ | Often strong evidence for shifting disease probability | Supports rule-in (LR+) / rule-out (LR-); assess post-test probability against the clinical action threshold |
| $5 - 10$ | $0.10 - 0.20$ | Moderate shift in probability | Meaningful diagnostic contribution |
| $2 - 5$ | $0.20 - 0.50$ | Small shift in probability | Weak; requires confirmatory testing |
| $1 - 2$ | $0.50 - 1.00$ | Negligible shift | Diagnostically uninformative |

---

## 3. Threshold Selection Strategies

| Objective | Method | Formula | Clinical Scenario |
| :--- | :--- | :--- | :--- |
| **Balanced** | Youden's Index ($J$) | $\max(J) = \text{Sens} + \text{Spec} - 1$ | Weights sensitivity and specificity equally; does not imply equal clinical costs or equal numbers of false positives and false negatives (a clinically utility-optimal cutoff may differ from the Youden-selected cutoff and depends on disease prevalence and relative error costs). |
| **Rule-Out** | Fixed High Sensitivity (illustrative) | $\text{Threshold at } \text{Sens} \ge 95\%$ | Illustrative target; clinical use requires validated, indication- and assay-specific pathways (e.g. clinical probability with age-adjusted D-dimer thresholds for PE; assay-specific serial results for high-sensitivity troponin). |
| **Rule-In** | Fixed High Specificity (illustrative) | $\text{Threshold at } \text{Spec} \ge 95\%$ | Illustrative target for minimizing false positives; high-risk clinical actions require assessing post-test probability against the relevant clinical action threshold based on pre-test probability and likelihood ratios. |
| **Geometric** | Closest to Top-Left | $\min \sqrt{(1 - \text{Sens})^2 + (1 - \text{Spec})^2}$ | Alternative Euclidean distance to ideal $(0, 1)$ ROC coordinate. |

---

## 4. DeLong Non-Parametric ROC AUC Variance

For a sample with $m$ diseased and $n$ non-diseased subjects, define placement values:

$$V_{10}(X_i) = \frac{1}{n} \sum_{j=1}^n \psi(X_i, Y_j), \quad V_{01}(Y_j) = \frac{1}{m} \sum_{i=1}^m \psi(X_i, Y_j)$$

where $\psi(X, Y) = 1$ if $X > Y$, $0.5$ if $X = Y$, and $0$ if $X < Y$.

$$\widehat{\text{AUC}} = \frac{1}{m} \sum_{i=1}^m V_{10}(X_i)$$

$$\widehat{\text{Var}}(\widehat{\text{AUC}}) = \frac{s_{10}^2}{m} + \frac{s_{01}^2}{n}$$

where $s_{10}^2$ and $s_{01}^2$ are the sample variances of the placements $V_{10}$ and $V_{01}$.

---

## 5. Vickers Decision Curve Analysis (DCA)

A model's clinical usefulness is measured by **Net Benefit (NB)** relative to the clinical decision threshold probability $p_t$:

$$\text{NB}_{\text{model}} = \frac{\text{True Positives}}{N} - \frac{\text{False Positives}}{N} \cdot \left(\frac{p_t}{1 - p_t}\right)$$

$$\text{NB}_{\text{all}} = \text{Prevalence} - (1 - \text{Prevalence}) \cdot \left(\frac{p_t}{1 - p_t}\right)$$

$$\text{NB}_{\text{none}} = 0$$

- The exchange rate $w = \frac{p_t}{1 - p_t}$ reflects the clinical harm of a false positive relative to a false negative.
- A model adds clinical value only across the threshold range $[p_{\text{min}}, p_{\text{max}}]$ where $\text{NB}_{\text{model}} > \max(\text{NB}_{\text{all}}, \text{NB}_{\text{none}})$.

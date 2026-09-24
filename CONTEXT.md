# CONTEXT.md — Domain Vocabulary & Naming Diary

Domain vocabulary, mathematical definitions, entity models, and clinical conventions for `medstat-core`.

## 1. Domain Terminology

| Term | Symbol / Code | Definition & Clinical Context |
| :--- | :--- | :--- |
| **Complete-Case** | `complete-case` | Analysis restricted exclusively to records with zero missing values. Valid strictly under Missing Completely at Random (MCAR). |
| **MICE** | `mice` | Multiple Imputation by Chained Equations. Iterative imputation using series of regression models under Missing at Random (MAR). |
| **Sample Flow Tracker** | `SampleFlowTracker` | CONSORT/STROBE audit tracker recording participant transitions: $N_{\text{initial}} \to N_{\text{excluded}} \to N_{\text{analyzed}}$. |
| **Standardized Mean Difference** | `SMD` | Difference in means or proportions divided by pooled standard deviation. Evaluates baseline balance; $\text{SMD} < 0.10$ signifies negligible imbalance. |
| **Firth Penalization** | `firth` | Penalization of the log-likelihood by Jeffrey's invariant prior $\frac{1}{2}\ln\lvert I(\beta)\rvert$, removing first-order bias and resolving monotone separation in small/sparse cohorts. |
| **Proportional Hazards** | `cox_ph` | Cox survival model assuming constant hazard ratios over time. Checked via Schoenfeld residual correlation ($p > 0.05$). |
| **Restricted Cubic Splines** | `rcs` | Piecewise cubic polynomials constrained to be linear past outer boundary knots, modeling non-linear biomarker curves. |
| **E-Value** | `e_value` | The minimum strength of association on the risk ratio scale that an unmeasured confounder must have with both exposure and outcome to fully explain away an observed association. |
| **Wilson Score Interval** | `wilson` | Asymmetric confidence interval for binomial proportions providing nominal coverage even near boundary proportions ($p \approx 0$ or $1$). |
| **DeLong Test** | `delong` | Non-parametric placement-value covariance estimation for ROC AUC standard errors and paired comparative testing. |
| **Decision Curve Analysis** | `dca` | Vickers decision-analytic metric evaluating clinical net benefit across threshold probabilities ($p_t$) against "Treat All" and "Treat None". |
| **Propensity Score Matching** | `psm` | Logistic regression modeling treatment probability followed by caliper-bounded nearest neighbor pairing. |
| **Limits of Agreement** | `bland_altman` | Bland-Altman interval $\bar{d} \pm 1.96 \cdot s_d$ containing 95% of paired measurement differences, accompanied by Carkeet (2015) exact CIs. |
| **Intraclass Correlation** | `icc` | Variance partition ratio quantifying reliability and agreement among clinical observers based on two-way ANOVA decomposition. |
| **DerSimonian-Laird** | `dl` | Non-iterative method of moments estimator for between-study variance ($\tau^2$) in random-effects meta-analysis. |
| **Egger's Test** | `egger` | Linear regression of standardized effect against precision assessing funnel plot asymmetry and publication bias. |

---

## 2. Entity Models & Data Structures

- **`SampleFlowTracker`**:
  - `initial_count: int`
  - `exclusions: list[tuple[str, int]]` (reason, count)
  - `final_count: int`
  - `to_dict() -> dict[str, Any]`
- **`Estimate`**:
  - `term: str` (covariate name)
  - `label: str` (display name)
  - `point_estimate: float`
  - `ci_lower: float`
  - `ci_upper: float`
  - `p_value: float`
  - `metric: str` ("OR", "HR", "Beta", "MD")
- **`EstimateTable`**:
  - `estimates: list[Estimate]`
  - `meta: ModelMeta` (model type, outcome, sample flow counts)
- **`DiagnosticResult`**:
  - `sensitivity: float`, `sensitivity_ci: tuple[float, float]`
  - `specificity: float`, `specificity_ci: tuple[float, float]`
  - `ppv: float`, `npv: float`
  - `lr_pos: float`, `lr_neg: float`, `dor: float`

---

## 3. Formatting & Precision Invariants

- **Intermediate Calculations**: Always retain full float precision (`float64`); never round intermediate numbers.
- **Odds / Hazard Ratios**: Display with 2 decimal places: `1.45 (95% CI, 1.12–1.88)`.
- **Percentages**: Display with 1 decimal place: `24.3%`.
- **P-Values**:
  - NEJM: $P = 0.04$, $P = 0.003$, $P < 0.001$ (with leading zero).
  - JAMA: $P = .04$, $P = .008$, $P < .001$ (no leading zero).
- **Units**: Must be explicitly rendered on continuous clinical variables (`mg/dL`, `mmHg`, `mL/min/1.73m²`).
- **Binary & Event Outcome Encoding**: Binary outcomes and survival event indicators must be strictly encoded as numeric `0` and `1` (`1 = Event`, `0 = Non-event`) across all CLI commands, YAML SAP specifications, and data cleaning pipelines. Text outcomes (`'Dead'`/`'Alive'`, `'Yes'`/`'No'`) are rejected to eliminate clinical event inversion.

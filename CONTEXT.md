# CONTEXT.md — Domain Vocabulary & Naming Diary

Domain vocabulary, mathematical definitions, entity models, and clinical conventions for `medstat-core`.

## 1. Domain Terminology

| Term | Symbol / Code | Definition & Clinical Context |
| :--- | :--- | :--- |
| **Complete-Case** | `complete-case` | Analysis restricted exclusively to records with zero missing values. Validity is conditional on assessing the missingness mechanism, target estimand, and sensitivity analysis; may also be appropriate when missingness is trivial and clinically uninformative, rather than stating validity strictly under MCAR. |
| **MICE** | `mice` | Multiple Imputation by Chained Equations. Iterative imputation using series of regression models under Missing at Random (MAR). |
| **Sample Flow Tracker** | `SampleFlowTracker` | CONSORT/STROBE audit tracker recording participant transitions: $N_{\text{initial}} \to N_{\text{excluded}} \to N_{\text{analyzed}}$. |
| **Standardized Mean Difference** | `SMD` | Difference in means or proportions divided by pooled standard deviation. Evaluates baseline balance; $\text{SMD} < 0.10$ signifies negligible imbalance. |
| **Firth Penalization** | `firth` | Penalization of the log-likelihood by Jeffrey's invariant prior $\frac{1}{2}\ln\lvert I(\beta)\rvert$, removing first-order bias and resolving monotone separation in small/sparse cohorts. |
| **Proportional Hazards** | `cox_ph` | Cox survival model assuming constant hazard ratios over time. Assessed in standard Cox CLI runs via Schoenfeld residual correlation test (`--schoenfeld`; Firth Cox runs do not compute this test); $p > 0.05$ indicates insufficient evidence against PH but does not establish proportionality — review residual plots and model context. |
| **Restricted Cubic Splines** | `rcs` | Piecewise cubic polynomials constrained to be linear past outer boundary knots, modeling non-linear biomarker curves. |
| **E-Value** | `e_value` | The minimum strength of association on the risk ratio scale that an unmeasured confounder must have with both exposure and outcome to fully explain away an observed association. |
| **Wilson Score Interval** | `wilson` | Asymmetric confidence interval for binomial proportions providing nominal coverage even near boundary proportions ($p \approx 0$ or $1$). |
| **DeLong Test** | `delong` | Non-parametric placement-value covariance estimation for ROC AUC standard errors and paired comparative testing. |
| **Decision Curve Analysis** | `dca` | Vickers decision-analytic metric evaluating clinical net benefit across threshold probabilities ($p_t$) against "Treat All" and "Treat None". |
| **Propensity Score Matching** | `psm` | Logistic regression modeling treatment probability followed by caliper-bounded nearest neighbor pairing. |
| **Limits of Agreement** | `bland_altman` | Bland-Altman interval $\bar{d} \pm 1.96 \cdot s_d$ containing 95% of paired measurement differences, accompanied by Bland–Altman (1999) large-sample approximate CIs. |
| **Intraclass Correlation** | `icc` | Variance partition ratio quantifying reliability and agreement among clinical observers based on two-way ANOVA decomposition. |
| **DerSimonian-Laird** | `dl` | Non-iterative method of moments estimator for between-study variance ($\tau^2$) in random-effects meta-analysis. |
| **Egger's Test** | `egger` | Linear regression of standardized effect against precision assessing funnel plot asymmetry and publication bias. |
| **Brier Score** | `brier_score` | Mean squared prediction error for probabilistic forecasts ($\text{Brier} = \frac{1}{N}\sum(y_i - \hat{p}_i)^2$); lower is better. Scaled Brier adjusts for baseline prevalence. |
| **Calibration Slope** | `calibration_slope` | Logistic recalibration coefficient: ideal slope = 1. Slope $\lt 1$ indicates overfitting; slope $\gt 1$ indicates underfitting. Paired with calibration intercept (ideal = 0). |
| **Integrated Calibration Index** | `ici` | Austin & Steyerberg (2019) mean absolute difference between LOWESS-smoothed observed and predicted probabilities, with E50, E90, and Emax quantiles. |
| **Hosmer-Lemeshow** | `hosmer_lemeshow` | Goodness-of-fit $\chi^2$ test across decile risk groups assessing logistic model calibration. Low power limits its use as a sole calibration indicator. |
| **Proportional Odds** | `proportional_odds` | Cumulative link model for ordinal outcomes assuming equal covariate effects across all threshold cuts: $\text{logit}(P(Y \ge j)) = \alpha_j + \beta^T X$. |
| **Brant Test** | `brant_test` | Wald-type hypothesis test (Brant 1990) assessing parallel slopes across ordinal cutpoints ($H_0: \beta_1 = \dots = \beta_{K-1}$). Omnibus and per-variable statistics. |
| **Cumulative Odds Ratio** | `cumulative_or` | Exponentiated slope $\exp(\beta)$ representing the odds ratio of being in a higher versus lower category per unit increase in predictor. |
| **Design Effect** | `design_effect` | Variance inflation factor from clustering: $\text{DEFF} = 1 + (\bar{m} - 1)\text{ICC}_{\text{cluster}}$. Quantifies effective sample size $N_{\text{eff}} = N / \text{DEFF}$. |
| **Cluster ICC** | `icc_cluster` | Intraclass correlation coefficient quantifying the proportion of total variance attributable to between-cluster differences in multi-center cohorts. |
| **Generalized Estimating Equations** | `gee` | Semi-parametric population-averaged regression accounting for within-cluster correlation using empirical robust (sandwich) standard errors. |
| **Random-Intercept Model** | `random_intercept` | Linear mixed-effects model decomposing outcome variance into fixed covariate effects and cluster-specific random shifts ($u_i \sim \mathcal{N}(0, \sigma_u^2)$). |
| **Shinystat Skill** | `shinystat` | Unified autonomous biostatistical intelligence skill that triangulates data reality with research questions, generates adaptive Python scripts, and enforces mandatory Grilling on ambiguity. |
| **Statistical Analysis Plan** | `SAP` | Formal specification of primary estimand, data cleaning/retention strategy, planned models, and reporting standards before execution. |
| **Gold Standard Validation** | `validate_gold_standard` | Pre-flight validation that drops missing values, rejects observed values outside $\{0, 1\}$ (and boolean types), and requires both classes only when `require_both_classes=True`. |
| **Primary Effect Extraction** | `extract_primary_effect` | Helper accepting an estimates map and returning a finite point estimate, or `np.nan` when no supported or unambiguous estimate is found. |
| **Zero-Variance SMD** | `smd_zero_variance` | Boundary condition in covariate balance: when pooled SD is 0 and group means differ, SMD is mathematically undefined and returns `np.nan` (rather than masking extreme imbalance as `0.0`), serialized as JSON `null`. |
| **Distinct Study Count** | `distinct_studies` | Independent sample size threshold ($k_{\text{distinct}} \ge 10$) and uniqueness verification enforced before running Egger's linear regression test to prevent spurious validity from multi-effect studies. |
| **Positional Boolean Masking** | `valid_mask` | Decoupling numpy design matrix slicing from pandas DataFrame index state via `.notna().to_numpy()` to prevent index misalignment across custom or non-standard indices. |
| **Autonomous Authority** | `autonomous_authority` | Agent mandate to independently execute sound biostatistical mechanics (normality testing, model upgrading to Firth/splines/GEE, missing data handling, 95% CIs) without halting to ask trivial user permission. |
| **Output Alignment Gate** | `output_alignment_gate` | Proactive inquiry protocol triggered when the user prompt omits a report format, prompting the user with options and recommending the Dual / Toggle View HTML report. |
| **Dual / Toggle View** | `dual_toggle_view` | Publication architecture placing interactive Plotly.js charts as the primary active view, with a tab or toggle button to switch to "Publication 300 DPI Figure (Static)" for journal submission and downloads. |
| **Publication 300 DPI Figure** | `pub_static_figure` | Journal-compliant 300 DPI static figure rendered via Matplotlib with colorblind-safe palettes and Thai typography fallback chain (`Sarabun`, `Thonburi`, `Sukhumvit Set`). |
| **Inverse Probability Weighting** | `iptw` | Propensity score weighting reweighting observational cohorts to create pseudo-populations where treatment assignment is independent of measured baseline covariates. |
| **Propensity Estimands** | `ate_att_atc_ato` | Target causal estimands in IPTW: ATE (whole eligible population), ATT (treated population), ATC (control population), and ATO (Overlap Weights, emphasizing clinical equipoise). |
| **Effective Sample Size** | `kish_neff` | Kish's effective sample size $N_{\text{eff}} = \frac{(\sum w_i)^2}{\sum w_i^2}$ evaluating statistical precision loss and variance inflation induced by unequal weighting. |
| **Cumulative Incidence Function** | `cif_aalen_johansen` | Non-parametric event probability estimator under competing risks ($\text{CIF}_k(t) = \int_0^t \hat{S}(u^-) d\hat{\Lambda}_k(u)$), avoiding the systematic upward bias of $1 - \text{Kaplan-Meier}$. |
| **Cause-Specific Hazard Ratio** | `cshr` | Cox model treating competing events as right-censored at failure time, estimating the instantaneous relative event rate among subjects currently event-free (etiological inference). |
| **Fine-Gray Hazard Ratio** | `fine_gray_shr` | Subdistribution hazard model maintaining subjects experiencing competing events in the risk set with decaying IPCW weights, estimating overall cumulative incidence (prognostic inference). |
| **Probability Conservation** | `cif_conservation` | Mandatory audit ensuring the sum of all cause-specific CIFs plus overall event-free survival probability equals unity ($\sum_k \text{CIF}_k(t) + \hat{S}(t) \approx 1.0$) across all follow-up horizons. |
| **Net Reclassification Improvement** | `nri` | Pencina et al. (2008, 2011) reclassification metric measuring net upward movement in events and net downward movement in non-events; supports both Category-Free (continuous) and Categorical NRI with $K \times K$ contingency matrices. |
| **Integrated Discrimination Improvement** | `idi` | Difference in discrimination slopes between updated and baseline risk models ($\text{IDI} = (\bar{p}_{\text{new}, e} - \bar{p}_{\text{old}, e}) - (\bar{p}_{\text{new}, ne} - \bar{p}_{\text{old}, ne})$). |
| **Time-Dependent ROC** | `time_dependent_roc` | Cumulative/Dynamic IPCW ROC estimator (Uno et al., 2007; Heagerty et al., 2000) evaluating longitudinal discrimination trajectories ($\text{AUROC}(t)$) across milestone horizons under right-censoring. |
| **TRIPOD+AI Internal Validation** | `bootstrap_internal_validation` | Harrell's bootstrap optimism correction protocol recommended by TRIPOD+AI (2024). Fits models in $B$ bootstrap resamples, tests on resamples and original cohort, and calculates optimism $\mathcal{O} = \frac{1}{B}\sum(M_{\text{boot}} - M_{\text{test}})$ to derive optimism-corrected performance ($M_{\text{apparent}} - \mathcal{O}$). |
| **Uniform Shrinkage Factor** | `shrinkage_factor` | Post-estimation shrinkage multiplier $\hat{s}$ (Copas 1983; van Houwelingen & le Cessie 1990) derived from average bootstrap calibration slope to prevent extreme overfitting in logistic models: $\hat{\beta}_{\text{shrunk}} = \hat{s} \cdot \hat{\beta}_{\text{orig}}$. |
| **Model Recalibration** | `recalibrate_prediction_model` | Model updating methodology (Steyerberg 2019) for validating prediction rules on new populations: Intercept-only updating (calibration-in-the-large, fixing slope $\beta=1.0$) vs Logistic recalibration (estimating updated $\alpha$ and $\beta$). |
| **Riley Sample Size Criteria** | `riley_sample_size` | Riley et al. (BMJ 2020; 368:m441) 4-criteria formulation for clinical prediction models: Criterion 1 (small margin in intercept), Criterion 2 (small MAPE $\le 0.05$), Criterion 3 (global shrinkage $\ge 0.90$), and Criterion 4 (small difference in apparent vs adjusted $R^2$). |
| **Obuchowski AUROC Power** | `diagnostic_auroc_sample_size` | Non-parametric sample size formulation for diagnostic ROC curves per Obuchowski (1998, 2004) and Hanley & McNeil (1982), accounting for case prevalence $\phi$ and control-to-case allocation ratio $R$. |

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
- **`IPTWResult`**:
  - `df_weighted: pd.DataFrame` (cohort augmented with `propensity_score` and `weight`)
  - `weights: np.ndarray`, `propensity_scores: np.ndarray`
  - `effective_sample_size: float`, `effective_sample_size_treated: float`, `effective_sample_size_control: float`
  - `diagnostics: dict[str, Any]` (estimand, stabilized, truncated, extreme weight flags, positivity violations)
  - Supports tuple unpacking: `df_weighted, diagnostics = iptw_result`
- **`CompetingRisksCIFResult`**:
  - `cif_table: pd.DataFrame` (time, CIF, Greenwood lower/upper CIs, standard error per event type)
  - `event_types: list[int]` (e.g. `[1, 2]`)
  - `horizons: dict[float, dict[int, dict[str, float]]]` (CIF estimates and 95% CIs at requested landmark times)
  - `overestimation_vs_km: dict[int, dict[float, float]]` (percentage point difference between Naive $1 - \text{KM}$ and Aalen-Johansen CIF)
  - `conservation_audit: pd.DataFrame` (verifies $\sum \text{CIF}_k + \hat{S} \approx 1.0$)
- **`DiagnosticResult`**:
  - `sensitivity: float`, `sensitivity_ci: tuple[float, float]`
  - `specificity: float`, `specificity_ci: tuple[float, float]`
  - `ppv: float`, `npv: float`
  - `lr_pos: float`, `lr_neg: float`, `dor: float`
- **`FigureResult`**:
  - `png_path: str` (path to 300 DPI exported image)
  - `alt_text: str` (accessibility text for web/markdown)
  - `caption: str` (publication caption with clinical/statistical notes)
  - `source_df: pd.DataFrame` (exact plotted data coordinates)
  - `csv_path: str` (audit trail data export)
- **`ReportDocument` (Report IR)**:
  - `title: str`, `authors: list[str]`, `date: str`, `institution: str`
  - `blocks: list[Block]` (`HeadingBlock`, `ParagraphBlock`, `TableBlock`, `FigureBlock`, `CalloutBlock`)
  - `results_dict: dict[str, Any]` (immutable source of truth for narrative numbers)
- **`IntegrityReport`**:
  - `passed: bool`, `untraced_numbers: list[float]`, `phi_violations: list[str]`
  - `warnings: list[str]`, `has_methods: bool`, `has_retention_flow: bool`, `has_causal_caveat: bool`

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
- **Figure Standards**: 300 DPI minimum resolution, colorblind-safe palettes (Okabe-Ito / Tol), Thai typography fallbacks (`Sarabun`, `Thonburi`, `Sukhumvit Set`), and 1 figure per slide in presentation outputs.
- **Numerical Traceability**: All narrative numbers are audited against `results_dict` within $\pm 0.02$ absolute or 1% relative error, including percentage-scale matching ($\pm 0.05$ / $\pm 0.005$). Common statistical/structural constants (single-digit indices 0–10, $\alpha$-levels, 95% / 99% CI levels, standard 100% scale, calendar years) are bypassed. An untraced rate of up to 15% (minimum 1 untraced value) is tolerated for non-statistical headings and structural annotations.
- **Zero-PHI Compliance**: Automated regex enforcement screens text, table cells, headers, and figure metadata for Hospital Numbers (`HN`), 13-digit Thai National IDs, phone numbers, dates of birth, and honorific-prefixed names (`Mr.`, `Mrs.`, `Ms.`, `Dr.`, `นาย`, `นาง`, `นพ.`, `พญ.`). Because arbitrary patient names without honorifics cannot be reliably detected by regex, automated passing does not prove absolute zero-PHI; manual de-identification review remains mandatory before external release.

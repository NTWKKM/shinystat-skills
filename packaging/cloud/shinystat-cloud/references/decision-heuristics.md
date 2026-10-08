# Biostatistical Decision Heuristics & Reference Manual (Cloud Sandbox Edition)

Reference guide for the `shinystat-cloud` autonomous agent in cloud code-execution environments (e.g. Claude Web Analysis Tool). Consult this manual during Pillar 2 (Research & Estimand Triangulation) to map clinical research questions to optimal biostatistical models and standalone recipes in [python-recipes.md](python-recipes.md).

---

## 1. Clinical Study Archetype to Statistical Model Mapping

| Study Archetype | Target Estimand / Goal | Primary Model / Method | Key Assumption / Diagnostics | Recommended Recipe / Library |
| :--- | :--- | :--- | :--- | :--- |
| **Type 1: Baseline Cohort** | Cohort characteristics | Table 1 (Mean/SD, Median/IQR, Count/%, SMD) | Normality inspection; Fisher exact fallback for small cells | [Recipe 1: Table 1 & SMD](python-recipes.md#1-table-1-baseline-characteristics--standardized-mean-difference-smd) |
| **Type 2: Binary Outcome** | Odds Ratio (OR), Risk factors | Multivariable Binary Logistic Regression | Linearity of log-odds; Crude & Adjusted OR table | [Recipe 10: Logistic Regression Table](python-recipes.md#10-multivariable-binary-logistic-regression-table) |
| **Type 2-Sparse: Separation** | Unbiased OR in rare events | Firth Penalized Logistic Regression | Profile likelihood CIs; resolves monotone separation | [Recipe 7: Firth Logistic Regression](python-recipes.md#7-firths-penalized-likelihood-logistic-regression) |
| **Type 2b: Ordinal Scale** | Cumulative Odds Ratio | Cumulative Logit Proportional Odds | Brant test for parallel lines; OrderedModel fallback | `statsmodels.miscmodels.ordinal_model.OrderedModel` |
| **Type 3: Survival / Time-to-Event** | Hazard Ratio (HR), Median survival | Kaplan-Meier, Log-Rank & Cox Proportional Hazards | Proportional hazards test (Schoenfeld residuals); L2 shrinkage | [Recipe 11: Survival Analysis Suite](python-recipes.md#11-survival-analysis-kaplan-meier-log-rank--cox-proportional-hazards) |
| **Type 4: Diagnostic Test Accuracy** | Sens, Spec, PPV, NPV, AUC, LRs | 2x2 Contingency Matrix & Empirical ROC | Wilson score CIs; Haldane correction; Single/Paired DeLong | [Recipe 2: 2x2 Diagnostics](python-recipes.md#2-2x2-diagnostic-accuracy-with-wilson-score--log-scale-95-cis) & [Recipe 3: ROC & DeLong](python-recipes.md#3-empirical-roc-optimal-cutoff--single--paired-delong-95-cis) |
| **Type 4b: Clinical Utility & Calib** | Net Benefit, Model Calibration | Decision Curve Analysis (DCA) & Recalibration | Brier score; Calibration slope & intercept; Austin-Steyerberg ICI | [Recipe 4: Calibration & DCA](python-recipes.md#4-model-calibration--vickers-decision-curve-analysis-dca) |
| **Type 5: Observational Causal** | Average Treatment Effect in Treated (ATT) | Propensity Score Matching (PSM, 1:1, Caliper 0.2 SD) | Austin (2009) balance (SMD < 0.10); Pair ID tracking; E-value | [Recipe 6: PSM & Balance](python-recipes.md#6-propensity-score-matching-psm-with-pair-ids--covariate-balance) & [Recipe 9: E-Value](python-recipes.md#9-vanderweele--ding-2017-sensitivity-e-value) |
| **Type 6: Device / Rater Agreement** | Limits of Agreement, Reliability | Bland-Altman Difference Analysis & ICC (1–3k) | Bland-Altman 1999 large-sample CIs; Shrout-Fleiss exact F CIs | [Recipe 5: Bland-Altman & ICC](python-recipes.md#5-observer-agreement-bland-altman--shrout-fleiss-1979-icc) |
| **Type 7: Systematic Meta-Analysis** | Pooled Effect (RR/OR/MD) | Inverse-Variance Fixed-Effects or DerSimonian-Laird | Condition on clinical variation; small $k < 5$ caution | Pure SciPy meta-analysis or `statsmodels` |
| **Type 8: Clustered / Multi-Center** | Population-averaged effect | Generalized Estimating Equations (GEE) | Cluster Design Effect ($\text{DEFF}$); Robust sandwich SEs | `statsmodels.genmod.generalized_estimating_equations.GEE` |

---

## 2. Decision Trees for Common Statistical Challenges

### A. Events Per Variable (EPV), Sparse Data & Separation
```
Calculate diagnostic EPV:
- Binary Logistic: EPV = min(Events, Non-Events) / P_parameters (counting multi-level factors as k-1)
- Cox Proportional Hazards: EPV = Observed Failures / P_parameters

├── EPV ≥ 10 & No Separation ──▶ Standard Multivariable Logistic (Recipe 10) / Cox PH (Recipe 11)
└── Sparse Data, Separation, or EPV < 10
    ├── Binary Logistic:
    │   ├── Strategy 1 (Etiological Effect): Firth penalized logistic regression with profile likelihood CIs (Recipe 7)
    │   ├── Strategy 2 (Prediction Model): L2 Ridge penalization / shrinkage to prevent overfitting (Riley et al., 2019)
    │   └── Strategy 3: Domain-guided clinical composite score or univariable reporting with sparse-data disclaimer
    └── Cox Proportional Hazards:
        ├── Strategy 1: L2-penalized Cox regression via lifelines (`CoxPHFitter(penalizer=0.1)`)
        ├── Strategy 2: Pre-specified clinical risk index / score reduction
        └── Strategy 3: Univariable survival analysis (Kaplan-Meier / univariable Cox) with sparse-data disclaimer
```

*Note on Modern EPV Evidence:* As demonstrated by Vittinghoff & McCulloch (2007) and van Smeden et al. (2016), EPV < 10 is an exploratory diagnostic indicator rather than an unconditional hard stop. Firth penalization resolves monotone separation and reduces first-order finite-sample parameter estimation bias (Heinze & Schemper, 2002); however, for prediction modeling, shrinkage/penalization is required to control overfitting.

### B. Missing Data Strategy
```
Audit Missingness Pattern & Mechanism (Recipe 8: Little's MCAR Test; note H0: MCAR; failing to reject does not prove MCAR)

├── Per-variable missingness < 5% in non-primary covariates ──▶ Complete-case analysis with documented justification
├── Missing 5%–40% under Missing at Random (MAR) ──▶ Multiple Imputation by Chained Equations (Recipe 12: MICE; continuous targets, M ≥ 5 datasets + Rubin's rules for inference; missing categorical requires variable-type-specific imputation)
├── Primary Outcome Missing ──▶ Never impute outcome; preserve true sample retention flow (N_initial ➔ N_analyzed)
└── Missingness Informative / MNAR ──▶ Halt for clinical domain review or sensitivity bounds
```

### C. Diagnostic Test Directionality & Cutpoint Selection
```
Identify Clinical Biomarker Mechanism

├── High biomarker indicates disease (e.g. Troponin, Lactate) ──▶ Direction: 'high' (Score ≥ Cutoff)
├── Low biomarker indicates disease (e.g. eGFR, Platelets) ──▶ Direction: 'low' (Score ≤ Cutoff; negate score for ROC)
└── Prespecified vs Exploratory Cutpoint
    ├── Prespecified clinical threshold ──▶ Report nominal 2x2 metrics (Recipe 2)
    └── Data-driven cutpoint (Youden's J) ──▶ Report with anti-p-hacking disclosure (Recipe 3)
```

---

## 3. Reporting & Publication Standards

- **NEJM Style**: 3 horizontal rules (top border 2px, header border 1px, bottom border 2px), zero vertical dividers, $P$-values formatted as $P = 0.04$ or $P < 0.001$. Render via [Recipe 13: Publication Table Formatter](python-recipes.md#13-publication-grade-html-table-renderer-nejm--jama-standards).
- **JAMA Style**: Clean minimal dividers, $P$-values formatted without leading zero: $P = .04$ or $P < .001$.
- **Precision Invariants**: Unrounded intermediate calculations (`float64`); OR/HR reported to 2 decimal places with 95% CIs; percentages reported to 1 decimal place; explicit units mandatory (`mg/dL`, `mL/min/1.73m²`).
- **Reporting Checklists**: Consult EQUATOR standards: STROBE (observational cohorts), CONSORT (trials), TRIPOD (prediction models), STARD (diagnostic accuracy), or PRISMA (systematic reviews).

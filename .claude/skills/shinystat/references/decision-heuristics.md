# Biostatistical Decision Heuristics & Reference Manual

Reference guide for the `shinystat` autonomous agent. Consult this manual during Pillar 2 (Research & Estimand Triangulation) to map clinical research questions to optimal biostatistical models and `medstat` core modules.

---

## 1. Clinical Study Archetype to Statistical Model Mapping

| Study Archetype | Target Estimand / Goal | Primary Model / Method | Key Assumption / Diagnostics | Recommended Module |
| :--- | :--- | :--- | :--- | :--- |
| **Type 1: Baseline Cohort** | Cohort characteristics | Table 1 (Mean/SD, Median/IQR, Count/%) | Normality inspection (Shapiro-Wilk / Skewness) | `medstat.data.clean`, `medstat.reporting.tables` |
| **Type 2: Binary Outcome** | Odds Ratio (OR), Risk factors | Multivariable Logistic Regression | $\text{EPV} \ge 10$, Linearity of continuous log-odds | `medstat.models.glm` |
| **Type 2-Sparse: Separation** | Unbiased OR in rare events | Firth Penalized Logistic Regression | Profile likelihood CIs; resolves monotone separation | `medstat.models.firth` |
| **Type 2b: Ordinal Scale** | Cumulative Odds Ratio | Cumulative Logit Proportional Odds | Brant's Wald test for parallel lines assumption | `medstat.models.ordinal` |
| **Type 3: Survival / Time-to-Event** | Hazard Ratio (HR) | Cox Proportional Hazards | Proportional hazards test (Grambsch-Therneau / Schoenfeld) | `medstat.models.survival` |
| **Type 4: Diagnostic Test Accuracy** | Sensitivity, Specificity, PPV, NPV, AUC | 2x2 Contingency Matrix & Empirical ROC | Wilson score 95% CIs; DeLong test for paired AUC; Directionality | `medstat.diagnostic` |
| **Type 4b: Clinical Utility & Calib** | Net Benefit, Model Calibration | Decision Curve Analysis (DCA) & Recalibration | Brier score, Calibration slope & intercept, Austin-Steyerberg ICI | `medstat.diagnostic.dca`, `medstat.diagnostic.calibration` |
| **Type 5: Observational Causal** | Average Treatment Effect in Treated (ATT) | Propensity Score Matching (PSM, 1:1, Caliper 0.2 SD) | Austin (2009) Covariate Balance ($|\text{SMD}| < 0.10$); VanderWeele E-value | `medstat.causal` |
| **Type 6: Device / Rater Agreement** | Limits of Agreement, Reliability | Bland-Altman Difference Analysis & ICC | Bland-Altman (1999) large-sample CIs; Shrout-Fleiss two-way ANOVA | `medstat.agreement` |
| **Type 7: Systematic Meta-Analysis** | Pooled Effect (RR/OR/MD) | DerSimonian-Laird Random-Effects | Cochran's $Q$, Higgins $I^2$; Egger's test if $k_{\text{distinct}} \ge 10$ | `medstat.meta` |
| **Type 8: Clustered / Multi-Center** | Population-averaged or Cluster Effect | GEE with robust sandwich SE or MixedLM | Cluster Design Effect ($\text{DEFF} = 1 + (\bar{m}-1)\text{ICC}_{\text{cluster}}$) | `medstat.models.multilevel` |

---

## 2. Decision Trees for Common Statistical Challenges

### A. Events Per Variable (EPV) & Sparse Data
```
Calculate EPV = min(Events, Non-Events) / P_covariates

├── EPV ≥ 10 ──▶ Proceed with standard GLM Logistic / Cox PH
└── EPV < 10 or Separation detected
    ├── Strategy 1 (Preferred): Firth penalized likelihood (`medstat.models.firth`)
    ├── Strategy 2: Pre-specified variable reduction or domain-guided composite score
    └── Strategy 3: Report univariable associations with explicit sparse-data disclaimer
```

### B. Missing Data Strategy
```
Audit Missingness Pattern & Mechanism

├── Total Missing < 5% and Clinically Uninformative ──▶ Complete-case analysis with documented justification
├── Missing 5%–40% under Missing at Random (MAR) ──▶ Multiple Imputation by Chained Equations (MICE)
├── Primary Outcome Missing ──▶ Never impute outcome; record as sample flow exclusion
└── Missingness Informative / Missing Not at Random (MNAR) ──▶ Halt at Grilling Gate for clinical review
```

### C. Diagnostic Test Directionality & Cutpoint Selection
```
Identify Clinical Biomarker Mechanism

├── High biomarker indicates disease (e.g. Troponin, Lactate) ──▶ Direction: 'high' (Score ≥ Cutoff)
├── Low biomarker indicates disease (e.g. eGFR, Platelets) ──▶ Direction: 'low' (Score ≤ Cutoff; negate score for ROC)
└── Prespecified vs Exploratory Cutpoint
    ├── Prespecified clinical threshold ──▶ Report nominal 2x2 metrics
    └── Data-driven cutpoint (Youden's J) ──▶ Report with anti-p-hacking disclosure and sensitivity analysis
```

---

## 3. Reporting & Table Styling Standards

- **NEJM Style**: 3 horizontal rules (top, header bottom, table bottom), no vertical dividers, $P$-values formatted as $P = 0.04$ or $P < 0.001$.
- **JAMA Style**: Clean minimal dividers, $P$-values formatted without leading zero: $P = .04$ or $P < .001$.
- **APA 7th Style**: Standard psychological and behavioral format with exact degrees of freedom.
- **Reporting Checklists**: Consult STROBE (observational cohorts), CONSORT (trials), TRIPOD (prediction models), STARD (diagnostic accuracy), or PRISMA (systematic reviews).

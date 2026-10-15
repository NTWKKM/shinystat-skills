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
| **Type 3b: Competing Risks Survival** | Subdistribution Hazard (SHR) & Cause-Specific HR (csHR) | Aalen-Johansen CIF & Fine-Gray Model (IPCW) | Competing risk probability conservation; Naive KM overestimation audit | `medstat.models.survival` |
| **Type 4: Diagnostic Test Accuracy** | Sensitivity, Specificity, PPV, NPV, AUC | 2x2 Contingency Matrix & Empirical ROC | Wilson score 95% CIs; DeLong test for paired AUC; Directionality | `medstat.diagnostic` |
| **Type 4b: Clinical Utility & Calib** | Net Benefit, Model Calibration | Decision Curve Analysis (DCA) & Recalibration | Brier score, Calibration slope & intercept, Austin-Steyerberg ICI | `medstat.diagnostic.dca`, `medstat.diagnostic.calibration` |
| **Type 5: Observational Causal (PSM)** | Average Treatment Effect in Treated (ATT) | Propensity Score Matching (PSM, 1:1, Caliper 0.2 SD) | Austin (2009) Covariate Balance (absolute SMD < 0.10); VanderWeele E-value | `medstat.causal` |
| **Type 5b: Causal Weighting (IPTW)** | Population Effect (ATE, ATT, ATC, Overlap ATO) | Inverse Probability Treatment Weighting (IPTW) | Kish's Effective Sample Size ($N_{\text{eff}} \ge 0.5 \times N$), Positivity (PS 0.05–0.95), Robust SE | `medstat.causal.iptw` |
| **Type 6: Device / Rater Agreement** | Limits of Agreement, Reliability | Bland-Altman Difference Analysis & ICC | Bland-Altman (1999) large-sample CIs; Shrout-Fleiss two-way ANOVA | `medstat.agreement` |
| **Type 7: Systematic Meta-Analysis** | Pooled Effect (RR/OR/MD) | Fixed-Effects or DerSimonian-Laird Random-Effects | Condition on clinical variation; avoid DerSimonian-Laird inverse-variance for rare events (exact/Peto not supported in `medstat.meta`); small $k < 5$ caution | `medstat.meta` |
| **Type 8: Clustered / Multi-Center** | Population-averaged or Cluster Effect | GEE with robust sandwich SE or MixedLM | Cluster Design Effect ($\text{DEFF} = 1 + (\bar{m}-1)\text{ICC}_{\text{cluster}}$) | `medstat.models.multilevel` |

---

## 2. Decision Trees for Common Statistical Challenges

### A. Events Per Variable (EPV) & Sparse Data
```
Calculate EPV (where P_parameters counts fitted degrees of freedom; e.g. k-1 for categorical factors, spline basis terms):
- Binary Logistic: EPV = min(Events, Non-Events) / P_parameters
- Cox Proportional Hazards: EPV = Observed Failures / P_parameters (do not use censored subjects)

├── EPV ≥ 10 & No Separation ──▶ Proceed with standard GLM Logistic / Cox PH
├── Monotone Separation Detected ──▶ Halt at Grilling Gate for clinical confirmation:
│   ├── Binary Logistic: (A) Firth penalized likelihood (`fit_firth_logistic` in `medstat.models.firth`), or (B) Remove/collapse separating covariate
│   └── Cox Proportional Hazards (Monotone Likelihood): (A) Firth penalized Cox (`fit_firth_cox` in `medstat.models.firth`), or (B) Remove/collapse separating covariate
└── Low EPV (< 10) without Separation
    ├── Binary Logistic:
    │   ├── Strategy 1 (Etiological Effect): Firth penalized logistic regression (`fit_firth_logistic` in `medstat.models.firth`)
    │   ├── Strategy 2 (Prediction Model): L2 Ridge penalization / shrinkage to prevent overfitting (Riley et al., 2019)
    │   ├── Strategy 3: Pre-specified variable reduction or domain-guided composite score
    │   └── Strategy 4: Report univariable associations with explicit sparse-data disclaimer
    └── Cox Proportional Hazards (Failures / Parameter < 10):
        ├── Strategy 1: Firth penalized Cox regression (`fit_firth_cox` in `medstat.models.firth`)
        ├── Strategy 2: Pre-specified covariate reduction (e.g. validated clinical risk score)
        └── Strategy 3: Univariable survival analysis (Kaplan-Meier / univariable Cox) with sparse-data disclaimer
```

### B. Missing Data Strategy
```
Audit Missingness Pattern & Mechanism

├── Total Missing < 5% and Clinically Uninformative ──▶ Complete-case analysis with documented justification
├── Missing 5%–40% under Missing at Random (MAR) ──▶ Multiple Imputation by Chained Equations (MICE)
├── Primary Outcome Missing ──▶ Never impute outcome; check protocol disposition (e.g., ITT vs per-protocol vs sensitivity); if unestablished, halt at Grilling Gate for explicit clinical disposition
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

### D. Observational Causal Inference: PSM vs. IPTW vs. Overlap Weights
```
Evaluate Treatment Assignment & Causal Target
├── Target Population
│   ├── Target = Treated Population (ATT) ──▶ 1:1 Nearest-Neighbor PSM with caliper 0.2 SD OR IPTW (estimand='ATT')
│   ├── Target = Whole Target Population (ATE) ──▶ IPTW (estimand='ATE') with stabilized weights
│   └── Significant Propensity Truncation / Lack of Overlap ──▶ Overlap Weights (estimand='ATO', Li et al. 2018)
├── Weight Stability & Positivity Audit
│   ├── Positivity Alert (PS < 0.05 or > 0.95) ──▶ Trim extreme tails OR switch to Overlap Weights (ATO)
│   ├── Extreme Weights (> 10.0) ──▶ Quantile Truncation (0.01, 0.99) OR Overlap Weights
│   └── Kish's Effective Sample Size (Neff < 0.5 * N) ──▶ Warning for variance inflation; use robust SE (HC1)
└── Balance Diagnostics: Austin (2009) Love Plot (absolute SMD < 0.10)
```

### E. Time-to-Event Analysis Under Competing Risks
```
Audit Event Types: Event of Interest vs. Competing Risks (e.g., CV death vs. non-CV death)
├── Single Failure Type (No Competing Risks) ──▶ Standard Kaplan-Meier & Cox Proportional Hazards
└── Multiple / Mutually Exclusive Failures
    ├── Cumulative Event Incidence / Absolute Risk Prediction:
    │   ├── DO NOT use Naive 1 - KM (treats competing events as censoring, estimating risk in a hypothetical world without competing events rather than real-world cumulative incidence; overestimates true risk)
    │   └── Use Aalen-Johansen Cumulative Incidence Function (CIF) (`fit_competing_risks_cif`)
    └── Multivariable Regression:
        ├── Etiological / Biological Question ──▶ Cause-Specific Cox Model (csHR)
        │   └── Censors competing events at failure time (measures rate in event-free patients)
        └── Prognostic / Clinical Risk Stratification ──▶ Fine-Gray Subdistribution Hazard Model (SHR)
            └── Uses IPCW counting process expansion (measures cumulative incidence effect)
        └── Best Practice: Report both csHR and SHR side-by-side with clinical takeaway
```

### F. Diagnostic Risk Reclassification & Incremental Value
```
Evaluating Added Value of Updated Biomarker/Model over Baseline
├── Discrimination Change ──▶ DeLong paired test for ΔAUROC (`compare_diagnostic_models`)
├── Net Reclassification Improvement (NRI)
│   ├── Continuous / Category-Free NRI ──▶ Primary metric when no established clinical thresholds exist (Pencina 2011)
│   ├── Categorical NRI ──▶ Use strictly when established clinical risk tiers exist (e.g. low <10%, intermediate 10-20%, high >20%)
│   ├── Separate Event & Non-Event Reporting ──▶ Mandatory per Kerr & Pepe (2014) to detect asymmetric reclassification
│   └── Calibration Guard ──▶ Enforce well-calibrated probabilities in [0, 1] (poorly calibrated models produce spurious positive NRI)
└── Integrated Discrimination Improvement (IDI)
    └── Difference in discrimination slopes (mean predicted risk difference between events and non-events)
```

### G. Time-Dependent ROC in Longitudinal Follow-up
```
Evaluating Diagnostic / Prognostic Biomarkers Over Follow-up Time
├── Censoring Handling ──▶ Inverse Probability of Censoring Weighting (IPCW) via Kaplan-Meier on censoring distribution G(t) (Uno et al. 2007, Heagerty et al. 2000)
├── Definition ──▶ Cumulative/Dynamic ROC: Cases = primary events by t (T_i <= t, delta_i = 1); Controls = event-free beyond t (T_i > t)
├── Under Competing Risks ──▶ When competing_event_code is provided, patients experiencing competing events by t (T_i <= t, delta_i = k_comp) are retained in the control set weighted by IPCW (Fine-Gray/Uno subdistribution definition); default mode (competing_event_code=None) requires strictly binary event coding {0, 1}
├── Horizon Selection ──▶ Clinically meaningful milestones (e.g., 30d, 90d, 1y, 3y, 5y); avoid horizons where remaining cases or controls < 5
└── Reporting ──▶ AUROC(t) trajectory curve with 95% bootstrap CI band, optimal Youden cutpoint, sensitivity, specificity per milestone
```

---

## 3. Reporting & Table Styling Standards

- **NEJM Style**: 3 horizontal rules (top, header bottom, table bottom), no vertical dividers, $P$-values formatted as $P = 0.04$ or $P < 0.001$.
- **JAMA Style**: Clean minimal dividers, $P$-values formatted without leading zero: $P = .04$ or $P < .001$.
- **APA 7th Style**: Standard psychological and behavioral format with exact degrees of freedom.
- **Reporting Checklists**: Consult STROBE (observational cohorts), CONSORT (trials), TRIPOD (prediction models), STARD (diagnostic accuracy), or PRISMA (systematic reviews).
- **Figures & Visualizations**: Consult [figures.md](figures.md) for 300 DPI publication plots and [report-builder.md](report-builder.md) for HTML Dual / Toggle View (interactive Plotly + 300 DPI static).

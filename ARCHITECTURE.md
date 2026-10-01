# ARCHITECTURE.md — System Architecture & Structural Diary

System architecture and structural specifications for `medstat-core` and the `medstat` CLI in `shinystat-skills`.

## 1. High-Level Architecture Overview

`medstat` is a headless, production-ready biostatistical computation engine and CLI application designed for offline-first clinical research, electronic health records (EHR) analytics, and automated biomedical manuscript preparation.

```
                          ┌───────────────────────────┐
                          │   medstat-master Skill    │
                          │   (Master Orchestrator)   │
                          └─────────────┬─────────────┘
                                        │ orchestrates
                                        ▼
                          ┌───────────────────────────┐
                          │    Atomic Agent Skills    │
                          │   (clean, models, diag,   │
                          │    causal-meta, report)   │
                          └─────────────┬─────────────┘
                                        │ invokes
                                        ▼
                          ┌───────────────────────────┐
                          │    medstat CLI & SAP      │
                          │   (click / YAML spec)     │
                          └─────────────┬─────────────┘
                                        │ executes
                                        ▼
      ┌─────────────────────────────────┴─────────────────────────────────┐
      │                      Core Calculation Modules                     │
      ├───────────────────┬───────────────────┬───────────────────────────┤
      │  medstat.data     │  medstat.models   │  medstat.diagnostic       │
      │  - clean          │  - glm / firth    │  - 2x2 contingency        │
      │  - missing (MICE) │  - survival (cox) │  - ROC & DeLong CIs       │
      │  - retention flow │  - splines (RCS)  │  - DCA net benefit        │
      ├───────────────────┼───────────────────┼───────────────────────────┤
      │  medstat.causal   │  medstat.meta     │  medstat.reporting        │
      │  - psm matching   │  - DL random-eff  │  - NEJM / JAMA / APA 7    │
      │  - balance & SMD  │  - forest data    │  - STROBE/CONSORT/TRIPOD  │
      │  - love plots     │  - Egger's test   │  - methods narrative      │
      ├───────────────────┴───────────────────┴───────────────────────────┤
      │                     medstat.agreement                             │
      │  - Bland-Altman LoA with Carkeet CIs                              │
      │  - Pure-SciPy Intraclass Correlation Coefficient (ICC) (No GPL)   │
      └───────────────────────────────────────────────────────────────────┘
```

---

## 2. Module Seams & Responsibilities

### `medstat.data`
- **`loader.py`**: Universal clinical tabular ingestion (.csv, .tsv, .xlsx, .parquet) with auto-encoding detection, whitespace stripping, and fuzzy column suggestion.
- **`clean.py`**: Missingness auditing, Little's MCAR multivariate test, Winsorization / IQR outlier handling, and strategy routing.
- **`missing.py`**: Implementation of `complete-case`, `mice` (via Bayesian ridge chained equations), `knn`, and `indicator` methods. Raises `MissingStrategyRequiredError` if missing data exists without an explicit strategy and documented justification.
- **`retention.py`**: `SampleFlowTracker` recording participant retention ($N_{\text{initial}} \to N_{\text{excluded}} \to N_{\text{analyzed}}$) with categorized reasons.

### `medstat.models`
- **`glm.py`**: OLS linear regression and standard binary logistic regression via `statsmodels`.
- **`firth.py`**: Firth penalized logistic regression and penalized Cox proportional hazards via `firthmodels` with profile likelihood confidence intervals.
- **`survival.py`**: Cox proportional hazards modeling via `lifelines` and Grambsch-Therneau Schoenfeld residual correlation tests.
- **`splines.py`**: Restricted cubic splines (RCS) with flexible knot placement via pure-Python `rcs_lib.py` for both Cox proportional hazards and multivariable logistic regression.
- **`sensitivity.py`**: VanderWeele & Ding E-value computation for point estimates and lower/upper confidence bounds.

### `medstat.diagnostic`
- **`accuracy.py`**: Complete 2x2 contingency metrics (Sensitivity, Specificity, PPV, NPV, LR+, LR-, DOR) with Wilson score confidence intervals and low-is-abnormal directionality handling.
- **`roc.py`**: Non-parametric empirical ROC curve, Youden's J cutpoint, and DeLong covariance matrix calculation for paired AUC comparisons.
- **`dca.py`**: Vickers Decision Curve Analysis calculating net benefit across threshold probabilities relative to "Treat All" and "Treat None".

### `medstat.causal` & `medstat.agreement`
- **`psm.py`**: Propensity score estimation via logistic regression, 1:1 nearest neighbor matching with logit standard deviation caliper, and matched cohort extraction.
- **`balance.py`**: Standardized Mean Difference (SMD) calculation for continuous and binary variables, and Austin 2009 Love plot data generation.
- **`mediation.py`**: Pure-Python parametric causal mediation analysis (Baron-Kenny / Imai) with quasi-Bayesian Monte Carlo confidence intervals for Average Causal Mediation Effect (ACME) and Average Direct Effect (ADE).
- **`bland_altman.py`**: Mean difference, 95% Limits of Agreement, and Carkeet (2015) confidence intervals. Supports both wide paired columns and long-format rater data.
- **`icc.py`**: Pure SciPy/NumPy two-way ANOVA decomposition computing Shrout & Fleiss (1979) forms (ICC1, ICC2, ICC3, ICC1k, ICC2k, ICC3k) with exact F-distribution confidence intervals.
- **`kappa.py`**: Inter-rater categorical agreement computing Cohen's Kappa for 2 raters and Fleiss' Kappa for $m \ge 2$ raters.

### `medstat.meta`
- **`models.py`**: Fixed-effects inverse variance and DerSimonian-Laird random-effects meta-analysis, Cochran's Q test, and Higgins $I^2$.
- **`forest.py`**: Forest plot structured data generation.
- **`bias.py`**: Egger's linear regression test for funnel plot asymmetry.

### `medstat.reporting`
- **`tables.py`**: Journal-compliant polymorphic HTML table rendering (NEJM, JAMA, APA 7) with strict border rules, no vertical dividers, and support for Table 1, Regression, Diagnostic accuracy, Bland-Altman, ICC, and Covariate balance.
- **`narrative.py`**: Automated biomedical Methods and Results narrative generation.
- **`checklists.py`**: Audited item checklists for STROBE, CONSORT, TRIPOD, STARD (diagnostic studies), and PRISMA (systematic reviews).

---

## 3. Data Flow & Contract Interfaces

```mermaid
flowchart LR
    Data[Clinical Ingestion: CSV / XLSX / TSV / Parquet] --> Profile[medstat profile: Data Health & Design Inference]
    Profile --> Clean[medstat.data.clean: Little's MCAR + Winsorization]
    Clean -->|Audited Cohort + Flow| Model[medstat.models / causal / diag / agreement]
    Model -->|JSON Estimates Contract| Report[medstat.reporting: Polymorphic Renderer]
    Report -->|HTML Table + Narrative + Checklist| Manuscript[Publication Draft]
```

All subcommands emit structured JSON contracts allowing easy composition into pipelines, agent tools, or downstream rendering engines.

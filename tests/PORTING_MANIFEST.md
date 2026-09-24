# PORTING_MANIFEST.md — Unit Test Migration & Parity Audit

Comprehensive audit manifest tracking the decoupling and porting of all 54 test modules from the legacy prototype (`tests/unit/`) to `medstat-core` (`tests/unit/`, `tests/e2e/`, and `tests/stress/`).

---

## 1. Porting Strategy & Scope

The legacy test suite contained a mixture of pure statistical calculations, data preparation routines, and Shiny UI/reactive components. Under **Requirement R1 (Headless Architecture)**, GUI-bound code was decoupled:
1. **Core Statistical & Clinical Modules**: Ported directly or expanded into dedicated headless unit/stress test suites.
2. **GPL-Infected Tests**: Replaced with pure-SciPy equivalents (e.g. `pingouin` ICC replaced with SciPy two-way ANOVA parity tests).
3. **Shiny/UI Specific Tests**: Explicitly retired (reactive state machines, Shiny notification handlers, UI CSS styles, and download helpers).

---

## 2. Manifest of All 54 Legacy Test Files

| # | Legacy Test File | Target in `shinystat-skills` | Porting Status | Rationale & Architectural Notes |
|---|:---|:---|:---|:---|
| 1 | `test_data_cleaning.py` | `tests/unit/test_data_cleaning.py` | **Ported** | Core cleaning, missing value detection, type inference. |
| 2 | `test_data_cleaning_advanced.py` | `tests/unit/test_data_cleaning.py` | **Ported** | Advanced anomaly detection and IQR winsorization. |
| 3 | `test_data_cleaning_workflow.py` | `tests/e2e/test_tier4_clinical_workflows.py` | **Ported & Upgraded** | End-to-end data preparation workflow with audited sample flow. |
| 4 | `test_data_quality.py` | `tests/unit/test_data_quality.py` | **Ported** | Data quality checks, column rules, and clinical schemas. |
| 5 | `test_data_quality_report.py` | `tests/unit/test_data_quality_report.py` | **Ported** | 5D Data Quality Dimensions (Completeness, Validity, etc.). |
| 6 | `test_littles_mcar.py` | `tests/unit/test_littles_mcar.py` | **Ported** | Little's multivariate MCAR chi-square test ($d^2$). |
| 7 | `test_missing_data.py` | `tests/unit/test_missing_data.py` | **Ported** | Missing data mechanisms and `MissingStrategyRequiredError`. |
| 8 | `test_multiple_imputation.py` | `tests/unit/test_multiple_imputation.py` | **Ported** | Chained equations (MICE), Rubin's pooling, and FMI. |
| 9 | `test_mi_reporting.py` | `tests/unit/test_sample_flow_retention.py` | **Consolidated** | Participant flow tracking through imputation stages. |
| 10 | `test_firth_regression.py` | `tests/stress/test_m1_stress_icc_firth.py` | **Ported & Deepened** | Profile Likelihood 95% CIs and LRT p-values under separation. |
| 11 | `test_bland_altman.py` | `tests/e2e/test_tier1_feature_coverage.py` | **Pending** | Standalone unit tests pending; covered in E2E. |
| 12 | `test_calibration_ici.py` | `tests/e2e/test_tier1_feature_coverage.py` | **Pending** | Standalone unit tests pending; covered in E2E. |
| 13 | `test_causal.py` | `tests/e2e/test_tier1_feature_coverage.py` | **Ported** | Propensity score matching, caliper, and SMD balance. |
| 14 | `test_dca.py` | `tests/e2e/test_tier1_feature_coverage.py` | **Ported** | Vickers Decision Curve Analysis net benefit curves. |
| 15 | `test_diag_returns.py` | `tests/e2e/test_tier2_boundary_corner_cases.py` | **Ported** | 2x2 contingency matrix, Sensitivity, Specificity, Wilson CIs. |
| 16 | `test_diagnostic_advanced.py` | `tests/e2e/test_tier1_feature_coverage.py` | **Ported** | Empirical ROC, Youden index, DeLong paired AUC test. |
| 17 | `test_effect_sizes.py` | `tests/e2e/test_tier1_feature_coverage.py` | **Pending** | Standalone unit tests pending; covered in E2E. |
| 18 | `test_fagan_nomogram.py` | `tests/e2e/test_tier1_feature_coverage.py` | **Pending** | Not yet ported to headless suite. |
| 19 | `test_formatting.py` | `tests/e2e/test_tier1_feature_coverage.py` | **Ported** | Medical journal precision and rounding rules. |
| 20 | `test_formatting_styles.py` | `tests/e2e/test_tier1_feature_coverage.py` | **Ported** | NEJM, JAMA, and APA 7 typography rules. |
| 21 | `test_glm.py` | `tests/e2e/test_tier1_feature_coverage.py` | **Ported** | Generalized linear models (logistic, linear regression). |
| 22 | `test_heterogeneity.py` | `tests/e2e/test_tier1_feature_coverage.py` | **Ported** | Cochran's Q and Higgins $I^2$ meta-analytic heterogeneity. |
| 23 | `test_linear_lib.py` | `tests/e2e/test_tier1_feature_coverage.py` | **Ported** | OLS regression, ANOVA decomposition, and parameter estimates. |
| 24 | `test_mediation.py` | `tests/e2e/test_tier1_feature_coverage.py` | **Conditional** | Causal mediation analysis (runs only when optional dependency installed). |
| 25 | `test_meta_analysis.py` | `tests/e2e/test_tier1_feature_coverage.py` | **Ported** | DerSimonian-Laird random effects, Forest plots, Egger's test. |
| 26 | `test_model_diagnostics.py` | `tests/e2e/test_tier1_feature_coverage.py` | **Pending** | Multicollinearity (VIF) and influence diagnostics pending. |
| 27 | `test_model_diagnostics_plots.py` | `tests/e2e/test_tier1_feature_coverage.py` | **Pending** | Diagnostic residual plot data structures pending. |
| 28 | `test_poisson_lib.py` | `tests/e2e/test_tier1_feature_coverage.py` | **Pending** | Poisson and negative binomial count regression pending. |
| 29 | `test_publication_renderer.py` | `tests/e2e/test_tier1_feature_coverage.py` | **Ported** | HTML publication table rendering without vertical borders. |
| 30 | `test_regression_publication.py` | `tests/e2e/test_tier1_feature_coverage.py` | **Ported** | Publication table styling for GLM and Cox models. |
| 31 | `test_repeated_measures.py` | `tests/stress/test_m1_stress_icc_firth.py` | **Ported & Re-engineered** | Replaced `pingouin` with pure-SciPy two-way ANOVA ICC. |
| 32 | `test_reporting_checklists.py` | `tests/e2e/test_tier1_feature_coverage.py` | **Ported** | STROBE, CONSORT, and TRIPOD checklist audits. |
| 33 | `test_sample_size.py` | `tests/e2e/test_tier1_feature_coverage.py` | **Ported** | Two-sample t-test, proportions, and survival sample sizing. |
| 34 | `test_sensitivity.py` | `tests/e2e/test_tier1_feature_coverage.py` | **Ported** | VanderWeele & Ding E-value calculations for point/CI bounds. |
| 35 | `test_sensitivity_fixes.py` | `tests/e2e/test_tier1_feature_coverage.py` | **Ported** | Numerical bounds handling in E-value edge cases. |
| 36 | `test_statistical_assumptions.py` | `tests/e2e/test_tier2_boundary_corner_cases.py`| **Ported** | Normality, homoscedasticity, and linearity assumption tests. |
| 37 | `test_statistics.py` | `tests/unit/` & `tests/e2e/` | **Ported** | Parametric and non-parametric bivariate hypothesis tests. |
| 38 | `test_survival_assumptions.py` | `tests/e2e/test_tier1_feature_coverage.py` | **Ported** | Schoenfeld residual tests for proportional hazards. |
| 39 | `test_survival_lib_patch.py` | `tests/e2e/test_tier1_feature_coverage.py` | **Ported** | Lifelines Cox PH wrapper and robust covariance estimation. |
| 40 | `test_tvc_lib.py` | `tests/e2e/test_tier1_feature_coverage.py` | **Pending** | Time-varying covariates in survival analysis pending. |
| 41 | `verify_table_one.py` | `tests/e2e/test_tier1_feature_coverage.py` | **Ported** | Table 1 baseline characteristics with continuous/binary SMDs. |
| 42 | `test_collinearity.py` | `tests/e2e/test_tier2_boundary_corner_cases.py`| **Ported** | Boundary handling of perfectly collinear feature matrices. |
| 43 | `test_correlation_returns.py` | `tests/e2e/test_tier1_feature_coverage.py` | **Ported** | Pearson and Spearman correlation coefficient returns. |
| 44 | `test_medical_edge_cases.py` | `tests/e2e/test_tier2_boundary_corner_cases.py`| **Ported** | Extreme clinical values, constant features, sparse events. |
| 45 | `test_advanced_stats.py` | `tests/e2e/test_tier1_feature_coverage.py` | **Ported** | Restricted cubic splines (RCS) knot placement & contrasts. |
| 46 | `test_color_palette.py` | `src/medstat/theme/palette.py` | **Headless Decoupled** | Decoupled from `tabs._common`; centralized headless palette. |
| 47 | `test_chi_html.py` | N/A | **Retired (UI)** | Shiny HTML modal rendering for Chi-Square dialogs. |
| 48 | `test_download_helpers.py` | N/A | **Retired (UI)** | Shiny session browser download handler callbacks. |
| 49 | `test_pdf_helpers.py` | N/A | **Retired (UI)** | Headless PDF export handled via Playwright extra, not Shiny. |
| 50 | `test_phase3_features.py` | N/A | **Retired (UI)** | Legacy Shiny UI tab state navigation. |
| 51 | `test_plotly_html_rendering.py`| N/A | **Retired (UI)** | HTML `<div>` injection for Shiny tab containers. |
| 52 | `test_state_machine.py` | N/A | **Retired (UI)** | Shiny reactive session state transitions. |
| 53 | `test_tab_diag_html_logic.py` | N/A | **Retired (UI)** | Shiny UI conditional rendering for diagnostic tab widgets. |
| 54 | `test_ui_ux_styles.py` | N/A | **Retired (UI)** | CSS class assertions for Bootstrap/Shiny UI buttons. |

---

## 3. Summary Statistics

- **Total Legacy Modules**: 54 files.
- **Ported to Headless Core (`tests/unit/`, `tests/e2e/`, `tests/stress/`)**: 37 files (68.5%).
- **Conditional / Optional Extra**: 1 file (1.9%).
- **Pending Porting**: 8 files (14.8%).
- **Retired GUI/Shiny UI Modules**: 8 files (14.8%).
- **Total Tests Currently Executed & Passing**: **255 Passed, 1 Skipped, 0 Failed**.

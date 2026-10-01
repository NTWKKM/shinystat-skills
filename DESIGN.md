# DESIGN.md — Architectural Decision Records & Trade-Off Diary

Architectural Decision Records (ADRs), rationale, constraints, and rejected alternatives for `medstat-core`.

---

## ADR 1: Pure SciPy/NumPy Two-Way ANOVA for ICC (Elimination of Pingouin)

### Context
Intraclass Correlation Coefficient (ICC) was previously computed using the `pingouin` package in the predecessor prototype. However, `pingouin` is licensed under GPL-3.0, which imposes copyleft restrictions on public releases and commercial distribution.

### Decision
Implement the complete Shrout & Fleiss (1979) and McGraw & Wong (1996) ICC taxonomy (`icc1`, `icc2`, `icc3`, `icc1k`, `icc2k`, `icc3k`) using pure SciPy/NumPy two-way ANOVA decomposition (`medstat.agreement.icc`). Calculate exact F-distribution confidence intervals directly via `scipy.stats.f`.

### Consequences
- **Status**: Accepted & Verified.
- **Licensing**: Repository is 100% compliant with permissive **Apache-2.0** licensing.
- **Parity**: Validated against `pingouin` across 19 unit & stress tests to within $10^{-6}$ numerical tolerance.
- **Performance**: Zero external package overhead; runs in sub-millisecond execution time.

---

## ADR 2: Decoupled Headless Library & CLI Architecture

### Context
The original implementation in the predecessor GUI prototype was tightly coupled to Shiny UI widgets (`shiny.ui.notification_show`, `tabs._common.get_color_palette`). AI agents and command-line automated pipelines require pure execution without web server runtimes or GUI dependencies.

### Decision
Extract all core statistical and data preparation routines into a standalone Python package `medstat-core` (`src/medstat/`). Provide a unified Click-based CLI executable `medstat` supporting both direct CLI flags and YAML Statistical Analysis Plan (`--spec analysis_plan.yaml`) execution.

### Consequences
- **Status**: Accepted & Verified.
- **Zero-Shiny Invariant**: Enforced by automated tests; zero imports of `shiny` or `shiny.ui` within `src/medstat/`.
- **Reproducibility**: Pre-registered YAML analysis plans can be executed in CI/CD pipelines, headless Docker containers, or agent workflows.

---

## ADR 3: Pure Python Firth Models vs R Oracle Validation

### Context
Firth's penalized likelihood estimation is essential for resolving monotone separation in sparse clinical cohorts. While `logistf` and `coxphf` are popular in R, runtime dependencies on R (`rpy2`) introduce severe deployment hurdles, platform inconsistencies, and Docker image bloat.

### Decision
Run all primary penalization in pure Python using `firthmodels >= 0.8.2` (logistic and Cox proportional hazards with profile likelihood confidence intervals). Retain R scripts (`test_firth.R`) strictly in `tests/benchmarks/r_scripts/` as an offline validation benchmark.

### Consequences
- **Status**: Accepted & Verified.
- **Deployment**: Zero runtime R dependency; installs cleanly via `pip install .` on any modern Python 3.12+ environment.
- **Validation**: Numerical parity confirmed within $10^{-4}$ tolerance on benchmark clinical datasets (`sex2`, `breast`).

---

## ADR 4: Mandatory Missing Data Strategy & Sample Flow Audit

### Context
Standard biostatistical practice frequently defaults to silent listwise deletion, implicitly assuming Missing Completely at Random (MCAR). In clinical research, this introduces substantial selection bias and violates CONSORT/STROBE guidelines.

### Decision
Enforce `MissingStrategyRequiredError` at runtime whenever missing values are detected without an explicit strategy (`complete-case`, `mice`, `knn`, `indicator`) and documented clinical justification. Every operation records an audited participant retention tracker:
$$N_{\text{initial}} \longrightarrow N_{\text{excluded}} \longrightarrow N_{\text{analyzed}}$$

### Consequences
- **Status**: Accepted & Verified.
- **Clinical Safety**: Eliminates accidental biased deletions; mandates explicit clinical assumptions.
- **Audit Trail**: Retention flow counts are emitted in CLI JSON outputs and HTML report headers.

---

## ADR 5: Self-Contained Atomic Agent Skills

### Context
AI agents loading skills frequently fail when skills rely on centralized shared directories or complex multi-skill routing networks.

### Decision
Package 5 atomic skills (`medstat-clean`, `medstat-models`, `medstat-diagnostic`, `medstat-causal-meta`, `medstat-report`), each with its own localized `references/` directory. Frontmatter descriptions are kept strictly under 1024 characters with front-loaded trigger words.

### Consequences
- **Status**: Accepted & Verified.
- **Portability**: Each skill directory can be copied independently into `.agents/skills/`, `.claude/skills/`, or `.cursor/skills/` without breaking any relative markdown links.
- **Installer**: Provided `scripts/install-skills.sh` automates single-command deployment across platforms.

---

## ADR 6: Strict Numeric 0/1 Encoding for Binary Outcomes & Event Indicators

### Context
In clinical regression and survival modeling, accepting raw string/categorical outcomes (e.g., `"Dead"`, `"Alive"`, `"Yes"`, `"No"`) risks silent clinical event inversion depending on alphabetical ordering or factor level assignment across different statistical packages (e.g., Python `statsmodels` vs R vs SAS).

### Decision
Enforce strict numeric `0` and `1` (`1 = Event`, `0 = Non-event`) encoding across all modeling CLI commands, YAML SAP execution engines, and skill instructions. Reject non-numeric outcome columns at CLI entry points with explicit `ClickException` messages instructing the user/agent to recode endpoints during the `medstat-clean` phase.

### Consequences
- **Status**: Accepted & Verified.
- **Clinical Safety**: Eliminates the catastrophic risk of inverse odds/hazard ratio estimation ($HR < 1$ mistaken for protective when it is harmful).
- **Separation of Concerns**: Data cleaning and recoding are isolated in `medstat-clean`, keeping `medstat-models` deterministic and unambiguous.

---

## ADR 7: Meta-Analysis Scale Alignment and FMI Boundary Regularization

### Context
In meta-analyses, study effect sizes and confidence intervals may be supplied on natural ratio or log scales. Intermingling unexponentiated log effects with exponentiated pooled metrics creates severe scale disharmony in published tables and forest plots. Furthermore, in Rubin's pooling under zero within-imputation variance ($\bar{W} \le 0, B > 0$), relative variance increase $r = \infty$, yielding an indeterminate float $\frac{\infty}{\infty} = \text{NaN}$ for Fraction of Missing Information (FMI).

### Decision
1. In `report_cmd`, enforce consistent exponentiation of both point estimates and confidence intervals for both individual studies and overall pooled results when ratio effect measures (OR/RR/HR) are specified with `log` scale metadata. Explicit scale metadata (`ci_scale` or `scale` as `log` or `natural`) is strictly required when providing confidence limits alongside `log_effect` or log-scale estimates; ambiguous records without scale metadata are rejected with a clear validation error to prevent misinterpretation of natural vs log confidence bounds.
2. In `pool_estimates`, evaluate the analytic limit $\lim_{r \to \infty} \text{FMI} = 1.0$ when $r = \infty$ ($\bar{W} \le 0, B > 0$), eliminating NaN values and accurately communicating complete between-imputation information dominance.
3. In analysis plan methods narrative, preserve specified `missing_strategy` without unevidenced complete-case defaults, and set unpooled single-imputation MICE narrative strategy to `None` to prevent false complete-case claims.

### Consequences
- **Status**: Accepted & Verified.
- **Precision & Reliability**: Forest plots and tables display coherent natural-scale numbers; FMI remains strictly defined on $[0, 1]$.

---

## ADR 8: Reporting Guidelines Modernization & Multi-Agent Parity

### Context
CodeRabbit AI review on PR #2 identified discrepancies across distributed skill copies (`.agent/`, `.agents/`, `.claude/`, `.cursor/`), outdated reporting checklist references (CONSORT 2010 instead of 2025, TRIPOD 2015 instead of TRIPOD+AI 2024), mechanical $I^2$ cutoffs contradicting Cochrane Chapter 10, and mismatched R benchmark scale limits.

### Decision
1. Modernize trial and prediction reporting standards across all skills to **CONSORT 2025** and **TRIPOD+AI (2024)**, revising analysis population descriptions to accommodate Intention-to-Treat, modified ITT, and per-protocol workflows.
2. Remove mechanical $I^2$ model switching thresholds; align meta-analysis guidance with Cochrane Chapter 10 by conditioning model choice on clinical/methodological variation assumptions.
3. Explicitly state the simulation context of Austin (2009) caliper matching (98–99% bias reduction in measured continuous baseline covariates, without eliminating unmeasured confounding).
4. Accurately relabel Limits of Agreement variance formulas as large-sample approximations (Bland & Altman 1999) matching the codebase implementation.
5. Establish `skills/` as the single authoritative source of truth and propagate identical contents to all 4 multi-agent platforms to ensure zero cross-platform drift.

### Consequences
- **Status**: Accepted & Verified.
- **Guideline Compliance**: Aligns the toolkit with current ICMJE and EQUATOR Network standards.
- **Cross-Platform Uniformity**: Guarantees identical agent instructions regardless of IDE or agent runtime.

---

## ADR 9: Master Biostatistical Orchestrator & Autonomous SAP Execution

### Context
Users interacting with AI agents often upload clinical spreadsheets (CSV, XLSX) without knowing which atomic skill (`medstat-clean`, `medstat-models`, `medstat-diagnostic`, `medstat-causal-meta`, `medstat-report`) to invoke. Forcing users to choose individual skills introduces friction and risks incorrect statistical method selection (e.g., selecting multivariable logistic regression for clustered rater data or treating text outcomes directly).

### Decision
Introduce `medstat-master` as the master orchestrator skill. It operates in two modes:
1. **Direct Autonomous Execution**: Automatically audits data health, establishes explicit event mapping (including "Dead"/"Alive", "Yes"/"No", "Recurred"/"Disease-Free"; stopping for clinician confirmation when event direction or censoring status is ambiguous), recodes binary endpoints to numeric `0/1` (ensuring in survival analysis that 1 = Event and 0 = Censored without inversion), infers study design, runs the clean $\to$ model $\to$ report pipeline, and returns finished manuscript tables when user intent is unambiguous.
2. **Statistical Analysis Proposal (SAP) Mode**: Synthesizes a structured 1-page clinical proposal aligning primary estimand, missingness mechanism, candidate model options, and target journal styles when ambiguity exists.

Downstream atomic skills are chained seamlessly via CLI subcommands without requiring manual skill switching.

### Consequences
- **Status**: Accepted & Verified.
- **User Experience**: Users can drop any clinical tabular file into chat and receive appropriate biostatistical analysis automatically.
- **Clinical Governance**: Enforces all core invariants (strict numeric 0/1 outcomes with verified event mapping, zero silent deletion with sample retention flow, Wilson CIs, DeLong AUC) centrally before any downstream model execution.

---

## ADR 10: Executable Documentation Contract & Skill CLI Parity

### Context
CodeRabbit AI review on PR #4 and local testing revealed that documentation and skill instructions drifted from actual CLI implementations:
1. Phantom CLI flags (`model fit`, `--y`, `--x`, `--event`, `--firth`, `meta dl`) were documented that did not exist in the Click CLI options.
2. SAP YAML spec templates used an imaginary schema (`dataset`, `cleaning`, `table1`, `primary_model`) that crashed `AnalysisPlan.from_yaml` at runtime.
3. Multiple skill mirror directories (`.agent/`, `.agents/`, `.claude/`, `.cursor/`, `skills/`) lacked automated drift detection, allowing out-of-sync documentation across agent platforms.

### Decision
1. **Strict CLI Parity**: All CLI snippets in `SKILL.md` and reference documents must reflect exact, verifiable Click CLI options (`medstat model --outcome/--exposure/--covariates/--method firth`, `medstat model --spec`, `medstat meta --data --effect-col --se-col --study-col --model random --method dl`).
2. **Executable Spec Templates**: All YAML templates in skill documentation must strictly adhere to the `AnalysisPlan` schema (`version`, `metadata`, `data`, `variables`, `models`, `reporting`) and execute successfully end-to-end via `medstat model --spec`.
3. **Automated Drift Enforcement**: Enforce multi-agent mirror parity and YAML template validity programmatically via unit tests (`tests/unit/test_skill_docs_drift.py`).

### Consequences
- **Status**: Accepted & Verified.
- **Reliability**: Any command or configuration copied by an AI agent or human analyst executes successfully without syntax crashes.
- **Drift Immunity**: Continuous testing fails immediately if mirror copies diverge or if CLI options change without updating skill documentation.

---

## ADR 11: Universal Clinical Ingestion & Automated Data Geometry Profiling

### Context
Clinical datasets originate from diverse hospital IT systems (EHR exports, registries, bedside ultrasound logs) across varied formats: Excel (`.xlsx`, `.xls`), Tab-Separated (`.tsv`), Parquet, and CSVs with divergent text encodings (`utf-8-sig`, `cp1252`, `latin1`) or trailing column whitespace. Furthermore, users and AI agents frequently encounter column name typos that caused uninformative `KeyError` crashes.

### Decision
1. Implement universal loader `load_clinical_data` in `medstat.data.loader` supporting CSV, TSV, Excel, and Parquet with automated fallback encodings (`utf-8`, `utf-8-sig`, `latin1`, `cp1252`) and automatic column name whitespace stripping.
2. Introduce `validate_columns` with `difflib.get_close_matches` providing intelligent suggestions when required columns are missing (e.g., *"Did you mean 'statin_rx' instead of 'tx_statin'?"*).
3. Introduce `medstat profile` command providing an instant one-shot clinical data health overview: cohort dimensions, missingness percentage, candidate clinical outcomes, survival endpoints, biomarker columns, and automated clinical study design inference.

### Consequences
- **Status**: Accepted & Verified.
- **Robustness**: Agents and clinicians can pass any standard spreadsheet directly without manual format pre-conversion.
- **Self-Healing Ergonomics**: Typo suggestions guide agents to self-correct variable names immediately without looping on failures.

---

## ADR 12: Biomarker Directionality & Safe Clinical Diagnostic Inference

### Context
In clinical diagnostic evaluation, high biomarker values typically indicate disease (e.g., Troponin, Lactate, Procalcitonin). However, critical clinical indicators are abnormal when *low* (e.g., Platelet count in severe thrombocytopenia, eGFR in renal failure, PaO2/FiO2 ratio in ARDS). Hardcoding $(y_{\text{score}} \ge \text{cutoff})$ inverted diagnostic accuracy (reporting sensitivity as $(1 - \text{sensitivity})$ and inverting ROC curve concordance).

### Decision
1. Add explicit `--direction [high|low]` option to `medstat diag` (defaulting to `high`).
2. When `--direction low` is specified:
   - Cutoff classification uses $(y_{\text{score}} \le \text{cutoff})$.
   - Biomarker scores are negated ($y_{\text{eff}} = -y_{\text{score}}$) for rank sweeps in ROC calculation, Youden's index, and paired DeLong AUC comparisons, ensuring positive concordance is strictly preserved.
3. Enforce that at least one analytical flag (`--cutoff`, `--roc`, `--compare-roc`, `--dca`, `--calibration`) is provided, preventing empty runs.

### Consequences
- **Status**: Accepted & Verified.
- **Clinical Safety**: Eliminates diagnostic inversion risk for low-is-abnormal laboratory and physiological tests.
- **Deterministic Validity**: True positives and true negatives align with bedside clinical definitions.

---

## ADR 13: Polymorphic Publication Reporting & Reporting Guidelines Modernization

### Context
Downstream reporting in `medstat report` previously crashed (`AttributeError: 'list' object has no attribute 'get'`) when handling list-based results such as Table 1 baseline records or Intraclass Correlation (ICC) tables. Furthermore, while STROBE, CONSORT, and TRIPOD were supported, diagnostic accuracy studies (STARD 2015) and systematic reviews / meta-analyses (PRISMA 2020) lacked structured reporting checklists.

### Decision
1. Implement polymorphic dispatch in `report_cmd` and `medstat.reporting.tables`:
   - Detects list schemas and renders `render_records_table`.
   - Polymorphically routes dictionary schemas into specialized publication renderers: `render_diagnostic_table`, `render_bland_altman_table`, and `render_balance_table`.
2. Expand `medstat.reporting.checklists` with comprehensive `get_stard_checklist()` (STARD 2015, 25 essential items) and `get_prisma_checklist()` (PRISMA 2020, 27 essential items), accessible directly via `medstat report --checklist [stard|prisma]`.

### Consequences
- **Status**: Accepted & Verified.
- **Publication Readiness**: Seamlessly converts all core statistical outputs (Table 1, GLM/Cox regression, diagnostic test accuracy, Bland-Altman LoA, ICC reliability, and Causal PSM covariate balance) into publication-styled HTML tables matching NEJM, JAMA, and APA 7 standards.
- **Guideline Completeness**: Full coverage across the major EQUATOR Network publication guidelines.

---

## ADR 14: Biostatistical Precision, Rater Agreement Rigor, and Probability-Scale DCA Calibration

### Context
Code review identified statistical subtleties across rater agreement, mediation, spline odds ratios, and diagnostic decision curve analysis:
1. Cohen's Kappa confidence intervals previously used the null standard error instead of the large-sample non-null standard error accounting for weights; Fleiss' Kappa lacked rater count uniformity checks and silently binned continuous ratings into quartiles.
2. Mediation with binary outcomes returned log-odds coefficients under OLS fallbacks without explicit scale labeling.
3. Logistic RCS summary tables reported exponentiated basis terms as odds ratios instead of contrast estimates relative to reference values.
4. DCA and calibration calculations accepted unoriented scores and arbitrary ranges outside $[0, 1]$.
5. Outlier detection lacked flag-only count recording and operated blindly on low-cardinality/binary columns.
6. Sample size calculations relied on silent default effect sizes and heuristic conversions.

### Decision
1. **Agreement**: Calculate separate Fleiss-Cohen-Everitt null SE (for hypothesis z-tests) and large-sample non-null SE (for 95% CIs) in `cohens_kappa`. Validate constant rater count per subject in `fleiss_kappa` and reject continuous ratings instead of quartile-binning them. Require explicit rater columns or long format.
2. **Mediation**: Reject silent OLS fallbacks in binary Logit fits, and label effect estimates explicitly with `"scale": "log_odds"` vs `"scale": "linear"`.
3. **Spline Contrasts**: Exclude basis terms and intercept from `odds_ratio` in `summary_df` (set to NaN), and construct a contrast matrix relative to `ref_value` to yield interpretable Odds Ratio trajectories across continuous exposure grids in `contrast_df`.
4. **DCA & Calibration**: Enforce risk probability orientation: if scores are bounded in $[0, 1]$, orient low-abnormality scores as $1 - p$; if continuous biomarkers are passed, orient via univariate logistic regression to map scores into valid probability scales.
5. **Data Cleaning & Loader**: Exclude binary/low-cardinality columns (distinct non-null values $\le 2$) from outlier processing and record per-column outlier counts when `--outlier-action flag` is passed. Support multi-encoding fallback (`utf-8`, `utf-8-sig`, `latin1`, `cp1252`) in universal data loader.
6. **Sample Size**: Require explicit parameter inputs (`--p1`/`--p2` for proportions, `--hazard-ratio` for survival, `--r` for correlation) and reject silent default fallbacks.

### Consequences
- **Status**: Accepted & Verified.
- **Statistical Rigor**: Eliminates false confidence interval coverage in weighted kappa, misleading spline basis ORs, and decision curve artifacts from unoriented raw biomarker scores.

[MEMORY_LEARN: Strict non-null standard errors for weighted kappa, contrast matrices for non-linear spline ORs, and probability-oriented DCA curves ensure rigorous biostatistical validity in automated clinical pipelines.]




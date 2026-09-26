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
1. **Direct Autonomous Execution**: Automatically audits data health, recodes binary endpoints to numeric `0/1`, infers study design, runs the clean $\to$ model $\to$ report pipeline, and returns finished manuscript tables when user intent is unambiguous.
2. **Statistical Analysis Proposal (SAP) Mode**: Synthesizes a structured 1-page clinical proposal aligning primary estimand, missingness mechanism, candidate model options, and target journal styles when ambiguity exists.

Downstream atomic skills are chained seamlessly via CLI subcommands without requiring manual skill switching.

### Consequences
- **Status**: Accepted & Verified.
- **User Experience**: Users can drop any clinical tabular file into chat and receive appropriate biostatistical analysis automatically.
- **Clinical Governance**: Enforces all core invariants (strict numeric 0/1 outcomes, zero silent deletion with sample retention flow, Wilson CIs, DeLong AUC) centrally before any downstream model execution.



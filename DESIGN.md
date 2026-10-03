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
1. Implement universal loader `load_clinical_data` in `medstat.data.loader` supporting CSV, TSV, Excel, and Parquet with automated fallback encodings (`utf-8`, `utf-8-sig`, `cp1252`, `latin1`) and automatic column name whitespace stripping.
2. Introduce `validate_columns` with `difflib.get_close_matches` providing intelligent suggestions when required columns are missing (e.g., *"Did you mean 'statin_rx' instead of 'tx_statin'?"*).
3. Introduce `medstat profile` command providing an instant one-shot clinical data health overview: cohort dimensions, missingness percentage, candidate clinical outcomes, survival endpoints, and limited heuristic study-design inference. All detected binary and survival-time endpoint candidates are heuristic and require clinical confirmation before use in an analysis (including when event meaning appears unambiguous); it does not identify biomarker columns or classify Type 4 diagnostic or Type 7 meta-analysis studies.

### Consequences
- **Status**: Accepted & Verified.
- **Robustness**: Excel support is limited to modern .xlsx files (legacy .xls files require conversion before loading); supported tabular formats include .csv, .tsv, .xlsx, and .parquet without manual pre-conversion.
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
2. Expand `medstat.reporting.checklists` with comprehensive `get_stard_checklist()` (STARD 2015, 30 essential items) and `get_prisma_checklist()` (PRISMA 2020, 27 essential items), accessible directly via `medstat report --checklist [stard|prisma]`.

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
5. **Data Cleaning & Loader**: Exclude binary/low-cardinality columns (distinct non-null values $\le 2$) from outlier processing and record per-column outlier counts when `--outlier-action flag` is passed. Support multi-encoding fallback (`utf-8`, `utf-8-sig`, `cp1252`, `latin1`) in universal data loader.
6. **Sample Size**: Require explicit parameter inputs (`--p1`/`--p2` for proportions, `--hazard-ratio` for survival, `--r` for correlation) and reject silent default fallbacks.

### Consequences
- **Status**: Accepted & Verified.
- **Statistical Rigor**: Eliminates false confidence interval coverage in weighted kappa, misleading spline basis ORs, and decision curve artifacts from unoriented raw biomarker scores.

[MEMORY_LEARN: Strict non-null standard errors for weighted kappa, contrast matrices for non-linear spline ORs, and probability-oriented DCA curves ensure rigorous biostatistical validity in automated clinical pipelines.]

---

## ADR 15: Adaptive Agent Scripting & Decoupled Execution Architecture (Evolution of medstat-master)

### Context
Users ingesting real-world clinical datasets (e.g., Thai hospital EHR exports with multi-row headers, notes, embedded dashboard summaries, and Thai locale strings) experienced execution errors in Antigravity. The previous orchestrator design suffered from two friction points:
1. Pre-flight PHI checks and external auditor dependencies caused fail-closed halts on clinical files.
2. Rigid canned CLI commands (`uv run medstat profile`, `uv run medstat clean`, etc.) assumed tidy, single-header tables starting at row 1, causing fatal parser errors on complex, multi-table spreadsheets.

### Decision
1. **Eliminate Mandatory PHI Blocker**: Remove the rigid pre-flight PHI auditor requirement from `medstat-master`, allowing agents to process clinical datasets without fail-closed stalls.
2. **Methodological Guidance (ระเบียนวิธี)**: Structure `medstat-master` as a methodological manual defining study designs (Types 1–7), statistical testing heuristics, variable mapping, and publication standards.
3. **Adaptive Python Scripting (ไม่ยึดติดกับสคริปต์สำเร็จรูป)**: Empower the agent to inspect the raw file layout, isolate the analytic cohort from embedded dashboard tables, and write/adapt customized Python scripts (`scratch/analyze.py`) using scientific libraries (`pandas`, `numpy`, `scipy.stats`, `statsmodels`, `sklearn`, `lifelines`) or `medstat` modules.
4. **Preserve Statistical Invariants**: Maintain strict numeric 0/1 outcome encoding, sample retention flow tracking ($N_{\text{initial}} \to N_{\text{excluded}} \to N_{\text{analyzed}}$), Wilson score CIs, DeLong variance, and NEJM/JAMA table formatting.

### Consequences
- **Status**: Accepted & Verified.
- **Robustness**: Non-standard clinical spreadsheets with metadata rows, Thai categories, and side-by-side summary blocks can now be ingested and analyzed dynamically without parser crashes.
- **Cross-Platform Uniformity**: Validated across all 4 multi-agent mirror platforms (`.agent/`, `.agents/`, `.claude/`, `.cursor/`) with zero documentation drift (154/154 unit tests passing).

[MEMORY_LEARN: Replacing rigid canned CLI execution with adaptive agent-driven Python scripting grounded in biostatistical methodology enables robust processing of messy real-world clinical spreadsheets while preserving clinical governance invariants.]

---

## ADR 16: Suite-Wide Expansion of Adaptive Python Scripting Protocol

### Context
Following ADR 15, `medstat-master` proved effective at handling complex, non-standard clinical datasets by empowering agents to adapt Python scripts from prototypes. However, specialized downstream skills (`medstat-clean`, `medstat-models`, `medstat-causal-meta`, `medstat-diagnostic`, `medstat-report`) still relied solely on rigid canned CLI examples. When agents operated in modular subtasks, they lacked explicit prototypes and references to core modules (`src/medstat/`), causing potential regression to naive data assumptions.

### Decision
1. **Universal Adaptive Protocol**: Expand the Adaptive Python Scripting Protocol to all 5 specialized skills:
   - `medstat-clean`: Prototype script for layout isolation, Little's MCAR, explicit imputation justification, Tukey IQR fences, and sample retention flow ($N_{\text{initial}} \to N_{\text{excluded}} \to N_{\text{analyzed}}$) referencing `src/medstat/data/`.
   - `medstat-models`: Prototype script for Table 1 with SMDs, multivariable logistic/GLM, Firth penalization, Cox PH, RCS splines, and VanderWeele E-values referencing `src/medstat/models/` and `src/medstat/reporting/`.
   - `medstat-causal-meta`: Prototype script for 1:1 nearest-neighbor PSM matching with logit caliper ($0.2 \times \text{SD}$), Austin 2009 balance check ($|\text{SMD}| < 0.10$), and Bland-Altman LoA referencing `src/medstat/causal/` and `src/medstat/agreement/`.
   - `medstat-diagnostic`: Prototype script for 2x2 contingency matrices with Wilson score 95% CIs, empirical ROC/AUC with DeLong variance, and Vickers Decision Curve Analysis (DCA) Net Benefit referencing `src/medstat/diagnostic/`.
   - `medstat-report`: Prototype script for rendering 3-horizontal-rule publication HTML tables (NEJM/JAMA), synthesizing automated Methods & Results narratives, and compiling STROBE/CONSORT/TRIPOD audits referencing `src/medstat/reporting/`.
2. **Mandatory Prototype Reference**: Agents are directed to review the provided prototype scripts or core `src/medstat/` implementations before executing scripts on raw clinical data:
   `[1. สำรวจโครงสร้างข้อมูลจริง] ──▶ [2. ดูสคริปต์ต้นแบบเพื่อยึดหลักชีวสถิติ] ──▶ [3. ปรับโค้ดให้เข้ากับข้อมูลและรัน]`
3. **Multi-Agent Mirror Parity**: Maintain exact byte-identical synchronization across canonical `skills/` and all 4 platform mirrors (`.agent/`, `.agents/`, `.claude/`, `.cursor/`), validated continuously via `test_skill_docs_drift.py`.

### Consequences
- **Status**: Accepted & Verified.
- **Biostatistical Rigor**: Agents can now flexibly analyze messy clinical data across any specific domain skill while strictly adhering to biostatistical ground truths.
- **Verification**: Zero documentation drift across 24 mirror files, 350/350 tests passing in full test suite.

[MEMORY_LEARN: Expanding adaptive prototype scripting across all domain skills provides agents with end-to-end flexibility for raw clinical data while enforcing biostatistical ground truths from core modules.]

---

## ADR 17: Autonomous Triangulation Decision Engine & Anti-Hallucination Governance Suite

### Context
While ADR 15 and ADR 16 provided agents with adaptive Python scripting flexibility to handle non-standard spreadsheets, testing revealed that without structured arbitrating logic, AI agents risk statistical hallucinations:
1. Mismatch between research proposal and raw data (e.g. attempting survival analysis when only binary vital status is available without follow-up duration, leading agents to fabricate synthetic time columns or loop on failed model fits).
2. Silent statistical assumptions (e.g. performing listwise deletion without MCAR testing, or reporting mean ± SD on highly skewed biomarker data without normality audits).
3. Inverted clinical concordance (e.g. evaluating low-is-abnormal biomarkers like eGFR or Platelets with Score >= Cutoff, causing inverted ROC curves with AUC < 0.50).
4. Sparse-data over-parameterization (e.g. fitting multivariable logistic regression with EPV < 10, producing quasi-complete separation and astronomical odds ratios).
5. Unsubstantiated clinical claims (e.g. concluding "treatments are equivalent" from p > 0.05, or confusing Pearson correlation with measurement agreement).

### Decision
1. **3-Pillar Triangulation Decision Engine**: Formally establish the arbitration protocol in `medstat-master` and references (`study-design-decision-tree.md`), requiring agents to triangulate:
   - **Pillar 1 (Proposal)**: PICO/PECO, target estimand, and primary research archetype (Types 1–7).
   - **Pillar 2 (Clinical Principles)**: Biological mechanisms, confounding by indication, non-linear thresholds, and directionality.
   - **Pillar 3 (Raw Data Reality)**: Data geometry, sample size $N$, events count, outcome formats, EPV, and missingness patterns.
2. **The 5 Anti-Hallucination Stop Gates**:
   - *Gate 1 (Contradiction Resolution)*: Halt when a time-to-event proposal lacks event or follow-up times. Use logistic regression only for a prespecified fixed-horizon outcome with complete ascertainment.
   - *Gate 2 (Silent Assumption Barrier)*: Zero silent listwise deletion; inspect distributions and model assumptions, then select summaries and tests based on the prespecified estimand, study design, and outcome scale.
   - *Gate 3 (Directionality Gate)*: Strict 0/1 numeric encoding; prespecify score direction from clinical meaning. If AUC is below 0.50, verify event mapping and score direction before any inversion.
   - *Gate 4 (Sparse Data & EPV Gate)*: Mandate calculation of diagnostic $\text{EPV} = \frac{\min(N_{\text{events}}, N_{\text{non-events}})}{P_{\text{parameters}}}$ (or failure events / parameters in Cox); treat EPV as a diagnostic under study-prespecified sparse-data criteria rather than an unconditional trigger, and enforce Firth penalized likelihood or variable reduction when quasi-complete separation, sparse-data bias, or estimation instability occurs.
   - *Gate 5 (Clinical Interpretation Guardrails)*: Report $P > 0.05$ as "insufficient evidence to reject the null hypothesis" (never "no difference"); report OR with incidence warning if $> 10\%$; strictly reject Pearson correlation for rater/device agreement.
3. **Domain Skill Hardening**:
   - `medstat-clean`: Strict ban on imputing primary outcome variables; physiological plausibility protection against naive Tukey fence outlier deletion.
   - `medstat-models`: Explicit EPV calculation and Firth fallback in master prototype script.
   - `medstat-diagnostic`: Automated directionality sanity check and anti-p-hacking cutpoint guidance.
   - `medstat-causal-meta`: Strict confounder selection (baseline only; no post-treatment mediators/colliders); agreement vs association invariant.
   - `medstat-report`: Absence of evidence reporting rule, OR vs RR warning, and uncertainty-first ICC interval reporting.
4. **Multi-Agent Mirror Synchronization**: Propagate byte-identical updates across canonical `skills/` and all 4 platform mirrors (`.agent/`, `.agents/`, `.claude/`, `.cursor/`).

### Consequences
- **Status**: Accepted & Verified.
- **Decision Rigor**: Agents make principled, autonomous, and decisive methodology choices anchored in clinical biostatistical ground truths.
- **Anti-Hallucination Defense**: Eliminates synthetic column fabrication, inverted ROC curves, separation artifacts, and misleading clinical claims.
- **Verification**: Zero documentation drift across all platform mirrors; 100% passing test suite.

[MEMORY_LEARN: 3-pillar triangulation (Proposal, Clinical Principles, Raw Data) paired with 5 deterministic anti-hallucination gates provides coding agents with decisive statistical judgment while preventing model mismatch and spurious claims.]

---

## ADR 18: Calibration Skill Documentation Parity (Resolving Code-Instruction Drift)

### Context
Audit of the restructured skill-set (2026-10-02) revealed that model calibration (`src/medstat/diagnostic/calibration.py`, 275 lines) was fully implemented — Brier score (with scaled Brier), calibration slope & intercept via logistic recalibration, Integrated Calibration Index (ICI / E50 / E90 / Emax per Austin & Steyerberg 2019), Hosmer-Lemeshow goodness-of-fit test, and Plotly calibration plot — and integrated into the CLI (`medstat diag --calibration`), polymorphic report tables, and narrative synthesis. However, none of the 6 skill instruction files referenced calibration capabilities, creating a documentation-code drift where agents could not discover or leverage the existing functionality.

### Decision
1. Document calibration as **Step 5** in `medstat-diagnostic` with prototype script referencing `src/medstat/diagnostic/calibration.py` functions.
2. Add **Core Rule 6** (Discrimination ≠ Calibration) to `medstat-diagnostic` as a positive anti-hallucination directive.
3. Add **Governance Rule 8** (Calibration Mandatory for Prediction Models) to `medstat-master` for TRIPOD-compliant prediction validation.
4. Cross-reference calibration in `medstat-models` Step 3 (Model Diagnostics) and `medstat-report` Core Rules.
5. Record calibration domain terms (`brier_score`, `calibration_slope`, `ici`, `hosmer_lemeshow`) in `CONTEXT.md`.
6. Record `calibration.py` module seam in `ARCHITECTURE.md`.

### Consequences
- **Status**: Accepted & Verified.
- **Zero New Code**: All changes are documentation-only; the underlying implementation and tests were already complete and passing.
- **Agent Discoverability**: Agents can now autonomously invoke calibration assessment for prediction models via skill instructions, prototype scripts, and CLI references.
- **TRIPOD Compliance**: Prediction model validation now includes discrimination (AUC) and calibration (Brier, slope, ICI) as a documented skill requirement.

[MEMORY_LEARN: Code-instruction drift — fully implemented features invisible to agents because skill documentation was never updated — is a systematic risk in adaptive scripting architectures. Audit skill instructions against actual CLI/module capabilities after every implementation sprint.]



---

## ADR 19: Ordinal Outcome Support via Proportional Odds Model and Brant Test

### Context
Clinical functional outcomes (e.g., Modified Rankin Scale [mRS 0–6] in stroke, Glasgow Coma Scale [GCS 3–15] in trauma, NYHA functional class I–IV in cardiology) have natural ordered gradations. Collapsing these endpoints into arbitrary binary thresholds (e.g., mRS 0–2 vs 3–6) loses statistical power and clinical nuance, while treating ordinal categories as continuous numbers in OLS regression violates distributional assumptions. Furthermore, fitting proportional odds without testing the parallel slopes assumption risks biased inference.

### Decision
1. Implement `src/medstat/models/ordinal.py` providing `fit_proportional_odds` using `statsmodels.miscmodels.ordinal_model.OrderedModel` with cumulative logit link.
2. Implement closed-form Brant's Wald test (`test_proportional_odds`, Brant 1990) computing both omnibus and per-variable test statistics for the parallel slopes assumption without external GPL packages.
3. Provide unconstrained multinomial logistic regression (`fit_multinomial_logistic`) as a fallback when proportional odds is violated.
4. Add Anti-Hallucination Gate 6 (Ordinal Scale Integrity Gate) and Study Design Type 2b to `medstat-master`.
5. Integrate `--type ordinal` and `--po-test` into `medstat model` CLI.

### Consequences
- **Status**: Accepted & Verified.
- **Statistical Validity**: Preserves ordinal clinical gradations with valid cumulative Odds Ratios ($\exp(\beta)$) and Wald confidence intervals.
- **Assumption Verification**: Automatically evaluates parallel slopes; guards against inappropriate linear or collapsed binary modeling.
- **Zero New Dependencies**: Leverages existing `statsmodels ≥ 0.14.0`.

[MEMORY_LEARN: Modeling multi-level ordinal clinical endpoints (mRS, GCS, NYHA) via cumulative logit with analytical Brant parallel-slopes testing preserves clinical gradation while preventing distributional violations.]

---

## ADR 20: Multilevel & Clustered Data Support via GEE and MixedLM

### Context
In multi-center clinical trials, health registry networks, and community hospital clusters (รพช.), patients within the same center share unmeasured institutional, geographic, or clinical practice characteristics. Standard GLMs assuming independent observations underestimate standard errors, inflate Type I error rates, and produce spuriously narrow confidence intervals.

### Decision
1. Implement `src/medstat/models/multilevel.py` providing:
   - `calculate_design_effect`: Computes cluster Intraclass Correlation ($\text{ICC}_{\text{cluster}}$), Design Effect ($\text{DEFF} = 1 + (\bar{m}-1)\text{ICC}$), and Effective Sample Size ($N_{\text{eff}} = N / \text{DEFF}$) via ANOVA variance decomposition.
   - `fit_gee`: Population-averaged Generalized Estimating Equations using `statsmodels.genmod.generalized_estimating_equations.GEE` with robust (sandwich) standard errors and exchangeable/independent/AR(1) correlation structures.
   - `fit_random_intercept`: Subject-specific linear mixed-effects model using `statsmodels.formula.api.mixedlm`.
2. Add Anti-Hallucination Gate 7 (Clustering & Independence Gate) and Study Design Type 8 to `medstat-master`.
3. Integrate `--type gee`, `--type mixed`, `--cluster <col>`, and `--corr-structure` into `medstat model` CLI, and `--cluster` into `medstat profile`.

### Consequences
- **Status**: Accepted & Verified.
- **Inference Rigor**: Robust sandwich standard errors prevent spurious statistical significance in clustered clinical data.
- **Sample Size Transparency**: Automatically reports Design Effect and effective sample size alongside nominal $N$.
- **Zero New Dependencies**: Implemented using existing `statsmodels ≥ 0.14.0`.

[MEMORY_LEARN: Multi-center clinical clustering requires variance adjustment; reporting Design Effect (DEFF) alongside population-averaged GEE robust standard errors guarantees valid inference under nested patient structures.]

---

## ADR 21: CodeRabbit PR#5 Quality & Clinical Biostatistics Remediation

### Context
Automated code and clinical biostatistics review by CodeRabbit AI on PR #5 identified 36 findings (23 Major, 12 Minor, 1 Nitpick). Key concerns spanned:
1. `src/medstat/models/multilevel.py`: `fit_random_intercept` used formula strings (`smf.mixedlm`) vulnerable to unquoted special characters/spaces and dummy syntax; missing covariate values were not explicitly detected before fitting GEE or MixedLM.
2. `src/medstat/cli/main.py`: GEE binomial family did not enforce strict numeric `{0, 1}` outcome validation prior to modeling; Egger's test count check checked raw row count instead of distinct study count (`nunique() >= 10`), risking false validity on multi-effect studies; conflicting effect measure CLI arguments were not rejected.
3. `src/medstat/causal/balance.py`: `calculate_smd` returned `0.0` when pooled SD was 0 even if group means differed, masking infinite/undefined clinical imbalance.
4. Public API exports: Functions required by reporting pipelines (`validate_gold_standard`, `extract_primary_effect`) were either missing or not exported in public package namespaces.
5. Unit tests: PR#5 verification tests used local shadow copies/mocks rather than testing true package imports.
6. Skill documentation: Canonical skill instructions contained small statistical and syntax gaps (e.g. missing `import numpy as np` in reporting, lack of expected cell count warnings, ambiguous fallback for survival time horizons, unverified outcome dropping).

### Decision
1. **Multilevel Matrix Formulation**: Refactored `fit_random_intercept` to pass direct design matrices (`sm.MixedLM(y_clean, exog.astype(float), groups=c_clean)`) and properly honor `add_constant`. Added explicit missingness checks raising `ValueError("Missing values detected in covariates X...")` across both GEE and MixedLM.
2. **CLI Guardrails**: Enforced strict numeric `{0, 1}` outcome verification for GEE binomial models. Enforced distinct study threshold (`df[study_col].dropna().nunique() >= 10`) for Egger's test. Added cross-column validation for meta-analysis effect measures.
3. **SMD Boundary Correctness**: Updated `calculate_smd` to return `np.nan` if pooled SD is 0 and means differ, returning `0.0` strictly when means are identical.
4. **Export Public Contract Utilities**: Implemented and exported `validate_gold_standard` in `medstat.diagnostic` and `extract_primary_effect` in `medstat.models`.
5. **Decoupled Unit Testing**: Refactored `tests/unit/test_pr5_coderabbit_fixes.py` to import and directly test production package code; added direct CLI tests for binary validation and duplicate study rejection.
6. **Skills Suite Hardening & Mirror Parity**: Hardened canonical skill instructions (`skills/`) for cell counts, survival horizons, Firth separation fallbacks, and complete-case protocol flags, and synchronized byte-for-byte across `.agent/`, `.agents/`, `.claude/`, and `.cursor/` mirrors.

### Consequences
- **Status**: Accepted & Verified.
- **Statistical Safety**: Prevents silent masking of infinite imbalance in balance metrics and prohibits invalid outcome types in GEE.
- **Robust Execution**: Formula parsing crashes eliminated in MixedLM with arbitrary column names.
- **100% Mirror Parity**: Verified by `tests/unit/test_skill_docs_drift.py` across all 5 skill directories.
- **Test Integrity**: Full suite passing (388/388 tests) with real production imports.

[MEMORY_LEARN: Zero-variance in covariate balance with differing group means represents an undefined/infinite imbalance that must yield NaN rather than 0.0 to prevent masking severe clinical cohort disparities.]

---

## ADR 22: CodeRabbit PR#5 Second-Pass Remediation & Security Hardening

### Context
Following initial remediation in commit `5c4f604`, CodeRabbit automated review completed a full re-review (`5397817652`) narrowing findings down to 19 items (9 Major, 9 Minor, 1 Nitpick). Key concerns addressed:
1. `src/medstat/models/ordinal.py`: `X_mat[y_series.index]` used label indexing which silently failed or re-indexed incorrectly when `X_mat` or `y_series` had non-standard, custom, or reset indices.
2. `src/medstat/causal/balance.py` & `src/medstat/cli/main.py`: `check_balance` produced non-finite float `np.nan` values for undefined SMDs, generating non-standard JSON (`NaN`) instead of valid JSON `null`.
3. `src/medstat/models/multilevel.py` & `main.py`: `calculate_design_effect` and mixed model CLI lacked explicit numeric outcome validation, allowing non-numeric outcomes to cause internal crashes during ANOVA decomposition.
4. `src/medstat/cli/main.py`: Egger's test allowed duplicate study IDs (when multiple rows had identical study names), violating linear regression observational independence.
5. `src/medstat/cli/main.py`: Ordinal cumulative odds E-value calculation lacked explicit documentation regarding common-outcome approximation ($RR \approx \sqrt{OR}$).
6. Skills Suite Security & Statistical Hardening: Prototype Python scripts lacked sandbox execution boundaries, risk-stratified zero-cell checks, strict binary outcome validation, and deterministic matching tie-breaking.

### Decision
1. **Positional Boolean Masking**: In `src/medstat/models/ordinal.py`, replaced label-based `y_series.index` with explicit positional boolean mask `valid_mask = y_raw.notna().to_numpy()` in both `fit_proportional_odds` and `fit_multinomial_logistic`.
2. **JSON Null Serialization for Balance SMDs**: In `src/medstat/causal/balance.py`, updated `check_balance` to convert non-finite SMDs to `None`. In `main.py` CLI causal command, applied `_clean_smd` helper so that `json.dump` outputs compliant `null`.
3. **Multilevel Numeric Guards**: Added explicit `is_numeric_dtype(y_raw)` validation in `calculate_design_effect` and the mixed model CLI branch.
4. **Egger Duplicate Study ID Rejection**: Added pre-flight `df[study_col].is_unique` check in `main.py` CLI before fitting Egger's regression, raising `click.BadParameter` if duplicate study IDs are present.
5. **Ordinal E-Value Assumption Annotation**: Set `rare_outcome=False` for ordinal CLI models and attached an explicit `assumption_note` documenting the square-root transformation.
6. **Skills Suite Sandboxing & Parity**: Embedded subprocess/sandbox security guidance across all 6 skills, enforced deterministic control matching tie-breaks, restricted zero-cell checks strictly to discrete variables, and synchronized all changes across all 4 mirrors (`.agent/`, `.agents/`, `.claude/`, `.cursor/`), confirmed by `test_skill_docs_drift.py`.

### Consequences
- **Status**: Accepted & Verified.
- **Robust Indexing**: Positional masking completely decouples model fitting from pandas DataFrame index state.
- **Strict JSON Standard**: Output JSON contains strictly valid `null` values for undefined SMDs.
- **Statistical Independence**: Prohibits invalid funnel plot asymmetry tests on multi-effect/clustered study records.
- **100% Test Pass**: 391/391 tests passing with zero lint or format warnings.

[MEMORY_LEARN: In pandas/numpy hybrid modeling pipelines, positional boolean masking (via `.to_numpy()`) prevents silent data corruption and slicing errors that occur with label-based Index alignment.]

---

## ADR 23: Protocol Justification, Pre-Post Balance Normalization & Clinical Model Family Hardening

### Context
Final CodeRabbit automated review items identified refinements in:
1. `src/medstat/models/ordinal.py`: `test_proportional_odds` still used `y_series.index` to slice `X_mat` on non-default indices.
2. `src/medstat/causal/balance.py`: `compare_pre_post_balance` lacked normalization for undefined SMDs (which may be `None` from `check_balance`), causing potential `TypeError` when calling `float(None)`.
3. `src/medstat/cli/main.py`: GEE Gaussian family lacked explicit rejection of non-numeric outcomes, and ordinal models allowed unordered text categories without establishing explicit ordinal rank.
4. Skills Suite Scaffolding:
   - `medstat-clean`: Outcome dropping in the scaffold defaulted to `True`, which could encourage unverified outcome exclusion.
   - `medstat-master`: Table 1 prototypes omitted group-specific analyzed and missing denominators; completion checklist stated automatic Firth for EPV < 10 rather than treating it as a risk screening alert.
   - `medstat-models`: Prototype script routed directly to Firth on EPV < 10 rather than issuing an alert and routing on zero cells or MLE failure.

### Decision
1. **Positional Masking & Length Guard in Brant Test**: Updated `test_proportional_odds` to build a positional boolean mask `valid_mask = y_raw.notna().to_numpy()` and assert `len(X) == len(y_raw)`.
2. **SMD Normalization Helper**: Added `_normalize_smd` in `src/medstat/causal/balance.py` coercing `None`, `np.nan`, or invalid values to `np.nan`, preserving boolean post-balance flags.
3. **CLI Outcome Type Enforcement**: In `src/medstat/cli/main.py`, rejected non-numeric outcomes for GEE Gaussian models, and required numeric or ordered pandas Categoricals for ordinal models (`--outcome`).
4. **Skills Suite Hardening**:
   - `medstat-clean`: Defaulted `protocol_permits_outcome_exclusion = False` and `protocol_rationale = None`.
   - `medstat-master`: Added group-specific analyzed/missing counts to continuous and categorical Table 1 prototypes; updated checklist item.
   - `medstat-models`: Changed EPV < 10 to a diagnostic warning, routing to Firth on zero cells or MLE failure.
5. **Mirror Parity**: Synchronized byte-for-byte across `.agent/`, `.agents/`, `.claude/`, and `.cursor/`, verified by `test_skill_docs_drift.py`.

### Consequences
- **Status**: Accepted & Verified.
- **Protocol Safety**: Eliminates accidental outcome exclusion without explicit PI/SAP protocol documentation.
- **Complete Test Coverage**: 394/394 tests passing with zero lint warnings.

[MEMORY_LEARN: Clinical baseline tables must report group-specific analyzed (n) and missing counts alongside summary statistics to prevent misinterpreting attrition as true balance.]

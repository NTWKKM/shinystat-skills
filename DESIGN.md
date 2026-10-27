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
5. **Decoupled Unit Testing**: Refactored `tests/unit/test_regression_guardrails_pr5.py` to import and directly test production package code; added direct CLI tests for binary validation and duplicate study rejection.
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

---

## ADR 24: GEE Time Validation, PSM Penalization Nonzero Alpha, Calibration Provenance & Outcome Recoding Parity

### Context
PR #5 latest review findings from CodeRabbit identified several clinical and statistical safety improvements:
1. `src/medstat/models/multilevel.py`: `fit_gee` allowed `struct_key="autoregressive"` without `time` to proceed through data preprocessing before encountering an error, rather than raising `ValueError` during argument validation.
2. `skills/medstat-causal-meta` & `src/medstat/causal/psm.py`: `fit_regularized(disp=False)` defaulted to `alpha=0.0` in statsmodels, resulting in unpenalized estimates during separation fallback.
3. `skills/medstat-report`: Calibration narrative generators allowed apparent calibration slope reporting (which is 1.0 by definition in development cohorts) and lacked validation requiring the complete metric quartet (`brier`, `slope`, `intercept`, `ici`) for external cohorts, along with non-finite metric validation. Duplicated JAMA and APA p-value branches existed.
4. `skills/medstat-clean`: Prototype outcome validation did not recode mapped text labels to integer codes nor cast validated codes to integer dtype before export.
5. `skills/medstat-diagnostic`: DCA fallback selected raw `test_score` when `predicted_risk` was missing, which violates the decision-analytic requirement for calibrated probabilities in $[0, 1]$.
6. `skills/medstat-models`: Table 1 group splitting inferred reference levels via naive lexical sort (`key=str`), which risks mislabeling active intervention arms as reference, and output generic `Group 0` and `Group 1` labels.

### Decision
1. **GEE Autoregressive Pre-flight Check**: Added immediate `ValueError` raising in `fit_gee` argument validation when `cov_struct == "autoregressive"` and `time is None`.
2. **Justified PSM Regularization Penalty**: Updated regularized propensity score fits to enforce per-column L1 alpha policy where covariates use sample-size-scaled penalty `alpha = 1.0 * len(y)` and the intercept identified by its all-ones values and position is unpenalized (`alpha = 0.0`) in both `src/medstat/causal/psm.py` and skill mirrors, retaining the fallback to Firth penalization.
3. **TRIPOD Calibration Reporting Compliance**: In `generate_methods_narrative` and `validate_calibration_metrics`, validated that all calibration metric values are finite numeric, rejected boolean values before numeric acceptance, supported an explicit `tautological_slope` flag to omit slope and intercept for in-sample apparent derivation models while allowing eligible apparent estimates to be reported, required the full metric quartet for external validation, and merged duplicated JAMA/APA p-value formatting branches.
4. **Clinical Outcome Recoding & Integer Typing**: Updated `medstat-clean` outcome validation to recode mapped string labels to protocol-defined integer codes and cast to integer dtype (`df_clean['outcome'].astype(int)`) before export across all 5 skill mirrors.
5. **DCA Calibrated Risk Guard**: Updated `medstat-diagnostic` DCA to require `predicted_risk` and skip analysis with an informative message if absent, eliminating invalid raw biomarker score calculations.
6. **Explicit Reference Group & Actual Treatment Labels**: Updated Table 1 group splitting in `medstat-models` to accept an optional `ref_level`, avoid lexical sorting, and report actual treatment group labels alongside standard summaries.
7. **Mirror Parity**: Synchronized all changes byte-for-byte across canonical `skills/` and mirrors (`.agent/`, `.agents/`, `.claude/`, `.cursor/`), confirmed by `test_skill_docs_drift.py`.

### Consequences
- **Status**: Accepted & Verified.
- **Statistical Safety**: Uncalibrated raw test scores can no longer corrupt DCA Net Benefit calculations; propensity scores cannot proceed unpenalized under separation.
- **Reporting Fidelity**: Narrative generation strictly follows TRIPOD guidelines regarding apparent vs external calibration.
- **Full Test Suite Pass**: 416/416 tests passing, 0 lint errors (`ruff check`), 100% formatted (`ruff format`).

[MEMORY_LEARN: Decision Curve Analysis (DCA) strictly requires calibrated probabilities in [0, 1]; passing uncalibrated raw continuous biomarkers corrupts net benefit calculations due to invalid threshold exchange rates.]

---

## ADR 25: Unified Shinystat Skill Architecture & Deterministic Grilling Gate

### Context
Maintaining multiple fragmented atomic skills (`medstat-clean`, `medstat-models`, `medstat-diagnostic`, `medstat-causal-meta`, `medstat-report`) caused cognitive fragmentation, high context load, and brittle inter-skill transitions. Furthermore, rigid execution models led agents to force canned commands onto messy real-world clinical datasets rather than exercising biostatistical judgment.

### Decision
1. **Unified Autonomous Skill (`shinystat`)**: Consolidate all domain skills into a single authoritative skill `skills/shinystat/SKILL.md` following `writing-for-agents` principles (concise core instructions, positive framing, checkable completion criteria).
2. **Hybrid Progressive Disclosure**: Disclose detailed clinical study archetypes, model selection tables, and decision trees in `skills/shinystat/references/decision-heuristics.md` behind a context pointer, keeping the main `SKILL.md` lean (~70 lines).
3. **Deterministic Grilling Gate**: Embed mandatory stop-and-grill rules (`❓ Q1 ... ➡️ Recommended`) when outcome/censoring mapping is ambiguous, research designs mismatch data geometry, missingness lacks clinical justification, or extreme sparse-data/separation occurs.
4. **Preserve Computational Engine**: Maintain `medstat-core` Python package and Click CLI unchanged (442 passing tests).
5. **Multi-Agent Mirror Parity & Automated Drift Defense**: Synchronize `shinystat` byte-for-byte across `.agent/`, `.agents/`, `.claude/`, `.cursor/`, and canonical `skills/`, verified continuously via `tests/unit/test_shinystat_skill.py`.

### Consequences
- **Status**: Accepted & Verified.
- **Context Efficiency**: Drastically reduced token footprint while amplifying autonomous statistical decision-making.
- **Anti-Hallucination Rigor**: Enforces hard-stop interviews before code execution whenever ambiguity is encountered.
- **Full Parity**: 100% identical mirror synchronization and 442/442 unit and E2E tests passing.

[MEMORY_LEARN: Consolidating modular skills into a single autonomous decision skill with progressive disclosure and deterministic grilling gates maximizes agent flexibility while eliminating silent assumptions and context sprawl.]

## ADR 26: Decoupled Shinystat Cloud Skill Package for Claude Web Sandbox

### Context
In cloud execution environments such as Claude Web (claude.ai Analysis Tool) or hosted Jupyter notebooks, the proprietary `medstat-core` Python package is not pre-installed. While standard data science libraries (`pandas`, `numpy`, `scipy`, `statsmodels`, `lifelines`, `scikit-learn`) are available, advanced biostatistical routines (Firth penalized regression, Austin & Steyerberg 2019 ICI, DeLong analytical variance, Shrout & Fleiss 1979 two-way ANOVA ICC, Bland-Altman 1999 LoA SE, Austin 2009 SMD, Little's MCAR test) risk execution errors if the agent attempts to import unavailable `medstat` modules.
Crucially, coupling cloud-specific recipes into the canonical `skills/shinystat` skill compromises the clean architectural design of the local agent workflows.

### Decision
1. **Pristine Canonical Skill**: Retain `skills/shinystat/` and all 4 platform mirrors (`.agent/`, `.agents/`, `.claude/`, `.cursor/`) in their clean, unlinked original state without cloud recipe references.
2. **Dedicated Cloud Skill (`packaging/cloud/shinystat-cloud/`)**: Package a dedicated autonomous skill specifically tailored for Claude Web. Embeds cloud sandbox execution guidance and links directly to `references/python-recipes.md`.
3. **Automated Cloud Packaging Script**: Provide `scripts/package-cloud-skill.sh` to build `~/Desktop/shinystat-cloud.zip` on demand without polluting the canonical skills tree.
4. **Desktop Artifacts**: Deliver both clean original `~/Desktop/shinystat.zip` and dedicated cloud `~/Desktop/shinystat-cloud.zip`.

### Consequences
- **Status**: Accepted & Verified.
- **Clean Architecture**: Canonical skill remains pristine while Claude Web receives a dedicated, self-contained cloud bundle.
- **Parity**: 442/442 unit and E2E tests passing.

[MEMORY_LEARN: Decoupling cloud-specific execution recipes into a dedicated export package keeps canonical repository skills pristine while empowering cloud-sandboxed agents with self-contained algorithm manuals.]

---

## ADR 27: Cloud Sandbox Biostatistical Hardening, Statistical Parity & Workflow Expansion

### Context
Code review and empirical audit of `packaging/cloud/shinystat-cloud/` revealed severe numerical inaccuracies, statistical biases, and usability blockers in the cloud sandbox recipes:
1. Little's MCAR test grouped data on continuous variable values instead of missingness patterns, dropping rows with missing values and inflating $df$ into hundreds.
2. Typecasting (`astype(int)`) occurred before dropping NaNs in DeLong and Calibration, silently turning missing patient outcomes into negative controls (`0`).
3. Categorical SMD returned `0.00` when $sd_{bin} = 0$ even under 100% disparity ($p_0=0, p_1=1$), masking extreme imbalance.
4. ICC accepted `alpha` but omitted confidence intervals entirely, violating SKILL.md completion criteria.
5. Firth logistic regression provided only Wald CIs, exploding to $>10,000$ on monotone separation, lacked step-halving, and had docstring formula typos.
6. PSM did not return `pair_id` (blocking paired/conditional analyses) and omitted post-match SMD balance diagnostics.
7. VanderWeele & Ding E-value applied $\sqrt{OR}$ unconditionally, halving confounder strength for rare clinical outcomes.
8. `decision-heuristics.md` recommended phantom `medstat.*` modules that do not exist in cloud sandboxes, and EPV < 10 was treated as an outdated hard stop.
9. Common biostatistical workflows (multivariable logistic table, Kaplan-Meier / Cox PH, MICE imputation, NEJM/JAMA table export) were missing, while Grilling Gate halted unpragmatically on any routine lab missingness.

### Decision
1. **Mathematical & Algorithmic Hardening**:
   - Replaced Little's MCAR with the true pattern-grouped EM ML algorithm (Little 1988), producing valid $df \ll N$.
   - Enforced NaN filtering on raw float inputs prior to integer casting across DeLong, calibration, and DCA.
   - Fixed categorical SMD to return `np.nan` on zero-variance disparity when $p_0 \ne p_1$.
   - Implemented exact F-distribution 95% CIs for all 6 Shrout & Fleiss (1979) ICC variants.
   - Added step-halving and Profile Likelihood CIs to Firth logistic regression; fixed formula docstring to $\sum_i h_i (0.5 - \pi_i) x_{ij}$.
   - Enhanced PSM to return `pair_id`, categorical dummy encoding, and pre/post SMD balance tables.
   - Added `rare_outcome` prevalence branching to VanderWeele & Ding E-value.
2. **Cloud Heuristics & Workflow Expansion**:
   - Purged all phantom `medstat.*` references from `decision-heuristics.md`, pointing directly to self-contained recipes.
   - Modernized EPV heuristics per Vittinghoff & McCulloch (2007), van Smeden et al. (2016), and Riley et al. (2019), routing Cox with sparse events to `lifelines` L2 penalization.
   - Added missing core recipes: Multivariable Logistic Table (Crude & Adjusted ORs), Survival Analysis Suite (KM median, Log-Rank, Cox PH, Schoenfeld test), MICE Imputer (protecting primary outcomes), and NEJM/JAMA Publication Table Formatter.
   - Enriched `SKILL.md` frontmatter with rich intent triggers, harmonized the 4 Pillars diagram, and added pragmatic batched defaults to the Grilling Gate (<5% complete-case, 5-40% MICE).
3. **Automated Verification**:
   - Implemented dedicated unit test suite `tests/unit/test_shinystat_cloud.py` (17/17 passing).
   - Rebuilt `~/Desktop/shinystat-cloud.zip` via `package-cloud-skill.sh`.

### Consequences
- **Status**: Accepted & Verified.
- **Biostatistical Validity**: Eliminates silent data corruption, Hauck-Donner explosions, and false zero balance metrics in cloud sandboxes.
- **Workflow Completeness**: Cloud agents can execute the full biostatistical lifecycle (Table 1 ➔ Logistic/Cox ➔ MICE ➔ Diagnostics/Agreement ➔ NEJM HTML tables) autonomously.
- **Full Test Suite Pass**: 459/459 tests passing suite-wide with zero failures.

[MEMORY_LEARN: Cloud code execution sandboxes require mathematically robust, self-contained Python recipes with pre-cast NaN filtering, exact ANOVA F-distribution intervals, and pattern-grouped EM algorithms to eliminate silent data corruption and dependency failures.]

---

## ADR 28: Multi-Imputation Rubin's Pooling & Survival Diagnostics Hardening for Cloud Sandbox

### Context
User re-testing of `shinystat-cloud.zip` (PR #8) identified 4 specific methodological and diagnostic improvements required prior to merge:
1. `impute_mice` returned a single imputed dataset; single imputation fails to account for imputation uncertainty, producing spuriously narrow standard errors in downstream inferential regression.
2. Schoenfeld proportional hazards diagnostics in survival analysis returned only a boolean pass/fail flag without reporting per-covariate test statistics and $p$-values, while Log-Rank was limited to 2 groups.
3. The ASCII architecture diagram header in `SKILL.md` still read "3-Pillar" while the section title was "The 4 Decision Pillars".
4. VanderWeele & Ding (2017) continuous hazard ratio conversion required explicit verification and unit tests for both rare and common outcome branches.

### Decision
1. **Multiple Imputation & Rubin's Pooling**:
   - Implemented `impute_mice_datasets` generating $M \ge 5$ stochastic imputed datasets via `IterativeImputer(estimator=BayesianRidge(), sample_posterior=True)`.
   - Implemented pure-Python `pool_estimates_rubin` applying Rubin's rules (Rubin 1987; Barnard & Rubin 1999) to combine point estimates and variance across $M$ imputations:
     $$\bar{\theta} = \frac{1}{M}\sum_{m=1}^M \hat{\theta}_m, \quad \bar{W} = \frac{1}{M}\sum_{m=1}^M \hat{V}_m, \quad B = \frac{1}{M-1}\sum_{m=1}^M (\hat{\theta}_m - \bar{\theta})^2, \quad T = \bar{W} + \left(1 + \frac{1}{M}\right)B$$
     incorporating Barnard-Rubin finite-sample adjusted degrees of freedom $\nu$, fraction of missing information ($FMI$), and 95% CIs.
   - Retained `impute_mice_single` (and backward-compatible alias `impute_mice`) with explicit docstring warnings that single imputation is strictly reserved for rapid exploratory data health profiling.
2. **Survival Suite Diagnostic Expansion**:
   - Upgraded Log-Rank testing to automatically branch: 2 groups $\to$ `logrank_test`; $>2$ groups $\to$ `multivariate_logrank_test`.
   - Enhanced Schoenfeld diagnostics via `proportional_hazard_test(cph, data, time_transform="rank")` returning per-variable test statistics, $p$-values, and an overall boolean check.
3. **Pillars Harmonization & Decision Trees**:
   - Harmonized diagram title in `SKILL.md` to `The 4 Decision Pillars Triangulation`.
   - Updated missing data heuristics in `SKILL.md` and `decision-heuristics.md` to require $M \ge 5$ stochastic datasets + Rubin's rules for inferential modeling.
4. **Unit Test Expansion & Verification**:
   - Expanded `tests/unit/test_shinystat_cloud.py` with tests for HR E-value branching (rare vs common, protective $HR < 1.0$), multivariate Log-Rank, per-variable Schoenfeld diagnostics, and Rubin's pooling variance inflation.
   - Rebuilt `~/Desktop/shinystat-cloud.zip`.

### Consequences
- **Status**: Accepted & Verified.
- **Inferential Validity**: MICE in cloud sandboxes now properly inflates standard errors to reflect missing data uncertainty, preventing anti-conservative clinical conclusions.
- **Diagnostic Transparency**: Clinicians can pinpoint specific covariates violating proportional hazards in Cox regression.
- **Full Test Suite Pass**: 19/19 cloud unit tests passing, 461/461 suite-wide tests passing with zero regressions.

[MEMORY_LEARN: Multiple imputation in clinical inference strictly requires M>=5 stochastic datasets with Rubin's rules pooling (W_bar + (1+1/M)B); single imputation treats imputed values as known constants and falsely narrows confidence intervals.]

---

## ADR 29: Outcome and Covariate Conditioning in MICE Imputation to Prevent Null Attenuation

### Context
Simulation audits and biostatistical review of `impute_mice_datasets` revealed a severe methodological bias:
1. Passing only `clean[features_to_impute]` into `IterativeImputer` excluded the primary outcome and other analytic covariates from the imputation model. Under this setup, imputed values are conditionally independent of the outcome, attenuating the true regression slope towards the null by ~40–50% (e.g. 0.0589 attenuated to 0.0287).
2. The clinical governance rule "never impute the primary outcome" was misinterpreted as "never include the outcome in the imputation model". According to foundational biostatistical literature (van Buuren, *Flexible Imputation of Missing Data*; Moons et al., 2006), the primary outcome **must** be included as a predictor in imputation models for missing covariates; only its values must remain unmodified.
3. If `features_to_impute` contains only a single variable and no other predictors are supplied, `IterativeImputer` has no features to condition on, degenerating to unconditional mean imputation with between-imputation variance $B = 0$ across all $M$ datasets.

### Decision
1. **Outcome & Covariate Conditioning**:
   - Updated `impute_mice_datasets` and `impute_mice_single` to assemble conditioning columns:
     $$\text{cols} = \text{Unique}(\text{features\_to\_impute} \cup \text{predictors} \cup [\text{outcome\_col}])$$
   - Fits `IterativeImputer` on all `cols`, but copies imputed values strictly back to `features_to_impute` (`clean[features_to_impute] = filled[features_to_impute]`), leaving the primary outcome and any complete covariates unmodified.
2. **Mean Imputation Guard**:
   - Added validation: `if len(cols) < 2: raise ValueError(...)` to block unconditional mean imputation when a single variable is passed with zero predictors and no outcome.
3. **Automated Verification**:
   - Added `test_mice_single_feature_with_outcome_conditioning`: verifies that single-variable imputation conditioned on outcome yields strictly positive between-imputation variance ($B > 0$) and fraction of missing information ($FMI > 0$).
   - Added `test_mice_attenuation_recovery_vs_full_data`: verifies that MICE pooled slope recovers the full-data regression slope without severe attenuation towards the null.
   - Rebuilt `~/Desktop/shinystat-cloud.zip`.

### Consequences
- **Status**: Accepted & Verified.
- **Biostatistical Validity**: Eliminates artificial attenuation towards the null, aligning cloud MICE recipes with international standards (van Buuren; Moons 2006).
- **Test Suite Pass**: 21/21 cloud unit tests passing, 463/463 workspace-wide tests passing.

[MEMORY_LEARN: In MICE imputation, never imputing the primary outcome means never overwriting or estimating missing outcome values; the outcome MUST still be included as a predictor when imputing covariates to prevent severe attenuation towards the null.]

---

## ADR 30: CodeRabbit PR #8 Remediation — Statistical Invariant Hardening, Strict Binary Validation, and Recipe Self-Containment

### Context
Comprehensive code review by CodeRabbit AI on GitHub PR #8 (`feat/cloud-skill-hardening`) against `main` identified 11 statistical, edge-case, and architectural findings across cloud recipes and documentation:
1. **Binary Outcome Silent Conversion**: `astype(int)` in DeLong, Calibration, and DCA allowed non-binary floats (e.g., $0.9 \to 0$, $1.9 \to 1$) and unverified class counts. Survival event indicators allowed arbitrary numbers or missing values.
2. **MICE Clinical Domain Constraints**: `IterativeImputer` with Gaussian `BayesianRidge` lacked bounds checking (`min_value`, `max_value`) for clinical metrics (e.g. vital signs), crashed on non-numeric columns, failed on 100% missing columns, and lacked validation for $m \ge 1$.
3. **Firth Logistic Regression Failure Handling**: Step-halving accepted parameter updates even when penalized log-likelihood deteriorated; convergence only checked parameter steps without verifying the adjusted score gradient; profile likelihood search fallback to Wald CI was silent without telemetry; design matrix collinearity was unverified.
4. **Rubin Pooling Barnard-Rubin Limit**: Setting $df = \infty$ when between-imputation variance $B = 0$ produced spuriously narrow standard normal confidence intervals instead of using the Barnard-Rubin (1999) finite-sample limit $\nu_{\mathrm{BR}} = \nu_{\mathrm{com}}\frac{\nu_{\mathrm{com}}+1}{\nu_{\mathrm{com}}+3}$.
5. **Little's MCAR Edge Cases**: Degenerate missingness structures with $df \le 0$ or completely missing datasets returned false non-significance ($p = 1.0$), and the EM algorithm lacked convergence and iteration telemetry.
6. **PSM Self-Containment & Balance Diagnostics**: SMD calculation relied on assumptions of existing scope instead of self-containment; dummy-encoded categorical confounders were omitted from balance tables; duplicate DataFrame indices corrupted 1:1 nearest-neighbor matching.
7. **Survival Cohort Retention Separation**: KM unadjusted analysis and Cox proportional hazards regression conflated cohorts instead of separating retention audits when covariates had missing values.
8. **Table 1 Column Formatting & SMD Zero Counts**: Header and body column string keys did not match, and categorical SMDs failed when a group had zero observations.
9. **ICC Zero Residual Boundary**: Perfect agreement with zero within-subject or error variance ($MSW = 0$ or $MSE = 0$) resulted in division-by-zero or NaN instead of ICC 1.0 with exact $[1.0, 1.0]$ bounds.
10. **HTML Escaping**: Recipe 13 rendered unescaped HTML strings, posing security risks in cloud web interfaces.
11. **Documentation Alignment**: 4-Pillars diagram numbering, missingness thresholds, and Schoenfeld diagnostic return keys required harmonization between `SKILL.md`, `decision-heuristics.md`, and `python-recipes.md`.

### Decision
1. **Strict Finite Binary Domain & Dimensionality Validation**:
   - Implemented strict $\{0, 1\}$ finite value verification (`np.isin(y, [0, 1]).all()`) and length equality checks across `auc_ci_delong`, `delong_paired_test`, `evaluate_calibration`, `calculate_dca`, `match_propensity_scores`, `fit_logistic_regression_table`, and `fit_survival_analysis_suite`.
   - Bounded probability inputs strictly to $[0, 1]$ in calibration and DCA with explicit apparent-performance disclosures.
2. **MICE Clinical Safeguards**:
   - Enforced $m \ge 1$, input validation blocking non-numeric columns and 100% missing variables, and integrated bounded imputation (`min_value`, `max_value`) to preserve physiological ranges while keeping primary outcomes strictly un-imputed.
3. **Firth Logistic Regression Convergence Hardening**:
   - Added `np.linalg.matrix_rank(X) == p` design matrix full-rank check.
   - Enforced monotonic penalized log-likelihood in step-halving.
   - Required dual convergence: parameter step norm $< tol$ and adjusted score gradient norm $< tol \cdot \max(1, N)$.
   - Added `"ci_method"` telemetry tracking (`"profile"` vs `"wald_fallback"`).
4. **Barnard-Rubin Finite-Sample Limit**:
   - Formulated degrees of freedom under $B = 0$:
     $$\nu_{\mathrm{BR}} = \nu_{\mathrm{com}}\frac{\nu_{\mathrm{com}}+1}{\nu_{\mathrm{com}}+3}, \quad \text{where } \nu_{\mathrm{com}} = n_{\mathrm{obs}} - k_{\mathrm{params}}$$
   - Added validation for non-negative standard errors, matching vector lengths, and safe $W = 0$ boundary handling.
5. **Little's MCAR Degeneracy & Telemetry**:
   - Bounded $df \le 0$ as untestable missingness returning NaN statistics and `converged=False`.
   - Added telemetry fields: `converged`, `n_iterations`, and `n_excluded_rows`.
6. **PSM Self-Containment, Categorical Balance & Re-indexing**:
   - Embedded standalone `calculate_smd` helper directly in Recipe 6.
   - Added `reset_index(drop=True)` to ensure index uniqueness.
   - Expanded balance diagnostics to encompass dummy-encoded categorical indicators alongside continuous covariates.
7. **Cohort Retention Audits & ICC Exactness**:
   - Attached explicit retention audits to logistic regression tables and separated KM vs Cox cohorts in survival analysis (`retention_km` and `retention_cox`).
   - Mapped ICC zero-residual boundaries to exact point 1.0, $[1.0, 1.0]$ CIs, and $F = \infty$.
8. **HTML Escaping & Documentation Harmonization**:
   - Escaped all HTML table components with `html.escape` in Recipe 13.
   - Harmonized 4-Pillars ASCII diagrams and missingness triage rules across `SKILL.md`, `decision-heuristics.md`, and `python-recipes.md`.
9. **CodeRabbit Gap Remediations (Gaps 1–3)**:
   - **MICE Continuous Target Scope**: Restricted `features_to_impute` strictly to continuous variables. Rejected binary/dummy indicators and boolean features with actionable `ValueError`. Disallowed missing categorical predictors while permitting fully observed dummy predictors to condition on without improper Gaussian imputation.
   - **DCA Apparent Performance Disclosure**: Attached explicit disclosure to `df.attrs["apparent_performance_warning"]` and boolean `df.attrs["internally_calibrated"]` whenever input scores fall outside $[0, 1]$ and trigger in-sample logistic recalibration.
   - **Isolated Recipe Execution & Deep Firth Tests**: Enhanced `test_each_recipe_is_self_contained` to call all 13 recipe entry-point functions with valid mini-fixtures in isolated namespaces. Added numerical profile likelihood endpoint checks ($2\Delta \ell^* \approx \chi^2_{1, 0.95}$ within 0.05), optimizer failure handling (`max_iter=1`), and monotonic log-likelihood progression under separation.
10. **Comprehensive Verification**:
   - Rebuilt `/Users/ntwkkm/Desktop/shinystat-cloud.zip`.
   - All 32 cloud tests passing, 474/474 workspace-wide tests passing with zero regressions.
11. **Proper Bayesian Linear Regression MICE Algorithm & Empirical Coverage Recovery**:
   - **Parameter Uncertainty vs Marginal Sampling**: `sklearn.impute.IterativeImputer(estimator=BayesianRidge(), sample_posterior=True)` takes independent marginal predictive draws for each missing observation using predictive variance $x^T \Sigma_\beta x + 1/\alpha$. While this predictive variance incorporates parameter variance $x^T \Sigma_\beta x$, independent row-wise marginal draws do not condition on a single shared draw of the regression vector $\beta^*$ across all missing records in an imputation dataset.
   - **Proper Bayesian Implementation**: Implemented textbook Proper Bayesian Linear Regression MICE (Rubin 1987; Schafer 1997; van Buuren 2018). In each chained cycle, requires $n_{\mathrm{obs}} > p$, full column rank, and non-degenerate residual sum of squares ($\text{RSS} > 1e-12$) to compute the exact unregularized flat-prior OLS posterior: deterministic linear relationships where the target is perfectly predicted are rejected with actionable errors rather than silently flooring variance. Draws residual variance $\sigma^{*2} \sim \text{Inv-}\chi^2(\nu, s^2)$ via $\chi^2$ sampling without tail modifications, draws a shared regression parameter vector $\beta^* \sim \mathcal{N}(\hat{\beta}, \sigma^{*2} (X_{\mathrm{obs}}^T X_{\mathrm{obs}})^{-1})$, and samples missing values $y_{\mathrm{mis}}^* \sim \mathcal{N}(X_{\mathrm{mis}} \beta^*, \sigma^{*2})$. Both MICE entry points require `max_iter` to be a positive integer ($\ge 1$).
   - **Bounded Sampling Model**: Bounded constraints (`min_value`, `max_value`) are drawn using truncated normal distributions (`scipy.stats.truncnorm`) rather than point-clipping, preserving continuous probability density without artificial point masses at boundaries, accompanied by strict bound ordering validation ($a < b$).
   - **Conditioning Integrity & Discrete Guard**: Requires conditioning-only predictors to be fully observed, mandates explicit cohort disposition for missing outcomes prior to MICE, rejects non-finite values, and rejects binary/low-cardinality discrete integer category encodings ({1, 2}, {1, 2, 3}) alongside declared categorical dtypes.
   - **Simulation Benchmark Evidence**: A Monte Carlo MAR simulation benchmark across 50 trials ($N=500$, true slope $\beta=0.06$, $M=10$) demonstrates empirical recovery for this specific evaluated scenario:
     - Pooled slope: $0.0602$ (vs full data $0.0592$, true $0.0600$).
     - Relative bias: $0.39\%$ (well within $|\text{bias}| < 10\%$).
     - Empirical 95% CI coverage: $94.00\%$ (with Monte Carlo SE $\approx 3.4$ percentage points, recovering coverage relative to marginal independent sampling).
   - **Guidance & Complete-Case Sensitivity**: Updated `SKILL.md` and `decision-heuristics.md` recommending $M \ge 5$ for $\le 30\%$ missingness, $M \ge 20$ for $> 30\%$ missingness, and qualifying that complete-case analysis serves as a sensitivity comparison whose agreement does not establish MAR or rule out MNAR bias.
12. **Firth Penalized Likelihood Enhancements**:
   - Monotonically non-decreasing penalized log-likelihood is tracked across iterations in `pll_history`.
   - When maximum iterations are exceeded or convergence fails, immediate fallback to Wald intervals is enforced (`ci_method = ["wald_fallback"]`), preventing computation of profile likelihood cutoffs on non-maximized iterates.
13. **Comprehensive Verification**:
   - Rebuilt `/Users/ntwkkm/Desktop/shinystat-cloud.zip`.
   - All cloud tests passing with zero regressions and clean `ruff` linter/formatter compliance.

### Consequences
- **Status**: Accepted & Verified.
- **Biostatistical Invariance**: Eliminates silent data coercion, improper inference with $df = \infty$, fractional dummy MICE draws, rank-deficient OLS regressions, point-mass boundary clipping, and unverified optimization in cloud sandbox environments.
- **Proper Parameter Uncertainty**: MICE parameter draws preserve joint coefficient uncertainty across missing records, recovering empirical coverage in MAR benchmark scenarios.
---

## ADR 31: Publication Medical Figures, Report Intermediate Representation (IR), Multi-Format Renderers, and Pillar 5 Reporting Integrity

### Context
Users requested comprehensive document and visual presentation generation across multiple formats (Word `.docx`, PowerPoint `.pptx`, Markdown `.md`, self-contained `.html`, and `.pdf`) with publication-quality clinical figures. Prior to this implementation, the biostatistical pipeline concluded with printed text or static HTML strings (Recipe 13), lacking graphical plotting and structured multi-format document rendering.

### Decision
1. **Recipe Contract Refactoring with Backward Compatibility**:
   - Extended the return dictionary contracts across all 13 recipes in `python-recipes.md` to include plot-ready keys (`roc_curve`, `optimal_point`, `km_curves`, `risk_table`, `calibration_curve`, `calibration_bins`, `ba_points`, `love_plot_data`, `forest_data`, `missingness_summary`).
   - Baseline regression snapshot testing (`test_recipes_snapshot.py`) verified that all pre-existing return keys remained backward-compatible.
2. **Clinical Figure Library (`medstat.figures`)**:
   - Enforced 300 DPI publication standards, colorblind-safe palettes (Okabe-Ito / Tol), and Thai typography fallback chains (`Sarabun`, `Thonburi`, `Sukhumvit Set`, `Arial`).
   - Implemented 10 clinical plot generators:
     - `plot_forest`: OR/HR/RR with logarithmic scaling and reference lines at 1.0.
     - `plot_kaplan_meier`: Survival curves with Hall-Wellner/log-log CIs, aligned risk tables, and log-rank p-values.
     - `plot_roc_curve`: Empirical ROC curves with Youden's J optimal cutpoint, DeLong 95% CIs, and paired test overlay.
     - `plot_calibration`: Decile bins, Wilson score CIs, LOESS curves, Brier score, and ICI.
     - `plot_dca`: Decision Curve Analysis Net Benefit against Treat All and Treat None.
     - `plot_bland_altman`: Pairwise difference vs average, 95% LoA, and Bland-Altman (1999) large-sample CIs.
     - `plot_love`: Austin 2009 standardized mean difference covariate balance before/after PSM with 0.10 threshold.
     - `plot_retention_flow`: Pure-Matplotlib STROBE / CONSORT participant retention flowchart.
     - Diagnostic plots: Missingness heatmaps, Schoenfeld residuals, and MICE density overlays.
   - Enforced `FigureResult(png_path, alt_text, caption, source_df, csv_path)` return contract with automatic CSV data export alongside every PNG for auditability.
3. **Report Intermediate Representation (`medstat.reporting.ir`)**:
   - Established format-agnostic `ReportDocument` as Single Source of Truth (SSOT).
   - Composed of typed blocks (`HeadingBlock`, `ParagraphBlock`, `TableBlock`, `FigureBlock`, `CalloutBlock`) linked to `results_dict`.
4. **Multi-Format Document Renderers (`medstat.reporting.renderers`)**:
   - `render_markdown`: GitHub Flavored Markdown with linked asset figures directory.
   - `render_html`: Standalone single-file HTML with embedded base64 images and ICMJE 3-rule CSS tables.
   - `render_docx`: Native Microsoft Word `.docx` via `python-docx` with OpenXML `<w:tblBorders>` 3-rule borders and 6.5 in figures.
   - `render_pptx`: Native PowerPoint 16:9 widescreen presentation via `python-pptx` (`13.333" x 7.5"`) with 1-figure-per-slide layout and structured takeaway cards.
   - `render_pdf`: Headless Chromium PDF printing via Playwright (`page.pdf()`), with LibreOffice (`soffice`) fallback.
5. **Pillar 5: Reporting Integrity (`medstat.reporting.integrity`)**:
   - Numerical Traceability: Every narrative number is cross-checked against `results_dict` ($\pm 0.02$ absolute, 1% relative, or $\pm 0.05$ on percentage scale), allowing up to 15% untraced numbers (min 1) for descriptive/narrative labels while ignoring common integers (1, 2, 5, 10, 95, 100).
   - Zero-PHI Enforcement: Deterministic regex scan identifying Hospital Numbers (`HN`), Thai 13-digit National IDs, phone numbers, honorific-prefixed personal names, and dates of birth across narrative, table headers, and figure alt-text. Passing automated screening flags absence of detected markers but requires mandatory clinical/manual PHI review prior to external release.
   - Causal Invariants: Observational PSM, Cox, and logistic/odds-ratio models must explicitly state unmeasured or residual confounding and E-values.
   - Retention and Methods: Enforces participant disposition accounting and statistical method disclosures.

### Consequences
- **Status**: Accepted & Verified.
- **Auditability**: Every figure exports its raw plotted data as CSV, and every document traces its narrative numbers to numerical model results.
- **Multi-Format Delivery**: Clinicians and researchers can request Word reports, PowerPoint slide decks, standalone HTML summaries, or PDF manuscripts seamlessly.
- **Full Test Suite Pass**: All 286 unit tests passing with zero regressions.

[MEMORY_LEARN: Medical document generation must enforce single-source-of-truth Report IR where every narrative number is traced back to results_dict, figures return 300 DPI PNGs with exported CSV source data, and observational causal models strictly mandate unmeasured confounding/E-value caveats.]

---

## ADR 32: Standalone HTML as the Default Deliverable Format Across Main and Cloud Skills

### Context
Previously, Grilling Gate Trigger 5 stated a default assumption of `docx` report when user preference was omitted. However, Word document generation requires additional local/cloud dependencies (`python-docx`), which may be unavailable or unnecessary when users simply expect immediate, interactive, portable biostatistical reports in browser-compatible environments (Claude Web, Antigravity preview, Jupyter/sandboxes). The user explicitly requested standardizing the default output format to HTML whenever unspecified.

### Decision
1. **Canonical Main Skill (`skills/shinystat/SKILL.md`)**:
   - Updated introductory Grilling Gate rule to explicitly exempt unspecified document formats from halting.
   - Updated Pillar 5 to establish self-contained HTML (`.html`) as the default deliverable format.
   - Updated Grilling Gate Trigger 5 and Completion Criteria: when output format is unspecified, the agent defaults to self-contained HTML (`.html` report + English + NEJM) without blocking execution.
   - Updated `skills/shinystat/references/report-builder.md` with explicit callout on HTML default.
2. **Dedicated Cloud Skill (`packaging/cloud/shinystat-cloud/SKILL.md`)**:
   - Synchronized Pillar 5, Grilling Trigger 5, and Completion Criteria to guarantee self-contained HTML (`.html`) is always generated when unspecified.
   - Updated `packaging/cloud/shinystat-cloud/references/report-builder.md` with explicit callout and primary format priority.
3. **Multi-Agent Mirror Synchronization**:
   - Synchronized canonical updates across `.agents/skills`, `.agent/skills`, `.claude/skills`, `.cursor/skills`, and global `~/.gemini/config/skills` via `install-skills.sh`.
   - Re-packaged `/Users/ntwkkm/Desktop/shinystat-cloud.zip` via `package-cloud-skill.sh`.

### Consequences
- **Status**: Accepted & Verified.
- **Immediate Portability**: Zero-friction deliverable inspection via web browser or artifact renderer without requiring Microsoft Office or `python-docx`.
- **Mirror Parity**: 100% byte-identical integrity preserved across all workspace mirrors and verified by automated unit tests.

[MEMORY_LEARN: Both canonical and cloud biostatistical skills must default to self-contained HTML reports with embedded figures when format is unspecified, bypassing unnecessary Grilling halts while keeping Office/PDF generation on-demand.]

---

## ADR 33: Hardening Numerical Traceability and Clinical Multi-Curve Auditability

### Context
During CodeRabbit automated and assertive review of PR #9 (`feat/multi-format-figures-reporting`), several edge-case data integrity concerns were identified:
1. In `integrity.py`, allowing a 1% relative tolerance check for numbers $\le 1.0$ caused subtle mismatches (e.g., $0.860$ vs $0.865$) to pass silently. Additionally, dividing by 100 on sub-one tokens allowed small p-values (e.g., $p = 0.02$) to falsely match tiny numbers (e.g., $0.004$).
2. Numeric tokens matching structural integers ($95$, $10$) were exempted before checking against `results_dict`, causing valid clinical metrics (e.g., $95\%$ sensitivity, $10$ events) to bypass traceability auditing.
3. In `plot_roc_curve`, when paired ROC models were supplied, `source_df` only recorded the primary curve coordinates, omitting the paired comparator curve from `FigureResult.source_df` and the exported CSV.
4. In standalone cloud environments without `python-docx`, the `generate_docx_report` fallback previously aliased directly to `generate_html_report`, writing HTML markup into `.docx` files.

### Decision
1. **Strict Precision Scaling**:
   - For values with $|tn| \le 1.0$, enforce `abs_diff <= abs_tol` scaled strictly by displayed decimal precision ($\min(0.02, 10^{-\text{digits}})$). Disable the relative tolerance branch for $|tn| \le 1.0$.
   - Restrict percentage-scale matching ($tn \approx kn \times 100$) strictly to tokens with explicit `%` markers or true percentage scales ($|tn| > 1.0$ and $|kn| \le 1.0$).
2. **Result-First Verification**:
   - Audit all extracted numbers against `doc.results_dict` first. Only evaluate structural context exemptions (e.g. `95% CI`, `Table 1`, `alpha = 0.05`) as a fallback for unmatched numbers.
3. **Multi-Curve Source Data Parity**:
   - Construct `source_df` in `plot_roc_curve` with concatenated primary and paired curve coordinates (`Model`, `False_Positive_Rate`, `True_Positive_Rate`, `Threshold`) before calling `export_source_data`.
4. **Cloud Path Redirect**:
   - In `packaging/cloud/shinystat-cloud/references/report-builder.md`, wrap the `ImportError` fallback to redirect explicit `.docx` extensions to `.html` before invoking `generate_html_report`.

### Consequences
- **Status**: Accepted & Verified.
- **Robust Clinical Governance**: Eliminates false-positive number matching and ensures full auditability across primary and paired clinical models.
- **Verification**: 421/421 tests passing across unit, e2e, and stress test suites.

[MEMORY_LEARN: Auditing narrative clinical statistics requires displayed-precision tolerance gates for fractions, result-first matching before structural exemptions, and multi-curve source data export in comparative ROC curves.]

---

## ADR 34: Clinical Presentation Pagination, Scientific-Notation Precision, and Causal Caveat Enforcement

### Context
During the completion and hardening of PR #9 (`feat/multi-format-figures-reporting`), several presentation rendering and reporting integrity edge cases were surfaced:
1. In `pptx.py`, tables exceeding 12 rows were truncated to the first 12 rows (`df.head(12)`), dropping clinical data in long demographic or multivariable regression tables. Furthermore, passing a 0-column DataFrame caused `ZeroDivisionError` in `python-pptx` (which computes `width // cols`).
2. In `integrity.py` and cloud `report-builder.md`, numeric tokens in scientific notation ($|val| \le 1.0$, e.g. `1.23e-4`) previously derived `dec_digits` without considering the exponent, leading to an artificially wide tolerance (`0.01` instead of $10^{-6}$) and allowing false-positive matches against unrelated small probabilities.
3. Observational and PSM reports lacking unmeasured confounding caveats only generated a warning; `passed` remained True, violating Pillar 5 safety gates.
4. Kaplan-Meier captions formatted $p < 0.001$ as `"0.000"` instead of `"< 0.001"`, contradicting the on-figure annotation.

### Decision
1. **Multi-Slide PowerPoint Pagination**:
   - Paginate tables with $> 12$ rows into continuation slides (`chunk_size = 12`) titled `"{base_title} (Part {chunk_idx + 1}/{n_chunks})"`, rendering full headers, alternating row fills, and footnotes showing row ranges (`Rows 1–12 of 25`).
   - Guard `_add_table_slide` with `if len(df.columns) == 0: return` to prevent `ZeroDivisionError` on empty-column DataFrames.
2. **Exponent-Aware Scientific Precision**:
   - Parse mantissa decimal places and subtract the exponent (`max(0, mantissa_dec - exp)`) to derive the true effective precision for scientific notation tokens.
3. **Causal Caveat Gating**:
   - Enforce `and has_causal_caveat` in `verify_report_integrity`'s `passed` calculation, ensuring observational models lacking discussion of unmeasured confounding fail audit status.
4. **KM Caption P-Value Formatting**:
   - Format `log_rank_p < 0.001` as `"< 0.001"` in `plot_kaplan_meier` caption.

### Consequences
- **Status**: Accepted & Verified.
- **Clinical Data Integrity**: Long clinical tables are fully preserved across PowerPoint decks; scientific-notation significance thresholds are strictly audited.
- **Verification**: 427/427 tests passing across unit and e2e test suites.

[MEMORY_LEARN: In reporting integrity audits and presentation renderers: (1) scientific notation tokens (|val| <= 1.0) must derive effective precision by parsing mantissa decimal places minus exponent; (2) PowerPoint table builders in python-pptx divide width by column count, so 0-column DataFrames must be guarded with an early return; (3) observational reporting integrity audits must gate passed status on explicit causal/confounding caveats.]

---

## ADR 35: Dual-Skill Distribution (shinystat & shinystat-cloud) via skills.sh Ecosystem

### Context
Users across diverse agent runtimes (Claude Code, Cursor, Windsurf, Antigravity, and Claude Web) need simple, frictionless installation mechanisms. While `scripts/install-skills.sh` handles local environments and `scripts/package-cloud-skill.sh` creates ZIPs for Claude Web, the wider AI agent ecosystem has coalesced around the open, telemetry-indexed package manager `skills.sh` (`npx skills add <owner>/<repo>`).
Previously, `skills/` housed only `skills/shinystat/`, while `shinystat-cloud` was sequestered in `packaging/cloud/`. As a result:
1. `npx skills add NTWKKM/shinystat-skills` discovered only `shinystat`.
2. Users in restricted cloud sandboxes or lightweight setups who only wanted the pure standalone Python recipes had no direct way to install `shinystat-cloud` via the `skills` CLI.

### Decision
1. **Dual-Skill Canonical Layout**:
   - Promote `shinystat-cloud` into `skills/shinystat-cloud/` alongside `skills/shinystat/`.
   - Maintain byte-identical synchronization across all 4 multi-agent platform mirrors (`.agent/skills/`, `.agents/skills/`, `.claude/skills/`, `.cursor/skills/`).
2. **skills.sh Telemetry Discovery**:
   - With both skills in `skills/`, `npx skills add NTWKKM/shinystat-skills` discovers both `shinystat` and `shinystat-cloud`.
   - Users can install selectively:
     - `npx skills add NTWKKM/shinystat-skills --skill shinystat --yes` (flagship engine)
     - `npx skills add NTWKKM/shinystat-skills --skill shinystat-cloud --yes` (standalone recipes)
     - Interactive multi-select prompt when no `--skill` flag is provided.
3. **Automated Parity & Validation**:
   - Update `tests/unit/test_shinystat_skill.py` with `test_shinystat_cloud_canonical_skill_exists_and_valid()` validating YAML frontmatter, character limits (< 1024), and reference document existence.
   - Enforce multi-agent mirror parity across both skills in continuous testing.

### Consequences
- **Status**: Accepted & Verified.
- **Ecosystem Ergonomics**: Enables 1-line installation across any modern coding agent.
- **Verification**: Byte-for-byte parity verified across 4 agent mirrors; tests passing in unit suite.

[MEMORY_LEARN: Structuring multiple atomic skills under canonical `skills/<skill-name>/` enables `skills.sh` to auto-discover individual skills, allowing selective installation via `--skill <name>` while maintaining multi-platform mirror synchronization.]

---

## ADR 36: Dual / Toggle View (Plotly.js + 300 DPI Publication PNG) with Isolated HTML Figures Mirror and Offline Resilience

### Context
In medical and clinical research, HTML reports serve two distinct audiences:
1. **Exploratory Clinicians & Researchers**: Desire interactive visualizations (Plotly.js) to hover over ROC curve thresholds to evaluate Sensitivity/Specificity at specific clinical cutoffs, zoom into Kaplan-Meier survival curves to inspect numbers at risk over specific timeframes, and inspect calibration deciles and DCA Net Benefit values.
2. **Journal Submission & Peer Review**: Require strict 300 DPI static bitmap figures (Matplotlib) adhering to ICMJE/NEJM/JAMA typographical standards and layout invariants, along with downloadable CSV datasets for numerical auditability.

Directly bundling Plotly.js into HTML files inflates report size by ~3.5–4 MB per report, while relying purely on external CDN can break in offline hospital intranets or strict cloud sandboxes (e.g. Claude Web Artifacts / iframe CSP). Furthermore, mutating existing `src/medstat/figures/*.py` core figure generators could inadvertently introduce regressions into Word, PowerPoint, PDF, and CLI workflows.

### Decision
1. **Isolated Mirror Folder Pattern (`src/medstat/figures/html/`)**:
   - Preserves all baseline `src/medstat/figures/*.py` and stat calculation modules 100% untouched.
   - Introduces `DualFigureResult` (subclass of `FigureResult` in `src/medstat/figures/html/base.py`) which holds both the 300 DPI Matplotlib PNG and the companion Plotly interactive specification (`plotly_json`, `plotly_fig`).
   - Implements companion dual generators (`plot_dual_roc_curve`, `plot_dual_kaplan_meier`, `plot_dual_calibration`, `plot_dual_dca`, `plot_dual_bland_altman`, `plot_dual_forest`, `plot_dual_love`).
2. **Dual / Toggle HTML Card (`render_html`)**:
   - Generates a figure card with segmented pill toggle buttons: `[Interactive (Plotly)]` and `[Publication 300 DPI (Static)]`.
   - Incorporates one-click direct download buttons for high-resolution 300 DPI PNG (`data:image/png;base64,...`) and raw source data CSV (`data:text/csv;charset=utf-8,...`).
3. **Offline-First & Sandbox Resilience (AGENTS.md Invariant)**:
   - Plotly.js CDN is injected only when interactive figures exist in the document.
   - If the CDN is blocked or the client is offline (`typeof Plotly === 'undefined'`), the document displays an informational notice banner with an immediate button `[สลับไปดู Publication Figure (300 DPI)]` to smoothly toggle to the static 300 DPI image without script errors or broken blank canvases.
   - Top tab buttons remain active and fully functional for manual switching.
4. **Print Parity (`@media print`)**:
   - When printing to paper or PDF (Cmd+P), CSS media queries hide the toolbar, suppress interactive canvas elements, and print the crisp 300 DPI publication image.
5. **Skill Reference Alignment**:
   - Synchronized `report-builder.md`, `figures.md`, `SKILL.md`, and `decision-heuristics.md` across both flagship `shinystat` and `shinystat-cloud` skills, maintaining 100% byte-identical mirror parity.

### Consequences
- **Status**: Accepted & Verified.
- **Safety**: Zero modifications to existing baseline `src/medstat/figures/` modules.
- **User Experience**: Clinicians can explore data dynamically or export publication-ready assets seamlessly.
- **Zero Drift**: Multi-agent mirror parity verified by automated unit tests.

[MEMORY_LEARN: Dual / Toggle View in clinical HTML reporting pairs exploratory Plotly interactivity with 300 DPI publication static images, using an isolated mirror folder (`figures/html/`) to protect baseline modules while enforcing offline CDN fallback and @media print parity.]

---

## ADR 37: Adaptive Clinical Inquiry and Harmonized Default HTML Deliverables across Canonical and Cloud Skills

### Context
User testing in Claude Web and local agent runtimes revealed two behavioral variances:
1. **Grilling Gate Suppression & Rigidity**: In previous revisions, rigid question templates (`❓ Q1`) paired with prohibition clauses (*"do NOT block execution with an unnecessary halt"*) inadvertently caused agents to either skip clarifying critical clinical ambiguities (due to action bias) or mechanically demand rigid questionnaire forms when intent was already unambiguous.
2. **Deliverable Disconnect**: Agents in chat sandboxes often concluded statistical analysis by merely rendering conversational Markdown summaries in chat rather than generating and delivering the default self-contained HTML publication report (`.html`).
3. **Skill Asymmetry**: Canonical `shinystat` retained legacy rigid gating and an outdated EPV hard stop, while `shinystat-cloud` evolved independently, creating cognitive drift across platforms.

### Decision
1. **Pruning & Positive Prompting (`/writing-for-agents`)**:
   - Replaced contradictory *"halt immediately... without halting"* directives with direct positive framing: Clarify genuine ambiguities directly with the user before analyzing; proceed directly when context is clear.
   - Removed rigid `❓ Q1 ... ➡️ ...` blocks in favor of flexible, domain-targeted inquiries (outcome definition, event/censoring codes, low-vs-high abnormal biomarker direction, study design archetype).
   - Removed anti-halt phrasing that discouraged agents from asking when genuine doubt existed.
2. **Explicit Deliverable Intent**:
   - Standardized Pillar 5 guidance in both skills to explicitly state that the default deliverable is a self-contained HTML report (`.html`) per `references/report-builder.md`, rather than outputting only conversational Markdown text in chat.
3. **100% Behavioral Parity**:
   - Synchronized decision models across both canonical `shinystat` (local CLI/`medstat`) and `shinystat-cloud` (standalone Python recipes), maintaining 100% mirror parity across `.agent/`, `.agents/`, `.claude/`, `.cursor/`, and `packaging/cloud/`.
   - Re-packaged `/Users/ntwkkm/Desktop/shinystat-cloud.zip`.

### Consequences
- **Status**: Accepted & Verified.
- **Ergonomics**: Agents ask natural, targeted questions when needed and avoid redundant interruptions when intent is clear.
- **Deliverable Consistency**: Users receive self-contained HTML reports with embedded figures and publication tables by default across all agent platforms.
- **Verification**: 40/40 unit tests passing with zero regressions and clean git status.

[MEMORY_LEARN: Framing clinical inquiry as adaptive, targeted clarification while explicitly anchoring default deliverables to self-contained HTML reports eliminates conversational truncations and questionnaire friction across both local and cloud AI agent runtimes.]

---

## ADR 38: Autonomous Biostatistical Authority, Output Format Alignment Gate & Dual / Toggle View HTML Recommendation Standard

### Context
User feedback and review under `/writing-for-agents` identified two operational challenges across both `shinystat` and `shinystat-cloud`:
1. **Decision Hesitancy on Biostatistical Mechanics**: Agents exhibited timidity when handling routine biostatistical mechanics (e.g. asking the user whether to use parametric vs non-parametric tests, whether to apply Firth penalization for sparse data/separation, or asking about knot placement and table styling), blurring the line between statistical expertise and clinical intent.
2. **Output Format Ambiguity**: When the user prompt did not specify a deliverable format, agents previously either defaulted quietly to HTML without user alignment or lapsed into rendering plain Markdown chat text, skipping the rich report-generation protocol.

### Decision
1. **Autonomous Biostatistical Authority**:
   - Explicitly empower the agent as the **Lead Clinical Biostatistician** across both local and cloud skills.
   - The agent exercises complete autonomy over statistical mechanics: distributional shape auditing (Mean [SD] vs Median [IQR]), non-parametric test routing, EPV diagnostics, Firth penalized likelihood execution for sparse data / separation, Schoenfeld assumption testing for Cox models, Restricted Cubic Spline modeling, GEE cluster adjustments, Little's MCAR testing, MICE execution, Youden's J cutpoint optimization, and VanderWeele E-value calculations.
   - Clarify the Grilling Gate boundary: reserve user clarification strictly for irreducible clinical and protocol ambiguities (primary outcome definition, event direction/censoring coding, unlabelled biomarker abnormality direction, or proposal-data geometry mismatches).
2. **Mandatory Output Format Alignment Gate (Ask Immediately if Unspecified)**:
   - When the user prompt does not specify a deliverable format, the agent **must proactively ask the user upfront** before finalizing the deliverable.
   - Present a structured menu recommending:
     - **(Recommended) Interactive HTML Report with Dual / Toggle View**: In the same self-contained HTML document, interactive Plotly.js charts serve as the primary active view (hover tooltips, zooming, cutpoint inspection), paired with a tab/button to switch to **"Publication 300 DPI Figure (Static)"** for downloading and journal submission, complete with ICMJE 3-rule tables and structured Methods & Results narratives.
     - **Manuscript Word Document (`.docx`)**: Formal Word report with 3-rule publication tables and 300 DPI figures.
     - **Executive PowerPoint Presentation (`.pptx`)**: 16:9 widescreen slides (1 figure per slide layout with key takeaway cards).
     - **Publication PDF (`.pdf`)** / **Markdown (`.md`)**.
     - **Quick In-Chat Summary Only**: Concise executive narrative and tables directly in chat.
3. **Dual / Toggle View Specification Alignment**:
   - Aligned tab button labels across renderers and recipes: Tab 1 = `Interactive (Plotly)` (primary view), Tab 2 = `"Publication 300 DPI Figure (Static)"`.
   - Maintained 100% byte-identical mirror parity across canonical `skills/`, all agent mirrors (`.agent/`, `.agents/`, `.claude/`, `.cursor/`), and `packaging/cloud/shinystat-cloud/`.
4. **CodeRabbit Statistical Rigor & Format Gate Remediation**:
   - *Estimand-Driven Test Selection*: Replaced skewness-only testing heuristics with estimand- and design-driven selection, explicitly noting that Mann-Whitney U tests stochastic shift rather than medians when group distribution shapes differ.
   - *EPV vs Separation Boundary & Estimand-Scoped Model Branching*: Decoupled low EPV finite-sample bias ($\text{EPV} < 10$, handled autonomously) from monotone separation (which halts at the Grilling Gate for clinical confirmation between penalization and covariate reduction). Explicitly qualified both the Sparse Data Bias rule and Autonomous Execution list across `skills/shinystat/SKILL.md`, `skills/shinystat-cloud/SKILL.md`, and all platform mirrors so that autonomous Firth penalization applies only when monotone separation does not trigger the Section 2 Grilling Gate; when monotone separation is present, the agent must follow the Grilling Gate. In `decision-heuristics.md`, separated the decision tree so monotone separation routes directly to the Grilling Gate with explicit model scoping: for Binary Logistic, (A) Firth penalized likelihood (`fit_firth_logistic` in `medstat` / Recipe 7 in cloud) vs (B) covariate reduction; for Cox Proportional Hazards (Monotone Likelihood), (A) Firth penalized Cox (`fit_firth_cox` in `medstat` / `CoxPHFitter(penalizer=0.1)` in cloud) vs (B) covariate reduction. For low EPV without separation, branched by research estimand (etiological Firth vs prediction L2 shrinkage).
   - *MAR Clinical Rationale & MNAR Sensitivity*: Required documenting clinical rationale supporting MAR before executing MICE, alongside mandatory complete-case benchmark reporting. Explicitly updated the Pragmatic Missing Data Defaults and Situations Requiring Mandatory Grilling / Inquiry across `skills/shinystat/SKILL.md`, `skills/shinystat-cloud/SKILL.md`, and all platform mirrors so that clinically plausible MNAR without a planned sensitivity analysis (pattern-mixture or delta-adjustment) mandates Grilling Gate clinical review before presenting pooled estimates, closing the limitation-only disclaimer loophole.
   - *Recipe 11 Penalized Cox Implementation Alignment*: Configured `fit_survival_analysis_suite` in Recipe 11 (`python-recipes.md`) with a default `penalizer=0.0` for unpenalized standard Cox analyses, added negative penalizer validation (`penalizer >= 0`), returned `cox_penalizer` in the output dictionary when covariates are present, and updated `decision-heuristics.md` and `SKILL.md` to pass `penalizer=0.1` explicitly for low-EPV and monotone likelihood strategies, maintaining parity with `medstat.models.survival.fit_cox_ph`.
   - *Qualified Dual-View HTML Rendering & Label Standardization*: Clarified in both skill references and `report-builder.md` that Dual / Toggle View pairs interactive Plotly charts with static publication figures when Plotly specifications are supplied (`plotly_figs`), explicitly stating that without a Plotly specification, the report contains only the static publication view. Standardized all static-view toggle button labels, fallback buttons, and static badges in `src/medstat/reporting/renderers/html.py` to `Publication 300 DPI Figure (Static)` and synchronized test assertions in `tests/unit/test_html_figures_mirror.py`.
   - *Observational Causal E-Values*: Explicitly restricted VanderWeele E-value computation to observational causal/exposure-outcome settings.
   - *Eliminated Format Gate Fallback Ambiguity*: Removed contradictory statements labelling HTML as an automatic 'default deliverable format when unspecified', ensuring the agent always waits for user format selection at the Output Format Alignment Gate.

### Consequences
- **Status**: Accepted & Verified.
- **Biostatistical Autonomy**: Agents act decisively on statistical decisions while respecting clinical boundaries.
- **Reporting Protocol Precision**: Proactive output format alignment ensures users always get their intended format without silent assumptions, with the Dual / Toggle View HTML report as the primary recommendation.
- **Parity & Tests**: 524/524 tests passing across the entire test suite with zero regressions or drift.

[MEMORY_LEARN: Empowering agents with Autonomous Biostatistical Authority over statistical mechanics while enforcing a mandatory Output Format Alignment Gate recommending Dual / Toggle View HTML reports guarantees decisive analytical execution and publication-grade deliverable fidelity.]

---

## ADR 39: Single Source of Truth (SSOT) Enforcement & Automated Skill Mirror Parity

### Context
In multi-agent environments, `shinystat-skills` maintains bundled Python scripts inside `skills/shinystat/scripts/medstat/` and mirrors across `.agent/`, `.agents/`, `.claude/`, and `.cursor/` for zero-setup portability. In commit `313a255` (PR #13), `src/medstat/reporting/renderers/html.py` was updated with standardized static view toggle labels, but the change was not propagated to the skill bundle, causing silent code drift. The existing mirror parity test only asserted equality between `skills/` and agent mirrors, leaving the seam between `src/medstat/` (the SSOT) and `skills/shinystat/scripts/medstat/` unvalidated.

### Decision
1. **Re-synchronize Code Drift**: Propagated latest `html.py` from `src/medstat/` across `skills/` and all multi-agent mirrors.
2. **Automated End-to-End Parity Tests**: 
   - Expanded `test_src_medstat_and_skills_bundled_medstat_byte_identical()` in `tests/unit/test_shinystat_skill.py` to directly verify 100% byte-identical parity of all 78 engine modules against `src/medstat/` across all 5 targets: `skills/shinystat/scripts/medstat/`, `.agent/skills/shinystat/scripts/medstat/`, `.agents/skills/shinystat/scripts/medstat/`, `.claude/skills/shinystat/scripts/medstat/`, and `.cursor/skills/shinystat/scripts/medstat/`.
   - Verified that `test_shinystat_multi_agent_mirrors_byte_identical` and `test_shinystat_cloud_canonical_skill_exists_and_valid` cover `SKILL.md`, `references/`, and `packaging/cloud/shinystat-cloud` with 100% byte-for-byte fidelity.
3. **CodeRabbit PR Review Optimization (`.coderabbit.yaml`)**:
   - Updated `path_filters` to explicitly exclude all mirror folders (`!.agent/**`, `!.agents/**`, `!.claude/**`, `!.cursor/**`, `!packaging/**`, `!skills/**/scripts/medstat/**`) while focusing exclusively on canonical files (`src/**`, `skills/**`, `tests/**`, `docs/**`, and root configs).
4. **Pytest Warning Elimination (62 -> 0 Warnings)**:
   - Fixed GEE QIC scale warning at source in `src/medstat/models/multilevel.py` by passing `scale=getattr(result, "scale", None)`.
   - Configured `filterwarnings` in `pyproject.toml` to elevate unexpected `FutureWarning` or `DeprecationWarning` from `medstat.*` to fatal errors, while safely filtering expected numerical warnings caused by intentional stress/boundary testing (separation, non-convergence, degenerate matrices, pingouin oracle comparisons).
5. **Installer Auto-Sync & `--sync` Flag**: Enhanced `scripts/install-skills.sh` to automatically rsync `src/medstat/` into `skills/shinystat/scripts/medstat/` before installation, and added a dedicated `--sync` / `--sync-only` CLI workflow that refreshes all workspace mirrors without creating transient backups.
6. **Living Contract Diary Preservation & Docs Organization**: Relocated historical milestone documentation (`ORIGINAL_REQUEST.md`, `PROJECT.md`, `TEST_INFRA.md`, `TEST_READY.md`) into `docs/` while strictly preserving `ARCHITECTURE.md`, `CONTEXT.md`, and `DESIGN.md` at the project root per Living Contract Diary invariants.

### Consequences
- **Status**: Accepted & Verified.
- **Zero-Drift Invariant**: Every single workspace mirror is tested directly against `src/medstat/` and canonical `skills/`; any unpropagated edit fails pytest deterministically.
- **Zero-Setup Usability**: Workspace agent mirrors remain committed and usable immediately out-of-the-box in Antigravity, Cursor, and Claude Code without manual install steps.
- **PR Review Efficiency**: CodeRabbit reviews only canonical files, saving token quota and eliminating duplicate comments.
- **Pristine Test Execution**: 530/530 tests pass in <20s with **0 warnings** and 100% clean lint/format.

[MEMORY_LEARN: Comprehensive multi-agent skill verification requires direct byte-identical parity tests from the package source (src/medstat) to every platform mirror (.agent, .agents, .claude, .cursor), paired with path_filters in .coderabbit.yaml to eliminate redundant PR review cycles.]

---

## ADR 40: Phase 2 Causal IPTW & Competing Risks Survival Architecture (Aalen-Johansen CIF & IPCW Fine-Gray Model)

### Context
Two critical biostatistical analytical pillars were previously incomplete in `shinystat-skills`:
1. **Causal Weighting (IPTW)**: Observational comparative effectiveness studies were limited to 1:1 Propensity Score Matching (PSM). PSM discards unmatched control or treated subjects, diminishing statistical power and restricting the target estimand primarily to the Average Treatment Effect on the Treated (ATT). Clinical researchers and epidemiologists frequently require Inverse Probability of Treatment Weighting (IPTW) to estimate the Average Treatment Effect across the entire eligible population (ATE), the Control (ATC), or the clinical equipoise target via Overlap Weights (ATO; Li et al., 2018). Furthermore, propensity score weighting without variance stabilization, positivity checking, or Kish's effective sample size ($N_{\text{eff}}$) risks extreme weight inflation and false precision.
2. **Competing Risks Survival Analysis**: In clinical oncology and cardiovascular trials, subjects frequently experience mutually exclusive competing terminal events (e.g., non-cardiovascular death competing with cardiovascular death, or local recurrence competing with distant metastasis). Standard Kaplan-Meier estimation treating competing events as uninformative right-censoring violates the independent censoring assumption, causing substantial upward bias in the cumulative event probability ($1 - \text{KM} > \text{true CIF}$). Furthermore, estimating only Cause-Specific Hazard Ratios (csHR) describes instantaneous event rates in event-free subjects (etiological mechanism) but fails to quantify overall cumulative incidence for prognosis and clinical risk stratification, which requires Fine-Gray Subdistribution Hazard Ratios (SHR).

### Decision
1. **Phase 2A: Inverse Probability of Treatment Weighting (`medstat.causal.iptw`)**:
   - Implemented `compute_iptw_weights` supporting 4 target estimands:
     - **ATE**: $w_i = \frac{T_i}{e_i} + \frac{1 - T_i}{1 - e_i}$ (with optional stabilized weights $sw_i = T_i \frac{p_T}{e_i} + (1 - T_i)\frac{1 - p_T}{1 - e_i}$).
     - **ATT**: $w_i = T_i + (1 - T_i)\frac{e_i}{1 - e_i}$.
     - **ATC**: $w_i = T_i\frac{1 - e_i}{e_i} + (1 - T_i)$.
     - **ATO (Overlap Weights)**: $w_i = T_i(1 - e_i) + (1 - T_i)e_i$, bounded in $[0, 1]$ and robust to poor positivity.
   - Enforced clinical guardrails: Kish's effective sample size ($N_{\text{eff}} = \frac{(\sum w_i)^2}{\sum w_i^2}$), positivity checking ($e_i < 0.05$ or $> 0.95$), extreme weight flags ($w_i > 10$ or $> 99^{\text{th}}$ percentile), and quantile truncation/trimming.
   - Designed `IPTWResult` supporting both attribute access and tuple unpacking `(df_weighted, diagnostics)` for backward compatibility.
   - Implemented `fit_iptw_outcome_model` supporting continuous (WLS), binary (logistic GLM with `freq_weights`), and survival (Cox PH with `weights_col`, `robust=True`), all using Huber-White robust sandwich standard errors (`HC0`/`HC1`).
   - Extended `medstat.causal.balance` to compute Bessel-corrected weighted sample variance ($s_w^2 = \frac{V_1^2}{V_1^2 - V_2} s_{\text{raw}}^2$), integrated weighted SMD columns in balance tables, and added square-marker weighted SMD traces to Love plots.
2. **Phase 2B: Competing Risks Survival Analysis (`medstat.models.survival`)**:
   - Implemented `fit_competing_risks_cif` using non-parametric `lifelines.AalenJohansenFitter(calculate_variance=True)`, Greenwood 95% confidence intervals, horizon evaluations at pre-specified follow-up times, probability conservation audit ($\sum_k \text{CIF}_k(t) + \hat{S}(t) \approx 1.0$), and automated Naive $1 - \text{Kaplan-Meier}$ overestimation calculation.
   - Resolved `lifelines` 0.30.3 version constraints: binary search step lookup via `index.searchsorted(eval_times, side="right") - 1` on cumulative density tables.
   - Implemented `fit_competing_risks_regression` reporting dual models:
     - **Cause-Specific Cox (`csHR`)**: Treats competing events as right-censored at event time (for etiological inference).
     - **Fine-Gray Subdistribution Hazard (`SHR`)**: Uses Inverse Probability of Censoring Weighting (IPCW) counting process dataset expansion, fitted via `lifelines.CoxPHFitter(entry_col, duration_col, event_col, weights_col, robust=False)`. Due to an upstream defect in `lifelines.fitters.coxph_fitter.SemiParametricPHFitter._compute_score_within_strata` which ignores entry times (`entries: None`, `_T unused`), combining `cluster_col` with `entry_col` in lifelines produces severely inflated cluster-robust standard errors (up to 4.8x). We resolved this by implementing an exact entry-aware cluster sandwich estimator (`entry_aware_cluster_sandwich`) based on Lin & Wei (1989) / Therneau & Grambsch (2000) counting process score residuals over $(T_{\text{start}}, T_{\text{stop}}]$, verified invariant to interval splitting to $10^{-9}$, paired with `score_gap` tie diagnostic monitoring and patient-level bootstrap fallback (`bootstrap_fine_gray_se`).
   - Designed publication 300 DPI step curves `plot_cif` with shaded 95% CIs and dashed $1 - \text{KM}$ overestimation curve, accompanied by companion interactive Plotly `plot_dual_cif`.
   - Created `render_competing_risks_table` in `medstat.reporting.tables` for publication-ready reporting of cause-specific vs subdistribution models.
3. **Cloud Recipes & Decision Heuristics**:
   - Created Recipe 14 (IPTW) and Recipe 15 (Competing Risks) in `skills/shinystat-cloud/references/python-recipes.md` with zero `medstat` dependency.
   - Added Decision Tree D (Causal Inference: PSM vs IPTW vs ATO) and Decision Tree E (Competing Risks: csHR vs Fine-Gray SHR) to `decision-heuristics.md` across canonical skills and multi-agent mirrors.

### Consequences
- **Status**: Accepted & Verified.
- **Zero New Dependencies**: Accomplished entirely within existing packages (`numpy`, `pandas`, `scipy`, `statsmodels`, `lifelines`).
- **Clinical & Statistical Rigor**: Overestimation of failure probability under competing risks is systematically prevented; causal weighting provides rigorous confounding adjustment with explicit positivity and effective sample size checks.
---

## ADR 41: Phase 3A Diagnostic Risk Reclassification (NRI & IDI) and Time-Dependent ROC (AUROC(t))

### Context
Clinical diagnostic and prognostic evaluation requires rigorous quantification of the incremental value added by novel biomarkers, risk scores, or multivariable models beyond standard baseline predictors:
1. **Diagnostic Risk Reclassification**: While overall discrimination ($\Delta\text{AUROC}$) tested via the DeLong paired test measures rank-order separation, it is notoriously insensitive to meaningful incremental clinical shifts in predicted probabilities. Clinicians and researchers require Net Reclassification Improvement (Continuous and Categorical NRI) and Integrated Discrimination Improvement (IDI; Pencina et al., 2008, 2011). However, uncalibrated probabilities or poorly configured thresholds produce spurious positive NRI (Kerr et al., 2014; Pepe et al., 2014). Transparent clinical reporting demands separate event ($NRI_e$) and non-event ($NRI_{ne}$) calculations, $K \times K$ contingency matrices, and probability calibration enforcement ($p \in [0, 1]$).
2. **Time-Dependent ROC in Longitudinal Cohorts**: In time-to-event survival settings with right-censoring (and competing risks), static binary ROC calculations are invalid because disease status changes dynamically over follow-up time. Clinical investigators require Cumulative/Dynamic Inverse Probability of Censoring Weighting (IPCW) ROC estimation ($\text{AUROC}(t)$; Uno et al., 2007; Heagerty et al., 2000) across milestone follow-up horizons (e.g., 30-day, 90-day, 1-year, 5-year), paired with longitudinal discrimination trajectory plots, optimal Youden cutoffs, and bootstrap confidence intervals.

### Decision
1. **Diagnostic Risk Reclassification Engine (`medstat.diagnostic.roc`)**:
   - Implemented `compute_reclassification_metrics`:
     - **Continuous (Category-Free) NRI**: $NRI_e = P(\text{up} \mid Y=1) - P(\text{down} \mid Y=1)$, $NRI_{ne} = P(\text{down} \mid Y=0) - P(\text{up} \mid Y=0)$, $NRI_{\text{total}} = NRI_e + NRI_{ne}$ with asymptotic standard errors and bootstrap CIs.
     - **Categorical NRI**: Evaluates movements across user-specified clinical risk cutoffs (e.g., `[0.10, 0.20]`), generating $K \times K$ transition contingency matrices for events, non-events, and combined cohorts, reclassification percentages (upward, downward, unchanged), and bootstrap CIs.
     - **Integrated Discrimination Improvement (IDI)**: Quantifies difference in discrimination slopes $(\bar{p}_{\text{new}, 1} - \bar{p}_{\text{old}, 1}) - (\bar{p}_{\text{new}, 0} - \bar{p}_{\text{old}, 0})$ with asymptotic SE, z-statistic, and relative IDI ($IDI / \text{slope}_{\text{old}}$).
     - **Safety Guards**: Miscalibration guard enforcing $p_0, p_1 \in [0, 1]$; separate event and non-event reporting per Kerr & Pepe guidelines.
   - Implemented `compare_diagnostic_models`: Combined DeLong paired test + Continuous NRI + Categorical NRI + IDI.
2. **Time-Dependent ROC Engine (`medstat.diagnostic.roc`)**:
   - Implemented `compute_time_dependent_roc`:
     - Cumulative/Dynamic IPCW estimator using Kaplan-Meier censoring distribution $G(t) = P(C > t)$.
     - Weighted concordance AUROC calculation algebraically identical to weighted trapezoidal integration.
     - Handles competing risks (subjects failing from competing events before horizon $t$ remain in the risk set as controls).
     - Evaluates optimal Youden cutoffs ($\max(\text{TPR}(c) - \text{FPR}(c))$), sensitivity, and specificity per horizon.
     - Bootstrap confidence intervals across milestones and dual-model trajectory comparison ($\Delta\text{AUROC}(t)$).
     - Sparse horizon safety guard: emits warnings and sets CI to NaN if horizon $t > \max(T)$ or remaining cases/controls $< 5$.
3. **Figures & Reporting Integration**:
   - Static Matplotlib (300 DPI) and interactive Plotly HTML figures: `plot_reclassification`, `plot_time_dependent_roc` (modes: `"auroc_trajectory"` and `"curves"`), `plot_dual_reclassification`, `plot_dual_time_dependent_roc`.
   - ICMJE/NEJM/JAMA table formatters: `format_reclassification_table` (with $K \times K$ contingency matrices), `format_time_dependent_roc_table`.
   - Automated narrative generation: `generate_reclassification_narrative`, `generate_time_dependent_roc_narrative`, and enhanced `generate_methods_narrative`.
   - CLI options: `medstat diag --p-old --p-new --thresholds --reclassification --time-dependent-roc --time-col --event-col --time-horizons`.
   - Cloud Recipes: Added Recipe 16 (NRI/IDI) and Recipe 17 (Time-Dependent ROC) to `python-recipes.md`.

### Consequences
- **Status**: Accepted & Verified.
- **Zero New Dependencies**: Uses standard `numpy`, `pandas`, `scipy`, `lifelines`.
- **Methodological Soundness**: Resolves the insensitivity of $\Delta\text{AUROC}$ and prevents the bias of static ROC in longitudinal cohorts.

[MEMORY_LEARN: In diagnostic incremental value assessment, ΔAUROC is often insensitive; combining DeLong tests with Category-Free NRI, Categorical NRI (with KxK matrices), and IDI under probability calibration guards provides robust evaluation. For survival data, IPCW Time-Dependent ROC prevents static classification bias across milestone horizons.]

---

## ADR 42: Phase 3B TRIPOD+AI Bootstrap Internal Validation, Model Recalibration, and Riley Minimum Sample Size Determination

### Context
In clinical prediction modeling and prognostic biomarker development:
1. **Model Overfitting & Apparent Optimism**: Derivation models evaluated on their training data exhibit optimistic apparent performance (inflated AUROC, attenuated Brier score, calibration slope > 1 on derivation but < 1 on new cohorts). Split-sample or cross-validation suffers from data loss and model instability in typical clinical sample sizes ($N < 5,000$). The TRIPOD+AI (2024) statement strongly recommends Harrell's bootstrap optimism-correction protocol to estimate honest out-of-sample performance and calculate a post-estimation uniform shrinkage factor ($\hat{s}$) to counter overfitting.
2. **Model Updating & Recalibration**: Clinical models transferred to new geographic, temporal, or hospital populations frequently miscalibrate due to baseline prevalence differences or case-mix variations. Researchers require model recalibration techniques (Steyerberg 2019): Intercept-only updating (calibration-in-the-large, fixing slope $\beta = 1.0$) vs Logistic recalibration (updating both $\alpha$ and $\beta$).
3. **Rigorous Sample Size Determination**: Clinical studies historically relied on crude 10 events-per-variable (EPP) heuristics, which frequently lead to severe model underpowering or overestimation. Riley et al. (BMJ 2020; 368:m441) established a rigorous 4-criteria formulation for clinical prediction models targeting global shrinkage ($\ge 0.90$), small margin in apparent vs adjusted $R^2$, precise intercept estimation, and small Mean Absolute Prediction Error (MAPE $\le 0.05$). Additionally, prospective diagnostic studies testing AUROC against a null benchmark require non-parametric power formulations per Obuchowski (1998, 2004) and Hanley & McNeil (1982) accounting for disease prevalence.

### Decision
1. **TRIPOD+AI Bootstrap Internal Validation Engine (`medstat.diagnostic.validation`)**:
   - Implemented `bootstrap_internal_validation`:
     - Fits derivation logistic regression on cohort of size $N$.
     - Evaluates apparent metrics ($M_{\text{app}}$): AUROC, Brier score, Cox-Snell and Nagelkerke $R^2$, calibration intercept, and calibration slope.
     - Performs $B$ bootstrap resamples ($B \ge 100$, default 200) with replacement.
     - In each replicate: fits bootstrap model, computes performance in bootstrap resample ($M_{\text{boot}}$) and in original cohort ($M_{\text{test}}$).
     - Calculates optimism $\mathcal{O}_m = \frac{1}{B}\sum (M_{\text{boot}} - M_{\text{test}})$ and optimism-corrected metric $M_{\text{opt\_corr}} = M_{\text{app}} - \mathcal{O}_m$.
     - Computes uniform shrinkage factor $\hat{s} = \text{mean}(\text{calibration\_slopes}_{\text{test}})$, and applies it to produce shrunk coefficients $\hat{\beta}_{\text{shrunk}} = \hat{s} \cdot \hat{\beta}_{\text{orig}}$.
     - Clinical guardrails: warns on EPP < 10, severe overfitting ($\hat{s} < 0.70$), or bootstrap non-convergence.
2. **Model Recalibration Engine (`medstat.diagnostic.validation`)**:
   - Implemented `recalibrate_prediction_model`:
     - Method `"intercept_only"`: $\text{logit}(Y) = \alpha + 1 \cdot \text{LP}$, solving $\sum \text{expit}(\alpha + \text{LP}_i) = \sum y_i$ via Brent's method.
     - Method `"logistic"`: $\text{logit}(Y) = \alpha + \beta \cdot \text{LP}$, updating intercept and calibration slope with 95% CIs and p-values.
     - Method `"both"`: Evaluates both methods simultaneously, reporting pre- and post-updating calibration intercept, slope, Brier score, Emax, ICI, and AUROC.
3. **Sample Size Engines (`medstat.power.sample_size`)**:
   - Implemented `calculate_riley_sample_size`:
     - Evaluates Riley Criteria 1–4, identifying the binding criterion, total $N$, required events, and anticipated EPP.
     - Automated deterministic numerical derivation of Cox-Snell $R^2_{CS}$ from user-supplied C-statistic or Nagelkerke $R^2$ using 64-node Gauss-Hermite quadrature integration (Riley et al. Stat Med 2021 / Austin & Steyerberg 2012) and Brent root finding.
     - Enforces conservative bound $R^2_{CS,\max} = 1 - \phi^{2\phi}(1 - \phi)^{2(1-\phi)}$.
   - Implemented `calculate_diagnostic_auroc_sample_size`:
     - Non-parametric Obuchowski (1998) / Hanley-McNeil (1982) sample size calculation given $\text{AUROC}_{\text{expected}}$, $\text{AUROC}_{\text{null}}$, prevalence $\phi$, $\alpha$, and power.
     - Computes variance terms $V_0, V_1$, required case count $n_{\text{cases}}$, control count $n_{\text{controls}}$, and total $N$.
4. **Publication Figures & Reporting Tables**:
   - Static Matplotlib (300 DPI) and interactive Plotly HTML figures: `plot_recalibration_comparison`, `plot_dual_recalibration_comparison`, `plot_bootstrap_optimism`, `plot_dual_bootstrap_optimism`.
   - Publication-grade HTML tables: `format_internal_validation_table`, `format_recalibration_table`, `format_riley_sample_size_table`.
   - Academic narrative generation: `generate_internal_validation_narrative`, `generate_recalibration_narrative`, `generate_riley_sample_size_narrative`, `generate_diagnostic_auroc_sample_size_narrative`.
   - CLI options: `medstat diag --internal-validation --recalibration`, and `medstat sample-size --type riley / diagnostic-auroc`.
   - Standalone Python Recipes 18, 19, and 20 in `skills/shinystat-cloud/references/python-recipes.md`.

### Consequences
- **Status**: Accepted & Verified.
- **Zero New Dependencies**: Accomplished with existing `scipy`, `numpy`, `pandas`, `statsmodels`.
- **Methodological Standards**: Full alignment with TRIPOD+AI guidelines for prediction model development and validation, and state-of-the-art power planning via Riley et al. and Obuchowski.

[MEMORY_LEARN: For clinical prediction models, TRIPOD+AI bootstrap internal validation computes honest optimism-corrected AUROC, Brier score, and uniform shrinkage factors (ŝ) to mitigate overfitting. For sample size planning, Riley et al. (BMJ 2020) 4-criteria formulation replaces fragile EPP heuristics by tying required N directly to target shrinkage and expected R².]

---

## ADR 43: `/writing-for-agents` Skill Architecture Elevation & PR #17 Statistical Rigor Hardening

### Context
In PR #17 (`docs/elevate-skills-writing-for-agents`), both canonical skills (`skills/shinystat/` and `skills/shinystat-cloud/`) were refactored using `/writing-for-agents` levers (front-loaded context pointers, progressive disclosure with anchor links, leading words `pre-flight`, `reconciled`, and `traceable`, positive framing, and checkable/exhaustive Completion Criteria). CodeRabbit assertive review on PR #17 identified three statistical and verification refinements:
1. **Scale-Aware Low-EPV Cox Penalization**: `Recipe 11` (`fit_survival_analysis_suite`) previously passed unstandardized predictors to `CoxPHFitter(penalizer=0.1)` under a fixed penalty recommendation. Because L2 ridge shrinkage depends on predictor scale, equivalent predictors in different measurement units receive unequal effective shrinkage unless continuous predictors are standardized or the penalty is data-validated.
2. **Design-Scoped Missing-Indicator Modeling**: High non-primary missingness (`> 40%`) previously offered `missing-indicator / pattern-mixture modeling` as a general option. In non-randomized observational studies, the missing-indicator method can introduce bias even under MCAR or MAR (Groenwold et al., *CMAJ* 2012), whereas its valid primary use is narrower (such as baseline-covariate adjustment in randomized trials).
3. **Exhaustive Numerical Traceability Completion Gate**: Both `SKILL.md` completion criteria previously inherited the raw `<= 1` / `15%` unmatched allowance without requiring manual verification of remaining unmatched numbers, which could permit a report with a single incorrect estimate or $P$-value to satisfy the completion gate.

### Decision
1. **Scale-Aware Cox Shrinkage (`Recipe 11`, `SKILL.md`, `decision-heuristics.md`)**:
   - Updated `fit_survival_analysis_suite` in `skills/shinystat-cloud/references/python-recipes.md` with optional `standardize_continuous: bool = False` and explicit `continuous_covariates: list[str] | None = None` metadata, z-score standardizing only explicitly declared continuous covariates (`(x - mean) / sd`, `ddof=1`), recording transformations in `cox_standardization` (`{"mean", "sd", "scale": "per 1 SD increase"}`), and rejecting unencoded or numeric-coded multi-level categorical covariates unless pre-encoded as binary `0/1` indicators.
   - Updated low-EPV and monotone-likelihood Cox guidance in `skills/shinystat-cloud/SKILL.md` and `skills/shinystat-cloud/references/decision-heuristics.md` to require scale-aware, data-validated L2 penalization (standardizing continuous predictors and reporting the transformation, or selecting and validating `penalizer` on the supported data), alongside clinical risk score reduction.
2. **Design-Scoped Missing-Indicator Guidance (`SKILL.md` & `decision-heuristics.md`)**:
   - Restricted missing-indicator modeling in the `> 40%` non-primary missingness branch across `skills/shinystat/` and `skills/shinystat-cloud/` to randomized-trial baseline-covariate adjustment or explicitly labeled sensitivity analyses alongside pattern-mixture modeling.
3. **Exhaustive Numerical Traceability Gate (`SKILL.md`)**:
   - Tightened Pillar 5 Numerical Traceability in the Completion Criteria of both `skills/shinystat/SKILL.md` and `skills/shinystat-cloud/SKILL.md` to require `len(untraced_numbers) == 0` or manual verification of every remaining value in `untraced_numbers` as a non-statistical token, ensuring any unverified estimate, CI bound, or $P$-value blocks completion.

### Consequences
- **Status**: Accepted & Verified.
- **Unit Invariance & Categorical Safety**: Verified via unit tests (`test_survival_analysis_suite_penalizer_options`) that 10x rescaling of continuous covariates yields identical hazard ratios under `standardize_continuous=True` with explicit `continuous_covariates` and `penalizer=0.1`, while numeric-coded multi-level categorical covariates are explicitly rejected unless pre-encoded as binary `0/1` indicators.
- **Mirror Parity**: Synchronized across all 4 multi-agent mirrors (`.agent/`, `.agents/`, `.claude/`, `.cursor/`) and `packaging/cloud/shinystat-cloud/`.

[MEMORY_LEARN: L2-penalized Cox regression requires explicit continuous-covariate metadata for standardization (never inferring continuity from nunique() > 2, which misclassifies numeric-coded categorical covariates); missing-indicator adjustment in high-missingness routing must be restricted to RCT baseline covariates or labeled sensitivity analyses; and agent completion gates must require manual verification of any remaining untraced numbers.]

---

## ADR 44: Integration of Journal-Grade Figures, Study Flow Diagrams, and Scientific Schematics

### Context
Clinical research manuscripts submitted to high-impact journals (*New England Journal of Medicine*, *JAMA*, *The Lancet*, *The BMJ*) require strict visual standards, vector TrueType fonts, L-frame axis styling, low-event inset zooms, and mathematically reconciled study flowcharts (CONSORT 2025/2010, PRISMA 2020 single/dual stream, STARD 2015, STROBE cohort flow). Pathophysiological mechanisms also require vector-first conceptual schematics (cellular biology, tissue cross-sections, ulcer craters, pressure injuries, vessel stenosis) without relying on heavy external software or cloud dependencies. The separate `journal-figures` skill contained valuable prototypes, but lacked integration with `shinystat`'s core computational pipeline, statistical models, and `SampleFlowTracker`.

### Decision
Merge and elevate all capabilities into `medstat-core` (`src/medstat/figures/`), `skills/shinystat/`, and `skills/shinystat-cloud/`:
1. **Core Architecture & SSOT**:
   - `medstat._phi`: SSOT regular expression engine for HIPAA/PDPA Zero-PHI compliance across reporting and figures (`assert_zero_phi_text`).
   - `FigureResult`: Extended with `vector_path` (TrueType PDF), `svg_path`, and `qa` dictionary alongside existing `png_path`, `caption`, and `csv_path`.
   - `medstat.figures.journal`: Six publication house styles (**NEJM**, **JAMA**, **LANCET**, **BMJ**, **CLINICAL**, **OKABE_ITO**) using `ggsci`-inspired and CVD-safe palettes (`JOURNAL_SPECS`), calibrated target widths (`single_col` 3.5", `one_half_col` 5.0", `double_col` 7.0", `docx` 6.5", `pptx` 11.5", `web` 7.5"), TrueType vector fonts (`pdf.fonttype=42`), L-frame spines (`style_axis`), and clinical report formatters (`format_estimate`, `format_p_value`, `format_num`).
   - `medstat.figures.qa`: Automated layout collision detection (`qa_layout`), CVD simulation (`cvd_check`), and Zero-PHI gate (`assert_zero_phi_text`).
2. **Study Flow Diagrams (`medstat.figures.flow`)**:
   - Top-down layout cursor `Flow` with strict mathematical reconciliation: raises `ValueError` if branch counts do not balance.
   - Implemented templates: CONSORT 2025 (reporting participants analyzed for primary outcome per Item 13c) & 2010 (`consort_rct`), PRISMA 2020 single- and dual-stream (`prisma_2020`), STARD 2015 diagnostic accuracy with $2 \times 2$ cells (`stard_flow`), STROBE cohort retention (`cohort_flow`, `cohort_flow_from_tracker` linked directly to `SampleFlowTracker`), and methods pipeline (`methods_pipeline`).
   - Automated export at 600 DPI PNG, vector PDF, SVG, and coordinate `_source.csv`.
3. **Vector Scientific Schematics (`medstat.figures.illustration`)**:
   - Vector canvas `Scene` powered by semantic art palette (`SEM`).
   - Anatomical elements: biological cells (RBC, platelet, neutrophil, lymphocyte, macrophage, fibroblast, bacterium, eukaryote), skin layers (`skin`, `crater`), wound healing phases (`wound_phase`, `wound_topdown` with shoelace area verification), pressure injury staging (`pressure_injury`, Stages 1–4), vessel stenosis (`artery_cross_section`, `vessel_longitudinal`, `selectin_lollipop`), and signaling cascades (`node`, `edge`, `cascade`).
   - Mandatory conceptual schematic disclaimer caption: `"Schematic illustration; not to scale; not derived from patient images."`
   - Automated export at 600 DPI PNG, vector PDF, SVG, and coordinate `_source.csv`.
4. **Statistical Graph Upgrades & 100% Backward Compatibility**:
   - Upgraded `plot_kaplan_meier` (aligned risk tables, NEJM zoomed inset with `//` axis break, fixed-inch label offset avoiding tick collisions), `plot_cif`, `plot_roc_curve`, `plot_calibration`, `plot_forest` (3-zone JAMA/BMJ layout), `plot_dca`, `plot_love`, and `plot_bland_altman`.
   - Default `journal=None` preserves 100% backward compatibility with all existing snapshots and tests.
5. **Skill Integration, Single Source of Truth & Grilling Gate #5 (`/writing-for-agents`)**:
   - Added reference guides: `references/house-styles.md`, `references/flows.md`, `references/illustration.md` in both `shinystat` and `shinystat-cloud` with 100% palette, width, and API parity against `medstat.figures`.
   - Updated Grilling Gate #5 (`Output Format & House Style Alignment Gate`) to list house-style choices (`NEJM` recommended, `JAMA`, `LANCET`, `BMJ`, `CLINICAL` / `OKABE_ITO`) while disclosing hex palettes, `TARGET_WIDTHS`, and formatting specifications in `references/house-styles.md` (single source of truth, avoiding `SKILL.md` sprawl).
   - Cloud recipes maintain pure Python execution with zero `medstat` imports.

### Consequences
- **Status**: Accepted & Verified.
- **Traceability & Integrity**: Every statistical plot exports coordinate CSVs (`_source.csv`, `_risktable.csv`); every flow diagram verifies count reconciliation.
- **Publication Grade**: Generates 600 DPI PNG + TrueType vector PDF + SVG with zero glyph distortion or font rasterization.
- **Backward Compatibility**: All 408 baseline unit tests pass with zero regressions.

[MEMORY_LEARN: For clinical publications, flow diagrams must enforce mathematical count reconciliation (N_parent - sum(N_excluded) == sum(N_branches)) at runtime; Kaplan-Meier low-event insets require explicit yticks and fixed-inch offsets to avoid risk-table tick collision; and scientific illustrations must carry an automated schematic disclosure banner.]

---

## ADR 45: Autonomous Figure Verification Subagent (`figure-tester`) and Exhaustive Testing Matrix

### Context
Ensuring zero defect across all clinical figures in `shinystat` required validating 7 major clinical categories, 5 journal house styles (**NEJM**, **JAMA**, **LANCET**, **BMJ**, **CLINICAL**), multiple canvas geometries, multi-format export bundles (PNG 300/600 DPI, TrueType PDF `fonttype=42`, SVG, audited `_source.csv`, and Dual View Plotly HTML), and 4 mandatory QA gates (`qa_layout` bounding box collision detection, `cvd_check` colorblind contrast, Zero-PHI regex scan, and mathematical flow reconciliation). Manual or partial testing risked undetected bounding box collisions or bundle export corruptions.

### Decision
1. **Subagent Specification & Discovery (`figure-tester`)**:
   - Registered `figure-tester` globally in `~/.gemini/config/agents/figure-tester.md` and across workspace mirrors (`.agent/agents/figure-tester.md`, `.agents/agents/figure-tester.md`).
   - Equipped with specialized figure QA instructions, `journal-figures` and `shinystat` skills, and write/command tools.
2. **Exhaustive Testing Matrix Suite**:
   - Implemented `tests/unit/test_all_figures_matrix.py` (10 pytest test cases) and `scripts/test_all_figures_matrix.py` (standalone 60-figure generation runner).
   - Covered:
     * Survival & Competing Risks: Stratified KM with aligned risk tables, censor ticks, NEJM zoomed inset, cause-specific CIF step curves, Greenwood 95% CIs, and Naive 1 - KM overlay.
     * Forest Plots: 3-zone layout on logarithmic ratio scale (HR/OR/Subdistribution HR) and linear difference scale (Beta/MD), custom header rows, and events columns.
     * Diagnostic & Prediction Validation: ROC curves with DeLong CIs and optimal cutpoint, paired ROC overlay, NRI scatter plot, Time-Dependent ROC trajectory and multi-horizon curves, decile calibration curves with Wilson score CIs and spike histograms, model recalibration comparison, and Vickers DCA net benefit.
     * Causal Balance & Agreement: Austin Love plots with 0.10 SMD threshold and multi-cohort balance, and Bland-Altman agreement with large-sample CIs.
     * Study Flow Diagrams: CONSORT 2025/2010, PRISMA 2020 single/dual-stream, STARD 2015 diagnostic flow, STROBE cohort flow, and methods pipeline.
     * Vector Scientific Schematics: Primitives (RBC, platelet, neutrophil, macrophage, fibroblast, bacteria), skin layers with crater, wound healing phases, and arterial stenosis %.
     * Diagnostics & Missingness: Missingness heatmap, Schoenfeld residuals, MICE density overlays.
     * Multi-format bundle outputs: 300/600 DPI PNG, TrueType PDF, SVG, CSV, and interactive Plotly HTML.
3. **Core Hardening & Bug Fixes**:
   - Hardened `plot_reclassification` against DataFrame boolean ambiguity by verifying `isinstance(reclass_data, dict)` before dictionary access.
   - Expanded `plot_bootstrap_optimism` x-axis limit multiplier to `max_val * 1.35` to guarantee zero bounding-box text collision with the legend.

### Consequences
- **Status**: Accepted & Verified.
- **Verification**: 100% test pass (61/61 pytest tests across figure test suites; 60/60 standalone figures generated with zero QA layout overlaps or PHI leaks).
- **Auditability**: Artifact outputs and QA ledgers are permanently audited and reproducible on any machine.

[MEMORY_LEARN: Expanding horizontal bar plot xlim to max_val * 1.35 prevents bounding box collisions between bar value labels and lower-right legend elements, ensuring 100% compliance with automated publication QA layout gates.]

---

## ADR 46: Actionable Proportional Hazards (PH) Violation Handling & Remedies

### Context
`SKILL.md` historically directed agents to "stratify or add time-varying covariates on PH violation", but suffered from four architectural defects:
1. `Recipe 11` only executed the Schoenfeld test; no executable remedy code existed.
2. The heuristic lacked operational decision logic: a nuisance covariate can be stratified (its HR disappears from the reporting table); an exposure cannot (requires piecewise/time-varying HR or RMST).
3. Evaluating $P \le 0.05$ across $k$ covariates ignored multiplicity (with 13 covariates, $\approx 0.65$ false violations are expected by chance).
4. The Diagnostics criterion was satisfied simply by *computing* Schoenfeld residuals, allowing detected violations to be silently ignored.

### Decision
1. **Single Source of Truth Rule (`references/decision-heuristics.md §2.E`)**:
   - Consolidated the decision rule in `references/decision-heuristics.md §2.E`.
   - `SKILL.md` and `Recipe 11` point to this rule using the canonical terms *violator* and *remedy*.
   - Updated the Model Convergence & Diagnostics checklist criterion from "Schoenfeld residuals ... computed" to *"every Schoenfeld violator carries a stated remedy"*.
2. **Operational Decision Logic & Multiplicity Partition**:
   - For $k$ covariates tested, define Bonferroni threshold $\alpha_{\text{Bonferroni}} = 0.05 / k$.
   - **Strong Violation** ($P \le \alpha_{\text{Bonferroni}}$): Definite violation requiring remedy.
   - **Weak Violation** ($\alpha_{\text{Bonferroni}} < P \le 0.05$): Potential violation; inspect `plot_schoenfeld_residuals` (figures.md §9) to decide.
   - **Nuisance Covariate Violator** $\to$ `refit_cox_with_strata(df, duration_col, event_col, covariates, strata_cols, exposure)`: refits with stratified baseline hazards (`strata=strata_cols`); non-strata HRs match lifelines (rtol 1e-8); strata covariates are omitted from the table; raises `ValueError` if `exposure` is in `strata_cols`.
   - **Exposure Violator** $\to$ `piecewise_hr(df, duration_col, event_col, exposure, covariates, cut_points)`: fits period-specific HRs via counting-process episode splitting and lifelines `CoxTimeVaryingFitter`.
   - **Dual Model Reporting**: Report both unadjusted/standard Cox and the remedy model, stating the remedy beside each violator in narrative/tables.
3. **Lifelines Variance Invariant**:
   - `piecewise_hr` uses model-based variance (`robust=False`). `cluster_col` and `robust=True` are strictly omitted because lifelines' robust/cluster sandwich estimator mishandles entry times (counting-process start intervals), inflating SEs.
4. **Known Limitation & Retained Plot Requirement**:
   - Schoenfeld tests with rank time transform evaluate monotonic correlations between residuals and ranked failure times. They can miss non-monotone effect trajectories (e.g. U-shaped, bathtub, or transient early spikes that average out across ranks).
   - Therefore, automated test significance is necessary but not sufficient: `plot_schoenfeld_residuals` remains an integral part of the decision rule.

### Consequences
- **Status**: Accepted & Verified.
- **Parity**: Recipe 11 code block is verified 100% identical in AST and normalized source text to `src/medstat/models/survival.py`.
- **Test Coverage**: Added `tests/unit/test_ph_violation_remedies.py` verifying strata HR equivalence, exposure ValueError guard, simulation recovery within 15% ($N=3000$, true HR 0.5 before $t=2$, 1.0 after), multiplicity list partitioning, and Recipe 11 AST parity.
- **Mirror Sync**: All workspace mirrors (`.agents`, `.agent`, `.claude`, `.cursor`, `packaging/cloud`) synchronized via `./scripts/install-skills.sh --sync` with zero diff.

[MEMORY_LEARN: In Cox proportional hazards diagnostics, Schoenfeld tests with rank time transform can miss non-monotone time-varying effects; automated Bonferroni P <= 0.05/k partitions strong vs weak violations, while plot_schoenfeld_residuals visual inspection remains mandatory before finalizing remedies.]

---

## ADR 47: Optional Polish Grilling Gate for Evidence-Base Grounding & Superscript Citations

### Context
In automated manuscript preparation and clinical report generation, AI agents risk two distinct failure modes:
1. **Evidence Mismatch & Phantom Citations**: Agents may invent fictional paper titles, misattribute clinical findings, or cite studies that do not exist or whose actual conclusions contradict the narrative claims. Furthermore, prose claims may overstep study designs (e.g. asserting causality on unadjusted observational cohorts).
2. **Citation Formatting Disconnect**: Narrative references often leak raw Markdown/HTML markup into exports or trigger false-positive "untraced number" failures during Pillar 5 numerical traceability auditing when citation numbers exceed standard statistical index limits.

### Decision
1. **Mandatory Inquiry at Gate #5 & Polish Grilling Gate #6**:
   - Integrated into the **Mandatory Alignment Gate (#5)** alongside Deliverable Format and House Style (if prompt explicitly requests finding supporting evidence e.g. "หาหลักฐานมาประกอบ", "find evidence", Gate #6 Full Polish auto-activates immediately):
     - **(Standard) Skip Polish (Direct Fast-Path)**: Proceeds immediately through the normal analysis workflow to deliver statistical findings, tables, and figures using user-provided or existing local sources (when user sources cite studies without full text or complete details, executes targeted primary study retrieval via `search_web`/MCP to format complete superscript citations `<sup>1</sup>`; no exploratory literature review; on-demand full polish remains available later).
     - **(Optional) Full Publication Polish (Execute Gate #6)**: Executes empirical literature verification via `search_web` / MCP databases, data claims reconciliation, and Vancouver/ICMJE superscript citations (`<sup>1</sup>`, Word runs).
   - **Data Grounding & Causal Language**: Audits clinical assertions in Discussion and Results against `results_dict` and study design. Evaluates permitted causal claims based on the prespecified estimand, study design, confounder control, and identification assumptions (does not treat weighting alone as sufficient or unweighted observational data as automatically prohibiting causal claims; reports associative framing where identification is unverified, with E-values where defined and appropriate on supported effect scales). Restricts the $P > \alpha$ "insufficient evidence to reject the null hypothesis" requirement to superiority tests, while preserving margin- and CI-based interpretation for prespecified non-inferiority or equivalence analyses. Enforces exact magnitude- and percentage-specific numerical traceability matching the SKILL.md SSOT rule.
   - **Empirical Literature Verification & Primary Source Retrieval**: When user sources cite papers without full text/details, or when verifying narrative claims, cross-audits every cited study/guideline using `search_web`, PubMed, or bibliographic services, while restricting `medical-terminology` MCP to clinical ontology codes (SNOMED, LOINC, RxNorm, ICD-10) and laboratory reference ranges. Enforces the **Anti-Phantom Citation Invariant** (zero hallucinated studies).
   - **Superscript Citations & House Style Reference List**: Generates sequential in-text numerical superscript citations (`<sup>1</sup>`, native Word `.docx` superscript runs, Markdown/LaTeX) paired with a complete bibliographic References section formatted in the target Journal House Style (`NEJM`, `JAMA`, `Lancet`, `BMJ`, `Clinical`).
2. **Reporting Integrity Structural Recognition**:
   - Enhanced `_is_structural_context` in `medstat.reporting.integrity` (`verify_report_integrity`) and `verify_standalone_integrity` in `report-builder.md` to recognize bounded citation numbers (`<sup>12</sup>`, `[12]`, `^[12]`, `Ref 12`, `12. Author`) and publication years (1900–2035) with recognizable bibliographic context (removing overbroad keywords and bare parentheses). Enforces plausible citation bounds and References-section requirements for branches 6c and 6e to preserve auditing for clinical values (`[250]`, `[12-15 days]`, `250. Patients were...`). Accurately calculates `citations_count` across bracketed and caret forms.
3. **Multi-Format Parity**:
   - Renderers (`html.py`, `docx.py`, `pptx.py`, `markdown.py`) format superscript tags cleanly with target-specific styling (Word OpenXML native superscript runs, PowerPoint Unicode superscripts via `_clean_slide_text` in both narrative and callout cards, HTML CSS `sup.citation`). HTML renderer restores anchor attributes (`href`, `id`, `name`) faithfully, emits `id="ref-N"` targets on reference paragraphs, and prevents bare carets from corrupting mathematical exponents and units ($R^2$, $m^2$).
4. **Multi-Agent Mirror Synchronization**:
   - Maintained byte-identical synchronization across canonical `skills/` and all 4 platform mirrors (`.agent/`, `.agents/`, `.claude/`, `.cursor/`, and `packaging/cloud/shinystat-cloud/`) via `./scripts/install-skills.sh --sync`.

### Consequences
- **Status**: Accepted & Verified.
- **Evidence Integrity**: Eliminates phantom citations and unsupported clinical claims from automated manuscript pipelines.
- **Publication Parity**: Generates seamless superscript citations across HTML, Word, PowerPoint, and Markdown matching ICMJE standards.
- **Traceability Coherence**: Citation numbers no longer trigger false-positive traceability failures during Pillar 5 audits.

[MEMORY_LEARN: Incorporating an optional Polish Grilling Gate with websearch/MCP literature verification enforces an anti-phantom citation invariant and aligns narrative claims with empirical data, while structural token recognition preserves numerical traceability in Pillar 5 audits.]

---

## ADR 48: User-Scoped References in Standard Mode vs Relevant Literature Enrichment in Full Polish with Mandatory Online Empirical Comparison

### Context
Following ADR 47, Gate #5 offered a binary choice between `(Standard) Skip Polish` and `(Optional) Full Publication Polish`. However, two ambiguities remained under `/writing-for-agents`:
1. **Unsolicited References Risk**: In Standard Mode, the scope of references was loosely phrased as "using user-provided or existing local sources", leaving room for agents to spontaneously insert external literature citations or exploratory references that the user never asked for.
2. **Empirical Evidence Grounding Gap**: When discussing statistical findings in Discussion or Results, agents could rely on generic priors or unverified assumptions rather than real empirical research benchmarks. Furthermore, even when users supplied local full-text documents or notes, references lacked cross-comparison against live primary publications online.

### Decision
1. **User-Scoped References in Standard Mode**:
   - In `(Standard) Skip Polish (Direct Fast-Path / User-Scoped References)`:
     - The Reference list is strictly scoped to references explicitly provided by the user in the prompt, notes, study protocol, or data dictionary.
     - When the user provides references, require authentic bibliographic metadata and direct claim support for each source; extract and present real empirical research figures (sample sizes, benchmark effect sizes, control event rates, 95% CIs) as comparable ONLY when the source actually reports them, matches the target estimand, population, outcome, and time horizon, and directly supports Discussion claims (citing non-empirical sources or sources with different clinical contexts for the claims or criteria they support without inventing or presenting non-comparable figures); online comparison against real research data is mandated only when the source reports relevant empirical findings; inserts sequential superscript citations (`<sup>1</sup>`, `<sup>2,3</sup>`), and formats a complete **References** section in the chosen **Journal House Style** (`NEJM`, `JAMA`, `Lancet`, `BMJ`, `Clinical`).
     - If the user provides NO references, standard deliverables omit the external literature References section entirely (reporting only statistical findings, tables, figures, and methodology).
     - Prohibit all spontaneous or unsolicited external literature citations in Standard Mode.
2. **Dynamic Bibliographic MCP Routing & Online Empirical Comparison**:
   - The agent dynamically discovers and prioritizes any connected literature or medical database MCP servers (e.g., PubMed MCP, PRISMA MCP, NEJM, Cochrane, Semantic Scholar, ScienceDirect, `medical-terminology`), seamlessly falling back to `search_web` / web queries when dedicated MCP servers are not mounted.
   - Enforce the **Zero-PHI Literature Query Boundary**: never transmit raw clinical notes, patient excerpts, or Protected Health Information (HN, patient name, DOB, Thai National ID, phone) to external literature MCPs or `search_web`. External queries must be constructed strictly from verified bibliographic metadata and de-identified medical topic descriptors.
   - For every user-provided reference or cited paper (even when the user supplies a local full-text document or clinical notes):
     - Query available MCPs or online sources if searchable to verify the cited source itself using bibliographic fields appropriate to the source type: for journal articles, full author list, title, journal, year, volume, and pages (or article number / e-locator when pages are not assigned), and DOI (when assigned); for clinical practice guidelines, consensus statements, government reports, or standards, issuing body/organization, title, year, version/report identifier, and official URL when no DOI exists; and for books or older publications, authors/editors, title, publisher, and year. Verify that the source directly supports the Discussion claim.
     - Extract and present real empirical research findings from published literature as comparable when reported and directly supporting the Discussion claim matching its estimand, population, outcome, and time horizon (citing non-empirical sources without figures for supported claims).
     - When literature services are unavailable or unreachable, authenticated local full text with source-appropriate bibliographic metadata satisfies verification ONLY when carrying an accepted provenance signal (user-provided file in workspace/prompt, publisher frontmatter/header, DOI/PMID header, or official organization publication cover); if neither online verification nor authenticated local text with provenance is available, halt explicitly.
3. **Relevant Literature Enrichment in Full Publication Polish**:
   - In `(Optional) Full Publication Polish (Execute Gate #6 / Relevant Literature Enrichment)`:
     - Execute all standard verifications, empirical Discussion grounding, superscript citations, and House Style References on user-provided sources.
     - In addition, actively conduct exploratory literature retrieval across available bibliographic MCPs / `search_web` to identify and integrate additional authentic, relevant published studies, clinical trials, and guidelines that directly match the user's research topic, population, and findings.
     - Extract and present real clinical benchmarks and comparison figures as comparable ONLY when reported by the source, matching the target estimand, population, outcome, and time horizon, and directly supporting a specific Discussion claim (otherwise cite the source for its conceptual claims without figures or omit it), format sequential superscript citations (`<sup>1</sup>`, `<sup>2,3</sup>`), and assemble a comprehensive References block in the locked Journal House Style.
   - Enforce the **Anti-Phantom Citation Invariant** strictly across all modes (failing closed with an explicit halt if verification is unavailable or conflicting).
4. **Runtime Requirements Pinning & Headless PDF Rendering**:
   - Pin all runtime dependencies in `skills/shinystat/requirements.txt` to exact tested versions (including `openpyxl==3.1.5` and `pyarrow==25.0.1` for Excel and Parquet data ingestion alongside `pandas`) to guarantee reproducible biostatistical execution across runs.
   - Document the Playwright Chromium installation requirement (`uv run --with playwright python -m playwright install chromium`, with `--with-deps` on Linux) to prevent headless PDF export failures in environments where Playwright is not pre-installed.
5. **Synchronization across Mirrors**:
   - Maintain canonical Single Source of Truth in `skills/` and propagate across all agent platform mirrors (`.agents/`, `.agent/`, `.claude/`, `.cursor/`, and `packaging/cloud/shinystat-cloud/`) via `./scripts/install-skills.sh --sync`.

### Consequences
- **Status**: Accepted & Verified.
- **Reference Cleanliness**: Eliminates uninvited external citations in standard deliverables while preserving seamless references when users provide them.
- **Empirical Rigor**: Ensures narrative Discussion points are compared against authentic real-world published data matching estimand and time horizon rather than hallucinations or model priors, while respecting non-empirical sources and guidelines lacking DOIs.
- **Verification Determinism**: Clearly defines verification outcomes when literature services are unavailable, permitting authenticated local full text with source-appropriate metadata and halting when verification cannot be achieved.
- **Execution Reproducibility**: Exact version pins covering CSV, Excel, and Parquet data sources eliminate subtle numerical differences or import breakages from future unpinned package resolutions.

[MEMORY_LEARN: In clinical report generation, standard reference lists must be strictly scoped to user-provided sources with online empirical comparison against authentic published data, while exploratory literature enrichment is reserved for Full Publication Polish.]

---

## ADR 49: Repository Deduplication, Progressive Disclosure Routing, Shared References, and Self-Healing CLI Ergonomics

### Context
Following rapid platform expansion across multiple AI agent harnesses (`.agent/`, `.agents/`, `.claude/`, `.cursor/`), the codebase suffered from 6x duplicate copies of Python code and skill instructions, leading to context pollution, split-brain drift risks, and high token overhead. Furthermore:
1. `skills/shinystat-cloud/references/python-recipes.md` was a 2,500+ line monolith that forced agents to ingest hundreds of unneeded lines into their context window.
2. 6 shared reference documents were duplicated 11 times across local and cloud skill directories.
3. Agents executing declarative YAML analysis plans lacked pre-flight syntax and schema validation, leading to mid-run runtime exceptions without structured remediation hints.

### Decision
1. **Repository Deduplication & SSOT Architecture (Phase 1)**:
   - Establish `src/medstat/` as the sole canonical source for all Python logic, and `skills/` as canonical source for skill prompts.
   - Centralize sub-agent definition at root `agents/figure-tester.md`.
   - Update `scripts/install-skills.sh` to install via relative symlinks (with `--symlink` default and `--copy` fallback).
   - Ignore build/mirror harnesses (`.agent/`, `.agents/`, `.claude/`, `.cursor/`, `packaging/`, `skills/shinystat/scripts/medstat/`) in `.gitignore`, `.ignore`, and `.cursorignore`, removing 517 duplicate entries from git index.
2. **Progressive Disclosure Cloud Recipes (Phase 2)**:
   - Partition monolithic `python-recipes.md` into 5 domain-specific files under `skills/shinystat-cloud/references/recipes/`:
     - `01-data-missing-table1.md` (Recipes 1, 8, 12, 13)
     - `02-regression-firth-ordinal.md` (Recipes 7, 9, 10)
     - `03-survival-competing-risks.md` (Recipes 11, 15)
     - `04-causal-iptw-psm.md` (Recipes 6, 14)
     - `05-diagnostic-meta-power.md` (Recipes 2, 3, 4, 5, 16, 17, 18, 19, 20)
   - Add a Progressive Disclosure Routing Table in `skills/shinystat-cloud/SKILL.md` to load only the required domain recipe.
   - Maintain dynamic concatenation in `package-cloud-skill.sh` and snapshot test fixtures for backward compatibility.
3. **Consolidated Shared References & Test Entrypoints (Phase 3)**:
   - Centralize the 6 shared clinical reference markdown files into `skills/_shared/references/` (`decision-heuristics.md`, `figures.md`, `flows.md`, `house-styles.md`, `illustration.md`, `report-builder.md`).
   - Deduplicate `scripts/test_all_figures_matrix.py` into a thin CLI runner delegating to `tests/unit/test_all_figures_matrix.py`.
4. **Self-Healing CLI Ergonomics & Pre-Flight Validation (Phase 4)**:
   - Implement `--schema` flag to emit the standard declarative YAML plan template.
   - Implement `--validate-only <plan.yaml>` to verify YAML syntax, required top-level keys, and dataset column availability without fitting models.
   - Provide structured `agent_remediation` error hints across CLI output, `IntegrityReport` (`integrity.py`), `fit_standard_logistic` (`glm.py`), and `schoenfeld_diagnostics` (`survival.py`).
   - Update `skills/shinystat/SKILL.md` requiring agents to run pre-flight `--validate-only` before starting analysis execution.

### Consequences
- **Status**: Accepted & Fully Verified.
- **Context Pollution Eliminated**: Code search tools (`ripgrep`, `grep`, `glob`) scan only canonical source files, reducing prompt tokens and search hallucination.
- **Progressive Token Savings**: Cloud sandbox agents load ~80% fewer recipe tokens per task by targeting specific domain modules.
- **Ergonomic Self-Healing**: Agents detect YAML syntax errors, missing columns, or PH violations with explicit, actionable remediation steps.
- **Zero Statistical Regression**: 100% of unit tests (466), stress tests (82), e2e tests (150), and baseline snapshot tests pass with identical mathematical values.

[MEMORY_LEARN: Deduplicating multi-agent skill harnesses via relative symlinks and progressive disclosure recipe routing reduces context pollution and token burn while preserving 100% statistical parity.]

---

## ADR 50: Production-Grade Pre-Flight Validation, Standalone Recipe Boundary Guards, and Statistical Invariant Hardening (PR #22 Review)

### Context
During code review for PR #22 (`refactor/repo-dedup-progressive-disclosure`), a comprehensive static and boundary audit identified edge-case handling gaps, schema divergence, and recipe boundary assumptions across CLI execution, biostatistical calculations, and cloud skill packaging:
1. `medstat model <spec.yaml>` caught schema validation errors in `AnalysisPlan.from_yaml(spec)` but lacked wrapping around the direct execution invocation (`plan.execute()`), which could raise uncaught runtime exceptions without emitting structured actionable remediation hints.
2. In `preflight_validate_spec`, YAML parsing permitted duplicate mapping keys, and dataset column extraction invoked `load_clinical_data` / `pd.read_csv`, loading entire large clinical cohorts into memory merely to check header names. Additionally, dataset path resolution used `spec_path.parent` instead of matching `main.py` execution resolution (`Path.cwd()`).
3. The `--schema` YAML template diverged from the executable `AnalysisPlan` dataclass (`project:` vs `metadata:` and `models:`).
4. Across the standalone cloud recipes (`01-data-missing-table1.md` through `05-diagnostic-meta-power.md`), numerical boundary protections were incomplete:
   - Little's MCAR EM loop did not check `max_iter >= 1`, allowed infinite values, and returned p-value on non-convergence.
   - Rubin pooling returned p = 1.0 on zero pooled SE with non-zero estimate.
   - Firth logistic clipped linear predictors with `np.clip` rather than using numerically stable log-likelihood and expit, and did not recompute information matrix at final beta.
   - Fine-Gray competing risks regression lacked explicit row retention metadata and did not return `fg_se_method` and `fg_score_gap` as serialized result columns.
   - Causal IPTW/PSM lacked categorical dummy encoding in `compute_iptw`, permitted non-binary treatment/outcome values, and allowed negative caliper SD.
   - Prediction model validation/DCA lacked empty complete-case cohort rejection, non-finite threshold rejection, zero-residual ICC(2) CI handling, and failed logistic recalibration fallback reporting.

### Decision
1. **Pre-Flight Ergonomics & Low-Overhead Column Inspection**:
   - Implemented `get_clinical_columns(path)` in `src/medstat/data/loader.py` to inspect column headers from CSV (chunksize 0/header line), Excel (openpyxl read_only max_row=1), and Parquet (metadata schema) without loading full datasets.
   - Added `_UniqueKeyLoader` in `src/medstat/cli/spec.py` to reject duplicate keys across YAML loading.
   - Hardened `preflight_validate_spec` to validate executable sections (`metadata`, `data`, `models`), resolve dataset paths against `Path.cwd()` first, and validate all model-referenced options (`p_recal`, `p_pred`, `time_col`, etc.).
   - Wrapped `plan.execute()` in `main.py` with `try...except Exception as e: emit_agent_actionable_error(e); raise SystemExit(1)`.
2. **Modular Recipe Boundary Hardening**:
   - `01-data-missing-table1.md`: Absolute SMD balance criterion ($|\text{SMD}| < 0.10$); discrete-code guard up to 5 levels; non-finite MCAR rejection; non-convergence p-value `np.nan`; Rubin pooling deterministic limit for zero SE.
   - `02-regression-firth-ordinal.md`: Removed `np.clip`; expit and stable logaddexp likelihood; recomputed information matrix at final beta; strict E-value validation; Logit outcome binary validation and convergence checks.
   - `03-survival-competing-risks.md`: Complete input validation in CIF and Fine-Gray; retention counts in `out.attrs["retention"]`; explicit `fg_se_method`, `fg_score_gap`, and confidence level columns in output rows.
   - `04-causal-iptw-psm.md`: Dummy encoding for categorical covariates in IPTW; binary treatment/outcome checks; non-negative caliper SD validation; `matched_X` schema reindexing; robust model type error.
   - `05-diagnostic-meta-power.md`: Wilson score count checks; 2x2 integer checks; empty DCA cohort & threshold checks; ICC(2) zero-residual CI fallback; tdROC integer event validation; bootstrap binary outcome validation, `cox_snell_r2` return, and zero-convergence fallback; recalibration explicit fallback reporting; Riley sample size positive finite margin validation.
3. **Canonical Reference Linking & Packaging Synchronization**:
   - Updated all links in `skills/shinystat-cloud/SKILL.md` to point to `../_shared/references/` and `references/recipes/`.
   - Updated `test_recipes_snapshot.py`, `test_shinystat_cloud.py`, and `test_shinystat_skill.py` to point to canonical modular recipes under `skills/shinystat-cloud/references/recipes/`.
   - Synchronized packaging through `scripts/package-cloud-skill.sh`.

### Consequences
- **Status**: Accepted & Fully Verified (699/699 tests passing).
- **Safety**: Complete numerical and boundary guardrails across all 20 biostatistical recipes.
- **Memory Efficiency**: CLI pre-flight checks execute with zero dataset memory footprint.
- **Resilience**: Agents receive actionable, structured error hints on invalid YAML or data specs before long-running model execution begins.

[MEMORY_LEARN: Pre-flight YAML and schema validation should inspect dataset schemas/headers using low-overhead metadata probes rather than full cohort loads, preventing out-of-memory overhead during agent planning.]






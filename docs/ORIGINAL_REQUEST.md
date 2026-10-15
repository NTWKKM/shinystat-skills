# Original User Request

## Initial Request — 2026-09-23T14:24:42Z

# Teamwork Project Prompt

Build and package `medstat-core`, a headless Python biostatistical calculation engine and CLI tool with 5 self-contained Agent Skills, decoupled from `/Users/ntwkkm/shiny-stat` and delivered into the clean workspace `/Users/ntwkkm/shinystat-skills` for public GitHub release.

Working directory: `/Users/ntwkkm/shinystat-skills`  
Source reference: `/Users/ntwkkm/shiny-stat`  
Integrity mode: development

---

## Requirements

### R1. Headless Biostatistical Engine (`medstat-core`)
- Decouple core statistical and calculation routines from `/Users/ntwkkm/shiny-stat/utils/` into a pure, headless Python library under `src/medstat/`.
- Strip all GUI/Shiny ties (`shiny.ui` notifications removed; `tabs._common.get_color_palette` centralized in `medstat.theme.palette`; `config` and `logger` made headless).
- Implement pure-Python / SciPy two-way ANOVA for Intraclass Correlation Coefficient (ICC) to eliminate GPL `pingouin` dependency, preserving a permissive **Apache-2.0** license.
- Run primary models in pure Python (`firthmodels >= 0.8.2` for penalized logistic and Cox with profile likelihood CIs, `rcs_lib.py` for splines). Retain R scripts strictly in `tests/benchmarks/r_scripts/` as an offline validation oracle.
- Modern packaging via `pyproject.toml` with `requires-python = ">=3.12"`, CLI script entry point `medstat = "medstat.cli:main"`, and optional extras `[causal]` (`econml`, `psmpy`) and `[pdf]` (`playwright`).

### R2. Clinically Sound Data Cleaning & Audited Sample Flow
- Audit missingness per variable with counts, percentages, and patterns.
- Enforce explicit strategy selection (`complete-case`, `mice`, `knn`, `indicator`) requiring documented clinical justification (no silent listwise deletion or MCAR assumptions).
- Track and emit an audited sample retention flow for every analysis:
  $$N_{initial} \to N_{excluded} \to N_{analyzed}$$
  recording specific reasons for exclusions (CONSORT/STROBE participant flow).

### R3. Unified CLI & Statistical Analysis Plan (SAP) Spec Engine
- Implement the `medstat` CLI with subcommands: `clean`, `table1`, `model`, `diag`, `causal`, `meta`, `agreement`, `sample-size`, and `report`.
- For model estimation (`medstat model`), accept an explicit YAML/JSON Statistical Analysis Plan (`analysis_plan.yaml`) defining outcome, exposure, covariates, data types, reference categories, interaction terms, spline knots, and missing data strategy.
- Generate self-documenting CLI help text as the single source of truth.

### R4. Five Self-Contained Agent Skills (`skills/`)
- Author 5 atomic skills in `skills/` designed strictly under `writing-for-agents`:
  1. `medstat-clean`: Missing audit, explicit imputation, outlier winsorization, sample flow tracking.
  2. `medstat-models`: Table 1 with SMDs, GLM, Cox PH, Schoenfeld test, Firth regression, RCS splines, E-value sensitivity.
  3. `medstat-diagnostic`: 2x2 accuracy (Sens/Spec/PPV/NPV/LR), ROC with DeLong 95% CIs and comparisons, DCA net benefit, Calibration.
  4. `medstat-causal-meta`: PSM balance & love plots, Bland-Altman, Pure-SciPy ICC, Meta-analysis $I^2$ and Forest plots.
  5. `medstat-report`: NEJM/JAMA/APA 7 publication HTML tables, Methods narrative generation, STROBE/CONSORT/TRIPOD audits.
- Each skill must have its own localized `references/` directory (fully portable, zero shared relative path breaks).
- Frontmatter descriptions strictly under 1024 characters with front-loaded leading words.
- Provide a setup script `scripts/install-skills.sh` for Antigravity, Claude Code, and Cursor environments.

### R5. Objective Verification & Parity Test Suite
- Port unit regression test suite from `shiny-stat/tests/unit/` to `tests/unit/` to ensure parity.
- Numerical parity benchmark against R oracle `test_firth.R` on benchmark datasets (`sex2`, `breast`).
- End-to-end task execution verifying the 5 skills against synthetic clinical fixtures.

---

## Acceptance Criteria

### Decoupling & Core Engine
- [ ] Zero references to `shiny`, `shiny.ui`, or `tabs.` in `src/medstat/`.
- [ ] Intraclass Correlation Coefficient (ICC) runs via pure SciPy/NumPy without importing `pingouin`.
- [ ] Package installs cleanly with `pip install -e .` under Python 3.12+.
- [ ] License file is valid Apache-2.0 and no GPL-infected code is present in core.

### Clinical Data Safety & Missingness
- [ ] Data cleaning module raises an error if missing data strategy is not explicitly specified.
- [ ] Sample retention flow metadata ($N_{initial} \to N_{excluded} \to N_{analyzed}$) is emitted in all analysis results.

### CLI & Spec Engine
- [ ] `medstat --help` and all subcommand help menus execute successfully with exit code 0.
- [ ] `medstat model --spec analysis_plan.yaml` successfully fits models with explicit reference levels and interaction terms without error.

### Skills & Agent Usability
- [ ] All 5 `SKILL.md` files validate YAML frontmatter and contain valid markdown without broken relative links.
- [ ] Each skill has its own localized `references/` directory.
- [ ] Descriptions are under 1024 characters with front-loaded leading words.

### Tests & Verification
- [ ] Ported unit test suite in `tests/unit/` passes 100% via `pytest`.
- [ ] Numerical parity test against `tests/benchmarks/` matches within $10^{-4}$ tolerance.
- [ ] 5 mock clinical workflow tasks execute end-to-end through CLI without errors.

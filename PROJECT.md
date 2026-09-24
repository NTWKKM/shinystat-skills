# Project: medstat-core

## Architecture
`medstat-core` is a pure headless Python biostatistical calculation engine and unified CLI tool with 5 self-contained Agent Skills, decoupled from `shiny-stat` and packaged for public GitHub release under Apache-2.0.

```
medstat-core/
├── pyproject.toml              # PEP 517/621, requires-python >= 3.12, CLI entrypoint, extras [causal], [pdf]
├── LICENSE                     # Apache-2.0 License (permissive, zero GPL pingouin copyleft)
├── README.md                   # Clean project documentation & CLI quickstart
├── src/
│   └── medstat/
│       ├── __init__.py         # Package root, version __version__ = "0.1.0"
│       ├── config.py           # Headless configuration
│       ├── logging.py          # Structured logging without Shiny UI ties
│       ├── theme/
│       │   └── palette.py      # Centralized color palettes (decoupled from tabs._common)
│       ├── data/
│       │   ├── clean.py        # Missingness audit, winsorization, Little's MCAR
│       │   ├── missing.py      # Imputation (MICE with Rubin's rules, KNN, indicator, complete-case)
│       │   ├── retention.py    # SampleFlowTracker (N_init -> N_excl -> N_anal) & ASCII flow diagrams
│       │   └── quality.py      # Multi-dimensional data quality scoring
│       ├── stats/
│       │   ├── descriptive.py  # Parametric & non-parametric summary stats
│       │   ├── bivariate.py    # T-test, Mann-Whitney, Chi-square, Fisher's exact, ANOVA, Kruskal
│       │   └── correlation.py  # Pearson, Spearman, Kendall
│       ├── models/
│       │   ├── glm.py          # Linear, Logistic, Poisson, Negative Binomial (statsmodels)
│       │   ├── survival.py     # Kaplan-Meier, Cox PH, Schoenfeld residuals (lifelines)
│       │   ├── firth.py        # Firth penalized logistic & Cox (firthmodels >= 0.8.2) with PL CIs & LRT
│       │   ├── splines.py      # Restricted cubic splines (rcs_lib.py via patsy)
│       │   └── sensitivity.py  # E-value calculations for unmeasured confounding
│       ├── diagnostic/
│       │   ├── accuracy.py     # 2x2 contingency, Sens, Spec, PPV, NPV, PLR, NLR, Wilson score CIs
│       │   ├── roc.py          # ROC curves, AUC, Youden index, DeLong 95% CIs and comparisons
│       │   ├── dca.py          # Decision Curve Analysis (net benefit across decision thresholds)
│       │   └── calibration.py  # Calibration curves, Brier score, Hosmer-Lemeshow
│       ├── causal/
│       │   ├── psm.py          # Propensity score matching, caliper, nearest neighbor
│       │   └── balance.py      # Standardized Mean Differences (SMD), Love plots
│       ├── meta/
│       │   ├── models.py       # Fixed-effect, Random-effects (DerSimonian-Laird), I^2, Tau^2
│       │   ├── forest.py       # Forest plot data generation
│       │   └── bias.py         # Funnel plots, Egger's regression test, Begg's test
│       ├── agreement/
│       │   ├── icc.py          # Pure SciPy/NumPy two-way ANOVA ICC (ICC1, ICC2, ICC3 single & avg, 95% CIs)
│       │   └── bland_altman.py # Bland-Altman mean difference, limits of agreement (LoA)
│       ├── power/
│       │   └── sample_size.py  # Sample size & power calculations
│       ├── reporting/
│       │   ├── table1.py       # Baseline characteristics Table 1 with SMDs and auto-testing
│       │   ├── tables.py       # APA 7, NEJM, JAMA HTML table formatters
│       │   └── narrative.py    # Automated Statistical Methods narrative generation
│       └── cli/
│           ├── __init__.py
│           ├── main.py         # Click group: clean, table1, model, diag, causal, meta, agreement, sample-size, report
│           └── spec.py         # SAP Spec Engine: YAML/JSON analysis_plan.yaml validation & execution
├── skills/
│   ├── medstat-clean/          # Missingness audit, explicit imputation, sample flow tracking
│   │   ├── SKILL.md
│   │   └── references/
│   ├── medstat-models/         # Table 1, GLM, Cox PH, Firth regression, RCS splines, E-values
│   │   ├── SKILL.md
│   │   └── references/
│   ├── medstat-diagnostic/     # 2x2 accuracy, ROC with DeLong, DCA net benefit, calibration
│   │   ├── SKILL.md
│   │   └── references/
│   ├── medstat-causal-meta/    # PSM balance, Love plots, pure-SciPy ICC, meta-analysis I^2
│   │   ├── SKILL.md
│   │   └── references/
│   └── medstat-report/         # NEJM/JAMA/APA 7 tables, Methods narrative, STROBE/CONSORT audits
│       ├── SKILL.md
│       └── references/
├── scripts/
│   └── install-skills.sh       # Multi-platform installer for Antigravity, Claude Code, Cursor
└── tests/
    ├── conftest.py             # Pytest forwarding wrapper (Python 3.9 -> .venv Python 3.12)
    ├── fixtures/               # Synthetic clinical datasets for E2E workflows
    ├── unit/                   # 541 ported unit tests across all analytical modules
    ├── benchmarks/             # R oracle parity benchmarks (firth logistic/cox, datasets sex2/breast)
    └── e2e/                    # 5 mock clinical workflow CLI tests
```

---

## Feature Inventory
Every feature from the Survey phase is enumerated here with its assigned milestone.

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Headless Statistical Core | Decouple all statistical routines into pure headless `src/medstat/` with zero Shiny/UI ties | M1 | Survey 1 |
| 2 | Modern Packaging & pyproject.toml | PEP 517/621 packaging with `requires-python = ">=3.12"`, optional extras `[causal]`, `[pdf]`, CLI entrypoint | M1 | Survey 1, 3 |
| 3 | Apache-2.0 License & Pingouin Elimination | Replace GPL pingouin with pure-Python/SciPy two-way ANOVA ICC (all 6 variants + CIs) | M1 | Survey 1, 3 |
| 4 | Centralized Palette & Styling | Relocate color palettes into `medstat.theme.palette`, strip notifications and GUI tags | M1 | Survey 1 |
| 5 | Firth Penalized Models & Splines | Primary pure-Python `firthmodels >= 0.8.2` (logistic, Cox) with PL CIs & LRT, plus RCS splines | M1 | Survey 1 |
| 6 | Pytest Forwarding Wrapper | `conftest.py` wrapper delegating system Python 3.9 to `.venv/bin/pytest` 3.12 with loop guard | M1 | Survey 3 |
| 7 | Missingness Audit & Little's MCAR | Missing counts, percentages, patterns, and Little's test in `medstat.data.clean` | M2 | Survey 2 |
| 8 | Explicit Missingness Strategy Gate | Raise `MissingStrategyRequiredError` if strategy not specified with clinical justification | M2 | Survey 2 |
| 9 | Audited Sample Retention Flow | `SampleFlowTracker` tracking $N_{initial} \to N_{excluded} \to N_{analyzed}$ with reasons & ASCII diagrams | M2 | Survey 2 |
| 10 | Imputation & MICE Rubin Pooling | Single imputation (KNN, indicator, complete-case) and MICE with Rubin's rules pooling | M2 | Survey 2 |
| 11 | Unified Click CLI Subcommands | Subcommands: `clean`, `table1`, `model`, `diag`, `causal`, `meta`, `agreement`, `sample-size`, `report` | M3 | Survey 2 |
| 12 | SAP Spec Engine (analysis_plan.yaml) | YAML/JSON specification parser validating types, reference levels, interactions, splines, missingness | M3 | Survey 2 |
| 13 | Self-Documenting Help Menus | Complete CLI `--help` for root and all subcommands returning exit code 0 | M3 | Survey 2 |
| 14 | 5 Agent Skills Packaging | Author atomic skills under `skills/` adhering strictly to `writing-for-agents` | M4 | Survey 3 |
| 15 | Localized Skill References | Portable `references/` directories per skill with zero broken relative paths | M4 | Survey 3 |
| 16 | Multi-Platform Skill Installer | `scripts/install-skills.sh` supporting Antigravity, Claude Code, and Cursor environments | M4 | Survey 3 |
| 17 | Ported Unit Regression Tests | Port all unit tests from `shiny-stat/tests/unit/` to `tests/unit/` with 100% pass rate | M5 | Survey 3 |
| 18 | R Oracle Parity Benchmarks ($10^{-4}$) | Numerical parity benchmarks against R `test_firth.R` on `sex2` and `breast` datasets within $10^{-4}$ | M5 | Survey 3 |
| 19 | 5 Mock Clinical Workflow CLI E2E Tests | End-to-end execution of 5 clinical workflows using synthetic fixtures via CLI | M5 | Survey 3 |

---

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Core Engine Decoupling & Pure-Python ICC | `pyproject.toml`, LICENSE, `conftest.py`, `src/medstat/` core stats, models (Firth, RCS), diagnostic, causal, meta, agreement (pure SciPy ICC), power, theme, reporting | none | IN_PROGRESS |
| M2 | Clinically Sound Data Cleaning & Audited Sample Flow | `medstat.data` (clean, missing, retention, quality), explicit missingness strategy enforcement, `SampleFlowTracker` | M1 | PLANNED |
| M3 | Unified CLI & SAP Spec Engine | `medstat.cli` (`main.py`, `spec.py`), click subcommands, `analysis_plan.yaml` validator and execution engine | M1, M2 | PLANNED |
| M4 | Five Self-Contained Agent Skills & Installer | `skills/` (5 skills with localized references, frontmatter < 1024 chars), `scripts/install-skills.sh` | M1, M2, M3 | PLANNED |
| M5 | Test Suite, R Parity Benchmarks & E2E Workflows | Port unit tests to `tests/unit/`, R benchmark parity in `tests/benchmarks/`, 5 clinical workflow CLI E2E tests | M1, M2, M3, M4 | PLANNED |

---

## Interface Contracts

### 1. `medstat.agreement.icc` ↔ Downstream Callers
```python
def calculate_icc(
    df: pd.DataFrame,
    targets: str,
    raters: str,
    ratings: str,
    icc_type: str = "ICC2",
) -> pd.DataFrame:
    """
    Returns pd.DataFrame with columns:
    ['Type', 'Description', 'ICC', 'F', 'df1', 'df2', 'pval', 'CI95%']
    Types: ICC1, ICC2, ICC3, ICC1k, ICC2k, ICC3k
    Pure SciPy/NumPy two-way ANOVA, zero pingouin dependency.
    """
```

### 2. `medstat.data.clean` ↔ Data Ingestion
```python
class MissingStrategyRequiredError(Exception):
    """Raised when missing data is present without an explicit, clinically justified strategy."""

def prepare_data_for_analysis(
    df: pd.DataFrame,
    required_cols: list[str],
    numeric_cols: list[str] | None = None,
    handle_missing: str | None = None,  # Must be explicitly 'complete-case', 'mice', 'knn', or 'indicator'
    missing_justification: str | None = None,  # Mandatory string if missing values present
    tracker: SampleFlowTracker | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]: ...
```

### 3. `medstat.data.retention` ↔ Clinical Reporting
```python
class SampleFlowTracker:
    def record_stage(self, stage_name: str, n_remaining: int, n_excluded: int, reason: str): ...
    def get_flow_summary(self) -> dict[str, Any]: ...  # N_initial, N_excluded, N_analyzed, stages
    def render_ascii_flow(self) -> str: ...
```

### 4. `medstat.cli.spec` ↔ CLI Execution
```python
class AnalysisPlan:
    @classmethod
    def from_yaml(cls, path: str | Path) -> "AnalysisPlan": ...
    def execute(self, df: pd.DataFrame) -> dict[str, Any]: ...
```

---

## Code Layout
- Target Workspace Root: `/Users/ntwkkm/shinystat-skills`
- Source Code: `/Users/ntwkkm/shinystat-skills/src/medstat/`
- Skills: `/Users/ntwkkm/shinystat-skills/skills/`
- Scripts: `/Users/ntwkkm/shinystat-skills/scripts/`
- Tests: `/Users/ntwkkm/shinystat-skills/tests/`
- Agent Metadata: `/Users/ntwkkm/shinystat-skills/.agents/`

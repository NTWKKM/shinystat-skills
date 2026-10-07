# shinystat / medstat-core: Headless Biostatistical Engine & Autonomous Agent Skill

`medstat-core` is a pure headless Python biostatistical calculation engine and CLI tool paired with the unified `shinystat` autonomous decision-making agent skill, decoupled from interactive UI frameworks and licensed under the permissive **Apache-2.0** license.

---

## Key Highlights

- **Pure Permissive Licensing (Apache-2.0)**: Replaced copyleft GPL dependencies (specifically `pingouin`) with native, pure-Python / NumPy / SciPy algorithms, including a full two-way ANOVA Intraclass Correlation Coefficient (ICC) implementation covering all 6 variants ($ICC(1,1), ICC(2,1), ICC(3,1), ICC(1,k), ICC(2,k), ICC(3,k)$) with exact F-tests and McGraw & Wong (1996) 95% confidence intervals.
- **Pure Python Firth Penalized Models**: First-class support for penalized logistic and Cox proportional hazards regression via `firthmodels >= 0.8.2` with Profile Likelihood confidence intervals and penalized Likelihood Ratio Tests (LRT), resolving quasi-complete and complete separation in sparse clinical datasets.
- **Restricted Cubic Splines (RCS)**: Non-linear relationship modeling and hazard ratio contrast curves via `patsy.cr` natural cubic splines.
- **Audited Participant Sample Flow**: Explicit missing data handling (`complete-case`, `mice`, `knn`, `indicator`) tracking attrition flows ($N_{initial} \to N_{excluded} \to N_{analyzed}$) conforming to CONSORT, STROBE, and TRIPOD standards.
- **Unified Click CLI & Adaptive Scripting**: Comprehensive subcommands (`clean`, `table1`, `model`, `diag`, `causal`, `meta`, `agreement`, `sample-size`, `report`) accepting Statistical Analysis Plans (`analysis_plan.yaml`), complemented by autonomous Python script adaptation for complex clinical datasets.

---

## Architecture & Code Layout

```
shinystat-skills/
├── pyproject.toml              # PEP 517/621 packaging (requires-python >= 3.12)
├── LICENSE                     # Apache-2.0 License
├── README.md                   # Project documentation
├── ARCHITECTURE.md             # System architecture & structural diary
├── CONTEXT.md                  # Domain vocabulary & naming diary
├── DESIGN.md                   # Architectural Decision Records (ADRs 1–25)
├── skills/                     # Canonical shinystat Agent Skill
│   └── shinystat/
│       ├── SKILL.md            # Autonomous decision core (4 pillars + Grilling Gate)
│       └── references/
│           └── decision-heuristics.md  # Archetype-to-model reference manual
├── .agent/skills/              # Antigravity Workspace mirror
├── .agents/skills/             # Antigravity / Cursor mirror
├── .claude/skills/             # Claude Code mirror
├── .cursor/skills/             # Cursor mirror
├── scripts/
│   └── install-skills.sh       # Automated multi-agent skill installer
├── src/medstat/
│   ├── __init__.py             # Public package exports
│   ├── config.py               # Headless configuration manager
│   ├── logging.py              # Headless structured logging & performance tracking
│   ├── theme/
│   │   └── palette.py          # Centralized clinical theme and palette constants
│   ├── stats/
│   │   ├── descriptive.py      # Parametric and non-parametric summary statistics
│   │   ├── bivariate.py        # T-tests, Mann-Whitney, Chi-square, Fisher, ANOVA, Kruskal
│   │   └── correlation.py      # Pearson, Spearman, Kendall correlation matrices & CIs
│   ├── models/
│   │   ├── glm.py              # Linear (OLS), Logistic, Poisson, Negative Binomial
│   │   ├── survival.py         # Kaplan-Meier, Cox PH, Schoenfeld assumption tests
│   │   ├── firth.py            # Firth penalized logistic & Cox with Profile Likelihood CIs
│   │   ├── splines.py          # Restricted cubic splines (RCS) & HR contrast curves
│   │   ├── ordinal.py          # Proportional odds cumulative logit & Brant test
│   │   ├── multilevel.py       # GEE with robust sandwich SEs & random-intercept MixedLM
│   │   └── sensitivity.py      # E-value sensitivity analysis for unmeasured confounding
│   ├── diagnostic/
│   │   ├── accuracy.py         # 2x2 contingency, Sens, Spec, PPV, NPV, LR+, LR-, DOR, Wilson CIs
│   │   ├── roc.py              # ROC curves, AUC, Youden index, DeLong 95% CIs and comparisons
│   │   ├── dca.py              # Decision Curve Analysis (net benefit across decision thresholds)
│   │   └── calibration.py      # Calibration curves, Brier score, ICI, Hosmer-Lemeshow
│   ├── causal/
│   │   ├── psm.py              # Propensity score matching, caliper, nearest neighbor, IPW
│   │   └── balance.py          # Standardized Mean Differences (SMD), Love plots
│   ├── meta/
│   │   ├── models.py           # Fixed-effect, Random-effects (DerSimonian-Laird), I^2, Tau^2
│   │   ├── forest.py           # Forest plot data generation
│   │   └── bias.py             # Funnel plots, Egger's regression test, Begg's test
│   ├── agreement/
│   │   ├── icc.py              # Pure SciPy/NumPy two-way ANOVA ICC (ICC1, ICC2, ICC3, 95% CIs)
│   │   ├── bland_altman.py     # Bland-Altman mean bias, limits of agreement (LoA) & CIs
│   │   └── kappa.py            # Cohen's and Fleiss' Kappa inter-rater agreement
│   ├── power/
│   │   └── sample_size.py      # Sample size & power calculations (means, proportions, survival)
│   └── reporting/
│       ├── table1.py           # Baseline characteristics Table 1 with SMDs and auto-testing
│       ├── tables.py           # APA 7, NEJM, JAMA HTML and text table formatters
│       ├── narrative.py        # Automated Statistical Methods narrative generation
│       └── checklists.py       # STROBE, CONSORT, TRIPOD, STARD, PRISMA audits
└── tests/
    ├── conftest.py             # Pytest forwarding wrapper & shared fixtures
    ├── fixtures/               # Synthetic clinical datasets
    ├── unit/                   # Comprehensive unit tests & skill parity tests
    ├── benchmarks/             # R oracle parity benchmarks
    └── e2e/                    # Clinical workflow CLI end-to-end tests
```

---

## Autonomous Agent Skill: `shinystat`

This project ships the unified **`shinystat` agent skill** (`skills/shinystat/` and mirrored in `.agent/`, `.agents/`, `.claude/`, `.cursor/`), designed following `writing-for-agents` principles with **Hybrid Progressive Disclosure**. Rather than forcing agents through rigid canned CLI commands or fragmented multi-skill handoffs, `shinystat` empowers coding agents to exercise autonomous biostatistical judgment across 4 core pillars:

### The 4 Decision Pillars

1. **Pillar 1: Data Reality Inspection**: Audits raw geometry, headers, missingness, and candidate sample sizes directly on raw data (`.csv`, `.xlsx`, `.tsv`, `.parquet`) without assuming pre-cleaned tidy shapes.
2. **Pillar 2: Research & Estimand Triangulation**: Triangulates research proposals (PICO/PECO) against 8 clinical study archetypes (Cohort, Case-Control, RCT, Diagnostic Accuracy, Ordinal, Survival, Agreement, Clustered) documented in `references/decision-heuristics.md`.
3. **Pillar 3: Adaptive Python Scripting**: Generates and adapts custom Python scripts combining `medstat` core calculation modules with standard scientific libraries (`pandas`, `scipy.stats`, `statsmodels`, `lifelines`, `scikit-learn`).
4. **Pillar 4: Biostatistical Safety Invariants**: Strictly enforces clinical safety rules:
   - **Numeric 0/1 encoding** for binary endpoints and survival event indicators (accepting explicitly mapped text labels like `"Yes"/"No"` or `"Dead"/"Alive"`, while rejecting unmapped text labels to eliminate clinical event inversion).
   - **Sample retention flow** ($N_{initial} \to N_{excluded} \to N_{analyzed}$) with explicit exclusion tracking.
   - **Zero silent listwise deletion**: Missingness audits (Little's MCAR) and justified handling (`complete-case`, `mice`, `knn`). Primary outcomes are never imputed.
   - **Wilson score 95% CIs** for proportions and 2×2 diagnostic test metrics.
   - **DeLong variance** for empirical ROC AUC comparisons.
   - **Distinct EPV rules**: $\text{EPV}_{\text{binary}} = \frac{\min(N_1, N_0)}{P}$ vs $\text{EPV}_{\text{Cox}} = \frac{N_{\text{failures}}}{P}$.

### Deterministic Grilling Gate

If requirements, outcome directions, missingness strategies, or statistical assumptions are ambiguous, unsupported by data, or affected by severe sparsity (EPV < 10 or quasi-complete separation), the agent **halts immediately** and executes an interactive interview (`❓ Q1 ... ➡️ Recommended`) before proceeding with code execution.

### Skills Directory Structure

Following the open `SKILL.md` standard with progressive disclosure:

```text
skills/shinystat/
├── SKILL.md                          # Autonomous decision core (4 pillars + Grilling Gate)
└── references/
    └── decision-heuristics.md        # Archetype-to-model selection & diagnostic reference
```

### Installing Skills for Your AI Coding Agent

You can install `shinystat` across all platforms using the provided installer script:

```bash
# Automated install into all supported environments (Antigravity, Claude, Cursor)
./scripts/install-skills.sh

# Or target a specific platform / scope:
./scripts/install-skills.sh --target antigravity --scope workspace
./scripts/install-skills.sh --target claude --scope global
```

Pick your platform and follow the corresponding manual setup if needed:

<details>
<summary><strong>Google Antigravity</strong> (Desktop / CLI / IDE)</summary>

**Auto-discovered** — no extra setup needed.

```bash
git clone https://github.com/NTWKKM/shinystat-skills.git
cd shinystat-skills
# Skills in .agent/skills/ are auto-discovered when this workspace is opened
```

Antigravity scans `.agent/skills/` (and `.agents/skills/`) at the workspace root and loads all `SKILL.md` files via progressive disclosure. Simply open the cloned directory as your workspace.

</details>

<details>
<summary><strong>Claude Code</strong> (Anthropic CLI)</summary>

Copy skills into your Claude Code skills directory:

```bash
git clone https://github.com/NTWKKM/shinystat-skills.git
mkdir -p ~/.claude/skills && cp -r shinystat-skills/.agent/skills/* ~/.claude/skills/
# Or for project-scoped:
mkdir -p .claude/skills && cp -r shinystat-skills/.agent/skills/* .claude/skills/
```

Claude Code discovers skills in `~/.claude/skills/` (global) or `.claude/skills/` (project). Each `SKILL.md` with valid YAML frontmatter registers as an available skill.

</details>

<details>
<summary><strong>Claude Web</strong> (claude.ai)</summary>

Upload each skill as a `.zip` file via **Upload a skill**:

```bash
# Option 1: Package dedicated cloud skill for Claude Web (recommended, includes standalone recipes)
./scripts/package-cloud-skill.sh ~/Desktop

# Option 2: Package original clean local skill
(cd skills && zip -r ~/Desktop/shinystat.zip shinystat)
```

Then in [claude.ai](https://claude.ai): Navigate to **Customize** → **Skills** → **+** → **Create skill** → **Upload a skill** and select the `.zip` file (e.g. `shinystat-cloud.zip`). Ensure **Code execution and file creation** is enabled. Claude reads the YAML frontmatter from `SKILL.md` inside the skill archive.

> **Note:** The dedicated cloud package (`shinystat-cloud.zip`) contains pure-Python standalone formulas in `references/python-recipes.md` so that Claude Web can execute all advanced biostatistical analyses directly in its sandbox using standard scientific libraries (`pandas`, `scipy.stats`, `statsmodels`, `lifelines`, `scikit-learn`) without needing `medstat-core` installed.

</details>

<details>
<summary><strong>OpenAI Codex</strong> (CLI)</summary>

Codex uses `AGENTS.md` for project-level instructions. Copy skill content into your project:

```bash
git clone https://github.com/NTWKKM/shinystat-skills.git
cd your-project

# Option A: Symlink the skills directory
mkdir -p .agents
ln -s /path/to/shinystat-skills/.agent/skills .agents/skills

# Option B: Copy and reference from AGENTS.md
mkdir -p .agents/skills
cp -r /path/to/shinystat-skills/.agent/skills/* .agents/skills/
```

Codex discovers `AGENTS.md` and `.agents/` directories by walking up from the CWD. Use `AGENTS.override.md` for machine-specific local tweaks (not committed to VCS).

</details>

<details>
<summary><strong>OpenClaw</strong></summary>

OpenClaw supports the standard `SKILL.md` format:

```bash
git clone https://github.com/NTWKKM/shinystat-skills.git

# Global install
mkdir -p ~/.agents/skills
cp -r shinystat-skills/.agent/skills/* ~/.agents/skills/

# Or workspace-scoped
mkdir -p .agents/skills
cp -r shinystat-skills/.agent/skills/* .agents/skills/

# Verify
openclaw skills list
```

OpenClaw scans `<workspace>/.agents/skills/`, `~/.agents/skills/`, and other configured paths. Skills are registered automatically — restart the agent session if newly added skills are not detected.

</details>

<details>
<summary><strong>Pi</strong> (Coding Agent by Mario Zechner)</summary>

Pi supports the `SKILL.md` open standard:

```bash
git clone https://github.com/NTWKKM/shinystat-skills.git

# Copy into Pi's skills directory
mkdir -p ~/.agents/skills
cp -r shinystat-skills/.agent/skills/* ~/.agents/skills/

# Or project-scoped
mkdir -p .agents/skills
cp -r shinystat-skills/.agent/skills/* .agents/skills/
```

Pi dynamically loads skills when a task matches the skill's description. Supports multiple LLM backends (Anthropic, OpenAI, DeepSeek, Google Gemini).

</details>

<details>
<summary><strong>Hermes</strong> (Nous Research)</summary>

Copy skills into Hermes' skills directory:

```bash
git clone https://github.com/NTWKKM/shinystat-skills.git

# Default location
mkdir -p ~/.hermes/skills
cp -r shinystat-skills/.agent/skills/* ~/.hermes/skills/

# Or configure external_dirs in ~/.hermes/config.yaml:
# skills:
#   external_dirs:
#     - /path/to/shinystat-skills/.agent/skills
```

Hermes detects skills automatically on next session. The `SKILL.md` format is portable across Hermes, Claude Code, and other SKILL.md-compatible agents.

</details>

<details>
<summary><strong>Meta Muse Code</strong></summary>

Muse Code reads skills from `.agents/skills/` at the project root:

```bash
git clone https://github.com/NTWKKM/shinystat-skills.git
cd your-project

# Copy or symlink
mkdir -p .agents/skills
cp -r /path/to/shinystat-skills/.agent/skills/* .agents/skills/
```

Muse Code also reads `AGENTS.md` at the project root for global instructions. Skills use the standard YAML frontmatter `SKILL.md` format.

</details>

### Enforced Clinical Rules (All Platforms)

Regardless of which agent platform you use, `shinystat` enforces:

- **Strict numeric 0/1 encoding**: outcome/event columns must be numeric `0`/`1` — explicitly mapped text labels like `"Dead"`/`"Alive"` or `"Yes"`/`"No"` are converted to 0/1, while unmapped text labels are rejected to eliminate event inversion.
- **Audited sample retention flow**: every analysis tracks $N_{initial} \to N_{excluded} \to N_{analyzed}$ with documented clinical justification; excluding records with missing primary outcomes requires reviewing the protocol's disposition (e.g., ITT vs per-protocol vs sensitivity analysis) rather than relying on a general complete-case justification alone.
- **No silent missing data deletion**: missingness audits (Little's MCAR) and explicit handling strategies are required; primary outcomes are never imputed.
- **Wilson score CIs only** for sensitivity/specificity and binomial proportions (not Wald).
- **DeLong variance only** for empirical ROC AUC confidence intervals and comparisons.
- **Separate EPV rules**: $\text{EPV}_{\text{binary}} = \frac{\min(N_1, N_0)}{P}$ vs $\text{EPV}_{\text{Cox}} = \frac{N_{\text{failures}}}{P}$.
- **Caliper 0.2×SD of logit propensity** for PSM, with Austin (2009) balance requirement (absolute SMD < 0.10).
- **Deterministic Grilling Gate**: mandates an interactive interview whenever ambiguity or conflicting study-data assumptions arise.

---

## Installation

### Option 1: Using `uv` (Recommended)

```bash
# Clone the repository
git clone https://github.com/NTWKKM/shinystat-skills.git
cd shinystat-skills

# Sync environment with dev dependencies
uv sync --all-extras

# Run CLI directly
uv run medstat --help
```

### Option 2: Standard `venv` & `pip`

```bash
# Clone the repository
git clone https://github.com/NTWKKM/shinystat-skills.git
cd shinystat-skills

# Create a virtual environment (Python >= 3.12 required)
python3.12 -m venv .venv
source .venv/bin/activate

# Install editable with dev dependencies
pip install -e '.[dev]'

# Optional extras for causal ML and automated PDF rendering:
pip install -e '.[dev,causal,pdf]'

# Verify CLI
medstat --help
```

---

## Quickstart

### Pure-SciPy Intraclass Correlation Coefficient (ICC)
```python
import pandas as pd
from medstat.agreement.icc import calculate_icc

df = pd.DataFrame({
    "Subject": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    "Rater": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
    "Score": [9, 6, 8, 7, 10, 8, 7, 7, 6, 9],
})

# Calculate all 6 ICC variants with 95% CIs via pure SciPy
icc_table = calculate_icc(df, targets="Subject", raters="Rater", ratings="Score")
print(icc_table[["Type", "Description", "ICC", "F", "df1", "df2", "pval", "CI95%"]])
```

### Firth Penalized Logistic Regression
```python
import numpy as np
import pandas as pd
from medstat.models.firth import fit_firth_logistic

# Dataset with complete separation
X = pd.DataFrame({"age": [20, 25, 30, 35, 40], "biomarker": [0, 0, 1, 1, 1]})
y = pd.Series([0, 0, 1, 1, 1])

res = fit_firth_logistic(y, X, ci_method="pl")
print("Odds Ratios & Profile Likelihood 95% CIs:")
print(res["summary_df"][["estimate", "odds_ratio", "ci_lower", "ci_upper", "p_value"]])
```

---

## Testing

The project includes an automatic forwarding wrapper in `tests/conftest.py` that transparently forwards execution to the Python 3.12 virtual environment (`.venv/bin/pytest`) even if executed from older system Python environments.

```bash
# Run test suite
pytest

# Run numerical parity benchmarks against R oracle
pytest tests/benchmarks/
```

---

## License

Apache-2.0. See `LICENSE` for details.

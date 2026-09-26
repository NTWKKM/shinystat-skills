# medstat-core: Headless Biostatistical Calculation Engine & CLI

`medstat-core` is a pure headless Python biostatistical calculation engine and CLI tool with 6 self-contained Agent Skills, decoupled from interactive UI frameworks and licensed under the permissive **Apache-2.0** license.

---

## Key Highlights

- **Pure Permissive Licensing (Apache-2.0)**: Replaced copyleft GPL dependencies (specifically `pingouin`) with native, pure-Python / NumPy / SciPy algorithms, including a full two-way ANOVA Intraclass Correlation Coefficient (ICC) implementation covering all 6 variants ($ICC(1,1), ICC(2,1), ICC(3,1), ICC(1,k), ICC(2,k), ICC(3,k)$) with exact F-tests and McGraw & Wong (1996) 95% confidence intervals.
- **Pure Python Firth Penalized Models**: First-class support for penalized logistic and Cox proportional hazards regression via `firthmodels >= 0.8.2` with Profile Likelihood confidence intervals and penalized Likelihood Ratio Tests (LRT), resolving quasi-complete and complete separation in sparse clinical datasets.
- **Restricted Cubic Splines (RCS)**: Non-linear relationship modeling and hazard ratio contrast curves via `patsy.cr` natural cubic splines.
- **Audited Participant Sample Flow**: Explicit missing data handling (`complete-case`, `mice`, `knn`, `indicator`) tracking attrition flows ($N_{initial} \to N_{excluded} \to N_{analyzed}$) conforming to CONSORT, STROBE, and TRIPOD standards.
- **Unified Click CLI**: Comprehensive subcommands (`clean`, `table1`, `model`, `diag`, `causal`, `meta`, `agreement`, `sample-size`, `report`) accepting Statistical Analysis Plans (`analysis_plan.yaml`).

---

## Architecture & Code Layout

```
medstat/
├── pyproject.toml              # PEP 517/621 packaging (requires-python >= 3.12)
├── LICENSE                     # Apache-2.0 License
├── README.md                   # Project documentation
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
│   │   └── bland_altman.py     # Bland-Altman mean bias, limits of agreement (LoA) & CIs
│   ├── power/
│   │   └── sample_size.py      # Sample size & power calculations (means, proportions, survival)
│   └── reporting/
│       ├── table1.py           # Baseline characteristics Table 1 with SMDs and auto-testing
│       ├── tables.py           # APA 7, NEJM, JAMA HTML and text table formatters
│       └── narrative.py        # Automated Statistical Methods narrative generation
├── .agent/skills/              # 5 self-contained Agent Skills (SKILL.md + references/)
└── tests/
    ├── conftest.py             # Pytest forwarding wrapper & shared fixtures
    ├── fixtures/               # Synthetic clinical datasets
    ├── unit/                   # Comprehensive unit tests
    ├── benchmarks/             # R oracle parity benchmarks
    └── e2e/                    # Clinical workflow CLI end-to-end tests
```

---

## Agent Skills

This project ships **6 agent skills** in `.agent/skills/` (and `skills/`) that teach AI coding agents how to use the `medstat` library correctly — enforcing clinical safety rules, correct statistical methods, and publication-grade output.

| Skill | Domain | Key Capabilities |
|:------|:-------|:-----------------|
| **medstat-master** | **Master Orchestrator** | **Ingests raw CSV/XLSX without requiring manual skill selection. Automatically audits data, infers clinical study design, formulates or executes a Statistical Analysis Plan (SAP), and orchestrates the downstream skills pipeline.** |
| **medstat-clean** | Data Cleaning | Missingness audit (Little's MCAR), imputation (MICE/KNN/indicator), sample-flow tracking, rejects silent listwise deletion |
| **medstat-models** | Regression & Survival | Table 1 (SMD), GLM/logistic, Cox PH + Schoenfeld, Firth penalized (sparse events), RCS splines, E-value sensitivity |
| **medstat-diagnostic** | Diagnostic Accuracy | 2×2 contingency (Wilson CI), ROC + DeLong CI, paired DeLong biomarker comparison, DCA net benefit |
| **medstat-causal-meta** | Causal & Meta-analysis | PSM (caliper 0.2×SD logit, SMD<0.10 balance), Love plot, ICC (pure SciPy), Bland-Altman, meta-analysis (DL + Egger's) |
| **medstat-report** | Publication Reporting | NEJM/JAMA/APA 7 styled tables, auto Methods narrative, STROBE/CONSORT/TRIPOD checklist audit |

### Skills Directory Structure

Each skill follows the open `SKILL.md` standard:

```text
.agent/skills/<skill-name>/
├── SKILL.md              # YAML frontmatter (name, description) + instructions
└── references/           # Detailed reference documentation
    └── *.md
```

### Installing Skills for Your AI Coding Agent

Skills are **instruction files** that teach AI agents how to call the `medstat` Python library. The Python library itself must also be installed separately (see [Installation](#installation) below).

Pick your platform and follow the corresponding setup:

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
cp -r shinystat-skills/.agent/skills/* ~/.claude/skills/
# Or for project-scoped:
cp -r shinystat-skills/.agent/skills/* .claude/skills/
```

Claude Code discovers skills in `~/.claude/skills/` (global) or `.claude/skills/` (project). Each `SKILL.md` with valid YAML frontmatter registers as an available skill.

</details>

<details>
<summary><strong>Claude Web</strong> (claude.ai Projects)</summary>

Upload each skill as a `.zip` file via **Upload a skill**:

```bash
# Create zip packages for each skill
cd shinystat-skills
for skill_dir in .agent/skills/*/; do
  skill_name=$(basename "$skill_dir")
  cd "$skill_dir" && zip -r ~/Desktop/"${skill_name}.zip" . && cd -
done
```

Then in [claude.ai](https://claude.ai): Project → **Upload a skill** → select each `.zip` file. Claude reads the YAML frontmatter from `SKILL.md` inside each zip.

> **Note:** Claude Web cannot execute Python code — skills serve as instructions only. Copy and run generated code locally.

</details>

<details>
<summary><strong>OpenAI Codex</strong> (CLI)</summary>

Codex uses `AGENTS.md` for project-level instructions. Copy skill content into your project:

```bash
git clone https://github.com/NTWKKM/shinystat-skills.git
cd your-project

# Option A: Symlink the skills directory
ln -s /path/to/shinystat-skills/.agent/skills .agents/skills

# Option B: Copy and reference from AGENTS.md
cp -r /path/to/shinystat-skills/.agent/skills .agents/skills
```

Codex discovers `AGENTS.md` and `.agents/` directories by walking up from the CWD. Use `AGENTS.override.md` for machine-specific local tweaks (not committed to VCS).

</details>

<details>
<summary><strong>OpenClaw</strong></summary>

OpenClaw supports the standard `SKILL.md` format:

```bash
git clone https://github.com/NTWKKM/shinystat-skills.git

# Global install
cp -r shinystat-skills/.agent/skills/* ~/.agents/skills/

# Or workspace-scoped
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
cp -r shinystat-skills/.agent/skills/* ~/.agents/skills/

# Or project-scoped
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
cp -r /path/to/shinystat-skills/.agent/skills/* .agents/skills/
```

Muse Code also reads `AGENTS.md` at the project root for global instructions. Skills use the standard YAML frontmatter `SKILL.md` format.

</details>

### Enforced Clinical Rules (All Platforms)

Regardless of which agent platform you use, these skills enforce:

- **Binary outcome only**: outcome/event columns must be numeric `0`/`1` — text labels (`"Dead"`/`"Alive"`) are rejected
- **No silent missing data deletion**: must explicitly specify `--strategy` (complete-case, mice, knn, indicator)
- **Wilson CI only** for sensitivity/specificity (not Wald)
- **DeLong variance only** for ROC AUC confidence intervals
- **Caliper 0.2×SD of logit propensity** for PSM, with SMD < 0.10 balance requirement

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

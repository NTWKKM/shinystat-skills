# medstat-core: Headless Biostatistical Calculation Engine & CLI

`medstat-core` is a pure headless Python biostatistical calculation engine and CLI tool with 5 self-contained Agent Skills, decoupled from interactive UI frameworks and licensed under the permissive **Apache-2.0** license.

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
├── skills/                     # 5 self-contained Agent Skills
└── tests/
    ├── conftest.py             # Pytest forwarding wrapper & shared fixtures
    ├── fixtures/               # Synthetic clinical datasets
    ├── unit/                   # Comprehensive unit tests
    ├── benchmarks/             # R oracle parity benchmarks
    └── e2e/                    # Clinical workflow CLI end-to-end tests
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/medstat.git
cd medstat

# Create a virtual environment (Python >= 3.12 required)
python3.12 -m venv .venv
source .venv/bin/activate

# Install editable with dev dependencies
pip install -e '.[dev]'

# Optional extras for causal ML and automated PDF rendering:
pip install -e '.[dev,causal,pdf]'
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

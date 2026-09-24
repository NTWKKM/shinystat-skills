"""
tests/conftest.py: Headless Pytest Configuration & Python 3.12 Forwarding Wrapper.

This file provides:
1. An execution forwarding wrapper that detects if pytest was invoked with Python < 3.12
   and forwards execution to `.venv/bin/pytest` with MEDSTAT_PYTEST_FORWARDED=1 guard.
2. Shared clinical test fixtures and benchmarks directory paths.
3. Clean removal of all legacy live Shiny server fixtures.
"""

import os
import subprocess
import sys
from pathlib import Path

# ==============================================================================
# 1. PEP 695 / Python 3.12 Forwarding Interceptor (Compatible with Python 3.8+)
# ==============================================================================
project_root = Path(__file__).resolve().parent.parent
venv_pytest = project_root / ".venv" / "bin" / "pytest"


def _forward_to_venv():
    if os.environ.get("MEDSTAT_PYTEST_FORWARDED"):
        return

    should_forward = sys.version_info < (3, 12)
    if not should_forward:
        try:
            import numpy  # noqa: F401
            import pandas  # noqa: F401
            import pytest  # noqa: F401
        except ImportError:
            should_forward = True

    if should_forward:
        if venv_pytest.exists():
            env = os.environ.copy()
            env["MEDSTAT_PYTEST_FORWARDED"] = "1"
            result = subprocess.run([str(venv_pytest)] + sys.argv[1:], env=env)
            sys.exit(result.returncode)
        elif sys.version_info < (3, 12):
            sys.stderr.write(
                "\n"
                + "=" * 70
                + "\n"
                + "❌ PYTHON VERSION INCOMPATIBILITY ERROR\n"
                + "medstat requires Python >= 3.12.\n"
                + f"Current environment is running Python {sys.version.split()[0]}.\n"
                + f"Virtual environment pytest not found at: {venv_pytest}\n"
                + "Please create a Python 3.12+ virtual environment:\n"
                + "   python3.12 -m venv .venv && source .venv/bin/activate\n"
                + "   pip install -e '.[dev,causal,pdf]'\n"
                + "=" * 70
                + "\n"
            )
            sys.exit(1)


_forward_to_venv()

# ==============================================================================
# 2. Shared Headless Biostatistics Test Fixtures
# ==============================================================================
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(scope="session")
def benchmark_dir() -> Path:
    """Path to offline R benchmark datasets and precomputed results."""
    return Path(__file__).resolve().parent / "benchmarks" / "python_results"


@pytest.fixture(scope="session")
def synthetic_clinical_cohort() -> pd.DataFrame:
    """
    Standard 1,600-patient synthetic clinical dataset containing demographics,
    vitals, lab biomarkers, survival times, and rater agreement measurements.
    """
    np.random.seed(42)
    n = 1600
    return pd.DataFrame(
        {
            "ID": np.arange(1, n + 1),
            "Age": np.random.normal(62, 12, n).clip(18, 95).round(1),
            "Sex": np.random.choice(["Male", "Female"], n, p=[0.52, 0.48]),
            "BMI": np.random.normal(27.5, 4.5, n).clip(15, 50).round(1),
            "Treatment": np.random.choice([0, 1], n, p=[0.5, 0.5]),
            "Diabetes": np.random.choice([0, 1], n, p=[0.75, 0.25]),
            "Hypertension": np.random.choice([0, 1], n, p=[0.60, 0.40]),
            "Outcome_Cured": np.random.choice([0, 1], n, p=[0.45, 0.55]),
            "Time_Months": np.random.exponential(24, n).clip(0.5, 60).round(1),
            "Status_Death": np.random.choice([0, 1], n, p=[0.80, 0.20]),
            "Biomarker_Score": np.random.normal(45, 15, n).round(2),
            "Gold_Standard_Disease": np.random.choice([0, 1], n, p=[0.70, 0.30]),
            "Rater1_Score": np.random.normal(100, 15, n).round(1),
            "Rater2_Score": np.random.normal(102, 16, n).round(1),
        }
    )


if __name__ == "__main__":
    if venv_pytest.exists():
        env = os.environ.copy()
        env["MEDSTAT_PYTEST_FORWARDED"] = "1"
        result = subprocess.run(
            [str(venv_pytest)] + sys.argv[1:], env=env, cwd=os.getcwd()
        )
        sys.exit(result.returncode)
    else:
        sys.exit(0)

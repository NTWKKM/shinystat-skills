"""
tests/e2e/conftest.py: Shared fixtures, paths, and CLI execution helpers for E2E tests.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import pytest
from click.testing import CliRunner

# Ensure src/ is on sys.path for direct module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def oncology_fixture_path() -> Path:
    p = FIXTURES_DIR / "oncology_survival.csv"
    assert p.exists(), f"Missing fixture: {p}"
    return p


@pytest.fixture(scope="session")
def sepsis_fixture_path() -> Path:
    p = FIXTURES_DIR / "sepsis_diagnostics.csv"
    assert p.exists(), f"Missing fixture: {p}"
    return p


@pytest.fixture(scope="session")
def pocus_fixture_path() -> Path:
    p = FIXTURES_DIR / "pocus_reliability.csv"
    assert p.exists(), f"Missing fixture: {p}"
    return p


@pytest.fixture(scope="session")
def cardiovascular_fixture_path() -> Path:
    p = FIXTURES_DIR / "cardiovascular_cohort.csv"
    assert p.exists(), f"Missing fixture: {p}"
    return p


@pytest.fixture(scope="session")
def meta_fixture_path() -> Path:
    p = FIXTURES_DIR / "multicenter_meta.csv"
    assert p.exists(), f"Missing fixture: {p}"
    return p


@pytest.fixture(scope="session")
def incomplete_fixture_path() -> Path:
    p = FIXTURES_DIR / "incomplete_clinical.csv"
    assert p.exists(), f"Missing fixture: {p}"
    return p


@pytest.fixture(scope="session")
def analysis_plan_path() -> Path:
    p = FIXTURES_DIR / "analysis_plan_example.yaml"
    assert p.exists(), f"Missing fixture: {p}"
    return p


@pytest.fixture
def medstat_cli_runner():
    """
    Returns a callable runner for medstat CLI.
    Attempts Click CliRunner first; falls back to subprocess if installed;
    skips gracefully if medstat CLI is not yet implemented (Progressive Testability).
    """

    def _run(args: list[str], catch_exceptions: bool = True) -> Tuple[int, str, str]:
        # 1. Check if medstat.cli.main can be imported
        try:
            from medstat.cli.main import cli

            runner = CliRunner()
            result = runner.invoke(cli, args, catch_exceptions=catch_exceptions)
            stderr = ""
            if result.exc_info and result.exc_info[1]:
                stderr = str(result.exc_info[1])
            return result.exit_code, result.output, stderr
        except ImportError:
            pass

        # 2. Check if medstat CLI entrypoint exists
        cli_main = SRC_DIR / "medstat" / "cli" / "main.py"
        if not cli_main.exists():
            pytest.skip("medstat.cli not yet implemented (scheduled for Milestone M3)")

        venv_medstat = PROJECT_ROOT / ".venv" / "bin" / "medstat"
        if venv_medstat.exists():
            cmd = [str(venv_medstat)] + args
        else:
            cmd = [sys.executable, "-m", "medstat.cli"] + args

        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC_DIR) + ":" + env.get("PYTHONPATH", "")
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
                env=env,
            )
            if proc.returncode != 0 and (
                "No module named 'medstat.cli'" in proc.stderr
                or "No module named medstat.cli" in proc.stderr
            ):
                pytest.skip(
                    "medstat.cli not yet importable (scheduled for Milestone M3)"
                )
            return proc.returncode, proc.stdout, proc.stderr
        except Exception as e:
            pytest.skip(f"medstat CLI not yet runnable: {e}")

    return _run


def require_medstat_module(module_name: str):
    """
    Decorator / helper to skip test if a specific medstat submodule is not yet implemented.
    Ensures Progressive Testability during multi-agent milestone implementation.
    """
    try:
        import importlib

        return importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        if e.name == module_name or (e.name and module_name.startswith(e.name)):
            pytest.skip(f"Required module '{module_name}' not yet implemented: {e}")
        raise

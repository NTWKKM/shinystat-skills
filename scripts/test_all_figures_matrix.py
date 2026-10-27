#!/usr/bin/env python3
"""
scripts/test_all_figures_matrix.py: Lightweight CLI runner wrapping tests/unit/test_all_figures_matrix.py
"""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest


def main() -> int:
    """Run tests/unit/test_all_figures_matrix.py via pytest runner."""
    target_test = REPO_ROOT / "tests" / "unit" / "test_all_figures_matrix.py"
    print("=" * 70)
    print("Running Comprehensive Biostatistical Figure Matrix Verification...")
    print(f"Target: {target_test}")
    print("=" * 70)
    exit_code = pytest.main(["-v", str(target_test)] + sys.argv[1:])
    return int(exit_code)


if __name__ == "__main__":
    sys.exit(main())

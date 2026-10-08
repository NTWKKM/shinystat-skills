#!/usr/bin/env python3
"""
medstat CLI entrypoint bundled with the shinystat agent skill.

Allows direct execution of medstat biostatistical commands without requiring
prior package installation into site-packages.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add scripts directory to sys.path so 'medstat' is directly importable
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from medstat.cli.main import cli
except ImportError as err:
    sys.stderr.write(
        f"Error importing medstat engine from {SCRIPTS_DIR}: {err}\n"
        "Ensure Python dependencies (pandas, scipy, statsmodels, lifelines, firthmodels) are installed.\n"
    )
    sys.exit(1)

if __name__ == "__main__":
    cli()

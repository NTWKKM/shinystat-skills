"""
medstat: Headless Biostatistical Calculation Engine & CLI for Clinical Research.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Clinical Biostatistics Core Team"
__license__ = "Apache-2.0"

from medstat.config import CONFIG, ConfigManager
from medstat.logging import get_logger

__all__ = [
    "__version__",
    "CONFIG",
    "ConfigManager",
    "get_logger",
]

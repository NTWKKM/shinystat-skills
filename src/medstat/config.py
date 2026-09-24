"""
Configuration Management System for medstat.

Centralized, headless configuration management with hierarchical key access,
environment variable overrides, and validation.
"""

from __future__ import annotations

import copy
import json
import os
import warnings
from pathlib import Path
from typing import Any, cast

from medstat.logging import get_logger

logger = get_logger(__name__)


class ConfigManager:
    """
    Centralized configuration management with hierarchical key access.

    Supports:
    - Nested dictionary access with dot notation
    - Default values and fallbacks
    - Environment variable overrides (MEDSTAT_*)
    - Config validation
    - Runtime updates
    """

    def __init__(self, config_dict: dict[str, Any] | None = None) -> None:
        self._config: dict[str, Any] = (
            config_dict if config_dict is not None else self._get_default_config()
        )
        self._env_prefix = "MEDSTAT_"
        self._load_env_overrides()
        self._sync_missing_legacy()

    def _sync_missing_legacy(self) -> None:
        analysis = self._config.get("analysis", {})
        missing = analysis.get("missing", {})
        if not isinstance(missing, dict):
            return

        legacy_strategy = analysis.get("missing_strategy")
        legacy_threshold = analysis.get("missing_threshold_pct")

        if "strategy" in missing and legacy_strategy != missing["strategy"]:
            analysis["missing_strategy"] = missing["strategy"]
        elif legacy_strategy is not None:
            missing["strategy"] = legacy_strategy

        if (
            "report_threshold_pct" in missing
            and legacy_threshold != missing["report_threshold_pct"]
        ):
            analysis["missing_threshold_pct"] = missing["report_threshold_pct"]
        elif legacy_threshold is not None:
            missing["report_threshold_pct"] = legacy_threshold

    @staticmethod
    def _get_default_config() -> dict[str, Any]:
        return {
            # ========== ANALYSIS SETTINGS ==========
            "analysis": {
                # Logistic Regression
                "logit_method": "auto",  # 'auto', 'firth', 'bfgs', 'default'
                "logit_max_iter": 100,
                "logit_screening_p": 0.20,
                "logit_min_cases": 10,
                # Variable Detection
                "var_detect_threshold": 10,
                "var_detect_decimal_pct": 0.30,
                # Reporting Style
                "publication_style": "nejm",  # 'nejm', 'jama', 'apa'
                # P-value Handling
                "pvalue_bounds_lower": 0.001,
                "pvalue_bounds_upper": 0.999,
                "pvalue_clip_tolerance": 0.00001,
                "pvalue_format_small": "<0.001",
                "pvalue_format_large": ">0.999",
                "significance_level": 0.05,
                # Survival Analysis
                "survival_method": "kaplan-meier",
                "cox_method": "efron",
                # Missing Data
                "missing": {
                    "strategy": "complete-case",
                    "user_defined_values": [],
                    "treat_empty_as_missing": True,
                    "report_missing": True,
                    "report_threshold_pct": 50,
                },
                "missing_strategy": "complete-case",
                "missing_threshold_pct": 50,
            },
            # ========== ADVANCED STATS SETTINGS ==========
            "stats": {
                "mcc_enable": True,
                "mcc_method": "fdr_bh",
                "mcc_alpha": 0.05,
                "vif_enable": True,
                "vif_threshold": 10,
                "ci_method": "auto",
            },
            # ========== LOGGING SETTINGS ==========
            "logging": {
                "enabled": True,
                "level": "INFO",
                "format": "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
                "date_format": "%Y-%m-%d %H:%M:%S",
                "file_enabled": False,
                "log_dir": "logs",
                "log_file": "medstat.log",
                "max_log_size": 10485760,
                "backup_count": 5,
                "console_enabled": True,
                "console_level": "INFO",
                "log_file_operations": True,
                "log_data_operations": True,
                "log_analysis_operations": True,
                "log_performance": True,
            },
            # ========== PERFORMANCE SETTINGS ==========
            "performance": {
                "enable_caching": True,
                "cache_ttl": 3600,
                "enable_compression": False,
                "num_threads": 4,
            },
            # ========== VALIDATION SETTINGS ==========
            "validation": {
                "strict_mode": False,
                "validate_inputs": True,
                "validate_outputs": True,
                "auto_fix_errors": True,
            },
            # ========== DEVELOPER SETTINGS ==========
            "debug": {
                "enabled": False,
                "verbose": False,
                "profile_performance": False,
                "show_timings": False,
            },
        }

    def _load_env_overrides(self) -> None:
        for key, value in os.environ.items():
            if key == "MEDSTAT_PYTEST_FORWARDED":
                continue
            if key.startswith(self._env_prefix):
                parts = key[len(self._env_prefix) :].lower().split("_")
                if len(parts) < 2:
                    continue
                section = parts[0]
                key_name = "_".join(parts[1:])
                target_key = f"{section}.{key_name}"
                existing_val = self.get(target_key)
                converted_val: Any = value
                if existing_val is not None:
                    try:
                        if isinstance(existing_val, bool):
                            converted_val = value.lower() in ("true", "1", "yes", "on")
                        elif isinstance(existing_val, int):
                            converted_val = int(value)
                        elif isinstance(existing_val, float):
                            converted_val = float(value)
                    except (ValueError, TypeError):
                        converted_val = value
                try:
                    self.update(target_key, converted_val)
                except (KeyError, ValueError, TypeError) as e:
                    warnings.warn(
                        f"Failed to set env override {key}={value}: {e}", stacklevel=2
                    )

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value: Any = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def update(self, key: str, value: Any) -> None:
        keys = key.split(".")
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                raise KeyError(f"Config path '{'.'.join(keys[:-1])}' does not exist")
            config = config[k]
        final_key = keys[-1]
        if final_key not in config:
            raise KeyError(f"Config key '{key}' does not exist")
        config[final_key] = value

        if key in ("analysis.missing_strategy", "analysis.missing.strategy"):
            analysis = self._config.get("analysis", {})
            missing = analysis.get("missing")
            if isinstance(missing, dict):
                missing["strategy"] = value
            analysis["missing_strategy"] = value
        elif key in (
            "analysis.missing_threshold_pct",
            "analysis.missing.report_threshold_pct",
        ):
            analysis = self._config.get("analysis", {})
            missing = analysis.get("missing")
            if isinstance(missing, dict):
                missing["report_threshold_pct"] = value
            analysis["missing_threshold_pct"] = value
        elif key.startswith("analysis.missing"):
            self._sync_missing_legacy()

    def set_nested(self, key: str, value: Any, create: bool = False) -> None:
        keys = key.split(".")
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                if create:
                    config[k] = {}
                else:
                    raise KeyError(f"Config path '{k}' does not exist")
            config = config[k]
        config[keys[-1]] = value

    def get_section(self, section: str) -> dict[str, Any]:
        result = self.get(section, {})
        return copy.deepcopy(result) if isinstance(result, dict) else result

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._config)

    def to_json(self, filepath: str | None = None, pretty: bool = True) -> str:
        try:
            json_str = json.dumps(self._config, indent=2 if pretty else None)
        except (TypeError, ValueError):
            logger.exception("Failed to serialize config to JSON")
            json_str = "{}"
        if filepath:
            try:
                Path(filepath).write_text(json_str)
            except OSError:
                logger.exception("Failed to write config to %s", filepath)
        return json_str

    def validate(self) -> tuple[bool, list[str]]:
        errors = []
        screening_p = cast(float | None, self.get("analysis.logit_screening_p"))
        if screening_p is None or not (0 < screening_p < 1):
            errors.append("analysis.logit_screening_p must be between 0 and 1")

        lower = cast(float | None, self.get("analysis.pvalue_bounds_lower"))
        upper = cast(float | None, self.get("analysis.pvalue_bounds_upper"))
        if lower is None or upper is None or not (lower < upper):
            errors.append("pvalue_bounds_lower must be < pvalue_bounds_upper")

        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.get("logging.level") not in valid_levels:
            errors.append(f"logging.level must be one of {valid_levels}")

        valid_methods = ["auto", "firth", "bfgs", "default"]
        if self.get("analysis.logit_method") not in valid_methods:
            errors.append(f"analysis.logit_method must be one of {valid_methods}")

        valid_styles = ["nejm", "jama", "apa", "lancet", "bmj"]
        if self.get("analysis.publication_style") not in valid_styles:
            errors.append(f"analysis.publication_style must be one of {valid_styles}")

        return len(errors) == 0, errors

    def __repr__(self) -> str:
        return f"ConfigManager({len(self._config)} sections)"


# Global config instance
CONFIG = ConfigManager()

# Threshold constants
COHEN_D_THRESHOLDS = {
    "negligible": 0.2,
    "small": 0.5,
    "medium": 0.8,
    "large": float("inf"),
}

ETA_SQUARED_THRESHOLDS = {
    "small": 0.01,
    "medium": 0.06,
    "large": 0.14,
}

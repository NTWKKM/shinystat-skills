"""
src/medstat/data/quality.py: Multi-Dimensional Clinical Data Quality Engine.

Implements:
1. 5 Quantitative Quality Dimensions (0-100):
   - Completeness (missingness rate)
   - Validity (range checks, type constraints, allowed sets)
   - Consistency (dirty numeric tokens, mixed types, cross-variable logic)
   - Uniqueness (duplicate rows and key collisions)
   - Plausibility (extreme statistical outliers >3x IQR, zero-variance continuous)
2. Declarative Schema validation (DataQualitySchema, ColumnRule, CrossVariableRule).
3. Composite quality score with clinical weighting and letter grades (A-F).
4. Full backward compatibility with legacy check_data_quality and DataQualityReport.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

# ==============================================================================
# Helper Functions for Column Type & Dirty Token Detection
# ==============================================================================


_CLEAN_SYMBOL_RE = re.compile(r"[<|>|,|%|$|€|£|฿|¥|\u2212|\u2013|−|–]")


def _is_numeric_column(
    series: pd.Series, total_rows: int
) -> tuple[bool, pd.Series, pd.Series, pd.Series, int]:
    """
    Determine whether a Series is numeric and identify non-standard or unparseable values.

    Returns:
        (is_numeric_col, numeric_strict, numeric_coerced, is_strict_nan, strict_nan_count)
    """
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    numeric_strict = pd.to_numeric(series, errors="coerce")

    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        # 1. Normalize typographic unicode minuses to ASCII hyphen-minus
        norm_s = (
            series.astype(str)
            .str.replace("−", "-", regex=False)
            .str.replace("–", "-", regex=False)
        )
        # 2. Normalize financial parentheses notation: '(100)' -> '-100'
        norm_s = norm_s.str.replace(r"^\s*\((.+)\)\s*$", r"-\1", regex=True)
        # 3. Strip currency symbols and comparison operators
        clean_s = norm_s.str.replace(_CLEAN_SYMBOL_RE, "", regex=True).str.strip()
        numeric_coerced = pd.to_numeric(clean_s, errors="coerce")
    else:
        numeric_coerced = numeric_strict

    valid_numeric_count = int(numeric_coerced.notna().sum())
    is_numeric_col = (
        (valid_numeric_count / total_rows) > 0.5 if total_rows > 0 else False
    )

    is_strict_nan = (
        numeric_strict.isna() & series.notna() & (series.astype(str).str.strip() != "")
    )
    strict_nan_count = int(is_strict_nan.sum())

    return (
        is_numeric_col,
        numeric_strict,
        numeric_coerced,
        is_strict_nan,
        strict_nan_count,
    )


def _format_row_list(rows: Sequence[Any], max_show: int = 400) -> str:
    """Format row indices into a comma-separated string."""
    if len(rows) == 0:
        return ""
    count = len(rows)
    if count <= max_show:
        return ",".join(map(str, rows))
    else:
        shown = ",".join(map(str, rows[:max_show]))
        remaining = count - max_show
        return f"{shown}, ... (+{remaining} more)"


def check_data_quality(df: pd.DataFrame) -> list[str]:
    """
    Generate human-readable data-quality warnings for each column.
    Maintains 100% parity with legacy check_data_quality output.
    """
    warnings: list[str] = []
    total_rows = len(df)

    if total_rows == 0:
        return warnings

    for i in range(df.shape[1]):
        col = df.columns[i]
        series = df.iloc[:, i]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]

        col_issues: list[str] = []

        # 0. Missing Data Check
        is_na = series.isna()
        missing_count = int(is_na.sum())
        if missing_count > 0:
            error_rows = df.index[is_na].tolist()
            row_str = _format_row_list(error_rows, max_show=10000)
            col_issues.append(f"Missing {missing_count} values at rows `{row_str}`.")

        (
            is_numeric_col,
            numeric_strict,
            _,
            is_strict_nan,
            strict_nan_count,
        ) = _is_numeric_column(series, total_rows)

        # CASE 1: Numeric Column
        if is_numeric_col:
            # Check for infinite values (+/- np.inf)
            is_inf = np.isinf(numeric_strict) & series.notna()
            if is_inf.any():
                inf_count = int(is_inf.sum())
                error_rows = df.index[is_inf].tolist()
                row_str = _format_row_list(error_rows, max_show=10000)
                col_issues.append(
                    f"Found {inf_count} non-standard values (infinite +/-Inf) at rows `{row_str}`."
                )

            if strict_nan_count > 0:
                error_rows = df.index[is_strict_nan].tolist()
                try:
                    bad_values = series.loc[is_strict_nan].unique()
                except TypeError:
                    bad_values = series.loc[is_strict_nan].map(str).unique()

                row_str = _format_row_list(error_rows, max_show=10000)
                val_str = _format_row_list(bad_values, max_show=20)
                col_issues.append(
                    f"Found {strict_nan_count} non-standard values at rows `{row_str}` (Values: `{val_str}`)."
                )

        # CASE 2: Categorical Column
        else:
            original_vals = series.astype(str).str.strip()
            is_numeric_in_text = (numeric_strict.notna()) & (original_vals != "")
            numeric_in_text_count = int(is_numeric_in_text.sum())

            if numeric_in_text_count > 0:
                error_rows = df.index[is_numeric_in_text].tolist()
                try:
                    bad_values = series.loc[is_numeric_in_text].unique()
                except TypeError:
                    bad_values = series.loc[is_numeric_in_text].map(str).unique()

                row_str = _format_row_list(error_rows, max_show=10000)
                val_str = _format_row_list(bad_values, max_show=20)
                col_issues.append(
                    f"Found {numeric_in_text_count} numeric values inside categorical column at rows `{row_str}` (Values: `{val_str}`)."
                )

            # Rare categories check: guard against unhashable nested types (lists, dicts)
            try:
                unique_count = int(series.nunique())
            except TypeError:
                try:
                    unique_count = int(series.dropna().map(str).nunique())
                except Exception:
                    unique_count = total_rows

            unique_ratio = unique_count / total_rows if total_rows > 0 else 0.0
            if unique_ratio < 0.8:
                try:
                    val_counts = series.value_counts()
                except TypeError:
                    val_counts = series.dropna().map(str).value_counts()
                rare_threshold = 5
                rare_mask = val_counts < rare_threshold
                rare_vals = val_counts[rare_mask].index.tolist()

                if rare_vals:
                    val_str = _format_row_list(rare_vals, max_show=20)
                    col_issues.append(
                        f"Found rare categories (<{rare_threshold} times): `{val_str}`."
                    )

        if col_issues:
            full_msg = " ".join(col_issues)
            warnings.append(f"**Column '{col}':** {full_msg}")

    return warnings


# ==============================================================================
# Declarative Validation Schema
# ==============================================================================


@dataclass
class ColumnRule:
    """Rules for validating an individual column."""

    dtype: str | None = None  # 'numeric', 'categorical', 'datetime', 'boolean'
    min_val: float | None = None  # Physical validity lower bound
    max_val: float | None = None  # Physical validity upper bound
    plausible_min: float | None = None  # Epidemiological plausibility lower bound
    plausible_max: float | None = None  # Epidemiological plausibility upper bound
    allowed_values: list[Any] | set[Any] | None = None  # Categorical whitelist
    regex_pattern: str | None = None  # String format regex
    nullable: bool = True
    unique: bool = False


@dataclass
class CrossVariableRule:
    """Cross-variable logical validation rule."""

    name: str
    condition: Callable[[pd.DataFrame], pd.Series] | str
    description: str
    severity: str = "warning"  # 'info', 'warning', 'critical'


@dataclass
class DataQualitySchema:
    """Declarative clinical validation schema."""

    columns: dict[str, ColumnRule] = field(default_factory=dict)
    cross_variable_rules: list[CrossVariableRule] = field(default_factory=list)
    id_column: str | None = None


# ==============================================================================
# Quality Reporting Engine
# ==============================================================================


class DataQualityReport:
    """
    Advanced Multi-Dimensional Data Quality Assessment Framework.
    Evaluates 5 dimensions: Completeness, Validity, Consistency, Uniqueness, Plausibility.
    """

    DEFAULT_WEIGHTS = {
        "completeness": 0.20,
        "validity": 0.20,
        "consistency": 0.20,
        "uniqueness": 0.20,
        "plausibility": 0.20,
    }

    def __init__(
        self,
        df: pd.DataFrame,
        schema: DataQualitySchema | None = None,
    ) -> None:
        self.df = df
        self.schema = schema or DataQualitySchema()
        self.total_rows = len(df)
        self.total_columns = len(df.columns)
        self.total_cells = df.size
        self.rule_errors: list[dict[str, Any]] = []

    def completeness_score(self) -> float:
        """Score 0-100: Percentage of non-missing cells."""
        if self.total_cells == 0:
            return 100.0
        missing_cells = int(self.df.isna().sum().sum())
        return 100.0 * (1.0 - (missing_cells / self.total_cells))

    def consistency_score(self) -> float:
        """
        Score 0-100: Detects dirty numeric tokens, type mismatch, and cross-variable violations.
        Scales cross-variable rule penalties by severity (info: 10.0, warning: 25.0, critical: 50.0).
        """
        if self.total_rows == 0:
            return 100.0

        consistency_scores = []
        for i in range(self.df.shape[1]):
            series = self.df.iloc[:, i]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            (
                is_num,
                numeric_strict,
                _,
                _,
                strict_nan_count,
            ) = _is_numeric_column(series, self.total_rows)

            if is_num:
                col_score = 100.0 * (1.0 - (strict_nan_count / self.total_rows))
                consistency_scores.append(col_score)
            else:
                original_vals = series.astype(str).str.strip()
                is_numeric_in_text = (numeric_strict.notna()) & (original_vals != "")
                numeric_count = int(is_numeric_in_text.sum())
                col_score = 100.0 * (1.0 - (numeric_count / self.total_rows))
                consistency_scores.append(col_score)

        base_score = float(np.mean(consistency_scores)) if consistency_scores else 100.0

        # Cross-variable rule checks if schema provided
        self.rule_errors = []
        if self.schema and self.schema.cross_variable_rules:
            severity_weights = {
                "info": 10.0,
                "warning": 25.0,
                "critical": 50.0,
            }
            rule_penalties = 0.0

            for rule in self.schema.cross_variable_rules:
                try:
                    # Scope evaluation to non-null rows of referenced columns
                    if isinstance(rule.condition, str):
                        backtick_tokens = re.findall(r"`([^`]+)`", rule.condition)
                        word_tokens = re.findall(
                            r"\b[A-Za-z_][A-Za-z0-9_]*\b", rule.condition
                        )
                        tokens = set(backtick_tokens).union(word_tokens)
                        cols_in_rule = [c for c in self.df.columns if c in tokens]
                    else:
                        cols_in_rule = []

                    if cols_in_rule:
                        non_null_mask = self.df[cols_in_rule].notna().all(axis=1)
                    else:
                        non_null_mask = pd.Series(True, index=self.df.index)

                    evaluable_rows = int(non_null_mask.sum())
                    if evaluable_rows == 0:
                        continue

                    if callable(rule.condition):
                        raw_valid = rule.condition(self.df)
                    else:
                        raw_valid = self.df.eval(rule.condition)

                    if isinstance(raw_valid, pd.Series):
                        valid_mask = raw_valid
                    elif isinstance(raw_valid, (bool, np.bool_)):
                        valid_mask = pd.Series(bool(raw_valid), index=self.df.index)
                    else:
                        continue

                    violations = int(((~valid_mask) & non_null_mask).sum())
                    violation_rate = violations / evaluable_rows
                    sev = (rule.severity or "warning").lower()
                    weight = severity_weights.get(sev, 25.0)
                    rule_penalties += violation_rate * weight

                except Exception as e:
                    self.rule_errors.append(
                        {
                            "rule": getattr(rule, "name", str(rule)),
                            "condition": str(getattr(rule, "condition", "")),
                            "error": str(e),
                        }
                    )

            base_score = max(0.0, base_score - min(100.0, rule_penalties))

        return base_score

    def uniqueness_score(self) -> float:
        """Score 0-100: Penalizes duplicate rows or key collisions (id_column or rule.unique)."""
        if self.total_rows == 0:
            return 100.0

        # Priority 1: Check id_column if specified
        if (
            self.schema
            and self.schema.id_column
            and self.schema.id_column in self.df.columns
        ):
            col_data = self.df[self.schema.id_column]
            series = (
                col_data.iloc[:, 0] if isinstance(col_data, pd.DataFrame) else col_data
            )
            dups = int(series.duplicated().sum())
            return 100.0 * (1.0 - (float(dups) / self.total_rows))

        # Priority 2: Check columns with rule.unique=True if specified
        unique_cols = [
            col
            for col, rule in (self.schema.columns.items() if self.schema else [])
            if rule.unique and col in self.df.columns
        ]
        if unique_cols:
            total_evaluated = len(unique_cols) * self.total_rows
            total_dups = 0
            for col in unique_cols:
                col_data = self.df[col]
                series = (
                    col_data.iloc[:, 0]
                    if isinstance(col_data, pd.DataFrame)
                    else col_data
                )
                total_dups += int(series.duplicated().sum())
            return max(0.0, 100.0 * (1.0 - (float(total_dups) / total_evaluated)))

        # Default: Full row duplicates
        dups = int(self.df.duplicated().sum())
        return 100.0 * (1.0 - (float(dups) / self.total_rows))

    def validity_score(self) -> float:
        """
        Score 0-100: Evaluates format, type, and range constraint conformance.
        Checks dtype, nullable, unique, min_val, max_val, allowed_values, and regex_pattern.
        Prevents unparseable string bypass and additive double-counting.
        """
        if self.total_cells == 0:
            return 100.0

        if not self.schema or not self.schema.columns:
            return 100.0

        evaluated_cells = 0
        invalid_cells = 0

        valid_bool_tokens = {
            True,
            False,
            0,
            1,
            0.0,
            1.0,
            "true",
            "false",
            "1",
            "0",
            "yes",
            "no",
            "y",
            "n",
            "t",
            "f",
        }

        for col, rule in self.schema.columns.items():
            if col not in self.df.columns:
                continue

            col_data = self.df[col]
            raw_series = (
                col_data.iloc[:, 0] if isinstance(col_data, pd.DataFrame) else col_data
            )

            # If nullable is False, evaluate all cells and flag NAs
            if rule.nullable:
                series = raw_series.dropna()
                is_invalid = pd.Series(False, index=series.index)
            else:
                series = raw_series
                is_invalid = raw_series.isna().copy()

            n = len(series)
            evaluated_cells += n
            if n == 0:
                continue

            non_null = series.dropna()
            if len(non_null) > 0:
                # 1. Per-column uniqueness if requested
                if rule.unique:
                    is_invalid.loc[non_null.index] |= non_null.duplicated(keep="first")

                # 2. Dtype validation
                if rule.dtype == "numeric":
                    num_s = pd.to_numeric(non_null, errors="coerce")
                    is_invalid.loc[non_null.index] |= num_s.isna()
                    if pd.api.types.is_bool_dtype(non_null):
                        is_invalid.loc[non_null.index] = True
                elif rule.dtype == "boolean":
                    s_norm = non_null.map(
                        lambda x: x.strip().lower() if isinstance(x, str) else x
                    )
                    is_invalid.loc[non_null.index] |= ~s_norm.isin(valid_bool_tokens)
                elif rule.dtype == "datetime":
                    dt_s = pd.to_datetime(non_null, format="mixed", errors="coerce")
                    is_invalid.loc[non_null.index] |= dt_s.isna()
                elif rule.dtype == "categorical":
                    if rule.allowed_values is not None:
                        is_invalid.loc[non_null.index] |= ~non_null.isin(
                            rule.allowed_values
                        )

                # 3. Physical validity bounds (min_val / max_val)
                if rule.min_val is not None or rule.max_val is not None:
                    num_s = pd.to_numeric(non_null, errors="coerce")
                    unparseable = num_s.isna() & (
                        non_null.astype(str).str.strip() != ""
                    )
                    is_invalid.loc[non_null.index] |= unparseable
                    if rule.min_val is not None:
                        is_invalid.loc[non_null.index] |= num_s.notna() & (
                            num_s < rule.min_val
                        )
                    if rule.max_val is not None:
                        is_invalid.loc[non_null.index] |= num_s.notna() & (
                            num_s > rule.max_val
                        )

                # 4. Allowed values whitelist (if not already checked under categorical)
                if rule.allowed_values is not None and rule.dtype != "categorical":
                    is_invalid.loc[non_null.index] |= ~non_null.isin(
                        rule.allowed_values
                    )

                # 5. Regular expression pattern (full string match)
                if rule.regex_pattern is not None:
                    is_invalid.loc[non_null.index] |= ~non_null.astype(
                        str
                    ).str.fullmatch(rule.regex_pattern)

            invalid_cells += int(is_invalid.sum())

        if evaluated_cells == 0:
            return 100.0
        return max(0.0, 100.0 * (1.0 - (invalid_cells / evaluated_cells)))

    def plausibility_score(self) -> float:
        """
        Score 0-100: Penalizes extreme outliers (>3*IQR Tukey's fence, MAD modified Z > 3.5 fallback,
        or 3.5-std fallback), schema plausible bounds (plausible_min/max), and zero-variance anomalies.
        """
        if self.total_rows == 0:
            return 100.0

        numeric_col_indices = []
        for i in range(self.df.shape[1]):
            s_col = self.df.iloc[:, i]
            if pd.api.types.is_numeric_dtype(s_col) and not pd.api.types.is_bool_dtype(
                s_col
            ):
                numeric_col_indices.append(i)

        if not numeric_col_indices:
            return 100.0

        total_values = 0
        outlier_count = 0

        for i in numeric_col_indices:
            col_name = self.df.columns[i]
            s_raw = self.df.iloc[:, i].dropna()
            s_num = pd.to_numeric(s_raw, errors="coerce")

            # Non-finite values (+/- inf) are immediate plausibility violations
            non_finite_mask = ~np.isfinite(s_num) & s_raw.notna()
            non_finite_count = int(non_finite_mask.sum())
            outlier_count += non_finite_count

            # Isolate finite values for distribution calculation
            s = s_num[np.isfinite(s_num)]
            n = len(s)

            total_values += len(s_raw)
            if n < 5:
                continue

            rule = (
                self.schema.columns.get(col_name)
                if (
                    self.schema
                    and self.schema.columns
                    and col_name in self.schema.columns
                )
                else None
            )

            # Schema plausible bounds (plausible_min/max)
            bound_outliers = pd.Series(False, index=s.index)
            if rule is not None:
                if rule.plausible_min is not None:
                    bound_outliers |= s < rule.plausible_min
                if rule.plausible_max is not None:
                    bound_outliers |= s > rule.plausible_max

            # Statistical outlier detection: Tukey's fence -> MAD fallback -> std fallback
            stat_outliers = pd.Series(False, index=s.index)
            q25, q75 = np.percentile(s, [25, 75])
            iqr = q75 - q25

            if iqr > 0:
                lower = q25 - 3.0 * iqr
                upper = q75 + 3.0 * iqr
                stat_outliers |= (s < lower) | (s > upper)
            else:
                med = float(np.median(s))
                abs_dev = np.abs(s - med)
                mad = float(np.median(abs_dev))
                if mad > 0:
                    # Modified Z-score (Iglewicz & Hoaglin 1993, threshold = 3.5)
                    mod_z = 0.6745 * abs_dev / mad
                    stat_outliers |= mod_z > 3.5
                else:
                    std = float(s.std())
                    if std > 0:
                        mean = float(s.mean())
                        z = np.abs(s - mean) / std
                        stat_outliers |= z > 3.5
                    elif std == 0.0 and n > 20:
                        outlier_count += int(n * 0.1)

            total_col_outliers = bound_outliers | stat_outliers
            outlier_count += int(total_col_outliers.sum())

        if total_values == 0:
            return 100.0
        return max(0.0, 100.0 * (1.0 - (outlier_count / total_values)))

    def composite_score(
        self,
        weights: dict[str, float] | None = None,
        scores: dict[str, float] | None = None,
    ) -> float:
        """Compute weighted composite data quality score (0-100)."""
        w = weights or self.DEFAULT_WEIGHTS
        calc_scores = scores or {
            "completeness": self.completeness_score(),
            "validity": self.validity_score(),
            "consistency": self.consistency_score(),
            "uniqueness": self.uniqueness_score(),
            "plausibility": self.plausibility_score(),
        }
        total_w = sum(w.get(k, 0.0) for k in calc_scores)
        composite = sum(calc_scores[k] * w.get(k, 0.0) for k in calc_scores) / total_w
        return round(composite, 1)

    def generate_report(
        self, weights: dict[str, float] | None = None
    ) -> dict[str, Any]:
        """
        Generate complete, structured quality assessment.
        Backward-compatible with legacy DataQualityReport dictionary layout.
        """
        scores = {
            "completeness": round(self.completeness_score(), 1),
            "consistency": round(self.consistency_score(), 1),
            "uniqueness": round(self.uniqueness_score(), 1),
            "validity": round(self.validity_score(), 1),
            "plausibility": round(self.plausibility_score(), 1),
        }
        overall = self.composite_score(weights, scores=scores)
        grade = self._score_to_grade(overall)
        issues = check_data_quality(self.df)
        recs = self._generate_recommendations(scores)

        return {
            "overall_score": overall,
            "grade": grade,
            "dimension_scores": scores,
            "issues": issues,
            "rule_errors": self.rule_errors,
            "recommendations": recs,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "total_cells": self.total_cells,
        }

    @staticmethod
    def _score_to_grade(score: float) -> str:
        if score >= 95.0:
            return "A"
        if score >= 85.0:
            return "B"
        if score >= 70.0:
            return "C"
        if score >= 50.0:
            return "D"
        return "F"

    def _generate_recommendations(self, scores: dict[str, float]) -> list[str]:
        recs = []
        if scores["completeness"] < 90.0:
            recs.append(
                "Data has significant missing values. Consider imputation or removing sparse columns."
            )
        if scores["consistency"] < 95.0:
            recs.append(
                "Found inconsistent data types (e.g., text in numeric columns). Use cleaning tools."
            )
        if scores["validity"] < 95.0:
            recs.append(
                "Values violate schema or physiological range constraints. Inspect flagged records."
            )
        if scores["uniqueness"] < 100.0:
            recs.append("Duplicate rows detected. Verify if this is expected.")
        if scores["plausibility"] < 95.0:
            recs.append(
                "Extreme statistical outliers detected (>3x IQR). Consider clinical winsorization or sensitivity check."
            )
        if not recs:
            recs.append(
                "Dataset meets high clinical quality standards across all dimensions."
            )
        return recs

    def render_ascii_summary(self) -> str:
        """Render a publication-ready ASCII summary table."""
        report = self.generate_report()
        scores = report["dimension_scores"]

        lines = [
            "==================================================",
            "           DATA QUALITY AUDIT REPORT              ",
            "==================================================",
            f"Overall Quality Score: {report['overall_score']}/100 (Grade: {report['grade']})",
            f"Cohort Dimensions:     {report['total_rows']:,} rows × {report['total_columns']:,} columns ({report['total_cells']:,} cells)",
            "--------------------------------------------------",
            "Dimension Breakdown:",
            f"  • Completeness:  {scores['completeness']:.1f}%",
            f"  • Validity:      {scores['validity']:.1f}%",
            f"  • Consistency:   {scores['consistency']:.1f}%",
            f"  • Uniqueness:    {scores['uniqueness']:.1f}%",
            f"  • Plausibility:  {scores['plausibility']:.1f}%",
            "--------------------------------------------------",
        ]
        if report["recommendations"]:
            lines.append("Clinical Recommendations:")
            for rec in report["recommendations"]:
                lines.append(f"  - {rec}")
            lines.append("==================================================")
        return "\n".join(lines)

"""
Descriptive Statistics and Distribution Assessment.

Provides parametric and non-parametric summaries for continuous and
categorical clinical variables.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats


def calculate_descriptive_stats(
    data: pd.Series | np.ndarray,
    name: str = "variable",
) -> dict[str, Any]:
    """
    Calculate comprehensive descriptive statistics for a continuous variable.

    Returns:
        Dictionary containing:
        - n: valid sample size
        - missing: missing count
        - mean, std, sem
        - median, q25, q75, iqr
        - min, max, range
        - skewness, kurtosis
        - normality_p: Shapiro-Wilk or D'Agostino p-value
        - is_normal: boolean (p >= 0.05)
    """
    if isinstance(data, pd.Series):
        s = data.dropna().to_numpy(dtype=float)
        missing_count = int(data.isna().sum())
    else:
        arr = np.asarray(data, dtype=float)
        valid_mask = ~np.isnan(arr)
        s = arr[valid_mask]
        missing_count = int((~valid_mask).sum())

    n = len(s)
    if n == 0:
        return {
            "name": name,
            "n": 0,
            "missing": missing_count,
            "mean": np.nan,
            "std": np.nan,
            "sem": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "iqr": np.nan,
            "min": np.nan,
            "max": np.nan,
            "range": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
            "normality_p": np.nan,
            "is_normal": False,
        }

    mean_val = float(np.mean(s))
    std_val = float(np.std(s, ddof=1)) if n > 1 else 0.0
    sem_val = std_val / np.sqrt(n) if n > 0 else 0.0
    median_val = float(np.median(s))
    q25_val = float(np.percentile(s, 25))
    q75_val = float(np.percentile(s, 75))
    iqr_val = float(q75_val - q25_val)
    min_val = float(np.min(s))
    max_val = float(np.max(s))

    skew_val = float(stats.skew(s, bias=False)) if n > 2 else 0.0
    kurt_val = float(stats.kurtosis(s, bias=False)) if n > 3 else 0.0

    # Normality test: Shapiro-Wilk for n <= 5000, D'Agostino omnibus for n > 5000
    normality_p = np.nan
    if 3 <= n <= 5000:
        try:
            _, normality_p = stats.shapiro(s)
            normality_p = float(normality_p)
        except Exception:
            normality_p = np.nan
    elif n > 5000:
        try:
            _, normality_p = stats.normaltest(s)
            normality_p = float(normality_p)
        except Exception:
            normality_p = np.nan

    is_normal = bool(normality_p >= 0.05) if not np.isnan(normality_p) else False

    return {
        "name": name,
        "n": n,
        "missing": missing_count,
        "mean": mean_val,
        "std": std_val,
        "sem": sem_val,
        "median": median_val,
        "q25": q25_val,
        "q75": q75_val,
        "iqr": iqr_val,
        "min": min_val,
        "max": max_val,
        "range": max_val - min_val,
        "skewness": skew_val,
        "kurtosis": kurt_val,
        "normality_p": normality_p,
        "is_normal": is_normal,
    }


def calculate_categorical_stats(
    data: pd.Series | np.ndarray,
    name: str = "variable",
) -> dict[str, Any]:
    """
    Calculate frequency and percentage breakdowns for a categorical variable.
    """
    if not isinstance(data, pd.Series):
        s = pd.Series(data)
    else:
        s = data

    n_total = len(s)
    n_missing = int(s.isna().sum())
    valid_s = s.dropna()
    n_valid = len(valid_s)

    counts = valid_s.value_counts()
    percentages = (counts / n_valid * 100.0) if n_valid > 0 else counts

    breakdown = []
    for cat, count in counts.items():
        pct = float(percentages[cat])
        breakdown.append(
            {
                "category": str(cat),
                "count": int(count),
                "percentage": pct,
                "label": f"{int(count)} ({pct:.1f}%)",
            }
        )

    return {
        "name": name,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_missing": n_missing,
        "n_categories": len(counts),
        "categories": breakdown,
    }


def summarize_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """
    Generate complete summary of all continuous and categorical columns in a DataFrame.
    """
    continuous = {}
    categorical = {}

    for col in df.columns:
        series = df[col]
        # Heuristic: numeric with > 10 unique values is continuous
        if pd.api.types.is_numeric_dtype(series) and series.nunique() > 10:
            continuous[col] = calculate_descriptive_stats(series, name=col)
        else:
            categorical[col] = calculate_categorical_stats(series, name=col)

    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "continuous": continuous,
        "categorical": categorical,
    }

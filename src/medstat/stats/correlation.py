"""
Correlation Analysis Module.

Provides Pearson, Spearman, and Kendall correlation matrices,
Fisher Z confidence intervals, p-values, and optional Plotly heatmap generation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats

from medstat.logging import get_logger
from medstat.theme.palette import get_color_palette

logger = get_logger(__name__)
COLORS = get_color_palette()


def compute_correlation_ci(
    r: float, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """
    Compute confidence interval for correlation coefficient using Fisher's Z transformation.
    """
    if not np.isfinite(r) or n < 4:
        return (np.nan, np.nan)

    eps = 1e-12
    r_clipped = np.clip(r, -1.0 + eps, 1.0 - eps)

    z = 0.5 * np.log((1.0 + r_clipped) / (1.0 - r_clipped))
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf((1.0 + confidence) / 2.0)

    z_lower = z - z_crit * se
    z_upper = z + z_crit * se

    r_lower = (np.exp(2.0 * z_lower) - 1.0) / (np.exp(2.0 * z_lower) + 1.0)
    r_upper = (np.exp(2.0 * z_upper) - 1.0) / (np.exp(2.0 * z_upper) + 1.0)

    return (float(r_lower), float(r_upper))


def interpret_correlation(r: float) -> str:
    """
    Interpret correlation strength and direction according to standard rules.
    """
    if not np.isfinite(r):
        return "N/A"

    abs_r = abs(r)
    if abs_r >= 0.9:
        strength = "Very Strong"
    elif abs_r >= 0.7:
        strength = "Strong"
    elif abs_r >= 0.5:
        strength = "Moderate"
    elif abs_r >= 0.3:
        strength = "Weak"
    else:
        strength = "Very Weak/Negligible"

    direction = "Positive" if r >= 0 else "Negative"
    return f"{strength} {direction}"


def compute_correlation_matrix(
    df: pd.DataFrame,
    cols: list[str],
    method: str = "pearson",
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict[str, Any] | None]:
    """
    Compute correlation matrix with p-values and summary statistics.

    Parameters:
        df: Input DataFrame.
        cols: Numeric column names to analyze (minimum 2).
        method: 'pearson', 'spearman', or 'kendall'.

    Returns:
        tuple: (corr_matrix, p_values_matrix, summary_dict)
    """
    if not cols or len(cols) < 2:
        return None, None, None

    clean_df = df[cols].dropna(how="all")
    if clean_df.empty:
        return None, None, None

    # Calculate correlation matrix
    corr_matrix = clean_df.corr(method=method)

    # Calculate pairwise p-values and sample sizes
    p_values = pd.DataFrame(index=cols, columns=cols, dtype=float)
    ci_lower = pd.DataFrame(index=cols, columns=cols, dtype=float)
    ci_upper = pd.DataFrame(index=cols, columns=cols, dtype=float)

    for i, col_i in enumerate(cols):
        for j, col_j in enumerate(cols):
            if i == j:
                p_values.iloc[i, j] = 1.0
                ci_lower.iloc[i, j] = 1.0
                ci_upper.iloc[i, j] = 1.0
            elif i > j:
                p_values.iloc[i, j] = p_values.iloc[j, i]
                ci_lower.iloc[i, j] = ci_lower.iloc[j, i]
                ci_upper.iloc[i, j] = ci_upper.iloc[j, i]
            else:
                pair = clean_df[[col_i, col_j]].dropna()
                n_pair = len(pair)
                if n_pair >= 3:
                    if method == "pearson":
                        r_val, p_val = stats.pearsonr(pair[col_i], pair[col_j])
                    elif method == "spearman":
                        r_val, p_val = stats.spearmanr(pair[col_i], pair[col_j])
                    elif method == "kendall":
                        r_val, p_val = stats.kendalltau(pair[col_i], pair[col_j])
                    else:
                        raise ValueError(f"Unsupported correlation method: {method}")

                    p_values.iloc[i, j] = float(p_val)
                    low, high = compute_correlation_ci(float(r_val), n_pair)
                    ci_lower.iloc[i, j] = low
                    ci_upper.iloc[i, j] = high
                else:
                    p_values.iloc[i, j] = np.nan
                    ci_lower.iloc[i, j] = np.nan
                    ci_upper.iloc[i, j] = np.nan

    summary = {
        "method": method,
        "n_variables": len(cols),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }

    return corr_matrix, p_values, summary


def pairwise_correlation(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    method: str = "pearson",
    confidence: float = 0.95,
) -> dict[str, Any]:
    """
    Calculate pairwise correlation between two continuous variables with CI.
    """
    valid = ~(np.isnan(x) | np.isnan(y))
    x_c = np.asarray(x, dtype=float)[valid]
    y_c = np.asarray(y, dtype=float)[valid]

    n = len(x_c)
    if n < 3:
        raise ValueError("Pairwise correlation requires at least 3 non-missing pairs.")

    if method == "pearson":
        r, p = stats.pearsonr(x_c, y_c)
    elif method == "spearman":
        r, p = stats.spearmanr(x_c, y_c)
    elif method == "kendall":
        r, p = stats.kendalltau(x_c, y_c)
    else:
        raise ValueError(f"Unknown correlation method: {method}")

    r_val = float(r)
    p_val = float(p)
    ci_low, ci_high = compute_correlation_ci(r_val, n, confidence=confidence)

    return {
        "method": method,
        "n": n,
        "r": r_val,
        "p_value": p_val,
        "ci_95": [ci_low, ci_high],
        "interpretation": interpret_correlation(r_val),
    }

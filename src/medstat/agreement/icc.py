"""
Pure-Python / NumPy / SciPy Intraclass Correlation Coefficient (ICC).

Implements two-way ANOVA formulation from Shrout & Fleiss (1979) and
McGraw & Wong (1996) for all 6 variants with zero GPL / pingouin dependencies.
Licensed under Apache-2.0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as stats


def calculate_icc(
    df: pd.DataFrame,
    targets: str,
    raters: str,
    ratings: str,
    icc_type: str | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compute Intraclass Correlation Coefficients (ICC) across 6 models using
    pure-Python, NumPy, and SciPy two-way ANOVA.

    Parameters:
        df: Input DataFrame in long format containing targets, raters, and ratings.
        targets: Name of the column representing subjects/targets.
        raters: Name of the column representing raters/judges.
        ratings: Name of the column representing numeric measurement scores.
        icc_type: Optional specific ICC form to return ('ICC1', 'ICC2', 'ICC3',
                  'ICC1k', 'ICC2k', 'ICC3k'). If None, all 6 variants are returned.
        alpha: Significance level for (1 - alpha) confidence intervals (default 0.05 for 95% CI).

    Returns:
        pd.DataFrame with columns:
        ['Type', 'Description', 'ICC', 'F', 'df1', 'df2', 'pval', 'CI95%']
    """
    # Pivot to wide format (n subjects x k raters)
    pivoted = df.pivot(index=targets, columns=raters, values=ratings)
    # Drop subjects with missing values across raters (balanced two-way ANOVA)
    clean_matrix = pivoted.dropna()

    n, k = clean_matrix.shape
    if n < 2 or k < 2:
        raise ValueError(
            f"ICC requires at least 2 subjects and 2 raters after cleaning (got n={n}, k={k})."
        )

    return _compute_icc_from_matrix(clean_matrix.values, icc_type=icc_type, alpha=alpha)


def calculate_icc_wide(
    df: pd.DataFrame,
    cols: list[str],
    icc_type: str | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compute ICC directly from wide-format DataFrame where each column is a rater.

    Parameters:
        df: Input DataFrame where columns are raters and rows are subjects.
        cols: List of column names representing raters (minimum 2).
        icc_type: Optional specific ICC form to return ('ICC1', 'ICC2', 'ICC3',
                  'ICC1k', 'ICC2k', 'ICC3k'). If None, all 6 variants are returned.
        alpha: Significance level for confidence intervals (default 0.05).

    Returns:
        pd.DataFrame with columns:
        ['Type', 'Description', 'ICC', 'F', 'df1', 'df2', 'pval', 'CI95%']
    """
    if len(cols) < 2:
        raise ValueError(f"ICC requires at least 2 rater columns (got {len(cols)}).")

    clean_matrix = df[cols].dropna()
    n, k = clean_matrix.shape
    if n < 2:
        raise ValueError(
            f"ICC requires at least 2 subjects with complete observations (got n={n})."
        )

    return _compute_icc_from_matrix(clean_matrix.values, icc_type=icc_type, alpha=alpha)


def _compute_icc_from_matrix(
    values: np.ndarray,
    icc_type: str | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Internal calculation of the 6 ICC variants from balanced n x k numpy array.
    """
    n, k = values.shape
    grand_mean = np.mean(values)

    # Row (targets/subjects) and column (raters/judges) sample means
    row_means = np.mean(values, axis=1)
    col_means = np.mean(values, axis=0)

    # Sums of Squares (Two-way ANOVA)
    SST = float(np.sum((values - grand_mean) ** 2))
    SSB = float(k * np.sum((row_means - grand_mean) ** 2))
    SSJ = float(n * np.sum((col_means - grand_mean) ** 2))
    SSE = max(float(SST - SSB - SSJ), 0.0)
    SSW = float(SST - SSB)

    # Degrees of Freedom
    df_B = n - 1
    df_J = k - 1
    df_E = (n - 1) * (k - 1)
    df_W = n * (k - 1)

    # Mean Squares
    MSB = SSB / df_B if df_B > 0 else 0.0
    MSJ = SSJ / df_J if df_J > 0 else 0.0
    MSE = SSE / df_E if df_E > 0 else 0.0
    MSW = SSW / df_W if df_W > 0 else 0.0

    # -------------------------------------------------------------------------
    # 1. Point Estimates
    # -------------------------------------------------------------------------
    denom_icc1 = MSB + (k - 1) * MSW
    icc1 = (MSB - MSW) / denom_icc1 if denom_icc1 != 0 else np.nan

    denom_icc2 = MSB + (k - 1) * MSE + (k / n) * (MSJ - MSE)
    icc2 = (MSB - MSE) / denom_icc2 if denom_icc2 != 0 else np.nan

    denom_icc3 = MSB + (k - 1) * MSE
    icc3 = (MSB - MSE) / denom_icc3 if denom_icc3 != 0 else np.nan

    icc1k = (MSB - MSW) / MSB if MSB != 0 else np.nan
    denom_icc2k = MSB + (MSJ - MSE) / n
    icc2k = (MSB - MSE) / denom_icc2k if denom_icc2k != 0 else np.nan
    icc3k = (MSB - MSE) / MSB if MSB != 0 else np.nan

    # -------------------------------------------------------------------------
    # 2. Hypothesis Testing & F-Statistics (H0: ICC = 0)
    # -------------------------------------------------------------------------
    # One-way: F = MSB / MSW
    f_obs_1 = MSB / MSW if MSW > 0 else np.nan
    p_1 = float(stats.f.sf(f_obs_1, df_B, df_W)) if not np.isnan(f_obs_1) else np.nan

    # Two-way: F = MSB / MSE
    f_obs_23 = MSB / MSE if MSE > 0 else np.nan
    p_23 = float(stats.f.sf(f_obs_23, df_B, df_E)) if not np.isnan(f_obs_23) else np.nan

    # -------------------------------------------------------------------------
    # 3. Confidence Intervals (McGraw & Wong 1996 / Shrout & Fleiss 1979)
    # -------------------------------------------------------------------------
    # Case 1: One-way Random (ICC1 & ICC1k)
    f_crit_1_low = stats.f.ppf(1 - alpha / 2, df_B, df_W)
    f_crit_1_high = stats.f.ppf(1 - alpha / 2, df_W, df_B)

    f_low_1 = f_obs_1 / f_crit_1_low if f_crit_1_low > 0 else np.nan
    f_high_1 = f_obs_1 * f_crit_1_high

    ci_icc1_low = (f_low_1 - 1) / (f_low_1 + k - 1)
    ci_icc1_high = (f_high_1 - 1) / (f_high_1 + k - 1)
    ci_icc1 = np.round(np.array([ci_icc1_low, ci_icc1_high]), 2)

    ci_icc1k_low = 1.0 - 1.0 / f_low_1 if f_low_1 > 0 else np.nan
    ci_icc1k_high = 1.0 - 1.0 / f_high_1 if f_high_1 > 0 else np.nan
    ci_icc1k = np.round(np.array([ci_icc1k_low, ci_icc1k_high]), 2)

    # Case 3: Two-way Fixed Consistency (ICC3 & ICC3k)
    f_crit_3_low = stats.f.ppf(1 - alpha / 2, df_B, df_E)
    f_crit_3_high = stats.f.ppf(1 - alpha / 2, df_E, df_B)

    f_low_3 = f_obs_23 / f_crit_3_low if f_crit_3_low > 0 else np.nan
    f_high_3 = f_obs_23 * f_crit_3_high

    ci_icc3_low = (f_low_3 - 1) / (f_low_3 + k - 1)
    ci_icc3_high = (f_high_3 - 1) / (f_high_3 + k - 1)
    ci_icc3 = np.round(np.array([ci_icc3_low, ci_icc3_high]), 2)

    ci_icc3k_low = 1.0 - 1.0 / f_low_3 if f_low_3 > 0 else np.nan
    ci_icc3k_high = 1.0 - 1.0 / f_high_3 if f_high_3 > 0 else np.nan
    ci_icc3k = np.round(np.array([ci_icc3k_low, ci_icc3k_high]), 2)

    # Case 2: Two-way Random Absolute Agreement (ICC2 & ICC2k)
    F_J = MSJ / MSE if MSE > 0 else 1.0
    r2 = icc2

    v_num = df_E * (k * r2 * F_J + n * (1.0 + (k - 1) * r2) - k * r2) ** 2
    v_den = df_B * (k * r2 * F_J) ** 2 + (n * (1.0 + (k - 1) * r2) - k * r2) ** 2
    v = max(v_num / v_den, 1.0) if v_den != 0 else 1.0

    f2_high = stats.f.ppf(1 - alpha / 2, df_B, v)
    f2_low = stats.f.ppf(1 - alpha / 2, v, df_B)

    denom_L2 = f2_high * (k * MSJ + (k * n - k - n) * MSE) + n * MSB
    L2 = n * (MSB - f2_high * MSE) / denom_L2 if denom_L2 != 0 else np.nan

    denom_U2 = k * MSJ + (k * n - k - n) * MSE + n * f2_low * MSB
    U2 = n * (f2_low * MSB - MSE) / denom_U2 if denom_U2 != 0 else np.nan

    ci_icc2 = np.round(np.array([L2, U2]), 2)

    ci_icc2k_low = (
        (k * L2) / (1.0 + (k - 1) * L2) if (1.0 + (k - 1) * L2) != 0 else np.nan
    )
    ci_icc2k_high = (
        (k * U2) / (1.0 + (k - 1) * U2) if (1.0 + (k - 1) * U2) != 0 else np.nan
    )
    ci_icc2k = np.round(np.array([ci_icc2k_low, ci_icc2k_high]), 2)

    # -------------------------------------------------------------------------
    # Assemble DataFrame
    # -------------------------------------------------------------------------
    results = [
        {
            "Type": "ICC1",
            "Description": "Single raters absolute",
            "ICC": float(icc1),
            "F": float(f_obs_1),
            "df1": int(df_B),
            "df2": int(df_W),
            "pval": float(p_1),
            "CI95%": ci_icc1,
        },
        {
            "Type": "ICC2",
            "Description": "Single random raters",
            "ICC": float(icc2),
            "F": float(f_obs_23),
            "df1": int(df_B),
            "df2": int(df_E),
            "pval": float(p_23),
            "CI95%": ci_icc2,
        },
        {
            "Type": "ICC3",
            "Description": "Single fixed raters",
            "ICC": float(icc3),
            "F": float(f_obs_23),
            "df1": int(df_B),
            "df2": int(df_E),
            "pval": float(p_23),
            "CI95%": ci_icc3,
        },
        {
            "Type": "ICC1k",
            "Description": "Average raters absolute",
            "ICC": float(icc1k),
            "F": float(f_obs_1),
            "df1": int(df_B),
            "df2": int(df_W),
            "pval": float(p_1),
            "CI95%": ci_icc1k,
        },
        {
            "Type": "ICC2k",
            "Description": "Average random raters",
            "ICC": float(icc2k),
            "F": float(f_obs_23),
            "df1": int(df_B),
            "df2": int(df_E),
            "pval": float(p_23),
            "CI95%": ci_icc2k,
        },
        {
            "Type": "ICC3k",
            "Description": "Average fixed raters",
            "ICC": float(icc3k),
            "F": float(f_obs_23),
            "df1": int(df_B),
            "df2": int(df_E),
            "pval": float(p_23),
            "CI95%": ci_icc3k,
        },
    ]

    res_df = pd.DataFrame(results)

    if icc_type is not None:
        filtered = res_df[res_df["Type"] == icc_type].copy()
        if not filtered.empty:
            return filtered

    return res_df

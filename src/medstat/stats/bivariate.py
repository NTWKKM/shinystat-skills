"""
Bivariate Statistical Tests and Effect Sizes.

Parametric and non-parametric hypothesis testing for two or more clinical groups.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats


def independent_ttest(
    g1: pd.Series | np.ndarray,
    g2: pd.Series | np.ndarray,
    equal_var: bool = False,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Independent two-sample t-test (defaults to Welch's t-test for unequal variances).

    Returns:
        dict with t_stat, df, p_value, mean1, mean2, mean_diff, ci_diff, cohens_d.
    """
    x1 = np.asarray(g1, dtype=float)[~np.isnan(g1)]
    x2 = np.asarray(g2, dtype=float)[~np.isnan(g2)]

    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        raise ValueError("Each group must have at least 2 non-missing observations.")

    m1, m2 = float(np.mean(x1)), float(np.mean(x2))
    v1, v2 = float(np.var(x1, ddof=1)), float(np.var(x2, ddof=1))
    mean_diff = m1 - m2

    res = stats.ttest_ind(x1, x2, equal_var=equal_var)
    t_stat = float(res.statistic)
    p_val = float(res.pvalue)

    if equal_var:
        df = n1 + n2 - 2
        sp = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / df)
        se_diff = sp * np.sqrt(1.0 / n1 + 1.0 / n2)
        cohens_d = (m1 - m2) / sp if sp > 0 else 0.0
    else:
        # Welch-Satterthwaite degrees of freedom
        se_diff = np.sqrt(v1 / n1 + v2 / n2)
        num = (v1 / n1 + v2 / n2) ** 2
        den = ((v1 / n1) ** 2) / (n1 - 1) + ((v2 / n2) ** 2) / (n2 - 1)
        df = num / den if den > 0 else float(n1 + n2 - 2)
        pooled_sd = np.sqrt((v1 + v2) / 2.0)
        cohens_d = (m1 - m2) / pooled_sd if pooled_sd > 0 else 0.0

    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=df)
    ci_low = float(mean_diff - t_crit * se_diff)
    ci_high = float(mean_diff + t_crit * se_diff)

    return {
        "test": "Welch's t-test" if not equal_var else "Student's t-test",
        "statistic": t_stat,
        "df": float(df),
        "p_value": p_val,
        "n1": n1,
        "n2": n2,
        "mean1": m1,
        "mean2": m2,
        "mean_diff": mean_diff,
        "ci_diff": [ci_low, ci_high],
        "cohens_d": float(cohens_d),
    }


def paired_ttest(
    pre: pd.Series | np.ndarray,
    post: pd.Series | np.ndarray,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Paired samples t-test.
    """
    valid = ~(np.isnan(pre) | np.isnan(post))
    x1 = np.asarray(pre, dtype=float)[valid]
    x2 = np.asarray(post, dtype=float)[valid]

    n = len(x1)
    if n < 2:
        raise ValueError("Paired t-test requires at least 2 complete pairs.")

    diffs = x1 - x2
    mean_diff = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1))
    se_diff = sd_diff / np.sqrt(n)

    res = stats.ttest_rel(x1, x2)
    t_stat = float(res.statistic)
    p_val = float(res.pvalue)
    df = n - 1

    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=df)
    ci_low = float(mean_diff - t_crit * se_diff)
    ci_high = float(mean_diff + t_crit * se_diff)
    cohens_d = mean_diff / sd_diff if sd_diff > 0 else 0.0

    return {
        "test": "Paired t-test",
        "statistic": t_stat,
        "df": df,
        "p_value": p_val,
        "n_pairs": n,
        "mean_diff": mean_diff,
        "sd_diff": sd_diff,
        "ci_diff": [ci_low, ci_high],
        "cohens_d": float(cohens_d),
    }


def mann_whitney_u(
    g1: pd.Series | np.ndarray,
    g2: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """
    Mann-Whitney U test (Wilcoxon rank-sum test) for two independent groups.
    """
    x1 = np.asarray(g1, dtype=float)[~np.isnan(g1)]
    x2 = np.asarray(g2, dtype=float)[~np.isnan(g2)]

    n1, n2 = len(x1), len(x2)
    if n1 == 0 or n2 == 0:
        raise ValueError("Both groups must have at least 1 observation.")

    res = stats.mannwhitneyu(x1, x2, alternative="two-sided")
    u_stat = float(res.statistic)
    p_val = float(res.pvalue)

    # Rank-biserial correlation effect size r = 1 - 2*U / (n1 * n2)
    rank_biserial = 1.0 - (2.0 * u_stat) / (n1 * n2)

    return {
        "test": "Mann-Whitney U",
        "statistic": u_stat,
        "p_value": p_val,
        "n1": n1,
        "n2": n2,
        "median1": float(np.median(x1)),
        "median2": float(np.median(x2)),
        "rank_biserial_r": float(rank_biserial),
    }


def wilcoxon_signed_rank(
    pre: pd.Series | np.ndarray,
    post: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """
    Wilcoxon signed-rank test for paired data.
    """
    valid = ~(np.isnan(pre) | np.isnan(post))
    x1 = np.asarray(pre, dtype=float)[valid]
    x2 = np.asarray(post, dtype=float)[valid]

    diffs = x1 - x2
    non_zero = diffs != 0
    if np.sum(non_zero) < 1:
        return {
            "test": "Wilcoxon signed-rank",
            "statistic": 0.0,
            "p_value": 1.0,
            "n_pairs": len(x1),
            "median_diff": 0.0,
        }

    res = stats.wilcoxon(x1, x2)
    return {
        "test": "Wilcoxon signed-rank",
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "n_pairs": len(x1),
        "median_diff": float(np.median(diffs)),
    }


def chi_square_test(
    table: pd.DataFrame | np.ndarray,
    correction: bool = True,
) -> dict[str, Any]:
    """
    Pearson's Chi-square test of independence for contingency tables.
    """
    arr = np.asarray(table, dtype=float)
    res = stats.chi2_contingency(arr, correction=correction)

    chi2 = float(res.statistic)
    p_val = float(res.pvalue)
    dof = int(res.dof)
    expected = res.expected_freq

    n = np.sum(arr)
    # Cramer's V
    min_dim = min(arr.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0.0

    return {
        "test": "Pearson Chi-Square",
        "statistic": chi2,
        "df": dof,
        "p_value": p_val,
        "n_total": int(n),
        "cramers_v": float(cramers_v),
        "expected_frequencies": expected.tolist(),
        "min_expected": float(np.min(expected)) if expected.size > 0 else 0.0,
    }


def fisher_exact_test(
    table_2x2: pd.DataFrame | np.ndarray,
) -> dict[str, Any]:
    """
    Fisher's exact test for 2x2 contingency table.
    """
    arr = np.asarray(table_2x2, dtype=int)
    if arr.shape != (2, 2):
        raise ValueError("Fisher exact test requires a 2x2 contingency table.")

    odds_ratio, p_val = stats.fisher_exact(arr, alternative="two-sided")
    return {
        "test": "Fisher's Exact Test",
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_val),
        "table": arr.tolist(),
    }


def one_way_anova(*groups: pd.Series | np.ndarray) -> dict[str, Any]:
    """
    One-way Analysis of Variance (ANOVA).
    """
    clean_groups = [
        np.asarray(g, dtype=float)[~np.isnan(g)] for g in groups if len(g) > 0
    ]
    k = len(clean_groups)
    if k < 2:
        raise ValueError("One-way ANOVA requires at least 2 groups.")

    n_total = sum(len(g) for g in clean_groups)
    df_between = k - 1
    df_within = n_total - k

    res = stats.f_oneway(*clean_groups)
    f_stat = float(res.statistic)
    p_val = float(res.pvalue)

    # Eta-squared = SS_between / SS_total
    grand_mean = np.mean(np.concatenate(clean_groups))
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in clean_groups)
    ss_total = sum(np.sum((g - grand_mean) ** 2) for g in clean_groups)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0.0

    return {
        "test": "One-way ANOVA",
        "statistic": f_stat,
        "df1": df_between,
        "df2": df_within,
        "p_value": p_val,
        "n_groups": k,
        "n_total": n_total,
        "eta_squared": float(eta_squared),
    }


def kruskal_wallis(*groups: pd.Series | np.ndarray) -> dict[str, Any]:
    """
    Kruskal-Wallis H test (non-parametric one-way ANOVA).
    """
    clean_groups = [
        np.asarray(g, dtype=float)[~np.isnan(g)] for g in groups if len(g) > 0
    ]
    k = len(clean_groups)
    if k < 2:
        raise ValueError("Kruskal-Wallis requires at least 2 groups.")

    res = stats.kruskal(*clean_groups)
    h_stat = float(res.statistic)
    p_val = float(res.pvalue)
    df = k - 1

    n_total = sum(len(g) for g in clean_groups)
    # Epsilon-squared effect size = H / ((N^2 - 1) / (N + 1)) = H / (N - 1)
    epsilon_squared = h_stat / (n_total - 1) if n_total > 1 else 0.0

    return {
        "test": "Kruskal-Wallis H",
        "statistic": h_stat,
        "df": df,
        "p_value": p_val,
        "n_groups": k,
        "n_total": n_total,
        "epsilon_squared": float(epsilon_squared),
    }

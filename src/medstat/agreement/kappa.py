"""
Cohen's and Fleiss' Kappa for Inter-Rater Agreement.

Computes unweighted/linear/quadratic Cohen's Kappa for two raters,
and Fleiss' generalized Kappa for fixed multi-rater categorical assessments.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def cohens_kappa(
    rater1: np.ndarray | pd.Series,
    rater2: np.ndarray | pd.Series,
    weights: str | None = None,
) -> dict[str, Any]:
    """
    Calculate Cohen's Kappa for two raters on categorical data.

    Parameters:
        rater1: First rater evaluations.
        rater2: Second rater evaluations.
        weights: None (unweighted), 'linear', or 'quadratic'.

    Returns:
        dict with kappa, se, 95% CI, observed agreement, and expected agreement.
    """
    s1 = pd.Series(rater1)
    s2 = pd.Series(rater2)
    valid = s1.notna() & s2.notna()
    s1 = s1[valid]
    s2 = s2[valid]

    categories = sorted(list(set(s1.unique()) | set(s2.unique())))
    k = len(categories)
    if k <= 1:
        return {
            "kappa": 1.0,
            "se": 0.0,
            "ci_lower": 1.0,
            "ci_upper": 1.0,
            "p_value": 0.0,
            "observed_agreement": 1.0,
            "expected_agreement": 1.0,
            "categories": [str(c) for c in categories],
        }

    cat_map = {cat: idx for idx, cat in enumerate(categories)}
    conf = np.zeros((k, k), dtype=float)
    for c1, c2 in zip(s1, s2):
        conf[cat_map[c1], cat_map[c2]] += 1.0

    n = np.sum(conf)
    if n == 0:
        raise ValueError("No valid paired ratings.")

    p_mat = conf / n
    row_margins = np.sum(p_mat, axis=1)
    col_margins = np.sum(p_mat, axis=0)

    p_observed = float(np.trace(p_mat))
    p_expected = float(np.sum(row_margins * col_margins))

    if weights in ("linear", "quadratic"):
        w = np.zeros((k, k), dtype=float)
        for i in range(k):
            for j in range(k):
                if weights == "linear":
                    w[i, j] = 1.0 - abs(i - j) / (k - 1)
                else:
                    w[i, j] = 1.0 - ((i - j) ** 2) / ((k - 1) ** 2)
        p_o_w = float(np.sum(w * p_mat))
        p_e_w = float(np.sum(w * np.outer(row_margins, col_margins)))
        kappa = (p_o_w - p_e_w) / (1.0 - p_e_w) if abs(1.0 - p_e_w) > 1e-12 else 1.0
    else:
        kappa = (
            (p_observed - p_expected) / (1.0 - p_expected)
            if abs(1.0 - p_expected) > 1e-12
            else 1.0
        )

    # Standard error approximation
    se = (
        np.sqrt(
            p_expected
            + p_expected**2
            - np.sum(row_margins * col_margins * (row_margins + col_margins))
        )
        / ((1.0 - p_expected) * np.sqrt(n))
        if abs(1.0 - p_expected) > 1e-12
        else 0.0
    )
    ci_lower = max(-1.0, float(kappa - 1.96 * se))
    ci_upper = min(1.0, float(kappa + 1.96 * se))
    z = (kappa / se) if se > 0 else 0.0
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))

    return {
        "kappa": float(kappa),
        "se": float(se),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_val,
        "observed_agreement": p_observed,
        "expected_agreement": p_expected,
        "n_subjects": int(n),
        "categories": [str(c) for c in categories],
        "weighting": weights or "unweighted",
    }


def fleiss_kappa(subject_category_matrix: np.ndarray) -> dict[str, Any]:
    """
    Calculate Fleiss' generalized Kappa for m raters per subject.

    Parameters:
        subject_category_matrix: Array of shape (N, k) where cell (i, j) is the
                                 number of raters who assigned subject i to category j.
    """
    mat = np.asarray(subject_category_matrix, dtype=float)
    N, k = mat.shape
    if N == 0 or k <= 1:
        return {"kappa": 1.0, "se": 0.0, "p_value": 0.0}

    # Number of raters per subject
    m = np.sum(mat[0, :])
    if m <= 1:
        raise ValueError("Fleiss' Kappa requires at least 2 ratings per subject.")

    # Proportion of all assignments to each category
    p_j = np.sum(mat, axis=0) / (N * m)
    P_e = float(np.sum(p_j**2))

    # Agreement for each subject
    P_i = (np.sum(mat**2, axis=1) - m) / (m * (m - 1))
    P_o = float(np.mean(P_i))

    denom = 1.0 - P_e
    kappa = (P_o - P_e) / denom if abs(denom) > 1e-12 else 1.0

    # Variance and SE under null hypothesis
    var_p = (2.0 / (N * m * (m - 1) * denom**2)) * (
        denom**2 - np.sum(p_j * (1.0 - p_j) * (1.0 - 2 * p_j))
    )
    se = float(np.sqrt(max(0.0, var_p)))
    z = (kappa / se) if se > 0 else 0.0
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))

    return {
        "kappa": float(kappa),
        "se": se,
        "ci_lower": max(-1.0, float(kappa - 1.96 * se)),
        "ci_upper": min(1.0, float(kappa + 1.96 * se)),
        "p_value": p_val,
        "observed_agreement": P_o,
        "expected_agreement": P_e,
        "n_subjects": int(N),
        "n_raters": int(m),
        "n_categories": int(k),
    }


def calculate_kappa(
    df: pd.DataFrame,
    rater1: str | None = None,
    rater2: str | None = None,
    targets: str | None = None,
    raters: str | None = None,
    ratings: str | None = None,
) -> dict[str, Any]:
    """
    High-level entry point to calculate either Cohen's or Fleiss' Kappa automatically.
    """
    # 1. Two named columns -> Cohen's Kappa
    if rater1 and rater2 and rater1 in df.columns and rater2 in df.columns:
        res = cohens_kappa(df[rater1], df[rater2])
        res["type"] = "cohen"
        res["raters"] = [rater1, rater2]
        return res

    # 2. Long format -> pivot to subject x rater
    if (
        targets
        and raters
        and ratings
        and all(c in df.columns for c in (targets, raters, ratings))
    ):
        pivot_df = df.pivot(index=targets, columns=raters, values=ratings)
    elif (
        "subject_id" in df.columns
        and "rater_id" in df.columns
        and any(
            c in df.columns for c in ("rating", "ratings", "score", "measurement_score")
        )
    ):
        score_col = next(
            c
            for c in df.columns
            if c in ("rating", "ratings", "score", "measurement_score")
        )
        pivot_df = df.pivot(index="subject_id", columns="rater_id", values=score_col)
    else:
        pivot_df = df.select_dtypes(include=[np.number, "category", "object"])

    cols = list(pivot_df.columns)
    if len(cols) == 2:
        res = cohens_kappa(pivot_df[cols[0]], pivot_df[cols[1]])
        res["type"] = "cohen"
        res["raters"] = [str(cols[0]), str(cols[1])]
        return res

    # If >2 raters:
    # If values are continuous, discretize into quartiles
    vals = pivot_df.values.flatten()
    if (
        pd.api.types.is_numeric_dtype(pivot_df.dtypes.iloc[0])
        and len(np.unique(vals[~pd.isna(vals)])) > 10
    ):
        # Bin continuous measurements into 4 categories
        try:
            binned = pd.qcut(
                pivot_df.stack(), q=4, labels=[1, 2, 3, 4], duplicates="drop"
            ).unstack()
            pivot_df = binned
        except Exception:
            pass

    # Build count matrix for Fleiss' Kappa
    cats = sorted(list(set(pivot_df.stack().dropna().unique())))
    cat_to_col = {c: i for i, c in enumerate(cats)}
    mat = np.zeros((len(pivot_df), len(cats)), dtype=int)
    for row_idx, (_, row) in enumerate(pivot_df.iterrows()):
        for val in row.dropna():
            if val in cat_to_col:
                mat[row_idx, cat_to_col[val]] += 1

    res = fleiss_kappa(mat)
    res["type"] = "fleiss"
    res["categories"] = [str(c) for c in cats]
    return res

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


def validate_categorical_ratings(ratings: Any) -> None:
    """
    Validate that ratings represent discrete categorical classes, rejecting continuous measurements.

    Continuous measurements are identified when numeric ratings exhibit continuous variation:
    - Fractional values with more than 5 unique levels or high uniqueness ratio (k / n > 0.3).
    - Or integer values with high cardinality (k > 20) and high uniqueness ratio (k / n > 0.35).

    Discrete category codes with fractions (e.g. CDR 0, 0.5, 1, 2, 3) or >10 categories
    with repeating observations are preserved as valid categorical ratings.
    """
    if isinstance(ratings, (pd.DataFrame, pd.Series)):
        vals = ratings.values.flatten()
    else:
        vals = np.asarray(ratings).flatten()

    valid_vals = vals[~pd.isna(vals)]
    if len(valid_vals) == 0:
        raise ValueError("No valid ratings found.")

    try:
        num_vals = pd.to_numeric(valid_vals)
    except (ValueError, TypeError):
        return

    n = len(num_vals)
    unique_vals = np.unique(num_vals)
    k = len(unique_vals)

    if n == 0 or k <= 1:
        return

    uniqueness_ratio = k / n
    has_fractions = bool(np.any(~np.isclose(num_vals, np.round(num_vals), atol=1e-8)))

    if has_fractions:
        # A fractional scale with <= 5 discrete levels (like CDR 0, 0.5, 1, 2, 3) is a valid ordinal scale.
        # But fractional measurements with > 5 unique levels and high uniqueness ratio are continuous.
        if (k > 5 and uniqueness_ratio > 0.3) or k > 15:
            raise ValueError(
                "Continuous ratings are not supported for Kappa. "
                "Ratings must be discrete categorical classes; for continuous scores, use ICC or Bland-Altman."
            )
    else:
        # Integer scales with > 20 unique levels and high uniqueness ratio are continuous measurements
        if k > 20 and uniqueness_ratio > 0.35:
            raise ValueError(
                "Continuous ratings are not supported for Kappa. "
                "Ratings must be discrete categorical classes; for continuous scores, use ICC or Bland-Altman."
            )


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

    # Validate that ratings are discrete categories
    validate_categorical_ratings(pd.concat([s1, s2]))

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
        p_o = float(np.sum(w * p_mat))
        p_e = float(np.sum(w * np.outer(row_margins, col_margins)))
    else:
        w = np.eye(k, dtype=float)
        p_o = p_observed
        p_e = p_expected

    denom = 1.0 - p_e
    kappa = (p_o - p_e) / denom if abs(denom) > 1e-12 else 1.0

    # 1. Null standard error (Fleiss, Cohen, & Everitt 1969) under H0: kappa = 0
    # Used for the z-test and p-value
    w_row = w @ col_margins  # \bar{w}_{i\cdot} = \sum_j w_{ij} p_{\cdot j}
    w_col = w.T @ row_margins  # \bar{w}_{\cdot j} = \sum_i w_{ij} p_{i\cdot}
    w_bar_sum = np.add.outer(w_row, w_col)  # \bar{w}_{i\cdot} + \bar{w}_{\cdot j}
    outer_margins = np.outer(row_margins, col_margins)

    var_null = (
        (np.sum(outer_margins * ((w - w_bar_sum) ** 2)) - p_e**2) / (n * (denom**2))
        if abs(denom) > 1e-12
        else 0.0
    )
    se_null = float(np.sqrt(max(0.0, var_null)))

    # 2. Large-sample non-null standard error for confidence interval
    # (Cohen 1968 / Fleiss, Cohen & Everitt 1969 / Cicchetti & Allison 1971)
    z_mat = w * denom - w_bar_sum * (1.0 - p_o)
    var_non_null = (
        (np.sum(p_mat * (z_mat**2)) - (p_o * p_e - 2.0 * p_e + p_o) ** 2)
        / (n * (denom**4))
        if abs(denom) > 1e-12
        else 0.0
    )
    se_ci = float(np.sqrt(max(0.0, var_non_null)))

    ci_lower = max(-1.0, float(kappa - 1.96 * se_ci))
    ci_upper = min(1.0, float(kappa + 1.96 * se_ci))
    z = (kappa / se_null) if se_null > 0 else 0.0
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))

    return {
        "kappa": float(kappa),
        "se": float(se_ci),
        "se_null": float(se_null),
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
    row_sums = np.sum(mat, axis=1)
    m = row_sums[0]
    if m <= 1:
        raise ValueError("Fleiss' Kappa requires at least 2 ratings per subject.")
    if not np.all(row_sums == m):
        raise ValueError(
            f"Fleiss' Kappa requires all subjects to have the same number of ratings. "
            f"Found varying rating counts: min={row_sums.min()}, max={row_sums.max()}."
        )

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
    # 1. Explicit two named columns -> Cohen's Kappa
    if rater1 is not None or rater2 is not None:
        if not (rater1 and rater2 and rater1 in df.columns and rater2 in df.columns):
            missing = [r for r in (rater1, rater2) if not r or r not in df.columns]
            raise ValueError(
                f"Explicit rater columns not found in DataFrame: {missing}"
            )
        validate_categorical_ratings(df[[rater1, rater2]])
        res = cohens_kappa(df[rater1], df[rater2])
        res["type"] = "cohen"
        res["raters"] = [rater1, rater2]
        return res

    # 2. Explicit long format -> pivot to subject x rater
    if targets is not None or raters is not None or ratings is not None:
        if not (
            targets
            and raters
            and ratings
            and all(c in df.columns for c in (targets, raters, ratings))
        ):
            missing = [
                c for c in (targets, raters, ratings) if not c or c not in df.columns
            ]
            raise ValueError(f"Long-format columns not found in DataFrame: {missing}")
        pivot_df = df.pivot(index=targets, columns=raters, values=ratings)

    # 3. Standard auto-detected long format columns
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

    # 4. No explicit rater columns resolved -> raise ValueError
    else:
        raise ValueError(
            "No explicit rater columns resolved from data. "
            "Specify rater1 and rater2 for two raters, or targets, raters, and ratings for long-format data."
        )

    cols = list(pivot_df.columns)
    if len(cols) < 2:
        raise ValueError(
            f"At least 2 raters are required to compute Kappa, found {len(cols)}."
        )

    if len(cols) == 2:
        validate_categorical_ratings(pivot_df[[cols[0], cols[1]]])
        res = cohens_kappa(pivot_df[cols[0]], pivot_df[cols[1]])
        res["type"] = "cohen"
        res["raters"] = [str(cols[0]), str(cols[1])]
        return res

    # If >2 raters:
    # Reject continuous ratings instead of quartile-binning them
    validate_categorical_ratings(pivot_df)

    # Validate that each subject has the same number of ratings before calling fleiss_kappa
    ratings_per_subj = pivot_df.notna().sum(axis=1)
    if ratings_per_subj.nunique() > 1:
        raise ValueError(
            f"Fleiss' Kappa requires all subjects to have the same number of ratings. "
            f"Found varying rating counts: min={ratings_per_subj.min()}, max={ratings_per_subj.max()}."
        )

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

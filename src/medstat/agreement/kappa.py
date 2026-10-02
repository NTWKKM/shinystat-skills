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


def validate_categorical_ratings(
    ratings: Any,
    categories: list[Any] | None = None,
) -> None:
    """
    Validate that ratings represent discrete categorical classes, rejecting continuous measurements.

    Categorical Input Contract:
    1. Declared Categories: If `categories` is explicitly provided, all observed ratings must belong
       to the declared category levels.
    2. Categorical / Discrete Dtypes: Pandas `category`, string, object, boolean, and integer dtypes
       represent discrete category levels by contract.
    3. Floating-Point Measurements: Un-declared floating-point values are validated to ensure they
       represent discrete rating levels (where multiple subjects share identical ratings) rather than
       continuous non-repeating measurements where nearly all values are distinct.
    """
    if isinstance(ratings, (pd.DataFrame, pd.Series)):
        vals = ratings.to_numpy().flatten()
    else:
        vals = np.asarray(ratings).flatten()

    valid_vals = vals[~pd.isna(vals)]
    if len(valid_vals) == 0:
        raise ValueError("No valid ratings found.")

    # 1. Declared categories contract: validate observed values against declared categories FIRST
    if categories is not None:
        declared_set = set(categories)
        unrecognized = set(valid_vals) - declared_set
        if unrecognized:
            raise ValueError(
                f"Observed ratings contain values not in declared categories: {unrecognized}"
            )
        return

    # 2. Categorical / discrete dtypes: early return only after declared categories contract
    if isinstance(ratings, pd.DataFrame):
        if all(
            isinstance(ratings[c].dtype, pd.CategoricalDtype) for c in ratings.columns
        ):
            return
    elif isinstance(ratings, pd.Series):
        if isinstance(ratings.dtype, pd.CategoricalDtype):
            return

    # 2. Non-numeric types (strings, objects, booleans)
    try:
        num_vals = pd.to_numeric(valid_vals)
    except (ValueError, TypeError):
        return

    n = len(num_vals)
    unique_vals = np.unique(num_vals)
    k = len(unique_vals)

    if n == 0 or k <= 1:
        return

    # Check whether the numbers have fractional components
    has_fractions = bool(np.any(~np.isclose(num_vals, np.round(num_vals), atol=1e-8)))

    if has_fractions:
        # A discrete fractional scale has repeating ratings across subjects (e.g. CDR 0, 0.5, 1, 2, 3
        # or an 18-level scale where subjects share categories).
        # Continuous measurements (e.g. lab concentrations, ultrasound dimensions) have essentially
        # no repeating observations (each subject has a distinct float measurement, k == n or k/n > 0.85).
        uniqueness_ratio = k / n
        if n >= 4 and (k == n or uniqueness_ratio > 0.85):
            raise ValueError(
                "Continuous ratings are not supported for Kappa. "
                "Ratings must be discrete categorical classes (or declare explicit category levels via categories=[...]); "
                "for continuous scores, use ICC or Bland-Altman."
            )


def cohens_kappa(
    rater1: np.ndarray | pd.Series,
    rater2: np.ndarray | pd.Series,
    weights: str | None = None,
    categories: list[Any] | None = None,
) -> dict[str, Any]:
    """
    Calculate Cohen's Kappa for two raters on categorical data.

    Parameters:
        rater1: First rater evaluations.
        rater2: Second rater evaluations.
        weights: None (unweighted), 'linear', or 'quadratic'.
        categories: Optional declared category levels. If specified, category order is
                    preserved for weighted Kappa and unobserved levels are supported.

    Returns:
        dict with kappa, se, 95% CI, observed agreement, and expected agreement.
    """
    s1 = pd.Series(rater1)
    s2 = pd.Series(rater2)
    valid = s1.notna() & s2.notna()
    s1 = s1[valid]
    s2 = s2[valid]

    # Validate that ratings are discrete categories under the categorical input contract
    validate_categorical_ratings(pd.concat([s1, s2]), categories=categories)

    if categories is not None:
        cat_list = list(categories)
    else:
        cats1: list[Any] = (
            list(s1.cat.categories)
            if isinstance(s1.dtype, pd.CategoricalDtype)
            else list(s1.unique())
        )
        cats2: list[Any] = (
            list(s2.cat.categories)
            if isinstance(s2.dtype, pd.CategoricalDtype)
            else list(s2.unique())
        )
        combined_set = set(cats1) | set(cats2)
        try:
            cat_list = sorted(list(combined_set))
        except TypeError:
            cat_list = list(combined_set)

    k = len(cat_list)
    if k <= 1:
        return {
            "kappa": None,
            "se": None,
            "se_null": None,
            "ci_lower": None,
            "ci_upper": None,
            "p_value": None,
            "observed_agreement": 1.0 if k == 1 and len(s1) > 0 else None,
            "expected_agreement": 1.0 if k == 1 and len(s1) > 0 else None,
            "n_subjects": int(len(s1)),
            "categories": [str(c) for c in cat_list],
            "weighting": weights or "unweighted",
            "note": (
                "Degenerate single-category agreement: Kappa and p-value are undefined "
                "when fewer than 2 distinct categories are present."
            ),
        }

    cat_map = {cat: idx for idx, cat in enumerate(cat_list)}
    conf = np.zeros((k, k), dtype=float)
    unmapped_pairs = 0
    for c1, c2 in zip(s1, s2):
        if c1 in cat_map and c2 in cat_map:
            conf[cat_map[c1], cat_map[c2]] += 1.0
        else:
            unmapped_pairs += 1

    if unmapped_pairs > 0:
        raise ValueError(
            f"{unmapped_pairs} non-missing rating pair(s) excluded from confusion count."
        )

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
    if abs(denom) <= 1e-12:
        return {
            "kappa": None,
            "se": None,
            "se_null": None,
            "ci_lower": None,
            "ci_upper": None,
            "p_value": None,
            "observed_agreement": p_observed,
            "expected_agreement": p_expected,
            "n_subjects": int(n),
            "categories": [str(c) for c in cat_list],
            "weighting": weights or "unweighted",
            "note": (
                "Degenerate agreement: expected agreement is 1.0 (denominator is zero); "
                "Kappa and p-value are undefined."
            ),
        }

    kappa = (p_o - p_e) / denom

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
        "categories": [str(c) for c in cat_list],
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
    if N == 0:
        raise ValueError("No subjects provided for Fleiss' Kappa.")
    if k <= 1:
        row_sums = np.sum(mat, axis=1) if k == 1 else np.array([])
        m = int(row_sums[0]) if len(row_sums) > 0 else 0
        return {
            "kappa": None,
            "se": None,
            "se_null": None,
            "ci_lower": None,
            "ci_upper": None,
            "ci_note": None,
            "p_value": None,
            "observed_agreement": 1.0 if k == 1 else None,
            "expected_agreement": 1.0 if k == 1 else None,
            "n_subjects": int(N),
            "n_raters": m,
            "n_categories": int(k),
            "note": (
                "Degenerate single-category agreement: Fleiss' Kappa and p-value are undefined "
                "when fewer than 2 distinct categories are present."
            ),
        }

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
    if abs(denom) <= 1e-12:
        return {
            "kappa": None,
            "se": None,
            "se_null": None,
            "ci_lower": None,
            "ci_upper": None,
            "ci_note": None,
            "p_value": None,
            "observed_agreement": P_o,
            "expected_agreement": P_e,
            "n_subjects": int(N),
            "n_raters": int(m),
            "n_categories": int(k),
            "note": (
                "Degenerate agreement: expected agreement is 1.0 (denominator is zero); "
                "Fleiss' Kappa and p-value are undefined."
            ),
        }

    kappa = (P_o - P_e) / denom

    # Variance and SE under null hypothesis
    var_p = (2.0 / (N * m * (m - 1) * denom**2)) * (
        denom**2 - np.sum(p_j * (1.0 - p_j) * (1.0 - 2 * p_j))
    )
    se_null = float(np.sqrt(max(0.0, var_p)))
    z = (kappa / se_null) if se_null > 0 else 0.0
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))

    return {
        "kappa": float(kappa),
        "se": se_null,
        "se_null": se_null,
        "ci_lower": None,
        "ci_upper": None,
        "ci_note": (
            "Confidence interval not computed: null-hypothesis standard error is valid for "
            "hypothesis testing (z-test), not for interval estimation. Non-null variance "
            "method required for valid coverage."
        ),
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
    categories: list[Any] | None = None,
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
        validate_categorical_ratings(df[[rater1, rater2]], categories=categories)
        res = cohens_kappa(df[rater1], df[rater2], categories=categories)
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
        validate_categorical_ratings(
            pivot_df[[cols[0], cols[1]]], categories=categories
        )
        res = cohens_kappa(pivot_df[cols[0]], pivot_df[cols[1]], categories=categories)
        res["type"] = "cohen"
        res["raters"] = [str(cols[0]), str(cols[1])]
        return res

    # If >2 raters:
    validate_categorical_ratings(pivot_df, categories=categories)

    # Validate that each subject has the same number of ratings before calling fleiss_kappa
    ratings_per_subj = pivot_df.notna().sum(axis=1)
    if ratings_per_subj.nunique() > 1:
        raise ValueError(
            f"Fleiss' Kappa requires all subjects to have the same number of ratings. "
            f"Found varying rating counts: min={ratings_per_subj.min()}, max={ratings_per_subj.max()}."
        )

    # Build count matrix for Fleiss' Kappa
    if categories is not None:
        cats = list(categories)
    else:
        cats = sorted(list(set(pivot_df.stack().dropna().unique())))
    cat_to_col = {c: i for i, c in enumerate(cats)}
    mat = np.zeros((len(pivot_df), len(cats)), dtype=int)
    for row_idx, (_, row) in enumerate(pivot_df.iterrows()):
        for val in row.dropna():
            if val not in cat_to_col:
                raise ValueError(
                    f"Rating '{val}' is not in the specified categories {cats}."
                )
            mat[row_idx, cat_to_col[val]] += 1

    res = fleiss_kappa(mat)
    res["type"] = "fleiss"
    res["categories"] = [str(c) for c in cats]
    return res

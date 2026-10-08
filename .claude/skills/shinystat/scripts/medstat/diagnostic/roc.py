"""
Receiver Operating Characteristic (ROC) and DeLong Test.

Computes empirical ROC curves, AUC, optimal cutoffs (Youden J),
DeLong 95% confidence intervals, and correlated paired DeLong comparison tests.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import roc_auc_score, roc_curve


def calculate_roc_curve(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    pos_label: int | str = 1,
) -> dict[str, Any]:
    """
    Compute empirical ROC curve points (FPR, TPR, thresholds) and AUC.
    """
    y_t = (np.asarray(y_true) == pos_label).astype(int)
    y_s = np.asarray(y_score, dtype=float)

    valid = ~(np.isnan(y_t) | np.isnan(y_s))
    y_t = y_t[valid]
    y_s = y_s[valid]

    if len(np.unique(y_t)) < 2:
        raise ValueError("ROC curve requires both positive and negative cases.")

    fpr, tpr, thresholds = roc_curve(y_t, y_s)
    auc = float(roc_auc_score(y_t, y_s))

    return {
        "auc": auc,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "n_pos": int(np.sum(y_t == 1)),
        "n_neg": int(np.sum(y_t == 0)),
    }


def find_optimal_threshold(
    fpr: list[float] | np.ndarray,
    tpr: list[float] | np.ndarray,
    thresholds: list[float] | np.ndarray,
    method: Literal["youden", "closest_topleft"] = "youden",
) -> dict[str, Any]:
    """
    Find optimal classification cutoff on ROC curve.
    """
    f = np.asarray(fpr, dtype=float)
    t = np.asarray(tpr, dtype=float)
    thresh = np.asarray(thresholds, dtype=float)

    if method == "youden":
        # Maximize Sensitivity + Specificity - 1 = TPR - FPR
        j_scores = t - f
        best_idx = int(np.argmax(j_scores))
    else:
        # Closest to (0, 1): minimize sqrt((1 - TPR)^2 + FPR^2)
        dist = np.sqrt((1.0 - t) ** 2 + f**2)
        best_idx = int(np.argmin(dist))

    return {
        "threshold": float(thresh[best_idx]),
        "sensitivity": float(t[best_idx]),
        "specificity": float(1.0 - f[best_idx]),
        "fpr": float(f[best_idx]),
        "tpr": float(t[best_idx]),
        "index": best_idx,
    }


def _delong_placements(
    y_true: np.ndarray, score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute DeLong placement values for positives (V10) and negatives (V01)."""
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    pos_scores = score[pos_idx]
    neg_scores = score[neg_idx]

    # Matrix comparison: (m x n)
    greater = (pos_scores[:, None] > neg_scores).astype(float)
    equal = (pos_scores[:, None] == neg_scores).astype(float)
    res = greater + 0.5 * equal

    v10 = res.mean(axis=1)  # average over negatives for each positive
    v01 = res.mean(axis=0)  # average over positives for each negative
    auc = float(v10.mean())

    return v10, v01, auc


def auc_ci_delong(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    pos_label: int | str = 1,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Compute AUC and DeLong analytical 95% confidence intervals (DeLong et al., 1988).
    """
    y_t = (np.asarray(y_true) == pos_label).astype(int)
    y_s = np.asarray(y_score, dtype=float)

    valid = ~(np.isnan(y_t) | np.isnan(y_s))
    y_t = y_t[valid]
    y_s = y_s[valid]

    n_pos = np.sum(y_t == 1)
    n_neg = np.sum(y_t == 0)

    if n_pos < 2 or n_neg < 2:
        raise ValueError("DeLong CI requires at least 2 positives and 2 negatives.")

    v10, v01, auc = _delong_placements(y_t, y_s)

    s10 = np.var(v10, ddof=1)
    s01 = np.var(v01, ddof=1)
    var_auc = (s10 / n_pos) + (s01 / n_neg)
    se_auc = np.sqrt(max(var_auc, 0.0))

    z = stats.norm.ppf(1.0 - alpha / 2.0)
    ci_low = max(0.0, auc - z * se_auc)
    ci_high = min(1.0, auc + z * se_auc)

    return {
        "auc": float(auc),
        "se": float(se_auc),
        "ci_lower": float(ci_low),
        "ci_upper": float(ci_high),
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
    }


def delong_paired_test(
    y_true: pd.Series | np.ndarray,
    score1: pd.Series | np.ndarray,
    score2: pd.Series | np.ndarray,
    pos_label: int | str = 1,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Paired DeLong test for comparing two correlated ROC curves evaluated on the same subjects.
    """
    y_t = (np.asarray(y_true) == pos_label).astype(int)
    s1 = np.asarray(score1, dtype=float)
    s2 = np.asarray(score2, dtype=float)

    valid = ~(np.isnan(y_t) | np.isnan(s1) | np.isnan(s2))
    y_t = y_t[valid]
    s1 = s1[valid]
    s2 = s2[valid]

    n_pos = np.sum(y_t == 1)
    n_neg = np.sum(y_t == 0)

    if n_pos < 2 or n_neg < 2:
        raise ValueError(
            "Paired DeLong test requires at least 2 positives and 2 negatives."
        )

    v10_1, v01_1, auc1 = _delong_placements(y_t, s1)
    v10_2, v01_2, auc2 = _delong_placements(y_t, s2)

    s10_1 = np.var(v10_1, ddof=1)
    s01_1 = np.var(v01_1, ddof=1)
    var1 = (s10_1 / n_pos) + (s01_1 / n_neg)

    s10_2 = np.var(v10_2, ddof=1)
    s01_2 = np.var(v01_2, ddof=1)
    var2 = (s10_2 / n_pos) + (s01_2 / n_neg)

    # Covariance between scores
    cov_10 = np.cov(v10_1, v10_2, ddof=1)[0, 1]
    cov_01 = np.cov(v01_1, v01_2, ddof=1)[0, 1]
    cov12 = (cov_10 / n_pos) + (cov_01 / n_neg)

    diff = auc1 - auc2
    var_diff = max(var1 + var2 - 2.0 * cov12, 1e-12)
    se_diff = np.sqrt(var_diff)

    z_stat = diff / se_diff
    p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))

    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    ci_low = diff - z_crit * se_diff
    ci_high = diff + z_crit * se_diff

    return {
        "auc1": float(auc1),
        "auc2": float(auc2),
        "difference": float(diff),
        "se_difference": float(se_diff),
        "ci_lower": float(ci_low),
        "ci_upper": float(ci_high),
        "z_statistic": float(z_stat),
        "p_value": float(p_val),
    }

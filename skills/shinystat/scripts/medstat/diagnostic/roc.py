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


def compute_reclassification_metrics(
    y_true: pd.Series | np.ndarray,
    p_old: pd.Series | np.ndarray,
    p_new: pd.Series | np.ndarray,
    thresholds: list[float] | None = None,
    n_bootstrap: int = 0,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Compute Net Reclassification Improvement (Continuous & Categorical NRI)
    and Integrated Discrimination Improvement (IDI) per Pencina et al. (2008, 2011).

    Enforces clinical guardrails:
    - Miscalibration Guard: Verifies probabilities lie in [0, 1].
    - Separate Reporting: Emits Event NRI and Non-event NRI components alongside total.
    """
    y_t = np.asarray(y_true, dtype=float)
    p_0 = np.asarray(p_old, dtype=float)
    p_1 = np.asarray(p_new, dtype=float)

    if len(y_t) != len(p_0) or len(y_t) != len(p_1):
        raise ValueError("y_true, p_old, and p_new must have identical lengths.")

    valid = ~(np.isnan(y_t) | np.isnan(p_0) | np.isnan(p_1))
    y_t = y_t[valid].astype(int)
    p_0 = p_0[valid]
    p_1 = p_1[valid]

    # Clinical Miscalibration Guard
    if (
        np.any(p_0 < -1e-7)
        or np.any(p_0 > 1.0 + 1e-7)
        or np.any(p_1 < -1e-7)
        or np.any(p_1 > 1.0 + 1e-7)
    ):
        raise ValueError(
            "NRI and IDI require calibrated probabilities in [0, 1]. Received values outside this range."
        )

    p_0 = np.clip(p_0, 0.0, 1.0)
    p_1 = np.clip(p_1, 0.0, 1.0)

    unique_y = np.unique(y_t)
    if len(unique_y) < 2 or not set(unique_y).issubset({0, 1}):
        raise ValueError("y_true must contain both 0 and 1 classes.")

    n_events = int(np.sum(y_t == 1))
    n_nonevents = int(np.sum(y_t == 0))
    n_total = len(y_t)

    # 1. Continuous (Category-Free) NRI
    # Upward: p_new > p_old + 1e-12; Downward: p_new < p_old - 1e-12
    up_cont = p_1 > p_0 + 1e-12
    down_cont = p_1 < p_0 - 1e-12

    mask_1 = y_t == 1
    mask_0 = y_t == 0

    n_up_e = int(np.sum(mask_1 & up_cont))
    n_down_e = int(np.sum(mask_1 & down_cont))
    p_up_e = n_up_e / n_events if n_events > 0 else 0.0
    p_down_e = n_down_e / n_events if n_events > 0 else 0.0
    nri_e_cont = p_up_e - p_down_e
    var_e_cont = (p_up_e + p_down_e - nri_e_cont**2) / n_events if n_events > 0 else 0.0
    se_e_cont = float(np.sqrt(max(0.0, var_e_cont)))
    z_e_cont = float(nri_e_cont / se_e_cont) if se_e_cont > 0.0 else 0.0
    p_val_e_cont = (
        float(2.0 * (1.0 - stats.norm.cdf(abs(z_e_cont))))
        if se_e_cont > 0.0
        else (1.0 if nri_e_cont == 0.0 else 0.0)
    )

    n_up_ne = int(np.sum(mask_0 & up_cont))
    n_down_ne = int(np.sum(mask_0 & down_cont))
    p_up_ne = n_up_ne / n_nonevents if n_nonevents > 0 else 0.0
    p_down_ne = n_down_ne / n_nonevents if n_nonevents > 0 else 0.0
    nri_ne_cont = p_down_ne - p_up_ne
    var_ne_cont = (
        (p_down_ne + p_up_ne - nri_ne_cont**2) / n_nonevents if n_nonevents > 0 else 0.0
    )
    se_ne_cont = float(np.sqrt(max(0.0, var_ne_cont)))
    z_ne_cont = float(nri_ne_cont / se_ne_cont) if se_ne_cont > 0.0 else 0.0
    p_val_ne_cont = (
        float(2.0 * (1.0 - stats.norm.cdf(abs(z_ne_cont))))
        if se_ne_cont > 0.0
        else (1.0 if nri_ne_cont == 0.0 else 0.0)
    )

    nri_total_cont = nri_e_cont + nri_ne_cont
    var_total_cont = var_e_cont + var_ne_cont
    se_total_cont = float(np.sqrt(max(0.0, var_total_cont)))
    z_total_cont = float(nri_total_cont / se_total_cont) if se_total_cont > 0.0 else 0.0
    p_val_total_cont = (
        float(2.0 * (1.0 - stats.norm.cdf(abs(z_total_cont))))
        if se_total_cont > 0.0
        else (1.0 if nri_total_cont == 0.0 else 0.0)
    )

    z_crit = 1.959963984540054
    res_nri_cont = {
        "total": {
            "estimate": float(nri_total_cont),
            "se": se_total_cont,
            "ci_lower": float(nri_total_cont - z_crit * se_total_cont),
            "ci_upper": float(nri_total_cont + z_crit * se_total_cont),
            "z_statistic": z_total_cont,
            "p_value": p_val_total_cont,
        },
        "events": {
            "estimate": float(nri_e_cont),
            "se": se_e_cont,
            "ci_lower": float(nri_e_cont - z_crit * se_e_cont),
            "ci_upper": float(nri_e_cont + z_crit * se_e_cont),
            "z_statistic": z_e_cont,
            "p_value": p_val_e_cont,
            "n_up": n_up_e,
            "n_down": n_down_e,
            "p_up": float(p_up_e),
            "p_down": float(p_down_e),
        },
        "nonevents": {
            "estimate": float(nri_ne_cont),
            "se": se_ne_cont,
            "ci_lower": float(nri_ne_cont - z_crit * se_ne_cont),
            "ci_upper": float(nri_ne_cont + z_crit * se_ne_cont),
            "z_statistic": z_ne_cont,
            "p_value": p_val_ne_cont,
            "n_up": n_up_ne,
            "n_down": n_down_ne,
            "p_up": float(p_up_ne),
            "p_down": float(p_down_ne),
        },
    }

    # 2. Categorical NRI
    res_nri_cat = None
    cat_old = None
    cat_new = None
    sorted_thresh = None
    cat_labels = None
    k_bins = 0

    if thresholds is not None and len(thresholds) > 0:
        sorted_thresh = sorted(list({float(t) for t in thresholds}))
        if any(t <= 0.0 or t >= 1.0 for t in sorted_thresh):
            raise ValueError(
                "Categorical NRI thresholds must be strictly within (0, 1)."
            )

        cat_old = np.digitize(p_0, sorted_thresh, right=False)
        cat_new = np.digitize(p_1, sorted_thresh, right=False)
        k_bins = len(sorted_thresh) + 1

        # Labels
        cat_labels = []
        for i in range(k_bins):
            if i == 0:
                cat_labels.append(f"< {sorted_thresh[0]:.2g}")
            elif i == k_bins - 1:
                cat_labels.append(f"≥ {sorted_thresh[-1]:.2g}")
            else:
                cat_labels.append(
                    f"{sorted_thresh[i - 1]:.2g} to <{sorted_thresh[i]:.2g}"
                )

        # Contingency matrices: row = old category, col = new category
        m_events = np.zeros((k_bins, k_bins), dtype=int)
        m_nonevents = np.zeros((k_bins, k_bins), dtype=int)

        for i in range(n_total):
            if y_t[i] == 1:
                m_events[cat_old[i], cat_new[i]] += 1
            else:
                m_nonevents[cat_old[i], cat_new[i]] += 1
        m_combined = m_events + m_nonevents

        up_cat = cat_new > cat_old
        down_cat = cat_new < cat_old

        n_up_e_cat = int(np.sum(mask_1 & up_cat))
        n_down_e_cat = int(np.sum(mask_1 & down_cat))
        p_up_e_cat = n_up_e_cat / n_events if n_events > 0 else 0.0
        p_down_e_cat = n_down_e_cat / n_events if n_events > 0 else 0.0
        nri_e_cat = p_up_e_cat - p_down_e_cat
        var_e_cat = (
            (p_up_e_cat + p_down_e_cat - nri_e_cat**2) / n_events
            if n_events > 0
            else 0.0
        )
        se_e_cat = float(np.sqrt(max(0.0, var_e_cat)))
        z_e_cat = float(nri_e_cat / se_e_cat) if se_e_cat > 0.0 else 0.0
        p_val_e_cat = (
            float(2.0 * (1.0 - stats.norm.cdf(abs(z_e_cat))))
            if se_e_cat > 0.0
            else (1.0 if nri_e_cat == 0.0 else 0.0)
        )

        n_up_ne_cat = int(np.sum(mask_0 & up_cat))
        n_down_ne_cat = int(np.sum(mask_0 & down_cat))
        p_up_ne_cat = n_up_ne_cat / n_nonevents if n_nonevents > 0 else 0.0
        p_down_ne_cat = n_down_ne_cat / n_nonevents if n_nonevents > 0 else 0.0
        nri_ne_cat = p_down_ne_cat - p_up_ne_cat
        var_ne_cat = (
            (p_down_ne_cat + p_up_ne_cat - nri_ne_cat**2) / n_nonevents
            if n_nonevents > 0
            else 0.0
        )
        se_ne_cat = float(np.sqrt(max(0.0, var_ne_cat)))
        z_ne_cat = float(nri_ne_cat / se_ne_cat) if se_ne_cat > 0.0 else 0.0
        p_val_ne_cat = (
            float(2.0 * (1.0 - stats.norm.cdf(abs(z_ne_cat))))
            if se_ne_cat > 0.0
            else (1.0 if nri_ne_cat == 0.0 else 0.0)
        )

        nri_total_cat = nri_e_cat + nri_ne_cat
        var_total_cat = var_e_cat + var_ne_cat
        se_total_cat = float(np.sqrt(max(0.0, var_total_cat)))
        z_total_cat = float(nri_total_cat / se_total_cat) if se_total_cat > 0.0 else 0.0
        p_val_total_cat = (
            float(2.0 * (1.0 - stats.norm.cdf(abs(z_total_cat))))
            if se_total_cat > 0.0
            else (1.0 if nri_total_cat == 0.0 else 0.0)
        )

        # Reclassification percentages across combined cohort
        pct_up = float(
            np.sum(m_combined[np.triu_indices(k_bins, k=1)]) / n_total * 100.0
        )
        pct_down = float(
            np.sum(m_combined[np.tril_indices(k_bins, k=-1)]) / n_total * 100.0
        )
        pct_same = float(np.trace(m_combined) / n_total * 100.0)

    # Bootstrap CIs for reclassification metrics
    boot_cat_low = None
    boot_cat_high = None
    boot_cont_low = None
    boot_cont_high = None
    boot_idi_low = None
    boot_idi_high = None

    if n_bootstrap > 0:
        rng = np.random.default_rng(random_state)
        boot_cat_vals = []
        boot_cont_vals = []
        boot_idi_vals = []
        idx_1 = np.where(mask_1)[0]
        idx_0 = np.where(mask_0)[0]
        has_cat = cat_old is not None and cat_new is not None

        for _ in range(n_bootstrap):
            b_1 = rng.choice(idx_1, size=len(idx_1), replace=True)
            b_0 = rng.choice(idx_0, size=len(idx_0), replace=True)
            b_idx = np.concatenate([b_1, b_0])
            p0_b = p_0[b_idx]
            p1_b = p_1[b_idx]
            y_b = y_t[b_idx]
            m1_b = y_b == 1
            m0_b = y_b == 0

            # Continuous NRI bootstrap
            nri_e_b = np.mean(p1_b[m1_b] > p0_b[m1_b] + 1e-12) - np.mean(
                p1_b[m1_b] < p0_b[m1_b] - 1e-12
            )
            nri_ne_b = np.mean(p1_b[m0_b] < p0_b[m0_b] - 1e-12) - np.mean(
                p1_b[m0_b] > p0_b[m0_b] + 1e-12
            )
            boot_cont_vals.append(nri_e_b + nri_ne_b)

            # IDI bootstrap
            idi_b = (np.mean(p1_b[m1_b]) - np.mean(p0_b[m1_b])) - (
                np.mean(p1_b[m0_b]) - np.mean(p0_b[m0_b])
            )
            boot_idi_vals.append(idi_b)

            # Categorical NRI bootstrap
            if has_cat:
                c_old = cat_old[b_idx]
                c_new = cat_new[b_idx]
                cat_e_b = np.mean(c_new[m1_b] > c_old[m1_b]) - np.mean(
                    c_new[m1_b] < c_old[m1_b]
                )
                cat_ne_b = np.mean(c_new[m0_b] < c_old[m0_b]) - np.mean(
                    c_new[m0_b] > c_old[m0_b]
                )
                boot_cat_vals.append(cat_e_b + cat_ne_b)

        boot_cont_low = float(np.percentile(boot_cont_vals, 2.5))
        boot_cont_high = float(np.percentile(boot_cont_vals, 97.5))
        boot_idi_low = float(np.percentile(boot_idi_vals, 2.5))
        boot_idi_high = float(np.percentile(boot_idi_vals, 97.5))
        if has_cat:
            boot_cat_low = float(np.percentile(boot_cat_vals, 2.5))
            boot_cat_high = float(np.percentile(boot_cat_vals, 97.5))

    if boot_cont_low is not None:
        res_nri_cont["total"]["ci_lower_boot"] = boot_cont_low
        res_nri_cont["total"]["ci_upper_boot"] = boot_cont_high

    if sorted_thresh is not None:
        res_nri_cat = {
            "thresholds": sorted_thresh,
            "categories": cat_labels,
            "total": {
                "estimate": float(nri_total_cat),
                "se": se_total_cat,
                "ci_lower": float(nri_total_cat - z_crit * se_total_cat),
                "ci_upper": float(nri_total_cat + z_crit * se_total_cat),
                "ci_lower_boot": boot_cat_low,
                "ci_upper_boot": boot_cat_high,
                "z_statistic": z_total_cat,
                "p_value": p_val_total_cat,
            },
            "events": {
                "estimate": float(nri_e_cat),
                "se": se_e_cat,
                "ci_lower": float(nri_e_cat - z_crit * se_e_cat),
                "ci_upper": float(nri_e_cat + z_crit * se_e_cat),
                "z_statistic": z_e_cat,
                "p_value": p_val_e_cat,
                "n_up": n_up_e_cat,
                "n_down": n_down_e_cat,
                "p_up": float(p_up_e_cat),
                "p_down": float(p_down_e_cat),
            },
            "nonevents": {
                "estimate": float(nri_ne_cat),
                "se": se_ne_cat,
                "ci_lower": float(nri_ne_cat - z_crit * se_ne_cat),
                "ci_upper": float(nri_ne_cat + z_crit * se_ne_cat),
                "z_statistic": z_ne_cat,
                "p_value": p_val_ne_cat,
                "n_up": n_up_ne_cat,
                "n_down": n_down_ne_cat,
                "p_up": float(p_up_ne_cat),
                "p_down": float(p_down_ne_cat),
            },
            "contingency_matrices": {
                "events": m_events.tolist(),
                "nonevents": m_nonevents.tolist(),
                "combined": m_combined.tolist(),
            },
            "reclassification_percentages": {
                "upward": pct_up,
                "downward": pct_down,
                "unchanged": pct_same,
            },
        }

    # 3. Integrated Discrimination Improvement (IDI)
    mean_p1_events = float(np.mean(p_1[mask_1]))
    mean_p0_events = float(np.mean(p_0[mask_1]))
    mean_p1_nonevents = float(np.mean(p_1[mask_0]))
    mean_p0_nonevents = float(np.mean(p_0[mask_0]))

    idi_val = (mean_p1_events - mean_p0_events) - (
        mean_p1_nonevents - mean_p0_nonevents
    )
    slope_old = mean_p0_events - mean_p0_nonevents
    slope_new = mean_p1_events - mean_p1_nonevents
    rel_idi = float(idi_val / slope_old) if slope_old > 1e-12 else None

    delta_p = p_1 - p_0
    var_delta_1 = float(np.var(delta_p[mask_1], ddof=1)) if n_events > 1 else 0.0
    var_delta_0 = float(np.var(delta_p[mask_0], ddof=1)) if n_nonevents > 1 else 0.0

    se_idi = float(
        np.sqrt(max(0.0, var_delta_1 / n_events + var_delta_0 / n_nonevents))
    )
    z_idi = float(idi_val / se_idi) if se_idi > 0.0 else 0.0
    p_val_idi = (
        float(2.0 * (1.0 - stats.norm.cdf(abs(z_idi))))
        if se_idi > 0.0
        else (1.0 if idi_val == 0.0 else 0.0)
    )

    res_idi = {
        "estimate": float(idi_val),
        "se": se_idi,
        "ci_lower": float(idi_val - z_crit * se_idi),
        "ci_upper": float(idi_val + z_crit * se_idi),
        "z_statistic": z_idi,
        "p_value": p_val_idi,
        "relative_idi": rel_idi,
        "discrimination_slope_old": float(slope_old),
        "discrimination_slope_new": float(slope_new),
        "mean_p_old_events": mean_p0_events,
        "mean_p_old_nonevents": mean_p0_nonevents,
        "mean_p_new_events": mean_p1_events,
        "mean_p_new_nonevents": mean_p1_nonevents,
    }

    if boot_idi_low is not None:
        res_idi["ci_lower_boot"] = boot_idi_low
        res_idi["ci_upper_boot"] = boot_idi_high

    return {
        "nri_continuous": res_nri_cont,
        "nri_categorical": res_nri_cat,
        "idi": res_idi,
        "n_events": n_events,
        "n_nonevents": n_nonevents,
        "n_total": n_total,
        "p_old": p_0,
        "p_new": p_1,
        "y_true": y_t,
        "thresholds": sorted_thresh
        if thresholds is not None and len(thresholds) > 0
        else None,
    }


def compare_diagnostic_models(
    y_true: pd.Series | np.ndarray,
    p_old: pd.Series | np.ndarray,
    p_new: pd.Series | np.ndarray,
    thresholds: list[float] | None = None,
    n_bootstrap: int = 0,
    random_state: int = 42,
    alpha: float = 0.05,
    pos_label: int | str = 1,
) -> dict[str, Any]:
    """
    Unified diagnostic model comparison combining:
    - Single & Paired DeLong ΔAUROC test
    - Continuous Net Reclassification Improvement (NRI)
    - Categorical NRI with K x K contingency matrices (if thresholds provided)
    - Integrated Discrimination Improvement (IDI)
    """
    y_arr = (
        pd.Series(y_true).to_numpy()
        if isinstance(y_true, pd.Series)
        else np.asarray(y_true)
    )
    if pd.isna(y_arr).any():
        raise ValueError("y_true contains missing or NaN values.")

    unique_vals = np.unique(y_arr)
    if len(unique_vals) != 2:
        raise ValueError(
            f"compare_diagnostic_models requires a binary outcome with exactly 2 distinct classes. Found {len(unique_vals)}: {unique_vals.tolist()}"
        )
    if pos_label not in unique_vals:
        raise ValueError(
            f"pos_label '{pos_label}' not found in y_true distinct values: {unique_vals.tolist()}"
        )

    y_t = (y_arr == pos_label).astype(int)
    p_0 = np.asarray(p_old, dtype=float)
    p_1 = np.asarray(p_new, dtype=float)

    # DeLong ROC evaluation
    roc_old = calculate_roc_curve(y_t, p_0, pos_label=1)
    ci_old = auc_ci_delong(y_t, p_0, pos_label=1, alpha=alpha)
    opt_old = find_optimal_threshold(
        roc_old["fpr"], roc_old["tpr"], roc_old["thresholds"]
    )

    roc_new = calculate_roc_curve(y_t, p_1, pos_label=1)
    ci_new = auc_ci_delong(y_t, p_1, pos_label=1, alpha=alpha)
    opt_new = find_optimal_threshold(
        roc_new["fpr"], roc_new["tpr"], roc_new["thresholds"]
    )

    # Paired DeLong test: score1 = p_new, score2 = p_old (so difference is ΔAUROC = AUC_new - AUC_old)
    delong_comp = delong_paired_test(y_t, p_1, p_0, pos_label=1, alpha=alpha)

    # Reclassification metrics
    reclass = compute_reclassification_metrics(
        y_true=y_t,
        p_old=p_0,
        p_new=p_1,
        thresholds=thresholds,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    reclass.pop("p_old", None)
    reclass.pop("p_new", None)
    reclass.pop("y_true", None)

    return {
        "model_old": {
            "auc": ci_old["auc"],
            "se": ci_old["se"],
            "ci_lower": ci_old["ci_lower"],
            "ci_upper": ci_old["ci_upper"],
            "optimal_cutoff": opt_old["threshold"],
            "sensitivity": opt_old["sensitivity"],
            "specificity": opt_old["specificity"],
            "fpr": roc_old["fpr"],
            "tpr": roc_old["tpr"],
            "thresholds": roc_old["thresholds"],
        },
        "model_new": {
            "auc": ci_new["auc"],
            "se": ci_new["se"],
            "ci_lower": ci_new["ci_lower"],
            "ci_upper": ci_new["ci_upper"],
            "optimal_cutoff": opt_new["threshold"],
            "sensitivity": opt_new["sensitivity"],
            "specificity": opt_new["specificity"],
            "fpr": roc_new["fpr"],
            "tpr": roc_new["tpr"],
            "thresholds": roc_new["thresholds"],
        },
        "delong": delong_comp,
        "reclassification": reclass,
    }


def _lookup_censoring_survival_left_limit(
    kmf_censor: Any,
    times: np.ndarray | pd.Series | list[float] | float,
    clip_min: float = 1e-3,
) -> np.ndarray | float:
    """Evaluate left-limit censoring survival probability G_hat(t-) = P(C >= t).

    Evaluates the survival probability strictly prior to t using
    searchsorted(timeline, t, side="left") - 1 clamped to index 0 (where S(0) = 1.0).
    """
    timeline = np.asarray(kmf_censor.survival_function_.index, dtype=float)
    vals = kmf_censor.survival_function_.iloc[:, 0].to_numpy(dtype=float)
    is_scalar = np.isscalar(times)
    t_arr = np.atleast_1d(np.asarray(times, dtype=float))
    idx = np.clip(np.searchsorted(timeline, t_arr, side="left") - 1, 0, len(vals) - 1)
    res = np.clip(vals[idx], clip_min, 1.0)
    return float(res[0]) if is_scalar else res


def _calculate_ipcw_roc_at_horizon(
    durations: np.ndarray,
    events: np.ndarray,
    scores: np.ndarray,
    horizon: float,
    g_durations: np.ndarray,
    g_horizon: float,
    competing_event_code: int | None = None,
    return_curve: bool = True,
) -> dict[str, Any]:
    """Calculate cumulative/dynamic IPCW AUROC and optional ROC curve at a single follow-up horizon."""
    t = float(horizon)
    cases_mask = (durations <= t) & (events == 1)

    if competing_event_code is not None:
        competing_mask = (durations <= t) & (events == competing_event_code)
        ctrl_mask = (durations > t) | competing_mask
    else:
        ctrl_mask = durations > t

    n_cases = int(np.sum(cases_mask))
    n_controls = int(np.sum(ctrl_mask))
    n_censored = int(np.sum((durations <= t) & (events == 0)))

    if n_cases < 1 or n_controls < 1:
        res_empty: dict[str, Any] = {
            "horizon": t,
            "n_cases": n_cases,
            "n_controls": n_controls,
            "n_censored": n_censored,
            "auroc": float("nan"),
        }
        if return_curve:
            res_empty.update(
                {
                    "optimal_threshold": float("nan"),
                    "sensitivity": float("nan"),
                    "specificity": float("nan"),
                    "fpr": [0.0, 1.0],
                    "tpr": [0.0, 1.0],
                    "thresholds": [float("inf"), float("-inf")],
                }
            )
        return res_empty

    # IPCW weights
    w_cases = 1.0 / np.clip(g_durations[cases_mask], 1e-3, 1.0)

    if competing_event_code is not None:
        w_ctrl = np.where(
            durations[ctrl_mask] > t,
            1.0 / max(g_horizon, 1e-3),
            1.0 / np.clip(g_durations[ctrl_mask], 1e-3, 1.0),
        )
    else:
        w_ctrl = np.full(n_controls, 1.0 / max(g_horizon, 1e-3))

    sc_cases = scores[cases_mask]
    sc_ctrl = scores[ctrl_mask]

    # O(N log N) Weighted Concordance AUROC(t) via sorting + np.searchsorted
    sort_idx = np.argsort(sc_ctrl)
    s0_sorted = sc_ctrl[sort_idx]
    w0_sorted = w_ctrl[sort_idx]
    cum_w0 = np.concatenate([[0.0], np.cumsum(w0_sorted)])

    idx_lt = np.searchsorted(s0_sorted, sc_cases, side="left")
    idx_gt = np.searchsorted(s0_sorted, sc_cases, side="right")
    concordance = cum_w0[idx_lt] + 0.5 * (cum_w0[idx_gt] - cum_w0[idx_lt])
    num = float(np.sum(w_cases * concordance))
    sum_w = float(np.sum(w_cases) * cum_w0[-1])
    auroc = float(num / sum_w) if sum_w > 0 else float("nan")

    if not return_curve:
        return {
            "horizon": t,
            "n_cases": n_cases,
            "n_controls": n_controls,
            "n_censored": n_censored,
            "auroc": auroc,
        }

    # Generate curve points across unique thresholds
    sum_w_cases = float(np.sum(w_cases))
    sum_w_ctrl = float(np.sum(w_ctrl))
    unique_thresh = np.sort(np.unique(np.concatenate([sc_cases, sc_ctrl])))[::-1]

    tpr_list = [0.0]
    fpr_list = [0.0]
    thresh_list = [float("inf")]

    for th in unique_thresh:
        tpr_val = (
            float(np.sum(w_cases[sc_cases >= th]) / sum_w_cases)
            if sum_w_cases > 0
            else 0.0
        )
        fpr_val = (
            float(np.sum(w_ctrl[sc_ctrl >= th]) / sum_w_ctrl) if sum_w_ctrl > 0 else 0.0
        )
        tpr_list.append(tpr_val)
        fpr_list.append(fpr_val)
        thresh_list.append(float(th))

    tpr_list.append(1.0)
    fpr_list.append(1.0)
    thresh_list.append(float("-inf"))

    # Optimal cutoff via Youden J = TPR - FPR
    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)
    j_scores = tpr_arr - fpr_arr
    best_idx = int(np.argmax(j_scores))

    opt_thresh = thresh_list[best_idx]
    if np.isneginf(opt_thresh):
        opt_thresh = float(unique_thresh[-1]) if len(unique_thresh) > 0 else 0.0
    elif np.isposinf(opt_thresh):
        opt_thresh = float(unique_thresh[0]) if len(unique_thresh) > 0 else 0.0

    return {
        "horizon": t,
        "n_cases": n_cases,
        "n_controls": n_controls,
        "n_censored": n_censored,
        "auroc": auroc,
        "optimal_threshold": float(opt_thresh),
        "sensitivity": float(tpr_arr[best_idx]),
        "specificity": float(1.0 - fpr_arr[best_idx]),
        "fpr": fpr_list,
        "tpr": tpr_list,
        "thresholds": thresh_list,
    }


def compute_time_dependent_roc(
    durations: pd.Series | np.ndarray,
    events: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray | None = None,
    time_horizons: list[float] | np.ndarray | None = None,
    competing_event_code: int | None = None,
    n_bootstrap: int = 200,
    random_state: int = 42,
    scores_old: pd.Series | np.ndarray | None = None,
    scores_new: pd.Series | np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Cumulative/Dynamic Inverse Probability of Censoring Weighting (IPCW)
    Time-Dependent ROC estimator (Uno et al., 2007; Heagerty et al., 2000).

    Computes AUROC(t), optimal Youden thresholds, and confidence intervals across
    clinical horizons t, with optional side-by-side comparison of two scores.

    Enforces clinical Horizon Boundary Guard:
    Emits a structured warning and sets CI to NaN if t > max(T) or remaining cases/controls < 5.
    """
    from lifelines import KaplanMeierFitter

    dur_arr = pd.Series(durations).to_numpy(dtype=float)
    ev_arr = pd.Series(events).to_numpy(dtype=float)

    is_comparison = False
    if scores_old is not None and scores_new is not None:
        is_comparison = True
        s_old = pd.Series(scores_old).to_numpy(dtype=float)
        s_new = pd.Series(scores_new).to_numpy(dtype=float)
        valid = ~(
            np.isnan(dur_arr) | np.isnan(ev_arr) | np.isnan(s_old) | np.isnan(s_new)
        )
        dur_arr = dur_arr[valid]
        ev_arr = ev_arr[valid].astype(int)
        s_old = s_old[valid]
        s_new = s_new[valid]
        s_arr = s_new
    elif isinstance(scores, dict):
        is_comparison = True
        s_old = pd.Series(scores.get("old", scores.get("baseline"))).to_numpy(
            dtype=float
        )
        s_new = pd.Series(scores.get("new", scores.get("updated"))).to_numpy(
            dtype=float
        )
        valid = ~(
            np.isnan(dur_arr) | np.isnan(ev_arr) | np.isnan(s_old) | np.isnan(s_new)
        )
        dur_arr = dur_arr[valid]
        ev_arr = ev_arr[valid].astype(int)
        s_old = s_old[valid]
        s_new = s_new[valid]
        s_arr = s_new
    else:
        if scores is None:
            raise ValueError("Must provide scores or both scores_old and scores_new.")
        s_arr = pd.Series(scores).to_numpy(dtype=float)
        valid = ~(np.isnan(dur_arr) | np.isnan(ev_arr) | np.isnan(s_arr))
        dur_arr = dur_arr[valid]
        ev_arr = ev_arr[valid].astype(int)
        s_arr = s_arr[valid]
        s_old = None
        s_new = None

    if (dur_arr < 0).any():
        raise ValueError("Time-dependent ROC durations must be non-negative (>= 0).")

    unique_ev = set(np.unique(ev_arr))
    if competing_event_code is None:
        if not unique_ev.issubset({0, 1}):
            raise ValueError(
                f"Event indicator must contain binary values (0 = censored, 1 = event). Found: {sorted(list(unique_ev))}"
            )
    else:
        if competing_event_code in (0, 1):
            raise ValueError(
                f"competing_event_code cannot be 0 or 1 (0 is censored, 1 is primary event). Received: {competing_event_code}"
            )
        if not unique_ev.issubset({0, 1, competing_event_code}):
            raise ValueError(
                f"Event indicator contains values outside {{0, 1, {competing_event_code}}}: {sorted(list(unique_ev))}"
            )

    n_obs = len(dur_arr)
    if n_obs < 10:
        raise ValueError("Time-dependent ROC requires at least 10 valid observations.")
    max_t = float(np.max(dur_arr))

    if time_horizons is None or len(time_horizons) == 0:
        # Default to 25th, 50th, 75th percentiles of observed event times
        ev_times = dur_arr[ev_arr == 1]
        if len(ev_times) >= 3:
            time_horizons = [float(np.percentile(ev_times, p)) for p in [25, 50, 75]]
        else:
            time_horizons = [max_t * 0.25, max_t * 0.5, max_t * 0.75]
    sorted_horizons = sorted([float(h) for h in time_horizons])
    if any(h <= 0 for h in sorted_horizons):
        raise ValueError("Time horizons must contain positive follow-up times (> 0).")

    # Fit KM estimator for censoring survival distribution G(t) = P(C > t)
    censor_event = (ev_arr == 0).astype(int)
    km_censor = KaplanMeierFitter()
    km_censor.fit(dur_arr, event_observed=censor_event)

    g_durations = _lookup_censoring_survival_left_limit(km_censor, dur_arr)

    results_by_horizon = {}
    comparison_by_horizon = {} if is_comparison else None
    non_sparse_horizons = []

    for h in sorted_horizons:
        g_h = max(float(km_censor.survival_function_at_times(h).iloc[0]), 1e-3)

        res_h = _calculate_ipcw_roc_at_horizon(
            durations=dur_arr,
            events=ev_arr,
            scores=s_arr,
            horizon=h,
            g_durations=g_durations,
            g_horizon=g_h,
            competing_event_code=competing_event_code,
            return_curve=True,
        )

        n_cases = res_h["n_cases"]
        n_controls = res_h["n_controls"]

        # Horizon Boundary Guard
        warning_msg = None
        is_sparse = False
        if h > max_t:
            warning_msg = f"Horizon t={h} exceeds maximum follow-up ({max_t:.2f}). Confidence interval marked as NaN."
            is_sparse = True
        elif n_cases < 5 or n_controls < 5:
            warning_msg = f"Sparse risk set at horizon t={h} (n_cases={n_cases}, n_controls={n_controls}). Minimum 5 required for stable variance. Confidence interval marked as NaN."
            is_sparse = True

        res_h["warning"] = warning_msg

        if is_sparse:
            res_h["ci_lower"] = float("nan")
            res_h["ci_upper"] = float("nan")
            res_h["se"] = float("nan")
            res_h["ci_method"] = None
        else:
            non_sparse_horizons.append(h)
            if n_bootstrap == 0:
                # Analytical Hanley-McNeil IPCW approximation
                auc_val = res_h["auroc"]
                if np.isnan(auc_val) or n_cases < 1 or n_controls < 1:
                    res_h["se"] = float("nan")
                    res_h["ci_lower"] = float("nan")
                    res_h["ci_upper"] = float("nan")
                    res_h["ci_method"] = None
                else:
                    q1 = auc_val / (2.0 - auc_val)
                    q2 = 2.0 * auc_val**2 / (1.0 + auc_val)
                    var_h = (
                        auc_val * (1.0 - auc_val)
                        + (n_cases - 1) * (q1 - auc_val**2)
                        + (n_controls - 1) * (q2 - auc_val**2)
                    ) / (n_cases * n_controls)
                    se_h = float(np.sqrt(max(0.0, var_h)))
                    res_h["se"] = se_h
                    res_h["ci_lower"] = max(
                        0.0, float(auc_val - 1.959963984540054 * se_h)
                    )
                    res_h["ci_upper"] = min(
                        1.0, float(auc_val + 1.959963984540054 * se_h)
                    )
                    res_h["ci_method"] = "hanley_mcneil"

        results_by_horizon[h] = res_h

        # Dual model comparison at horizon h
        if is_comparison and s_old is not None and s_new is not None:
            res_old_h = _calculate_ipcw_roc_at_horizon(
                dur_arr,
                ev_arr,
                s_old,
                h,
                g_durations,
                g_h,
                competing_event_code,
                return_curve=False,
            )
            res_new_h = res_h

            delta_auc = (
                res_new_h["auroc"] - res_old_h["auroc"]
                if not (np.isnan(res_new_h["auroc"]) or np.isnan(res_old_h["auroc"]))
                else float("nan")
            )

            comp_entry: dict[str, Any] = {
                "horizon": h,
                "auroc_old": res_old_h["auroc"],
                "auroc_new": res_new_h["auroc"],
                "delta_auroc": delta_auc,
                "n_cases": n_cases,
                "n_controls": n_controls,
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "se": float("nan"),
                "p_value": float("nan"),
                "ci_method": None,
            }
            comparison_by_horizon[h] = comp_entry

    # Bootstrap CI estimation (fit KM fitter ONCE per replicate across all horizons)
    if n_bootstrap > 0 and len(non_sparse_horizons) > 0:
        rng = np.random.default_rng(random_state)
        boot_indices = [
            rng.choice(n_obs, size=n_obs, replace=True) for _ in range(n_bootstrap)
        ]
        boot_aucs_by_h: dict[float, list[float]] = {h: [] for h in non_sparse_horizons}
        boot_deltas_by_h: dict[float, list[float]] = {
            h: [] for h in non_sparse_horizons
        }

        for b_idx in boot_indices:
            d_b = dur_arr[b_idx]
            e_b = ev_arr[b_idx]
            s_b = s_arr[b_idx]
            km_b = KaplanMeierFitter()
            km_b.fit(d_b, event_observed=(e_b == 0).astype(int))
            g_dur_b = _lookup_censoring_survival_left_limit(km_b, d_b)

            for h in non_sparse_horizons:
                g_h_b = max(float(km_b.survival_function_at_times(h).iloc[0]), 1e-3)
                b_res = _calculate_ipcw_roc_at_horizon(
                    d_b,
                    e_b,
                    s_b,
                    h,
                    g_dur_b,
                    g_h_b,
                    competing_event_code,
                    return_curve=False,
                )
                if not np.isnan(b_res["auroc"]):
                    boot_aucs_by_h[h].append(b_res["auroc"])

                if is_comparison and s_old is not None and s_new is not None:
                    b_old = _calculate_ipcw_roc_at_horizon(
                        d_b,
                        e_b,
                        s_old[b_idx],
                        h,
                        g_dur_b,
                        g_h_b,
                        competing_event_code,
                        return_curve=False,
                    )
                    if not (np.isnan(b_old["auroc"]) or np.isnan(b_res["auroc"])):
                        boot_deltas_by_h[h].append(b_res["auroc"] - b_old["auroc"])

        for h in non_sparse_horizons:
            b_aucs = boot_aucs_by_h[h]
            if len(b_aucs) >= 10:
                results_by_horizon[h]["ci_lower"] = float(np.percentile(b_aucs, 2.5))
                results_by_horizon[h]["ci_upper"] = float(np.percentile(b_aucs, 97.5))
                results_by_horizon[h]["se"] = float(np.std(b_aucs, ddof=1))
                results_by_horizon[h]["ci_method"] = "bootstrap"
            else:
                results_by_horizon[h]["ci_lower"] = float("nan")
                results_by_horizon[h]["ci_upper"] = float("nan")
                results_by_horizon[h]["se"] = float("nan")
                results_by_horizon[h]["ci_method"] = None

            if is_comparison and comparison_by_horizon is not None:
                b_deltas = boot_deltas_by_h[h]
                d_auc = comparison_by_horizon[h]["delta_auroc"]
                if len(b_deltas) >= 10 and not np.isnan(d_auc):
                    se_d = float(np.std(b_deltas, ddof=1))
                    comparison_by_horizon[h]["se"] = se_d
                    comparison_by_horizon[h]["ci_lower"] = float(
                        np.percentile(b_deltas, 2.5)
                    )
                    comparison_by_horizon[h]["ci_upper"] = float(
                        np.percentile(b_deltas, 97.5)
                    )
                    z_d = float(d_auc / se_d) if se_d > 0.0 else 0.0
                    comparison_by_horizon[h]["p_value"] = float(
                        2.0 * (1.0 - stats.norm.cdf(abs(z_d)))
                    )
                    comparison_by_horizon[h]["ci_method"] = "bootstrap"
                else:
                    comparison_by_horizon[h]["ci_lower"] = float("nan")
                    comparison_by_horizon[h]["ci_upper"] = float("nan")
                    comparison_by_horizon[h]["se"] = float("nan")
                    comparison_by_horizon[h]["p_value"] = float("nan")
                    comparison_by_horizon[h]["ci_method"] = None

    trajectory = {
        "horizons": sorted_horizons,
        "auroc": [results_by_horizon[h]["auroc"] for h in sorted_horizons],
        "ci_lower": [results_by_horizon[h]["ci_lower"] for h in sorted_horizons],
        "ci_upper": [results_by_horizon[h]["ci_upper"] for h in sorted_horizons],
    }

    out: dict[str, Any] = {
        "time_horizons": sorted_horizons,
        "results_by_horizon": results_by_horizon,
        "trajectory": trajectory,
        "ci_method": "bootstrap" if n_bootstrap > 0 else "hanley_mcneil",
    }
    if is_comparison:
        out["comparison"] = comparison_by_horizon

    return out

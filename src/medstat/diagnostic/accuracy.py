"""
2x2 Diagnostic Accuracy and Contingency Table Metrics.

Calculates sensitivity, specificity, positive/negative predictive values,
likelihood ratios, diagnostic odds ratio, and Wilson score confidence intervals.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats


def calculate_ci_wilson_score(
    k: float, n: float, ci: float = 0.95
) -> tuple[float, float]:
    """
    Compute the Wilson score confidence interval for a binomial proportion.
    """
    if n <= 0 or np.isnan(k) or np.isnan(n):
        return (np.nan, np.nan)

    alpha = 1.0 - ci
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    p = k / n
    denom = 1.0 + (z**2) / n
    center = p + (z**2) / (2.0 * n)
    half_width = z * np.sqrt((p * (1.0 - p) + (z**2) / (4.0 * n)) / n)

    low = max(0.0, (center - half_width) / denom)
    high = min(1.0, (center + half_width) / denom)
    return (float(low), float(high))


def calculate_2x2_metrics(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    ci: float = 0.95,
) -> dict[str, Any]:
    """
    Compute comprehensive diagnostic metrics from 2x2 contingency counts.
    """
    n_pos = tp + fn
    n_neg = fp + tn
    total = tp + fp + fn + tn

    if total == 0:
        raise ValueError("Total sample size in 2x2 table cannot be 0.")

    # Sensitivity & Specificity
    sens = tp / n_pos if n_pos > 0 else np.nan
    spec = tn / n_neg if n_neg > 0 else np.nan
    sens_ci = calculate_ci_wilson_score(tp, n_pos, ci=ci)
    spec_ci = calculate_ci_wilson_score(tn, n_neg, ci=ci)

    # PPV & NPV
    n_test_pos = tp + fp
    n_test_neg = fn + tn
    ppv = tp / n_test_pos if n_test_pos > 0 else np.nan
    npv = tn / n_test_neg if n_test_neg > 0 else np.nan
    ppv_ci = calculate_ci_wilson_score(tp, n_test_pos, ci=ci)
    npv_ci = calculate_ci_wilson_score(tn, n_test_neg, ci=ci)

    # Accuracy
    acc = (tp + tn) / total
    acc_ci = calculate_ci_wilson_score(tp + tn, total, ci=ci)

    # Likelihood Ratios with log-method CIs
    z = stats.norm.ppf(1.0 - (1.0 - ci) / 2.0)

    # LR+ = Sens / (1 - Spec) = (TP / n_pos) / (FP / n_neg)
    if (1.0 - spec) > 0 and not np.isnan(sens):
        lr_plus = sens / (1.0 - spec)
        # Var(ln LR+) approx = 1/TP - 1/n_pos + 1/FP - 1/n_neg
        if tp > 0 and fp > 0:
            se_ln_lrp = np.sqrt(1.0 / tp - 1.0 / n_pos + 1.0 / fp - 1.0 / n_neg)
            lr_plus_ci = (
                float(np.exp(np.log(lr_plus) - z * se_ln_lrp)),
                float(np.exp(np.log(lr_plus) + z * se_ln_lrp)),
            )
        else:
            lr_plus_ci = (np.nan, np.nan)
    else:
        lr_plus = np.nan
        lr_plus_ci = (np.nan, np.nan)

    # LR- = (1 - Sens) / Spec = (FN / n_pos) / (TN / n_neg)
    if spec > 0 and not np.isnan(sens):
        lr_minus = (1.0 - sens) / spec
        if fn > 0 and tn > 0:
            se_ln_lrm = np.sqrt(1.0 / fn - 1.0 / n_pos + 1.0 / tn - 1.0 / n_neg)
            lr_minus_ci = (
                float(np.exp(np.log(lr_minus) - z * se_ln_lrm)),
                float(np.exp(np.log(lr_minus) + z * se_ln_lrm)),
            )
        else:
            lr_minus_ci = (np.nan, np.nan)
    else:
        lr_minus = np.nan
        lr_minus_ci = (np.nan, np.nan)

    # Diagnostic Odds Ratio (DOR) = (TP * TN) / (FP * FN)
    if fp * fn > 0 and tp * tn > 0:
        dor = float((tp * tn) / (fp * fn))
        se_ln_dor = np.sqrt(1.0 / tp + 1.0 / tn + 1.0 / fp + 1.0 / fn)
        dor_ci = (
            float(np.exp(np.log(dor) - z * se_ln_dor)),
            float(np.exp(np.log(dor) + z * se_ln_dor)),
        )
    else:
        # Haldane-Anscombe 0.5 correction for zero cells
        dor = float(((tp + 0.5) * (tn + 0.5)) / ((fp + 0.5) * (fn + 0.5)))
        se_ln_dor = np.sqrt(
            1.0 / (tp + 0.5) + 1.0 / (tn + 0.5) + 1.0 / (fp + 0.5) + 1.0 / (fn + 0.5)
        )
        dor_ci = (
            float(np.exp(np.log(dor) - z * se_ln_dor)),
            float(np.exp(np.log(dor) + z * se_ln_dor)),
        )

    # Youden's Index
    youden_j = (sens + spec - 1.0) if not (np.isnan(sens) or np.isnan(spec)) else np.nan

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "total": int(total),
        "prevalence": float(n_pos / total),
        "sensitivity": float(sens),
        "sensitivity_ci": [float(sens_ci[0]), float(sens_ci[1])],
        "specificity": float(spec),
        "specificity_ci": [float(spec_ci[0]), float(spec_ci[1])],
        "ppv": float(ppv),
        "ppv_ci": [float(ppv_ci[0]), float(ppv_ci[1])],
        "npv": float(npv),
        "npv_ci": [float(npv_ci[0]), float(npv_ci[1])],
        "accuracy": float(acc),
        "accuracy_ci": [float(acc_ci[0]), float(acc_ci[1])],
        "lr_plus": float(lr_plus) if np.isfinite(lr_plus) else np.nan,
        "lr_plus_ci": [float(lr_plus_ci[0]), float(lr_plus_ci[1])],
        "lr_minus": float(lr_minus) if np.isfinite(lr_minus) else np.nan,
        "lr_minus_ci": [float(lr_minus_ci[0]), float(lr_minus_ci[1])],
        "dor": float(dor),
        "dor_ci": [float(dor_ci[0]), float(dor_ci[1])],
        "youden_index": float(youden_j),
    }


def calculate_diagnostic_accuracy(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    pos_label: int | str = 1,
    ci: float = 0.95,
) -> dict[str, Any]:
    """
    Calculate 2x2 diagnostic metrics directly from true and predicted binary vectors.
    """
    y_t = (np.asarray(y_true) == pos_label).astype(int)
    y_p = (np.asarray(y_pred) == pos_label).astype(int)

    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))

    return calculate_2x2_metrics(tp=tp, fp=fp, fn=fn, tn=tn, ci=ci)

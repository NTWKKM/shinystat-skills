# Diagnostic Accuracy, Calibration & Agreement Recipes (Cloud Sandbox)

> **Scope:** Recipes 2–5: 2x2 diagnostic accuracy with Wilson score CIs, empirical ROC and single / paired DeLong, calibration and Vickers DCA, Bland-Altman LoA and Shrout-Fleiss ICC.

---

## 2. 2x2 Diagnostic Accuracy with Wilson Score & Log-Scale 95% CIs

Computes clinical diagnostic performance metrics with asymmetric Wilson score intervals (proportions) and log-scale intervals (likelihood ratios and diagnostic odds ratios) with Haldane-Anscombe zero-cell correction.

```python
import numpy as np
import pandas as pd
import scipy.stats as stats


def calculate_ci_wilson_score(k: int, n: int, ci: float = 0.95) -> tuple[float, float]:
    """Compute asymmetric Wilson score confidence interval for binomial proportion."""
    if not isinstance(k, (int, np.integer)) or not isinstance(n, (int, np.integer)):
        raise ValueError(
            f"k and n must be integers; got k={type(k).__name__}, n={type(n).__name__}"
        )
    if n <= 0:
        raise ValueError(f"n must be strictly positive (> 0); got n={n}")
    if k < 0 or k > n:
        raise ValueError(f"k must satisfy 0 <= k <= n; got k={k}, n={n}")
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci must be strictly between 0 and 1; got ci={ci}")
    alpha = 1.0 - ci
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    p = k / n
    denom = 1.0 + (z**2) / n
    center = p + (z**2) / (2.0 * n)
    half_width = z * np.sqrt((p * (1.0 - p) + (z**2) / (4.0 * n)) / n)
    return (
        float(max(0.0, (center - half_width) / denom)),
        float(min(1.0, (center + half_width) / denom)),
    )


def calculate_2x2_metrics(tp: int, fp: int, fn: int, tn: int, ci: float = 0.95) -> dict:
    """Calculate sensitivity, specificity, PPV, NPV, LR+, LR-, and DOR with complete 95% CIs."""
    for count_val, name in [(tp, "tp"), (fp, "fp"), (fn, "fn"), (tn, "tn")]:
        if (
            not np.isfinite(count_val)
            or not float(count_val).is_integer()
            or count_val < 0
        ):
            raise ValueError(
                f"Contingency cell count '{name}' must be a finite non-negative integer (>= 0); got {count_val}."
            )
    tp, fp, fn, tn = int(tp), int(fp), int(fn), int(tn)
    if tp + fp + fn + tn == 0:
        raise ValueError("Total sample size in 2x2 contingency table must be > 0.")

    alpha = 1.0 - ci
    z = stats.norm.ppf(1.0 - alpha / 2.0)

    n_pos = tp + fn
    n_neg = fp + tn
    sens = tp / n_pos if n_pos > 0 else np.nan
    spec = tn / n_neg if n_neg > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    # Haldane-Anscombe 0.5 correction for zero-cells in ratio statistics
    adj = 0.5 if (tp == 0 or fp == 0 or fn == 0 or tn == 0) else 0.0
    tp_a, fp_a, fn_a, tn_a = tp + adj, fp + adj, fn + adj, tn + adj
    sens_a = tp_a / (tp_a + fn_a)
    spec_a = tn_a / (fp_a + tn_a)

    # Likelihood Ratios with log-scale CIs (Simel et al., 1995 / Altman 2000)
    lr_pos = sens_a / (1.0 - spec_a) if (1.0 - spec_a) > 0 else np.nan
    se_log_lr_pos = np.sqrt(
        1.0 / tp_a - 1.0 / (tp_a + fn_a) + 1.0 / fp_a - 1.0 / (fp_a + tn_a)
    )
    lr_pos_ci = (
        float(np.exp(np.log(lr_pos) - z * se_log_lr_pos)),
        float(np.exp(np.log(lr_pos) + z * se_log_lr_pos)),
    )

    lr_neg = (1.0 - sens_a) / spec_a if spec_a > 0 else np.nan
    se_log_lr_neg = np.sqrt(
        1.0 / fn_a - 1.0 / (tp_a + fn_a) + 1.0 / tn_a - 1.0 / (fp_a + tn_a)
    )
    lr_neg_ci = (
        float(np.exp(np.log(lr_neg) - z * se_log_lr_neg)),
        float(np.exp(np.log(lr_neg) + z * se_log_lr_neg)),
    )

    # Diagnostic Odds Ratio (DOR) with Woolf log-scale CI
    dor = (tp_a * tn_a) / (fp_a * fn_a) if (fp_a * fn_a) > 0 else np.nan
    se_log_dor = np.sqrt(1.0 / tp_a + 1.0 / fp_a + 1.0 / fn_a + 1.0 / tn_a)
    dor_ci = (
        float(np.exp(np.log(dor) - z * se_log_dor)),
        float(np.exp(np.log(dor) + z * se_log_dor)),
    )

    def _wilson_or_nan(k: int, n: int) -> tuple[float, float]:
        # Undefined proportion (zero denominator): no interval.
        return calculate_ci_wilson_score(k, n, ci) if n > 0 else (np.nan, np.nan)

    return {
        "Sensitivity": (float(sens), _wilson_or_nan(tp, n_pos)),
        "Specificity": (float(spec), _wilson_or_nan(tn, n_neg)),
        "PPV": (float(ppv), _wilson_or_nan(tp, tp + fp)),
        "NPV": (float(npv), _wilson_or_nan(tn, tn + fn)),
        "LR_pos": (float(lr_pos), lr_pos_ci),
        "LR_neg": (float(lr_neg), lr_neg_ci),
        "DOR": (float(dor), dor_ci),
        "contingency_matrix": [[tp, fn], [fp, tn]],
        "confusion_table": pd.DataFrame(
            {"Actual_Positive": [tp, fn], "Actual_Negative": [fp, tn]},
            index=["Predicted_Positive", "Predicted_Negative"],
        ),
    }
```

---

---

## 3. Empirical ROC, Optimal Cutoff & Single / Paired DeLong 95% CIs

Computes empirical ROC curve, Youden's J optimal cutpoint, and DeLong covariance matrix (DeLong et al., 1988) for single-model CIs and paired ROC curve comparisons ($AUC_1 - AUC_2$). Drops missing values safely prior to integer casting.

```python
import numpy as np
import scipy.stats as stats
from sklearn.metrics import roc_curve


def _delong_placements(y_true: np.ndarray, y_score: np.ndarray):
    """Compute positive (v10) and negative (v01) placement values for DeLong test.

    Rank-based (Sun & Xu 2014): midranks reproduce the pairwise kernel
    1[pos > neg] + 0.5 * 1[pos == neg] exactly, in O(N log N) memory-light time.
    """
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    m, n = len(pos_scores), len(neg_scores)
    r_all = stats.rankdata(np.concatenate([pos_scores, neg_scores]))
    r_pos = stats.rankdata(pos_scores)
    r_neg = stats.rankdata(neg_scores)
    v10 = (r_all[:m] - r_pos) / n
    v01 = 1.0 - (r_all[m:] - r_neg) / m
    auc = float(v10.mean())
    return v10, v01, auc


def auc_ci_delong(
    y_true, y_score, direction: str = "high", alpha: float = 0.05
) -> dict:
    """Compute empirical AUC and DeLong analytical 95% confidence interval with directionality."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be strictly between 0 and 1; got {alpha}")
    if direction not in ("high", "low"):
        raise ValueError(f"direction must be 'high' or 'low'; got '{direction}'")
    y_raw = np.asarray(y_true, dtype=float)
    s_raw = np.asarray(y_score, dtype=float)
    if len(y_raw) != len(s_raw):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_raw)}) and y_score ({len(s_raw)}) must have identical lengths."
        )
    valid = ~(np.isnan(y_raw) | np.isnan(s_raw))
    y_valid = y_raw[valid]
    if not np.all(np.isin(y_valid, [0, 1])):
        raise ValueError(
            "Binary outcome y_true must contain strictly binary values (0 and 1)."
        )
    y_t = y_valid.astype(int)
    y_s = s_raw[valid]
    if direction == "low":
        y_s = -y_s

    n_pos, n_neg = int(np.sum(y_t == 1)), int(np.sum(y_t == 0))
    if n_pos < 2 or n_neg < 2:
        raise ValueError("Requires at least 2 positives and 2 negatives.")

    v10, v01, auc = _delong_placements(y_t, y_s)
    s10 = np.var(v10, ddof=1)
    s01 = np.var(v01, ddof=1)
    se_auc = float(np.sqrt(max((s10 / n_pos) + (s01 / n_neg), 0.0)))

    z = stats.norm.ppf(1.0 - alpha / 2.0)
    fpr, tpr, thresholds = roc_curve(y_t, y_s)
    youden_j = tpr - fpr
    best_idx = int(np.argmax(youden_j))
    opt_thresh = float(thresholds[best_idx])
    if direction == "low":
        opt_thresh = -opt_thresh

    return {
        "auc": float(auc),
        "se": se_auc,
        "ci_lower": float(max(0.0, auc - z * se_auc)),
        "ci_upper": float(min(1.0, auc + z * se_auc)),
        "optimal_threshold": opt_thresh,
        "sensitivity": float(tpr[best_idx]),
        "specificity": float(1.0 - fpr[best_idx]),
        "youden_j": float(youden_j[best_idx]),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "fpr": fpr,
        "tpr": tpr,
        "roc_curve": {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": -thresholds if direction == "low" else thresholds,
        },
        "optimal_point": {
            "fpr": float(fpr[best_idx]),
            "tpr": float(tpr[best_idx]),
            "cutoff": opt_thresh,
            "youden_j": float(youden_j[best_idx]),
        },
    }


def delong_paired_test(y_true, score1, score2, alpha: float = 0.05) -> dict:
    """Paired DeLong comparative test between two correlated ROC curves on the same cohort."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be strictly between 0 and 1; got {alpha}")
    y_raw = np.asarray(y_true, dtype=float)
    s1_raw = np.asarray(score1, dtype=float)
    s2_raw = np.asarray(score2, dtype=float)
    if len(y_raw) != len(s1_raw) or len(y_raw) != len(s2_raw):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_raw)}), score1 ({len(s1_raw)}), and score2 ({len(s2_raw)}) must have identical lengths."
        )
    valid = ~(np.isnan(y_raw) | np.isnan(s1_raw) | np.isnan(s2_raw))
    y_valid = y_raw[valid]
    if not np.all(np.isin(y_valid, [0, 1])):
        raise ValueError(
            "Binary outcome y_true must contain strictly binary values (0 and 1)."
        )
    y_t = y_valid.astype(int)
    s1 = s1_raw[valid]
    s2 = s2_raw[valid]

    n_pos, n_neg = int(np.sum(y_t == 1)), int(np.sum(y_t == 0))
    if n_pos < 2 or n_neg < 2:
        raise ValueError("Paired DeLong requires at least 2 positives and 2 negatives.")

    v10_1, v01_1, auc1 = _delong_placements(y_t, s1)
    v10_2, v01_2, auc2 = _delong_placements(y_t, s2)

    # Var(AUC1 - AUC2) from paired placement differences: algebraically equal to
    # var1 + var2 - 2*cov12, without the cancellation that makes small variances look zero.
    d10 = v10_1 - v10_2
    d01 = v01_1 - v01_2
    var_diff = float(np.var(d10, ddof=1) / n_pos + np.var(d01, ddof=1) / n_neg)
    # Genuinely zero only when every paired placement difference is identical.
    zero_variance = bool(np.ptp(d10) == 0.0 and np.ptp(d01) == 0.0)

    diff = auc1 - auc2
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    warning = None

    if not zero_variance and var_diff > 0.0:
        se_diff = float(np.sqrt(var_diff))
        z_stat = diff / se_diff
        p_val = float(2.0 * stats.norm.sf(abs(z_stat)))
        ci_l, ci_u = float(diff - z_crit * se_diff), float(diff + z_crit * se_diff)
    elif abs(diff) <= 1e-12:
        # Identical rankings: no difference to test.
        se_diff, z_stat, p_val, ci_l, ci_u = 0.0, 0.0, 1.0, float(diff), float(diff)
    else:
        # Degenerate placements (e.g. perfect separation): asymptotic variance is zero,
        # so the normal test is undefined. Report inference as unavailable.
        se_diff, z_stat, p_val, ci_l, ci_u = (np.nan,) * 5
        warning = (
            "DeLong variance of the AUC difference is zero (degenerate placements); "
            "z-test and CI unavailable. Use a bootstrap or exact method."
        )

    return {
        "auc1": float(auc1),
        "auc2": float(auc2),
        "difference": float(diff),
        "se_difference": se_diff,
        "ci_lower": ci_l,
        "ci_upper": ci_u,
        "z_statistic": float(z_stat),
        "p_value": p_val,
        "warning": warning,
    }
```

---

---

## 4. Model Calibration & Vickers Decision Curve Analysis (DCA)

Evaluates clinical prediction model goodness-of-fit (Austin & Steyerberg 2019 ICI with span 0.75, logistic recalibration slope & intercept) and Vickers & Elkin (2006) decision curve analysis.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import logit
from statsmodels.nonparametric.smoothers_lowess import lowess


def evaluate_calibration(y_true, y_pred) -> dict:
    """Compute Brier Score, Calibration Slope/Intercept, and Austin & Steyerberg (2019) ICI."""
    y_raw = np.asarray(y_true, dtype=float)
    p_raw = np.asarray(y_pred, dtype=float)
    if len(y_raw) != len(p_raw):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_raw)}) and y_pred ({len(p_raw)}) must have identical lengths."
        )
    valid = ~(np.isnan(y_raw) | np.isnan(p_raw))
    y_valid = y_raw[valid]
    if len(y_valid) == 0:
        raise ValueError(
            "No complete-case observations remaining for calibration evaluation."
        )
    if not np.all(np.isin(y_valid, [0, 1])):
        raise ValueError(
            "Binary outcome y_true must contain strictly binary values (0 and 1)."
        )
    p_valid = p_raw[valid]
    if (p_valid < 0.0).any() or (p_valid > 1.0).any():
        raise ValueError("Predicted probabilities y_pred must be in the range [0, 1].")
    y_t = y_valid.astype(int)
    y_p = np.clip(p_valid, 1e-6, 1.0 - 1e-6)

    # 1. Brier score
    brier = float(np.mean((y_t - y_p) ** 2))
    prev = float(np.mean(y_t))
    scaled_brier = (
        1.0 - (brier / (prev * (1.0 - prev))) if (0.0 < prev < 1.0) else np.nan
    )

    # 2. Calibration slope & intercept via logistic recalibration
    log_odds = logit(y_p)
    try:
        X_slope = sm.add_constant(log_odds)
        model_slope = sm.Logit(y_t, X_slope).fit(disp=False)
        slope = float(model_slope.params[1])
    except Exception:
        slope = np.nan

    try:
        model_inter = sm.Logit(y_t, np.ones_like(y_t), offset=log_odds).fit(disp=False)
        intercept = float(model_inter.params[0])
    except Exception:
        intercept = np.nan

    # 3. Austin & Steyerberg (2019) ICI using LOWESS smoother (span = 0.75)
    order = np.argsort(y_p)
    smooth_obs = lowess(y_t[order], y_p[order], frac=0.75, it=0, return_sorted=False)
    smooth_obs = np.clip(smooth_obs, 0.0, 1.0)
    abs_err = np.abs(y_p[order] - smooth_obs)

    # 4. Decile calibration bins for plotting
    n_bins = 10
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.unique(np.percentile(y_p, quantiles))
    if len(bin_edges) < 2:
        bin_edges = np.array([float(np.min(y_p)) - 1e-5, float(np.max(y_p)) + 1e-5])
    else:
        bin_edges[-1] += 1e-5
    p_means, o_rates, b_counts, ci_l_list, ci_u_list = [], [], [], [], []
    for b in range(len(bin_edges) - 1):
        mask = (y_p >= bin_edges[b]) & (y_p < bin_edges[b + 1])
        cnt = int(np.sum(mask))
        if cnt > 0:
            pm = float(np.mean(y_p[mask]))
            om = float(np.mean(y_t[mask]))
            # Binomial Wilson score CI
            k_ev = int(np.sum(y_t[mask]))
            z_val = 1.96
            denom = 1.0 + (z_val**2) / cnt
            center = om + (z_val**2) / (2.0 * cnt)
            hw = z_val * np.sqrt((om * (1.0 - om) + (z_val**2) / (4.0 * cnt)) / cnt)
            ci_l = float(max(0.0, (center - hw) / denom))
            ci_u = float(min(1.0, (center + hw) / denom))
        else:
            pm, om, ci_l, ci_u = np.nan, np.nan, np.nan, np.nan
        p_means.append(pm)
        o_rates.append(om)
        b_counts.append(cnt)
        ci_l_list.append(ci_l)
        ci_u_list.append(ci_u)

    return {
        "brier_score": brier,
        "scaled_brier": float(scaled_brier),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "ici": float(np.mean(abs_err)),
        "e50": float(np.median(abs_err)),
        "e90": float(np.percentile(abs_err, 90)),
        "emax": float(np.max(abs_err)),
        "calibration_bins": {
            "pred_mean": np.array(p_means),
            "obs_rate": np.array(o_rates),
            "counts": np.array(b_counts),
            "ci_lower": np.array(ci_l_list),
            "ci_upper": np.array(ci_u_list),
        },
        "calibration_curve": {
            "pred_smooth": y_p[order],
            "obs_smooth": smooth_obs,
        },
    }


def calculate_dca(y_true, y_pred, thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """Calculate Vickers & Elkin (2006) Clinical Net Benefit across decision thresholds.

    Validation & Performance Note:
    If predicted scores fall outside [0, 1], an in-sample logistic recalibration is fitted.
    Evaluating net benefit on the same dataset used for recalibration reflects apparent
    development performance. For rigorous validation, evaluate on independent test data.
    """
    y_raw = np.asarray(y_true, dtype=float)
    p_raw = np.asarray(y_pred, dtype=float)
    if len(y_raw) != len(p_raw):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_raw)}) and y_pred ({len(p_raw)}) must have identical lengths."
        )
    valid = ~(np.isnan(y_raw) | np.isnan(p_raw))
    y_valid = y_raw[valid]
    if len(y_valid) == 0:
        raise ValueError(
            "No complete cases remaining after removing missing values."
        )
    if not np.all(np.isin(y_valid, [0, 1])):
        raise ValueError(
            "Binary outcome y_true must contain strictly binary values (0 and 1)."
        )
    if len(np.unique(y_valid)) < 2:
        raise ValueError(
            "Binary outcome y_true must contain both event classes (0 and 1)."
        )
    y_t = y_valid.astype(int)
    y_p = p_raw[valid]

    # Map raw biomarker to probability if not in [0, 1]
    internally_calibrated = False
    if (y_p < 0.0).any() or (y_p > 1.0).any():
        internally_calibrated = True
        calib_fit = sm.Logit(y_t, sm.add_constant(y_p)).fit(disp=False)
        y_p = calib_fit.predict(sm.add_constant(y_p))

    n = len(y_t)
    prevalence = float(np.mean(y_t))
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    else:
        thresholds = np.asarray(thresholds, dtype=float)
        if len(thresholds) == 0:
            raise ValueError("Thresholds array cannot be empty.")
        if not np.all(np.isfinite(thresholds)) or np.any(thresholds <= 0.0) or np.any(thresholds >= 1.0):
            raise ValueError("All decision thresholds must be finite and strictly between 0 and 1.")

    records = []
    for pt in thresholds:
        weight = pt / (1.0 - pt)
        tp = np.sum((y_p >= pt) & (y_t == 1))
        fp = np.sum((y_p >= pt) & (y_t == 0))
        nb_model = (tp / n) - (fp / n) * weight
        nb_all = prevalence - (1.0 - prevalence) * weight
        records.append(
            {"threshold": pt, "net_benefit": float(nb_model), "strategy": "Model"}
        )
        records.append(
            {"threshold": pt, "net_benefit": float(nb_all), "strategy": "Treat All"}
        )
        records.append({"threshold": pt, "net_benefit": 0.0, "strategy": "Treat None"})

    res_df = pd.DataFrame(records)
    if internally_calibrated:
        res_df.attrs["apparent_performance_warning"] = (
            "Input scores fell outside [0, 1] and were calibrated in-sample using logistic regression. "
            "Net benefit estimates reflect apparent development performance and may be over-optimistic."
        )
    else:
        res_df.attrs["apparent_performance_warning"] = None
    res_df.attrs["internally_calibrated"] = internally_calibrated
    return res_df
```

---

---

## 5. Observer Agreement: Bland-Altman & Shrout-Fleiss (1979) ICC

Evaluates quantitative agreement and inter-rater reliability using pure NumPy/SciPy two-way ANOVA decomposition (Shrout & Fleiss 1979; McGraw & Wong 1996) with **exact F-distribution 95% Confidence Intervals** across all 6 forms.

```python
import numpy as np
import pandas as pd
import scipy.stats as stats


def calculate_bland_altman(m1, m2, ci: float = 0.95) -> dict:
    """Compute Bland-Altman mean difference and 95% Limits of Agreement with 1999 large-sample CIs."""
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci must be strictly between 0 and 1; got {ci}")
    v1 = np.asarray(m1, dtype=float)
    v2 = np.asarray(m2, dtype=float)
    if len(v1) != len(v2):
        raise ValueError(
            f"Length mismatch: m1 ({len(v1)}) and m2 ({len(v2)}) must have identical lengths."
        )
    valid = np.isfinite(v1) & np.isfinite(v2)
    diffs = v1[valid] - v2[valid]
    means = (v1[valid] + v2[valid]) / 2.0
    n = len(diffs)
    if n < 2:
        raise ValueError(
            "Bland-Altman difference analysis requires at least 2 valid paired measurements."
        )

    mean_diff = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1))
    se_mean = sd_diff / np.sqrt(n)

    t_crit = stats.t.ppf(1.0 - (1.0 - ci) / 2.0, df=n - 1)
    # Limits of agreement always cover 95% of differences; `ci` sets only the CI level.
    z_loa = stats.norm.ppf(0.975)
    upper_loa = mean_diff + z_loa * sd_diff
    lower_loa = mean_diff - z_loa * sd_diff

    # Standard error of Limits of Agreement (Bland & Altman, 1999)
    se_loa = float(np.sqrt((1.0 / n + (z_loa**2) / (2.0 * (n - 1))) * (sd_diff**2)))

    return {
        "n_pairs": n,
        "mean_diff": mean_diff,
        "ci_mean_diff": (mean_diff - t_crit * se_mean, mean_diff + t_crit * se_mean),
        "sd_diff": sd_diff,
        "upper_loa": upper_loa,
        "ci_upper_loa": (upper_loa - t_crit * se_loa, upper_loa + t_crit * se_loa),
        "lower_loa": lower_loa,
        "ci_lower_loa": (lower_loa - t_crit * se_loa, lower_loa + t_crit * se_loa),
        "ba_points": {
            "means": means,
            "diffs": diffs,
            "mean_diff": mean_diff,
            "loa_lower": lower_loa,
            "loa_upper": upper_loa,
            "ci_mean_diff": (
                float(mean_diff - t_crit * se_mean),
                float(mean_diff + t_crit * se_mean),
            ),
            "ci_loa_lower": (
                float(lower_loa - t_crit * se_loa),
                float(lower_loa + t_crit * se_loa),
            ),
            "ci_loa_upper": (
                float(upper_loa - t_crit * se_loa),
                float(upper_loa + t_crit * se_loa),
            ),
        },
    }


def calculate_icc_matrix(values: np.ndarray, alpha: float = 0.05) -> pd.DataFrame:
    """Compute all 6 Shrout & Fleiss (1979) ICC variants with exact F-distribution 95% CIs."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be strictly between 0 and 1; got {alpha}")
    mat = np.asarray(values, dtype=float)
    if mat.ndim != 2:
        raise ValueError(f"ICC input must be a 2D matrix; got ndim={mat.ndim}")
    if not np.all(np.isfinite(mat)):
        raise ValueError(
            "ICC matrix contains missing or non-finite values. Impute or subset complete cases."
        )
    n, k = mat.shape
    if n < 2 or k < 2:
        raise ValueError("ICC requires at least 2 targets and 2 raters.")

    grand_mean = np.mean(mat)
    # Check all-constant matrix: target variance is zero, reliability undefined
    if np.all(mat == grand_mean):
        return pd.DataFrame(
            [
                {
                    "Type": t,
                    "Description": d,
                    "ICC": np.nan,
                    "CI_lower": np.nan,
                    "CI_upper": np.nan,
                    "F": np.nan,
                    "pval": np.nan,
                }
                for t, d in [
                    ("ICC1", "One-way random (single)"),
                    ("ICC2", "Two-way random agreement"),
                    ("ICC3", "Two-way mixed consistency"),
                    ("ICC1k", "One-way random (average)"),
                    ("ICC2k", "Two-way random agreement (avg)"),
                    ("ICC3k", "Two-way mixed consistency (avg)"),
                ]
            ]
        )

    row_means = np.mean(mat, axis=1)
    col_means = np.mean(mat, axis=0)

    SST = float(np.sum((mat - grand_mean) ** 2))
    SSB = float(k * np.sum((row_means - grand_mean) ** 2))
    SSJ = float(n * np.sum((col_means - grand_mean) ** 2))
    SSE = max(float(SST - SSB - SSJ), 0.0)
    SSW = max(float(SST - SSB), 0.0)

    df_B, df_J, df_E, df_W = n - 1, k - 1, (n - 1) * (k - 1), n * (k - 1)
    MSB = SSB / df_B if df_B > 0 else 0.0
    MSJ = SSJ / df_J if df_J > 0 else 0.0
    MSE = SSE / df_E if df_E > 0 else 0.0
    MSW = SSW / df_W if df_W > 0 else 0.0

    # Perfect agreement boundary handling
    # 1. Model 1
    if MSW == 0 and MSB > 0:
        icc1 = 1.0
        icc1k = 1.0
        ci_icc1 = (1.0, 1.0)
        ci_icc1k = (1.0, 1.0)
        f_obs_1 = np.inf
        p_1 = 0.0
    else:
        icc1 = (
            (MSB - MSW) / (MSB + (k - 1) * MSW)
            if (MSB + (k - 1) * MSW) != 0
            else np.nan
        )
        icc1k = (MSB - MSW) / MSB if MSB != 0 else np.nan
        f_obs_1 = MSB / MSW if MSW > 0 else np.nan
        p_1 = (
            float(stats.f.sf(f_obs_1, df_B, df_W)) if not np.isnan(f_obs_1) else np.nan
        )
        f1_l = stats.f.ppf(1.0 - alpha / 2.0, df_B, df_W)
        f1_u = stats.f.ppf(1.0 - alpha / 2.0, df_W, df_B)
        fl1 = f_obs_1 / f1_l if (not np.isnan(f_obs_1) and f1_l > 0) else np.nan
        fu1 = f_obs_1 * f1_u if not np.isnan(f_obs_1) else np.nan
        ci_icc1 = (
            (fl1 - 1.0) / (fl1 + k - 1.0) if not np.isnan(fl1) else np.nan,
            (fu1 - 1.0) / (fu1 + k - 1.0) if not np.isnan(fu1) else np.nan,
        )
        ci_icc1k = (
            1.0 - 1.0 / fl1 if (not np.isnan(fl1) and fl1 > 0) else np.nan,
            1.0 - 1.0 / fu1 if (not np.isnan(fu1) and fu1 > 0) else np.nan,
        )

    # 2. Model 3 (Consistency)
    if MSE == 0 and MSB > 0:
        icc3 = 1.0
        icc3k = 1.0
        ci_icc3 = (1.0, 1.0)
        ci_icc3k = (1.0, 1.0)
        f_obs_23 = np.inf
        p_23 = 0.0
    else:
        denom_icc3 = MSB + (k - 1) * MSE
        icc3 = (MSB - MSE) / denom_icc3 if denom_icc3 != 0 else np.nan
        icc3k = (MSB - MSE) / MSB if MSB != 0 else np.nan
        f_obs_23 = MSB / MSE if MSE > 0 else np.nan
        p_23 = (
            float(stats.f.sf(f_obs_23, df_B, df_E))
            if not np.isnan(f_obs_23)
            else np.nan
        )
        f3_l = stats.f.ppf(1.0 - alpha / 2.0, df_B, df_E)
        f3_u = stats.f.ppf(1.0 - alpha / 2.0, df_E, df_B)
        fl3 = f_obs_23 / f3_l if (not np.isnan(f_obs_23) and f3_l > 0) else np.nan
        fu3 = f_obs_23 * f3_u if not np.isnan(f_obs_23) else np.nan
        ci_icc3 = (
            (fl3 - 1.0) / (fl3 + k - 1.0) if not np.isnan(fl3) else np.nan,
            (fu3 - 1.0) / (fu3 + k - 1.0) if not np.isnan(fu3) else np.nan,
        )
        ci_icc3k = (
            1.0 - 1.0 / fl3 if (not np.isnan(fl3) and fl3 > 0) else np.nan,
            1.0 - 1.0 / fu3 if (not np.isnan(fu3) and fu3 > 0) else np.nan,
        )

    # 3. Model 2 (Absolute Agreement)
    if MSE == 0 and MSJ == 0 and MSB > 0:
        icc2 = 1.0
        icc2k = 1.0
        ci_icc2 = (1.0, 1.0)
        ci_icc2k = (1.0, 1.0)
    else:
        denom_icc2 = MSB + (k - 1) * MSE + (k / n) * (MSJ - MSE)
        icc2 = (MSB - MSE) / denom_icc2 if denom_icc2 != 0 else np.nan
        denom_icc2k = MSB + (MSJ - MSE) / n
        icc2k = (MSB - MSE) / denom_icc2k if denom_icc2k != 0 else np.nan
        if MSE == 0 and MSJ > 0:
            ci_icc2 = (np.nan, np.nan)
            ci_icc2k = (np.nan, np.nan)
        else:
            F_J = MSJ / MSE if MSE > 0 else 1.0
            r2 = icc2 if not np.isnan(icc2) else 0.0
            v_num = df_E * (k * r2 * F_J + n * (1.0 + (k - 1) * r2) - k * r2) ** 2
            v_den = df_B * (k * r2 * F_J) ** 2 + (n * (1.0 + (k - 1) * r2) - k * r2) ** 2
            v = max(v_num / v_den, 1.0) if v_den != 0 else 1.0
            f2_u = stats.f.ppf(1.0 - alpha / 2.0, df_B, v)
            f2_l = stats.f.ppf(1.0 - alpha / 2.0, v, df_B)
            denom_L2 = f2_u * (k * MSJ + (k * n - k - n) * MSE) + n * MSB
            L2 = n * (MSB - f2_u * MSE) / denom_L2 if denom_L2 != 0 else np.nan
            denom_U2 = k * MSJ + (k * n - k - n) * MSE + n * f2_l * MSB
            U2 = n * (f2_l * MSB - MSE) / denom_U2 if denom_U2 != 0 else np.nan
            ci_icc2 = (L2, U2)
            ci_icc2k = (
                (k * L2) / (1.0 + (k - 1) * L2) if (1.0 + (k - 1) * L2) != 0 else np.nan,
                (k * U2) / (1.0 + (k - 1) * U2) if (1.0 + (k - 1) * U2) != 0 else np.nan,
            )

    return pd.DataFrame(
        [
            {
                "Type": "ICC1",
                "Description": "One-way random (single)",
                "ICC": icc1,
                "CI_lower": ci_icc1[0],
                "CI_upper": ci_icc1[1],
                "F": f_obs_1,
                "pval": p_1,
            },
            {
                "Type": "ICC2",
                "Description": "Two-way random agreement",
                "ICC": icc2,
                "CI_lower": ci_icc2[0],
                "CI_upper": ci_icc2[1],
                "F": f_obs_23,
                "pval": p_23,
            },
            {
                "Type": "ICC3",
                "Description": "Two-way mixed consistency",
                "ICC": icc3,
                "CI_lower": ci_icc3[0],
                "CI_upper": ci_icc3[1],
                "F": f_obs_23,
                "pval": p_23,
            },
            {
                "Type": "ICC1k",
                "Description": "One-way random (average)",
                "ICC": icc1k,
                "CI_lower": ci_icc1k[0],
                "CI_upper": ci_icc1k[1],
                "F": f_obs_1,
                "pval": p_1,
            },
            {
                "Type": "ICC2k",
                "Description": "Two-way random agreement (avg)",
                "ICC": icc2k,
                "CI_lower": ci_icc2k[0],
                "CI_upper": ci_icc2k[1],
                "F": f_obs_23,
                "pval": p_23,
            },
            {
                "Type": "ICC3k",
                "Description": "Two-way mixed consistency (avg)",
                "ICC": icc3k,
                "CI_lower": ci_icc3k[0],
                "CI_upper": ci_icc3k[1],
                "F": f_obs_23,
                "pval": p_23,
            },
        ]
    )
```

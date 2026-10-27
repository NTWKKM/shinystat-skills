# Diagnostics, Agreement, Validation & Sample Size Recipes (Cloud Sandbox)

> **Scope:** 2x2 diagnostic accuracy, Wilson score CIs, empirical ROC/DeLong test, calibration, Vickers DCA, Bland-Altman LoA, Shrout-Fleiss ICC, NRI/IDI reclassification, time-dependent ROC, TRIPOD+AI bootstrap internal validation, model recalibration, and Riley/Obuchowski sample size determinations.



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

    return {
        "Sensitivity": (float(sens), calculate_ci_wilson_score(tp, n_pos, ci)),
        "Specificity": (float(spec), calculate_ci_wilson_score(tn, n_neg, ci)),
        "PPV": (float(ppv), calculate_ci_wilson_score(tp, tp + fp, ci)),
        "NPV": (float(npv), calculate_ci_wilson_score(tn, tn + fn, ci)),
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
    """Compute positive (v10) and negative (v01) placement values for DeLong test."""
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    greater = (pos_scores[:, None] > neg_scores).astype(float)
    equal = (pos_scores[:, None] == neg_scores).astype(float)
    comp = greater + 0.5 * equal
    v10 = comp.mean(axis=1)
    v01 = comp.mean(axis=0)
    auc = float(v10.mean())
    return v10, v01, auc


def auc_ci_delong(
    y_true, y_score, direction: str = "high", alpha: float = 0.05
) -> dict:
    """Compute empirical AUC and DeLong analytical 95% confidence interval with directionality."""
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

    var1 = (np.var(v10_1, ddof=1) / n_pos) + (np.var(v01_1, ddof=1) / n_neg)
    var2 = (np.var(v10_2, ddof=1) / n_pos) + (np.var(v01_2, ddof=1) / n_neg)
    cov12 = (np.cov(v10_1, v10_2, ddof=1)[0, 1] / n_pos) + (
        np.cov(v01_1, v01_2, ddof=1)[0, 1] / n_neg
    )

    diff = auc1 - auc2
    var_diff = max(var1 + var2 - 2.0 * cov12, 1e-12)
    se_diff = float(np.sqrt(var_diff))

    z_stat = diff / se_diff
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)

    return {
        "auc1": float(auc1),
        "auc2": float(auc2),
        "difference": float(diff),
        "se_difference": se_diff,
        "ci_lower": float(diff - z_crit * se_diff),
        "ci_upper": float(diff + z_crit * se_diff),
        "z_statistic": float(z_stat),
        "p_value": p_val,
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
    v1 = np.asarray(m1, dtype=float)
    v2 = np.asarray(m2, dtype=float)
    if len(v1) != len(v2):
        raise ValueError(
            f"Length mismatch: m1 ({len(v1)}) and m2 ({len(v2)}) must have identical lengths."
        )
    valid = ~(np.isnan(v1) | np.isnan(v2))
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
    z_loa = stats.norm.ppf(1.0 - (1.0 - ci) / 2.0)
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
    mat = np.asarray(values, dtype=float)
    if np.isnan(mat).any():
        raise ValueError(
            "ICC matrix contains missing values. Impute or subset complete cases."
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

---

---

## 16. Diagnostic Risk Reclassification: Continuous & Categorical NRI and IDI

Computes Pencina et al. (2008, 2011) Net Reclassification Improvement (Continuous and Categorical NRI) and Integrated Discrimination Improvement (IDI).
Enforces clinical guardrails: checks probabilities lie in $[0, 1]$, and reports Event NRI and Non-event NRI separately per Kerr et al. (2014) guidelines.

```python
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats


def compute_reclassification_metrics(
    y_true: pd.Series | np.ndarray,
    p_old: pd.Series | np.ndarray,
    p_new: pd.Series | np.ndarray,
    thresholds: list[float] | None = None,
    n_bootstrap: int = 0,
    random_state: int = 42,
) -> dict[str, Any]:
    """Calculate Continuous NRI, Categorical NRI, and IDI with asymptotic and bootstrap CIs."""
    y_t = np.asarray(y_true, dtype=float)
    p_0 = np.asarray(p_old, dtype=float)
    p_1 = np.asarray(p_new, dtype=float)

    if len(y_t) != len(p_0) or len(y_t) != len(p_1):
        raise ValueError("y_true, p_old, and p_new must have identical lengths.")

    valid = ~(np.isnan(y_t) | np.isnan(p_0) | np.isnan(p_1))
    y_valid = y_t[valid]
    if len(y_valid) == 0:
        raise ValueError("No complete cases remaining after removing missing values.")
    if not np.all(np.isin(y_valid, [0.0, 1.0])):
        raise ValueError("y_true must contain strictly binary values (0 and 1).")
    if len(np.unique(y_valid)) < 2:
        raise ValueError("y_true must contain both event classes (0 and 1).")
    y_t = y_valid.astype(int)
    p_0 = p_0[valid]
    p_1 = p_1[valid]

    # Miscalibration Guard
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

    # 1. Continuous NRI
    up_cont = p_1 > p_0 + 1e-12
    down_cont = p_1 < p_0 - 1e-12

    mask_1 = y_t == 1
    mask_0 = y_t == 0

    n_up_e = int(np.sum(mask_1 & up_cont))
    n_down_e = int(np.sum(mask_1 & down_cont))
    p_up_e = n_up_e / n_events if n_events > 0 else 0.0
    p_down_e = n_down_e / n_events if n_events > 0 else 0.0
    nri_e_cont = p_up_e - p_down_e
    var_e_cont = (
        (p_up_e + p_down_e - nri_e_cont**2) / n_events if n_events > 0 else 0.0
    )
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
        (p_down_ne + p_up_ne - nri_ne_cont**2) / n_nonevents
        if n_nonevents > 0
        else 0.0
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
    z_total_cont = (
        float(nri_total_cont / se_total_cont) if se_total_cont > 0.0 else 0.0
    )
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

        cat_labels = []
        for i in range(k_bins):
            if i == 0:
                cat_labels.append(f"< {sorted_thresh[0]:.2g}")
            elif i == k_bins - 1:
                cat_labels.append(f"≥ {sorted_thresh[-1]:.2g}")
            else:
                cat_labels.append(f"{sorted_thresh[i-1]:.2g} to <{sorted_thresh[i]:.2g}")

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
            (p_up_e_cat + p_down_e_cat - nri_e_cat**2) / n_events if n_events > 0 else 0.0
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
        z_total_cat = (
            float(nri_total_cat / se_total_cat) if se_total_cat > 0.0 else 0.0
        )
        p_val_total_cat = (
            float(2.0 * (1.0 - stats.norm.cdf(abs(z_total_cat))))
            if se_total_cat > 0.0
            else (1.0 if nri_total_cat == 0.0 else 0.0)
        )

        pct_up = float(np.sum(m_combined[np.triu_indices(k_bins, k=1)]) / n_total * 100.0)
        pct_down = float(np.sum(m_combined[np.tril_indices(k_bins, k=-1)]) / n_total * 100.0)
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

    # 3. IDI
    mean_p1_events = float(np.mean(p_1[mask_1]))
    mean_p0_events = float(np.mean(p_0[mask_1]))
    mean_p1_nonevents = float(np.mean(p_1[mask_0]))
    mean_p0_nonevents = float(np.mean(p_0[mask_0]))

    idi_val = (mean_p1_events - mean_p0_events) - (mean_p1_nonevents - mean_p0_nonevents)
    slope_old = mean_p0_events - mean_p0_nonevents
    slope_new = mean_p1_events - mean_p1_nonevents
    rel_idi = float(idi_val / slope_old) if slope_old > 1e-12 else None

    delta_p = p_1 - p_0
    var_delta_1 = float(np.var(delta_p[mask_1], ddof=1)) if n_events > 1 else 0.0
    var_delta_0 = float(np.var(delta_p[mask_0], ddof=1)) if n_nonevents > 1 else 0.0

    se_idi = float(np.sqrt(max(0.0, var_delta_1 / n_events + var_delta_0 / n_nonevents)))
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
    }
```

---

---

## 17. Time-Dependent ROC: Cumulative/Dynamic IPCW Estimator (AUROC(t))

Cumulative/Dynamic Inverse Probability of Censoring Weighting (IPCW) Time-Dependent ROC estimator (Uno et al., 2007; Heagerty et al., 2000).
Computes longitudinal discrimination trajectories, optimal Youden cutoffs, and bootstrap confidence intervals across clinical follow-up horizons.

```python
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
from lifelines import KaplanMeierFitter


def _lookup_censoring_survival_left_limit(
    kmf_censor: Any,
    times: np.ndarray | pd.Series | list[float] | float,
    clip_min: float = 1e-3,
) -> np.ndarray | float:
    """Evaluate left-limit censoring survival probability G_hat(t-) = P(C >= t)."""
    timeline = np.asarray(kmf_censor.survival_function_.index, dtype=float)
    vals = kmf_censor.survival_function_.iloc[:, 0].to_numpy(dtype=float)
    is_scalar = np.isscalar(times)
    t_arr = np.atleast_1d(np.asarray(times, dtype=float))
    idx = np.clip(np.searchsorted(timeline, t_arr, side="left") - 1, 0, len(vals) - 1)
    res = np.clip(vals[idx], clip_min, 1.0)
    return float(res[0]) if is_scalar else res


def _calc_ipcw_roc_point(
    durations: np.ndarray,
    events: np.ndarray,
    scores: np.ndarray,
    horizon: float,
    g_durations: np.ndarray,
    g_horizon: float,
    competing_event_code: int | None = None,
    return_curve: bool = True,
) -> dict[str, Any]:
    """Single horizon weighted IPCW AUROC calculation."""
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

    # O(N log N) fast concordance
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

    sum_w_cases = float(np.sum(w_cases))
    sum_w_ctrl = float(np.sum(w_ctrl))
    unique_th = np.sort(np.unique(np.concatenate([sc_cases, sc_ctrl])))[::-1]

    tpr_list = [0.0]
    fpr_list = [0.0]
    th_list = [float("inf")]
    for th in unique_th:
        tpr_list.append(float(np.sum(w_cases[sc_cases >= th]) / sum_w_cases) if sum_w_cases > 0 else 0.0)
        fpr_list.append(float(np.sum(w_ctrl[sc_ctrl >= th]) / sum_w_ctrl) if sum_w_ctrl > 0 else 0.0)
        th_list.append(float(th))
    tpr_list.append(1.0)
    fpr_list.append(1.0)
    th_list.append(float("-inf"))

    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)
    best_idx = int(np.argmax(tpr_arr - fpr_arr))
    opt_th = th_list[best_idx]
    if np.isneginf(opt_th):
        opt_th = float(unique_th[-1]) if len(unique_th) > 0 else 0.0
    elif np.isposinf(opt_th):
        opt_th = float(unique_th[0]) if len(unique_th) > 0 else 0.0

    return {
        "horizon": t,
        "n_cases": n_cases,
        "n_controls": n_controls,
        "n_censored": n_censored,
        "auroc": auroc,
        "optimal_threshold": float(opt_th),
        "sensitivity": float(tpr_arr[best_idx]),
        "specificity": float(1.0 - fpr_arr[best_idx]),
        "fpr": fpr_list,
        "tpr": tpr_list,
        "thresholds": th_list,
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
    """Cumulative/Dynamic IPCW Time-Dependent ROC suite."""
    dur_arr = pd.Series(durations).to_numpy(dtype=float)
    ev_arr = pd.Series(events).to_numpy(dtype=float)

    allowed_event_codes = (
        {0.0, 1.0}
        if competing_event_code is None
        else {0.0, 1.0, float(competing_event_code)}
    )

    def _validate_and_cast_events(ev_sub: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(ev_sub)):
            raise ValueError("Event values must be finite.")
        if not np.all(np.floor(ev_sub) == ev_sub):
            raise ValueError("Event values must be integer-valued codes.")
        if not np.all(np.isin(ev_sub, list(allowed_event_codes))):
            raise ValueError(f"Event values must be in {allowed_event_codes}.")
        return ev_sub.astype(int)

    is_comparison = False
    if scores_old is not None and scores_new is not None:
        is_comparison = True
        s_old = pd.Series(scores_old).to_numpy(dtype=float)
        s_new = pd.Series(scores_new).to_numpy(dtype=float)
        valid = ~(np.isnan(dur_arr) | np.isnan(ev_arr) | np.isnan(s_old) | np.isnan(s_new))
        dur_arr, ev_arr, s_old, s_new = (
            dur_arr[valid],
            _validate_and_cast_events(ev_arr[valid]),
            s_old[valid],
            s_new[valid],
        )
        s_arr = s_new
    elif isinstance(scores, dict):
        is_comparison = True
        s_old = pd.Series(scores.get("old", scores.get("baseline"))).to_numpy(dtype=float)
        s_new = pd.Series(scores.get("new", scores.get("updated"))).to_numpy(dtype=float)
        valid = ~(np.isnan(dur_arr) | np.isnan(ev_arr) | np.isnan(s_old) | np.isnan(s_new))
        dur_arr, ev_arr, s_old, s_new = (
            dur_arr[valid],
            _validate_and_cast_events(ev_arr[valid]),
            s_old[valid],
            s_new[valid],
        )
        s_arr = s_new
    else:
        if scores is None:
            raise ValueError("Must provide scores or both scores_old and scores_new.")
        s_arr = pd.Series(scores).to_numpy(dtype=float)
        valid = ~(np.isnan(dur_arr) | np.isnan(ev_arr) | np.isnan(s_arr))
        dur_arr, ev_arr, s_arr = (
            dur_arr[valid],
            _validate_and_cast_events(ev_arr[valid]),
            s_arr[valid],
        )
        s_old, s_new = None, None

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
    max_t = float(np.max(dur_arr)) if n_obs > 0 else 0.0

    if time_horizons is None or len(time_horizons) == 0:
        ev_times = dur_arr[ev_arr == 1]
        time_horizons = [float(np.percentile(ev_times, p)) for p in [25, 50, 75]] if len(ev_times) >= 3 else [max_t * 0.5]
    sorted_horizons = sorted([float(h) for h in time_horizons])
    if any(h <= 0 for h in sorted_horizons):
        raise ValueError("Time horizons must contain positive follow-up times (> 0).")

    kmc = KaplanMeierFitter()
    kmc.fit(dur_arr, event_observed=(ev_arr == 0).astype(int))
    g_dur = _lookup_censoring_survival_left_limit(kmc, dur_arr)

    results_by_horizon = {}
    comparison_by_horizon = {} if is_comparison else None
    non_sparse_horizons = []

    for h in sorted_horizons:
        g_h = max(float(kmc.survival_function_at_times(h).iloc[0]), 1e-3)
        res_h = _calc_ipcw_roc_point(dur_arr, ev_arr, s_arr, h, g_dur, g_h, competing_event_code, return_curve=True)

        is_sparse = (h > max_t) or (res_h["n_cases"] < 5) or (res_h["n_controls"] < 5)
        res_h["warning"] = "Sparse risk set or beyond follow-up" if is_sparse else None

        if is_sparse:
            res_h["ci_lower"] = float("nan")
            res_h["ci_upper"] = float("nan")
            res_h["se"] = float("nan")
            res_h["ci_method"] = None
        else:
            non_sparse_horizons.append(h)
            if n_bootstrap == 0:
                auc_val = res_h["auroc"]
                if np.isnan(auc_val) or res_h["n_cases"] < 1 or res_h["n_controls"] < 1:
                    res_h["se"] = float("nan")
                    res_h["ci_lower"] = float("nan")
                    res_h["ci_upper"] = float("nan")
                    res_h["ci_method"] = None
                else:
                    q1 = auc_val / (2.0 - auc_val)
                    q2 = 2.0 * auc_val**2 / (1.0 + auc_val)
                    var_h = (
                        auc_val * (1.0 - auc_val)
                        + (res_h["n_cases"] - 1) * (q1 - auc_val**2)
                        + (res_h["n_controls"] - 1) * (q2 - auc_val**2)
                    ) / (res_h["n_cases"] * res_h["n_controls"])
                    se_h = float(np.sqrt(max(0.0, var_h)))
                    res_h["se"] = se_h
                    res_h["ci_lower"] = max(0.0, float(auc_val - 1.959963984540054 * se_h))
                    res_h["ci_upper"] = min(1.0, float(auc_val + 1.959963984540054 * se_h))
                    res_h["ci_method"] = "hanley_mcneil"
                    if res_h.get("n_censored", 0) > 0:
                        cens_msg = (
                            f"Horizon t={h} has {res_h['n_censored']} censored observations before t. "
                            "Hanley-McNeil confidence interval is an unweighted approximation that does not account for "
                            "IPCW censoring weight uncertainty; n_bootstrap >= 200 is recommended when censoring is present."
                        )
                        res_h["warning"] = (
                            f"{res_h['warning']}; {cens_msg}"
                            if res_h.get("warning")
                            else cens_msg
                        )

        results_by_horizon[h] = res_h

        if is_comparison and s_old is not None and s_new is not None:
            res_old_h = _calc_ipcw_roc_point(dur_arr, ev_arr, s_old, h, g_dur, g_h, competing_event_code, return_curve=False)
            delta = (
                res_h["auroc"] - res_old_h["auroc"]
                if not (np.isnan(res_h["auroc"]) or np.isnan(res_old_h["auroc"]))
                else float("nan")
            )
            comp_entry = {
                "horizon": h,
                "auroc_old": res_old_h["auroc"],
                "auroc_new": res_h["auroc"],
                "delta_auroc": delta,
                "n_cases": res_h["n_cases"],
                "n_controls": res_h["n_controls"],
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "se": float("nan"),
                "p_value": float("nan"),
                "ci_method": None,
            }
            comparison_by_horizon[h] = comp_entry

    if n_bootstrap > 0 and len(non_sparse_horizons) > 0:
        rng = np.random.default_rng(random_state)
        boot_indices = [rng.choice(n_obs, size=n_obs, replace=True) for _ in range(n_bootstrap)]
        boot_aucs_by_h = {h: [] for h in non_sparse_horizons}
        boot_deltas_by_h = {h: [] for h in non_sparse_horizons}

        for b_idx in boot_indices:
            d_b, e_b, s_b = dur_arr[b_idx], ev_arr[b_idx], s_arr[b_idx]
            km_b = KaplanMeierFitter().fit(d_b, event_observed=(e_b == 0).astype(int))
            g_dur_b = _lookup_censoring_survival_left_limit(km_b, d_b)

            for h in non_sparse_horizons:
                g_h_b = max(float(km_b.survival_function_at_times(h).iloc[0]), 1e-3)
                b_res = _calc_ipcw_roc_point(d_b, e_b, s_b, h, g_dur_b, g_h_b, competing_event_code, return_curve=False)
                if not np.isnan(b_res["auroc"]):
                    boot_aucs_by_h[h].append(b_res["auroc"])

                if is_comparison and s_old is not None and s_new is not None:
                    bo = _calc_ipcw_roc_point(d_b, e_b, s_old[b_idx], h, g_dur_b, g_h_b, competing_event_code, return_curve=False)
                    if not (np.isnan(bo["auroc"]) or np.isnan(b_res["auroc"])):
                        boot_deltas_by_h[h].append(b_res["auroc"] - bo["auroc"])

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
                    comparison_by_horizon[h]["ci_lower"] = float(np.percentile(b_deltas, 2.5))
                    comparison_by_horizon[h]["ci_upper"] = float(np.percentile(b_deltas, 97.5))
                    z_d = float(d_auc / se_d) if se_d > 0 else 0.0
                    comparison_by_horizon[h]["p_value"] = float(2.0 * (1.0 - stats.norm.cdf(abs(z_d))))
                    comparison_by_horizon[h]["ci_method"] = "bootstrap"

    trajectory = {
        "horizons": sorted_horizons,
        "auroc": [results_by_horizon[h]["auroc"] for h in sorted_horizons],
        "ci_lower": [results_by_horizon[h]["ci_lower"] for h in sorted_horizons],
        "ci_upper": [results_by_horizon[h]["ci_upper"] for h in sorted_horizons],
    }

    out = {
        "time_horizons": sorted_horizons,
        "results_by_horizon": results_by_horizon,
        "trajectory": trajectory,
        "ci_method": "bootstrap" if n_bootstrap > 0 else "hanley_mcneil",
    }
    if is_comparison:
        out["comparison"] = comparison_by_horizon

    top_warnings = [
        results_by_horizon[h]["warning"]
        for h in sorted_horizons
        if results_by_horizon[h].get("warning")
    ]
    if top_warnings:
        out["warnings"] = top_warnings
        if n_bootstrap == 0 and any(
            results_by_horizon[h].get("ci_method") == "hanley_mcneil"
            and results_by_horizon[h].get("n_censored", 0) > 0
            for h in sorted_horizons
        ):
            out["warning"] = (
                "Hanley-McNeil confidence interval is an unweighted approximation that does not account for "
                "IPCW censoring weight uncertainty; n_bootstrap >= 200 is recommended when censoring is present."
            )
        else:
            out["warning"] = top_warnings[0]

    return out
```

---

---

## 18. TRIPOD+AI Optimism-Corrected Bootstrap Internal Validation & Uniform Shrinkage

Computes Efron-Gong optimism-corrected performance metrics (AUROC, Brier score, Cox-Snell $R^2$, Nagelkerke $R^2$, calibration-in-the-large, calibration slope) across $B$ bootstrap resamples. Derives the uniform shrinkage factor $\hat{s}$ and re-estimates the intercept $\alpha_0$ to preserve sample prevalence on future predictions.

```python
import math
from typing import Any

import numpy as np
import pandas as pd
import scipy.optimize as opt
import statsmodels.api as sm
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score


def bootstrap_internal_validation(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    n_bootstrap: int = 200,
    random_state: int = 42,
    missing_strategy: str = "error",
) -> dict[str, Any]:
    """TRIPOD+AI Optimism-Corrected Bootstrap Internal Validation and Uniform Shrinkage."""
    norm_strat = str(missing_strategy).lower().strip()
    if norm_strat not in ("error", "complete_case"):
        raise ValueError(
            f"Unsupported missing_strategy '{missing_strategy}'. Must be 'error' or 'complete_case'."
        )

    cols = [outcome] + list(predictors)
    n_initial = len(df)
    missing_counts = {
        c: int(df[c].isna().sum()) for c in cols if df[c].isna().sum() > 0
    }
    sub_df = df[cols].dropna()
    n_excluded = n_initial - len(sub_df)

    if n_excluded > 0 and norm_strat == "error":
        counts_str = ", ".join(f"{c} ({cnt})" for c, cnt in missing_counts.items())
        raise ValueError(
            f"Missing values detected in input columns: {counts_str}. "
            "Please perform complete-case filtering or multiple imputation before running internal validation."
        )

    exclusion_summary = {
        "n_initial": n_initial,
        "n_excluded": n_excluded,
        "n_retained": len(sub_df),
        "missing_counts": missing_counts,
    }

    if len(sub_df) < 10:
        raise ValueError(f"Insufficient non-missing observations (N = {len(sub_df)}).")

    y_orig_raw = sub_df[outcome].to_numpy(dtype=float)
    if not np.all(np.isin(y_orig_raw, [0.0, 1.0])):
        raise ValueError(
            f"Binary outcome '{outcome}' must contain strictly binary values (0 and 1)."
        )
    if len(np.unique(y_orig_raw)) < 2:
        raise ValueError(
            f"Binary outcome '{outcome}' must contain both event classes (0 and 1)."
        )
    y_orig = y_orig_raw.astype(int)

    # Check for categorical / non-float predictor dtypes (object, string, category, bool)
    pred_df = sub_df[predictors].copy()
    has_categorical = any(
        col_dtype.name in ("object", "category", "string", "bool")
        or isinstance(col_dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(col_dtype)
        or pd.api.types.is_string_dtype(col_dtype)
        or pd.api.types.is_object_dtype(col_dtype)
        for col_dtype in [pred_df[c].dtype for c in predictors]
    )

    if has_categorical:
        X_expanded = pd.get_dummies(pred_df, drop_first=True, dtype=float)
        model_predictors = list(X_expanded.columns)
        X_orig = X_expanded.to_numpy(dtype=float)
    else:
        model_predictors = list(predictors)
        X_orig = pred_df.to_numpy(dtype=float)

    n_total = len(y_orig)
    n_events = int(np.sum(y_orig))
    n_non_events = n_total - n_events
    n_minority = min(n_events, n_non_events)
    n_predictors = len(model_predictors)
    epp = float(n_minority / n_predictors) if n_predictors > 0 else 0.0

    # 1. Fit initial model with separation/singular fallback
    X_orig_const = sm.add_constant(X_orig, has_constant="add")
    warnings = []
    try:
        orig_model = sm.Logit(y_orig, X_orig_const).fit(disp=False)
        converged = getattr(orig_model, "mle_retvals", {}).get("converged", True)
        if not converged or np.any(np.abs(orig_model.params) > 50.0):
            raise RuntimeError("Model separation or non-convergence detected.")
        orig_params = orig_model.params
    except Exception as e:
        warnings.append(
            f"Standard MLE logit failed or encountered quasi-complete separation ({e}); falling back to regularized logit."
        )
        reg_model = sm.Logit(y_orig, X_orig_const).fit_regularized(
            method="l1", alpha=1e-4, disp=False
        )
        orig_params = reg_model.params

    lp_orig = np.dot(X_orig_const, orig_params)
    p_orig = expit(lp_orig)

    def _calc_metrics(
        y_true: np.ndarray, p_pred: np.ndarray, lp: np.ndarray
    ) -> dict[str, float]:
        n = len(y_true)
        p_c = np.clip(p_pred, 1e-15, 1.0 - 1e-15)
        prev = np.clip(float(np.mean(y_true)), 1e-15, 1.0 - 1e-15)
        ll_null = float(
            np.sum(y_true * np.log(prev) + (1.0 - y_true) * np.log(1.0 - prev))
        )
        ll_model = float(
            np.sum(y_true * np.log(p_c) + (1.0 - y_true) * np.log(1.0 - p_c))
        )
        cs_r2 = max(0.0, 1.0 - math.exp(-2.0 * (ll_model - ll_null) / n))
        r2_max = max(1e-12, 1.0 - math.exp(2.0 * ll_null / n))
        nag_r2 = min(1.0, max(0.0, cs_r2 / r2_max))
        auc = (
            float(roc_auc_score(y_true, p_pred))
            if len(np.unique(y_true)) == 2
            else 0.5
        )
        brier = float(np.mean((y_true - p_pred) ** 2))

        # Intercept offset
        sum_y = float(np.sum(y_true))
        if sum_y <= 0:
            inter = -50.0
        elif sum_y >= n:
            inter = 50.0
        else:
            try:
                inter = float(
                    opt.brentq(
                        lambda a: float(np.sum(expit(a + lp)) - sum_y),
                        -50.0,
                        50.0,
                    )
                )
            except Exception:
                inter = 0.0

        # Slope
        if np.allclose(lp, lp[0]):
            slope = 1.0
        else:
            try:
                mod_slope = sm.Logit(y_true, sm.add_constant(lp)).fit(disp=False)
                slope = float(mod_slope.params[1])
            except Exception:
                slope = 1.0

        return {
            "auroc": auc,
            "brier_score": brier,
            "cox_snell_r2": cs_r2,
            "nagelkerke_r2": nag_r2,
            "calibration_intercept": inter,
            "calibration_slope": slope,
        }

    apparent = _calc_metrics(y_orig, p_orig, lp_orig)

    # 2. Bootstrap loop
    rng = np.random.default_rng(random_state)
    boot_records = {k: [] for k in apparent}
    test_records = {k: [] for k in apparent}
    optimism_records = {k: [] for k in apparent}
    test_slopes = []
    n_successful = 0

    for _ in range(n_bootstrap):
        boot_idx = rng.choice(n_total, size=n_total, replace=True)
        y_b = y_orig[boot_idx]
        X_b = X_orig[boot_idx]
        if len(np.unique(y_b)) < 2:
            continue
        # Skip samples with zero variance in any dummy/predictor column
        if np.any(np.var(X_b, axis=0) <= 1e-12):
            continue
        X_b_const = sm.add_constant(X_b, has_constant="add")
        try:
            boot_mod = sm.Logit(y_b, X_b_const).fit(disp=False)
            b_params = boot_mod.params
            if np.isnan(b_params).any() or np.isinf(b_params).any():
                continue
        except Exception:
            continue

        lp_b = np.dot(X_b_const, b_params)
        p_b = expit(lp_b)
        m_b = _calc_metrics(y_b, p_b, lp_b)

        lp_test = np.dot(X_orig_const, b_params)
        p_test = expit(lp_test)
        m_test = _calc_metrics(y_orig, p_test, lp_test)

        n_successful += 1
        for k in apparent:
            boot_records[k].append(m_b[k])
            test_records[k].append(m_test[k])
            optimism_records[k].append(m_b[k] - m_test[k])
        test_slopes.append(m_test["calibration_slope"])

    # Summary
    metrics_summary = {}
    for k in apparent:
        app_val = apparent[k]
        if n_successful == 0:
            opt_mean = np.nan
            b_mean = np.nan
            t_mean = np.nan
            corrected = np.nan
        else:
            opt_mean = float(np.mean(optimism_records[k])) if optimism_records[k] else 0.0
            b_mean = float(np.mean(boot_records[k])) if boot_records[k] else app_val
            t_mean = float(np.mean(test_records[k])) if test_records[k] else app_val
            corrected = app_val - opt_mean
            if k in ("auroc", "nagelkerke_r2", "cox_snell_r2"):
                corrected = max(0.0, min(1.0, corrected))
            elif k == "brier_score":
                corrected = max(0.0, corrected)
        metrics_summary[k] = {
            "apparent": float(app_val),
            "boot_mean": float(b_mean) if not np.isnan(b_mean) else np.nan,
            "test_mean": float(t_mean) if not np.isnan(t_mean) else np.nan,
            "optimism": float(opt_mean) if not np.isnan(opt_mean) else np.nan,
            "optimism_corrected": float(corrected) if not np.isnan(corrected) else np.nan,
        }

    orig_intercept = float(orig_params[0])
    orig_betas = [float(b) for b in orig_params[1:]]

    if n_successful == 0:
        shrinkage_factor = np.nan
        shrunk_intercept = np.nan
        shrunk_betas = [np.nan for _ in orig_betas]
    else:
        raw_s = float(np.mean(test_slopes)) if test_slopes else 1.0
        shrinkage_factor = float(np.clip(raw_s, 0.01, 1.0))
        shrunk_betas = [shrinkage_factor * b for b in orig_betas]
        lp_shrunk_no_inter = np.dot(X_orig, shrunk_betas)

        # Re-estimate intercept
        target_events = float(np.sum(y_orig))
        try:
            shrunk_intercept = float(
                opt.brentq(
                    lambda a: float(np.sum(expit(a + lp_shrunk_no_inter)) - target_events),
                    -50.0,
                    50.0,
                )
            )
        except Exception:
            prev_orig = np.clip(target_events / n_total, 1e-6, 1.0 - 1e-6)
            shrunk_intercept = float(logit(prev_orig) - np.mean(lp_shrunk_no_inter))

    orig_coef_dict = {"Intercept": orig_intercept}
    shrunk_coef_dict = {"Intercept": shrunk_intercept}
    for name, ob, sb in zip(model_predictors, orig_betas, shrunk_betas):
        orig_coef_dict[name] = ob
        shrunk_coef_dict[name] = sb

    if epp < 10.0:
        warnings.append(
            f"Low events per predictor (EPP = {epp:.2f} < 10), indicating elevated risk of model overfitting."
        )
    if not np.isnan(shrinkage_factor) and shrinkage_factor < 0.80:
        warnings.append(
            f"Uniform shrinkage factor s_hat = {shrinkage_factor:.3f} (< 0.80) indicates substantial overfitting."
        )
    if n_successful == 0:
        warnings.append(
            "Zero bootstrap resamples converged successfully; optimism and shrinkage could not be estimated."
        )
    elif n_successful < n_bootstrap * 0.5:
        warnings.append(
            f"Fewer than 50% of bootstrap resamples converged ({n_successful}/{n_bootstrap})."
        )

    return {
        "n_total": n_total,
        "n_events": n_events,
        "n_predictors": n_predictors,
        "events_per_predictor": epp,
        "n_bootstrap": n_bootstrap,
        "n_successful_replicates": n_successful,
        "missing_data_summary": exclusion_summary,
        "apparent_metrics": apparent,
        "metrics": metrics_summary,
        "shrinkage_factor": shrinkage_factor,
        "shrunk_coefficients": {
            "original": orig_coef_dict,
            "shrunk": shrunk_coef_dict,
        },
        "warnings": warnings,
    }
```

---

---

## 19. Prediction Model Recalibration and Updating (Logistic & Intercept-Only)

Adjusts existing prediction models to a new target population or cohort. Supports Intercept-Only updating (calibration-in-the-large, shifting mean predicted risk to the observed outcome incidence while fixing slope=1) and Logistic Recalibration (updating both calibration intercept $\alpha$ and calibration slope $\beta$).

```python
from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy.optimize as opt
import statsmodels.api as sm
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score


def recalibrate_prediction_model(
    y_true: pd.Series | np.ndarray,
    p_pred: pd.Series | np.ndarray,
    method: Literal[
        "intercept_only", "calibration_in_the_large", "logistic", "both"
    ] = "logistic",
    clip_eps: float = 1e-6,
) -> dict[str, Any]:
    if (
        not isinstance(clip_eps, (int, float))
        or not np.isfinite(clip_eps)
        or not (0.0 < float(clip_eps) < 0.5)
    ):
        raise ValueError(
            f"clip_eps ({clip_eps}) must be a finite number strictly between 0 and 0.5."
        )
    clip_eps = float(clip_eps)

    norm_m = (
        "intercept_only"
        if method in ("intercept_only", "calibration_in_the_large")
        else method
    )
    if norm_m not in ("intercept_only", "logistic", "both"):
        raise ValueError(f"Unsupported recalibration method: {method}")

    y_arr = np.asarray(y_true, dtype=float).ravel()
    p_arr = np.asarray(p_pred, dtype=float).ravel()

    if len(y_arr) != len(p_arr):
        raise ValueError(
            f"Length mismatch between y_true (N={len(y_arr)}) and p_pred (N={len(p_arr)})."
        )

    n_invalid_y = int(np.sum(~np.isfinite(y_arr)))
    n_invalid_p = int(np.sum(~np.isfinite(p_arr)))
    n_excluded = n_invalid_y + n_invalid_p
    if n_excluded > 0:
        raise ValueError(
            f"Non-finite values detected in input arrays: y_true has {n_invalid_y} non-finite value(s), "
            f"p_pred has {n_invalid_p} non-finite value(s) ({n_excluded} non-finite values total). "
            "Please perform complete-case filtering or multiple imputation before running recalibration."
        )

    n = len(y_arr)
    if n == 0:
        raise ValueError("Cannot perform recalibration on empty arrays (N = 0).")
    if n < 10:
        raise ValueError(f"Insufficient observations for recalibration (N = {n}).")

    unique_y = np.unique(y_arr)
    if not np.array_equal(unique_y, [0.0, 1.0]):
        raise ValueError(
            f"y_true must contain strictly binary classes {{0, 1}} with both classes present, got {unique_y}."
        )

    if np.any(p_arr < 0.0) or np.any(p_arr > 1.0):
        raise ValueError(
            "Predicted probabilities p_pred must lie strictly within [0, 1]."
        )

    p_clipped = np.clip(p_arr, clip_eps, 1.0 - clip_eps)
    lp_orig = logit(p_clipped)

    def _calc_suite(y: np.ndarray, p: np.ndarray, lp: np.ndarray) -> dict[str, float]:
        sum_y = float(np.sum(y))
        try:
            inter = float(
                opt.brentq(
                    lambda a: float(np.sum(expit(a + lp)) - sum_y), -50.0, 50.0
                )
            )
        except Exception:
            inter = 0.0

        if np.var(lp) <= 1e-12 or np.allclose(lp, lp[0]):
            slope = float("nan")
        else:
            try:
                mod_s = sm.Logit(y, sm.add_constant(lp)).fit(disp=False)
                if not getattr(mod_s, "mle_retvals", {}).get("converged", True):
                    slope = float("nan")
                else:
                    slope = float(mod_s.params[1])
                    if not np.isfinite(slope):
                        slope = float("nan")
            except Exception:
                slope = float("nan")

        brier = float(np.mean((y - p) ** 2))
        auc = (
            float(roc_auc_score(y, p))
            if len(np.unique(y)) == 2
            else 0.5
        )

        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess

            smoothed = lowess(y, p, frac=0.67, return_sorted=False)
            cal_err = np.abs(smoothed - p)
            emax = float(np.max(cal_err))
            ici = float(np.mean(cal_err))
        except Exception:
            emax = float("nan")
            ici = float("nan")

        return {
            "calibration_intercept": inter,
            "calibration_slope": slope,
            "brier_score": brier,
            "emax": emax,
            "ici": ici,
            "auroc": auc,
        }

    pre_metrics = _calc_suite(y_arr, p_clipped, lp_orig)

    sum_y = float(np.sum(y_arr))
    try:
        alpha_inter = float(
            opt.brentq(
                lambda a: float(np.sum(expit(a + lp_orig)) - sum_y), -50.0, 50.0
            )
        )
    except Exception:
        alpha_inter = 0.0
    p_inter = expit(alpha_inter + lp_orig)

    X = sm.add_constant(lp_orig)
    logistic_fallback = False
    try:
        fit_mod = sm.Logit(y_arr, X).fit(disp=False)
        converged = getattr(fit_mod, "mle_retvals", {}).get("converged", True)
        if not converged:
            raise RuntimeError("Logistic recalibration failed to converge.")
        alpha_log = float(fit_mod.params[0])
        beta_log = float(fit_mod.params[1])
        alpha_ci = [float(c) for c in fit_mod.conf_int()[0]]
        beta_ci = [float(c) for c in fit_mod.conf_int()[1]]
        p_log = expit(alpha_log + beta_log * lp_orig)
    except Exception:
        logistic_fallback = True
        alpha_log = alpha_inter
        beta_log = 1.0
        alpha_ci = [alpha_inter, alpha_inter]
        beta_ci = [1.0, 1.0]
        p_log = p_inter

    if norm_m == "intercept_only":
        lp_recal = alpha_inter + lp_orig
        post_metrics = _calc_suite(y_arr, p_inter, lp_recal)
        return {
            "method": "intercept_only",
            "n": n,
            "pre": pre_metrics,
            "post": post_metrics,
            "recalibration_parameters": {
                "method": "intercept_only",
                "alpha": alpha_inter,
                "beta": 1.0,
            },
            "p_recalibrated": p_inter,
            "p_recalibrated_intercept_only": p_inter,
            "p_recalibrated_logistic": p_log,
        }
    elif norm_m == "logistic":
        reported_method = "intercept_only_fallback" if logistic_fallback else "logistic"
        lp_recal = alpha_log + beta_log * lp_orig
        post_metrics = _calc_suite(y_arr, p_log, lp_recal)
        return {
            "method": reported_method,
            "n": n,
            "pre": pre_metrics,
            "post": post_metrics,
            "recalibration_parameters": {
                "method": reported_method,
                "alpha": alpha_log,
                "alpha_ci": alpha_ci,
                "beta": beta_log,
                "beta_ci": beta_ci,
            },
            "p_recalibrated": p_log,
            "p_recalibrated_intercept_only": p_inter,
            "p_recalibrated_logistic": p_log,
        }
    else:  # both
        post_inter = _calc_suite(y_arr, p_inter, alpha_inter + lp_orig)
        post_log = _calc_suite(y_arr, p_log, alpha_log + beta_log * lp_orig)

        return {
            "method": "both",
            "n": n,
            "pre": pre_metrics,
            "post_intercept_only": post_inter,
            "post_logistic": post_log,
            "post": post_log,
            "logistic_fallback": logistic_fallback,
            "parameters_intercept_only": {"alpha": alpha_inter, "beta": 1.0},
            "parameters_logistic": {
                "alpha": alpha_log,
                "beta": beta_log,
                "fallback": logistic_fallback,
            },
            "p_recalibrated": p_log,
            "p_recalibrated_intercept_only": p_inter,
            "p_recalibrated_logistic": p_log,
        }
```

---

---

## 20. Sample Size Determination for Clinical Prediction Models (Riley et al. 2020) & Diagnostic AUROC (Obuchowski 1998)

Calculates the minimum required sample size for multivariable prediction model development per the 4 criteria of Riley et al. (BMJ 2020;368:m441), including numerical derivation of $R^2_{CS}$ from C-statistic or Nagelkerke $R^2$. Also determines required case and control sample sizes for diagnostic ROC test evaluation against a null benchmark per Obuchowski (1998) and Hanley & McNeil (1982).

```python
import math
from typing import Any

import numpy as np
import scipy.optimize as opt
import scipy.stats as stats
from scipy.special import expit, log_expit


def _approximate_cox_snell_r2_from_c_stat(
    c_stat: float,
    prevalence: float,
    deg: int = 64,
) -> float:
    """Exact deterministic Gauss-Hermite numerical integration of Riley et al. (2021)."""
    c = float(c_stat)
    phi = float(prevalence)
    if not (0.5 < c < 1.0):
        raise ValueError(f"c_statistic ({c}) must be strictly between 0.5 and 1.0.")
    if not (0.0 < phi < 1.0):
        raise ValueError(f"prevalence ({phi}) must be strictly between 0 and 1.")

    mu1 = math.sqrt(2.0) * float(stats.norm.ppf(c))
    nodes, weights = np.polynomial.hermite.hermgauss(deg)
    z = np.sqrt(2.0) * nodes
    inv_sqrt_pi = 1.0 / math.sqrt(math.pi)

    def f_alpha(alpha0: float) -> float:
        e0 = inv_sqrt_pi * float(np.sum(weights * expit(alpha0 + mu1 * z)))
        e1 = inv_sqrt_pi * float(np.sum(weights * expit(alpha0 + mu1 * (z + mu1))))
        return (1.0 - phi) * e0 + phi * e1 - phi

    alpha0 = float(opt.brentq(f_alpha, -50.0, 50.0))

    u0 = alpha0 + mu1 * z
    u1 = alpha0 + mu1 * (z + mu1)
    ll_0 = inv_sqrt_pi * float(np.sum(weights * log_expit(-u0)))
    ll_1 = inv_sqrt_pi * float(np.sum(weights * log_expit(u1)))
    ll_model = (1.0 - phi) * ll_0 + phi * ll_1

    ll_null = phi * math.log(phi) + (1.0 - phi) * math.log(1.0 - phi)
    r2_cs = -math.expm1(-2.0 * (ll_model - ll_null))
    r2_max = -math.expm1(2.0 * ll_null)

    return float(min(r2_max, max(0.0, r2_cs)))


def calculate_riley_sample_size(
    n_parameters: int,
    prevalence: float,
    c_statistic: float | None = None,
    cox_snell_r2: float | None = None,
    nagelkerke_r2: float | None = None,
    shrinkage_target: float = 0.90,
    intercept_margin: float = 0.05,
    optimism_margin: float = 0.05,
    mape_target: float | None = None,
) -> dict[str, Any]:
    """Riley et al. (2020) sample size for developing clinical prediction models."""
    if n_parameters < 1:
        raise ValueError(f"n_parameters must be >= 1, got {n_parameters}")
    if not (0.0 < prevalence < 1.0):
        raise ValueError(f"prevalence must be strictly in (0, 1), got {prevalence}")
    if not (0.5 < shrinkage_target < 1.0):
        raise ValueError(
            f"shrinkage_target must be strictly in (0.5, 1.0), got {shrinkage_target}"
        )
    if not (math.isfinite(intercept_margin) and intercept_margin > 0.0):
        raise ValueError(
            f"intercept_margin must be finite and > 0, got {intercept_margin}"
        )
    if not (math.isfinite(optimism_margin) and optimism_margin > 0.0):
        raise ValueError(
            f"optimism_margin must be finite and > 0, got {optimism_margin}"
        )
    if mape_target is not None and not (math.isfinite(mape_target) and mape_target > 0.0):
        raise ValueError(f"mape_target must be finite and > 0, got {mape_target}")

    phi = float(prevalence)
    p_params = int(n_parameters)
    s_target = float(shrinkage_target)

    ll_null = phi * math.log(phi) + (1.0 - phi) * math.log(1.0 - phi)
    r2_max = -math.expm1(2.0 * ll_null)

    if cox_snell_r2 is not None:
        cs_r2 = float(cox_snell_r2)
        if not (0.0 < cs_r2 < r2_max):
            raise ValueError(
                f"cox_snell_r2 ({cs_r2}) must be strictly between 0 and R2_max ({r2_max:.4f})."
            )
    elif nagelkerke_r2 is not None:
        nag_r2 = float(nagelkerke_r2)
        if not (0.0 < nag_r2 < 1.0):
            raise ValueError(
                f"nagelkerke_r2 ({nag_r2}) must be strictly between 0 and 1."
            )
        cs_r2 = nag_r2 * r2_max
    elif c_statistic is not None:
        c_stat = float(c_statistic)
        cs_r2 = _approximate_cox_snell_r2_from_c_stat(c_stat, phi)
    else:
        raise ValueError("Must provide c_statistic, cox_snell_r2, or nagelkerke_r2.")

    # Criterion 1: Overall risk/intercept
    n1 = int(math.ceil(((1.96 / intercept_margin) ** 2) * phi * (1.0 - phi)))

    # Criterion 2: MAPE (van Smeden 2019)
    target_m = (
        mape_target if mape_target is not None else (0.05 if phi <= 0.5 else None)
    )
    if target_m is not None:
        min_phi = min(phi, 1.0 - phi)
        exp_val = (
            -0.508
            + 0.259 * math.log(min_phi)
            + 0.504 * math.log(p_params)
            - math.log(target_m)
        ) / 0.544
        n2 = int(math.ceil(math.exp(exp_val)))
    else:
        n2 = None

    # Criterion 3: Uniform shrinkage
    arg_s = 1.0 - cs_r2 / s_target
    if arg_s <= 0.0:
        raise ValueError(
            f"Target shrinkage factor ({s_target:.2f}) must be strictly greater than anticipated Cox-Snell R2 ({cs_r2:.4f})."
        )
    denom_s = (s_target - 1.0) * math.log(arg_s)
    n3 = 1000000 if denom_s <= 0 else int(math.ceil(p_params / denom_s))

    # Criterion 4: Small optimism in R2
    s_req = cs_r2 / (cs_r2 + optimism_margin * r2_max)
    arg_opt = 1.0 - cs_r2 / s_req
    if s_req >= 1.0 or arg_opt <= 0.0:
        n4 = n3
    else:
        denom_opt = (s_req - 1.0) * math.log(arg_opt)
        n4 = n3 if denom_opt <= 0 else int(math.ceil(p_params / denom_opt))

    crit_dict: dict[str, Any] = {
        "Criterion 1 (Intercept)": n1,
        "Criterion 3 (Shrinkage)": n3,
        "Criterion 4 (Optimism)": n4,
    }
    if n2 is not None:
        crit_dict["Criterion 2 (MAPE)"] = n2
    else:
        crit_dict["Criterion 2 (MAPE)"] = None

    valid_crits = {k: v for k, v in crit_dict.items() if v is not None}
    n_total = max(valid_crits.values())
    binding = max(valid_crits, key=valid_crits.get)  # type: ignore[arg-type]
    n_events = int(math.ceil(n_total * phi))
    epp = float(n_events / p_params)

    return {
        "n_total": n_total,
        "n_events": n_events,
        "n_parameters": p_params,
        "prevalence": phi,
        "events_per_parameter": epp,
        "cox_snell_r2": cs_r2,
        "nagelkerke_r2": float(cs_r2 / r2_max),
        "r2_max": float(r2_max),
        "shrinkage_target": s_target,
        "binding_criterion": binding,
        "criteria": {
            "criterion1_intercept": {"n": n1, "margin": intercept_margin},
            "criterion2_mape": {
                "n": n2,
                "target_mape": target_m,
                "status": "evaluated"
                if n2 is not None
                else "N/A (default MAPE target 0.05 is only validated for prevalence <= 0.50; pass mape_target explicitly to evaluate)",
            },
            "criterion3_shrinkage": {"n": n3, "target_shrinkage": s_target},
            "criterion4_optimism": {"n": n4, "optimism_margin": optimism_margin},
        },
    }


def calculate_diagnostic_auroc_sample_size(
    auroc_expected: float,
    auroc_null: float = 0.50,
    prevalence: float = 0.50,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> dict[str, Any]:
    """Obuchowski (1998) / Hanley-McNeil (1982) diagnostic AUROC sample size."""
    theta1 = float(auroc_expected)
    theta0 = float(auroc_null)
    phi = float(prevalence)

    if not (0.0 <= theta0 < theta1 <= 1.0):
        raise ValueError(
            f"auroc_null ({theta0}) must be strictly less than auroc_expected ({theta1}) within [0, 1]."
        )
    if not (0.0 < phi < 1.0):
        raise ValueError(f"prevalence ({phi}) must be strictly in (0, 1).")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha ({alpha}) must be strictly in (0, 1).")
    if not (0.0 < power < 1.0):
        raise ValueError(f"power ({power}) must be strictly in (0, 1).")

    z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0 if two_sided else 1.0 - alpha))
    z_beta = float(stats.norm.ppf(power))

    R = (1.0 - phi) / phi

    def _v(t: float) -> float:
        q1 = t / (2.0 - t)
        q2 = 2.0 * (t**2) / (1.0 + t)
        return float((q2 - t**2) + (q1 - t**2) / R)

    var_theta1 = _v(theta1)
    var_theta0 = _v(theta0)

    numerator = (z_alpha * math.sqrt(var_theta0) + z_beta * math.sqrt(var_theta1)) ** 2
    delta = (theta1 - theta0) ** 2
    n_cases = int(math.ceil(numerator / delta))
    n_controls = int(math.ceil(n_cases * R))

    return {
        "n_cases": n_cases,
        "n_controls": n_controls,
        "total_n": n_cases + n_controls,
        "n_total": n_cases + n_controls,
        "auroc_expected": theta1,
        "auroc_null": theta0,
        "prevalence": phi,
        "allocation_ratio_controls_to_cases": float(R),
        "alpha": alpha,
        "power": power,
        "two_sided": two_sided,
    }
```

# Standalone Python Biostatistical Recipes & Formulas (Cloud Sandbox Edition)

Reference implementations for standalone execution in standard data science environments (such as Claude Web Analysis Tool, Jupyter, Google Colab) where standard libraries (`pandas`, `numpy`, `scipy`, `statsmodels`, `lifelines`, `scikit-learn`) are available **without** requiring custom external packages (zero `medstat` imports).

## Recipe Index

| # | Topic & Heading Anchor | Trigger Branch |
| :--- | :--- | :--- |
| **Recipe 1** | [Table 1 & Austin (2009) SMD](#1-table-1-baseline-characteristics--standardized-mean-difference-smd) | Baseline cohort summary, Fisher fallback, zero-variance guard |
| **Recipe 2** | [2x2 Diagnostic Accuracy](#2-2x2-diagnostic-accuracy-with-wilson-score--log-scale-95-cis) | Sens, Spec, PPV, NPV, LR+/LR-, DOR (Wilson & Haldane CIs) |
| **Recipe 3** | [Empirical ROC & DeLong CIs](#3-empirical-roc-optimal-cutoff--single--paired-delong-95-cis) | AUROC, Youden cutoff, single/paired DeLong test |
| **Recipe 4** | [Calibration & Vickers DCA](#4-model-calibration--vickers-decision-curve-analysis-dca) | Brier score, calibration slope/intercept, ICI, net benefit |
| **Recipe 5** | [Bland-Altman & Shrout-Fleiss ICC](#5-observer-agreement-bland-altman--shrout-fleiss-1979-icc) | Continuous LoA (1999 SEs) & ANOVA rater ICC (exact F CIs) |
| **Recipe 6** | [Propensity Score Matching (PSM)](#6-propensity-score-matching-psm-with-pair-ids--covariate-balance) | 1:1 nearest-neighbor matching (0.2 SD caliper), pair IDs, SMD |
| **Recipe 7** | [Firth Penalized Logistic Regression](#7-firths-penalized-likelihood-logistic-regression) | Sparse binary data ($\text{EPV}_{\text{binary}} < 10$) or separation (profile CIs) |
| **Recipe 8** | [Little's (1988) MCAR Test](#8-littles-1988-mcar-multivariate-missingness-test) | Pre-flight multivariate missingness pattern audit ($H_0: \text{MCAR}$) |
| **Recipe 9** | [VanderWeele & Ding E-Value](#9-vanderweele--ding-2017-sensitivity-e-value) | Unmeasured confounding sensitivity for observational OR/RR/HR |
| **Recipe 10** | [Multivariable Logistic Table](#10-multivariable-binary-logistic-regression-table) | Binary outcome regression ($\text{EPV}_{\text{binary}} \ge 10$, crude & adjusted ORs) |
| **Recipe 11** | [Survival Suite (KM, Log-Rank, Cox)](#11-survival-analysis-kaplan-meier-log-rank--cox-proportional-hazards) | Time-to-event survival, Schoenfeld PH check, L2-penalized Cox |
| **Recipe 12** | [MICE & Rubin's Rules Pooling](#12-multiple-imputation-by-chained-equations-mice--rubins-rules) | $5\%\text{--}40\%$ MAR covariate missingness (Bayesian draws, continuous targets) |
| **Recipe 13** | [Publication HTML Table Renderer](#13-publication-grade-html-table-renderer-nejm--jama-standards) | NEJM / JAMA 3-horizontal-rule HTML table formatting |
| **Recipe 14** | [IPTW & Weighted Balance](#14-inverse-probability-of-treatment-weighting-iptw--weighted-balance) | ATE/ATT/ATC/ATO weights, Kish $N_{\text{eff}}$, Bessel SMD, Huber-White SEs |
| **Recipe 15** | [Competing Risks (CIF & Fine-Gray)](#15-competing-risks-survival-analysis-aalen-johansen-cif--fine-gray-model) | Aalen-Johansen CIF, Naive $1-\text{KM}$ audit, csHR, Fine-Gray SHR |
| **Recipe 16** | [Diagnostic Reclassification](#16-diagnostic-risk-reclassification-continuous--categorical-nri-and-idi) | Incremental value via Continuous & Categorical NRI (event/non-event) & IDI |
| **Recipe 17** | [Time-Dependent ROC ($\text{AUROC}(t)$)](#17-time-dependent-roc-cumulativedynamic-ipcw-estimator-auroct) | Longitudinal discrimination across horizons under right-censoring |
| **Recipe 18** | [TRIPOD+AI Bootstrap Validation](#18-tripodai-optimism-corrected-bootstrap-internal-validation--uniform-shrinkage) | Optimism correction ($B \ge 200$), uniform shrinkage $\hat{s}$, intercept $\alpha_0$ |
| **Recipe 19** | [Prediction Model Recalibration](#19-prediction-model-recalibration-and-updating-logistic--intercept-only) | Intercept-only and logistic model updating in external cohorts |
| **Recipe 20** | [Sample Size (Riley & Obuchowski)](#20-sample-size-determination-for-clinical-prediction-models-riley-et-al-2020--diagnostic-auroc-obuchowski-1998) | Prediction model 4-criteria sample size & diagnostic AUROC power |

---

## 1. Table 1 Baseline Characteristics & Standardized Mean Difference (SMD)

Computes publication-grade baseline patient characteristics with Austin (2009) Standardized Mean Differences ($\text{SMD} < 0.10$ denotes balance). Handles zero-variance disparity correctly (returning `np.nan` rather than misleading `0.00`) and provides Fisher's exact test fallback for sparse cell counts.

```python
import numpy as np
import pandas as pd
import scipy.stats as stats


def calculate_smd(treated: np.ndarray, control: np.ndarray) -> float:
    """Calculate Austin (2009) Standardized Mean Difference for continuous variables."""
    t = np.asarray(treated, dtype=float)
    c = np.asarray(control, dtype=float)
    t = t[~np.isnan(t)]
    c = c[~np.isnan(c)]
    if len(t) == 0 or len(c) == 0:
        return np.nan
    mean_t, mean_c = np.mean(t), np.mean(c)
    var_t = np.var(t, ddof=1) if len(t) > 1 else 0.0
    var_c = np.var(c, ddof=1) if len(c) > 1 else 0.0
    pooled_sd = np.sqrt((var_t + var_c) / 2.0)
    if pooled_sd == 0:
        return 0.0 if mean_t == mean_c else np.nan
    return float((mean_t - mean_c) / pooled_sd)


def calculate_binary_smd(p0: float, p1: float) -> float:
    """Calculate Austin (2009) Standardized Mean Difference for binary proportions."""
    if np.isnan(p0) or np.isnan(p1):
        return np.nan
    sd_bin = np.sqrt((p0 * (1.0 - p0) + p1 * (1.0 - p1)) / 2.0)
    if sd_bin == 0:
        return 0.0 if p0 == p1 else np.nan
    return float((p1 - p0) / sd_bin)


def generate_table_one(
    df: pd.DataFrame,
    strata: str,
    continuous_vars: list[str],
    categorical_vars: list[str],
    nonnormal_vars: list[str] | None = None,
) -> pd.DataFrame:
    """Build stratified Table 1 with parametric/non-parametric tests and SMD."""
    nonnormal = set(nonnormal_vars or [])
    clean_df = df.dropna(subset=[strata]).copy()
    groups = sorted(clean_df[strata].unique())
    rows = []

    # Unified group column mapping to prevent count vs measurement column divergence
    col_labels = {
        g: f"{strata}={g} (N = {int(np.sum(clean_df[strata] == g))})" for g in groups
    }

    # 1. Total counts header
    total_counts = {"Characteristic": f"Overall (N = {len(clean_df)})"}
    for g in groups:
        total_counts[col_labels[g]] = f"n = {int(np.sum(clean_df[strata] == g))}"
    total_counts["P-value"] = ""
    total_counts["SMD"] = ""
    rows.append(total_counts)

    # 2. Continuous variables
    for var in continuous_vars:
        row = {"Characteristic": var}
        group_vals = [
            clean_df.loc[clean_df[strata] == g, var].dropna().to_numpy(dtype=float)
            for g in groups
        ]

        if var in nonnormal:
            for g, vals in zip(groups, group_vals):
                if len(vals) > 0:
                    med = np.median(vals)
                    q25, q75 = np.percentile(vals, [25, 75])
                    row[col_labels[g]] = f"{med:.1f} [{q25:.1f}, {q75:.1f}]"
                else:
                    row[col_labels[g]] = "-"
            try:
                pval = (
                    stats.kruskal(*group_vals).pvalue
                    if len(groups) > 1 and all(len(v) for v in group_vals)
                    else np.nan
                )
            except Exception:
                pval = np.nan
        else:
            for g, vals in zip(groups, group_vals):
                if len(vals) > 0:
                    mean = np.mean(vals)
                    sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                    row[col_labels[g]] = f"{mean:.1f} ({sd:.1f})"
                else:
                    row[col_labels[g]] = "-"
            try:
                pval = (
                    stats.f_oneway(*group_vals).pvalue
                    if len(groups) > 1 and all(len(v) for v in group_vals)
                    else np.nan
                )
            except Exception:
                pval = np.nan

        row["P-value"] = (
            f"{pval:.3f}"
            if (not np.isnan(pval) and pval >= 0.001)
            else ("<0.001" if not np.isnan(pval) else "")
        )
        if len(groups) == 2 and len(group_vals[0]) > 0 and len(group_vals[1]) > 0:
            smd_val = calculate_smd(group_vals[1], group_vals[0])
            row["SMD"] = f"{smd_val:.2f}" if not np.isnan(smd_val) else "-"
        else:
            row["SMD"] = ""
        rows.append(row)

    # 3. Categorical variables
    for var in categorical_vars:
        var_clean = clean_df.dropna(subset=[var])
        missing_cnt = len(clean_df) - len(var_clean)
        header_text = f"**{var}**" + (
            f" (Missing: {missing_cnt})" if missing_cnt > 0 else ""
        )
        rows.append({"Characteristic": header_text, "P-value": "", "SMD": ""})

        categories = sorted(var_clean[var].unique())
        contingency = pd.crosstab(var_clean[var], var_clean[strata])

        # P-value calculation with Fisher fallback for 2x2 with small expected counts
        pval = np.nan
        if contingency.size > 0:
            try:
                if contingency.shape == (2, 2) and (contingency.values < 5).any():
                    _, pval = stats.fisher_exact(contingency.values)
                else:
                    _, pval, _, _ = stats.chi2_contingency(contingency)
            except Exception:
                pval = np.nan

        p_str = (
            f"{pval:.3f}"
            if (not np.isnan(pval) and pval >= 0.001)
            else ("<0.001" if not np.isnan(pval) else "")
        )

        for idx, cat in enumerate(categories):
            row = {"Characteristic": f"  {cat}"}
            for g in groups:
                sub_g = var_clean[var_clean[strata] == g]
                cnt = int(np.sum(sub_g[var] == cat))
                pct = (cnt / len(sub_g) * 100.0) if len(sub_g) > 0 else np.nan
                row[col_labels[g]] = f"{cnt} ({pct:.1f}%)" if not np.isnan(pct) else "-"
            row["P-value"] = p_str if idx == 0 else ""

            if len(groups) == 2:
                g0_sub = var_clean[var_clean[strata] == groups[0]]
                g1_sub = var_clean[var_clean[strata] == groups[1]]
                if len(g0_sub) == 0 or len(g1_sub) == 0:
                    smd_cat = np.nan
                else:
                    p0 = float(np.mean(g0_sub[var] == cat))
                    p1 = float(np.mean(g1_sub[var] == cat))
                    smd_cat = calculate_binary_smd(p0, p1)
                row["SMD"] = f"{smd_cat:.2f}" if not np.isnan(smd_cat) else "-"
            else:
                row["SMD"] = ""
            rows.append(row)

    return pd.DataFrame(rows)
```

---

## 2. 2x2 Diagnostic Accuracy with Wilson Score & Log-Scale 95% CIs

Computes clinical diagnostic performance metrics with asymmetric Wilson score intervals (proportions) and log-scale intervals (likelihood ratios and diagnostic odds ratios) with Haldane-Anscombe zero-cell correction.

```python
import numpy as np
import pandas as pd
import scipy.stats as stats


def calculate_ci_wilson_score(k: int, n: int, ci: float = 0.95) -> tuple[float, float]:
    """Compute asymmetric Wilson score confidence interval for binomial proportion."""
    if n <= 0:
        return (np.nan, np.nan)
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
        if int(count_val) < 0:
            raise ValueError(
                f"Contingency cell count '{name}' must be non-negative (>= 0)."
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
    if not np.all(np.isin(y_valid, [0, 1])):
        raise ValueError(
            "Binary outcome y_true must contain strictly binary values (0 and 1)."
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

    records = []
    for pt in thresholds:
        if pt <= 0.0 or pt >= 1.0:
            continue
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

## 6. Propensity Score Matching (PSM) with Pair IDs & Covariate Balance

Performs 1:1 nearest-neighbor matching without replacement bounded by Austin (2011) 0.2 caliper of logit propensity score. Returns the matched cohort with **`pair_id`** column (enabling paired t-test, conditional logistic regression, and cluster-robust standard errors) alongside pre- and post-match SMD balance diagnostics.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import logit


def match_propensity_scores(
    df: pd.DataFrame,
    treatment_col: str,
    confounders: list[str],
    caliper_sd: float = 0.2,
) -> dict:
    """1:1 Nearest-Neighbor PSM with caliper, returning pair IDs, retention audit, and pre/post SMD balance."""

    # Self-contained Austin (2009) SMD helper to guarantee standalone execution
    def _calc_smd_local(t_arr, c_arr):
        t = np.asarray(t_arr, dtype=float)
        c = np.asarray(c_arr, dtype=float)
        t = t[~np.isnan(t)]
        c = c[~np.isnan(c)]
        if len(t) == 0 or len(c) == 0:
            return np.nan
        mean_t, mean_c = np.mean(t), np.mean(c)
        var_t = np.var(t, ddof=1) if len(t) > 1 else 0.0
        var_c = np.var(c, ddof=1) if len(c) > 1 else 0.0
        pooled_sd = np.sqrt((var_t + var_c) / 2.0)
        if pooled_sd == 0:
            return 0.0 if mean_t == mean_c else np.nan
        return float((mean_t - mean_c) / pooled_sd)

    n_initial = len(df)
    clean = (
        df.dropna(subset=[treatment_col] + confounders).copy().reset_index(drop=True)
    )
    n_analyzed = len(clean)
    n_excluded = n_initial - n_analyzed

    # Validate treatment column strictly binary
    treatment_vals = clean[treatment_col].to_numpy()
    if not np.all(np.isin(treatment_vals, [0, 1])):
        raise ValueError(
            f"Treatment column '{treatment_col}' must contain strictly binary values (0 and 1)."
        )
    y = treatment_vals.astype(int)

    # Handle categorical confounders via dummy encoding
    X_raw = pd.get_dummies(clean[confounders], drop_first=True, dtype=float)
    X = sm.add_constant(X_raw)

    # Fit propensity score model
    ps_model = sm.Logit(y, X).fit(disp=False)
    clean["ps"] = ps_model.predict(X)
    clean["logit_ps"] = logit(np.clip(clean["ps"], 1e-5, 1.0 - 1e-5))

    caliper = caliper_sd * float(np.std(clean["logit_ps"]))

    treated = clean[clean[treatment_col] == 1].copy()
    control = clean[clean[treatment_col] == 0].copy()

    # Sort treated by propensity score for deterministic nearest-neighbor matching
    treated = treated.sort_values(by="logit_ps", ascending=False)
    avail_ctrl = control.copy()

    matched_records = []
    pair_counter = 1

    for t_idx, t_row in treated.iterrows():
        if len(avail_ctrl) == 0:
            break
        diffs = np.abs(avail_ctrl["logit_ps"] - t_row["logit_ps"])
        min_diff = diffs.min()
        if min_diff <= caliper:
            best_match_idx = diffs.idxmin()

            t_entry = t_row.to_dict()
            t_entry["pair_id"] = pair_counter
            matched_records.append(t_entry)

            c_entry = avail_ctrl.loc[best_match_idx].to_dict()
            c_entry["pair_id"] = pair_counter
            matched_records.append(c_entry)

            avail_ctrl = avail_ctrl.drop(index=best_match_idx)
            pair_counter += 1

    matched_df = pd.DataFrame(matched_records)
    if len(matched_df) > 0:
        matched_clean = matched_df.reset_index(drop=True)
        matched_X = pd.get_dummies(
            matched_clean[confounders], drop_first=True, dtype=float
        )
    else:
        matched_X = pd.DataFrame(columns=X_raw.columns)

    # Compute balance diagnostics across ALL modeled confounders (including dummy-encoded levels)
    balance_rows = []
    for var in X_raw.columns:
        t_pre = X_raw.loc[clean[treatment_col] == 1, var].to_numpy()
        c_pre = X_raw.loc[clean[treatment_col] == 0, var].to_numpy()
        smd_pre = _calc_smd_local(t_pre, c_pre)

        if len(matched_df) > 0 and var in matched_X.columns:
            t_post = matched_X.loc[matched_clean[treatment_col] == 1, var].to_numpy()
            c_post = matched_X.loc[matched_clean[treatment_col] == 0, var].to_numpy()
            smd_post = _calc_smd_local(t_post, c_post)
        else:
            smd_post = np.nan

        balance_rows.append(
            {
                "Variable": var,
                "Pre-Match SMD": smd_pre,
                "Post-Match SMD": smd_post,
                "Balanced (<0.10)": abs(smd_post) < 0.10
                if not np.isnan(smd_post)
                else False,
            }
        )

    bal_df = pd.DataFrame(balance_rows)
    love_plot_data = {
        "covariates": list(bal_df["Variable"]),
        "smd_raw": [
            float(x) if not np.isnan(x) else np.nan for x in bal_df["Pre-Match SMD"]
        ],
        "smd_matched": [
            float(x) if not np.isnan(x) else np.nan for x in bal_df["Post-Match SMD"]
        ],
        "threshold": 0.10,
    }

    return {
        "matched_df": matched_df,
        "balance_df": bal_df,
        "love_plot_data": love_plot_data,
        "n_matched_pairs": pair_counter - 1,
        "n_unmatched_treated": len(treated) - (pair_counter - 1),
        "retention_audit": {
            "n_initial": n_initial,
            "n_excluded": n_excluded,
            "n_analyzed": n_analyzed,
            "exclusion_reasons": f"Missing values in {[treatment_col] + confounders}"
            if n_excluded > 0
            else "None",
        },
    }
```

---

## 7. Firth's Penalized Likelihood Logistic Regression

When sample sizes are small or monotone separation occurs ($\text{EPV} < 10$), standard MLE estimates explode ($\text{OR} \to \infty$). Firth's penalized logistic regression modifies the score equation by Jeffreys invariant prior: $U^*(\beta) = U(\beta) + a(\beta)$, where $a_j = \frac{1}{2}\text{tr}(I^{-1}\frac{\partial I}{\partial \beta_j}) = \sum_i h_i (0.5 - \pi_i) x_{ij}$. Reduces first-order parameter estimation bias (Heinze & Schemper, 2002). Implements **step-halving** to guarantee likelihood monotonicity and **profile likelihood confidence intervals** to eliminate Hauck-Donner explosion.

```python
import numpy as np
import scipy.stats as stats
import scipy.optimize as optimize


def _firth_penalized_loglik(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    eta = np.clip(X @ beta, -30, 30)
    pi = 1.0 / (1.0 + np.exp(-eta))
    W = pi * (1.0 - pi)
    I = (X.T * W) @ X
    sign, logdet = np.linalg.slogdet(I)
    if sign <= 0:
        return -np.inf
    ll = np.sum(
        y * np.log(np.clip(pi, 1e-15, 1.0))
        + (1.0 - y) * np.log(np.clip(1.0 - pi, 1e-15, 1.0))
    )
    return float(ll + 0.5 * logdet)


def fit_firth_logistic(
    X: np.ndarray,
    y: np.ndarray,
    fit_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict:
    """Firth penalized logistic regression with Newton-Raphson step-halving and Profile Likelihood CIs."""
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()

    if not np.all(np.isin(y_arr, [0, 1])):
        raise ValueError("Binary outcome y must contain strictly 0 and 1.")

    if fit_intercept and not np.allclose(X_arr[:, 0], 1.0):
        X_arr = np.column_stack([np.ones(len(y_arr)), X_arr])

    n, p = X_arr.shape
    rank = np.linalg.matrix_rank(X_arr)
    if rank < p:
        raise ValueError(
            f"Design matrix is rank-deficient (rank {rank} < {p} parameters). Collinear features must be removed."
        )

    beta = np.zeros(p)
    current_pll = _firth_penalized_loglik(beta, X_arr, y_arr)
    pll_history = [float(current_pll)]

    converged = False
    for iteration in range(max_iter):
        eta = np.clip(X_arr @ beta, -30, 30)
        pi = 1.0 / (1.0 + np.exp(-eta))
        W = pi * (1.0 - pi)

        I = (X_arr.T * W) @ X_arr
        try:
            I_inv = np.linalg.inv(I)
        except np.linalg.LinAlgError:
            I_inv = np.linalg.pinv(I)

        H = np.sum((X_arr @ I_inv) * X_arr, axis=1) * W
        u_star = X_arr.T @ (y_arr - pi + H * (0.5 - pi))
        step = I_inv @ u_star

        # Step-halving to enforce monotonically increasing penalized log-likelihood
        alpha_step = 1.0
        beta_new = beta + step
        new_pll = _firth_penalized_loglik(beta_new, X_arr, y_arr)
        while new_pll < current_pll and alpha_step > 1e-4:
            alpha_step *= 0.5
            beta_new = beta + alpha_step * step
            new_pll = _firth_penalized_loglik(beta_new, X_arr, y_arr)

        if new_pll < current_pll:
            # Monotonicity failed even after step halving
            converged = False
            break

        param_step = np.max(np.abs(beta_new - beta))
        score_norm = np.max(np.abs(u_star))
        beta = beta_new
        current_pll = new_pll
        pll_history.append(float(current_pll))

        # Both parameter movement and score gradient must satisfy convergence
        if param_step < tol and score_norm < (tol * max(1.0, float(n))):
            converged = True
            break

    se = np.sqrt(np.diag(I_inv))

    # Profile Likelihood Confidence Intervals (chi2 cutoff = 3.841)
    # A profile likelihood interval requires the maximized penalized likelihood at convergence.
    # If the model fails to converge, profile inference must not be reported; fallback to Wald intervals.
    ci_lower = np.zeros(p)
    ci_upper = np.zeros(p)
    if not converged:
        ci_lower = beta - 1.96 * se
        ci_upper = beta + 1.96 * se
        ci_method = ["wald_fallback"] * p
    else:
        ci_method = ["profile"] * p
        target_pll = current_pll - 0.5 * stats.chi2.ppf(0.95, df=1)

        for j in range(p):
            other_idx = [idx for idx in range(p) if idx != j]

            def profile_obj(b_j):
                if len(other_idx) == 0:
                    return (
                        _firth_penalized_loglik(np.array([b_j]), X_arr, y_arr)
                        - target_pll
                    )

                def nuisance_loss(nuis):
                    b_full = np.zeros(p)
                    b_full[j] = b_j
                    b_full[other_idx] = nuis
                    return -_firth_penalized_loglik(b_full, X_arr, y_arr)

                res = optimize.minimize(
                    nuisance_loss, beta[other_idx], method="Nelder-Mead"
                )
                if not res.success:
                    return np.nan
                return -res.fun - target_pll

            # Root search with explicit Wald fallback tracking
            try:
                low_guess = beta[j] - 2.5 * se[j]
                while profile_obj(low_guess) > 0 and low_guess > beta[j] - 20 * se[j]:
                    low_guess -= se[j]
                val_low = profile_obj(low_guess)
                val_mid = profile_obj(beta[j])
                if np.isnan(val_low) or np.isnan(val_mid) or val_low * val_mid > 0:
                    raise ValueError("Profile root bracket failure")
                ci_lower[j] = optimize.brentq(profile_obj, low_guess, beta[j])
            except Exception:
                ci_lower[j] = beta[j] - 1.96 * se[j]
                ci_method[j] = "wald_fallback"

            try:
                high_guess = beta[j] + 2.5 * se[j]
                while profile_obj(high_guess) > 0 and high_guess < beta[j] + 20 * se[j]:
                    high_guess += se[j]
                val_high = profile_obj(high_guess)
                val_mid = profile_obj(beta[j])
                if np.isnan(val_high) or np.isnan(val_mid) or val_high * val_mid > 0:
                    raise ValueError("Profile root bracket failure")
                ci_upper[j] = optimize.brentq(profile_obj, beta[j], high_guess)
            except Exception:
                ci_upper[j] = beta[j] + 1.96 * se[j]
                ci_method[j] = "wald_fallback"

    forest_items = []
    start_j = 1 if fit_intercept else 0
    for j in range(start_j, p):
        cov_idx = (j - 1) if fit_intercept else j
        v_name = f"Covariate_{cov_idx}"
        forest_items.append(
            {
                "term": v_name,
                "estimate": float(np.exp(beta[j])),
                "ci_lower": float(np.exp(ci_lower[j])),
                "ci_upper": float(np.exp(ci_upper[j])),
                "scale": "OR",
            }
        )

    return {
        "coefficients": beta,
        "standard_errors": se,
        "odds_ratios": np.exp(beta),
        "ci_lower": np.exp(ci_lower),
        "ci_upper": np.exp(ci_upper),
        "converged": converged,
        "ci_method": ci_method,
        "pll_history": pll_history,
        "forest_data": forest_items,
    }
```

---

## 8. Little's (1988) MCAR Multivariate Missingness Test

Pattern-grouped Expectation-Maximization (EM) ML parameter estimation under multivariate normality with Little's $d^2$ distance statistic: $d^2 = \sum_s n_s (\bar{y}_{s, obs} - \hat{\mu}_{obs})^T \hat{\Sigma}_{obs, obs}^{-1} (\bar{y}_{s, obs} - \hat{\mu}_{obs})$, with degrees of freedom $df = \sum p_s - P$. Tests the null hypothesis $H_0: \text{MCAR}$. Note that a non-significant test does not establish MAR or distinguish MAR from MNAR. Ridge covariance adjustment represents a numerical regularization approximation.

```python
import numpy as np
import pandas as pd
import scipy.stats as stats


def littles_mcar_test(
    df: pd.DataFrame, variables: list[str], max_iter: int = 100, tol: float = 1e-5
) -> dict:
    """Perform Roderick Little's (1988) MCAR test via pattern-grouped EM ML estimation.

    Statistical & Regularization Notes:
    - Tests the null hypothesis H0: Missing Completely at Random (MCAR). A non-significant
      result does NOT establish MAR or rule out MNAR.
    - Ridge covariance stabilization (Sigma + epsilon * I) is a numerical regularization
      approximation to ensure positive definiteness in small or collinear patterns.
    """
    data = df[variables].to_numpy(dtype=float, copy=True)
    N, P = data.shape
    nan_mask = np.isnan(data)

    if not np.any(nan_mask):
        return {
            "chi2": 0.0,
            "df": 0,
            "p_value": 1.0,
            "mcar_rejected": False,
            "n_patterns": 1,
            "converged": True,
            "n_iterations": 0,
            "n_excluded_rows": 0,
        }

    # Validate that data is not completely missing
    if nan_mask.all():
        raise ValueError(
            "Dataset is completely missing (100% NaN); Little's MCAR test cannot be computed."
        )

    for j, col_name in enumerate(variables):
        if nan_mask[:, j].all():
            raise ValueError(
                f"Variable '{col_name}' has 0 observed values (100% missing); MCAR cannot be tested."
            )

    # Drop rows that are completely missing across all tested variables
    row_all_nan = nan_mask.all(axis=1)
    n_excluded_rows = int(np.sum(row_all_nan))
    if n_excluded_rows > 0:
        data = data[~row_all_nan]
        nan_mask = nan_mask[~row_all_nan]
        N = len(data)

    if N == 0:
        raise ValueError(
            "No rows with observed values remain; Little's MCAR test cannot be computed."
        )

    # Group into unique missingness patterns using boolean indicator tuples
    obs_mask = ~nan_mask
    patterns_dict = {}
    for i, row in enumerate(obs_mask):
        key = tuple(row)
        patterns_dict.setdefault(key, []).append(i)

    parsed_patterns = []
    total_p_s = 0
    for key, row_indices in patterns_dict.items():
        O_s = np.array([j for j, obs in enumerate(key) if obs], dtype=int)
        M_s = np.array([j for j, obs in enumerate(key) if not obs], dtype=int)
        p_s = len(O_s)
        n_s = len(row_indices)
        total_p_s += p_s
        parsed_patterns.append(
            {
                "indices": np.array(row_indices, dtype=int),
                "O_s": O_s,
                "M_s": M_s,
                "p_s": p_s,
                "n_s": n_s,
            }
        )

    df_stat = total_p_s - P
    if df_stat <= 0:
        return {
            "chi2": np.nan,
            "df": int(df_stat),
            "p_value": np.nan,
            "mcar_rejected": False,
            "n_patterns": len(parsed_patterns),
            "converged": False,
            "n_iterations": 0,
            "n_excluded_rows": n_excluded_rows,
            "note": "Untestable: degrees of freedom <= 0 (insufficient overlap across missingness patterns).",
        }

    # EM Algorithm initialization
    mu = np.nanmean(data, axis=0)
    data_init = data.copy()
    for j in range(P):
        nan_j = np.isnan(data_init[:, j])
        data_init[nan_j, j] = mu[j]
    Sigma = np.cov(data_init, rowvar=False, ddof=1)
    if Sigma.ndim == 0:
        Sigma = np.array([[float(Sigma)]])
    scale_ridge = 1e-6 * (np.trace(Sigma) / P if np.trace(Sigma) > 0 else 1.0)
    Sigma += np.eye(P) * scale_ridge

    # EM loop
    converged = False
    n_iterations = 0
    for iteration in range(max_iter):
        n_iterations = iteration + 1
        sum_y = np.zeros(P)
        sum_yy = np.zeros((P, P))
        for pat in parsed_patterns:
            O_s, M_s, n_s, idx = pat["O_s"], pat["M_s"], pat["n_s"], pat["indices"]
            Y_pat = data[idx]
            if len(M_s) == 0:
                sum_y += np.sum(Y_pat, axis=0)
                sum_yy += Y_pat.T @ Y_pat
            else:
                Sigma_OO = Sigma[np.ix_(O_s, O_s)]
                Sigma_MO = Sigma[np.ix_(M_s, O_s)]
                Sigma_MM = Sigma[np.ix_(M_s, M_s)]
                try:
                    W = np.linalg.solve(
                        Sigma_OO + np.eye(len(O_s)) * scale_ridge, Sigma_MO.T
                    ).T
                except np.linalg.LinAlgError:
                    W = Sigma_MO @ np.linalg.pinv(Sigma_OO)
                C_MM = 0.5 * (Sigma_MM - W @ Sigma_MO.T + (Sigma_MM - W @ Sigma_MO.T).T)
                Y_imp = Y_pat.copy()
                Y_imp[:, M_s] = mu[M_s] + (Y_pat[:, O_s] - mu[O_s]) @ W.T
                sum_y += np.sum(Y_imp, axis=0)
                sum_yy += Y_imp.T @ Y_imp
                sum_yy[np.ix_(M_s, M_s)] += n_s * C_MM

        mu_next = sum_y / N
        Sigma_next = (sum_yy / N) - np.outer(mu_next, mu_next)
        Sigma_next = 0.5 * (Sigma_next + Sigma_next.T) + np.eye(P) * scale_ridge

        if max(np.max(np.abs(mu_next - mu)), np.max(np.abs(Sigma_next - Sigma))) < tol:
            mu, Sigma = mu_next, Sigma_next
            converged = True
            break
        mu, Sigma = mu_next, Sigma_next

    # Calculate Little's d2 distance statistic
    d2 = 0.0
    for pat in parsed_patterns:
        O_s, n_s, idx = pat["O_s"], pat["n_s"], pat["indices"]
        if len(O_s) == 0:
            continue
        y_bar_obs = np.mean(data[idx][:, O_s], axis=0)
        diff = y_bar_obs - mu[O_s]
        Sigma_OO = Sigma[np.ix_(O_s, O_s)]
        try:
            inv_Sigma_OO = np.linalg.inv(Sigma_OO + np.eye(len(O_s)) * scale_ridge)
        except np.linalg.LinAlgError:
            inv_Sigma_OO = np.linalg.pinv(Sigma_OO)
        d2 += n_s * float(diff.T @ inv_Sigma_OO @ diff)

    p_val = float(stats.chi2.sf(d2, df=df_stat))
    missing_summary = pd.DataFrame(
        {
            "Variable": variables,
            "N_Missing": [int(np.sum(np.isnan(df[v]))) for v in variables],
            "Pct_Missing": [float(np.mean(np.isnan(df[v])) * 100.0) for v in variables],
        }
    )
    return {
        "chi2": float(d2),
        "df": int(df_stat),
        "p_value": float(p_val),
        "mcar_rejected": bool(p_val < 0.05),
        "n_patterns": len(parsed_patterns),
        "converged": converged,
        "n_iterations": n_iterations,
        "n_excluded_rows": n_excluded_rows,
        "missingness_summary": missing_summary,
    }
```

---

## 9. VanderWeele & Ding (2017) Sensitivity E-Value

Calculates the minimum strength of association on the risk ratio scale that an unmeasured confounder must have with both exposure and outcome to fully explain away an observed association. Respects outcome prevalence: uses $RR \approx OR$ for rare outcomes ($<10\%$) and $\sqrt{OR}$ only for common outcomes.

```python
import numpy as np


def calculate_e_value(
    estimate: float,
    lower: float | None = None,
    upper: float | None = None,
    estimate_type: str = "RR",
    rare_outcome: bool = False,
) -> dict:
    """Calculate VanderWeele & Ding (2017) E-value for RR, OR, or HR with prevalence branching."""
    est = float(estimate)
    if est <= 0.0:
        raise ValueError("Estimate must be strictly positive (> 0) on ratio scale.")
    low = float(lower) if lower is not None else None
    up = float(upper) if upper is not None else None
    if low is not None and low <= 0.0:
        raise ValueError("lower limit must be strictly positive (> 0) on ratio scale.")
    if up is not None and up <= 0.0:
        raise ValueError("upper limit must be strictly positive (> 0) on ratio scale.")
    if low is not None and up is not None and low > up:
        raise ValueError(
            f"lower confidence limit ({low}) cannot exceed upper confidence limit ({up})."
        )

    # Conversion to Risk Ratio (RR) scale
    if estimate_type == "OR":
        if not rare_outcome:
            # Common outcome approximation per Ding & VanderWeele (2016)
            est = np.sqrt(est)
            if low is not None and low > 0:
                low = np.sqrt(low)
            if up is not None and up > 0:
                up = np.sqrt(up)
    elif estimate_type == "HR":
        if not rare_outcome:
            # VanderWeele & Ding (2017) HR conversion for common outcome
            def hr_to_rr(h):
                return (
                    (1.0 - 0.5 ** np.sqrt(h)) / (1.0 - 0.5 ** np.sqrt(1.0 / h))
                    if h != 1.0
                    else 1.0
                )

            est = hr_to_rr(est)
            if low is not None and low > 0:
                low = hr_to_rr(low)
            if up is not None and up > 0:
                up = hr_to_rr(up)

    rr_star = est if est >= 1.0 else (1.0 / est)
    e_val_point = rr_star + np.sqrt(rr_star * (rr_star - 1.0))

    e_val_ci = 1.0
    if est >= 1.0 and low is not None and low > 1.0:
        e_val_ci = low + np.sqrt(low * (low - 1.0))
    elif est < 1.0 and up is not None and up < 1.0:
        inv_u = 1.0 / up
        e_val_ci = inv_u + np.sqrt(inv_u * (inv_u - 1.0))

    return {
        "e_value_point": float(e_val_point),
        "e_value_ci": float(e_val_ci),
        "scale_note": f"Approximated as RR ({'rare outcome: unadjusted' if rare_outcome else 'common outcome: transformed'})",
    }
```

---

## 10. Multivariable Binary Logistic Regression Table

Fits univariable and multivariable logistic regression using `statsmodels` and builds a complete publication table reporting **Crude OR (95% CI)**, **Adjusted OR (95% CI)**, and p-values.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_logistic_regression_table(
    df: pd.DataFrame, outcome: str, covariates: list[str]
) -> pd.DataFrame:
    """Generate publication-ready Logistic Regression table with Crude & Adjusted ORs."""
    n_initial = len(df)
    clean = df.dropna(subset=[outcome] + covariates).copy()
    n_analyzed = len(clean)
    n_excluded = n_initial - n_analyzed

    outcome_vals = clean[outcome].to_numpy()
    if not np.all(np.isin(outcome_vals, [0, 1])):
        raise ValueError(
            f"Outcome column '{outcome}' must contain strictly binary values (0 and 1)."
        )
    y = outcome_vals.astype(int)

    # 1. Multivariable model
    X_multi = sm.add_constant(clean[covariates].astype(float))
    fit_multi = sm.Logit(y, X_multi).fit(disp=False)

    rows = []
    for var in covariates:
        # Univariable model (Crude OR)
        X_uni = sm.add_constant(clean[[var]].astype(float))
        fit_uni = sm.Logit(y, X_uni).fit(disp=False)

        crude_b = fit_uni.params[var]
        crude_ci = fit_uni.conf_int().loc[var]
        crude_p = fit_uni.pvalues[var]
        crude_str = f"{np.exp(crude_b):.2f} ({np.exp(crude_ci[0]):.2f}–{np.exp(crude_ci[1]):.2f})"

        # Multivariable (Adjusted OR)
        adj_b = fit_multi.params[var]
        adj_ci = fit_multi.conf_int().loc[var]
        adj_p = fit_multi.pvalues[var]
        adj_str = (
            f"{np.exp(adj_b):.2f} ({np.exp(adj_ci[0]):.2f}–{np.exp(adj_ci[1]):.2f})"
        )

        p_val_str = f"{adj_p:.3f}" if adj_p >= 0.001 else "<0.001"

        rows.append(
            {
                "Variable": var,
                "Crude OR (95% CI)": crude_str,
                "Adjusted OR (95% CI)": adj_str,
                "P-value": p_val_str,
            }
        )

    res_df = pd.DataFrame(rows)
    res_df.attrs["retention_audit"] = {
        "n_initial": n_initial,
        "n_excluded": n_excluded,
        "n_analyzed": n_analyzed,
        "exclusion_reasons": f"Missing values in {[outcome] + covariates}"
        if n_excluded > 0
        else "None",
    }
    return res_df
```

---

## 11. Survival Analysis: Kaplan-Meier, Log-Rank & Cox Proportional Hazards

Computes Kaplan-Meier survival estimates with median survival times, Log-Rank hypothesis test (supports 2 or more groups), and multivariable Cox Proportional Hazards table with per-variable Grambsch-Therneau Schoenfeld residual tests.

```python
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import (
    logrank_test,
    multivariate_logrank_test,
    proportional_hazard_test,
)


def fit_survival_analysis_suite(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    strata_col: str | None = None,
    covariates: list[str] | None = None,
    penalizer: float = 0.0,
    standardize_continuous: bool = False,
    continuous_covariates: list[str] | None = None,
) -> dict:
    """Run Kaplan-Meier, Log-Rank, and Cox PH modeling with per-covariate Schoenfeld test and optional scale-aware standardization."""
    n_initial = len(df)

    # 1. Unadjusted Kaplan-Meier cohort: subset duration, event, and strata (preserves KM cohort)
    km_cols = [duration_col, event_col] + ([strata_col] if strata_col else [])
    clean_km = df.dropna(subset=km_cols).copy()

    # Validate event indicator strictly binary and duration non-negative
    ev_km = clean_km[event_col].to_numpy()
    if not np.all(np.isin(ev_km, [0, 1])):
        raise ValueError(
            f"Event indicator '{event_col}' must contain strictly binary values (0 = Censored, 1 = Event)."
        )
    dur_km = clean_km[duration_col].to_numpy()
    if (dur_km < 0).any():
        raise ValueError(
            f"Duration column '{duration_col}' must contain non-negative values (>= 0)."
        )

    # Kaplan-Meier curve & median survival
    km_results = {}
    km_curves = {}
    risk_table = pd.DataFrame()
    if strata_col:
        groups = clean_km[strata_col].unique()
        for g in groups:
            sub = clean_km[clean_km[strata_col] == g]
            kmf = KaplanMeierFitter()
            kmf.fit(sub[duration_col], sub[event_col], label=str(g))
            km_results[f"Median Survival ({g})"] = float(kmf.median_survival_time_)
            ci_df = kmf.confidence_interval_
            cens_t = sub.loc[sub[event_col] == 0, duration_col].to_numpy()
            km_curves[str(g)] = {
                "timeline": np.asarray(kmf.timeline),
                "survival": np.asarray(kmf.survival_function_[str(g)]),
                "ci_lower": np.asarray(ci_df.iloc[:, 0]),
                "ci_upper": np.asarray(ci_df.iloc[:, 1]),
                "censored_times": cens_t,
                "censored_survival": np.asarray(kmf.survival_function_at_times(cens_t))
                if len(cens_t) > 0
                else np.array([]),
            }

        # Risk table at milestone intervals
        max_t = float(clean_km[duration_col].max())
        milestones = np.linspace(0, max_t, 5).round(1)
        if len(milestones) > 0 and milestones[-1] > max_t:
            milestones[-1] = float(np.floor(max_t * 10) / 10)
        r_rows = []
        for g in groups:
            sub = clean_km[clean_km[strata_col] == g]
            row_dict = {"Group": str(g)}
            for m in milestones:
                row_dict[f"t={m}"] = int(np.sum(sub[duration_col] >= m))
            r_rows.append(row_dict)
        risk_table = pd.DataFrame(r_rows)

        # Log-rank test (2 groups or multivariate for >2 groups)
        if len(groups) == 2:
            lr_res = logrank_test(
                clean_km.loc[clean_km[strata_col] == groups[0], duration_col],
                clean_km.loc[clean_km[strata_col] == groups[1], duration_col],
                clean_km.loc[clean_km[strata_col] == groups[0], event_col],
                clean_km.loc[clean_km[strata_col] == groups[1], event_col],
            )
            km_results["Log-Rank P-value"] = float(lr_res.p_value)
        elif len(groups) > 2:
            lr_res = multivariate_logrank_test(
                clean_km[duration_col], clean_km[strata_col], clean_km[event_col]
            )
            km_results["Multivariate Log-Rank P-value"] = float(lr_res.p_value)
    else:
        kmf = KaplanMeierFitter()
        kmf.fit(clean_km[duration_col], clean_km[event_col], label="Overall")
        km_results["Overall Median Survival"] = float(kmf.median_survival_time_)
        ci_df = kmf.confidence_interval_
        cens_t = clean_km.loc[clean_km[event_col] == 0, duration_col].to_numpy()
        km_curves["Overall"] = {
            "timeline": np.asarray(kmf.timeline),
            "survival": np.asarray(kmf.survival_function_["Overall"]),
            "ci_lower": np.asarray(ci_df.iloc[:, 0]),
            "ci_upper": np.asarray(ci_df.iloc[:, 1]),
            "censored_times": cens_t,
            "censored_survival": np.asarray(kmf.survival_function_at_times(cens_t))
            if len(cens_t) > 0
            else np.array([]),
        }

    # 2. Cox Proportional Hazards model on complete-covariate cohort
    cox_table = pd.DataFrame()
    schoenfeld_summary = {}
    forest_items = []
    cox_standardization = {}
    n_analyzed_cox = 0
    if covariates:
        cox_cols = [duration_col, event_col] + covariates
        clean_cox = df.dropna(subset=cox_cols).copy()
        n_analyzed_cox = len(clean_cox)
        ev_cox = clean_cox[event_col].to_numpy()
        if not np.all(np.isin(ev_cox, [0, 1])):
            raise ValueError(
                f"Event indicator '{event_col}' must contain strictly binary values."
            )

        if penalizer < 0:
            raise ValueError("penalizer must be >= 0")
        cox_data = clean_cox[cox_cols].copy()
        for var in covariates:
            col_s = cox_data[var]
            if not pd.api.types.is_numeric_dtype(col_s) or pd.api.types.is_bool_dtype(col_s):
                raise ValueError(
                    f"Covariate '{var}' must be numeric; encode categorical covariates as binary 0/1 indicators before Cox fitting."
                )
        if standardize_continuous:
            if continuous_covariates is None:
                raise ValueError(
                    "continuous_covariates must be specified explicitly when standardize_continuous=True."
                )
            cont_set = set(continuous_covariates)
            unknown_cont = cont_set - set(covariates)
            if unknown_cont:
                raise ValueError(
                    f"continuous_covariates {sorted(unknown_cont)} must be present in covariates."
                )
            for var in covariates:
                col_s = cox_data[var]
                if var in cont_set:
                    mean_v = float(col_s.mean())
                    sd_v = float(col_s.std(ddof=1))
                    if sd_v <= 0:
                        raise ValueError(
                            f"Continuous covariate '{var}' has zero variance and cannot be standardized."
                        )
                    cox_data[var] = (col_s - mean_v) / sd_v
                    cox_standardization[var] = {
                        "mean": mean_v,
                        "sd": sd_v,
                        "scale": "per 1 SD increase",
                    }
                else:
                    if not np.all(np.isin(col_s.to_numpy(), [0, 1])):
                        raise ValueError(
                            f"Non-continuous covariate '{var}' must be pre-encoded as binary 0/1 indicator(s) before Cox fitting."
                        )
        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(cox_data, duration_col=duration_col, event_col=event_col)

        summary = cph.summary
        rows = []
        for var in covariates:
            hr = float(summary.loc[var, "exp(coef)"])
            ci_l = float(summary.loc[var, "exp(coef) lower 95%"])
            ci_u = float(summary.loc[var, "exp(coef) upper 95%"])
            p_val = float(summary.loc[var, "p"])
            rows.append(
                {
                    "Covariate": var,
                    "Hazard Ratio (95% CI)": f"{hr:.2f} ({ci_l:.2f}–{ci_u:.2f})",
                    "P-value": f"{p_val:.3f}" if p_val >= 0.001 else "<0.001",
                }
            )
            forest_items.append(
                {
                    "term": var,
                    "estimate": hr,
                    "ci_lower": ci_l,
                    "ci_upper": ci_u,
                    "p_value": p_val,
                    "scale": "HR",
                }
            )
        cox_table = pd.DataFrame(rows)

        # Proportional hazards test per covariate
        try:
            prop_test = proportional_hazard_test(cph, cox_data, time_transform="rank")
            schoenfeld_summary["p_values"] = {
                k: float(v) for k, v in prop_test.summary["p"].items()
            }
            schoenfeld_summary["test_statistics"] = {
                k: float(v) for k, v in prop_test.summary["test_statistic"].items()
            }
            passed = bool((prop_test.summary["p"] > 0.05).all())
            schoenfeld_summary["no_ph_violation_detected"] = passed
            schoenfeld_summary["PH_Assumptions_Passed"] = (
                passed  # backward compatibility
            )
            schoenfeld_summary["PH_Note"] = (
                "No proportional hazards violation detected (p > 0.05)."
                if passed
                else "Potential PH assumption violation detected (p <= 0.05)."
            )
        except Exception as e:
            schoenfeld_summary["PH_Note"] = str(e)

    return {
        "km_summary": km_results,
        "km_curves": km_curves,
        "risk_table": risk_table,
        "cox_table": cox_table,
        "cox_penalizer": penalizer if covariates else None,
        "cox_standardization": cox_standardization if covariates else None,
        "forest_data": forest_items,
        "schoenfeld_diagnostics": schoenfeld_summary,
        "retention_km": {
            "n_initial": n_initial,
            "n_analyzed": len(clean_km),
            "n_excluded": n_initial - len(clean_km),
        },
        "retention_cox": {
            "n_initial": n_initial,
            "n_analyzed": n_analyzed_cox,
            "n_excluded": n_initial - n_analyzed_cox,
        }
        if covariates
        else None,
    }
```

---

## 12. Multiple Imputation by Chained Equations (MICE) & Rubin's Rules

Provides multiple stochastic imputations ($M \ge 5$) via **Proper Bayesian Linear Regression MICE** (Rubin 1987; Schafer 1997; van Buuren 2018), strictly enforcing that **primary outcomes are never imputed**. Jointly samples parameter uncertainty per cycle ($\sigma^{*2} \sim \text{Inv-}\chi^2(\nu, s^2)$, $\beta^* \sim N(\hat{\beta}, \sigma^{*2}(X^TX)^{-1})$) and missing values ($y_{\mathrm{mis}}^* \sim N(X_{\mathrm{mis}}\beta^*, \sigma^{*2})$) at every chained cycle. Drawing a shared parameter vector per cycle correctly propagates model estimation uncertainty across all missing records, whereas independent marginal predictions fail to preserve joint coefficient variation. Configured strictly for continuous/numeric clinical features with optional boundary constraints (`min_value`, `max_value`) sampled via truncated Gaussian distributions. Imputation targets (`features_to_impute`) must be continuous variables. Fully observed dummy-encoded variables may be supplied in `predictors` to serve as conditioning features. Missing categorical targets require variable-type-specific imputation (e.g., logistic/polytomous regression) and must not be included in `features_to_impute`. Includes pure-Python **Rubin's rules pooling** (Rubin 1987; Barnard & Rubin 1999) with finite-sample degrees of freedom adjustment. Ratio effect measures (OR, HR) must be pooled on the log scale and exponentiated.

```python
import numpy as np
import pandas as pd
import scipy.stats as stats


def impute_mice_datasets(
    df: pd.DataFrame,
    features_to_impute: list[str],
    outcome_col: str | None = None,
    predictors: list[str] | None = None,
    m: int = 5,
    max_iter: int = 10,
    random_state: int = 42,
    min_value: float | dict[str, float] | None = None,
    max_value: float | dict[str, float] | None = None,
    target_types: dict[str, str] | None = None,
) -> list[pd.DataFrame]:
    """Generate M stochastic imputed datasets for valid inferential analysis under MAR.

    Clinical Governance Invariants:
    1. Never impute the primary outcome variable (its original values remain untouched).
    2. Primary outcome and additional covariates MUST be included as predictors in the
       imputation model per standard biostatistical methodology (van Buuren; Moons et al. 2006)
       to prevent severe attenuation towards the null.
    3. Proper Bayesian draws: Jointly samples parameter uncertainty (beta*, sigma*) and missing values.
       In an evaluated 50-trial MAR simulation benchmark, recovered empirical coverage (~94%, MCSE ~3.4 pp).
    4. Continuous imputation targets only: features_to_impute must be continuous variables.
       Value-based guards screen against binary and low-cardinality discrete integers ({1, 2}, {1, 2, 3}),
       but explicit target-type validation remains essential. Missing categorical targets require
       variable-type-specific models.
    5. Observed values of features_to_impute are preserved intact; only missing values are imputed.
    6. Bounded sampling: When min_value or max_value is supplied, draws are generated via truncated normal
       sampling rather than clipping, preserving continuous probability density without boundary point masses.
       Note that bounding modifies the conditional model; inferential claims should be qualified accordingly.
    7. Conditioning completeness: Conditioning-only predictors must be fully observed. Primary outcome must
       not contain missing values without prior explicit cohort disposition.
    """
    if not isinstance(m, int) or isinstance(m, bool) or m < 1:
        raise ValueError(
            f"Number of imputations m must be an integer >= 1; got {m!r} "
            f"of type {type(m).__name__}."
        )

    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
        raise ValueError(
            f"max_iter must be an integer >= 1; got {max_iter!r} "
            f"of type {type(max_iter).__name__}."
        )

    if outcome_col and outcome_col in features_to_impute:
        raise ValueError(
            "Clinical governance invariant: Never impute the primary outcome variable!"
        )

    cols = list(
        dict.fromkeys(
            features_to_impute
            + (predictors or [])
            + ([outcome_col] if outcome_col else [])
        )
    )
    if len(cols) < 2:
        raise ValueError(
            "MICE requires >= 1 additional predictor (or outcome variable) to condition on; "
            "otherwise chained imputation degenerates to unconditional mean imputation."
        )

    # Validate that features_to_impute are strictly continuous
    for feat in features_to_impute:
        if df[feat].isna().all():
            raise ValueError(
                f"Column '{feat}' has 0 observed values (100% missing) and cannot be imputed."
            )
        if (
            isinstance(df[feat].dtype, pd.CategoricalDtype)
            or df[feat].dtype.name == "category"
        ):
            raise ValueError(
                f"Column '{feat}' is declared as categorical. "
                "Recipe 12 implements Gaussian MICE for continuous variables. "
                "Missing categorical targets require variable-type-specific imputation."
            )
        if target_types and target_types.get(feat) in {
            "categorical",
            "ordinal",
            "binary",
        }:
            raise ValueError(
                f"Column '{feat}' is declared as {target_types.get(feat)}. "
                "Recipe 12 implements Gaussian MICE strictly for continuous targets."
            )
        if pd.api.types.is_bool_dtype(df[feat]) or not pd.api.types.is_numeric_dtype(
            df[feat]
        ):
            raise ValueError(
                f"Column '{feat}' in features_to_impute is non-numeric ({df[feat].dtype}). "
                "Recipe 12 implements Gaussian MICE for continuous variables. "
                "Missing categorical targets require variable-type-specific imputation."
            )
        non_null_vals = df[feat].dropna().unique()
        vals_set = set(non_null_vals)
        if len(non_null_vals) <= 2:
            raise ValueError(
                f"Column '{feat}' in features_to_impute is binary/dummy ({vals_set}) with <= 2 unique observed values. "
                "Binary/dummy variables cannot be imputed using Gaussian linear regression MICE; "
                "Gaussian posterior draws produce fractional values and cannot enforce binary {0, 1} "
                "or group membership. Restrict features_to_impute to continuous variables."
            )
        if vals_set.issubset({1, 2, 3, 1.0, 2.0, 3.0}) or vals_set.issubset(
            {0, 1, 2, 0.0, 1.0, 2.0}
        ):
            raise ValueError(
                f"Column '{feat}' in features_to_impute has discrete integer category codes ({vals_set}). "
                "Categorical and ordinal variables must not be imputed via Gaussian linear regression MICE; "
                "they require variable-type-specific models (e.g. polytomous/ordinal logistic MICE)."
            )

    # Validate conditioning columns, predictors, and outcome
    for c in cols:
        if df[c].isna().all():
            raise ValueError(
                f"Column '{c}' has 0 observed values (100% missing) and cannot be conditioned upon."
            )
        if pd.api.types.is_bool_dtype(df[c]) or not pd.api.types.is_numeric_dtype(
            df[c]
        ):
            raise ValueError(
                f"Column '{c}' is non-numeric ({df[c].dtype}). "
                "All features in MICE conditioning matrix must be numeric. "
                "Categorical predictors must be dummy-encoded before inclusion in 'predictors'."
            )
        obs_c = df[c].dropna().values
        if len(obs_c) > 0 and not np.all(np.isfinite(obs_c)):
            raise ValueError(
                f"Column '{c}' contains non-finite values (inf / -inf). "
                "Non-finite values must be cleaned prior to MICE."
            )

    if predictors:
        for pred in predictors:
            if pred not in features_to_impute and df[pred].isna().any():
                non_null_p = df[pred].dropna().unique()
                if len(non_null_p) <= 2:
                    raise ValueError(
                        f"Predictor '{pred}' is binary/dummy and contains missing values. "
                        "Gaussian MICE cannot impute categorical variables. "
                        "Categorical predictors must be fully observed to condition on in Gaussian MICE."
                    )
                raise ValueError(
                    f"Conditioning predictor '{pred}' contains missing values. "
                    "Conditioning-only predictors must be fully observed. "
                    "If this continuous predictor requires imputation, include it in 'features_to_impute'."
                )

    if outcome_col and df[outcome_col].isna().any():
        raise ValueError(
            f"Primary outcome column '{outcome_col}' contains missing values. "
            "Clinical governance invariant: Never impute the primary outcome variable. "
            "Rows with missing outcome must have an explicit cohort disposition (e.g. complete-case exclusion) prior to MICE."
        )

    # Validate bound ordering
    for feat in features_to_impute:
        low_val = (
            min_value[feat]
            if isinstance(min_value, dict) and feat in min_value
            else (min_value if isinstance(min_value, (int, float)) else None)
        )
        high_val = (
            max_value[feat]
            if isinstance(max_value, dict) and feat in max_value
            else (max_value if isinstance(max_value, (int, float)) else None)
        )
        if low_val is not None and high_val is not None and low_val >= high_val:
            raise ValueError(
                f"Invalid bound ordering for feature '{feat}': min_value ({low_val}) must be strictly less than max_value ({high_val})."
            )

    rng = np.random.default_rng(random_state)
    datasets = []
    for m_idx in range(m):
        clean = df.copy()
        # Initialize missing values in features_to_impute with random draws from observed
        for feat in features_to_impute:
            mis = clean[feat].isna()
            obs_vals = clean.loc[~mis, feat].values
            if len(obs_vals) > 0 and mis.any():
                clean.loc[mis, feat] = rng.choice(obs_vals, size=mis.sum())

        # Proper Bayesian chained equations iterations
        for it in range(max_iter):
            for feat in features_to_impute:
                mis_mask = df[feat].isna()
                if not mis_mask.any():
                    continue
                obs_mask = ~mis_mask
                pred_cols = [c for c in cols if c != feat]

                X_obs = np.column_stack(
                    [np.ones(obs_mask.sum()), clean.loc[obs_mask, pred_cols].values]
                )
                y_obs = df.loc[obs_mask, feat].values
                X_mis = np.column_stack(
                    [np.ones(mis_mask.sum()), clean.loc[mis_mask, pred_cols].values]
                )

                n_obs, p = X_obs.shape
                if n_obs <= p:
                    raise ValueError(
                        f"Insufficient observed cases for feature '{feat}': n_obs ({n_obs}) must be strictly greater "
                        f"than number of parameters p ({p}) to form a valid Bayesian regression posterior."
                    )
                rank_obs = np.linalg.matrix_rank(X_obs)
                if rank_obs < p:
                    raise ValueError(
                        f"Design matrix for feature '{feat}' is rank-deficient (rank {rank_obs} < {p} parameters). "
                        "Collinear predictors or zero-variance columns must be removed before Bayesian regression MICE."
                    )

                XtX = X_obs.T @ X_obs
                Xty = X_obs.T @ y_obs
                beta_hat = np.linalg.solve(XtX, Xty)
                residuals = y_obs - X_obs @ beta_hat
                rss = float(np.sum(residuals**2))
                if rss <= 1e-12:
                    raise ValueError(
                        f"Degenerate residual sum of squares for feature '{feat}' (RSS <= 1e-12). "
                        "The target is perfectly predicted by conditioning features; "
                        "flat-prior Bayesian regression variance posterior is improper for deterministic relationships."
                    )
                df_resid = n_obs - p
                s2 = rss / df_resid

                # 1. Parameter uncertainty: Draw sigma*^2 ~ Inv-Chi2(df_resid, s2)
                g = rng.chisquare(df_resid)
                if g <= 0.0 or not np.isfinite(g):
                    raise RuntimeError(
                        f"Numerical failure in chi-square posterior draw for feature '{feat}'."
                    )
                sigma_sq = (df_resid * s2) / g
                if not np.isfinite(sigma_sq) or sigma_sq <= 0.0:
                    raise RuntimeError(
                        f"Non-finite or non-positive variance posterior draw for feature '{feat}'."
                    )
                sigma = np.sqrt(sigma_sq)

                # 2. Parameter uncertainty: Draw beta* ~ N(beta_hat, sigma^2 * (XtX)^-1)
                V_inv = np.linalg.inv(XtX)
                L = np.linalg.cholesky(V_inv)
                beta_star = beta_hat + sigma * (L @ rng.normal(size=p))

                # 3. Residual uncertainty: Draw y_mis* ~ N(X_mis @ beta*, sigma^2) with truncated normal support under bounds
                mu = X_mis @ beta_star
                low_val = (
                    min_value[feat]
                    if isinstance(min_value, dict) and feat in min_value
                    else (min_value if isinstance(min_value, (int, float)) else None)
                )
                high_val = (
                    max_value[feat]
                    if isinstance(max_value, dict) and feat in max_value
                    else (max_value if isinstance(max_value, (int, float)) else None)
                )

                if low_val is not None or high_val is not None:
                    a_param = (low_val - mu) / sigma if low_val is not None else -np.inf
                    b_param = (
                        (high_val - mu) / sigma if high_val is not None else np.inf
                    )
                    y_imp = stats.truncnorm.rvs(
                        a=a_param, b=b_param, loc=mu, scale=sigma, random_state=rng
                    )
                else:
                    y_imp = mu + sigma * rng.normal(size=len(X_mis))

                clean.loc[mis_mask, feat] = y_imp

        # Primary outcome column preserved untouched
        if outcome_col:
            clean[outcome_col] = df[outcome_col]

        datasets.append(clean)
    return datasets


def pool_estimates_rubin(
    point_estimates: list[float],
    standard_errors: list[float],
    n_obs: int | None = None,
    k_params: int = 1,
    alpha: float = 0.05,
) -> dict:
    """Pool point estimates and standard errors across M imputations using Rubin's rules.

    Clinical & Statistical Invariant:
    When pooling ratio effect measures (such as Odds Ratios or Hazard Ratios), pooling
    MUST be performed on the log scale (log OR, log HR) with log-scale standard errors,
    and then exponentiated for reporting and hypothesis testing.
    """
    m = len(point_estimates)
    if len(point_estimates) != len(standard_errors):
        raise ValueError(
            f"Length mismatch: point_estimates ({len(point_estimates)}) and "
            f"standard_errors ({len(standard_errors)}) must have identical lengths."
        )
    if m < 2:
        raise ValueError("Rubin pooling requires at least m=2 imputed datasets.")

    theta = np.asarray(point_estimates, dtype=float)
    ses = np.asarray(standard_errors, dtype=float)

    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(ses)):
        raise ValueError(
            "point_estimates and standard_errors must contain strictly finite values (no NaN or inf)."
        )

    if (ses < 0.0).any():
        raise ValueError("standard_errors must be strictly non-negative (>= 0).")

    if n_obs is not None and n_obs <= k_params:
        raise ValueError(
            f"n_obs ({n_obs}) must be strictly greater than k_params ({k_params})."
        )

    V = ses**2
    theta_bar = float(np.mean(theta))
    W_bar = float(np.mean(V))
    B = float(np.var(theta, ddof=1))
    T = float(W_bar + (1.0 + 1.0 / m) * B)
    se = float(np.sqrt(T))

    # Degrees of freedom with Barnard & Rubin (1999) finite-sample adjustment
    if n_obs is not None:
        nu_com = max(1.0, float(n_obs - k_params))
        nu_com_adj = ((nu_com + 1.0) / (nu_com + 3.0)) * nu_com
    else:
        nu_com = None
        nu_com_adj = None

    if B > 0 and W_bar > 0:
        r = float((1.0 + 1.0 / m) * B / W_bar)
        df_old = (m - 1) * (1.0 + 1.0 / r) ** 2
        if nu_com_adj is not None:
            lam = r / (r + 1.0)
            df_obs = nu_com_adj * (1.0 - lam)
            df = float((df_old * df_obs) / (df_old + df_obs))
        else:
            df = float(df_old)
        fmi = (
            float((r + 2.0 / (df + 3.0)) / (r + 1.0))
            if np.isfinite(df)
            else float(r / (r + 1.0))
        )
    elif B == 0 and W_bar > 0:
        r = 0.0
        df = float(nu_com_adj) if nu_com_adj is not None else np.inf
        fmi = 0.0
    elif B > 0 and W_bar == 0:
        r = np.inf
        df = float(m - 1)
        fmi = 1.0
    else:  # B == 0 and W_bar == 0
        r = 0.0
        df = float(nu_com_adj) if nu_com_adj is not None else np.inf
        fmi = 0.0

    t_stat = theta_bar / se if se > 0 else 0.0
    if np.isfinite(df) and df > 0:
        p_val = float(2.0 * stats.t.sf(abs(t_stat), df))
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    else:
        p_val = float(2.0 * stats.norm.sf(abs(t_stat)))
        t_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))

    ci_lower = float(theta_bar - t_crit * se)
    ci_upper = float(theta_bar + t_crit * se)

    return {
        "pooled_estimate": theta_bar,
        "pooled_se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_val,
        "within_variance": W_bar,
        "between_variance": B,
        "total_variance": T,
        "df": df,
        "fmi": fmi,
        "m_imputations": m,
    }


def impute_mice_single(
    df: pd.DataFrame,
    features_to_impute: list[str],
    outcome_col: str | None = None,
    predictors: list[str] | None = None,
    max_iter: int = 10,
    random_state: int = 42,
    min_value: float | dict[str, float] | None = None,
    max_value: float | dict[str, float] | None = None,
    target_types: dict[str, str] | None = None,
) -> pd.DataFrame:
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
        raise ValueError(
            f"max_iter must be an integer >= 1; got {max_iter!r} "
            f"of type {type(max_iter).__name__}."
        )

    if outcome_col and outcome_col in features_to_impute:
        raise ValueError(
            "Clinical governance invariant: Never impute the primary outcome variable!"
        )

    cols = list(
        dict.fromkeys(
            features_to_impute
            + (predictors or [])
            + ([outcome_col] if outcome_col else [])
        )
    )
    if len(cols) < 2:
        raise ValueError(
            "MICE requires >= 1 additional predictor (or outcome variable) to condition on; "
            "otherwise chained imputation degenerates to unconditional mean imputation."
        )

    # Validate that features_to_impute are strictly continuous
    for feat in features_to_impute:
        if df[feat].isna().all():
            raise ValueError(
                f"Column '{feat}' has 0 observed values (100% missing) and cannot be imputed."
            )
        if (
            isinstance(df[feat].dtype, pd.CategoricalDtype)
            or df[feat].dtype.name == "category"
        ):
            raise ValueError(
                f"Column '{feat}' is declared as categorical. "
                "Recipe 12 implements Gaussian MICE for continuous variables. "
                "Missing categorical targets require variable-type-specific imputation."
            )
        if target_types and target_types.get(feat) in {
            "categorical",
            "ordinal",
            "binary",
        }:
            raise ValueError(
                f"Column '{feat}' is declared as {target_types.get(feat)}. "
                "Recipe 12 implements Gaussian MICE strictly for continuous targets."
            )
        if pd.api.types.is_bool_dtype(df[feat]) or not pd.api.types.is_numeric_dtype(
            df[feat]
        ):
            raise ValueError(
                f"Column '{feat}' in features_to_impute is non-numeric ({df[feat].dtype}). "
                "Recipe 12 implements Gaussian MICE for continuous variables. "
                "Missing categorical targets require variable-type-specific imputation."
            )
        non_null_vals = df[feat].dropna().unique()
        vals_set = set(non_null_vals)
        if len(non_null_vals) <= 2:
            raise ValueError(
                f"Column '{feat}' in features_to_impute is binary/dummy ({vals_set}) with <= 2 unique observed values. "
                "Binary/dummy variables cannot be imputed using Gaussian linear regression MICE; "
                "Gaussian posterior draws produce fractional values and cannot enforce binary {0, 1} "
                "or group membership. Restrict features_to_impute to continuous variables."
            )
        if vals_set.issubset({1, 2, 3, 1.0, 2.0, 3.0}) or vals_set.issubset(
            {0, 1, 2, 0.0, 1.0, 2.0}
        ):
            raise ValueError(
                f"Column '{feat}' in features_to_impute has discrete integer category codes ({vals_set}). "
                "Categorical and ordinal variables must not be imputed via Gaussian linear regression MICE; "
                "they require variable-type-specific models."
            )

    # Validate conditioning columns, predictors, and outcome
    for c in cols:
        if df[c].isna().all():
            raise ValueError(
                f"Column '{c}' has 0 observed values (100% missing) and cannot be conditioned upon."
            )
        if pd.api.types.is_bool_dtype(df[c]) or not pd.api.types.is_numeric_dtype(
            df[c]
        ):
            raise ValueError(
                f"Column '{c}' is non-numeric ({df[c].dtype}). "
                "All features in MICE conditioning matrix must be numeric. "
                "Categorical predictors must be dummy-encoded before inclusion in 'predictors'."
            )
        obs_c = df[c].dropna().values
        if len(obs_c) > 0 and not np.all(np.isfinite(obs_c)):
            raise ValueError(
                f"Column '{c}' contains non-finite values (inf / -inf). "
                "Non-finite values must be cleaned prior to MICE."
            )

    if predictors:
        for pred in predictors:
            if pred not in features_to_impute and df[pred].isna().any():
                non_null_p = df[pred].dropna().unique()
                if len(non_null_p) <= 2:
                    raise ValueError(
                        f"Predictor '{pred}' is binary/dummy and contains missing values. "
                        "Gaussian MICE cannot impute categorical variables. "
                        "Categorical predictors must be fully observed to condition on in Gaussian MICE."
                    )
                raise ValueError(
                    f"Conditioning predictor '{pred}' contains missing values. "
                    "Conditioning-only predictors must be fully observed. "
                    "If this continuous predictor requires imputation, include it in 'features_to_impute'."
                )

    if outcome_col and df[outcome_col].isna().any():
        raise ValueError(
            f"Primary outcome column '{outcome_col}' contains missing values. "
            "Clinical governance invariant: Never impute the primary outcome variable. "
            "Rows with missing outcome must have an explicit cohort disposition (e.g. complete-case exclusion) prior to MICE."
        )

    # Validate bound ordering
    for feat in features_to_impute:
        low_val = (
            min_value[feat]
            if isinstance(min_value, dict) and feat in min_value
            else (min_value if isinstance(min_value, (int, float)) else None)
        )
        high_val = (
            max_value[feat]
            if isinstance(max_value, dict) and feat in max_value
            else (max_value if isinstance(max_value, (int, float)) else None)
        )
        if low_val is not None and high_val is not None and low_val >= high_val:
            raise ValueError(
                f"Invalid bound ordering for feature '{feat}': min_value ({low_val}) must be strictly less than max_value ({high_val})."
            )

    clean = df.copy()
    # Initialize missing entries with column mean
    for feat in features_to_impute:
        mis = clean[feat].isna()
        if mis.any():
            clean.loc[mis, feat] = df[feat].mean()

    for it in range(max_iter):
        for feat in features_to_impute:
            mis_mask = df[feat].isna()
            if not mis_mask.any():
                continue
            obs_mask = ~mis_mask
            pred_cols = [c for c in cols if c != feat]

            X_obs = np.column_stack(
                [np.ones(obs_mask.sum()), clean.loc[obs_mask, pred_cols].values]
            )
            y_obs = df.loc[obs_mask, feat].values
            X_mis = np.column_stack(
                [np.ones(mis_mask.sum()), clean.loc[mis_mask, pred_cols].values]
            )

            n_obs, p = X_obs.shape
            if n_obs <= p:
                raise ValueError(
                    f"Insufficient observed cases for feature '{feat}': n_obs ({n_obs}) must be strictly greater "
                    f"than number of parameters p ({p}) to form a valid Bayesian regression posterior."
                )
            rank_obs = np.linalg.matrix_rank(X_obs)
            if rank_obs < p:
                raise ValueError(
                    f"Design matrix for feature '{feat}' is rank-deficient (rank {rank_obs} < {p} parameters). "
                    "Collinear predictors or zero-variance columns must be removed before Bayesian regression MICE."
                )

            XtX = X_obs.T @ X_obs
            Xty = X_obs.T @ y_obs
            beta_hat = np.linalg.solve(XtX, Xty)
            residuals = y_obs - X_obs @ beta_hat
            rss = float(np.sum(residuals**2))
            if rss <= 1e-12:
                raise ValueError(
                    f"Degenerate residual sum of squares for feature '{feat}' (RSS <= 1e-12). "
                    "The target is perfectly predicted by conditioning features; "
                    "deterministic linear relationships cannot be modeled as continuous stochastic processes."
                )

            # Deterministic prediction (posterior mean) for exploratory profiling
            y_imp = X_mis @ beta_hat

            low_val = (
                min_value[feat]
                if isinstance(min_value, dict) and feat in min_value
                else (min_value if isinstance(min_value, (int, float)) else None)
            )
            high_val = (
                max_value[feat]
                if isinstance(max_value, dict) and feat in max_value
                else (max_value if isinstance(max_value, (int, float)) else None)
            )

            if low_val is not None:
                y_imp = np.maximum(y_imp, low_val)
            if high_val is not None:
                y_imp = np.minimum(y_imp, high_val)

            clean.loc[mis_mask, feat] = y_imp

    if outcome_col:
        clean[outcome_col] = df[outcome_col]

    return clean


# Backward-compatible alias for exploratory single imputation
impute_mice = impute_mice_single
```

---

## 13. Publication-Grade HTML Table Renderer (NEJM / JAMA Standards)

Renders any summary DataFrame into an ICMJE-compliant publication table with 3 horizontal rules (top border, header border, bottom border), zero vertical dividers, and standard clinical typography with full data escaping.

```python
import html
import pandas as pd


def render_publication_table(
    df: pd.DataFrame,
    title: str = "Table 1. Cohort Characteristics",
    style: str = "nejm",
    footnote: str | None = None,
) -> str:
    """Render pandas DataFrame into journal-compliant 3-rule publication HTML with escaped data."""
    border_color = "#000000" if style == "nejm" else "#333333"
    escaped_title = html.escape(str(title))
    table_html = f"""<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 20px 0;">
  <p style="font-weight: bold; font-size: 15px; margin-bottom: 8px;">{escaped_title}</p>
  <table style="border-collapse: collapse; width: 100%; border-top: 2px solid {border_color}; border-bottom: 2px solid {border_color}; font-size: 13px;">
    <thead>
      <tr style="border-bottom: 1px solid {border_color}; text-align: left;">
"""
    for col in df.columns:
        escaped_col = html.escape(str(col))
        table_html += f'        <th style="padding: 6px 10px; font-weight: 600;">{escaped_col}</th>\n'
    table_html += "      </tr>\n    </thead>\n    <tbody>\n"

    for _, row in df.iterrows():
        table_html += "      <tr>\n"
        for col in df.columns:
            val = str(row[col]) if row[col] is not None else ""
            escaped_val = html.escape(val)
            table_html += f'        <td style="padding: 5px 10px;">{escaped_val}</td>\n'
        table_html += "      </tr>\n"

    table_html += "    </tbody>\n  </table>\n"

    if footnote is None:
        footnote_text = "Data presented as mean (SD), median [IQR], or count (%). P-values calculated using two-sided tests."
    else:
        footnote_text = footnote

    if footnote_text:
        escaped_footnote = html.escape(str(footnote_text))
        table_html += f'  <p style="font-size: 11px; color: #666; margin-top: 6px;">{escaped_footnote}</p>\n'

    table_html += "</div>"
    return table_html
```

---

## 14. Inverse Probability of Treatment Weighting (IPTW) & Weighted Balance

Standalone implementation for causal inference via propensity score weighting supporting ATE, ATT, ATC, and Overlap Weights (ATO; Li et al., 2018). Computes stabilized weights, quantile truncation, Kish's Effective Sample Size (\$N_{\text{eff}}\$), Bessel-corrected weighted SMD, and Huber-White sandwich variance estimation.

```python
from typing import Any, Literal
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from lifelines import CoxPHFitter


def compute_iptw(
    df: pd.DataFrame,
    treatment_col: str,
    covariates: list[str],
    estimand: Literal["ATE", "ATT", "ATC", "ATO"] = "ATE",
    stabilized: bool = True,
    truncate_quantiles: tuple[float, float] | None = (0.01, 0.99),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute IPTW weights with Kish's Neff, truncation, and positivity diagnostics."""
    clean = df[[treatment_col] + covariates].dropna().copy()
    T = clean[treatment_col].to_numpy(dtype=float)

    # 1. Fit Propensity Score via Logistic Regression
    X = sm.add_constant(clean[covariates])
    ps_model = sm.Logit(T, X).fit(disp=False)
    ps = np.clip(ps_model.predict(X).to_numpy(dtype=float), 1e-4, 1.0 - 1e-4)
    clean["propensity_score"] = ps

    p_t = float(np.mean(T))
    estimand_u = estimand.upper()

    # 2. Raw Weights by Estimand
    if estimand_u == "ATE":
        if stabilized:
            w_raw = np.where(T == 1, p_t / ps, (1.0 - p_t) / (1.0 - ps))
        else:
            w_raw = np.where(T == 1, 1.0 / ps, 1.0 / (1.0 - ps))
    elif estimand_u == "ATT":
        w_raw = np.where(T == 1, 1.0, ps / (1.0 - ps))
    elif estimand_u == "ATC":
        w_raw = np.where(T == 1, (1.0 - ps) / ps, 1.0)
    elif estimand_u == "ATO":
        w_raw = np.where(T == 1, 1.0 - ps, ps)
    else:
        raise ValueError(f"Unsupported estimand: {estimand}")

    # 3. Truncation
    if truncate_quantiles is not None and estimand_u != "ATO":
        w_low, w_high = np.quantile(w_raw, truncate_quantiles[0]), np.quantile(w_raw, truncate_quantiles[1])
        w_final = np.clip(w_raw, w_low, w_high)
    else:
        w_final = w_raw.copy()

    clean["iptw_weight"] = w_final

    # 4. Diagnostics & Kish's Neff
    n_eff_treated = float((np.sum(w_final[T == 1]) ** 2) / np.sum(w_final[T == 1] ** 2))
    n_eff_control = float((np.sum(w_final[T == 0]) ** 2) / np.sum(w_final[T == 0] ** 2))
    n_eff_total = float((np.sum(w_final) ** 2) / np.sum(w_final ** 2))

    diagnostics = {
        "estimand": estimand_u,
        "stabilized": stabilized and estimand_u == "ATE",
        "n_eff": {"treated": round(n_eff_treated, 1), "control": round(n_eff_control, 1), "total": round(n_eff_total, 1)},
        "extreme_weights_gt_10": int(np.sum(w_final > 10.0)),
        "positivity_violations": int(np.sum((ps < 0.05) | (ps > 0.95))),
    }

    out_df = df.copy()
    out_df["propensity_score"] = clean["propensity_score"]
    out_df["iptw_weight"] = clean["iptw_weight"]
    return out_df, diagnostics


def calculate_weighted_smd(
    treated: np.ndarray,
    control: np.ndarray,
    weights_treated: np.ndarray,
    weights_control: np.ndarray,
) -> float:
    """Calculate Bessel-corrected weighted SMD for continuous variables."""
    t, c = np.asarray(treated, dtype=float), np.asarray(control, dtype=float)
    wt, wc = np.asarray(weights_treated, dtype=float), np.asarray(weights_control, dtype=float)

    mask_t = np.isfinite(t) & np.isfinite(wt) & (wt > 0)
    mask_c = np.isfinite(c) & np.isfinite(wc) & (wc > 0)
    t, wt = t[mask_t], wt[mask_t]
    c, wc = c[mask_c], wc[mask_c]

    if len(t) == 0 or len(c) == 0:
        return np.nan

    mean_t = float(np.sum(wt * t) / np.sum(wt))
    mean_c = float(np.sum(wc * c) / np.sum(wc))

    # Bessel-corrected weighted variance: s_w^2 = V1 / (V1^2 - V2) * sum(w_i * (x_i - mean_w)^2)
    v1_t, v2_t = np.sum(wt), np.sum(wt ** 2)
    var_t = (v1_t / (v1_t ** 2 - v2_t)) * np.sum(wt * (t - mean_t) ** 2) if (v1_t ** 2 - v2_t) > 0 else 0.0

    v1_c, v2_c = np.sum(wc), np.sum(wc ** 2)
    var_c = (v1_c / (v1_c ** 2 - v2_c)) * np.sum(wc * (c - mean_c) ** 2) if (v1_c ** 2 - v2_c) > 0 else 0.0

    pooled_sd = np.sqrt((var_t + var_c) / 2.0)
    if pooled_sd == 0:
        return 0.0 if mean_t == mean_c else np.nan
    return float((mean_t - mean_c) / pooled_sd)


def fit_weighted_outcome_model(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    weights_col: str = "iptw_weight",
    model_type: Literal["continuous", "binary", "survival"] = "continuous",
    duration_col: str | None = None,
) -> dict[str, Any]:
    """Fit weighted outcome model with robust sandwich standard errors (Huber-White)."""
    clean = df.dropna(subset=[outcome_col, treatment_col, weights_col]).copy()
    weights = clean[weights_col].to_numpy(dtype=float)
    T = clean[treatment_col].to_numpy(dtype=float)

    if model_type == "continuous":
        Y = clean[outcome_col].to_numpy(dtype=float)
        X = sm.add_constant(T)
        res = sm.WLS(Y, X, weights=weights).fit(cov_type="HC1")
        return {
            "model_type": "continuous",
            "effect": float(res.params[1]),
            "robust_se": float(res.bse[1]),
            "ci_lower": float(res.conf_int()[1][0]),
            "ci_upper": float(res.conf_int()[1][1]),
            "p_value": float(res.pvalues[1]),
        }
    elif model_type == "binary":
        Y = clean[outcome_col].to_numpy(dtype=float)
        X = sm.add_constant(T)
        res = sm.GLM(Y, X, family=sm.families.Binomial(), freq_weights=weights).fit(cov_type="HC1")
        or_val = float(np.exp(res.params[1]))
        ci = res.conf_int()[1]
        return {
            "model_type": "binary",
            "odds_ratio": or_val,
            "or_ci_lower": float(np.exp(ci[0])),
            "or_ci_upper": float(np.exp(ci[1])),
            "robust_se": float(res.bse[1]),
            "p_value": float(res.pvalues[1]),
        }
    elif model_type == "survival":
        if duration_col is None:
            raise ValueError("duration_col must be specified for survival outcome.")
        cph = CoxPHFitter()
        cph.fit(clean[[duration_col, outcome_col, treatment_col, weights_col]], duration_col=duration_col, event_col=outcome_col, weights_col=weights_col, robust=True)
        hr_val = float(cph.hazard_ratios_[treatment_col])
        ci_row = cph.confidence_intervals_.loc[treatment_col]
        return {
            "model_type": "survival",
            "hazard_ratio": hr_val,
            "hr_ci_lower": float(np.exp(ci_row.iloc[0])),
            "hr_ci_upper": float(np.exp(ci_row.iloc[1])),
            "robust_se": float(cph.standard_errors_[treatment_col]),
            "p_value": float(cph.summary.loc[treatment_col, "p"]),
        }
```

---

## 15. Competing Risks Survival Analysis (Aalen-Johansen CIF & Fine-Gray Model)

Standalone implementation for non-parametric Cumulative Incidence Function (Aalen-Johansen) with Naive \$1 - \text{Kaplan-Meier}\$ overestimation audit, Cause-Specific Cox regression (etiological rate ratio), and Fine-Gray Subdistribution Hazard model via IPCW counting process expansion (clinical prognosis).

```python
from typing import Any
import numpy as np
import pandas as pd
import scipy.stats as stats
from lifelines import AalenJohansenFitter, CoxPHFitter, KaplanMeierFitter


def fit_competing_risks_cif(
    durations: np.ndarray | pd.Series,
    event_observed: np.ndarray | pd.Series,
    event_of_interest: int = 1,
    competing_event: int = 2,
    horizons: list[float] | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Fit Aalen-Johansen CIF curves and audit Naive 1 - KM overestimation bias."""
    dur = np.asarray(durations, dtype=float)
    evt = np.asarray(event_observed, dtype=int)

    # 1. Fit Aalen-Johansen
    ajf1 = AalenJohansenFitter(calculate_variance=True)
    ajf1.fit(dur, evt, event_of_interest=event_of_interest)

    ajf2 = AalenJohansenFitter(calculate_variance=True)
    ajf2.fit(dur, evt, event_of_interest=competing_event)

    # 2. Fit Naive KM (censoring competing risks)
    km_naive = KaplanMeierFitter(alpha=alpha)
    km_naive.fit(dur, event_observed=(evt == event_of_interest).astype(int))

    def _lookup(series: pd.Series | pd.DataFrame, eval_t: np.ndarray) -> np.ndarray:
        s = series.iloc[:, 0] if isinstance(series, pd.DataFrame) else series
        idx = s.index.searchsorted(eval_t, side="right") - 1
        return np.where(idx >= 0, s.iloc[np.maximum(idx, 0)].to_numpy(dtype=float), 0.0)

    timeline = np.sort(np.unique(np.concatenate([[0.0], dur[evt > 0]])))
    cif1 = _lookup(ajf1.cumulative_density_, timeline)
    var1 = np.maximum(_lookup(ajf1.variance_, timeline), 0.0)
    cif2 = _lookup(ajf2.cumulative_density_, timeline)
    s_naive = _lookup(km_naive.survival_function_, timeline)
    naive = np.clip(1.0 - s_naive, 0.0, 1.0)

    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    cif1_l = np.clip(cif1 - z * np.sqrt(var1), 0.0, 1.0)
    cif1_u = np.clip(cif1 + z * np.sqrt(var1), 0.0, 1.0)

    cif_table = pd.DataFrame({
        "time": timeline,
        "cif_primary": cif1,
        "cif_primary_ci_lower": cif1_l,
        "cif_primary_ci_upper": cif1_u,
        "cif_competing": cif2,
        "naive_1_minus_km": naive,
        "overestimation_bias": np.maximum(naive - cif1, 0.0),
    })

    horizon_table = None
    if horizons:
        h_arr = np.asarray(horizons, dtype=float)
        h_cif1 = _lookup(ajf1.cumulative_density_, h_arr)
        h_var1 = np.maximum(_lookup(ajf1.variance_, h_arr), 0.0)
        h_naive = np.clip(1.0 - _lookup(km_naive.survival_function_, h_arr), 0.0, 1.0)
        h_rows = []
        for i, h in enumerate(h_arr):
            b_abs = float(h_naive[i] - h_cif1[i])
            b_rel = float(b_abs / h_cif1[i] * 100.0) if h_cif1[i] > 1e-4 else 0.0
            h_rows.append({
                "horizon": h,
                "cif_primary": round(float(h_cif1[i]), 4),
                "naive_1_minus_km": round(float(h_naive[i]), 4),
                "overestimation_abs": round(b_abs, 4),
                "overestimation_pct": round(b_rel, 2),
            })
        horizon_table = pd.DataFrame(h_rows)

    return {
        "ajf_primary": ajf1,
        "ajf_competing": ajf2,
        "km_naive": km_naive,
        "cif_table": cif_table,
        "horizon_summary": horizon_table,
    }


def fit_competing_risks_regression(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    covariates: list[str],
    event_of_interest: int = 1,
    competing_event: int = 2,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Fit dual Cause-Specific Cox and Fine-Gray Subdistribution Hazard models side by side."""
    clean = df[[duration_col, event_col] + covariates].dropna().copy()
    clean["_id"] = np.arange(len(clean))

    # 1. Cause-Specific Cox Model
    df_cs = clean.copy()
    df_cs["_cs_event"] = (df_cs[event_col] == event_of_interest).astype(int)
    cph_cs = CoxPHFitter(alpha=alpha)
    cph_cs.fit(df_cs[[duration_col, "_cs_event"] + covariates], duration_col=duration_col, event_col="_cs_event", robust=True)

    # 2. Fine-Gray Model via IPCW counting process expansion
    km_censor = KaplanMeierFitter()
    km_censor.fit(clean[duration_col], event_observed=(clean[event_col] == 0).astype(int))
    g_sf = km_censor.survival_function_

    fail_times = np.sort(clean.loc[clean[event_col] == event_of_interest, duration_col].unique())

    def _get_G(t: float) -> float:
        idx = g_sf.index.searchsorted(t, side="left") - 1
        return max(float(g_sf.iloc[idx, 0]) if idx >= 0 else 1.0, 1e-4)

    exp_rows = []
    for _, row in clean.iterrows():
        s_id = row["_id"]
        t_i = float(row[duration_col])
        e_i = int(row[event_col])
        cov_dict = {cov: row[cov] for cov in covariates}

        if e_i == event_of_interest:
            r = {"_id": s_id, "_start": 0.0, "_stop": t_i, "_event": 1, "_weight": 1.0, **cov_dict}
            exp_rows.append(r)
        elif e_i == 0:
            r = {"_id": s_id, "_start": 0.0, "_stop": t_i, "_event": 0, "_weight": 1.0, **cov_dict}
            exp_rows.append(r)
        elif e_i == competing_event:
            r0 = {"_id": s_id, "_start": 0.0, "_stop": t_i, "_event": 0, "_weight": 1.0, **cov_dict}
            exp_rows.append(r0)
            subseq = fail_times[fail_times > t_i]
            g_ti = _get_G(t_i)
            prev_t = t_i
            for tj in subseq:
                w_ij = max(_get_G(tj) / g_ti, 1e-4)
                exp_rows.append({"_id": s_id, "_start": prev_t, "_stop": tj, "_event": 0, "_weight": w_ij, **cov_dict})
                prev_t = tj

    cph_fg = CoxPHFitter(alpha=alpha)
    cph_fg.fit(pd.DataFrame(exp_rows), duration_col="_stop", entry_col="_start", event_col="_event", weights_col="_weight", cluster_col="_id", robust=True)

    # 3. Side-by-side comparison
    comp_rows = []
    for cov in covariates:
        cs_hr = float(cph_cs.hazard_ratios_[cov])
        cs_ci = cph_cs.confidence_intervals_.loc[cov]
        cs_p = float(cph_cs.summary.loc[cov, "p"])

        fg_shr = float(cph_fg.hazard_ratios_[cov])
        fg_ci = cph_fg.confidence_intervals_.loc[cov]
        fg_p = float(cph_fg.summary.loc[cov, "p"])

        comp_rows.append({
            "covariate": cov,
            "cause_specific_hr": round(cs_hr, 3),
            "cs_95_ci": f"{cs_hr:.3f} ({np.exp(cs_ci.iloc[0]):.3f} to {np.exp(cs_ci.iloc[1]):.3f})",
            "cs_p_value": cs_p,
            "fine_gray_shr": round(fg_shr, 3),
            "fg_95_ci": f"{fg_shr:.3f} ({np.exp(fg_ci.iloc[0]):.3f} to {np.exp(fg_ci.iloc[1]):.3f})",
            "fg_p_value": fg_p,
            "profile": "Divergent" if (cs_p < 0.05) != (fg_p < 0.05) else "Concordant",
        })

    return pd.DataFrame(comp_rows)
```

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
    y_t = y_t[valid].astype(int)
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

    is_comparison = False
    if scores_old is not None and scores_new is not None:
        is_comparison = True
        s_old = pd.Series(scores_old).to_numpy(dtype=float)
        s_new = pd.Series(scores_new).to_numpy(dtype=float)
        valid = ~(np.isnan(dur_arr) | np.isnan(ev_arr) | np.isnan(s_old) | np.isnan(s_new))
        dur_arr, ev_arr, s_old, s_new = dur_arr[valid], ev_arr[valid].astype(int), s_old[valid], s_new[valid]
        s_arr = s_new
    elif isinstance(scores, dict):
        is_comparison = True
        s_old = pd.Series(scores.get("old", scores.get("baseline"))).to_numpy(dtype=float)
        s_new = pd.Series(scores.get("new", scores.get("updated"))).to_numpy(dtype=float)
        valid = ~(np.isnan(dur_arr) | np.isnan(ev_arr) | np.isnan(s_old) | np.isnan(s_new))
        dur_arr, ev_arr, s_old, s_new = dur_arr[valid], ev_arr[valid].astype(int), s_old[valid], s_new[valid]
        s_arr = s_new
    else:
        if scores is None:
            raise ValueError("Must provide scores or both scores_old and scores_new.")
        s_arr = pd.Series(scores).to_numpy(dtype=float)
        valid = ~(np.isnan(dur_arr) | np.isnan(ev_arr) | np.isnan(s_arr))
        dur_arr, ev_arr, s_arr = dur_arr[valid], ev_arr[valid].astype(int), s_arr[valid]
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

    y_orig = sub_df[outcome].to_numpy(dtype=float)

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
        opt_mean = float(np.mean(optimism_records[k])) if optimism_records[k] else 0.0
        b_mean = float(np.mean(boot_records[k])) if boot_records[k] else app_val
        t_mean = float(np.mean(test_records[k])) if test_records[k] else app_val
        corrected = app_val - opt_mean
        if k in ("auroc", "nagelkerke_r2"):
            corrected = max(0.0, min(1.0, corrected))
        elif k == "brier_score":
            corrected = max(0.0, corrected)
        metrics_summary[k] = {
            "apparent": float(app_val),
            "boot_mean": float(b_mean),
            "test_mean": float(t_mean),
            "optimism": float(opt_mean),
            "optimism_corrected": float(corrected),
        }

    raw_s = float(np.mean(test_slopes)) if test_slopes else 1.0
    shrinkage_factor = float(np.clip(raw_s, 0.01, 1.0))

    # Shrunk coefficients
    orig_intercept = float(orig_params[0])
    orig_betas = [float(b) for b in orig_params[1:]]
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
    if shrinkage_factor < 0.80:
        warnings.append(
            f"Uniform shrinkage factor s_hat = {shrinkage_factor:.3f} (< 0.80) indicates substantial overfitting."
        )
    if n_successful < n_bootstrap * 0.5:
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
    try:
        fit_mod = sm.Logit(y_arr, X).fit(disp=False)
        alpha_log = float(fit_mod.params[0])
        beta_log = float(fit_mod.params[1])
        alpha_ci = [float(c) for c in fit_mod.conf_int()[0]]
        beta_ci = [float(c) for c in fit_mod.conf_int()[1]]
        p_log = expit(alpha_log + beta_log * lp_orig)
    except Exception:
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
        lp_recal = alpha_log + beta_log * lp_orig
        post_metrics = _calc_suite(y_arr, p_log, lp_recal)
        return {
            "method": "logistic",
            "n": n,
            "pre": pre_metrics,
            "post": post_metrics,
            "recalibration_parameters": {
                "method": "logistic",
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
            "parameters_intercept_only": {"alpha": alpha_inter, "beta": 1.0},
            "parameters_logistic": {"alpha": alpha_log, "beta": beta_log},
            "p_recalibrated": p_log,
            "p_recalibrated_intercept_only": p_inter,
            "p_recalibrated_logistic": p_log,
        }
```

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



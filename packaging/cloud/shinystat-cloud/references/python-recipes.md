# Standalone Python Biostatistical Recipes & Formulas (Cloud Sandbox Edition)

Reference implementations for standalone execution in standard data science environments (such as Claude Web Analysis Tool, Jupyter, Google Colab) where standard libraries (`pandas`, `numpy`, `scipy`, `statsmodels`, `lifelines`, `scikit-learn`) are available **without** requiring custom external packages (zero `medstat` imports).

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
    nonnormal_vars: list[str] | None = None
) -> pd.DataFrame:
    """Build stratified Table 1 with parametric/non-parametric tests and SMD."""
    nonnormal = set(nonnormal_vars or [])
    clean_df = df.dropna(subset=[strata]).copy()
    groups = sorted(clean_df[strata].unique())
    rows = []
    
    # Unified group column mapping to prevent count vs measurement column divergence
    col_labels = {g: f"{strata}={g} (N = {int(np.sum(clean_df[strata] == g))})" for g in groups}

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
        group_vals = [clean_df.loc[clean_df[strata] == g, var].dropna().to_numpy(dtype=float) for g in groups]
        
        if var in nonnormal:
            for g, vals in zip(groups, group_vals):
                if len(vals) > 0:
                    med = np.median(vals)
                    q25, q75 = np.percentile(vals, [25, 75])
                    row[col_labels[g]] = f"{med:.1f} [{q25:.1f}, {q75:.1f}]"
                else:
                    row[col_labels[g]] = "-"
            try:
                pval = stats.kruskal(*group_vals).pvalue if len(groups) > 1 and all(len(v) for v in group_vals) else np.nan
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
                pval = stats.f_oneway(*group_vals).pvalue if len(groups) > 1 and all(len(v) for v in group_vals) else np.nan
            except Exception:
                pval = np.nan
                
        row["P-value"] = f"{pval:.3f}" if (not np.isnan(pval) and pval >= 0.001) else ("<0.001" if not np.isnan(pval) else "")
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
        header_text = f"**{var}**" + (f" (Missing: {missing_cnt})" if missing_cnt > 0 else "")
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

        p_str = f"{pval:.3f}" if (not np.isnan(pval) and pval >= 0.001) else ("<0.001" if not np.isnan(pval) else "")

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
    return (float(max(0.0, (center - half_width) / denom)), float(min(1.0, (center + half_width) / denom)))

def calculate_2x2_metrics(tp: int, fp: int, fn: int, tn: int, ci: float = 0.95) -> dict:
    """Calculate sensitivity, specificity, PPV, NPV, LR+, LR-, and DOR with complete 95% CIs."""
    for count_val, name in [(tp, "tp"), (fp, "fp"), (fn, "fn"), (tn, "tn")]:
        if int(count_val) < 0:
            raise ValueError(f"Contingency cell count '{name}' must be non-negative (>= 0).")
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
    se_log_lr_pos = np.sqrt(1.0 / tp_a - 1.0 / (tp_a + fn_a) + 1.0 / fp_a - 1.0 / (fp_a + tn_a))
    lr_pos_ci = (float(np.exp(np.log(lr_pos) - z * se_log_lr_pos)), float(np.exp(np.log(lr_pos) + z * se_log_lr_pos)))

    lr_neg = (1.0 - sens_a) / spec_a if spec_a > 0 else np.nan
    se_log_lr_neg = np.sqrt(1.0 / fn_a - 1.0 / (tp_a + fn_a) + 1.0 / tn_a - 1.0 / (fp_a + tn_a))
    lr_neg_ci = (float(np.exp(np.log(lr_neg) - z * se_log_lr_neg)), float(np.exp(np.log(lr_neg) + z * se_log_lr_neg)))

    # Diagnostic Odds Ratio (DOR) with Woolf log-scale CI
    dor = (tp_a * tn_a) / (fp_a * fn_a) if (fp_a * fn_a) > 0 else np.nan
    se_log_dor = np.sqrt(1.0 / tp_a + 1.0 / fp_a + 1.0 / fn_a + 1.0 / tn_a)
    dor_ci = (float(np.exp(np.log(dor) - z * se_log_dor)), float(np.exp(np.log(dor) + z * se_log_dor)))

    return {
        "Sensitivity": (float(sens), calculate_ci_wilson_score(tp, n_pos, ci)),
        "Specificity": (float(spec), calculate_ci_wilson_score(tn, n_neg, ci)),
        "PPV": (float(ppv), calculate_ci_wilson_score(tp, tp + fp, ci)),
        "NPV": (float(npv), calculate_ci_wilson_score(tn, tn + fn, ci)),
        "LR_pos": (float(lr_pos), lr_pos_ci),
        "LR_neg": (float(lr_neg), lr_neg_ci),
        "DOR": (float(dor), dor_ci),
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

def auc_ci_delong(y_true, y_score, direction: str = "high", alpha: float = 0.05) -> dict:
    """Compute empirical AUC and DeLong analytical 95% confidence interval with directionality."""
    y_raw = np.asarray(y_true, dtype=float)
    s_raw = np.asarray(y_score, dtype=float)
    if len(y_raw) != len(s_raw):
        raise ValueError(f"Length mismatch: y_true ({len(y_raw)}) and y_score ({len(s_raw)}) must have identical lengths.")
    valid = ~(np.isnan(y_raw) | np.isnan(s_raw))
    y_valid = y_raw[valid]
    if not np.all(np.isin(y_valid, [0, 1])):
        raise ValueError("Binary outcome y_true must contain strictly binary values (0 and 1).")
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
    }

def delong_paired_test(y_true, score1, score2, alpha: float = 0.05) -> dict:
    """Paired DeLong comparative test between two correlated ROC curves on the same cohort."""
    y_raw = np.asarray(y_true, dtype=float)
    s1_raw = np.asarray(score1, dtype=float)
    s2_raw = np.asarray(score2, dtype=float)
    if len(y_raw) != len(s1_raw) or len(y_raw) != len(s2_raw):
        raise ValueError(f"Length mismatch: y_true ({len(y_raw)}), score1 ({len(s1_raw)}), and score2 ({len(s2_raw)}) must have identical lengths.")
    valid = ~(np.isnan(y_raw) | np.isnan(s1_raw) | np.isnan(s2_raw))
    y_valid = y_raw[valid]
    if not np.all(np.isin(y_valid, [0, 1])):
        raise ValueError("Binary outcome y_true must contain strictly binary values (0 and 1).")
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
    cov12 = (np.cov(v10_1, v10_2, ddof=1)[0, 1] / n_pos) + (np.cov(v01_1, v01_2, ddof=1)[0, 1] / n_neg)

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
        raise ValueError(f"Length mismatch: y_true ({len(y_raw)}) and y_pred ({len(p_raw)}) must have identical lengths.")
    valid = ~(np.isnan(y_raw) | np.isnan(p_raw))
    y_valid = y_raw[valid]
    if not np.all(np.isin(y_valid, [0, 1])):
        raise ValueError("Binary outcome y_true must contain strictly binary values (0 and 1).")
    p_valid = p_raw[valid]
    if (p_valid < 0.0).any() or (p_valid > 1.0).any():
        raise ValueError("Predicted probabilities y_pred must be in the range [0, 1].")
    y_t = y_valid.astype(int)
    y_p = np.clip(p_valid, 1e-6, 1.0 - 1e-6)

    # 1. Brier score
    brier = float(np.mean((y_t - y_p) ** 2))
    prev = float(np.mean(y_t))
    scaled_brier = 1.0 - (brier / (prev * (1.0 - prev))) if (0.0 < prev < 1.0) else np.nan

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

    return {
        "brier_score": brier,
        "scaled_brier": float(scaled_brier),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "ici": float(np.mean(abs_err)),
        "e50": float(np.median(abs_err)),
        "e90": float(np.percentile(abs_err, 90)),
        "emax": float(np.max(abs_err)),
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
        raise ValueError(f"Length mismatch: y_true ({len(y_raw)}) and y_pred ({len(p_raw)}) must have identical lengths.")
    valid = ~(np.isnan(y_raw) | np.isnan(p_raw))
    y_valid = y_raw[valid]
    if not np.all(np.isin(y_valid, [0, 1])):
        raise ValueError("Binary outcome y_true must contain strictly binary values (0 and 1).")
    y_t = y_valid.astype(int)
    y_p = p_raw[valid]

    # Map raw biomarker to probability if not in [0, 1]
    if (y_p < 0.0).any() or (y_p > 1.0).any():
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
        records.append({"threshold": pt, "net_benefit": float(nb_model), "strategy": "Model"})
        records.append({"threshold": pt, "net_benefit": float(nb_all), "strategy": "Treat All"})
        records.append({"threshold": pt, "net_benefit": 0.0, "strategy": "Treat None"})

    return pd.DataFrame(records)
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
        raise ValueError(f"Length mismatch: m1 ({len(v1)}) and m2 ({len(v2)}) must have identical lengths.")
    valid = ~(np.isnan(v1) | np.isnan(v2))
    diffs = v1[valid] - v2[valid]
    n = len(diffs)
    if n < 2:
        raise ValueError("Bland-Altman difference analysis requires at least 2 valid paired measurements.")

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
    }

def calculate_icc_matrix(values: np.ndarray, alpha: float = 0.05) -> pd.DataFrame:
    """Compute all 6 Shrout & Fleiss (1979) ICC variants with exact F-distribution 95% CIs."""
    mat = np.asarray(values, dtype=float)
    if np.isnan(mat).any():
        raise ValueError("ICC matrix contains missing values. Impute or subset complete cases.")
    n, k = mat.shape
    if n < 2 or k < 2:
        raise ValueError("ICC requires at least 2 targets and 2 raters.")

    grand_mean = np.mean(mat)
    # Check all-constant matrix: target variance is zero, reliability undefined
    if np.all(mat == grand_mean):
        return pd.DataFrame([
            {"Type": t, "Description": d, "ICC": np.nan, "CI_lower": np.nan, "CI_upper": np.nan, "F": np.nan, "pval": np.nan}
            for t, d in [
                ("ICC1", "One-way random (single)"),
                ("ICC2", "Two-way random agreement"),
                ("ICC3", "Two-way mixed consistency"),
                ("ICC1k", "One-way random (average)"),
                ("ICC2k", "Two-way random agreement (avg)"),
                ("ICC3k", "Two-way mixed consistency (avg)"),
            ]
        ])

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
        icc1 = (MSB - MSW) / (MSB + (k - 1) * MSW) if (MSB + (k - 1) * MSW) != 0 else np.nan
        icc1k = (MSB - MSW) / MSB if MSB != 0 else np.nan
        f_obs_1 = MSB / MSW if MSW > 0 else np.nan
        p_1 = float(stats.f.sf(f_obs_1, df_B, df_W)) if not np.isnan(f_obs_1) else np.nan
        f1_l = stats.f.ppf(1.0 - alpha / 2.0, df_B, df_W)
        f1_u = stats.f.ppf(1.0 - alpha / 2.0, df_W, df_B)
        fl1 = f_obs_1 / f1_l if (not np.isnan(f_obs_1) and f1_l > 0) else np.nan
        fu1 = f_obs_1 * f1_u if not np.isnan(f_obs_1) else np.nan
        ci_icc1 = ((fl1 - 1.0) / (fl1 + k - 1.0) if not np.isnan(fl1) else np.nan, (fu1 - 1.0) / (fu1 + k - 1.0) if not np.isnan(fu1) else np.nan)
        ci_icc1k = (1.0 - 1.0 / fl1 if (not np.isnan(fl1) and fl1 > 0) else np.nan, 1.0 - 1.0 / fu1 if (not np.isnan(fu1) and fu1 > 0) else np.nan)

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
        p_23 = float(stats.f.sf(f_obs_23, df_B, df_E)) if not np.isnan(f_obs_23) else np.nan
        f3_l = stats.f.ppf(1.0 - alpha / 2.0, df_B, df_E)
        f3_u = stats.f.ppf(1.0 - alpha / 2.0, df_E, df_B)
        fl3 = f_obs_23 / f3_l if (not np.isnan(f_obs_23) and f3_l > 0) else np.nan
        fu3 = f_obs_23 * f3_u if not np.isnan(f_obs_23) else np.nan
        ci_icc3 = ((fl3 - 1.0) / (fl3 + k - 1.0) if not np.isnan(fl3) else np.nan, (fu3 - 1.0) / (fu3 + k - 1.0) if not np.isnan(fu3) else np.nan)
        ci_icc3k = (1.0 - 1.0 / fl3 if (not np.isnan(fl3) and fl3 > 0) else np.nan, 1.0 - 1.0 / fu3 if (not np.isnan(fu3) and fu3 > 0) else np.nan)

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
        ci_icc2k = ((k * L2) / (1.0 + (k - 1) * L2) if (1.0 + (k - 1) * L2) != 0 else np.nan,
                    (k * U2) / (1.0 + (k - 1) * U2) if (1.0 + (k - 1) * U2) != 0 else np.nan)

    return pd.DataFrame([
        {"Type": "ICC1", "Description": "One-way random (single)", "ICC": icc1, "CI_lower": ci_icc1[0], "CI_upper": ci_icc1[1], "F": f_obs_1, "pval": p_1},
        {"Type": "ICC2", "Description": "Two-way random agreement", "ICC": icc2, "CI_lower": ci_icc2[0], "CI_upper": ci_icc2[1], "F": f_obs_23, "pval": p_23},
        {"Type": "ICC3", "Description": "Two-way mixed consistency", "ICC": icc3, "CI_lower": ci_icc3[0], "CI_upper": ci_icc3[1], "F": f_obs_23, "pval": p_23},
        {"Type": "ICC1k", "Description": "One-way random (average)", "ICC": icc1k, "CI_lower": ci_icc1k[0], "CI_upper": ci_icc1k[1], "F": f_obs_1, "pval": p_1},
        {"Type": "ICC2k", "Description": "Two-way random agreement (avg)", "ICC": icc2k, "CI_lower": ci_icc2k[0], "CI_upper": ci_icc2k[1], "F": f_obs_23, "pval": p_23},
        {"Type": "ICC3k", "Description": "Two-way mixed consistency (avg)", "ICC": icc3k, "CI_lower": ci_icc3k[0], "CI_upper": ci_icc3k[1], "F": f_obs_23, "pval": p_23},
    ])
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
    caliper_sd: float = 0.2
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
    clean = df.dropna(subset=[treatment_col] + confounders).copy().reset_index(drop=True)
    n_analyzed = len(clean)
    n_excluded = n_initial - n_analyzed

    # Validate treatment column strictly binary
    treatment_vals = clean[treatment_col].to_numpy()
    if not np.all(np.isin(treatment_vals, [0, 1])):
        raise ValueError(f"Treatment column '{treatment_col}' must contain strictly binary values (0 and 1).")
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
        matched_X = pd.get_dummies(matched_clean[confounders], drop_first=True, dtype=float)
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

        balance_rows.append({
            "Variable": var,
            "Pre-Match SMD": smd_pre,
            "Post-Match SMD": smd_post,
            "Balanced (<0.10)": abs(smd_post) < 0.10 if not np.isnan(smd_post) else False
        })

    return {
        "matched_df": matched_df,
        "balance_df": pd.DataFrame(balance_rows),
        "n_matched_pairs": pair_counter - 1,
        "n_unmatched_treated": len(treated) - (pair_counter - 1),
        "retention_audit": {
            "n_initial": n_initial,
            "n_excluded": n_excluded,
            "n_analyzed": n_analyzed,
            "exclusion_reasons": f"Missing values in {[treatment_col] + confounders}" if n_excluded > 0 else "None",
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
    ll = np.sum(y * np.log(np.clip(pi, 1e-15, 1.0)) + (1.0 - y) * np.log(np.clip(1.0 - pi, 1e-15, 1.0)))
    return float(ll + 0.5 * logdet)

def fit_firth_logistic(
    X: np.ndarray,
    y: np.ndarray,
    fit_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-6
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
        raise ValueError(f"Design matrix is rank-deficient (rank {rank} < {p} parameters). Collinear features must be removed.")

    beta = np.zeros(p)
    current_pll = _firth_penalized_loglik(beta, X_arr, y_arr)

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

        # Both parameter movement and score gradient must satisfy convergence
        if param_step < tol and score_norm < (tol * max(1.0, float(n))):
            converged = True
            break

    se = np.sqrt(np.diag(I_inv))

    # Profile Likelihood Confidence Intervals (chi2 cutoff = 3.841) with explicit fallback tracking
    ci_lower = np.zeros(p)
    ci_upper = np.zeros(p)
    ci_method = ["profile"] * p
    target_pll = current_pll - 0.5 * stats.chi2.ppf(0.95, df=1)

    for j in range(p):
        other_idx = [idx for idx in range(p) if idx != j]
        def profile_obj(b_j):
            if len(other_idx) == 0:
                return _firth_penalized_loglik(np.array([b_j]), X_arr, y_arr) - target_pll
            def nuisance_loss(nuis):
                b_full = np.zeros(p)
                b_full[j] = b_j
                b_full[other_idx] = nuis
                return -_firth_penalized_loglik(b_full, X_arr, y_arr)
            res = optimize.minimize(nuisance_loss, beta[other_idx], method="Nelder-Mead")
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

    return {
        "coefficients": beta,
        "standard_errors": se,
        "odds_ratios": np.exp(beta),
        "ci_lower": np.exp(ci_lower),
        "ci_upper": np.exp(ci_upper),
        "converged": converged,
        "ci_method": ci_method,
    }
```

---

## 8. Little's (1988) MCAR Multivariate Missingness Test

Pattern-grouped Expectation-Maximization (EM) ML parameter estimation under multivariate normality with Little's $d^2$ distance statistic: $d^2 = \sum_s n_s (\bar{y}_{s, obs} - \hat{\mu}_{obs})^T \hat{\Sigma}_{obs, obs}^{-1} (\bar{y}_{s, obs} - \hat{\mu}_{obs})$, with degrees of freedom $df = \sum p_s - P$. Tests the null hypothesis $H_0: \text{MCAR}$. Note that a non-significant test does not establish MAR or distinguish MAR from MNAR. Ridge covariance adjustment represents a numerical regularization approximation.

```python
import numpy as np
import pandas as pd
import scipy.stats as stats

def littles_mcar_test(df: pd.DataFrame, variables: list[str], max_iter: int = 100, tol: float = 1e-5) -> dict:
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
        raise ValueError("Dataset is completely missing (100% NaN); Little's MCAR test cannot be computed.")

    for j, col_name in enumerate(variables):
        if nan_mask[:, j].all():
            raise ValueError(f"Variable '{col_name}' has 0 observed values (100% missing); MCAR cannot be tested.")

    # Drop rows that are completely missing across all tested variables
    row_all_nan = nan_mask.all(axis=1)
    n_excluded_rows = int(np.sum(row_all_nan))
    if n_excluded_rows > 0:
        data = data[~row_all_nan]
        nan_mask = nan_mask[~row_all_nan]
        N = len(data)

    if N == 0:
        raise ValueError("No rows with observed values remain; Little's MCAR test cannot be computed.")

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
        parsed_patterns.append({
            "indices": np.array(row_indices, dtype=int),
            "O_s": O_s,
            "M_s": M_s,
            "p_s": p_s,
            "n_s": n_s,
        })

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
            "note": "Untestable: degrees of freedom <= 0 (insufficient overlap across missingness patterns)."
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
                    W = np.linalg.solve(Sigma_OO + np.eye(len(O_s)) * scale_ridge, Sigma_MO.T).T
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
    return {
        "chi2": float(d2),
        "df": int(df_stat),
        "p_value": float(p_val),
        "mcar_rejected": bool(p_val < 0.05),
        "n_patterns": len(parsed_patterns),
        "converged": converged,
        "n_iterations": n_iterations,
        "n_excluded_rows": n_excluded_rows,
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
    rare_outcome: bool = False
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
        raise ValueError(f"lower confidence limit ({low}) cannot exceed upper confidence limit ({up}).")

    # Conversion to Risk Ratio (RR) scale
    if estimate_type == "OR":
        if not rare_outcome:
            # Common outcome approximation per Ding & VanderWeele (2016)
            est = np.sqrt(est)
            if low is not None and low > 0: low = np.sqrt(low)
            if up is not None and up > 0: up = np.sqrt(up)
    elif estimate_type == "HR":
        if not rare_outcome:
            # VanderWeele & Ding (2017) HR conversion for common outcome
            def hr_to_rr(h):
                return (1.0 - 0.5**np.sqrt(h)) / (1.0 - 0.5**np.sqrt(1.0 / h)) if h != 1.0 else 1.0
            est = hr_to_rr(est)
            if low is not None and low > 0: low = hr_to_rr(low)
            if up is not None and up > 0: up = hr_to_rr(up)

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
    df: pd.DataFrame,
    outcome: str,
    covariates: list[str]
) -> pd.DataFrame:
    """Generate publication-ready Logistic Regression table with Crude & Adjusted ORs."""
    n_initial = len(df)
    clean = df.dropna(subset=[outcome] + covariates).copy()
    n_analyzed = len(clean)
    n_excluded = n_initial - n_analyzed

    outcome_vals = clean[outcome].to_numpy()
    if not np.all(np.isin(outcome_vals, [0, 1])):
        raise ValueError(f"Outcome column '{outcome}' must contain strictly binary values (0 and 1).")
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
        adj_str = f"{np.exp(adj_b):.2f} ({np.exp(adj_ci[0]):.2f}–{np.exp(adj_ci[1]):.2f})"
        
        p_val_str = f"{adj_p:.3f}" if adj_p >= 0.001 else "<0.001"
        
        rows.append({
            "Variable": var,
            "Crude OR (95% CI)": crude_str,
            "Adjusted OR (95% CI)": adj_str,
            "P-value": p_val_str
        })
        
    res_df = pd.DataFrame(rows)
    res_df.attrs["retention_audit"] = {
        "n_initial": n_initial,
        "n_excluded": n_excluded,
        "n_analyzed": n_analyzed,
        "exclusion_reasons": f"Missing values in {[outcome] + covariates}" if n_excluded > 0 else "None",
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
from lifelines.statistics import logrank_test, multivariate_logrank_test, proportional_hazard_test

def fit_survival_analysis_suite(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    strata_col: str | None = None,
    covariates: list[str] | None = None
) -> dict:
    """Run Kaplan-Meier, Log-Rank, and Cox PH modeling with per-covariate Schoenfeld test."""
    n_initial = len(df)
    
    # 1. Unadjusted Kaplan-Meier cohort: subset duration, event, and strata (preserves KM cohort)
    km_cols = [duration_col, event_col] + ([strata_col] if strata_col else [])
    clean_km = df.dropna(subset=km_cols).copy()

    # Validate event indicator strictly binary and duration non-negative
    ev_km = clean_km[event_col].to_numpy()
    if not np.all(np.isin(ev_km, [0, 1])):
        raise ValueError(f"Event indicator '{event_col}' must contain strictly binary values (0 = Censored, 1 = Event).")
    dur_km = clean_km[duration_col].to_numpy()
    if (dur_km < 0).any():
        raise ValueError(f"Duration column '{duration_col}' must contain non-negative values (>= 0).")

    # Kaplan-Meier curve & median survival
    km_results = {}
    if strata_col:
        groups = clean_km[strata_col].unique()
        for g in groups:
            sub = clean_km[clean_km[strata_col] == g]
            kmf = KaplanMeierFitter()
            kmf.fit(sub[duration_col], sub[event_col], label=str(g))
            km_results[f"Median Survival ({g})"] = float(kmf.median_survival_time_)
        
        # Log-rank test (2 groups or multivariate for >2 groups)
        if len(groups) == 2:
            lr_res = logrank_test(
                clean_km.loc[clean_km[strata_col] == groups[0], duration_col],
                clean_km.loc[clean_km[strata_col] == groups[1], duration_col],
                clean_km.loc[clean_km[strata_col] == groups[0], event_col],
                clean_km.loc[clean_km[strata_col] == groups[1], event_col]
            )
            km_results["Log-Rank P-value"] = float(lr_res.p_value)
        elif len(groups) > 2:
            lr_res = multivariate_logrank_test(clean_km[duration_col], clean_km[strata_col], clean_km[event_col])
            km_results["Multivariate Log-Rank P-value"] = float(lr_res.p_value)
    else:
        kmf = KaplanMeierFitter()
        kmf.fit(clean_km[duration_col], clean_km[event_col])
        km_results["Overall Median Survival"] = float(kmf.median_survival_time_)

    # 2. Cox Proportional Hazards model on complete-covariate cohort
    cox_table = pd.DataFrame()
    schoenfeld_summary = {}
    n_analyzed_cox = 0
    if covariates:
        cox_cols = [duration_col, event_col] + covariates
        clean_cox = df.dropna(subset=cox_cols).copy()
        n_analyzed_cox = len(clean_cox)
        ev_cox = clean_cox[event_col].to_numpy()
        if not np.all(np.isin(ev_cox, [0, 1])):
            raise ValueError(f"Event indicator '{event_col}' must contain strictly binary values.")

        cph = CoxPHFitter()
        cox_data = clean_cox[cox_cols]
        cph.fit(cox_data, duration_col=duration_col, event_col=event_col)
        
        summary = cph.summary
        rows = []
        for var in covariates:
            hr = float(summary.loc[var, "exp(coef)"])
            ci_l = float(summary.loc[var, "exp(coef) lower 95%"])
            ci_u = float(summary.loc[var, "exp(coef) upper 95%"])
            p_val = float(summary.loc[var, "p"])
            rows.append({
                "Covariate": var,
                "Hazard Ratio (95% CI)": f"{hr:.2f} ({ci_l:.2f}–{ci_u:.2f})",
                "P-value": f"{p_val:.3f}" if p_val >= 0.001 else "<0.001"
            })
        cox_table = pd.DataFrame(rows)

        # Proportional hazards test per covariate
        try:
            prop_test = proportional_hazard_test(cph, cox_data, time_transform="rank")
            schoenfeld_summary["p_values"] = {k: float(v) for k, v in prop_test.summary["p"].items()}
            schoenfeld_summary["test_statistics"] = {k: float(v) for k, v in prop_test.summary["test_statistic"].items()}
            passed = bool((prop_test.summary["p"] > 0.05).all())
            schoenfeld_summary["no_ph_violation_detected"] = passed
            schoenfeld_summary["PH_Assumptions_Passed"] = passed  # backward compatibility
            schoenfeld_summary["PH_Note"] = "No proportional hazards violation detected (p > 0.05)." if passed else "Potential PH assumption violation detected (p <= 0.05)."
        except Exception as e:
            schoenfeld_summary["PH_Note"] = str(e)

    return {
        "km_summary": km_results,
        "cox_table": cox_table,
        "schoenfeld_diagnostics": schoenfeld_summary,
        "retention_km": {"n_initial": n_initial, "n_analyzed": len(clean_km), "n_excluded": n_initial - len(clean_km)},
        "retention_cox": {"n_initial": n_initial, "n_analyzed": n_analyzed_cox, "n_excluded": n_initial - n_analyzed_cox} if covariates else None,
    }
```

---

## 12. Multiple Imputation by Chained Equations (MICE) & Rubin's Rules

Provides multiple stochastic imputations ($M \ge 5$) via Scikit-Learn `IterativeImputer` with Bayesian Ridge regression, strictly enforcing that **primary outcomes are never imputed**. Configured for continuous/numeric clinical features with optional boundary constraints (`min_value`, `max_value`). Categorical confounders must be dummy-encoded before imputation. Includes pure-Python **Rubin's rules pooling** (Rubin 1987; Barnard & Rubin 1999) with finite-sample degrees of freedom adjustment. Ratio effect measures (OR, HR) must be pooled on the log scale and exponentiated.

```python
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

def impute_mice_datasets(
    df: pd.DataFrame,
    features_to_impute: list[str],
    outcome_col: str | None = None,
    predictors: list[str] | None = None,
    m: int = 5,
    max_iter: int = 10,
    random_state: int = 42,
    min_value: float | dict[str, float] | None = None,
    max_value: float | dict[str, float] | None = None
) -> list[pd.DataFrame]:
    """Generate M stochastic imputed datasets for valid inferential analysis under MAR.
    
    Clinical Governance Invariants:
    1. Never impute the primary outcome variable (its original values remain untouched).
    2. Primary outcome and additional covariates MUST be included as predictors in the
       imputation model per standard biostatistical methodology (van Buuren; Moons et al. 2006)
       to prevent severe attenuation towards the null.
    3. Continuous/numeric feature scope: Categorical features must be dummy-encoded before imputation.
    4. Observed values of features_to_impute are preserved intact; only missing values are imputed.
    """
    if m < 1:
        raise ValueError(f"Number of imputations m must be >= 1 (got {m}).")

    if outcome_col and outcome_col in features_to_impute:
        raise ValueError("Clinical governance invariant: Never impute the primary outcome variable!")

    cols = list(dict.fromkeys(features_to_impute + (predictors or []) + ([outcome_col] if outcome_col else [])))
    if len(cols) < 2:
        raise ValueError(
            "MICE requires >= 1 additional predictor (or outcome variable) to condition on; "
            "otherwise chained imputation degenerates to unconditional mean imputation."
        )

    # Validate feature types and missingness
    for c in cols:
        if df[c].isna().all():
            raise ValueError(f"Column '{c}' has 0 observed values (100% missing) and cannot be imputed or conditioned upon.")
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(
                f"Column '{c}' is non-numeric ({df[c].dtype}). "
                "Recipe 12 implements Gaussian MICE for continuous variables. "
                "Categorical features must be numerically/dummy encoded before imputation."
            )

    # Build min/max bounds arrays for IterativeImputer
    if isinstance(min_value, dict):
        min_arr = np.array([min_value.get(c, -np.inf) for c in cols])
    elif min_value is not None:
        min_arr = min_value
    else:
        min_arr = -np.inf

    if isinstance(max_value, dict):
        max_arr = np.array([max_value.get(c, np.inf) for c in cols])
    elif max_value is not None:
        max_arr = max_value
    else:
        max_arr = np.inf

    datasets = []
    for i in range(m):
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=max_iter,
            random_state=random_state + i * 100,
            sample_posterior=True,
            min_value=min_arr,
            max_value=max_arr
        )
        clean = df.copy()
        filled = pd.DataFrame(
            imputer.fit_transform(clean[cols].astype(float)),
            columns=cols,
            index=clean.index
        )
        # Preserve observed values strictly, impute only missing entries
        for feat in features_to_impute:
            missing_mask = clean[feat].isna()
            clean.loc[missing_mask, feat] = filled.loc[missing_mask, feat]

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
    alpha: float = 0.05
) -> dict:
    """Pool point estimates and standard errors across M imputations using Rubin's rules.
    
    Clinical & Statistical Invariant:
    When pooling ratio effect estimates (such as Odds Ratios or Hazard Ratios), pooling
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
        raise ValueError("point_estimates and standard_errors must contain strictly finite values (no NaN or inf).")

    if (ses < 0.0).any():
        raise ValueError("standard_errors must be strictly non-negative (>= 0).")

    if n_obs is not None and n_obs <= k_params:
        raise ValueError(f"n_obs ({n_obs}) must be strictly greater than k_params ({k_params}).")

    V = ses ** 2
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
        fmi = float((r + 2.0 / (df + 3.0)) / (r + 1.0)) if np.isfinite(df) else float(r / (r + 1.0))
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
    max_value: float | dict[str, float] | None = None
) -> pd.DataFrame:
    """Deterministic single MICE imputation strictly for rapid exploratory data health profiling."""
    if outcome_col and outcome_col in features_to_impute:
        raise ValueError("Clinical governance invariant: Never impute the primary outcome variable!")

    cols = list(dict.fromkeys(features_to_impute + (predictors or []) + ([outcome_col] if outcome_col else [])))
    if len(cols) < 2:
        raise ValueError(
            "MICE requires >= 1 additional predictor (or outcome variable) to condition on; "
            "otherwise chained imputation degenerates to unconditional mean imputation."
        )

    for c in cols:
        if df[c].isna().all():
            raise ValueError(f"Column '{c}' has 0 observed values (100% missing) and cannot be imputed or conditioned upon.")
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(
                f"Column '{c}' is non-numeric ({df[c].dtype}). "
                "Recipe 12 implements Gaussian MICE for continuous variables. "
                "Categorical features must be numerically/dummy encoded before imputation."
            )

    if isinstance(min_value, dict):
        min_arr = np.array([min_value.get(c, -np.inf) for c in cols])
    elif min_value is not None:
        min_arr = min_value
    else:
        min_arr = -np.inf

    if isinstance(max_value, dict):
        max_arr = np.array([max_value.get(c, np.inf) for c in cols])
    elif max_value is not None:
        max_arr = max_value
    else:
        max_arr = np.inf

    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=max_iter,
        random_state=random_state,
        sample_posterior=False,
        min_value=min_arr,
        max_value=max_arr
    )
    clean = df.copy()
    filled = pd.DataFrame(
        imputer.fit_transform(clean[cols].astype(float)),
        columns=cols,
        index=clean.index
    )
    for feat in features_to_impute:
        missing_mask = clean[feat].isna()
        clean.loc[missing_mask, feat] = filled.loc[missing_mask, feat]

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
    footnote: str | None = None
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

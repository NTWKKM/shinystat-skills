# Standalone Python Biostatistical Recipes & Formulas

Reference implementations extracted from `medstat-core` for standalone execution in standard data science environments (e.g. Claude Web sandbox, Jupyter, Google Colab) where `pandas`, `numpy`, `scipy`, `statsmodels`, `lifelines`, and `scikit-learn` are available without the `medstat` package installed.

---

## 1. Table 1 Baseline Characteristics & Standardized Mean Difference (SMD)

Computes publication-grade baseline patient characteristics with Austin (2009) Standardized Mean Differences ($\text{SMD} < 0.10$ denotes balance).

```python
import numpy as np
import pandas as pd
import scipy.stats as stats

def calculate_smd(treated: np.ndarray, control: np.ndarray) -> float:
    """Calculate Austin (2009) Standardized Mean Difference."""
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

def generate_table_one(df: pd.DataFrame, strata: str, continuous_vars: list[str], categorical_vars: list[str], nonnormal_vars: list[str] | None = None) -> pd.DataFrame:
    """Build stratified Table 1 with parametric/non-parametric tests and SMD."""
    nonnormal = set(nonnormal_vars or [])
    groups = sorted(df[strata].dropna().unique())
    rows = []
    
    # Total counts
    total_counts = {"Characteristic": "Overall (N = {})".format(len(df))}
    for g in groups:
        total_counts[f"{strata}={g} (N = {np.sum(df[strata] == g)})"] = ""
    total_counts["P-value"] = ""
    total_counts["SMD"] = ""
    rows.append(total_counts)

    # Continuous variables
    for var in continuous_vars:
        row = {"Characteristic": var}
        group_vals = [df.loc[df[strata] == g, var].dropna().to_numpy() for g in groups]
        if var in nonnormal:
            for g, vals in zip(groups, group_vals):
                med = np.median(vals) if len(vals) else np.nan
                q25, q75 = np.percentile(vals, [25, 75]) if len(vals) else (np.nan, np.nan)
                row[f"{strata}={g}"] = f"{med:.1f} [{q25:.1f}, {q75:.1f}]"
            pval = stats.kruskal(*group_vals).pvalue if len(groups) > 1 and all(len(v) for v in group_vals) else np.nan
        else:
            for g, vals in zip(groups, group_vals):
                mean, sd = np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                row[f"{strata}={g}"] = f"{mean:.1f} ({sd:.1f})"
            pval = stats.f_oneway(*group_vals).pvalue if len(groups) > 1 and all(len(v) for v in group_vals) else np.nan
        row["P-value"] = f"{pval:.3f}" if pval >= 0.001 else "<0.001" if not np.isnan(pval) else ""
        row["SMD"] = f"{calculate_smd(group_vals[0], group_vals[1]):.2f}" if len(groups) == 2 else ""
        rows.append(row)

    # Categorical variables
    for var in categorical_vars:
        rows.append({"Characteristic": f"**{var}**", "P-value": "", "SMD": ""})
        categories = sorted(df[var].dropna().unique())
        contingency = pd.crosstab(df[var], df[strata])
        chi2, pval, _, _ = stats.chi2_contingency(contingency) if contingency.size > 0 else (np.nan, np.nan, 0, 0)
        for cat in categories:
            row = {"Characteristic": f"  {cat}"}
            for g in groups:
                sub = df[df[strata] == g]
                cnt = np.sum(sub[var] == cat)
                pct = (cnt / len(sub) * 100) if len(sub) else 0.0
                row[f"{strata}={g}"] = f"{cnt} ({pct:.1f}%)"
            row["P-value"] = f"{pval:.3f}" if pval >= 0.001 else "<0.001" if not np.isnan(pval) else ""
            if len(groups) == 2:
                p0 = np.mean(df.loc[df[strata] == groups[0], var] == cat)
                p1 = np.mean(df.loc[df[strata] == groups[1], var] == cat)
                sd_bin = np.sqrt((p0 * (1 - p0) + p1 * (1 - p1)) / 2.0)
                row["SMD"] = f"{(p1 - p0) / sd_bin:.2f}" if sd_bin > 0 else "0.00"
            rows.append(row)

    return pd.DataFrame(rows)
```

---

## 2. 2x2 Diagnostic Accuracy with Wilson Score 95% CIs

Computes clinical diagnostic performance metrics with asymmetric Wilson score intervals and likelihood ratios.

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
    """Calculate sensitivity, specificity, PPV, NPV, LR+, LR-, and DOR."""
    n_pos = tp + fn
    n_neg = fp + tn
    sens = tp / n_pos if n_pos > 0 else np.nan
    spec = tn / n_neg if n_neg > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    
    # Likelihood ratios
    lr_pos = sens / (1.0 - spec) if (1.0 - spec) > 0 else np.nan
    lr_neg = (1.0 - sens) / spec if spec > 0 else np.nan
    dor = (tp * tn) / (fp * fn) if (fp * fn) > 0 else np.nan
    
    return {
        "Sensitivity": (sens, calculate_ci_wilson_score(tp, n_pos, ci)),
        "Specificity": (spec, calculate_ci_wilson_score(tn, n_neg, ci)),
        "PPV": (ppv, calculate_ci_wilson_score(tp, tp + fp, ci)),
        "NPV": (npv, calculate_ci_wilson_score(tn, tn + fn, ci)),
        "LR_pos": lr_pos,
        "LR_neg": lr_neg,
        "DOR": dor,
    }
```

---

## 3. Empirical ROC, Optimal Cutoff & DeLong Analytical 95% CI

Computes empirical ROC curve, Youden's J optimal cutpoint, and DeLong covariance matrix (DeLong et al., 1988) without bootstrap resampling.

```python
import numpy as np
import scipy.stats as stats
from sklearn.metrics import roc_curve, roc_auc_score

def auc_ci_delong(y_true: np.ndarray, y_score: np.ndarray, alpha: float = 0.05) -> dict:
    """Compute empirical AUC and DeLong analytical 95% confidence interval."""
    y_t = np.asarray(y_true).astype(int)
    y_s = np.asarray(y_score, dtype=float)
    valid = ~(np.isnan(y_t) | np.isnan(y_s))
    y_t, y_s = y_t[valid], y_s[valid]

    pos_scores = y_s[y_t == 1]
    neg_scores = y_s[y_t == 0]
    m, n = len(pos_scores), len(neg_scores)
    if m < 2 or n < 2:
        raise ValueError("Requires at least 2 positives and 2 negatives.")

    # Placement values
    comp = (pos_scores[:, None] > neg_scores).astype(float) + 0.5 * (pos_scores[:, None] == neg_scores).astype(float)
    v10 = comp.mean(axis=1)  # (m,)
    v01 = comp.mean(axis=0)  # (n,)
    auc = float(v10.mean())

    s10 = np.var(v10, ddof=1)
    s01 = np.var(v01, ddof=1)
    var_auc = (s10 / m) + (s01 / n)
    se_auc = np.sqrt(max(var_auc, 0.0))

    z = stats.norm.ppf(1.0 - alpha / 2.0)
    ci_low = max(0.0, auc - z * se_auc)
    ci_high = min(1.0, auc + z * se_auc)

    # Youden J cutoff
    fpr, tpr, thresholds = roc_curve(y_t, y_s)
    youden_j = tpr - fpr
    best_idx = int(np.argmax(youden_j))
    
    return {
        "auc": auc,
        "se": se_auc,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "optimal_threshold": float(thresholds[best_idx]),
        "sensitivity": float(tpr[best_idx]),
        "specificity": float(1.0 - fpr[best_idx]),
        "youden_j": float(youden_j[best_idx]),
    }
```

---

## 4. Model Calibration & Decision Curve Analysis (DCA)

Evaluates clinical prediction model goodness-of-fit (Austin & Steyerberg 2019 ICI, logistic recalibration slope) and net benefit across decision thresholds (Vickers & Elkin 2006).

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import logit
from statsmodels.nonparametric.smoothers_lowess import lowess

def evaluate_calibration(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute Brier Score, Calibration Slope/Intercept, and Integrated Calibration Index (ICI)."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.clip(np.asarray(y_pred, dtype=float), 1e-6, 1.0 - 1e-6)

    # 1. Brier score
    brier = float(np.mean((y_t - y_p) ** 2))
    prev = float(np.mean(y_t))
    scaled_brier = 1.0 - (brier / (prev * (1.0 - prev))) if prev > 0 else np.nan

    # 2. Calibration slope & intercept via logistic recalibration
    log_odds = logit(y_p)
    X_slope = sm.add_constant(log_odds)
    model_slope = sm.Logit(y_t, X_slope).fit(disp=False)
    slope = float(model_slope.params[1])

    model_inter = sm.Logit(y_t, np.ones_like(y_t), offset=log_odds).fit(disp=False)
    intercept = float(model_inter.params[0])

    # 3. Austin & Steyerberg (2019) ICI using LOWESS smoother
    order = np.argsort(y_p)
    smooth_obs = lowess(y_t[order], y_p[order], frac=2.0 / 3.0, it=0, return_sorted=False)
    abs_err = np.abs(y_p[order] - smooth_obs)
    ici = float(np.mean(abs_err))
    e50 = float(np.median(abs_err))
    e90 = float(np.percentile(abs_err, 90))
    emax = float(np.max(abs_err))

    return {
        "brier_score": brier,
        "scaled_brier": scaled_brier,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "ici": ici,
        "e50": e50,
        "e90": e90,
        "emax": emax,
    }

def calculate_dca(y_true: np.ndarray, y_pred: np.ndarray, thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """Calculate Vickers & Elkin (2006) Clinical Net Benefit across decision thresholds."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=float)
    n = len(y_t)
    prevalence = np.mean(y_t)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    records = []
    for pt in thresholds:
        weight = pt / (1.0 - pt)
        tp = np.sum((y_p >= pt) & (y_t == 1))
        fp = np.sum((y_p >= pt) & (y_t == 0))
        nb_model = (tp / n) - (fp / n) * weight
        nb_all = prevalence - (1.0 - prevalence) * weight
        records.append({"threshold": pt, "net_benefit": nb_model, "strategy": "Model"})
        records.append({"threshold": pt, "net_benefit": nb_all, "strategy": "Treat All"})
        records.append({"threshold": pt, "net_benefit": 0.0, "strategy": "Treat None"})

    return pd.DataFrame(records)
```

---

## 5. Observer Agreement: Bland-Altman & Intraclass Correlation (ICC)

Evaluates quantitative agreement and inter-rater reliability using pure NumPy/SciPy two-way ANOVA decomposition (Shrout & Fleiss 1979; McGraw & Wong 1996) and Bland-Altman (1999) large-sample variance.

```python
import numpy as np
import pandas as pd
import scipy.stats as stats

def calculate_bland_altman(m1: np.ndarray, m2: np.ndarray, ci: float = 0.95) -> dict:
    """Compute Bland-Altman mean difference and 95% Limits of Agreement with 1999 large-sample CIs."""
    diffs = np.asarray(m1, dtype=float) - np.asarray(m2, dtype=float)
    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1))
    se_mean = sd_diff / np.sqrt(n)
    
    t_crit = stats.t.ppf(1.0 - (1.0 - ci) / 2.0, df=n - 1)
    z_loa = stats.norm.ppf(1.0 - (1.0 - ci) / 2.0)
    upper_loa = mean_diff + z_loa * sd_diff
    lower_loa = mean_diff - z_loa * sd_diff

    # Standard error of Limits of Agreement (Bland & Altman 1999)
    se_loa = float(np.sqrt((1.0 / n + (z_loa**2) / (2.0 * (n - 1))) * (sd_diff**2)))

    return {
        "mean_diff": mean_diff,
        "ci_mean_diff": (mean_diff - t_crit * se_mean, mean_diff + t_crit * se_mean),
        "sd_diff": sd_diff,
        "upper_loa": upper_loa,
        "ci_upper_loa": (upper_loa - t_crit * se_loa, upper_loa + t_crit * se_loa),
        "lower_loa": lower_loa,
        "ci_lower_loa": (lower_loa - t_crit * se_loa, lower_loa + t_crit * se_loa),
    }

def calculate_icc_matrix(values: np.ndarray, alpha: float = 0.05) -> pd.DataFrame:
    """Compute all 6 Shrout & Fleiss (1979) ICC variants via pure two-way ANOVA."""
    n, k = values.shape  # n targets, k raters
    grand_mean = np.mean(values)
    row_means = np.mean(values, axis=1)
    col_means = np.mean(values, axis=0)

    SST = float(np.sum((values - grand_mean) ** 2))
    SSB = float(k * np.sum((row_means - grand_mean) ** 2))
    SSJ = float(n * np.sum((col_means - grand_mean) ** 2))
    SSE = max(float(SST - SSB - SSJ), 0.0)
    SSW = float(SST - SSB)

    df_B, df_J, df_E, df_W = n - 1, k - 1, (n - 1) * (k - 1), n * (k - 1)
    MSB = SSB / df_B if df_B > 0 else 0.0
    MSJ = SSJ / df_J if df_J > 0 else 0.0
    MSE = SSE / df_E if df_E > 0 else 0.0
    MSW = SSW / df_W if df_W > 0 else 0.0

    # Shrout & Fleiss (1979) formulas
    icc1 = (MSB - MSW) / (MSB + (k - 1) * MSW)
    icc2 = (MSB - MSE) / (MSB + (k - 1) * MSE + (k / n) * (MSJ - MSE))
    icc3 = (MSB - MSE) / (MSB + (k - 1) * MSE)
    icc1k = (MSB - MSW) / MSB
    icc2k = (MSB - MSE) / (MSB + (MSJ - MSE) / n)
    icc3k = (MSB - MSE) / MSB

    # F-tests
    f_obs = MSB / MSE if MSE > 0 else np.nan
    pval = float(stats.f.sf(f_obs, df_B, df_E)) if not np.isnan(f_obs) else np.nan

    return pd.DataFrame([
        {"Type": "ICC1", "Description": "One-way random (single)", "ICC": icc1, "F": MSB / MSW, "pval": stats.f.sf(MSB / MSW, df_B, df_W)},
        {"Type": "ICC2", "Description": "Two-way random agreement", "ICC": icc2, "F": f_obs, "pval": pval},
        {"Type": "ICC3", "Description": "Two-way mixed consistency", "ICC": icc3, "F": f_obs, "pval": pval},
        {"Type": "ICC1k", "Description": "One-way random (average)", "ICC": icc1k, "F": MSB / MSW, "pval": stats.f.sf(MSB / MSW, df_B, df_W)},
        {"Type": "ICC2k", "Description": "Two-way random agreement (avg)", "ICC": icc2k, "F": f_obs, "pval": pval},
        {"Type": "ICC3k", "Description": "Two-way mixed consistency (avg)", "ICC": icc3k, "F": f_obs, "pval": pval},
    ])
```

---

## 6. Propensity Score Matching (PSM) with Logit Caliper

1:1 nearest-neighbor matching without replacement bounded by Austin (2011) 0.2 caliper of logit propensity score.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import logit

def match_propensity_scores(df: pd.DataFrame, treatment_col: str, confounders: list[str], caliper_sd: float = 0.2) -> pd.DataFrame:
    """1:1 Nearest-Neighbor Propensity Score Matching within caliper."""
    clean = df.dropna(subset=[treatment_col] + confounders).copy()
    X = sm.add_constant(clean[confounders])
    y = clean[treatment_col].astype(int)

    # Estimate propensity score
    ps_model = sm.Logit(y, X).fit(disp=False)
    clean["ps"] = ps_model.predict(X)
    clean["logit_ps"] = logit(np.clip(clean["ps"], 1e-5, 1.0 - 1e-5))

    caliper = caliper_sd * np.std(clean["logit_ps"])

    treated = clean[clean[treatment_col] == 1].copy()
    control = clean[clean[treatment_col] == 0].copy()

    matched_indices = []
    avail_ctrl = control.copy()

    for idx, t_row in treated.iterrows():
        diffs = np.abs(avail_ctrl["logit_ps"] - t_row["logit_ps"])
        min_diff = diffs.min()
        if min_diff <= caliper:
            best_match_idx = diffs.idxmin()
            matched_indices.extend([idx, best_match_idx])
            avail_ctrl = avail_ctrl.drop(index=best_match_idx)

    matched_df = clean.loc[matched_indices].copy()
    return matched_df
```

---

## 7. Firth's Penalized Likelihood Regression Algorithm

When sample sizes are small or monotone separation occurs ($\text{EPV} < 10$), standard MLE estimates explode ($\text{OR} \to \infty$). Firth's penalized logistic regression modifies the score equation by Jeffreys invariant prior: $U^*(\beta) = U(\beta) + a(\beta)$, where $a_j = \frac{1}{2}\text{tr}(I^{-1}\frac{\partial I}{\partial \beta_j}) = \sum_i (h_i - 2h_i \pi_i) x_{ij}$.

```python
import numpy as np
import scipy.stats as stats

def fit_firth_logistic(X: np.ndarray, y: np.ndarray, max_iter: int = 50, tol: float = 1e-6) -> dict:
    """Pure-Python Firth penalized logistic regression via modified Newton-Raphson scoring."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape
    beta = np.zeros(p)

    for iteration in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        pi = 1.0 / (1.0 + np.exp(-eta))
        W = pi * (1.0 - pi)
        
        # Fisher Information Matrix I = X^T W X
        XtW = X.T * W
        I = XtW @ X
        try:
            I_inv = np.linalg.inv(I)
        except np.linalg.LinAlgError:
            I_inv = np.linalg.pinv(I)

        # Hat matrix diagonals: h_i = diag(W^{1/2} X I^{-1} X^T W^{1/2})
        H = np.sum((X @ I_inv) * X, axis=1) * W

        # Modified Firth Score: U* = X^T (y - pi + h * (0.5 - pi))
        u_star = X.T @ (y - pi + H * (0.5 - pi))

        step = I_inv @ u_star
        beta_new = beta + step

        if np.max(np.abs(step)) < tol:
            beta = beta_new
            break
        beta = beta_new

    # Standard errors and Wald tests
    se = np.sqrt(np.diag(I_inv))
    z = beta / se
    p_vals = 2.0 * (1.0 - stats.norm.cdf(np.abs(z)))
    or_point = np.exp(beta)
    ci_low = np.exp(beta - 1.96 * se)
    ci_high = np.exp(beta + 1.96 * se)

    return {
        "coefficients": beta,
        "standard_errors": se,
        "odds_ratios": or_point,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "p_values": p_vals,
    }
```

---

## 8. Little's MCAR Multivariate Missingness Test

Tests whether missing data across multivariate continuous variables satisfies Missing Completely At Random (Little 1988).

```python
import numpy as np
import pandas as pd
import scipy.stats as stats

def littles_mcar_test(df: pd.DataFrame, variables: list[str]) -> dict:
    """Perform Little's (1988) MCAR multivariate distance test."""
    data = df[variables].dropna(how="all").copy()
    n, p = len(data), len(variables)

    # Global ML estimates (EM-ready initial values)
    mu_hat = data.mean().to_numpy()
    cov_hat = data.cov().to_numpy()

    # Identify missingness patterns
    missing_patterns = data.isna().astype(int)
    pattern_groups = data.groupby(list(missing_patterns.columns))

    d_squared = 0.0
    df_mcar = 0

    for _, group in pattern_groups:
        if len(group) == 0:
            continue
        obs_cols = [c for c in variables if not group[c].isna().any()]
        if len(obs_cols) == 0:
            continue
        obs_idx = [variables.index(c) for c in obs_cols]
        sub_mu = mu_hat[obs_idx]
        sub_cov = cov_hat[np.ix_(obs_idx, obs_idx)]

        try:
            sub_cov_inv = np.linalg.inv(sub_cov)
        except np.linalg.LinAlgError:
            sub_cov_inv = np.linalg.pinv(sub_cov)

        grp_mean = group[obs_cols].mean().to_numpy()
        diff = grp_mean - sub_mu
        d_squared += len(group) * float(diff.T @ sub_cov_inv @ diff)
        df_mcar += len(obs_cols)

    df_mcar = max(df_mcar - p, 1)
    p_value = float(stats.chi2.sf(d_squared, df=df_mcar))

    return {
        "chi2": d_squared,
        "df": df_mcar,
        "p_value": p_value,
        "mcar_rejected": p_value < 0.05,
    }
```

---

## 9. VanderWeele & Ding (2017) Sensitivity E-Value

Calculates the minimum strength of association an unmeasured confounder must have with both exposure and outcome to explain away the observed effect.

```python
import numpy as np

def calculate_e_value(estimate: float, lower: float | None = None, upper: float | None = None, estimate_type: str = "RR") -> dict:
    """Calculate VanderWeele & Ding E-value for RR, OR, or HR."""
    est = float(estimate)
    if estimate_type in ["OR", "HR"]:
        est = np.sqrt(est)  # Square-root approximation for non-rare outcomes
        if lower is not None: lower = np.sqrt(lower)
        if upper is not None: upper = np.sqrt(upper)

    rr_star = est if est >= 1.0 else (1.0 / est)
    e_val_point = rr_star + np.sqrt(rr_star * (rr_star - 1.0))

    e_val_ci = 1.0
    if est >= 1.0 and lower is not None and lower > 1.0:
        e_val_ci = lower + np.sqrt(lower * (lower - 1.0))
    elif est < 1.0 and upper is not None and upper < 1.0:
        inv_u = 1.0 / upper
        e_val_ci = inv_u + np.sqrt(inv_u * (inv_u - 1.0))

    return {
        "e_value_point": float(e_val_point),
        "e_value_ci": float(e_val_ci),
    }
```

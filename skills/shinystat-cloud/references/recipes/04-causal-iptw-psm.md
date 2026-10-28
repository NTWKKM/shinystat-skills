# Observational Causal Inference (PSM & IPTW) Recipes (Cloud Sandbox)

> **Scope:** Propensity Score Matching (1:1 caliper matching), pair identification, Inverse Probability of Treatment Weighting (ATE/ATT/ATC/ATO), Kish effective sample size, and weighted balance diagnostics.



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

    if not confounders:
        raise ValueError("confounders list must not be empty.")

    n_initial = len(df)
    clean = (
        df.dropna(subset=[treatment_col] + confounders).copy().reset_index(drop=True)
    )
    for col in [treatment_col] + confounders:
        if pd.api.types.is_numeric_dtype(clean[col]):
            vals = clean[col].to_numpy(dtype=float)
            if np.isinf(vals).any():
                raise ValueError(
                    f"Column '{col}' contains infinite values (inf or -inf)."
                )
    n_analyzed = len(clean)
    n_excluded = n_initial - n_analyzed

    # Validate treatment column strictly binary
    treatment_vals = clean[treatment_col].to_numpy()
    if not np.all(np.isin(treatment_vals, [0, 1])):
        raise ValueError(
            f"Treatment column '{treatment_col}' must contain strictly binary values (0 and 1)."
        )
    if set(treatment_vals) != {0, 1}:
        raise ValueError(
            f"Treatment column '{treatment_col}' must contain both classes (0 and 1)."
        )
    y = treatment_vals.astype(int)

    # Handle categorical confounders via dummy encoding
    X_raw = pd.get_dummies(clean[confounders], drop_first=True, dtype=float)
    X = sm.add_constant(X_raw)

    # Fit propensity score model
    ps_model = sm.Logit(y, X).fit(disp=False)
    clean["ps"] = ps_model.predict(X)
    clean["logit_ps"] = logit(np.clip(clean["ps"], 1e-5, 1.0 - 1e-5))

    if not np.isfinite(caliper_sd) or caliper_sd < 0.0:
        raise ValueError(
            f"caliper_sd must be finite and non-negative (>= 0); got {caliper_sd}"
        )
    caliper = caliper_sd * float(np.std(clean["logit_ps"], ddof=1))

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

    if len(matched_records) > 0:
        matched_df = pd.DataFrame(matched_records)
        matched_clean = matched_df.reset_index(drop=True)
        matched_X = pd.get_dummies(
            matched_clean[confounders], drop_first=True, dtype=float
        ).reindex(columns=X_raw.columns, fill_value=0.0)
    else:
        matched_df = pd.DataFrame(columns=list(clean.columns) + ["pair_id"])
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

    bal_df = pd.DataFrame(
        balance_rows,
        columns=[
            "Variable",
            "Pre-Match SMD",
            "Post-Match SMD",
            "Balanced (<0.10)",
        ],
    )
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
    for col in [treatment_col] + covariates:
        if pd.api.types.is_numeric_dtype(clean[col]):
            vals = clean[col].to_numpy(dtype=float)
            if np.isinf(vals).any():
                raise ValueError(
                    f"Column '{col}' contains infinite values (inf or -inf)."
                )
    T = clean[treatment_col].to_numpy(dtype=float)
    if not np.all(np.isfinite(T)) or not np.all(np.isin(T, [0.0, 1.0])):
        raise ValueError(
            f"Treatment column '{treatment_col}' must contain strictly binary values (0 and 1)."
        )
    if set(T) != {0.0, 1.0}:
        raise ValueError(
            f"Treatment column '{treatment_col}' must contain both treated and control subjects (both 0 and 1)."
        )

    # 1. Fit Propensity Score via Logistic Regression with dummy encoding for categorical covariates
    X_raw = pd.get_dummies(clean[covariates], drop_first=True, dtype=float)
    X = sm.add_constant(X_raw)
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
        q_low, q_high = truncate_quantiles
        if not (0.0 <= q_low <= 1.0 and 0.0 <= q_high <= 1.0):
            raise ValueError(
                f"truncate_quantiles values must be within [0, 1]; got {truncate_quantiles}"
            )
        if q_low > q_high:
            raise ValueError(
                f"truncate_quantiles low ({q_low}) cannot exceed high ({q_high})"
            )
        w_low, w_high = (
            np.quantile(w_raw, q_low),
            np.quantile(w_raw, q_high),
        )
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

    if np.any(wt < 0.0) or np.any(wc < 0.0):
        raise ValueError("Weights cannot contain negative values.")

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
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError(
            f"Weights column '{weights_col}' must contain strictly finite and positive (> 0) values."
        )
    T = clean[treatment_col].to_numpy(dtype=float)
    if not np.all(np.isfinite(T)) or not np.all(np.isin(T, [0.0, 1.0])):
        raise ValueError(
            f"Treatment column '{treatment_col}' must contain strictly finite binary values (0 and 1)."
        )
    if set(T) != {0.0, 1.0}:
        raise ValueError(
            f"Treatment column '{treatment_col}' must contain both classes (0 and 1)."
        )

    if model_type == "continuous":
        Y = clean[outcome_col].to_numpy(dtype=float)
        if not np.all(np.isfinite(Y)):
            raise ValueError(
                f"Outcome column '{outcome_col}' contains non-finite values (inf or nan)."
            )
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
        if not np.all(np.isin(Y, [0.0, 1.0])):
            raise ValueError(
                f"Binary outcome column '{outcome_col}' must contain strictly 0 and 1 values."
            )
        if set(Y) != {0.0, 1.0}:
            raise ValueError(
                f"Binary outcome column '{outcome_col}' must contain both classes (0 and 1)."
            )
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
        surv_clean = clean.dropna(
            subset=[duration_col, outcome_col, treatment_col, weights_col]
        ).copy()
        raw_dur = surv_clean[duration_col].to_numpy(dtype=float)
        raw_evt = surv_clean[outcome_col].to_numpy(dtype=float)
        valid_mask = (
            np.isfinite(raw_dur)
            & (raw_dur > 0)
            & np.isfinite(raw_evt)
            & np.isin(raw_evt, [0.0, 1.0])
        )
        surv_clean = surv_clean[valid_mask].copy()
        if len(surv_clean) == 0:
            raise ValueError(
                "No valid observations remaining for weighted Cox survival model."
            )
        cph = CoxPHFitter()
        cph.fit(
            surv_clean[[duration_col, outcome_col, treatment_col, weights_col]],
            duration_col=duration_col,
            event_col=outcome_col,
            weights_col=weights_col,
            robust=True,
        )
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
    else:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. Supported types are 'continuous', 'binary', 'survival'."
        )
```

---

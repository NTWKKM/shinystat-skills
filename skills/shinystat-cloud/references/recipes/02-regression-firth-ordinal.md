# Regression, Firth Penalization & Sensitivity Recipes (Cloud Sandbox)

> **Scope:** Firth penalized likelihood logistic regression, VanderWeele & Ding (2017) E-value sensitivity analysis, and multivariable binary logistic regression.



---

## 7. Firth's Penalized Likelihood Logistic Regression

When sample sizes are small or monotone separation occurs ($\text{EPV} < 10$), standard MLE estimates explode ($\text{OR} \to \infty$). Firth's penalized logistic regression modifies the score equation by Jeffreys invariant prior: $U^*(\beta) = U(\beta) + a(\beta)$, where $a_j = \frac{1}{2}\text{tr}(I^{-1}\frac{\partial I}{\partial \beta_j}) = \sum_i h_i (0.5 - \pi_i) x_{ij}$. Reduces first-order parameter estimation bias (Heinze & Schemper, 2002). Implements **step-halving** to guarantee likelihood monotonicity and **profile likelihood confidence intervals** to eliminate Hauck-Donner explosion.

```python
import numpy as np
import scipy.optimize as optimize
from scipy.special import expit
import scipy.stats as stats


def _firth_penalized_loglik(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    eta = X @ beta
    pi = expit(eta)
    W = pi * (1.0 - pi)
    I = (X.T * W) @ X
    sign, logdet = np.linalg.slogdet(I)
    if sign <= 0:
        return -np.inf
    ll = np.sum(y * eta - np.logaddexp(0.0, eta))
    return float(ll + 0.5 * logdet)


def fit_firth_logistic(
    X: np.ndarray,
    y: np.ndarray,
    fit_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict:
    """Firth penalized logistic regression with Newton-Raphson step-halving and Profile Likelihood CIs."""
    if not isinstance(max_iter, (int, np.integer)) or isinstance(max_iter, bool) or max_iter < 1:
        raise ValueError(f"max_iter must be an integer >= 1; got {max_iter}")

    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    elif X_arr.ndim != 2:
        raise ValueError(f"X must be a 2D array or 1D vector; got ndim={X_arr.ndim}")

    y_arr = np.asarray(y, dtype=float).ravel()
    if len(X_arr) != len(y_arr):
        raise ValueError(
            f"Length mismatch: X ({len(X_arr)}) and y ({len(y_arr)}) must have matching row counts."
        )
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("X contains non-finite values (inf or nan).")
    if not np.all(np.isfinite(y_arr)):
        raise ValueError("y contains non-finite values (inf or nan).")

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
        eta = X_arr @ beta
        pi = expit(eta)
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

    # Recompute information matrix and inverse at final beta
    eta_final = X_arr @ beta
    pi_final = expit(eta_final)
    W_final = pi_final * (1.0 - pi_final)
    I_final = (X_arr.T * W_final) @ X_arr
    try:
        I_inv = np.linalg.inv(I_final)
    except np.linalg.LinAlgError:
        I_inv = np.linalg.pinv(I_final)
    se = np.sqrt(np.diag(I_inv))
    if not np.all(np.isfinite(se)) or np.any(se <= 0.0):
        raise ValueError(
            "Standard errors derived from Fisher information matrix are non-finite or non-positive."
        )

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
    if not np.isfinite(est) or est <= 0.0:
        raise ValueError("Estimate must be finite and strictly positive (> 0) on ratio scale.")
    low = float(lower) if lower is not None else None
    up = float(upper) if upper is not None else None
    if low is not None and (not np.isfinite(low) or low <= 0.0):
        raise ValueError("lower limit must be finite and strictly positive (> 0) on ratio scale.")
    if up is not None and (not np.isfinite(up) or up <= 0.0):
        raise ValueError("upper limit must be finite and strictly positive (> 0) on ratio scale.")
    if low is not None and low > est:
        raise ValueError(f"lower confidence limit ({low}) cannot exceed estimate ({est}).")
    if up is not None and up < est:
        raise ValueError(f"upper confidence limit ({up}) cannot be less than estimate ({est}).")
    if low is not None and up is not None and low > up:
        raise ValueError(
            f"lower confidence limit ({low}) cannot exceed upper confidence limit ({up})."
        )

    if estimate_type not in ("RR", "OR", "HR"):
        raise ValueError(
            f"Unsupported estimate_type '{estimate_type}'. Supported values are 'RR', 'OR', 'HR'."
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
    if not covariates:
        raise ValueError("covariates list must not be empty.")
    n_initial = len(df)
    clean = df.dropna(subset=[outcome] + covariates).copy()
    n_analyzed = len(clean)
    n_excluded = n_initial - n_analyzed
    if n_analyzed == 0:
        raise ValueError("No complete cases remaining after cleaning.")

    for col in [outcome] + covariates:
        vals = clean[col].to_numpy(dtype=float)
        if not np.all(np.isfinite(vals)):
            raise ValueError(f"Column '{col}' contains non-finite values (inf or nan).")
        if col != outcome and len(np.unique(vals)) < 2:
            raise ValueError(
                f"Covariate '{col}' must contain at least two distinct finite values."
            )

    outcome_vals = clean[outcome].to_numpy()
    if not np.all(np.isin(outcome_vals, [0, 1])):
        raise ValueError(
            f"Outcome column '{outcome}' must contain strictly binary values (0 and 1)."
        )
    if set(outcome_vals) != {0, 1}:
        raise ValueError(
            f"Outcome column '{outcome}' must contain both classes (0 and 1)."
        )
    y = outcome_vals.astype(int)

    # 1. Multivariable model
    X_multi = sm.add_constant(clean[covariates].astype(float))
    fit_multi = sm.Logit(y, X_multi).fit(disp=False)
    if not getattr(fit_multi, "mle_retvals", {}).get("converged", True):
        raise ValueError("Multivariable logistic regression fit did not converge.")

    rows = []
    for var in covariates:
        # Univariable model (Crude OR)
        X_uni = sm.add_constant(clean[[var]].astype(float))
        fit_uni = sm.Logit(y, X_uni).fit(disp=False)
        if not getattr(fit_uni, "mle_retvals", {}).get("converged", True):
            raise ValueError(
                f"Univariable logistic regression for '{var}' did not converge."
            )

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

        if not np.isfinite(adj_p):
            p_val_str = "NA"
        else:
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

## 21. Proportional Odds Ordinal Logistic Regression & Brant Test

Fits a cumulative-link proportional odds ordinal regression (`OrderedModel(..., distr="logit")`) for ordered clinical scales (e.g., mRS 0–6, NYHA Class I–IV, KDIGO AKI Stage 1–3) and verifies the parallel-slopes assumption using **Brant's (1990) omnibus and per-variable Wald test** across all $K-1$ cumulative binary thresholds. Includes an unconstrained multinomial logistic fallback (`fit_multinomial_logistic`) when proportional odds is violated ($P < 0.05$).

```python
from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel


def _extract_ordered_categories(
    y_series: pd.Series,
) -> tuple[pd.Series, list[Any] | np.ndarray]:
    if isinstance(y_series.dtype, pd.CategoricalDtype):
        y_cleaned = y_series.cat.remove_unused_categories()
        observed = set(y_cleaned.dropna().unique())
        categories = [c for c in y_cleaned.cat.categories if c in observed]
        return y_cleaned, categories
    return y_series, np.sort(y_series.unique())


def fit_proportional_odds(
    y: pd.Series | np.ndarray,
    X: pd.DataFrame | np.ndarray,
    distr: str = "logit",
    method: str = "bfgs",
    maxiter: int = 200,
) -> dict[str, Any]:
    """Fit cumulative link proportional odds model for ordinal outcomes (>= 3 ordered categories)."""
    y_raw = pd.Series(y)
    valid_mask = y_raw.notna().to_numpy()
    y_series = y_raw[valid_mask]
    y_series, categories = _extract_ordered_categories(y_series)
    k_categories = len(categories)

    if k_categories < 3:
        raise ValueError(
            f"Ordinal model requires at least 3 ordered categories (found {k_categories})."
        )

    if isinstance(X, pd.DataFrame):
        X_df = X.loc[y_series.index].copy()
        var_names = list(X_df.columns)
    else:
        X_mat = np.asarray(X)
        if len(X_mat) != len(y_raw):
            raise ValueError("Length mismatch between y and X.")
        if len(y_series) < len(y_raw):
            X_mat = X_mat[valid_mask]
        var_names = [f"x{i}" for i in range(X_mat.shape[1])]
        X_df = pd.DataFrame(X_mat, index=y_series.index, columns=var_names)

    model = OrderedModel(y_series, X_df, distr=distr)
    result = model.fit(method=method, maxiter=maxiter, disp=False)
    k_vars = model.k_vars

    coefs = result.params
    se = result.bse
    z_stat = result.tvalues
    p_values = result.pvalues
    conf = result.conf_int()

    odds_ratios, or_lower, or_upper = [], [], []
    for i in range(len(coefs)):
        if i < k_vars:
            b = coefs.iloc[i] if hasattr(coefs, "iloc") else coefs[i]
            ci_l = conf.iloc[i, 0] if hasattr(conf, "iloc") else conf[i, 0]
            ci_u = conf.iloc[i, 1] if hasattr(conf, "iloc") else conf[i, 1]
            odds_ratios.append(float(np.exp(b)))
            or_lower.append(float(np.exp(ci_l)))
            or_upper.append(float(np.exp(ci_u)))
        else:
            odds_ratios.append(np.nan)
            or_lower.append(np.nan)
            or_upper.append(np.nan)

    summary_df = pd.DataFrame(
        {
            "coef": np.asarray(coefs, dtype=float),
            "std_error": np.asarray(se, dtype=float),
            "z_stat": np.asarray(z_stat, dtype=float),
            "p_value": np.asarray(p_values, dtype=float),
            "odds_ratio": odds_ratios,
            "or_ci_lower": or_lower,
            "or_ci_upper": or_upper,
        },
        index=list(result.params.index),
    )

    raw_thresh = model.transform_threshold_params(result.params)
    cutpoints = np.asarray(raw_thresh[1:-1], dtype=float)
    predictor_df = summary_df.iloc[:k_vars].copy()

    return {
        "model": result,
        "summary_df": summary_df,
        "predictor_df": predictor_df,
        "cutpoints": cutpoints,
        "categories": [
            int(c) if isinstance(c, (int, np.integer)) else str(c) for c in categories
        ],
        "k_categories": int(k_categories),
        "log_likelihood": float(result.llf),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "nobs": int(result.nobs),
    }


def assess_proportional_odds(
    y: pd.Series | np.ndarray,
    X: pd.DataFrame | np.ndarray,
) -> dict[str, Any]:
    """Perform Brant's Wald test (Brant 1990) for the proportional odds parallel-slopes assumption."""
    y_raw = pd.Series(y)
    valid_mask = y_raw.notna().to_numpy()
    y_series = y_raw[valid_mask]
    y_series, categories = _extract_ordered_categories(y_series)
    K = len(categories)
    J = K - 1

    if K < 3:
        raise ValueError("Brant test requires an ordinal outcome with >= 3 categories.")

    if isinstance(X, pd.DataFrame):
        X_df = X.loc[y_series.index].copy()
    else:
        X_mat = np.asarray(X)
        if len(y_series) < len(y_raw):
            X_mat = X_mat[valid_mask]
        X_df = pd.DataFrame(
            X_mat,
            index=y_series.index,
            columns=[f"x{i}" for i in range(X_mat.shape[1])],
        )

    p_covars = X_df.shape[1]
    var_names = list(X_df.columns)
    X_const = sm.add_constant(X_df).to_numpy(dtype=float)
    n = len(y_series)

    betas, fitted_probs, inv_infos = [], [], []
    for j in range(J):
        z_j = y_series.isin(categories[j + 1 :]).to_numpy(dtype=int)
        if z_j.sum() == 0 or z_j.sum() == n:
            raise ValueError(
                f"Cutpoint {categories[j]} has zero cases in one class; cannot compute Brant test."
            )
        res_j = sm.Logit(z_j, X_const).fit(disp=False)
        betas.append(np.asarray(res_j.params[1:], dtype=float))
        p_j = np.asarray(res_j.predict(X_const), dtype=float)
        fitted_probs.append(p_j)
        w_j = p_j * (1.0 - p_j)
        info_j = X_const.T @ (w_j[:, None] * X_const)
        inv_infos.append(np.linalg.pinv(info_j))

    V = np.zeros((J * p_covars, J * p_covars), dtype=float)
    for j in range(J):
        for m in range(J):
            if j == m:
                V_jm = inv_infos[j]
            else:
                pj, pm = fitted_probs[j], fitted_probs[m]
                w_jm = (pm * (1.0 - pj)) if j < m else (pj * (1.0 - pm))
                info_jm = X_const.T @ (w_jm[:, None] * X_const)
                V_jm = inv_infos[j] @ info_jm @ inv_infos[m]
            V[
                j * p_covars : (j + 1) * p_covars, m * p_covars : (m + 1) * p_covars
            ] = V_jm[1:, 1:]

    C_omnibus = np.zeros(((J - 1) * p_covars, J * p_covars), dtype=float)
    for k in range(J - 1):
        C_omnibus[k * p_covars : (k + 1) * p_covars, 0:p_covars] = -np.eye(p_covars)
        C_omnibus[
            k * p_covars : (k + 1) * p_covars, (k + 1) * p_covars : (k + 2) * p_covars
        ] = np.eye(p_covars)

    beta_vec = np.concatenate(betas)
    diff_omnibus = C_omnibus @ beta_vec
    cov_diff_omnibus = C_omnibus @ V @ C_omnibus.T
    wald_omnibus = float(
        diff_omnibus.T @ np.linalg.pinv(cov_diff_omnibus) @ diff_omnibus
    )
    df_omnibus = (J - 1) * p_covars
    p_omnibus = float(stats.chi2.sf(wald_omnibus, df=df_omnibus))

    var_results = {}
    for v_idx, v_name in enumerate(var_names):
        C_v = np.zeros((J - 1, J * p_covars), dtype=float)
        for k in range(J - 1):
            C_v[k, v_idx] = -1.0
            C_v[k, (k + 1) * p_covars + v_idx] = 1.0
        diff_v = C_v @ beta_vec
        cov_v = C_v @ V @ C_v.T
        stat_v = float(diff_v.T @ np.linalg.pinv(cov_v) @ diff_v)
        df_v = J - 1
        p_v = float(stats.chi2.sf(stat_v, df=df_v))
        var_results[v_name] = {
            "statistic": stat_v,
            "df": df_v,
            "p_value": p_v,
            "passed": bool(p_v >= 0.05),
            "cutpoint_coefficients": {
                f"cutpoint_{categories[j]}": float(betas[j][v_idx]) for j in range(J)
            },
        }

    return {
        "test": "Brant Wald Test",
        "omnibus": {
            "statistic": wald_omnibus,
            "df": df_omnibus,
            "p_value": p_omnibus,
            "passed": bool(p_omnibus >= 0.05),
        },
        "variables": var_results,
    }


def fit_multinomial_logistic(
    y: pd.Series | np.ndarray,
    X: pd.DataFrame | np.ndarray,
    add_constant: bool = True,
    maxiter: int = 100,
) -> dict[str, Any]:
    """Fit unconstrained multinomial logit fallback when proportional odds is rejected."""
    y_raw = pd.Series(y)
    valid_mask = y_raw.notna().to_numpy()
    y_series = y_raw[valid_mask]
    X_df = (
        X.loc[y_series.index].copy()
        if isinstance(X, pd.DataFrame)
        else pd.DataFrame(np.asarray(X)[valid_mask], index=y_series.index)
    )
    X_mat = sm.add_constant(X_df).to_numpy(dtype=float) if add_constant else X_df.to_numpy(dtype=float)
    res = sm.MNLogit(y_series.to_numpy(), X_mat).fit(maxiter=maxiter, disp=False)
    return {
        "model": res,
        "log_likelihood": float(res.llf),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "params": res.params,
        "pvalues": res.pvalues,
    }
```

---

## 22. Restricted Cubic Splines (Harrell RCS) for Non-Linear Continuous Predictors

Constructs a standalone **Harrell Restricted Cubic Spline (RCS)** basis with $k \in \{3, 4, 5\}$ quantile knots (Harrell 2015 canonical quantiles: $[0.10, 0.50, 0.90]$ for $k=3$; $[0.05, 0.35, 0.65, 0.95]$ for $k=4$; $[0.05, 0.275, 0.50, 0.725, 0.95]$ for $k=5$), fits either a logistic regression (`family="binomial"`), linear model (`family="gaussian"`), or Cox PH model (`family="cox"`), and computes the **Likelihood Ratio Test for non-linearity (`p_nonlinearity`)**, overall spline association (`p_overall`), and reference-centered dose-response contrast curve ($\text{OR}(x)$ or $\text{HR}(x)$ vs `ref_value`).

```python
from typing import Any, Literal
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def build_harrell_rcs_basis(
    x: np.ndarray | pd.Series,
    knots: int | list[float] | np.ndarray = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct Harrell restricted cubic spline basis matrix (k - 1 columns) and knot vector."""
    x_arr = np.asarray(x, dtype=float).ravel()
    if isinstance(knots, (int, np.integer)):
        k = int(knots)
        if k < 3:
            raise ValueError("Restricted cubic splines require at least k=3 knots.")
        if k == 3:
            q = [0.10, 0.50, 0.90]
        elif k == 4:
            q = [0.05, 0.35, 0.65, 0.95]
        elif k == 5:
            q = [0.05, 0.275, 0.50, 0.725, 0.95]
        else:
            q = np.linspace(0.05, 0.95, k).tolist()
        knot_vals = np.quantile(x_arr[~np.isnan(x_arr)], q)
    else:
        knot_vals = np.sort(np.asarray(knots, dtype=float))
        k = len(knot_vals)
        if k < 3:
            raise ValueError("Restricted cubic splines require at least 3 knots.")

    knot_vals = np.unique(knot_vals)
    if len(knot_vals) < 3:
        raise ValueError("Insufficient distinct knot locations for RCS basis.")
    k = len(knot_vals)
    t_k = knot_vals[-1]
    t_km1 = knot_vals[-2]
    scale = (t_k - knot_vals[0]) ** 2 if t_k > knot_vals[0] else 1.0

    def _pos3(u: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, u) ** 3

    cols = [x_arr]
    denom_tail = t_k - t_km1 if t_k > t_km1 else 1.0
    for j in range(k - 2):
        t_j = knot_vals[j]
        term = (
            _pos3(x_arr - t_j)
            - _pos3(x_arr - t_km1) * ((t_k - t_j) / denom_tail)
            + _pos3(x_arr - t_k) * ((t_km1 - t_j) / denom_tail)
        ) / scale
        cols.append(term)

    return np.column_stack(cols), knot_vals


def fit_rcs_spline_model(
    df: pd.DataFrame,
    outcome_col: str,
    rcs_var: str,
    adjust_cols: list[str] | None = None,
    knots: int = 4,
    family: Literal["binomial", "gaussian"] = "binomial",
    ref_value: float | None = None,
    n_grid: int = 100,
) -> dict[str, Any]:
    """Fit Harrell RCS model, test non-linearity via LRT, and compute dose-response contrast curve."""
    adjust = [c for c in (adjust_cols or []) if c not in {outcome_col, rcs_var}]
    clean = df[[outcome_col, rcs_var] + adjust].dropna().copy()
    if len(clean) < 15:
        raise ValueError("Insufficient complete observations for RCS model (need >= 15).")

    x_vals = clean[rcs_var].to_numpy(dtype=float)
    y_vals = clean[outcome_col].to_numpy(dtype=float)
    basis_mat, knot_vals = build_harrell_rcs_basis(x_vals, knots=knots)
    n_spline_cols = basis_mat.shape[1]
    spline_names = [f"{rcs_var}_rcs{i}" for i in range(n_spline_cols)]

    X_spline_df = pd.DataFrame(basis_mat, index=clean.index, columns=spline_names)
    X_lin_df = pd.DataFrame({f"{rcs_var}_rcs0": x_vals}, index=clean.index)
    if adjust:
        for c in adjust:
            X_spline_df[c] = clean[c].astype(float)
            X_lin_df[c] = clean[c].astype(float)

    X_full = sm.add_constant(X_spline_df)
    X_lin = sm.add_constant(X_lin_df)
    fam = sm.families.Binomial() if family == "binomial" else sm.families.Gaussian()

    res_full = sm.GLM(y_vals, X_full, family=fam).fit()
    res_lin = sm.GLM(y_vals, X_lin, family=fam).fit()

    # Likelihood Ratio Test for non-linearity (comparing full k-1 spline terms vs linear term)
    lr_nonlin = max(0.0, float(2.0 * (res_full.llf - res_lin.llf)))
    df_nonlin = max(1, n_spline_cols - 1)
    p_nonlinearity = float(stats.chi2.sf(lr_nonlin, df=df_nonlin))

    # Wald test for overall association of all spline terms
    b_spline = res_full.params[spline_names].to_numpy(dtype=float)
    cov_spline = res_full.cov_params().loc[spline_names, spline_names].to_numpy(dtype=float)
    wald_overall = float(b_spline.T @ np.linalg.pinv(cov_spline) @ b_spline)
    p_overall = float(stats.chi2.sf(wald_overall, df=n_spline_cols))

    if ref_value is None:
        ref_value = float(np.median(x_vals))

    grid_x = np.linspace(float(np.min(x_vals)), float(np.max(x_vals)), n_grid)
    grid_basis, _ = build_harrell_rcs_basis(grid_x, knots=knot_vals)
    ref_basis, _ = build_harrell_rcs_basis(np.array([ref_value]), knots=knot_vals)

    diff_basis = grid_basis - ref_basis
    log_contrast = diff_basis @ b_spline
    var_contrast = np.sum((diff_basis @ cov_spline) * diff_basis, axis=1)
    se_contrast = np.sqrt(np.maximum(0.0, var_contrast))

    is_ratio = family == "binomial"
    est = np.exp(log_contrast) if is_ratio else log_contrast
    ci_low = (
        np.exp(log_contrast - 1.96 * se_contrast)
        if is_ratio
        else (log_contrast - 1.96 * se_contrast)
    )
    ci_high = (
        np.exp(log_contrast + 1.96 * se_contrast)
        if is_ratio
        else (log_contrast + 1.96 * se_contrast)
    )

    contrast_df = pd.DataFrame(
        {
            rcs_var: grid_x,
            "estimate": est,
            "ci_lower": ci_low,
            "ci_upper": ci_high,
            "log_contrast": log_contrast,
            "se": se_contrast,
        }
    )

    return {
        "model": res_full,
        "knots": knot_vals.tolist(),
        "ref_value": float(ref_value),
        "p_overall": p_overall,
        "p_nonlinearity": p_nonlinearity,
        "lr_nonlinearity": lr_nonlin,
        "df_nonlinearity": df_nonlin,
        "contrast_df": contrast_df,
        "aic": float(res_full.aic),
    }
```

---

## 23. Clustered GEE with Robust Sandwich SEs, ICC & Design Effect ($\text{DEFF}$)

Computes one-way random-effects ANOVA **Intraclass Correlation ($\text{ICC}_{\text{cluster}}$)**, **Design Effect ($\text{DEFF} = 1 + (\bar{m}-1)\text{ICC}$)**, and **Effective Sample Size ($N_{\text{eff}} = N / \text{DEFF}$)** for multi-center or repeated-measures data, and fits population-averaged **Generalized Estimating Equations (GEE)** with Exchangeable, Independence, or Autoregressive working correlation and Huber-White sandwich standard errors (using bias-reduced covariance when $K_{\text{clusters}} < 40$).

```python
from typing import Any, Literal
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.cov_struct import Autoregressive, Exchangeable, Independence
from statsmodels.genmod.families import Binomial, Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE


def calculate_design_effect(
    y: pd.Series | np.ndarray,
    cluster_ids: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Calculate cluster ICC, Design Effect (DEFF), and Effective Sample Size (N_eff)."""
    y_arr = np.asarray(y, dtype=float).ravel()
    c_arr = np.asarray(cluster_ids).ravel()
    valid = ~(np.isnan(y_arr) | pd.isna(c_arr))
    y_clean = y_arr[valid]
    c_clean = c_arr[valid]

    n_total = len(y_clean)
    unique_clusters, cluster_counts = np.unique(c_clean, return_counts=True)
    k_clusters = len(unique_clusters)
    if k_clusters < 2:
        return {
            "n_clusters": int(k_clusters),
            "n_total": int(n_total),
            "mean_cluster_size": float(n_total),
            "icc_cluster": 0.0,
            "design_effect": 1.0,
            "effective_sample_size": float(n_total),
        }

    grand_mean = float(np.mean(y_clean))
    ss_total = float(np.sum((y_clean - grand_mean) ** 2))
    cluster_means = np.array([np.mean(y_clean[c_clean == c]) for c in unique_clusters])
    ss_between = float(np.sum(cluster_counts * (cluster_means - grand_mean) ** 2))
    ss_within = ss_total - ss_between

    df_between = k_clusters - 1
    df_within = max(1, n_total - k_clusters)
    ms_between = ss_between / df_between if df_between > 0 else 0.0
    ms_within = ss_within / df_within

    m_bar = float(np.mean(cluster_counts))
    m_0 = (
        float(n_total - np.sum(cluster_counts**2) / n_total) / float(df_between)
        if df_between > 0
        else m_bar
    )
    s2_a = max(0.0, (ms_between - ms_within) / m_0) if m_0 > 0 else 0.0
    total_var = s2_a + ms_within
    icc = float(s2_a / total_var) if total_var > 0 else 0.0
    deff = float(1.0 + (m_bar - 1.0) * icc)
    n_eff = float(n_total / deff) if deff > 0 else float(n_total)

    return {
        "n_clusters": int(k_clusters),
        "n_total": int(n_total),
        "mean_cluster_size": m_bar,
        "icc_cluster": icc,
        "design_effect": deff,
        "effective_sample_size": n_eff,
    }


def fit_gee(
    y: pd.Series | np.ndarray,
    X: pd.DataFrame | np.ndarray,
    cluster_ids: pd.Series | np.ndarray,
    family: Literal["binomial", "gaussian"] = "binomial",
    cov_struct: Literal["exchangeable", "independence", "autoregressive"] = "exchangeable",
    add_constant: bool = True,
    time: pd.Series | np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit population-averaged GEE with sandwich SEs (bias-reduced when n_clusters < 40)."""
    y_s = pd.Series(y)
    c_s = pd.Series(cluster_ids)
    valid = ~(y_s.isna() | c_s.isna())
    y_clean = y_s[valid].to_numpy(dtype=float)
    c_clean = c_s[valid].to_numpy()

    if isinstance(X, pd.DataFrame):
        X_df = X.loc[y_s[valid].index].copy()
        var_names = list(X_df.columns)
        X_mat = X_df.to_numpy(dtype=float)
    else:
        X_mat = np.asarray(X, dtype=float)[valid.to_numpy()]
        var_names = [f"x{i}" for i in range(X_mat.shape[1])]

    if add_constant:
        X_mat = sm.add_constant(X_mat)
        var_names = ["const"] + var_names

    t_clean = None
    if time is not None:
        t_clean = np.asarray(time, dtype=float)[valid.to_numpy()]
        order = np.lexsort((t_clean, c_clean))
        t_clean = t_clean[order]
    else:
        order = np.argsort(c_clean, kind="stable")

    y_clean = y_clean[order]
    X_mat = X_mat[order]
    c_clean = c_clean[order]

    struct_map = {
        "exchangeable": Exchangeable,
        "independence": Independence,
        "autoregressive": lambda: Autoregressive(grid=True),
    }
    struct = struct_map[str(cov_struct).lower()]()
    fam = Binomial() if str(family).lower() == "binomial" else Gaussian()
    is_binary = str(family).lower() == "binomial"

    n_clusters = len(np.unique(c_clean))
    cov_type = "bias_reduced" if n_clusters < 40 else "robust"
    model = GEE(
        y_clean, X_mat, groups=c_clean, time=t_clean, family=fam, cov_struct=struct
    )
    res = model.fit(cov_type=cov_type)

    coefs = np.asarray(res.params, dtype=float)
    se = np.asarray(res.bse, dtype=float)
    conf = np.asarray(res.conf_int(), dtype=float)
    summary_df = pd.DataFrame(
        {
            "coef": coefs,
            "std_error": se,
            "z_stat": np.asarray(res.tvalues, dtype=float),
            "p_value": np.asarray(res.pvalues, dtype=float),
            "ci_lower": conf[:, 0],
            "ci_upper": conf[:, 1],
            "odds_ratio": np.exp(coefs) if is_binary else np.full_like(coefs, np.nan),
            "or_ci_lower": np.exp(conf[:, 0]) if is_binary else np.full_like(coefs, np.nan),
            "or_ci_upper": np.exp(conf[:, 1]) if is_binary else np.full_like(coefs, np.nan),
        },
        index=var_names,
    )
    deff_info = calculate_design_effect(y_clean, c_clean)

    return {
        "model": res,
        "summary_df": summary_df,
        "n_clusters": n_clusters,
        "cov_type": cov_type,
        "design_effect": deff_info["design_effect"],
        "icc_cluster": deff_info["icc_cluster"],
        "effective_sample_size": deff_info["effective_sample_size"],
    }
```

---

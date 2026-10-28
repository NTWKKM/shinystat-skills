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

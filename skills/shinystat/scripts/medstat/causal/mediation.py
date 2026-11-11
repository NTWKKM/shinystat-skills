"""
Causal Mediation Analysis Module.

Estimates Average Causal Mediation Effect (ACME / Indirect Effect),
Average Direct Effect (ADE), Total Effect, and Proportion Mediated
using the product-of-coefficients method when mediator and outcome models are linear, and
Imai et al. (2010) quasi-Bayesian g-computation on the probability scale when either is binary
(the logit product alpha*beta is not a causal effect because of non-collapsibility).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from medstat.data.missing import prepare_data_for_analysis


def run_mediation(
    df: pd.DataFrame,
    treatment: str,
    mediator: str,
    outcome: str,
    covariates: list[str] | None = None,
    n_sims: int = 1000,
    seed: int = 42,
    missing_strategy: str | None = None,
    missing_justification: str | None = None,
) -> dict[str, Any]:
    """
    Perform parametric causal mediation analysis.

    Parameters:
        df: Clinical DataFrame.
        treatment: Binary or continuous exposure/treatment variable.
        mediator: Intermediate mediator variable.
        outcome: Primary clinical outcome variable.
        covariates: Optional baseline confounders.
        n_sims: Number of Monte Carlo draws for confidence intervals (default 1000).
        seed: Random seed for reproducibility.
        missing_strategy: Strategy for handling missing data ('complete-case', 'knn', 'indicator'). Note: 'mice' is not supported.
        missing_justification: Documented clinical rationale for missing data strategy under STROBE/CONSORT.

    Returns:
        dict containing ACME, ADE, Total Effect, Proportion Mediated, 95% CIs, and sample retention info.
    """
    if (
        missing_strategy is not None
        and missing_strategy.lower().replace("_", "-") == "mice"
    ):
        raise NotImplementedError(
            "MICE multiple imputation is not supported for causal mediation analysis because combining "
            "quasi-Bayesian Monte Carlo mediation estimates across multiple imputations (Rubin's rules) "
            "is not yet implemented. Please use 'complete-case', 'knn', or 'indicator'."
        )

    covar_list = covariates or []
    all_cols = [treatment, mediator, outcome] + covar_list
    df_clean, missing_info = prepare_data_for_analysis(
        df,
        required_cols=all_cols,
        handle_missing=missing_strategy,
        missing_justification=missing_justification,
        disallowed_strategies={"mice"},
    )

    if len(df_clean) < 10:
        raise ValueError("Insufficient observations for causal mediation analysis.")

    def _is_binary(col: pd.Series) -> bool:
        return bool(col.nunique() == 2 and set(col.unique()).issubset({0, 1, 0.0, 1.0}))

    is_binary_mediator = _is_binary(df_clean[mediator])
    is_binary_outcome = _is_binary(df_clean[outcome])

    # 1. Mediator model: M ~ Treatment + Covariates (OLS, or Logit for a binary mediator)
    X_med = sm.add_constant(df_clean[[treatment] + covar_list])
    y_med = df_clean[mediator]
    if is_binary_mediator:
        try:
            med_model = sm.Logit(y_med, X_med).fit(disp=False)
        except Exception as e:
            raise ValueError(f"Failed to fit logistic mediator model: {e}")
    else:
        med_model = sm.OLS(y_med, X_med).fit()

    alpha_1 = float(med_model.params[treatment])
    se_alpha_1 = float(med_model.bse[treatment])

    # 2. Outcome model: Y ~ Treatment + Mediator + Covariates
    X_out = sm.add_constant(df_clean[[treatment, mediator] + covar_list])
    y_out = df_clean[outcome]

    if is_binary_outcome:
        try:
            out_model = sm.Logit(y_out, X_out).fit(disp=False)
        except Exception as e:
            raise ValueError(f"Failed to fit logistic outcome model: {e}")
    else:
        out_model = sm.OLS(y_out, X_out).fit()

    beta_trt = float(out_model.params[treatment])
    beta_med = float(out_model.params[mediator])
    se_beta_med = float(out_model.bse[mediator])

    rng = np.random.default_rng(seed)

    if (is_binary_outcome or is_binary_mediator) and not _is_binary(df_clean[treatment]):
        # The g-computation contrast is E[Y(1, .)] - E[Y(0, .)]; for a non 0/1 treatment it
        # would be an extrapolated 1-vs-0 contrast rather than an observed one.
        raise ValueError(
            f"Treatment '{treatment}' must be coded 0/1 when the mediator or outcome is binary "
            "(Imai g-computation estimates the 1 vs 0 contrast)."
        )

    if is_binary_outcome or is_binary_mediator:
        # Non-linear models: the product alpha_1 * beta_med is not a causal effect on any
        # scale (non-collapsibility of the logit). Use Imai et al. (2010) g-computation of
        # E[Y(t, M(t'))] on the probability / mean scale with quasi-Bayesian parameter draws.
        sim = _imai_quasi_bayes(
            df_clean,
            treatment,
            mediator,
            covar_list,
            med_model,
            out_model,
            is_binary_mediator,
            is_binary_outcome,
            n_sims,
            rng,
        )
        point = sim["point"]
        draws = sim["draws"]

        def _ci(key: str) -> tuple[float, float]:
            return (
                float(np.percentile(draws[key], 2.5)),
                float(np.percentile(draws[key], 97.5)),
            )

        acme_point, ade_point, total_point = point["acme"], point["ade"], point["total"]
        prop_med = (acme_point / total_point) if abs(total_point) > 1e-9 else 0.0
        acme_ci, ade_ci, total_ci = _ci("acme"), _ci("ade"), _ci("total")
        # Add-one Monte Carlo estimator: never exactly 0 with finitely many draws.
        n_draws = len(draws["acme"])
        tail = min(np.sum(draws["acme"] <= 0), np.sum(draws["acme"] >= 0))
        p_value_acme = float(min(1.0, 2.0 * (tail + 1) / (n_draws + 1)))
        scale = "risk_difference" if is_binary_outcome else "mean_difference"
        method_desc = (
            "Imai et al. (2010) quasi-Bayesian g-computation "
            f"({'logistic' if is_binary_mediator else 'linear'} mediator, "
            f"{'logistic' if is_binary_outcome else 'linear'} outcome; "
            f"{scale.replace('_', ' ')} scale, treatment contrast 1 vs 0)"
        )
        extra = {
            "acme_treated": point["acme_1"],
            "acme_control": point["acme_0"],
            "ade_treated": point["ade_1"],
            "ade_control": point["ade_0"],
        }
    else:
        # Linear mediator and outcome: product of coefficients equals the Imai ACME.
        acme_point = alpha_1 * beta_med
        ade_point = beta_trt
        total_point = ade_point + acme_point
        prop_med = (acme_point / total_point) if abs(total_point) > 1e-9 else 0.0

        sim_alpha_1 = rng.normal(alpha_1, se_alpha_1, size=n_sims)
        cov_params = out_model.cov_params()
        mean_joint = np.array([beta_trt, beta_med], dtype=float)
        cov_joint = cov_params.loc[
            [treatment, mediator], [treatment, mediator]
        ].to_numpy()
        sim_joint = rng.multivariate_normal(mean_joint, cov_joint, size=n_sims)
        sim_acme = sim_alpha_1 * sim_joint[:, 1]
        sim_ade = sim_joint[:, 0]
        sim_total = sim_ade + sim_acme

        acme_ci = (
            float(np.percentile(sim_acme, 2.5)),
            float(np.percentile(sim_acme, 97.5)),
        )
        ade_ci = (
            float(np.percentile(sim_ade, 2.5)),
            float(np.percentile(sim_ade, 97.5)),
        )
        total_ci = (
            float(np.percentile(sim_total, 2.5)),
            float(np.percentile(sim_total, 97.5)),
        )

        # Sobel test p-value for indirect effect
        sobel_se = np.sqrt(alpha_1**2 * se_beta_med**2 + beta_med**2 * se_alpha_1**2)
        z_stat = (acme_point / sobel_se) if sobel_se > 0 else 0.0
        p_value_acme = float(2.0 * stats.norm.sf(abs(z_stat)))
        scale = "linear"
        method_desc = "Quasi-Bayesian Monte Carlo & Baron-Kenny (Linear Scale)"
        extra = {}

    return {
        "treatment": treatment,
        "mediator": mediator,
        "outcome": outcome,
        "n_input": len(df),
        "n_observations": len(df_clean),
        "n_excluded": missing_info["rows_excluded"],
        "missing_counts": missing_info["missing_counts"],
        "scale": scale,
        "effect_scale": scale,
        "acme": acme_point,
        "acme_ci": acme_ci,
        "acme_pvalue": p_value_acme,
        "indirect_effect": acme_point,
        "indirect_ci": acme_ci,
        "ade": ade_point,
        "ade_ci": ade_ci,
        "direct_effect": ade_point,
        "total_effect": total_point,
        "total_ci": total_ci,
        "prop_mediated": prop_med,
        "method": method_desc,
        **extra,
    }


def _imai_quasi_bayes(
    df: pd.DataFrame,
    treatment: str,
    mediator: str,
    covariates: list[str],
    med_model: Any,
    out_model: Any,
    binary_mediator: bool,
    binary_outcome: bool,
    n_sims: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Natural effects by g-computation: E[Y(t, M(t'))] averaged over the empirical covariates.

    ACME(t) = E[Y(t, M(1))] - E[Y(t, M(0))]; ADE(t) = E[Y(1, M(t))] - E[Y(0, M(t))];
    reported ACME / ADE are the averages over t in {0, 1}; total = E[Y(1, M(1))] - E[Y(0, M(0))].
    """
    n = len(df)
    cov = df[covariates].to_numpy(dtype=float) if covariates else np.zeros((n, 0))
    ones = np.ones((n, 1))

    def x_med(t: float) -> np.ndarray:
        return np.column_stack([ones, np.full(n, t), cov])

    def x_out(t: float, m: np.ndarray) -> np.ndarray:
        return np.column_stack([ones, np.full(n, t), m, cov])

    sigma_m = 0.0 if binary_mediator else float(np.sqrt(med_model.scale))
    z_m = rng.standard_normal(n)  # common random numbers for a continuous mediator

    def mean_y(gamma: np.ndarray, beta: np.ndarray, t: float, t_m: float) -> float:
        lin_m = x_med(t_m) @ gamma
        if binary_mediator:
            p1 = 1.0 / (1.0 + np.exp(-lin_m))
            ey1 = x_out(t, np.ones(n)) @ beta
            ey0 = x_out(t, np.zeros(n)) @ beta
            if binary_outcome:
                ey1, ey0 = 1.0 / (1.0 + np.exp(-ey1)), 1.0 / (1.0 + np.exp(-ey0))
            return float(np.mean(p1 * ey1 + (1.0 - p1) * ey0))
        m = lin_m + sigma_m * z_m
        ey = x_out(t, m) @ beta
        if binary_outcome:
            ey = 1.0 / (1.0 + np.exp(-ey))
        return float(np.mean(ey))

    def effects(gamma: np.ndarray, beta: np.ndarray) -> dict[str, float]:
        e = {
            (t, tm): mean_y(gamma, beta, t, tm) for t in (0.0, 1.0) for tm in (0.0, 1.0)
        }
        acme_1, acme_0 = e[(1, 1)] - e[(1, 0)], e[(0, 1)] - e[(0, 0)]
        ade_1, ade_0 = e[(1, 1)] - e[(0, 1)], e[(1, 0)] - e[(0, 0)]
        return {
            "acme": 0.5 * (acme_1 + acme_0),
            "ade": 0.5 * (ade_1 + ade_0),
            "total": e[(1, 1)] - e[(0, 0)],
            "acme_1": acme_1,
            "acme_0": acme_0,
            "ade_1": ade_1,
            "ade_0": ade_0,
        }

    gamma_hat = med_model.params.to_numpy(dtype=float)
    beta_hat = out_model.params.to_numpy(dtype=float)
    point = effects(gamma_hat, beta_hat)

    gammas = rng.multivariate_normal(
        gamma_hat, med_model.cov_params().to_numpy(), size=n_sims
    )
    betas = rng.multivariate_normal(
        beta_hat, out_model.cov_params().to_numpy(), size=n_sims
    )
    sims = [effects(g, b) for g, b in zip(gammas, betas)]
    draws = {k: np.array([d[k] for d in sims]) for k in ("acme", "ade", "total")}
    return {"point": point, "draws": draws}

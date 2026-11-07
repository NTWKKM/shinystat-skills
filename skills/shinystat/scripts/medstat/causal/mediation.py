"""
Causal Mediation Analysis Module.

Estimates Average Causal Mediation Effect (ACME / Indirect Effect),
Average Direct Effect (ADE), Total Effect, and Proportion Mediated
using the parametric product-of-coefficients method with quasi-Bayesian
Monte Carlo simulation for standard errors and confidence intervals (Imai et al. 2010, Baron & Kenny 1986).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

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

    # 1. Mediator model: M ~ Treatment + Covariates (OLS)
    X_med = sm.add_constant(df_clean[[treatment] + covar_list])
    y_med = df_clean[mediator]
    med_model = sm.OLS(y_med, X_med).fit()

    alpha_1 = float(med_model.params[treatment])
    se_alpha_1 = float(med_model.bse[treatment])

    # 2. Outcome model: Y ~ Treatment + Mediator + Covariates
    X_out = sm.add_constant(df_clean[[treatment, mediator] + covar_list])
    y_out = df_clean[outcome]

    is_binary_outcome = bool(
        y_out.nunique() == 2 and set(y_out.unique()).issubset({0, 1, 0.0, 1.0})
    )
    if is_binary_outcome:
        try:
            out_model = sm.Logit(y_out, X_out).fit(disp=False)
        except Exception as e:
            raise ValueError(f"Failed to fit logistic outcome model: {e}")
        scale = "log_odds"
        method_desc = "Baron-Kenny Product of Coefficients (Log-Odds Scale)"
    else:
        out_model = sm.OLS(y_out, X_out).fit()
        scale = "linear"
        method_desc = "Quasi-Bayesian Monte Carlo & Baron-Kenny (Linear Scale)"

    beta_trt = float(out_model.params[treatment])

    beta_med = float(out_model.params[mediator])
    se_beta_med = float(out_model.bse[mediator])

    # 3. Effect estimation
    # Indirect Effect (ACME) = alpha_1 * beta_med
    acme_point = alpha_1 * beta_med
    ade_point = beta_trt
    total_point = ade_point + acme_point
    prop_med = (acme_point / total_point) if abs(total_point) > 1e-9 else 0.0

    # 4. Quasi-Bayesian Monte Carlo Confidence Intervals
    rng = np.random.default_rng(seed)
    sim_alpha_1 = rng.normal(alpha_1, se_alpha_1, size=n_sims)

    # Jointly sample treatment and mediator parameters from outcome model covariance
    cov_params = out_model.cov_params()
    mean_joint = np.array([beta_trt, beta_med], dtype=float)
    cov_joint = cov_params.loc[[treatment, mediator], [treatment, mediator]].to_numpy()
    sim_joint = rng.multivariate_normal(mean_joint, cov_joint, size=n_sims)
    sim_beta_trt = sim_joint[:, 0]
    sim_beta_med = sim_joint[:, 1]

    sim_acme = sim_alpha_1 * sim_beta_med
    sim_ade = sim_beta_trt
    sim_total = sim_ade + sim_acme

    acme_ci = (
        float(np.percentile(sim_acme, 2.5)),
        float(np.percentile(sim_acme, 97.5)),
    )
    ade_ci = (float(np.percentile(sim_ade, 2.5)), float(np.percentile(sim_ade, 97.5)))
    total_ci = (
        float(np.percentile(sim_total, 2.5)),
        float(np.percentile(sim_total, 97.5)),
    )

    # Sobel test p-value for indirect effect
    sobel_se = np.sqrt(alpha_1**2 * se_beta_med**2 + beta_med**2 * se_alpha_1**2)
    z_stat = (acme_point / sobel_se) if sobel_se > 0 else 0.0
    from scipy import stats

    p_value_acme = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))

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
    }

"""
Sample Size & Statistical Power Calculations for Study Design.

Provides analytical sample size and power estimations for:
- Two independent means (Student's / Welch's t-test)
- Two independent proportions (Chi-square / Z-test)
- Survival analysis (Schoenfeld log-rank formula)
- Pearson correlation
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import scipy.optimize as opt
import scipy.stats as stats
import statsmodels.stats.power as smp
import statsmodels.stats.proportion as smprop
from scipy.special import expit, log_expit


def calculate_sample_size_means(
    mean1: float,
    mean2: float,
    sd1: float,
    sd2: float | None = None,
    alpha: float = 0.05,
    power: float = 0.80,
    ratio: float = 1.0,
    alternative: Literal["two-sided", "larger", "smaller"] = "two-sided",
) -> dict[str, Any]:
    """
    Calculate required sample size to compare two independent means.
    """
    s2 = sd1 if sd2 is None else sd2
    if sd1 <= 0 or s2 <= 0:
        raise ValueError("Standard deviations must be strictly positive.")

    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha}")
    if not (0.0 < power < 1.0):
        raise ValueError(f"power must be strictly between 0 and 1, got {power}")
    if power <= alpha:
        raise ValueError(
            f"power ({power}) must be strictly greater than alpha ({alpha})"
        )

    sd_pooled = math.sqrt((sd1**2 + s2**2) / 2.0)
    diff = abs(mean1 - mean2)
    if diff == 0:
        raise ValueError("Effect size is zero (means are identical).")

    effect_size = diff / sd_pooled

    analysis = smp.TTestIndPower()
    n1 = analysis.solve_power(
        effect_size=effect_size,
        power=power,
        alpha=alpha,
        ratio=ratio,
        alternative=alternative,
    )

    n1_int = int(math.ceil(float(n1)))
    n2_int = int(math.ceil(float(n1) * ratio))

    return {
        "n1": n1_int,
        "n2": n2_int,
        "total_n": n1_int + n2_int,
        "effect_size_d": float(effect_size),
        "alpha": alpha,
        "power": power,
        "ratio": ratio,
    }


def calculate_sample_size_t_test(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
    ratio: float = 1.0,
    alternative: Literal["two-sided", "larger", "smaller"] = "two-sided",
) -> int:
    """
    Calculate sample size per group for independent two-sample t-test given Cohen's d.

    Parameters:
        effect_size: Cohen's d (must be > 0).
        alpha: Type I error rate (default 0.05).
        power: Statistical power 1 - beta (default 0.80).
        ratio: Ratio of sample size in group 2 to group 1 (default 1.0).
        alternative: 'two-sided', 'larger', or 'smaller'.

    Returns:
        int: Required sample size per group (n1).
    """
    if effect_size <= 0:
        raise ValueError("Effect size must be strictly positive.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha}")
    if not (0.0 < power < 1.0):
        raise ValueError(f"power must be strictly between 0 and 1, got {power}")
    if power <= alpha:
        raise ValueError(
            f"power ({power}) must be strictly greater than alpha ({alpha})"
        )

    analysis = smp.TTestIndPower()
    n = analysis.solve_power(
        effect_size=effect_size,
        power=power,
        alpha=alpha,
        ratio=ratio,
        alternative=alternative,
    )
    return int(math.ceil(float(n)))


def calculate_sample_size_proportions(
    p1: float,
    p2: float,
    alpha: float = 0.05,
    power: float = 0.80,
    ratio: float = 1.0,
    alternative: Literal["two-sided", "larger", "smaller"] = "two-sided",
) -> dict[str, Any]:
    """
    Calculate required sample size to compare two independent proportions.
    Uses Cohen's h transformation for arc-sine power calculations.
    """
    if not (0 < p1 < 1 and 0 < p2 < 1):
        raise ValueError("Proportions must be strictly between 0 and 1.")

    effect_size = smprop.proportion_effectsize(p1, p2)
    if abs(effect_size) < 1e-6:
        raise ValueError("Effect size is negligible (proportions are identical).")

    analysis = smp.NormalIndPower()
    n1 = analysis.solve_power(
        effect_size=abs(effect_size),
        power=power,
        alpha=alpha,
        ratio=ratio,
        alternative=alternative,
    )

    n1_int = int(math.ceil(float(n1)))
    n2_int = int(math.ceil(float(n1) * ratio))

    return {
        "n1": n1_int,
        "n2": n2_int,
        "total_n": n1_int + n2_int,
        "cohens_h": float(effect_size),
        "alpha": alpha,
        "power": power,
        "ratio": ratio,
    }


def calculate_sample_size_survival(
    hazard_ratio: float,
    p_event: float = 0.50,
    alpha: float = 0.05,
    power: float = 0.80,
    ratio: float = 1.0,
) -> dict[str, Any]:
    """
    Calculate required number of events and subjects for survival analysis (Schoenfeld 1981).
    """
    hr = float(hazard_ratio)
    if hr <= 0 or hr == 1.0:
        raise ValueError("Hazard ratio must be positive and not equal to 1.0.")

    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    z_beta = stats.norm.ppf(power)

    # Proportion in treatment group: p1 = ratio / (1 + ratio)
    p_trt = ratio / (1.0 + ratio)
    p_ctrl = 1.0 - p_trt

    # Schoenfeld formula for required events E:
    # E = ((z_alpha + z_beta)^2) / (p1 * p2 * (ln(HR))^2)
    num = (z_alpha + z_beta) ** 2
    den = p_trt * p_ctrl * (math.log(hr) ** 2)
    required_events = math.ceil(num / den)

    # Total sample size adjusted for expected event rate
    total_n = math.ceil(required_events / p_event) if p_event > 0 else required_events
    n1 = math.ceil(total_n * p_trt)
    n2 = total_n - n1

    return {
        "required_events": int(required_events),
        "total_n": int(total_n),
        "n_treated": int(n1),
        "n_control": int(n2),
        "hazard_ratio": hr,
        "alpha": alpha,
        "power": power,
    }


def calculate_sample_size_correlation(
    r: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict[str, Any]:
    """
    Calculate required sample size for testing Pearson correlation coefficient.
    """
    abs_r = abs(r)
    if abs_r <= 0 or abs_r >= 1:
        raise ValueError("Correlation r must be between -1 and 1 and non-zero.")

    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    z_beta = stats.norm.ppf(power)

    # Fisher's Z transformation
    c = 0.5 * math.log((1.0 + abs_r) / (1.0 - abs_r))
    n = ((z_alpha + z_beta) / c) ** 2 + 3.0

    return {
        "required_n": int(math.ceil(n)),
        "r": r,
        "alpha": alpha,
        "power": power,
    }


def _approximate_cox_snell_r2_from_c_stat(
    c_stat: float,
    prevalence: float,
    deg: int = 64,
) -> float:
    """
    Exact deterministic numerical derivation of Cox-Snell R2 from target C-statistic
    and outcome prevalence per Riley et al. (Stat Med 2021) / Austin & Steyerberg (2012).

    Given target c-statistic c in (0.5, 1.0) and outcome prevalence phi in (0, 1),
    under the two-component normal mixture model:
      - Non-events (Y=0): LP ~ N(0, 1)
      - Events (Y=1): LP ~ N(mu1, 1), where mu1 = sqrt(2) * Phi^(-1)(c)
      - The true log-odds ratio of LP is beta1 = mu1.

    Uses deterministic Gauss-Hermite quadrature (deg=64) to:
      1. Solve for intercept alpha0 via Brent's method such that:
         (1 - phi) * E[expit(alpha0 + mu1 * Z)] + phi * E[expit(alpha0 + mu1 * (Z + mu1))] = phi
      2. Compute expected model log-likelihood:
         E[ell_model] = (1 - phi) * E[ln(1 - expit(alpha0 + mu1 * Z))] + phi * E[ln(expit(alpha0 + mu1 * (Z + mu1)))]
      3. Compute expected null log-likelihood:
         E[ell_null] = phi * ln(phi) + (1 - phi) * ln(1 - phi)
      4. Return Cox-Snell R2 = 1 - exp(-2 * (E[ell_model] - E[ell_null]))
    """
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

    # 1. Solve for alpha0 via Brent's method
    def f_alpha(alpha0: float) -> float:
        e0 = inv_sqrt_pi * float(np.sum(weights * expit(alpha0 + mu1 * z)))
        e1 = inv_sqrt_pi * float(np.sum(weights * expit(alpha0 + mu1 * (z + mu1))))
        return (1.0 - phi) * e0 + phi * e1 - phi

    alpha0 = float(opt.brentq(f_alpha, -50.0, 50.0))

    # 2. Compute expected model log-likelihood using numerically stable log_expit
    u0 = alpha0 + mu1 * z
    u1 = alpha0 + mu1 * (z + mu1)
    ll_0 = inv_sqrt_pi * float(np.sum(weights * log_expit(-u0)))
    ll_1 = inv_sqrt_pi * float(np.sum(weights * log_expit(u1)))
    ll_model = (1.0 - phi) * ll_0 + phi * ll_1

    # 3. Compute expected null log-likelihood
    ll_null = phi * math.log(phi) + (1.0 - phi) * math.log(1.0 - phi)

    # 4. Return Cox-Snell R2 bounded by (0, R2_max)
    r2_cs = 1.0 - math.exp(-2.0 * (ll_model - ll_null))
    r2_max = 1.0 - math.exp(2.0 * ll_null)

    return float(min(r2_max - 1e-6, max(1e-6, r2_cs)))


approximate_cox_snell_r2_from_c_stat = _approximate_cox_snell_r2_from_c_stat


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
    """
    Calculate minimum required sample size for developing a multivariable clinical
    prediction model for binary outcomes per Riley et al. (BMJ 2020; 368:m441).

    Parameters:
        n_parameters: Number of candidate predictor parameters (excluding intercept).
        prevalence: Anticipated outcome prevalence / proportion (0 < phi < 1).
        c_statistic: Anticipated C-statistic / AUROC (0.5 < c < 1.0).
        cox_snell_r2: Anticipated Cox-Snell R2 (0 < R2_CS < R2_max).
        nagelkerke_r2: Anticipated Nagelkerke R2 (0 < R2_Nag < 1.0).
        shrinkage_target: Target uniform shrinkage factor S (default: 0.90).
        intercept_margin: Margin of error delta for overall intercept/risk (default: 0.05).
        optimism_margin: Allowed difference in apparent vs adjusted R2 (default: 0.05).
        mape_target: Target Mean Absolute Prediction Error (optional, van Smeden 2019).

    Returns:
        dict: Required sample size, events, EPP, and criteria breakdown.
    """
    if n_parameters < 1:
        raise ValueError(f"n_parameters must be >= 1, got {n_parameters}")
    if not (0.0 < prevalence < 1.0):
        raise ValueError(f"prevalence must be strictly in (0, 1), got {prevalence}")
    if not (0.5 < shrinkage_target < 1.0):
        raise ValueError(
            f"shrinkage_target must be strictly in (0.5, 1.0), got {shrinkage_target}"
        )
    if not (0.0 < intercept_margin < 1.0):
        raise ValueError(
            f"intercept_margin must be strictly in (0, 1), got {intercept_margin}"
        )
    if not (0.0 < optimism_margin < 1.0):
        raise ValueError(
            f"optimism_margin must be strictly in (0, 1), got {optimism_margin}"
        )

    phi = float(prevalence)
    p_params = int(n_parameters)
    s_target = float(shrinkage_target)

    # R2_max for binary logistic model
    # R2_max = 1 - [phi^phi * (1-phi)^(1-phi)]^2
    ll_null = phi * math.log(phi) + (1.0 - phi) * math.log(1.0 - phi)
    r2_max = 1.0 - math.exp(2.0 * ll_null)

    # Determine Cox-Snell R2
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
        raise ValueError(
            "Must provide at least one model strength measure: c_statistic, cox_snell_r2, or nagelkerke_r2."
        )

    # 1. Criterion 1: Precise estimation of overall intercept / risk
    # n1 = ceil((1.96 / delta)^2 * phi * (1 - phi))
    n_crit1 = int(math.ceil(((1.96 / intercept_margin) ** 2) * phi * (1.0 - phi)))

    # 2. Criterion 2: Mean Absolute Prediction Error (MAPE, van Smeden 2019)
    # RvS-2: n = exp((-0.508 + 0.259*ln(phi) + 0.504*ln(p) - ln(m)) / 0.544)
    target_m = (
        mape_target if mape_target is not None else (0.05 if phi <= 0.5 else None)
    )
    if target_m is not None:
        min_phi = min(phi, 1.0 - phi)
        exponent = (
            -0.508
            + 0.259 * math.log(min_phi)
            + 0.504 * math.log(p_params)
            - math.log(target_m)
        ) / 0.544
        n_crit2 = int(math.ceil(math.exp(exponent)))
    else:
        n_crit2 = None

    # 3. Criterion 3: Uniform shrinkage factor (overfitting minimization)
    # n3 = ceil(P / ((S - 1) * ln(1 - R2_CS / S)))
    arg_s = 1.0 - cs_r2 / s_target
    if arg_s <= 0.0:
        raise ValueError(
            f"Target shrinkage factor ({s_target:.2f}) must be strictly greater than Cox-Snell R2 ({cs_r2:.4f}). "
            f"Increase shrinkage_target to at least {min(0.99, cs_r2 + 0.05):.2f}."
        )
    denom_s = (s_target - 1.0) * math.log(arg_s)
    if denom_s <= 0:
        n_crit3 = 1000000
    else:
        n_crit3 = int(math.ceil(p_params / denom_s))

    # 4. Criterion 4: Small optimism in R2 (apparent vs adjusted)
    # S_req = R2_CS / (R2_CS + delta_R2 * R2_max)
    s_req = cs_r2 / (cs_r2 + optimism_margin * r2_max)
    arg_opt = 1.0 - cs_r2 / s_req
    if s_req >= 1.0 or arg_opt <= 0.0:
        n_crit4 = n_crit3
    else:
        denom_opt = (s_req - 1.0) * math.log(arg_opt)
        if denom_opt <= 0:
            n_crit4 = n_crit3
        else:
            n_crit4 = int(math.ceil(p_params / denom_opt))

    # Binding criterion & overall sample size
    eval_dict: dict[str, int] = {
        "Criterion 1 (Intercept)": n_crit1,
        "Criterion 3 (Shrinkage)": n_crit3,
        "Criterion 4 (Optimism)": n_crit4,
    }
    if mape_target is not None and n_crit2 is not None:
        eval_dict["Criterion 2 (MAPE)"] = n_crit2

    n_total = max(eval_dict.values())
    binding_crit = max(eval_dict, key=eval_dict.get)  # type: ignore[arg-type]
    n_events = int(math.ceil(n_total * phi))
    epp = float(n_events / p_params)

    return {
        "n_total": n_total,
        "n_events": n_events,
        "n_parameters": p_params,
        "prevalence": phi,
        "events_per_parameter": epp,
        "cox_snell_r2": float(cs_r2),
        "nagelkerke_r2": float(cs_r2 / r2_max),
        "r2_max": float(r2_max),
        "shrinkage_target": s_target,
        "binding_criterion": binding_crit,
        "criteria": {
            "criterion1_intercept": {
                "n": n_crit1,
                "margin": intercept_margin,
                "description": "Precise estimation of overall risk/intercept",
            },
            "criterion2_mape": {
                "n": n_crit2,
                "target_mape": target_m,
                "description": "Mean Absolute Prediction Error (van Smeden 2019)",
            },
            "criterion3_shrinkage": {
                "n": n_crit3,
                "target_shrinkage": s_target,
                "description": f"Minimizing overfitting (shrinkage >= {s_target:.2f})",
            },
            "criterion4_optimism": {
                "n": n_crit4,
                "optimism_margin": optimism_margin,
                "s_required": float(s_req),
                "description": "Small difference in apparent and adjusted R2",
            },
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
    """
    Calculate required sample size for testing diagnostic AUROC against a null value
    per Obuchowski (1998, 2004) and Hanley & McNeil (1982).

    Parameters:
        auroc_expected: Anticipated AUROC of the diagnostic test (theta_1).
        auroc_null: Null hypothesis AUROC (theta_0, default: 0.50).
        prevalence: Anticipated case proportion in clinical cohort (0 < phi < 1).
        alpha: Type I error rate (default: 0.05).
        power: Statistical power 1 - beta (default: 0.80).
        two_sided: Whether to conduct a two-sided hypothesis test (default: True).

    Returns:
        dict: Required cases (n_cases), controls (n_controls), total N, and design details.
    """
    theta1 = float(auroc_expected)
    theta0 = float(auroc_null)
    phi = float(prevalence)

    if not (0.50 <= theta0 < theta1 < 1.0):
        raise ValueError(
            f"Expected AUROC ({theta1}) must be strictly greater than null AUROC ({theta0}) and < 1.0."
        )
    if not (0.0 < phi < 1.0):
        raise ValueError(f"prevalence must be strictly in (0, 1), got {phi}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be strictly in (0, 1), got {alpha}")
    if not (0.0 < power < 1.0):
        raise ValueError(f"power must be strictly in (0, 1), got {power}")
    if power <= alpha:
        raise ValueError(
            f"power ({power}) must be strictly greater than alpha ({alpha})"
        )

    # Control-to-case ratio R = (1 - phi) / phi
    r_ratio = (1.0 - phi) / phi

    def _calc_v(theta: float) -> float:
        q1 = theta / (2.0 - theta)
        q2 = (2.0 * (theta**2)) / (1.0 + theta)
        return float(
            theta * (1.0 - theta) + (q1 - theta**2) + (q2 - theta**2) / r_ratio
        )

    v0 = _calc_v(theta0)
    v1 = _calc_v(theta1)

    z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0 if two_sided else 1.0 - alpha))
    z_beta = float(stats.norm.ppf(power))

    # Obuchowski / Hanley-McNeil required number of cases:
    # N_cases = (z_alpha * sqrt(V0) + z_beta * sqrt(V1))^2 / (theta1 - theta0)^2
    numerator = (z_alpha * math.sqrt(v0) + z_beta * math.sqrt(v1)) ** 2
    denominator = (theta1 - theta0) ** 2
    n_cases_req = numerator / denominator

    n_cases = int(math.ceil(n_cases_req))
    n_controls = int(math.ceil(n_cases * r_ratio))
    n_total = n_cases + n_controls

    return {
        "n_cases": n_cases,
        "n_controls": n_controls,
        "total_n": n_total,
        "n_total": n_total,
        "prevalence": phi,
        "auroc_expected": theta1,
        "auroc_null": theta0,
        "alpha": alpha,
        "power": power,
        "two_sided": two_sided,
        "control_to_case_ratio": r_ratio,
        "allocation_ratio_controls_to_cases": r_ratio,
        "variance_null": v0,
        "variance_expected": v1,
    }

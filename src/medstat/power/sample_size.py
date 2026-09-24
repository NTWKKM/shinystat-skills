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

import scipy.stats as stats
import statsmodels.stats.power as smp
import statsmodels.stats.proportion as smprop


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

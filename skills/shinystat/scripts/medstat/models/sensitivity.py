"""
Sensitivity Analysis for Unmeasured Confounding & Robustness.

Implements E-value formulas (VanderWeele & Ding 2017) and non-parametric
bootstrap confidence intervals.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np

from medstat.logging import get_logger

logger = get_logger(__name__)


def calculate_e_value(
    estimate: float,
    lower: float | None = None,
    upper: float | None = None,
    estimate_type: Literal["RR", "OR", "HR"] = "RR",
    rare_outcome: bool = False,
) -> dict[str, Any]:
    """
    Calculate E-value for a risk ratio (RR), odds ratio (OR), or hazard ratio (HR).

    The E-value defines the minimum strength of association on the risk ratio scale
    that an unmeasured confounder must have with both the treatment and the outcome
    to fully explain away the observed treatment-outcome association.

    Formula:
        E-value = RR + sqrt(RR * (RR - 1))
    """
    if estimate is None or not np.isfinite(estimate):
        raise ValueError("Estimate must be a finite numeric value.")

    if estimate <= 0:
        raise ValueError("Estimate must be strictly positive (> 0).")

    original_estimate = float(estimate)
    est = float(estimate)
    l_bound = (
        float(lower)
        if (lower is not None and np.isfinite(lower) and lower > 0)
        else None
    )
    u_bound = (
        float(upper)
        if (upper is not None and np.isfinite(upper) and upper > 0)
        else None
    )

    # Conversion for OR/HR to approximate RR for common outcomes
    if not rare_outcome:
        if estimate_type == "OR":
            est = np.sqrt(est)
            if l_bound is not None:
                l_bound = np.sqrt(l_bound)
            if u_bound is not None:
                u_bound = np.sqrt(u_bound)
        elif estimate_type == "HR":

            def _hr_to_rr(hr: float) -> float:
                if hr <= 0:
                    return hr
                return float(
                    (1.0 - 0.5 ** np.sqrt(hr)) / (1.0 - 0.5 ** np.sqrt(1.0 / hr))
                )

            est = _hr_to_rr(est)
            if l_bound is not None:
                l_bound = _hr_to_rr(l_bound)
            if u_bound is not None:
                u_bound = _hr_to_rr(u_bound)

    # Invert protective effects (RR < 1)
    if est < 1.0:
        est_prime = 1.0 / est
        # Limit closest to null is 1 / upper
        l_prime = 1.0 / u_bound if (u_bound is not None and u_bound > 0) else None
    else:
        est_prime = est
        l_prime = l_bound

    def _e_val(val: float | None) -> float:
        if val is None or val <= 1.0:
            return 1.0
        return float(val + np.sqrt(val * (val - 1.0)))

    e_est = _e_val(est_prime)
    e_ci = _e_val(l_prime)

    return {
        "original_estimate": original_estimate,
        "estimate_type": estimate_type,
        "rare_outcome": rare_outcome,
        "e_value_estimate": round(e_est, 3),
        "e_value_ci_limit": round(e_ci, 3),
        "interpretation": f"An unmeasured confounder associated with both the exposure and outcome by a risk ratio of at least {e_est:.2f}-fold each could explain away the estimate, but weaker confounding could not.",
    }


def bootstrap_confidence_interval(
    data: np.ndarray | list[float],
    statistic_func: Callable[[np.ndarray], float],
    n_iterations: int = 2000,
    alpha: float = 0.05,
    method: Literal["percentile", "bca", "basic"] = "percentile",
    random_seed: int | None = 42,
) -> dict[str, Any]:
    """
    Calculate bootstrap confidence intervals for any scalar statistic.
    """
    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)

    if n < 2:
        raise ValueError("Need at least 2 non-missing observations for bootstrap.")

    rng = np.random.default_rng(random_seed)
    boot_stats = np.empty(n_iterations)

    for i in range(n_iterations):
        sample = rng.choice(arr, size=n, replace=True)
        boot_stats[i] = statistic_func(sample)

    point_est = float(statistic_func(arr))
    se_boot = float(np.std(boot_stats, ddof=1))

    reported_method = method
    if method == "percentile":
        low = float(np.percentile(boot_stats, 100 * (alpha / 2.0)))
        high = float(np.percentile(boot_stats, 100 * (1.0 - alpha / 2.0)))
    elif method == "basic":
        q_low = float(np.percentile(boot_stats, 100 * (alpha / 2.0)))
        q_high = float(np.percentile(boot_stats, 100 * (1.0 - alpha / 2.0)))
        low = 2 * point_est - q_high
        high = 2 * point_est - q_low
    else:  # bca fallback to percentile if acceleration fails
        logger.warning(
            "BCa bootstrap interval requested but acceleration not computed; falling back to percentile method."
        )
        reported_method = "percentile"
        low = float(np.percentile(boot_stats, 100 * (alpha / 2.0)))
        high = float(np.percentile(boot_stats, 100 * (1.0 - alpha / 2.0)))

    return {
        "point_estimate": point_est,
        "ci_lower": low,
        "ci_upper": high,
        "se_bootstrap": se_boot,
        "method": reported_method,
        "n_iterations": n_iterations,
    }

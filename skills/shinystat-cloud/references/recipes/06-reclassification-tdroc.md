# Risk Reclassification & Time-Dependent ROC Recipes (Cloud Sandbox)

> **Scope:** Recipes 16–17: continuous / categorical NRI and IDI, and cumulative/dynamic IPCW time-dependent ROC (AUROC(t)).

---

## 16. Diagnostic Risk Reclassification: Continuous & Categorical NRI and IDI

Computes Pencina et al. (2008, 2011) Net Reclassification Improvement (Continuous and Categorical NRI) and Integrated Discrimination Improvement (IDI).
Enforces clinical guardrails: checks probabilities lie in $[0, 1]$, and reports Event NRI and Non-event NRI separately per Kerr et al. (2014) guidelines.

```python
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats


def _wald_nri(estimate: float, var: float, n: int) -> tuple[float, float, float]:
    """Return (se, z, p) for an NRI component; NaN when the Wald variance is degenerate.

    A zero variance arises when n < 2 or every subject moves the same way (p_up or
    p_down equal to 1, or no movement at all); the normal approximation is then undefined.
    """
    if n < 2 or not np.isfinite(var) or var <= 0.0:
        return float("nan"), float("nan"), float("nan")
    se = float(np.sqrt(var))
    z = float(estimate / se)
    return se, z, float(2.0 * stats.norm.sf(abs(z)))


def compute_reclassification_metrics(
    y_true: pd.Series | np.ndarray,
    p_old: pd.Series | np.ndarray,
    p_new: pd.Series | np.ndarray,
    thresholds: list[float] | None = None,
    n_bootstrap: int = 0,
    random_state: int = 42,
) -> dict[str, Any]:
    """Calculate Continuous NRI, Categorical NRI, and IDI with asymptotic and bootstrap CIs."""
    y_t = np.asarray(y_true, dtype=float)
    p_0 = np.asarray(p_old, dtype=float)
    p_1 = np.asarray(p_new, dtype=float)

    if len(y_t) != len(p_0) or len(y_t) != len(p_1):
        raise ValueError("y_true, p_old, and p_new must have identical lengths.")

    valid = ~(np.isnan(y_t) | np.isnan(p_0) | np.isnan(p_1))
    y_valid = y_t[valid]
    if len(y_valid) == 0:
        raise ValueError("No complete cases remaining after removing missing values.")
    if not np.all(np.isin(y_valid, [0.0, 1.0])):
        raise ValueError("y_true must contain strictly binary values (0 and 1).")
    if len(np.unique(y_valid)) < 2:
        raise ValueError("y_true must contain both event classes (0 and 1).")
    y_t = y_valid.astype(int)
    p_0 = p_0[valid]
    p_1 = p_1[valid]

    # Miscalibration Guard
    if (
        np.any(p_0 < -1e-7)
        or np.any(p_0 > 1.0 + 1e-7)
        or np.any(p_1 < -1e-7)
        or np.any(p_1 > 1.0 + 1e-7)
    ):
        raise ValueError(
            "NRI and IDI require calibrated probabilities in [0, 1]. Received values outside this range."
        )

    p_0 = np.clip(p_0, 0.0, 1.0)
    p_1 = np.clip(p_1, 0.0, 1.0)

    unique_y = np.unique(y_t)
    if len(unique_y) < 2 or not set(unique_y).issubset({0, 1}):
        raise ValueError("y_true must contain both 0 and 1 classes.")

    n_events = int(np.sum(y_t == 1))
    n_nonevents = int(np.sum(y_t == 0))
    n_total = len(y_t)

    # 1. Continuous NRI
    up_cont = p_1 > p_0 + 1e-12
    down_cont = p_1 < p_0 - 1e-12

    mask_1 = y_t == 1
    mask_0 = y_t == 0

    n_up_e = int(np.sum(mask_1 & up_cont))
    n_down_e = int(np.sum(mask_1 & down_cont))
    p_up_e = n_up_e / n_events if n_events > 0 else 0.0
    p_down_e = n_down_e / n_events if n_events > 0 else 0.0
    nri_e_cont = p_up_e - p_down_e
    var_e_cont = (
        (p_up_e + p_down_e - nri_e_cont**2) / n_events if n_events > 0 else 0.0
    )
    se_e_cont, z_e_cont, p_val_e_cont = _wald_nri(nri_e_cont, var_e_cont, n_events)

    n_up_ne = int(np.sum(mask_0 & up_cont))
    n_down_ne = int(np.sum(mask_0 & down_cont))
    p_up_ne = n_up_ne / n_nonevents if n_nonevents > 0 else 0.0
    p_down_ne = n_down_ne / n_nonevents if n_nonevents > 0 else 0.0
    nri_ne_cont = p_down_ne - p_up_ne
    var_ne_cont = (
        (p_down_ne + p_up_ne - nri_ne_cont**2) / n_nonevents
        if n_nonevents > 0
        else 0.0
    )
    se_ne_cont, z_ne_cont, p_val_ne_cont = _wald_nri(
        nri_ne_cont, var_ne_cont, n_nonevents
    )

    nri_total_cont = nri_e_cont + nri_ne_cont
    # Total inference requires both components to be estimable.
    var_total_cont = (
        var_e_cont + var_ne_cont
        if np.isfinite(se_e_cont) and np.isfinite(se_ne_cont)
        else float("nan")
    )
    se_total_cont, z_total_cont, p_val_total_cont = _wald_nri(
        nri_total_cont, var_total_cont, n_total
    )

    z_crit = 1.959963984540054
    res_nri_cont = {
        "total": {
            "estimate": float(nri_total_cont),
            "se": se_total_cont,
            "ci_lower": float(nri_total_cont - z_crit * se_total_cont),
            "ci_upper": float(nri_total_cont + z_crit * se_total_cont),
            "z_statistic": z_total_cont,
            "p_value": p_val_total_cont,
        },
        "events": {
            "estimate": float(nri_e_cont),
            "se": se_e_cont,
            "ci_lower": float(nri_e_cont - z_crit * se_e_cont),
            "ci_upper": float(nri_e_cont + z_crit * se_e_cont),
            "z_statistic": z_e_cont,
            "p_value": p_val_e_cont,
            "n_up": n_up_e,
            "n_down": n_down_e,
            "p_up": float(p_up_e),
            "p_down": float(p_down_e),
        },
        "nonevents": {
            "estimate": float(nri_ne_cont),
            "se": se_ne_cont,
            "ci_lower": float(nri_ne_cont - z_crit * se_ne_cont),
            "ci_upper": float(nri_ne_cont + z_crit * se_ne_cont),
            "z_statistic": z_ne_cont,
            "p_value": p_val_ne_cont,
            "n_up": n_up_ne,
            "n_down": n_down_ne,
            "p_up": float(p_up_ne),
            "p_down": float(p_down_ne),
        },
    }

    # 2. Categorical NRI
    res_nri_cat = None
    cat_old = None
    cat_new = None
    sorted_thresh = None
    cat_labels = None
    k_bins = 0

    if thresholds is not None and len(thresholds) > 0:
        sorted_thresh = sorted(list({float(t) for t in thresholds}))
        if any(not np.isfinite(t) or t <= 0.0 or t >= 1.0 for t in sorted_thresh):
            raise ValueError(
                "Categorical NRI thresholds must be finite and strictly within (0, 1)."
            )

        cat_old = np.digitize(p_0, sorted_thresh, right=False)
        cat_new = np.digitize(p_1, sorted_thresh, right=False)
        k_bins = len(sorted_thresh) + 1

        cat_labels = []
        for i in range(k_bins):
            if i == 0:
                cat_labels.append(f"< {sorted_thresh[0]:.2g}")
            elif i == k_bins - 1:
                cat_labels.append(f"≥ {sorted_thresh[-1]:.2g}")
            else:
                cat_labels.append(f"{sorted_thresh[i-1]:.2g} to <{sorted_thresh[i]:.2g}")

        m_events = np.zeros((k_bins, k_bins), dtype=int)
        m_nonevents = np.zeros((k_bins, k_bins), dtype=int)

        for i in range(n_total):
            if y_t[i] == 1:
                m_events[cat_old[i], cat_new[i]] += 1
            else:
                m_nonevents[cat_old[i], cat_new[i]] += 1
        m_combined = m_events + m_nonevents

        up_cat = cat_new > cat_old
        down_cat = cat_new < cat_old

        n_up_e_cat = int(np.sum(mask_1 & up_cat))
        n_down_e_cat = int(np.sum(mask_1 & down_cat))
        p_up_e_cat = n_up_e_cat / n_events if n_events > 0 else 0.0
        p_down_e_cat = n_down_e_cat / n_events if n_events > 0 else 0.0
        nri_e_cat = p_up_e_cat - p_down_e_cat
        var_e_cat = (
            (p_up_e_cat + p_down_e_cat - nri_e_cat**2) / n_events if n_events > 0 else 0.0
        )
        se_e_cat, z_e_cat, p_val_e_cat = _wald_nri(nri_e_cat, var_e_cat, n_events)

        n_up_ne_cat = int(np.sum(mask_0 & up_cat))
        n_down_ne_cat = int(np.sum(mask_0 & down_cat))
        p_up_ne_cat = n_up_ne_cat / n_nonevents if n_nonevents > 0 else 0.0
        p_down_ne_cat = n_down_ne_cat / n_nonevents if n_nonevents > 0 else 0.0
        nri_ne_cat = p_down_ne_cat - p_up_ne_cat
        var_ne_cat = (
            (p_down_ne_cat + p_up_ne_cat - nri_ne_cat**2) / n_nonevents
            if n_nonevents > 0
            else 0.0
        )
        se_ne_cat, z_ne_cat, p_val_ne_cat = _wald_nri(
            nri_ne_cat, var_ne_cat, n_nonevents
        )

        nri_total_cat = nri_e_cat + nri_ne_cat
        var_total_cat = (
            var_e_cat + var_ne_cat
            if np.isfinite(se_e_cat) and np.isfinite(se_ne_cat)
            else float("nan")
        )
        se_total_cat, z_total_cat, p_val_total_cat = _wald_nri(
            nri_total_cat, var_total_cat, n_total
        )

        pct_up = float(np.sum(m_combined[np.triu_indices(k_bins, k=1)]) / n_total * 100.0)
        pct_down = float(np.sum(m_combined[np.tril_indices(k_bins, k=-1)]) / n_total * 100.0)
        pct_same = float(np.trace(m_combined) / n_total * 100.0)

    # Bootstrap CIs for reclassification metrics
    boot_cat_low = None
    boot_cat_high = None
    boot_cont_low = None
    boot_cont_high = None
    boot_idi_low = None
    boot_idi_high = None

    if n_bootstrap > 0:
        rng = np.random.default_rng(random_state)
        boot_cat_vals = []
        boot_cont_vals = []
        boot_idi_vals = []
        idx_1 = np.where(mask_1)[0]
        idx_0 = np.where(mask_0)[0]
        has_cat = cat_old is not None and cat_new is not None

        for _ in range(n_bootstrap):
            b_1 = rng.choice(idx_1, size=len(idx_1), replace=True)
            b_0 = rng.choice(idx_0, size=len(idx_0), replace=True)
            b_idx = np.concatenate([b_1, b_0])
            p0_b = p_0[b_idx]
            p1_b = p_1[b_idx]
            y_b = y_t[b_idx]
            m1_b = y_b == 1
            m0_b = y_b == 0

            # Continuous NRI bootstrap
            nri_e_b = np.mean(p1_b[m1_b] > p0_b[m1_b] + 1e-12) - np.mean(
                p1_b[m1_b] < p0_b[m1_b] - 1e-12
            )
            nri_ne_b = np.mean(p1_b[m0_b] < p0_b[m0_b] - 1e-12) - np.mean(
                p1_b[m0_b] > p0_b[m0_b] + 1e-12
            )
            boot_cont_vals.append(nri_e_b + nri_ne_b)

            # IDI bootstrap
            idi_b = (np.mean(p1_b[m1_b]) - np.mean(p0_b[m1_b])) - (
                np.mean(p1_b[m0_b]) - np.mean(p0_b[m0_b])
            )
            boot_idi_vals.append(idi_b)

            # Categorical NRI bootstrap
            if has_cat:
                c_old = cat_old[b_idx]
                c_new = cat_new[b_idx]
                cat_e_b = np.mean(c_new[m1_b] > c_old[m1_b]) - np.mean(
                    c_new[m1_b] < c_old[m1_b]
                )
                cat_ne_b = np.mean(c_new[m0_b] < c_old[m0_b]) - np.mean(
                    c_new[m0_b] > c_old[m0_b]
                )
                boot_cat_vals.append(cat_e_b + cat_ne_b)

        boot_cont_low = float(np.percentile(boot_cont_vals, 2.5))
        boot_cont_high = float(np.percentile(boot_cont_vals, 97.5))
        boot_idi_low = float(np.percentile(boot_idi_vals, 2.5))
        boot_idi_high = float(np.percentile(boot_idi_vals, 97.5))
        if has_cat:
            boot_cat_low = float(np.percentile(boot_cat_vals, 2.5))
            boot_cat_high = float(np.percentile(boot_cat_vals, 97.5))

    if boot_cont_low is not None:
        res_nri_cont["total"]["ci_lower_boot"] = boot_cont_low
        res_nri_cont["total"]["ci_upper_boot"] = boot_cont_high

    if sorted_thresh is not None:
        res_nri_cat = {
            "thresholds": sorted_thresh,
            "categories": cat_labels,
            "total": {
                "estimate": float(nri_total_cat),
                "se": se_total_cat,
                "ci_lower": float(nri_total_cat - z_crit * se_total_cat),
                "ci_upper": float(nri_total_cat + z_crit * se_total_cat),
                "ci_lower_boot": boot_cat_low,
                "ci_upper_boot": boot_cat_high,
                "z_statistic": z_total_cat,
                "p_value": p_val_total_cat,
            },
            "events": {
                "estimate": float(nri_e_cat),
                "se": se_e_cat,
                "ci_lower": float(nri_e_cat - z_crit * se_e_cat),
                "ci_upper": float(nri_e_cat + z_crit * se_e_cat),
                "z_statistic": z_e_cat,
                "p_value": p_val_e_cat,
                "n_up": n_up_e_cat,
                "n_down": n_down_e_cat,
                "p_up": float(p_up_e_cat),
                "p_down": float(p_down_e_cat),
            },
            "nonevents": {
                "estimate": float(nri_ne_cat),
                "se": se_ne_cat,
                "ci_lower": float(nri_ne_cat - z_crit * se_ne_cat),
                "ci_upper": float(nri_ne_cat + z_crit * se_ne_cat),
                "z_statistic": z_ne_cat,
                "p_value": p_val_ne_cat,
                "n_up": n_up_ne_cat,
                "n_down": n_down_ne_cat,
                "p_up": float(p_up_ne_cat),
                "p_down": float(p_down_ne_cat),
            },
            "contingency_matrices": {
                "events": m_events.tolist(),
                "nonevents": m_nonevents.tolist(),
                "combined": m_combined.tolist(),
            },
            "reclassification_percentages": {
                "upward": pct_up,
                "downward": pct_down,
                "unchanged": pct_same,
            },
        }

    # 3. IDI
    mean_p1_events = float(np.mean(p_1[mask_1]))
    mean_p0_events = float(np.mean(p_0[mask_1]))
    mean_p1_nonevents = float(np.mean(p_1[mask_0]))
    mean_p0_nonevents = float(np.mean(p_0[mask_0]))

    idi_val = (mean_p1_events - mean_p0_events) - (mean_p1_nonevents - mean_p0_nonevents)
    slope_old = mean_p0_events - mean_p0_nonevents
    slope_new = mean_p1_events - mean_p1_nonevents
    rel_idi = float(idi_val / slope_old) if slope_old > 1e-12 else None

    delta_p = p_1 - p_0
    if n_events < 2 or n_nonevents < 2:
        # Sample variance of delta_p is not estimable in a group of one: no inference.
        se_idi = z_idi = p_val_idi = float("nan")
    else:
        var_delta_1 = float(np.var(delta_p[mask_1], ddof=1))
        var_delta_0 = float(np.var(delta_p[mask_0], ddof=1))
        se_idi = float(np.sqrt(var_delta_1 / n_events + var_delta_0 / n_nonevents))
        if se_idi > 0.0:
            z_idi = float(idi_val / se_idi)
            p_val_idi = float(2.0 * stats.norm.sf(abs(z_idi)))
        else:
            # Zero SE (including zero IDI): the normal test is undefined, so no inference.
            se_idi = z_idi = p_val_idi = float("nan")

    res_idi = {
        "estimate": float(idi_val),
        "se": se_idi,
        "ci_lower": float(idi_val - z_crit * se_idi),
        "ci_upper": float(idi_val + z_crit * se_idi),
        "z_statistic": z_idi,
        "p_value": p_val_idi,
        "relative_idi": rel_idi,
        "discrimination_slope_old": float(slope_old),
        "discrimination_slope_new": float(slope_new),
        "mean_p_old_events": mean_p0_events,
        "mean_p_old_nonevents": mean_p0_nonevents,
        "mean_p_new_events": mean_p1_events,
        "mean_p_new_nonevents": mean_p1_nonevents,
    }

    if boot_idi_low is not None:
        res_idi["ci_lower_boot"] = boot_idi_low
        res_idi["ci_upper_boot"] = boot_idi_high

    return {
        "nri_continuous": res_nri_cont,
        "nri_categorical": res_nri_cat,
        "idi": res_idi,
        "n_events": n_events,
        "n_nonevents": n_nonevents,
        "n_total": n_total,
    }
```

---

---

## 17. Time-Dependent ROC: Cumulative/Dynamic IPCW Estimator (AUROC(t))

Cumulative/Dynamic Inverse Probability of Censoring Weighting (IPCW) Time-Dependent ROC estimator (Uno et al., 2007; Heagerty et al., 2000).
Computes longitudinal discrimination trajectories, optimal Youden cutoffs, and bootstrap confidence intervals across clinical follow-up horizons.

```python
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
from lifelines import KaplanMeierFitter


def _lookup_censoring_survival_left_limit(
    kmf_censor: Any,
    times: np.ndarray | pd.Series | list[float] | float,
    clip_min: float = 1e-3,
) -> np.ndarray | float:
    """Evaluate left-limit censoring survival probability G_hat(t-) = P(C >= t)."""
    timeline = np.asarray(kmf_censor.survival_function_.index, dtype=float)
    vals = kmf_censor.survival_function_.iloc[:, 0].to_numpy(dtype=float)
    is_scalar = np.isscalar(times)
    t_arr = np.atleast_1d(np.asarray(times, dtype=float))
    raw_idx = np.searchsorted(timeline, t_arr, side="left") - 1
    idx = np.clip(raw_idx, 0, len(vals) - 1)
    # No timeline point strictly before t: G(t-) = G(0-) = 1.
    res = np.where(raw_idx < 0, 1.0, np.clip(vals[idx], clip_min, 1.0))
    return float(res[0]) if is_scalar else res


def _calc_ipcw_roc_point(
    durations: np.ndarray,
    events: np.ndarray,
    scores: np.ndarray,
    horizon: float,
    g_durations: np.ndarray,
    g_horizon: float,
    competing_event_code: int | None = None,
    return_curve: bool = True,
) -> dict[str, Any]:
    """Single horizon weighted IPCW AUROC calculation."""
    t = float(horizon)
    cases_mask = (durations <= t) & (events == 1)

    if competing_event_code is not None:
        competing_mask = (durations <= t) & (events == competing_event_code)
        ctrl_mask = (durations > t) | competing_mask
    else:
        ctrl_mask = durations > t

    n_cases = int(np.sum(cases_mask))
    n_controls = int(np.sum(ctrl_mask))
    n_censored = int(np.sum((durations <= t) & (events == 0)))

    if n_cases < 1 or n_controls < 1:
        res_empty: dict[str, Any] = {
            "horizon": t,
            "n_cases": n_cases,
            "n_controls": n_controls,
            "n_censored": n_censored,
            "auroc": float("nan"),
        }
        if return_curve:
            res_empty.update(
                {
                    "optimal_threshold": float("nan"),
                    "sensitivity": float("nan"),
                    "specificity": float("nan"),
                    "fpr": [0.0, 1.0],
                    "tpr": [0.0, 1.0],
                    "thresholds": [float("inf"), float("-inf")],
                }
            )
        return res_empty

    w_cases = 1.0 / np.clip(g_durations[cases_mask], 1e-3, 1.0)
    if competing_event_code is not None:
        w_ctrl = np.where(
            durations[ctrl_mask] > t,
            1.0 / max(g_horizon, 1e-3),
            1.0 / np.clip(g_durations[ctrl_mask], 1e-3, 1.0),
        )
    else:
        w_ctrl = np.full(n_controls, 1.0 / max(g_horizon, 1e-3))

    sc_cases = scores[cases_mask]
    sc_ctrl = scores[ctrl_mask]

    # O(N log N) fast concordance
    sort_idx = np.argsort(sc_ctrl)
    s0_sorted = sc_ctrl[sort_idx]
    w0_sorted = w_ctrl[sort_idx]
    cum_w0 = np.concatenate([[0.0], np.cumsum(w0_sorted)])

    idx_lt = np.searchsorted(s0_sorted, sc_cases, side="left")
    idx_gt = np.searchsorted(s0_sorted, sc_cases, side="right")
    concordance = cum_w0[idx_lt] + 0.5 * (cum_w0[idx_gt] - cum_w0[idx_lt])
    num = float(np.sum(w_cases * concordance))
    sum_w = float(np.sum(w_cases) * cum_w0[-1])
    auroc = float(num / sum_w) if sum_w > 0 else float("nan")

    if not return_curve:
        return {
            "horizon": t,
            "n_cases": n_cases,
            "n_controls": n_controls,
            "n_censored": n_censored,
            "auroc": auroc,
        }

    sum_w_cases = float(np.sum(w_cases))
    sum_w_ctrl = float(np.sum(w_ctrl))
    unique_th = np.sort(np.unique(np.concatenate([sc_cases, sc_ctrl])))[::-1]

    # Weighted counts at or above each threshold via sorted cumulative sums (O(N log N)).
    def _weight_at_or_above(sc: np.ndarray, w: np.ndarray, total: float) -> np.ndarray:
        order = np.argsort(sc, kind="stable")
        cum = np.concatenate([[0.0], np.cumsum(w[order])])
        below = cum[np.searchsorted(sc[order], unique_th, side="left")]
        return (total - below) / total if total > 0 else np.zeros(len(unique_th))

    tpr_list = [0.0] + _weight_at_or_above(sc_cases, w_cases, sum_w_cases).tolist()
    fpr_list = [0.0] + _weight_at_or_above(sc_ctrl, w_ctrl, sum_w_ctrl).tolist()
    th_list = [float("inf")] + [float(th) for th in unique_th]
    tpr_list.append(1.0)
    fpr_list.append(1.0)
    th_list.append(float("-inf"))

    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)
    best_idx = int(np.argmax(tpr_arr - fpr_arr))
    opt_th = th_list[best_idx]

    return {
        "horizon": t,
        "n_cases": n_cases,
        "n_controls": n_controls,
        "n_censored": n_censored,
        "auroc": auroc,
        "optimal_threshold": float(opt_th),
        "sensitivity": float(tpr_arr[best_idx]),
        "specificity": float(1.0 - fpr_arr[best_idx]),
        "fpr": fpr_list,
        "tpr": tpr_list,
        "thresholds": th_list,
    }


def compute_time_dependent_roc(
    durations: pd.Series | np.ndarray,
    events: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray | None = None,
    time_horizons: list[float] | np.ndarray | None = None,
    competing_event_code: int | None = None,
    n_bootstrap: int = 200,
    random_state: int = 42,
    scores_old: pd.Series | np.ndarray | None = None,
    scores_new: pd.Series | np.ndarray | None = None,
) -> dict[str, Any]:
    """Cumulative/Dynamic IPCW Time-Dependent ROC suite."""
    dur_arr = pd.Series(durations).to_numpy(dtype=float)
    ev_arr = pd.Series(events).to_numpy(dtype=float)
    if len(dur_arr) != len(ev_arr):
        raise ValueError(
            f"Length mismatch: durations ({len(dur_arr)}) and events ({len(ev_arr)}) must have identical lengths."
        )

    allowed_event_codes = (
        {0.0, 1.0}
        if competing_event_code is None
        else {0.0, 1.0, float(competing_event_code)}
    )

    def _validate_and_cast_events(ev_sub: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(ev_sub)):
            raise ValueError("Event values must be finite.")
        if not np.all(np.floor(ev_sub) == ev_sub):
            raise ValueError("Event values must be integer-valued codes.")
        if not np.all(np.isin(ev_sub, list(allowed_event_codes))):
            raise ValueError(f"Event values must be in {allowed_event_codes}.")
        return ev_sub.astype(int)

    is_comparison = False
    if scores_old is not None and scores_new is not None:
        is_comparison = True
        s_old = pd.Series(scores_old).to_numpy(dtype=float)
        s_new = pd.Series(scores_new).to_numpy(dtype=float)
        if len(s_old) != len(dur_arr) or len(s_new) != len(dur_arr):
            raise ValueError(
                f"Length mismatch: durations ({len(dur_arr)}), scores_old ({len(s_old)}), and scores_new ({len(s_new)}) must have identical lengths."
            )
        valid = ~(
            np.isnan(dur_arr)
            | np.isnan(ev_arr)
            | np.isnan(s_old)
            | np.isnan(s_new)
        )
        dur_arr, ev_arr, s_old, s_new = (
            dur_arr[valid],
            _validate_and_cast_events(ev_arr[valid]),
            s_old[valid],
            s_new[valid],
        )
        if np.isinf(dur_arr).any():
            raise ValueError("Time-dependent ROC durations contain infinite values.")
        if np.isinf(s_old).any() or np.isinf(s_new).any():
            raise ValueError("Time-dependent ROC scores contain infinite values.")
        s_arr = s_new
    elif isinstance(scores, dict):
        is_comparison = True
        s_old = pd.Series(scores.get("old", scores.get("baseline"))).to_numpy(dtype=float)
        s_new = pd.Series(scores.get("new", scores.get("updated"))).to_numpy(dtype=float)
        if len(s_old) != len(dur_arr) or len(s_new) != len(dur_arr):
            raise ValueError(
                f"Length mismatch: durations ({len(dur_arr)}), scores['old'] ({len(s_old)}), and scores['new'] ({len(s_new)}) must have identical lengths."
            )
        valid = ~(
            np.isnan(dur_arr)
            | np.isnan(ev_arr)
            | np.isnan(s_old)
            | np.isnan(s_new)
        )
        dur_arr, ev_arr, s_old, s_new = (
            dur_arr[valid],
            _validate_and_cast_events(ev_arr[valid]),
            s_old[valid],
            s_new[valid],
        )
        if np.isinf(dur_arr).any():
            raise ValueError("Time-dependent ROC durations contain infinite values.")
        if np.isinf(s_old).any() or np.isinf(s_new).any():
            raise ValueError("Time-dependent ROC scores contain infinite values.")
        s_arr = s_new
    else:
        if scores is None:
            raise ValueError("Must provide scores or both scores_old and scores_new.")
        s_arr = pd.Series(scores).to_numpy(dtype=float)
        if len(s_arr) != len(dur_arr):
            raise ValueError(
                f"Length mismatch: durations ({len(dur_arr)}) and scores ({len(s_arr)}) must have identical lengths."
            )
        valid = ~(np.isnan(dur_arr) | np.isnan(ev_arr) | np.isnan(s_arr))
        dur_arr, ev_arr, s_arr = (
            dur_arr[valid],
            _validate_and_cast_events(ev_arr[valid]),
            s_arr[valid],
        )
        if np.isinf(dur_arr).any():
            raise ValueError("Time-dependent ROC durations contain infinite values.")
        if np.isinf(s_arr).any():
            raise ValueError("Time-dependent ROC scores contain infinite values.")
        s_old, s_new = None, None

    if (dur_arr < 0).any():
        raise ValueError("Time-dependent ROC durations must be non-negative (>= 0).")

    unique_ev = set(np.unique(ev_arr))
    if competing_event_code is None:
        if not unique_ev.issubset({0, 1}):
            raise ValueError(
                f"Event indicator must contain binary values (0 = censored, 1 = event). Found: {sorted(list(unique_ev))}"
            )
    else:
        if competing_event_code in (0, 1):
            raise ValueError(
                f"competing_event_code cannot be 0 or 1 (0 is censored, 1 is primary event). Received: {competing_event_code}"
            )
        if not unique_ev.issubset({0, 1, competing_event_code}):
            raise ValueError(
                f"Event indicator contains values outside {{0, 1, {competing_event_code}}}: {sorted(list(unique_ev))}"
            )

    n_obs = len(dur_arr)
    if n_obs < 10:
        raise ValueError("Time-dependent ROC requires at least 10 valid observations.")
    max_t = float(np.max(dur_arr)) if n_obs > 0 else 0.0

    if time_horizons is None or len(time_horizons) == 0:
        ev_times = dur_arr[ev_arr == 1]
        time_horizons = [float(np.percentile(ev_times, p)) for p in [25, 50, 75]] if len(ev_times) >= 3 else [max_t * 0.5]
    sorted_horizons = sorted([float(h) for h in time_horizons])
    if any(h <= 0 for h in sorted_horizons):
        raise ValueError("Time horizons must contain positive follow-up times (> 0).")

    kmc = KaplanMeierFitter()
    kmc.fit(dur_arr, event_observed=(ev_arr == 0).astype(int))
    g_dur = _lookup_censoring_survival_left_limit(kmc, dur_arr)

    results_by_horizon = {}
    comparison_by_horizon = {} if is_comparison else None
    non_sparse_horizons = []

    for h in sorted_horizons:
        g_h = max(float(kmc.survival_function_at_times(h).iloc[0]), 1e-3)
        res_h = _calc_ipcw_roc_point(dur_arr, ev_arr, s_arr, h, g_dur, g_h, competing_event_code, return_curve=True)

        is_sparse = (h > max_t) or (res_h["n_cases"] < 5) or (res_h["n_controls"] < 5)
        res_h["warning"] = "Sparse risk set or beyond follow-up" if is_sparse else None

        if is_sparse:
            res_h["ci_lower"] = float("nan")
            res_h["ci_upper"] = float("nan")
            res_h["se"] = float("nan")
            res_h["ci_method"] = None
        else:
            non_sparse_horizons.append(h)
            if n_bootstrap == 0:
                auc_val = res_h["auroc"]
                if np.isnan(auc_val) or res_h["n_cases"] < 1 or res_h["n_controls"] < 1:
                    res_h["se"] = float("nan")
                    res_h["ci_lower"] = float("nan")
                    res_h["ci_upper"] = float("nan")
                    res_h["ci_method"] = None
                else:
                    q1 = auc_val / (2.0 - auc_val)
                    q2 = 2.0 * auc_val**2 / (1.0 + auc_val)
                    var_h = (
                        auc_val * (1.0 - auc_val)
                        + (res_h["n_cases"] - 1) * (q1 - auc_val**2)
                        + (res_h["n_controls"] - 1) * (q2 - auc_val**2)
                    ) / (res_h["n_cases"] * res_h["n_controls"])
                    se_h = float(np.sqrt(max(0.0, var_h)))
                    res_h["se"] = se_h
                    res_h["ci_lower"] = max(0.0, float(auc_val - 1.959963984540054 * se_h))
                    res_h["ci_upper"] = min(1.0, float(auc_val + 1.959963984540054 * se_h))
                    res_h["ci_method"] = "hanley_mcneil"
                    if res_h.get("n_censored", 0) > 0:
                        cens_msg = (
                            f"Horizon t={h} has {res_h['n_censored']} censored observations before t. "
                            "Hanley-McNeil confidence interval is an unweighted approximation that does not account for "
                            "IPCW censoring weight uncertainty; n_bootstrap >= 200 is recommended when censoring is present."
                        )
                        res_h["warning"] = (
                            f"{res_h['warning']}; {cens_msg}"
                            if res_h.get("warning")
                            else cens_msg
                        )

        results_by_horizon[h] = res_h

        if is_comparison and s_old is not None and s_new is not None:
            res_old_h = _calc_ipcw_roc_point(dur_arr, ev_arr, s_old, h, g_dur, g_h, competing_event_code, return_curve=False)
            delta = (
                res_h["auroc"] - res_old_h["auroc"]
                if not (np.isnan(res_h["auroc"]) or np.isnan(res_old_h["auroc"]))
                else float("nan")
            )
            comp_entry = {
                "horizon": h,
                "auroc_old": res_old_h["auroc"],
                "auroc_new": res_h["auroc"],
                "delta_auroc": delta,
                "n_cases": res_h["n_cases"],
                "n_controls": res_h["n_controls"],
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "se": float("nan"),
                "p_value": float("nan"),
                "ci_method": None,
            }
            comparison_by_horizon[h] = comp_entry

    if n_bootstrap > 0 and len(non_sparse_horizons) > 0:
        rng = np.random.default_rng(random_state)
        boot_indices = [rng.choice(n_obs, size=n_obs, replace=True) for _ in range(n_bootstrap)]
        boot_aucs_by_h = {h: [] for h in non_sparse_horizons}
        boot_deltas_by_h = {h: [] for h in non_sparse_horizons}

        for b_idx in boot_indices:
            d_b, e_b, s_b = dur_arr[b_idx], ev_arr[b_idx], s_arr[b_idx]
            km_b = KaplanMeierFitter().fit(d_b, event_observed=(e_b == 0).astype(int))
            g_dur_b = _lookup_censoring_survival_left_limit(km_b, d_b)

            for h in non_sparse_horizons:
                g_h_b = max(float(km_b.survival_function_at_times(h).iloc[0]), 1e-3)
                b_res = _calc_ipcw_roc_point(d_b, e_b, s_b, h, g_dur_b, g_h_b, competing_event_code, return_curve=False)
                if not np.isnan(b_res["auroc"]):
                    boot_aucs_by_h[h].append(b_res["auroc"])

                if is_comparison and s_old is not None and s_new is not None:
                    bo = _calc_ipcw_roc_point(d_b, e_b, s_old[b_idx], h, g_dur_b, g_h_b, competing_event_code, return_curve=False)
                    if not (np.isnan(bo["auroc"]) or np.isnan(b_res["auroc"])):
                        boot_deltas_by_h[h].append(b_res["auroc"] - bo["auroc"])

        for h in non_sparse_horizons:
            b_aucs = boot_aucs_by_h[h]
            if len(b_aucs) >= 10:
                results_by_horizon[h]["ci_lower"] = float(np.percentile(b_aucs, 2.5))
                results_by_horizon[h]["ci_upper"] = float(np.percentile(b_aucs, 97.5))
                results_by_horizon[h]["se"] = float(np.std(b_aucs, ddof=1))
                results_by_horizon[h]["ci_method"] = "bootstrap"
            else:
                results_by_horizon[h]["ci_lower"] = float("nan")
                results_by_horizon[h]["ci_upper"] = float("nan")
                results_by_horizon[h]["se"] = float("nan")
                results_by_horizon[h]["ci_method"] = None

            if is_comparison and comparison_by_horizon is not None:
                b_deltas = boot_deltas_by_h[h]
                d_auc = comparison_by_horizon[h]["delta_auroc"]
                if len(b_deltas) >= 10 and not np.isnan(d_auc):
                    se_d = float(np.std(b_deltas, ddof=1))
                    comparison_by_horizon[h]["se"] = se_d
                    comparison_by_horizon[h]["ci_lower"] = float(np.percentile(b_deltas, 2.5))
                    comparison_by_horizon[h]["ci_upper"] = float(np.percentile(b_deltas, 97.5))
                    # Zero bootstrap SE: the normal test is undefined, so no p-value.
                    comparison_by_horizon[h]["p_value"] = (
                        float(2.0 * stats.norm.sf(abs(d_auc / se_d)))
                        if se_d > 0
                        else float("nan")
                    )
                    comparison_by_horizon[h]["ci_method"] = "bootstrap"

    trajectory = {
        "horizons": sorted_horizons,
        "auroc": [results_by_horizon[h]["auroc"] for h in sorted_horizons],
        "ci_lower": [results_by_horizon[h]["ci_lower"] for h in sorted_horizons],
        "ci_upper": [results_by_horizon[h]["ci_upper"] for h in sorted_horizons],
    }

    out = {
        "time_horizons": sorted_horizons,
        "results_by_horizon": results_by_horizon,
        "trajectory": trajectory,
        "ci_method": "bootstrap" if n_bootstrap > 0 else "hanley_mcneil",
    }
    if is_comparison:
        out["comparison"] = comparison_by_horizon

    top_warnings = [
        results_by_horizon[h]["warning"]
        for h in sorted_horizons
        if results_by_horizon[h].get("warning")
    ]
    if top_warnings:
        out["warnings"] = top_warnings
        if n_bootstrap == 0 and any(
            results_by_horizon[h].get("ci_method") == "hanley_mcneil"
            and results_by_horizon[h].get("n_censored", 0) > 0
            for h in sorted_horizons
        ):
            out["warning"] = (
                "Hanley-McNeil confidence interval is an unweighted approximation that does not account for "
                "IPCW censoring weight uncertainty; n_bootstrap >= 200 is recommended when censoring is present."
            )
        else:
            out["warning"] = top_warnings[0]

    return out
```

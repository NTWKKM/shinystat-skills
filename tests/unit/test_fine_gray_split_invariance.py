"""
tests/unit/test_fine_gray_split_invariance.py: Drop-in tests for Recipe 15 (Fine-Gray entry-aware cluster sandwich).

Verifies interval splitting invariance, tie handling (score_gap), agreement with bootstrap,
and upstream lifelines cluster canary defect.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
import pytest
from lifelines import CoxPHFitter

from medstat.models.survival import (
    _fit_fine_gray,
    bootstrap_fine_gray_se,
    entry_aware_cluster_sandwich,
    expand_fine_gray,
    fit_competing_risks_regression,
)

sandwich = entry_aware_cluster_sandwich
fit_fg = _fit_fine_gray
expand = expand_fine_gray
COVS = ["x1", "x2"]


def _competing(n=400, seed=7, coarse=0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1, x2 = rng.binomial(1, 0.5, n), rng.normal(size=n)
    t1 = rng.exponential(1 / (0.15 * np.exp(-0.6 * x1 + 0.3 * x2)))
    t2 = rng.exponential(1 / (0.10 * np.exp(0.4 * x1)))
    c = rng.uniform(0, 12, n)
    t = np.minimum.reduce([t1, t2, c])
    e = np.where(t == t1, 1, np.where(t == t2, 2, 0))
    if coarse:
        t = np.maximum(np.ceil(t / coarse) * coarse, coarse)
    return pd.DataFrame({"t": t, "e": e, "x1": x1, "x2": x2})


def _one_row_per_patient(d: pd.DataFrame) -> pd.DataFrame:
    out = d[COVS].astype(float).copy()
    out["_id"], out["_start"], out["_stop"] = np.arange(len(d)), 0.0, d["t"].to_numpy()
    out["_event"], out["_weight"] = (d["e"] == 1).astype(int).to_numpy(), 1.0
    return out


def _split(df: pd.DataFrame, step: float) -> pd.DataFrame:
    """Re-express identical follow-up as consecutive (start, stop] pieces; the event stays in the last piece."""
    grid = np.arange(step, df["_stop"].max() + step, step)
    parts = []
    for lo, hi in zip(np.r_[0.0, grid], np.r_[grid, np.inf]):
        p = df[df["_stop"] > lo].copy()
        p["_start"] = lo
        cut = p["_stop"] > hi
        p.loc[cut, "_stop"], p.loc[cut, "_event"] = hi, 0
        parts.append(p)
    return pd.concat(parts, ignore_index=True)


def _beta_bread(df):
    fit = fit_fg(df, COVS, alpha=0.05)
    return (
        fit.params_.reindex(COVS).to_numpy(),
        fit.variance_matrix_.loc[COVS, COVS].to_numpy(),
    )


@pytest.fixture(scope="module")
def unsplit():
    one = _one_row_per_patient(_competing())
    fit = fit_fg(one, COVS, alpha=0.05)
    beta = fit.params_.reindex(COVS).to_numpy()
    bread = fit.variance_matrix_.loc[COVS, COVS].to_numpy()
    return one, beta, bread


@pytest.mark.parametrize("step", [4.0, 2.0, 1.0, 0.5, 0.1])
def test_sandwich_invariant_to_interval_splitting(unsplit, step):
    one, beta, bread = unsplit
    se_ref, gap_ref = sandwich(one, COVS, beta, bread)
    se_split, gap_split = sandwich(_split(one, step), COVS, beta, bread)
    np.testing.assert_allclose(se_split, se_ref, rtol=1e-9, atol=0)
    assert gap_split == pytest.approx(gap_ref, abs=1e-9)


def test_sandwich_invariant_split_at_exact_event_time(unsplit):
    """Verifies (start, stop] boundary condition when an interval is split exactly at an observed event time."""
    one, beta, bread = unsplit
    event_times = np.sort(one.loc[one["_event"] == 1, "_stop"].unique())
    cut_time = event_times[len(event_times) // 2]
    assert cut_time < one["_stop"].max()

    p1 = one.copy()
    cut_mask = p1["_stop"] > cut_time
    p1.loc[cut_mask, "_stop"] = cut_time
    p1.loc[cut_mask, "_event"] = 0
    p2 = one[cut_mask].copy()
    p2["_start"] = cut_time
    split = pd.concat([p1, p2], ignore_index=True)

    # Confirm split created intervals starting exactly at cut_time
    assert np.any(split["_start"] == cut_time)

    se_ref, gap_ref = sandwich(one, COVS, beta, bread)
    se_split, gap_split = sandwich(split, COVS, beta, bread)
    np.testing.assert_allclose(se_split, se_ref, rtol=1e-9, atol=0)
    assert gap_split == pytest.approx(gap_ref, abs=1e-9)


def test_refit_on_split_data_keeps_se(unsplit):
    one, _, _ = unsplit
    se_ref, _ = sandwich(one, COVS, *_beta_bread(one))
    split = _split(one, 0.5)
    se_split, _ = sandwich(split, COVS, *_beta_bread(split))
    np.testing.assert_allclose(se_split, se_ref, rtol=1e-3)


def test_matches_lifelines_robust_on_unsplit(unsplit):
    one, beta, bread = unsplit
    ref = CoxPHFitter().fit(
        one[COVS + ["_stop", "_event"]],
        duration_col="_stop",
        event_col="_event",
        robust=True,
    )
    se, _ = sandwich(one, COVS, beta, bread)
    np.testing.assert_allclose(
        se, ref.standard_errors_.reindex(COVS).to_numpy(), rtol=1e-3
    )


def test_lifelines_cluster_canary(unsplit):
    """Documents the upstream defect. When lifelines fixes it this skips: remove the workaround."""
    one, _, _ = unsplit
    split = _split(one, 0.25)
    se_one = (
        CoxPHFitter()
        .fit(
            one.drop(columns="_weight"),
            duration_col="_stop",
            entry_col="_start",
            event_col="_event",
            cluster_col="_id",
        )
        .standard_errors_["x1"]
    )
    se_split = (
        CoxPHFitter()
        .fit(
            split.drop(columns="_weight"),
            duration_col="_stop",
            entry_col="_start",
            event_col="_event",
            cluster_col="_id",
        )
        .standard_errors_["x1"]
    )
    if se_split / se_one < 1.05:
        pytest.skip(
            "lifelines cluster SE is now split-invariant; entry_aware_cluster_sandwich is no longer needed"
        )
    assert se_split / se_one > 1.05


def test_point_estimates_match_lifelines_cluster_fit():
    ex = expand(_competing(), "t", "e", COVS)
    ours = fit_fg(ex, COVS, alpha=0.05).params_.reindex(COVS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = (
            CoxPHFitter()
            .fit(
                ex,
                duration_col="_stop",
                entry_col="_start",
                event_col="_event",
                weights_col="_weight",
                cluster_col="_id",
            )
            .params_.reindex(COVS)
        )
    np.testing.assert_allclose(ours.to_numpy(), ref.to_numpy(), rtol=1e-8)


def test_sandwich_agrees_with_bootstrap():
    d = _competing()
    ex = expand(d, "t", "e", COVS)
    beta, bread = _beta_bread(ex)
    se, gap = sandwich(ex, COVS, beta, bread)
    boot = bootstrap_fine_gray_se(d, "t", "e", COVS, n_boot=100, seed=0)
    assert gap < 0.05
    ratio = se / boot
    assert np.all((ratio > 0.8) & (ratio < 1.2)), (
        ratio
    )  # bootstrap B=100 carries ~7% SE noise


def test_heavy_ties_raise_score_gap_and_warn():
    d = _competing(coarse=3.0)
    with pytest.warns(UserWarning, match="bootstrap"):
        out = fit_competing_risks_regression(d, "t", "e", COVS)
    assert out.attrs["fg_score_gap"] > 0.20


def test_fine_data_no_tie_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fit_competing_risks_regression(_competing(), "t", "e", COVS)
    assert not any("score_gap" in str(w.message) for w in caught)
    assert out.attrs["fg_se_method"] == "sandwich"


def test_regression_ci_is_not_inflated():
    d = _competing()
    out = fit_competing_risks_regression(d, "t", "e", COVS).set_index("covariate")
    lo, hi = (float(x) for x in re.findall(r"[\d.]+", out.loc["x1", "fg_95_ci"])[1:3])
    se_ci = np.log(hi / lo) / (2 * 1.959964)
    ex = expand(d, "t", "e", COVS)
    naive = fit_fg(ex, COVS, alpha=0.05).standard_errors_["x1"]
    assert (
        0.9 < se_ci / naive < 1.25
    )  # a cluster-sandwich bug would put this near 1.4 or higher


def test_invalid_se_method():
    with pytest.raises(ValueError, match="se_method"):
        fit_competing_risks_regression(
            _competing(), "t", "e", COVS, se_method="cluster_col"
        )


def test_bootstrap_invalid_n_boot():
    d = _competing()
    with pytest.raises(ValueError, match="n_boot must be at least 2"):
        bootstrap_fine_gray_se(d, "t", "e", COVS, n_boot=1)
    with pytest.raises(ValueError, match="n_boot must be at least 2"):
        bootstrap_fine_gray_se(d, "t", "e", COVS, n_boot=0)
    with pytest.raises(ValueError, match="n_boot must be at least 2"):
        fit_competing_risks_regression(
            d, "t", "e", COVS, se_method="bootstrap", n_boot=1
        )

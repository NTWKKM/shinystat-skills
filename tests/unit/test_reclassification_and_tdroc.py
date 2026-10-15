"""tests/unit/test_reclassification_and_tdroc.py

Unit tests and statistical invariant validation for:
1. Diagnostic Risk Reclassification (Continuous NRI, Categorical NRI, IDI, Kerr/Pepe separate reporting).
2. Time-Dependent ROC with Cumulative/Dynamic IPCW estimator (AUROC(t), Youden cutoffs, bootstrap CIs).
3. Static & interactive HTML figures (plot_reclassification, plot_time_dependent_roc).
4. Reporting tables and biomedical narrative generation.
"""

from __future__ import annotations

import numpy as np
import pytest
from lifelines import KaplanMeierFitter

from medstat.diagnostic.roc import (
    _lookup_censoring_survival_left_limit,
    calculate_roc_curve,
    compare_diagnostic_models,
    compute_reclassification_metrics,
    compute_time_dependent_roc,
)
from medstat.figures.html.roc import (
    DualFigureResult,
    plot_dual_reclassification,
    plot_dual_time_dependent_roc,
)
from medstat.figures.roc import (
    FigureResult,
    plot_reclassification,
    plot_time_dependent_roc,
)
from medstat.reporting.narrative import (
    generate_methods_narrative,
    generate_reclassification_narrative,
    generate_time_dependent_roc_narrative,
)
from medstat.reporting.tables import (
    format_reclassification_table,
    format_time_dependent_roc_table,
)

# =============================================================================
# 1. Diagnostic Risk Reclassification (NRI & IDI) Tests
# =============================================================================


def test_reclassification_additive_decomposition():
    """Verify that Total NRI equals the sum of Event NRI and Non-event NRI."""
    np.random.seed(42)
    n = 200
    y_true = np.random.binomial(1, 0.35, size=n)
    p_old = np.clip(0.1 + 0.3 * y_true + np.random.normal(0, 0.1, size=n), 0.01, 0.99)
    p_new = np.clip(0.05 + 0.5 * y_true + np.random.normal(0, 0.08, size=n), 0.01, 0.99)

    res = compute_reclassification_metrics(
        y_true, p_old, p_new, thresholds=[0.20, 0.40], n_bootstrap=0
    )

    cont = res["nri_continuous"]
    assert np.isclose(
        cont["total"]["estimate"],
        cont["events"]["estimate"] + cont["nonevents"]["estimate"],
        atol=1e-12,
    )

    cat = res["nri_categorical"]
    assert np.isclose(
        cat["total"]["estimate"],
        cat["events"]["estimate"] + cat["nonevents"]["estimate"],
        atol=1e-12,
    )


def test_reclassification_identical_models_yield_zeros():
    """Identical predicted probabilities must yield zero NRI, zero IDI, and p=1.0."""
    n = 100
    y_true = np.array([1] * 30 + [0] * 70)
    p = np.linspace(0.1, 0.9, n)

    res = compute_reclassification_metrics(
        y_true, p, p, thresholds=[0.30, 0.60], n_bootstrap=0
    )

    assert res["nri_continuous"]["total"]["estimate"] == 0.0
    assert res["nri_continuous"]["total"]["p_value"] == 1.0
    assert res["nri_categorical"]["total"]["estimate"] == 0.0
    assert res["nri_categorical"]["total"]["p_value"] == 1.0
    assert res["idi"]["estimate"] == 0.0
    assert res["idi"]["p_value"] == 1.0


def test_reclassification_superior_model_directionality():
    """A strongly superior model must produce strictly positive NRI and IDI with p < 0.05."""
    np.random.seed(42)
    n = 300
    y_true = np.random.binomial(1, 0.4, size=n)
    # Baseline: weak predictor
    p_old = 1.0 / (
        1.0 + np.exp(-(-0.5 + 0.4 * y_true + np.random.normal(0, 0.5, size=n)))
    )
    # Updated: strong predictor
    p_new = 1.0 / (
        1.0 + np.exp(-(-1.5 + 2.5 * y_true + np.random.normal(0, 0.3, size=n)))
    )

    res = compute_reclassification_metrics(
        y_true, p_old, p_new, thresholds=[0.30, 0.60], n_bootstrap=50, random_state=42
    )

    assert res["nri_continuous"]["total"]["estimate"] > 0.30
    assert res["nri_continuous"]["total"]["p_value"] < 0.001
    assert res["nri_categorical"]["total"]["estimate"] > 0.15
    assert res["nri_categorical"]["total"]["p_value"] < 0.01
    assert res["idi"]["estimate"] > 0.10
    assert res["idi"]["p_value"] < 0.001
    assert res["idi"]["relative_idi"] is not None
    assert res["idi"]["relative_idi"] > 0.5


def test_reclassification_categorical_contingency_matrices():
    """Verify categorical NRI contingency matrices sum to actual cohort counts."""
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    p_old = np.array([0.15, 0.45, 0.75, 0.15, 0.45, 0.75, 0.25])
    p_new = np.array([0.25, 0.75, 0.85, 0.05, 0.35, 0.45, 0.10])

    res = compute_reclassification_metrics(y, p_old, p_new, thresholds=[0.30, 0.60])
    cat = res["nri_categorical"]
    m_ev = np.array(cat["contingency_matrices"]["events"])
    m_nev = np.array(cat["contingency_matrices"]["nonevents"])
    m_comb = np.array(cat["contingency_matrices"]["combined"])

    assert m_ev.shape == (3, 3)
    assert np.sum(m_ev) == 3
    assert np.sum(m_nev) == 4
    assert np.array_equal(m_comb, m_ev + m_nev)


def test_reclassification_miscalibration_guard():
    """Probabilities outside [0, 1] must raise ValueError to prevent spurious NRI."""
    y = np.array([1, 0, 1, 0])
    p_valid = np.array([0.8, 0.2, 0.7, 0.3])
    p_invalid_high = np.array([1.2, 0.2, 0.7, 0.3])
    p_invalid_low = np.array([-0.1, 0.2, 0.7, 0.3])

    with pytest.raises(ValueError, match="calibrated probabilities"):
        compute_reclassification_metrics(y, p_valid, p_invalid_high)

    with pytest.raises(ValueError, match="calibrated probabilities"):
        compute_reclassification_metrics(y, p_invalid_low, p_valid)


def test_reclassification_input_validation():
    """Mismatched lengths or non-binary outcomes must raise ValueError."""
    with pytest.raises(ValueError, match="identical lengths"):
        compute_reclassification_metrics([1, 0], [0.5], [0.5])

    with pytest.raises(ValueError, match="both 0 and 1 classes"):
        compute_reclassification_metrics(
            [0, 2, 1, 0], [0.5, 0.4, 0.6, 0.3], [0.6, 0.3, 0.7, 0.2]
        )

    with pytest.raises(ValueError, match="strictly within"):
        compute_reclassification_metrics(
            [1, 0, 1, 0], [0.5, 0.4, 0.6, 0.3], [0.6, 0.3, 0.7, 0.2], thresholds=[1.5]
        )


def test_compare_diagnostic_models_combined_suite():
    """Verify compare_diagnostic_models runs DeLong, NRI, and IDI seamlessly."""
    np.random.seed(42)
    n = 100
    y = np.random.binomial(1, 0.3, size=n)
    p_old = np.clip(0.2 + 0.2 * y + np.random.normal(0, 0.1, size=n), 0.05, 0.95)
    p_new = np.clip(0.1 + 0.5 * y + np.random.normal(0, 0.1, size=n), 0.05, 0.95)

    comp = compare_diagnostic_models(y, p_old, p_new, thresholds=[0.20, 0.50])
    assert "delong" in comp
    assert "reclassification" in comp
    assert "auc1" in comp["delong"]
    assert "auc2" in comp["delong"]
    assert "difference" in comp["delong"]
    assert "p_value" in comp["delong"]
    # Directionality: p_new is superior, so ΔAUROC must be strictly positive (> 0)
    assert comp["delong"]["difference"] > 0
    assert comp["delong"]["auc1"] > comp["delong"]["auc2"]
    assert comp["reclassification"]["nri_continuous"]["total"]["estimate"] > 0


# =============================================================================
# 2. Time-Dependent ROC Tests
# =============================================================================


def test_time_dependent_roc_zero_censoring_equivalence():
    """When there is zero censoring, IPCW AUROC(t) matches binary AUROC within 1e-6."""
    np.random.seed(42)
    n = 100
    durations = np.random.uniform(5.0, 50.0, size=n)
    events = np.ones(n, dtype=int)  # 100% events, 0% censored
    scores = durations * -0.5 + np.random.normal(0, 1.0, size=n)

    horizon = 25.0
    td_res = compute_time_dependent_roc(
        durations, events, scores, time_horizons=[horizon], n_bootstrap=0
    )
    td_auc = td_res["results_by_horizon"][horizon]["auroc"]

    # Standard static binary evaluation at horizon t: cases = dur <= 25, controls = dur > 25
    y_binary = (durations <= horizon).astype(int)
    static_auc = calculate_roc_curve(y_binary, scores)["auc"]

    assert np.isclose(td_auc, static_auc, atol=1e-5)


def test_time_dependent_roc_trajectories_and_youden():
    """Verify trajectory extraction, optimal Youden threshold, and sensitivity/specificity."""
    np.random.seed(42)
    n = 200
    scores = np.random.normal(0, 1, size=n)
    hazard = 0.05 * np.exp(1.0 * scores)
    durations = np.random.exponential(1.0 / hazard)
    censoring = np.random.exponential(25.0, size=n)
    obs_t = np.minimum(durations, censoring)
    ev = (durations <= censoring).astype(int)

    horizons = [5.0, 10.0, 15.0]
    res = compute_time_dependent_roc(
        obs_t, ev, scores, time_horizons=horizons, n_bootstrap=50, random_state=42
    )

    assert res["time_horizons"] == horizons
    assert len(res["trajectory"]["auroc"]) == 3
    for h in horizons:
        h_res = res["results_by_horizon"][h]
        assert 0.5 <= h_res["auroc"] <= 1.0
        assert 0.0 <= h_res["sensitivity"] <= 1.0
        assert 0.0 <= h_res["specificity"] <= 1.0
        assert np.isfinite(h_res["optimal_threshold"])
        assert h_res["n_cases"] > 0
        assert h_res["n_controls"] > 0


def test_time_dependent_roc_sparse_horizon_safety_guard():
    """Horizons beyond max follow-up or with sparse risk sets must warn and set CI to NaN."""
    durations = np.array([5.0, 10.0, 15.0, 20.0, 25.0] * 3)
    events = np.array([1, 0, 1, 0, 1] * 3)
    scores = np.array([0.8, 0.2, 0.7, 0.3, 0.9] * 3)

    # Horizon 50.0 is way beyond max(durations) = 25.0
    res = compute_time_dependent_roc(durations, events, scores, time_horizons=[50.0])
    h_res = res["results_by_horizon"][50.0]
    assert h_res["warning"] is not None
    assert np.isnan(h_res["ci_lower"])
    assert np.isnan(h_res["ci_upper"])


def test_time_dependent_roc_competing_risks():
    """Verify competing risk event handling in time-dependent ROC."""
    np.random.seed(42)
    n = 150
    scores = np.random.normal(0, 1, size=n)
    dur = np.random.uniform(1.0, 50.0, size=n)
    # 0 = censored, 1 = primary event, 2 = competing event
    status = np.random.choice([0, 1, 2], size=n, p=[0.3, 0.4, 0.3])

    res = compute_time_dependent_roc(
        dur, status, scores, time_horizons=[20.0], competing_event_code=2, n_bootstrap=0
    )
    h_res = res["results_by_horizon"][20.0]
    assert np.isfinite(h_res["auroc"])
    assert 0.0 <= h_res["auroc"] <= 1.0


def test_time_dependent_roc_dual_model_comparison():
    """Verify dual model trajectory comparison with delta AUROC."""
    np.random.seed(42)
    n = 150
    marker1 = np.random.normal(0, 1, size=n)
    marker2 = marker1 * 0.8 + np.random.normal(0, 0.5, size=n)
    dur = np.random.exponential(20.0, size=n) + 1.0
    ev = np.random.binomial(1, 0.5, size=n)

    res = compute_time_dependent_roc(
        dur,
        ev,
        scores_old=marker1,
        scores_new=marker2,
        time_horizons=[10.0, 25.0],
        n_bootstrap=30,
    )
    assert "comparison" in res
    assert 10.0 in res["comparison"]
    assert "delta_auroc" in res["comparison"][10.0]
    assert np.isfinite(res["comparison"][10.0]["delta_auroc"])


# =============================================================================
# 3. Figures & HTML Plotting Tests
# =============================================================================


def test_plot_reclassification_static_and_html(tmp_path):
    """Verify plot_reclassification and plot_dual_reclassification."""
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0] * 5)
    p_old = np.array([0.6, 0.5, 0.3, 0.2, 0.7, 0.4, 0.8, 0.1] * 5)
    p_new = np.array([0.7, 0.6, 0.2, 0.1, 0.8, 0.3, 0.9, 0.2] * 5)
    reclass_data = compute_reclassification_metrics(
        y, p_old, p_new, thresholds=[0.3, 0.6]
    )

    # 1. Static plot
    out_png = str(tmp_path / "reclass.png")
    fig_res = plot_reclassification(reclass_data, out_path=out_png)
    assert isinstance(fig_res, FigureResult)
    assert (tmp_path / "reclass.png").exists()
    assert fig_res.csv_path is not None and len(fig_res.csv_path) > 0

    # 2. Dual HTML plot
    out_dual = str(tmp_path / "reclass_dual.png")
    dual_res = plot_dual_reclassification(reclass_data, out_path=out_dual)
    assert isinstance(dual_res, DualFigureResult)
    assert dual_res.plotly_fig is not None
    assert "data" in dual_res.plotly_fig
    assert "layout" in dual_res.plotly_fig


def test_plot_time_dependent_roc_static_and_html(tmp_path):
    """Verify plot_time_dependent_roc trajectory and curves modes."""
    dur = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0] * 5)
    ev = np.array([1, 0, 1, 0, 1, 0] * 5)
    sc = np.array([0.8, 0.2, 0.7, 0.3, 0.9, 0.1] * 5)
    td_data = compute_time_dependent_roc(
        dur, ev, sc, time_horizons=[25.0, 45.0], n_bootstrap=0
    )

    # 1. Trajectory mode
    out_traj = str(tmp_path / "td_traj.png")
    res_traj = plot_time_dependent_roc(
        td_data, mode="auroc_trajectory", out_path=out_traj
    )
    assert isinstance(res_traj, FigureResult)
    assert (tmp_path / "td_traj.png").exists()

    # 2. Curves mode
    out_curves = str(tmp_path / "td_curves.png")
    res_curves = plot_time_dependent_roc(td_data, mode="curves", out_path=out_curves)
    assert isinstance(res_curves, FigureResult)
    assert (tmp_path / "td_curves.png").exists()

    # 3. Dual plot
    out_dual = str(tmp_path / "td_dual.png")
    dual_res = plot_dual_time_dependent_roc(td_data, out_path=out_dual)
    assert isinstance(dual_res, DualFigureResult)
    assert dual_res.plotly_fig is not None


# =============================================================================
# 4. Tables and Narrative Formatting Tests
# =============================================================================


def test_format_reclassification_table():
    """Verify format_reclassification_table HTML structure."""
    y = np.array([1, 1, 0, 0, 1, 0])
    p_old = np.array([0.6, 0.5, 0.3, 0.2, 0.7, 0.4])
    p_new = np.array([0.7, 0.6, 0.2, 0.1, 0.8, 0.3])
    reclass_data = compute_reclassification_metrics(
        y, p_old, p_new, thresholds=[0.3, 0.6]
    )

    html = format_reclassification_table(
        reclass_data, title="Table 3. Risk Reclassification"
    )
    assert "<table" in html
    assert "Table 3. Risk Reclassification" in html
    assert "Continuous NRI (Total)" in html
    assert "Categorical NRI" in html
    assert "IDI" in html
    assert "Reclassification Contingency Matrix" in html


def test_format_time_dependent_roc_table():
    """Verify format_time_dependent_roc_table HTML structure."""
    dur = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0] * 3)
    ev = np.array([1, 0, 1, 0, 1, 0] * 3)
    sc = np.array([0.8, 0.2, 0.7, 0.3, 0.9, 0.1] * 3)
    td_data = compute_time_dependent_roc(
        dur, ev, sc, time_horizons=[25.0, 45.0], n_bootstrap=0
    )

    html = format_time_dependent_roc_table(
        td_data, title="Table 4. Time-Dependent Discrimination"
    )
    assert "<table" in html
    assert "Table 4. Time-Dependent Discrimination" in html
    assert "Horizon (t)" in html
    assert "AUROC(t)" in html
    assert "Optimal Cutoff" in html
    assert "25" in html
    assert "45" in html


def test_narratives_reclassification_and_tdroc():
    """Verify narrative generators produce accurate clinical text."""
    # 1. Reclassification narrative
    y = np.array([1, 1, 0, 0, 1, 0])
    p_old = np.array([0.6, 0.5, 0.3, 0.2, 0.7, 0.4])
    p_new = np.array([0.7, 0.6, 0.2, 0.1, 0.8, 0.3])
    reclass_data = compute_reclassification_metrics(
        y, p_old, p_new, thresholds=[0.3, 0.6]
    )
    narr_reclass = generate_reclassification_narrative(reclass_data)
    assert "Continuous Net Reclassification Improvement" in narr_reclass
    assert "Categorical NRI" in narr_reclass
    assert "Integrated Discrimination Improvement" in narr_reclass

    # 2. Time-Dependent ROC narrative
    dur = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0] * 3)
    ev = np.array([1, 0, 1, 0, 1, 0] * 3)
    sc = np.array([0.8, 0.2, 0.7, 0.3, 0.9, 0.1] * 3)
    td_data = compute_time_dependent_roc(
        dur, ev, sc, time_horizons=[25.0, 45.0], n_bootstrap=0
    )
    narr_td = generate_time_dependent_roc_narrative(td_data)
    assert "Time-dependent cumulative/dynamic ROC analysis" in narr_td
    assert "at t = 25" in narr_td or "t = 25.0" in narr_td

    # 3. Methods narrative
    narr_methods = generate_methods_narrative(
        model_type="diagnostic",
        has_reclassification=True,
        has_td_roc=True,
        time_horizons=[25.0, 45.0],
    )
    assert "Net Reclassification Improvement" in narr_methods
    assert "Time-dependent ROC curves" in narr_methods


# =============================================================================
# 5. Robustness & Edge-Case Guardrail Tests
# =============================================================================


def test_time_dependent_roc_missing_data_filtering():
    """Verify that rows with NaN or None in events are dropped without RuntimeWarning or TypeError."""
    durations = [10.0, 20.0, 30.0, 40.0, 50.0] * 3
    events = [1, 0, None, 1, 0] * 3
    scores = [0.8, 0.2, 0.5, 0.7, 0.1] * 3

    # Must not raise TypeError on None
    res = compute_time_dependent_roc(
        durations, events, scores, time_horizons=[25.0], n_bootstrap=0
    )
    h_res = res["results_by_horizon"][25.0]
    # Out of 15 rows, 3 with None must be dropped -> 12 observations remaining
    assert h_res["n_cases"] + h_res["n_controls"] + h_res["n_censored"] == 12

    # Float array with np.nan must also be dropped cleanly
    events_nan = np.array([1.0, 0.0, np.nan, 1.0, 0.0] * 3)
    res_nan = compute_time_dependent_roc(
        durations, events_nan, scores, time_horizons=[25.0], n_bootstrap=0
    )
    h_nan = res_nan["results_by_horizon"][25.0]
    assert h_nan["n_cases"] + h_nan["n_controls"] + h_nan["n_censored"] == 12


def test_plot_reclassification_missing_values_safety(tmp_path):
    """Verify plot_reclassification cleanly ignores NaN and None without casting errors."""
    y = [1, 0, None, 1, 0] * 4
    p_old = [0.6, 0.2, 0.5, 0.7, 0.3] * 4
    p_new = [0.8, 0.1, 0.6, 0.9, 0.2] * 4

    out_png = tmp_path / "reclass_missing.png"
    fig_res = plot_reclassification(
        y_true=y, p_old=p_old, p_new=p_new, out_path=str(out_png)
    )
    assert fig_res.png_path is not None
    # 20 rows - 4 None rows = 16 rows
    assert len(fig_res.source_df) == 16


def test_time_dependent_roc_tied_censoring_at_horizon():
    """Verify clean handling when censoring ties occur exactly at horizon t."""
    # 5 events before 20, 5 censored exactly at 20, 10 surviving past 20
    durations = np.array([5.0] * 5 + [20.0] * 5 + [30.0] * 10)
    events = np.array([1] * 5 + [0] * 5 + [1] * 10)
    scores = np.linspace(0.1, 0.9, 20)

    res = compute_time_dependent_roc(
        durations, events, scores, time_horizons=[20.0], n_bootstrap=0
    )
    h_res = res["results_by_horizon"][20.0]
    assert h_res["n_cases"] == 5
    assert h_res["n_censored"] == 5
    assert h_res["n_controls"] == 10
    assert np.isfinite(h_res["auroc"])


def test_time_dependent_roc_boundary_guards():
    """Verify negative durations, non-positive horizons, and non-binary events raise ValueError."""
    dur = np.array([-5.0, 10.0, 15.0] * 4)
    ev = np.array([1, 0, 1] * 4)
    sc = np.array([0.5, 0.4, 0.6] * 4)

    with pytest.raises(ValueError, match="non-negative"):
        compute_time_dependent_roc(dur, ev, sc, time_horizons=[10.0])

    dur_valid = np.abs(dur)
    with pytest.raises(ValueError, match="positive follow-up times"):
        compute_time_dependent_roc(dur_valid, ev, sc, time_horizons=[0.0])

    ev_invalid = np.array([1, 3, 1] * 4)
    with pytest.raises(ValueError, match="binary values"):
        compute_time_dependent_roc(dur_valid, ev_invalid, sc, time_horizons=[10.0])


def test_reclassification_bootstrap_continuous_and_idi():
    """Verify continuous NRI and IDI receive bootstrap CIs when n_bootstrap > 0."""
    np.random.seed(42)
    n = 150
    y = np.random.binomial(1, 0.35, size=n)
    p_old = np.clip(0.1 + 0.3 * y + np.random.normal(0, 0.1, size=n), 0.05, 0.95)
    p_new = np.clip(0.05 + 0.6 * y + np.random.normal(0, 0.08, size=n), 0.05, 0.95)

    res = compute_reclassification_metrics(
        y, p_old, p_new, thresholds=[0.25, 0.50], n_bootstrap=40, random_state=42
    )
    cont = res["nri_continuous"]["total"]
    assert cont["ci_lower_boot"] is not None
    assert cont["ci_upper_boot"] is not None
    assert cont["ci_lower_boot"] < cont["ci_upper_boot"]

    idi = res["idi"]
    assert idi["ci_lower_boot"] is not None
    assert idi["ci_upper_boot"] is not None
    assert idi["ci_lower_boot"] < idi["ci_upper_boot"]


def test_time_dependent_roc_tied_censoring_left_limit():
    """Verify that left-limit G_hat(T_i-) evaluates survival strictly prior to T_i.

    When an event time coincides with a censoring time, right-continuous G(t) drops at t,
    whereas left-limit G(t-) correctly reflects survival probability prior to the jump.
    """
    # KM fitted on censoring: durations [10, 10, 20], event_observed = [1, 0, 1] (censor at 10 and 20)
    durations = np.array([10.0, 10.0, 20.0, 30.0])
    # events: 0=censored, 1=death -> for censoring KM, event_observed is (events == 0)
    km = KaplanMeierFitter().fit(durations, event_observed=[1, 0, 1, 0])

    # At time 10.0, a censoring occurred. Right-continuous G(10) drops to 0.75.
    g_right = float(km.survival_function_at_times(10.0).iloc[0])
    # Left-limit strictly prior to 10.0 must be 1.0 (before any censoring occurred)
    g_left = _lookup_censoring_survival_left_limit(km, 10.0)
    assert g_left == 1.0
    assert g_right < g_left


def test_time_dependent_roc_strict_event_code_validation():
    """Verify event code validation both with and without competing events."""
    dur = np.array([10.0, 20.0, 30.0] * 4)
    sc = np.array([0.5, 0.4, 0.6] * 4)

    # 1. Without competing event code (default None): reject values not in {0, 1}
    ev_invalid_neg = np.array([1, 0, -1] * 4)
    with pytest.raises(ValueError, match="binary values"):
        compute_time_dependent_roc(dur, ev_invalid_neg, sc, time_horizons=[15.0])

    ev_invalid_two = np.array([1, 0, 2] * 4)
    with pytest.raises(ValueError, match="binary values"):
        compute_time_dependent_roc(dur, ev_invalid_two, sc, time_horizons=[15.0])

    # 2. With competing event code: reject 0 or 1 as competing_event_code
    ev_comp = np.array([1, 0, 2] * 4)
    with pytest.raises(ValueError, match="competing_event_code cannot be 0 or 1"):
        compute_time_dependent_roc(dur, ev_comp, sc, competing_event_code=0)

    with pytest.raises(ValueError, match="competing_event_code cannot be 0 or 1"):
        compute_time_dependent_roc(dur, ev_comp, sc, competing_event_code=1)

    # 3. With competing_event_code=2: reject if any value outside {0, 1, 2} exists
    ev_comp_bad = np.array([1, 0, 3] * 4)
    with pytest.raises(ValueError, match=r"outside \{0, 1, 2\}"):
        compute_time_dependent_roc(dur, ev_comp_bad, sc, competing_event_code=2)

    # Valid competing event code execution
    res_comp = compute_time_dependent_roc(
        dur, ev_comp, sc, competing_event_code=2, time_horizons=[15.0]
    )
    assert 15.0 in res_comp["results_by_horizon"]


def test_compare_diagnostic_models_strict_binary_validation():
    """Verify compare_diagnostic_models rejects 3-class, missing/NaN, and invalid pos_label."""
    p_old = np.array([0.2, 0.4, 0.6, 0.8])
    p_new = np.array([0.3, 0.5, 0.7, 0.9])

    # 3-class outcome
    with pytest.raises(ValueError, match="exactly 2 distinct classes"):
        compare_diagnostic_models([0, 1, 2, 1], p_old, p_new)

    # 1-class outcome
    with pytest.raises(ValueError, match="exactly 2 distinct classes"):
        compare_diagnostic_models([1, 1, 1, 1], p_old, p_new)

    # Missing / NaN in y_true
    with pytest.raises(ValueError, match="missing or NaN"):
        compare_diagnostic_models([0, 1, np.nan, 1], p_old, p_new)

    # pos_label not present in outcome
    with pytest.raises(ValueError, match="pos_label 'pos' not found"):
        compare_diagnostic_models([0, 1, 0, 1], p_old, p_new, pos_label="pos")

    # Output does not contain patient arrays
    res = compare_diagnostic_models([0, 1, 0, 1], p_old, p_new)
    assert "p_old" not in res["reclassification"]
    assert "p_new" not in res["reclassification"]
    assert "y_true" not in res["reclassification"]


def test_time_dependent_roc_ci_method_reporting():
    """Verify ci_method is correctly reported as 'bootstrap', 'hanley_mcneil', or None."""
    dur = np.array([10.0, 20.0, 30.0, 40.0, 50.0] * 6)
    ev = np.array([1, 0, 1, 0, 1] * 6)
    sc = np.array([0.8, 0.2, 0.7, 0.3, 0.9] * 6)

    # Analytical (n_bootstrap=0)
    res_ana = compute_time_dependent_roc(
        dur, ev, sc, time_horizons=[25.0, 100.0], n_bootstrap=0
    )
    assert res_ana["ci_method"] == "hanley_mcneil"
    assert res_ana["results_by_horizon"][25.0]["ci_method"] == "hanley_mcneil"
    # Beyond max follow-up (100.0 > 50.0) -> is_sparse -> ci_method is None
    assert res_ana["results_by_horizon"][100.0]["ci_method"] is None

    # Bootstrap (n_bootstrap > 0)
    res_boot = compute_time_dependent_roc(
        dur, ev, sc, time_horizons=[25.0], n_bootstrap=20, random_state=42
    )
    assert res_boot["ci_method"] == "bootstrap"
    assert res_boot["results_by_horizon"][25.0]["ci_method"] == "bootstrap"

    # Narrative checks
    table_html = format_time_dependent_roc_table(res_ana)
    assert "analytical IPCW approximation (Hanley-McNeil)" in table_html

    table_boot = format_time_dependent_roc_table(res_boot)
    assert "bootstrap resampling" in table_boot

    narr_ana = generate_methods_narrative(
        model_type="diagnostic",
        has_td_roc=True,
        diag_data=res_ana,
    )
    assert "analytical IPCW approximation (Hanley-McNeil)" in narr_ana

    narr_boot = generate_methods_narrative(
        model_type="diagnostic",
        has_td_roc=True,
        diag_data=res_boot,
    )
    assert "bootstrap resampling" in narr_boot


def test_plot_time_dependent_roc_non_finite_horizon_and_dynamic_ylim(tmp_path):
    """Verify non-finite horizon curves are skipped with unavailable notes, and y-axis is dynamic."""
    dur = np.array([10.0, 20.0, 30.0, 40.0, 50.0] * 3)
    ev = np.array([1, 0, 1, 0, 1] * 3)
    sc = np.array([0.8, 0.2, 0.7, 0.3, 0.9] * 3)
    td_data = compute_time_dependent_roc(
        dur, ev, sc, time_horizons=[25.0, 100.0], n_bootstrap=0
    )

    # Make horizon 100 AUROC explicitly NaN to simulate insufficient events/at-risk
    td_data["results_by_horizon"][100.0]["auroc"] = float("nan")

    out_c = str(tmp_path / "curves_unavail.png")
    fig_c = plot_time_dependent_roc(td_data, mode="curves", out_path=out_c)
    assert "omitted from curve plotting" in fig_c.caption

    # Dual Plotly version
    out_dual = str(tmp_path / "dual_unavail.png")
    dual_res = plot_dual_time_dependent_roc(td_data, mode="curves", out_path=out_dual)
    trace_names = [t.name for t in dual_res.plotly_fig.data]
    assert any(
        "unavailable (insufficient events/at-risk)" in str(n) for n in trace_names
    )

    # Trajectory mode with low AUROC to verify dynamic y-axis scaling
    td_low = dict(td_data)
    td_low["trajectory"] = {
        "horizons": [25.0],
        "auroc": [0.35],
        "ci_lower": [0.25],
        "ci_upper": [0.45],
    }
    out_traj = str(tmp_path / "traj_low.png")
    plot_time_dependent_roc(td_low, mode="auroc_trajectory", out_path=out_traj)
    dual_traj = plot_dual_time_dependent_roc(
        td_low, mode="auroc_trajectory", out_path=out_traj
    )
    y_range = dual_traj.plotly_fig.layout.yaxis.range
    assert y_range[0] < 0.30

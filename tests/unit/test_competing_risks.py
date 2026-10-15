"""
tests/unit/test_competing_risks.py: Unit tests for Competing Risks CIF (Aalen-Johansen) and Fine-Gray regression.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from medstat.figures.html.survival import plot_dual_cif
from medstat.figures.survival import plot_cif
from medstat.models.survival import (
    CompetingRisksCIFResult,
    fit_competing_risks_cif,
    fit_competing_risks_regression,
)
from medstat.reporting.tables import render_competing_risks_table


@pytest.fixture
def synthetic_competing_risks() -> pd.DataFrame:
    """Create reproducible synthetic competing risks cohort."""
    np.random.seed(42)
    n = 200
    age = np.random.normal(60, 10, n)
    treatment = np.random.binomial(1, 0.5, n)

    # Latent failure times
    t1 = np.random.exponential(15.0 / np.exp(0.5 * treatment + 0.02 * age))
    t2 = np.random.exponential(20.0 / np.exp(-0.2 * treatment + 0.03 * age))
    c = np.random.uniform(5, 30, n)

    time = np.minimum(np.minimum(t1, t2), c)
    event = np.zeros(n, dtype=int)
    event[(t1 <= t2) & (t1 <= c)] = 1  # Event of interest
    event[(t2 < t1) & (t2 <= c)] = 2  # Competing event

    return pd.DataFrame(
        {
            "id": np.arange(n),
            "time": time,
            "event": event,
            "treatment": treatment,
            "age": age,
        }
    )


def test_fit_competing_risks_cif_basics(synthetic_competing_risks: pd.DataFrame):
    df = synthetic_competing_risks
    res = fit_competing_risks_cif(
        durations=df["time"],
        event_observed=df["event"],
        event_of_interest=1,
        competing_event=2,
        horizons=[5.0, 10.0, 15.0],
    )

    assert isinstance(res, CompetingRisksCIFResult)
    assert res.cif_table is not None
    assert len(res.cif_table) > 0
    assert "time" in res.cif_table.columns
    assert "cif_primary" in res.cif_table.columns
    assert "cif_competing" in res.cif_table.columns
    assert "naive_1_minus_km" in res.cif_table.columns
    assert "bias_overestimation" in res.cif_table.columns

    # Probability conservation check
    assert res.probability_conservation_max_error < 0.05

    # Naive KM overestimation check: Naive >= CIF1 at all time points
    naive = res.cif_table["naive_1_minus_km"].to_numpy()
    cif1 = res.cif_table["cif_primary"].to_numpy()
    assert (naive >= cif1 - 1e-4).all()

    # Horizons
    assert res.horizon_summary is not None
    assert len(res.horizon_summary) == 3
    assert "overestimation_abs" in res.horizon_summary.columns
    assert "overestimation_relative_pct" in res.horizon_summary.columns


def test_fit_competing_risks_regression(synthetic_competing_risks: pd.DataFrame):
    df = synthetic_competing_risks
    res = fit_competing_risks_regression(
        df=df,
        duration_col="time",
        event_col="event",
        covariates=["treatment", "age"],
        event_of_interest=1,
        competing_event=2,
        id_col="id",
    )

    assert "model_cause_specific" in res
    assert "model_fine_gray" in res
    assert "comparison_table" in res

    comp_df = res["comparison_table"]
    assert len(comp_df) == 2
    assert "cause_specific_hr" in comp_df.columns
    assert "fine_gray_shr" in comp_df.columns
    assert "cs_p_value" in comp_df.columns
    assert "fg_p_value" in comp_df.columns

    # Treatment hazard ratio should be positive and match reference values from deterministic cohort
    trt_row = comp_df[comp_df["covariate"] == "treatment"].iloc[0]
    assert trt_row["cause_specific_hr"] > 0
    assert trt_row["fine_gray_shr"] > 0
    assert trt_row["cs_ci_lower"] < trt_row["cs_ci_upper"]
    assert trt_row["fg_ci_lower"] < trt_row["fg_ci_upper"]

    # Reference value assertions on deterministic fixture (np.random.seed(42), n=200)
    assert trt_row["fine_gray_shr"] == pytest.approx(1.9883, abs=0.05)
    assert trt_row["cause_specific_hr"] == pytest.approx(1.8725, abs=0.05)


def test_plot_cif_and_dual_cif(tmp_path, synthetic_competing_risks: pd.DataFrame):
    df = synthetic_competing_risks
    res = fit_competing_risks_cif(
        durations=df["time"],
        event_observed=df["event"],
        horizons=[10.0],
    )

    out_file = tmp_path / "cif_plot.png"
    fig_res = plot_cif(res, out_path=out_file)
    assert out_file.exists()
    assert Path(fig_res.png_path) == out_file
    assert fig_res.source_df is not None

    dual_res = plot_dual_cif(res, out_path=tmp_path / "cif_dual.png")
    assert dual_res.plotly_fig is not None


def test_render_competing_risks_table(synthetic_competing_risks: pd.DataFrame):
    df = synthetic_competing_risks
    res = fit_competing_risks_regression(
        df=df,
        duration_col="time",
        event_col="event",
        covariates=["treatment", "age"],
    )

    html_nejm = render_competing_risks_table(
        title="Primary Cardiovascular Failure vs Competing Non-CV Mortality",
        cr_data=res,
        style="NEJM",
    )
    assert "<table" in html_nejm
    assert "Cause-Specific HR" in html_nejm
    assert "Fine-Gray SHR" in html_nejm
    assert "treatment" in html_nejm


def test_competing_risks_distinct_event_validation(
    synthetic_competing_risks: pd.DataFrame,
):
    df = synthetic_competing_risks
    # Equal event_of_interest and competing_event must raise ValueError
    with pytest.raises(ValueError, match="must be distinct"):
        fit_competing_risks_cif(
            durations=df["time"],
            event_observed=df["event"],
            event_of_interest=1,
            competing_event=1,
        )

    with pytest.raises(ValueError, match="must be distinct"):
        fit_competing_risks_regression(
            df=df,
            duration_col="time",
            event_col="event",
            covariates=["treatment"],
            event_of_interest=1,
            competing_event=1,
        )


def test_competing_risks_missing_event_handling(
    synthetic_competing_risks: pd.DataFrame,
):
    df = synthetic_competing_risks.copy()
    # Insert NaN into event_observed; CIF should filter it out without crashing on int conversion
    evt = df["event"].astype(float).copy()
    evt.iloc[0] = np.nan
    evt.iloc[1] = np.nan

    res = fit_competing_risks_cif(
        durations=df["time"],
        event_observed=evt,
        event_of_interest=1,
        competing_event=2,
    )
    assert res.cif_table is not None
    assert len(res.cif_table) > 0

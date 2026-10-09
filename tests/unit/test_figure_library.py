"""
tests/unit/test_figure_library.py: Unit tests for the publication figure library.
"""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from medstat.figures import (
    FigureResult,
    plot_bland_altman,
    plot_calibration,
    plot_dca,
    plot_forest,
    plot_kaplan_meier,
    plot_love,
    plot_mice_diagnostics,
    plot_missingness_map,
    plot_retention_flow,
    plot_roc_curve,
    plot_schoenfeld_residuals,
)
from tests.fixtures.synthetic_clinical import (
    generate_synthetic_clinical_cohort,
    generate_synthetic_meta_analysis_data,
)


@pytest.fixture
def tmp_fig_dir(tmp_path):
    d = tmp_path / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_plot_forest(tmp_fig_dir):
    df_meta = generate_synthetic_meta_analysis_data()
    out_png = tmp_fig_dir / "test_forest.png"

    res = plot_forest(
        df_meta,
        scale="OR",
        title="Odds Ratios across Studies (การศึกษาเปรียบเทียบ)",
        out_path=out_png,
    )
    assert isinstance(res, FigureResult)
    assert Path(res.png_path).exists()
    assert Path(res.png_path).stat().st_size > 10000
    assert Path(res.csv_path).exists()
    assert len(res.source_df) == len(df_meta)
    assert "OR" in res.alt_text


def test_plot_kaplan_meier(tmp_fig_dir):
    out_png = tmp_fig_dir / "test_km.png"

    # Build km_curves dict
    km_data = {
        "Control": {
            "timeline": np.array([0, 30, 90, 180, 365]),
            "survival": np.array([1.0, 0.95, 0.88, 0.82, 0.75]),
            "ci_lower": np.array([1.0, 0.91, 0.82, 0.75, 0.67]),
            "ci_upper": np.array([1.0, 0.98, 0.93, 0.87, 0.81]),
            "censored_times": np.array([45, 120, 200]),
            "censored_survival": np.array([0.93, 0.85, 0.80]),
        },
        "Treated": {
            "timeline": np.array([0, 30, 90, 180, 365]),
            "survival": np.array([1.0, 0.98, 0.94, 0.90, 0.86]),
            "ci_lower": np.array([1.0, 0.94, 0.89, 0.84, 0.79]),
            "ci_upper": np.array([1.0, 1.0, 0.97, 0.94, 0.91]),
            "censored_times": np.array([60, 150, 300]),
            "censored_survival": np.array([0.96, 0.92, 0.88]),
        },
    }
    risk_table = pd.DataFrame(
        [
            {
                "Group": "Control",
                "t=0.0": 100,
                "t=90.0": 88,
                "t=180.0": 82,
                "t=365.0": 75,
            },
            {
                "Group": "Treated",
                "t=0.0": 100,
                "t=90.0": 94,
                "t=180.0": 90,
                "t=365.0": 86,
            },
        ]
    )

    res = plot_kaplan_meier(
        km_data,
        risk_table=risk_table,
        log_rank_p=0.042,
        title="Kaplan-Meier Survival by Treatment (อัตราการรอดชีวิต)",
        out_path=out_png,
    )
    assert Path(res.png_path).exists()
    assert Path(res.png_path).stat().st_size > 15000
    assert Path(res.csv_path).exists()
    assert "Log-rank" in res.caption


def test_plot_roc_curve(tmp_fig_dir):
    out_png = tmp_fig_dir / "test_roc.png"
    roc_data = {
        "fpr": np.array([0.0, 0.05, 0.15, 0.30, 0.60, 1.0]),
        "tpr": np.array([0.0, 0.50, 0.75, 0.88, 0.96, 1.0]),
        "thresholds": np.array([10.0, 8.0, 5.0, 3.0, 1.0, 0.1]),
        "auc": 0.865,
        "ci_lower": 0.812,
        "ci_upper": 0.918,
        "optimal_point": {"fpr": 0.15, "tpr": 0.75, "cutoff": 5.0},
        "name": "Troponin Biomarker",
    }

    res = plot_roc_curve(
        roc_data, title="ROC Curve (ความแม่นยำในการวินิจฉัย)", out_path=out_png
    )
    assert Path(res.png_path).exists()
    assert Path(res.png_path).stat().st_size > 12000
    assert Path(res.csv_path).exists()
    assert "AUC" in res.caption


def test_plot_calibration(tmp_fig_dir):
    out_png = tmp_fig_dir / "test_calib.png"
    calib_bins = {
        "pred_mean": np.linspace(0.05, 0.95, 10),
        "obs_rate": np.linspace(0.06, 0.92, 10),
        "ci_lower": np.linspace(0.02, 0.85, 10),
        "ci_upper": np.linspace(0.12, 0.97, 10),
        "counts": np.array([50] * 10),
    }
    calib_curve = {
        "pred_smooth": np.linspace(0.01, 0.99, 100),
        "obs_smooth": np.linspace(0.02, 0.97, 100),
    }

    res = plot_calibration(
        calib_bins,
        calib_curve=calib_curve,
        brier_score=0.124,
        ici=0.018,
        title="Calibration Deciles (การสอบเทียบโมเดล)",
        out_path=out_png,
    )
    assert Path(res.png_path).exists()
    assert Path(res.png_path).stat().st_size > 12000
    assert Path(res.csv_path).exists()


def test_plot_dca(tmp_fig_dir):
    out_png = tmp_fig_dir / "test_dca.png"
    th = np.linspace(0.05, 0.80, 20)
    dca_records = []
    for pt in th:
        dca_records.append(
            {
                "threshold": pt,
                "net_benefit": max(0.20 - 0.25 * pt, 0.0),
                "strategy": "Model",
            }
        )
        dca_records.append(
            {
                "threshold": pt,
                "net_benefit": max(0.15 - 0.40 * pt, -0.1),
                "strategy": "Treat All",
            }
        )
        dca_records.append(
            {"threshold": pt, "net_benefit": 0.0, "strategy": "Treat None"}
        )
    dca_df = pd.DataFrame(dca_records)

    res = plot_dca(dca_df, title="Decision Curve Analysis", out_path=out_png)
    assert Path(res.png_path).exists()
    assert Path(res.png_path).stat().st_size > 12000
    assert Path(res.csv_path).exists()


def test_plot_bland_altman(tmp_fig_dir):
    out_png = tmp_fig_dir / "test_ba.png"
    ba_data = {
        "means": np.random.normal(120, 10, 80),
        "diffs": np.random.normal(2.5, 4.0, 80),
        "mean_diff": 2.5,
        "loa_upper": 10.34,
        "loa_lower": -5.34,
        "ci_mean_diff": (1.62, 3.38),
        "ci_loa_upper": (8.82, 11.86),
        "ci_loa_lower": (-6.86, -3.82),
    }

    res = plot_bland_altman(
        ba_data, units="mmHg", title="Blood Pressure Device Agreement", out_path=out_png
    )
    assert Path(res.png_path).exists()
    assert Path(res.png_path).stat().st_size > 12000
    assert Path(res.csv_path).exists()


def test_plot_love(tmp_fig_dir):
    out_png = tmp_fig_dir / "test_love.png"
    love_data = {
        "covariates": ["Age", "BMI", "eGFR", "Hypertension", "Diabetes"],
        "smd_raw": [0.35, 0.28, 0.22, 0.41, 0.18],
        "smd_matched": [0.04, 0.03, 0.05, 0.02, 0.04],
        "threshold": 0.10,
    }

    res = plot_love(love_data, title="Love Plot (สมดุลตัวแปร)", out_path=out_png)
    assert Path(res.png_path).exists()
    assert Path(res.png_path).stat().st_size > 12000
    assert Path(res.csv_path).exists()


def test_plot_retention_flow(tmp_fig_dir):
    out_png = tmp_fig_dir / "test_retention.png"
    retention_data = {
        "n_initial": 520,
        "n_excluded": 20,
        "n_analyzed": 500,
        "exclusion_reasons": ["Declined consent (n = 12)", "Age < 18 (n = 8)"],
    }

    res = plot_retention_flow(
        retention_data, title="CONSORT Flow Diagram", out_path=out_png
    )
    assert Path(res.png_path).exists()
    assert Path(res.png_path).stat().st_size > 10000
    assert Path(res.csv_path).exists()


def test_plot_diagnostics(tmp_fig_dir):
    df = generate_synthetic_clinical_cohort(n=100, seed=42)

    # 1. Missingness map
    res_miss = plot_missingness_map(
        df,
        columns=["age", "bmi", "lab_crp", "lab_ldl"],
        out_path=tmp_fig_dir / "miss.png",
    )
    assert Path(res_miss.png_path).exists()

    # 2. Schoenfeld residuals
    times = np.linspace(10, 365, 50)
    resids = np.random.normal(0, 0.5, 50)
    res_schoen = plot_schoenfeld_residuals(
        times,
        resids,
        var_name="Diabetes",
        p_value=0.28,
        out_path=tmp_fig_dir / "schoen.png",
    )
    assert Path(res_schoen.png_path).exists()

    # 3. MICE diagnostics
    obs = np.random.normal(100, 15, 60)
    imp = [np.random.normal(98, 14, 40), np.random.normal(102, 16, 40)]
    res_mice = plot_mice_diagnostics(
        obs, imp, var_name="CRP", out_path=tmp_fig_dir / "mice.png"
    )
    assert Path(res_mice.png_path).exists()

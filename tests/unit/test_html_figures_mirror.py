"""
tests/unit/test_html_figures_mirror.py: Unit tests for Dual/Toggle HTML figures mirror and renderer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from medstat.figures.base import FigureResult
from medstat.figures.html import (
    DualFigureResult,
    plot_dual_bland_altman,
    plot_dual_calibration,
    plot_dual_dca,
    plot_dual_forest,
    plot_dual_kaplan_meier,
    plot_dual_love,
    plot_dual_roc_curve,
)
from medstat.reporting.ir import ReportDocument
from medstat.reporting.renderers import render_html
from medstat.reporting.renderers.html import _encode_csv_data_uri


@pytest.fixture
def tmp_fig_dir(tmp_path: Path) -> Path:
    d = tmp_path / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_dual_figure_result_contract():
    """Verifies DualFigureResult serialization and backward compatibility."""
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))

    # Test initialization with plotly_fig
    res1 = DualFigureResult(
        png_path="test.png",
        alt_text="Alt",
        caption="Caption",
        source_df=df,
        csv_path="test.csv",
        plotly_fig=fig,
    )
    assert res1.plotly_json is not None
    assert "data" in json.loads(res1.plotly_json)
    assert res1.to_dict()["has_interactive"] is True

    # Test initialization without interactive figure
    res2 = DualFigureResult(
        png_path="test.png",
        alt_text="Alt",
        caption="Caption",
        source_df=df,
        csv_path="test.csv",
    )
    assert res2.plotly_json is None
    assert res2.to_dict()["has_interactive"] is False


def test_plot_dual_roc_curve(tmp_fig_dir: Path):
    """Verifies plot_dual_roc_curve produces both 300 DPI PNG and Plotly ROC curve."""
    roc_data = {
        "fpr": [0.0, 0.1, 0.2, 0.5, 1.0],
        "tpr": [0.0, 0.6, 0.8, 0.9, 1.0],
        "thresholds": [5.0, 4.0, 3.0, 2.0, 1.0],
        "auc": 0.85,
        "ci_lower": 0.78,
        "ci_upper": 0.92,
        "optimal_point": {"fpr": 0.1, "tpr": 0.6, "cutoff": 4.0},
        "name": "Biomarker A",
    }
    out_png = tmp_fig_dir / "dual_roc.png"

    res = plot_dual_roc_curve(
        roc_data=roc_data,
        title="Biomarker A ROC",
        out_path=out_png,
    )

    assert isinstance(res, DualFigureResult)
    assert Path(res.png_path).exists()
    assert Path(res.csv_path).exists()
    assert res.plotly_json is not None
    spec = json.loads(res.plotly_json)
    assert len(spec["data"]) >= 3  # chance diagonal, ROC curve, optimal cutpoint


def test_plot_dual_kaplan_meier(tmp_fig_dir: Path):
    """Verifies plot_dual_kaplan_meier produces step curves with CI bands."""
    km_data = {
        "Treatment": {
            "timeline": np.array([0, 10, 20, 30]),
            "survival": np.array([1.0, 0.9, 0.8, 0.7]),
            "ci_lower": np.array([1.0, 0.8, 0.7, 0.6]),
            "ci_upper": np.array([1.0, 0.95, 0.88, 0.8]),
        },
        "Control": {
            "timeline": np.array([0, 10, 20, 30]),
            "survival": np.array([1.0, 0.8, 0.6, 0.5]),
            "ci_lower": np.array([1.0, 0.7, 0.5, 0.4]),
            "ci_upper": np.array([1.0, 0.88, 0.7, 0.6]),
        },
    }
    out_png = tmp_fig_dir / "dual_km.png"

    res = plot_dual_kaplan_meier(
        km_curves=km_data,
        log_rank_p=0.015,
        title="Survival Analysis",
        out_path=out_png,
    )

    assert isinstance(res, DualFigureResult)
    assert Path(res.png_path).exists()
    assert res.plotly_json is not None
    spec = json.loads(res.plotly_json)
    assert "Log-rank p = 0.015" in spec["layout"]["title"]["text"]


def test_plot_dual_calibration(tmp_fig_dir: Path):
    """Verifies plot_dual_calibration produces decile points and ideal line."""
    calib_bins = {
        "pred_mean": np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
        "obs_rate": np.array([0.08, 0.32, 0.48, 0.72, 0.88]),
        "ci_lower": np.array([0.04, 0.25, 0.40, 0.65, 0.80]),
        "ci_upper": np.array([0.14, 0.40, 0.58, 0.80, 0.95]),
    }
    out_png = tmp_fig_dir / "dual_calib.png"

    res = plot_dual_calibration(
        calib_bins=calib_bins,
        brier_score=0.085,
        ici=0.021,
        title="Model Calibration",
        out_path=out_png,
    )

    assert isinstance(res, DualFigureResult)
    assert Path(res.png_path).exists()
    assert res.plotly_json is not None
    spec = json.loads(res.plotly_json)
    assert "Brier = 0.085" in spec["layout"]["title"]["text"]

    decile_trace = next(
        t for t in spec["data"] if t.get("name") == "Decile Bins (95% CI)"
    )
    assert "customdata" in decile_trace
    assert decile_trace["customdata"][0] == [0.04, 0.14]
    assert "%{customdata[0]:.3f}" in decile_trace["hovertemplate"]
    assert "%{customdata[1]:.3f}" in decile_trace["hovertemplate"]


def test_plot_dual_dca(tmp_fig_dir: Path):
    """Verifies plot_dual_dca produces Net Benefit curves."""
    df_dca = pd.DataFrame(
        {
            "threshold": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
            "net_benefit": [0.15, 0.10, 0.05, 0.12, 0.08, 0.02],
            "strategy": [
                "Model",
                "Model",
                "Model",
                "Treat All",
                "Treat All",
                "Treat All",
            ],
        }
    )
    out_png = tmp_fig_dir / "dual_dca.png"

    res = plot_dual_dca(
        dca_df=df_dca,
        title="Decision Curve Analysis",
        out_path=out_png,
    )

    assert isinstance(res, DualFigureResult)
    assert Path(res.png_path).exists()
    assert res.plotly_json is not None


def test_plot_dual_bland_altman(tmp_fig_dir: Path):
    """Verifies plot_dual_bland_altman produces limits of agreement plot."""
    means = np.array([50.0, 60.0, 70.0, 80.0, 90.0])
    diffs = np.array([1.2, -0.8, 2.1, -1.5, 0.4])
    ba_data = {
        "means": means,
        "diffs": diffs,
        "mean_diff": float(np.mean(diffs)),
        "loa_upper": float(np.mean(diffs) + 1.96 * np.std(diffs, ddof=1)),
        "loa_lower": float(np.mean(diffs) - 1.96 * np.std(diffs, ddof=1)),
        "method1": "Device A",
        "method2": "Device B",
    }
    out_png = tmp_fig_dir / "dual_ba.png"

    res = plot_dual_bland_altman(
        ba_data=ba_data,
        units="mmHg",
        title="BP Monitor Agreement",
        out_path=out_png,
    )

    assert isinstance(res, DualFigureResult)
    assert Path(res.png_path).exists()
    assert res.plotly_json is not None


def test_plot_dual_forest(tmp_fig_dir: Path):
    """Verifies plot_dual_forest produces effect estimates with error bars."""
    data = [
        {"term": "Age >= 60", "estimate": 1.45, "ci_lower": 1.10, "ci_upper": 1.92},
        {"term": "Diabetes", "estimate": 2.10, "ci_lower": 1.55, "ci_upper": 2.85},
    ]
    out_png = tmp_fig_dir / "dual_forest.png"

    res = plot_dual_forest(
        data=data,
        scale="OR",
        title="Risk Factors Forest Plot",
        out_path=out_png,
    )

    assert isinstance(res, DualFigureResult)
    assert Path(res.png_path).exists()
    assert res.plotly_json is not None


def test_plot_dual_love(tmp_fig_dir: Path):
    """Verifies plot_dual_love produces covariate balance plot."""
    love_data = {
        "covariates": ["Age", "BMI", "Hypertension"],
        "smd_raw": [0.25, 0.18, 0.32],
        "smd_matched": [0.04, 0.02, 0.05],
    }
    out_png = tmp_fig_dir / "dual_love.png"

    res = plot_dual_love(
        love_data=love_data,
        title="Covariate Balance",
        out_path=out_png,
    )

    assert isinstance(res, DualFigureResult)
    assert Path(res.png_path).exists()
    assert res.plotly_json is not None

    # Verify custom threshold propagation
    res_custom = plot_dual_love(
        love_data=love_data,
        threshold=0.15,
        title="Custom Threshold Balance",
        out_path=tmp_fig_dir / "dual_love_015.png",
    )
    spec_custom = json.loads(res_custom.plotly_json)
    thresh_lines = [
        s for s in spec_custom["layout"].get("shapes", []) if s.get("x0") == 0.15
    ]
    assert len(thresh_lines) == 1
    thresh_anns = [
        a
        for a in spec_custom["layout"].get("annotations", [])
        if "0.15" in a.get("text", "")
    ]
    assert len(thresh_anns) == 1


def test_render_html_dual_toggle_view(tmp_path: Path):
    """Verifies render_html renders Dual Toggle UI, offline notice, and download buttons."""
    fig_png = tmp_path / "roc.png"
    roc_data = {
        "fpr": [0.0, 0.2, 1.0],
        "tpr": [0.0, 0.8, 1.0],
        "auc": 0.88,
        "name": "Model 1",
    }
    dual_res = plot_dual_roc_curve(roc_data=roc_data, out_path=fig_png)

    doc = ReportDocument(title="Diagnostic Performance Report")
    doc.add_heading("Results", level=1)
    doc.add_figure(dual_res)

    out_html = tmp_path / "report_dual.html"
    render_html(doc, out_html)

    content = out_html.read_text(encoding="utf-8")

    # Verify CDN script presence
    assert "https://cdn.plot.ly/plotly-2.35.2.min.js" in content

    # Verify Dual Toggle UI tabs
    assert "Interactive (Plotly)" in content
    assert "Publication 300 DPI (Static)" in content
    assert 'class="fig-tab-btn active"' in content

    # Verify Download Action Buttons
    assert "Download 300 DPI PNG" in content
    assert "CSV Data" in content
    assert "data:text/csv;charset=utf-8" in content

    # Verify Offline Resilience & Banner
    assert "offline-banner" in content
    assert "โหมดออฟไลน์ (Offline Mode)" in content
    assert "สลับไปดู Publication Figure (300 DPI)" in content
    assert "switchFigView(id, 'static');" in content

    # Verify Print Media Styles
    assert "@media print" in content
    assert ".figure-toolbar { display: none !important; }" in content


def test_render_html_static_only_figure(tmp_path: Path):
    """Verifies render_html renders clean static-only card when no Plotly data is present."""
    png_path = tmp_path / "static.png"
    # Create simple PNG
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    fig.savefig(png_path)
    plt.close(fig)

    source_df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    csv_path = tmp_path / "static.csv"
    source_df.to_csv(csv_path, index=False)

    static_res = FigureResult(
        png_path=str(png_path),
        alt_text="Static Plot",
        caption="Figure 1. Static Plot",
        source_df=source_df,
        csv_path=str(csv_path),
    )

    doc = ReportDocument(title="Static Report")
    doc.add_figure(static_res)

    out_html = tmp_path / "report_static.html"
    render_html(doc, out_html)

    content = out_html.read_text(encoding="utf-8")

    # Verify Plotly CDN is NOT included when not needed
    assert "https://cdn.plot.ly/plotly-2.35.2.min.js" not in content

    # Verify static badge is present without interactive tab buttons
    assert "Publication Figure (300 DPI)" in content
    assert "Interactive (Plotly)" not in content
    assert "Download 300 DPI PNG" in content


def test_render_html_escapes_less_than_in_plotly_json(tmp_path: Path):
    """Verifies render_html escapes '<' as '\\u003c' in embedded plotly_json script tags."""
    fig_png = tmp_path / "roc_escape.png"
    roc_data = {
        "fpr": [0.0, 0.2, 1.0],
        "tpr": [0.0, 0.8, 1.0],
        "auc": 0.88,
        "name": "Model <Alpha> </script><script>alert(1)</script>",
    }
    dual_res = plot_dual_roc_curve(
        roc_data=roc_data,
        title="Cutoff < 0.05",
        out_path=fig_png,
    )

    doc = ReportDocument(title="Test Escaping")
    doc.add_figure(dual_res)

    out_html = tmp_path / "report_escape.html"
    render_html(doc, out_html)

    content = out_html.read_text(encoding="utf-8")

    match = re.search(
        r'<script type="application/json" id="plotly-spec-1">\s*(.*?)\s*</script>',
        content,
        re.DOTALL,
    )
    assert match is not None, "Plotly spec script tag not found"
    script_json = match.group(1)

    # Verify no unescaped '<' exists inside the embedded JSON script element
    assert "<" not in script_json
    assert "\\u003c" in script_json

    # Verify JSON parsing faithfully reproduces original string content
    parsed = json.loads(script_json)
    assert "Cutoff < 0.05" in parsed["layout"]["title"]["text"]
    alpha_trace = next(
        t for t in parsed["data"] if "Model <Alpha>" in t.get("name", "")
    )
    assert "</script>" in alpha_trace["name"]


def test_encode_csv_data_uri_missing_file_emits_warning(tmp_path: Path):
    """Verifies that _encode_csv_data_uri warns with the CSV path when the file is missing."""
    missing_csv = tmp_path / "nonexistent_source.csv"
    with pytest.warns(UserWarning, match=re.escape(str(missing_csv.resolve()))):
        res = _encode_csv_data_uri(missing_csv)
    assert res is None


def test_encode_csv_data_uri_read_exception_emits_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verifies that _encode_csv_data_uri warns with the CSV path when reading an existing file throws an exception."""
    existing_csv = tmp_path / "error_file.csv"
    existing_csv.write_text("data", encoding="utf-8")

    orig_read_text = Path.read_text

    def mock_read_text(self, *args, **kwargs):
        if self.name == "error_file.csv":
            raise PermissionError("Permission denied")
        return orig_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", mock_read_text)

    with pytest.warns(
        UserWarning, match=r"Failed to read CSV source data from.*error_file\.csv"
    ):
        res = _encode_csv_data_uri(existing_csv)
    assert res is None


def test_render_html_missing_csv_emits_warning_and_omits_download(tmp_path: Path):
    """Verifies render_html warns when CSV file is missing, but still outputs report without CSV button."""
    png_path = tmp_path / "test_warn_fig.png"
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    fig.savefig(png_path)
    plt.close(fig)

    missing_csv = tmp_path / "missing_data_audit.csv"
    fig_res = FigureResult(
        png_path=str(png_path),
        alt_text="Audit Figure",
        caption="Audit Caption",
        source_df=pd.DataFrame({"x": [1]}),
        csv_path=str(missing_csv),
    )
    doc = ReportDocument(title="Audit Report")
    doc.add_figure(fig_res)

    out_html = tmp_path / "audit_report.html"
    with pytest.warns(UserWarning, match=re.escape(str(missing_csv.resolve()))):
        render_html(doc, out_html)

    content = out_html.read_text(encoding="utf-8")
    assert "Download 300 DPI PNG" in content
    assert "CSV Data" not in content


def test_render_html_unreadable_csv_emits_warning_and_omits_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verifies render_html warns when an existing CSV raises an exception on read, but still outputs report without CSV button."""
    png_path = tmp_path / "test_read_err_fig.png"
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    fig.savefig(png_path)
    plt.close(fig)

    existing_csv = tmp_path / "unreadable_audit.csv"
    existing_csv.write_text("x,y\n1,2", encoding="utf-8")

    fig_res = FigureResult(
        png_path=str(png_path),
        alt_text="Audit Figure",
        caption="Audit Caption",
        source_df=pd.DataFrame({"x": [1]}),
        csv_path=str(existing_csv),
    )
    doc = ReportDocument(title="Audit Report")
    doc.add_figure(fig_res)

    orig_read_text = Path.read_text

    def mock_read_text(self, *args, **kwargs):
        if self.name == "unreadable_audit.csv":
            raise OSError("Simulated disk read I/O error")
        return orig_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", mock_read_text)

    out_html = tmp_path / "read_err_report.html"
    with pytest.warns(
        UserWarning, match=r"Failed to read CSV source data from.*unreadable_audit\.csv"
    ):
        render_html(doc, out_html)

    content = out_html.read_text(encoding="utf-8")
    assert "Download 300 DPI PNG" in content
    assert "CSV Data" not in content

"""
tests/unit/test_journal_figures.py: Verification of journal styling, formatters,
QA layout gate, Zero-PHI scan, and FigureResult contract extensions.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from medstat.figures.base import FigureResult
from medstat.figures.journal import (
    JOURNAL_SPECS,
    TARGET_WIDTHS,
    apply_journal_theme,
    format_estimate,
    format_num,
    format_p_value,
    style_axis,
)
from medstat.figures.qa import assert_zero_phi_text, cvd_check, qa_layout
from medstat.figures.styles import save_figure_bundle


def test_journal_specs_contain_all_five_styles():
    for j in ["NEJM", "JAMA", "LANCET", "BMJ", "CLINICAL"]:
        assert j in JOURNAL_SPECS
        assert "palette" in JOURNAL_SPECS[j]
        assert len(JOURNAL_SPECS[j]["palette"]) >= 6
        assert "bg_manuscript" in JOURNAL_SPECS[j]


def test_format_p_value_rules():
    # NEJM: standard P=0.03, P<0.001
    assert format_p_value(0.031, "NEJM") == "P=0.03"
    assert format_p_value(0.0004, "NEJM") == "P<0.001"
    assert format_p_value(0.031, "NEJM", bare=True) == "0.03"

    # JAMA: P = .03, P < .001, P > .99 (no leading zero)
    assert format_p_value(0.031, "JAMA") == "P = .03"
    assert format_p_value(0.0004, "JAMA") == "P < .001"
    assert format_p_value(0.995, "JAMA") == "P > .99"
    assert format_p_value(0.031, "JAMA", bare=True) == ".03"

    # LANCET: mid-dot, lowercase p, 4 dp for 0.0001-0.001
    assert format_p_value(0.014, "LANCET") == "p=0·01"
    assert format_p_value(0.0004, "LANCET") == "p=0·0004"
    assert format_p_value(0.00005, "LANCET") == "p<0·0001"
    assert format_p_value(0.0004, "LANCET", bare=True) == "0·0004"

    # BMJ: P = 0.03, P < 0.001 (keeps leading zero, spaced)
    assert format_p_value(0.031, "BMJ") == "P = 0.03"
    assert format_p_value(0.0004, "BMJ") == "P < 0.001"
    assert format_p_value(0.031, "BMJ", bare=True) == "0.03"

    # CLINICAL: P=0.03, P<0.001
    assert format_p_value(0.031, "CLINICAL") == "P=0.03"


def test_format_estimate_rules():
    # HR 1.45 (1.12 to 1.88)
    nejm_est = format_estimate(1.45, 1.12, 1.88, "HR", journal="NEJM")
    assert "HR, 1.45" in nejm_est
    assert "95% CI" in nejm_est

    jama_est = format_estimate(1.45, 1.12, 1.88, "HR", journal="JAMA")
    assert "HR, 1.45" in jama_est

    lancet_est = format_estimate(1.45, 1.12, 1.88, "HR", journal="LANCET")
    assert "1·45" in lancet_est
    assert "1·12" in lancet_est

    bmj_est = format_estimate(1.45, 1.12, 1.88, "HR", journal="BMJ")
    assert "HR 1.45 (95% CI 1.12 to 1.88)" in bmj_est


def test_assert_zero_phi_text_raises_on_violations():
    # Valid clinical terms
    assert_zero_phi_text(["Survival Probability", "Cohort (n = 120)", "Stage 2"])

    # Violations
    with pytest.raises(ValueError, match="Zero-PHI Violation"):
        assert_zero_phi_text(["Patient HN 123456"])

    with pytest.raises(ValueError, match="Zero-PHI Violation"):
        assert_zero_phi_text(["ID: 1100500123456"])

    with pytest.raises(ValueError, match="Zero-PHI Violation"):
        assert_zero_phi_text(["Dr. Somchai report"])

    with pytest.raises(ValueError, match="Zero-PHI Violation"):
        assert_zero_phi_text(["นพ. สมชาย"])


def test_qa_layout_gate():
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [0, 1])
    ax.set_title("Clean Title")
    qa = qa_layout(fig)
    assert qa["passed"] is True
    plt.close(fig)

    # Intentionally collide two texts at the exact same coordinates
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.text(0.5, 0.5, "Long Text Colliding A", fontsize=12)
    ax.text(0.5, 0.5, "Long Text Colliding B", fontsize=12)
    qa_col = qa_layout(fig)
    assert qa_col["passed"] is False
    assert len(qa_col["text_overlaps"]) > 0
    plt.close(fig)


def test_cvd_check_computation():
    palette = {
        "c1": "#BC3C29",
        "c2": "#0072B5",
        "c3": "#E18727",
    }
    df = cvd_check(pairs=[("c1", "c2"), ("c1", "c3")], palette=palette, min_de=12.0)
    assert isinstance(df, pd.DataFrame)
    assert "deltaE" in df.columns
    assert "needs_cue" in df.columns
    assert len(df) == 2 * 4  # 2 pairs * 4 vision modes (normal, deutan, protan, tritan)


def test_figure_result_contract_extensions(tmp_path):
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    res = FigureResult(
        png_path=str(tmp_path / "plot.png"),
        alt_text="Test",
        caption="Caption",
        source_df=df,
        csv_path=str(tmp_path / "plot.csv"),
        vector_path=str(tmp_path / "plot.pdf"),
        svg_path=str(tmp_path / "plot.svg"),
        qa={"passed": True},
    )
    d = res.to_dict()
    assert d["vector_path"] == str(tmp_path / "plot.pdf")
    assert d["svg_path"] == str(tmp_path / "plot.svg")
    assert d["qa"] == {"passed": True}


def test_save_figure_bundle_generates_pdf_truetype(tmp_path):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [1, 0])
    png_p = tmp_path / "test_bundle.png"
    p_png, p_pdf, p_svg = save_figure_bundle(
        fig, png_p, dpi=300, export_vector=True, export_svg=True
    )
    assert p_png.endswith(".png")
    assert p_pdf is not None and p_pdf.endswith(".pdf")
    assert p_svg is not None and p_svg.endswith(".svg")
    assert (tmp_path / "test_bundle.png").exists()
    assert (tmp_path / "test_bundle.pdf").exists()
    assert (tmp_path / "test_bundle.svg").exists()

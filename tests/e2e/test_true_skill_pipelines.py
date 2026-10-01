"""
tests/e2e/test_true_skill_pipelines.py: End-to-End Skill Pipeline Chaining Tests.

Verifies true autonomous biostatistics pipelines:
1. medstat profile: Universal ingestion (.csv, .tsv, .xlsx, .parquet) and design inference.
2. Table 1 -> medstat report: Polymorphic HTML rendering of baseline patient characteristics.
3. Diagnostic testing -> medstat report: Polymorphic HTML rendering of 2x2 matrix and Wilson CIs.
4. Bland-Altman -> medstat report: Polymorphic HTML rendering of Mean Bias and Limits of Agreement.
5. Intraclass Correlation (ICC) -> medstat report: Polymorphic HTML rendering of rater reliability.
6. Causal PSM -> medstat report: Polymorphic HTML rendering of Austin (2009) covariate balance.
7. Reporting checklists: STARD and PRISMA guideline markdown audit exports.
"""

import pandas as pd


def test_e2e_profile_universal_formats(
    medstat_cli_runner, cardiovascular_fixture_path, tmp_path
):
    """Test medstat profile across CSV, TSV, Excel (.xlsx), and Parquet formats."""
    # Read base fixture
    df = pd.read_csv(cardiovascular_fixture_path)

    tsv_file = tmp_path / "cohort.tsv"
    xlsx_file = tmp_path / "cohort.xlsx"
    parquet_file = tmp_path / "cohort.parquet"

    df.to_csv(tsv_file, sep="\t", index=False)
    df.to_excel(xlsx_file, index=False)
    df.to_parquet(parquet_file, index=False)

    for path in [cardiovascular_fixture_path, tsv_file, xlsx_file, parquet_file]:
        ec, out, _ = medstat_cli_runner(["profile", "--data", str(path)])
        assert ec == 0, f"Profile failed on {path.suffix}: {out}"
        assert "CLINICAL DATASET HEALTH PROFILE" in out
        assert "Inferred Design:" in out


def test_e2e_table1_to_report_pipeline(
    medstat_cli_runner, cardiovascular_fixture_path, tmp_path
):
    """Verify Table 1 JSON output piped cleanly into medstat report."""
    t1_json = tmp_path / "table1.json"
    report_html = tmp_path / "table1_report.html"

    # Step 1: Generate Table 1 JSON
    ec1, out1, _ = medstat_cli_runner(
        [
            "table1",
            "--data",
            str(cardiovascular_fixture_path),
            "--group",
            "statin_rx",
            "--output",
            str(t1_json),
        ]
    )
    assert ec1 == 0, f"Table 1 generation failed: {out1}"
    assert t1_json.exists()

    # Step 2: Render publication HTML table via medstat report
    ec2, out2, _ = medstat_cli_runner(
        [
            "report",
            "--results",
            str(t1_json),
            "--style",
            "nejm",
            "--format",
            "html",
            "--output",
            str(report_html),
        ]
    )
    assert ec2 == 0, f"Report generation failed: {out2}"
    assert report_html.exists()

    content = report_html.read_text(encoding="utf-8")
    assert "<table" in content.lower()
    assert "table 1" in content.lower() or "characteristic" in content.lower()


def test_e2e_diagnostic_to_report_pipeline(
    medstat_cli_runner, sepsis_fixture_path, tmp_path
):
    """Verify Diagnostic Accuracy JSON piped cleanly into medstat report."""
    diag_json = tmp_path / "diag.json"
    report_html = tmp_path / "diag_report.html"

    # Step 1: Run diagnostic accuracy analysis with direction handling
    ec1, out1, _ = medstat_cli_runner(
        [
            "diag",
            "--data",
            str(sepsis_fixture_path),
            "--gold-standard",
            "sepsis_confirmed_2x2",
            "--test-col",
            "lactate",
            "--cutoff",
            "2.0",
            "--direction",
            "high",
            "--roc",
            "--dca",
            "--output",
            str(diag_json),
        ]
    )
    assert ec1 == 0, f"Diagnostic testing failed: {out1}"
    assert diag_json.exists()

    # Step 2: Render publication HTML table via medstat report
    ec2, out2, _ = medstat_cli_runner(
        [
            "report",
            "--results",
            str(diag_json),
            "--style",
            "jama",
            "--format",
            "html",
            "--output",
            str(report_html),
        ]
    )
    assert ec2 == 0, f"Report generation failed: {out2}"
    assert report_html.exists()

    content = report_html.read_text(encoding="utf-8")
    assert "<table" in content.lower()
    assert "sensitivity" in content.lower() or "specificity" in content.lower()


def test_e2e_bland_altman_to_report_pipeline(
    medstat_cli_runner, pocus_fixture_path, tmp_path
):
    """Verify Bland-Altman JSON piped cleanly into medstat report."""
    # Pivot long-format POCUS reliability into paired columns for Bland-Altman
    df_long = pd.read_csv(pocus_fixture_path)
    df_piv = (
        df_long.pivot(
            index="subject_id", columns="rater_id", values="measurement_score"
        )
        .dropna()
        .reset_index()
    )
    rater_cols = [c for c in df_piv.columns if c != "subject_id"]
    r1, r2 = str(rater_cols[0]), str(rater_cols[1])
    piv_csv = tmp_path / "pocus_paired.csv"
    df_piv.to_csv(piv_csv, index=False)

    ba_json = tmp_path / "bland_altman.json"
    report_html = tmp_path / "ba_report.html"

    # Step 1: Compute Bland-Altman agreement
    ec1, out1, _ = medstat_cli_runner(
        [
            "agreement",
            "bland-altman",
            "--data",
            str(piv_csv),
            "--m1",
            r1,
            "--m2",
            r2,
            "--output",
            str(ba_json),
        ]
    )
    assert ec1 == 0, f"Bland-Altman failed: {out1}"
    assert ba_json.exists()

    # Step 2: Render publication HTML table via medstat report
    ec2, out2, _ = medstat_cli_runner(
        [
            "report",
            "--results",
            str(ba_json),
            "--style",
            "apa7",
            "--format",
            "html",
            "--output",
            str(report_html),
        ]
    )
    assert ec2 == 0, f"Report generation failed: {out2}"
    assert report_html.exists()

    content = report_html.read_text(encoding="utf-8")
    assert "<table" in content.lower()
    assert "mean difference" in content.lower() or "bias" in content.lower()


def test_e2e_icc_to_report_pipeline(medstat_cli_runner, pocus_fixture_path, tmp_path):
    """Verify ICC JSON piped cleanly into medstat report."""
    icc_json = tmp_path / "icc.json"
    report_html = tmp_path / "icc_report.html"

    # Step 1: Compute ICC
    ec1, out1, _ = medstat_cli_runner(
        [
            "agreement",
            "icc",
            "--data",
            str(pocus_fixture_path),
            "--targets",
            "subject_id",
            "--raters",
            "rater_id",
            "--ratings",
            "measurement_score",
            "--output",
            str(icc_json),
        ]
    )
    assert ec1 == 0, f"ICC failed: {out1}"
    assert icc_json.exists()

    # Step 2: Render publication HTML table via medstat report
    ec2, out2, _ = medstat_cli_runner(
        [
            "report",
            "--results",
            str(icc_json),
            "--style",
            "nejm",
            "--format",
            "html",
            "--output",
            str(report_html),
        ]
    )
    assert ec2 == 0, f"Report generation failed: {out2}"
    assert report_html.exists()

    content = report_html.read_text(encoding="utf-8")
    assert "<table" in content.lower()
    assert "icc" in content.lower() or "type" in content.lower()


def test_e2e_causal_psm_to_report_pipeline(
    medstat_cli_runner, cardiovascular_fixture_path, tmp_path
):
    """Verify Causal PSM JSON piped cleanly into medstat report."""
    psm_json = tmp_path / "psm.json"
    report_html = tmp_path / "psm_report.html"

    # Step 1: Run Propensity Score Matching
    ec1, out1, _ = medstat_cli_runner(
        [
            "causal",
            "psm",
            "--data",
            str(cardiovascular_fixture_path),
            "--treatment",
            "statin_rx",
            "--covariates",
            "age,sbp,ldl",
            "--caliper",
            "0.2",
            "--balance-check",
            "--output",
            str(psm_json),
        ]
    )
    assert ec1 == 0, f"Causal PSM failed: {out1}"
    assert psm_json.exists()

    # Step 2: Render publication HTML table via medstat report
    ec2, out2, _ = medstat_cli_runner(
        [
            "report",
            "--results",
            str(psm_json),
            "--style",
            "nejm",
            "--format",
            "html",
            "--output",
            str(report_html),
        ]
    )
    assert ec2 == 0, f"Report generation failed: {out2}"
    assert report_html.exists()

    content = report_html.read_text(encoding="utf-8")
    assert "<table" in content.lower()
    assert (
        "smd" in content.lower()
        or "balance" in content.lower()
        or "covariate" in content.lower()
    )


def test_e2e_stard_and_prisma_checklists(medstat_cli_runner, tmp_path):
    """Verify STARD and PRISMA checklist exports."""
    stard_md = tmp_path / "stard.md"
    prisma_md = tmp_path / "prisma.md"

    ec1, out1, _ = medstat_cli_runner(
        ["report", "--checklist", "stard", "--output", str(stard_md)]
    )
    assert ec1 == 0, f"STARD checklist failed: {out1}"
    assert stard_md.exists()
    stard_text = stard_md.read_text(encoding="utf-8")
    assert "# STARD Compliance Checklist" in stard_text
    assert "Index test" in stard_text

    ec2, out2, _ = medstat_cli_runner(
        ["report", "--checklist", "prisma", "--output", str(prisma_md)]
    )
    assert ec2 == 0, f"PRISMA checklist failed: {out2}"
    assert prisma_md.exists()
    prisma_text = prisma_md.read_text(encoding="utf-8")
    assert "# PRISMA Compliance Checklist" in prisma_text
    assert "Synthesis results" in prisma_text

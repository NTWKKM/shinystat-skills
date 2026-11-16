"""Unit tests for medstat.reporting.provenance (RunManifest, AnalysisLineage, integrity)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from medstat.reporting.integrity import verify_report_integrity
from medstat.reporting.ir import ReportDocument
from medstat.reporting.provenance import (
    REPAIR_CAP,
    AnalysisLineage,
    AnalysisNode,
    FrozenNodeError,
    LineageError,
    RepairCapExceededError,
    RunManifest,
    add_reproducibility_appendix,
    build_run_manifest,
    hash_dataframe,
    hash_plan,
    plan_delta,
)
from medstat.reporting.renderers import render_html, render_markdown


def test_hash_dataframe_canonical_and_fail_closed():
    df1 = pd.DataFrame({"b": [1.0, np.nan, 3.0], "a": [10, 20, 30]})
    df2 = pd.DataFrame({"a": [10, 20, 30], "b": [1.0, None, 3.0]})
    assert hash_dataframe(df1) == hash_dataframe(df2)

    # Row order matters
    df_reordered = df1.iloc[[1, 0, 2]].reset_index(drop=True)
    assert hash_dataframe(df1) != hash_dataframe(df_reordered)

    # Value change matters
    df_val = df1.copy()
    df_val.loc[0, "a"] = 11
    assert hash_dataframe(df1) != hash_dataframe(df_val)

    # Dtype change matters (int64 vs float64)
    df_dtype = df1.copy()
    df_dtype["a"] = df_dtype["a"].astype(float)
    assert hash_dataframe(df1) != hash_dataframe(df_dtype)

    with pytest.raises(TypeError):
        hash_dataframe({"a": [1, 2]})  # type: ignore[arg-type]


def test_hash_plan_and_plan_delta():
    p1 = {
        "model": {"estimator": "cox", "penalizer": 0.0},
        "missing": {"strategy": "mice"},
    }
    p2 = {
        "missing": {"strategy": "mice"},
        "model": {"penalizer": 0.0, "estimator": "cox"},
    }
    assert hash_plan(p1) == hash_plan(p2)

    # bool vs int distinction
    assert hash_plan({"flag": True}) != hash_plan({"flag": 1})
    assert plan_delta({"flag": True}, {"flag": 1}) == {"flag": (True, 1)}

    child = {
        "model": {"estimator": "firth", "penalizer": 0.5},
        "missing": {"strategy": "mice"},
        "extra": "note",
    }
    delta = plan_delta(p1, child)
    assert delta == {
        "extra": ("<absent>", "note"),
        "model.estimator": ("cox", "firth"),
        "model.penalizer": (0.0, 0.5),
    }

    with pytest.raises(TypeError):
        hash_plan(["not", "a", "mapping"])  # type: ignore[arg-type]


def test_build_run_manifest_zero_phi_and_timezone():
    df = pd.DataFrame({"patient_secret_col": [1, 2, 3], "outcome": [0, 1, 0]})
    plan = {"model": {"estimator": "logistic"}}
    manifest = build_run_manifest(
        df,
        plan=plan,
        seed=42,
        packages=("pandas", "nonexistent-pkg-xyz-999"),
    )
    assert isinstance(manifest, RunManifest)
    assert manifest.data_sha256 == hash_dataframe(df)
    assert manifest.plan_sha256 == hash_plan(plan)
    assert manifest.n_rows == 3
    assert manifest.n_cols == 2
    assert manifest.seed == 42
    assert manifest.package_versions["nonexistent-pkg-xyz-999"] == "not installed"
    assert manifest.package_versions["pandas"] != "not installed"
    assert manifest.created_at.endswith("+07:00")

    serialized = str(dataclasses.asdict(manifest))
    assert "patient_secret_col" not in serialized
    assert "/" not in manifest.data_sha256


def test_lineage_freeze_delta_and_repair_cap():
    lineage = AnalysisLineage()
    p_plan = {"model": {"estimator": "logistic"}, "missing": {"strategy": "mice"}}
    lineage.add_primary("primary", "Adjusted mortality risk", p_plan)

    with pytest.raises(LineageError):
        lineage.add_primary("primary_2", "Second primary", p_plan)

    # Empty delta sensitivity fails
    with pytest.raises(LineageError):
        lineage.add_sensitivity("sens_same", "primary", "Same plan", dict(p_plan))

    # Unknown parent fails
    with pytest.raises(LineageError):
        lineage.add_sensitivity(
            "sens_orphan",
            "missing_parent",
            "Orphan",
            {"model": {"estimator": "firth"}, "missing": {"strategy": "mice"}},
        )

    # Exploratory cannot parent sensitivity
    lineage.add_exploratory("exp_1", "Subgroup check", {"subgroup": "age>65"})
    with pytest.raises(LineageError):
        lineage.add_sensitivity(
            "sens_from_exp",
            "exp_1",
            "Off exploratory",
            {"subgroup": "age>70"},
        )

    # Repair provisional node up to REPAIR_CAP, then halt
    assert REPAIR_CAP == 2
    lineage.repair(
        "primary",
        {
            "model": {"estimator": "logistic", "maxiter": 200},
            "missing": {"strategy": "mice"},
        },
        "Non-convergence #1",
    )
    lineage.repair(
        "primary",
        {
            "model": {"estimator": "logistic", "maxiter": 500},
            "missing": {"strategy": "mice"},
        },
        "Non-convergence #2",
    )
    with pytest.raises(RepairCapExceededError) as exc_info:
        lineage.repair(
            "primary",
            {"model": {"estimator": "firth"}, "missing": {"strategy": "mice"}},
            "Non-convergence #3",
        )
    assert "Gate #7" in exc_info.value.agent_remediation

    # record_failure also enforces cap
    lineage.add_exploratory("exp_fail", "Failing exploratory", {"x": 1})
    lineage.record_failure("exp_fail", "Err 1")
    lineage.record_failure("exp_fail", "Err 2")
    with pytest.raises(RepairCapExceededError):
        lineage.record_failure("exp_fail", "Err 3")

    # Mark primary answered -> frozen
    lineage.mark_answered("primary", "OR 1.45 (95% CI, 1.12-1.88)")
    with pytest.raises(FrozenNodeError) as frozen_exc:
        lineage.repair("primary", p_plan, "Post-answer tweak")
    assert frozen_exc.value.agent_remediation
    with pytest.raises(FrozenNodeError):
        lineage.mark_answered("primary", "OR 1.50")

    # Valid sensitivity child off answered primary
    lineage.add_sensitivity(
        "sens_cc",
        "primary",
        "Complete-case sensitivity",
        {
            "model": {"estimator": "logistic", "maxiter": 500},
            "missing": {"strategy": "complete-case"},
        },
    )
    lineage.mark_answered("sens_cc", "OR 1.42 (95% CI, 1.09-1.85)")

    assert lineage.violations() == []
    lines = lineage.summary_lines()
    assert lines[0].startswith(
        "- [primary · answered] Adjusted mortality risk: OR 1.45"
    )
    assert (
        "  - [sensitivity · answered] Complete-case sensitivity (Δ missing.strategy): OR 1.42"
        in lines[1]
    )

    # Direct structural violation via malformed AnalysisNode is caught by violations()
    bad_lineage = AnalysisLineage()
    bad_lineage.nodes.append(
        AnalysisNode(
            node_id="sens_bad",
            role="sensitivity",
            question="Missing parent",
            plan=p_plan,
            plan_sha256=hash_plan(p_plan),
            parent_id="nonexistent",
        )
    )
    assert bad_lineage.violations()


def test_reproducibility_appendix_and_integrity_checks(tmp_path: Path):
    df = pd.DataFrame({"exposure": [0, 1, 0, 1], "outcome": [0, 1, 0, 0]})
    plan = {"model": {"estimator": "logistic"}}
    results = {"or": 1.45, "ci_low": 1.12, "ci_high": 1.88, "n": 4}

    # Without manifest -> warning only, still passes
    doc_no_manifest = ReportDocument(title="No Manifest", results_dict=results)
    doc_no_manifest.add_paragraph(
        "Methods: cohort retention flow analyzed 4 patients; OR 1.45 (95% CI, 1.12-1.88)."
    )
    audit_warn = verify_report_integrity(doc_no_manifest)
    assert audit_warn.passed is True
    assert audit_warn.has_manifest is False
    assert any("manifest" in w.lower() for w in audit_warn.warnings)

    # With manifest + answered lineage + synthetic 13-digit-prefix sha256 collision test
    manifest = build_run_manifest(df, plan=plan, seed=42)
    colliding_manifest = dataclasses.replace(
        manifest,
        data_sha256="0123456789012" + "a" * 51,
        plan_sha256="081234567890" + "b" * 52,
    )
    lineage = AnalysisLineage()
    lineage.add_primary("primary", "Primary mortality model", plan)
    lineage.mark_answered("primary", "OR 1.45 (95% CI, 1.12-1.88)")
    lineage.add_sensitivity(
        "sens_cc",
        "primary",
        "Complete-case sensitivity",
        {"model": {"estimator": "firth"}},
    )
    lineage.mark_answered("sens_cc", "OR 1.45")

    doc = ReportDocument(
        title="Lineage Report",
        results_dict=results,
        manifest=colliding_manifest,
        lineage=lineage,
    )
    doc.add_paragraph(
        "Methods: cohort retention flow analyzed 4 patients (hash sha256:012345678901); OR 1.45 (95% CI, 1.12-1.88)."
    )
    add_reproducibility_appendix(doc)

    # Passes audit with real manifest when data matches
    doc_real = ReportDocument(
        title="Real Manifest",
        results_dict=results,
        manifest=manifest,
        lineage=lineage,
    )
    doc_real.add_paragraph(
        "Methods: cohort retention flow analyzed 4 patients; OR 1.45 (95% CI, 1.12-1.88)."
    )
    add_reproducibility_appendix(doc_real)

    audit_ok = verify_report_integrity(
        doc_real,
        data=df,
        delivered_text="\n".join(lineage.summary_lines()),
    )
    assert audit_ok.passed is True, (
        f"phi={audit_ok.phi_violations}, untraced={audit_ok.untraced_numbers}, lineage={audit_ok.lineage_violations}"
    )
    assert audit_ok.has_manifest is True
    assert audit_ok.lineage_violations == []

    # Colliding hash prefix does not trigger false PHI or untraced number
    audit_col = verify_report_integrity(doc)
    assert audit_col.passed is True, (
        f"phi={audit_col.phi_violations}, untraced={audit_col.untraced_numbers}"
    )

    # Renders cleanly to MD and HTML
    md_out = render_markdown(doc_real, tmp_path / "report.md")
    html_out = render_html(doc_real, tmp_path / "report.html")
    assert Path(md_out).exists() and Path(html_out).exists()

    # Data mismatch fails audit
    df_changed = df.copy()
    df_changed.loc[0, "outcome"] = 1
    audit_mismatch = verify_report_integrity(doc_real, data=df_changed)
    assert audit_mismatch.passed is False
    assert any("hash" in v.lower() for v in audit_mismatch.lineage_violations)

    # Provisional primary fails final audit
    prov_lineage = AnalysisLineage()
    prov_lineage.add_primary("primary", "Unfinished primary", plan)
    doc_prov = ReportDocument(
        title="Provisional",
        results_dict=results,
        manifest=manifest,
        lineage=prov_lineage,
    )
    doc_prov.add_paragraph("Methods: cohort retention flow OR 1.45.")
    audit_prov = verify_report_integrity(doc_prov, data=df)
    assert audit_prov.passed is False
    assert any("provisional" in v.lower() for v in audit_prov.lineage_violations)

    # PHI in plan_delta fails audit
    phi_lineage = AnalysisLineage()
    phi_lineage.add_primary("primary", "Primary", {"filter": "age >= 18"})
    phi_lineage.mark_answered("primary", "OR 1.45")
    phi_lineage.add_sensitivity(
        "sens_phi", "primary", "Filtered", {"filter": "HN: 123456"}
    )
    phi_lineage.mark_answered("sens_phi", "OR 1.45")
    doc_phi = ReportDocument(
        title="PHI Delta",
        results_dict=results,
        manifest=manifest,
        lineage=phi_lineage,
    )
    doc_phi.add_paragraph("Methods: cohort retention flow OR 1.45.")
    audit_phi = verify_report_integrity(doc_phi, data=df)
    assert audit_phi.passed is False
    assert audit_phi.phi_violations

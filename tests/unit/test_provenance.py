"""Unit tests for medstat.reporting.provenance (RunManifest, AnalysisLineage, integrity)."""

from __future__ import annotations

import copy
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

    # Missing value must not collide with literal "<NA>" string
    df_missing = pd.DataFrame({"x": [None, "a"]})
    df_literal_na = pd.DataFrame({"x": ["<NA>", "a"]})
    assert hash_dataframe(df_missing) != hash_dataframe(df_literal_na)

    # Categorical categories order and ordered flag affect digest
    df_cat_ab = pd.DataFrame(
        {"grp": pd.Categorical(["a", "b"], categories=["a", "b"], ordered=True)}
    )
    df_cat_ba = pd.DataFrame(
        {"grp": pd.Categorical(["a", "b"], categories=["b", "a"], ordered=True)}
    )
    df_cat_unord = pd.DataFrame(
        {"grp": pd.Categorical(["a", "b"], categories=["a", "b"], ordered=False)}
    )
    assert hash_dataframe(df_cat_ab) != hash_dataframe(df_cat_ba)
    assert hash_dataframe(df_cat_ab) != hash_dataframe(df_cat_unord)

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

    # bool vs int distinction (scalar and nested sequences/mappings)
    assert hash_plan({"flag": True}) != hash_plan({"flag": 1})
    assert plan_delta({"flag": True}, {"flag": 1}) == {"flag": (True, 1)}
    assert plan_delta({"seq": [True, [1]]}, {"seq": [1, [True]]}) == {
        "seq": ([True, [1]], [1, [True]])
    }
    assert plan_delta({"seq": [(True, 0)]}, {"seq": [(1, 0)]}) == {
        "seq": ([(True, 0)], [(1, 0)])
    }
    assert plan_delta({"items": [{"flag": True}]}, {"items": [{"flag": 1}]}) == {
        "items": ([{"flag": True}], [{"flag": 1}])
    }

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

    # In-place mutation of answered plan is blocked (dict, list, set), while add_primary keeps mutable dict
    lin_freeze = AnalysisLineage()
    collision_set = set(range(80))
    for i in range(0, 45):
        collision_set.remove(i)
    p_node = lin_freeze.add_primary(
        "primary",
        "Freeze check",
        {
            "model": {"estimator": "logistic"},
            "covariates": ["age"],
            "tags": {"primary", "adjusted", "sensitivity_ready"},
            "ids": collision_set,
        },
    )
    assert isinstance(p_node.plan, dict)
    lin_freeze.mark_answered("primary", "OR 1.45")
    assert isinstance(p_node.plan, dict)
    assert lin_freeze.violations() == []
    with pytest.raises(FrozenNodeError):
        p_node.plan["model"]["estimator"] = "firth"
    with pytest.raises(FrozenNodeError):
        p_node.plan["covariates"].append("sex")
    with pytest.raises(FrozenNodeError):
        p_node.plan["tags"].add("extra")
    with pytest.raises(FrozenNodeError):
        p_node.plan = {"model": {"estimator": "firth"}}
    assert dataclasses.asdict(p_node)["plan"]["model"]["estimator"] == "logistic"

    # Unsupported mutable value types are rejected when freezing
    lin_bad_type = AnalysisLineage()
    lin_bad_type.add_primary("primary", "Unsupported", {"buf": bytearray(b"abc")})
    with pytest.raises(TypeError, match="Unsupported plan value type"):
        lin_bad_type.mark_answered("primary", "OR 1.00")

    # Repairing a parent after sensitivity children exist raises LineageError
    lin_parent_child = AnalysisLineage()
    lin_parent_child.add_primary("primary", "Primary", {"a": 1})
    lin_parent_child.add_sensitivity("sens", "primary", "Child", {"a": 2})
    with pytest.raises(LineageError):
        lin_parent_child.repair("primary", {"a": 3}, "Late repair")

    # Tampered plan_sha256 and stale plan_delta are both caught by violations()
    lin_tamper = AnalysisLineage()
    t_node = lin_tamper.add_primary("primary", "Tampered", {"flag": True})
    t_node.plan["flag"] = 1
    assert any("plan_sha256" in v for v in lin_tamper.violations())

    # Set vs mapping with reserved-looking key must not collide in hash_plan / violations()
    assert hash_plan({"covariates": {"age"}}) != hash_plan(
        {"covariates": {"__set__": ["age"]}}
    )
    lin_set_swap = AnalysisLineage()
    s_swap_node = lin_set_swap.add_primary(
        "primary", "Set swap", {"covariates": {"age"}}
    )
    s_swap_node.plan["covariates"] = {"__set__": ["age"]}
    with pytest.raises(LineageError, match="plan_sha256"):
        lin_set_swap.mark_answered("primary", "Answered")
    assert any("plan_sha256" in v for v in lin_set_swap.violations())

    lin_stale = AnalysisLineage()
    lin_stale.add_primary("primary", "Primary", {"flag": True})
    lin_stale.mark_answered("primary", "OR 1.00")
    s_node = lin_stale.add_sensitivity("sens", "primary", "Sens", {"flag": 1})
    s_node.plan_delta = {"flag": (1, 1)}
    assert any("stale plan_delta" in v for v in lin_stale.violations())


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


def _get_nested(mapping: dict, dotted_key: str, default=None):
    cur = mapping
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _set_nested(mapping: dict, dotted_key: str, value) -> None:
    parts = dotted_key.split(".")
    cur = mapping
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = copy.deepcopy(value)


def _del_nested(mapping: dict, dotted_key: str) -> None:
    parts = dotted_key.split(".")
    cur = mapping
    for part in parts[:-1]:
        if not isinstance(cur, dict) or part not in cur:
            return
        cur = cur[part]
    if isinstance(cur, dict):
        cur.pop(parts[-1], None)


def _reconcile_provisional_primary(
    lineage: AnalysisLineage, updated_primary: dict
) -> None:
    """Reconcile a provisional primary and all provisional descendants in topological order.

    Halts with ValueError before committing any mutation if an answered descendant exists
    or if a provisional child's intended change conflicts with its updated parent plan.
    """
    primary = lineage._get("primary")
    if primary.status != "provisional" or any(
        n.status == "answered" for n in lineage.nodes[1:]
    ):
        raise ValueError(
            "Cannot reconcile in place when primary or any descendant is answered."
        )

    old_plans: dict[str, dict] = {
        n.node_id: copy.deepcopy(n.plan) for n in lineage.nodes
    }
    updated_plans: dict[str, dict] = {"primary": copy.deepcopy(updated_primary)}
    updated_parents: dict[str, str | None] = {"primary": None}
    updated_deltas: dict[str, dict] = {"primary": {}}
    retired: set[str] = set()
    reparent_to: dict[str, str] = {}
    kept_ids: list[str] = ["primary"]

    for node in lineage.nodes[1:]:
        orig_parent_id = node.parent_id or ""
        target_parent_id = orig_parent_id
        was_reparented = False
        if orig_parent_id in retired:
            if orig_parent_id not in reparent_to:
                retired.add(node.node_id)
                continue
            target_parent_id = reparent_to[orig_parent_id]
            was_reparented = True

        if target_parent_id not in updated_plans:
            retired.add(node.node_id)
            continue

        parent_node = lineage._get(target_parent_id)
        if parent_node.role == "exploratory":
            retired.add(node.node_id)
            continue

        new_parent_plan = updated_plans[target_parent_id]
        old_parent_plan = old_plans[orig_parent_id]
        reapplied = copy.deepcopy(new_parent_plan)

        for dotted_key, (_, child_val) in node.plan_delta.items():
            old_pval = _get_nested(old_parent_plan, dotted_key, "<absent>")
            new_pval = _get_nested(new_parent_plan, dotted_key, "<absent>")
            if old_pval != new_pval and new_pval != child_val:
                raise ValueError(
                    f"Conflict on '{node.node_id}' key '{dotted_key}': "
                    f"parent changed {old_pval!r} -> {new_pval!r}, child had {child_val!r}."
                )
            if child_val == "<absent>":
                _del_nested(reapplied, dotted_key)
            else:
                _set_nested(reapplied, dotted_key, child_val)

        new_delta = plan_delta(new_parent_plan, reapplied)
        if not new_delta:
            retired.add(node.node_id)
            if not was_reparented:
                reparent_to[node.node_id] = target_parent_id
            continue

        updated_plans[node.node_id] = reapplied
        updated_parents[node.node_id] = target_parent_id
        updated_deltas[node.node_id] = new_delta
        kept_ids.append(node.node_id)

    # Commit atomically only after validation succeeds.
    primary.plan = updated_plans["primary"]
    primary.plan_sha256 = hash_plan(primary.plan)
    kept_nodes: list[AnalysisNode] = [primary]
    for nid in kept_ids[1:]:
        n = lineage._get(nid)
        n.parent_id = updated_parents[nid]
        n.plan = updated_plans[nid]
        n.plan_sha256 = hash_plan(n.plan)
        n.plan_delta = updated_deltas[nid]
        kept_nodes.append(n)
    lineage.nodes = kept_nodes


def test_provisional_primary_reconciliation_retains_child_and_reconciles_grandchild():
    """Topological reconciliation updates retained children, their grandchildren, and retired-child descendants."""
    lineage = AnalysisLineage()
    p_old = {
        "model": {"estimator": "logistic"},
        "missing_strategy": "complete-case",
        "m": None,
        "seed": 1,
    }
    c_firth = {
        "model": {"estimator": "firth"},
        "missing_strategy": "complete-case",
        "m": None,
        "seed": 1,
    }
    g_firth_trim = {
        "model": {"estimator": "firth"},
        "missing_strategy": "complete-case",
        "m": None,
        "seed": 1,
        "trim": True,
    }
    c_mice = {
        "model": {"estimator": "logistic"},
        "missing_strategy": "mice",
        "m": 5,
        "seed": 1,
    }
    g_mice_sub = {
        "model": {"estimator": "logistic"},
        "missing_strategy": "mice",
        "m": 5,
        "seed": 1,
        "subgroup": "elderly",
    }
    g_mice_seed2 = {
        "model": {"estimator": "logistic"},
        "missing_strategy": "mice",
        "m": 5,
        "seed": 2,
    }
    gg_mice_seed2_sub = {
        "model": {"estimator": "logistic"},
        "missing_strategy": "mice",
        "m": 5,
        "seed": 2,
        "subgroup": "female",
    }

    lineage.add_primary("primary", "Primary effect", p_old)
    lineage.add_sensitivity("sens_firth", "primary", "Firth sensitivity", c_firth)
    lineage.add_sensitivity(
        "sens_firth_trim", "sens_firth", "Trimmed Firth", g_firth_trim
    )
    lineage.add_sensitivity("sens_mice", "primary", "MICE sensitivity", c_mice)
    lineage.add_sensitivity(
        "sens_mice_sub", "sens_mice", "Elderly MICE subgroup", g_mice_sub
    )
    lineage.add_sensitivity("sens_mice_seed2", "sens_mice", "MICE seed 2", g_mice_seed2)
    lineage.add_sensitivity(
        "sens_mice_seed2_sub",
        "sens_mice_seed2",
        "Female MICE seed 2 subgroup",
        gg_mice_seed2_sub,
    )

    # Gate #4 switches provisional primary to MICE (m=5, seed=2) without calling repair().
    updated_primary = {
        "model": {"estimator": "logistic"},
        "missing_strategy": "mice",
        "m": 5,
        "seed": 2,
    }
    _reconcile_provisional_primary(lineage, updated_primary)

    assert [n.node_id for n in lineage.nodes] == [
        "primary",
        "sens_firth",
        "sens_firth_trim",
        "sens_mice_sub",
    ]
    primary = lineage._get("primary")
    assert primary.failed_attempts == []

    # Retained child inherited MICE (m=5, seed=2) while keeping nested Firth estimator.
    firth_node = lineage._get("sens_firth")
    assert firth_node.parent_id == "primary"
    assert firth_node.plan == updated_primary | {"model": {"estimator": "firth"}}
    assert "model.estimator" not in firth_node.plan
    assert firth_node.plan_delta == {"model.estimator": ("logistic", "firth")}

    # Grandchild of retained child inherited MICE (m=5, seed=2) AND nested Firth.
    firth_trim_node = lineage._get("sens_firth_trim")
    assert firth_trim_node.parent_id == "sens_firth"
    assert firth_trim_node.plan == updated_primary | {
        "model": {"estimator": "firth"},
        "trim": True,
    }
    assert "model.estimator" not in firth_trim_node.plan
    assert firth_trim_node.plan_delta == {"trim": ("<absent>", True)}

    # Grandchild of retired sens_mice was reparented to primary with clean delta,
    # whereas sens_mice_seed2 (differing only by seed=2) and its subgroup child were removed.
    mice_sub_node = lineage._get("sens_mice_sub")
    assert mice_sub_node.parent_id == "primary"
    assert mice_sub_node.plan == updated_primary | {"subgroup": "elderly"}
    assert mice_sub_node.plan_delta == {"subgroup": ("<absent>", "elderly")}

    assert lineage.violations() == []
    lineage.mark_answered("primary", "OR 1.10 (95% CI 0.69-1.77, P = 0.69)")
    lineage.mark_answered("sens_firth", "OR 1.12 (95% CI 0.70-1.80, P = 0.64)")
    lineage.mark_answered("sens_firth_trim", "OR 1.15 (95% CI 0.71-1.85, P = 0.58)")
    lineage.mark_answered("sens_mice_sub", "OR 1.08 (95% CI 0.66-1.76, P = 0.75)")
    assert lineage.violations() == []

    df = pd.DataFrame({"outcome": [0, 1, 0, 1], "x": [1.0, 2.0, 3.0, 4.0]})
    manifest = build_run_manifest(df, plan=updated_primary, seed=42)
    doc = ReportDocument(
        title="Topological Lineage Reconciliation",
        results_dict={
            "primary": {"or": 1.10, "ci_l": 0.69, "ci_u": 1.77, "p": 0.69},
            "sens_firth": {"or": 1.12, "ci_l": 0.70, "ci_u": 1.80, "p": 0.64},
            "sens_firth_trim": {"or": 1.15, "ci_l": 0.71, "ci_u": 1.85, "p": 0.58},
            "sens_mice_sub": {"or": 1.08, "ci_l": 0.66, "ci_u": 1.76, "p": 0.75},
        },
        manifest=manifest,
        lineage=lineage,
    )
    doc.add_paragraph(
        "Methods: cohort retention flow OR 1.10 (95% CI 0.69-1.77, P = 0.69)."
    )
    add_reproducibility_appendix(doc)
    audit = verify_report_integrity(doc, data=df)
    assert audit.passed is True, audit.lineage_violations
    assert audit.has_manifest is True
    assert audit.lineage_violations == []


def test_provisional_primary_reconciliation_halts_on_conflict():
    """Conflicting child change (m=10 vs updated primary m=5) halts before mutating lineage."""
    lineage = AnalysisLineage()
    p_old = {
        "model": {"estimator": "logistic"},
        "missing_strategy": "complete-case",
        "m": None,
    }
    c_firth = {
        "model": {"estimator": "firth"},
        "missing_strategy": "complete-case",
        "m": None,
    }
    c_m10 = {
        "model": {"estimator": "logistic"},
        "missing_strategy": "mice",
        "m": 10,
    }
    lineage.add_primary("primary", "Primary", p_old)
    lineage.add_sensitivity("sens_firth", "primary", "Firth sensitivity", c_firth)
    lineage.add_sensitivity("sens_m10", "primary", "MICE m=10", c_m10)

    updated_primary = {
        "model": {"estimator": "logistic"},
        "missing_strategy": "mice",
        "m": 5,
    }
    with pytest.raises(ValueError, match="Conflict on 'sens_m10' key 'm'"):
        _reconcile_provisional_primary(lineage, updated_primary)

    # Original lineage—including the earlier non-conflicting Firth sibling—remains intact.
    assert [n.node_id for n in lineage.nodes] == ["primary", "sens_firth", "sens_m10"]
    assert lineage._get("primary").plan == p_old
    assert lineage._get("primary").plan_sha256 == hash_plan(p_old)
    assert lineage._get("sens_firth").plan == c_firth
    assert lineage._get("sens_m10").plan == c_m10
    assert lineage.violations() == []

"""Run manifest and analysis-lineage tracking for reproducible clinical reporting."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal
from zoneinfo import ZoneInfo

import pandas as pd

if TYPE_CHECKING:
    from medstat.reporting.ir import ReportDocument

Role = Literal["primary", "sensitivity", "exploratory"]
Status = Literal["provisional", "answered"]

REPAIR_CAP = 2
DEFAULT_PACKAGES: tuple[str, ...] = (
    "medstat",
    "pandas",
    "numpy",
    "scipy",
    "statsmodels",
    "lifelines",
    "scikit-learn",
)
_ABSENT = "<absent>"
_TZ_BANGKOK = ZoneInfo("Asia/Bangkok")


class FrozenNodeError(RuntimeError):
    """Raised when attempting to mutate an answered (frozen) analysis node."""

    def __init__(
        self,
        message: str,
        agent_remediation: str = (
            "An answered node is frozen and cannot be repaired or re-answered. "
            "Branch a sensitivity child via add_sensitivity() instead."
        ),
    ) -> None:
        super().__init__(message)
        self.agent_remediation = agent_remediation


class RepairCapExceededError(RuntimeError):
    """Raised when a provisional node exceeds REPAIR_CAP failed repair attempts."""

    def __init__(
        self,
        message: str,
        agent_remediation: str = (
            "Repair cap (2) reached without an answer. Halt execution and ask the user (Gate #7)."
        ),
    ) -> None:
        super().__init__(message)
        self.agent_remediation = agent_remediation


class LineageError(ValueError):
    """Raised when an analysis lineage violates structural or pre-specification rules."""

    def __init__(
        self,
        message: str,
        agent_remediation: str = (
            "Ensure at most one primary node, a valid non-exploratory parent for each sensitivity "
            "node, and a non-empty plan delta between parent and child."
        ),
    ) -> None:
        super().__init__(message)
        self.agent_remediation = agent_remediation


def _normalize_mapping(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _normalize_mapping(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_mapping(v) for v in obj]
    if isinstance(obj, tuple):
        return [_normalize_mapping(v) for v in obj]
    return obj


def _values_equal(a: Any, b: Any) -> bool:
    if type(a) is bool or type(b) is bool:
        return type(a) is type(b) and a == b
    return a == b


def _flatten_mapping(mapping: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for k in sorted(mapping.keys(), key=str):
        key_str = str(k)
        full_key = f"{prefix}.{key_str}" if prefix else key_str
        val = mapping[k]
        if isinstance(val, Mapping) and len(val) > 0:
            flat.update(_flatten_mapping(val, full_key))
        else:
            flat[full_key] = val
    return flat


def hash_dataframe(df: pd.DataFrame) -> str:
    """Computes a canonical SHA-256 digest of a DataFrame (column-order independent, row/dtype sensitive)."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"hash_dataframe expects a pandas DataFrame, got {type(df).__name__}"
        )
    sorted_cols = sorted(df.columns, key=lambda c: (str(c), repr(c)))
    df_sorted = df.loc[:, sorted_cols]
    dtype_line = "|".join(f"{col}:{df_sorted[col].dtype}" for col in sorted_cols)
    csv_body = df_sorted.to_csv(
        index=False,
        float_format="%.17g",
        lineterminator="\n",
        na_rep="<NA>",
    )
    payload = f"{dtype_line}\n{csv_body}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def hash_plan(plan: Mapping[str, Any]) -> str:
    """Computes a canonical SHA-256 digest of an analysis plan mapping."""
    if not isinstance(plan, Mapping):
        raise TypeError(f"hash_plan expects a Mapping, got {type(plan).__name__}")
    normalized = _normalize_mapping(plan)
    canonical = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def plan_delta(
    parent: Mapping[str, Any], child: Mapping[str, Any]
) -> dict[str, tuple[Any, Any]]:
    """Returns dotted-path differences ``{path: (parent_val, child_val)}`` between two plans."""
    if not isinstance(parent, Mapping) or not isinstance(child, Mapping):
        raise TypeError("plan_delta expects Mapping arguments for parent and child")
    flat_parent = _flatten_mapping(parent)
    flat_child = _flatten_mapping(child)
    delta: dict[str, tuple[Any, Any]] = {}
    for key in sorted(set(flat_parent) | set(flat_child)):
        pv = flat_parent.get(key, _ABSENT)
        cv = flat_child.get(key, _ABSENT)
        if not _values_equal(pv, cv):
            delta[key] = (copy.deepcopy(pv), copy.deepcopy(cv))
    return delta


@dataclass(frozen=True)
class RunManifest:
    """Zero-PHI provenance record capturing data, plan, seed, and runtime environment."""

    data_sha256: str
    n_rows: int
    n_cols: int
    plan_sha256: str | None
    seed: int | None
    package_versions: dict[str, str]
    python_version: str
    created_at: str


def build_run_manifest(
    df: pd.DataFrame,
    plan: Mapping[str, Any] | None = None,
    seed: int | None = None,
    packages: Sequence[str] = DEFAULT_PACKAGES,
) -> RunManifest:
    """Builds a Zero-PHI RunManifest for an analysed cohort DataFrame and optional plan."""
    data_hash = hash_dataframe(df)
    plan_hash = hash_plan(plan) if plan is not None else None
    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    return RunManifest(
        data_sha256=data_hash,
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        plan_sha256=plan_hash,
        seed=seed,
        package_versions=versions,
        python_version=platform.python_version(),
        created_at=datetime.now(_TZ_BANGKOK).isoformat(timespec="seconds"),
    )


@dataclass
class AnalysisNode:
    """Single analysis run node in an AnalysisLineage.

    A node stays ``provisional`` while errors, non-convergence, or halted gates prevent an
    answer. Once it produces an estimate—or ``NaN`` with a fail-closed reason—calling
    ``mark_answered`` sets ``status="answered"`` and freezes it permanently.
    """

    node_id: str
    role: Role
    question: str
    plan: dict[str, Any]
    plan_sha256: str
    parent_id: str | None = None
    plan_delta: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    status: Status = "provisional"
    headline: str | None = None
    failed_attempts: list[str] = field(default_factory=list)


class AnalysisLineage:
    """Tracks pre-specified primary, sensitivity-delta, and exploratory analyses."""

    def __init__(self) -> None:
        self.nodes: list[AnalysisNode] = []

    def _get(self, node_id: str) -> AnalysisNode:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise LineageError(f"Unknown analysis node '{node_id}'.")

    def _ensure_unique_id(self, node_id: str) -> None:
        if not node_id or not node_id.strip():
            raise LineageError("node_id must be a non-empty string.")
        if any(n.node_id == node_id for n in self.nodes):
            raise LineageError(f"Duplicate analysis node_id '{node_id}'.")

    def add_primary(
        self, node_id: str, question: str, plan: Mapping[str, Any]
    ) -> AnalysisNode:
        self._ensure_unique_id(node_id)
        if any(n.role == "primary" for n in self.nodes):
            raise LineageError(
                "A primary analysis node already exists; only one primary analysis is allowed."
            )
        plan_copy = copy.deepcopy(_normalize_mapping(plan))
        node = AnalysisNode(
            node_id=node_id,
            role="primary",
            question=question,
            plan=plan_copy,
            plan_sha256=hash_plan(plan_copy),
        )
        self.nodes.append(node)
        return node

    def add_sensitivity(
        self,
        node_id: str,
        parent_id: str,
        question: str,
        plan: Mapping[str, Any],
    ) -> AnalysisNode:
        self._ensure_unique_id(node_id)
        parent = self._get(parent_id)
        if parent.role == "exploratory":
            raise LineageError(
                f"Sensitivity node '{node_id}' cannot branch from exploratory node '{parent_id}'."
            )
        plan_copy = copy.deepcopy(_normalize_mapping(plan))
        delta = plan_delta(parent.plan, plan_copy)
        if not delta:
            raise LineageError(
                f"Sensitivity node '{node_id}' has an empty plan_delta relative to parent '{parent_id}'."
            )
        node = AnalysisNode(
            node_id=node_id,
            role="sensitivity",
            question=question,
            plan=plan_copy,
            plan_sha256=hash_plan(plan_copy),
            parent_id=parent_id,
            plan_delta=delta,
        )
        self.nodes.append(node)
        return node

    def add_exploratory(
        self, node_id: str, question: str, plan: Mapping[str, Any]
    ) -> AnalysisNode:
        self._ensure_unique_id(node_id)
        plan_copy = copy.deepcopy(_normalize_mapping(plan))
        node = AnalysisNode(
            node_id=node_id,
            role="exploratory",
            question=question,
            plan=plan_copy,
            plan_sha256=hash_plan(plan_copy),
        )
        self.nodes.append(node)
        return node

    def repair(
        self, node_id: str, plan: Mapping[str, Any], reason: str
    ) -> AnalysisNode:
        node = self._get(node_id)
        if node.status == "answered":
            raise FrozenNodeError(
                f"Node '{node_id}' is already answered and frozen; cannot repair in place."
            )
        if len(node.failed_attempts) >= REPAIR_CAP:
            raise RepairCapExceededError(
                f"Node '{node_id}' reached repair cap ({REPAIR_CAP}) after: {node.failed_attempts}"
            )
        plan_copy = copy.deepcopy(_normalize_mapping(plan))
        new_delta: dict[str, tuple[Any, Any]] = {}
        if node.parent_id is not None:
            parent = self._get(node.parent_id)
            new_delta = plan_delta(parent.plan, plan_copy)
            if node.role == "sensitivity" and not new_delta:
                raise LineageError(
                    f"Repaired sensitivity node '{node_id}' has an empty plan_delta relative to '{node.parent_id}'."
                )
        node.failed_attempts.append(reason)
        node.plan = plan_copy
        node.plan_sha256 = hash_plan(plan_copy)
        node.plan_delta = new_delta
        return node

    def record_failure(self, node_id: str, reason: str) -> None:
        node = self._get(node_id)
        if node.status == "answered":
            raise FrozenNodeError(
                f"Node '{node_id}' is already answered and frozen; cannot record provisional failure."
            )
        if len(node.failed_attempts) >= REPAIR_CAP:
            raise RepairCapExceededError(
                f"Node '{node_id}' reached repair cap ({REPAIR_CAP}) after: {node.failed_attempts}"
            )
        node.failed_attempts.append(reason)

    def mark_answered(self, node_id: str, headline: str) -> AnalysisNode:
        node = self._get(node_id)
        if node.status == "answered":
            raise FrozenNodeError(f"Node '{node_id}' is already answered and frozen.")
        if not headline or not str(headline).strip():
            raise LineageError(
                f"Node '{node_id}' requires a non-empty headline result to be marked answered."
            )
        node.status = "answered"
        node.headline = str(headline).strip()
        return node

    def violations(self) -> list[str]:
        issues: list[str] = []
        by_id: dict[str, AnalysisNode] = {}
        primary_count = 0
        for node in self.nodes:
            if node.node_id in by_id:
                issues.append(f"Duplicate node_id '{node.node_id}'.")
            by_id[node.node_id] = node
            if node.role == "primary":
                primary_count += 1
        if primary_count > 1:
            issues.append(
                f"Multiple primary nodes ({primary_count}); expected at most 1."
            )
        for node in self.nodes:
            if node.role == "sensitivity":
                if not node.parent_id or node.parent_id not in by_id:
                    issues.append(
                        f"Sensitivity node '{node.node_id}' references missing parent '{node.parent_id}'."
                    )
                else:
                    parent = by_id[node.parent_id]
                    if parent.role == "exploratory":
                        issues.append(
                            f"Sensitivity node '{node.node_id}' cannot branch from exploratory parent '{parent.node_id}'."
                        )
                    if not plan_delta(parent.plan, node.plan):
                        issues.append(
                            f"Sensitivity node '{node.node_id}' has empty plan_delta relative to '{parent.node_id}'."
                        )
            if node.status == "answered" and (
                not node.headline or not str(node.headline).strip()
            ):
                issues.append(
                    f"Answered node '{node.node_id}' is missing a headline result."
                )
            if len(node.failed_attempts) > REPAIR_CAP:
                issues.append(
                    f"Node '{node.node_id}' exceeded repair cap ({len(node.failed_attempts)} > {REPAIR_CAP})."
                )
        return issues

    def summary_lines(self) -> list[str]:
        children_by_parent: dict[str | None, list[AnalysisNode]] = {}
        node_ids = {n.node_id for n in self.nodes}
        for node in self.nodes:
            key = node.parent_id if node.parent_id in node_ids else None
            children_by_parent.setdefault(key, []).append(node)

        lines: list[str] = []
        visited: set[str] = set()

        def _visit(node: AnalysisNode, depth: int) -> None:
            if node.node_id in visited:
                return
            visited.add(node.node_id)
            indent = "  " * depth
            headline = node.headline if node.headline else "pending"
            if node.role == "sensitivity":
                delta_keys = ", ".join(sorted(node.plan_delta.keys())) or "none"
                prefix = indent or "  "
                lines.append(
                    f"{prefix}- [{node.role} · {node.status}] {node.question} (Δ {delta_keys}): {headline}"
                )
            else:
                lines.append(
                    f"{indent}- [{node.role} · {node.status}] {node.question}: {headline}"
                )
            for child in children_by_parent.get(node.node_id, []):
                _visit(child, depth + 1)

        for root in children_by_parent.get(None, []):
            _visit(root, 0)
        for node in self.nodes:
            if node.node_id not in visited:
                _visit(node, 0)
        return lines


def add_reproducibility_appendix(doc: ReportDocument) -> ReportDocument:
    """Appends a Zero-PHI Reproducibility & Analysis Lineage appendix to ``doc``."""
    if doc.manifest is None and doc.lineage is None:
        return doc
    doc.add_heading("Reproducibility & Analysis Lineage", level=2)
    if doc.manifest is not None:
        m = doc.manifest
        pkg_str = ", ".join(f"{k}={v}" for k, v in m.package_versions.items())
        rows = [
            ("Data SHA-256", f"sha256:{m.data_sha256[:12]}"),
            ("Dataset Shape", f"{m.n_rows} rows x {m.n_cols} cols"),
            (
                "Plan SHA-256",
                f"sha256:{m.plan_sha256[:12]}" if m.plan_sha256 else "None",
            ),
            ("Seed", str(m.seed) if m.seed is not None else "None"),
            ("Python", m.python_version),
            ("Packages", pkg_str),
            ("Created At", m.created_at),
        ]
        manifest_df = pd.DataFrame(rows, columns=["Field", "Value"])
        doc.add_table(manifest_df, caption="Run Manifest (Zero-PHI Provenance)")
    if doc.lineage is not None and doc.lineage.nodes:
        doc.add_paragraph("\n".join(doc.lineage.summary_lines()))
    return doc

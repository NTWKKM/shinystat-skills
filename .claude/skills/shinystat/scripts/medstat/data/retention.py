"""
src/medstat/data/retention.py: Audited Participant Retention Flow & Diagrams.

Implements:
1. SampleFlowTracker: Tracks N_initial -> N_excluded -> N_analyzed with stage reasons.
2. Invariant enforcement: n_start - n_excluded = n_remaining at every transition.
3. Multi-reason breakdown and DataFrame filter gateway (apply_filter).
4. CONSORT trial arm branching (branch_arms).
5. Clean ASCII/Unicode flow diagram rendering and Mermaid export.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd


class FlowDesign(str, Enum):
    STROBE = "STROBE"  # Observational studies (cohort, case-control, cross-sectional)
    CONSORT = "CONSORT"  # Randomized controlled trials
    GENERIC = "GENERIC"  # Generic pipeline


@dataclass
class ExclusionReason:
    """Detailed exclusion reason within a stage."""

    reason: str
    count: int
    percentage: float = 0.0  # Percentage of stage starting N
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "count": self.count,
            "percentage": round(self.percentage, 2),
            "metadata": self.metadata,
        }


@dataclass
class FlowStage:
    """A single stage in the sample retention flow."""

    stage_id: int
    stage_name: str
    n_start: int
    n_excluded: int
    n_remaining: int
    retention_rate_stage: float  # (n_remaining / n_start) * 100
    retention_rate_overall: float  # (n_remaining / n_initial) * 100
    reasons: list[ExclusionReason] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "stage_name": self.stage_name,
            "n_start": self.n_start,
            "n_excluded": self.n_excluded,
            "n_remaining": self.n_remaining,
            "retention_rate_stage": round(self.retention_rate_stage, 2),
            "retention_rate_overall": round(self.retention_rate_overall, 2),
            "reasons": [r.to_dict() for r in self.reasons],
            "description": self.description,
        }


class SampleFlowTracker:
    """
    Audited participant retention flow tracker adhering to CONSORT and STROBE standards.

    Tracks sequential reductions from N_initial -> N_excluded -> N_analyzed,
    enforcing conservation invariants and rendering ASCII flow diagrams.
    """

    def __init__(
        self,
        initial_n: int | None = None,
        n_initial: int | None = None,
        initial_name: str = "Initial Enrolled / Assessed",
        design: str | FlowDesign = FlowDesign.STROBE,
    ) -> None:
        init_val = n_initial if n_initial is not None else initial_n
        if init_val is not None and init_val < 0:
            raise ValueError(f"Initial sample size cannot be negative: {init_val}")
        self.design = FlowDesign(design) if isinstance(design, str) else design
        self.initial_name = initial_name
        self.initial_n = init_val
        self.stages: list[FlowStage] = []
        self._current_n = init_val
        self._arms: dict[str, SampleFlowTracker] = {}

    def set_initial(self, n: int, name: str | None = None) -> None:
        """Set or update the initial cohort size."""
        if n < 0:
            raise ValueError(f"Initial sample size cannot be negative: {n}")
        self.initial_n = n
        self._current_n = n
        if name:
            self.initial_name = name

    @property
    def current_n(self) -> int:
        return self._current_n if self._current_n is not None else 0

    @property
    def final_n(self) -> int:
        if not self.stages:
            return self.current_n
        return self.stages[-1].n_remaining

    @property
    def total_excluded(self) -> int:
        if self.initial_n is None:
            return 0
        return self.initial_n - self.final_n

    @property
    def overall_retention_rate(self) -> float:
        if not self.initial_n or self.initial_n == 0:
            return 0.0
        return (self.final_n / self.initial_n) * 100.0

    def record_stage(
        self,
        stage_name: str,
        n_remaining: int,
        n_excluded: int | None = None,
        reason: str | list[str | tuple[str, int]] = "",
        details: dict[str, Any] | None = None,
    ) -> FlowStage:
        """
        Record a stage transition in the retention flow.

        Parameters:
            stage_name: Descriptive name of the stage.
            n_remaining: Number of participants remaining after exclusions.
            n_excluded: Number of participants excluded at this stage (computed if None).
            reason: String reason or list of (reason, count) tuples.
            details: Optional metadata dictionary.
        """
        if self.initial_n is None:
            computed_excluded = n_excluded if n_excluded is not None else 0
            self.set_initial(n_remaining + computed_excluded)

        n_start = self.current_n

        if n_excluded is None:
            n_excluded = n_start - n_remaining

        # Invariant validations
        if n_remaining < 0:
            raise ValueError(f"Remaining sample size cannot be negative: {n_remaining}")
        if n_excluded < 0:
            raise ValueError(f"Excluded sample size cannot be negative: {n_excluded}")
        if n_remaining + n_excluded != n_start:
            raise ValueError(
                f"Conservation invariant violated at stage '{stage_name}': "
                f"n_start ({n_start}) != n_remaining ({n_remaining}) + n_excluded ({n_excluded})"
            )

        # Parse detailed exclusion reasons
        parsed_reasons: list[ExclusionReason] = []
        if isinstance(reason, str):
            if reason or n_excluded > 0:
                pct = (n_excluded / n_start * 100.0) if n_start > 0 else 0.0
                parsed_reasons.append(
                    ExclusionReason(
                        reason=reason or "Excluded", count=n_excluded, percentage=pct
                    )
                )
        elif isinstance(reason, list):
            for item in reason:
                if isinstance(item, tuple):
                    r_text, r_count = item
                    r_pct = (r_count / n_start * 100.0) if n_start > 0 else 0.0
                    parsed_reasons.append(
                        ExclusionReason(reason=r_text, count=r_count, percentage=r_pct)
                    )
                elif isinstance(item, str):
                    parsed_reasons.append(
                        ExclusionReason(reason=item, count=0, percentage=0.0)
                    )

        stage_id = len(self.stages) + 1
        ret_stage = (n_remaining / n_start * 100.0) if n_start > 0 else 0.0
        ret_overall = (
            (n_remaining / self.initial_n * 100.0)
            if self.initial_n and self.initial_n > 0
            else 0.0
        )

        stage = FlowStage(
            stage_id=stage_id,
            stage_name=stage_name,
            n_start=n_start,
            n_excluded=n_excluded,
            n_remaining=n_remaining,
            retention_rate_stage=ret_stage,
            retention_rate_overall=ret_overall,
            reasons=parsed_reasons,
            description=details.get("description", "") if details else "",
        )

        self.stages.append(stage)
        self._current_n = n_remaining
        return stage

    def apply_filter(
        self,
        df: pd.DataFrame,
        condition: pd.Series | np.ndarray | Callable[[pd.DataFrame], pd.Series],
        stage_name: str,
        reason: str,
    ) -> pd.DataFrame:
        """
        Apply a filter condition to a DataFrame and record the retention stage automatically.
        """
        if self.initial_n is None:
            self.set_initial(len(df))

        mask = condition(df) if callable(condition) else condition
        if not isinstance(mask, (pd.Series, np.ndarray)):
            raise TypeError(
                "Filter condition must evaluate to a boolean Series or ndarray"
            )

        df_filtered = df[mask].copy()
        n_remaining = len(df_filtered)
        n_excluded = len(df) - n_remaining

        self.record_stage(
            stage_name=stage_name,
            n_remaining=n_remaining,
            n_excluded=n_excluded,
            reason=reason,
        )
        return df_filtered

    def branch_arms(
        self, arm_allocations: dict[str, int]
    ) -> dict[str, SampleFlowTracker]:
        """
        Branch the current cohort into trial arms (CONSORT trial design).
        """
        for arm_name, n_arm in arm_allocations.items():
            if n_arm < 0:
                raise ValueError(
                    f"Arm allocation for '{arm_name}' cannot be negative: {n_arm}"
                )

        total_allocated = sum(arm_allocations.values())
        if total_allocated != self.current_n:
            raise ValueError(
                f"Sum of arm allocations ({total_allocated}) does not match current cohort N ({self.current_n})"
            )

        for arm_name, n_arm in arm_allocations.items():
            arm_tracker = SampleFlowTracker(
                initial_n=n_arm,
                initial_name=f"Allocated to {arm_name}",
                design=FlowDesign.CONSORT,
            )
            self._arms[arm_name] = arm_tracker

        return self._arms

    def get_flow_summary(self) -> dict[str, Any]:
        """
        Return structured, serializable flow metadata.
        Provides both lowercase and uppercase keys for test and contract compatibility.
        """
        init_val = self.initial_n if self.initial_n is not None else 0
        final_val = self.final_n
        excl_val = self.total_excluded

        summary: dict[str, Any] = {
            "n_initial": init_val,
            "n_excluded": excl_val,
            "n_analyzed": final_val,
            "initial_n": init_val,
            "final_n": final_val,
            "total_excluded": excl_val,
            "N_initial": init_val,
            "N_excluded": excl_val,
            "N_analyzed": final_val,
            "design": self.design.value,
            "initial_name": self.initial_name,
            "retention_rate_pct": round(self.overall_retention_rate, 2),
            "stages": [s.to_dict() for s in self.stages],
        }
        if self._arms:
            summary["arms"] = {
                name: arm.get_flow_summary() for name, arm in self._arms.items()
            }
        return summary

    def to_dict(self) -> dict[str, Any]:
        return self.get_flow_summary()

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def render_ascii_flow(self, style: str = "unicode", box_width: int = 58) -> str:
        """
        Render a clean ASCII or Unicode participant retention flow diagram.
        """
        is_unicode = style.lower() == "unicode"

        tl = "┌" if is_unicode else "+"
        tr = "┐" if is_unicode else "+"
        bl = "└" if is_unicode else "+"
        br = "┘" if is_unicode else "+"
        h = "─" if is_unicode else "-"
        v = "│" if is_unicode else "|"
        arrow = "├─►" if is_unicode else "+-->"
        bullet = "•" if is_unicode else "*"

        lines: list[str] = []

        def make_box(title: str, subtitle: str) -> list[str]:
            inner_w = box_width - 2
            t_str = f" {title}"[:inner_w].ljust(inner_w)
            s_str = f" {subtitle}"[:inner_w].ljust(inner_w)
            return [
                f"{tl}{h * inner_w}{tr}",
                f"{v}{t_str}{v}",
                f"{v}{s_str}{v}",
                f"{bl}{h * inner_w}{br}",
            ]

        # Initial box
        init_n = self.initial_n if self.initial_n is not None else 0
        lines.extend(make_box(f"Stage 0: {self.initial_name}", f"N = {init_n:,}"))

        mid_space = " " * (box_width // 2)

        for s in self.stages:
            # Dropdown connector
            lines.append(f"{mid_space}{v}")
            # Exclusion branch
            excl_pct = (s.n_excluded / s.n_start * 100) if s.n_start > 0 else 0.0
            lines.append(
                f"{mid_space}{arrow} Excluded (N = {s.n_excluded:,}, {excl_pct:.1f}%):"
            )
            if s.reasons:
                for r in s.reasons:
                    count_str = f": {r.count:,}" if r.count > 0 else ""
                    lines.append(f"{mid_space}{v}   {bullet} {r.reason}{count_str}")
            else:
                lines.append(f"{mid_space}{v}   {bullet} Excluded at {s.stage_name}")
            lines.append(f"{mid_space}{v}")

            # Next stage box
            ret_str = (
                f"N = {s.n_remaining:,} ({s.retention_rate_overall:.1f}% of initial)"
            )
            lines.extend(make_box(f"Stage {s.stage_id}: {s.stage_name}", ret_str))

        # Check for branched arms
        if self._arms:
            lines.append(f"{mid_space}{v}")
            lines.append(
                f"{mid_space}{v}--- Allocated into {len(self._arms)} trial arms:"
            )
            for arm_name, arm_tracker in self._arms.items():
                arm_summary = arm_tracker.get_flow_summary()
                lines.append(
                    f"{mid_space}    • Arm [{arm_name}]: N = {arm_summary['initial_n']:,} "
                    f"-> Analyzed: N = {arm_summary['final_n']:,}"
                )

        return "\n".join(lines)

    def to_mermaid(self) -> str:
        """Generate Mermaid.js flowchart markdown string adhering to CONSORT trial layout."""
        lines = ["flowchart TD"]
        init_n = self.initial_n if self.initial_n is not None else 0
        clean_init_name = self.initial_name.replace('"', "'")
        lines.append(f'  S0["{clean_init_name}<br>N = {init_n:,}"]')

        def _format_reasons(reasons: list[ExclusionReason], n_excluded: int) -> str:
            if not reasons:
                return f"Excluded: N = {n_excluded:,}"
            if len(reasons) == 1:
                r_clean = reasons[0].reason.replace('"', "'")
                return (
                    f"Excluded: N = {n_excluded:,}<br>{r_clean}"
                    if r_clean
                    else f"Excluded: N = {n_excluded:,}"
                )
            items = []
            for r in reasons:
                r_clean = r.reason.replace('"', "'")
                if not r_clean:
                    continue
                if r.count > 0:
                    items.append(f"• {r_clean}: n = {r.count:,}")
                else:
                    items.append(f"• {r_clean}")
            if items:
                return f"Excluded: N = {n_excluded:,}<br>" + "<br>".join(items)
            return f"Excluded: N = {n_excluded:,}"

        prev_node = "S0"
        for s in self.stages:
            node_id = f"S{s.stage_id}"
            excl_id = f"E{s.stage_id}"
            excl_label = _format_reasons(s.reasons, s.n_excluded)
            clean_stage_name = s.stage_name.replace('"', "'")
            lines.append(f'  {excl_id}["{excl_label}"]')
            lines.append(
                f'  {node_id}["{clean_stage_name}<br>N = {s.n_remaining:,} ({s.retention_rate_overall:.1f}%)"]'
            )
            lines.append(f"  {prev_node} --> {excl_id}")
            lines.append(f"  {prev_node} --> {node_id}")
            prev_node = node_id

        if self._arms:
            for i, (arm_name, arm_tracker) in enumerate(self._arms.items(), 1):
                clean_arm_name = arm_name.replace('"', "'")
                arm_node_id = f"Arm{i}_0"
                arm_init_n = (
                    arm_tracker.initial_n
                    if arm_tracker.initial_n is not None
                    else arm_tracker.current_n
                )
                lines.append(
                    f'  {arm_node_id}["Allocated to {clean_arm_name}<br>N = {arm_init_n:,}"]'
                )
                lines.append(f"  {prev_node} --> {arm_node_id}")

                arm_prev = arm_node_id
                for s in arm_tracker.stages:
                    arm_stage_id = f"Arm{i}_S{s.stage_id}"
                    arm_excl_id = f"Arm{i}_E{s.stage_id}"
                    arm_excl_label = _format_reasons(s.reasons, s.n_excluded)
                    clean_arm_stage_name = s.stage_name.replace('"', "'")
                    lines.append(f'  {arm_excl_id}["{arm_excl_label}"]')
                    lines.append(
                        f'  {arm_stage_id}["{clean_arm_stage_name}<br>N = {s.n_remaining:,} ({s.retention_rate_overall:.1f}%)"]'
                    )
                    lines.append(f"  {arm_prev} --> {arm_excl_id}")
                    lines.append(f"  {arm_prev} --> {arm_stage_id}")
                    arm_prev = arm_stage_id

        return "\n".join(lines)

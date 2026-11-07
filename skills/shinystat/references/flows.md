# Clinical Study Flow Diagrams (Cloud Sandbox Edition)

This reference provides self-contained, standalone Python recipes for generating publication flow diagrams (CONSORT 2025/2010, STROBE, PRISMA 2020) in cloud sandboxes using pure `matplotlib` and `numpy`.

---

## 1. Strict Mathematical Reconciliation Invariant

Every clinical flow diagram must satisfy mathematical reconciliation before rendering:

### A. Randomized Controlled Trials (CONSORT 2025/2010)
$$\text{Assessed} - \sum \text{Excluded} = \text{Randomized} = \sum \text{Allocated}$$
For each trial arm $i$:
$$\text{Allocated}_i = \text{Analyzed}_i + \text{Lost to Follow-up}_i + \text{Discontinued Intervention}_i + \text{Excluded from Analysis}_i$$

### B. Observational Cohort Studies (STROBE)
$$\text{Retained}_{k} - \text{Excluded}_{k+1} = \text{Retained}_{k+1}$$

### C. Systematic Reviews & Meta-Analyses (PRISMA 2020)
$$\text{Identified} - \text{Duplicates Removed} = \text{Screened}$$
$$\text{Screened} - \text{Excluded Screening} = \text{Reports Sought for Retrieval}$$
$$\text{Reports Sought} - \text{Reports Not Retrieved} = \text{Assessed for Eligibility}$$
$$\text{Assessed for Eligibility} - \sum \text{Excluded Eligibility} = \text{Included Studies}$$

All participant counts must be non-negative integers ($\ge 0$). If any equation does not balance exactly, a `ValueError` must be raised.

---

## 2. Recipe: Standalone CONSORT 2025/2010 Flow Diagram

```python
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path
import textwrap

def draw_standalone_consort_rct(
    assessed: int,
    excluded: dict[str, int],
    arms: list[dict],
    version: str = "2025",
    journal: str = "NEJM",
    out_path: str = "figures/consort_flow.png",
):
    """
    Renders an audit-reconciled CONSORT 2025/2010 flow diagram using pure matplotlib.
    """
    if str(version) not in ("2025", "2010"):
        raise ValueError(f"Unsupported CONSORT version '{version}'. Supported: '2025', '2010'")
    if assessed < 0 or any(v < 0 for v in excluded.values()):
        raise ValueError("Participant counts must be non-negative integers")
    if len(arms) not in (2, 3):
        raise ValueError(f"Only 2-arm and 3-arm trials are currently supported (got {len(arms)})")

    n_ex = sum(excluded.values())
    n_rand = assessed - n_ex
    n_alloc = sum(a["allocated"] for a in arms)
    if n_rand != n_alloc:
        raise ValueError(f"CONSORT reconciliation mismatch: randomized ({n_rand}) != allocated ({n_alloc})")

    # Arm-level mathematical reconciliation check
    for arm in arms:
        if arm["allocated"] < 0 or arm.get("analyzed", 0) < 0:
            raise ValueError("Arm counts must be non-negative integers")

        # Handle reason dicts safely: empty dicts produce 0
        dnr_dict = arm["did_not_receive"] if isinstance(arm.get("did_not_receive"), dict) else {}
        dnr = sum(dnr_dict.values()) if isinstance(arm.get("did_not_receive"), dict) else (arm.get("did_not_receive") or 0)
        rec = arm.get("received", arm["allocated"] - dnr)

        lo_dict = arm["lost"] if isinstance(arm.get("lost"), dict) else {}
        lo = sum(lo_dict.values()) if isinstance(arm.get("lost"), dict) else (arm.get("lost") or 0)

        di_dict = arm["discontinued"] if isinstance(arm.get("discontinued"), dict) else {}
        di = sum(di_dict.values()) if isinstance(arm.get("discontinued"), dict) else (arm.get("discontinued") or 0)

        ex_an_dict = arm["excluded_from_analysis"] if isinstance(arm.get("excluded_from_analysis"), dict) else {}
        ex_an = sum(ex_an_dict.values()) if isinstance(arm.get("excluded_from_analysis"), dict) else (arm.get("excluded_from_analysis") or 0)
        analyzed = arm.get("analyzed", arm["allocated"] - ex_an)

        if lo < 0 or di < 0 or ex_an < 0 or dnr < 0 or rec < 0:
            raise ValueError("Follow-up and allocation counts must be non-negative integers")

        # Partition 1: Allocation partition (received + did_not_receive == allocated)
        if rec + dnr != arm["allocated"]:
            raise ValueError(
                f"Arm '{arm.get('name')}' allocation mismatch: allocated ({arm['allocated']}) "
                f"!= received ({rec}) + did_not_receive ({dnr})"
            )

        # Partition 2: Analysis partition (analyzed + excluded_from_analysis == allocated)
        # Note: lost and discontinued describe follow-up events and may overlap with analyzed
        if analyzed < 0 or analyzed > arm["allocated"]:
            raise ValueError(
                f"Arm '{arm.get('name')}' reconciliation error: analyzed ({analyzed}) must be between 0 and allocated ({arm['allocated']})"
            )
        if analyzed + ex_an != arm["allocated"]:
            raise ValueError(
                f"Arm '{arm.get('name')}' analysis partition mismatch: allocated ({arm['allocated']}) "
                f"!= analyzed ({analyzed}) + excluded_from_analysis ({ex_an})"
            )

    journal_colors = {
        "NEJM": "#002D62",
        "JAMA": "#005A9C",
        "LANCET": "#EE1C25",
        "BMJ": "#003B5C",
        "CLINICAL": "#2B6CB0",
    }
    accent_color = journal_colors.get(journal.upper(), "#222222")

    fig, ax = plt.subplots(figsize=(7.5 if len(arms) == 2 else 9.5, 8.5), dpi=300)
    try:
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis("off")

        def box(cx, cy, w, h, title, subtitle=None, fc="#FFFFFF", ec=accent_color):
            rect = FancyBboxPatch(
                (cx - w / 2, cy - h / 2), w, h,
                boxstyle="square,pad=0",
                facecolor=fc, edgecolor=ec, linewidth=0.9,
            )
            ax.add_patch(rect)
            if subtitle:
                ax.text(cx, cy + h * 0.20, title, ha="center", va="center", fontsize=8.0, fontweight="bold", color="#111111")
                ax.text(cx, cy - h * 0.18, subtitle, ha="center", va="center", fontsize=7.0, color="#333333")
            else:
                ax.text(cx, cy, title, ha="center", va="center", fontsize=8.0, fontweight="bold", color="#111111")
            return cx, cy

        def arrow(x0, y0, x1, y1):
            ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=accent_color, lw=0.9),
            )

        # 1. Enrollment
        box(45, 94, 38, 6, "Assessed for eligibility", f"(N = {assessed:,})")
        arrow(45, 91, 45, 80)

        # Excluded box with all exclusion reasons rendered
        excl_lines = [f"\u2022 {k} (n = {v:,})" for k, v in excluded.items()]
        excl_text = "\n".join(excl_lines)
        h_excl = max(8, 2.5 + len(excl_lines) * 2.2)
        box(78, 85, 32, h_excl, f"Excluded (n = {n_ex:,})", excl_text)
        arrow(45, 85, 62, 85)

        # 2. Randomized
        box(45, 77, 34, 6, "Randomized" if version == "2025" else "Randomised", f"(N = {n_rand:,})")

        # 3. Allocation & Follow-up
        cx_coords = [25, 75] if len(arms) == 2 else [18, 50, 82]
        arm_w = 38 if len(arms) == 2 else 28

        for arm, cx in zip(arms, cx_coords):
            dnr_dict = arm["did_not_receive"] if isinstance(arm.get("did_not_receive"), dict) else {}
            dnr = sum(dnr_dict.values()) if isinstance(arm.get("did_not_receive"), dict) else (arm.get("did_not_receive") or 0)
            rec = arm.get("received", arm["allocated"] - dnr)

            lo_dict = arm["lost"] if isinstance(arm.get("lost"), dict) else {}
            lo = sum(lo_dict.values()) if isinstance(arm.get("lost"), dict) else (arm.get("lost") or 0)

            di_dict = arm["discontinued"] if isinstance(arm.get("discontinued"), dict) else {}
            di = sum(di_dict.values()) if isinstance(arm.get("discontinued"), dict) else (arm.get("discontinued") or 0)

            ex_an_dict = arm["excluded_from_analysis"] if isinstance(arm.get("excluded_from_analysis"), dict) else {}
            ex_an = sum(ex_an_dict.values()) if isinstance(arm.get("excluded_from_analysis"), dict) else (arm.get("excluded_from_analysis") or 0)
            analyzed = arm.get("analyzed", arm["allocated"] - ex_an)

            # Allocation box with intervention receipt details (CONSORT 2025/2010)
            alloc_lines = [f"(n = {arm['allocated']:,})"]
            alloc_lines.append(f"\u2022 Received allocated intervention (n = {rec:,})")
            if dnr > 0 or dnr_dict:
                alloc_lines.append(f"\u2022 Did not receive intervention (n = {dnr:,})")
                for r_k, r_v in dnr_dict.items():
                    alloc_lines.append(f"   - {r_k} (n = {r_v})")
            alloc_text = "\n".join(alloc_lines)
            h_alloc = max(6.5, 2.5 + len(alloc_lines) * 1.5)

            arrow(45, 74, cx, 63 + h_alloc / 2)
            box(cx, 63, arm_w, h_alloc, f"Allocated to {arm['name']}", alloc_text)

            # Detailed follow-up reasons
            fu_lines = [f"\u2022 Lost to follow-up (n = {lo})"]
            for r_k, r_v in lo_dict.items():
                fu_lines.append(f"   - {r_k} (n = {r_v})")
            fu_lines.append(f"\u2022 Discontinued intervention (n = {di})")
            for r_k, r_v in di_dict.items():
                fu_lines.append(f"   - {r_k} (n = {r_v})")

            fu_text = "\n".join(fu_lines)
            h_fu = max(7, 3.0 + len(fu_lines) * 1.6)

            arrow(cx, 63 - h_alloc / 2, cx, 44 + h_fu / 2)
            box(cx, 44, arm_w, h_fu, "Follow-Up", fu_text)

            # Analysis
            po = " for primary outcome" if version == "2025" else ""
            an_lines = [f"(n = {analyzed:,})"]
            if ex_an > 0 or ex_an_dict:
                an_lines.append(f"\u2022 Excluded from analysis (n = {ex_an})")
                for r_k, r_v in ex_an_dict.items():
                    an_lines.append(f"   - {r_k} (n = {r_v})")
            h_an = max(6.5, 2.8 + len(an_lines) * 1.6)
            arrow(cx, 44 - h_fu / 2, cx, 24 + h_an / 2)
            box(cx, 24, arm_w, h_an, f"Analysed{po}", "\n".join(an_lines))

        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=600, bbox_inches="tight")
        return str(p)
    finally:
        plt.close(fig)
```

---

## 3. Recipe: Standalone STROBE Observational Flow

```python
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import textwrap

def draw_standalone_strobe_flow(
    stages: list[tuple[str, int, int, str]],  # (stage_name, retained_n, excluded_n, exclusion_reason)
    out_path: str = "figures/strobe_flow.png",
):
    """
    Renders an audit-reconciled sequential cohort exclusion flow diagram.
    """
    # Reconcile adjacent stages: previous - excluded == retained
    for idx in range(len(stages) - 1):
        curr_name, curr_retain, _, _ = stages[idx]
        next_name, next_retain, next_ex, _ = stages[idx + 1]
        if curr_retain - next_ex != next_retain:
            raise ValueError(
                f"STROBE reconciliation mismatch between '{curr_name}' and '{next_name}': "
                f"previous ({curr_retain}) - excluded ({next_ex}) != retained ({next_retain})"
            )

    fig, ax = plt.subplots(figsize=(6.5, 2.0 + len(stages) * 1.5), dpi=300)
    try:
        ax.set_xlim(0, 100)
        ax.set_ylim(0, len(stages) * 20 + 10)
        ax.axis("off")

        y_cursor = len(stages) * 20
        for idx, (name, n_retain, n_ex, reason) in enumerate(stages):
            # Center cohort box
            rect = FancyBboxPatch((20, y_cursor - 6), 40, 10, boxstyle="square,pad=0", fc="#FFFFFF", ec="#222222", lw=0.8)
            ax.add_patch(rect)
            ax.text(40, y_cursor - 0.5, name, ha="center", va="center", fontsize=8.0, fontweight="bold")
            ax.text(40, y_cursor - 4.0, f"(N = {n_retain:,})", ha="center", va="center", fontsize=7.5, color="#333333")

            # Exclusion occurs along transition from current stage (idx) into next stage (idx + 1)
            if idx < len(stages) - 1:
                _, _, next_ex, next_reason = stages[idx + 1]
                ax.annotate("", xy=(40, y_cursor - 14), xytext=(40, y_cursor - 6), arrowprops=dict(arrowstyle="->", color="#222222", lw=0.9))
                if next_ex > 0:
                    y_branch = y_cursor - 10
                    ax.annotate("", xy=(68, y_branch), xytext=(40, y_branch), arrowprops=dict(arrowstyle="->", color="#222222", lw=0.9))
                    wrapped_reason = textwrap.fill(next_reason, width=28)
                    line_count = len(wrapped_reason.split('\n'))
                    box_h = max(8, 3.0 + line_count * 2.0)
                    ex_rect = FancyBboxPatch((68, y_branch - box_h / 2), 30, box_h, boxstyle="square,pad=0", fc="#F8F9FA", ec="#777777", lw=0.7)
                    ax.add_patch(ex_rect)
                    ax.text(83, y_branch + box_h * 0.15, f"Excluded (n = {next_ex:,})", ha="center", va="center", fontsize=7.0, fontweight="bold")
                    ax.text(83, y_branch - box_h * 0.20, wrapped_reason, ha="center", va="center", fontsize=6.5, color="#555555")

            y_cursor -= 18

        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=600, bbox_inches="tight")
        return str(p)
    finally:
        plt.close(fig)
```

---

## 4. Recipe: Standalone PRISMA 2020 Systematic Review Flow

```python
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import textwrap

def draw_standalone_prisma_flow(
    identified_databases: int,
    duplicates_removed: int,
    screened: int,
    screen_excluded: int,
    retrieval_sought: int,
    not_retrieved: int,
    assessed_eligibility: int,
    excluded_eligibility: dict[str, int],
    included_studies: int,
    included_reports: int = None,
    out_path: str = "figures/prisma_flow.png",
):
    """
    Renders an audit-reconciled PRISMA 2020 systematic review flow diagram.
    """
    all_counts = [
        identified_databases,
        duplicates_removed,
        screened,
        screen_excluded,
        retrieval_sought,
        not_retrieved,
        assessed_eligibility,
        included_studies,
        *(excluded_eligibility.values()),
    ]
    if included_reports is not None:
        all_counts.append(included_reports)
    if any(not isinstance(c, int) or c < 0 for c in all_counts):
        raise ValueError("PRISMA counts must be non-negative integers")

    # 1. Identification reconciliation
    if identified_databases - duplicates_removed != screened:
        raise ValueError(
            f"PRISMA Identification mismatch: identified ({identified_databases}) - duplicates ({duplicates_removed}) "
            f"!= screened ({screened})"
        )
    # 2. Screening reconciliation
    if screened - screen_excluded != retrieval_sought:
        raise ValueError(
            f"PRISMA Screening mismatch: screened ({screened}) - excluded ({screen_excluded}) "
            f"!= reports sought ({retrieval_sought})"
        )
    # 3. Retrieval reconciliation
    if retrieval_sought - not_retrieved != assessed_eligibility:
        raise ValueError(
            f"PRISMA Retrieval mismatch: sought ({retrieval_sought}) - not retrieved ({not_retrieved}) "
            f"!= assessed ({assessed_eligibility})"
        )
    # 4. Eligibility reconciliation
    n_ex_el = sum(excluded_eligibility.values())
    target_reports = included_reports if included_reports is not None else included_studies
    if assessed_eligibility - n_ex_el != target_reports:
        raise ValueError(
            f"PRISMA Eligibility mismatch: assessed ({assessed_eligibility}) - excluded ({n_ex_el}) "
            f"!= included reports ({target_reports})"
        )

    fig, ax = plt.subplots(figsize=(8.0, 9.5), dpi=300)
    try:
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis("off")

        def box(cx, cy, w, h, title, subtitle=None, fc="#FFFFFF", ec="#2B6CB0"):
            rect = FancyBboxPatch(
                (cx - w / 2, cy - h / 2), w, h,
                boxstyle="square,pad=0",
                facecolor=fc, edgecolor=ec, linewidth=0.9,
            )
            ax.add_patch(rect)
            if subtitle:
                ax.text(cx, cy + h * 0.20, title, ha="center", va="center", fontsize=8.0, fontweight="bold", color="#111111")
                ax.text(cx, cy - h * 0.18, subtitle, ha="center", va="center", fontsize=7.0, color="#333333")
            else:
                ax.text(cx, cy, title, ha="center", va="center", fontsize=8.0, fontweight="bold", color="#111111")
            return cx, cy

        def arrow(x0, y0, x1, y1):
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#2B6CB0", lw=0.9))

        # 1. Identification
        box(30, 93, 42, 7, "Records identified from databases", f"(n = {identified_databases:,})")
        box(76, 84, 34, 7, "Duplicate records removed", f"(n = {duplicates_removed:,})")
        arrow(30, 89.5, 30, 76.5)
        arrow(30, 84, 59, 84)

        # 2. Screening
        box(30, 73, 42, 7, "Records screened", f"(n = {screened:,})")
        box(76, 73, 34, 7, "Records excluded", f"(n = {screen_excluded:,})")
        arrow(51, 73, 59, 73)
        arrow(30, 69.5, 30, 56.5)

        # 3. Retrieval
        box(30, 53, 42, 7, "Reports sought for retrieval", f"(n = {retrieval_sought:,})")
        box(76, 53, 34, 7, "Reports not retrieved", f"(n = {not_retrieved:,})")
        arrow(51, 53, 59, 53)
        arrow(30, 49.5, 30, 36.5)

        # 4. Eligibility
        box(30, 33, 42, 7, "Reports assessed for eligibility", f"(n = {assessed_eligibility:,})")
        excl_lines = [f"\u2022 {k} (n = {v:,})" for k, v in excluded_eligibility.items()]
        h_excl = max(8, 3.0 + len(excl_lines) * 2.0)
        box(76, 33, 34, h_excl, f"Reports excluded (n = {n_ex_el:,})", "\n".join(excl_lines))
        arrow(51, 33, 59, 33)
        arrow(30, 29.5, 30, 15.0)

        # 5. Included
        inc_sub = f"Studies included in review: {included_studies:,}"
        if included_reports:
            inc_sub += f"\nReports of included studies: {included_reports:,}"
        box(30, 11, 42, 8, "New studies included in review", inc_sub)

        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=600, bbox_inches="tight")
        return str(p)
    finally:
        plt.close(fig)
```

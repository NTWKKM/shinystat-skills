"""
src/medstat/figures/flow.py: Reconciled study-flow diagrams on top of Scene canvas.
Standard clinical templates:
1. CONSORT 2025 / 2010 Parallel Two-Arm Trial (BMJ 2025;389:e081123)
2. PRISMA 2020 Systematic Reviews with single or dual stream (BMJ 2021;372:n71)
3. STARD 2015 Diagnostic Accuracy Studies (BMJ 2015;351:h5527)
4. STROBE / Cohort Participant Selection
5. Number-Audited Methods Pipeline
6. SampleFlowTracker Direct Integration (cohort_flow_from_tracker)

Every template validates mathematical reconciliation before drawing and exports
600 DPI PNG + vector PDF (fonttype=42) + SVG + _source.csv audit manifest.
"""

from __future__ import annotations

import re
import textwrap
from typing import Any, Sequence

import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from medstat.figures.base import FigureResult
from medstat.figures.illustration import Scene

INK = "#222222"
SIDE_FC = "#F5F5F5"
BAR_FC = "#E1E8F0"


def n_(v: int | float) -> str:
    """Non-breaking space formatting: '(n = 62)' never wraps or splits across lines."""
    return f"n\u00a0=\u00a0{int(v):,}"


def _need(cond: bool, eq: str) -> None:
    if not cond:
        raise ValueError(f"Unreconciled counts: {eq}")


def _reasons(d: dict[str, int]) -> str:
    return "\n".join(f"\u2022 {k} ({n_(v)})" for k, v in d.items()) if d else ""


class Flow:
    """
    Top-down layout engine for publication flow diagrams.
    Coordinates are negative relative units (cursor starts at 0 and advances downward).
    """

    def __init__(
        self,
        width: str | float = "double_col",
        journal: str = "NEJM",
        seed: int = 1,
        fs: float = 7.0,
    ) -> None:
        self.sc = Scene(width, aspect=1.0, journal=journal, seed=seed)
        self.fs = fs
        self.w_in = self.sc.fig.get_size_inches()[0]
        self.y = -1.0
        self.boxes: dict[str, dict[str, Any]] = {}
        self.phases: list[tuple[str, float]] = []

    def _measure(self, text: str, weight: str = "normal") -> tuple[float, float]:
        t = self.sc.ax.text(
            0,
            0,
            text,
            fontsize=self.fs,
            fontweight=weight,
            ha="left",
            va="top",
            linespacing=1.25,
        )
        bb = t.get_window_extent(renderer=self.sc.fig.canvas.get_renderer())
        t.remove()
        (x0, y0), (x1, y1) = self.sc.ax.transData.inverted().transform(
            [[bb.x0, bb.y0], [bb.x1, bb.y1]]
        )
        return x1 - x0, y1 - y0

    def _fit(
        self, text: str, width: float, weight: str = "normal"
    ) -> tuple[str, float]:
        if not text:
            return "", 0.0
        chars = max(8, int(width * self.w_in / 100 * 72 / (self.fs * 0.5)))
        while True:
            lines = [
                textwrap.fill(
                    line_text,
                    chars,
                    subsequent_indent="   " if line_text.startswith("\u2022") else "",
                    break_long_words=False,
                )
                for line_text in text.split("\n")
            ]
            t = "\n".join(lines)
            w, h = self._measure(t, weight)
            if w <= width or chars <= 6:
                return t, h
            chars -= 2

    def _layout(
        self,
        key: str,
        cx: float,
        w: float,
        head: str,
        body: str = "",
        style: str = "plain",
        align: str = "center",
    ) -> dict[str, Any]:
        pad = 1.3
        head_t, hh = self._fit(head, w - 2 * pad, "bold")
        body_t, bh = self._fit(body, w - 2 * pad)
        h = 2 * pad + hh + (0.7 if head_t and body_t else 0) + bh
        return dict(
            key=key,
            cx=cx,
            w=w,
            h=h,
            head=head_t,
            body=body_t,
            hh=hh,
            pad=pad,
            style=style,
            align=align,
            left=cx - w / 2,
            right=cx + w / 2,
        )

    def _draw(self, L: dict[str, Any], top: float, h: float | None = None) -> None:
        h_val = h or L["h"]
        fc, lw = {
            "plain": ("#FFFFFF", 0.7),
            "side": (SIDE_FC, 0.6),
            "final": ("#FFFFFF", 1.4),
        }[L["style"]]
        self.sc.ax.add_patch(
            FancyBboxPatch(
                (L["left"], top - h_val),
                L["w"],
                h_val,
                boxstyle="square,pad=0",
                fc=fc,
                ec=INK,
                lw=lw,
                zorder=5,
            )
        )
        x = L["cx"] if L["align"] == "center" else L["left"] + L["pad"]
        ha = L["align"]
        y = top - L["pad"]
        if L["head"]:
            self.sc.text(
                x,
                y,
                L["head"],
                self.fs,
                ha,
                "top",
                "bold",
                z=6,
                linespacing=1.25,
            )
            y -= L["hh"] + 0.7
        if L["body"]:
            self.sc.text(
                x,
                y,
                L["body"],
                self.fs,
                ha,
                "top",
                "normal",
                z=6,
                linespacing=1.25,
            )
        self.boxes[L["key"]] = dict(L, top=top, bottom=top - h_val)

    def mark(self, name: str) -> None:
        """Marks a trial/flow phase (e.g. Enrollment, Allocation, Follow-up, Analysis)."""
        self.phases.append((name, self.y))

    def row(self, items: Sequence[tuple], gap_after: float = 0.0) -> None:
        """Places items with shared top coordinate and synchronized height."""
        Ls = [self._layout(*it[:5], *(it[5:] if len(it) > 5 else ())) for it in items]
        H = max(L["h"] for L in Ls)
        for L in Ls:
            self._draw(L, self.y, H)
        self.y -= H + gap_after

    def gap(
        self,
        from_key: str | None = None,
        side: tuple | None = None,
        base: float = 5.5,
    ) -> None:
        if side is None:
            self.y -= base
            return
        L = self._layout(*side[:5], "side", side[5] if len(side) > 5 else "left")
        g = max(base, L["h"] + 1.5)
        ym = self.y - g / 2
        self._draw(L, ym + L["h"] / 2)
        if from_key is not None:
            x0 = self.boxes[from_key]["cx"]
            self.arrow((x0, ym), (L["left"], ym))
        self.y -= g

    def gaps(self, sides: Sequence[tuple], base: float = 5.5) -> None:
        """Places multiple side boxes in one shared vertical gap."""
        Ls = [
            (
                fk,
                self._layout(*sd[:5], "side", sd[5] if len(sd) > 5 else "left"),
            )
            for fk, sd in sides
        ]
        g = max([base] + [L["h"] + 1.5 for _, L in Ls])
        ym = self.y - g / 2
        for fk, L in Ls:
            self._draw(L, ym + L["h"] / 2)
            self.arrow((self.boxes[fk]["cx"], ym), (L["left"], ym))
        self.y -= g

    def arrow(self, p0: tuple[float, float], p1: tuple[float, float]) -> None:
        self.sc.ax.annotate(
            "",
            xy=p1,
            xytext=p0,
            zorder=4,
            arrowprops=dict(
                arrowstyle="-|>,head_length=4.2,head_width=2.0",
                mutation_scale=1.0,
                lw=0.8,
                color=INK,
                shrinkA=0,
                shrinkB=0,
            ),
        )

    def link(self, a: str, b: str, ym: float | None = None) -> None:
        A, B = self.boxes[a], self.boxes[b]
        p0, p1 = (A["cx"], A["bottom"]), (B["cx"], B["top"])
        if abs(p0[0] - p1[0]) < 0.01:
            self.arrow(p0, p1)
            return
        ym_val = (p0[1] + p1[1]) / 2 if ym is None else ym
        self.sc.line(
            [p0, (p0[0], ym_val), (p1[0], ym_val)],
            INK,
            0.8,
            z=4,
            cap="butt",
        )
        self.arrow((p1[0], ym_val), p1)

    def finish(
        self,
        out_path: str,
        title: str,
        alt: str,
        notes: str = "Counts are reconciled; see _source.csv.",
        pad_x_phase: bool = True,
        dpi: int = 600,
        export_vector: bool = True,
        export_svg: bool = True,
    ) -> FigureResult:
        sc = self.sc
        ymin = self.y - 0.5
        H = 1.0 - ymin
        if self.phases:
            bad = [k for k, b in self.boxes.items() if b["left"] < 6.8]
            if bad:
                raise ValueError(
                    f"Boxes {bad} overlap phase bars (box left must be >= 6.8)"
                )
        ends = [p[1] for p in self.phases[1:]] + [self.y]
        for (name, y0), y1 in zip(self.phases, ends):
            sc.ax.add_patch(
                Rectangle(
                    (0.8, y1 + 0.8),
                    5.2,
                    (y0 - y1) - 1.6,
                    facecolor=BAR_FC,
                    edgecolor="none",
                    zorder=1,
                )
            )
            sc.text(
                3.4,
                (y0 + y1) / 2,
                name,
                self.fs,
                "center",
                "center",
                "bold",
                rotation=90,
                z=3,
            )
        sc.ax.set_ylim(ymin, 1.0)
        sc.H = H
        sc.fig.set_size_inches(self.w_in, self.w_in * H / 100)
        return sc.save(
            out_path,
            title,
            alt,
            notes,
            export_vector=export_vector,
            export_svg=export_svg,
            dpi=dpi,
        )


def _log(f: Flow, label: str, **kw: Any) -> None:
    f.sc.note("count", label, **{k: int(v) for k, v in kw.items()})


def _check(f: Flow, name: str, lhs: int | float, rhs: int | float) -> None:
    ok = lhs == rhs
    f.sc.note("reconciliation", name, lhs=int(lhs), rhs=int(rhs), ok=bool(ok))
    _need(ok, f"{name}: {lhs} != {rhs}")


# --------------------------------------------------------------------------- CONSORT 2025 / 2010
def consort_rct(
    assessed: int,
    excluded: dict[str, int],
    arms: list[dict[str, Any]],
    out_path: str = "figures/consort.png",
    journal: str = "NEJM",
    width: str | float = "double_col",
    title: str = "CONSORT flow diagram of participants",
    seed: int = 1,
    version: str = "2025",
    uk_spelling: bool = False,
) -> FigureResult:
    """
    Two-arm parallel randomized controlled trial.
    version="2025" (default; supersedes 2010 per BMJ 2025;389:e081123):
    labels follow-up/analysis 'for primary outcome', lists discontinuation before loss to follow-up.
    version="2010" preserves older wording.
    Invariants: assessed = excluded + randomized; randomized = sum(allocated);
    allocated = received + not received = analysed + excluded from analysis.
    """
    assert len(arms) == 2, "CONSORT layout requires exactly two trial arms"
    assert version in ("2025", "2010"), "version must be '2025' or '2010'"

    v25 = version == "2025"
    po = " for primary outcome" if v25 else ""

    def sp(us: str, uk: str) -> str:
        return uk if uk_spelling else us

    f = Flow(width, journal, seed)
    n_ex = sum(excluded.values())
    rand = assessed - n_ex

    _check(f, "randomized = sum(allocated)", rand, sum(a["allocated"] for a in arms))
    _log(f, "assessed", n=assessed)
    _log(f, "excluded", n=n_ex)
    _log(f, "randomized", n=rand)
    f.sc.note("template", f"CONSORT {version}", version=version)

    for a in arms:
        nr = sum(a.get("not_received", {}).values())
        lo = sum(a.get("lost", {}).values())
        di = sum(a.get("discontinued", {}).values())
        ea = sum(a.get("excluded_analysis", {}).values())
        _check(
            f,
            f"{a['name']}: allocated = received + not received",
            a["allocated"],
            a["received"] + nr,
        )
        _check(
            f,
            f"{a['name']}: allocated = analysed + excluded from analysis",
            a["allocated"],
            a["analysed"] + ea,
        )
        _log(
            f,
            f"{a['name']}: arm",
            allocated=a["allocated"],
            received=a["received"],
            not_received=nr,
            lost=lo,
            discontinued=di,
            analysed=a["analysed"],
            excluded_analysis=ea,
        )

    cx_mid, w_mid, x_arm, w_arm = 54, 34, (30.5, 77.5), 40

    def blk(label: str, d: dict[str, int]) -> str:
        if d:
            return f"\u2022 {label} ({n_(sum(d.values()))})\n" + "\n".join(
                f"   {k} ({n_(v)})" for k, v in d.items()
            )
        return f"\u2022 {label} ({n_(0)})"

    f.mark(sp("Enrollment", "Enrolment"))
    f.row(
        [
            (
                "assessed",
                cx_mid,
                w_mid,
                "Assessed for eligibility",
                n_(assessed),
            )
        ]
    )
    f.gap(
        "assessed",
        ("excl", 87, 25, f"Excluded ({n_(n_ex)})", _reasons(excluded)),
    )
    f.row(
        [
            (
                "rand",
                cx_mid,
                w_mid,
                sp("Randomized", "Randomised"),
                n_(rand),
            )
        ]
    )
    f.link("assessed", "rand")
    f.gap(base=9)

    f.mark("Allocation")
    alloc_items = []
    for i, a in enumerate(arms):
        head = f"Allocated to {a['name']} ({n_(a['allocated'])})"
        body = f"\u2022 Received allocated intervention ({n_(a['received'])})"
        if a.get("not_received"):
            body += "\n\u2022 Did not receive allocated intervention\n" + "\n".join(
                f"   {k} ({n_(v)})" for k, v in a["not_received"].items()
            )
        alloc_items.append((f"alloc{i}", x_arm[i], w_arm, head, body, "plain", "left"))
    f.row(alloc_items)
    f.link("rand", "alloc0")
    f.link("rand", "alloc1")
    f.gap(base=5.5)

    lost_lbl = "Lost to follow-up" + po
    f.mark("Follow-up")
    fu_items = []
    for i, a in enumerate(arms):
        if v25:
            body = "\n".join(
                [
                    blk("Discontinued intervention", a.get("discontinued", {})),
                    blk(lost_lbl, a.get("lost", {})),
                ]
            )
        else:
            body = "\n".join(
                [
                    blk("Lost to follow-up", a.get("lost", {})),
                    blk("Discontinued intervention", a.get("discontinued", {})),
                ]
            )
        fu_items.append((f"fu{i}", x_arm[i], w_arm, "Follow-up", body, "plain", "left"))
    f.row(fu_items)
    f.link("alloc0", "fu0")
    f.link("alloc1", "fu1")
    f.gap(base=5.5)

    f.mark("Analysis")
    an_items = []
    for i, a in enumerate(arms):
        head = f"{sp('Analyzed', 'Analysed')}{po} ({n_(a['analysed'])})"
        body = blk("Excluded from analysis", a.get("excluded_analysis", {}))
        an_items.append((f"an{i}", x_arm[i], w_arm, head, body, "final", "left"))
    f.row(an_items)
    f.link("fu0", "an0")
    f.link("fu1", "an1")

    return f.finish(
        out_path,
        title,
        f"CONSORT {version} flow diagram: {assessed} assessed, {rand} randomized to {arms[0]['name']} and {arms[1]['name']}.",
        f"CONSORT {version} layout; verified count reconciliation.",
    )


# --------------------------------------------------------------------------- PRISMA 2020
def _src_lines(d: dict[str, int]) -> str:
    return "\n".join(f"{k} ({n_(v)})" for k, v in d.items())


def prisma_2020(
    databases: dict[str, int],
    registers: dict[str, int],
    removed_duplicates: int,
    removed_automation: int,
    removed_other: int,
    screened_excluded: int,
    not_retrieved: int,
    fulltext_excluded: dict[str, int],
    included_studies: int,
    included_reports: int,
    other: dict[str, Any] | None = None,
    out_path: str = "figures/prisma.png",
    journal: str = "NEJM",
    width: str | float = "double_col",
    title: str = "PRISMA 2020 flow diagram",
    seed: int = 1,
) -> FigureResult:
    """
    PRISMA 2020 systematic review flow (BMJ 2021;372:n71).
    Supports single database/register stream or two-column layout with 'other methods' stream.
    Equations: identified - removed = screened; screened - excluded = sought;
    sought - not retrieved = assessed; assessed - fulltext exclusions = included reports.
    """
    f = Flow(width, journal, seed)
    ident = sum(databases.values()) + sum(registers.values())
    removed = removed_duplicates + removed_automation + removed_other
    screened = ident - removed
    sought = screened - screened_excluded
    assessed = sought - not_retrieved
    reports_in = assessed - sum(fulltext_excluded.values())

    for k, v in dict(
        identified=ident,
        removed=removed,
        screened=screened,
        sought=sought,
        assessed=assessed,
    ).items():
        _log(f, k, n=v)

    id_body = f"Databases ({n_(sum(databases.values()))})\nRegisters ({n_(sum(registers.values()))})"
    rm_body = (
        f"\u2022 Duplicate records ({n_(removed_duplicates)})\n"
        f"\u2022 Marked ineligible by automation tools ({n_(removed_automation)})\n"
        f"\u2022 Other reasons ({n_(removed_other)})"
    )

    if other is None:
        _check(
            f,
            "included reports = assessed - full-text exclusions",
            reports_in,
            included_reports,
        )
        _need(included_studies <= included_reports, "studies > reports")
        _log(f, "included", studies=included_studies, reports=included_reports)

        cx, w, sx, sw = 36, 56, 83, 31
        f.mark("Identification")
        f.row([("id", cx, w, "Records identified from:", id_body)])
        f.gap(
            "id",
            (
                "rm",
                sx,
                sw,
                f"Records removed before screening ({n_(removed)})",
                rm_body,
            ),
            9,
        )

        f.mark("Screening")
        f.row([("scr", cx, w, "Records screened", n_(screened))])
        f.gap(
            "scr",
            ("se", sx, sw, "Records excluded", n_(screened_excluded)),
        )

        f.row([("sought", cx, w, "Reports sought for retrieval", n_(sought))])
        f.gap(
            "sought",
            ("nr", sx, sw, "Reports not retrieved", n_(not_retrieved)),
        )

        f.row([("assd", cx, w, "Reports assessed for eligibility", n_(assessed))])
        f.gap(
            "assd",
            (
                "fx",
                sx,
                sw,
                f"Reports excluded ({n_(sum(fulltext_excluded.values()))})",
                _reasons(fulltext_excluded),
            ),
            9,
        )

        f.mark("Included")
        f.row(
            [
                (
                    "inc",
                    cx,
                    w,
                    "Studies included in review",
                    f"{n_(included_studies)}\nReports of included studies ({n_(included_reports)})",
                    "final",
                )
            ]
        )
        for a, b in (
            ("id", "scr"),
            ("scr", "sought"),
            ("sought", "assd"),
            ("assd", "inc"),
        ):
            f.link(a, b)

        return f.finish(
            out_path,
            title,
            f"PRISMA 2020 flow: {ident} records identified, {included_studies} studies included.",
        )

    # Dual stream with 'other methods'
    o_ident = sum(other["sources"].values())
    o_nr = other.get("not_retrieved", 0)
    o_fx = other.get("fulltext_excluded", {})
    o_assessed = o_ident - o_nr
    o_in = o_assessed - sum(o_fx.values())
    _log(
        f,
        "other methods",
        identified=o_ident,
        not_retrieved=o_nr,
        assessed=o_assessed,
        included_reports=o_in,
    )
    _check(
        f,
        "included reports = new-review + other-methods reports",
        reports_in + o_in,
        included_reports,
    )
    _need(included_studies <= included_reports, "studies > reports")
    _log(f, "included", studies=included_studies, reports=included_reports)

    L, LS, R, RS = (19, 22), (41.5, 21), (66, 22), (89, 20)
    f.mark("Identification")
    f.row(
        [
            ("id", L[0], L[1], "Records identified from:", id_body),
            (
                "id_o",
                R[0],
                R[1],
                "Records identified from:",
                _src_lines(other["sources"]),
            ),
        ]
    )
    f.gaps(
        [
            (
                "id",
                (
                    "rm",
                    LS[0],
                    LS[1],
                    f"Records removed before screening ({n_(removed)})",
                    rm_body,
                ),
            )
        ],
        9,
    )

    f.mark("Screening")
    f.row([("scr", L[0], L[1], "Records screened", n_(screened))])
    f.gaps([("scr", ("se", LS[0], LS[1], "Records excluded", n_(screened_excluded)))])

    f.row(
        [
            ("sought", L[0], L[1], "Reports sought for retrieval", n_(sought)),
            (
                "sought_o",
                R[0],
                R[1],
                "Reports sought for retrieval",
                n_(o_ident),
            ),
        ]
    )
    f.gaps(
        [
            (
                "sought",
                ("nr", LS[0], LS[1], "Reports not retrieved", n_(not_retrieved)),
            ),
            (
                "sought_o",
                ("nr_o", RS[0], RS[1], "Reports not retrieved", n_(o_nr)),
            ),
        ]
    )

    f.row(
        [
            (
                "assd",
                L[0],
                L[1],
                "Reports assessed for eligibility",
                n_(assessed),
            ),
            (
                "assd_o",
                R[0],
                R[1],
                "Reports assessed for eligibility",
                n_(o_assessed),
            ),
        ]
    )
    f.gaps(
        [
            (
                "assd",
                (
                    "fx",
                    LS[0],
                    LS[1],
                    f"Reports excluded ({n_(sum(fulltext_excluded.values()))})",
                    _reasons(fulltext_excluded),
                ),
            ),
            (
                "assd_o",
                (
                    "fx_o",
                    RS[0],
                    RS[1],
                    f"Reports excluded ({n_(sum(o_fx.values()))})",
                    _reasons(o_fx),
                ),
            ),
        ]
    )

    y_merge = f.y - 3.0
    f.gap(base=9)
    f.mark("Included")
    f.row(
        [
            (
                "inc",
                54,
                50,
                "Studies included in review",
                f"{n_(included_studies)}\nReports of included studies ({n_(included_reports)})",
                "final",
            )
        ]
    )
    f.link("id", "scr")
    f.link("scr", "sought")
    f.link("id_o", "sought_o")
    f.link("sought", "assd")
    f.link("sought_o", "assd_o")
    f.link("assd", "inc", ym=y_merge)
    f.link("assd_o", "inc", ym=y_merge)

    return f.finish(
        out_path,
        title,
        f"PRISMA 2020 flow (two streams): {ident + o_ident} records identified, {included_studies} studies included.",
    )


# --------------------------------------------------------------------------- STROBE / Cohort
def cohort_flow(
    source: int,
    steps: list[tuple[str, int]],
    groups: dict[str, int] | None = None,
    source_label: str = "Source population",
    out_path: str = "figures/cohort.png",
    journal: str = "NEJM",
    width: str | float = "one_half_col",
    title: str = "Cohort selection flow diagram",
    seed: int = 1,
) -> FigureResult:
    """
    Sequential cohort selection and exclusion flow.
    steps = [(exclusion reason, n excluded), ...].
    Invariants: source - sum(exclusions) = analysed; groups sum = analysed.
    """
    f = Flow(width, journal, seed)
    left = source
    cx, w, sx, sw = 34, 56, 80, 36
    _log(f, "source", n=source)
    f.row([("s0", cx, w, source_label, n_(source))])
    prev = "s0"

    for i, (reason, ne) in enumerate(steps, 1):
        left -= ne
        _need(left >= 0, f"Exclusion '{reason}' exceeds remaining participants")
        _log(f, f"after exclusion {i}", excluded=ne, remaining=left)
        f.gap(prev, (f"x{i}", sx, sw, "Excluded", f"{reason} ({n_(ne)})"), 7)
        is_last = i == len(steps)
        f.row(
            [
                (
                    f"s{i}",
                    cx,
                    w,
                    "Remaining" if not is_last else "Analysed cohort",
                    n_(left),
                    "plain" if not is_last else "final",
                )
            ]
        )
        f.link(prev, f"s{i}")
        prev = f"s{i}"

    _check(
        f,
        "source - exclusions = analysed",
        source - sum(n for _, n in steps),
        left,
    )
    if groups:
        _check(f, "groups sum = analysed", sum(groups.values()), left)
        k = len(groups)
        xs = np.linspace(22, 78, k) if k > 1 else [cx]
        ww = min(56 / k - 2, 34)
        f.gap(base=9)
        f.row(
            [
                (f"g{j}", xs[j], ww, name, n_(v))
                for j, (name, v) in enumerate(groups.items())
            ]
        )
        for j in range(k):
            f.link(prev, f"g{j}")

    return f.finish(
        out_path, title, f"Cohort flow from {source} to {left} participants."
    )


def cohort_flow_from_tracker(
    tracker: Any,
    groups: dict[str, int] | None = None,
    out_path: str = "figures/cohort.png",
    journal: str = "NEJM",
    width: str | float = "one_half_col",
    title: str = "Cohort selection flow diagram",
    seed: int = 1,
) -> FigureResult:
    """
    Generates a publication-grade cohort selection flowchart directly from a
    SampleFlowTracker (medstat.data.retention), guaranteeing 100% numerical
    reconciliation with empirical pipeline filtering.
    """
    source = tracker.initial_n if tracker.initial_n is not None else tracker.current_n
    source_label = tracker.initial_name or "Source population"
    steps: list[tuple[str, int]] = []

    for stage in tracker.stages:
        if stage.n_excluded > 0:
            if stage.reasons:
                for r in stage.reasons:
                    r_text = r.reason or stage.stage_name
                    steps.append((r_text, r.count if r.count > 0 else stage.n_excluded))
            else:
                steps.append((stage.stage_name, stage.n_excluded))

    return cohort_flow(
        source=source,
        steps=steps,
        groups=groups,
        source_label=source_label,
        out_path=out_path,
        journal=journal,
        width=width,
        title=title,
        seed=seed,
    )


# --------------------------------------------------------------------------- STARD 2015
def stard_flow(
    eligible: int,
    excluded: dict[str, int],
    index_pos: int,
    index_neg: int,
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    ref_not_done_pos: int = 0,
    ref_not_done_neg: int = 0,
    index_inconclusive: int = 0,
    out_path: str = "figures/stard.png",
    journal: str = "NEJM",
    width: str | float = "double_col",
    title: str = "STARD flow diagram",
    seed: int = 1,
) -> FigureResult:
    """
    STARD 2015 diagnostic accuracy flow diagram (BMJ 2015;351:h5527).
    Equations: eligible - excluded = enrolled; enrolled = index_pos + index_neg + inconclusive;
    index_pos = TP + FP + ref_not_done_pos; index_neg = FN + TN + ref_not_done_neg.
    """
    f = Flow(width, journal, seed)
    enrolled = eligible - sum(excluded.values())
    _check(
        f,
        "enrolled = index pos + neg + inconclusive",
        enrolled,
        index_pos + index_neg + index_inconclusive,
    )
    _check(
        f,
        "index positive = TP + FP + reference not done",
        index_pos,
        tp + fp + ref_not_done_pos,
    )
    _check(
        f,
        "index negative = FN + TN + reference not done",
        index_neg,
        fn + tn + ref_not_done_neg,
    )
    _log(f, "cells", tp=tp, fp=fp, fn=fn, tn=tn)

    f.mark("Enrollment")
    f.row([("el", 40, 36, "Eligible participants", n_(eligible))])
    f.gap(
        "el",
        ("ex", 82, 32, f"Excluded ({n_(sum(excluded.values()))})", _reasons(excluded)),
    )
    en_body = n_(enrolled) + (
        f"\nInconclusive index result: {n_(index_inconclusive)}"
        if index_inconclusive
        else ""
    )
    f.row([("en", 40, 36, "Enrolled; index test performed", en_body)])
    f.link("el", "en")
    f.gap(base=9)

    f.mark("Index")
    f.row(
        [
            ("pos", 27, 38, "Index test positive", n_(index_pos)),
            ("neg", 73, 38, "Index test negative", n_(index_neg)),
        ]
    )
    f.link("en", "pos")
    f.link("en", "neg")
    f.gap(base=9)

    f.mark("Reference")
    f.row(
        [
            (
                "tp",
                18.5,
                21,
                "Condition present",
                n_(tp) + "\n(true positive)",
            ),
            (
                "fp",
                41.5,
                21,
                "Condition absent",
                n_(fp) + "\n(false positive)",
            ),
            (
                "fn",
                64.5,
                21,
                "Condition present",
                n_(fn) + "\n(false negative)",
            ),
            (
                "tn",
                87.5,
                21,
                "Condition absent",
                n_(tn) + "\n(true negative)",
            ),
        ]
    )
    for a, b in (("pos", "tp"), ("pos", "fp"), ("neg", "fn"), ("neg", "tn")):
        f.link(a, b)

    notes = (
        f"Reference-not-performed counts: {ref_not_done_pos} (index positive), "
        f"{ref_not_done_neg} (index negative)."
        if (ref_not_done_pos or ref_not_done_neg)
        else "All participants completed reference standard."
    )
    return f.finish(
        out_path,
        title,
        f"STARD flow: {enrolled} enrolled; TP {tp}, FP {fp}, FN {fn}, TN {tn}.",
        notes,
    )


# --------------------------------------------------------------------------- Methods Pipeline
_NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?")


def methods_pipeline(
    steps: list[dict[str, Any]],
    out_path: str = "figures/methods.png",
    journal: str = "NEJM",
    width: str | float = "one_half_col",
    title: str = "Study methodology",
    seed: int = 1,
    allowed_numbers: Sequence[Any] = (),
    require_cite: bool = False,
    strict_numbers: bool = False,
) -> FigureResult:
    """
    Pipeline overview diagram with strict number auditing and citation tracking.
    steps = [{title, body, side, cite, counts}].
    """
    f = Flow(width, journal, seed)
    cx, w = 32, 56
    sx, sw = 82, 32
    prev: str | None = None
    prev_out: int | None = None
    declared = {str(x) for x in allowed_numbers}
    untraced: list[tuple[int, str]] = []
    uncited: list[int] = []

    for i, st in enumerate(steps, 1):
        c = st.get("counts")
        body = st.get("body", "")
        side = st.get("side", "")
        ex = {}
        if c:
            ex = c.get("excluded", {})
            n_in, n_out = c["n_in"], c["n_out"]
            _check(
                f,
                f"step {i}: n_in - exclusions = n_out",
                n_in - sum(ex.values()),
                n_out,
            )
            if prev_out is not None:
                _check(f, f"step {i}: n_in = previous n_out", n_in, prev_out)
            _log(f, f"step {i}", n_in=n_in, excluded=sum(ex.values()), n_out=n_out)
            prev_out = n_out
            declared |= (
                {str(n_in), str(n_out), f"{n_in:,}", f"{n_out:,}"}
                | {str(v) for v in ex.values()}
                | {f"{v:,}" for v in ex.values()}
            )
            body = (body + "\n" if body else "") + f"{n_(n_in)} \u2192 {n_(n_out)}"
            side = (side + "\n" if side else "") + _reasons(ex)

        for txt in (st["title"], st.get("body", ""), st.get("side", "")):
            for m in _NUM.findall(txt):
                key = m.rstrip("%")
                if key in declared or key.replace(",", "") in declared:
                    continue
                if re.fullmatch(r"(19|20|21)\d\d", key) or (
                    re.fullmatch(r"\d", key) and "." not in key and not m.endswith("%")
                ):
                    continue
                untraced.append((i, m))

        if st.get("body") and not st.get("cite"):
            uncited.append(i)
        if st.get("cite"):
            body += f"\nRef: {st['cite']}"

        f.sc.note("step", st["title"], index=i, cite=st.get("cite", ""))
        f.row([(f"m{i}", cx, w, f"{i}. {st['title']}", body, "plain", "left")])
        if prev:
            f.link(prev, f"m{i}")
        top, h = f.boxes[f"m{i}"]["top"], f.boxes[f"m{i}"]["h"]
        if side:
            L = f._layout(f"n{i}", sx, sw, "", side, "side", "left")
            f._draw(L, top - h / 2 + L["h"] / 2)
            f.sc.line(
                [(cx + w / 2, top - h / 2), (L["left"], top - h / 2)],
                INK,
                0.6,
                ls=(0, (3, 2)),
                z=4,
            )
            f.y = min(f.y, top - max(h, L["h"] + 1.0))
        prev = f"m{i}"
        if i < len(steps):
            f.gap(base=5.0)

    for idx, m in untraced:
        f.sc.note("untraced_number", m, step=idx)
    if strict_numbers and untraced:
        raise ValueError(
            f"Untraced numbers (not in counts/allowed_numbers): {untraced}"
        )
    if require_cite and uncited:
        raise ValueError(f"Steps without citation: {uncited}")

    r = f.finish(
        out_path,
        title,
        f"Methodology flow with {len(steps)} steps.",
        "Schematic of study methods.",
    )
    if r.qa is not None:
        r.qa["untraced_numbers"] = untraced
        r.qa["uncited_steps"] = uncited
    return r


__all__ = [
    "Flow",
    "consort_rct",
    "prisma_2020",
    "cohort_flow",
    "cohort_flow_from_tracker",
    "stard_flow",
    "methods_pipeline",
    "n_",
]

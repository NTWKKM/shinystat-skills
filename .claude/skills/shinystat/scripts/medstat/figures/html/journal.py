"""
src/medstat/figures/html/journal.py: Pure-Python Plotly spec styling synchronization
with journal house styles (NEJM, JAMA, LANCET, BMJ, CLINICAL).
"""

from __future__ import annotations

from typing import Any

from medstat.figures.journal import JOURNAL_SPECS


def sync_plotly_to_journal(
    plotly_spec: dict[str, Any],
    journal: str = "NEJM",
    canvas: str = "manuscript",
) -> dict[str, Any]:
    """
    Applies journal colorway, background colors, and typography to a Plotly dictionary specification.
    """
    j_key = journal.upper() if journal and journal.upper() in JOURNAL_SPECS else "NEJM"
    spec = JOURNAL_SPECS[j_key]
    palette = spec["palette"]
    bg = spec["bg_editorial"] if canvas == "editorial" else spec["bg_manuscript"]

    out = {
        "data": list(plotly_spec.get("data", [])),
        "layout": dict(plotly_spec.get("layout", {})),
    }
    out["layout"]["colorway"] = palette
    out["layout"]["paper_bgcolor"] = bg
    out["layout"]["plot_bgcolor"] = bg
    out["layout"]["font"] = {
        "family": "Arial, Helvetica, Sarabun, sans-serif",
        "size": 12,
        "color": "#222222",
    }
    for idx, trace in enumerate(out["data"]):
        if (
            isinstance(trace, dict)
            and "line" in trace
            and isinstance(trace["line"], dict)
        ):
            if "color" not in trace["line"]:
                trace_copy = dict(trace)
                line_copy = dict(trace["line"])
                line_copy["color"] = palette[idx % len(palette)]
                trace_copy["line"] = line_copy
                out["data"][idx] = trace_copy
    return out


__all__ = ["sync_plotly_to_journal"]

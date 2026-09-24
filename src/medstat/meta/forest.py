"""
Forest Plot Generation for Meta-Analysis.

Constructs publication-ready forest plot datasets and interactive Plotly figures.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go

from medstat.theme.palette import get_color_palette

COLORS = get_color_palette()


def generate_forest_data(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """
    Extract structured forest plot coordinates from meta-analysis results.
    Supports:
    - generate_forest_data(meta_results)
    - generate_forest_data(df, effect_col, se_col, study_col, meta_results)
    """
    if len(args) == 1 and isinstance(args[0], dict):
        meta_results = args[0]
    elif len(args) >= 5 and isinstance(args[4], dict):
        meta_results = args[4]
    elif "meta_results" in kwargs:
        meta_results = kwargs["meta_results"]
    elif len(args) > 0 and isinstance(args[-1], dict):
        meta_results = args[-1]
    else:
        raise ValueError("Valid meta_results dictionary required.")

    studies_df = meta_results["studies"]
    is_ratio = meta_results["is_ratio"]

    study_records = []
    for idx, row in studies_df.iterrows():
        effect = float(
            row.get("effect_size", row.get("log_effect", row.get("effect", 0.0)))
        )
        ci_lower = float(row.get("ci_lower", effect - 1.96 * float(row.get("se", 0.1))))
        ci_upper = float(row.get("ci_upper", effect + 1.96 * float(row.get("se", 0.1))))
        weight_pct = float(row.get("weight_re_pct", row.get("weight_fe_pct", 0.0)))

        study_records.append(
            {
                "study": str(row.get("study", f"Study {idx}")),
                "effect": effect,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "weight_pct": weight_pct,
            }
        )

    re = meta_results["random_effects"]
    summary_diamond = {
        "label": "Overall (Random Effects)",
        "effect": float(re["effect_disp"]),
        "ci_lower": float(re["ci_lower"]),
        "ci_upper": float(re["ci_upper"]),
    }

    return {
        "studies": study_records,
        "summary": summary_diamond,
        "is_ratio": is_ratio,
        "heterogeneity": meta_results["heterogeneity"],
    }


def create_forest_plot(meta_results: dict[str, Any]) -> go.Figure:
    """
    Generate an interactive Plotly Forest Plot.
    """
    data = generate_forest_data(meta_results)
    studies = data["studies"]
    summary = data["summary"]
    is_ratio = data["is_ratio"]
    het = data["heterogeneity"]

    fig = go.Figure()

    y_labels = [s["study"] for s in studies] + [summary["label"]]
    y_pos = list(range(len(studies) - 1, -1, -1)) + [-1]

    # Study points & CIs
    for i, s in enumerate(studies):
        y = y_pos[i]
        fig.add_trace(
            go.Scatter(
                x=[s["ci_lower"], s["ci_upper"]],
                y=[y, y],
                mode="lines",
                line=dict(color=COLORS["primary"], width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        # Size proportional to weight
        marker_size = max(6, int(np.sqrt(s["weight_pct"]) * 3.5))
        fig.add_trace(
            go.Scatter(
                x=[s["effect"]],
                y=[y],
                mode="markers",
                marker=dict(color=COLORS["primary"], size=marker_size, symbol="square"),
                name=s["study"],
                hovertemplate=f"Study: {s['study']}<br>Effect: {s['effect']:.2f} [{s['ci_lower']:.2f}, {s['ci_upper']:.2f}]<br>Weight: {s['weight_pct']:.1f}%<extra></extra>",
                showlegend=False,
            )
        )

    # Summary diamond at bottom (y = -1)
    d_x = [
        summary["ci_lower"],
        summary["effect"],
        summary["ci_upper"],
        summary["effect"],
        summary["ci_lower"],
    ]
    d_y = [-1, -0.7, -1, -1.3, -1]
    fig.add_trace(
        go.Scatter(
            x=d_x,
            y=d_y,
            fill="toself",
            fillcolor=COLORS["danger"],
            line=dict(color=COLORS["danger"]),
            name=summary["label"],
            hovertemplate=f"Overall Effect: {summary['effect']:.2f} [{summary['ci_lower']:.2f}, {summary['ci_upper']:.2f}]<extra></extra>",
            showlegend=False,
        )
    )

    # Null reference line
    null_val = 1.0 if is_ratio else 0.0
    fig.add_vline(x=null_val, line_dash="dash", line_color="gray")

    # Title with heterogeneity
    het_str = f"Heterogeneity: I² = {het['I2']:.1f}%, τ² = {het['tau2']:.3f}, p = {het['p_value']:.3f}"

    measure = meta_results.get("effect_measure")
    if is_ratio:
        if measure and ("OR" in str(measure).upper() or "ODDS" in str(measure).upper()):
            x_title = "Odds Ratio (log scale)"
        elif measure and (
            "RR" in str(measure).upper() or "RISK" in str(measure).upper()
        ):
            x_title = "Risk Ratio (log scale)"
        elif measure and (
            "HR" in str(measure).upper() or "HAZARD" in str(measure).upper()
        ):
            x_title = "Hazard Ratio (log scale)"
        elif measure:
            x_title = f"{measure} (log scale)"
        else:
            x_title = "Risk Ratio (log scale)"
    else:
        x_title = "Effect Size"

    fig.update_layout(
        title=f"Meta-Analysis Forest Plot<br><sup>{het_str}</sup>",
        xaxis_title=x_title,
        yaxis=dict(
            tickmode="array",
            tickvals=y_pos,
            ticktext=y_labels,
        ),
        xaxis_type="log" if is_ratio else "linear",
        template="plotly_white",
        height=max(450, len(y_pos) * 35),
    )

    return fig

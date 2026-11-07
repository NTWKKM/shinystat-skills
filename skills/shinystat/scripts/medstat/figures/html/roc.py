"""
src/medstat/figures/html/roc.py: Dual-view (Plotly + 300 DPI Matplotlib) ROC curve generator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from medstat.figures.html.base import DualFigureResult
from medstat.figures.roc import (
    plot_reclassification,
    plot_roc_curve,
    plot_time_dependent_roc,
)


def plot_dual_roc_curve(
    roc_data: dict[str, Any],
    paired_roc_data: dict[str, Any] | None = None,
    paired_p_value: float | None = None,
    title: str = "Receiver Operating Characteristic (ROC) Curve",
    out_path: str | Path = "figures/roc_curve.png",
    target: Literal["docx", "pptx", "web"] = "docx",
) -> DualFigureResult:
    """
    Renders both publication-grade 300 DPI Matplotlib figure and interactive Plotly ROC curve.
    """
    res = plot_roc_curve(
        roc_data=roc_data,
        paired_roc_data=paired_roc_data,
        paired_p_value=paired_p_value,
        title=title,
        out_path=out_path,
        target=target,
    )

    fpr = np.asarray(roc_data["fpr"], dtype=float)
    tpr = np.asarray(roc_data["tpr"], dtype=float)
    auc = float(roc_data.get("auc", 0.5))
    ci_l = roc_data.get("ci_lower")
    ci_u = roc_data.get("ci_upper")
    model_name = str(roc_data.get("name", "Model 1"))

    if ci_l is not None and ci_u is not None:
        label_1 = f"{model_name}: AUC = {auc:.3f} (95% CI, {ci_l:.3f}–{ci_u:.3f})"
    else:
        label_1 = f"{model_name}: AUC = {auc:.3f}"

    fig = go.Figure()

    # 45-degree chance diagonal
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="#94A3B8", width=1.5),
            name="Chance Diagonal (AUC = 0.500)",
            hoverinfo="skip",
        )
    )

    # Primary model ROC curve
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            line=dict(color="#0284C7", width=2.5),
            name=label_1,
            hovertemplate=(
                "<b>" + model_name + "</b><br>"
                "1 - Specificity (FPR): %{x:.3f}<br>"
                "Sensitivity (TPR): %{y:.3f}<extra></extra>"
            ),
        )
    )

    # Optimal cutoff marker
    opt_pt = roc_data.get("optimal_point")
    if opt_pt:
        opt_fpr = float(opt_pt["fpr"])
        opt_tpr = float(opt_pt["tpr"])
        opt_cut = float(opt_pt["cutoff"])
        opt_sens = float(roc_data.get("sensitivity", opt_tpr))
        opt_spec = float(roc_data.get("specificity", 1.0 - opt_fpr))

        fig.add_trace(
            go.Scatter(
                x=[opt_fpr],
                y=[opt_tpr],
                mode="markers",
                marker=dict(size=10, color="#DC2626", symbol="circle"),
                name=f"Optimal Cutpoint ({opt_cut:.2f})",
                hovertemplate=(
                    f"<b>Optimal Cutpoint: {opt_cut:.3f}</b><br>"
                    f"Sensitivity: {opt_sens:.1%}<br>"
                    f"Specificity: {opt_spec:.1%}<br>"
                    f"FPR: {opt_fpr:.3f}<extra></extra>"
                ),
            )
        )

    # Paired model curve if available
    if paired_roc_data is not None:
        fpr_2 = np.asarray(paired_roc_data["fpr"], dtype=float)
        tpr_2 = np.asarray(paired_roc_data["tpr"], dtype=float)
        auc_2 = float(paired_roc_data.get("auc", 0.5))
        ci_l2 = paired_roc_data.get("ci_lower")
        ci_u2 = paired_roc_data.get("ci_upper")
        model_name_2 = str(paired_roc_data.get("name", "Model 2"))

        if ci_l2 is not None and ci_u2 is not None:
            label_2 = (
                f"{model_name_2}: AUC = {auc_2:.3f} (95% CI, {ci_l2:.3f}–{ci_u2:.3f})"
            )
        else:
            label_2 = f"{model_name_2}: AUC = {auc_2:.3f}"

        if paired_p_value is not None:
            label_2 += f" (p = {paired_p_value:.3f})"

        fig.add_trace(
            go.Scatter(
                x=fpr_2,
                y=tpr_2,
                mode="lines",
                line=dict(color="#D97706", width=2.2, dash="dot"),
                name=label_2,
                hovertemplate=(
                    "<b>" + model_name_2 + "</b><br>"
                    "1 - Specificity (FPR): %{x:.3f}<br>"
                    "Sensitivity (TPR): %{y:.3f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#0F172A")),
        xaxis=dict(
            title="False Positive Rate (1 - Specificity)",
            range=[0, 1.02],
            showgrid=True,
            gridcolor="#F1F5F9",
        ),
        yaxis=dict(
            title="True Positive Rate (Sensitivity)",
            range=[0, 1.02],
            showgrid=True,
            gridcolor="#F1F5F9",
        ),
        template="plotly_white",
        legend=dict(
            yanchor="bottom",
            y=0.02,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="#E2E8F0",
            borderwidth=1,
        ),
        margin=dict(l=50, r=30, t=50, b=50),
        height=500,
    )

    return DualFigureResult(
        png_path=res.png_path,
        alt_text=res.alt_text,
        caption=res.caption,
        source_df=res.source_df,
        csv_path=res.csv_path,
        plotly_fig=fig,
    )


def plot_dual_time_dependent_roc(
    td_roc_data: dict[str, Any],
    mode: Literal["curves", "auroc_trajectory"] = "curves",
    title: str | None = None,
    out_path: str | Path = "figures/td_roc.png",
    target: Literal["docx", "pptx", "web"] = "docx",
) -> DualFigureResult:
    """
    Renders both publication-grade 300 DPI Matplotlib figure and interactive Plotly time-dependent ROC.
    """
    res = plot_time_dependent_roc(
        td_roc_data=td_roc_data,
        mode=mode,
        title=title,
        out_path=out_path,
        target=target,
    )

    fig = go.Figure()
    horizons = td_roc_data.get("time_horizons", [])
    results_by_horizon = td_roc_data.get("results_by_horizon", {})

    palette = ["#0284C7", "#0F172A", "#D97706", "#0D9488", "#8B5CF6", "#DC2626"]

    if mode == "curves":
        # Chance diagonal
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="#94A3B8", width=1.5),
                name="Chance Diagonal (AUC = 0.500)",
                hoverinfo="skip",
            )
        )

        for idx, t in enumerate(horizons):
            res_h = results_by_horizon.get(t, results_by_horizon.get(str(t), {}))
            if not res_h:
                continue
            auc_val = float(res_h.get("auroc", 0.5))
            if not np.isfinite(auc_val) or np.isnan(auc_val):
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        marker=dict(size=0, opacity=0),
                        showlegend=True,
                        name=f"t = {t:g}: unavailable (insufficient events/at-risk)",
                        hoverinfo="none",
                    )
                )
                continue

            fpr = res_h.get("fpr", [0, 1])
            tpr = res_h.get("tpr", [0, 1])
            ci_l = res_h.get("ci_lower")
            ci_u = res_h.get("ci_upper")
            color = palette[idx % len(palette)]

            if (
                ci_l is not None
                and not np.isnan(ci_l)
                and ci_u is not None
                and not np.isnan(ci_u)
            ):
                label = (
                    f"t = {t:g}: AUC = {auc_val:.3f} (95% CI, {ci_l:.3f}–{ci_u:.3f})"
                )
            else:
                label = f"t = {t:g}: AUC = {auc_val:.3f}"

            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    line=dict(color=color, width=2.2),
                    name=label,
                    hovertemplate=(
                        f"<b>Horizon t={t:g}</b><br>"
                        "1 - Specificity (FPR): %{x:.3f}<br>"
                        "Sensitivity (TPR): %{y:.3f}<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            title=dict(
                text=title or "Time-Dependent Receiver Operating Characteristic Curves",
                font=dict(size=15, color="#0F172A"),
            ),
            xaxis=dict(
                title="False Positive Rate (1 - Specificity)",
                range=[0, 1.02],
                showgrid=True,
                gridcolor="#F1F5F9",
            ),
            yaxis=dict(
                title="True Positive Rate (Sensitivity)",
                range=[0, 1.02],
                showgrid=True,
                gridcolor="#F1F5F9",
            ),
            template="plotly_white",
            legend=dict(
                yanchor="bottom",
                y=0.02,
                xanchor="right",
                x=0.98,
                bgcolor="rgba(255, 255, 255, 0.85)",
                bordercolor="#E2E8F0",
                borderwidth=1,
            ),
            margin=dict(l=50, r=30, t=50, b=50),
            height=500,
        )

    else:  # auroc_trajectory
        comp_dict = td_roc_data.get("comparison")
        all_plotted_lowers: list[float] = []

        if comp_dict:
            h_vals = sorted(list(comp_dict.keys()), key=lambda x: float(x))
            x_vals = [float(x) for x in h_vals]
            old_aucs = [float(comp_dict[x]["auroc_old"]) for x in h_vals]
            new_aucs = [float(comp_dict[x]["auroc_new"]) for x in h_vals]

            all_plotted_lowers.extend(old_aucs)
            all_plotted_lowers.extend(new_aucs)

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=old_aucs,
                    mode="lines+markers",
                    line=dict(color="#64748B", dash="dash", width=2.0),
                    marker=dict(size=7),
                    name="Baseline Model",
                    hovertemplate="Baseline AUC(t): %{y:.3f} at t=%{x}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=new_aucs,
                    mode="lines+markers",
                    line=dict(color="#0284C7", width=2.5),
                    marker=dict(size=8, symbol="square"),
                    name="Updated Model",
                    hovertemplate="Updated AUC(t): %{y:.3f} at t=%{x}<extra></extra>",
                )
            )
        else:
            traj = td_roc_data.get("trajectory", {})
            x_vals = [float(x) for x in traj.get("horizons", horizons)]
            y_vals = [float(y) for y in traj.get("auroc", [])]
            ci_l = [
                float(val_l) if val_l is not None and not np.isnan(val_l) else y_vals[i]
                for i, val_l in enumerate(traj.get("ci_lower", []))
            ]
            ci_u = [
                float(u) if u is not None and not np.isnan(u) else y_vals[i]
                for i, u in enumerate(traj.get("ci_upper", []))
            ]

            all_plotted_lowers.extend(ci_l)
            all_plotted_lowers.extend(y_vals)

            fig.add_trace(
                go.Scatter(
                    x=x_vals + x_vals[::-1],
                    y=ci_u + ci_l[::-1],
                    fill="toself",
                    fillcolor="rgba(2, 132, 199, 0.15)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=True,
                    name="95% CI",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines+markers",
                    line=dict(color="#0284C7", width=2.5),
                    marker=dict(size=7),
                    name="AUROC(t)",
                    hovertemplate="AUROC(t): %{y:.3f} at t=%{x}<extra></extra>",
                )
            )

        valid_lowers = [
            float(v) for v in all_plotted_lowers if v is not None and np.isfinite(v)
        ]
        min_lower = float(np.nanmin(valid_lowers)) if valid_lowers else 0.45
        y_min = max(0.0, min(0.45, min_lower - 0.02))

        fig.add_trace(
            go.Scatter(
                x=[min(x_vals), max(x_vals)] if len(x_vals) > 0 else [0, 1],
                y=[0.5, 0.5],
                mode="lines",
                line=dict(dash="dot", color="#94A3B8", width=1.2),
                name="Chance (0.50)",
                hoverinfo="skip",
            )
        )

        fig.update_layout(
            title=dict(
                text=title or "Longitudinal Discrimination Trajectory (AUROC(t))",
                font=dict(size=15, color="#0F172A"),
            ),
            xaxis=dict(
                title="Follow-up Time Horizon",
                showgrid=True,
                gridcolor="#F1F5F9",
            ),
            yaxis=dict(
                title="Cumulative/Dynamic AUROC",
                range=[y_min, 1.02],
                showgrid=True,
                gridcolor="#F1F5F9",
            ),
            template="plotly_white",
            legend=dict(
                yanchor="bottom",
                y=0.02,
                xanchor="left",
                x=0.02,
                bgcolor="rgba(255, 255, 255, 0.85)",
                bordercolor="#E2E8F0",
                borderwidth=1,
            ),
            margin=dict(l=50, r=30, t=50, b=50),
            height=500,
        )

    return DualFigureResult(
        png_path=res.png_path,
        alt_text=res.alt_text,
        caption=res.caption,
        source_df=res.source_df,
        csv_path=res.csv_path,
        plotly_fig=fig,
    )


def plot_dual_reclassification(
    reclass_data: dict[str, Any] | None = None,
    p_old: pd.Series | np.ndarray | None = None,
    p_new: pd.Series | np.ndarray | None = None,
    y_true: pd.Series | np.ndarray | None = None,
    thresholds: list[float] | None = None,
    title: str = "Risk Reclassification (Baseline vs Updated Model)",
    out_path: str | Path = "figures/reclassification.png",
    target: Literal["docx", "pptx", "web"] = "docx",
) -> DualFigureResult:
    """
    Renders both publication-grade 300 DPI Matplotlib figure and interactive Plotly risk reclassification plot.
    """
    res = plot_reclassification(
        reclass_data=reclass_data,
        p_old=p_old,
        p_new=p_new,
        y_true=y_true,
        thresholds=thresholds,
        title=title,
        out_path=out_path,
        target=target,
    )

    df_source = res.source_df
    p_0 = df_source["P_Old"].to_numpy()
    p_1 = df_source["P_New"].to_numpy()
    y_t = df_source["Y_True"].to_numpy()

    idx_0 = y_t == 0
    idx_1 = y_t == 1

    fig = go.Figure()

    # Diagonal equality line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="#94A3B8", width=1.5),
            name="No Reclassification (p_new = p_old)",
            hoverinfo="skip",
        )
    )

    # Non-events (Y=0)
    fig.add_trace(
        go.Scatter(
            x=p_0[idx_0],
            y=p_1[idx_0],
            mode="markers",
            marker=dict(size=7, color="#0284C7", opacity=0.55),
            name=f"Non-Events (Y=0, n={int(np.sum(idx_0))})",
            hovertemplate=(
                "<b>Non-Event (Y=0)</b><br>"
                "Baseline Risk: %{x:.3f}<br>"
                "Updated Risk: %{y:.3f}<br>"
                "Shift: %{text:.3f}<extra></extra>"
            ),
            text=(p_1[idx_0] - p_0[idx_0]),
        )
    )

    # Events (Y=1)
    fig.add_trace(
        go.Scatter(
            x=p_0[idx_1],
            y=p_1[idx_1],
            mode="markers",
            marker=dict(size=8, color="#DC2626", symbol="triangle-up", opacity=0.75),
            name=f"Events (Y=1, n={int(np.sum(idx_1))})",
            hovertemplate=(
                "<b>Event (Y=1)</b><br>"
                "Baseline Risk: %{x:.3f}<br>"
                "Updated Risk: %{y:.3f}<br>"
                "Shift: %{text:.3f}<extra></extra>"
            ),
            text=(p_1[idx_1] - p_0[idx_1]),
        )
    )

    shapes = []
    if thresholds:
        for th in thresholds:
            shapes.append(
                dict(
                    type="line",
                    x0=th,
                    x1=th,
                    y0=0,
                    y1=1,
                    line=dict(color="#CBD5E1", dash="dot", width=1),
                )
            )
            shapes.append(
                dict(
                    type="line",
                    x0=0,
                    x1=1,
                    y0=th,
                    y1=th,
                    line=dict(color="#CBD5E1", dash="dot", width=1),
                )
            )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#0F172A")),
        xaxis=dict(
            title="Baseline Model Predicted Risk (p_old)",
            range=[-0.02, 1.02],
            showgrid=True,
            gridcolor="#F1F5F9",
        ),
        yaxis=dict(
            title="Updated Model Predicted Risk (p_new)",
            range=[-0.02, 1.02],
            showgrid=True,
            gridcolor="#F1F5F9",
        ),
        shapes=shapes,
        template="plotly_white",
        legend=dict(
            yanchor="bottom",
            y=0.02,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="#E2E8F0",
            borderwidth=1,
        ),
        margin=dict(l=50, r=30, t=50, b=50),
        height=520,
    )

    return DualFigureResult(
        png_path=res.png_path,
        alt_text=res.alt_text,
        caption=res.caption,
        source_df=res.source_df,
        csv_path=res.csv_path,
        plotly_fig=fig,
    )

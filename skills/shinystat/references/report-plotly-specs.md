# Standalone Pure-Python Plotly Spec Generators (Zero Dependencies)

Companion to [report-builder.md](report-builder.md): load only when the HTML report needs interactive `plotly_figs`. Each `build_plotly_*_spec` returns a plain `{"data": [...], "layout": {...}}` dict to pass into `generate_html_report(..., plotly_figs=[...])`.

## 1. Spec Generators (`build_plotly_*_spec`)

In cloud sandboxes (such as Claude Web Analysis Tool), external packages like `medstat` or `plotly` are not available. The functions below construct 100% compliant Plotly JSON dictionaries (`{"data": [...], "layout": {...}}`) in pure Python using native data types (`dict`, `list`, `float`, `str`). Pass the resulting dictionary objects directly into `generate_html_report(..., plotly_figs=[...])` to enable full interactive exploration alongside 300 DPI publication figures.

```python
import math


def _clean_float(val, default=None):
    """Safely converts val to finite float or returns default."""
    if val is None:
        return default
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def build_plotly_roc_spec(
    roc_data: dict,
    paired_roc_data: dict = None,
    title: str = "Receiver Operating Characteristic (ROC) Curve",
) -> dict:
    """Builds a pure-dict Plotly specification for an empirical ROC curve with zero dependencies."""
    clean_pairs = []
    for x, y in zip(roc_data.get("fpr", []), roc_data.get("tpr", [])):
        try:
            fx, fy = float(x), float(y)
            if math.isfinite(fx) and math.isfinite(fy):
                clean_pairs.append((fx, fy))
        except (TypeError, ValueError):
            continue
    fpr = [p[0] for p in clean_pairs]
    tpr = [p[1] for p in clean_pairs]
    auc = _clean_float(roc_data.get("auc"))
    ci_l = _clean_float(roc_data.get("ci_lower"))
    ci_u = _clean_float(roc_data.get("ci_upper"))
    model_name = str(roc_data.get("name", "Model 1"))

    if auc is not None:
        if ci_l is not None and ci_u is not None:
            label_1 = f"{model_name}: AUC = {auc:.3f} (95% CI, {ci_l:.3f}–{ci_u:.3f})"
        else:
            label_1 = f"{model_name}: AUC = {auc:.3f}"
    else:
        label_1 = f"{model_name}: AUC = N/A"

    traces = [
        {
            "type": "scatter",
            "x": [0, 1],
            "y": [0, 1],
            "mode": "lines",
            "line": {"dash": "dash", "color": "#94A3B8", "width": 1.5},
            "name": "Chance Diagonal (AUC = 0.500)",
            "hoverinfo": "skip",
        },
        {
            "type": "scatter",
            "x": fpr,
            "y": tpr,
            "mode": "lines",
            "line": {"color": "#0284C7", "width": 2.5},
            "name": label_1,
            "hovertemplate": (
                f"<b>{model_name}</b><br>"
                "1 - Specificity (FPR): %{x:.3f}<br>"
                "Sensitivity (TPR): %{y:.3f}<extra></extra>"
            ),
        },
    ]

    opt_pt = roc_data.get("optimal_point")
    if opt_pt:
        opt_fpr = _clean_float(opt_pt.get("fpr"))
        opt_tpr = _clean_float(opt_pt.get("tpr"))
        if opt_fpr is not None and opt_tpr is not None:
            opt_cut = _clean_float(opt_pt.get("cutoff"), default=0.0)
            opt_sens = _clean_float(roc_data.get("sensitivity"), default=opt_tpr)
            opt_spec = _clean_float(roc_data.get("specificity"), default=1.0 - opt_fpr)
            traces.append({
                "type": "scatter",
                "x": [opt_fpr],
                "y": [opt_tpr],
                "mode": "markers",
                "marker": {"size": 10, "color": "#DC2626", "symbol": "circle"},
                "name": f"Optimal Cutpoint ({opt_cut:.2f})",
                "hovertemplate": (
                    f"<b>Optimal Cutpoint: {opt_cut:.3f}</b><br>"
                    f"Sensitivity: {opt_sens:.1%}<br>"
                    f"Specificity: {opt_spec:.1%}<br>"
                    f"FPR: {opt_fpr:.3f}<extra></extra>"
                ),
            })

    if paired_roc_data:
        p_pairs = []
        for x, y in zip(paired_roc_data.get("fpr", []), paired_roc_data.get("tpr", [])):
            try:
                fx, fy = float(x), float(y)
                if math.isfinite(fx) and math.isfinite(fy):
                    p_pairs.append((fx, fy))
            except (TypeError, ValueError):
                continue
        p_fpr = [p[0] for p in p_pairs]
        p_tpr = [p[1] for p in p_pairs]
        p_auc = _clean_float(paired_roc_data.get("auc"))
        p_ci_l = _clean_float(paired_roc_data.get("ci_lower"))
        p_ci_u = _clean_float(paired_roc_data.get("ci_upper"))
        p_name = str(paired_roc_data.get("name", "Model 2"))
        if p_auc is not None:
            if p_ci_l is not None and p_ci_u is not None:
                p_label = f"{p_name}: AUC = {p_auc:.3f} (95% CI, {p_ci_l:.3f}–{p_ci_u:.3f})"
            else:
                p_label = f"{p_name}: AUC = {p_auc:.3f}"
        else:
            p_label = f"{p_name}: AUC = N/A"
        traces.append({
            "type": "scatter",
            "x": p_fpr,
            "y": p_tpr,
            "mode": "lines",
            "line": {"color": "#D97706", "width": 2.0},
            "name": p_label,
            "hovertemplate": (
                f"<b>{p_name}</b><br>"
                "1 - Specificity (FPR): %{x:.3f}<br>"
                "Sensitivity (TPR): %{y:.3f}<extra></extra>"
            ),
        })

    layout = {
        "title": {"text": title, "x": 0.5, "xanchor": "center"},
        "xaxis": {
            "title": {"text": "1 - Specificity (False Positive Rate)"},
            "range": [-0.02, 1.02],
            "dtick": 0.2,
            "gridcolor": "#F1F5F9",
            "zeroline": False,
        },
        "yaxis": {
            "title": {"text": "Sensitivity (True Positive Rate)"},
            "range": [-0.02, 1.02],
            "dtick": 0.2,
            "gridcolor": "#F1F5F9",
            "zeroline": False,
        },
        "template": "plotly_white",
        "legend": {
            "x": 0.45,
            "y": 0.1,
            "bgcolor": "rgba(255,255,255,0.9)",
            "bordercolor": "#E2E8F0",
            "borderwidth": 1,
        },
        "hovermode": "closest",
        "margin": {"l": 60, "r": 30, "t": 60, "b": 60},
    }
    return {"data": traces, "layout": layout}


def build_plotly_km_spec(
    km_data: dict,
    title: str = "Kaplan-Meier Survival Curves",
    x_label: str = "Time",
    y_label: str = "Survival Probability",
) -> dict:
    """Builds a pure-dict Plotly specification for stratified Kaplan-Meier curves with 95% CI ribbons."""
    colors = ["#0284C7", "#D97706", "#059669", "#7C3AED", "#DC2626"]
    traces = []

    for idx, (group_name, curve) in enumerate(km_data.items()):
        color = colors[idx % len(colors)]
        raw_t = curve.get("timeline", [])
        raw_s = curve.get("survival", [])
        raw_cil = curve.get("ci_lower")
        raw_ciu = curve.get("ci_upper")

        timeline, survival, ci_lower, ci_upper = [], [], [], []
        has_ci = (
            raw_cil is not None
            and raw_ciu is not None
            and len(raw_cil) == len(raw_t)
            and len(raw_ciu) == len(raw_t)
        )

        for i in range(len(raw_t)):
            try:
                t_val = float(raw_t[i])
                s_val = float(raw_s[i])
                if not (math.isfinite(t_val) and math.isfinite(s_val)):
                    continue
                timeline.append(t_val)
                survival.append(s_val)
                if has_ci:
                    try:
                        cl = float(raw_cil[i])
                        cu = float(raw_ciu[i])
                        if math.isfinite(cl) and math.isfinite(cu):
                            ci_lower.append(cl)
                            ci_upper.append(cu)
                        else:
                            ci_lower.append(None)
                            ci_upper.append(None)
                    except (TypeError, ValueError):
                        ci_lower.append(None)
                        ci_upper.append(None)
            except (TypeError, ValueError, IndexError):
                continue

        valid_ci_indices = [
            k for k in range(len(timeline))
            if has_ci and ci_lower[k] is not None and ci_upper[k] is not None
        ]
        if valid_ci_indices:
            step_t = []
            step_ci_u = []
            step_ci_l = []
            for idx_pos, k in enumerate(valid_ci_indices):
                if idx_pos == 0:
                    step_t.append(timeline[k])
                    step_ci_u.append(ci_upper[k])
                    step_ci_l.append(ci_lower[k])
                else:
                    prev_k = valid_ci_indices[idx_pos - 1]
                    step_t.extend([timeline[k], timeline[k]])
                    step_ci_u.extend([ci_upper[prev_k], ci_upper[k]])
                    step_ci_l.extend([ci_lower[prev_k], ci_lower[k]])

            traces.append({
                "type": "scatter",
                "x": step_t,
                "y": step_ci_u,
                "mode": "lines",
                "line": {"width": 0, "color": "rgba(0,0,0,0)"},
                "showlegend": False,
                "hoverinfo": "skip",
            })
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            traces.append({
                "type": "scatter",
                "x": step_t,
                "y": step_ci_l,
                "mode": "lines",
                "line": {"width": 0, "color": "rgba(0,0,0,0)"},
                "fill": "tonexty",
                "fillcolor": f"rgba({r}, {g}, {b}, 0.15)",
                "showlegend": False,
                "hoverinfo": "skip",
            })

        traces.append({
            "type": "scatter",
            "x": timeline,
            "y": survival,
            "mode": "lines",
            "line": {"color": color, "width": 2.5, "shape": "hv"},
            "name": str(group_name),
            "hovertemplate": (
                f"<b>{group_name}</b><br>"
                f"{x_label}: %{{x:.1f}}<br>"
                "Survival: %{y:.3f}<extra></extra>"
            ),
        })

        if "censored_times" in curve and curve["censored_times"] and timeline:
            cens_times = [
                float(x)
                for x in curve["censored_times"]
                if _clean_float(x) is not None
            ]
            cens_surv = []
            for ct in cens_times:
                s_val = survival[0]
                for t, s in zip(timeline, survival):
                    if t <= ct:
                        s_val = s
                    else:
                        break
                cens_surv.append(s_val)

            if cens_times:
                traces.append({
                    "type": "scatter",
                    "x": cens_times,
                    "y": cens_surv,
                    "mode": "markers",
                    "marker": {"symbol": "cross", "size": 6, "color": color},
                    "name": f"{group_name} (Censored)",
                    "showlegend": False,
                    "hovertemplate": f"Censored at {x_label} %{{x:.1f}}<extra></extra>",
                })

    layout = {
        "title": {"text": title, "x": 0.5, "xanchor": "center"},
        "xaxis": {
            "title": {"text": x_label},
            "gridcolor": "#F1F5F9",
            "zeroline": False,
        },
        "yaxis": {
            "title": {"text": y_label},
            "range": [-0.02, 1.05],
            "dtick": 0.2,
            "gridcolor": "#F1F5F9",
            "zeroline": False,
        },
        "template": "plotly_white",
        "legend": {
            "x": 0.75,
            "y": 0.95,
            "bgcolor": "rgba(255,255,255,0.9)",
            "bordercolor": "#E2E8F0",
            "borderwidth": 1,
        },
        "hovermode": "closest",
        "margin": {"l": 60, "r": 30, "t": 60, "b": 60},
    }
    return {"data": traces, "layout": layout}


def build_plotly_calibration_spec(
    calib_bins: dict,
    title: str = "Model Calibration Curve",
    brier_score: float = None,
    ici: float = None,
) -> dict:
    """Builds a pure-dict Plotly specification for a decile calibration curve with Wilson 95% CIs."""
    raw_pred = calib_bins.get("pred_mean", [])
    raw_obs = calib_bins.get("obs_rate", [])
    raw_cil = calib_bins.get("ci_lower")
    raw_ciu = calib_bins.get("ci_upper")
    has_ci = (
        raw_cil is not None
        and raw_ciu is not None
        and len(raw_cil) == len(raw_obs)
        and len(raw_ciu) == len(raw_obs)
    )

    pred_mean, obs_rate, ci_low, ci_high = [], [], [], []
    for idx, (p, o) in enumerate(zip(raw_pred, raw_obs)):
        try:
            fp, fo = float(p), float(o)
            if math.isfinite(fp) and math.isfinite(fo):
                pred_mean.append(fp)
                obs_rate.append(fo)
                if has_ci:
                    try:
                        fl, fh = float(raw_cil[idx]), float(raw_ciu[idx])
                        if math.isfinite(fl) and math.isfinite(fh):
                            ci_low.append(fl)
                            ci_high.append(fh)
                        else:
                            ci_low.append(None)
                            ci_high.append(None)
                    except (TypeError, ValueError):
                        ci_low.append(None)
                        ci_high.append(None)
        except (TypeError, ValueError):
            continue

    obs_trace = {
        "type": "scatter",
        "x": pred_mean,
        "y": obs_rate,
        "mode": "markers+lines",
        "line": {"color": "#0284C7", "width": 2.0},
        "marker": {"size": 8, "color": "#0284C7", "symbol": "circle"},
    }

    if has_ci and len(ci_low) == len(obs_rate) and any(
        l is not None and h is not None and math.isfinite(l) and math.isfinite(h) and l <= h and (h > o or o > l)
        for l, h, o in zip(ci_low, ci_high, obs_rate)
    ):
        error_y_plus = [
            h - o if (h is not None and math.isfinite(h) and h >= o) else 0
            for h, o in zip(ci_high, obs_rate)
        ]
        error_y_minus = [
            o - l if (l is not None and math.isfinite(l) and o >= l) else 0
            for l, o in zip(ci_low, obs_rate)
        ]
        ci_hover = [
            f"<br>95% CI: [{l:.3f}, {h:.3f}]"
            if (l is not None and h is not None and math.isfinite(l) and math.isfinite(h))
            else ""
            for l, h in zip(ci_low, ci_high)
        ]
        obs_trace.update({
            "error_y": {
                "type": "data",
                "symmetric": False,
                "array": error_y_plus,
                "arrayminus": error_y_minus,
                "color": "#0284C7",
                "thickness": 1.5,
                "width": 4,
            },
            "customdata": ci_hover,
            "name": "Decile Bins (95% CI)",
            "hovertemplate": (
                "<b>Decile Group</b><br>"
                "Predicted Risk: %{x:.3f}<br>"
                "Observed Rate: %{y:.3f}"
                "%{customdata}<extra></extra>"
            ),
        })
    else:
        obs_trace.update({
            "name": f"Observed vs Predicted ({len(pred_mean)} bins)",
            "hovertemplate": (
                "<b>Bin Group</b><br>"
                "Predicted Risk: %{x:.3f}<br>"
                "Observed Rate: %{y:.3f}<extra></extra>"
            ),
        })

    traces = [
        {
            "type": "scatter",
            "x": [0, 1],
            "y": [0, 1],
            "mode": "lines",
            "line": {"dash": "dash", "color": "#94A3B8", "width": 1.5},
            "name": "Perfect Calibration",
            "hoverinfo": "skip",
        },
        obs_trace,
    ]

    clean_brier = _clean_float(brier_score)
    clean_ici = _clean_float(ici)
    annotations = []
    annot_text = []
    if clean_brier is not None:
        annot_text.append(f"Brier Score: {clean_brier:.3f}")
    if clean_ici is not None:
        annot_text.append(f"ICI: {clean_ici:.3f}")
    if annot_text:
        annotations.append({
            "text": "<br>".join(annot_text),
            "xref": "paper",
            "yref": "paper",
            "x": 0.05,
            "y": 0.95,
            "showarrow": False,
            "bgcolor": "rgba(255,255,255,0.9)",
            "bordercolor": "#CBD5E1",
            "borderwidth": 1,
            "font": {"size": 12},
        })

    layout = {
        "title": {"text": title, "x": 0.5, "xanchor": "center"},
        "xaxis": {
            "title": {"text": "Mean Predicted Probability"},
            "range": [-0.02, 1.02],
            "dtick": 0.2,
            "gridcolor": "#F1F5F9",
            "zeroline": False,
        },
        "yaxis": {
            "title": {"text": "Observed Event Rate"},
            "range": [-0.02, 1.02],
            "dtick": 0.2,
            "gridcolor": "#F1F5F9",
            "zeroline": False,
        },
        "template": "plotly_white",
        "legend": {
            "x": 0.55,
            "y": 0.1,
            "bgcolor": "rgba(255,255,255,0.9)",
            "bordercolor": "#E2E8F0",
            "borderwidth": 1,
        },
        "annotations": annotations,
        "margin": {"l": 60, "r": 30, "t": 60, "b": 60},
    }
    return {"data": traces, "layout": layout}


def build_plotly_dca_spec(
    dca_data: list[dict] | dict,
    title: str = "Decision Curve Analysis (Net Benefit)",
) -> dict:
    """Builds a pure-dict Plotly specification for Decision Curve Analysis (Vickers DCA)."""
    strategies = {}
    if isinstance(dca_data, dict) and "threshold" not in dca_data:
        strategies = dca_data
    elif isinstance(dca_data, list):
        for row in dca_data:
            strat = row.get("strategy", "Model")
            if strat not in strategies:
                strategies[strat] = {"threshold": [], "net_benefit": []}
            strategies[strat]["threshold"].append(row.get("threshold"))
            strategies[strat]["net_benefit"].append(row.get("net_benefit"))
    else:
        strategies = {"Model": dca_data}

    traces = []
    color_map = {
        "treat all": ("#64748B", "dot", 1.5),
        "treat none": ("#94A3B8", "solid", 1.0),
        "all": ("#64748B", "dot", 1.5),
        "none": ("#94A3B8", "solid", 1.0),
    }

    model_colors = ["#0284C7", "#D97706", "#059669", "#7C3AED"]
    model_idx = 0

    for strat_name, vals in strategies.items():
        # Pair threshold and net_benefit, filter finite values, and sort strictly by ascending threshold
        pairs = []
        raw_th = vals.get("threshold", [])
        raw_nb = vals.get("net_benefit", [])
        for x, y in zip(raw_th, raw_nb):
            try:
                fx, fy = float(x), float(y)
                if math.isfinite(fx) and math.isfinite(fy):
                    pairs.append((fx, fy))
            except (TypeError, ValueError):
                continue
        pairs.sort(key=lambda p: p[0])
        th = [p[0] for p in pairs]
        nb = [p[1] for p in pairs]

        key = str(strat_name).lower().strip()
        if key in color_map:
            col, dash_style, width = color_map[key]
        else:
            col = model_colors[model_idx % len(model_colors)]
            dash_style = "solid"
            width = 2.5
            model_idx += 1

        traces.append({
            "type": "scatter",
            "x": th,
            "y": nb,
            "mode": "lines",
            "line": {"color": col, "width": width, "dash": dash_style},
            "name": str(strat_name),
            "hovertemplate": (
                f"<b>{strat_name}</b><br>"
                "Threshold Probability: %{x:.2f}<br>"
                "Net Benefit: %{y:.3f}<extra></extra>"
            ),
        })

    layout = {
        "title": {"text": title, "x": 0.5, "xanchor": "center"},
        "xaxis": {
            "title": {"text": "Threshold Probability (Pt)"},
            "range": [-0.02, 1.02],
            "dtick": 0.1,
            "gridcolor": "#F1F5F9",
            "zeroline": True,
        },
        "yaxis": {
            "title": {"text": "Net Benefit"},
            "gridcolor": "#F1F5F9",
            "zeroline": True,
        },
        "template": "plotly_white",
        "legend": {
            "x": 0.75,
            "y": 0.95,
            "bgcolor": "rgba(255,255,255,0.9)",
            "bordercolor": "#E2E8F0",
            "borderwidth": 1,
        },
        "hovermode": "closest",
        "margin": {"l": 60, "r": 30, "t": 60, "b": 60},
    }
    return {"data": traces, "layout": layout}


def build_plotly_bland_altman_spec(
    ba_data: dict,
    units: str = "",
    title: str = "Bland-Altman Agreement Plot",
) -> dict:
    """Builds a pure-dict Plotly specification for a Bland-Altman agreement plot."""
    means, diffs = [], []
    for m, d in zip(ba_data.get("means", []), ba_data.get("diffs", [])):
        try:
            fm, fd = float(m), float(d)
            if math.isfinite(fm) and math.isfinite(fd):
                means.append(fm)
                diffs.append(fd)
        except (TypeError, ValueError):
            continue

    sample_mean_d = sum(diffs) / len(diffs) if diffs else None
    mean_diff = _clean_float(ba_data.get("mean_diff"))
    if mean_diff is None:
        mean_diff = sample_mean_d

    loa_upper = _clean_float(ba_data.get("loa_upper"))
    loa_lower = _clean_float(ba_data.get("loa_lower"))
    if (loa_upper is None or loa_lower is None) and len(diffs) >= 2 and sample_mean_d is not None:
        variance = sum((d - sample_mean_d) ** 2 for d in diffs) / (len(diffs) - 1)
        sd = math.sqrt(variance)
        bias_center = mean_diff if mean_diff is not None else sample_mean_d
        if loa_upper is None:
            loa_upper = bias_center + 1.96 * sd
        if loa_lower is None:
            loa_lower = bias_center - 1.96 * sd

    min_x = min(means) if means else 0.0
    max_x = max(means) if means else 1.0
    pad = (max_x - min_x) * 0.05 if max_x > min_x else 0.1
    x_range = [min_x - pad, max_x + pad]
    unit_str = f" ({units})" if units else ""

    traces = [
        {
            "type": "scatter",
            "x": means,
            "y": diffs,
            "mode": "markers",
            "marker": {"size": 6, "color": "#0284C7", "opacity": 0.65},
            "name": "Paired Differences",
            "hovertemplate": (
                f"Mean: %{{x:.2f}}{unit_str}<br>"
                f"Diff: %{{y:.2f}}{unit_str}<extra></extra>"
            ),
        },
        {
            "type": "scatter",
            "x": x_range,
            "y": [0.0, 0.0],
            "mode": "lines",
            "line": {"dash": "dot", "color": "#94A3B8", "width": 1.0},
            "name": "Zero Bias (0.0)",
            "hoverinfo": "skip",
        },
    ]

    if mean_diff is not None:
        traces.append({
            "type": "scatter",
            "x": x_range,
            "y": [mean_diff, mean_diff],
            "mode": "lines",
            "line": {"color": "#0F172A", "width": 2.0},
            "name": f"Mean Bias: {mean_diff:.2f}{unit_str}",
            "hovertemplate": f"Mean Bias = {mean_diff:.2f}{unit_str}<extra></extra>",
        })
    if loa_upper is not None:
        traces.append({
            "type": "scatter",
            "x": x_range,
            "y": [loa_upper, loa_upper],
            "mode": "lines",
            "line": {"dash": "dash", "color": "#DC2626", "width": 1.5},
            "name": f"+1.96 SD (Upper LoA): {loa_upper:.2f}{unit_str}",
            "hovertemplate": f"Upper LoA = {loa_upper:.2f}{unit_str}<extra></extra>",
        })
    if loa_lower is not None:
        traces.append({
            "type": "scatter",
            "x": x_range,
            "y": [loa_lower, loa_lower],
            "mode": "lines",
            "line": {"dash": "dash", "color": "#DC2626", "width": 1.5},
            "name": f"-1.96 SD (Lower LoA): {loa_lower:.2f}{unit_str}",
            "hovertemplate": f"Lower LoA = {loa_lower:.2f}{unit_str}<extra></extra>",
        })

    layout = {
        "title": {"text": title, "x": 0.5, "xanchor": "center"},
        "xaxis": {
            "title": {"text": f"Mean of Measurements{unit_str}"},
            "range": x_range,
            "gridcolor": "#F1F5F9",
            "zeroline": False,
        },
        "yaxis": {
            "title": {"text": f"Difference (Method A - Method B){unit_str}"},
            "gridcolor": "#F1F5F9",
            "zeroline": False,
        },
        "template": "plotly_white",
        "legend": {
            "x": 0.65,
            "y": 0.98,
            "bgcolor": "rgba(255,255,255,0.9)",
            "bordercolor": "#E2E8F0",
            "borderwidth": 1,
        },
        "hovermode": "closest",
        "margin": {"l": 60, "r": 30, "t": 60, "b": 60},
    }
    return {"data": traces, "layout": layout}


def build_plotly_forest_spec(
    forest_data: list[dict],
    scale: str = "OR",
    title: str = "Multivariable Effect Estimates (Forest Plot)",
) -> dict:
    """Builds a pure-dict Plotly specification for a clinical Forest plot."""
    is_log = scale.upper() in ("OR", "HR", "RR")
    null_val = 1.0 if is_log else 0.0

    terms, estimates, ci_lower, ci_upper, customdata = [], [], [], [], []
    for d in forest_data:
        try:
            e = float(d["estimate"])
            l = float(d["ci_lower"])
            u = float(d["ci_upper"])
            if math.isfinite(e) and math.isfinite(l) and math.isfinite(u):
                if is_log and (e <= 0 or l <= 0 or u <= 0):
                    continue
                terms.append(str(d.get("term", "")))
                estimates.append(e)
                ci_lower.append(l)
                ci_upper.append(u)
                p = _clean_float(d.get("p_value"))
                p_str = f"p = {p:.3f}" if p is not None else ""
                customdata.append([l, u, p_str])
        except (TypeError, ValueError, KeyError):
            continue

    error_x_plus = [u - e for u, e in zip(ci_upper, estimates)]
    error_x_minus = [e - l for l, e in zip(ci_lower, estimates)]

    traces = [
        {
            "type": "scatter",
            "x": [null_val, null_val],
            "y": [-0.5, max(len(terms) - 0.5, 0.5)],
            "mode": "lines",
            "line": {"dash": "dash", "color": "#94A3B8", "width": 1.5},
            "name": f"Null Effect ({null_val})",
            "hoverinfo": "skip",
        },
        {
            "type": "scatter",
            "x": estimates,
            "y": terms,
            "mode": "markers",
            "marker": {"size": 9, "color": "#0284C7", "symbol": "square"},
            "error_x": {
                "type": "data",
                "symmetric": False,
                "array": error_x_plus,
                "arrayminus": error_x_minus,
                "color": "#0284C7",
                "thickness": 1.5,
                "width": 5,
            },
            "customdata": customdata,
            "name": f"Estimate ({scale})",
            "hovertemplate": (
                "<b>%{y}</b><br>"
                f"{scale}: %{{x:.2f}}<br>"
                "95% CI: [%{customdata[0]:.2f}, %{customdata[1]:.2f}]<br>"
                "%{customdata[2]}<extra></extra>"
            ),
        },
    ]

    layout = {
        "title": {"text": title, "x": 0.5, "xanchor": "center"},
        "xaxis": {
            "title": {"text": f"Effect Size ({scale}, 95% CI)"},
            "type": "log" if is_log else "linear",
            "gridcolor": "#F1F5F9",
            "zeroline": not is_log,
        },
        "yaxis": {
            "title": {"text": ""},
            "type": "category",
            "autorange": "reversed",
            "gridcolor": "#F1F5F9",
        },
        "template": "plotly_white",
        "showlegend": False,
        "margin": {"l": 160, "r": 30, "t": 60, "b": 60},
    }
    return {"data": traces, "layout": layout}


def build_plotly_love_spec(
    love_data: list[dict],
    threshold: float = 0.10,
    title: str = "Covariate Balance (Love Plot)",
) -> dict:
    """Builds a pure-dict Plotly specification for an Austin 2009 Love plot of Standardized Mean Differences."""
    clean_thresh = _clean_float(threshold, default=0.10)
    variables, pre_smds, post_smds = [], [], []
    for d in love_data:
        try:
            pre = abs(float(d["pre_smd"]))
            post = abs(float(d["post_smd"]))
            if math.isfinite(pre) and math.isfinite(post):
                variables.append(str(d.get("variable", "")))
                pre_smds.append(pre)
                post_smds.append(post)
        except (TypeError, ValueError, KeyError):
            continue

    traces = [
        {
            "type": "scatter",
            "x": [clean_thresh, clean_thresh],
            "y": [-0.5, max(len(variables) - 0.5, 0.5)],
            "mode": "lines",
            "line": {"dash": "dash", "color": "#DC2626", "width": 1.5},
            "name": f"Threshold ({clean_thresh:.2f})",
            "hoverinfo": "skip",
        },
        {
            "type": "scatter",
            "x": pre_smds,
            "y": variables,
            "mode": "markers",
            "marker": {"size": 8, "color": "#94A3B8", "symbol": "circle"},
            "name": "Unadjusted (Pre-Match)",
            "hovertemplate": "<b>%{y}</b><br>Pre-Match SMD: %{x:.3f}<extra></extra>",
        },
        {
            "type": "scatter",
            "x": post_smds,
            "y": variables,
            "mode": "markers",
            "marker": {"size": 9, "color": "#0284C7", "symbol": "diamond"},
            "name": "Matched (Post-Match)",
            "hovertemplate": "<b>%{y}</b><br>Post-Match SMD: %{x:.3f}<extra></extra>",
        },
    ]

    max_smd = max(max(pre_smds, default=0.2), max(post_smds, default=0.2), clean_thresh)
    layout = {
        "title": {"text": title, "x": 0.5, "xanchor": "center"},
        "xaxis": {
            "title": {"text": "Absolute Standardized Mean Difference (SMD)"},
            "range": [-0.01, max_smd * 1.15],
            "gridcolor": "#F1F5F9",
            "zeroline": True,
            "zerolinecolor": "#E2E8F0",
        },
        "yaxis": {
            "title": {"text": ""},
            "type": "category",
            "autorange": "reversed",
            "gridcolor": "#F1F5F9",
        },
        "template": "plotly_white",
        "legend": {
            "x": 0.65,
            "y": 0.98,
            "bgcolor": "rgba(255,255,255,0.9)",
            "bordercolor": "#E2E8F0",
            "borderwidth": 1,
        },
        "margin": {"l": 160, "r": 30, "t": 60, "b": 60},
    }
    return {"data": traces, "layout": layout}
```

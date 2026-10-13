# Cloud Multi-Format Report Builder & Pillar 5 Integrity Reference

In cloud sandboxes (e.g. Claude Web Artifacts), `medstat` is not installed as an external package. This reference provides self-contained, pure-Python reporting recipes that assemble biostatistical findings into publication-grade documents (Markdown, self-contained HTML, Word `.docx`, and PowerPoint `.pptx`) with zero custom dependencies.

> [!IMPORTANT]
> **Output Format Alignment Gate**: When the user prompt does not explicitly specify a deliverable format, **proactively ask the user upfront** before finalizing the deliverable, recommending **(Dual / Toggle View) Self-Contained HTML (`generate_html_report`, `.html`)** where interactive Plotly.js charts serve as the primary view, paired with a tab/button to switch to "Publication 300 DPI Figure (Static)" for journal submission and downloads.

---

## 1. Cloud Report Generation Architecture

1. **Default & Primary Formats (Zero Extra Dependencies)**:
   - **Self-Contained HTML (`.html`)**: **Default deliverable format** when unspecified. Always available; embeds plots via base64 data URIs and applies ICMJE 3-rule CSS styling.
   - **Markdown (`.md`)**: Always available; formats tables via `pandas.DataFrame.to_markdown()` and links figure assets.
2. **Office Document Fallbacks (`python-docx` / `python-pptx`)**:
   - If `docx` or `pptx` packages are installed in the sandbox, use the standalone snippets below upon explicit user request.
   - If either package is missing (`ImportError`), automatically fall back to generating **Self-Contained HTML**.

---

## 2. Standalone Document Generation Recipes

### A. Markdown Report (`generate_markdown_report`)

```python
from pathlib import Path
import pandas as pd


def generate_markdown_report(
    title: str,
    results: dict,
    tables: dict[str, pd.DataFrame],
    out_path: str = "report.md",
) -> str:
    lines = [
        f"# {title}\n",
        f"**Date:** {results.get('date', 'N/A')} | **Analysis Cohort:** N = {results.get('n', 'N/A')}\n\n---\n",
    ]

    # Executive Summary / Narrative
    lines.append("## Executive Summary\n")
    if "narrative" in results:
        lines.append(f"{results['narrative']}\n")

    # Tables
    for caption, df in tables.items():
        lines.append(f"\n### {caption}\n")
        try:
            lines.append(df.to_markdown(index=False) + "\n")
        except Exception:
            lines.append(df.to_string(index=False) + "\n")

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("\n".join(lines), encoding="utf-8")
    return str(out_p)
```

### B. Self-Contained HTML Report with Dual / Toggle View

Generates a standalone single-file HTML report with ICMJE 3-rule tables and **Dual / Toggle View** (interactive Plotly charts with instant toggle to 300 DPI publication figures, download buttons, and offline resilience):

```python
import base64
import html
import json
from pathlib import Path
import pandas as pd


def generate_html_report(
    title: str,
    results: dict,
    tables: dict[str, pd.DataFrame],
    figure_paths: list[str] = None,
    plotly_figs: list[any] = None,  # optional list of go.Figure matching figure_paths
    out_path: str = "report.html",
) -> str:
    has_plotly = bool(plotly_figs)
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
    ]
    if has_plotly:
        parts.append("<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>")

    parts.extend([
        "<style>",
        "  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #1e293b; max-width: 920px; margin: 40px auto; padding: 0 20px; background: #fafafa; }",
        "  .container { background: #fff; padding: 40px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }",
        "  h1 { font-size: 26px; border-bottom: 2px solid #0f172a; padding-bottom: 10px; margin-top: 0; }",
        "  h2 { font-size: 20px; color: #0284c7; margin-top: 30px; }",
        "  table.pub-table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 14px; }",
        "  table.pub-table thead tr { border-top: 2px solid #0f172a; border-bottom: 1px solid #0f172a; }",
        "  table.pub-table tbody tr:last-child { border-bottom: 2px solid #0f172a; }",
        "  table.pub-table th, table.pub-table td { padding: 8px 12px; text-align: left; }",
        "  .figure-card { margin: 30px 0; border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; background: #fff; }",
        "  .figure-toolbar { display: flex; justify-content: space-between; align-items: center; padding: 8px 14px; background: #f8fafc; border-bottom: 1px solid #e2e8f0; flex-wrap: wrap; gap: 8px; }",
        "  .fig-tabs { display: inline-flex; background: #e2e8f0; border-radius: 6px; padding: 2px; }",
        "  .fig-btn { border: none; background: transparent; padding: 5px 12px; border-radius: 5px; font-size: 12.5px; cursor: pointer; color: #475569; }",
        "  .fig-btn.active { background: #fff; color: #0f172a; font-weight: 600; box-shadow: 0 1px 2px rgba(0,0,0,0.06); }",
        "  .fig-actions a { padding: 4px 10px; border: 1px solid #cbd5e1; border-radius: 4px; font-size: 12px; text-decoration: none; color: #334155; background: #fff; }",
        "  .figure-view { padding: 16px; text-align: center; }",
        "  .figure-view img { max-width: 100%; height: auto; border-radius: 4px; }",
        "  .offline-banner { margin: 12px; padding: 12px; background: #fffbeb; border: 1px solid #fde68a; border-radius: 6px; text-align: left; font-size: 13px; color: #92400e; }",
        "  .caption { font-size: 13px; color: #64748b; font-style: italic; margin-top: 6px; }",
        "  @media print { .figure-toolbar { display: none !important; } .view-interactive { display: none !important; } .view-static { display: block !important; } }",
        "</style></head><body><div class='container'>",
        f"<h1>{html.escape(title)}</h1>",
    ])

    # Narrative
    if "narrative" in results:
        parts.append(f"<p>{html.escape(results['narrative'])}</p>")

    # Tables
    for caption, df in tables.items():
        parts.append(f"<div class='caption'><strong>{html.escape(caption)}</strong></div>")
        parts.append("<table class='pub-table'><thead><tr>")
        for col in df.columns:
            parts.append(f"<th>{html.escape(str(col))}</th>")
        parts.append("</tr></thead><tbody>")
        for _, row in df.iterrows():
            parts.append("<tr>")
            for col in df.columns:
                val = str(row[col]) if pd.notna(row[col]) else ""
                parts.append(f"<td>{html.escape(val)}</td>")
            parts.append("</tr>")
        parts.append("</tbody></table>")

    # Figures (Dual / Toggle View)
    if figure_paths:
        for idx, fig_p_str in enumerate(figure_paths):
            fp = Path(fig_p_str)
            if not fp.exists():
                continue
            b64 = base64.b64encode(fp.read_bytes()).decode("utf-8")
            plt_fig = plotly_figs[idx] if plotly_figs and idx < len(plotly_figs) else None
            idx_str = str(idx + 1)
            stem = fp.stem

            parts.append(f"<div class='figure-card' id='fig-card-{idx_str}'>")
            parts.append("<div class='figure-toolbar'>")
            if plt_fig:
                parts.append("<div class='fig-tabs'>")
                parts.append(f"<button type='button' class='fig-btn active' id='tab-i-{idx_str}' onclick=\"switchFig('{idx_str}', 'i')\">Interactive (Plotly)</button>")
                parts.append(f"<button type='button' class='fig-btn' id='tab-s-{idx_str}' onclick=\"switchFig('{idx_str}', 's')\">Publication 300 DPI Figure (Static)</button>")
                parts.append("</div>")
            else:
                parts.append("<span style='font-size:12.5px; font-weight:600;'>Publication 300 DPI Figure (Static)</span>")
            parts.append(f"<div class='fig-actions'><a href='data:image/png;base64,{b64}' download='{stem}.png'>Download 300 DPI PNG</a></div>")
            parts.append("</div>")

            if plt_fig:
                raw_json = (
                    plt_fig.to_json()
                    if hasattr(plt_fig, "to_json")
                    else json.dumps(plt_fig)
                )
                plotly_json = raw_json.replace("<", "\\u003c")
                parts.append(f"<div class='figure-view view-interactive' id='view-i-{idx_str}'>")
                parts.append(f"<div id='plotly-div-{idx_str}' style='min-height:480px; width:100%;'></div>")
                parts.append(f"<div id='offline-msg-{idx_str}' class='offline-banner' style='display:none;'>📡 โหมดออฟไลน์: ไม่สามารถโหลด Plotly CDN ได้ กรุณาสลับไปดูรูป 300 DPI <button onclick=\"switchFig('{idx_str}', 's')\" style='margin-left:8px; cursor:pointer;'>สลับไปดูรูป 300 DPI</button></div>")
                parts.append("</div>")
                parts.append(f"<div class='figure-view view-static' id='view-s-{idx_str}' style='display:none;'>")
                parts.append(f"<img src='data:image/png;base64,{b64}' alt='{stem}'>")
                parts.append("</div>")
                parts.append(f"<script type='application/json' id='spec-{idx_str}'>{plotly_json}</script>")
            else:
                parts.append(f"<div class='figure-view view-static' id='view-s-{idx_str}'>")
                parts.append(f"<img src='data:image/png;base64,{b64}' alt='{stem}'>")
                parts.append("</div>")
            parts.append("</div>")

    if has_plotly:
        parts.append("""
<script>
function switchFig(id, view) {
  var vi = document.getElementById('view-i-' + id);
  var vs = document.getElementById('view-s-' + id);
  var ti = document.getElementById('tab-i-' + id);
  var ts = document.getElementById('tab-s-' + id);
  if (view === 'i') {
    if (typeof Plotly === 'undefined') {
      var msg = document.getElementById('offline-msg-' + id);
      if (msg) msg.style.display = 'block';
      return;
    }
    if (vi) vi.style.display = 'block';
    if (vs) vs.style.display = 'none';
    if (ti) ti.classList.add('active');
    if (ts) ts.classList.remove('active');
    var div = document.getElementById('plotly-div-' + id);
    if (div && window.Plotly) Plotly.Plots.resize(div);
  } else {
    if (vi) vi.style.display = 'none';
    if (vs) vs.style.display = 'block';
    if (ti) ti.classList.remove('active');
    if (ts) ts.classList.add('active');
  }
}
document.addEventListener('DOMContentLoaded', function() {
  if (typeof Plotly === 'undefined') {
    document.querySelectorAll('.figure-card').forEach(function(c) {
      var id = c.id.replace('fig-card-', '');
      var msg = document.getElementById('offline-msg-' + id);
      if (msg) msg.style.display = 'block';
      switchFig(id, 's');
    });
    return;
  }
  document.querySelectorAll('script[type=\"application/json\"][id^=\"spec-\"]').forEach(function(s) {
    var id = s.id.replace('spec-', '');
    var div = document.getElementById('plotly-div-' + id);
    if (div) {
      try {
        var spec = JSON.parse(s.textContent);
        Plotly.newPlot(div, spec.data || [], spec.layout || {}, {responsive: true, displaylogo: false});
      } catch(e) { switchFig(id, 's'); }
    }
  });
});
</script>
""")

    parts.append("</div></body></html>")
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("\n".join(parts), encoding="utf-8")
    return str(out_p)
```

### B.1. Standalone Pure-Python Plotly Spec Generators (Zero Dependencies)

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
    auc = _clean_float(roc_data.get("auc"), default=0.5)
    ci_l = _clean_float(roc_data.get("ci_lower"))
    ci_u = _clean_float(roc_data.get("ci_upper"))
    model_name = str(roc_data.get("name", "Model 1"))

    if ci_l is not None and ci_u is not None:
        label_1 = f"{model_name}: AUC = {auc:.3f} (95% CI, {ci_l:.3f}–{ci_u:.3f})"
    else:
        label_1 = f"{model_name}: AUC = {auc:.3f}"

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
        p_auc = _clean_float(paired_roc_data.get("auc"), default=0.5)
        p_ci_l = _clean_float(paired_roc_data.get("ci_lower"))
        p_ci_u = _clean_float(paired_roc_data.get("ci_upper"))
        p_name = str(paired_roc_data.get("name", "Model 2"))
        if p_ci_l is not None and p_ci_u is not None:
            p_label = f"{p_name}: AUC = {p_auc:.3f} (95% CI, {p_ci_l:.3f}–{p_ci_u:.3f})"
        else:
            p_label = f"{p_name}: AUC = {p_auc:.3f}"
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
                if has_ci:
                    cl = float(raw_cil[i])
                    cu = float(raw_ciu[i])
                    if not (math.isfinite(cl) and math.isfinite(cu)):
                        continue
                    ci_lower.append(cl)
                    ci_upper.append(cu)
                timeline.append(t_val)
                survival.append(s_val)
            except (TypeError, ValueError, IndexError):
                continue

        if has_ci and len(timeline) == len(ci_lower) == len(ci_upper) and timeline:
            step_t = []
            step_ci_u = []
            step_ci_l = []
            for i in range(len(timeline)):
                if i == 0:
                    step_t.append(timeline[0])
                    step_ci_u.append(ci_upper[0])
                    step_ci_l.append(ci_lower[0])
                else:
                    step_t.extend([timeline[i], timeline[i]])
                    step_ci_u.extend([ci_upper[i - 1], ci_upper[i]])
                    step_ci_l.extend([ci_lower[i - 1], ci_lower[i]])

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
    raw_cil = calib_bins.get("ci_lower", raw_obs)
    raw_ciu = calib_bins.get("ci_upper", raw_obs)

    pred_mean, obs_rate, ci_low, ci_high = [], [], [], []
    for p, o, l, h in zip(raw_pred, raw_obs, raw_cil, raw_ciu):
        try:
            fp, fo, fl, fh = float(p), float(o), float(l), float(h)
            if math.isfinite(fp) and math.isfinite(fo) and math.isfinite(fl) and math.isfinite(fh):
                pred_mean.append(fp)
                obs_rate.append(fo)
                ci_low.append(fl)
                ci_high.append(fh)
        except (TypeError, ValueError):
            continue

    error_y_plus = [h - o for h, o in zip(ci_high, obs_rate)]
    error_y_minus = [o - l for l, o in zip(ci_low, obs_rate)]
    customdata = [[l, h] for l, h in zip(ci_low, ci_high)]

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
        {
            "type": "scatter",
            "x": pred_mean,
            "y": obs_rate,
            "mode": "markers+lines",
            "line": {"color": "#0284C7", "width": 2.0},
            "marker": {"size": 8, "color": "#0284C7", "symbol": "circle"},
            "error_y": {
                "type": "data",
                "symmetric": False,
                "array": error_y_plus,
                "arrayminus": error_y_minus,
                "color": "#0284C7",
                "thickness": 1.5,
                "width": 4,
            },
            "customdata": customdata,
            "name": "Decile Bins (95% CI)",
            "hovertemplate": (
                "<b>Decile Group</b><br>"
                "Predicted Risk: %{x:.3f}<br>"
                "Observed Rate: %{y:.3f}<br>"
                "95% CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<extra></extra>"
            ),
        },
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

### C. Standalone Word (`.docx`) Report (when `python-docx` is installed)

```python
try:
    import docx
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import parse_xml
    from docx.oxml.ns import nsdecls

    def generate_docx_report(
        title: str,
        results: dict,
        tables: dict[str, pd.DataFrame],
        figure_paths: list[str] = None,
        out_path: str = "report.docx",
    ) -> str:
        doc = docx.Document()
        doc.add_heading(title, level=0)

        if "narrative" in results:
            doc.add_paragraph(results["narrative"])

        for caption, df in tables.items():
            p = doc.add_paragraph()
            p.add_run(caption).bold = True
            tbl = doc.add_table(rows=len(df) + 1, cols=len(df.columns))
            # Format header
            for j, col in enumerate(df.columns):
                tbl.cell(0, j).text = str(col)
                tbl.cell(0, j).paragraphs[0].runs[0].bold = True
            # Populate data
            for i, row in df.iterrows():
                for j, col in enumerate(df.columns):
                    val = str(row[col]) if pd.notna(row[col]) else ""
                    tbl.cell(i + 1, j).text = val
            # Apply 3-rule borders
            tblPr = tbl._tbl.tblPr
            borders = parse_xml(
                f"<w:tblBorders {nsdecls('w')}>\n"
                f'  <w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>\n'
                f'  <w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/>\n'
                f'  <w:insideH w:val="none"/>\n'
                f'  <w:left w:val="none"/>\n'
                f'  <w:right w:val="none"/>\n'
                f'  <w:insideV w:val="none"/>\n'
                f"</w:tblBorders>"
            )
            tblPr.append(borders)

        if figure_paths:
            for fp in figure_paths:
                p = Path(fp)
                if p.exists():
                    doc.add_picture(str(p), width=Inches(6.5))

        doc.save(out_path)
        return out_path

except ImportError:
    # Automatic fallback when python-docx is unavailable
    def generate_docx_report(
        title: str,
        results: dict,
        tables: dict[str, pd.DataFrame],
        figure_paths: list[str] = None,
        out_path: str = "report.docx",
    ) -> str:
        if Path(out_path).suffix.lower() == ".docx":
            out_path = str(Path(out_path).with_suffix(".html"))
        return generate_html_report(
            title, results, tables, figure_paths, out_path=out_path
        )
```

---

## 3. Standalone Reporting Integrity Verification

Before releasing any report, run the standalone verification routine:

```python
import re
from pathlib import Path
import pandas as pd


def verify_standalone_integrity(
    narrative_text: str, results_dict: dict, table_dfs: list[pd.DataFrame] = None
) -> dict:
    """Verifies numerical traceability and screens for PHI without importing medstat."""
    violations = []

    # 1. PHI Scan
    phi_patterns = [
        (
            re.compile(
                r"\b(?:HN|hn|H\.N\.)\s*[:#/-]?\s*\d{2,}(?:[-/]\d{2,})+\b|\b(?:HN|hn|H\.N\.)\s*[:#/-]?\s*\d{4,}\b"
            ),
            "Hospital Number (HN)",
        ),
        (
            re.compile(r"\b\d{1}[-\s]?\d{4}[-\s]?\d{5}[-\s]?\d{2}[-\s]?\d{1}\b"),
            "Thai Citizen ID (13 digits)",
        ),
        (
            re.compile(
                r"\b(?:0[689]\d{1}[-\s]?\d{3}[-\s]?\d{4}|0[2-57]\d{1}[-\s]?\d{3}[-\s]?\d{3})\b"
            ),
            "Phone Number",
        ),
        (
            re.compile(
                r"\b(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?|นาย|นาง|นางสาว|นพ\.?|พญ\.?|ดร\.?)\s+[A-Za-zก-๙]{2,}\b"
            ),
            "Personal Name Marker",
        ),
        (
            re.compile(
                r"\b(?:DOB|dob|Date of Birth|วันเกิด)\s*[:#-]?\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b",
                re.I,
            ),
            "Date of Birth",
        ),
    ]

    for pat, label in phi_patterns:
        for m in pat.findall(narrative_text):
            violations.append(f"{label}: '{m}'")

    if table_dfs:
        for df in table_dfs:
            if df is not None and isinstance(df, pd.DataFrame):
                for col in df.columns:
                    col_str = str(col)
                    for pat, label in phi_patterns:
                        for m in pat.findall(col_str):
                            violations.append(f"{label}: '{m}'")
                    for val in df[col].dropna():
                        val_str = str(val)
                        for pat, label in phi_patterns:
                            for m in pat.findall(val_str):
                                violations.append(f"{label}: '{m}'")

    # 2. Observational Causal / E-value Caveat Check
    keys_str = " ".join(results_dict.keys()).lower()
    is_causal = any(
        k in keys_str
        for k in ["psm", "propensity", "cox", "hazard_ratio", "odds_ratio", "logistic"]
    )
    has_caveat = True
    if is_causal:
        has_caveat = any(
            phrase in narrative_text.lower()
            for phrase in [
                "unmeasured confounding",
                "residual confounding",
                "confound",
                "e-value",
            ]
        )

    # 3. Numerical Traceability Check
    def _flatten_nums(d):
        nums = []
        if isinstance(d, dict):
            for v in d.values():
                nums.extend(_flatten_nums(v))
        elif hasattr(d, "tolist") and callable(d.tolist):
            nums.extend(_flatten_nums(d.tolist()))
        elif isinstance(d, (list, tuple, set)):
            for v in d:
                nums.extend(_flatten_nums(v))
        elif isinstance(d, (int, float)) and not isinstance(d, bool):
            nums.append(float(d))
        return nums

    known_nums = _flatten_nums(results_dict)
    clean_text = narrative_text.replace(",", "")
    raw_matches = list(
        re.finditer(
            r"(?<!\w)(?<![a-zA-Z]-)(?:[-+]?\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?",
            clean_text,
        )
    )
    ignored_structural = {
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        95.0,
        99.0,
        100.0,
        0.05,
        0.01,
        0.001,
    }
    untraced_nums = []
    for m in raw_matches:
        raw_str = m.group(0)
        try:
            val = float(raw_str)
        except ValueError:
            continue
        if not (-1e7 < val < 1e7):
            continue

        if abs(val) <= 1.0:
            if re.search(r"[eE]", raw_str):
                parts = re.split(r"[eE]", raw_str)
                mantissa = parts[0]
                try:
                    exp = int(parts[1])
                except (IndexError, ValueError):
                    exp = 0
                mantissa_dec = len(mantissa.split(".")[1]) if "." in mantissa else 0
                dec_digits = max(0, mantissa_dec - exp)
            else:
                dec_digits = len(raw_str.split(".")[1]) if "." in raw_str else 2
            abs_tol = min(0.02, 10.0 ** (-dec_digits))
        else:
            abs_tol = 0.02

        matched = False
        for kn in known_nums:
            if abs(val - kn) <= abs_tol:
                matched = True
                break
            if abs(val) > 1.0 and abs(kn) > 1e-9 and abs(val - kn) / abs(kn) <= 0.01:
                matched = True
                break
            if abs(val) > 1.0 and abs(kn) <= 1.0 and abs(val - kn * 100.0) <= 0.05:
                matched = True
                break

        if not matched:
            start = m.start()
            window_before = clean_text[max(0, start - 30) : start].split("\n")[-1]
            is_counter = bool(
                re.search(
                    r"\b(?:Table|Figure|Fig\.?|Tier|Phase|Stage|Grade|Step|Level|Item|Section)\s*$",
                    window_before,
                    re.IGNORECASE,
                )
            )
            date_prefix_pattern = (
                r"(?:"
                r"\b(?:in|year|years|during|period|from|dated|since|until|between)(?:\s+(?:the\s+)?(?:calendar|fiscal|academic|study)?(?:\s*year)?)?(?:\s+"
                r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?))?"
                r"|\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
                r"|\b(?:19\d{2}|20\d{2})\s*(?:-|–|/|\b(?:to|through|until|and)\b)"
                r")\s*$"
            )
            is_year = (
                2010.0 <= val <= 2035.0
                and "." not in raw_str
                and bool(re.search(date_prefix_pattern, window_before, re.IGNORECASE))
            )
            is_structural_10 = val == 10.0 and is_counter
            if not (val in ignored_structural or is_structural_10 or is_year):
                untraced_nums.append(val)

    untraced_rate = len(untraced_nums) / max(len(raw_matches), 1)
    traceability_passed = len(untraced_nums) <= 1 or untraced_rate <= 0.15

    return {
        "passed": len(violations) == 0 and has_caveat and traceability_passed,
        "phi_violations": violations,
        "has_causal_caveat": has_caveat,
        "untraced_numbers": untraced_nums,
        "traceability_passed": traceability_passed,
    }
```

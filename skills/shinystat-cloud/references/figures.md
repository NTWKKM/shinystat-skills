# Publication-Grade Clinical Figures Reference (Cloud Sandbox Standalone Edition)

All figures in `shinystat-cloud` are generated using **standalone `matplotlib` and `numpy`/`scipy` recipes** (static 300/600 DPI PNG, PDF with TrueType `pdf.fonttype = 42`, and SVG) and **zero-dependency pure-Python Plotly dictionary builders** (interactive HTML).

---

## 1. Universal Figure Quality & Layout Contract

Every clinical figure produced in a cloud Python sandbox must satisfy these visual and geometric standards:

1. **Journal House Style Synchronization**: Apply the locked Journal House Style (`NEJM`, `JAMA`, `Lancet`, `BMJ`, or `Clinical`) from [house-styles.md](house-styles.md) before creating axes (`apply_journal_style(style)`).
2. **TrueType Vector Export**: Set `plt.rcParams["pdf.fonttype"] = 42` and `plt.rcParams["ps.fonttype"] = 42` so exported PDFs keep editable text paths.
3. **Spine Despining**: Hide top and right spines (`ax.spines["top"].set_visible(False)`, `ax.spines["right"].set_visible(False)`).
4. **Uncertainty & Sample Size**: Include 95% confidence bands or error bars on all point estimates, and display Number-at-Risk tables below survival and cumulative incidence curves.
5. **Collision-Free Layout**: Call `fig.tight_layout()` and export with `dpi=300` (or `dpi=600` for line art) and `bbox_inches="tight"`.

---

## 2. Available Plot Recipes

### 1. Kaplan-Meier Survival Curve with Number-at-Risk Table
- **Primary Recipe**: Use `plot_km_with_risk_table` in [03-survival-competing-risks.md](recipes/03-survival-competing-risks.md) (Recipe 6).
- **Required Elements**:
  - Stepwise Greenwood 95% CI ribbon (`where="post"`).
  - Censoring tick marks (`|`) along each survival curve.
  - Aligned lower subplot (`gridspec_kw={"height_ratios": [3.5, 1]}`) displaying exact Numbers at Risk at each major x-axis tick.
  - Log-rank $P$-value and Hazard Ratio (95% CI) callout in the upper-right or lower-left corner.

### 2. Competing Risks Cumulative Incidence Function (CIF)
- **Primary Recipe**: Use `compute_aalen_johansen_cif` and `fit_fine_gray_crr` in [03-survival-competing-risks.md](recipes/03-survival-competing-risks.md) (Recipe 8).
- **Standalone Plot Template**:
```python
import matplotlib.pyplot as plt
import numpy as np


def plot_standalone_cif(
    cif_primary: dict,
    cif_competing: dict | None = None,
    title: str = "Cumulative Incidence Function (Aalen-Johansen)",
    x_label: str = "Follow-up Time",
    y_label: str = "Cumulative Incidence",
    out_path: str = "cif_plot.png",
) -> str:
    plt.rcParams["pdf.fonttype"] = 42
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    t1 = np.asarray(cif_primary["timeline"], dtype=float)
    y1 = np.asarray(cif_primary["cif"], dtype=float)
    ax.step(t1, y1, where="post", color="#BC3C29", lw=2.0, label=cif_primary.get("label", "Primary Event"))
    if "ci_lower" in cif_primary and "ci_upper" in cif_primary:
        ax.fill_between(
            t1,
            cif_primary["ci_lower"],
            cif_primary["ci_upper"],
            step="post",
            color="#BC3C29",
            alpha=0.18,
        )
    if cif_competing is not None:
        t2 = np.asarray(cif_competing["timeline"], dtype=float)
        y2 = np.asarray(cif_competing["cif"], dtype=float)
        ax.step(t2, y2, where="post", color="#0072B5", lw=1.8, ls="--", label=cif_competing.get("label", "Competing Event"))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(x_label, fontweight="bold")
    ax.set_ylabel(y_label, fontweight="bold")
    ax.set_title(title, loc="left", fontweight="bold", pad=10)
    ax.set_ylim(0.0, min(1.0, max(0.25, float(np.max(y1)) * 1.25)))
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path
```

### 3. Multivariable / Subgroup / Meta-Analysis Forest Plot
- **Primary Recipe**: Use `fit_firth_logistic` / `fit_multivariable_logistic` in [02-regression-firth-ordinal.md](recipes/02-regression-firth-ordinal.md) or `run_meta_analysis` in [05-diagnostic-meta-power.md](recipes/05-diagnostic-meta-power.md) (Recipe 24).
- **Standalone Plot Template**:
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_standalone_forest(
    df: pd.DataFrame,
    label_col: str = "Variable",
    est_col: str = "OR",
    ci_low_col: str = "ci_lower",
    ci_high_col: str = "ci_upper",
    null_value: float = 1.0,
    log_scale: bool = True,
    xlabel: str = "Adjusted Odds Ratio (95% CI)",
    title: str = "Multivariable Forest Plot",
    out_path: str = "forest_plot.png",
) -> str:
    plt.rcParams["pdf.fonttype"] = 42
    work = df.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.0, max(3.2, 0.45 * len(work) + 1.2)), dpi=300)
    y_pos = np.arange(len(work))
    est = work[est_col].to_numpy(dtype=float)
    low = work[ci_low_col].to_numpy(dtype=float)
    high = work[ci_high_col].to_numpy(dtype=float)

    ax.axvline(null_value, color="#64748B", ls="--", lw=1.1, zorder=1)
    ax.errorbar(
        est,
        y_pos,
        xerr=[est - low, high - est],
        fmt="s",
        color="#0072B5",
        ecolor="#1E293B",
        elinewidth=1.4,
        capsize=3,
        markersize=6,
        zorder=3,
    )
    if log_scale:
        ax.set_xscale("log")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(work[label_col].astype(str))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_title(title, loc="left", fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path
```

### 4. Diagnostic ROC Curve with DeLong 95% CI & Optimal Threshold
- **Primary Recipe**: Use `auc_ci_delong` and `delong_paired_test` in [05-diagnostic-meta-power.md](recipes/05-diagnostic-meta-power.md) (Recipe 3).

### 5. Calibration Belt & Decision Curve Analysis (DCA)
- **Primary Recipe**: Use `evaluate_calibration` and `calculate_dca` in [05-diagnostic-meta-power.md](recipes/05-diagnostic-meta-power.md) (Recipe 4).

### 6. Propensity Score Balance (Love Plot)
- **Primary Recipe**: Use `match_propensity_scores` or `compute_iptw_weights` in [04-causal-iptw-psm.md](recipes/04-causal-iptw-psm.md) (Recipes 6 & 15), plotting $|\text{SMD}|$ pre- and post-adjustment with a vertical threshold line at $|\text{SMD}| = 0.10$.

### 7. Bland-Altman Method Agreement Plot
- **Primary Recipe**: Use `calculate_bland_altman` in [05-diagnostic-meta-power.md](recipes/05-diagnostic-meta-power.md) (Recipe 5), marking mean bias and $\pm 1.96\text{ SD}$ Limits of Agreement (LoA) with 95% CI shading.

### 8. Restricted Cubic Splines (RCS) Dose-Response Curve
- **Primary Recipe**: Use `fit_rcs_spline_model` in [02-regression-firth-ordinal.md](recipes/02-regression-firth-ordinal.md) (Recipe 22), plotting `contrast_df["estimate"]` and `ci_lower`/`ci_upper` across the continuous predictor range with annotated overall and non-linearity $P$-values.

### 9. Cox Proportional Hazards Schoenfeld Residuals Diagnostic Plot
- **Primary Recipe**: Use `fit_cox_ph_with_schoenfeld` in [03-survival-competing-risks.md](recipes/03-survival-competing-risks.md) (Recipe 8).
- **Standalone Plot Template**:
```python
import matplotlib.pyplot as plt
import numpy as np


def plot_schoenfeld_residuals(
    event_times: np.ndarray,
    scaled_residuals: np.ndarray,
    covariate_name: str = "Covariate",
    ph_p_value: float | None = None,
    out_path: str = "schoenfeld_residuals.png",
) -> str:
    """Plot scaled Schoenfeld residuals against time with a smoothed trend line."""
    plt.rcParams["pdf.fonttype"] = 42
    t = np.asarray(event_times, dtype=float).ravel()
    r = np.asarray(scaled_residuals, dtype=float).ravel()
    order = np.argsort(t)
    t, r = t[order], r[order]

    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=300)
    ax.scatter(t, r, color="#64748B", alpha=0.55, s=18, label="Scaled Residuals")
    ax.axhline(0.0, color="#94A3B8", ls="--", lw=1.0)
    if len(t) >= 3:
        coef = np.polyfit(t, r, deg=1)
        ax.plot(t, np.polyval(coef, t), color="#BC3C29", lw=2.0, label="Linear Trend")

    subtitle = f" (Schoenfeld P = {ph_p_value:.3f})" if ph_p_value is not None else ""
    ax.set_title(f"Schoenfeld Residuals: {covariate_name}{subtitle}", loc="left", fontweight="bold")
    ax.set_xlabel("Analysis Time", fontweight="bold")
    ax.set_ylabel(f"Beta(t) for {covariate_name}", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path
```

---

## 3. Standalone Interactive HTML Figures (Zero-Dependency Plotly Dicts)

When generating interactive HTML reports with Dual / Toggle View in cloud sandboxes, use the 7 pure-Python dictionary builders in [report-builder.md §1B](report-builder.md#1b-standalone-pure-python-plotly-spec-generators-zero-dependencies):
- `build_plotly_roc_spec`
- `build_plotly_km_spec`
- `build_plotly_forest_spec`
- `build_plotly_calibration_spec`
- `build_plotly_dca_spec`
- `build_plotly_love_spec`
- `build_plotly_bland_altman_spec`

Pass the resulting dictionary list into `generate_html_report(..., plotly_specs=[...])` along with the static PNG paths in `figure_paths=[...]`.

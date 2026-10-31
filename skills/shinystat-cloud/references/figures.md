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

- **Primary Recipe**: Use `fit_survival_analysis_suite` in [03-survival-competing-risks.md](recipes/03-survival-competing-risks.md) (Recipe 11).
- **Standalone Plot Template**:
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_standalone_km_with_risk_table(
    km_groups: dict[str, pd.DataFrame],
    risk_counts: dict[str, list[int]] | None = None,
    time_ticks: list[float] | None = None,
    title: str = "Kaplan-Meier Survival Estimate",
    x_label: str = "Follow-up Time",
    y_label: str = "Survival Probability",
    out_path: str = "km_plot.png",
) -> str:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    if risk_counts and time_ticks:
        if any(len(counts) != len(time_ticks) for counts in risk_counts.values()):
            raise ValueError("All risk_counts lists must have equal length to time_ticks.")

    fig, (ax, ax_tbl) = plt.subplots(
        2, 1, figsize=(6.8, 5.2), dpi=300, gridspec_kw={"height_ratios": [3.5, 1]}, sharex=True
    )
    palette = ["#0072B5", "#BC3C29", "#20854E", "#E18727"]
    for idx, (label, df_g) in enumerate(km_groups.items()):
        color = palette[idx % len(palette)]
        t = np.asarray(df_g["timeline"] if "timeline" in df_g else df_g["time"], dtype=float)
        s = np.asarray(df_g["survival"] if "survival" in df_g else df_g["estimate"], dtype=float)
        ax.step(t, s, where="post", color=color, lw=2.0, label=str(label))
        if "ci_lower" in df_g and "ci_upper" in df_g:
            ax.fill_between(
                t, df_g["ci_lower"], df_g["ci_upper"], step="post", color=color, alpha=0.18
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(y_label, fontweight="bold")
    ax.set_title(title, loc="left", fontweight="bold", pad=10)
    ax.legend(frameon=False, loc="lower left")

    ax_tbl.spines["top"].set_visible(False)
    ax_tbl.spines["right"].set_visible(False)
    ax_tbl.spines["left"].set_visible(False)
    ax_tbl.yaxis.set_ticks([])
    if risk_counts and time_ticks:
        ax_tbl.set_xticks(time_ticks)
        for idx, (label, counts) in enumerate(risk_counts.items()):
            y_row = 1.0 - (idx + 0.5) / max(1, len(risk_counts))
            ax_tbl.text(time_ticks[0], y_row, f"{label}: ", ha="right", va="center", fontweight="bold", fontsize=8.5)
            for tx, cnt in zip(time_ticks, counts):
                ax_tbl.text(tx, y_row, str(cnt), ha="center", va="center", fontsize=8.5)
    ax_tbl.set_xlabel(x_label, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path
```

### 2. Competing Risks Cumulative Incidence Function (CIF)

- **Primary Recipe**: Use `fit_competing_risks_cif` and `fit_competing_risks_regression` in [03-survival-competing-risks.md](recipes/03-survival-competing-risks.md) (Recipe 15).
- **Standalone Plot Template**:
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_standalone_cif(
    cif_primary: dict | pd.DataFrame,
    cif_competing: dict | pd.DataFrame | None = None,
    risk_counts: dict[str, list[int]] | None = None,
    time_ticks: list[float] | None = None,
    title: str = "Cumulative Incidence Function (Aalen-Johansen)",
    x_label: str = "Follow-up Time",
    y_label: str = "Cumulative Incidence",
    out_path: str = "cif_plot.png",
) -> str:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    if risk_counts and time_ticks:
        if any(len(counts) != len(time_ticks) for counts in risk_counts.values()):
            raise ValueError("All risk_counts lists must have equal length to time_ticks.")

    has_risk = bool(risk_counts and time_ticks)
    if has_risk:
        fig, (ax, ax_tbl) = plt.subplots(
            2, 1, figsize=(6.8, 5.2), dpi=300, gridspec_kw={"height_ratios": [3.5, 1]}, sharex=True
        )
    else:
        fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
        ax_tbl = None

    src = cif_primary["cif_table"] if isinstance(cif_primary, dict) and "cif_table" in cif_primary else cif_primary
    t_key = "time" if "time" in src else "timeline"
    y_key = "cif_primary" if "cif_primary" in src else "cif"
    t1 = np.asarray(src[t_key], dtype=float)
    y1 = np.asarray(src[y_key], dtype=float)
    label1 = cif_primary.get("label", "Primary Event") if isinstance(cif_primary, dict) else "Primary Event"
    ax.step(t1, y1, where="post", color="#BC3C29", lw=2.0, label=label1)

    max_vals = [float(np.nanmax(y1)) if len(y1) else 0.0]
    ci_low_col = "cif_primary_ci_lower" if "cif_primary_ci_lower" in src else ("ci_lower" if "ci_lower" in src else None)
    ci_high_col = "cif_primary_ci_upper" if "cif_primary_ci_upper" in src else ("ci_upper" if "ci_upper" in src else None)
    if ci_low_col and ci_high_col:
        ci_low = np.asarray(src[ci_low_col], dtype=float)
        ci_high = np.asarray(src[ci_high_col], dtype=float)
        ax.fill_between(t1, ci_low, ci_high, step="post", color="#BC3C29", alpha=0.18)
        if len(ci_high):
            max_vals.append(float(np.nanmax(ci_high)))

    if cif_competing is not None:
        src2 = cif_competing["cif_table"] if isinstance(cif_competing, dict) and "cif_table" in cif_competing else cif_competing
        t2_key = "time" if "time" in src2 else "timeline"
        y2_key = "cif_competing" if "cif_competing" in src2 else ("cif_primary" if "cif_primary" in src2 else "cif")
        t2 = np.asarray(src2[t2_key], dtype=float)
        y2 = np.asarray(src2[y2_key], dtype=float)
        label2 = cif_competing.get("label", "Competing Event") if isinstance(cif_competing, dict) else "Competing Event"
        ax.step(t2, y2, where="post", color="#0072B5", lw=1.8, ls="--", label=label2)
        if len(y2):
            max_vals.append(float(np.nanmax(y2)))
        ci2_high_col = "cif_competing_ci_upper" if "cif_competing_ci_upper" in src2 else ("ci_upper" if "ci_upper" in src2 else None)
        if ci2_high_col:
            ci2_high = np.asarray(src2[ci2_high_col], dtype=float)
            if len(ci2_high):
                max_vals.append(float(np.nanmax(ci2_high)))
    elif "cif_competing" in src:
        y2 = np.asarray(src["cif_competing"], dtype=float)
        ax.step(t1, y2, where="post", color="#0072B5", lw=1.8, ls="--", label="Competing Event")
        if len(y2):
            max_vals.append(float(np.nanmax(y2)))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel(y_label, fontweight="bold")
    ax.set_title(title, loc="left", fontweight="bold", pad=10)
    ax.set_ylim(0.0, min(1.0, max(0.25, max(max_vals) * 1.25)))
    ax.legend(frameon=False, loc="upper left")

    if ax_tbl is not None and risk_counts and time_ticks:
        ax_tbl.spines["top"].set_visible(False)
        ax_tbl.spines["right"].set_visible(False)
        ax_tbl.spines["left"].set_visible(False)
        ax_tbl.yaxis.set_ticks([])
        ax_tbl.set_xticks(time_ticks)
        for idx, (label, counts) in enumerate(risk_counts.items()):
            y_row = 1.0 - (idx + 0.5) / max(1, len(risk_counts))
            ax_tbl.text(time_ticks[0], y_row, f"{label}: ", ha="right", va="center", fontweight="bold", fontsize=8.5)
            for tx, cnt in zip(time_ticks, counts):
                ax_tbl.text(tx, y_row, str(cnt), ha="center", va="center", fontsize=8.5)
        ax_tbl.set_xlabel(x_label, fontweight="bold")
    else:
        ax.set_xlabel(x_label, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path
```

### 3. Multivariable / Subgroup / Meta-Analysis Forest Plot

- **Primary Recipe**: Use `fit_firth_logistic` in [02-regression-firth-ordinal.md](recipes/02-regression-firth-ordinal.md) (Recipe 1), or `run_meta_analysis` in [05-diagnostic-meta-power.md](recipes/05-diagnostic-meta-power.md) (Recipe 24).
- **Scale & Axis Guidance**: For ratio effect measures (OR / RR / HR), use `log_scale=True`, `null_value=1.0`, and ratio axis labels (e.g. `xlabel="Adjusted Odds Ratio (95% CI)"`). For continuous difference effect measures (MD / SMD), use `log_scale=False`, `null_value=0.0`, and difference axis labels (e.g. `xlabel="Mean Difference (95% CI)"` or `xlabel="Standardized Mean Difference (95% CI)"`).
- **Standalone Plot Template**:
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_standalone_forest(
    df: pd.DataFrame,
    label_col: str = "term",
    est_col: str = "estimate",
    ci_low_col: str = "ci_lower",
    ci_high_col: str = "ci_upper",
    null_value: float = 1.0,
    log_scale: bool = True,
    xlabel: str = "Adjusted Odds Ratio (95% CI)",
    title: str = "Multivariable Forest Plot",
    out_path: str = "forest_plot.png",
) -> str:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    work = df.iloc[::-1].reset_index(drop=True)
    if label_col not in work.columns:
        for cand in ("Variable", "study", "covariate", "term"):
            if cand in work.columns:
                label_col = cand
                break
    if est_col not in work.columns:
        for cand in ("OR", "odds_ratio", "hr", "effect_size", "estimate"):
            if cand in work.columns:
                est_col = cand
                break

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
- **Standalone Plot Template**:
```python
import matplotlib.pyplot as plt
import numpy as np


def plot_standalone_roc(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_val: float,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
    optimal_point: dict | None = None,
    title: str = "Receiver Operating Characteristic (ROC)",
    out_path: str = "roc_plot.png",
) -> str:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    fig, ax = plt.subplots(figsize=(5.5, 5.2), dpi=300)
    lbl = f"AUROC = {auc_val:.3f}" + (
        f" (95% CI, {ci_lower:.3f}–{ci_upper:.3f})" if ci_lower is not None and ci_upper is not None else ""
    )
    ax.plot(fpr, tpr, color="#0072B5", lw=2.2, label=lbl)
    ax.plot([0, 1], [0, 1], color="#94A3B8", ls="--", lw=1.2, label="Chance (0.50)")
    if optimal_point is not None:
        opt_fpr = float(optimal_point["fpr"])
        opt_tpr = float(optimal_point["tpr"])
        opt_cutoff = optimal_point.get("cutoff")
        opt_lbl = f"Optimal (cutoff = {opt_cutoff:.3f})" if opt_cutoff is not None else "Optimal Cutpoint"
        ax.scatter([opt_fpr], [opt_tpr], color="#BC3C29", s=45, zorder=5, label=opt_lbl)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("1 - Specificity (False Positive Rate)", fontweight="bold")
    ax.set_ylabel("Sensitivity (True Positive Rate)", fontweight="bold")
    ax.set_title(title, loc="left", fontweight="bold", pad=10)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path
```

### 5. Calibration Belt & Decision Curve Analysis (DCA)

- **Primary Recipe**: Use `evaluate_calibration` and `calculate_dca` in [05-diagnostic-meta-power.md](recipes/05-diagnostic-meta-power.md) (Recipe 4).
- **Standalone Plot Template**:
```python
import matplotlib.pyplot as plt
import pandas as pd


def plot_standalone_calibration_dca(
    calib_df: pd.DataFrame | dict,
    dca_df: pd.DataFrame | dict,
    out_path: str = "calibration_dca.png",
) -> str:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    if isinstance(calib_df, dict):
        if "calibration_bins" in calib_df:
            calib_df = pd.DataFrame(calib_df["calibration_bins"])
        else:
            calib_df = pd.DataFrame(calib_df)
    if isinstance(dca_df, dict):
        dca_df = pd.DataFrame(dca_df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=300)
    ax1.plot([0, 1], [0, 1], color="#94A3B8", ls="--", lw=1.2, label="Ideal")
    ax1.plot(calib_df["pred_mean"], calib_df["obs_rate"], "o-", color="#0072B5", lw=2.0, label="Observed")
    if "ci_lower" in calib_df.columns and "ci_upper" in calib_df.columns:
        yerr = [
            calib_df["obs_rate"] - calib_df["ci_lower"],
            calib_df["ci_upper"] - calib_df["obs_rate"],
        ]
        ax1.errorbar(
            calib_df["pred_mean"],
            calib_df["obs_rate"],
            yerr=yerr,
            fmt="none",
            ecolor="#0072B5",
            elinewidth=1.2,
            capsize=3,
            label="95% CI",
        )
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_xlabel("Predicted Probability", fontweight="bold")
    ax1.set_ylabel("Observed Event Frequency", fontweight="bold")
    ax1.set_title("A. Calibration Plot", loc="left", fontweight="bold")
    ax1.legend(frameon=False)

    if "strategy" in dca_df.columns:
        m_df = dca_df[dca_df["strategy"] == "Model"]
        all_df = dca_df[dca_df["strategy"] == "Treat All"]
        none_df = dca_df[dca_df["strategy"] == "Treat None"]
        if not m_df.empty:
            ax2.plot(m_df["threshold"], m_df["net_benefit"], color="#BC3C29", lw=2.0, label="Prediction Model")
        if not all_df.empty:
            ax2.plot(all_df["threshold"], all_df["net_benefit"], color="#64748B", ls="--", lw=1.4, label="Treat All")
        if not none_df.empty:
            ax2.plot(none_df["threshold"], none_df["net_benefit"], color="#1E293B", ls=":", lw=1.2, label="Treat None")
        else:
            ax2.axhline(0.0, color="#1E293B", ls=":", lw=1.2, label="Treat None")
    else:
        if "nb_model" in dca_df.columns:
            ax2.plot(dca_df["threshold"], dca_df["nb_model"], color="#BC3C29", lw=2.0, label="Prediction Model")
        if "nb_all" in dca_df.columns:
            ax2.plot(dca_df["threshold"], dca_df["nb_all"], color="#64748B", ls="--", lw=1.4, label="Treat All")
        ax2.axhline(0.0, color="#1E293B", ls=":", lw=1.2, label="Treat None")

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_xlabel("Threshold Probability", fontweight="bold")
    ax2.set_ylabel("Net Benefit", fontweight="bold")
    ax2.set_title("B. Decision Curve Analysis", loc="left", fontweight="bold")
    ax2.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path
```

### 6. Propensity Score Balance (Love Plot)

- **Primary Recipe**: Use `match_propensity_scores` in [04-causal-iptw-psm.md](recipes/04-causal-iptw-psm.md) (Recipe 6) which returns `balance_df` directly, or compute weighted SMDs via `calculate_weighted_smd` (Recipe 14) formatted into a balance table with columns `Variable`, `Pre-Match SMD`, and `Post-Match SMD` (or `smd_unadjusted`, `smd_adjusted`), plotting $|\text{SMD}|$ pre- and post-adjustment with a vertical threshold line at $|\text{SMD}| = 0.10$.
- **Standalone Plot Template**:
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_standalone_love(
    balance_df: pd.DataFrame,
    var_col: str = "Variable",
    unadj_col: str = "Pre-Match SMD",
    adj_col: str = "Post-Match SMD",
    out_path: str = "love_plot.png",
) -> str:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    work = balance_df.iloc[::-1].reset_index(drop=True)
    if var_col not in work.columns:
        for cand in ("Variable", "covariate", "term", "variable"):
            if cand in work.columns:
                var_col = cand
                break
    if unadj_col not in work.columns:
        for cand in ("Pre-Match SMD", "smd_unadjusted", "Unadjusted", "smd_pre"):
            if cand in work.columns:
                unadj_col = cand
                break
    if adj_col not in work.columns:
        for cand in ("Post-Match SMD", "smd_adjusted", "Adjusted", "smd_post", "smd_matched"):
            if cand in work.columns:
                adj_col = cand
                break
    y = np.arange(len(work))
    fig, ax = plt.subplots(figsize=(6.5, max(3.2, 0.4 * len(work) + 1.2)), dpi=300)
    ax.axvline(0.10, color="#BC3C29", ls="--", lw=1.2, label="Threshold (|SMD| = 0.10)")
    ax.plot(np.abs(work[unadj_col]), y, "o", color="#64748B", label="Unadjusted")
    ax.plot(np.abs(work[adj_col]), y, "s", color="#0072B5", label="Adjusted")
    ax.set_yticks(y)
    ax.set_yticklabels(work[var_col].astype(str))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Absolute Standardized Mean Difference (|SMD|)", fontweight="bold")
    ax.set_title("Covariate Balance (Love Plot)", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path
```

### 7. Bland-Altman Method Agreement Plot

- **Primary Recipe**: Use `calculate_bland_altman` in [05-diagnostic-meta-power.md](recipes/05-diagnostic-meta-power.md) (Recipe 5), marking mean bias and $\pm 1.96\text{ SD}$ Limits of Agreement (LoA) with 95% CI shading.
- **Standalone Plot Template**:
```python
import matplotlib.pyplot as plt
import numpy as np


def plot_standalone_bland_altman(
    m1: np.ndarray,
    m2: np.ndarray,
    ba_result: dict | None = None,
    title: str = "Bland-Altman Agreement Plot",
    out_path: str = "bland_altman.png",
) -> str:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    v1 = np.asarray(m1, dtype=float)
    v2 = np.asarray(m2, dtype=float)
    if len(v1) != len(v2):
        raise ValueError(f"Length mismatch: m1 ({len(v1)}) and m2 ({len(v2)}) must have identical lengths.")
    valid = np.isfinite(v1) & np.isfinite(v2)
    a = v1[valid]
    b = v2[valid]
    n = len(a)
    if n < 2:
        raise ValueError("Bland-Altman difference analysis requires at least 2 valid paired measurements.")
    means = 0.5 * (a + b)
    diffs = a - b

    if ba_result is not None:
        bias = float(ba_result["mean_diff"])
        loa_high = float(ba_result["upper_loa"])
        loa_low = float(ba_result["lower_loa"])
        ci_bias = ba_result.get("ci_mean_diff")
        ci_loa_high = ba_result.get("ci_upper_loa")
        ci_loa_low = ba_result.get("ci_lower_loa")
    else:
        from scipy.stats import t as t_dist

        bias = float(np.mean(diffs))
        sd = float(np.std(diffs, ddof=1))
        loa_low, loa_high = bias - 1.96 * sd, bias + 1.96 * sd
        se_mean = sd / np.sqrt(n)
        se_loa = float(np.sqrt((1.0 / n + (1.96**2) / (2.0 * (n - 1))) * (sd**2)))
        t_crit = float(t_dist.ppf(0.975, df=n - 1)) if n > 1 else 1.96
        ci_bias = (bias - t_crit * se_mean, bias + t_crit * se_mean)
        ci_loa_high = (loa_high - 1.96 * se_loa, loa_high + 1.96 * se_loa)
        ci_loa_low = (loa_low - 1.96 * se_loa, loa_low + 1.96 * se_loa)

    fig, ax = plt.subplots(figsize=(6.2, 4.5), dpi=300)
    ax.scatter(means, diffs, color="#0072B5", alpha=0.65, s=24)
    if ci_bias is not None:
        ax.axhspan(ci_bias[0], ci_bias[1], color="#1E293B", alpha=0.12, label="95% CI (Bias)")
    if ci_loa_high is not None:
        ax.axhspan(ci_loa_high[0], ci_loa_high[1], color="#BC3C29", alpha=0.10, label="95% CI (LoA)")
    if ci_loa_low is not None:
        ax.axhspan(ci_loa_low[0], ci_loa_low[1], color="#BC3C29", alpha=0.10)

    ax.axhline(bias, color="#1E293B", lw=1.8, label=f"Mean Bias = {bias:.2f}")
    ax.axhline(loa_high, color="#BC3C29", ls="--", lw=1.4, label=f"+1.96 SD = {loa_high:.2f}")
    ax.axhline(loa_low, color="#BC3C29", ls="--", lw=1.4, label=f"-1.96 SD = {loa_low:.2f}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Mean of Two Measurements", fontweight="bold")
    ax.set_ylabel("Difference (Method 1 - Method 2)", fontweight="bold")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path
```

### 8. Restricted Cubic Splines (RCS) Dose-Response Curve

- **Primary Recipe**: Use `fit_rcs_spline_model` in [02-regression-firth-ordinal.md](recipes/02-regression-firth-ordinal.md) (Recipe 22), plotting `contrast_df["estimate"]` and `ci_lower`/`ci_upper` across the continuous predictor range with annotated overall and non-linearity $P$-values.
- **Standalone Plot Template**:
```python
import matplotlib.pyplot as plt
import pandas as pd


def plot_standalone_rcs(
    contrast_df: pd.DataFrame,
    rcs_var: str | None = None,
    x_col: str | None = None,
    family: str = "binomial",
    p_overall: float | None = None,
    p_nonlinear: float | None = None,
    x_label: str = "Continuous Exposure",
    y_label: str | None = None,
    title: str = "Restricted Cubic Spline Dose-Response",
    out_path: str = "rcs_curve.png",
) -> str:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fam_clean = str(family).lower()
    if fam_clean not in ("binomial", "gaussian", "cox"):
        raise ValueError(
            f"Unsupported family '{family}'; must be one of 'binomial', 'gaussian', or 'cox'."
        )

    target_col = x_col or rcs_var
    if target_col is not None:
        if target_col not in contrast_df.columns:
            raise KeyError(
                f"Requested predictor column '{target_col}' not found in contrast_df columns: {list(contrast_df.columns)}"
            )
        col = target_col
    else:
        cand_cols = [
            c
            for c in contrast_df.columns
            if c not in ("estimate", "ci_lower", "ci_upper", "log_contrast", "se")
        ]
        if not cand_cols:
            raise KeyError(
                f"Could not infer predictor column from contrast_df columns: {list(contrast_df.columns)}"
            )
        col = cand_cols[0]

    x = contrast_df[col].to_numpy(dtype=float)
    y = contrast_df["estimate"].to_numpy(dtype=float)
    lo = contrast_df["ci_lower"].to_numpy(dtype=float)
    hi = contrast_df["ci_upper"].to_numpy(dtype=float)

    is_gaussian = fam_clean == "gaussian"
    null_val = 0.0 if is_gaussian else 1.0
    if y_label is None:
        if is_gaussian:
            y_label = "Mean Difference (95% CI)"
        elif fam_clean == "cox":
            y_label = "Adjusted Hazard Ratio (95% CI)"
        else:
            y_label = "Adjusted Odds Ratio (95% CI)"

    fig, ax = plt.subplots(figsize=(6.4, 4.5), dpi=300)

    ax.axhline(null_val, color="#94A3B8", ls="--", lw=1.1)
    ax.plot(x, y, color="#0072B5", lw=2.2, label="Spline Point Estimate")
    ax.fill_between(x, lo, hi, color="#0072B5", alpha=0.18, label="95% CI")
    if p_overall is not None and p_nonlinear is not None:
        ax.text(
            0.04,
            0.92,
            f"P-overall = {p_overall:.3f}\nP-non-linear = {p_nonlinear:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(x_label, fontweight="bold")
    ax.set_ylabel(y_label, fontweight="bold")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path
```

### 9. Cox Proportional Hazards Schoenfeld Residuals Diagnostic Plot

- **Primary Recipe**: Use `schoenfeld_diagnostics` and `fit_survival_analysis_suite` in [03-survival-competing-risks.md](recipes/03-survival-competing-risks.md) (Recipe 11).
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
    plt.rcParams["ps.fonttype"] = 42
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

When generating interactive HTML reports with Dual / Toggle View in cloud sandboxes, use the pure-Python dictionary builders in [report-builder.md §B1](report-builder.md#b1-standalone-pure-python-plotly-spec-generators-zero-dependencies):
- `build_plotly_roc_spec`
- `build_plotly_km_spec`
- `build_plotly_forest_spec`
- `build_plotly_calibration_spec`
- `build_plotly_dca_spec`
- `build_plotly_love_spec`
- `build_plotly_bland_altman_spec`

Pass the resulting dictionary list into `generate_html_report(..., plotly_figs=[...])` along with the static PNG paths in `figure_paths=[...]`.

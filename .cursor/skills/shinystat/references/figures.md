# Clinical Figure Library Reference

`medstat.figures` provides publication-grade, reproducible figure generation for clinical trials, observational studies, and diagnostic biomarker validation.

---

## 1. Architectural Principles & Standards

### Publication Specifications
- **Resolution**: 300 DPI strictly enforced across all exported images.
- **Dimensions**:
  - `docx` target: Width 6.5 in (fits standard 8.5" x 11" page with 1.0" margins).
  - `pptx` target: Width 11.5–12.0 in (fits 16:9 widescreen slides).
  - `web` target: Max width 900 px with responsive SVG/high-DPI PNG rendering.
- **Colorblind-Safe Palettes**: Okabe-Ito / Tol palettes (`#0072B2`, `#D55E00`, `#009E73`, `#CC79A7`, `#E69F00`).
- **Thai Typography Support**: Built-in fallback chain (`Sarabun`, `Thonburi`, `Sukhumvit Set`, `Arial`) ensures clean Thai text rendering without square-box glyph errors.
- **Zero-PHI Guarantee**: Figure axis labels, legends, and hover texts strictly prohibit raw patient identifiers (HN, Thai citizen IDs, patient names).

### Figure Return Contract
Every plotting function returns a `FigureResult` dataclass:
```python
@dataclass
class FigureResult:
    png_path: str  # Absolute or relative path to saved 300 DPI PNG
    alt_text: str  # Accessibility description for web and markdown
    caption: str  # Publication-ready figure caption with statistical notes
    source_df: pd.DataFrame  # Underlying numerical values plotted
    csv_path: str  # Path to exported raw data CSV for auditability
```

---

## 2. Available Plot Recipes

### 1. Forest Plot (`plot_forest`)
Visualizes effect estimates (Odds Ratios, Hazard Ratios, Risk Ratios) from multivariable models or meta-analyses.
- **Features**: Logarithmic scale, reference vertical dashed line at 1.0, point estimates proportional to sample weight, annotated values and 95% CIs on right margin.
- **Usage**:
```python
from medstat.figures.forest import plot_forest

res = plot_forest(
    model_df,  # columns: ['term', 'estimate', 'ci_lower', 'ci_upper', 'p_value']
    scale="OR",
    title="Multivariable Predictors of 30-Day Mortality",
    out_path="figures/forest_mortality.png",
    target="docx",
)
```

### 2. Kaplan-Meier Survival Curve (`plot_kaplan_meier`)
Produces stratified survival curves with aligned Numbers at Risk table.
- **Features**: Shaded 95% confidence bands (Hall-Wellner / log-log), tick marks for censored events, aligned risk table positioned directly below the time axis, log-rank test p-value.
- **Usage**:
```python
from medstat.figures.survival import plot_kaplan_meier

res = plot_kaplan_meier(
    km_data,  # Dict[group_name -> {timeline, survival, ci_lower, ci_upper, censored_times}]
    risk_table,  # DataFrame with columns: ['Group', 't=0.0', 't=30.0', 't=90.0', 't=180.0', 't=365.0']
    log_rank_p=0.038,
    title="Kaplan-Meier Survival by Treatment Arm",
    out_path="figures/km_survival.png",
    target="docx",
)
```

### 3. ROC Curve & DeLong CI (`plot_roc_curve`)
Evaluates diagnostic discrimination accuracy.
- **Features**: Empirical True Positive vs False Positive rates, 45-degree chance diagonal, optimal cutoff marker via Youden's J index, DeLong 95% CI label, optional paired overlay comparison.
- **Usage**:
```python
from medstat.figures.roc import plot_roc_curve

res = plot_roc_curve(
    roc_data={
        "fpr": fpr_arr,
        "tpr": tpr_arr,
        "auc": 0.865,
        "ci_lower": 0.812,
        "ci_upper": 0.918,
        "optimal_point": {"cutoff": 5.0, "fpr": 0.15, "tpr": 0.75},
        "name": "Biomarker Model",
    },
    out_path="figures/roc_biomarker.png",
    target="docx",
)
```

### 4. Calibration Curve (`plot_calibration`)
Validates agreement between predicted risk probabilities and observed event frequencies.
- **Features**: Decile grouping, Wilson score 95% error bars, LOESS smoothed curve, 45-degree ideal reference, Brier score and Integrated Calibration Index (ICI) annotation.
- **Usage**:
```python
from medstat.figures.calibration import plot_calibration

res = plot_calibration(
    calib_bins={
        "pred_mean": pred_mean_arr,
        "obs_rate": obs_rate_arr,
        "counts": counts_arr,
        "ci_lower": ci_lower_arr,
        "ci_upper": ci_upper_arr,
    },
    brier_score=0.124,
    ici=0.021,
    out_path="figures/calibration.png",
    target="docx",
)
```

### 5. Decision Curve Analysis (`plot_dca`)
Quantifies clinical net benefit across decision threshold probabilities.
- **Features**: Net benefit comparison against "Treat All" and "Treat None" strategies across threshold ranges (1% to 50%).
- **Usage**:
```python
from medstat.figures.dca import plot_dca

res = plot_dca(
    dca_data=dca_df,  # Long-form DataFrame with columns: ['strategy', 'threshold', 'net_benefit']
    out_path="figures/dca_net_benefit.png",
    target="docx",
)
```

### 6. Bland-Altman Agreement Plot (`plot_bland_altman`)
Assesses agreement between two continuous clinical measurement methods or devices.
- **Features**: Difference vs average, horizontal mean bias line, upper and lower 95% Limits of Agreement (LoA), Bland & Altman (1999) confidence intervals.
- **Usage**:
```python
from medstat.figures.agreement import plot_bland_altman

res = plot_bland_altman(
    ba_data={
        "means": means_arr,
        "diffs": diffs_arr,
        "mean_diff": -0.85,
        "loa_upper": 4.12,
        "loa_lower": -5.82,
        "ci_mean_diff": (-1.2, -0.5),
        "ci_loa_upper": (3.5, 4.7),
        "ci_loa_lower": (-6.4, -5.2),
    },
    units="mmHg",
    title="Bland-Altman Agreement: Non-invasive vs Arterial Line",
    out_path="figures/bland_altman_bp.png",
    target="docx",
)
```

### 7. Love Plot (`plot_love`)
Visualizes covariate balance before and after Propensity Score Matching (PSM) or Inverse Probability of Treatment Weighting (IPTW).
- **Features**: Absolute Standardized Mean Differences (SMD), Austin (2009) 0.10 threshold line, distinct markers supporting 2-way (Unadjusted vs Matched or Weighted) and 3-way balance comparison (Unadjusted vs Matched vs IPTW).
- **Column Contract**:
  - **2-Way Balance (PSM)**: Requires `'Variable'` (or `'Covariate'`), `'Pre_Match_SMD'` (or `'Unadjusted_SMD'`), and `'Post_Match_SMD'` (or `'Matched_SMD'`).
  - **2-Way Balance (IPTW)**: Requires `'Variable'` (or `'Covariate'`), `'Pre_Match_SMD'` (or `'Unadjusted_SMD'`), and `'Weighted_SMD'` (or `'Post_Weight_SMD'`).
  - **3-Way Balance (PSM vs IPTW)**: Requires `'Variable'`, `'Pre_Match_SMD'` (or `'Unadjusted_SMD'`), `'Post_Match_SMD'` (or `'Matched_SMD'`), and `'Weighted_SMD'` (or `'Post_Weight_SMD'`).
- **Usage**:
```python
from medstat.figures.balance import plot_love

# 2-way or 3-way balance DataFrame
res = plot_love(
    love_data=love_df,  # e.g., Columns: ['Variable', 'Unadjusted_SMD', 'Matched_SMD', 'Weighted_SMD']
    threshold=0.10,
    out_path="figures/love_plot.png",
    target="docx",
)
```

### 8. Participant Retention Flowchart (`plot_retention_flow`)
Generates STROBE / CONSORT participant disposition flowcharts.
- **Features**: Assessed for eligibility, excluded with breakdown, randomized / assigned, lost to follow-up, analyzed cohorts.
- **Usage**:
```python
from medstat.figures.retention import plot_retention_flow

res = plot_retention_flow(
    retention_data={
        "n_initial": 1250,
        "n_excluded": 165,
        "n_analyzed": 1085,
        "exclusion_reasons": ["Prior cardiac surgery: 120", "Missing key lab: 45"],
    },
    title="Participant Retention Flow Diagram (STROBE / CONSORT)",
    out_path="figures/strobe_retention_flow.png",
    target="docx",
)
```

### 9. Statistical Diagnostics (`plot_missingness_map`, `plot_schoenfeld_residuals`, `plot_mice_diagnostics`)
Automates pre-modeling and post-modeling diagnostic validation plots.

### 10. Cumulative Incidence Function (`plot_cif`)
Visualizes non-parametric Aalen-Johansen CIF curves alongside Naive $1 - \text{Kaplan-Meier}$ overestimation audit.
- **Features**: Event of interest CIF with Greenwood 95% confidence bands, competing risk CIF curves, and Naive $1 - \text{KM}$ curve illustrating bias when competing events are treated as independent censoring.
- **Column Contract**: `cif_table` DataFrame must contain `'time'`, `'cif_primary'`, and `'cif_competing'` columns (with optional `'cif_primary_ci_lower'`, `'cif_primary_ci_upper'`, `'cif_competing_ci_lower'`, `'cif_competing_ci_upper'`, `'naive_1_minus_km'`).
- **Usage**:
```python
from medstat.figures.survival import plot_cif

res = plot_cif(
    cif_data=cif_df,  # Columns: ['time', 'cif_primary', 'cif_competing', 'naive_1_minus_km']
    title="Cumulative Incidence of Primary Event vs Competing Risks",
    out_path="figures/cif_curves.png",
    target="docx",
)
```

### 9. Risk Reclassification Plot (`plot_reclassification`)
Scatter plot comparing predicted probabilities of baseline vs updated models, color-coded by event status ($Y=1$ events in red, $Y=0$ non-events in blue), with equality line and threshold grid.
- **Usage**:
```python
from medstat.figures.roc import plot_reclassification

res = plot_reclassification(
    reclass_data,  # Output dict from compute_reclassification_metrics
    title="Risk Reclassification: Model 1 vs Model 2",
    out_path="figures/reclassification.png",
)
```

### 10. Time-Dependent ROC Trajectory & Multi-Horizon Curves (`plot_time_dependent_roc`)
Visualizes longitudinal discrimination trajectories ($\text{AUROC}(t)$ with 95% CI bands) or multi-horizon ROC curves.
- **Usage**:
```python
from medstat.figures.roc import plot_time_dependent_roc

# AUROC trajectory over time
res = plot_time_dependent_roc(
    td_data,  # Output dict from compute_time_dependent_roc
    mode="auroc_trajectory",
    title="Time-Dependent Discrimination Trajectory",
    out_path="figures/td_roc_trajectory.png",
)

# Multi-horizon ROC curves
res_curves = plot_time_dependent_roc(
    td_data,
    mode="curves",
    title="Time-Dependent ROC Curves across Follow-up Horizons",
    out_path="figures/td_roc_curves.png",
)
```

### 11. Recalibration Comparison Plot (`plot_recalibration_comparison`)
Compares pre-recalibration vs post-recalibration calibration curves against the ideal 45-degree reference line, annotated with intercept, slope, and Brier score.
- **Usage**:
```python
from medstat.figures.calibration import plot_recalibration_comparison

res = plot_recalibration_comparison(
    recal_res,  # Output dict from recalibrate_prediction_model
    title="Prediction Model Recalibration",
    out_path="figures/recalibration.png",
)
```

### 12. Bootstrap Optimism Plot (`plot_bootstrap_optimism`)
Grouped horizontal bar chart of **Apparent vs Optimism-Corrected** metrics (including `1 - Brier Score` where applicable) and the uniform shrinkage factor callout.
- **Usage**:
```python
from medstat.figures.calibration import plot_bootstrap_optimism

res = plot_bootstrap_optimism(
    val_res,  # Output dict from bootstrap_internal_validation
    title="TRIPOD+AI Bootstrap Optimism Validation",
    out_path="figures/bootstrap_optimism.png",
)
```

---

## 3. Interactive HTML Figures (`medstat.figures.html`)

For interactive HTML reports, use the mirror submodule `medstat.figures.html`. It generates a `DualFigureResult` (subclass of `FigureResult`) containing both a publication-standard 300 DPI Matplotlib figure and a Plotly interactive figure:

- **`plot_dual_roc_curve(roc_data, ...)`**: Empirical ROC curve with interactive threshold/sensitivity/specificity tooltips and chance diagonal.
- **`plot_dual_reclassification(reclass_data, ...)`**: Interactive scatter plot of baseline vs updated risks with hover tooltips and event/non-event toggles.
- **`plot_dual_time_dependent_roc(td_data, ...)`**: Interactive longitudinal AUROC trajectory with CI bounds or multi-horizon ROC curves.
- **`plot_dual_recalibration_comparison(recal_res, ...)`**: Interactive pre vs post recalibration curve comparison with hover tooltips and calibration slope/intercept metrics.
- **`plot_dual_bootstrap_optimism(val_res, ...)`**: Interactive grouped horizontal bar chart of **Apparent vs Optimism-Corrected** metrics (including `1 - Brier Score` where applicable) and the uniform shrinkage factor callout.
- **`plot_dual_kaplan_meier(km_curves, risk_table, ...)`**: Step-function survival curves with 95% CI bands and hover tooltips.
- **`plot_dual_cif(cif_data, ...)`**: Non-parametric Cumulative Incidence Function (Aalen-Johansen) with Naive $1 - \text{KM}$ audit and interactive hover.
- **`plot_dual_calibration(calib_bins, ...)`**: Decile calibration points with 95% Wilson CI error bars and Brier/ICI annotations.
- **`plot_dual_dca(dca_data, ...)`**: Vickers Decision Curve Analysis comparing model net benefit against Treat All and Treat None.
- **`plot_dual_bland_altman(ba_data, ...)`**: Paired difference plot with mean bias, 95% Limits of Agreement, and confidence intervals.
- **`plot_dual_forest(data, ...)`**: Forest plot with log-scale axis and interactive point estimates.
- **`plot_dual_love(love_data, ...)`**: Austin 2009 Love plot comparing 2-way and 3-way covariate balance relative to 0.10 SMD threshold.

```python
from medstat.figures.html import plot_dual_roc_curve, DualFigureResult

dual_fig = plot_dual_roc_curve(roc_data, out_path="figures/roc_curve.png")
# Pass directly to ReportDocument; render_html will automatically render the Dual / Toggle View
doc.add_figure(dual_fig)
```


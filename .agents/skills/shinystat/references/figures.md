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
    png_path: str         # Absolute or relative path to saved 300 DPI PNG
    alt_text: str         # Accessibility description for web and markdown
    caption: str          # Publication-ready figure caption with statistical notes
    source_df: pd.DataFrame  # Underlying numerical values plotted
    csv_path: str         # Path to exported raw data CSV for auditability
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
    estimate_type="OR",
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
    km_data,       # Dict[group_name -> {timeline, survival, ci_lower, ci_upper, censored_times}]
    risk_table,    # DataFrame with columns: [Group, 0, 30, 90, 180, 365]
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
        "thresholds": thresh_arr,
        "auc": 0.865,
        "ci_lower": 0.812,
        "ci_upper": 0.918,
        "optimal_point": {"cutoff": 5.0, "fpr": 0.15, "tpr": 0.75},
        "model_name": "Biomarker Model",
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
    calibration_data={
        "mean_pred": mean_pred_bins,
        "obs_freq": obs_freq_bins,
        "bin_counts": counts,
        "brier_score": 0.124,
        "ici": 0.021,
    },
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
    dca_data=dca_results,  # thresholds, net_benefit_model, net_benefit_all, net_benefit_none
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
    method_a=df["device_a"].values,
    method_b=df["device_b"].values,
    method_a_name="Device A (Non-invasive)",
    method_b_name="Device B (Arterial Line)",
    unit="mmHg",
    out_path="figures/bland_altman_bp.png",
    target="docx",
)
```

### 7. Love Plot (`plot_love`)
Visualizes covariate balance before and after Propensity Score Matching (PSM).
- **Features**: Absolute Standardized Mean Differences (SMD), Austin (2009) 0.10 threshold line, distinct markers for Pre-match vs Post-match balance.
- **Usage**:
```python
from medstat.figures.balance import plot_love

res = plot_love(
    love_plot_data,  # DataFrame: [Variable, SMD_Unmatched, SMD_Matched]
    out_path="figures/love_plot_psm.png",
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
    flow_data={
        "assessed": 1250,
        "excluded": {"Prior cardiac surgery": 120, "Missing key lab": 45},
        "allocated_arm_a": 542,
        "allocated_arm_b": 543,
        "analyzed_arm_a": 540,
        "analyzed_arm_b": 541,
    },
    out_path="figures/strobe_retention_flow.png",
    target="docx",
)
```

### 9. Statistical Diagnostics (`plot_missingness_map`, `plot_schoenfeld_residuals`, `plot_mice_diagnostics`)
Automates pre-modeling and post-modeling diagnostic validation plots.

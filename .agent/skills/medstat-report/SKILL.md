---
name: medstat-report
description: Publication-grade clinical report rendering, NEJM/JAMA/APA 7 styling, automated Methods and Results narrative generation, and reporting guideline audits (STROBE, CONSORT, TRIPOD). Use when compiling clinical study findings into publication-ready tables, generating compliant statistical methods narratives, formatting survival/logistic/diagnostic tables for medical journal submission, or auditing reporting checklist compliance.
---

# medstat-report: Publication HTML Tables, Methods Narratives & Guidelines

Dissemination engine for rendering publication-quality tables conforming to top medical journal standards (NEJM, JAMA, APA 7), synthesizing compliant statistical methods text, and generating reporting checklists.

## Core Rules

1. **Strict Journal Typography**: Enforce the exact rules of the target journal (NEJM double top border and symbol footnotes; JAMA minimal horizontal rules, decimal precision, and exact p-values without leading zeros; APA 7 italicized statistics).
2. **Automated Methods Narrative**: When `--narrative` is specified, pair quantitative tables with an automated Methods text describing model type, confounder selection, missing data handling, and significance criteria.
3. **Guideline Compliance**: Accompany observational studies with STROBE audits, randomized trials with CONSORT, and prediction models with TRIPOD (or TRIPOD+AI where applicable).
4. **Clinical Interpretation & Anti-Hallucination Invariants**:
   - *Absence of Evidence*: Never report $P > 0.05$ as "demonstrating no difference" or "proving equivalence." State: "insufficient evidence to reject the null hypothesis" and discuss the 95% CI width.
   - *Odds Ratio vs Risk*: The degree to which an Odds Ratio diverges from Relative Risk depends on both outcome incidence and effect size (divergence grows as incidence and effect size increase). Estimates must be explicitly labeled as Odds Ratios without loose substitution of "risk" or "relative risk".
   - *Uncertainty-First ICC Reporting*: For Intraclass Correlation Coefficients (ICC), report the 95% confidence interval and its spanning clinical reliability tier (Koo & Li 2016) rather than interpreting point estimates in isolation.
   - *Cumulative Odds & GEE*: For ordinal outcomes, report cumulative odds ratios clearly stating the direction (e.g., odds of higher vs. lower categories). For clustered data using GEE, report marginal effects with robust sandwich standard errors, alongside cluster ICC and Design Effect (DEFF).
5. **Calibration Reporting Context**: When calibration data is present, the narrative distinguishes in-sample apparent estimates (where calibration slope is not reported for in-sample apparent probability estimates) from external validation calibration (where Brier score, calibration slope/intercept, and ICI are fully reported). This distinction prevents misleading calibration claims from non-cross-validated model predictions.

## Execution Sequence

```
[1. INGEST RESULTS] ──▶ [2. CHOOSE JOURNAL STYLE] ──▶ [3. GENERATE NARRATIVE] ──▶ [4. CHECKLIST AUDIT]
```

### Step 1: Render Publication-Grade HTML Tables

Convert statistical model outputs or meta-analysis results into publication-ready HTML tables:

```bash
# NEJM Journal Style (New England Journal of Medicine)
medstat report --results <model_results.json> \
  --style nejm \
  --output table_nejm.html

# JAMA Journal Style (Journal of the American Medical Association)
medstat report --results <model_results.json> \
  --style jama \
  --output table_jama.html

# APA 7th Edition Style
medstat report --results <model_results.json> \
  --style apa7 \
  --output table_apa7.html
```

- **Output Invariants**:
  - Point estimates (OR, HR, RR, MD) formatted with 95% confidence intervals: e.g. `1.45 (95% CI, 1.12–1.88)`.
  - P-values formatted per journal convention (JAMA: $P = .04$, $P < .001$; NEJM: $P = 0.04$, $P < 0.001$).
  - Consult [references/journal-styles.md](references/journal-styles.md) for detailed typographical rules.

### Step 2: Synthesize Statistical Methods & Results Narratives

Generate automated manuscript-ready text from model results or pre-registered SAP:

```bash
# Generate statistical methods and findings narrative
medstat report --results <model_results.json> \
  --narrative \
  --output narrative_report.html
```

- Synthesizes descriptive statistics, model formulation, covariate adjustment, missing data handling strategy, and effect size interpretations with exact 95% CIs.
- Pre-formats narrative paragraphs for copy-paste into clinical manuscript drafts.

### Step 3: Audit Reporting Guideline Checklists

Generate structured compliance checklists across major biomedical reporting guidelines:

```bash
# STROBE Checklist (Observational Studies: Cohort, Case-Control, Cross-Sectional)
medstat report --checklist strobe --output strobe_checklist.md

# CONSORT Checklist (Randomized Controlled Trials)
medstat report --checklist consort --output consort_checklist.md

# TRIPOD Checklist (Clinical Prediction Models)
medstat report --checklist tripod --output tripod_checklist.md
```

- Each checklist audits required reporting items: participant flow, eligibility criteria, missingness handling, bias mitigation, confounder adjustments, and sensitivity analysis.

---

## Adaptive Python Scripting Protocol (ปรับแต่งสคริปต์รายงานผลมาตรฐานวารสารและการจัดตารางตามข้อมูลจริง)

> **Core Philosophy**: Never execute rigid canned scripts that make naive assumptions about file structure. The agent is empowered with full autonomy to write, adapt, and run Python scripts (`scratch/report.py`) tailored to formatting clinical study findings, survival metrics, adjusted odds ratios, or rater reliability statistics into publication-ready tables (NEJM, JAMA, APA 7) and automated methods narratives.
>
> 🔒 **Subprocess & Script Execution Safety**:
> When generating and running reporting scripts (`scratch/report.py`), enforce execution controls: disable shell/subprocess access, limit file reads strictly to the designated dataset and referenced prototype/core modules, limit file writes strictly to scratch and designated output paths, and ensure no access to credentials or environment secrets. Require explicit user confirmation if the runtime cannot enforce these sandbox controls.
>
> ⚠️ **Mandatory Directive — ต้องดู Script ต้นแบบประกอบเสมอ (Review Prototype Scripts First)**:
> แม้จะให้อิสระ Agent ในการเขียนและปรับ Python Script เองตามสภาพข้อมูลจริง แต่ **Agent ต้องเปิดดูและอ้างอิงสคริปต์ต้นแบบ (Prototype Scripts)** หรือศึกษาการคำนวณในโมดูลแกนกลาง `src/medstat/reporting/` เสมอ เพื่อยึดมาตรฐานความถูกต้องทางชีวสถิติการแพทย์:
> - **การจัดรูปแบบตารางมาตรฐานวารสาร (NEJM / JAMA / APA 7)**: ดูโครงสร้าง HTML table 3 เส้นนอน (three horizontal rules: top border, mid header border, bottom table border) และไม่มีเส้นแนวตั้ง (no vertical borders) จาก `src/medstat/reporting/tables.py`
> - **การสังเคราะห์ Methods & Results Narrative**: ดูการเรียบเรียงวรรคข้อความอัตโนมัติจาก `src/medstat/reporting/narrative.py`
> - **การตรวจสอบ Reporting Guidelines (STROBE, CONSORT, TRIPOD)**: ดูการสร้าง compliance checklist จาก `src/medstat/reporting/checklists.py`
>
> **วงจรการทำงานของ Agent**:
> `[1. นำผลลัพธ์แบบจำลองทางสถิติมาจัดกลุ่ม] ──▶ [2. ดูสคริปต์ต้นแบบเพื่อยึดหลักวารสาร] ──▶ [3. สร้างตาราง publication และรายงาน]`

### Master Prototype Script for Publication Reporting (สคริปต์ต้นแบบมาตรฐาน)

Agent ควรนำโครงสร้างและฟังก์ชันของสคริปต์ต้นแบบนี้ไปปรับแต่งลงใน workspace (เช่น `scratch/report.py`) เพื่อจัดรูปแบบตารางและเนื้อหารายงาน:

```python
import html
import numpy as np
import pandas as pd

# 1. ORGANIZE MODEL ESTIMATES & CLINICAL METRICS
model_records = [
    {"Variable": "Treatment Arm (Drug B vs Drug A)", "Estimate": "1.85", "CI": "(1.24 - 2.76)", "p_value": "0.003"},
    {"Variable": "Age (per 10-year increase)", "Estimate": "1.12", "CI": "(1.02 - 1.23)", "p_value": "0.018"},
    {"Variable": "Female Sex", "Estimate": "0.88", "CI": "(0.65 - 1.19)", "p_value": "0.412"},
    {"Variable": "Baseline SBP >= 140 mmHg", "Estimate": "1.45", "CI": "(1.08 - 1.95)", "p_value": "0.014"},
]

# 2. RENDER NEJM / JAMA / APA PUBLICATION HTML TABLE

def format_p_value(p_val_str, style="NEJM"):
    if p_val_str is None or pd.isna(p_val_str) or str(p_val_str).strip() in ("NA", "—", "-", "", "nan", "NaN"):
        return "—"
    if isinstance(p_val_str, (bool, np.bool_)):
        raise ValueError(f"p-value cannot be a boolean (got {p_val_str}).")
    try:
        p = float(p_val_str)
        if not np.isfinite(p) or not (0.0 <= p <= 1.0):
            raise ValueError(f"p-value must fall within [0, 1] (got {p}).")
        if style.upper() in ("JAMA", "APA", "APA7"):  # JAMA & APA 7: no leading zero (p cannot exceed 1)
            if p < 0.001:
                return "<.001"
            elif p < 0.01:
                return f"{p:.3f}".lstrip("0")
            elif p > 0.99:
                return ">.99"
            else:
                return f"{p:.2f}".lstrip("0")
        else:  # NEJM
            if p < 0.001:
                return "<0.001"
            elif p < 0.01:
                return f"{p:.3f}"
            elif p > 0.99:
                return ">0.99"
            else:
                return f"{p:.2f}"
    except (ValueError, TypeError) as e:
        if "must fall within" in str(e):
            raise
        return "—"


def render_publication_html_table(records, style="NEJM", title="Table 2. Multivariable Logistic Regression Analysis", adjustment_vars=None, ci_method="profile likelihood", measure_name="Odds Ratio"):
    style_upper = style.upper()
    if style_upper == "NEJM":
        top_border = "border-top: 3px double #000;"
        p_header = "P Value"
    elif style_upper in ("JAMA", "APA", "APA7"):
        top_border = "border-top: 1px solid #000;"
        p_header = "<em>P</em> Value"
    else:
        raise ValueError(f"Unsupported table style: {style}. Supported styles are 'NEJM', 'JAMA', 'APA', 'APA7'.")

    adj_prefix = "Adjusted " if adjustment_vars else "Unadjusted "
    est_header = f"{adj_prefix}{html.escape(str(measure_name))}"
    title_escaped = html.escape(str(title))
    
    measure_plural = f"{measure_name}s" if not str(measure_name).endswith("s") else str(measure_name)
    if adjustment_vars:
        covar_str = ", ".join(html.escape(str(v)) for v in adjustment_vars)
        footnote_text = f"* {html.escape(measure_plural)} were adjusted for {covar_str}. Confidence intervals are {html.escape(str(ci_method))}-based."
    else:
        footnote_text = f"* Unadjusted {html.escape(measure_plural.lower())}. Confidence intervals are {html.escape(str(ci_method))}-based."
    
    html_out = f"""
    <div style="font-family: 'Times New Roman', Times, serif; max-width: 800px; margin: 20px auto;">
      <h3 style="margin-bottom: 8px; font-weight: bold;">{title_escaped}</h3>
      <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 14px;">
        <thead>
          <tr style="{top_border} border-bottom: 1px solid #000;">
            <th style="padding: 6px 8px;">Characteristic / Variable</th>
            <th style="padding: 6px 8px; text-align: right;">{est_header}</th>
            <th style="padding: 6px 8px; text-align: right;">95% Confidence Interval</th>
            <th style="padding: 6px 8px; text-align: right;">{p_header}</th>
          </tr>
        </thead>
        <tbody>
    """
    for r in records:
        p_str = format_p_value(r.get('p_value', ''), style=style)
        var_esc = html.escape(str(r.get('Variable', '')))
        est_esc = html.escape(str(r.get('Estimate', '')))
        ci_esc = html.escape(str(r.get('CI', '')))
        p_esc = html.escape(str(p_str))
        html_out += f"""
          <tr>
            <td style="padding: 6px 8px;">{var_esc}</td>
            <td style="padding: 6px 8px; text-align: right;">{est_esc}</td>
            <td style="padding: 6px 8px; text-align: right;">{ci_esc}</td>
            <td style="padding: 6px 8px; text-align: right;">{p_esc}</td>
          </tr>
        """
    html_out += f"""
        </tbody>
        <tfoot>
          <tr style="border-bottom: 2px solid #000;">
            <td colspan="4" style="padding: 8px 4px; font-size: 12px; color: #333;">
              {footnote_text}
            </td>
          </tr>
        </tfoot>
      </table>
    </div>
    """
    return html_out

# 3. GENERATE STATISTICAL METHODS NARRATIVE FROM ACTUAL ANALYSIS METADATA
def generate_methods_narrative(
    study_design="Retrospective Cohort",
    primary_outcome="30-day Mortality",
    model_type="Multivariable logistic regression",
    confounders=None,
    missing_data_strategy="complete-case analysis",
    guideline="STROBE",
    tests=None,
    two_sided=True,
    alpha=0.05,
    calibration_provenance=None,  # None (not assessed), "apparent" (in-sample), or "external" (independent cohort)
    calibration_metrics=None,  # e.g. {"slope": 0.94, "intercept": -0.03, "brier": 0.112, "ici": 0.021}
    fit_metadata=None,  # Optional dict, e.g. {"tautological_slope": True} when slope is 1.0 by construction
):
    if confounders is None:
        raise ValueError("confounders list must be explicitly provided from actual analysis metadata (do not use arbitrary defaults)")
    if tests is None:
        tests = "Welch's t-test or Mann-Whitney U test for continuous variables and Pearson Chi-Square or Fisher's exact test for categorical variables"

    confounder_str = ", ".join(confounders)
    sig_clause = ""
    if two_sided is not None and alpha is not None:
        sided_str = "two-sided" if two_sided else "one-sided"
        sig_clause = f"All tests were {sided_str}, with p < {alpha} considered statistically significant. "
    elif alpha is not None:
        sig_clause = f"Statistical tests used a significance threshold of p < {alpha}. "

    if confounders:
        adj_clause = f"adjusting for prespecified confounders ({', '.join(confounders)})"
    else:
        adj_clause = "without covariate adjustment (unadjusted model)"

    calib_clause = ""
    if calibration_provenance is not None:
        if calibration_provenance not in ("apparent", "external"):
            raise ValueError("calibration_provenance must be 'apparent', 'external', or None.")
        if not calibration_metrics:
            raise ValueError("calibration_metrics must be provided when calibration_provenance is set.")
        for k, v in calibration_metrics.items():
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.number)) or not np.isfinite(v):
                raise ValueError(f"Calibration metric '{k}' must be a finite numeric value (got {v}).")
            if k.lower() in ("brier", "ici") and not 0.0 <= float(v) <= 1.0:
                raise ValueError(f"Calibration metric '{k}' must lie within [0, 1] (got {v}).")
        if calibration_provenance == "apparent":
            # For apparent estimates, calibration slope is excluded per TRIPOD / core reporting rules
            # (apparent slope is 1.0 / tautological and uninformative in-sample)
            filtered_metrics = {k: v for k, v in calibration_metrics.items() if k.lower() != "slope"}
            if not filtered_metrics:
                raise ValueError("Apparent validation requires at least one non-slope calibration metric (e.g., Brier score).")
        else:
            required_metrics = {"brier", "slope", "intercept", "ici"}
            missing = required_metrics - {k.lower() for k in calibration_metrics}
            if missing:
                raise ValueError(
                    f"External validation requires calibration metrics {sorted(required_metrics)}; missing: {sorted(missing)}."
                )
            filtered_metrics = calibration_metrics
        labels = {
            "slope": "calibration slope",
            "intercept": "calibration intercept",
            "brier": "Brier score",
            "ici": "ICI",
        }
        metric_str = ", ".join(f"{labels.get(k.lower(), k)} {v:.3f}" for k, v in filtered_metrics.items())
        if calibration_provenance == "apparent":
            calib_clause = (
                f"Calibration was assessed in the development sample ({metric_str}); these are apparent (in-sample) "
                "estimates that are optimistic and do not constitute external validation. "
            )
        else:
            calib_clause = f"Calibration was assessed in an independent external validation cohort ({metric_str}). "

    guideline_clause = (
        f"Reporting conformed to {guideline} guidelines based on verified checklist audit evidence for {study_design.lower()} studies."
        if fit_metadata and fit_metadata.get("checklist_audit_verified")
        else f"Methods description was structured in accordance with {guideline} reporting guidelines for {study_design.lower()} studies."
    )

    text = (
        f"Statistical Analysis: Continuous and categorical baseline variables were compared using {tests}. "
        f"Missing data were addressed via {missing_data_strategy}. "
        f"{model_type} was fitted to evaluate associations with {primary_outcome}, "
        f"{adj_clause}. "
        f"Effect estimates were reported with corresponding 95% confidence intervals. "
        f"{calib_clause}"
        f"{sig_clause}"
        f"{guideline_clause}"
    )
    return text


# Export HTML table and narrative
html_output = render_publication_html_table(model_records, style="NEJM")
with open("publication_table.html", "w", encoding="utf-8") as f:
    f.write(html_output)

narrative_text = generate_methods_narrative(
    study_design="Retrospective Cohort",
    primary_outcome="30-day Mortality",
    model_type="Multivariable logistic regression",
    confounders=["Age", "Female Sex", "Baseline SBP"],
    missing_data_strategy="complete-case analysis",
    guideline="STROBE",
    two_sided=True,
    alpha=0.05,
)
print("Methods Narrative:\n", narrative_text)
```

The agent may freely incorporate `medstat` modules (e.g. `from medstat.reporting.tables import render_records_table`, `from medstat.reporting.narrative import synthesize_methods_text`, `from medstat.reporting.checklists import generate_strobe_checklist`) or standard libraries as appropriate.

---

## Completion Criteria

- [ ] Statistical analysis results ingested and validated from JSON output.
- [ ] Publication table rendered in target journal format (NEJM, JAMA, or APA 7) with zero vertical lines and correct CI notation.
- [ ] Methods and results narrative paragraphs generated and reviewed (when `--narrative` is requested).
- [ ] Appropriate reporting checklist (STROBE, CONSORT, or TRIPOD) audited.

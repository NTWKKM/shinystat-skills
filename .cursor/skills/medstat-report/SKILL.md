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
import pandas as pd

# 1. ORGANIZE MODEL ESTIMATES & CLINICAL METRICS
model_records = [
    {"Variable": "Treatment Arm (Drug B vs Drug A)", "Estimate": "1.85", "CI": "(1.24 - 2.76)", "p_value": "0.003"},
    {"Variable": "Age (per 10-year increase)", "Estimate": "1.12", "CI": "(1.02 - 1.23)", "p_value": "0.018"},
    {"Variable": "Female Sex", "Estimate": "0.88", "CI": "(0.65 - 1.19)", "p_value": "0.412"},
    {"Variable": "Baseline SBP >= 140 mmHg", "Estimate": "1.45", "CI": "(1.08 - 1.95)", "p_value": "0.014"},
]

# 2. RENDER NEJM / JAMA PUBLICATION HTML TABLE
def format_p_value(p_val_str, style="NEJM"):
    try:
        p = float(p_val_str)
        if style.upper() == "JAMA":
            if p < 0.001:
                return "<.001"
            elif p >= 0.99:
                return ">.99"
            else:
                return f"{p:.3f}".lstrip("0")
        else:  # NEJM
            if p < 0.001:
                return "<0.001"
            elif p >= 0.99:
                return ">0.99"
            else:
                return f"{p:.3f}"
    except (ValueError, TypeError):
        return p_val_str

import html

def render_publication_html_table(records, style="NEJM", title="Table 2. Multivariable Logistic Regression Analysis", adjustment_vars=None, ci_method="profile likelihood"):
    is_nejm = style.upper() == "NEJM"
    top_border = "border-top: 3px double #000;" if is_nejm else "border-top: 1px solid #000;"
    p_header = "P Value" if is_nejm else "<em>P</em> Value"
    title_escaped = html.escape(str(title))
    
    if adjustment_vars:
        covar_str = ", ".join(html.escape(str(v)) for v in adjustment_vars)
        footnote_text = f"* Odds ratios were adjusted for {covar_str}. Confidence intervals are {html.escape(str(ci_method))}-based."
    else:
        footnote_text = f"* Unadjusted estimates. Confidence intervals are {html.escape(str(ci_method))}-based."
    
    html_out = f"""
    <div style="font-family: 'Times New Roman', Times, serif; max-width: 800px; margin: 20px auto;">
      <h3 style="margin-bottom: 8px; font-weight: bold;">{title_escaped}</h3>
      <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 14px;">
        <thead>
          <tr style="{top_border} border-bottom: 1px solid #000;">
            <th style="padding: 6px 8px;">Characteristic / Variable</th>
            <th style="padding: 6px 8px; text-align: right;">Adjusted Odds Ratio</th>
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
):
    if confounders is None:
        confounders = ["age", "sex", "hypertension"]
    if tests is None:
        tests = "Welch's t-test or Mann-Whitney U test for continuous variables and Pearson Chi-Square or Fisher's exact test for categorical variables"

    confounder_str = ", ".join(confounders)
    text = (
        f"Statistical Analysis: Continuous and categorical baseline variables were compared using {tests}. "
        f"Missing data were addressed via {missing_data_strategy}. "
        f"{model_type} was fitted to evaluate associations with {primary_outcome}, "
        f"adjusting for prespecified confounders ({confounder_str}). "
        f"Effect estimates were reported with corresponding 95% confidence intervals. "
        f"All tests were two-sided, with p < 0.05 considered statistically significant. "
        f"Reporting conformed to {guideline} guidelines for {study_design.lower()} studies."
    )
    return text

# Export HTML table and narrative
html_output = render_publication_html_table(model_records, style="NEJM")
with open("publication_table.html", "w", encoding="utf-8") as f:
    f.write(html_output)

narrative_text = generate_methods_narrative()
print("Methods Narrative:\n", narrative_text)
```

The agent may freely incorporate `medstat` modules (e.g. `from medstat.reporting.tables import render_records_table`, `from medstat.reporting.narrative import synthesize_methods_text`, `from medstat.reporting.checklists import generate_strobe_checklist`) or standard libraries as appropriate.

---

## Completion Criteria

- [ ] Statistical analysis results ingested and validated from JSON output.
- [ ] Publication table rendered in target journal format (NEJM, JAMA, or APA 7) with zero vertical lines and correct CI notation.
- [ ] Methods and results narrative paragraphs generated and reviewed (when `--narrative` is requested).
- [ ] Appropriate reporting checklist (STROBE, CONSORT, or TRIPOD) audited.

---
name: medstat-report
description: Publication-grade clinical report rendering, NEJM/JAMA/APA 7 styling, automated Methods and Results narrative generation, and reporting guideline audits (STROBE, CONSORT, TRIPOD). Use when compiling clinical study findings into publication-ready tables, generating compliant statistical methods narratives, formatting survival/logistic/diagnostic tables for medical journal submission, or auditing reporting checklist compliance.
---

# medstat-report: Publication HTML Tables, Methods Narratives & Guidelines

Dissemination engine for rendering publication-quality tables conforming to top medical journal standards (NEJM, JAMA, APA 7), synthesizing compliant statistical methods text, and generating reporting checklists.

## Core Rules

1. **Strict Journal Typography**: Enforce the exact rules of the target journal (NEJM double top border and symbol footnotes; JAMA minimal horizontal rules, decimal precision, and exact p-values without leading zeros; APA 7 italicized statistics).
2. **Automated Methods Narrative**: Every analysis report must pair quantitative tables with an automated Methods text describing model type, confounder selection, missing data handling, and significance criteria.
3. **Guideline Compliance**: Accompany observational studies with STROBE audits, randomized trials with CONSORT, and prediction/diagnostic studies with TRIPOD.

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
medstat report --checklist strobe --output strobe_checklist.html

# CONSORT Checklist (Randomized Controlled Trials)
medstat report --checklist consort --output consort_checklist.html

# TRIPOD Checklist (Clinical Prediction and Diagnostic Models)
medstat report --checklist tripod --output tripod_checklist.html
```

- Each checklist audits required reporting items: participant flow, eligibility criteria, missingness handling, bias mitigation, confounder adjustments, and sensitivity analysis.

## Completion Criteria

- [ ] Statistical analysis results ingested and validated from JSON output.
- [ ] Publication table rendered in target journal format (NEJM, JAMA, or APA 7) with zero vertical lines and correct CI notation.
- [ ] Methods and results narrative paragraphs generated and reviewed.
- [ ] Appropriate reporting checklist (STROBE, CONSORT, or TRIPOD) audited.

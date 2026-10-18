# Clinical Study Flow Diagrams Reference (CONSORT, PRISMA, STARD, STROBE)

This reference documents the automated, mathematically reconciled flow diagram generator (`medstat.figures.flow`).

---

## 1. Core Engineering Principles

1. **Strict Mathematical Reconciliation**:
   - Every split and branch is checked programmatically:
     $$\sum N_{\text{branches}} = N_{\text{parent}} - N_{\text{excluded}}$$
   - Any mathematical mismatch immediately raises an explicit error detailing the discrepancy (no silent number drift).
2. **Top-Down Layout Cursor (`Flow`)**:
   - Layout coordinates are automatically computed using vertical cursor advances and horizontal centering.
   - Text boxes automatically wrap with proper indentation for bullet items.
3. **Publication-Grade Export**:
   - **600 DPI** PNG for high-definition micro-typography.
   - Vector PDF with TrueType fonts (`pdf.fonttype = 42`).
   - Clean vector SVG for responsive web embedding.
   - Machine-auditable `_source.csv` recording bounding boxes, text, counts, and reconciliation status.

---

## 2. CONSORT 2025 & 2010 Flow Diagrams (`consort_rct`)

Complies with the latest CONSORT guidelines:
- **CONSORT 2025**: Explicitly reports randomized participants analyzed for primary outcome (Item 13c) and itemizes attrition/discontinuation reasons.
- **CONSORT 2010**: Classic 4-phase trial flow (Enrollment $\to$ Allocation $\to$ Follow-up $\to$ Analysis).

### Python Usage
```python
from medstat.figures.flow import consort_rct

res = consort_rct(
    assessed=1250,
    excluded={
        "Not meeting inclusion criteria": 210,
        "Declined to participate": 85,
        "Other reasons (geographic/transport)": 35,
    },
    arms=[
        {
            "name": "Intervention (Drug A + SOC)",
            "allocated": 460,
            "received": 452,
            "not_received": 8,
            "lost": 12,
            "discontinued": 18,
            "analyzed": 430,
            "excluded_analysis": 0,
        },
        {
            "name": "Control (Placebo + SOC)",
            "allocated": 460,
            "received": 458,
            "not_received": 2,
            "lost": 15,
            "discontinued": 20,
            "analyzed": 425,
            "excluded_analysis": 0,
        },
    ],
    version="2025",  # or "2010"
    journal="NEJM",  # "NEJM", "JAMA", "LANCET", "BMJ", or "CLINICAL"
    out_path="figures/consort_flow.png",
)
```

---

## 3. PRISMA 2020 Systematic Review Flow (`prisma_2020`)

Supports both single-stream (databases and registers only) and dual-stream (incorporating previous review search results or other methods like websites, citations, and trial registries).

### Python Usage (Dual-Stream)
```python
from medstat.figures.flow import prisma_2020

res = prisma_2020(
    identification={
        "databases": 4200,
        "registers": 350,
        "other": 85,
    },
    duplicates=1200,
    automation_excluded=450,
    records_screened=2985,  # (4200 + 350 + 85) - 1200 - 450 = 2985
    records_excluded=2450,
    reports_sought=535,
    reports_not_retrieved=35,
    reports_assessed=500,
    reasons_excluded={
        "Ineligible patient population": 220,
        "Wrong intervention / dosing": 140,
        "No primary clinical endpoint reported": 65,
    },
    studies_included=75,
    previous_studies=25,    # Triggers dual-stream flow
    journal="LANCET",
    out_path="figures/prisma_flow.png",
)
```

---

## 4. STARD 2015 Diagnostic Accuracy Flow (`stard_flow`)

Documents participant recruitment, index test execution, reference standard verification, indeterminate/inconclusive results, and final $2 \times 2$ diagnostic cross-tabulation.

### Python Usage
```python
from medstat.figures.flow import stard_flow

res = stard_flow(
    enrolled=600,
    excluded={"Did not meet clinical suspicion criteria": 45, "Consent withdrawn": 15},
    index_name="Novel Point-of-Care Biomarker",
    index_positive=210,
    index_negative=310,
    index_inconclusive=20,
    ref_name="Composite Gold Standard (Biopsy / PCR)",
    cells={
        "TP": 185,
        "FP": 25,
        "FN": 15,
        "TN": 295,
        "ref_missing": 20,
    },
    journal="JAMA",
    out_path="figures/stard_flow.png",
)
```

---

## 5. STROBE Observational Cohort Flow (`cohort_flow` & `SampleFlowTracker`)

Visualizes sequential inclusion/exclusion stages in clinical cohorts and links seamlessly with `medstat.data.retention.SampleFlowTracker`.

### Direct Integration with `SampleFlowTracker`
```python
from medstat.data.retention import SampleFlowTracker
from medstat.figures.flow import cohort_flow_from_tracker

tracker = SampleFlowTracker(initial_n=2500, initial_name="All Hospitalized Sepsis Patients")
tracker.record_stage("Age >= 18 years", 2400, 100, reason="Pediatric (<18 years)")
tracker.record_stage("Complete Baseline Lactate", 2150, 250, reason="Missing admission lactate")
tracker.record_stage("ICU Length of Stay >= 24h", 1980, 170, reason="Died or discharged <24h")

res = cohort_flow_from_tracker(
    tracker,
    groups={"30-Day Survivors": 1540, "30-Day In-Hospital Mortality": 440},
    journal="BMJ",
    out_path="figures/cohort_retention_flow.png",
)
```

---

## 6. Methods & ML Pipeline Architecture (`methods_pipeline`)

Renders linear or branching analytical workflows (data extraction, preprocessing, feature engineering, model training, cross-validation, and external testing).

```python
from medstat.figures.flow import methods_pipeline

res = methods_pipeline(
    stages=[
        {"title": "EHR Extraction", "desc": "12,450 ICU admissions (MIMIC-IV)"},
        {"title": "Preprocessing", "desc": "MICE imputation (M=20), outlier truncation"},
        {"title": "Feature Engineering", "desc": "RCS splines (4 knots), time-varying shock index"},
        {"title": "Model Training", "desc": "Firth penalized logistic & Fine-Gray CIF"},
        {"title": "Validation", "desc": "Bootstrap (B=500), Calibration, Decision Curve Analysis"},
    ],
    journal="NEJM",
    out_path="figures/methods_workflow.png",
)
```

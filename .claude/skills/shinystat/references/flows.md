# Clinical Study Flow Diagrams Reference (`medstat.figures.flow`)

Reached from `SKILL.md` Pillar 5. Every flow template is **reconciled**: participant counts are checked before drawing, any mismatch raises `ValueError` naming the violated equation, and every count and equation check is exported to `_source.csv`.

---

## 1. Core Engineering Principles

1. **Strict Mathematical Reconciliation**:
   - Every split and branch is verified programmatically ($\sum N_{\text{branches}} = N_{\text{parent}} - N_{\text{excluded}}$).
   - Any mismatch raises `ValueError("Unreconciled counts: ...")` immediately.
2. **Top-Down Layout Cursor (`Flow`)**:
   - Places boxes via `row(items)`, `gap(from_key, side=...)`, `link(a, b)`, and `mark(phase)` (rotated phase bars; boxes start right of `x = 6.8`, verified in `finish()`), wrapping text to fit and using `n_()` non-breaking spaces so `(n = 62)` never splits across lines.
3. **Publication-Grade Bundle (`FigureResult`)**:
   - **600 DPI** PNG (`png_path`), TrueType PDF (`vector_path`, `pdf.fonttype = 42`), SVG (`svg_path`), layout/Zero-PHI gate (`qa["passed"] == True`), and **traceable** `_source.csv` (`csv_path`).

---

## 2. CONSORT 2025 & 2010 Flow Diagrams (`consort_rct`)

- **CONSORT 2025** (`version="2025"`, default; BMJ 2025;389:e081123): labels follow-up and analysis boxes `"for primary outcome"` and lists intervention discontinuation before loss to follow-up (check box wording against the target journal's current CONSORT figure template before submission).
- **CONSORT 2010** (`version="2010"`): classic 4-phase trial wording.
- **Equations enforced**: `assessed = sum(excluded) + randomized`; `randomized = sum(allocated)`; per arm: `allocated = received + sum(not_received) = analysed + sum(excluded_analysis)`.

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
            "not_received": {"Declined infusion": 8},
            "lost": {"Moved away": 12},
            "discontinued": {"Adverse event": 18},
            "analysed": 430,
            "excluded_analysis": {"Missing primary endpoint": 30},
        },
        {
            "name": "Control (Placebo + SOC)",
            "allocated": 460,
            "received": 458,
            "not_received": {"Withdrew consent": 2},
            "lost": {"Unreachable": 15},
            "discontinued": {"Protocol withdrawal": 20},
            "analysed": 425,
            "excluded_analysis": {"Missing primary endpoint": 35},
        },
    ],
    version="2025",  # or "2010"
    journal="NEJM",
    out_path="figures/consort_flow.png",
)
```

---

## 3. PRISMA 2020 Systematic Review Flow (`prisma_2020`)

Supports single-stream (databases and registers) or dual-stream (`other={"sources": {...}, "not_retrieved": n, "fulltext_excluded": {...}}`).
- **Equations enforced**: `identified - removed = screened`; `screened - screened_excluded = sought`; `sought - not_retrieved = assessed`; `assessed - sum(fulltext_excluded) = included_reports` (summed across both streams when `other` is provided); `included_studies <= included_reports`.

```python
from medstat.figures.flow import prisma_2020

res = prisma_2020(
    databases={"PubMed": 2800, "Embase": 1400},
    registers={"ClinicalTrials.gov": 350},
    removed_duplicates=1200,
    removed_automation=450,
    removed_other=0,
    screened_excluded=2365,
    not_retrieved=35,
    fulltext_excluded={
        "Ineligible patient population": 220,
        "Wrong intervention / dosing": 140,
        "No primary clinical endpoint": 65,
    },
    included_studies=90,
    included_reports=105,  # 75 from databases/registers + 30 from other methods
    other={
        "sources": {"Websites": 35, "Citation searching": 50},
        "not_retrieved": 10,
        "fulltext_excluded": {"Duplicate cohort": 45},
    },
    journal="LANCET",
    out_path="figures/prisma_flow.png",
)
```

---

## 4. STARD 2015 Diagnostic Accuracy Flow (`stard_flow`)

- **Equations enforced**: `enrolled = eligible - sum(excluded) = index_pos + index_neg + index_inconclusive`; `index_pos = tp + fp + ref_not_done_pos`; `index_neg = fn + tn + ref_not_done_neg`.

```python
from medstat.figures.flow import stard_flow

res = stard_flow(
    eligible=600,
    excluded={"Did not meet suspicion criteria": 45, "Consent withdrawn": 15},
    index_pos=210,
    index_neg=310,
    index_inconclusive=20,
    tp=185,
    fp=25,
    fn=15,
    tn=295,
    ref_not_done_pos=0,
    ref_not_done_neg=0,
    journal="JAMA",
    out_path="figures/stard_flow.png",
)
```

---

## 5. STROBE Observational Cohort Flow (`cohort_flow` & `cohort_flow_from_tracker`)

Links directly to `SampleFlowTracker` (`medstat.data.retention`) or takes explicit `(reason, n_excluded)` steps.
- **Equations enforced**: `source - sum(steps) = analysed`; `sum(groups.values()) = analysed`.

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

## 6. Methods & ML Pipeline Overview (`methods_pipeline`)

Audits step counts (`n_in - sum(excluded) = n_out`, chained across steps), flags untraced numbers (`qa["untraced_numbers"]`), and checks citations (`qa["uncited_steps"]`; `strict_numbers=True` / `require_cite=True` raise `ValueError`).

```python
from medstat.figures.flow import methods_pipeline

res = methods_pipeline(
    steps=[
        {
            "title": "EHR Cohort Extraction",
            "body": "Adult ICU admissions (2022–2024)",
            "side": "Source: Tertiary HIS",
            "cite": "STROBE",
            "counts": {"n_in": 2500, "excluded": {"Age < 18": 100}, "n_out": 2400},
        },
        {
            "title": "Biomarker Quality Gate",
            "body": "Valid baseline arterial lactate",
            "cite": "Surviving Sepsis 2021",
            "counts": {"n_in": 2400, "excluded": {"Missing/hemolyzed": 420}, "n_out": 1980},
        },
    ],
    strict_numbers=True,
    journal="NEJM",
    out_path="figures/methods_workflow.png",
)
```

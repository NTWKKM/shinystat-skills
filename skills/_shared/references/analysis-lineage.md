# Analysis Lineage & Run Manifest Protocol

Reproducibility and anti-p-hacking discipline adapted from OpenResearch's frozen-node contract for clinical biostatistics (ADR 87).

---

## 1. Pre-Specified Primary Freezes Once Answered

- Register the single `primary` node during **Plan** (`SKILL.md §0` step 3) from the user's pre-specified research question and SAP.
- Once a run **answers** the node (returns a valid estimate, including a non-significant result or `NaN` with a documented fail-closed reason), mark it `answered` and **freeze** it permanently.
- Never edit or overwrite an answered node to "improve" a $P$-value or chase statistical significance; a disappointing or null finding is still a result.

---

## 2. Sensitivity Analyses Branch as Plan Deltas

- Every planned or mid-run sensitivity analysis (complete-case vs MICE, Firth vs standard MLE, E-value / MNAR delta-adjustment, subgroup/per-protocol check) is a **child node** (`role="sensitivity"`) branched off the primary node or an answered sensitivity parent.
- Each sensitivity child carries a non-empty `plan_delta` (the exact parameter/strategy difference from its parent) evaluated on the **same dataset hash** (`data_sha256`).
- Keep one named methodological change per sensitivity child so each comparison stays attributable.

---

## 3. Provisional vs Answered (Repair vs Freeze)

| Run Outcome | Node State | Allowed Action |
| :--- | :--- | :--- |
| Script/syntax error, missing column mapping, solver non-convergence, singular Hessian, mid-run Gate #3/#4 halt | **Provisional** | **Repair in place** (`repair()`) and re-run the same node |
| Converged estimate (significant or non-significant) | **Answered** | **Freeze** (`mark_answered()`); branch a child for any new variant |
| Undefined inference (`NaN` with explicit fail-closed reason, zero-variance SMD, absent class) | **Answered** | **Freeze** (`mark_answered()`); report the `NaN` and reason |

---

## 4. Repair Cap (2 Attempts → Gate #7)

- A provisional node allows at most **2 repair attempts** (`REPAIR_CAP = 2`).
- Immediately after calling `repair()` or `record_failure()` on a second failed attempt, explicitly check `len(node.failed_attempts) >= REPAIR_CAP` and **halt and ask the user** via **Grilling Gate #7** rather than relying on `RepairCapExceededError` (which `repair()` and `record_failure()` raise only on a subsequent call) or iterating blindly.
- Also **halt immediately and ask the user** via **Grilling Gate #7** if the same setup failure recurs on a second node.

---

## 5. Exploratory Analyses

- Any post-hoc or data-driven exploration requested or discovered during analysis must be registered with `role="exploratory"`.
- Label exploratory results explicitly as **"Exploratory (hypothesis-generating)"** in narrative, tables, and turn summaries.
- An exploratory node can **never** parent a `sensitivity` node and can **never** be silently promoted to `primary`.

---

## 6. Run Manifest (Zero-PHI Provenance)

- Build the **Run Manifest** once per analysed dataset during **Plan** (`build_run_manifest` in `medstat.reporting.provenance` for local runs, or [report-builder.md §4](report-builder.md#4-standalone-run-manifest--analysis-lineage) for cloud sandboxes).
- A manifest records: `data_sha256` (column-order independent, row- and dtype-sensitive), `n_rows`, `n_cols`, `plan_sha256`, `seed`, `package_versions`, `python_version`, and `created_at` (`Asia/Bangkok`), with **zero file paths or column names**.
- Attach `manifest` and `lineage` to `ReportDocument` (local: call `add_reproducibility_appendix(doc)`) and pass `data=df` to `verify_report_integrity` / `verify_standalone_integrity`.
- Display hashes in reports using the short prefix `sha256:<first 12 hex>` so 13-digit Thai ID regex collisions never arise.

---

## 7. One-Line-Per-Analysis Turn Summary

- Close any turn that executes or updates models with a compact tree summary (`AnalysisLineage.summary_lines()` / `lineage_summary_lines()`):
  - `- [primary · answered] <question>: <headline>`
  - `  - [sensitivity · answered] <question> (Δ <changed keys>): <headline>`
- Every numeric value quoted in the turn summary must trace to `results_dict` and be audited via `delivered_text`.

---

## 8. Output Artifact Folder Layout

Organize multi-file deliverables under one descriptive topic directory instead of scattering files at the workspace root:

```text
<topic-slug>/
  <topic-slug>-report.html   # or .docx / .pptx / .pdf / .md
  figures/
    <figure-stem>.png
    <figure-stem>.pdf
    <figure-stem>.svg
    <figure-stem>.csv
  data/                      # only when data files are requested; Zero-PHI verified
```

Create `figures/` or `data/` only when files exist for them (no empty scaffolding).

---

## 9. In-Chat Summary Single-Representation Rule

- When `#5a` is **In-Chat Summary** (no file deliverable), present **one primary representation** per comparison (a concise table, **or** a figure, **or** prose with headline values)—never repeat the exact same numbers in both a table and a prose list in the same reply.
- File deliverables (`.html`, `.docx`, `.pptx`, `.pdf`, `.md`) retain **Exhaustive Tables** alongside figures per journal reporting standards (§3).

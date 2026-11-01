# Subagent Context-Offload Protocol (`subagent-protocol.md`)

Reference protocol for `shinystat` and `shinystat-cloud` when the runtime provides a subagent-spawning tool and **Subagent Offload** is selected at **Grilling Gate #5**.

---

## 1. Capability-Conditional Activation & Parallel Topology

- **Activation Condition**: Engage this protocol only when the active agent runtime exposes a tool for spawning subagents and the user selects **Subagent Offload** at **Grilling Gate #5**. When no subagent tool exists or the user omits a preference, default to **Direct in Main Agent**.
- **Role Division**:
  1. **Main Agent (*Orchestrator & Clinical Gatekeeper*)**:
     - Runs **Pillar 1** pre-flight data inspection (`df.info()`, `df.describe()`, missingness %, numeric `0/1` coding check) and **Pillar 2** archetype triangulation.
     - Conducts the **single consolidated pre-flight inquiry** with the user for any triggered `#1`, `#2`, `#4` (when pre-flight confirms $> 40\%$ non-primary missingness or clinically plausible MNAR without sensitivity analysis), and `#5` items; computes **pre-flight** $\text{EPV}_{\text{binary}}$ / $\text{EPV}_{\text{Cox}}$ (during Pillar 1 when outcome/event coding is unambiguous, or immediately after resolving Gate `#1`/`#2` in the consolidated turn before model fitting).
     - Writes `<output_dir>/sap_handoff.json` and dispatches Subagent(s).
     - Performs **independent on-disk integrity verification** (§4) before delivering final outputs to the user.
  2. **Subagent B (*Biostatistical Execution & Report Subagent*)**:
     - Always dispatched in **Subagent Offload** mode.
     - Reads `<output_dir>/sap_handoff.json` and loads only the required statistical references (`decision-heuristics.md`, `figures.md`, `flows.md`, `house-styles.md`, `illustration.md`, `report-builder.md §1–§3`, and target `references/recipes/01-*.md` to `05-*.md` in `shinystat-cloud`).
     - Executes the statistical models, tracks `SampleFlowTracker` / `retention_audit`, exports 300/600 DPI figure bundles (`.png`, TrueType `.pdf` with `fonttype=42`, `.svg`, `_source.csv` with `qa["passed"] == True`), and writes `<output_dir>/results_dict.json` and `<output_dir>/retention_audit.json`.
  3. **Subagent A (*Literature & Citation Verification Subagent*)**:
     - Dispatched **in parallel with Subagent B** only when **Full Publication Polish** is selected at Gate `#5` or when user-provided references require online verification.
     - Reads `<output_dir>/sap_handoff.json` and loads `report-builder.md §4` (*Evidence-Base Verification & Superscript Citations Pipeline*).
     - Queries available literature MCPs or web search tools strictly under the **Zero-PHI Literature Query Boundary**, verifies source-appropriate metadata and comparable empirical benchmarks (**Anti-Phantom Citation Invariant**), and writes `<output_dir>/citations_verified.json`.
  4. **Report Assembly & Delivery Gate**:
     - Once active branches (`Subagent B`, and `Subagent A` when engaged) reach `COMPLETED`, Subagent B (or the report compilation step) merges `<output_dir>/results_dict.json` and figure bundles—plus `<output_dir>/citations_verified.json` only when Subagent A was engaged (compiling without citation input when Subagent A was not dispatched)—into the target deliverable (`.html`, `.docx`, `.pptx`, `.pdf`, or `.md`), and writes `<output_dir>/report_narrative.txt` and `<output_dir>/table_dfs.json` (with non-rendered cohort indexes dropped via `df.reset_index(drop=True)` so patient identifiers cannot appear) from the exact finalized post-merge narrative and tables used to render that deliverable.
     - **Post-Assembly Discussion & Results Audit**: When §4 integrity verification is engaged, after the finalized report narrative is produced but before delivery to the user, the Main Agent audits the finalized Discussion and Results against `<output_dir>/results_dict.json` and the study design (verifying point estimates, 95% CIs, p-values, effect directions, and study-archetype causal caveats), and halts delivery immediately if the audit fails.

---

## 2. SAP Hand-Off Brief Schema (`<output_dir>/sap_handoff.json`)

The Main Agent writes `<output_dir>/sap_handoff.json` to disk before spawning any subagent.

- **Zero-PHI Hand-Off Invariant**: Never embed raw patient rows, free-text clinical notes, or patient identifiers (HN, National ID, name, DOB, phone) inside `sap_handoff.json` or subagent prompt strings. Pass only file paths, column names, and aggregate pre-flight statistics.

```json
{
  "schema_version": "1.0",
  "data_path": "/abs/path/to/clinical_cohort.csv",
  "output_dir": "/abs/path/to/output",
  "column_mapping": {
    "outcome": "mortality_30d",
    "exposure": "early_vasopressor",
    "covariates": ["age_years", "sofa_score", "lactate_mmol_l"],
    "time_col": null,
    "event_col": null,
    "cluster_col": null,
    "biomarker_direction": "high",
    "numeric_coding_verified": true
  },
  "preflight_profile": {
    "n_initial": 450,
    "missingness_pct_by_col": {
      "lactate_mmol_l": 8.4,
      "sofa_score": 1.1
    },
    "epv_metric": "EPV_binary",
    "epv_value": 14.2,
    "study_archetype": "Type 5: Observational Causal"
  },
  "gate_decisions": {
    "resolved_gates_1_to_4": {},
    "deliverable_format": "html",
    "house_style": "NEJM",
    "polish_mode": "standard",
    "execution_context_mode": "subagent_offload"
  }
}
```

---

## 3. Mid-Run HALT, Parallel Synchronization & On-Disk Checkpoint/Resume Protocol

Because **Gate #3** (monotone separation), **Gate #4** ($> 40\%$ non-primary missingness or plausible MNAR without sensitivity analysis), and **Gate #6** (unverified, retracted, or contradictory citation) are detected *during* model fitting or literature verification inside a subagent—while only the Main Agent communicates with the user—subagents must follow this deterministic checkpoint-and-halt protocol:

### A. On-Disk Checkpointing & Row-Level Data Isolation
1. Create an isolated control directory `<output_dir>/.scratch_checkpoint/` for non-PHI checkpoint metadata, and create a restricted workspace directory **outside** the user-facing `<output_dir>` (e.g., `<scratch_dir>/shinystat_ckpt_<run_id>/` via `tempfile.mkdtemp(prefix="shinystat_ckpt_")`) for any row-level cohort checkpoints.
2. If Subagent B has cleaned or recoded the cohort before halting, save the intermediate cohort inside the restricted external workspace `<scratch_dir>/shinystat_ckpt_<run_id>/cleaned_cohort.parquet` (or `.csv`) outside `<output_dir>`, while keeping the executable script at `<output_dir>/analysis_script.py`.
3. Write a role-specific checkpoint-state file `<output_dir>/.scratch_checkpoint/checkpoint_state_<role>.json` (e.g., `checkpoint_state_biostat_execution.json` or `checkpoint_state_literature_verifier.json`):
   ```json
   {
     "subagent_role": "biostat_execution | literature_verifier",
     "completed_steps": ["preflight_clean", "retention_tracker", "table1"],
     "cleaned_data_path": "/abs/path/to/scratch/shinystat_ckpt_<run_id>/cleaned_cohort.parquet",
     "retention_audit_path": "/abs/path/to/output/retention_audit.json",
     "partial_results_path": "/abs/path/to/output/results_dict.json",
     "script_path": "/abs/path/to/output/analysis_script.py"
   }
   ```

### B. Structured HALT Payload (`<output_dir>/.scratch_checkpoint/halt_payload_<role>.json`)
The halting subagent writes its payload to disk and returns the exact JSON structure to the Main Agent:
```json
{
  "status": "HALT_FOR_GRILLING_GATE",
  "subagent_role": "biostat_execution",
  "gate_id": "#3",
  "trigger_summary": "Monotone separation detected for covariate 'mechanical_ventilation' in multivariable logistic regression.",
  "options_for_user": [
    "A: Apply Firth penalized likelihood logistic regression with profile likelihood 95% CIs",
    "B: Remove or collapse 'mechanical_ventilation' from the multivariable model"
  ],
  "checkpoint_path": "/abs/path/to/output/.scratch_checkpoint/checkpoint_state_biostat_execution.json",
  "script_path": "/abs/path/to/output/analysis_script.py"
}
```

### C. Parallel Sibling Completion & Single-Turn Halt Aggregation
When Subagent A and Subagent B run in parallel and one branch halts:
1. **Sibling Continues to Completion**: Let the unaffected sibling subagent run to completion and write its output (`results_dict.json` or `citations_verified.json`) to `<output_dir>` without cancelling or killing it.
2. **Wait & Aggregate**: The Main Agent waits until both parallel branches reach a terminal state (`COMPLETED` or `HALT_FOR_GRILLING_GATE`), aggregates all triggered HALT payloads, and presents a **single consolidated question turn** to the user.
3. **Selective Resume**: After the user responds, the Main Agent updates `sap_handoff.json["gate_decisions"]["resolved_gates_1_to_4"]` (or citation instructions) and resumes **only** the halted subagent branch(es) from its role-specific `<output_dir>/.scratch_checkpoint/checkpoint_state_<role>.json` without re-running the already-completed sibling branch or repeating pre-flight cleaning.
4. **Deferred Compilation**: Final report compilation runs only after all active branches return `COMPLETED`.

---

## 4. Independent Main-Agent On-Disk Integrity Verification & Scratch Cleanup

Before presenting the final deliverable to the user, the Main Agent must independently verify the on-disk artifacts rather than relying on a subagent's conversational claim that `IntegrityReport.passed == True`:

1. **Artifact Existence & Non-Zero Size Check**:
   - Confirm that the deliverable file (`.html`, `.docx`, `.pptx`, `.pdf`, or `.md`), `<output_dir>/results_dict.json`, and every figure's bundle files (`.png`, TrueType `.pdf`, `.svg`, `_source.csv`) exist on disk with byte size $> 0$.
2. **Independent Execution of Pillar 5 Integrity Gate**:
   - **In `shinystat` (Local Engine)**:
     - The Main Agent reloads `<output_dir>/report_narrative.txt`, `<output_dir>/results_dict.json`, and `<output_dir>/table_dfs.json`.
     - It reconstructs the exact finalized document by parsing `report_narrative.txt` into `ParagraphBlock`s (and `HeadingBlock`s where headings exist) and reconstructing each table entry from `table_dfs.json` into a `pandas.DataFrame` via `pd.DataFrame(data=item["data"], columns=item["columns"])` wrapped in a `TableBlock(df=df, caption=item.get("caption", "Table"), footnote=item.get("footnote"))`.
     - It instantiates `ReportDocument(title="Clinical Analysis Report", blocks=reconstructed_blocks, results_dict=results_dict)` and invokes `verify_report_integrity(doc)`. Delivery halts if `IntegrityReport.passed is not True`.
     - After `verify_report_integrity` consumes `<output_dir>/table_dfs.json`, remove `table_dfs.json` from the output workspace.
   - **In `shinystat-cloud` (Standalone Cloud Sandbox)**: To keep the Main Agent's context window lean without loading the full `report-builder.md` file:
     - During final report assembly (§1.4), Subagent B writes `<output_dir>/report_narrative.txt` (the exact finalized post-merge narrative text rendered into the deliverable), `<output_dir>/table_dfs.json` (serialized as a JSON list of table dicts `[df.reset_index(drop=True).to_dict(orient="split") for df in table_dfs]` without non-rendered cohort indexes so patient identifiers cannot appear in `table_dfs.json`), and `<output_dir>/.scratch_checkpoint/verify_integrity.py`.
     - `<output_dir>/.scratch_checkpoint/verify_integrity.py` loads `<output_dir>/report_narrative.txt`, `<output_dir>/results_dict.json`, and `<output_dir>/table_dfs.json`, reconstructs every table entry into a `pandas.DataFrame` via `pd.DataFrame(data=item["data"], columns=item["columns"])` (exiting non-zero immediately if `table_dfs.json` is missing, malformed, or any entry fails `isinstance(df, pd.DataFrame)` reconstruction), calls `verify_standalone_integrity` (from `report-builder.md §3`), and exits non-zero (`sys.exit(1)`) unless the returned result has `passed == True`, `len(untraced_numbers) == 0`, `len(phi_violations) == 0`, and `has_causal_caveat == True`.
     - The Main Agent executes `python <output_dir>/.scratch_checkpoint/verify_integrity.py` after final report assembly and treats a missing script or a non-zero exit code as a failed integrity check.
     - Immediately after `verify_standalone_integrity` consumes `<output_dir>/table_dfs.json`, remove `table_dfs.json` from the output workspace.
3. **Scratch Row-Level Checkpoint Cleanup**:
   - Across all terminal outcomes—(a) successful completion (`passed == True`), (b) verification failure or unrecoverable halt, and (c) abandoned or superseded resumes—`unlink` and remove the external row-level checkpoint directory (`<scratch_dir>/shinystat_ckpt_<run_id>/cleaned_cohort.parquet` / `.csv`), `<output_dir>/table_dfs.json` (if still present), and `.scratch_checkpoint/` staging artifacts so row-level checkpoints cannot remain in deliverables or on disk.

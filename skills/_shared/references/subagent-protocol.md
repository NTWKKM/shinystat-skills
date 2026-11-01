# Subagent Context-Offload Protocol (`subagent-protocol.md`)

Reference protocol for `shinystat` and `shinystat-cloud` when the runtime provides a subagent-spawning tool and **Subagent Offload** is selected at **Grilling Gate #5**.

---

## 1. Capability-Conditional Activation & Parallel Topology

- **Activation Condition**: Engage this protocol only when the active agent runtime exposes a tool for spawning subagents and the user selects **Subagent Offload** at **Grilling Gate #5**. When no subagent tool exists or the user omits a preference, default to **Direct in Main Agent**.
- **Role Division**:
  1. **Main Agent (*Orchestrator & Clinical Gatekeeper*)**:
     - Runs **Pillar 1** pre-flight profiling (`df.info()`, `df.describe()`, missingness %, numeric `0/1` coding check, and pre-flight $\text{EPV}_{\text{binary}}$ / $\text{EPV}_{\text{Cox}}$) and **Pillar 2** archetype triangulation.
     - Conducts the **single consolidated pre-flight inquiry** with the user for any triggered `#1`, `#2`, and `#5` items.
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
  4. **Report Assembly**:
     - Once both active branches (`Subagent B`, and `Subagent A` when engaged) reach `COMPLETED`, Subagent B (or the report compilation step) merges `<output_dir>/results_dict.json`, figure bundles, and `<output_dir>/citations_verified.json` into the target deliverable (`.html`, `.docx`, `.pptx`, `.pdf`, or `.md`).

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
1. Create an isolated staging directory `<output_dir>/.scratch_checkpoint/` (excluded from user-facing deliverable bundles and archives).
2. If Subagent B has cleaned or recoded the cohort prior to halting, save the intermediate cohort inside `<output_dir>/.scratch_checkpoint/cleaned_cohort.parquet` (or `.csv`) and save the executable script to `<output_dir>/analysis_script.py`.
3. Write `<output_dir>/.scratch_checkpoint/checkpoint_state.json`:
   ```json
   {
     "subagent_role": "biostat_execution | literature_verifier",
     "completed_steps": ["preflight_clean", "retention_tracker", "table1"],
     "cleaned_data_path": "/abs/path/to/output/.scratch_checkpoint/cleaned_cohort.parquet",
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
  "checkpoint_path": "/abs/path/to/output/.scratch_checkpoint/checkpoint_state.json",
  "script_path": "/abs/path/to/output/analysis_script.py"
}
```

### C. Parallel Sibling Completion & Single-Turn Halt Aggregation
When Subagent A and Subagent B run in parallel and one branch halts:
1. **Sibling Continues to Completion**: Do **not** cancel or kill the unaffected sibling subagent. Let the unaffected branch finish its work and write its output (`results_dict.json` or `citations_verified.json`) to `<output_dir>`.
2. **Wait & Aggregate**: The Main Agent waits until both parallel branches reach a terminal state (`COMPLETED` or `HALT_FOR_GRILLING_GATE`), aggregates all triggered HALT payloads, and presents a **single consolidated question turn** to the user.
3. **Selective Resume**: After the user responds, the Main Agent updates `sap_handoff.json["gate_decisions"]["resolved_gates_1_to_4"]` (or citation instructions) and resumes **only** the halted subagent branch(es) from `<output_dir>/.scratch_checkpoint/checkpoint_state.json` without re-running the already-completed sibling branch or repeating pre-flight cleaning.
4. **Deferred Compilation**: Final report compilation runs only after all active branches return `COMPLETED`.

---

## 4. Independent Main-Agent On-Disk Integrity Verification & Scratch Cleanup

The Main Agent must **never** trust a subagent's conversational claim that `IntegrityReport.passed == True`. Before presenting the final deliverable to the user, the Main Agent must independently verify the on-disk artifacts:

1. **Artifact Existence & Non-Zero Size Check**:
   - Confirm that the deliverable file (`.html`, `.docx`, `.pptx`, `.pdf`, or `.md`), `<output_dir>/results_dict.json`, and every figure's bundle files (`.png`, TrueType `.pdf`, `.svg`, `_source.csv`) exist on disk with byte size $> 0$.
2. **Independent Execution of Pillar 5 Integrity Gate**:
   - **In `shinystat` (Local Engine)**: The Main Agent runs a lightweight verification check importing `verify_report_integrity` from `medstat.reporting.integrity` against the serialized `ReportDocument` / narrative and `<output_dir>/results_dict.json`.
   - **In `shinystat-cloud` (Standalone Cloud Sandbox)**: To keep the Main Agent's context window lean without loading the full `report-builder.md` file:
     - Subagent B writes the standalone `verify_standalone_integrity` function (from `report-builder.md §3`) to `<output_dir>/.scratch_checkpoint/verify_integrity.py` alongside `<output_dir>/results_dict.json` and `<output_dir>/report_narrative.txt`.
     - The Main Agent executes `python <output_dir>/.scratch_checkpoint/verify_integrity.py` (or reads strictly `report-builder.md §3` if the script was not emitted) to independently confirm `passed == True`, `len(untraced_numbers) == 0`, `len(phi_violations) == 0`, and `has_causal_caveat == True`.
3. **Scratch Row-Level Checkpoint Cleanup**:
   - Immediately after independent verification confirms `passed == True`, delete any temporary row-level cohort files in `<output_dir>/.scratch_checkpoint/` (e.g., `cleaned_cohort.parquet` / `.csv`) so no unbundled row-level patient data remains in the output workspace.

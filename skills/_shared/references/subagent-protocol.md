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
     - Dispatched **in parallel with Subagent B** when **Full Publication Polish** is selected at Gate `#5` or when user-provided references are supplied for verification (whether online via literature MCPs/web search or offline via authenticated local-source verification in Standard mode).
     - Reads `<output_dir>/sap_handoff.json` and loads `report-builder.md §4` (*Evidence-Base Verification & Superscript Citations Pipeline*).
     - Queries available literature MCPs or web search tools strictly under the **Zero-PHI Literature Query Boundary** (or verifies local user-provided citations against DOI/PMID/bibliographic metadata offline when literature services are unavailable), verifies source-appropriate metadata and comparable empirical benchmarks (**Anti-Phantom Citation Invariant**), and writes `<output_dir>/citations_verified.json`.
  4. **Report Assembly & Delivery Gate**:
     - Once active branches (`Subagent B`, and `Subagent A` when engaged) reach `COMPLETED`:
       - **File Deliverable Path**: When an external deliverable is requested (`.html`, `.docx`, `.pptx`, `.pdf`, or `.md`), Subagent B (or the report compilation step) merges `<run_output_dir>/results_dict.json` and figure bundles—plus `<run_output_dir>/citations_verified.json` whenever Subagent A produced citation input (compiling without citation input and not requiring that file when Subagent A was not dispatched or after the user accepts the no-citation fallback for a failed Subagent A)—into the target deliverable, and writes `<run_output_dir>/report_narrative.txt` and `<run_output_dir>/table_dfs.json` (or `<run_output_dir>/table_dfs_<run_id>.json`, with non-rendered cohort indexes dropped via `df.reset_index(drop=True)` so patient identifiers cannot appear) inside `run_output_dir` (`<output_dir>/run_<run_id>/`) from the exact finalized post-merge narrative and tables used to render that deliverable.
       - **In-Chat Summary Path**: When **In-Chat Summary** is selected (with no external file deliverable requested), Subagent B merges `<run_output_dir>/results_dict.json`—plus `<run_output_dir>/citations_verified.json` whenever Subagent A produced citation input (compiling without citation input and not requiring that file when Subagent A was not dispatched or after the user accepts the no-citation fallback for a failed Subagent A)—directly into the markdown summary written to `<run_output_dir>/report_narrative.txt` and writes `<run_output_dir>/table_dfs.json` (or `<run_output_dir>/table_dfs_<run_id>.json`, with dropped indexes); the Main Agent displays and verifies the narrative and tables directly in chat without requiring an unrequested file deliverable, preserving all integrity and citation audits.
       - **Terminal Staging Table Cleanup**: Regardless of whether the external-deliverable or In-Chat Summary path was used, the terminal cleanup steps remove both the run-specific table files (`<run_output_dir>/table_dfs_<run_id>.json` and `<run_output_dir>/table_dfs.json`) and any legacy `<output_dir>/table_dfs.json` whenever present, on every terminal outcome (successful delivery, verification failure, or unrecoverable halt).
     - **Finalized Narrative Citation Verification**: When Subagent A was engaged, citation verification must not be restricted to the initial hand-off state. After report assembly produces the finalized narrative (`report_narrative.txt` / deliverable) and before delivery to the user, the Main Agent (or Subagent A in a post-assembly pass) reruns verification against the finalized Discussion and narrative text to confirm that every in-text superscript citation and empirical benchmark comparison matches verified sources and satisfies the **Anti-Phantom Citation Invariant**; when Subagent A was not engaged, compilation and delivery proceed without citation input or citation audit.
     - **Post-Assembly Discussion & Results Audit**: When §4 integrity verification is engaged, after the finalized report narrative is produced but before delivery to the user, the Main Agent audits the finalized Discussion and Results against `<output_dir>/results_dict.json` and the study design (verifying point estimates, 95% CIs, p-values, effect directions, and study-archetype causal caveats), and halts delivery immediately if the audit fails.

---

## 2. SAP Hand-Off Brief Schema (`<output_dir>/sap_handoff.json`)

The Main Agent writes `<output_dir>/sap_handoff.json` to disk before spawning any subagent.

- **Zero-PHI Hand-Off Invariant**: Never embed raw patient rows, free-text clinical notes, or patient identifiers (HN, National ID, name, DOB, phone) inside `sap_handoff.json` or subagent prompt strings. Pass only file paths, column names, and aggregate pre-flight statistics.
- **Concurrent Run Namespace & Ownership Invariant**: To prevent concurrent runs from overwriting, corrupting, or prematurely deleting each other's state, every run must assign a unique `run_id` (e.g. timestamp or UUID suffix) and establish a distinct per-run output directory `run_output_dir` (`<output_dir>/run_<run_id>/`) for ALL artifacts—including `sap_handoff.json`, scripts, figures, `results_dict.json`, `retention_audit.json`, checkpoints (`.scratch_checkpoint_<run_id>/`), deliverables, and staging tables (`table_dfs.json` / `table_dfs_<run_id>.json`). Do not rely on checkpoint-only namespacing within a shared output folder. Orchestrators must verify `run_id` ownership before reading or cleaning state, and enforce that cleanup unlinks only artifacts belonging strictly to that specific run's directory.

```json
{
  "schema_version": "1.0",
  "run_id": "run_20261101_181027",
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
    "citation_instructions": {
      "version": 1,
      "gate_6_decision": null,
      "guidance": null
    },
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
1. Create an isolated control directory `<output_dir>/.scratch_checkpoint_<run_id>/` for non-PHI checkpoint metadata, and create a restricted workspace directory **outside** the user-facing `<output_dir>` using `tempfile.mkdtemp(prefix=f"shinystat_ckpt_{run_id}_", dir=scratch_dir)` for any row-level cohort checkpoints. Store the exact returned directory path (`row_level_checkpoint_dir`).
2. If Subagent B has cleaned or recoded the cohort before halting, save the intermediate cohort inside the restricted external workspace `<row_level_checkpoint_dir>/cleaned_cohort.parquet` (or `.csv`) outside `<output_dir>`, record this exact path in `cleaned_data_path`, while keeping the executable script at `<output_dir>/analysis_script.py`.
3. Write a role-specific checkpoint-state file `<output_dir>/.scratch_checkpoint_<run_id>/checkpoint_state_<role>.json` (e.g., `checkpoint_state_biostat_execution.json` or `checkpoint_state_literature_verifier.json`):
   ```json
   {
     "run_id": "run_20261101_181027",
     "subagent_role": "biostat_execution | literature_verifier",
     "completed_steps": ["preflight_clean", "retention_tracker", "table1"],
     "preprocessing_provenance": {
       "missing_data_strategy": "complete_case | mice | indicator",
       "imputation_m": 10,
       "exclusion_criteria": "missing_lactate_dropped"
     },
     "cleaned_data_path": "/abs/path/to/scratch/shinystat_ckpt_<run_id>/cleaned_cohort.parquet",
     "retention_audit_path": "/abs/path/to/output/retention_audit.json",
     "partial_results_path": "/abs/path/to/output/results_dict.json",
     "script_path": "/abs/path/to/output/analysis_script.py"
   }
   ```

### B. Structured HALT Payload (`<output_dir>/.scratch_checkpoint_<run_id>/halt_payload_<role>.json`)
The halting subagent writes its payload to disk and returns the exact JSON structure to the Main Agent:
```json
{
  "run_id": "run_20261101_181027",
  "status": "HALT_FOR_GRILLING_GATE",
  "subagent_role": "biostat_execution",
  "gate_id": "#3",
  "trigger_summary": "Monotone separation detected for covariate 'mechanical_ventilation' in multivariable logistic regression.",
  "options_for_user": [
    "A: Apply Firth penalized likelihood logistic regression with profile likelihood 95% CIs",
    "B: Remove or collapse 'mechanical_ventilation' from the multivariable model"
  ],
  "checkpoint_path": "/abs/path/to/output/.scratch_checkpoint_<run_id>/checkpoint_state_biostat_execution.json",
  "script_path": "/abs/path/to/output/analysis_script.py"
}
```

### C. Single-Branch & Parallel Sibling Halt Handling, Aggregation & Resume

1. **Single-Branch Execution Halt**:
   - When only Subagent B is dispatched (no Subagent A) and it halts for Gate #3 (monotone separation) or Gate #4 (missingness / MNAR):
   - Subagent B writes `<output_dir>/.scratch_checkpoint_<run_id>/halt_payload_biostat_execution.json` and halts.
   - The Main Agent immediately presents the gate question to the user.
   - Upon the user's decision, the Main Agent updates `sap_handoff.json["gate_decisions"]["resolved_gates_1_to_4"]`, applies Gate #4 provenance reconciliation (invalidating cached data, cohort-dependent steps, and stale partial results if the missingness strategy changed), and resumes Subagent B from its checkpoint.
2. **Parallel Sibling Completion & Single-Turn Halt Aggregation**:
   - When Subagent A and Subagent B run in parallel and one branch halts:
     - **Sibling Continues to Completion**: Let the unaffected sibling subagent run to completion and write its output (`results_dict.json` or `citations_verified.json`) to `<output_dir>` without cancelling or killing it.
     - **Wait & Aggregate**: The Main Agent waits until both parallel branches reach a terminal state (`COMPLETED` or `HALT_FOR_GRILLING_GATE`), aggregates all triggered HALT payloads, and presents a **single consolidated question turn** to the user.
     - **Branch Failure Recovery**: If a parallel branch fails (encounters an unhandled error or crashes) while its sibling has halted:
       - If the failure is in an optional literature branch (Subagent A) due to transient network or tool availability, the Main Agent may retry Subagent A once or offer the user to fall back to Standard offline compilation without citations alongside the sibling's Gate halt.
       - If the failure is in the core biostatistical branch (Subagent B) or is unrecoverable, declare the run unrecoverable, perform run-scoped cleanup of temporary scratch checkpoints for that `run_id`, while preserving the error log and any pending HALT payload for user diagnostic visibility before exiting.
3. **Selective Resume & Preprocessing Provenance Invalidation**:
   - **Gate #1–#4 Decision & Provenance Reconciliation**: After the user responds, the Main Agent updates `sap_handoff.json["gate_decisions"]["resolved_gates_1_to_4"]`. If a Gate #4 decision modifies the missing-data handling strategy or conflicts with the recorded `preprocessing_provenance` in `checkpoint_state_biostat_execution.json` (e.g. switching from complete-case listwise deletion to MICE, or altering MNAR sensitivity parameters): the Main Agent and Subagent B must NOT reuse incompatible data at `cleaned_data_path`. Invalidate `cleaned_data_path`, rerun cohort cleaning and sample retention tracking (`SampleFlowTracker` / `retention_audit`), purge cohort-dependent steps from `completed_steps` (e.g. remove `"table1"`, `"retention_tracker"`, and downstream model steps), purge stale cohort-dependent entries from `partial_results` / `results_dict` (such as baseline demographics and unadjusted counts, preserving complete-case sensitivity results only when explicitly labeled as sensitivity analysis) from raw data under the new missingness strategy, update `retention_audit.json` and `cleaned_data_path`, and only then resume Subagent B's model fitting. When Gate #4 decisions or Gate #3 separation resolutions remain compatible with existing preprocessing, preserve the selective-resume behavior and load `cleaned_data_path` directly without repeating data cleaning.
   - **Gate #6 Citation Decision Contract**: When resuming Subagent A following a Gate #6 halt, the Main Agent increments `sap_handoff.json["gate_decisions"]["citation_instructions"]["version"]`, sets `gate_6_decision` (e.g., `"omit"`, `"replace_with_user_source"`, or `"re_verify"`), and records user guidance. Subagent A is required to read and apply `citation_instructions` when resuming the citation branch so it deterministically follows the user's latest Gate #6 decision.
   - **Targeted Branch Execution**: Resume **only** the halted subagent branch(es) from its role-specific `<output_dir>/.scratch_checkpoint_<run_id>/checkpoint_state_<role>.json` without re-running the already-completed sibling branch.
4. **Deferred Compilation**: Final report compilation runs only after all active branches return `COMPLETED`.

---

## 4. Independent Main-Agent On-Disk Integrity Verification & Scratch Cleanup

Before presenting the final deliverable to the user, the Main Agent must independently verify the on-disk artifacts rather than relying on a subagent's conversational claim that `IntegrityReport.passed == True`:

1. **Artifact Existence & Non-Zero Size Check**:
   - Confirm that the deliverable file (`.html`, `.docx`, `.pptx`, `.pdf`, or `.md`), `<output_dir>/results_dict.json`, and every figure's bundle files (`.png`, TrueType `.pdf`, `.svg`, `_source.csv`) exist on disk with byte size $> 0$.
2. **Independent Execution of Pillar 5 Integrity Gate & Reconciliation Audit**:
   - **Comprehensive Document Model Coverage**: Both local and cloud verification paths must check the finalized `ReportDocument` or serialize every rendered block—including `HeadingBlock`s, `ParagraphBlock`s, `CalloutBlock`s (note/warning text), `FigureBlock`s (figure caption and alt-text), and `TableBlock`s (table headers, cells, captions, and footnotes)—ensuring all rendered narrative and metadata undergo Zero-PHI and numerical traceability scanning.
   - **In `shinystat` (Local Engine)**:
     - The Main Agent reloads `<run_output_dir>/report_narrative_<run_id>.txt` (or `<run_output_dir>/report_narrative.txt`), `<run_output_dir>/results_dict.json`, and `<run_output_dir>/table_dfs_<run_id>.json` (or `<run_output_dir>/table_dfs.json`).
     - It verifies the finalized `ReportDocument` or reconstructs the exact finalized document by parsing every rendered block: `ParagraphBlock`s, `HeadingBlock`s, `CalloutBlock`s, `FigureBlock`s (with caption/alt-text), and reconstructing each table entry from `table_dfs_<run_id>.json` into a `pandas.DataFrame` via `pd.DataFrame(data=item["data"], columns=item["columns"])` wrapped in a `TableBlock(df=df, caption=item.get("caption", "Table"), footnote=item.get("footnote"))`.
     - It instantiates `ReportDocument(title=finalized_title, blocks=reconstructed_blocks, results_dict=results_dict)` (using the serialized finalized title from the rendered report narrative or metadata rather than a generic replacement, ensuring any personal identifiers or numeric claims in the document title undergo automated Zero-PHI and numerical traceability scanning) and invokes `verify_report_integrity(doc)`. Delivery halts if `IntegrityReport.passed is not True`.
     - After `verify_report_integrity` consumes `<run_output_dir>/table_dfs_<run_id>.json`, remove both `<run_output_dir>/table_dfs_<run_id>.json` and `<run_output_dir>/table_dfs.json` (as well as any legacy `<output_dir>/table_dfs.json`, whenever present) from the output workspace.
   - **In `shinystat-cloud` (Standalone Cloud Sandbox)**: To keep the Main Agent's context window lean without loading the full `report-builder.md` file:
     - During final report assembly (§1.4), Subagent B writes `<run_output_dir>/report_narrative_<run_id>.txt` (serializing *every* rendered text block into text: headings, paragraphs, callouts, figure captions, figure alt-text, table captions, and table footnotes), `<run_output_dir>/table_dfs_<run_id>.json` (serialized as a JSON list of table dicts `[df.reset_index(drop=True).to_dict(orient="split") for df in table_dfs]` without non-rendered cohort indexes so patient identifiers cannot appear in `table_dfs_<run_id>.json`), and `<run_output_dir>/.scratch_checkpoint_<run_id>/verify_integrity.py`.
     - `<run_output_dir>/.scratch_checkpoint_<run_id>/verify_integrity.py` loads `<run_output_dir>/report_narrative_<run_id>.txt`, `<run_output_dir>/results_dict.json`, and `<run_output_dir>/table_dfs_<run_id>.json`, reconstructs every table entry into a `pandas.DataFrame` via `pd.DataFrame(data=item["data"], columns=item["columns"])` (exiting non-zero immediately if `table_dfs_<run_id>.json` is missing, malformed, or any entry fails `isinstance(df, pd.DataFrame)` reconstruction), calls `verify_standalone_integrity` (from `report-builder.md §3`), and exits non-zero (`sys.exit(1)`) unless the returned result has `passed == True`, `len(untraced_numbers) == 0`, and `len(phi_violations) == 0`. Furthermore, the verification gate dynamically checks study design and claims: `has_causal_caveat == True` is strictly required when the study archetype is observational (e.g. Type 5: Observational Causal, Type 2: Etiologic Multivariable) or when causal language is detected in the narrative text, but is not falsely required for purely descriptive (Type 1) or non-causal prediction (Type 3) studies without causal claims.
     - The Main Agent executes `python <run_output_dir>/.scratch_checkpoint_<run_id>/verify_integrity.py` after final report assembly and treats a missing script or a non-zero exit code as a failed integrity check.
     - Retain `<run_output_dir>/table_dfs_<run_id>.json` until the Main Agent completes the independent table-to-results numerical reconciliation audit below; remove both `<run_output_dir>/table_dfs_<run_id>.json` and `<run_output_dir>/table_dfs.json` (as well as any legacy `<output_dir>/table_dfs.json`, whenever present) from the output workspace only after the reconciliation audit completes.
   - **Independent Table-to-Results Numerical Reconciliation Audit**:
     - In both local and cloud verification paths, passing automated narrative checks is necessary but not sufficient.
     - The Main Agent executes an independent audit reconciling every numerical value displayed in table cells (`TableBlock`s / `table_dfs`) directly against `results_dict`.
     - Every effect estimate, odds ratio ($\text{OR}$), hazard ratio ($\text{HR}$), risk ratio ($\text{RR}$), $95\%$ confidence interval bound, $P$-value, and cohort retention/sample count displayed in tables must match `results_dict` within numerical precision tolerance.
     - If this audit detects any mismatch—including altered odds ratios, transposed CI limits, or divergent counts—delivery must HALT immediately, even if narrative traceability and Zero-PHI scans passed without error.
   - **Mandatory Manual Review for Unprefixed Patient Names**:
     - Automated regex scanning in `verify_report_integrity` and `verify_standalone_integrity` detects standardized clinical identifiers (HN, Citizen ID, Phone, DOB, and title-prefixed names).
     - As specified in the Skill Completion Criteria (`SKILL.md §Completion Criteria`), passing automated integrity checks does **NOT** substitute for the clinician's manual review.
     - The Main Agent must perform an explicit manual review of the finalized narrative text, callout boxes, figure annotations, and table cells to confirm zero unprefixed patient names (e.g. standalone first/last names without honorifics or staff names mistaken for patients). Delivery halts if any unredacted personal identifier is found.
3. **Scratch Row-Level Checkpoint Cleanup & Ownership Verification**:
   - Across all terminal outcomes—(a) successful completion (`passed == True`), (b) verification failure or unrecoverable halt, and (c) abandoned or superseded resumes—verify `run_id` ownership before removing any staging artifacts.
   - Selectively `unlink` and remove the exact external row-level checkpoint directory returned by `mkdtemp` (`<row_level_checkpoint_dir>/cleaned_cohort.parquet` / `.csv` and directory `<scratch_dir>/shinystat_ckpt_<run_id>_*`), `<run_output_dir>/table_dfs_<run_id>.json` and `<run_output_dir>/table_dfs.json` (as well as any legacy `<output_dir>/table_dfs.json`, whenever present), and the isolated `<run_output_dir>/.scratch_checkpoint_<run_id>/` staging directory belonging strictly to that `run_id`, ensuring concurrent or adjacent runs in shared directories cannot overwrite or delete each other's state and row-level checkpoints cannot remain in deliverables or on disk.

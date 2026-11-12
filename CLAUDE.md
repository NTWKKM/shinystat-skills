# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`medstat` (`src/medstat/`) is a headless clinical biostatistics engine and CLI. `skills/` ships it to agents twice: `shinystat` bundles the engine, and `shinystat-cloud` re-implements it as 24 standalone recipes for sandboxes without `medstat`. Three relations hold the repository together, each with a word used below:

- **canonical → mirror**: files copied by `scripts/sync-skills.sh`.
- **parity**: the engine and its recipe compute the same number.
- **fail closed**: undefined or ambiguous input raises, or returns `NaN` with a reason.

## Done when

A change is done when every line that applies to it holds:

- `./scripts/sync-skills.sh --check` passes, and each mirror change sits in the same commit as its canonical source.
- Engine and recipe are in parity for the behaviour changed, with a test in `tests/unit/test_reference_parity.py` or `tests/unit/test_shinystat_cloud.py` that fails without the change.
- `uv run --extra dev pytest` is green.
- `uv run ruff check src tests scripts` is clean.
- The contract describing the changed behaviour states what the code now does (see Contracts).

## Commands

```bash
uv run --extra dev pytest                   # full suite; CI sets MPLBACKEND=Agg
uv run pytest tests/unit/test_reference_parity.py::test_firth_logistic_matches_r_logistf   # one test
uv run ruff check src tests scripts
./scripts/sync-skills.sh                    # canonical -> mirror
./scripts/package-cloud-skill.sh            # packaging/cloud/shinystat-cloud.zip
UPDATE_RECIPES_SNAPSHOT=1 uv run pytest tests/unit/test_recipes_snapshot.py   # re-baseline recipe outputs
```

- Select tests by path or `-k`: the markers in `pyproject.toml` are applied to no test.
- R oracle tests live in `tests/unit/test_reference_parity.py`; `tests/benchmarks/` holds their CSVs and the R script only.
- Format only the lines you change: `ruff format` is unenforced and the tree is not format-clean.
- `FutureWarning` and `DeprecationWarning` raised from `medstat.*` fail the test that triggers them.

## Canonical and mirror

Edit the canonical file, then run `./scripts/sync-skills.sh`. Mirrors are committed copies (ADR 69).

| Canonical | Mirror |
| :--- | :--- |
| `src/medstat/` | `skills/shinystat/scripts/medstat/` |
| `skills/_shared/references/*.md` | `skills/shinystat/references/` and `skills/shinystat-cloud/references/` (cloud keeps its own `figures.md`) |

Canonical in place, with no upstream: both `SKILL.md` files, `skills/shinystat-cloud/references/figures.md`, `skills/shinystat-cloud/references/recipes/`, `skills/shinystat/references/api-quickref.md` (pinned to engine signatures by a test), `skills/shinystat/scripts/medstat_cli.py`, `skills/shinystat/requirements.txt`.

- pytest writes mirrors: several tests call `package-cloud-skill.sh`, which syncs first. A drifted tree returns from a test run with `skills/` modified.
- CI pushes a sync commit onto a drifted same-repo branch, and fails a drifted fork PR or `v*` tag. Sync before pushing so CI has nothing to add.
- `.agent/`, `.agents/`, `.cursor/` and `packaging/` are gitignored output of `scripts/install-skills.sh` and `package-cloud-skill.sh`. Regenerate them with those scripts.

## Parity

Each estimator, variance formula, validation rule and fallback lives in `src/medstat/` and again in a recipe under `skills/shinystat-cloud/references/recipes/`. Change both in one commit (ADR 75).

Recipe markdown is executable: tests `exec` its fenced `python` blocks and compare the results with `tests/fixtures/recipes_baseline_snapshot.json` and with the engine. A `python` fence there therefore runs standalone on pandas, scipy, statsmodels, lifelines and scikit-learn. An intended numerical change re-baselines the snapshot and records the reason in its ADR.

## Prose is tested

`tests/unit/test_shinystat_skill.py` and `tests/unit/test_shinystat_cloud.py` assert on the wording of `SKILL.md`, the reference files, `README.md` and `DESIGN.md`. Run both after editing any of them.

`DESIGN.md` headings are `## ADR N: Title`, sequential from 1, each with `### Context`, `### Decision`, `### Consequences`. A review round on an existing change appends a `Round N` bullet to that ADR's Consequences.

## Contracts

- `ARCHITECTURE.md`, module seams: read before adding a module, moving a function between `medstat.*` packages, or adding a recipe or reference file.
- `CONTEXT.md`, glossary, entity models and display precision: read before naming a concept, changing a result structure, or formatting a number.
- `DESIGN.md`, the reason behind each statistical choice: grep `ADR N` or the method name before changing an estimator or a default. Add or extend an ADR when behaviour changes.

## Engine

`data → models | diagnostic | causal | meta | agreement | power → figures → reporting.ir → reporting.integrity → reporting.renderers`

- `ReportDocument` (`reporting/ir.py`) is the renderers' only input. Computation happens upstream; a renderer formats.
- Every number in narrative text traces to `results_dict` (`reporting/integrity.py`). Add the value there with the sentence that states it.
- Each static figure in `figures/` has a `plot_dual_*` counterpart in `figures/html/`. `tests/unit/test_all_figures_matrix.py` runs every figure family against the five journal house styles.
- PHI patterns live in `_phi.py` alone; figures and reporting import them.
- `cli/spec.py` validation errors carry `agent_remediation` hints that agents act on. Update the hint together with the option it describes.

## Fail closed

Engine and recipes fail closed, and tests hold them to it:

- Undefined inference (zero variance, degenerate bootstrap, an absent class) returns `NaN` with a reason.
- An unrecognised option or effect scale raises. A fallback that does run is disclosed in the result (`fallback_reason`, `ci_methods`).
- Missing data without a strategy and justification raises `MissingStrategyRequiredError`. Rows missing the outcome or score are excluded and counted.
- Outcomes and event indicators are numeric 0/1 (competing risks: 0 censored, 1 event of interest, 2+ competing). Unmapped text labels raise.
- Flow diagram counts reconcile or raise.

Fixed methods: Wilson CIs for proportions, DeLong variance for AUC, Firth with `penalty_weight=0.5`, IPTW weights as probability weights with sandwich variance. Dependencies stay permissively licensed (ADR 1 replaced GPL `pingouin`). Fixtures and examples use synthetic data with mock IDs such as `PATIENT_MOCK_ONC_001`.

# AGENTS.md

- Read `CLAUDE.md` before the first edit in this repository: it defines `Done when`, canonical → mirror, parity, fail closed, and the tests that guard skill and ADR prose.
- Run ad-hoc Python snippets against local modules with `PYTHONPATH=src uv run python - <<'PY'` (`pyproject.toml` sets `pythonpath = ["src"]` for `pytest` only).
- Target a single test file during iteration (`uv run pytest tests/unit/test_<name>.py`), then finish with `uv run ruff check src tests scripts`, `uv run --extra dev pytest`, and every remaining line of `Done when` in `CLAUDE.md`.
- Locate relevant sections of `DESIGN.md`, `ARCHITECTURE.md`, and `CONTEXT.md` via `rg -n` and open only the matching slice; `Contracts` in `CLAUDE.md` says which file answers which question.

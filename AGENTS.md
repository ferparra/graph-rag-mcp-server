# Repository Guidelines

## Project Structure & Modules
- `src/`: Core code (e.g., `mcp_server.py`, `unified_store.py`, `semantic_chunker.py`, `fs_indexer.py`, `entry_points.py`).
- `tests/`: Unit, integration, e2e, eval, and benchmarks with pytest markers (`unit`, `integration`, `e2e`, `eval`, `slow`).
- `scripts/`: Ops utilities (`reindex.py`, `reindex_watch.py`, `enrich_para_taxonomy.py`, `run_tests.py`).
- `configs/`, `docs/`, `examples/`, `rules/`, `rule-tests/`, `utils/`: Supporting assets and examples.
- `.env` / `.env.example`: Local configuration (do not commit secrets).

## Build, Test, and Development
- Install deps: `uv sync` (requires `uv`).
- Run installer: `uv run install.py` (detects clients, configures env).
- Start MCP (stdio): `uvx --python 3.13 --from . graph-rag-mcp-stdio`.
- Start MCP (HTTP): `uv run graph-rag-mcp-http --port 8765`.
- Index vault: `uv run scripts/reindex.py all` (status: `status`).
- Tests (fast): `uv run scripts/run_tests.py fast -v`.
- Tests (all + coverage): `uv run scripts/run_tests.py all --coverage`.
- Lint/typecheck tests: `uv run scripts/run_tests.py lint` / `typecheck`.

## Codebase Tools (sg / ast-grep)
- Config: `sgconfig.yml` at repo root; rules in `rules/`, tests in `rule-tests/`, utils in `utils/`.
- Search by AST pattern (Python):
  - `sg run -l python -p 'def $NAME($ARGS): $BODY' src`
  - `sg run -l python -p 'class $C { $BODY }' src`
- Scan via config: `sg scan` (uses `sgconfig.yml` globs and rule dirs).
- Test rules: `sg test` (runs tests under `rule-tests/`).
- Tips: prefer `sg` over `rg` for structural queries; pass a custom config with `sg -c sgconfig.yml` if needed.
- Sample rule: `rules/no-print.yml` flags Python `print()`; run its tests with `sg test` or scan with `sg scan --filter no-print`.

## Coding Style & Conventions
- Python 3.13, 4-space indentation, type hints required for new/changed code.
- Naming: `snake_case` for modules/functions, `PascalCase` for classes, `UPPER_CASE` for constants.
- Pydantic models for data contracts; prefer explicit `@dataclass`/models over dicts at boundaries.
- Keep functions small and pure; log at INFO in server paths; raise typed errors.
- Quality gates via `pytest-flake8` and `pytest-mypy` (see Testing).

## Testing Guidelines
- Framework: `pytest` with markers (`unit`, `integration`, `e2e`, `eval`, `slow`, `requires_chroma`, `requires_gemini`, `requires_vault`).
- Coverage: threshold 80% (`pytest.ini`).
- Naming: files `tests/**/test_*.py`; focused, deterministic tests; use fixtures in `tests/conftest.py`.
- Examples:
  - Unit: `uv run python -m pytest tests/unit -m unit -q`
  - Integration: `uv run python -m pytest tests/integration -m integration -q`
  - Collection only: `uv run python -m pytest --collect-only -q`

## Commit & Pull Requests
- Conventional Commits: `feat:`, `fix:`, `refactor:`, `chore:`, etc. (see git history).
- PRs must include: clear description, linked issues, test coverage for changes, and updates to `README.md`/`SETUP.md`/`docs/` if behavior changes.
- Before opening PR: `uv run scripts/run_tests.py all --coverage` and ensure ≥80% coverage, no flake8/mypy failures.

## Security & Configuration
- Secrets: never commit API keys; use `.env` (copy from `.env.example`).
- Required env: `GEMINI_API_KEY`, vault paths; optional Chroma settings (`OBSIDIAN_RAG_*`).
- Local data: ChromaDB at `.chroma_db/` by default—safe to delete/rebuild when reindexing.

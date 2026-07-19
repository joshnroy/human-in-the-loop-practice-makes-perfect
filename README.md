# human-in-the-loop-practice-makes-perfect
Ask Josh for access to the notion page: https://app.notion.com/p/joshnroy/Human-in-the-Loop-Practice-Makes-Perfect-37133470fbc580aab736c283e49ee5db?source=copy_link

## Setup

Uses the `hitl-pmp` conda environment (Python 3.10).

```bash
conda activate hitl-pmp
pip install -e ".[dev]"
```

## Structure

- `src/hitl_pmp/` — core, reusable, tested library code
- `tests/` — unit tests for `hitl_pmp` (mirrors `src/hitl_pmp/`)
- `analysis/` — scripts/notebooks that use `hitl_pmp` to produce results and figures for the project

`analysis/` imports from `hitl_pmp`; `hitl_pmp` never depends on `analysis/`.

## Checks

```bash
pytest              # tests
ruff check .        # lint
ruff format .       # format
mypy src            # typecheck
```

All three run in CI (`.github/workflows/ci.yml`) on every push/PR to `main`.
Optionally, run `pre-commit install` to run lint/format/typecheck locally before each commit.

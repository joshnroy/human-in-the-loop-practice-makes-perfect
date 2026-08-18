# human-in-the-loop-practice-makes-perfect

Design doc and paper notes live in Notion — ask Josh for access.

## Setup

```bash
conda activate hitl-pmp          # Python 3.10
pip install -e ".[dev]"
```

Or prefix any command with `scripts/with_env.sh`, which resolves the environment itself.
Fast Downward and the KINDER simulator need extra steps — see `CLAUDE.md`.

## Structure

- `src/hitl_pmp/` — the library. `core/` holds the abstract interfaces; `environments/`,
  `methods/`, `humans/`, `planning/`, `recording/` and `results_writer/` hold concrete
  implementations.
- `tests/` — mirrors `src/hitl_pmp/` 1:1.
- `analysis/` — post-run scripts that read a run's `--output-dir` back in and produce
  figures and tables. `analysis/` imports from `hitl_pmp`; never the reverse.
- `scripts/` — operational entrypoints that *drive* runs, notably `run_sweep.py`.
- `docs/experiment-logs/` — dated, permanent records of experiments actually run.

## Running

```bash
python -m hitl_pmp.cli --env lightswitch --method skill-oracle --output-dir out/
```

Both `--env` and `--method` are required; all flags are named.

## Checks

```bash
pytest ; ruff check . ; ruff format --check . ; mypy src ; lint-imports ; scripts/check_doc_links.sh
```

All of these run in CI on every push and PR to `main`. `lint-imports` must print
`Contracts: 1 kept, 0 broken.` Conventions, traps and process rules are in `CLAUDE.md`.

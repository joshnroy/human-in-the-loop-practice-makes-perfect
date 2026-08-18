---
name: hitl-env
description: Set up and verify the hitl-pmp environment - conda activation, FD_EXEC_PATH, the PYTHONPATH worktree trap, and the five-check verification gate. Use before running any Python, pytest, mypy, ruff, lint-imports, or sweep command in this repo.
when_to_use: Before the first Python/pytest/mypy/ruff/lint-imports command in a session or worktree; when an import resolves to the wrong checkout; when a ModuleNotFoundError appears; before pushing a branch; before launching a sweep.
---

# hitl-pmp environment and verification

## 1. Setup

**Prefix every command with `scripts/with_env.sh`.** That is the whole setup:

```bash
scripts/with_env.sh pytest
scripts/with_env.sh python -m scripts.run_sweep --env lightswitch ...
```

It activates the `hitl-pmp` conda env, sets `FD_EXEC_PATH`, sets `PYTHONPATH` to **its
own** checkout's `src/` (derived from the script's location, never `$PWD`), exports
`MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl`, then `exec`s your command. Run it with no
arguments to print what it resolved.

**Why a wrapper.** An agent sandbox refuses `source …/conda.sh && conda activate` ("runs a
string through source") and `FD_EXEC_PATH=… cmd` ("too complex to verify"); a bare
`export` runs but **does not persist** — the next command gets a fresh shell. Plain `&&`
chaining works. The wrapper sets the environment inside the same process as your command,
which is why it works at all. Humans in an interactive shell can keep using
`conda activate` directly.

**Fallback if the wrapper is unavailable:** call the env's binaries by absolute path,
`/home/josh/miniconda3/envs/hitl-pmp/bin/python`. That gets you the right interpreter but
**not** `PYTHONPATH`, so §2's trap is live.

## 2. The PYTHONPATH trap — the load-bearing line

**A git worktree silently imports the *main* checkout's library unless `PYTHONPATH` is
set.** The editable install's `.pth` points at an absolute path: the main checkout's
`src/`. So a worktree can run its own `scripts/run_sweep.py` against a different
worktree's library and nothing errors — the run just measures the wrong thing. This has
already produced one near-invalid measurement.

Run this before trusting any result. The printed path **must** start with the directory
you are working in:

```bash
scripts/with_env.sh python -c "import hitl_pmp; print(hitl_pmp.__file__)"
```

`PYTHONPATH=src` (relative) appears in older transcripts. It works only while cwd is the
worktree root and breaks silently otherwise. Use an absolute path.

## 3. The verification gate

```bash
scripts/with_env.sh pytest ; scripts/with_env.sh ruff check . ; scripts/with_env.sh ruff format --check . ; scripts/with_env.sh mypy src ; scripts/with_env.sh lint-imports ; scripts/check_doc_links.sh
```

Use `;` rather than `&&` so one failure does not hide the others. Each check gets its own
`with_env.sh` — the wrapper `exec`s a single command, it is not a shell you stay inside.

`lint-imports` must print exactly `Contracts: 1 kept, 0 broken.` "0 kept, 1 broken" means
the `core` / `environments` / `methods` dependency direction was violated — a real
architectural error, not a lint nit.

**Budget.** The full suite is ~2 min 20 s; lint/typecheck/import-linter are cheap once
warm (mypy ~10 s cold). Run targeted tests while iterating
(`scripts/with_env.sh pytest tests/<path> -x`, or `-k <expr>`); run the full gate once
before pushing. Re-running everything *is* correct when chasing a failure whose blast
radius you do not know, or after touching `core/`. Never skip it entirely before reporting
done. "Push and let CI check it" is not cheaper — CI is ~4.5 min wall-clock with a slower
path back to a fix. Get the current test count with
`scripts/with_env.sh pytest --collect-only -q | tail -1` rather than trusting a written-down
number.

## 4. Known sharp edges

- **`ruff format` reformats Python inside Markdown** (verified on ruff 0.16.1). A
  ```` ```python ```` block in any `.md` file is formatted like a source file, so a
  deliberately-ugly snippet fails `ruff format --check .`. Use a `text`/`console` fence for
  code that must stay verbatim.
- **`pgrep -cf hitl_pmp.cli` self-matches** — the wrapper shell carries the pattern. Run
  `pgrep -af '[h]itl_pmp\.cli'` **directly**, not inside `$(...)`, which spawns another
  shell that reintroduces the self-match.
- **Plain `git commit` works.** `pre-commit` is not installed here. *If* someone installs
  it, its `lint-imports` hook is `language: system` and resolves from the ambient PATH, so
  it fails whenever the conda env is not active in the committing shell — only then, use
  `git commit --no-verify` and rely on the manual gate.

## 5. Running sweeps

`CLAUDE.md`'s "Running experiments" section carries the rules: fixed seeds through
`scripts/run_sweep.py`, `analysis/` is post-run only, always run inside
`systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue`, budget `--max-workers`
across *all* agents (~22 concurrent runs total), and populate the worktree's own
`reference/` so the sweep does not read submodules out of the shared checkout. Three
numbers that live only here:

- **~450 MB cgroup-charged per worker** in normal operation for the light domains;
  **~2.23 GiB per Tossing3D run**, about 5x what the tooling assumes. Size the cap
  accordingly: `16G` for a full sweep, `2-8G` for probes so a leak hits a wall in seconds.
- **Both printed `__file__`s must be under the worktree** after
  `scripts/update_reference_repos.sh`:
  `scripts/with_env.sh python -c "import kinder, kinder_models; print(kinder.__file__, kinder_models.__file__)"`.
  Populating `reference/` costs ~1.1 GB per worktree, so it buys isolation only for runs
  that need it. A sweep needs it; a unit-test run does not.
- **Every run is already timed** — `run_sweep` writes `timing.json` beside each
  `stats.json` (elapsed, exit status, and *two separate* concurrency signals: this sweep's
  in-flight child count and a machine-wide sample that includes other agents' runs; do not
  conflate them). Read it with `analysis/run_timing.py --results-root <dir>`, never by hand
  and never from directory mtimes. Do not add timing instrumentation — it is already there.

---
name: hitl-env
description: Set up and verify the hitl-pmp environment - conda activation, FD_EXEC_PATH, the PYTHONPATH worktree trap, and the five-check verification gate. Use before running any Python, pytest, mypy, ruff, lint-imports, or sweep command in this repo.
when_to_use: Before the first Python/pytest/mypy/ruff/lint-imports command in a session or worktree; when an import resolves to the wrong checkout; when a ModuleNotFoundError appears; before pushing a branch; before launching a sweep.
---

# hitl-pmp environment and verification

One source of truth for setup, the import hazard, and the verification gate. Preloaded
into the `hitl-experiment` agent type; also invocable directly as `/hitl-env`.

## 1. Environment setup

**Prefix every command with `scripts/with_env.sh`.** That is the whole setup:

```bash
scripts/with_env.sh pytest
scripts/with_env.sh ruff check .
scripts/with_env.sh mypy src
scripts/with_env.sh python -m scripts.run_sweep --env lightswitch ...
```

The wrapper activates the `hitl-pmp` conda env, sets `FD_EXEC_PATH`, and sets
`PYTHONPATH` to **its own** checkout's `src/` (derived from the script's location, never
`$PWD`, so it is right wherever you invoke it from), then `exec`s your command. Run it
with **no arguments** to print what it resolved, which doubles as the §2 sanity check:

```bash
scripts/with_env.sh
```

```text
REPO_ROOT     /…/worktrees/agent-…
conda env     hitl-pmp
python        /home/josh/miniconda3/envs/hitl-pmp/bin/python
PYTHONPATH    /…/worktrees/agent-…/src
FD_EXEC_PATH  /home/josh/Documents/repos/research/downward
hitl_pmp      /…/worktrees/agent-…/src/hitl_pmp/__init__.py
```

### Why a wrapper, and not the three `source`/`export` lines this used to say

Because an agent sandbox — the primary audience for this skill — **cannot execute them**.
Measured directly in a worktree-isolated sandbox:

| form | result |
| --- | --- |
| `source …/conda.sh && conda activate hitl-pmp` | **refused** — "runs a string through source, which can't be verified" |
| `FD_EXEC_PATH=… some-command` | **refused** — "too complex to verify" |
| `export PYTHONPATH=…` as its own command | runs, but **does not persist**: the next command gets a fresh shell and `printenv PYTHONPATH` exits 1 |
| `a && b` (plain chaining, no `cd`, no `source`) | works |

So the old block failed twice over: two of its three lines were refused outright, and the
one that ran was discarded before the next command. Every agent that hit this had to
invent the same wrapper for itself. Now it is in the repo.

Note the third row: **an `export` in one call is gone by the next**, so there is no
"run this once per shell" for an agent. The wrapper sets the environment inside the same
process as your command, which is why it works at all.

Humans in an interactive shell can keep using `conda activate` directly — the wrapper is
additive, not a replacement. If you are in a shell where the environment really does
persist, `conda activate hitl-pmp` plus the cwd-independent
`export PYTHONPATH="$(git rev-parse --show-toplevel)/src"` is still fine.

### Fallback if the wrapper is unavailable

Call the env's binaries by absolute path — verified to work under the same restrictions:

```bash
/home/josh/miniconda3/envs/hitl-pmp/bin/python -c "import hitl_pmp; print(hitl_pmp.__file__)"
```

This gets you the right interpreter but **not** `PYTHONPATH`, so §2's trap is live: check
the printed path before trusting anything.

## 2. The PYTHONPATH trap — the load-bearing line

**A git worktree silently imports the *main* checkout's library unless `PYTHONPATH` is set.**

The editable install's `.pth` file points at an absolute path: the main checkout's `src/`.
So in a worktree, `import hitl_pmp` resolves to someone else's code, while
`python -m scripts.run_sweep` loads the driver from cwd. A worktree can therefore run
its own driver against a different worktree's library, and nothing errors — the run just
measures the wrong thing. This has already produced one near-invalid measurement.

Measured in a fresh worktree on this machine:

```text
WITHOUT PYTHONPATH -> <main checkout>/src/hitl_pmp/__init__.py     # WRONG: the main checkout
WITH    PYTHONPATH -> <this worktree>/src/hitl_pmp/__init__.py     # correct
```

### Sanity check — run this before trusting any result

```bash
scripts/with_env.sh python -c "import hitl_pmp; print(hitl_pmp.__file__)"
```

(Bare `scripts/with_env.sh` prints the same line among the rest of the resolved
environment — see §1.)

The printed path **must** start with the directory you are working in. If it names a
different checkout, stop and fix `PYTHONPATH` before running anything else.

Note that `PYTHONPATH=src` (relative) appears in older transcripts. It works only while
cwd is the worktree root and breaks silently otherwise. Prefer an absolute path.

## 3. The verification gate

CI (`.github/workflows/ci.yml`) runs all of these on every push and PR to `main`. Run
the whole gate locally before pushing:

```bash
scripts/with_env.sh pytest ; scripts/with_env.sh ruff check . ; scripts/with_env.sh ruff format --check . ; scripts/with_env.sh mypy src ; scripts/with_env.sh lint-imports
```

Use `;` rather than `&&` so that one failure does not hide the others. Each check gets
its own `with_env.sh` because the wrapper `exec`s a single command — it is not a shell
you stay inside.

`lint-imports` must print exactly:

```text
Contracts: 1 kept, 0 broken.
```

"0 kept, 1 broken" means the `core` / `environments` / `methods` dependency direction was
violated — a real architectural error, not a lint nit.

## 4. Verification budget

The full suite takes **about 2 min 20 s**. Lint, typecheck and import-linter are cheap
once warm (~0.3 s total); `mypy` costs ~10 s cold in a fresh worktree.

An exact test count is deliberately **not** pinned here — it goes stale on every merge,
and a stale number in a skill is worse than no number, because the next agent trusts it.
Get the current one instead:

```bash
scripts/with_env.sh pytest --collect-only -q | tail -1
```

Across one measured session, 20 subagents ran the full suite 38 times — 28 of those were
repeats within a single agent, an estimated **55-68 minutes** of wall clock.

Guidance, not a prohibition:

- **While iterating**, run targeted tests:
  `scripts/with_env.sh pytest tests/<path>/test_<file>.py`, or
  `scripts/with_env.sh pytest -k <expr>`. Add `-x` to stop at the first failure.
- **Once, before pushing**, run the full gate above.
- Re-running the full suite *is* correct when you are chasing a real failure whose blast
  radius you do not yet know, or after a change that touches shared `core/` interfaces.
  Some of those 28 repeats were agents legitimately iterating. The waste is re-running
  the whole suite to check a change you already know is local.
- Never skip the full suite entirely before reporting done — CI will run it anyway, and
  finding out from CI costs more than finding out locally.

"Push and let CI check it" is **not** a cheap substitute for the local gate. CI is about
**4.5 min wall-clock** (lint 73-83 s, typecheck 102-115 s, test 252-281 s, run in
parallel) — roughly what the local gate costs. You pay the same wait, just less
immediately and with a slower path back to a fix.

## 5. Known sharp edges

- **`ruff format` reformats Python inside Markdown.** Verified on ruff 0.16.1: a
  ```` ```python ```` block in any `.md` file is formatted like a source file, so a
  deliberately-ugly snippet in a README will fail `ruff format --check .`. Run
  `ruff format` on Markdown files you edit, or use a non-`python` fence (`text`,
  `console`) for code that must stay verbatim.
- **Plain `git commit` works — but the `lint-imports` pre-commit hook is fragile on PATH
  if you enable it.** `pre-commit install` is optional and is **not** currently installed
  (`.git/hooks/pre-commit` does not exist, and commits here succeed without any flag), so
  do not reach for `--no-verify` by default.

  *If* you or someone else runs `pre-commit install`: `.pre-commit-config.yaml` declares
  `import-linter` as a `local` hook with `language: system`, so it resolves `lint-imports`
  from the ambient PATH. That binary exists only inside the `hitl-pmp` conda env, so the
  hook fails whenever the env is not active in the committing shell — which is common in
  an agent shell. In that situation, and only then, use `git commit --no-verify` and rely
  on the manual gate in §3, which is what CI runs anyway.
- **`pgrep -cf hitl_pmp.cli` self-matches.** The command's own wrapper shell has the
  pattern in its command line, so the naive form reports a phantom running job. Run
  `pgrep -af '[h]itl_pmp\.cli'` **directly** — not inside `$(...)`, which spawns another
  shell that reintroduces the self-match.

## 6. Running sweeps

Drive runs through `scripts/run_sweep.py` (fixed seeds, the
`<results-root>/<method>/<seed>/` layout `analysis/` globs for), never a hand-rolled
shell loop. `analysis/` scripts are post-run only: they read `--output-dir` output back
in and never drive a `Method` themselves.

### Always run a sweep in a memory-capped cgroup

```bash
systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
  scripts/with_env.sh python -m scripts.run_sweep --output-dir <dir> ...
```

This is not optional hygiene — it prevents a whole class of session loss. A tmux pane
runs in its own systemd scope, and both the system and user managers here have
`DefaultOOMPolicy=stop` (verified). Under that policy, when the kernel OOM-kills *any*
process in the unit, systemd tears down the **entire unit**. On 2026-08-03 one leaking
process reached ~48 GB and took down the whole session — two mid-flight agents plus hours
of sweep compute. The journal recorded
`tmux-spawn-….scope: The kernel OOM killer killed some processes in this unit`, then
`Failed with result 'oom-kill'` two seconds later; scope peak was 57.5 G memory and
7.9 G swap.

There is no headroom to absorb a spike: **swap is 100% consumed and all of it is tmpfs**
(process anonymous swap is 0.00 GB), so a spike goes straight to the OOM killer.

Running under `systemd-run --user --scope` with `OOMPolicy=continue` was verified
end-to-end to kill only the child (exit 137) while the shell and the tmux scope survive.

Sizing: about **450 MB cgroup-charged per worker** in normal operation. Use `16G` for a
full sweep, and **2-8 G for probes and one-off runs** so a leak hits a wall in seconds
rather than minutes. `systemd-run` is at `/usr/bin/systemd-run`.

### Budget concurrency across all agents, not per sweep

`scripts/run_sweep.py`'s `--max-workers` defaults to `os.cpu_count()`, which is **24 on
this machine — per sweep**. Two agents each taking the default put 48 runs on 24 cores.
At 28 concurrent runs each run gets 0.665 of a core: **1.4x slower with zero throughput
gain.**

- **Check before launching**, every time:

  ```bash
  pgrep -af '[h]itl_pmp\.cli'
  cat /proc/loadavg
  ```

  (Use this form, not `pgrep -cf hitl_pmp.cli` — see §5 on the self-match.)

- **Target roughly 22 concurrent runs in total across every agent on the box**, not per
  sweep. Treat ~22 as a working budget, not a measured optimum: it is derived from the
  saturation arithmetic above, **not directly observed**. Finding the true knee needs a
  quiet machine for ~90 minutes, which has not been done.
- **Pass `--max-workers` explicitly** whenever another agent is already running. Do not
  take the default by omission.
- **Yielding is always safe.** Concurrency has been measured *not* to perturb results:
  13,105 FD process observations showed max lifetime 0 s against a 10 s budget, and
  `stats.json` was byte-identical between a low-load probe and a 20-way concurrent sweep.
  Staggering costs wall-clock only, never validity — so when in doubt, take fewer workers.

### Every run is already timed — read it, do not re-measure it

`run_sweep` writes a **`timing.json` beside each run's `stats.json`**, inside that run's
own `--output-dir`. It records start and end time (timezone-aware ISO 8601, plus epoch
seconds), `elapsed_seconds` from `time.monotonic()`, the exit status, `--max-workers` and
the CPU count, and **two separate concurrency signals**: this sweep's own in-flight child
count, and a machine-wide sample (`hitl_pmp.cli` process count and 1-minute load average)
that includes every *other* agent's runs too. Do not conflate those two — regressing
wall-clock on the sweep-local count while the box was shared attributes someone else's
load to your `--max-workers`.

It is a separate file on purpose: `stats.json`'s byte-stability is what verifies a change
did not alter results, and a timestamp inside it would break that for every open PR.
Nothing in `timing.json` is ever an input to a reproducibility comparison.

Read it back with the post-run analysis script — never by hand, and never from directory
mtimes (they answer a different question: when a file was last written, not when its run
began):

```bash
scripts/with_env.sh python analysis/run_timing.py --results-root <dir>
scripts/with_env.sh python analysis/run_timing.py --results-root <dir> --per-run
```

So "how long does a run take, and does concurrency actually help?" is answerable from
recorded data. It prints `No timing.json found ...` for sweeps that predate the record.

**Do not add timing instrumentation to `run_sweep` — it is already there** (PR #56). An
earlier version of this skill listed its absence as a known limitation and suggested
adding it; that work has shipped.

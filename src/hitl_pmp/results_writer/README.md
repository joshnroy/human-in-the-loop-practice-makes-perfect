# `results_writer/` — pluggable observers of a run's results

One abstraction (`ResultsWriter`), one static list of concrete implementations
(`registry.RESULTS_WRITERS`), and one place the harness drives them
(`method_runner.py`). Adding an observer is **a class plus one line in the registry**;
the harness does not change.

| file | holds |
| --- | --- |
| `results_writer.py` | `ResultsWriter` — the ABC and its hook contract |
| `types.py` | `RunSummaryScalars`, `CheckpointScalars` — the flat payloads, derived from `Metrics` in one place |
| `registry.py` | `RESULTS_WRITERS` — the static list every run is offered |
| `run_progress.py` | `RunProgressWriter` — `progress.jsonl`, always on, no flag |
| `wandb_writer.py` | `WandbResultsWriter` — `--record-wandb`, the first concrete one |

## The hooks, and why there are only two

`method_runner.py` owns exactly two boundaries a generic observer can key on:

- **`record_checkpoint(metrics=)`** — an evaluation sweep just finished and was
  recorded, so `metrics.evaluations[-1]` is that sweep. This is the same boundary
  `progress.jsonl` and `competence_log.jsonl` already use, and the same
  `num_online_transitions` learning-curve x-axis. Per *sweep*, never per environment
  step: a step-level hook would be thousands of calls per run for no analytic gain.
- **`close(metrics=)`** — the run is over. Fired from a `finally`, matching
  `recording.LoopRecorder.close`, so a crashed run still flushes whatever it recorded
  and no backend handle is left open.

Both are concrete no-ops on the ABC, so a writer implements only the boundary it cares
about. The single abstract method is `open_if_requested`, because interpreting the
writer's own condition is the whole of what makes one writer different from another.

## Opt-in is a writer's choice, not the mechanism's

`open_if_requested` asks each registered writer *does this run want you*. A flag is one
way to answer that and not the only one:

| writer | condition | behaviour with no flags at all |
| --- | --- | --- |
| `WandbResultsWriter` | `--record-wandb` **and** `--output-dir` | declines |
| `RunProgressWriter` | `--output-dir` alone | **writes** |

`progress.jsonl` is deliberately always on — instrumentation you have to remember to
switch on is not available for the run you did not expect to need it for, and it costs
a few hundred bytes and one `write` per sweep. So no flag was invented for it to fit
the method's name; the contract already admitted it.

**`open_if_requested` takes `num_cycles` as well as `args`** because that is the one
thing an always-on progress writer needs and `args` cannot supply. `sweeps_total` is
`num_cycles + 1`, and `num_cycles` is a *method-CLI decision*: `SkillOracleCli` passes
the literal `0` to `MethodRunner.run` and its namespace has no `num_cycles` at all,
while `PracticeCycleCli` declares a `--num-cycles` flag. A writer that re-derived it
with `getattr(args, "num_cycles", 0)` would be correct for both of today's methods by
coincidence and silently wrong — a wrong ETA denominator — for the first method that
computes its own. Writers that do not need it ignore it.

## Why a `ResultsWriter` is a real instance, not a static-method container

`core/README.md`'s dividing line is whether a class carries genuine per-run state. A
`ResultsWriter` does — `WandbResultsWriter` holds a live W&B run handle, and any
file-backed writer holds an open handle. That is the same reason
`CompetenceLogRecorder`, `EpisodeTraceRecorder` and `RunProgressWriter` are real
pydantic instances while `HumanOracle` and `Renderer` are static containers.

## Why it is not in `core/`

`episode_traces.py` states the rule this follows: *a recorder carries real per-run
state, and `core/README.md`'s existing precedent — `recording.LoopRecorder`, kept out
of `core` for the identical reason — is that a stateful recorder never crosses into
`core`; only plain data does.* So this package sits in `practice_loop.py`'s layer,
below `method_runner.py` and above `core`, pinned by the import-linter contract in
`pyproject.toml`. It imports `Metrics` and nothing else.

## What is *not* a `ResultsWriter`, and why

This is the boundary of the mechanism, not a to-do list. Three of the five existing
outputs are structurally out of reach of a harness-level hook, and the fourth must not
be movable at all.

| output | written by | why it is not a `ResultsWriter` |
| --- | --- | --- |
| `stats.json` | `method_runner.py`, end of run | It is the run's **product**, not an observation of it, and its byte-stability is what proves a change did not alter results. Nothing that can be added to a list should be able to move that write. |
| `config_snapshot.json` | `method_runner.py`, end of run | Same: provenance of the run, written unconditionally with `--output-dir`, and deliberately never-raising — the opposite of a writer's fail-loudly contract. |
| `timing.json` | `scripts/run_sweep.py`, **parent process**, after the child exits | No in-run hook can produce it. It measures the child's wall-clock *as observed from outside*, including process spawn, which nothing inside the run can see. |
| `sampler_draws.jsonl` | a `Method`'s sampler, via `methods/practice_makes_perfect/cli.py` | Its event is a sampler consultation deep inside a `Method`. The harness never sees one, so there is no hook here that could fire. |
| `episode_traces.jsonl` | `PracticeLoop._evaluate`, per evaluation episode | Its event is one *episode*, a level below the sweep boundary these hooks sit on. |

`progress.jsonl` (`run_progress.py`) was the one existing output that *does* fire at
exactly `record_checkpoint`'s boundary, and it **has since been adopted** — it is a
registered `ResultsWriter` and `method_runner.py` no longer names it. The concern that
kept it out at first was that an always-on output did not fit an opt-in shape; that
turned out to be about the method's *name* rather than its contract, which asks each
writer to decide for itself. The one genuine obstacle was `sweeps_total`, whose
`num_cycles` is not in `args`; `open_if_requested` now takes it. Its bytes are pinned
against a pre-adoption run in `tests/results_writer/test_run_progress.py`.

## `--record-wandb`

See `wandb_writer.py`'s module docstring for the full rationale. The short version:

- **Optional dependency**, imported lazily. `pip install -e ".[wandb]"`. CI does not
  install it; the package imports, typechecks and tests without it, and the
  `wandb`-backed tests gate on `importlib.util.find_spec("wandb")`.
- **Offline by default.** If `WANDB_MODE` is unset the writer passes `mode="offline"`,
  so no run blocks on the network, a credential-less machine still works, and a sweep of
  ~22 concurrent runs opens no sockets. Sync afterwards with
  `wandb sync <output-dir>/wandb/offline-run-*`.
- **One flag, not five.** Project, entity and group come from W&B's own environment
  variables (`WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_RUN_GROUP`), and
  `scripts/run_sweep.py` already forwards `os.environ` to every child, so one export
  configures a whole grid. The project defaults to `hitl-pmp` — one per repo, because
  cross-run comparison does not work across projects.
- **`wandb sweep`/`wandb agent` are deliberately not adopted.** Their agent draws its
  own hyperparameter values, which is directly at odds with this project's fixed-seed
  discipline; `scripts/run_sweep.py` keeps ownership of the grid and W&B's *grouping*
  gives what its sweeps would have.

### Two limits worth stating plainly

- **`stats.json` byte-identity is guaranteed and tested**
  (`tests/results_writer/test_wandb_writer.py`). `timing.json` is **not**: any W&B call
  costs wall-clock, so a `--record-wandb` run's `elapsed_seconds` is not strictly
  comparable to one without. Offline mode makes that small; it has not been measured.
- **W&B is an index and a viewer, never the system of record.**
  `docs/experiment-logs/` stays the durable, reviewed, committed record. A W&B run page
  is not citable six months out the way a committed log entry is, and
  `scripts/check_doc_links.sh` already bans a URL in a committed doc.

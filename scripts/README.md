# scripts

Operational entrypoints that *drive* runs, as opposed to
[`../analysis/`](../analysis/README.md), which strictly reads their output
afterwards. That split is the same one `CLAUDE.md` documents: an `analysis/`
script must never run a simulation, so anything that launches one belongs here.
Mirrors the sibling `hitl-practice` repo's own `scripts/` convention.

Most of these shell out to `python -m hitl_pmp.cli` rather than importing
`hitl_pmp` — `run_sweep.py` deliberately so, since a sweep is exactly a sequence
of CLI invocations and nothing more. `render_tossing3d_demo.py` is the exception
and says why in its own docstring: it needs an in-process handle on the live
simulator to install a per-tick frame sink, which no command line can express.
Either way `lint-imports`' contract is unaffected — its `root_packages` is
`["hitl_pmp"]`, so `scripts/` sits outside it entirely.

The shell-out rule is also one-directional: `analysis/run_timing.py` imports
`RunTiming` from `run_sweep.py` (reader ← writer), which is fine — the boundary
being protected is `scripts/` never reaching *into* the library uninvited.

## `with_env.sh`

Runs one command inside a fully set-up environment — the `hitl-pmp` conda env,
`FD_EXEC_PATH`, and a `PYTHONPATH` pointing at **this** checkout's `src/` —
then `exec`s it:

```bash
scripts/with_env.sh pytest
scripts/with_env.sh python -m scripts.run_sweep --env lightswitch ...
scripts/with_env.sh          # no command: print the resolved environment
```

It exists because the three-line `source`/`export` setup it replaces cannot be
run by an agent: a worktree-isolated sandbox refuses `source …` and `VAR=x cmd`
outright, and gives every command a fresh shell, so an `export` in one call is
gone by the next. Every agent that hit this wrote the same wrapper for itself.
Humans in an interactive shell can keep using `conda activate` directly — this is
additive.

`PYTHONPATH` is derived from the script's own location, never `$PWD`, so it stays
correct wherever it is invoked from. That is the point: a worktree that does not
set it silently imports the *main* checkout's library, because the editable
install's `.pth` file holds an absolute path — and nothing errors, so the run just
measures the wrong thing. `tests/scripts/test_with_env.py` pins it (skipped where
there is no conda, e.g. CI).

## `run_sweep.py`

Runs a (method × seed) grid in parallel and writes each run to
`<results-root>/<method>/<seed>/`, which is exactly the layout the `analysis/`
scripts glob for. Replaces hand-rolled shell loops, which were rewritten
per-experiment and therefore never reviewed, tested, or reproducible.

```bash
python -m scripts.run_sweep \
    --env lightswitch \
    --methods ees random-skills skill-oracle \
    --num-seeds 10 \
    --results-root results/ees \
    --shared-args "--grid-size 25 --num-test-tasks 10" \
    --method-args "ees=--num-cycles 10 --max-steps-per-interaction 150" \
    --method-args "random-skills=--num-cycles 10 --max-steps-per-interaction 150"
```

- `--shared-args` go to every run; `--method-args` (repeatable) go to one method
  only. The distinction is load-bearing rather than convenient: methods do not
  share a flag set, and `--method skill-oracle` rejects `--num-cycles` outright.
- `--max-workers` defaults to the CPU count. Each child is pinned to a single
  math thread (`OMP_NUM_THREADS=1`), since workers already run concurrently and
  letting each grab every core just oversubscribes. That pinning cannot change
  results — thread-count independence is pinned by
  `tests/scripts/test_reproducibility.py`.
- A failing run is reported, not raised: one bad seed must not abort the other
  29. The command exits non-zero if any run failed, and each run's stdout/stderr
  is saved to its own `log.txt`. That covers a run that never *started* too — a
  spawn that raises is reported as `returncode == -1` with the traceback as its
  output, rather than escaping and taking the sweep down with it.
- **Failures appear on stderr the moment they happen** — see below. If you are
  watching a sweep, watch stderr.
- `--max-spawn-attempts` (default 3) retries a run that could not be **launched**.
  See below — the distinction it draws is the important part.

## Watching a sweep: failures are reported live, on stderr

A sweep is 40–80 runs of ~40 minutes. If seed 3 fails two minutes in, whoever is
watching — a human or an agent monitoring the process — needs to see it *then* and
decide whether to cancel, not discover at minute 90 that 60 runs failed the same
way. So:

- **Failures and retry notices go to `stderr`; ordinary progress (`[ok]` lines,
  the startup banner, the final summary) goes to `stdout`.** Filter on stderr
  alone to see only what you might act on; `2>&1` still interleaves both in order
  for a human.
- **They are emitted from inside the worker, the instant that run finishes** —
  not when the sweep ends and not when its result is collected. This matters
  because `execute` collects with `list(executor.map(...))`, which yields in
  **submission order**: reporting at the consumption point would hold seed 3's
  failure until seed 0 finished. `test_a_failure_is_emitted_while_the_sweep_is_
  still_running` pins the ordering, not merely the presence of the line.
- **Everything is `flush=True`.** Python block-buffers stdout/stderr when they are
  not a TTY — exactly the case when an agent captures the output, and exactly when
  immediacy matters most. Without it a watcher sees nothing until the process
  exits.
- **One line per failure, not a traceback.** The notice carries the method, seed,
  returncode, a short reason (the child's last output line, which is where a
  Python traceback puts its exception and argparse its error), and the path to
  that run's `log.txt`. Sixty identically-failing runs should be sixty scannable
  notices; the full tracebacks stay in the per-run logs:

```
[FAILED rc=9] ees seed=3: RuntimeError: CUDA out of memory -- full output: results/ees/3/log.txt
```

None of this changes control flow. The sweep still runs to completion and still
exits non-zero at the end if anything failed — this is purely about *when and
where* the report shows up, so that cancelling early is a decision someone can
actually make.

## Retrying a failed *launch*, never a failed *run*

`subprocess.run` **raising** means the child never started. On this box the real
case is memory pressure making `fork()` raise `OSError`, and it is transient: a
sibling worker finishing frees gigabytes within seconds, so the same command
would very likely launch a moment later. Losing a ~40-minute run to that is pure
waste, so it is retried — 3 attempts by default, backing off 2s then 4s.
`--max-spawn-attempts 1` disables retrying entirely.

**A child that started and exited non-zero is never retried.** `--seed` fully
determines a run (`tests/scripts/test_reproducibility.py` pins this), so
re-running it would burn another ~40 minutes reproducing the identical failure. A
sweep with one genuinely broken seed must cost 1x, not 3x. Conflating the two
events is the expensive mistake here, and
`test_a_run_that_started_and_exited_non_zero_is_never_retried` pins that exactly
one spawn happens on that path.

Retries are recorded, never silent — a sweep that quietly retried a dozen times
is a machine-health signal someone needs to see:

- each retry prints a `[retry] <method> seed=N: spawn attempt i/3 raised OSError`
  line **to stderr, immediately**, and the run's own status line gains
  `(3 spawn attempts)`;
- every failed attempt's traceback (not just the last) is prepended to `log.txt`;
- `spawn_attempts` is recorded in `timing.json`;
- the sweep summary reports how many runs needed more than one spawn.

**Retrying cannot change results.** A retried attempt is the identical command
with the identical `--seed`, so a run that launches on attempt 2 produces exactly
what attempt 1 would have. The one measurement it perturbs is `elapsed_seconds`,
which includes the backoff by design — that is genuinely time the sweep spent —
so `spawn_attempts > 1` is how a wall-clock analysis identifies (and if wanted,
excludes) such a run.

## `timing.json`: how long a run took, and against how much load

Each run also writes `<results-root>/<method>/<seed>/timing.json` (a `RunTiming`)
recording start/end timestamps, elapsed wall-clock, exit status, and the machine's
concurrency. Before this existed there was no recorded run *start* time anywhere —
a directory is created ~100 ms before its `stats.json` is written, so mtimes were a
start-time proxy only by accident — and "how long does a run take, and does adding
concurrency help?" had to be re-measured by hand every time it was asked. Read it
back with [`analysis/run_timing.py`](../analysis/README.md).

**It is a separate file on purpose.** `stats.json` is the serialized `Metrics`, and
its byte-stability is how this project verifies that a change didn't alter results
(same seed → identical bytes). A timestamp inside it would break that check for
every PR that uses it and make runs non-reproducible by construction. Nothing in
`timing.json` is ever an input to a reproducibility comparison; exclude it by name.

**Per run, not per sweep.** One file per run survives an interrupted sweep or an
OOM (every run that already finished keeps its record), can't clobber another
sweep writing into the same results root, and needs no append logic. A sweep-level
summary would be purely derived — `sweep_id` already groups the records, and each
one carries the sweep's `--max-workers` and CPU count. Re-running a sweep
overwrites it, the same as `log.txt`.

**Two concurrency signals, deliberately not merged:**

| field | scope | meaning |
| --- | --- | --- |
| `sweep_runs_in_flight_at_start` / `_at_end` | this sweep only | how many of *this* sweep's children were running. A **lower bound** on real competition — other agents run their own sweeps on the same box. |
| `machine_at_start` / `machine_at_end` → `cli_processes` | whole machine | every running `hitl_pmp.cli`, this sweep's and everyone else's. This is the one to regress wall-clock against. |
| `machine_at_*` → `load_average_1min` | whole machine | `os.getloadavg()[0]`, a ~1-minute *damped* average: the start sample describes the machine the run **entered**, the end sample approximates what it **experienced**. |

Conflating the first two would attribute another agent's load to this sweep's
`--max-workers`. All four are point samples at run entry and exit, not
time-weighted averages; `--max-workers` bounds the sweep-local count in between.
The two counts are also on different scales: the sweep-local one **includes** the
run itself, the machine-wide one **excludes** it (its samples straddle the child,
which does not exist yet at the start and has exited by the end), so a reader
comparing them adds one — `analysis/run_timing.py` does.
`cli_processes` is `null` rather than `0` where there is no `/proc` (macOS) —
"unknown" and "none in flight" are different facts. Elapsed comes from
`time.monotonic()`, not from subtracting the timestamps, so an NTP step mid-run
can't corrupt it — though it *does* include any spawn-retry backoff, which is
what `spawn_attempts` (defaulted to 1, so records written before retrying existed
still read truthfully) is there to let you spot.

## `render_tossing3d_demo.py`

Renders one oracle episode on Tossing3D as a smooth GIF, using KINDER's own
rendering and GIF tooling rather than this repo's `core.Renderer` path.

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m scripts.render_tossing3d_demo \
    --output docs/tossing3d_skill_oracle_demo.gif --seed 0 --check-scene-bg
```

- **Why it is not `--output-dir`'s `episode.mp4`.** A `hitl_pmp` transition is a
  whole *skill* — several hundred MuJoCo control ticks — and `core.Renderer` is
  one frame per transition by construction, so that path can only ever produce a
  4-frame storyboard. This taps `KinderBackend.capture_frames_into` to render per
  tick instead, exactly as KINDER's `scripts/generate_demo_video.py` does.
- **It needs the `tossing3d` extra**, so run it from the KINDER virtualenv (see
  `src/hitl_pmp/environments/tossing3d/README.md`). It also wants `gifsicle` on
  `PATH` for `kinder.gif_utils.optimize_gif`; without it KINDER prints a warning
  and skips optimisation, which is a 12 MB GIF rather than 1.9 MB.
- `--check-scene-bg` re-runs the rollout with KINDER's plain scene and asserts the
  cube trajectory is bit-identical, i.e. that the background is purely cosmetic.
  It is: that is what licenses rendering the demo with it and sweeping without.

## Seeds are fixed

`--num-seeds 10` means seeds **0..9** exactly — never a random draw. A sweep has
to regenerate the same numbers when re-run months later, and the paper's protocol
("we run 10 random seeds of each approach") means a fixed set of seeds, not
randomly chosen ones.

One `--seed` integer determines a run's results completely: it seeds task
sampling (`LightSwitchTasks`), skill/parameter sampling (each `Method`'s own RNG),
and torch training. `tests/scripts/test_reproducibility.py` asserts this
end-to-end through the real CLI — same seed → identical `stats.json`, different
seeds → different `stats.json`, and neither depends on the math thread count.
It's tested at that level deliberately: any one component reaching for an
unseeded global would break reproducibility without breaking a narrower test.

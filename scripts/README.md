# scripts

Operational entrypoints that *drive* runs, as opposed to
[`../analysis/`](../analysis/README.md), which strictly reads their output
afterwards. That split is the same one `CLAUDE.md` documents: an `analysis/`
script must never run a simulation, so anything that launches one belongs here.
Mirrors the sibling `hitl-practice` repo's own `scripts/` convention.

Nothing here imports from `hitl_pmp` — these shell out to `python -m hitl_pmp.cli`
— so the package's own layering contract (`lint-imports`) is unaffected by them.
That rule is one-directional: `analysis/run_timing.py` imports `RunTiming` from
`run_sweep.py` (reader ← writer), which is fine — the boundary being protected is
`scripts/` never reaching *into* the library.

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

## `tossing3d_oracle_demo.py`

Renders KINDER's Tossing3D oracle rollout to a GIF, one per toss standoff, and
writes them to `docs/`. The two stock-config clips —
[`docs/tossing3d_oracle_standoff_1p35.gif`](../docs/tossing3d_oracle_standoff_1p35.gif)
and
[`docs/tossing3d_oracle_standoff_1p55.gif`](../docs/tossing3d_oracle_standoff_1p55.gif)
— are exactly what its defaults produce; a third,
[`docs/tossing3d_oracle_coincident_standoff_1p35.gif`](../docs/tossing3d_oracle_coincident_standoff_1p35.gif),
comes from `--task-config coincident-bin-goal` and is described further down.

| standoff | cube comes to rest | `_check_goals()` |
| --- | --- | --- |
| 1.35 | `x=2.2197 y=0.0103 z=0.0444` — **in the bin** (`bin_0` is at `x=2.2301`) | `False` |
| 1.55 | `x=2.0268 y=0.0105 z=0.0249` — bare floor, short of the bin | `True` |

Same seed, same skills, same parameters; only `move_to_target`'s standoff differs,
and every skill terminates on its own (71 / 23 / 16 / 18 steps in both). **The
throw that lands in the bin scores nothing and the one that misses scores**, because
`Tossing3D-o1`'s goal predicate is `["on", "cube_0", "blocks_goal_region"]` — a
ground region the bin merely sits near. The clips draw that region, so this reads
without the caption; each also carries its own measured numbers burned into the
frame, so one clip on its own still makes the point.

It drives a simulator, so it is here and not in `analysis/`. Unlike `run_sweep.py`
it does not shell out to `hitl_pmp.cli`: there is no CLI surface for KINDER at all
on `main` (`environments/tossing3d/` is not merged — see
[`docs/tossing3d-integration-status.md`](../docs/tossing3d-integration-status.md)),
and this deliberately depends on nothing but upstream, so it survives that work
landing, changing, or staying closed. It imports no `hitl_pmp` module.

```bash
# KINDER is an optional extra and is NOT in the hitl-pmp conda env.
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -p OOMPolicy=continue \
    /path/to/kinder-venv/bin/python scripts/tossing3d_oracle_demo.py --output-dir docs
```

Three things about running it are easy to get wrong, and each costs an hour:

- **The memory cap is not optional.** KINDER leaks roughly one PyBullet client and
  ~150 MB per skill execution; a kernel OOM on this box has taken a whole login
  session down before. Read the scope's cgroup `memory.max` back to confirm the cap
  is real rather than assuming the flag took.
- **The distribution is `kindergarden`; the import package is `kinder`.** The venv
  additionally needs `pydantic` (this repo's own convention, not KINDER's) and wants
  `gifsicle` on `PATH` — without it `kinder.gif_utils.optimize_gif` prints a warning,
  skips, and leaves a ~6 MB clip.
- **`MUJOCO_GL` must be `egl`, with some `DISPLAY` set, before KINDER is imported.**
  `register_all_environments()` rewrites `MUJOCO_GL` to `osmesa` when `DISPLAY` is
  unset, `import mujoco` then raises, `_check_deps` **swallows** it, and every
  Dynamic3D env silently vanishes into `NameNotFound` from `kinder.make`. The script
  handles this itself in `configure_headless_rendering`; the ordering is pinned by a
  test, because setting the variable after the import is a no-op.

### The goal region is drawn, using KINDER's own mechanism

The clips shade `blocks_goal_region` in translucent blue. Without it the central
finding is invisible: you watch a cube land squarely in a bin and are told that
scored a failure, with nothing on screen to show where the goal actually was. With
it, the goal box (`x ∈ [1.8500, 2.1500]`) and the bin footprint (`x ∈ [2.0801,
2.3801]` at seed 125) are both in frame, their **7.0 cm overlap on x** is visible,
and so is the fact that the cube comes to rest at `x = 2.2197` — past the far edge
of the goal box, inside the bin. On the coincident config the same overlay reads the
opposite way: the blue box *fills* the bin. `--no-goal-region` renders clean.

The caption's legend line carries both measured brackets — the goal box and the bin
footprint — so the two clips of standoff 1.35 can be compared number for number
without leaving the frame.

**This is not an overlay painted onto the image.** KINDER creates a box *site* for
every region at construction and calls `visualize_regions()` unconditionally
(`envs.py:455` for ground regions, `:542`/`:568` for fixtures). The sites exist and
are rendered; they are invisible only because their alpha is `0` — either the
`[1, 0, 0, 0]` default (`objects/base.py:370` and `:900`) or, for this scene's goal
region, a `[0.2, 0.6, 0.2, 0.0]` set in the task JSON. So the first-class route is an
`"rgba"` key on that JSON entry, and it is unavailable here: the task JSON is under
`reference/`, a third-party checkout this repo reads and never modifies. The script
writes the same value into the compiled `MjModel` instead, **after** `env.reset()` —
reset rebuilds the scene XML and recompiles the model, so an earlier write is
discarded with the model it was made on.

It is cosmetic by construction (a site is a massless, collision-free marker) and
that was checked rather than assumed: with the overlay on, both standoffs reproduce
their rest positions, their `_check_goals()` values and their 71 / 23 / 16 / 18 skill
steps exactly.

The site is located by reading the model's site names back and matching, never by a
hardcoded string — the name is assembled as `{fixture}_{region}_region_{index}` deep
inside upstream's XML builder, and five of this scene's six sites are regions, so the
match has to discriminate. It resolves to `ground_blocks_goal_region_region_0`, and
the caption's `x ∈ […]` bracket is read off that same site's `pos`/`size`, so the
shaded box and the printed numbers cannot drift apart. Zero matches, or more than
one, raises with every site name listed.

### `--task-config coincident-bin-goal`: put the bin back on the goal region

`--task-config` picks the scene. `stock` (the default) overrides nothing — KINDER
loads the task JSON it registered, so a stock run is byte-identical to one from
before the flag existed, which is what keeps the two committed stock clips
regenerable. `coincident-bin-goal` points KINDER at
[`scripts/task_configs/Tossing3D-o1-coincident.json`](task_configs/Tossing3D-o1-coincident.json)
through `ObjectCentricRobotEnv`'s own `task_config_path` parameter
(`envs.py:79`, which takes an absolute path verbatim at `:101-105`). Nothing under
`reference/` is modified.

The file is upstream's `Tossing3D-o1.json` with **exactly one line different**:

```
bin_init_region.ranges:  [[2.23, -0.0005, 2.231, 0.0005]]  ->  [[2.0, -0.0005, 2.001, 0.0005]]
blocks_goal_region:      byte-identical, untouched
```

**x = 2.0 is upstream's own number.** `Tossing3D-o2.json` already ships the bin
there, beside a `blocks_goal_region` byte-identical to o1's — the two files'
`blocks_goal_region` entries hash to the same
`53219d0f7a5b54781668539a848a27a2c8ae5fce9c69aed385f7f93987fe2735`. So this is a
pairing upstream publishes, not one invented here; the bin drifted away from the goal
region in upstream commit `1183de7` and o2 never followed it. Moving the *bin* rather
than the *goal region* also leaves the region that scores untouched, so every number
ever measured against this domain stays comparable.

It is **opt-in and constrained**. An unknown `--task-config` value is rejected by
`argparse` at parse time; `resolve_task_config` raises on one anyway; and
`require_task_config_applies` raises if the config is paired with an `--env-id` it
was not measured on, rather than silently doing nothing. There is exactly one
non-stock config and it is the one with provenance.

#### It is verified against live geometry, not JSON literals

Neither box is where the JSON says it is: a ground region's range is inflated by
`ground_placement_threshold = 0.05` per side before it becomes a region, and the bin's
placement is *sampled* from a 1 mm-wide `bin_init_region`, so its footprint shifts
slightly per seed. Every run therefore measures both from the compiled model — the
goal box from the live sim-backed `Region.bbox`, the bin from the world positions of
the five box geoms hanging off body `bin_0` (upstream's `Bin` builder sets no geom
names, so they can only be found through the body):

| | seed | goal box x | bin footprint x | overlap | worst edge off by |
| --- | --- | --- | --- | --- | --- |
| stock | 0 | [1.850000, 2.150000] | [2.080813, 2.380813] | 0.0692 m | 0.2308 m |
| stock | 125 | [1.850000, 2.150000] | [2.080101, 2.380101] | 0.0699 m | 0.2301 m |
| coincident | 0 | [1.850000, 2.150000] | **[1.850813, 2.150813]** | 0.2999 m | **0.0008 m** |
| coincident | 125 | [1.850000, 2.150000] | **[1.850101, 2.150101]** | 0.2999 m | **0.0001 m** |

The goal region is fixed, so it does not move with the seed; the bin does, which is
why both seeds are recorded rather than one constant. `verify_coincidence` re-checks
this on every coincident run and **raises** if the worst edge ever exceeds 5 mm — a
bound the measurements clear by a factor of 6, and that stock misses by 46×.

#### `--sweep`, and the standoff that shows the fix

Moving the bin changes the **physics**, not only the scoring: the bin now sits where
the cube used to land, so it is an obstacle in the flight path and the standoffs that
solved before are not the ones that solve now. `--sweep` runs the identical physics
but records no frames and writes no file, which is how a standoff gets chosen —
rendering dominates the per-rollout cost and a search discards every frame anyway.

Over 11 standoffs on the coincident config, **6/11 solve**:

| standoff | cube at rest | where | `_check_goals()` |
| --- | --- | --- | --- |
| 1.20 | `x=2.1064 z=0.0440` | in the bin | `True` |
| 1.25 | `x=2.0902 z=0.0443` | in the bin | `True` |
| 1.30 | `x=2.0402 z=0.0444` | in the bin | `True` |
| 1.35 | `x=1.9902 z=0.0444` | in the bin | `True` |
| 1.40 | `x=2.0952 z=0.0442` | in the bin | `True` |
| 1.425 | `x=2.0345 z=0.0443` | in the bin | `True` |
| 1.45 | `x=1.6870 z=0.0249` | bare floor | `False` |
| 1.50 | `x=1.7623 z=0.0248` | bare floor | `False` |
| 1.55 | `x=1.8243 z=0.0249` | bare floor | `False` |
| 1.60 | `x=1.7739 z=0.0249` | bare floor | `False` |
| 1.65 | `x=1.7238 z=0.0249` | bare floor | `False` |

The solving band `[1.20, 1.425]` is contiguous and 0.225 m wide, so this is not a
knife-edge that happened to be found. Note what the two columns say together: on this
config, **11/11 rollouts agree — every cube that came to rest in the bin scored, and
every one that did not, did not.** The five failures all bounce off the bin's near
wall (now at `x ≈ 1.8601`) and land short of it.

Standoff **1.35** is the committed one, because it lands the cube in the bin on *both*
configs and so renders the whole finding as one pair of clips:

| config | cube at rest | `_check_goals()` | clip |
| --- | --- | --- | --- |
| stock | `x=2.2197 y=0.0103 z=0.0444` (bin at `x=2.2301`) | `False` | `tossing3d_oracle_standoff_1p35.gif` |
| coincident | `x=1.9902 y=0.0105 z=0.0444` (bin at `x=2.0001`) | `True` | `tossing3d_oracle_coincident_standoff_1p35.gif` |

In both the cube lands about a centimetre short of the bin's own centre, on the bin's
interior floor (`z = 0.0444` against a `0.0249` start height, in both). Same skills,
same parameters, same seed, same 71 / 23 / 16 / 18 step counts. The only difference is
which task JSON was loaded — and the verdict flips.

```bash
# sweep (no rendering, no output files)
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -p OOMPolicy=continue \
    /path/to/kinder-venv/bin/python scripts/tossing3d_oracle_demo.py \
    --sweep --task-config coincident-bin-goal --standoffs 1.20 1.25 1.30 1.35 1.40 1.425

# the committed coincident clip
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -p OOMPolicy=continue \
    /path/to/kinder-venv/bin/python scripts/tossing3d_oracle_demo.py \
    --output-dir docs --task-config coincident-bin-goal --standoffs 1.35
```

### `--camera task_view`, not `agentview_1`

Upstream's demo (`reference/kindergarden/scripts/generate_demo_video.py`) sets
`agentview_1`, but only `if "TidyBot" in env_id` — and this env id is
`kinder/Tossing3D-o1-v0`, which does not contain `TidyBot`. That camera is not in
this scene's `camera_names` at all (`frontview`, `birdview`, `agentview`,
`sideview`, `task_view`, `robot_base`, `robot_wrist`), and `set_render_camera`
stores the name without validating it, so it renders a near-static shot of a wall:
6/32 sampled frames unique, against 32/32 for `task_view`. The script now rejects
an unknown camera name outright rather than rendering the wrong thing quietly.

`task_view` is the camera Tossing3D's own task config defines, and it is the only
one of the seven that frames the task. Its limitation, since it is a fixed camera:
the bin sits at the right-hand edge, so the cube's landing is at the frame border.
The pick, the drive to the barrier, the swing and the cube in flight are all clearly
visible.

`--scene-bg` (on by default) is what makes this the MimicLabs `lab2` scene the task
JSON names, and is what the ~1 GB asset download is for; `--no-scene-bg` renders a
scene literally named `simple` — a bare ground plane, upstream's *unit test* setting,
useful only as a fast smoke test. Physics is unaffected by either. fps comes from
`env.metadata["render_fps"]` (20 here), never a hardcoded value.

### Why the clips are 64 frames and a 128-colour palette

They are **committed**, so their size is a review concern rather than a preference.
The rollout is 128 steps of a wood-textured lab floor, so nearly every pixel changes
every frame and GIF's inter-frame compression buys almost nothing: at `--every 1`
with upstream's own `--colors 256 --lossy 80` the optimised clip is 3.9 MB, against
a whole-repo `.git` of 26 MB. Keeping every 2nd frame and halving the playback fps
(so the clip still runs at real speed) is the dominant saving, and the palette does
the rest:

| clip | raw | optimised |
| --- | --- | --- |
| stock, standoff 1.35 | 6,360,661 B | 1,471,267 B |
| stock, standoff 1.55 | 6,343,483 B | 1,573,497 B |
| coincident, standoff 1.35 | 7,840,225 B | 1,583,009 B |

Upstream's exact settings are still one flag away
(`--every 1 --colors 256 --lossy 80`).

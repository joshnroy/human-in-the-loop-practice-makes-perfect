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
  is saved to its own `log.txt`.

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
can't corrupt it.

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

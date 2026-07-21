# scripts

Operational entrypoints that *drive* runs, as opposed to
[`../analysis/`](../analysis/README.md), which strictly reads their output
afterwards. That split is the same one `CLAUDE.md` documents: an `analysis/`
script must never run a simulation, so anything that launches one belongs here.
Mirrors the sibling `hitl-practice` repo's own `scripts/` convention.

Nothing here imports from `hitl_pmp` — these shell out to `python -m hitl_pmp.cli`
— so the package's own layering contract (`lint-imports`) is unaffected by them.

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

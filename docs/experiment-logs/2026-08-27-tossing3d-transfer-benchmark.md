# Standard EES transfer benchmark — protocol registered before the full run

## Question / goal

Does autonomous practice with a recoverable same-side bin improve success on the
original far-side Tossing3D tasks? Report mean and standard deviation across run seeds,
and retain videos of every practice session.

## Background

The prior 16-action video used one same-side evaluation task and 200 sampler iterations.
It established a physical sequence, not benchmark performance or learning improvement.
The established later Tossing3D protocol uses 100 cycles × 20 actions, ten seeds and ten
test tasks, as documented in the 100-cycle plateau and reset-free remeasurement logs.

## Hypothesis

Repeated autonomous recovery supplies throw outcomes for the learned sampler. Whether
that learning transfers to the far-side scene is an empirical question; practice may
also stall after an unrecoverable landing. Neither failure is removed from the results.

## Guidance given

Use the same standard settings as previous experiments, ten far-side test tasks,
mean/std learning curves, practice videos, and draft PRs. Preserve TDD and reset-free
practice rather than silently restoring the cube between attempts.

## Methods

- Run seeds: 0–9, no replacement or best-seed selection.
- Practice: same-side scene, canonical scene seed 125, continuous state across cycles.
- Evaluation: stock barrier scene; ten tasks sampled once per run and reused at every
  checkpoint. The test seed stream is derived from each run seed, as in prior experiments.
- Budget: 100 cycles, maximum 20 actions per cycle, baseline plus 100 evaluation sweeps.
- Learning: normal EES defaults, including 10,000 sampler training iterations; no demo
  goal-pursuit cap, no human reset skill, no periodic or interval resets.
- Record every practice period with the standard period recorder. Evaluation uses a
  separate simulator and separate replay log, never resetting the practice environment.
- Primary plot: success fraction per run seed, mean ± one **sample standard deviation**
  (`ddof=1`) across the ten seeds, not SEM or a confidence interval.
- Align by cycle. Plot actual cumulative practice actions separately, because stalled
  runs need not share an action-count grid. Show individual seed trajectories too.
- Require all ten seeds and 101 checkpoints, each with denominator ten. Do not silently
  omit incomplete runs or substitute the two-cycle preflight for the full budget.

Reproduce:

```bash
scripts/with_env.sh python -m scripts.tossing3d_transfer_benchmark --results-root artifacts/tossing3d-transfer-standard --max-workers 10
scripts/with_env.sh python -m analysis.practice_makes_perfect.tossing3d_transfer_curve --results-root artifacts/tossing3d-transfer-standard --output-dir artifacts/tossing3d-transfer-standard/plots
```

Each seed writes `ees/<seed>/stats.json`, `config_snapshot.json`, sampler/competence
logs and `period_videos/practice/cycle_0000.mp4` through `cycle_0099.mp4`. Existing seed
output directories are refused, preventing accidental overwrites. The sweep runner
limits each child to one math thread. Concurrency changes wall time, not the protocol.

TDD: the recipe and aggregation tests first failed on missing modules. Tests now pin
the standard budget and far-side layout, verify sample SD against a known matrix,
reject unequal checkpoint grids/wrong task counts, and require all seed directories
and matching experiment configurations.

## Results

Pending the full run. The two-cycle preflight checks wiring and must not be presented
as the final learning curve. No learning-improvement claim is made in advance.

## Recommendation

Keep the PR draft until all runs are complete and the recorded curve and practice
failures have been inspected. If a seed fails to complete, report that explicitly;
do not compute a final curve from a selectively completed subset.

# Fixing the evaluation protocol: a fixed test set

`PracticeLoop` re-sampled the evaluation set *inside every sweep*, so each point on a
learning curve was measured on a different set of test tasks. This log records the
before/after that justifies changing it, and — as importantly — the decomposition that
caught a second change which **cancelled** the benefit and had to be removed.

## The bug

```python
for i in range(num_test_tasks):
    task = problem.sample_test_task()   # fresh tasks, every sweep
```

predicators does not do this: `BaseEnv.get_test_tasks` generates once and caches
(`envs/base_env.py:180-193`), so every cycle is scored on the same tasks and the curve
isolates policy change.

Re-sampling stacks task-sampling variance on top of the policy change the curve is
supposed to measure. Worse, it is not symmetric noise: a Method whose competence is
uneven across the task distribution swings between a favourable and an unfavourable
draw from one cycle to the next, which reads as instability that the method does not
actually have.

## Experiment

Light Switch EES, 10 seeds, `--grid-size 25 --num-test-tasks 10`, 10 cycles x 150
steps. Arms run **sequentially, not in parallel**, deliberately: Fast Downward is
wrapped in a 10-second *wall-clock* timeout, so unequal CPU load between arms would
turn into unequal planning-failure rates and contaminate exactly the variance
measurement being made.

The endpoint is useless here — Light Switch EES saturates at 100% on every seed under
every arm — so the readout is mid-curve variance plus two stability statistics:
**downward steps** (a monotone learner should not regress) and their total magnitude.

| metric | before | **test-set fix only** | test set + train pool |
|---|---|---|---|
| pooled across-seed sd (climb) | 16.6 | **10.4** | 15.7 |
| downward steps (of 100 pairs) | 5 | **2** | 6 |
| drop magnitude (pts) | 60 | **20** | 90 |
| final mean % | 100 | **100** | 97 |

The fix does what it claims: ~37% less across-seed variance, regressions cut from 5 to
2, and a visibly faster climb (99% at 450 transitions versus 78% before).

## What the decomposition caught

The first version of this change also replaced the unbounded train-task stream with a
cached pool of 50 sampled with replacement — another genuine fidelity gap versus
predicators (`approaches/online_nsrt_learning_approach.py:66-67`). Bundled together,
the measured benefit **disappeared**: sd back to 15.7, *more* regressions than before
the fix, and the endpoint falling off 100%.

So the train-pool change was removed from this PR. It stays on the fix list, but needs
its own PR and its own evidence rather than riding along with a protocol fix. This is
precisely the failure mode the repo's one-feature-per-PR rule exists to prevent, and
the only reason it was caught is that the before/after was run at all — the bundled
version would otherwise have shipped looking like a win.

## What this fix does NOT explain

The original motivation was Ball-Ring, where our EES showed ~3x the reference's
per-seed variance (sd 34-38 versus 12) and seeds appearing to collapse from 100% to 0%
within a single cycle. The moving test set was the leading suspect for **both**.

It only explains the first. With the test set fixed — the same 10 tasks every sweep —
EES still produces curves like `[0, .2, 0, 1.0, 1.0, .1, .1]`. A policy that solves
10/10 and then 1/10 *on identical tasks* is real degradation, not measurement noise, so
that hypothesis is falsified.

| symptom | cause | status |
|---|---|---|
| elevated across-seed variance | moving test set | fixed here |
| catastrophic 100% -> 0% collapse | sampler convergence (below) | separate |

The collapse turned out to be a **sampler-convergence artifact**. Holding everything
else fixed and varying only the gradient-step budget:

| sampler iters | Light Switch EES curve |
|---|---|
| 300 | `0, .2, 0, 1.0, 1.0, .1, .1, .1, .2` — oscillates |
| 1000 | `0, .3, .7, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0` — climbs and holds |
| 5000 | `0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0` — immediate, stable |

An underconverged classifier's argmax moves between refits, so the policy is
effectively re-rolled every cycle. This also means the port's own default is
indefensible: `max_train_iters = 1000` against `n_iter_no_change = 5000`, so early
stopping **can never fire** and training always halts at exactly 1000 steps. The
paper's config uses 100000 (`active_sampler_learning.yaml:112`, global — `grid_row`
does not override it) and predicators' own default is 10000 (`settings.py:572`).

That is why `test_ees_learns_to_solve_light_switch_over_practice_cycles` changed here:
it was pinned at 300 iterations, asserting that EES learns at a budget where it
demonstrably cannot, and passed only because the re-sampled evaluation set handed its
final sweep a separate draw.

## Reproducing

```bash
# before: git stash / check out main
python -m scripts.run_sweep --env lightswitch --methods ees --num-seeds 10 \
  --results-root results/evalproto-before \
  --shared-args "--grid-size 25 --num-test-tasks 10" \
  --method-args "ees=--num-cycles 10 --max-steps-per-interaction 150"
# after: same command on this branch, into results/evalproto-after
```

Run the arms **one at a time**, for the wall-clock-timeout reason above.

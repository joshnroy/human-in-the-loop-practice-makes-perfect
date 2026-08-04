# Rescuing the robot more often, with training held fixed (Tossing Room)

TBD-HEADLINE

![gap vs reset interval](./2026-08-04-tossingroom-reset-interval-gap.png)

## The question, and why it needed new code

Tossing Room has exactly one genuinely terminal failure:

| family | `Throw`? | a miss costs |
|---|---|---|
| `EMPTY` | no — `MoveRoom`xk + `Press` | nothing; the family is deterministic |
| `TRASH` | yes | a round trip to the pile for a fresh item — expensive, recoverable |
| `RECYCLING` | yes | **terminal** — pile in room 3, recycling bin in room 1, `blocked_right_from = 2` makes room 3 unreachable once the item is gone, so Fast Downward correctly reports no plan |

`PracticeLoop.run` used to reset the environment only at the top of each practice
cycle, so the harness handed out a free reset every `--max-steps-per-interaction`
steps. That caps what stranding can cost at "you wasted the rest of this period".

**Hypothesis.** Rescue a stranded robot sooner and it wastes less experience, so
`RECYCLING` suffers less and the (TRASH − RECYCLING) gap shrinks as resets get
more frequent.

PR #39 tried to test this by varying the period length with `--num-cycles`
inverted to hold transitions fixed. It could not: `--num-cycles` sets the number
of free resets **and** the number of sampler refits with one number, so its arms
ended ~40 competence points apart on identical experience, and its measured gap
peaked mid-curve rather than at the fewest-resets end. That is the training
difference asserting itself, not irreversibility.

So this experiment starts with a code change. `PracticeLoop.run` gains
`practice_reset_interval` (CLI: `--practice-reset-interval`), which puts the
environment back to the **current** practice task's initial state every k steps
*inside* a period, without ending the cycle and therefore without firing
`end_cycle()`. Resets are not charged as transitions. `None` — the default — is
exactly the old behaviour.

Two details that had to be right for the manipulation to be clean:

* **`Method.observe_environment_reset`.** EES scores a skill by checking its
  `add_effects` on the *next* state it sees. Without a hook fired immediately
  before each within-period reset, every skill executed just before a reset would
  be judged against a freshly reset environment — where `InBin`/`RobotInRoom`
  effects essentially never hold — and recorded as a **failure** into both its
  competence model and its sampler's training data. That mislabelling scales with
  reset frequency: ~225 false failures in arm A against 0 in arm D, out of 2500
  skill executions. It would have degraded exactly the sampler whose quality
  determines TRASH and RECYCLING success, in the direction that *masks* the
  hypothesis. The hook deliberately does **not** fire at the period boundary,
  where the last skill has always gone unobserved — that asymmetry is what makes
  every arm drop exactly one observation per period.
  `test_the_number_of_observed_outcomes_does_not_depend_on_the_reset_interval`
  runs the real method at four intervals and counts observed outcomes: 14 in every
  arm.
* **`Metrics.num_practice_resets`.** Counts resets as they happen and rides into
  `stats.json`, so "the arms differed in reset frequency" is a measurement rather
  than a restatement of the flag.

## Design

| arm | `--practice-reset-interval` | resets/period | total free resets |
|---|---|---|---|
| A | 10 | 10 | 250 |
| B | 25 | 4 | 100 |
| C | 50 | 2 | 50 |
| D | 100 (= period length, the old behaviour) | 1 | 25 |

`--num-cycles 25` and `--max-steps-per-interaction 100` in **every** arm, so all
four get 25 sampler refits over 2500 online transitions. A 10x range in reset
frequency with training pinned. 20 paired seeds (0..19, fixed by `run_sweep`),
`--num-test-tasks 30`, `--sampler-max-train-iters 10000`, `--env tossingroom
--method ees`.

**Primary metric: the within-arm (TRASH − RECYCLING) final success gap, paired by
seed.** Prediction if the hypothesis holds: the gap shrinks from D toward A.

### Not comparable to PR #39

Two things changed at once relative to that experiment, both deliberately:

* the **test-set composition** is now deterministic (14 TRASH / 14 RECYCLING / 2
  EMPTY per seed, from #41) where #39 sampled it from `goal_weights` and got
  16/10/4 at seed 0, 11/12/7 at seed 1;
* **20 seeds** instead of 10.

So absolute numbers here are measured on a different evaluation set from #39's and
must not be read across. Within these four arms everything stays paired and
comparable, which is what the experiment needs.

### The noise floor, and what this design could have found

The gap is a difference of two binomial proportions, so at the worst case p = 0.5
its per-seed sd from task sampling alone is
`100 * sqrt(0.25/14 + 0.25/14)` = **18.9pp**. That is the floor: an observed gap
sd near it means the spread is how few tasks each seed holds, not what the agent
learned.

#39's floor was 20.7pp on a *sampled* composition whose per-family counts also
moved seed to seed. The improvement here is small in the floor and large in the
seed count, which is the right trade: #39's largest-effect arm had sd 49.3 against
that 20.7pp floor, i.e. genuine seed-to-seed heterogeneity that no number of extra
tasks per seed can reduce — only more seeds can.

TBD-MDE

## Result 0: the manipulation happened

TBD-MANIPULATION

## Result 1: the precondition holds (or does not)

TBD-PRECONDITION

## Result 2: the gap against reset frequency

TBD-GAP

![per-family learning curves](./2026-08-04-tossingroom-reset-interval-curves.png)

## Reproducing

```bash
conda activate hitl-pmp
export FD_EXEC_PATH=/path/to/downward

for arm_interval in armA:10 armB:25 armC:50 armD:100; do
  arm=${arm_interval%%:*}; interval=${arm_interval##*:}
  python -m scripts.run_sweep \
    --env tossingroom \
    --methods ees \
    --num-seeds 20 \
    --results-root "$RESULTS/$arm" \
    --shared-args "--num-test-tasks 30 --practice-reset-interval $interval" \
    --method-args "ees=--num-cycles 25 --max-steps-per-interaction 100 --sampler-max-train-iters 10000" \
    --max-workers 5
done

python -m analysis.practice_makes_perfect.tossingroom_reset_interval \
  --arm "armA=$RESULTS/armA" --arm "armB=$RESULTS/armB" \
  --arm "armC=$RESULTS/armC" --arm "armD=$RESULTS/armD" \
  --aggregate-output docs/experiment-logs/2026-08-04-tossingroom-reset-interval.json

python -m analysis.practice_makes_perfect.tossingroom_reset_interval \
  --arms-json docs/experiment-logs/2026-08-04-tossingroom-reset-interval.json \
  --output docs/experiment-logs/2026-08-04-tossingroom-reset-interval-gap.png \
  --curves-output docs/experiment-logs/2026-08-04-tossingroom-reset-interval-curves.png
```

Per-seed results are machine-local (see
`docs/experiment-logs/2026-08-03-cross-machine-reproducibility.md`); the committed
`2026-08-04-tossingroom-reset-interval.json` is the record that travels, and
comparisons should be made at arm level.

## Limitations

TBD-LIMITATIONS

# Rescuing the robot more often, with training held fixed (Tossing Room)

**On the pre-specified metric the answer is no, and this time the design can
actually say so.** With `--num-cycles 25` and `--max-steps-per-interaction 100`
pinned in every arm — 25 sampler refits and exactly 2500 online transitions
everywhere — a 10x change in how often the robot gets a free reset moves the final
(TRASH − RECYCLING) gap by **+0.36pp, exact Wilcoxon p = 0.9531**. The four arm
means are +1.8 / +0.7 / +0.4 / +2.1 pp for reset intervals of 10/25/50/100 steps.
No trend: per-seed slope +0.03pp per doubling, **p = 0.7959**.

**But the null is measured at a ceiling, and that matters.** By 2500 transitions
every arm sits at 95–99% on both throw families, so the final gap has almost
nowhere to move. The tight sds (3.2–12.5pp against an 18.9pp noise floor) are
saturation, not precision — 13–16 of 20 seeds have a gap of *exactly* zero. The
design's minimum detectable effect on the extreme contrast is 9.35pp, so a gap
effect that large is excluded and a smaller one is not.

**The methodological result is the one worth keeping: the confound that sank PR #39
is gone.** Paired armD − armA, final competence differs by **−1.8pp on TRASH
(p = 0.3750)** and **−2.1pp on RECYCLING (p = 0.5156)** — against #39's −40.4 and
−44.7pp at p = 0.0156. Transitions are 2500 in every seed of every arm, sd 0.
`EMPTY` is exactly 100% everywhere, sd 0. The arms differ in reset frequency and
in nothing else that was measured.

**And post-hoc, reset frequency turns out to matter a great deal — just not to the
endpoint.** Resetting every 10 steps instead of every 100 raises the area under
the learning curve by **+18.4pp on RECYCLING (p = 0.0001)** and **+11.1pp on TRASH
(p = 0.0007)**, and by **exactly 0.0pp on the deterministic `EMPTY` control**. The
harness's reset frequency is a real driver of sample efficiency on the stochastic
families. Whether it is *specifically* about irreversibility is **not
established**: RECYCLING gains 7.3pp more than TRASH, but p = 0.0623, and ~49
seeds would be needed for 80% power.

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
  runs the real method (EES, Fast Downward in the loop) at four intervals and
  counts observed outcomes: `n − 1` per period of `n` steps at every interval.
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

**Primary metric, fixed before the runs: the within-arm (TRASH − RECYCLING) final
success gap, paired by seed.** Prediction if the hypothesis holds: the gap shrinks
from D toward A.

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

**Minimum detectable effect at 20 paired seeds**, at the spread actually observed
(`(z_0.975 + z_0.80) * sd / sqrt(n)`):

| quantity | observed sd | MDE |
|---|---|---|
| final gap, armD − armA | 14.9 | **9.35pp** |
| per-seed trend slope | 4.02 | 2.52pp per doubling |
| final TRASH, armD − armA | 6.1 | 3.8pp |
| final RECYCLING, armD − armA | 12.9 | 8.1pp |

So the null below excludes a final-gap effect of ~9pp or larger, and says nothing
about smaller ones.

## Result 0: the manipulation happened, and nothing else moved with it

All three checks below are read *before* any p-value, because each decides whether
the p-values mean anything.

**Free resets that actually happened** (`Metrics.num_practice_resets`, read back
out of each run's own `stats.json`):

| arm | interval | expected | observed (min..max over 20 seeds) |
|---|---|---|---|
| A | 10 | 250 | **250..250** |
| B | 25 | 100 | **100..100** |
| C | 50 | 50 | **50..50** |
| D | 100 | 25 | **25..25** |

**Test-set composition**: 14 TRASH / 14 RECYCLING / 2 EMPTY in every seed of every
arm. **0 violations** across all 80 runs. Asserted rather than assumed, and the
expected split is asked of `TossingRoomTasks.test_goal_type_counts()` rather than
hardcoded in the analysis.

**Experience**: exactly **2500** online transitions in every seed of every arm,
sd 0. This is not automatic — a mid-period reset can revive a robot whose practice
planner had nothing applicable left and would otherwise have raised
`InteractionComplete`, buying the frequently-reset arms extra steps. It did not
happen.

**All 80 runs exited 0** (20/20 in each arm).

## Result 1: the precondition holds — the arms really are equally trained

Paired across the same 20 seeds, exact tests, arm D (100) minus arm A (10):

| family | mean difference | sd | Wilcoxon p | sign-flip p | MDE at n = 20 |
|---|---|---|---|---|---|
| `TRASH` | −1.8pp | 6.1 | 0.3750 | 0.3750 | 3.8pp |
| `RECYCLING` | −2.1pp | 12.9 | 0.5156 | 0.5547 | 8.1pp |
| `EMPTY` | +0.0pp | 0.0 | 1.0000 | 1.0000 | 0.0pp |

Compare PR #39's same table: **−40.4pp and −44.7pp, both p = 0.0156**. That
experiment's arms sat at different points on a gap-versus-progress curve and its
cross-arm gap comparison was confounded at a magnitude far larger than anything it
reported. These arms are level, so a cross-arm gap difference here would be
attributable to reset frequency.

`EMPTY` is again a **perfect control** — 100% in every seed of every arm, sd
exactly 0. Whatever moves is confined to the two families that use the stochastic
`Throw`. It is also the reason the control cannot do more work than that: an
untrained policy already solves it, so it sits on the ceiling.

## Result 2: the final gap does not move with reset frequency

| arm | interval | resets | final TRASH | final RECYCLING | final EMPTY | **gap (T−R)** |
|---|---|---|---|---|---|---|
| A | 10 | 250 | 99.3 ± 2.2 | 97.5 ± 5.8 | 100.0 ± 0.0 | **+1.8 ± 6.5** |
| B | 25 | 100 | 95.7 ± 17.6 | 95.0 ± 17.5 | 100.0 ± 0.0 | **+0.7 ± 3.2** |
| C | 50 | 50 | 97.9 ± 4.7 | 97.5 ± 4.8 | 100.0 ± 0.0 | **+0.4 ± 4.9** |
| D | 100 | 25 | 97.5 ± 5.8 | 95.4 ± 10.4 | 100.0 ± 0.0 | **+2.1 ± 12.5** |

**Trend tests**, all paired on the same 20 seeds, all exact by enumeration:

| test | mean | sd | Wilcoxon p | sign-flip p | n for 80% power |
|---|---|---|---|---|---|
| gap, armD − armA | +0.36pp | 14.92 | 0.9531 | 1.0000 | ~13,707 |
| per-seed slope, pp per doubling | +0.03 | 4.02 | 0.7959 | 0.9778 | ~135,590 |
| per-seed Spearman rho vs interval | −0.13 | 0.58 | 0.2845 | 0.3263 | ~154 |

Nothing is close to p < 0.05, and the Spearman rho is *negative* — the opposite of
the hypothesis's direction. Tested at n = 20 across seeds rather than n = 4 across
arm means, because Spearman on four points bottoms out at p = 0.083 even for
perfect monotonicity.

**Within-arm gap against zero — is RECYCLING behind TRASH at the end at all?**

| arm | mean gap | Wilcoxon p | exact-zero seeds |
|---|---|---|---|
| A | +1.8pp | 0.3750 | 14/20 |
| B | +0.7pp | 0.6250 | 16/20 |
| C | +0.4pp | 0.9062 | 14/20 |
| D | +2.1pp | 0.5000 | 13/20 |

No arm establishes even the premise at the final checkpoint.

**Noise floor against observed sd.** Every arm is *below* the 18.9pp floor —
6.5 / 3.2 / 4.9 / 12.5, i.e. 34% / 17% / 26% / 66% of it. **That is saturation,
not precision.** A gap of exactly zero is forced whenever both families are at
100%, and 13–16 of 20 seeds are in that state. This is the same trap PR #39's log
flagged for its own arm A, and it applies to all four arms here.

**Per-seed final RECYCLING %, because an sd here is often one collapsed seed:**

```text
armA  100 100 100 100 100 100 100  93 100 100  86 100 100  93 100  79 100 100 100 100
armB   93 100 100 100 100 100  93 100 100 100  93 100  21 100 100 100 100 100 100 100
armC   93 100 100  86  86 100 100 100 100 100 100  93 100 100 100  93 100 100 100 100
armD  100 100 100  86 100 100  64 100 100 100 100 100  71 100  86 100 100 100 100 100
```

One seed in arm B collapses to 21%; arm D has two seeds at 64% and 71%. Arm A's
worst is 79%. That ordering is directionally consistent with the hypothesis and
far too thin to claim anything from.

![per-family learning curves](./2026-08-04-tossingroom-reset-interval-curves.png)

## Result 3 (POST-HOC): the effect is in the trajectory, not the endpoint

The learning curves above are not saturated, and they plainly differ. Everything in
this section was added **after** seeing that the pre-specified endpoint metric was
measured at a ceiling, and carries correspondingly less evidential weight.

The statistic is the **normalised area under each seed's learning curve** — the
mean success rate over all 26 checkpoints. Chosen over "transitions to reach X%"
because that one is censored: a seed that never reaches the threshold has no value
(one ends at 21%), which would silently drop the worst seeds and flatter whichever
arm they fall in. Averaging the curve is defined for every seed.

An unweighted mean over checkpoints is only comparable across arms if checkpoint
*i* sits at the same transition count in every arm. It does: across all 80 runs
and all three families there is exactly **one** distinct transition grid,
`0, 100, ..., 2500`. Checked rather than inferred from the equal totals — a period
that ended early and was made up later would land on 2500 while sampling the curve
at different x.

| arm | interval | RECYCLING AUC | TRASH AUC | EMPTY AUC |
|---|---|---|---|---|
| A | 10 | **75.9 ± 12.2** | **79.2 ± 9.9** | 100.0 ± 0.0 |
| B | 25 | 68.2 ± 16.8 | 77.7 ± 11.6 | 100.0 ± 0.0 |
| C | 50 | 60.8 ± 18.2 | 74.5 ± 12.3 | 100.0 ± 0.0 |
| D | 100 | 57.5 ± 14.9 | 68.1 ± 9.5 | 100.0 ± 0.0 |

**Monotone in reset frequency on both throw families, and flat on the control.**
Paired armA (10) minus armD (100), exact:

| family | mean speed-up | sd | Wilcoxon p | sign-flip p |
|---|---|---|---|---|
| `RECYCLING` | **+18.4pp** | 18.9 | **0.0001** | **0.0001** |
| `TRASH` | **+11.1pp** | 11.8 | **0.0007** | **0.0005** |
| `EMPTY` | +0.0pp | 0.0 | 1.0000 | 1.0000 |

So the harness's reset frequency is a **real and large driver of sample
efficiency** on the stochastic families, worth ~18 points of area-under-curve on
the irreversible one over a 10x range. It is not an artifact of the control
family, which does not move at all.

**Is it about irreversibility, though?** Not established. There is an obvious
non-irreversibility mechanism that predicts exactly this: a reset teleports the
robot back to the pile room, so it spends fewer steps traversing and more steps
throwing, which speeds up *any* throw-dependent family. Irreversibility predicts
something stronger — that the terminal family benefits **more**:

| test | mean | sd | Wilcoxon p | sign-flip p | MDE at n = 20 | n for 80% power |
|---|---|---|---|---|---|---|
| RECYCLING speed-up − TRASH speed-up | +7.3pp | 18.4 | 0.0623 | 0.0912 | 11.5pp | **~49** |

The sign is right and the magnitude is substantial, but **p = 0.0623 is not
significant** and this is a post-hoc test besides. ~49 paired seeds would settle
it, and seeds pool cleanly with these — 20 are already banked.

The same picture from the other direction, the mean (TRASH − RECYCLING) gap across
all checkpoints rather than at the last one:

| arm | interval | mean gap over training | sd | vs zero, Wilcoxon p |
|---|---|---|---|---|
| A | 10 | +3.2pp | 13.3 | 0.5958 |
| B | 25 | +9.4pp | 16.5 | **0.0266** |
| C | 50 | +13.8pp | 16.5 | **0.0003** |
| D | 100 | +10.6pp | 15.7 | **0.0055** |

Mid-training, RECYCLING *is* behind TRASH in three of four arms — the premise the
final checkpoint could not establish. And the arm with the most frequent resets is
the one where that gap is smallest and indistinguishable from zero. But armD − armA
is +7.3pp at **p = 0.0623** (the same contrast as the differential above, as it
must be), and the ordering is non-monotone (C > D). **Not established.**

## Concurrency and reproducibility, measured

All four arms ran concurrently at `--max-workers 5` each, inside
`systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue`, sharing the
box with another agent's sweep. PR #39 validated concurrency at loads 9–27; this
window ran hotter, so it was re-measured rather than assumed:

* **29,925 live Fast Downward process observations** over 900 samples at 1 Hz, at
  loads **22.2–46.4** on 24 cores. Max process lifetime **3s** against the 10s
  wall-clock budget; **8** observations at or above 2s; **zero** at or above 5s.
* **Byte-identity.** Arm A seed 0 completed during the hottest window (load
  50–57) and was re-run afterwards at load 24–40. The two `stats.json` files are
  **identical** (md5 `d6ca15518a46c7280cf3b8a6fef9a03a`). A spurious FD timeout
  necessarily changes the plan and therefore the trajectory, so identity across
  that load difference rules one out.

> One measurement bug is recorded here because it nearly became a reported number.
> The first FD sampler grepped for `downward` in `ps` output and returned a max
> lifetime of 10s with 6 observations past half the budget — which would have been
> alarming. The rows were another agent's `FD_EXEC_PATH=... pytest` bash wrapper,
> whose command line merely *mentions* the FD checkout path. Every real
> `fast-downward.py` process was at 0s. The tightened sampler matches
> `fast-downward.py` and excludes shell wrappers. This is the same class of bug PR
> #39 recorded (a partial `ps` row read as a 4-billion-second lifetime); a loose
> `ps` filter has now produced a false alarm twice on this project.

**Default preserved, verified end to end rather than argued.** Arm D's interval
(100) equals the period length, so it should be exactly the behaviour that shipped
before the flag existed. Seed 0 was run once through the sweep with
`--practice-reset-interval 100` and once with the flag **omitted entirely**, same
seed, same everything else. The two `stats.json` files are **byte-identical**,
including `num_practice_resets` (25 either way — the period-opening reset was
always there and is now simply counted). That is the real method with Fast Downward
in the loop over 25 cycles, not a unit test against a fake.

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

## Limitations, stated plainly

* **The pre-specified metric was measured at a ceiling.** At 2500 transitions this
  configuration saturates both throw families in most seeds, so the final gap had
  ~9pp of resolution and roughly nothing to resolve. A future version of this
  experiment should either shorten the budget or harden the domain so the final
  checkpoint is not on the ceiling — the right move is probably to stop the run
  around 1200–1500 transitions, where the curves are still separated.
* **Everything in Result 3 is post-hoc.** The area-under-curve statistics were
  chosen after seeing that the endpoint metric was saturated. They are reported
  because the effect is large and the control is clean, not because they were
  planned.
* **The irreversibility-specific claim is unestablished** (p = 0.0623, ~49 seeds
  needed). A generic "a reset saves wasted traversal, which helps any throw
  family" mechanism explains the headline speed-up just as well, and this design
  cannot separate the two.
* **`EMPTY` cannot do much work as a control.** It is solved by an untrained
  policy, so it sits on the ceiling and could only ever have shown movement
  downward. Its value here is that it shows *no* generic harness artifact, not
  that it bounds the effect.
* **Numbers are not comparable to PR #39's** — different test-set composition and
  different seed count.
* Per-seed results are machine-local; compare at arm level.

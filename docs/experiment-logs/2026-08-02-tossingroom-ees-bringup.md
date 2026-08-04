# Tossing Room: EES learns the throw force, and the evaluation horizon was hiding it

**Result.** `TossingRoomProblem.max_episode_steps` was `2 * num_rooms + 2 = 16` against
a 5-skill solve — eleven spare steps, every one of them a free retry of the domain's
single stochastic skill. An **unpracticed** EES scored **94.7%** that way, five points
under the skill oracle, having learned nothing at all; at the corrected horizon it
scores 62.7%. Porting Light Switch's `H_eval` convention (longest shortest solve + 2 =
7) makes the curve legible without touching a single practice step.

With the metric fixed, EES's learned sampler is visibly doing the thing it is supposed
to do: it drives `Throw`'s force onto each task's own `target_force`, median greedy
error **0.253 → 0.053** (the tolerance is 0.1), per-throw hit rate **22% → 78%** against
a 19% chance rate, and greedy throws per throw-episode **2.57 → 1.28** against a floor
of 1. It stops retrying because it stops missing.

![EES vs. the random-skills lower bound](2026-08-02-tossingroom-ees-curves.png)

At the corrected horizon the curve has somewhere to go: EES starts at 62.3% — an
unpracticed policy guessing the throw force — and ends at 99.0%, against a
random-skills floor that never leaves 7%. All three sampler-iteration arms share the
same first sweep to the decimal (62.3%), which is the sanity check that the budget
cannot matter before any training has happened.

![The throw force converges onto each task's target](2026-08-02-tossingroom-throw-convergence.png)

The mechanism, from inside the same runs (3 seeds, 10k sampler iterations): the median
greedy throw error first crosses into the `throw_tolerance` band at 750 transitions,
comes back out at 900, and settles below it from 1050 on — and the retry count falls
toward one throw per episode as it does. The non-monotonicity is real and is discussed
below, not smoothed away. Detail and per-seed numbers in
[the section below](#does-ees-learn-yes--it-stops-missing-so-it-stops-retrying).

## The bracket

| arm | % of evaluation tasks solved | source |
|---|---|---|
| skill oracle (upper) | **100%** (30/30) | `tests/environments/tossingroom/test_integration.py`, pinned by CI |
| random skills (lower) | **6.7%** (sd 2.7, worst seed **0%**) | 10 seeds × 30 test tasks, this log |
| EES, practiced (10k iters) | **99.0%** (sd 3.2, worst seed 90%) | 10 seeds × 30 test tasks, this log |
| EES, unpracticed, **old** horizon 16 | **94.7%** (sd 3.9, worst seed 87%) | 10 seeds × 30 test tasks, this log |
| EES, unpracticed, **new** horizon 7 | **62.7%** (sd 14.7, worst seed 37%) | 10 seeds × 30 test tasks, this log |

The oracle row is cited, not re-derived: re-running a sweep to reproduce a CI assertion
would spend compute to learn nothing. The random-skills row is worth saying out loud —
one of its ten seeds solved **nothing at all**, which is the shape of a genuine floor
rather than a weak method.

The problem is row four. Before the horizon change, EES's *floor* — a fresh
`EesMethod` that has never been trained — sat 5 points under the oracle's ceiling and
88 points over the random-skills floor, having learned nothing whatsoever. There was no
room left in the metric for practice to show up in.

## Why: the horizon was buying retries, not measuring competence

`Throw` is the domain's only stochastic skill. A failed throw is not terminal — the
robot still holds the item and is still in the bin room — so the next policy step
simply replans to `Throw` again. The horizon therefore sets, silently, how many
independent draws the evaluation grants.

Every constant below was read out of the source rather than the paper: a random force
is `TossingRoomSkills.sample_params`'s `Uniform(0, 1)`; a task's `target_force` is
`TossingRoomTasks`' `Uniform(0.5, 1.0)` (`target_low`/`target_high`); a throw lands iff
`|force − target| < throw_tolerance = 0.1` (`TossingRoomEnvironment._apply_throw`); and
the goal families are drawn from `goal_weights = (0.4 RECYCLING, 0.4 TRASH, 0.2 EMPTY)`.
Integrating the overlap gives a chance hit probability of **exactly 0.19** (the window
is the full 0.2 wide for targets in [0.5, 0.9] and clipped by the force's upper bound
above that). Measured over the 1133 throws an unpracticed EES issued below, the realised
rate is **0.198** — so the arithmetic here is not a model fitted after the fact.

At horizon `H` a `TRASH` episode spends 4 actions getting into position and can throw
`H − 4` times; `RECYCLING` spends 3 and can throw `H − 3`; `EMPTY` needs no throw at
all and always solves within 4. At `p = 0.19` that predicts
`0.4(1 − 0.81¹²) + 0.4(1 − 0.81¹³) + 0.2 = 94.2%` at `H = 16`, and `61.5%` at `H = 7`.

Instrumented over **300 episodes (10 seeds × 30 test tasks)** — the same protocol as
the real arms below — with an unpracticed EES (a fresh `EesMethod`, no practice at
all):

| horizon | solved (mean over seeds) | sd across seeds | worst seed | `Throw`/episode | max in one episode |
|---|---|---|---|---|---|
| **16** (`2 * num_rooms + 2`, old) | **94.7%** | 3.9 | 87% | 3.78 | 13 |
| 12 | 87.7% | 8.3 | 73% | 3.42 | 9 |
| 9 (`num_rooms + 2`) | 76.0% | 12.6 | 53% | 2.84 | 6 |
| 8 | 69.3% | 14.8 | 43% | 2.53 | 5 |
| **7** (longest solve + 2, new) | **62.7%** | 14.7 | 37% | 2.16 | 4 |
| 6 | 52.7% | 12.6 | 33% | 1.69 | 3 |
| 5 (longest solve, no spare) | 42.3% | 10.8 | 27% | 1.11 | 2 |

Predicted at `p = 0.19`, against measured, at every horizon in the table:

| horizon | 16 | 12 | 9 | 8 | 7 | 6 | 5 |
|---|---|---|---|---|---|---|---|
| predicted | 94.2% | 86.6% | 74.8% | 68.8% | 61.5% | 52.5% | 41.4% |
| measured | 94.7% | 87.7% | 76.0% | 69.3% | 62.7% | 52.7% | 42.3% |

Seven horizons, every one within 1.3 points and every one *slightly under* — exactly
the bias the realised 0.198 rather than 0.190 predicts. The horizon's effect on this
metric is entirely accounted for by counting free redraws; nothing else is needed to
explain it.

`Pickup` (240), `MoveRoom` (787) and `Press` (60) are **identical at every horizon** —
literally the same integers, not merely close. Only the throw count moves (1133 at
H=16, 648 at H=7). The horizon was measuring patience, not skill.

Two independent measurements agree on the `H = 7` figure, which is worth stating
because the whole PR turns on it: this probe puts an unpracticed EES at **62.7%**, and
the first evaluation sweep of the ten-seed EES arms below — a completely separate set
of runs, driven through the CLI rather than this script — comes out at **62.3%** in all
three sampler-iteration arms.

Every horizon in that table comes from **one** rollout set, not seven. `run_task_episode`
checks the goal at the top of each iteration and only then calls the policy, and the
policy replans from the current state with no history, so truncating the horizon to `H`
stops the *same* trajectory earlier: success at `H` is exactly `steps_to_success ≤ H`.
Running each episode once at `H = 16` and reading off prefixes therefore gives every
horizon exactly, and perfectly paired — no seven separate runs whose RNG streams would
diverge after the first extra throw and make the comparison unpaired.

## The fix, and why this number

Light Switch's horizon is `grid_size + 2`, cited to the paper's Appendix F. Its solve
is `grid_size − 1` moves plus one toggle, so the load-bearing quantity is **two spare
actions**, not the cell count — the coincidence with `grid_size` is an artifact of the
light sitting at the far end. Ball-Ring likewise pins `H_eval = 8` against a ~6-skill
plan.

Tossing Room's was self-described as "a generous bound", derived from nothing. It now
computes the longest shortest solve this layout admits and adds two:

| goal family | shortest solve (default layout) |
|---|---|
| TRASH | `Pickup` + 3 × `MoveRoom` (3→4→5→6) + `Throw` = **5** |
| RECYCLING | `Pickup` + 2 × `MoveRoom` (3→2→1) + `Throw` = 4 |
| EMPTY | 3 × `MoveRoom` + `Press` = 4 |

so `max_episode_steps() == 7`. Room-to-room distance is a closed form, not a graph
search: the rooms are a 1-D hallway whose only blocked edge is the rightward step out
of `blocked_right_from`. Targets that edge makes unreachable are skipped — they are
unsolvable at any horizon, so letting them set the budget would only hand the solvable
goals extra retries.

**This changes only the measurement.** An interaction period's length is
`--max-steps-per-interaction`, which `PracticeLoop` applies directly and which this
does not touch, so no Method receives more or less practice experience than before.
The oracle throws at the known target and solves in exactly the shortest plan, so
`test_integration.py`'s 30/30 assertion is untouched — verified, not assumed.

Written test-first: four assertions in `tests/environments/tossingroom/test_problem.py`
that all fail against `2 * num_rooms + 2`, including one that pins the property the old
formula got wrong — padding a layout with rooms nobody walks to must not buy extra
attempts (`num_rooms=40` still gives 7).

## Does EES learn? Yes — it stops missing, so it stops retrying

The success rate says *that* EES improves; it does not say the improvement is the throw
force rather than something else. This looks directly at the learned quantity. Every
`Throw` issued during an evaluation sweep, greedy only (epsilon never fires at
evaluation time), at the shipped `H = 7`, 3 seeds × 10 cycles × 150 steps, 30 test
tasks, 10000 sampler iterations:

| transitions | solved % | median \|force − target\| | within tolerance | greedy throws per throw-episode |
|---|---|---|---|---|
| 0 (pre-practice) | 64.4 | 0.253 | 22% | 2.57 |
| 150 | 57.8 | 0.249 | **18%** | 2.67 |
| 300 | 75.6 | 0.188 | 36% | 1.98 |
| 450 | 76.7 | 0.233 | 38% | 1.97 |
| 600 | 70.0 | 0.188 | 34% | 2.04 |
| 750 | 96.7 | 0.048 | 84% | 1.22 |
| 900 | 86.7 | 0.125 | 48% | 1.79 |
| 1050 | 94.4 | 0.069 | 63% | 1.50 |
| 1200 | 93.3 | 0.063 | 66% | 1.42 |
| 1350 | 98.9 | 0.048 | 84% | 1.19 |
| 1500 | 96.7 | 0.053 | 78% | 1.28 |

The learned quantity moves, and it moves in the direction the success rate needs.
Median greedy error falls **0.253 → 0.053**, from a quarter of the force range down to
half the tolerance — i.e. from outside the band the throw has to land in, to inside it.
The per-throw hit rate goes **22% → 78%** against a chance rate of 19%, and the policy
correspondingly stops retrying: **2.57 → 1.28** greedy throws per throw-episode, against
a floor of exactly 1.

**That last column is the horizon-robust version of the claim**, and the one worth
keeping: a policy that has learned the force needs one throw, and a policy that is
guessing needs however many the horizon allows. It halved while the solve rate went
64% → 97%. It stopped retrying because it stopped missing.

**Convergence is noisy, not clean.** The 900-transition sweep gives back most of the
750 sweep's gain (84% → 48% within tolerance, median error 0.048 → 0.125) before
recovering. And every one of the three seeds has at least one sweep where the greedy
sampler is *below* its own 19% chance rate — seed 1 at 150 transitions (13%), seed 2 at
150 (17%), seed 0 at 600 (15%) — which is what an underconstrained classifier
confidently picking a wrong region looks like. Those excursions land at *different*
sweeps per seed, so they mostly wash out of the population curve: the ten-seed 10k arm's
mean has exactly one substantial downward step, at 150 transitions (62.3% → 56.7%), and
after that climbs steadily to 99.0% with only two sub-half-point wobbles once it is up
against the ceiling. **The per-seed instability is real; the mean curve is not made of
it** — in particular, no dip appears at 450–600 where seed 0's excursion sits.

**Provenance.** These traces come from the same runs that produced the curves above,
not a re-implementation: `TracingEesMethod`/`TracingEnvironment` are ordinary subclass
overrides driven through the real `PracticeLoop`, and seed 0's per-sweep success
sequence — 18, 19, 21, 20, 14, 27, 27, 30, 29, 29, 30 out of 30 — comes out *identical*
to `results/tossingroom-10000/ees/0/stats.json` from the CLI-driven arm. The
instrumentation does not perturb the run it measures.

**These numbers are not comparable to the same table taken at `H = 16`.** The per-sweep
statistic pools every greedy throw in the sweep, and a task the policy is bad at
contributes more throws than one it is good at — so a longer horizon over-samples the
failures and reports a worse median error and a lower hit rate for the *same* policy.
The old horizon's version of this table is superseded, not carried forward.

**A consistency check on the whole story.** The same retry arithmetic that explained the
unpracticed score, run forward with the *learned* hit rate instead of the chance one,
predicts `0.4(1 − 0.22³) + 0.4(1 − 0.22⁴) + 0.2 ≈ 99.5%` at `H = 7` — against 96.7%
measured on these three seeds and 99.0% on the full ten-seed arm. Consistent to within
what 90 and 300 episodes can resolve, though this direction of the argument is the
weaker one: a *learned* greedy sampler's retries are not independent draws the way a
random sampler's are, since re-throwing at a task the classifier has wrong tends to
reproduce the same wrong force. Treat it as corroboration, not as a second derivation.

## The sampler-iteration grid: a null, and an underpowered one

![The sampler-iteration grid](2026-08-02-tossingroom-sampler-grid.png)

| sampler iters | final % (mean) | sd | worst seed | per-seed final, in seed order |
|---|---|---|---|---|
| 1 000 | 99.0 | 3.2 | 90.0 | 100, 100, 100, **90**, 100, 100, 100, 100, 100, 100 |
| 10 000 | 99.0 | 3.2 | 90.0 | 100, **90**, 100, 100, 100, 100, 100, 100, 100, 100 |
| 100 000 | 98.3 | 4.2 | 86.7 | **86.7**, 100, 100, 100, 100, 100, 100, 100, **96.7**, 100 |

`--sampler-max-train-iters` buys nothing here. The arms are run on the same ten seeds,
so every comparison is **paired** and is tested that way — an unpaired test on a paired
design has already produced a wrong p-value in this project. The test is an exact
Wilcoxon signed-rank over all `2ⁿ` sign assignments (n = 10 on a bounded,
ceiling-clipped percentage is neither large nor normal, so neither the normal
approximation nor a t-test applies):

| pair | metric | effective n | median diff | p |
|---|---|---|---|---|
| 1k vs 10k | final % solved | 2 | 0.0 | 1.000 |
| 1k vs 10k | transitions to 100% | 6 | 0.0 | 1.000 |
| 1k vs 100k | final % solved | 3 | 0.0 | 0.750 |
| 1k vs 100k | transitions to 100% | 8 | 0.0 | 0.789 |
| 10k vs 100k | final % solved | 3 | 0.0 | 0.750 |
| 10k vs 100k | transitions to 100% | 7 | 0.0 | 0.906 |

**This is the opposite of Ball-Ring.** The same three-value grid there
([2026-07-29-eval-protocol-fidelity.md](2026-07-29-eval-protocol-fidelity.md)) found
10 000 beating 1 000 by **+33 points** — 34% → 67% mean, worst seed 0% → 30%, and
*three* seeds collapsed to 0% at 1 000 against none at 10 000 — reported there at
p = 0.016. (That log does not record which test produced the p; the effect size and the
collapse counts are what this comparison rests on, and those are unambiguous.)

The domains differ in exactly the way that predicts it: Ball-Ring's sampler has to
learn a 2-D grasp/place parameterisation against geometric constraints, while Tossing
Room's has to learn a **single scalar** — one throw force against one per-task target,
on a 1-D interval where the answer is 20% of the range wide. A thousand iterations
already finds it, so more cannot help.

Tossing Room is not the first domain here to behave this way. That same grid found
Light Switch saturating identically (final means 100 / 100 / 99, "the domain saturates,
so it cannot discriminate these settings at all"). Two of three domains cannot see this
lever; the one that can is the one with the hardest sampler.

**How underpowered is this, exactly?** The honest answer is not an MDE from a t-test,
because the limiting factor is not sample size but **censoring**. Nine or ten of ten
seeds sit *exactly* at the oracle's 100% ceiling in every arm, so most pairs tie and
drop out: `n` falls from 10 to 2–3 on the endpoint. An exact signed-rank at `n` pairs
cannot produce a two-sided p below `2 / 2ⁿ`, so **`n = 3` has a p floor of 0.25** — the
100k arms could have been beaten by every seed that moved and still not have cleared
0.05. The endpoint on this domain is not a low-power measurement; it is an unusable
one.

That is also the honest contrast with Ball-Ring: a +33-point effect on an arm whose
seeds ranged from 0% to well below the ceiling had room to move consistently in one
direction across seeds, which is exactly what a signed-rank needs to reach a small p.
Tossing Room has no headroom in which to be consistent.

The `transitions to 100%` column exists for exactly this reason — it is the only
statistic left with any range once the endpoint saturates, since a seed that reaches
100% at 450 transitions and one that reaches it at 1200 both score 100 at the end. It
keeps 6–8 pairs instead of 2–3, and it is *also* null: median difference 0 transitions,
p ≥ 0.789, and the median seed in all three arms first hits 100% at the same 900
transitions. Two independent metrics, no effect in either.

**What this does not license.** "1 000 is enough for Tossing Room" is supported. "The
sampler budget doesn't matter" is not — Ball-Ring is a live counterexample in this same
codebase, and nothing here rules out a difference smaller than this design can see.

## Reproducing

```bash
# the bracket's lower bound and the three EES arms -- ARMS RUN ONE AT A TIME
python -m scripts.run_sweep --env tossingroom --methods random-skills --num-seeds 10 \
  --results-root results/tossingroom-random --shared-args "--num-test-tasks 30" \
  --method-args "random-skills=--num-cycles 10 --max-steps-per-interaction 150"
for iters in 1000 10000 100000; do
  python -m scripts.run_sweep --env tossingroom --methods ees --num-seeds 10 \
    --results-root results/tossingroom-$iters --shared-args "--num-test-tasks 30" \
    --method-args "ees=--num-cycles 10 --max-steps-per-interaction 150 \
                        --sampler-max-train-iters $iters"
done

# the curves, the grid figure, and the paired tests -- reads results/ only
python -m analysis.practice_makes_perfect.tossingroom_comparison \
  --arm "EES (1k sampler iters)=results/tossingroom-1000" \
  --arm "EES (10k)=results/tossingroom-10000" \
  --arm "EES (100k)=results/tossingroom-100000" \
  --random-skills-root results/tossingroom-random \
  --output docs/experiment-logs/2026-08-02-tossingroom-ees-curves.png \
  --grid-output docs/experiment-logs/2026-08-02-tossingroom-sampler-grid.png

# the horizon probe (JSON committed, so the table regenerates without re-running)
python -m scripts.tossingroom_horizon_sweep --max-horizon 16 --num-seeds 10 \
  --num-test-tasks 30 \
  --output docs/experiment-logs/2026-08-02-tossingroom-horizon-sweep.json
python -m analysis.practice_makes_perfect.tossingroom_horizon_table \
  --traces docs/experiment-logs/2026-08-02-tossingroom-horizon-sweep.json

# the throw-force traces (JSON committed, likewise)
python -m scripts.tossingroom_throw_traces --label "EES (10k)" \
  --sampler-max-train-iters 10000 --num-seeds 3 \
  --output docs/experiment-logs/2026-08-02-tossingroom-throw-traces.json
python -m analysis.practice_makes_perfect.tossingroom_throw_convergence \
  --traces docs/experiment-logs/2026-08-02-tossingroom-throw-traces.json \
  --output docs/experiment-logs/2026-08-02-tossingroom-throw-convergence.png
```

Both JSONs are committed next to this file, so every table and figure here regenerates
from the `analysis/` half alone — no re-run, and no chance of a number drifting from the
data that produced it.

Fast Downward's timeout is wall-clock, so arms must be run **sequentially** — unequal
CPU load between arms turns into unequal planning-failure rates and contaminates
exactly the comparison being made. Every arm above is seeded 0..9 by `run_sweep`, and
both probe scripts fix their own seeds the same way, so the horizon table reproduces
bit-for-bit: it was re-derived from scratch for this writeup and came back identical to
the run that motivated the change.

---

## Follow-up: a missed `Throw` now releases the item

The horizon fix above treats the symptom. The **mechanism** was that a miss cost nothing:
`_apply_throw` only mutated state on success, so after a miss the robot still held the
item and still stood in the bin room — a bit-identical state, and the very next step
re-threw for free. The horizon merely set how many free re-rolls the evaluation granted.

Throwing now releases the item whether or not it lands:

```python
+ next_state.set(obj=self.robot, feature_name="holding", feature_val=0.0)
  if robot_room == bin_room and abs(raw_force - target) < self.throw_tolerance:
      next_state.set(obj=bin_obj, feature_name="count", feature_val=count + 1.0)
- next_state.set(obj=self.robot, feature_name="holding", feature_val=0.0)
```

The thrown item is **gone, not recoverable**. Items are singleton discriminators carrying
only `(kind, target_force)` with no position, so "it is lying near the bin" is not
representable — and making it so would reintroduce exactly the cheap retry this removes.
The only way to try again is a fresh item from the limitless pile, costing a round trip
to the start room: affordable inside a 100-step practice period, impossible inside an
evaluation horizon of longest-solve + 2. Same dynamics in both modes; the **budget** does
the work, which is how Ball-Ring already makes a failed placement terminal.

### Effect on an unpracticed policy

| | miss was free | miss releases |
|---|---|---|
| unpracticed EES | 62.7% | **38.7%** |
| `Throw` actions / 300 episodes | 648 | **240** |
| throws per episode | 2.16 | **0.80** |

`0.80` is the confirmation: exactly one throw per throw-task, zero for the `EMPTY` family
(20% of tasks, solved by `Press`). The retry channel is closed. 38.7% is what a single
attempt predicts — `0.2 x 100 + 0.8 x 19 = 35.2%`, against a standard error of ~2.8pp.

### The sampler-iteration grid is now a *valid* null, not a censored one

10 seeds, 30 test tasks, 25 cycles x 100 steps:

| arm | mean | sd | worst seed | seeds at ceiling |
|---|---|---|---|---|
| random skills | 3.7 | 4.0 | 0.0 | 0/10 |
| EES, 1000 | 93.3 | **14.8** | **56.7** | 8/10 |
| EES, 10000 | **95.0** | **5.0** | 83.3 | 3/10 |
| EES, 100000 | 94.0 | 6.0 | 83.3 | 3/10 |

Paired over the same seeds: all comparisons p ~ 0.68-0.91, with only 2/10 ties.

That last column is why this null is worth more than the previous one. Under the old
dynamics 9-10 of 10 seeds sat exactly on the ceiling, so paired seeds tied, the effective
n collapsed to 2-3 and the exact Wilcoxon floored at p = 0.25 — the endpoint could not
have detected an effect of any size. Now the arms have real spread and the test reports a
genuine null: **on this domain the sampler-iteration count does not change the success
rate.** That remains the opposite of Ball-Ring, where 10000 beat 1000 by +33 points.

One signal the old metric could not show: **1000 iterations is bimodal.** Its per-seed
finals are `[56, 76, 100 x 8]` — eight perfect runs and two poor ones — against
`83-100` for 10000. Same mean, but **8.7x the variance** (sd 14.8 vs 5.0) and a worst seed
27 points lower. Reporting only the mean would call these identical; they are not, and the
more-trained sampler is the more *reliable* one even though fewer of its seeds are perfect.

### What this does not change

The horizon fix stands on its own: a 16-step budget for a 5-skill solve is miscalibrated
regardless, and `longest_shortest_solve() + 2` ports Light Switch's `H_eval` convention.
The two changes are complementary — the horizon sets an honest budget, and the release
makes the budget mean something.

Note one further behavioural consequence: throwing in the *wrong* room also releases the
item now. That follows from "throwing is a release"; only an empty-handed throw remains a
true no-op.

### What the trained policy actually does, one clip per goal family

The clips below are **not fresh demo runs**. Each is the `10000`-iteration arm's own run
at that seed, re-run with `--num-render-checkpoints 2` and verified to reproduce
`results-release/ees10000/ees/<seed>/stats.json` **evaluation-for-evaluation** — so these
are literally the policies behind the table above, at its final sweep (25 cycles, 2500
transitions), on that run's first test task. Sampler budget is the `main` default,
**10000** (`--sampler-max-train-iters` left unset).

**Seed rule: the lowest seed whose first test task belongs to that family.** A seed fixes
the whole test set, so which family test task 0 lands in is fixed too (families are drawn
`0.4/0.4/0.2`); that gives TRASH → seed 0, RECYCLING → seed 5, EMPTY → seed 4. The rule is
deliberately not success-selected — **seed 0 is the arm's worst seed** (25/30 = 83.3%
final, the low end of the `83-100` range quoted above), and it is used anyway.

Success is read off the bin-count badge in the last frame (`T:1` / `R:1` / `R:0 T:0`), not
off the frame count; the force in each label is rounded to 2dp by the renderer, so it is
quoted approximately.

**TRASH, trained** — `Pickup(trash)`, `MoveRoom` 3→4→5→6, `Throw` at ≈0.77 against this
task's `target_force` 0.712. Five actions, the shortest solve this family admits, and the
throw lands (`T:1`): the learned force is inside the 0.1 tolerance window on the first
attempt, so there is no second attempt to make.

![EES, trained, TRASH](2026-08-03-tossingroom-ees-trash.gif)

**TRASH, unpracticed** (same seed, same task, the sweep *before* any practice) — the same
five-action plan, `Throw` at ≈0.73, and it **also lands**. Shown deliberately rather than
quietly dropped: a uniform force lands within 0.1 of 0.712 about 19% of the time, and this
is one of those times. It is the whole thesis of this PR in one frame — an unpracticed
policy's apparent competence here is a coin flip, not a learned sampler, and at the old
horizon it also had eleven more coin flips to spend.

![EES, unpracticed, TRASH](2026-08-03-tossingroom-ees-trash-untrained.gif)

**RECYCLING, trained** — `Pickup(recycling)`, `MoveRoom` 3→2→1, `Throw` at ≈0.95 against
`target_force` 0.988, error ≈0.04 against a 0.1 tolerance. Four actions, again the shortest
solve, `R:1`.

![EES, trained, RECYCLING](2026-08-03-tossingroom-ees-recycling.gif)

**RECYCLING, unpracticed** — the same approach, then `Throw` at ≈0.38 against 0.988: a miss
by 0.6, the item is released, and the remaining three frames are `no-op (no plan)`. Fast
Downward is right that there is no plan. The pile is in room 3, the recycling bin in room 1,
and `blocked_right_from = 2` makes stepping right from room 2 back into room 3 impossible —
so **for the RECYCLING family a missed throw is terminal at any horizon**, not merely
expensive. The "fetch a fresh item" retry route the release change leaves open exists only
for TRASH, whose bin (room 6) sits on the reachable side of the ledge.

![EES, unpracticed, RECYCLING](2026-08-03-tossingroom-ees-recycling-untrained.gif)

**EMPTY, trained** — `MoveRoom` 3→4→5→6 then `Press`, four actions, both bins to zero
(`R:0 T:0`). This family contains no `Throw` at all, so it is deterministic and neither the
release change nor the horizon change touches it; its unpracticed clip is identical and is
not reproduced here.

![EES, trained, EMPTY](2026-08-03-tossingroom-ees-empty.gif)

These are 1280x240 at 2 fps, against the July skill-oracle clips'
(`2026-07-24-tossingroom-oracle-*.gif`) 1000x188 at ~6 fps — a `TossingRoomRenderer`
figure-size and `render_fps` difference, not a regression. File sizes match that precedent
(22-26 KB each).

#### Reproducing the clips

```bash
# one run per family; `--num-render-checkpoints 2` records the pre-practice sweep as
# episode_000000.mp4 and the final one as episode_002500.mp4. Run these ONE AT A TIME.
for seed in 0 5 4; do
  python -m hitl_pmp.cli --env tossingroom --method ees --seed $seed \
    --num-test-tasks 30 --num-cycles 25 --max-steps-per-interaction 100 \
    --sampler-max-train-iters 10000 --num-render-checkpoints 2 \
    --output-dir results/demo-$seed
done
```

Each run's `stats.json` `evaluations` must equal `results-release/ees10000/ees/$seed/`'s
exactly; all three did. The mp4 → gif step quantises to a shared 64-colour palette before
writing, which is what keeps these at the ~25 KB of the oracle precedent rather than
~130 KB: an mp4 round trip adds per-pixel compression noise everywhere, which defeats GIF's
inter-frame delta encoding until the palette collapses it again.

```python
import imageio
import numpy as np
from PIL import Image

frames = [np.asarray(f) for f in imageio.get_reader(video_path)]
palette = Image.fromarray(np.concatenate(frames, axis=0)).quantize(
    colors=64, method=Image.Quantize.MEDIANCUT
)
images = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
images[0].save(
    gif_path,
    save_all=True,
    append_images=images[1:],
    duration=500,  # 2 fps, matching TossingRoomCli.render_fps
    loop=0,
    optimize=True,
    disposal=1,
)
```

`VideoWriter.write_gif` remains the plain mp4 → gif path; this is a post-processing step on
top of the same frames, deliberately kept in this log rather than pushed into `core/` —
this PR's source diff is two files, and a rendering-size tweak does not belong in it.

#### A negative result: `--goal-type` is not the right way to make these

The obvious approach — `--goal-type trash` to force a deterministic single-family demo — is
wrong, and expensively so. `forced_goal_type` pins **training** tasks as well as test ones,
which is a different experiment from any arm in this log. Run at seed 0 with the protocol
above it scores **2/30**, with a sweep sequence that flips between 30/30 and 2/30 to the
end (its last seven sweeps are `30, 2, 30, 30, 2, 30, 2`): a test set that is 100% throw
tasks amplifies the
sampler's per-seed instability into an all-or-nothing signal, where the mixed distribution's
20% `Press`-only tasks would have floored it around 6/30. Use `--goal-type` for a
*deterministic* demo of a fixed policy (the skill oracle, as in PR #25); do not use it to
demonstrate a learning method.

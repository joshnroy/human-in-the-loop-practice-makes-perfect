# Tossing Room: EES learns the throw force, and the evaluation horizon was hiding it

**Result.** EES's learned sampler drives `Throw`'s force onto each task's own
`target_force`: the median greedy error falls from 0.34 (chance) to 0.03, and the
per-throw hit rate goes 19% → 100%. The success-rate curve could not show this,
because `TossingRoomProblem.max_episode_steps` was `2 * num_rooms + 2 = 16` against a
5-skill solve — eleven spare steps, every one of them a free retry of the single
stochastic skill. An **unpracticed** EES scored 94.7% that way (every one of 10 seeds
between 87% and 100%). Porting Light Switch's `H_eval` convention (longest shortest
solve + 2 = 7) makes the curve legible without touching a single practice step.

![EES vs. the random-skills lower bound](2026-08-02-tossingroom-ees-curves.png)

At the corrected horizon the curve has somewhere to go: EES starts at 62.3% — an
unpracticed policy guessing the throw force — and ends at 99.0%, against a
random-skills floor that never leaves 7%. All three sampler-iteration arms share the
same first sweep to the decimal (62.3%), which is the sanity check that the budget
cannot matter before any training has happened.

![The throw force converges onto each task's target](2026-08-02-tossingroom-throw-convergence.png)

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

## Does EES learn? Yes — and this is horizon-independent

Measured *before* the horizon change, so it cannot be an artifact of it. Every `Throw`
issued during an evaluation sweep, greedy only (epsilon never fires at evaluation
time), seed 0, 10 cycles × 150 steps, 30 test tasks, 10000 sampler iterations:

| sweep | greedy throws | mean \|force − target\| | within tolerance |
|---|---|---|---|
| 0 (pre-practice) | 124 | 0.339 | 19% |
| 1 | 101 | 0.272 | 21% |
| 2 | 107 | 0.322 | 20% |
| 3 | 205 | 0.328 | **6%** |
| 4 | 209 | 0.332 | **6%** |
| 5 | 40 | 0.102 | 65% |
| 6 | 27 | 0.043 | 96% |
| 7 | 29 | 0.053 | 90% |
| 8 | 26 | 0.034 | 100% |
| 9 | 26 | 0.036 | 100% |
| 10 | 26 | 0.033 | 100% |

Two things worth naming:

1. **The count is the tell.** 26 greedy throws across 30 test tasks is one throw per
   throw-goal task (the rest are the `EMPTY` family, which needs none). The policy
   went from needing 124 attempts to needing 26 — i.e. it stopped retrying, because it
   stopped missing.
2. **It gets worse before it gets better.** At sweeps 3–4 the classifier's argmax is
   *below* chance (6% versus 19%) and attempts nearly double. An underconstrained
   classifier confidently picking a wrong region is worse than picking at random. This
   is the same instability the Ball-Ring convergence work saw, and it is what the
   dip at 600 transitions in the success curve below is made of.

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

**This is the opposite of Ball-Ring**, where the same grid found 10 000 beating 1 000
by **+33 points (p = 0.016)**. The domains differ in exactly the way that predicts it:
Ball-Ring's sampler has to learn a 2-D grasp/place parameterisation against geometric
constraints, while Tossing Room's has to learn a **single scalar** — one throw force
against one per-task target, on a 1-D interval where the answer is 20% of the range
wide. A thousand iterations already finds it, so more cannot help.

**How underpowered is this, exactly?** The honest answer is not an MDE from a t-test,
because the limiting factor is not sample size but **censoring**. Nine or ten of ten
seeds sit *exactly* at the oracle's 100% ceiling in every arm, so most pairs tie and
drop out: `n` falls from 10 to 2–3 on the endpoint. An exact signed-rank at `n` pairs
cannot produce a two-sided p below `2 / 2ⁿ`, so **`n = 3` has a p floor of 0.25** — the
100k arms could have been beaten by every seed that moved and still not have cleared
0.05. The endpoint on this domain is not a low-power measurement; it is an unusable
one.

That is also the honest contrast with Ball-Ring: +33 points at p = 0.016 means the
effect was near-unanimous in sign across seeds, which is what a signed-rank needs.
Tossing Room has no headroom in which to be unanimous.

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

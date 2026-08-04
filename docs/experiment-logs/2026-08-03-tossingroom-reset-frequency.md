# Trading cycles against steps cannot isolate reset frequency (Tossing Room)

This experiment set out to ask whether a longer practice period (fewer free resets)
makes the irreversible `RECYCLING` family fall further behind the recoverable
`TRASH` family. **It could not answer that question** — the design varies reset
frequency and sampler-refit count with the same flag — but it established four
other things, three of which nobody was looking for.

**What it established:**

1. **[Concurrency does not perturb results on this machine.](#concurrency-does-not-perturb-results-here)**
   13,105 live Fast Downward process observations at a max lifetime of **0s**
   against a 10s wall-clock budget, at loads 9–27 on 24 cores; and `stats.json`
   **byte-identical** between a low-load probe and a 20-way concurrent sweep, on
   both extreme arms. **This retires the standing sequential-arms rule** in the
   Ball-Ring logs.
2. **[At fixed experience, `--num-cycles` sets competence — the transition count does
   not.](#result-1-at-fixed-experience---num-cycles-sets-competence-not-the-transition-count)**
   Arms A and D consumed identical 2500 online transitions (sd 0) and ended ~40
   points apart: armD − armA is **TRASH −40.4pp, RECYCLING −44.7pp, both
   p = 0.0156** (paired exact).
3. **[That 40pp *is* the confound, measured](#result-2-the-design-confounds-resets-with-refits--do-not-re-tread).**
   `--num-cycles` sets the number of free resets *and* the number of `end_cycle()`
   sampler refits with one number, so no sample size could have separated them —
   and the cost of that entanglement is 40 points, far larger than any gap
   difference the experiment set out to detect. Filed below as a **do-not-re-tread**
   record.
4. **`EMPTY` is a perfect control** — 100% in every seed of every arm, sd exactly 0,
   difference exactly 0 — so all movement is confined to the two families that use
   the stochastic `Throw`. That result directly motivated reallocating `EMPTY`'s
   share of the evaluation budget (**PR #41**, a fixed 14/14/2 test-set
   composition).

**What must not be claimed.** This experiment says nothing about whether reset
frequency affects the price of an irreversible action, in either direction. The
measured (TRASH − RECYCLING) gap does not trend with practice-period length, but a
design in which the arms differ by 40 competence points could not have told a real
trend apart from a training difference — so the null is a fact about *this data*,
not a finding about the harness. The premise is not established either: **no arm
shows RECYCLING behind TRASH at p < 0.05** (arm C, the near-miss, is p = 0.0547).
The corrected design is [below](#result-2-the-design-confounds-resets-with-refits--do-not-re-tread).

![gap vs practice-period length](./2026-08-03-tossingroom-reset-freq-gap.png)

## Concurrency does not perturb results here

The Ball-Ring logs run arms **sequentially**, on the stated reasoning that "Fast
Downward's timeout is wall-clock, so concurrent arms bias each other"
(`2026-08-03-ballring-iters.md`, `2026-07-24-ballring-ees.md`). That caution was
reasonable and it is measurable. Measured, it is **unnecessary**, and it has been
costing serial wall-clock on every sweep since it was written.

**Direct evidence — byte-identity.** Seed 0 of both extreme arms was run twice:
once at low load (2 concurrent runs) and once inside the full 20-way concurrent
sweep. `evaluations`, `breakdowns`, and the whole `stats.json` are **identical** in
both arms. This is the strong form of the check: a spurious FD timeout necessarily
changes the plan and therefore the trajectory, so identity across a 10x difference
in machine load proves no timeout perturbation occurred.

**Direct evidence — planner headroom.** 600 samples at 1 Hz over 10 minutes of the
full sweep, **13,105 live Fast Downward process observations**: max process
lifetime **0s** against the 10s budget, and **zero** observations at or above 5s.

| sample | concurrent runs | load avg (24 cores) | max FD lifetime |
|---|---|---|---|
| 09:41 | 22 | 14.0 | 0s |
| 09:46 | 25 | 26.9 | 0s |
| 09:51-10:01 | 13-23 | 9.1-27 | 0s (13,105 observations) |

**There is no planning-failure rate in the logs to report.** `PlanningFailure` is
caught silently at three sites in `ees_method.py` (`refresh_planning_progress_plans`,
the goal-phase plan, and `_practice_plan`) and nothing is printed, so a per-run
failure count cannot be recovered from `log.txt`. Byte-identity is the instrument
that replaces it, and it is a stronger one.

> **A measurement bug worth recording**, because it nearly became a reported number.
> The first FD sampler reported a max process lifetime of **4,123,168,608 seconds**.
> Under load `ps` occasionally emits a partial row whose first field is not an
> elapsed time, and `sort -rn | head -1` takes the max over *all* rows, so a single
> malformed line poisons the maximum for an entire sampling run. Inspecting raw `ps`
> output showed every live FD process at 0s. The rewritten sampler rejects
> non-numeric and implausible (>86400s) values and separately counts observations at
> or past half the budget.

Concurrency was therefore used deliberately: four `run_sweep` processes at once,
weighted toward the long pole (arm A `--max-workers 9`, B 5, C 3, D 3), ~20
concurrent against 24 cores. All 40 runs succeeded; no arm had a failure. The scope
of this claim is *this machine* — 24 cores, this FD build, this domain's planning
problems. It is a measurement, repeatable by the same instruments, not a general
theorem about wall-clock timeouts.

## The question

`PracticeLoop.run` calls `problem.reset_to_task(task=task)` at the top of every
practice cycle:

```python
for cycle in range(num_cycles):
    task = problem.sample_train_task()
    policy = method.get_practice_policy(task=task)
    state = problem.reset_to_task(task=task)
    for _ in range(max_steps_per_interaction):
        ...
```

So the harness hands the robot a **free environment reset every
`--max-steps-per-interaction` steps**. That is faithful to predicators
(`main.py:301-302` resets per interaction request) and correct for a
reproduction. But it also caps what an irreversible action can cost: the worst
outcome is "you wasted the rest of this practice period", never "you wasted the
rest of the run".

Tossing Room has exactly one genuinely terminal failure family:

| family | `Throw`? | a miss costs |
|---|---|---|
| `EMPTY` | no — `MoveRoom`xk + `Press` | nothing; the family is deterministic |
| `TRASH` | yes | a round trip to the pile for a fresh item — expensive, recoverable |
| `RECYCLING` | yes | **terminal** — pile in room 3, recycling bin in room 1, `blocked_right_from = 2` makes room 3 unreachable once the item is gone, so Fast Downward correctly reports no plan |

At the shipped config (25 cycles x 100 steps, 10 seeds, 30 test tasks) RECYCLING
trails TRASH at every checkpoint but does not stall:

| family | 0 | 300 | 600 | 900 | 1200 | 1500 | 1800 | 2100 | 2500 |
|---|---|---|---|---|---|---|---|---|---|
| `EMPTY` | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| `TRASH` | 28% | 28% | 54% | 78% | 76% | 86% | 86% | 86% | 98% |
| `RECYCLING` | 18% | 19% | 39% | 44% | 55% | 81% | 81% | 83% | 92% |

**Hypothesis.** The penalty for irreversibility is bounded by the reset
frequency. Longer practice periods (fewer free resets) mean a stranded robot
wastes more experience, so RECYCLING should fall further behind TRASH. Shorter
periods should shrink the gap.

## Design

Total online transitions are held at 2500 and cycles are traded against steps:

| arm | `--num-cycles` | `--max-steps-per-interaction` | free resets |
|---|---|---|---|
| A | 50 | 50 | 50 |
| B | 25 | 100 | 25 (shipped) |
| C | 10 | 250 | 10 |
| D | 5 | 500 | 5 |

10 seeds each (0..9, fixed by `run_sweep`), `--num-test-tasks 30`,
`--sampler-max-train-iters 10000`, `--env tossingroom --method ees`.

**Primary metric: the within-arm (TRASH - RECYCLING) final success gap, paired by
seed.** Not the absolute rates. `--num-cycles` also sets how many times the
sampler refits (`method.end_cycle()` runs once per cycle) and how many evaluation
sweeps run, so arm D gets 5 refits where arm A gets 50 and **absolute success
rates are not comparable across arms**. Both families inside one arm see the same
refits, so the within-arm gap cancels that confound's effect on the *level*.

**That is only half of what the metric needs, and the missing half is the whole
experiment.** The gap cancels the level, but the gap is *itself* a function of
training progress, and a hump-shaped one: near zero while both families sit at the
floor, largest mid-training when TRASH pulls ahead, near zero again once both
saturate. The shipped-config table above traces exactly that hump -- 10pp at
transition 0, **34pp at 900**, 6pp at 2500. Since `--num-cycles` sets progress,
the arms can sit at different points on that hump and their final gaps can differ
for that reason alone, with no contribution from irreversibility whatsoever.

So a cross-arm difference in the gap is attributable to reset frequency **only if
the arms end at comparable competence**. That is a measurable precondition, not an
assumption, and it is checked before any p-value is read:

- **final TRASH level, paired across arms** -- are the arms equally trained?
- **final EMPTY level, paired across arms** -- the control. No `Throw`, no
  stochastic skill, deterministic `MoveRoom`xk + `Press` plan. If EMPTY moves,
  something other than irreversibility is driving the result.

## Result 1: at fixed experience, `--num-cycles` sets competence, not the transition count

Paired across the same 10 seeds, exact tests, arm D (500 steps) minus arm A (50):

| family | mean difference | sd | Wilcoxon p | sign-flip p | n for 80% power |
|---|---|---|---|---|---|
| `TRASH` | **-40.4pp** | 38.1 | **0.0156** | **0.0156** | 7 |
| `RECYCLING` | **-44.7pp** | 35.9 | **0.0156** | **0.0156** | 5 |
| `EMPTY` | **+0.0pp** | 0.0 | 1.0000 | 1.0000 | infinite (zero effect) |

Every arm reached **exactly 2500 online transitions, sd 0** — the experience budget
is identical by construction, not approximately equal. Yet both throw families move
by ~40 points between the extreme arms, and both are significant. **Transition count
alone does not determine how much this method learns in this harness**; how the same
transitions are divided into cycles does. That was not what the experiment was
looking for, and it is the most portable thing in it.

It is also the precondition failing: the arms are nowhere near equally trained, so
they do not sit at comparable points on the hump, and the designed cross-arm
comparison is confounded at a magnitude far larger than any gap difference it
reports.

**`EMPTY` is a perfect control**: 100% in every seed of every arm, sd exactly 0,
difference exactly 0. So whatever moves across arms is confined to the two families
that use the stochastic `Throw` — this is not some generic harness artifact. It is
also the reason the control cannot do more work than that: `EMPTY` is solved by an
untrained policy, so it sits on the ceiling and could not have shown movement. A
test family that is 100% in all 40 runs is spending evaluation budget to learn
nothing, which is why PR #41 cuts it to 2 of 30 tasks and moves the rest to the two
families that vary.

**What this does *not* license saying.** It is tempting to sharpen this to "learning
is driven by sampler refits, not by transitions" — same 2500 transitions, 40 points
apart, and refits are the obvious channel. But `--num-cycles` moves refits and
resets *together* in this design, so the 40pp is attributable to either, and this
experiment cannot separate them. The honest statement is the heading's: **holding
transitions fixed at 2500, final competence depends strongly on `--num-cycles`, and
this design cannot say which of its two effects is responsible.** Separating them
needs an arm that varies refit count at fixed reset frequency, which is not one of
the four run here.

## Result 2: the design confounds resets with refits — do not re-tread

This is a structural failure, not a statistical one: `--num-cycles` sets both
quantities with one number, so **no sample size could have separated them**. Filed
in the form of the handoff's refuted-hypotheses table so the same experiment is not
rebuilt later:

| design | what it was meant to isolate | why it cannot work | corrected design |
|---|---|---|---|
| Hold total transitions at 2500, trade `--num-cycles` against `--max-steps-per-interaction` (50x50, 25x100, 10x250, 5x500) | how often the harness hands out a free reset | `--num-cycles` is simultaneously the reset count **and** the `end_cycle()` refit count. One flag, two treatments, perfectly collinear across arms — measured cost: **40.4pp TRASH / 44.7pp RECYCLING** between the extreme arms at identical experience (p = 0.0156). The within-arm gap cancels the *level* but not the hump-shaped dependence of the gap on progress. | Vary a **within-period reset interval** while holding `--num-cycles` (and so refits, and so evaluation sweeps) **fixed**: reset to the current practice task's initial state every k steps *inside* a period, without ending the cycle. `practice_reset_interval` on `josh/experiment/reset-interval` does exactly this; `None` reproduces today's behaviour. |

Two things the corrected design needs to be honest rather than merely implemented,
both already handled on that branch and both worth knowing before anyone rebuilds
it:

- A `Method` must be told the state the environment is about to leave
  (`observe_environment_reset`). EES scores a skill against its `add_effects` on the
  *next* state it sees, so without the hook every skill executed just before a reset
  is judged against a freshly reset environment — where `InBin`-style effects
  essentially never hold — and recorded as a failure into both its competence model
  and its sampler's training data. That mislabelling **scales with reset frequency**,
  which is exactly the quantity being varied.
- The manipulation must be verifiable from the run's own output
  (`Metrics.num_practice_resets`) rather than from an argument about the loop's
  arithmetic.

## Result 3: the measured gap across arms

| arm | period | resets | final TRASH | final RECYCLING | final EMPTY | **gap (T-R)** |
|---|---|---|---|---|---|---|
| A | 50 | 50 | 96.8 ± 6.7 | 98.0 ± 6.3 | 100.0 ± 0.0 | **-1.2 ± 6.3** |
| B | 100 | 25 | 97.1 ± 6.7 | 91.7 ± 16.2 | 100.0 ± 0.0 | **+5.4 ± 19.0** |
| C | 250 | 10 | 68.2 ± 34.7 | 32.4 ± 39.8 | 100.0 ± 0.0 | **+35.8 ± 49.3** |
| D | 500 | 5 | 56.5 ± 36.5 | 53.3 ± 36.2 | 100.0 ± 0.0 | **+3.2 ± 23.8** |

The gap is **non-monotone** in practice-period length, and the arm with the *fewest*
free resets sits near zero. Read the final-TRASH column next to the gap column: the
gap is largest exactly where TRASH is mid-curve (arm C, 68%), and small at both the
saturated end (arms A and B, ~97%) and the barely-trained end (arm D, 57%, where
RECYCLING at 53% has not separated from TRASH either). That is the shape the hump
predicts from the competence differences of Result 1, with no reset effect required.

Three tests of the trend, all paired on the same 10 seeds, all exact:

| test | mean | sd | Wilcoxon p | sign-flip p | n for 80% power |
|---|---|---|---|---|---|
| gap, armD (500) - armA (50) | +4.33pp | 22.72 | 0.6875 | 0.5938 | ~216 |
| per-seed slope, pp per doubling of period | +4.27 | 7.37 | 0.1055 | 0.0977 | ~23 |
| per-seed Spearman rho of gap vs period | +0.05 | 0.55 | 0.5469 | 0.7617 | ~787 |

**Nothing reaches p < 0.05.** The per-seed slope is the only borderline quantity
(p ≈ 0.10, ~23 seeds would resolve it), but running those 23 seeds would buy
nothing: its *direction* is uninterpretable while the arms differ by 40 points in
competence, since a positive slope is exactly what the hump predicts with no
irreversibility effect at all.

The trend is tested at n = 10 across *seeds* rather than n = 4 across arm means
because Spearman on four arm means bottoms out at p = 0.083 even for perfect
monotonicity, so it could never reach significance. Both per-seed forms are
reported; the slope is the one to read.

## Result 4: no arm establishes the gap at all

Within-arm (TRASH - RECYCLING) against zero:

| arm | mean gap | Wilcoxon p | sign-flip p | exact-zero seeds |
|---|---|---|---|---|
| A | -1.2pp | 1.0000 | 1.0000 | 8/10 |
| B | +5.4pp | 0.6250 | 0.5000 | 5/10 |
| C | **+35.8pp** | **0.0547** | **0.0547** | 1/10 |
| D | +3.2pp | 0.8438 | 0.7188 | 2/10 |

Not one arm reaches p < 0.05. Arm C is the near-miss. So at 10 seeds this data does
not establish even the premise — that the irreversible family trails the
recoverable one — let alone that the trailing grows with practice-period length.

Arms A and B are at the ceiling (8/10 and 5/10 seeds tie at exactly zero because
both families are at 100%), which collapses the effective n of the paired test.

## Result 5: holding progress fixed, the gap is flat

The un-confounded companion. Read each seed's gap at the first checkpoint where its
*own* TRASH rate reaches a level every seed of every arm attains (43.8%), rather
than at a common transition count:

| arm | seeds matched | mean progress-matched gap | sd |
|---|---|---|---|
| A | 10/10 | +30.4pp | 30.8 |
| B | 10/10 | +38.2pp | 29.1 |
| C | 10/10 | +34.4pp | 32.1 |
| D | 10/10 | +22.0pp | 17.5 |

All ten seeds match in every arm, and only 2/10 match before any practice, so this
comparison has real content rather than being the untrained gap in disguise. Hold
training progress fixed and the gap is **flat across arms**. (It also shows the gap
is genuinely ~20-40pp mid-training; the near-zero final gaps in arms A/B are
saturation, not absence.)

This is the closest thing here to a clean cross-arm comparison, and it is still not
a clean one: matching on TRASH competence is a post-hoc alignment on an outcome, not
a randomised control of the treatment.

## How much of the spread is just task sampling?

`goal_weights` is `(0.4, 0.4, 0.2)` and is *sampled*, so 30 test tasks split
unevenly — a seed holds as few as 10 RECYCLING tasks, so its rate moves in steps of
10pp.

> **This sampled composition was superseded by PR #41, and the boundary matters when
> reading this log beside newer results.** Every number here was produced under the
> *sampled* test set described above, so the noise-floor analysis below is correct for
> the design as run. Tossing Room runs produced after `29e97fa` draw a **fixed 14
> TRASH / 14 RECYCLING / 2 EMPTY** split instead — a different evaluation set, whose
> per-family rates are **not directly comparable** to these.

| arm | observed gap sd | predicted from binomial task sampling alone | ratio |
|---|---|---|---|
| A | 6.3pp | 20.7pp | 31% |
| B | 19.0pp | 20.7pp | 92% |
| C | 49.3pp | 20.7pp | 239% |
| D | 23.8pp | 20.7pp | 115% |

This table is what tells a reader what the design could and could not have found.
**Arm A's 31% is not precision** — it is saturation. 8 of its 10 seeds tie at
exactly zero because both families are at 100%, which produces a small sd for a
reason that has nothing to do with measurement quality. Arms B and D sit at
essentially the noise floor, meaning their spread is mostly *which* tasks each seed
drew, and more seeds would not help — more test tasks per family would. Only arm C
carries spread well above the floor, and that is genuine seed heterogeneity rather
than measurement noise: 4 of its 10 seeds finish with RECYCLING at exactly **0%**.

Per-seed final RECYCLING, showing that collapse is concentrated, not spread:

```text
armA   100  100  100  100  100  100   80  100  100  100
armB    50   83  100  100   83  100  100  100  100  100
armC     0    0   36  100    8    0    0  100   50   30
armD     0   17   50   36  100  100   40  100   60   30
```

Four zeros in arm C, one in arm D. If the hypothesis were driving this, arm D
(fewest resets) should show the most collapse. It shows less.

## Per-family learning curves

![per-family learning curves](./2026-08-03-tossingroom-reset-freq-curves.png)

One panel per arm, mean ± standard error over 10 seeds. Levels are **not**
comparable across panels — the number of cycles is the number of sampler refits, so
the panels differ in training as well as in reset frequency. That is the confound of
Result 2, drawn.

## What was verified

- **Arm B reproduces the shipped Tossing Room configuration digit for digit** —
  EMPTY `100/100/100/100/100/100/100/100/100`, TRASH
  `28/28/54/78/76/86/86/86/98`, RECYCLING `18/19/39/44/55/81/81/83/92`, identical
  to the table recorded before this experiment. This validates the whole pipeline
  end to end and independently confirms the runs used this worktree's code rather
  than the editable install's (`import hitl_pmp` resolves to the main checkout by
  default; every run was launched with `PYTHONPATH` pinned to the worktree).
- All 40 runs (4 arms x 10 seeds) exited 0. Every arm reached exactly 2500 online
  transitions, sd 0 — no arm was short-changed on experience.
- Every seed drew the same test set in every arm (pinned by
  `test_every_arm_draws_the_same_test_set_so_denominators_match_across_arms`), so
  the cross-arm pairing is valid.
- The exact tests are computed here, not quoted from scipy, and pinned against
  hand-computable ground truth (an all-positive sample of size n has two-sided
  p = 2/2^n exactly) plus a worked textbook case.

## Limitations

- **The stated question is unanswered and this design cannot answer it.** See
  Result 2 for the do-not-re-tread record and the corrected design.
- **30 test tasks split by sampled `goal_weights` is too few per family.** The
  binomial noise floor on the gap is 20.7pp, which two of four arms sit at. More
  seeds do not fix this; more test tasks per family do — PR #41 does exactly that.
  These runs predate it, so they are the last Tossing Room results measured on the
  sampled test set; anything after `29e97fa` uses the fixed 14/14/2 split and is not
  directly comparable.
- **Arms A and B are saturated**, so their gaps are compressed against the ceiling
  and their paired tests have an effective n of 2-5 rather than 10.
- **The concurrency result is machine-scoped.** It is a measurement on 24 cores with
  this FD build and this domain's planning problems, not a general claim that
  wall-clock planning timeouts never bite.
- Per-seed results are machine-local and do not reproduce bit-for-bit across
  machines (see `2026-08-03-cross-machine-reproducibility.md`); the committed
  aggregate `2026-08-03-tossingroom-reset-freq.json` is the record that travels,
  and comparisons should be made at arm level.

## Reproducing

Runs (results written outside the repo — `.gitignore` has a bare `results/` that
matches at any depth):

```bash
export FD_EXEC_PATH=/path/to/downward
python -m scripts.run_sweep \
  --env tossingroom --methods ees --num-seeds 10 \
  --results-root "$SCRATCH/rf/sweep/armA" \
  --shared-args "--num-test-tasks 30 --sampler-max-train-iters 10000" \
  --method-args "ees=--num-cycles 50 --max-steps-per-interaction 50" \
  --max-workers 9
```

and likewise `armB` (25 x 100), `armC` (10 x 250), `armD` (5 x 500).

Aggregate, then regenerate every figure, table and p-value above from the
aggregate alone:

```bash
python -m analysis.practice_makes_perfect.tossingroom_reset_frequency \
  --arm armA="$SCRATCH/rf/sweep/armA" --arm armB="$SCRATCH/rf/sweep/armB" \
  --arm armC="$SCRATCH/rf/sweep/armC" --arm armD="$SCRATCH/rf/sweep/armD" \
  --aggregate-output docs/experiment-logs/2026-08-03-tossingroom-reset-freq.json

python -m analysis.practice_makes_perfect.tossingroom_reset_frequency \
  --arms-json docs/experiment-logs/2026-08-03-tossingroom-reset-freq.json \
  --output docs/experiment-logs/2026-08-03-tossingroom-reset-freq-gap.png \
  --curves-output docs/experiment-logs/2026-08-03-tossingroom-reset-freq-curves.png
```

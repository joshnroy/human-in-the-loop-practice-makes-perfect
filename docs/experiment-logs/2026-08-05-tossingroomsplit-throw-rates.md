# Two throws, two samplers, once the sampler has to learn a function: the endpoint gap becomes real and recycling gets 70/567 of the learned-sampler draws

**TL;DR.** Re-run of the split-throw experiment against the domain in which a throw's
required force is an unobserved function of the bin's `throw_distance` and the item's
`weight` rather than a `target_force` feature sitting in each sampler's own input row.
Vanilla EES, 10 fixed seeds, 2500 online transitions each, 14 TRASH / 14 RECYCLING / 2
EMPTY per seed. Four things, one of which changes direction.

1. **The structural claim survives, and is stronger than before.** Across 250 practice
   periods `ThrowRecycling` was attempted **0 or 1 times and never twice** — 142 periods at
   1, 108 at 0. `ThrowTrash`'s ceiling is again exactly **12**, now reached in 45 of 250
   periods. The measured attempt ratio is **893:142 = 6.29:1**, up from 3.79:1 and closer
   to the 12:1 the layout arithmetic predicted, though still well short of it.
2. **The endpoint result reverses from a null to a significant one, and this is the
   headline.** TRASH finishes **131/140** and RECYCLING **76/140** — a **+39.29pp** gap
   against a **16.74pp** MDE, with a paired Wilcoxon over seeds at **n = 10, p = 0.0039**.
   The previous run could not run that test at all: 7 of 10 seeds tied, dropping the
   effective n to 3 and the attainable p floor to 0.2500. **No seeds tie now**, so the test
   the log said "cannot fire" fires, and it rejects.
3. **Learning is less of a switch than it was.** **190/260** TRASH and **199/260**
   RECYCLING seed-checkpoints sit at an extreme (≥12/14 or ≤4/14). For TRASH the
   in-between count more than doubles, 33/260 → **70/260**. Some of the curve is now a
   genuine climb rather than an averaging artifact — but three quarters of
   seed-checkpoints still sit at one end, so the pooled RECYCLING line must still **not**
   be read as a family improving steadily.
4. **The recycling sampler gets 70 of the two throws' 567 learned-sampler draws** — 4 to
   11 per seed (median 6.5) in an entire 2500-transition run, against trash's 34 to 73
   (median 45.5). Its longest run of consecutive all-missing practice periods reaches **6**
   (seeds 0 and 3) against trash's worst of **4**.

![per-skill throw rates](./2026-08-05-tossingroomsplit-throw-rates.png)

![per-seed spread](./2026-08-05-tossingroomsplit-throw-rates-per-seed.png)

## Question / goal

`ThrowTrash` and `ThrowRecycling` are two lifted skills with two independent samplers of
identical architecture. The layout gives them wildly different practice budgets — trash is
a retryable round trip, recycling is one-way across a ledge that closes behind the robot.
**Do they learn at correspondingly different rates, and what does the difference actually
consist of?**

## Background

### What changed under this experiment, and why the old numbers are withdrawn

This page previously carried a run collected against a domain in which each throw's
classifier input row **contained the answer**. `EesMethod` builds each row as
`[1.0] + concat(state[obj] for obj in ground_skill.objects) + params`, and index 4 was
`item.target_force` while the dynamics landed a throw iff
`|force − item.target_force| < 0.1`. Both samplers were being asked to learn
`|x₁₀ − x₄| < 0.1`, a comparison between two of their own inputs. Measured over 80
applicable groundings of one throw, only **2 of the 10** state-plus-force columns carried
signal.

The replacement gives each bin a per-task **`throw_distance`** and each item a per-task
**`weight`**, and makes the required force
`reference_force + 0.2·(distance − 2) + 0.4·(weight − 1)` — five constants that live on the
environment and never enter a `State`. Signal columns go **2/10 → 3/11** per throw. Two
things were held fixed on purpose: a uniformly random force still lands with probability
**0.2** on every task, and the best fixed force a state-blind sampler could choose still
lands well under half of throws. What moved is the low-sample regime: an offline probe of
the same `MlpBinaryClassifier` put argmax-of-100 success at 16 labelled throws at **0.37**
against the identity's **0.70**, converging to ~0.99 by 160 either way.

`build_task` now draws four uniforms per task instead of two, so **the 30 evaluation tasks
of a given seed are different tasks**. The previous numbers are therefore **not
re-scorable — they are incomparable**, exactly as the capacity-1 redesign made the run
before them incomparable. Kept here as history so the record of what was claimed is not
deleted:

| claim, previous run (identity representation) | what it said | what this re-run measures |
|---|---|---|
| attempt ratio | 618:163 = **3.79:1** | **893:142 = 6.29:1** — survives in substance, moves toward the prediction |
| recycling never attempted twice in a period | 163 at 1, 87 at 0 | **142 at 1, 108 at 0** — survives exactly |
| trash's per-period ceiling | **12**, in 30 of 250 periods | **12**, in 45 of 250 periods |
| `MoveRoom` share of practice | 19,224/24,750 | **18,913/24,750** |
| endpoint | TRASH 140/140, RECYCLING 124/140 | **TRASH 131/140, RECYCLING 76/140** |
| endpoint gap | +11.43pp, **null result** (MDE 16.74pp), Wilcoxon could not fire | **+39.29pp, p = 0.0039 — established** |
| AUC difference | +41.13pp, p = 0.0020 | **+42.31pp, p = 0.0020** — survives almost unchanged |
| per greedy attempt | trash 224/347 vs recycling 37/83, +19.98pp | **trash 288/497 vs recycling 22/70, +26.52pp** — same direction, larger |
| learning is a **switch** | 227/260 and 209/260 at an extreme | **190/260 and 199/260** — weaker for trash |
| competence | 0.927 trash, 0.800 recycling | **0.907 trash, 0.742 recycling** — same order |

### Why the split exists at all

`EesMethod.sampler` keys its `LearnedSkillSampler` dict by `skill_name`, so two names give
two classifiers with independent weights on the same architecture. Each throw learns only
from its own attempts, with no transfer from the other. The layout then decides how many
attempts each gets: trash is a retryable round trip from the pile; recycling sits behind a
one-way ledge, and since a throw always releases the item, reaching the recycling bin ends
that period's chance of another go.

The split also sharpens the representation defect. Within one throw skill the bound bin,
item and room never vary, so every column that was an affine copy of the `kind` bit in the
unsplit domain is a flat **constant** here: **9 constant / 0 redundant / 3 free**. Under
`target_force` those three free columns were the answer, the dial, and nothing else.

## Hypothesis

Registered before the original sweep and reproduced unedited, because a re-run does not get
to rewrite what was predicted:

> Trash: pickup → walk 3 rooms → throw → walk back = 8 steps, so a 100-step period should
> buy roughly **12 attempts**. Recycling: pickup → step across the ledge → throw, with no
> way back and no second item, so **exactly 1 attempt per practice period, ever**. Expected
> ratio **≈ 1:12**. Recycling should learn far more slowly, **at roughly that ratio** — and,
> with the shared-sampler transfer channel removed, **possibly not at all within a practice
> budget.**

Carried into this re-run: that making the sampler learn a *function* would hurt the skill
with fewer samples much more than the one with many, because the two budgets (~89 and ~14
attempts per seed) sit on opposite sides of the offline learning curve. So the TRASH /
RECYCLING gap was expected to **widen**, and the endpoint null was expected to become
resolvable.

## Guidance given

- Re-run every arm the representation change invalidates; **rewrite everything that reports
  a number** rather than appending, and state any deviation from the protocol.
- Fixed seeds via `scripts/run_sweep.py`, never randomly drawn. **Time one seed first and
  report it before launching the sweep.** Results must land outside the agent worktree.
- **Counts as `x/y` everywhere** — prose, tables, axis labels, annotations.
- **Figures, not just tables**, with **per-seed spread**.
- Check specifically: **is learning still a switch**, and **does the TRASH/RECYCLING gap
  survive**? Report the endpoint **and** the AUC.
- Report the binomial noise floor `sqrt(0.25/n_a + 0.25/n_b)` and the MDE. **Do not claim a
  difference below it.**
- Record that the earlier numbers were withdrawn and why — do not delete that history.

## Methods

| | |
|---|---|
| domain | `tossingroomsplit`, throw-representation branch |
| method | `ees`, vanilla. **One experiment, no arms** — the comparison is between two skills inside the same runs |
| seeds | 10, fixed at 0–9 (`scripts/run_sweep.py`, never randomly drawn) |
| protocol | `--num-cycles 25 --max-steps-per-interaction 100` → exactly **2500** online transitions in every seed |
| evaluation | `--num-test-tasks 30`, fixed composition **14 TRASH / 14 RECYCLING / 2 EMPTY** per seed |
| horizon | `longest_shortest_solve() + 2` = **12**, confirmed in every trace |

**Deviations from the previous protocol: none.** Same seeds, same cycles, same steps, same
test-task count and composition, same two-step sweep-then-trace collection. One *analysis*
statistic had to be rebased rather than re-run; see below.

```bash
# 1. The sweep. Writes results/<root>/ees/<seed>/{stats.json,timing.json,config_snapshot.json}.
python -m scripts.run_sweep \
  --env tossingroomsplit --methods ees --num-seeds 10 \
  --results-root results/tossingroomsplit-fn-throws \
  --shared-args "--num-test-tasks 30" \
  --method-args "ees=--num-cycles 25 --max-steps-per-interaction 100" \
  --max-workers 10

# 2. The per-skill traces, one process per seed (they are serial within a process).
python -m scripts.tossingroomsplit_skill_traces \
  --label ees --seeds <k> --num-cycles 25 \
  --max-steps-per-interaction 100 --num-test-tasks 30 \
  --output results/tossingroomsplit-fn-traces/shard-<k>.json

# 3. The analysis. Post-run only; it never drives a simulation.
python -m analysis.practice_makes_perfect.tossingroomsplit_throw_rates \
  --traces results/tossingroomsplit-fn-traces/shard-{0..9}.json \
  --results-root results/tossingroomsplit-fn-throws \
  --output docs/experiment-logs/2026-08-05-tossingroomsplit-throw-rates.png \
  --per-seed-output docs/experiment-logs/2026-08-05-tossingroomsplit-throw-rates-per-seed.png
```

### Compute, measured rather than guessed

| | wall clock | per-run mean |
|---|---|---|
| single-seed calibration (alone, 1 worker) | **3 min 4 s** (184.0 s) | — |
| sweep, 10 seeds at 10 workers | **3 min 35 s** (214.6 s) | 189.5 s |
| traces, 10 processes | **4 min 31 s** (271 s) | — |

Peak RSS for a single run, measured alone: **923 MB**. 10/10 runs succeeded; no launch
failures and no retries were printed to stderr. The per-run mean under 10-way concurrency
(189.5 s) is about 3% above the 184.0 s the same work took alone.

### Why step 2 exists, and how it is checked

`stats.json` is the serialized `core.Metrics`: tasks solved per sweep with a per-task goal
breakdown. That is the right record of outcomes. But the question here is about the two
*skills* — how often each was practised, how often each succeeded, whether it actually
landed, and what force it chose — and none of that leaves `EesMethod`'s internals. So the
collector subclasses the real method to read it out.

That the two are the **same ten runs** is a checked fact, not an argument from determinism:

* `tests/scripts/test_tossingroomsplit_skill_traces.py::test_tracing_does_not_perturb_the_run`
  requires a traced and an untraced run at the same seed to produce **equal** per-sweep
  `(transitions, solved, total)` triples.
* The analysis **refuses to print** unless every traced seed reproduces its real
  `stats.json` exactly. It reported: `consistency gate: all 10 traced seeds reproduce their
  swept stats.json exactly`.
* The trace collection was run **twice, independently**, and produced **10/10 byte-identical
  shards**.

**One further reconciliation, exact.** 250 practice periods × 100 steps = 25,000
transitions; EES leaves the last skill of each period unobserved by construction, so 250
outcomes are unobservable. Observed practice attempts across all skills total **24,750** —
25,000 − 250, to the unit.

### One statistic was rebased, and it is not a rename

The previous run's sharpest single number was a count of greedy draws **below 0.4**. That
was well founded when a task's required force was drawn `U(0.5, 1.0)` at tolerance 0.1: a
force below `0.5 − 0.1` missed *whatever* task it was aiming at, so "chose a force no task
could have wanted" was a property of the choice alone.

**That statistic cannot be taken on this domain any more.** The required force now spans
`[0.1, 0.9]`, so every force in the `U(0, 1)` draw range is right for *some* task and the
count is identically 0/N. It is replaced by the **per-grounding** miss: greedy draws
further than **3× the tolerance (0.30)** from the force that grounding actually required.
The collector already recorded the required force alongside the chosen one, so this needed
no new instrumentation and no re-collection. The two numbers are **not comparable**, and the
old one is not quoted anywhere on this page.

### What this design can detect

Every gap below is quoted against the floor `sqrt(0.25/n_a + 0.25/n_b)` and the 80%-power
two-sided MDE, which is 2.80 of them.

| comparison | denominators | noise floor | MDE |
|---|---|---|---|
| goal family, final sweep | 140 vs 140 | 5.98pp | **16.74pp** |
| all practice attempts | 893 vs 142 | 4.52pp | **12.66pp** |
| learned-sampler attempts | 497 vs 70 | 6.38pp | **17.88pp** |
| epsilon-random draws | 396 vs 72 | 6.41pp | **17.95pp** |

The floor is driven by the **smaller** arm throughout, so piling on more trash attempts
cannot buy resolution the recycling side does not have.

## Results

### 1. The attempt budget: the structural claim survives, and moves toward the prediction

| skill | attempts per period | periods |
|---|---|---|
| `ThrowRecycling` | **0 or 1, and never more** | 142 at 1, 108 at 0 |
| `ThrowTrash` | 0 through **12** | 130 at 0, 20 at 1, 13 at 2, 11 at 3, 4 at 4, 4 at 5, 1 at 6, 2 at 8, 1 at 9, 2 at 10, 17 at 11, 45 at 12 |

Across 250 practice periods there is **not one** with two recycling attempts, and trash's
ceiling is again exactly **12** — now reached in 45 of 250 periods against 30 before. Both
were the parts of the prediction the layout guarantees, and both hold.

**The ratio is 893:142 = 6.29:1, against 3.79:1 before and a predicted 12:1.** It moved
toward the prediction because EES now chooses to practise trash more often — 120 of 250
periods have at least one trash attempt, against 106 before — which is what a skill whose
competence estimate is genuinely falling looks like.

| skill | observed practice attempts |
|---|---|
| `MoveRoom` | **18,913/24,750** |
| `PressRecycling` | **3,371/24,750** |
| `PickupTrash` | 932/24,750 |
| `ThrowTrash` | **893/24,750** |
| `PressTrash` | 357/24,750 |
| `PickupRecycling` | 142/24,750 |
| `ThrowRecycling` | **142/24,750** |

`PressRecycling` at 3,371/24,750 is a stranded robot pressing an already-empty recycling
button on the far side of the ledge — 24 times the recycling throw count, and essentially
unchanged from the previous run's 3,684. That is practice budget spent on a no-op, and the
representation change does not touch it.

Per seed the ratio ranges from **3.1** (seed 9, 56 vs 18) to **11.9** (seed 6, 131 vs 11),
so the pooled 6.29 is a central value with a factor-of-two spread either side. A
single-seed reading of this domain would still mislead badly.

### 2. The endpoint gap is now established, and this is the direction change

| | final sweep |
|---|---|
| TRASH | **131/140** |
| RECYCLING | **76/140** |
| gap | **+39.29pp** |
| binomial noise floor | 5.98pp |
| minimum detectable effect (80%) | **16.74pp** |
| paired Wilcoxon over seeds | **n = 10, W = 54.0, p = 0.0039** (p floor at this n: 0.0020) |

**+39.29pp against a 16.74pp MDE, and a paired test that both fires and rejects.** The
previous run measured +11.43pp — below its MDE — and its Wilcoxon had an effective n of 3
after ties, with an attainable p floor of 0.2500 equal to the p it returned. That log's
own recommendation was "report this domain by curve statistic, never by endpoint, because
the endpoint test cannot fire."

**It fires now because the ties are gone.** Seven of ten seeds previously finished
14/14 against 14/14. Not one seed ties in this run:

| seed | TRASH peak | TRASH final | RECYCLING peak | RECYCLING final |
|---|---|---|---|---|
| 0 | 14/14 | 13/14 | 14/14 | 8/14 |
| 1 | 14/14 | 14/14 | **5/14** | **1/14** |
| 2 | 14/14 | 13/14 | 14/14 | 9/14 |
| 3 | 14/14 | 13/14 | **6/14** | 5/14 |
| 4 | 14/14 | 14/14 | 14/14 | 12/14 |
| 5 | 14/14 | 14/14 | 9/14 | 8/14 |
| 6 | 14/14 | 14/14 | 10/14 | 8/14 |
| 7 | 14/14 | 12/14 | **5/14** | 4/14 |
| 8 | 14/14 | 14/14 | 10/14 | 9/14 |
| 9 | 13/14 | 10/14 | 13/14 | 12/14 |

Every seed reaches TRASH 13/14 or 14/14 at its peak and finishes at 10/14 or better.
RECYCLING finishes between **1/14 and 12/14**, and only four seeds ever *reach* 13/14 or
14/14 at any checkpoint — none of which holds it. **The previous run's "eight seeds at
14/14 and one catastrophic seed" has become "no seed solves this family."**

**Across the whole curve the picture is unchanged and still significant.** Mean area under
the per-family curve is **74.20** for TRASH against **31.90** for RECYCLING, a **+42.31pp**
difference, paired Wilcoxon n = 10, **p = 0.0020** — the previous run's +41.13pp at the same
p. So the AUC statistic was measuring something real about the layout that survived a
change of representation, while the endpoint statistic was previously floor-limited by
TRASH saturating.

`EMPTY` is **520/520** across all 260 seed-sweeps — 20/20 in every sweep of every seed,
pre-practice included. It contains no throw, so neither sampler can touch it; it is the
deterministic control and it did not move.

### 3. The learning-rate result: slower by roughly the attempt ratio, and now not finishing

Transitions at which each family first reaches a given share of its own 140 test tasks:

| level | TRASH | RECYCLING | ratio |
|---|---|---|---|
| 25% | 200 | 1300 | **6.5x** |
| 50% | 400 | 1800 | **4.5x** |
| 75% | 1000 | **never** | — |
| 90% | 1600 | **never** | — |

Against a measured attempt ratio of **6.29:1**. The predicted *relationship* holds at the
two thresholds RECYCLING reaches — 6.5x and 4.5x against 6.29 — and the previous run's
drift downward with threshold repeats. **RECYCLING never reaches 75% at all**, where before
it reached 75% at 2100 transitions. That is the clearest single statement of what the
representation change cost the low-budget skill.

### 4. Per practice attempt, trash is the better sampler, by more than before

The add-effect audit first, because it is what makes the rest readable:

| skill | landed / attempts | EES scored / attempts | thrown into a prefilled bin | scored successes that missed |
|---|---|---|---|---|
| `ThrowTrash` | **356/893** | **356/893** | **0/893** | **0/356** |
| `ThrowRecycling` | **32/142** | **32/142** | **0/142** | **0/32** |

Landed and scored are the same column, and the prefilled-bin channel capacity-1 closed is
measured closed rather than assumed. In the top-right panel of the first figure the "EES
scored a success" line sits exactly underneath "actually landed" for both skills.

Over **all** practice attempts, trash lands 356/893 (39.9%) against recycling's 32/142
(22.5%) — **+17.33pp against a 12.66pp MDE**. Established.

| per learned-sampler attempt | `ThrowTrash` | `ThrowRecycling` | gap | MDE |
|---|---|---|---|---|
| landed | **288/497** (58.0%) | **22/70** (31.4%) | **+26.52pp** | 17.88pp |

**+26.52pp against a 17.88pp MDE: detectable, same direction as the previous run's
+19.98pp and larger.** The epsilon-random draws remain the control and still say the two
throws are comparably hard: `ThrowTrash` lands **68/396** (17.2%) on randomly chosen forces
and `ThrowRecycling` **10/72** (13.9%), a 3.3pp gap against a **17.95pp** MDE — well inside
it, so **no difference in intrinsic difficulty is claimed**. Both sit near the 0.2
first-principles base rate the representation change deliberately preserved. The difference
between the two skills is in what each sampler learned.

> **A caveat that limits this comparison, and it is not one this experiment can close.** A
> parallel investigation of `LearnedSkillSampler.sample` found that `is_fitted` returns
> `True` for the single-class shortcut, so `sample` takes the argmax branch on an all-equal
> score vector and returns `candidates[0]` while reporting `was_random = False`. The
> "learned-sampler" pool above is therefore **not purely** draws that reflect a trained
> classifier's belief — some of it is the first candidate the caller happened to draw,
> labelled greedy. Both skills are affected, but the skill with fewer observations spends
> longer in that state, so the recycling column is the one likely to be flattered or
> maligned by it. **The +26.52pp gap should be read as provisional pending that fix**, and
> the all-attempt comparison (+17.33pp, which does not depend on the greedy/random split) is
> the more robust of the two.

### 5. Learning is a switch, but less of one than it was

Scoring every (seed, checkpoint) against 12/14 and 4/14:

| skill | at an extreme (≥12/14 or ≤4/14) | anywhere in between | seed-checkpoints |
|---|---|---|---|
| `ThrowTrash` | **190/260** | **70/260** | 260 |
| `ThrowRecycling` | **199/260** | 61/260 | 260 |

Previously 227/260 and 209/260. **TRASH's in-between count more than doubles, 33/260 →
70/260**, which is the offline learning curve showing through: a skill that used to jump
from nothing to solved in one refit now spends real time partway. RECYCLING moves much
less, 51/260 → 61/260.

The bottom-left panel of the per-seed figure is this table drawn, and it also shows what the
counts do not: RECYCLING's mass is now spread across 0–9 out of 14 rather than piled at both
ends, while TRASH still has 83 seed-checkpoints at exactly 14/14. **Describing RECYCLING as
"still climbing" would still be a statement about the mean rather than about any seed** —
the per-seed panel shows several seeds flat near 2/14 for 1200 transitions and then a step —
but the shape is no longer as cleanly binary as the previous run reported.

### 6. The mechanism: seventy learned-sampler draws across ten whole runs

Of the two throws' **567** learned-sampler draws across all ten seeds, recycling gets **70**:

| seed | greedy `ThrowTrash` draws | greedy `ThrowRecycling` draws |
|---|---|---|
| 0 | 48 | 8 |
| 1 | 70 | 5 |
| 2 | 36 | 10 |
| 3 | 63 | 6 |
| 4 | 34 | 11 |
| 5 | 44 | 5 |
| 6 | 73 | 4 |
| 7 | 45 | 7 |
| 8 | 46 | 5 |
| 9 | 38 | 9 |
| **total** | **497/567** | **70/567** |

**A whole 2500-transition run gives the recycling sampler between 4 and 11 chances to
correct itself, median 6.5.** With `--exploration-epsilon 0.5` half of every attempt is a
coin flip that teaches the classifier nothing about its own belief, so 142 recycling
attempts become 70 learned draws spread over 25 practice periods.

**And 70 is on the wrong side of the offline curve.** The probe of this exact classifier put
argmax success at 16 labelled throws at 0.37 and at 80 at 0.94. Trash's ~50 draws per seed
sit near the top of that curve; recycling's ~7 sit near the bottom. That is the mechanism in
one sentence: *the layout rations recycling to a sample count at which this classifier
cannot yet represent the relation.*

| skill | greedy draws missing their own grounding by more than 0.30 | longest run of consecutive all-missing practice periods, per seed |
|---|---|---|
| `ThrowTrash` | **77/497** | 2, 1, 2, 4, 1, 1, 3, 2, 1, 1 |
| `ThrowRecycling` | **25/70** | **6, 5, 4, 6, 5, 1, 3, 3, 3, 4** |

Trash's worst run of all-missing periods across ten seeds is 4; recycling reaches 6, in two
seeds. The same caveat as in (4) applies to the 25/70 — the "greedy" label includes draws a
single-class classifier made — so it is reported as the shape it is rather than leaned on.

### 7. Competence, re-measured

| | final competence | measured learned-sampler landing rate |
|---|---|---|
| `ThrowTrash` | 0.907 | 288/497 (58.0%) |
| `ThrowRecycling` | 0.742 | 22/70 (31.4%) |

Competence ranks the two skills in the **correct** order, as it did in the previous run once
the add-effect defect was closed. It remains a substantial **overestimate** of both — 0.907
against 58.0% and 0.742 against 31.4% — because `OptimisticSkillCompetenceModel` is a
windowed estimate under a Beta(10, 1) prior whose mean is 0.909. And the overestimate is
still worse for the skill with fewer observations: recycling's competence never falls below
0.721 across the whole run, while trash's drops to 0.577 at 600 transitions and then climbs,
which is what an estimate actually tracking data looks like. Competence is what
`skill_costs()` turns into `-log(competence)` plan edge costs and what `score_ground_skill`
extrapolates when choosing what to practise, so a skill whose estimate barely moves off the
prior is a skill the planner has little reason to prioritise — which is part of why
recycling gets 70 draws.

## Verdict on the prediction

| claim | verdict |
|---|---|
| recycling gets **exactly 1 attempt per practice period, ever** | **held exactly** — 0 or 1 in all 250 periods, never 2 (142 at 1, 108 at 0) |
| trash gets **~12 attempts** in a 100-step period | **held, conditionally** — 12 is the observed ceiling, reached in 45 of 250 periods, but 130 of 250 have zero |
| attempt ratio **≈ 1:12** | **still refuted, but less badly** — measured **6.29:1** (893:142), against 3.79:1 before |
| recycling learns **far slower, at roughly the attempt ratio** | **held** — 6.5x and 4.5x against a 6.29:1 ratio |
| **possibly not at all** within a practice budget | **effectively held** — RECYCLING never reaches 75% of its family, and finishes 76/140 with no seed above 12/14 |
| learning is a **switch**, not a curve | **weakened** — 190/260 and 199/260 at an extreme, against 227/260 and 209/260 |
| *(previous run)* the endpoint gap is a **null result** the test cannot even fire on | **reversed** — +39.29pp, n = 10, p = 0.0039 |

## Video

The previous version of this page carried an evaluation-progression clip whose analysis was
written from the frames, and what it read off those frames was the thrown force against the
task's `target_force` ("this task's recycling `target_force` is 0.72, so the winning window
is 0.62–0.82"). There is no `target_force` any more and that task no longer exists, so every
number in that prose is unrecoverable rather than merely stale. The file has been removed
with the section rather than left to be read as current. Re-recording against this run is a
worthwhile follow-up and is not fabricated here.

## Recommendation

1. **The endpoint is now the right statistic to pre-specify, alongside AUC.** The previous
   recommendation — "report by curve statistic, never by endpoint" — was correct *for a
   design in which TRASH saturated at 140/140 and seven seeds tied*. That is no longer the
   case: the endpoint resolves at p = 0.0039 and the AUC at p = 0.0020, and they agree.
2. **Fix `LearnedSkillSampler.sample` before quoting any greedy-vs-random split from this
   domain again.** The three defects found in it mean the "learned-sampler attempt" column
   mixes trained draws with `candidates[0]`. Results (4) and (6) are flagged provisional for
   that reason; results (1), (2), (3) and (5) do not depend on the split.
3. **The recycling sampler's problem is sample count, and the representation change made
   that bite.** 70/567 of the learned draws, 4–11 per run, against an offline curve that
   needs tens. The obvious intervention is a **cheaper reset** — the per-cycle `PERIOD`
   reset is the only thing that returns the robot to the pile, so the recycling budget is
   one throw per cycle by construction. `--practice-reset-interval` already exists and a
   reset-interval arm on this domain would test it directly.
4. **`PressRecycling` at 3,371/24,750 is still worth its own look.** It is 24x the recycling
   throw count and is almost entirely a stranded robot pressing an already-empty button.
5. **Do not compare any number here to any number measured before the representation
   change**, and in particular do not quote the withdrawn "greedy draws below 0.4" figure:
   that statistic is not measurable on this domain and its replacement is not on the same
   scale.
6. **Nothing here argues for or against the split as a design.** No shared-sampler arm was
   run. Result (4) compares two skills within one run, not two architectures.

## Raw data

* [`2026-08-05-tossingroomsplit-throw-rates.json`](./2026-08-05-tossingroomsplit-throw-rates.json)
  — all ten seeds' per-period skill tallies (attempts, successes, landings, prefilled-bin
  attempts, the epsilon-random split of each, and every learned-sampler force with the force
  its grounding required), per-cycle competence, and per-sweep evaluation records with their
  goal-family breakdowns. **Every count on this page re-derives from it**, and none is
  reconstructed by multiplying a percentage by *n*. Checked rather than asserted: re-running
  the analysis with `--traces` pointing at this one committed file instead of the ten shards
  produces a report identical apart from the output filename it prints.
* [`2026-08-05-tossingroomsplit-throw-rates.png`](./2026-08-05-tossingroomsplit-throw-rates.png)
  — the pooled figure.
* [`2026-08-05-tossingroomsplit-throw-rates-per-seed.png`](./2026-08-05-tossingroomsplit-throw-rates-per-seed.png)
  — the per-seed spread, the seed-checkpoint distribution, and what each sampler answered.

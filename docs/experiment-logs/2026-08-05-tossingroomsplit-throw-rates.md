# Two throws, two samplers on the capacity-1 domain: recycling gets 83/430 of the learned-sampler draws, and its learning is a switch one seed never flips

**TL;DR.** Re-run of PR #71's split-throw experiment against the **capacity-1**
`tossingroomsplit` domain, replacing a pre-#74 run whose every number is withdrawn.
Vanilla EES, 10 fixed seeds, 2500 online transitions each, 14 TRASH / 14 RECYCLING / 2
EMPTY per seed. Four things.

1. **The structural claim survives, and it is the only one that does.** Across 250 practice
   periods `ThrowRecycling` was attempted **0 or 1 times and never twice** — 163 periods at
   1, 87 at 0. `ThrowTrash`'s ceiling is again exactly **12**. The measured attempt ratio is
   **618:163 = 3.79:1**, close to the pre-#74 run's 4.0:1 and nowhere near the predicted
   12:1.
2. **The pre-#74 headline is reversed.** Per learned-sampler attempt, `ThrowTrash` lands
   **224/347** against `ThrowRecycling`'s **37/83** — **+19.98pp** against a **17.12pp** MDE.
   Trash is the better sampler. The old log's "recycling is the better sampler" was the
   scoring defect talking, and the audit now reads **0/618** and **0/163** prefilled with
   **0/286** and **0/50** scored-but-missed: the channel is closed, measured rather than
   assumed.
3. **Learning is a switch, not a curve — the same shape the Tossing Room baseline found.**
   **227/260** TRASH seed-checkpoints and **209/260** RECYCLING seed-checkpoints sit at an
   extreme (≥12/14 or ≤4/14); only 33/260 and 51/260 are anywhere in between. Seed 1's
   RECYCLING reaches **9/14** at 100 transitions and ends at **1/14**, frozen there for the
   last 19 of 26 checkpoints. The pooled RECYCLING line must **not** be read as a family
   climbing steadily.
4. **And there is a mechanism, in the data rather than in a video.** The recycling sampler
   gets **83/430** of the two throws' learned-sampler draws — **5 to 11 per seed (median
   7.5) in an entire 2500-transition run**, against trash's 25 to 52 (median 34). **24/83**
   of them are forces below **0.4**, which no task in this domain can want. Its longest run
   of consecutive all-missing practice periods reaches **9** (seed 2) against trash's worst
   of **2**. A recycling sampler that convinces itself wrong gets roughly one datapoint per
   period to unconvince it; trash gets up to twelve.

![per-skill throw rates](./2026-08-05-tossingroomsplit-throw-rates.png)

![per-seed spread](./2026-08-05-tossingroomsplit-throw-rates-per-seed.png)

## Question / goal

`ThrowTrash` and `ThrowRecycling` are two lifted skills with two independent samplers of
identical architecture. The layout gives them wildly different practice budgets — trash is a
retryable round trip, recycling is one-way across a ledge that closes behind the robot. **Do
they learn at correspondingly different rates, and what does the difference actually consist
of?**

## Background

### The earlier numbers were withdrawn, and why

This page previously carried a run collected against the **pre-capacity-1**
`tossingroomsplit`. [#74](https://github.com/joshnroy/human-in-the-loop-practice-makes-perfect/pull/74)
then changed the **dynamics**, not merely the scoring:

* a bin holds **at most one item**, and a throw at a full one is **refused** — the item stays
  in hand and nothing happens;
* each bin has **its own emptying button** beside it, and a press empties only that bin;
* both throws carry their bin's empty atom as a **precondition and a delete effect**;
* `EMPTY` prefills **exactly one item per bin** and became an **ordering** task;
* the evaluation horizon went **7 → 12**, derived from `EMPTY`'s new 10-action shortest solve.

Because those are dynamics changes the old results were **not re-scorable — they were
incomparable**, and that included the figures the old log argued were structural. The
withdrawn numbers, kept here as history so the record of what was claimed is not deleted:

| withdrawn claim (pre-#74) | what it said | what this re-run measures |
|---|---|---|
| attempt ratio | **799:200 = 4.0:1** | **618:163 = 3.79:1** — survives in substance |
| recycling never attempted twice in a period | 200 at 1, 50 at 0 | **163 at 1, 87 at 0** — survives exactly |
| `MoveRoom` share of practice | **22,648/24,750** | **19,224/24,750** — moved, and `PressRecycling` went 55 → **3,684** |
| endpoint | TRASH 127/140, RECYCLING 122/140 | **TRASH 140/140, RECYCLING 124/140** |
| per greedy attempt | recycling **better**, 51/94 vs 152/433 | **reversed** — trash 224/347 vs recycling 37/83 |
| scored successes that missed | **313/532** for trash | **0/286** — the defect is closed |
| competence ranks the two skills **backwards** | 0.859 trash vs 0.844 recycling | **does not survive** — 0.927 vs 0.800, the correct order |
| AUC difference | +12.94pp, p = 0.1934 | **+41.13pp, p = 0.0020** |

Two instrumentation defects were fixed before this re-run and are not results: `_observe_throw`
used to record a **refused** throw as a landing whenever the force was good, and a
non-vacuity test constructed the now-closed defect.

### Why the split exists at all

`EesMethod.sampler` keys its `LearnedSkillSampler` dict by `skill_name`, so two names give two
classifiers with independent weights on the same architecture. Each throw learns only from its
own attempts, with no transfer from the other. The layout then decides how many attempts each
gets: trash is a retryable round trip from the pile; recycling sits behind a one-way ledge, and
since a throw always releases the item, reaching the recycling bin ends that period's chance of
another go.

## Hypothesis

Registered before the original sweep and reproduced unedited, because a re-run does not get to
rewrite what was predicted:

> Trash: pickup → walk 3 rooms → throw → walk back = 8 steps, so a 100-step period should buy
> roughly **12 attempts**. Recycling: pickup → step across the ledge → throw, with no way back
> and no second item, so **exactly 1 attempt per practice period, ever**. Expected ratio
> **≈ 1:12**. Recycling should learn far more slowly, **at roughly that ratio** — and, with the
> shared-sampler transfer channel removed, **possibly not at all within a practice budget.**

Carried into the re-run, from what the fresh Tossing Room baseline found on the sibling domain:
that per-family learning would prove to be a **switch** rather than a curve, and that
recycling's disadvantage would show up as a sampler stuck on a confident wrong force.

## Guidance given

- Re-run the whole experiment against the capacity-1 domain; **rewrite everything that reports
  a number**, and state any deviation from the original protocol.
- Fixed seeds via `scripts/run_sweep.py`, never randomly drawn. **Time one seed first.**
- **Counts as `x/y` everywhere** — prose, tables, axis labels, annotations. Never a bare
  percentage.
- **Figures, not just tables**, with **per-seed spread**: a bar chart of two means hides one
  seed driving the effect.
- Check for the two things the fresh Tossing Room baseline found: **learning as a switch**
  (per-seed extremes, not a pooled mean described as "still climbing"), and **RECYCLING's
  mechanism** — a sampler pinned on a wrong force with one datapoint per period to escape it.
- Report the binomial noise floor `sqrt(0.25/n_a + 0.25/n_b)` and the MDE the design can
  detect. **Do not claim a difference below it.**
- Background must **record that the earlier numbers were withdrawn and why** — do not delete
  that history.
- Results must land somewhere durable, outside the agent worktree.

## Methods

| | |
|---|---|
| domain | `tossingroomsplit` at capacity-1 (PR #70, tracking #74) |
| method | `ees`, vanilla. **One experiment, no arms** — the comparison is between two skills inside the same runs |
| seeds | 10, fixed at 0–9 (`scripts/run_sweep.py`, never randomly drawn) |
| protocol | `--num-cycles 25 --max-steps-per-interaction 100` → exactly **2500** online transitions in every seed |
| evaluation | `--num-test-tasks 30`, fixed composition **14 TRASH / 14 RECYCLING / 2 EMPTY** per seed |
| horizon | `longest_shortest_solve() + 2` = **12**, confirmed in every trace |

**Deviations from the original protocol: one, and it is additive.** The trace collector now
also records, per learned-sampler throw, the **force it chose and the target it was aiming at**.
Nothing else changed — same seeds, same cycles, same steps, same test-task count and
composition. The collector consumes no randomness and changes no control flow, and the
traced/swept consistency gate below is what checks that.

```bash
# 1. The sweep. Writes results/<root>/ees/<seed>/{stats.json,timing.json,config_snapshot.json}.
python -m scripts.run_sweep \
  --env tossingroomsplit --methods ees --num-seeds 10 \
  --results-root results/tossingroomsplit-cap1-throws \
  --shared-args "--num-test-tasks 30" \
  --method-args "ees=--num-cycles 25 --max-steps-per-interaction 100" \
  --max-workers 10

# 2. The per-skill traces, one process per seed (they are serial within a process).
python -m scripts.tossingroomsplit_skill_traces \
  --label ees --seeds <k> --num-cycles 25 \
  --max-steps-per-interaction 100 --num-test-tasks 30 \
  --output results/tossingroomsplit-cap1-traces/shard-<k>.json

# 3. The analysis. Post-run only; it never drives a simulation.
python -m analysis.practice_makes_perfect.tossingroomsplit_throw_rates \
  --traces results/tossingroomsplit-cap1-traces/shard-{0..9}.json \
  --results-root results/tossingroomsplit-cap1-throws \
  --output docs/experiment-logs/2026-08-05-tossingroomsplit-throw-rates.png \
  --per-seed-output docs/experiment-logs/2026-08-05-tossingroomsplit-throw-rates-per-seed.png
```

### Compute, measured rather than guessed

| | wall clock | per-run range |
|---|---|---|
| single-seed calibration (alone, 1 worker) | **2 min 42 s** (162.4 s) | — |
| sweep, 10 seeds at 10 workers | **4 min 0 s** (240.3 s) | 179.2 s – 240.1 s |
| traces, 10 processes | **3 min 43 s** (223 s) | — |

10/10 runs succeeded; no launch failures and no retries were printed to stderr. The per-run
range runs *above* the 162.4 s the same work took alone because a first batch of trace processes
was launched while the sweep was still live and then killed — up to 20 runs were briefly
resident. The sweep's own numbers are unaffected (a run is determined by its `--seed`); only its
wall clock is.

### Why step 2 exists, and how it is checked

`stats.json` is the serialized `core.Metrics`: tasks solved per sweep with a per-task goal
breakdown. That is the right record of outcomes. But the question here is about the two
*skills* — how often each was practised, how often each succeeded, whether it actually landed,
and now what force it chose — and none of that leaves `EesMethod`'s internals. So the collector
subclasses the real method to read it out, exactly as `scripts/tossingroom_throw_traces.py`
already does.

That the two are the **same ten runs** is a checked fact, not an argument from determinism:

* `tests/scripts/test_tossingroomsplit_skill_traces.py::test_tracing_does_not_perturb_the_run`
  requires a traced and an untraced run at the same seed to produce **equal** per-sweep
  `(transitions, solved, total)` triples.
* The analysis **refuses to print** unless every traced seed reproduces its real `stats.json`
  exactly, and treats a traced seed *missing* from the sweep as a disagreement rather than a
  skip. It reported: `consistency gate: all 10 traced seeds reproduce their swept stats.json
  exactly`.

**One further reconciliation, exact.** 250 practice periods × 100 steps = 25,000 transitions;
EES leaves the last skill of each period unobserved by construction, so 250 outcomes are
unobservable. Observed practice attempts across all skills total **24,750** — 25,000 − 250, to
the unit.

### What this design can detect

Every gap below is quoted against the floor `sqrt(0.25/n_a + 0.25/n_b)` and the 80%-power
two-sided MDE, which is 2.80 of them.

| comparison | denominators | noise floor | MDE |
|---|---|---|---|
| goal family, final sweep | 140 vs 140 | 5.98pp | **16.74pp** |
| all practice attempts | 618 vs 163 | 4.40pp | **12.33pp** |
| learned-sampler attempts | 347 vs 83 | 6.11pp | **17.12pp** |
| epsilon-random draws | 271 vs 80 | 6.36pp | **17.82pp** |

The family MDE is coarse — one task is 7.1pp on a 14-task family, so the design resolves about
two and a half tasks per seed. The floor is driven by the **smaller** arm throughout, so piling
on more trash attempts cannot buy resolution the recycling side does not have.

## Results

### 1. The attempt budget: the structural claim survives the redesign intact

| skill | attempts per period | periods |
|---|---|---|
| `ThrowRecycling` | **0 or 1, and never more** | 163 at 1, 87 at 0 |
| `ThrowTrash` | 0 through **12** | 144 at 0, 35 at 1, 12 at 2, 8 at 3, 1 at 4, 4 at 5, 3 at 6, 2 at 7, 1 at 9, 10 at 11, 30 at 12 |

Across 250 practice periods there is **not one** with two recycling attempts, and trash's
ceiling is again exactly **12** — reached in 30 of 250 periods. Both of those were flagged as
*pending* by the withdrawal, and both come back unchanged. The one-way ledge does what the
layout said it would, and capacity-1 did not alter it.

**The ratio is 618:163 = 3.79:1, not 12:1.** The reason is unchanged from the pre-#74 run and is
the part the arithmetic never modelled: EES does not choose to practise trash in most periods
(144 of 250 have zero trash attempts), and almost all of the practice budget goes on walking.

| skill | observed practice attempts |
|---|---|
| `MoveRoom` | **19,224/24,750** |
| `PressRecycling` | **3,684/24,750** |
| `PickupTrash` | 636/24,750 |
| `ThrowTrash` | **618/24,750** |
| `PressTrash` | 262/24,750 |
| `PickupRecycling` | 163/24,750 |
| `ThrowRecycling` | **163/24,750** |

**`PressRecycling` at 3,684/24,750 is new and it is the largest change in this table.** Pre-#74
there was one shared `Press` at 55/24,750 total. Capacity-1 gave each bin its own button, put
the recycling one in room 1 behind the ledge, and the result is that a robot stranded on the far
side spends its remaining steps pressing an already-empty recycling button — which is exactly
the late-practice behaviour the Tossing Room baseline's full-loop recording showed and could
only describe qualitatively. Here it is a count: **3,684 of 24,750 practice actions**, against
163 recycling throws.

Per seed the ratio ranges from **2.5** (seed 6, 40 vs 16) to **5.5** (seed 9, 83 vs 15), so the
pooled 3.79 is a genuine central value rather than one run's accident — but a single-seed reading
of this domain would still mislead by a factor of two in either direction.

### 2. The learning-rate result: roughly 4x slower, roughly matching the attempt ratio

Transitions at which each family first reaches a given share of its own 140 test tasks:

| level | TRASH | RECYCLING | ratio |
|---|---|---|---|
| 25% | 100 | 500 | **5.0x** |
| 50% | 400 | 1700 | **4.2x** |
| 75% | 600 | 2100 | 3.5x |
| 90% | 600 | **never** | — |

Against a measured attempt ratio of **3.79:1**. The predicted *relationship* holds — recycling is
slower by roughly the factor by which it is practised less — but the pre-#74 log's claim that it
held **"to two significant figures at both thresholds"** does **not** survive: the ratios here
are 5.0x, 4.2x and 3.5x against 3.79, and they drift monotonically downward as the threshold
rises. The honest statement is "the same order, matching within about 30%", and the drift is the
convergence in (4) seen from the other side.

### 3. Per practice attempt, trash is the better sampler — the opposite of what the defective run said

The add-effect audit first, because it is what makes the rest readable. On the pre-#74 run a
throw into an already-non-empty bin was scored a success at any force, and it happened to trash
constantly and to recycling never. **Capacity-1 closed that, and this run measures it closed
rather than assuming it:**

| skill | landed / attempts | EES scored / attempts | thrown into a prefilled bin | scored successes that missed |
|---|---|---|---|---|
| `ThrowTrash` | **286/618** | **286/618** | **0/618** | **0/286** |
| `ThrowRecycling` | **50/163** | **50/163** | **0/163** | **0/50** |

Landed and scored are now the *same column*. In the top-right panel of the first figure the
"EES scored a success" line sits exactly underneath "actually landed" for both skills, which is
what a closed channel looks like.

With the labels honest, the per-attempt comparison reverses:

| per greedy (learned-sampler) attempt | `ThrowTrash` | `ThrowRecycling` | gap | MDE |
|---|---|---|---|---|
| landed | **224/347** (64.6%) | **37/83** (44.6%) | **+19.98pp** | 17.12pp |

**+19.98pp against a 17.12pp MDE: detectable, and pointing the other way from the pre-#74 run's
−19.15pp.** The old finding — "recycling is the better sampler" — was an artifact of 313 of
trash's 532 recorded successes being throws that missed. It is withdrawn.

The epsilon-random draws remain the control, and they still say the two throws are comparably
hard: `ThrowTrash` lands **62/271** on randomly chosen forces and `ThrowRecycling` **13/80**, a
6.6pp gap against a **17.82pp** MDE — well inside it, so **no difference in intrinsic difficulty
is claimed**. Both sit near the ~19% first-principles base rate for a `U(0, 1)` force against a
`U(0.5, 1.0)` target at tolerance 0.1. The difference between the two skills is in what each
sampler learned, and now the sampler with more practice is the better one.

### 4. The endpoint is still a null result, and the per-seed test cannot even fire

| | final sweep |
|---|---|
| TRASH | **140/140** |
| RECYCLING | **124/140** |
| gap | +11.43pp |
| binomial noise floor | 5.98pp |
| minimum detectable effect (80%) | **16.74pp** |
| paired Wilcoxon over seeds | n = 3 after ties, W = 6.0, **p = 0.2500** |

**+11.43pp sits below the 16.74pp MDE, so the endpoint gap is not established** — and this is a
different null from the pre-#74 one (+3.57pp), for a different reason. Seven of ten seeds tie
exactly, at 14/14 against 14/14, which drops the Wilcoxon's effective n to 3 and its **attainable
p floor to 0.2500** — the same number the test actually returned. **At this n the test could not
have rejected under any outcome whatsoever**, so its p-value carries no information and should
not be quoted as evidence of anything.

**Across the whole curve the picture is different, and this time it is significant.** Mean area
under the per-family curve is **84.23** for TRASH against **43.10** for RECYCLING, a **+41.13pp**
difference, paired Wilcoxon n = 10, **p = 0.0020**. The pre-#74 run measured +12.94pp at
p = 0.1934 and could not resolve it. This one can, and by a wide margin — the entire effect lives
in the shape of the curve, exactly where the old log said to look.

`EMPTY` is **20/20 at every one of the 26 evaluation sweeps of every seed**, pre-practice
included — 260/260 seed-checkpoints. It is the deterministic control, and it measures the
symbolic model rather than either sampler.

### 5. Learning is a switch, not a curve — per skill, and one seed never flips it

The pooled RECYCLING line in the first figure climbs from 22/140 to 124/140 and looks like a
family improving steadily. It is not. Scoring every (seed, checkpoint) against 12/14 and 4/14 —
the same thresholds the Tossing Room baseline used:

| skill | at an extreme (≥12/14 or ≤4/14) | anywhere in between | seed-checkpoints |
|---|---|---|---|
| `ThrowTrash` | **227/260** | 33/260 | 260 |
| `ThrowRecycling` | **209/260** | 51/260 | 260 |

The split domain shows the same shape as `tossingroom`, on both skills; the Tossing Room
baseline reports 221/260 there, which is quoted from that log rather than re-measured here. Seeds
sit at one end or the other and snap between them at a seed-specific moment; the smooth pooled
line is an averaging artifact, and **describing RECYCLING as "still climbing" would be a
statement about the mean rather than about any seed.**

Per seed, the peak each family reached and where it ended:

| seed | TRASH peak | TRASH final | RECYCLING peak | RECYCLING final |
|---|---|---|---|---|
| 0 | 14/14 | 14/14 | 14/14 | 14/14 |
| 1 | 14/14 | 14/14 | **9/14** | **1/14** |
| 2 | 14/14 | 14/14 | 14/14 | 14/14 |
| 3 | 14/14 | 14/14 | 13/14 | 13/14 |
| 4 | 14/14 | 14/14 | 14/14 | 14/14 |
| 5 | 14/14 | 14/14 | 14/14 | 14/14 |
| 6 | 14/14 | 14/14 | 14/14 | 14/14 |
| 7 | 14/14 | 14/14 | 14/14 | 14/14 |
| 8 | 14/14 | 14/14 | 14/14 | **12/14** |
| 9 | 14/14 | 14/14 | 14/14 | 14/14 |

**Every one of the ten seeds finishes TRASH at 14/14.** RECYCLING is eight seeds at 14/14, one
at 13/14, and **seed 1 at 1/14**. Seed 1 is not a slow learner: its RECYCLING score across the 26
checkpoints is `6, 9, 0, 6, 3, 8, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1` —
it reached **9/14 at 100 transitions**, fell out, and sat at **1/14 for the last 19 of 26
checkpoints**. That is the split domain's version of the Tossing Room seed that reached 10/14
mid-run and ended at 3/14, and the whole of the pooled 124/140 rather than 140/140 is that one
seed plus two tasks elsewhere.

### 6. The mechanism: eight learned-sampler datapoints per run, and one of them is enough to get stuck

This is the sharpest thing the experiment says, and it is now measured rather than inferred from
a video.

**How little the recycling sampler is ever asked.** Of the two throws' **430** learned-sampler
draws across all ten seeds, recycling gets **83**:

| seed | greedy `ThrowTrash` draws | greedy `ThrowRecycling` draws |
|---|---|---|
| 0 | 34 | 11 |
| 1 | 30 | 11 |
| 2 | 39 | 11 |
| 3 | 30 | 7 |
| 4 | 34 | 8 |
| 5 | 40 | 7 |
| 6 | 25 | 5 |
| 7 | 29 | 10 |
| 8 | 34 | 6 |
| 9 | 52 | 7 |
| **total** | **347/430** | **83/430** |

**A whole 2500-transition run gives the recycling sampler between 5 and 11 chances to correct
itself, median 7.5.** With `--exploration-epsilon 0.5` half of every attempt is a coin flip that teaches the
classifier nothing about its own belief, so the 163 recycling attempts become 83 learned draws,
and those 83 are spread over 25 practice periods × 10 seeds.

**What it answers with them.** A target force is drawn `U(0.5, 1.0)` and the tolerance is 0.1, so
**any force below 0.4 misses whatever task it was aiming at** — a wrong answer that needs no
reference to the particular target.

| skill | greedy draws below 0.4 | longest run of consecutive all-missing practice periods, per seed |
|---|---|---|
| `ThrowTrash` | **44/347** | 1, 1, 1, 0, 0, 1, 0, 1, 1, 2 |
| `ThrowRecycling` | **24/83** | **5, 7, 9, 3, 1, 1, 1, 2, 4, 5** |

**Trash's worst run of all-missing periods across ten seeds is 2. Recycling reaches 9.** The
bottom-right panel of the per-seed figure is this table drawn: several orange traces leave the
reachable band entirely and stay near 0.00 for the middle third of the run, while no blue trace
does.

Seed 1 is the case in full. Its eleven greedy recycling forces, in practice-period order:

| period | 3 | 4 | 5 | 7 | 9 | 10 | 11 | 17 | 18 | 21 | 22 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| force | 0.40 | 0.65 | 0.89 | **0.03** | **0.00** | **0.23** | **0.01** | **0.17** | **0.15** | 0.67 | **0.28** |

**Eight of eleven are below the reachable band, and seven of those are consecutive draws spanning
periods 7 to 18** — a confident, stable, wrong answer, not scatter. Its RECYCLING score is frozen
at 1/14 across exactly that stretch. This is the same picture the Tossing Room baseline's video
showed (a sampler pinned near 0.00 force for four consecutive checkpoints); the difference is
that here it is 11 numbers from the trace rather than six frames.

**The asymmetry is the finding.** A trash sampler that convinces itself of a wrong force gets up
to twelve corrections in the next 100-step period. A recycling sampler gets **one**, and only in
periods where EES chooses to practise it at all — so the same wrong belief costs the two skills
completely different amounts of time to escape, and for one seed in ten it never was escaped
inside the budget.

### 7. Competence, re-measured: no longer backwards, still an overestimate

| | final competence | measured greedy landing rate |
|---|---|---|
| `ThrowTrash` | 0.927 | 224/347 (64.6%) |
| `ThrowRecycling` | 0.800 | 37/83 (44.6%) |

**The pre-#74 finding that EES's competence estimate ranks the two skills backwards does not
survive.** It ranked trash 0.859 below recycling's measured performance then because the
add-effect defect was feeding trash fabricated successes; with the defect closed, competence
ranks the two in the correct order at the endpoint and throughout the second half of the run.

Two things about it are still worth recording. It remains a substantial **overestimate** of both
— 0.927 against a measured 64.6% and 0.800 against 44.6% — because `OptimisticSkillCompetenceModel`
is a windowed estimate under a Beta(10, 1) prior whose mean is 0.909. And the overestimate is
**worse for the skill with fewer observations**: recycling's competence never falls below 0.742
across the whole run, while trash's drops to 0.543 at 500 transitions and then climbs, which is
what an estimate that is actually tracking data looks like. Competence is what `skill_costs()`
turns into `-log(competence)` plan edge costs and what `score_ground_skill` extrapolates when
choosing what to practise, so a skill whose estimate never moves off the prior is a skill the
planner has no reason to prioritise.

## Video

[`2026-08-05-tossingroomsplit-cap1-eval-progression-recycling.mp4`](./2026-08-05-tossingroomsplit-cap1-eval-progression-recycling.mp4)
— seed 5's own evaluation of its test task 0 at 0 / 500 / 1000 / 1500 / 2000 / 2500 transitions,
each segment preceded by a title card naming its transition count.

**Seed choice is a rule, not a selection:** the lowest seed whose test task 0 belongs to the
RECYCLING family. Seeds 0, 5 and 7 qualify; seed 5 is used because it is the one whose
per-checkpoint scores show the switch flipping *inside* the recorded window — `3, 5, 6, 8, 6,
14` across transitions 1500–2000 — so the clip contains both sides of it rather than only the
trained end. It is deliberately not success-selected: every seed but seed 1 ends this family at
13/14 or 14/14.

The recorded run reproduces `results/tossingroomsplit-cap1-throws/ees/5/stats.json`
**byte-for-byte** (identical SHA-256), so rendering did not perturb the run it recorded.

**Every number below is read off the frames.** This task's recycling `target_force` is
**0.72**, so with `throw_tolerance` 0.1 the winning window is 0.62–0.82.

| checkpoint | thrown force | frames | outcome |
|---|---|---|---|
| 0 | **0.38** | 13 | miss — below the reachable band |
| 500 | **0.30** | 13 | miss — below the reachable band |
| 1000 | **0.84** | 13 | miss — inside the band, over *this* target |
| 1500 | **0.92** | 13 | miss — inside the band, over *this* target |
| 2000 | **0.77** | 5 | **solved** |
| 2500 | **0.74** | 5 | **solved** |

The approach is identical at all six: `PickupRecycling(robot, recycling, room_3, pile)`,
`MoveRoom` 3→2, `MoveRoom` 2→1, `ThrowRecycling`. Only the force changes, and the whole
progression is legible in that one number. **Every failure ends the same way** — the item is
released, room 3 becomes unreachable across the one-way ledge, Fast Downward correctly reports
no plan, and the episode spends its remaining **eight frames** emitting `no-op (no plan)` until
the horizon of 12 expires. A missed recycling throw is terminal at any horizon, which is why
this family cannot retry inside an evaluation episode either, not just inside a practice period.

Two things the clip shows that the tables cannot. The first two checkpoints are the
**pinned-below-the-band** regime the trace counts (24/83 of all greedy recycling draws): 0.38
and 0.30 are wrong for *every* task in the domain, not just this one. The middle two are a
different failure — 0.84 and 0.92 are perfectly reachable forces that simply overshoot this
task's 0.72, i.e. a sampler that has escaped the wrong region but not yet learned to condition
on the target. The switch between 1500 and 2000 is the sampler crossing from the second regime
into the tolerance window, and the episode collapses from 13 frames to 5 the moment it does.

## Verdict on the prediction

| claim | verdict |
|---|---|
| recycling gets **exactly 1 attempt per practice period, ever** | **held exactly** — 0 or 1 in all 250 periods, never 2 (163 at 1, 87 at 0) |
| trash gets **~12 attempts** in a 100-step period | **held, conditionally** — 12 is the observed ceiling, reached in 30 periods, but 144 of 250 have zero |
| attempt ratio **≈ 1:12** | **refuted** — measured **3.79:1** (618:163) |
| recycling learns **far slower, at roughly the attempt ratio** | **held, loosely** — 5.0x / 4.2x / 3.5x against a 3.79:1 ratio. The pre-#74 "two significant figures" version is withdrawn |
| **possibly not at all** within a practice budget | **refuted for 9 seeds, held for 1** — 124/140 pooled, but seed 1 ends at 1/14 having peaked at 9/14 |
| *(pre-#74, now withdrawn)* recycling is the **better** sampler per attempt | **reversed** — trash 224/347 vs recycling 37/83, +19.98pp against a 17.12pp MDE |
| *(pre-#74, now withdrawn)* competence ranks the two skills **backwards** | **does not survive** — 0.927 vs 0.800 is the correct order |
| *(new)* learning is a **switch**, not a curve | **held** — 227/260 and 209/260 seed-checkpoints at an extreme |

## Recommendation

1. **Report this domain by curve statistic, never by endpoint.** The pre-specified endpoint is a
   null result at +11.43pp against a 16.74pp MDE, and the per-seed Wilcoxon cannot fire at all
   (7 of 10 seeds tie, p floor 0.2500 = the p returned). The AUC comparison resolves the same
   effect at **+41.13pp, p = 0.0020**. Pre-specify AUC or transitions-to-threshold next time.
2. **The recycling sampler's problem is sample count, and it is measurable now.** 83/430 of the
   learned-sampler draws, 5-11 per run against trash's 25-52, 24/83 of them outside the reachable
   band, and
   all-missing streaks up to 9 periods. The obvious intervention is a **cheaper reset** — the
   per-cycle `PERIOD` reset is the only thing that returns the robot to the pile, so the
   recycling budget is exactly one throw per cycle by construction. A reset-interval arm on this
   domain would test it directly, and `--practice-reset-interval` already exists.
3. **`PressRecycling` at 3,684/24,750 is worth its own look.** It is 22x the recycling throw count
   and is almost entirely a stranded robot pressing an already-empty button. That is practice
   budget being spent on a no-op, and it is the largest single change the capacity-1 redesign made
   to what practice *does* in this domain.
4. **Do not quote the competence model as a learning curve**, even now that it ranks correctly. It
   overestimates both skills by 20–30 points and never moves off the Beta(10, 1) prior for the
   skill with few observations — which is the skill the planner most needs a real estimate of.
5. **10 seeds is enough for the AUC claim and not for the endpoint one.** The +41.13pp AUC
   difference is resolved at p = 0.0020; the +11.43pp endpoint gap needs roughly four times the
   seeds, and would still be measuring the wrong thing.
6. **Nothing here argues for or against the split as a design.** No shared-sampler arm was run.
   Result (3) compares two skills within one run, not two architectures.

## Raw data

* [`2026-08-05-tossingroomsplit-throw-rates.json`](./2026-08-05-tossingroomsplit-throw-rates.json)
  — all ten seeds' per-period skill tallies (attempts, successes, landings, prefilled-bin
  attempts, the epsilon-random split of each, and every learned-sampler force with the target it
  aimed at), per-cycle competence, and per-sweep evaluation records with their goal-family
  breakdowns. **Every count on this page re-derives from it**, and none is reconstructed by
  multiplying a percentage by *n*. Checked rather than asserted: re-running the analysis with
  `--traces` pointing at this one committed file instead of the ten shards produces a
  **byte-identical** report.
* [`2026-08-05-tossingroomsplit-throw-rates.png`](./2026-08-05-tossingroomsplit-throw-rates.png)
  — the pooled figure.
* [`2026-08-05-tossingroomsplit-throw-rates-per-seed.png`](./2026-08-05-tossingroomsplit-throw-rates-per-seed.png)
  — the per-seed spread, the seed-checkpoint distribution, and what each sampler answered.
* [`2026-08-05-tossingroomsplit-cap1-eval-progression-recycling.mp4`](./2026-08-05-tossingroomsplit-cap1-eval-progression-recycling.mp4)
  — the video above.

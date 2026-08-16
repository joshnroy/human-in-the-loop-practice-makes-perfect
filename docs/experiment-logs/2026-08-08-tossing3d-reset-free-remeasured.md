# A reset-free robot on Tossing3D does not practise less — it stops practising, on 10/10 seeds after one cycle

> **STALENESS NOTE (added 2026-08-16, at the two-skill migration).** Every Tossing3D
> number on this page was measured on the **three-skill** decomposition: `Pick`
> (sampling distance and rotation) -> `MoveToThrowPose` (sampling a standoff) -> `Toss`
> (sampling release speed and gripper release millisecond), over six predicates
> including `RobotAtSuccessfulThrowPose`. That domain no longer exists. Upstream replaced
> it with two skills -- a parameterless `pick_cube` and a composed
> `move_to_toss_location_and_toss` taking four parameters -- and this repo followed, so:
> the pick has **no** continuous parameters and therefore no sampler at all; the standoff
> is now the composed toss's first parameter, drawn from upstream's `(1.25, 1.45)` rather
> than this repo's `(1.10, 1.75)`; release speed is drawn from `(115, 140)` deg/s rather
> than `(60, 140)`; the gripper release millisecond from `(700, 840)` rather than
> `(300, 1400)`; `RobotAtSuccessfulThrowPose` and the whole `THROW_RANGE` calibration are
> deleted; and `OnGround` now accepts a cube resting on any face rather than only the one
> it started on. Two measured grasp fixes (centre grasp, approach settling) also reach the
> pick for the first time.
>
> **Nothing on this page is edited or recomputed.** It is a correct description of the
> domain that was actually in effect when these runs happened. It is simply not evidence
> about the two-skill domain, and no count here is directly comparable to a re-run on it.

> **Staleness note (2026-08-10).** This page's per-skill counts — `Pick 1/1`,
> `MoveToThrowPose 1/5`, `Toss 1/1`, the `3`–`13` transition-index range where the toss
> happens, and any other figure that traces back to how many `MoveToThrowPose` draws the
> sampler needed before succeeding — were measured with `THROW_STANDOFF_BOUNDS =
> (0.45, 1.75)`. That lower bound has since moved to `1.10` to fix a
> `cuboid_barrier`-collision defect (see the staleness note atop
> `2026-08-06-tossing3d-ees-first-real.md`), roughly halving the sampler's range and
> doubling the derived acceptance band's share of it — a narrower, proportionally
> easier-to-hit range plausibly changes how many draws it takes to land a throw. **The
> stranding finding itself is unaffected**: it is about what happens once the cube is past
> the barrier, after which no skill is applicable regardless of how the standoff was
> sampled to get there. Nothing on this page is recomputed or edited.

**Answer: reset-free practice collapses on this domain, and the score gap is a symptom
rather than the finding.** Late window `24.9/100` against `scheduled`'s `80.8/100`, lower
on **`10/10`** seeds, mean `-5.59` tasks per seed, exact paired sign-flip
**`p = 0.001953`** — the smallest value attainable at ten paired seeds — against an MDE of
`1.94`. This is **not** a null result.

**The mechanism, measured rather than inferred: every seed strands at cycle 1.** The
reset-free arm took **`77`** practice transitions in total across ten 100-cycle runs,
against `3350` for `scheduled`. **`990/1000`** of its practice cycles took **zero** steps,
and **`10/10`** seeds stranded from cycle **1** and never recovered — a robot that
practised once and then stood still for ninety-nine cycles. It ends at `24.9/100`,
numerically level with the `24/100` `random-skills` reference from #133; **that is a
descriptive comparison, not a tested one**, and #133 measured its baseline at 20 cycles
and before #160 split the evaluation environment, so the two were not produced under the
same conditions.

**The `scheduled` arm came back byte-identical to #178's committed runs on `10/10` seeds**,
which is the independent check that #179's fix touched only the arm it was meant to.

## Question / goal

**What does a genuinely reset-free arm cost on Tossing3D?** The question could not be
asked until now: `--practice-reset-policy never` was a no-op on this domain, so the arm
that carried the name had never actually been run.

## Background

`--practice-reset-policy never` exists to turn off the per-period reset, so that "how
much is a free reset worth?" is an empirical question rather than an assumption. #160
gave Tossing3D its own evaluation `Problem` specifically so the flag could mean something
here — without one, every evaluation episode's `reset_to_task` writes into the practice
environment.

#178 ran the flag on this domain for the first time, at 100 cycles × 10 seeds, and found
in passing that it had changed nothing: the two arms' `stats.json` files differed in
exactly one field, `num_practice_resets` (`100` against `0`), on `10/10` seeds.
`PracticeLoop` sampled the train task *before* the reset-policy branch, and
`Tossing3DTasks.build_task` could only obtain an initial `State` by rebuilding the MuJoCo
scene — so the scene was rebuilt every cycle whatever the flag said, and the counter
correctly reported `0` for the branch it counts while the condition it certifies was
false.

#179 fixed that. `Tasks` gained `sample_train_task_in_place`, and Tossing3D overrides it
to pair the environment's **current** state with the fixed goal, touching no simulator.
This is the first measurement of the arm — not a re-analysis of an old one.

### What "reset-free" means on this domain, which is not what it means elsewhere

After the fix, a reset-free run practises in **one scene for its whole length** —
whatever `hard_reset` left behind (`canonical_seed`, upstream's 125). That is not an
implementation shortcut and not a limitation to engineer around. On Tossing3D the only
way to obtain a new initial state is `env.reset(seed=...)`, so **handing the robot a new
scene and resetting it are the same physical act**. Josh accepted this explicitly:

> one task is fine, we'll add variability later - this actually mirrors more of what
> happens in the real world robot

So the arm is "no reset" and "no scene variety" *inseparably*, and every number below is a
number about both together. A scratch third simulator to manufacture scene variety was
considered and rejected as incoherent: the practice robot would be handed tasks describing
a world it does not inhabit.

### The domain feature that makes this the interesting case

Tossing3D is here **because its actions are irreversible**. A tossed cube ends up past an
immovable 5 m barrier and no skill brings it back, and `Toss` deletes `Reachable`. That is
precisely the condition under which "ending an episode does not imply a free reset" — the
assumption this whole project breaks from Gym on.

## Hypothesis

Registered in writing before any 100-cycle `stats.json` from this sweep was read:

> **The genuinely reset-free arm ends *below* the scheduled arm.**
>
> 1. **One scene for the whole run.** The sampler sees a single point of the scene
>    distribution and is tested across held-out scenes.
> 2. **The throw is irreversible.** Without a per-period reset the robot can be stranded
>    in a post-toss state with nothing applicable, so practice periods produce no throw to
>    learn from.
>
> **Counter-consideration, stated so a small gap is not read as a surprise.** Tossing3D's
> only learnable quantity is the approach standoff, and `bin_init_region` is degenerate —
> the bin does not move — so the correct standoff is a **constant** (#133). Practising in
> one scene may therefore cost the sampler very little. If mechanism 1 dominates the gap
> should be small; if mechanism 2 dominates it should be large.

**Confirmed in direction, and mechanism 2 dominates completely.** The gap is large
(`-5.59` tasks per seed), and the stranding measurement below shows mechanism 1 barely
gets a chance to matter: the robot stops practising after a single cycle, so the size of
its scene distribution is close to irrelevant.

## Guidance given

- **The fix shape was Josh's:** specify the state at construction — under `never`, build
  the training task from the current state plus the fixed goal, with no simulator
  operation at all.
- **The consequence was accepted explicitly** (quoted above): one scene for the whole run,
  and variability added later.
- **A scratch/third simulator was considered and rejected** as incoherent.
- `scheduled` keeps its current behaviour, and was expected to come back **byte-identical**
  to #178's committed runs. A difference there was a stop-and-report condition, not
  something to explain.
- Read the conditions out of the committed runs' own `config_snapshot.json` rather than
  reconstructing them from prose.
- Per-seed spread, a paired test across the ten seeds, and an MDE beside any null result.

## Methods

Two arms, `ees` only, 10 fixed seeds (`0-9`), 100 cycles, driven by `scripts/run_sweep.py`
inside a memory-capped **detached systemd service** (a `--scope` dies with the calling
shell and has already destroyed a long sweep here). Both arms ran concurrently at
`--max-workers 10`, i.e. 20 concurrent runs in one wave; the cap was verified in the
kernel (`memory.max` = 44 GiB) rather than trusted from the command line, and peak usage
sat around 22 GB.

**Conditions were not reconstructed from prose.** They were read out of #178's own
committed `config_snapshot.json` and then *checked*: a 2-cycle probe was run with the
candidate flag set and its `config_snapshot.json["args"]` compared key for key against the
committed one, ignoring only `num_cycles`, `output_dir` and `seed`. **`24/24` keys matched
on both arms.** The only flag easy to miss is
`reproduce_predicators_explore_target_only = False`, passed explicitly as
`--no-reproduce-predicators-explore-target-only`.

### The two arms are scored on the same test tasks

Worth stating explicitly, because #179's whole point is that the arms' *train* task
sequences are no longer identical on this domain. **The evaluation sets are.** Both arms
draw their ten test tasks from the separate evaluation `Problem` (#160), whose `Tasks` is
constructed identically from the same seed and derives the same test scene-seed stream in
both arms — and nothing in the practice loop can advance it. What diverges is the
practice-side train stream, which under `never` is not drawn from at all
(pinned by `test_sampling_a_train_task_in_place_leaves_the_scene_seed_stream_untouched`).
So the comparison is two robots measured on identical held-out scenes.

### Provenance of the committed runs

Every one of the 20 `config_snapshot.json` files records `git_commit = 2e79dde` with
`git_dirty = false`. That commit was later amended away while assembling this log, so it
is **not** an ancestor of this branch's head — stated plainly rather than left for a
reader to trip over. It does not weaken the provenance: `de997e0` (#179's fix) **is** an
ancestor of it, and `git diff -- src/` between `de997e0`, `2e79dde` and this branch's head
is **empty**. The runs were produced by the fix, from a clean tree, by exactly the library
code this PR ships.

### The decision rule, fixed before any final number was read

Reused unchanged from #178, because the volatility it was designed for is a property of
the domain rather than of that experiment — Tossing3D's per-seed score swings several
tasks between adjacent sweeps with no learning event.

- A seed's score is its **mean solved count over the last 10 sweeps** (`LATE`), never a
  single final sweep.
- **Paired** across the ten seeds, because both arms ran the same fixed seed set. An
  unpaired test would discard exactly the structure the design bought.
- Exact paired sign-flip on per-seed `never − scheduled`, with the **MDE reported beside
  it, always**.
- Progress was watched live during the run, as #178 also did. That is harmless precisely
  because the rule above was fixed in writing beforehand.

### A second endpoint, added before any final score was read

`transitions per cycle`, and the **stranding onset** derived from it: the first cycle of
the *terminal* run of zero-transition cycles — terminal-from-here rather than "the first
gap", the same definition `pickup_weight_stranding.py` uses on Tossing Room, so the two
experiments read side by side.

It was added because "the reset-free arm scored lower" and "the reset-free arm stopped
practising" are very different claims that a score cannot separate, and the live progress
made the second look likely. Adding it *before* reading any final score is what keeps it a
measurement rather than a search for an explanation.

## Results

### The `scheduled` arm is untouched — `10/10` byte-identical

Every one of the ten re-run `scheduled/<seed>/stats.json` files is byte-for-byte the file
#178 committed. `sha256`, not a field diff:

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `sha256` prefix | `d29bc8c2` | `8a0f6cad` | `dbadb7d3` | `f2d007fa` | `38910158` | `bab036ac` | `8b21cd2f` | `9d2c2201` | `6202d758` | `5bf838bf` |
| identical | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

This is the check that matters most for trusting everything else: #179 changed the code
path both arms run through, and `scheduled` is the arm every committed Tossing3D number
sits on.

### The reset-free arm collapses to the non-learning baseline

| measure | `scheduled` | `never` |
| --- | --- | --- |
| **`LATE` window (last 10 sweeps)** | **`80.8/100`** | **`24.9/100`** |
| seeds where `never` is lower | — | `10/10` |
| mean per-seed difference | — | `-5.59` tasks |
| exact paired sign-flip | — | **`p = 0.001953`** |
| MDE at 80% power | — | `1.94` tasks per seed |

`p = 0.001953` is `2/1024`, the **smallest two-sided p an exact sign-flip can return at
ten paired seeds** — every seed moved the same way, so the test is saturated. The observed
effect (`5.59`) is nearly three times the MDE (`1.94`).

**`never`'s `24.9/100` is numerically level with `random-skills`' `24/100` (#133)** — a
hundred cycles of reset-free practice left the robot about where a robot that consults no
sampler at all already was. **This is a description of two numbers, not a test.** They
were not measured under the same conditions (#133 ran 20 cycles and predates #160's
separate evaluation `Problem`), and no comparison between them is claimed here beyond the
arithmetic. The tested claim on this page is `never` against `scheduled`, which were run
paired, on the same seeds, under identical conditions.

![Two learning curves against practice cycle, ten faint per-seed lines under a bold pooled mean for each arm. The scheduled arm in blue climbs from about 2 to about 8 out of 10 within a dozen cycles and stays there; the reset-free arm in red never leaves the dotted random-skills reference line at 2.4 for the whole hundred cycles, well below the dashed skill-oracle line at 10.](2026-08-08-tossing3d-reset-free-remeasured-curves.png)

**Figure 1. The reset-free arm never leaves the non-learning baseline.** Bold pooled mean
over faint per-seed lines, both arms on the same axes. `skill-oracle` (`100/100`) and
`random-skills` (`24/100`), both from #133, are drawn as **reference lines rather than
curves**: neither learns, so a curve would invite a reader to look for a trend in a
constant. The red band's per-seed haze is as wide as the blue's — this is not a quieter
robot, it is a robot whose score is entirely evaluation noise.

![Ten lines, one per seed, joining each seed's mean score over the last ten sweeps under the scheduled policy to the same quantity under the reset-free policy. Every line slopes steeply down, from a cluster between 7 and 9 out of 10 to a spread between 0 and 6.5, with most landing at or below the dotted random-skills reference line at 2.4.](2026-08-08-tossing3d-reset-free-remeasured-paired.png)

**Figure 2. Every seed falls, and most fall to the floor.** Plotted per seed rather than
as two bars because with ten seeds a bar chart of two means hides one seed driving the
whole movement — here nothing is hidden, all ten move the same way. Seed 9 (`8.6 → 6.5`)
is the mildest and seed 0 (`9.0 → 0.0`) the most complete.

### The mechanism: it stops practising, it does not practise badly

| measure | `scheduled` | `never` |
| --- | --- | --- |
| total practice transitions (10 seeds × 100 cycles) | `3350` | **`77`** |
| seeds ever stranded | `0/10` | **`10/10`** |
| stranding onset | — | **cycle `1` on every seed** |
| practice cycles taking zero steps | `0/1000` | **`990/1000`** |

![Cumulative practice transitions against practice cycle, one line per seed, left panel. The ten scheduled lines rise steadily and linearly to between 316 and 354 by cycle 100. The ten reset-free lines rise only in the first cycle, to between 3 and 13, and are then perfectly flat for the remaining ninety-nine cycles. A shaded region and dashed vertical line mark cycle 1 onward as stranded (0 transitions/cycle, 10/10 never-arm seeds), and a marker on seed 0's curve is labelled "seed 0: last action (Toss) at transition 7". A small right panel plots the toss transition index for all ten never-arm seeds as a dot strip -- 3, 3, 5, 5, 5, 7, 12, 12, 12, 13 -- with seed 0's value of 7 marked in a darker colour near the median of 6.0.](2026-08-08-tossing3d-reset-free-remeasured-practice.png)

**Figure 3. Ten flat lines, and where each one stops.** A robot that keeps practising is
a line that keeps rising; a stranded one is a line that goes flat and stays flat. The
annotation is derived from the per-cycle transition and skill-attempt record
(`practice_outcomes_per_cycle` and `evaluations`), never from where a line visually goes
flat: `toss_transition_index` locates the last cycle with any activity, confirms from
that cycle's own attempt counts that `Toss` was among the skills attempted there, and
reports the cumulative transition count as of its end.

**Two different claims sit on this figure, and they should not be conflated.** *Every*
never-arm seed strands at **cycle 1** — that is the shaded region and dashed line, uniform
across all ten seeds, matching the `10/10` in the table above. The **transition index** at
which the toss happens is not uniform: it ranges `3`–`13` across seeds (the right panel),
because it depends on how many failed `MoveToThrowPose` draws the sampler needed before
succeeding. Seed 0's marked value, `7`, is one seed's particular number, not a claim about
every seed — it was chosen because its toss-transition index sits closest to the ten-seed
median (`6.0`), and because it is the seed already used as the worked example throughout
this write-up.

**Read at the skill level, from the runs' own `practice_outcomes_per_cycle`,** the picture
is unambiguous. Cycle 0 is *identical* between the arms — `Pick` `1/1`,
`MoveToThrowPose` `1/5`, `Toss` `1/1`, 7 transitions on seed 0 — because both start from
the same `hard_reset` scene. From cycle 1 the reset-free arm records **zero attempts of
every skill**, for the remaining ninety-nine cycles.

So the reset-free robot's *entire* experience for a 100-cycle run is one practice period.
`3` to `13` transitions, once, and then nothing.

### Two facts, and only one of them is about the domain

`990/1000` is a property of **(this domain × EES)**, not of the domain alone, and the two
halves should not be quoted as one.

**The domain fact — checked against the operators, not inferred from prose.** `Toss`
deletes both `Holding(?robot, ?cube)` and `Reachable(?cube, ?barrier)`
(`environments/tossing3d/skills.py`, `Tossing3DSkills.TOSS`). In the resulting state
**none of the three skills is applicable**: `Pick` requires `Reachable`, which is gone;
`MoveToThrowPose` and `Toss` both require `Holding`, which is gone. There is genuinely
nothing to do, and no skill in this domain retrieves a cube from past the barrier. This
half is method-independent.

**The method fact.** EES's planner finds no applicable skill and raises
`InteractionComplete`, so the period ends immediately and the cost shows up as a *zero*.
A method that acted anyway — `random-skills` samples a skill regardless — would instead
burn its full `--max-steps-per-interaction` on no-ops and failing controllers every cycle,
learning exactly as little while recording ~2000 transitions rather than `77`. So
`990/1000` idle cycles is EES *correctly declining to act* in a situation the domain has
made hopeless. **The domain fact is the finding; the idle count is how this particular
robot's version of it happens to be denominated.**

### Per seed

| seed | `scheduled` `LATE` | `never` `LATE` | difference | `never` transitions | `never` idle cycles |
| --- | --- | --- | --- | --- | --- |
| 0 | 9.0 | 0.0 | `-9.0` | 7 | `99/100` |
| 1 | 7.7 | 3.3 | `-4.4` | 12 | `99/100` |
| 2 | 8.4 | 3.6 | `-4.8` | 3 | `99/100` |
| 3 | 7.4 | 3.2 | `-4.2` | 13 | `99/100` |
| 4 | 7.9 | 0.0 | `-7.9` | 5 | `99/100` |
| 5 | 8.2 | 3.9 | `-4.3` | 3 | `99/100` |
| 6 | 8.4 | 0.6 | `-7.8` | 12 | `99/100` |
| 7 | 8.0 | 1.0 | `-7.0` | 5 | `99/100` |
| 8 | 7.2 | 2.8 | `-4.4` | 5 | `99/100` |
| 9 | 8.6 | 6.5 | `-2.1` | 12 | `99/100` |

### What this does *not* establish

- **It does not decompose the two mechanisms.** No-reset and no-scene-variety are
  inseparable on this domain by construction, so `-5.59` is a number about both together.
  In practice the stranding is so total that scene variety barely gets a chance to matter
  — but that is an argument from the mechanism data, not a decomposition.
- **It does not measure "the value of a reset" in general.** It measures it on a domain
  chosen for having irreversible actions. Tossing Room's reset-free arms behave quite
  differently and should not be read off this page.
- **It is not a claim that reset-free practice is unworkable.** It is a claim that
  reset-free practice *without any means of recovery* is, here. What a recovery mechanism
  buys is the sibling question — see the recommendation.

### Cost

`analysis/run_timing.py` over both arms, at 20 concurrent runs on 24 cores with other
jobs on the box: `scheduled` median `6058.4 s` per run (`6133.2 s` wall for the arm),
`never` median `4838.4 s` (`5497.7 s` wall). The reset-free arm is cheaper only because
its practice periods are empty; it still pays for all 101 evaluation sweeps. **Not
comparable to #178's `5013.9 s`**, which was measured at `--max-workers 8` against a
much quieter machine.

## Recommendation

**Cite this as the motivating measurement for the human in the loop, not as "reset-free is
worse".** It is the cleanest evidence this project has that the premise it is built on is
real: on a domain with irreversible actions and no way to recover, a robot practising
without a free reset does not degrade gracefully — it takes one action and stops, and a
hundred cycles of budget buys nothing at all.

**Quote the domain fact, not the idle count.** The durable claim is *"after one toss, no
skill in this domain is applicable and none retrieves the cube"* — checked against the
operators, true of any method. `990/1000` idle cycles is EES's particular denomination of
that fact (see above), and a different method would produce a different number for the
same underlying situation.

Three follow-ups, in the order the evidence argues for them:

1. **This is now the obvious place to point a `HumanOracle` at.** The ladder built in
   #147–#151 measured rescue-on-stuck recovering reset-free practice on Tossing Room
   (`227/300` against `112/300`). Tossing3D is the harder and more honest case, because
   here the robot cannot recover *at all* on its own — the gap a human closes is the whole
   `80.8` versus `24.9`, not a fraction of it. Running that ladder on this domain is a
   direct, high-value experiment that this PR makes possible.
2. **Give the reset-free arm scene variety only as a separate, argued change.** It is not
   an engineering gap to close quietly: on this domain a new scene *is* a reset, so
   anything that supplies one is supplying a rescue under another name. If a future
   experiment needs it, it should say what physical act it corresponds to.
3. **Do not re-run this to get a tighter p-value.** `p = 0.001953` is the floor at ten
   paired seeds and every seed moved the same way. More seeds would buy precision on an
   effect whose mechanism is already read directly off `990/1000` idle cycles.

## Artifacts

- Runs, both arms, 10 seeds each:
  [`2026-08-08-tossing3d-reset-free-remeasured-runs/`](2026-08-08-tossing3d-reset-free-remeasured-runs)
  (`stats.json`, `config_snapshot.json` and `timing.json` per run; `sampler_draws.jsonl`
  and `log.txt` are deliberately not committed — tens of MB per run, and machine-specific
  noise respectively).
- Analysis: `analysis/practice_makes_perfect/tossing3d_reset_free_arms.py`, covered by
  `tests/analysis/practice_makes_perfect/test_tossing3d_reset_free_arms.py`.
- Figures: [curves](2026-08-08-tossing3d-reset-free-remeasured-curves.png),
  [paired](2026-08-08-tossing3d-reset-free-remeasured-paired.png),
  [practice](2026-08-08-tossing3d-reset-free-remeasured-practice.png).

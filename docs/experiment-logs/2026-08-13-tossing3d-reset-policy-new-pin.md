# The reset-free collapse survives the pin bump, but it no longer needs a throw to happen

**Answer: the finding reproduces, and the mechanism widened.** Re-measured at the KINDER
pins bumped on 2026-08-13, the reset-free arm still collapses — late window `12.2/100`
against `scheduled`'s `73.4/100`, lower on **`10/10`** seeds, mean `-6.12` tasks per seed,
exact paired sign-flip **`p = 0.001953`** against an MDE of `0.98`. All ten seeds still
strand from cycle **`1`** and never recover, and **`990/1000`** practice cycles still take
zero steps. This is not a null result.

**What changed is the route in.** The 2026-08-08 page's mechanism was one sentence: the
robot throws the cube past an immovable barrier, `Toss` deletes `Reachable`, and nothing
is applicable afterwards. At these pins **`6/10`** seeds strand that way. The other
**`4/10`** never attempt a `Toss` at all: they end with the gripper shut on nothing —
`HandEmpty` false because the gripper is closed, `Holding` false because the cube is back
on the floor — which is a deadlock the published mechanism does not describe. `Pick` needs
`HandEmpty`; `MoveToThrowPose` and `Toss` need `Holding`; neither holds, so nothing is
applicable, for a completely different physical reason.

**The dead end itself is unanimous: `10/10` seeds end in a state where no skill's
preconditions are satisfied,** checked by evaluating the domain's own predicate
classifiers on each run's own final recorded feature vector. So "the reset-free robot runs
out of actions" is if anything *more* robust than published — it is not one failure mode,
it is two.

## Question / goal

**Does the reset-free collapse on Tossing3D survive a pin bump that changed the throw?**
`docs/experiment-logs/2026-08-08-tossing3d-reset-free-remeasured.md` is this project's
motivating measurement: the cleanest evidence that a robot practising without a free reset
on an irreversible domain does not degrade gracefully but stops entirely. Everything
downstream — the case for a `HumanOracle` at all — rests on it. PR #246 moved both KINDER
pins, and with them the dynamics that measurement was taken under, so it is worth knowing
whether the result is a property of the domain or of one pin.

## Background

`--practice-reset-policy never` turns off the per-period reset. On Tossing3D that is
inseparably also "no scene variety", because the only way to obtain a new initial state is
`env.reset(seed=...)` — handing the robot a new scene and resetting it are the same
physical act. #179 made the flag real on this domain (before it, `Tossing3DTasks.build_task`
rebuilt the MuJoCo scene every cycle whatever the flag said); #178 and #160 are the
measurement and the separate evaluation `Problem` it needs. The 2026-08-08 page is the
first and so far only measurement of the arm.

**What moved underneath it.** The 2026-08-08 runs' own `config_snapshot.json` files record
`kindergarden 4113237` and `kinder_models 11eace5`. PR #246 bumped `reference/kindergarden`
to `c9f00e8` and `reference/kinder-baselines` to `9e88126`. The two must bump together: `MujocoEnv.step`
now asserts a control schedule covers the period exactly, and `TossController` emits the
matching full-width schedule. The consequence for this experiment is recorded in
`CLAUDE.md`: the 1 kHz release scheduling moves the canonical oracle rollout's resting *x*
by **+41.6 mm**, from `1.9901` to `2.0318`. That is a real change to where a thrown cube
lands, on a domain whose success criterion is a 0.30 m box. So byte-identity with the
2026-08-08 runs is not available and was not sought; the comparison here is at the level
of the finding.

## Hypothesis

Registered before any `stats.json` from this sweep was read:

> **The collapse reproduces.** The published mechanism is a claim about the domain's
> operators, not about the throw's ballistics: `Toss` unconditionally deletes `Holding`
> and `Reachable`, `Pick` requires `Reachable`, and the barrier is a wall the base cannot
> cross. A 41.6 mm change in where the cube lands changes *which* side of the scored box
> it lands on; it does not give the robot a way to retrieve it. So `never` should still
> strand, still on `10/10` seeds, and the gap should still be large and one-directional.
>
> **The score levels are expected to move**, in both arms, because the throw moved. A
> changed `scheduled` plateau is not evidence of a defect.

**Confirmed, and then some.** The collapse reproduced at full strength. The part the
hypothesis got wrong by omission is that it assumed a `Toss` was the only way in — it was
at the old pins, on `10/10` seeds, and it is not now.

## Guidance given

- Analysis and write-up only. **Do not re-run the sweep**; the runs were already complete.
- Compare at the level of the *finding*, not the bytes: does `never` still collapse, do
  all ten seeds still strand at cycle 1, is the gap the same size.
- If the mechanism changed, that is the headline. If it did not, that is also a real
  result — a robustness check on the project's motivating measurement.
- Verify the pins and commit from each run's own `config_snapshot.json` rather than
  trusting the brief.
- Per-seed spread, a **paired** test across the ten seeds, and an MDE beside any null
  result. Counts as `x/y`, never a bare percentage.
- Never edit, restate or recompute a published number; add a marked note beside it.

## Methods

Two arms, `ees` only, 10 fixed seeds (`0-9`), 100 cycles, `--env tossing3d`, driven by
`scripts/run_sweep.py`. Both arms ran concurrently at `--max-workers 10`, i.e. 20
concurrent runs on 24 cores. 101 evaluation sweeps per run, 10 test tasks per sweep,
`101/101` sweeps completed on `10/10` seeds in both arms.

### Provenance, read from the runs rather than from the brief

All **`20/20`** `config_snapshot.json` files agree, with no exceptions:

| field | value |
| --- | --- |
| `git_commit` | `54e87947cdf9972c6358ef356a602f39e53ed9e8`, `git_dirty = false` |
| `kindergarden_commit` | `c9f00e82f94c807f5a92c76a29f55cc572cdd2a2`, clean |
| `kinder_models_commit` | `9e881264d6868a391fff8e3090b9ea44bea1d231`, clean |
| `fast_downward_commit` | `6230635ccff53e1df38ead53b057a2a0e9160275` |

`54e8794` is **not** an ancestor of this branch's head: it was the pre-squash form of the
pin bump, which merged as `e516455`. Stated plainly rather than left for a reader to trip
over, and it does not weaken the provenance — `git diff` between `54e8794` and this
branch's head is **empty** for both `src/` and `reference/`, so the runs were produced by
exactly the library code and exactly the two gitlinks this branch ships.

The two arms' resolved argument namespaces were compared key for key. Each snapshot carries
34 keys; three are expected to differ (`practice_reset_policy`, `seed`, `output_dir`), and
the remaining **`31/31` match on all `20/20` runs**. So the only manipulated variable is the
one under test.

`scripts/update_reference_repos.sh --check` reports `skipped 3/3`, `not initialised`, exit
`2`, in this worktree — the expected state, since `git worktree add` does not populate
submodules and this analysis needs no simulator. It is not evidence about the sweep; the
snapshots above are.

### The decision rule, unchanged from #178 and #179

Reused rather than reinvented, because the volatility it was designed for is a property of
the domain: Tossing3D's per-seed score swings several tasks between adjacent sweeps with no
learning event.

- A seed's score is its **mean solved count over the last 10 sweeps** (`LATE`), never a
  single final sweep.
- **Paired** across the ten seeds, because both arms ran the same fixed seed set.
- Exact paired sign-flip on per-seed `never − scheduled`, **with the MDE beside it**.
- Stranding onset is the first cycle of the *terminal* run of zero-transition cycles —
  terminal-from-here, not "the first gap", the same definition `pickup_weight_stranding.py`
  uses on Tossing Room.

### One thing added, and why it is a measurement rather than a search for an explanation

`analysis/practice_makes_perfect/tossing3d_reset_free_arms.py` already refused to describe
this data. Its `toss_transition_index` derives the figure annotation from the per-cycle
skill-attempt record and **raises** if the last active cycle recorded no `Toss` — a guard
written in #179 precisely so that a stall with a different cause could not be silently
labelled as a throw. Running it on this sweep raised. That refusal is how the mechanism
change was found; it was not looked for.

Two readers were added in response, both **post-run only**:

- `ended_on_a_toss`, which reports the route instead of assuming it, from `stats.json`
  alone — so it is directly comparable against the committed 2026-08-08 runs.
- `held_predicates` / `applicable_skills`, which rebuild each run's **final recorded
  state** from its own `sampler_draws.jsonl` and evaluate it with the domain's *own*
  classifiers and the *actual* `Tossing3DSkills` preconditions, rather than a
  re-transcription of either. A draw records the features of the objects its ground skill
  binds; merging forward over the file is exact here rather than convenient, because the
  robot and cube are bound by all three skills and the barrier and bin are immovable within
  an episode — and a reset-free run is one episode end to end.

`toss_transition_index` is kept, still raising, rather than relaxed. Naming an event that
did not happen is worse than refusing to.

## Results

### The collapse reproduces, and the gap is slightly wider

| measure | `scheduled` | `never` |
| --- | --- | --- |
| **`LATE` window (last 10 sweeps)** | **`73.4/100`** | **`12.2/100`** |
| seeds where `never` is lower | — | `10/10` |
| mean per-seed difference | — | `-6.12` tasks |
| exact paired sign-flip | — | **`p = 0.001953`** |
| MDE at 80% power | — | `0.98` tasks per seed |

`p = 0.001953` is `2/1024`, the smallest two-sided value an exact sign-flip can return at
ten paired seeds — every seed moved the same way, so the test is saturated. The observed
effect (`6.12`) is more than six times the MDE (`0.98`).

**Both arms' absolute levels moved**, which is expected and is not evidence of a defect:
the throw itself changed. Against the 2026-08-08 page's published `80.8/100` and
`24.9/100` — quoted, not recomputed — `scheduled` is lower and `never` is lower still.
`never` now sits **below** the `24/100` `random-skills` reference from #133 on `10/10`
seeds (its best seed is `2.3/10`, against that line's `2.4/10`). **That is a description of
two numbers, not a test**: #133 ran 20 cycles and predates #160's separate evaluation
`Problem`, so the two were not produced under the same conditions, and nothing here is
claimed about the comparison beyond the arithmetic.

![Two learning curves against practice cycle, ten faint per-seed lines under a bold pooled mean for each arm. The scheduled arm in blue climbs from about 1 to about 7 out of 10 over the first sixty cycles and drifts slowly upward after; the reset-free arm in orange stays flat between 1 and 1.5 for the whole hundred cycles, below the dotted random-skills reference line at 2.4 and far below the dotted skill-oracle line at 10.](2026-08-13-tossing3d-reset-policy-new-pin-curves.png)

**Figure 1. The reset-free arm never leaves the floor.** Bold pooled mean over faint
per-seed lines, both arms on the same axes. Neither arm splits into subgroups — every seed
within an arm behaves the same way — so each gets one bold line, and the legend carries
`n=10` for both. `skill-oracle` (`100/100`) and `random-skills` (`24/100`), both from #133,
are drawn as **reference lines rather than curves**: neither learns, so a curve would
invite a reader to look for a trend in a constant.

![Ten lines, one per seed, joining each seed's mean score over the last ten sweeps under the scheduled policy to the same quantity under the reset-free policy. Every line slopes steeply down, from a cluster between 5.8 and 8.5 out of 10 to a cluster between 0.4 and 2.3, all of them ending below the dotted random-skills reference line at 2.4.](2026-08-13-tossing3d-reset-policy-new-pin-paired.png)

**Figure 2. Every seed falls, and every seed lands on the floor.** Plotted per seed rather
than as two bars because with ten seeds a bar chart of two means hides one seed driving the
whole movement — here nothing is hidden, all ten move the same way. Seed 7 (`5.8 → 1.8`) is
the mildest and seed 3 (`8.5 → 0.4`) the steepest.

### It still stops practising rather than practising badly

| measure | `scheduled` | `never` |
| --- | --- | --- |
| total practice transitions (10 seeds × 100 cycles) | `3069` | **`35`** |
| seeds ever stranded | `0/10` | **`10/10`** |
| stranding onset | — | **cycle `1` on every seed** |
| practice cycles taking zero steps | `0/1000` | **`990/1000`** |

`10/10` seeds stranded from cycle `1`, and `990/1000` idle cycles, are **identical** to the
2026-08-08 page. What moved is the size of the single practice period that does happen:
`35` transitions across all ten seeds, against the `77` published at the old pins, ranging
`1`–`9` per seed against `3`–`13`. Fewer `MoveToThrowPose` draws are now needed before one
succeeds, and two seeds get only a single transition before the world closes.

![Cumulative practice transitions against practice cycle, one line per seed, left panel. The ten scheduled lines rise steadily and linearly to between 296 and 338 by cycle 100. The ten reset-free lines rise only in the first cycle, to between 1 and 9, and are then perfectly flat for the remaining ninety-nine cycles. A shaded region and dashed vertical line mark cycle 1 onward as stranded (0 transitions per cycle, 10/10 never-arm seeds), and a marker on seed 0's curve is labelled "seed 0: last action (Pick) at transition 1". A small right panel plots every never-arm seed's last practice action as a dot strip against transition index, each row labelled with the skills that cycle attempted -- six rows reading MoveToThrowPose+Pick+Toss, two reading MoveToThrowPose+Pick, and two reading Pick alone.](2026-08-13-tossing3d-reset-policy-new-pin-practice.png)

**Figure 3. Ten flat lines, and what each one stopped doing.** A robot that keeps
practising is a line that keeps rising; a stranded one goes flat and stays flat. The
annotation is derived from the per-cycle transition and skill-attempt record, never from
where a line visually goes flat. **The right panel is where the mechanism change is
visible**: it used to read `Toss` on every row.

Two claims sit on this figure and should not be conflated. *Every* never-arm seed strands
at **cycle 1** — the shaded region, uniform across all ten. The **transition index** at
which it stops is not uniform: `1`–`9`, median `3.0`, because it depends on how many draws
the sampler needed. Seed 0's marked value, `1`, is one seed's number.

### The mechanism: two routes into the same dead end

**`6/10` seeds strand the published way.** Their last active cycle attempted a `Toss`. On
`5/6` of them (1, 4, 5, 6, 9) the cube ended past the barrier, so `Reachable` is false; on
the sixth (7) the cube stayed on the robot's side but is not `OnGround`. Either way `Pick`
cannot fire, and with `Holding` gone neither can the other two.

**`4/10` seeds (0, 2, 3, 8) never attempt a `Toss` at all.** All four end with the gripper
**closed on nothing**: `pos_gripper = 1.0`, so `HandEmpty` is false, while the cube is on
the floor (`z ≈ 0.025` against a `bb_z` of `0.05`) and therefore below `HOLDING_HEIGHT`, so
`Holding` is false too. The cube is still on the robot's own side of the barrier —
`Reachable` holds on all four — and it is still physically retrievable. The robot simply
has no operator whose preconditions are met, because `Pick` is the only skill that starts
from an empty hand and its hand is not empty.

| route | seeds | last cycle attempted | why nothing is applicable |
| --- | --- | --- | --- |
| ended on a `Toss` | `6/10` — 1, 4, 5, 6, 7, 9 | `Pick`+`MoveToThrowPose`+`Toss` | `Reachable` gone (`5/6`) or `OnGround` gone (`1/6`); `Holding` gone on all six |
| never threw | `4/10` — 0, 2, 3, 8 | `Pick` (2), `Pick`+`MoveToThrowPose` (2) | `HandEmpty` **and** `Holding` both false — shut gripper, cube on the floor |
| **no skill applicable** | **`10/10`** | — | — |

![Two panels of dot matrices, one row per never-arm seed, ordered with the six that ended on a Toss below the four that never threw. The left panel ticks which of HandEmpty, Holding, OnGround, Reachable and RobotAtSuccessfulThrowPose hold in each run's final recorded state; every seed that ended on a Toss has HandEmpty ticked and Holding crossed, while every seed that never threw has both HandEmpty and Holding crossed and Reachable ticked. The right panel shows Pick, MoveToThrowPose and Toss, and every cell in it is crossed for all ten seeds.](2026-08-13-tossing3d-reset-policy-new-pin-stranding.png)

**Figure 4. Two routes, one dead end.** Left, the predicates that hold in each run's final
recorded state; right, the skills whose preconditions are therefore satisfied — nothing, on
every seed. The `HandEmpty` column is the whole story: ticked on every seed that threw,
crossed on every seed that did not. Evaluated with the domain's own classifiers and the
actual operator preconditions, on features the runs themselves recorded.

### Two facts, and only one of them is about the domain

The 2026-08-08 page drew this line and it still holds, with one half now wider.

**The domain fact.** After a `Toss`, no skill in this domain is applicable, and none
retrieves the cube. That is checked against the operators and is method-independent. **What
this sweep adds is that it is not the only such state.** A shut gripper with the cube on the
floor is a second dead end reachable without any irreversible action at all — the
irreversibility is in the *symbol layer*, not only in the barrier. Nothing in `Pick`,
`MoveToThrowPose` or `Toss` opens a gripper that closed on nothing.

**The method fact.** `990/1000` idle cycles is EES's particular denomination of that: its
planner finds no applicable skill and ends the period immediately, so the cost shows up as
a zero. A method that acted regardless would burn its full `--max-steps-per-interaction`
learning exactly as little while recording thousands of transitions.

### Per seed

| seed | `scheduled` `LATE` | `never` `LATE` | difference | `never` transitions | route |
| --- | --- | --- | --- | --- | --- |
| 0 | 6.1 | 0.9 | `-5.2` | 1 | never threw |
| 1 | 7.3 | 1.7 | `-5.6` | 9 | ended on `Toss` |
| 2 | 7.4 | 0.9 | `-6.5` | 2 | never threw |
| 3 | 8.5 | 0.4 | `-8.1` | 2 | never threw |
| 4 | 7.7 | 1.5 | `-6.2` | 5 | ended on `Toss` |
| 5 | 7.1 | 0.9 | `-6.2` | 3 | ended on `Toss` |
| 6 | 7.8 | 0.7 | `-7.1` | 3 | ended on `Toss` |
| 7 | 5.8 | 1.8 | `-4.0` | 4 | ended on `Toss` |
| 8 | 7.7 | 1.1 | `-6.6` | 1 | never threw |
| 9 | 8.0 | 2.3 | `-5.7` | 5 | ended on `Toss` |

Every never-arm seed took `99/100` idle cycles.

### What this does *not* establish

- **It does not decompose no-reset from no-scene-variety.** They are inseparable on this
  domain by construction, so `-6.12` is a number about both together. The stranding is so
  total that scene variety barely gets a chance to matter — but that is an argument from
  the mechanism data, not a decomposition.
- **It does not explain *why* `Pick` now fails on seeds 0 and 8, or why the cube is dropped
  on seeds 2 and 3.** The runs record that it happened, not the contact dynamics behind it.
  Attributing the shut-gripper deadlock specifically to the pin bump would need the same
  sweep at the old pins, which was not run. What *is* established from the committed
  2026-08-08 runs is that the deadlock did not occur there: `10/10` of those seeds ended
  their last active cycle on a `Toss`, checked from their own `stats.json` rather than from
  that page's prose.
- **It does not measure "the value of a reset" in general** — only on a domain chosen for
  having irreversible actions.
- **It is not a claim that reset-free practice is unworkable**, only that reset-free
  practice *without any means of recovery* is, here.

### Cost

`analysis/run_timing.py`, per arm, at 20 concurrent runs on 24 cores: `scheduled` median
`6795.8 s` per run (`6908.6 s` wall for the arm), `never` median `5235.3 s` (`5835.1 s`
wall). The reset-free arm is cheaper only because its practice periods are empty; it still
pays for all 101 evaluation sweeps. `run_timing.py` globs `*/*/timing.json`, so it must be
pointed at each arm directory rather than at the sweep root, which is one level deeper here.

**No video.** The practice period is what strands, and nothing renders it: `--record-full-loop`
was off, and the `episode*.mp4` files these runs wrote are *evaluation* episodes. A clip of
the shut-gripper deadlock would need a re-run, which this task deliberately did not do.

## Recommendation

**Keep citing the 2026-08-08 page as the motivating measurement, and cite this one for the
mechanism.** The headline claim — a robot practising without a free reset on an irreversible
domain takes a couple of actions and then stops for ninety-nine cycles — is now measured
twice, at two different sets of dynamics, with the same `10/10` and the same `990/1000`.
That is about as robust as a ten-seed result on this domain gets.

**Restate the durable claim slightly wider than the old page did.** "After one toss, no
skill in this domain is applicable" is true and still the cleanest example, but it is a
special case of "this domain has dead-end states the robot cannot leave, and it reaches one
within a couple of actions". The shut-gripper deadlock reaches the same place with no
irreversible physical act at all, which is a *stronger* argument for a human in the loop,
not a weaker one: even a robot that never throws anything away still needs rescuing.

Three follow-ups, in the order the evidence argues for them:

1. **Point the `HumanOracle` ladder at this domain.** It is the obvious next experiment and
   this re-measurement removes the last reason to wait — the target is not a one-pin
   artefact. The gap a human would close is the whole `73.4` versus `12.2`.
2. **Decide whether the shut-gripper deadlock is a domain defect or a domain feature.**
   If a real TidyBot could open its gripper and try again, the skill set is missing an
   operator and `4/10` of this sweep's stranding is an artefact of our own modelling. If it
   could not, this is the domain working as intended. That question is worth one person-hour
   before any more Tossing3D experiments are read.
3. **Do not re-run this for a tighter p-value.** `p = 0.001953` is the floor at ten paired
   seeds and every seed moved the same way.

## Artifacts

- Runs, both arms, 10 seeds each:
  [`2026-08-13-tossing3d-reset-policy-new-pin-runs/`](2026-08-13-tossing3d-reset-policy-new-pin-runs)
  (`stats.json`, `config_snapshot.json` and `timing.json` per run, plus
  `sampler_draws.jsonl` for the `never` arm, which is what the stranding analysis reads
  back — 1–9 lines per run there against ~300 per `scheduled` run, which is why only one
  arm's is carried. `log.txt` is deliberately not committed: machine-specific noise.)
- Analysis: `analysis/practice_makes_perfect/tossing3d_reset_free_arms.py`, covered by
  `tests/analysis/practice_makes_perfect/test_tossing3d_reset_free_arms.py`.
- Figures: [curves](2026-08-13-tossing3d-reset-policy-new-pin-curves.png),
  [paired](2026-08-13-tossing3d-reset-policy-new-pin-paired.png),
  [practice](2026-08-13-tossing3d-reset-policy-new-pin-practice.png),
  [stranding](2026-08-13-tossing3d-reset-policy-new-pin-stranding.png).

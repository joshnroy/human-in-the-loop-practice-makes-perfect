# Tossing3D: EES against the random-skills floor

**Status: a reproduction result.** It says what EES's learning curve looks like on a
KINDER domain whose success is irreversible. It does **not** test the irreversibility
hypothesis, and cannot — see [What this cannot tell us](#what-this-cannot-tell-us),
which is the section to read before any number below.

## What was run

Ten fixed seeds (0..9) per arm, both arms through `scripts/run_sweep.py` so the
`(method x seed)` grid and the `<root>/<method>/<seed>/` layout are the ones
`analysis/` globs for. Seeds are shared across arms, so **every comparison here is
paired** and is tested as such.

```bash
python -m scripts.run_sweep \
  --env tossing3d --methods ees random-skills --num-seeds 10 \
  --results-root <root> --max-workers 12 \
  --shared-args "--num-cycles 10 --max-steps-per-interaction 150"
```

The protocol flags match the Tossing Room bring-up (`2026-08-02-tossingroom-ees-bringup.md`)
so the two domains' curves are read on the same axes: 10 cycles x 150 steps = 1500
online transitions, evaluated on the default 10 held-out test tasks before practice and
after each cycle.

Analysis is post-run only, off the written `stats.json`:

```bash
python -m analysis.practice_makes_perfect.ees \
  --results-root <root> --output docs/experiment-logs/2026-08-04-tossing3d-ees-curves.png
python -m analysis.practice_makes_perfect.tossing3d_comparison --results-root <root>
```

**Every success rate below is the count of evaluation episodes solved**, `x/y`, with a
percentage alongside only where it aids reading. The counts are not derived from the
percentages — they are what `Metrics` recorded in the first place: `record_evaluation`
writes `(num_online_transitions, num_solved, num_total)` triples, and those triples are
committed here as
[`2026-08-04-tossing3d-arms.json`](2026-08-04-tossing3d-arms.json) — every seed of
both arms at all 11 checkpoints, copied verbatim out of the runs' own `stats.json`, so
each number in this log can be re-derived from a file in this repo rather than from a
results directory that no longer exists.

*Differences* of rates stay in percentage points throughout — paired differences, sds
and gaps are not counts of anything, and no denominator is invented for them.

## Running it at all: the integration needed a fix first

The committed integration was green on everything CI runs, but CI does not have
`kindergarden`, so the eight tests that genuinely drive MuJoCo were skipping. Run with
the optional dependency installed, two things showed up.

**A memory leak that made a sweep impossible.** A 40-step run reached 18.7 GB RSS and
was still climbing at ~112 MB/s. The cause is a seam between two reasonable-looking
pieces: KINDER's `LiftedParameterizedController.ground` is
`return self.controller_cls(objects)`, so grounding mints a *new* controller per call,
and each new controller's `reset` stands up its own `PyBulletSim` — a
`p.connect(p.DIRECT)` plus the Kinova URDF and meshes — which nothing on KINDER's side
ever disconnects. `KinderBackend` already cached the *lifted* controllers to avoid
exactly this and its docstring named the failure mode, but that cache cannot help, since
the leak is per *grounding*. Measured at ~150 MB per `Pick` and ~315 MB per `Toss`;
resets were never the problem (50 of them are flat).

Fixed by releasing the client in `_run`'s `finally`. Memoizing the grounding instead was
tried first and is wrong — `PyBulletSim` carries held-object state `reset` does not
clear, so every `Pick` after the first silently fails.

| | 60 skill executions | 600 skill executions |
| --- | --- | --- |
| before | 8.4 GB, still climbing | (would not survive) |
| after | ~0.66 GB | ~0.66 GB (+122 MB total, all warmup) |

Releasing only frees a resource, so the physics are unchanged: seeds 0 and 2 reproduce
the pre-fix landing positions to three decimals, including the swing = 1.0 overshoot to
x = 2.216.

### The goal region was wrong, and these numbers are the re-run

The first version of this experiment scored success against the wrong box, and every
number in it understated the truth. Recording it here because the correction is the
reason this log's results changed, not a footnote to them.

`blocks_goal_region`'s task JSON reads `ranges[0] = [1.9, -0.1, 0.0, 2.1, 0.1, 0.1]`,
and this domain's `InGoalRegion` predicate tested the cube against that literal. **KINDER
never compares anything against it.** `MujocoGround._create_regions`
(`envs/dynamic3d/objects/base.py:874-881`) inflates the range by
`ground_placement_threshold = 0.05` (`base.py:840`) on every side, clamping z at 0, and
stores the result as `Region.bbox` — which is what `Region.check_in_region`
(`base.py:148-185`) does its inclusive per-axis test on, and what `_check_goals`
(`envs.py:1053-1167`) therefore decides success by.

| | x | y | z |
| --- | --- | --- | --- |
| what we tested | [1.90, 2.10] | [-0.10, 0.10] | [0.00, 0.10] |
| **what KINDER tests** | **[1.85, 2.15]** | **[-0.15, 0.15]** | **[0.00, 0.15]** |

Confirmed against the live simulator on both variants, and by both of `Region.bbox`'s
code paths (the MuJoCo-site read and the XML fallback) agreeing. Our box was 2/3 of the
true width on x — the axis a toss controls — so the error was systematic and
one-directional: every landing in the two 5 cm shells was a KINDER success scored here
as a failure.

`goal_region_bounds()` now reads `Region.bbox` back from upstream rather than
re-deriving the inflation, so the two cannot drift apart again, and a fidelity test pins
them element-wise. The old test suite could not have caught this: its property test
walked 12 random states, and a random walk of whole skills essentially never lands the
cube in a 5 cm boundary shell. Tests that deliberately probe those shells were added
alongside the fix.

**This required a re-run, not a rescore.** The predicate is the goal atom
(`tasks.py:66-69`) and a `Toss` add-effect (`skills.py:73`), and `run_task_episode`
returns early on `is_satisfied` — so the corrected box changes which episodes get run,
what EES's competence signal sees, and how long episodes last, not merely how finished
trajectories are scored. Both arms were re-run from scratch in a single sweep. Every
number below is from that re-run; the superseded figures are given as *was → now* where
the comparison is informative.

**One genuinely red fidelity test.**
`test_a_full_power_toss_overshoots_the_goal_region` asserted the overshoot on seed 1 —
the one seed where it does not happen. There the grasp is marginal and the cube slips
out during `move_to_target`, landing at x ~ 1.58 having never been tossed, so the test
was measuring the `Pick`, not the swing. It now asserts on seeds 0 and 2, the same seed
pair `test_the_oracle_swing_actually_reaches_the_goal_region` was already tolerating by
asserting 2 of 3 rather than 3 of 3.

### The swing dial, measured

Landing x after the oracle's Pick and MoveToThrowPose, per swing. These are landing
positions — pure physics — so they are unchanged by the goal-region correction below;
only which of them count as solved changes.

The goal region is x in [1.85, 2.15] (see [the goal-region
correction](#the-goal-region-was-wrong-and-these-numbers-are-the-re-run)). The bin's
footprint starts at 2.08, so the region and the bin **overlap** on x in [2.08, 2.15]:
landing in the bin is not itself a failure. What makes the dial worth learning is that
the extremes miss on both sides — a weak swing drops short, and KINDER's own demo toss
(swing = 1.0) sails out to 2.22, past the region's far edge.

| swing | 0.25 | 0.50 | 0.60 | 0.75 | 0.90 | 1.00 |
| --- | --- | --- | --- | --- | --- | --- |
| seed 0 | 1.418 | 1.657 | **1.990** | **1.914** | **2.015** | 2.216 |
| seed 2 | 1.424 | 1.656 | **1.989** | **1.915** | **2.014** | 2.216 |

Bold = inside the goal region. **No sampled swing changes verdict under the corrected
box** — 1.657 is short of 1.85 and 2.216 is past 2.15 — so this table reads identically
either way, and `ORACLE_SWING = 0.75` was re-checked rather than retuned.

That also means the sampled points only bracket the solving band to somewhere inside
(0.50, 1.00); the finer `[0.57, 0.93]` figure previously quoted was interpolation
against the narrow box, not a measurement, and is not restated. Roughly half the
sampler's `[0.25, 1.25]` prior lands in the region — which is why this domain's
**pre-practice** checkpoint is not near zero, and why EES's own starting point, not just
the random-skills floor, is the reference that matters.

Seed 1 is absent because its grasp is marginal and it drops the cube during the move for
every swing, so it measures the Pick.

## Results

![EES vs the random-skills floor on Tossing3D](2026-08-04-tossing3d-ees-curves.png)

Fraction of the 10 held-out evaluation tasks solved vs online transitions; solid = mean
over seeds 0..9, shading = standard error. The paper's Figure 4 view.

| arm | seeds | pre-practice | end of training | sd | worst seed | best seed |
| --- | --- | --- | --- | --- | --- | --- |
| **EES** | 10 | 33/100 (33.0%) | **67/100** (67.0%) | 22.1 | 4/10 | 10/10 |
| random skills | 10 | 14/100 (14.0%) | 21/100 (21.0%) | 7.4 | 1/10 | 3/10 |

Each arm evaluates 10 held-out tasks per seed on 10 seeds, so the arm column is a
genuine pooled count of 100 evaluation episodes, not a mean of rates given a
denominator after the fact. The `sd` is the spread of the ten per-seed rates in points
— it is *not* a binomial spread on the pooled count, and should not be read as one.

Per-seed end-of-training, EES: `[7/10, 6/10, 9/10, 8/10, 4/10, 5/10, 5/10, 4/10, 10/10, 9/10]`
— as percentages, `[70, 60, 90, 80, 40, 50, 50, 40, 100, 90]`.
Per-seed end-of-training, random skills: `[3/10, 2/10, 3/10, 1/10, 2/10, 2/10, 2/10, 3/10, 1/10, 2/10]`
— as percentages, `[30, 20, 30, 10, 20, 20, 20, 30, 10, 20]`.

Arms share seeds 0..9, so both comparisons are paired and are tested with the exact
Wilcoxon signed-rank (all 2^n sign assignments enumerated).

| comparison | mean paired difference | test | verdict |
| --- | --- | --- | --- |
| EES end vs **its own pre-practice checkpoint** | +34.0 pp (sd 23.2) | p = 0.0039, n = 9 non-tied | **established** |
| EES end vs **random skills** end | +46.0 pp (sd 25.5) | p = 0.0020, n = 10 non-tied | **established** |
| random skills end vs its own start | +7.0 pp (sd 12.5) | p = 0.1094, n = 7 non-tied | **not established** — 26 paired seeds would be needed for 80% power |

So EES learns on this domain: it roughly doubles its own untrained success rate, and
ends three times the non-learning floor. The floor itself is flat within noise across
the whole budget (18/100 – 26/100 at every checkpoint after the first), which is what a
floor should look like. *(This range previously read "21–26%", which understated it: the
900-transition checkpoint is 18/100, as the checkpoint table below has always recorded.
Corrected in place rather than deleted. The reading is unchanged — at p ≈ 0.22 on 100
draws the binomial sd is ~4.1 points, so 18–26 sits inside ±1 sd and "flat within noise"
stands, as does the separately-reported +7.0 pp rise being **not established**.)*

### The mean dips before it climbs — but the dip is not established

The *mean* curve is not monotone: it falls from 33% pre-practice to 20% at 300
transitions before climbing to 67%. Tempting as "EES gets worse before it gets better"
is, **that decrease does not survive a test** and is not claimed here:

| comparison | mean paired difference | test | verdict |
| --- | --- | --- | --- |
| EES worst post-practice checkpoint (300) vs pre-practice | −13.0 pp (sd 20.6) | p = 0.1328, n = 8 non-tied | **not established** — 20 paired seeds would be needed for 80% power |

Per-seed differences: `[-30, -40, 0, 0, -10, -10, +10, -10, -50, +10]` — seven seeds at
or below zero, but two go up and the spread swamps the mean. The 300-transition
checkpoint was also chosen *post hoc*, as the worst of ten by mean, which inflates
significance rather than deflating it; a p of 0.13 under that selection is weaker than
it looks, not stronger.

So: the dip is a feature of this sample's mean, not a demonstrated effect.
`tossing3d_comparison.py` runs this test on every invocation so the claim cannot drift
back into being asserted.

The checkpoints, for reference — evaluation episodes solved out of the 100 run at each
(10 held-out tasks x 10 seeds):

| transitions | 0 | 150 | 300 | 450 | 600 | 750 | 900 | 1050 | 1200 | 1350 | 1500 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EES | 33/100 | 21/100 | 20/100 | 25/100 | 49/100 | 40/100 | 64/100 | 58/100 | 59/100 | 61/100 | **67/100** |
| random skills | 14/100 | 22/100 | 23/100 | 24/100 | 22/100 | 24/100 | 18/100 | 26/100 | 22/100 | 21/100 | 21/100 |

Each denominator is 100 because every seed contributes the same 10 tasks; the pooled
rate and the mean of the ten per-seed rates therefore coincide exactly, and the plotted
percentages above are these counts over 100.

This still matters for reading short runs, whether or not the dip is real: a pilot at 90
transitions showed 6/10 -> 1/10 and looked like EES actively degrading. Whatever that
was, it was not the end state — the same configuration run to 1500 transitions reaches
67%. Do not read a few hundred transitions on this domain as a trend in either
direction.

The pre-practice checkpoint is high here for a structural reason — EES *plans* the
correct skill sequence from the start, and only the sampler is untrained, so its
starting point already inherits the ~36% of the swing prior that lands in the goal
region (see the swing table above). Random skills has to discover the sequence too,
which is why it starts at 14%. That is also why the two comparisons in the table above
answer different questions and both are reported.

A tempting mechanism for the dip — early practice data is mostly failure, so the
refitted sampler is briefly worse than the uniform prior it replaced — is **not tested
here**, and since the dip itself is not established, nothing in this log should be read
as evidence for it.

## The progression, at four checkpoints

`--num-render-checkpoints 4` on seed 0, recorded from before any practice through the
end of training. The renderer is a storyboard — one labelled frame per skill, not a
smooth video — so each frame names the ground skill, its sampled continuous parameters,
and the cube's resulting x against the goal region.

| transitions | what the demo episode does |
| --- | --- |
| **0** | `Pick(params=[0.59, -0.37])` fails outright; the cube never leaves the floor at x = 0.48. Runs the full horizon. |
| **450** | Tossed clean past everything to x = 2.86, then `no-op (no plan)`. The cube is unreachable and nothing recovers it. |
| **1050** | `Toss(params=[0.89])` → cube at x = **2.01**, inside [1.90, 2.10]. Solved in the minimum 3 skills. |
| **1500** | `Toss(params=[0.84])` → cube at x = **2.00**. Solved in 3. |

![0 transitions — the Pick fails](2026-08-04-tossing3d-ees-000000.gif)
![450 transitions — overshoots to 2.86, then no plan](2026-08-04-tossing3d-ees-000450.gif)
![1050 transitions — swing 0.89, cube at 2.01](2026-08-04-tossing3d-ees-001050.gif)
![1500 transitions — swing 0.84, cube at 2.00](2026-08-04-tossing3d-ees-001500.gif)

Two things worth reading off these rather than off the curve.

**The swing converges into the measured band.** 0.89 and 0.84 both sit inside the
[0.57, 0.93] band the swing table above establishes, and the trained episodes finish in
3 skills where the untrained ones run the horizon out.

**The 450-transition frame is the irreversibility, on screen.** The cube is at x = 2.86,
past the goal region and past the bin, and the label is literally `no-op (no plan)` —
there is no skill sequence back. This is the failure mode the domain exists to exhibit.
Note what happens next anyway: the episode ends, the harness resets for free, and the
run recovers to 90% by 1500. That is exactly the confound described below — the picture
shows the irreversibility, and the experiment still cannot measure its cost.

**These GIFs are from a separate run, not the seed-0 run in the curve.** Its trajectory
differs from the sweep's seed 0 (`[6,3,1,1,7,7,7,7,8,9,9]` vs `[6,3,3,4,7,7,7,8,8,9,7]`
solved-out-of-10 per checkpoint), so they illustrate the behaviour rather than being the
data behind the curve.

**The likely cause is thread count, not the render flag.** This run also invoked
`hitl_pmp.cli` directly, so it missed the `OMP_NUM_THREADS=1` that `run_sweep` pins on
every child — and the section above establishes that this alone moves a Tossing3D run
substantially, since the invalid first re-run check diverged for exactly that reason and
became IDENTICAL once the pinning matched. So `--num-render-checkpoints` is *not* shown
to perturb anything; the one variable known to differ here already explains it. A
matched pair of `run_sweep` invocations differing only in the render flag would confirm
that, and is not done here.

They are also ~60–100 KB rather than the ~25 KB of the Tossing Room precedent, using the
same shared-64-colour-palette recipe. A 640x528 3D render with smooth shading simply has
more entropy than a 2D storyboard; 32 colours only buys ~15%.

## Can the GPU help? Rendering already uses it; nothing else can

The box has an idle RTX 5090, and a Tossing3D run costs ~1.1 s per transition, so it is
a fair question. Measured, per subsystem — **no change is warranted**:

| subsystem | GPU path? | measured | verdict |
| --- | --- | --- | --- |
| MuJoCo physics | **none available** | 8.0 ms per env step = 200 substeps x 0.040 ms | dominant cost, and unreachable |
| Rendering | **already on GPU** | `GL_RENDERER = NVIDIA GeForce RTX 5090/PCIe/SSE2`, 1.20 ms/frame | nothing to do |
| PyBullet IK / motion planning | none | `p.connect(p.DIRECT)` is headless CPU by design | unreachable |
| EES sampler MLP (torch) | possible, unwise | 44 ms per 200 full-batch steps, 1 thread | would be slower, and would break reproducibility |

**Physics is the cost and has no GPU path.** GPU MuJoCo means MJX (JAX-backed), and
`grep` finds no `mjx` and no `jax` anywhere in `kindergarden` or `kinder-models` — it is
plain `mujoco.mj_step`. KINDER steps `SIMULATION_TIMESTEP = 0.0005 s` at
`control_frequency = 10 Hz`, i.e. **200 substeps per env step**
(`envs/dynamic3d/mujoco_utils.py:149`), so the 8.0 ms is 0.040 ms per substep — normal
full-speed CPU MuJoCo, not something running slowly. Getting this on the GPU would mean
porting Tossing3D to MJX, and MJX's advantage is *thousands of environments stepped in
parallel*, not single-environment latency; for one sequential env it is typically slower
than the C backend. Not worth it for this config, and it would be a fidelity risk on a
domain whose whole point is running the benchmark's own simulator unmodified.

**Rendering was the one real risk and it is already fine.** `MUJOCO_GL=egl` could
silently fall back to a software rasteriser, which is a classic cause of render-bound
runs; it has not — the GL renderer string names the 5090. At 1.20 ms/frame against a
storyboard of a handful of frames per recorded episode, it is not a measurable part of a
sweep anyway.

**The sampler must stay on CPU.** It is a 32x32 MLP over a few hundred rows; at that
size transfer overhead dominates and the GPU loses. More importantly this repo verifies
changes by byte-identical `stats.json`, and the section below is direct evidence that
perturbing torch's numerics changes Tossing3D results *substantially* — moving the
sampler to CUDA would do exactly that. (Incidentally the KINDER venv's torch is
`2.13.0+cpu`, a CPU-only build, so the sweep never had a GPU option in the first place.)

## Was the sweep contaminated by its own concurrency?

The sweep ran 12 concurrent runs on a 24-core box. The thing that could break under that
load is Fast Downward's **10 s wall-clock timeout** — a starved FD would return no plan,
and a run's curve would then be measuring timeouts rather than learning.

**This cannot be checked the obvious way.** `ees_method.py` swallows `PlanningFailure`
at all three call sites without logging it, so a timeout and a genuinely unreachable
goal look identical, and neither appears in `log.txt`. Grepping the per-run logs for a
planning-failure rate is simply not a check this harness supports today.

What the evidence actually says, in descending order of strength:

- **Cross-agent, on this same box** (reset-frequency work, Tossing Room): 13,105 live FD
  process observations, **maximum lifetime 0 s** against the 10 s budget, at loads from
  9 to 27; not one observation reached even 5 s. And `stats.json` byte-identical between
  a low-load probe and a 20-way concurrent sweep on both extreme arms. A spurious
  timeout would necessarily change the plan and hence the trajectory, so identity rules
  one out. This is the strongest evidence available and it says concurrency is safe.
- **This domain, low concurrency:** two identical `run_sweep --max-workers 1`
  invocations produced bit-identical `stats.json`. The pipeline is deterministic.
- **This domain, directly: the re-run diff came back IDENTICAL.** EES seed 0 re-run
  alone through `run_sweep --max-workers 1` reproduced its 12-way sweep result exactly:

  ```text
  sweep    (12-way concurrent, OMP=1): [6, 3, 3, 4, 7, 7, 7, 8, 8, 9, 7]
  recheck3 (1 worker,          OMP=1): [6, 3, 3, 4, 7, 7, 7, 8, 8, 9, 7]
  ```

  A starved FD would have returned a different plan and hence a different trajectory, so
  this rules out timeout contamination for this seed.

**A wrong turn worth recording, because it nearly became a finding.** The *first* attempt
at that check returned DIFFERS, and substantially (`[6,1,4,1,2,1,0,6,7,7,7]`). It was
invalid: it invoked `hitl_pmp.cli` **directly**, while `run_sweep` pins
`OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` on every child (`run_sweep.py:153`). So it
compared a many-thread run against a one-thread run and would have reported "concurrency
perturbs Tossing3D results" when the actual variable was torch's thread count. Re-running
through `run_sweep` — identical in every respect except load — gave the IDENTICAL above.

Two things follow. Any future re-run comparison on this repo must go through
`run_sweep`, not the CLI, or it is not measuring what it thinks. And **the sampler's
numerics are thread-count dependent**, so a `--seed` determines a run only at a fixed
thread count; that is worth knowing independently of this experiment.

**Bottom line for this log's numbers.** Concurrency did not contaminate the sweep. And
the arm-level comparison would not have been at risk either way, since both arms ran
interleaved inside the *same* sweep under the same load, so any load effect would apply
equally to both and the paired differences would stand regardless.

## What this cannot tell us

Tossing3D's defining property is that **a tossed cube cannot be retrieved** — success or
miss, it ends up past an immovable barrier and no skill brings it back. The project's V1
proposal names exactly this as EES's predicted failure mode: *"when the goal is reached
(ex: Tossing 3d), it can't reset, even though it did everything right."*

**The measurement above cannot speak to that hypothesis in either direction**, for two
concrete reasons.

1. **The harness hands out a free reset.** `PracticeLoop.run` calls
   `problem.reset_to_task(task=task)` at the top of every practice cycle, and
   `run_task_episode` resets per evaluation episode. So the environment is restored for
   free every `--max-steps-per-interaction` steps. That is faithful to predicators and
   correct for a reproduction — and it supplies precisely the free resets the
   irreversibility hypothesis is about removing. With that reset in place Tossing3D does
   not stall, so a clean curve here is evidence about EES's sampler learning, not about
   irreversibility.
2. **Human intervention is not representable.** `Metrics.num_human_interventions()`
   returns a hardcoded `(0.0, 0)` because no `Method` in this repo ever calls
   `Problem.execute_human_command`. The cost that the hypothesis predicts irreversibility
   imposes therefore has nowhere to be recorded, whatever the environment does.

This is not a hypothetical concern: the same thing already happened on Tossing Room,
where the RECYCLING family makes a missed throw terminal and EES still learned 18% ->
92%.

Changing the reset behaviour is a separate design decision and belongs in a separate PR.
It is deliberately **not** done here — this log establishes the reproduction baseline
that such a change would be measured against.

## Limitations

- **Per-seed numbers are machine-local.** Compare at arm level. Two identical
  `run_sweep` invocations on this box produced bit-identical `stats.json`, so `--seed`
  fully determines a run *here*; that is a determinism claim, not a portability one.
- **The planning-failure rate cannot be read out of the run logs at all.**
  `ees_method.py` catches `PlanningFailure` silently at all three of its call sites
  (lines 386, 778, 802) and never logs it, so a Fast Downward *timeout* is
  indistinguishable from a goal that is genuinely unreachable — neither reaches
  `log.txt`. Grepping the per-run logs for planning failures, the obvious way to check
  whether 12-way concurrency starved FD against its 10 s budget, is therefore not a
  check that can be performed. That is a gap in the harness's observability, not
  something specific to this domain.

  What was substituted: re-running a seed alone and diffing `stats.json`, since a
  timeout would necessarily change the plan and therefore the trajectory. That came back
  IDENTICAL — see
  [Was the sweep contaminated by its own concurrency?](#was-the-sweep-contaminated-by-its-own-concurrency)
  below, including the first, invalid attempt at it.
- **A `--seed` determines a run only at a fixed thread count.** Established accidentally
  (see the same section): the same seed run with and without `OMP_NUM_THREADS=1` diverges
  substantially. `run_sweep` pins it on every child, so sweeps are consistent, but a run
  driven straight through `hitl_pmp.cli` is not comparable to one from a sweep.
- **10 evaluation tasks per checkpoint**, so a single seed's curve moves in 10-point
  steps. The arm-level mean over 10 seeds is the readable quantity.
- **`o2` is not supported** (two cubes; the symbolic layer here is single-cube).
- **The goal is KINDER's `blocks_goal_region` verbatim**, not "in the bin" — and
  "verbatim" means the region KINDER *tests*, x ∈ [1.85, 2.15], which is the task JSON's
  range inflated by `ground_placement_threshold`. The region and the bin overlap on
  x ∈ [2.08, 2.15]; the bin does not sit past it. See the environment README and
  [the goal-region correction](#the-goal-region-was-wrong-and-these-numbers-are-the-re-run).

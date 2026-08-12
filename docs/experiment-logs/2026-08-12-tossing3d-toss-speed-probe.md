# Tossing3D: how far a speed dial on `Toss` can actually reach

**2026-08-12.** Rung-0 feasibility probe for the velocity-controller task. 280 cells:
three profile-scaling modes x a commanded-speed grid x 10 scene seeds, all at standoff
`ORACLE_THROW_STANDOFF = 1.35` with `Pick` held at the oracle grasp.

Measured at `reference/kinder-baselines` pin **`3524010`**, which contains upstream
PR #103 (base-motion collision-checking switched **on**). Verified at run time rather
than assumed: every result JSON records the `kinder_models.__file__` it ran against, and
the tree it names was checked to contain the uncommented `obstacle_geoms` lines.

![The ceiling, the dial's authority in metres, whether the label is missable in both directions, and R5](2026-08-12-tossing3d-toss-speed-probe.png)

## What was asked

`Toss` has `param_dim=0`, so nothing about the throw can be practised. The proposal is to
make the throw's energy a parameter by scaling the three limits `TossController.reset`
passes inline to `_trapezoidal_motion_profile` (`max_vel=140`, `max_accel=300`,
`max_decel=200` deg/s). Four questions had to be answered before anything was built:

1. Does scaling more than `max_vel` lift the release-speed ceiling?
2. What is the dial's authority, in metres of cube displacement?
3. Does the release pose stay speed-invariant (design-doc risk **R5**)?
4. Is there an **upper** failure edge, so the success label is missable in both
   directions rather than only from below?

A fifth question came from the pin: every earlier Tossing3D number was measured with
base-motion collision-checking off, so the probe was re-run rather than replayed.

## Modes

| mode | scales | what it is |
| --- | --- | --- |
| `vel` | `max_vel` | the design doc's option A |
| `vel-accel` | `max_vel`, `max_accel` | **the design doc's option D, as worded** |
| `vel-accel-decel` | all three | option D extended to the third limit |

At 140 deg/s all three are upstream's literals exactly, so that column is upstream's own
toss and every mode agrees there to the last digit (achieved 147.58 deg/s, range 1.344 m,
`10/10` solved in all three).

A solve needs the cube to rest with `range` in **[1.212, 1.512] m** — the goal region's
x extent `[1.850, 2.150]` minus the base's mean x at release, `0.638`.

## Result 1 — the pin changed nothing. Null result.

`vel` repeats the pre-pin grid exactly (60-240 deg/s by 20, seeds 0-4 among the 10), so
the two runs pair cell for cell.

| quantity | pre-pin `11eace5` | post-pin `3524010` | paired difference |
| --- | --- | --- | --- |
| range (m), mean over 50 paired cells | 1.2957 | 1.2958 | +0.0001, sd 0.0004, max abs 0.0017 |
| achieved release speed (deg/s) | 137.849 | 137.849 | 0.0000 |
| solved | 30/50 | 30/50 | **0/50 verdicts flipped** |

Paired t-test on range `t = 1.759, p = 0.085` (n = 50); Wilcoxon signed-rank
`W = 167.5, p = 0.274`. Nothing significant, and the effect size is 1.7 mm at worst —
smaller than the 6-42 mm seed-to-seed spread within a single cell.

**Turning base-motion collision-checking on does not perturb the throw at standoff 1.35.**
The pre-pin rung-0 numbers therefore stand. That could not have been known without
re-running, and it is the reason the re-run was worth its wall-clock.

## Result 2 — option D as the design doc words it does not work

`vel-accel` neither lifts the ceiling nor gives a usable dial. Its outcome **folds back on
itself**: 420 deg/s is indistinguishable from 140 deg/s.

| commanded | 140 | 180 | 220 | 260 | 300 | 340 | 380 | 420 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| achieved at release (deg/s) | 147.6 | 169.4 | 209.5 | 176.6 | 208.0 | 190.7 | 170.7 | 162.8 |
| range (m) | 1.344 | 1.460 | 1.692 | 1.449 | 1.688 | 1.703 | 1.537 | 1.447 |
| solved | 10/10 | 9/10 | 0/10 | 9/10 | 2/10 | 1/10 | 6/10 | 10/10 |

The mechanism is in the profile arithmetic and needs no physics: with `max_decel` fixed,
the deceleration phase grows as the **square** of the scale factor while the acceleration
phase grows only linearly. The profile crosses into its triangular branch, the release
point — which fires at a fixed fraction of *distance* — walks in and out of the cruise
phase as the scale factor rises, and the commanded release speed oscillates instead of
climbing. Total authority is **0.359 m**, *less* than option A's 0.460 m.

A sampler cannot learn a parameter that maps two far-apart values onto the same outcome.
**This variant should not be shipped**, and it is the one the design doc names.

## Result 3 — scaling all three limits does lift the ceiling, and buys an upper edge

| commanded | 60 | 100 | 140 | 180 | 220 | 260 | 300 | 340 | 380 | 420 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| achieved at release (deg/s) | 64.6 | 105.7 | 147.6 | 169.4 | 233.8 | 218.2 | 252.6 | 280.5 | 289.8 | 271.3 |
| range (m) | 0.954 | 1.147 | 1.344 | 1.652 | 1.846 | 1.748 | 2.055 | 2.288 | 2.288 | 1.984 |
| solved | 0/10 | 0/10 | 10/10 | 2/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| peak torque fraction | 0.284 | 0.436 | 0.586 | 0.738 | 0.875 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 |

- **Ceiling lifted.** Achieved release speed reaches 289.8 deg/s against `vel`'s hard
  saturation at 175.4 — a factor of 1.65.
- **Authority nearly tripled**: 0.954-2.288 m, a span of **1.334 m**, against `vel`'s
  0.460 m.
- **An upper failure edge exists for the first time.** Solves go `0/10` at 60 and 100,
  `10/10` at 140, `2/10` at 180, then `0/10` from 220 upward. The label is genuinely
  missable in *both* directions, which is the property the design doc's requirement 2
  asks for and which `MoveToThrowPose`'s old `NearBin` lacked.

## Result 4 — R4 is now true at the top, which is new

The pre-pin probe found **zero** torque saturation anywhere (peak fraction 0.725), and
recorded R4 as false in the good direction. That was measured over a range option A could
reach. Push harder and it stops holding:

| mode | saturated control steps | peak torque fraction | first saturating speed |
| --- | --- | --- | --- |
| `vel` | 0/1941 | 0.725 | none |
| `vel-accel` | 0/1270 | 0.884 | none |
| `vel-accel-decel` | **88/1610** | **1.000** | 260 deg/s |

Above 260 deg/s commanded the arm is torque-limited, and it shows in the data: range stops
tracking the dial (2.288 m at both 340 and 380, then *falling* to 1.984 m at 420) and
achieved release speed peaks at 380 and drops. **The dial is only trustworthy below the
saturation onset.**

Over the monotone, unsaturated window **60-220 deg/s commanded**, `vel-accel-decel` gives
range 0.954-1.846 m — **0.892 m of authority, still 1.9x option A's**, strictly increasing,
and straddling the [1.212, 1.512] m solve band with room on both sides.

## Result 5 — R5 fails, and option D makes it worse, as predicted

The release configuration is not speed-invariant. Peak-to-peak spread of the achieved
release configuration over each whole grid, worst joint (joint 6 in all three modes):

| mode | joint 6 spread | joint 4 | joint 2 |
| --- | --- | --- | --- |
| `vel` | 10.53 deg | 5.61 | 3.36 |
| `vel-accel` | 13.58 deg | 8.76 | 3.83 |
| `vel-accel-decel` | **19.72 deg** | 14.48 | 5.88 |

Two causes, both structural. Release is checked once per `_CONTROL_DT = 0.1 s` step, so the
release fraction lands wherever the discrete grid allows — measured 0.467 to 0.625 against
the nominal 0.46 — and the profile shortens as speed rises (14 control steps at 60 deg/s,
4 at 420), which coarsens that quantisation. On top of that the PD loop lags more at speed.

This was flagged in advance as the cost of option D and it is confirmed: scaling all three
limits roughly **doubles** the spread relative to option A. It does not block the design —
the dial still moves the cube monotonically over the usable window — but "the trajectory
otherwise stays the same" is false in fact, not merely at the margins, and the fix (a
smaller control timestep for the toss specifically) is a real follow-up rather than a
nicety.

## Result 6 — the ballistic range model is wrong, again

The design doc proposes `range = k * v^2`, fitted through the origin. Fitted to this data
it is **worse than predicting the mean** in every mode and every subsetting:

| fit | `vel` | `vel-accel` | `vel-accel-decel` |
| --- | --- | --- | --- |
| ballistic `k*v^2`, vs commanded, whole grid | R^2 = -9.97 | -26.66 | -2.56 |
| ballistic `k*v^2`, vs achieved, whole grid | R^2 = -4.38 | -1.87 | -0.40 |
| affine `a + b*v`, vs commanded, whole grid | R^2 = 0.853 | 0.064 | 0.843 |
| affine `a + b*v`, vs achieved, whole grid | R^2 = 0.941 | 0.674 | **0.960** |
| affine `a + b*v`, vs commanded, <= 220 only | R^2 = 0.890 | 0.856 | **0.969** |

The affine fit carries a large speed-independent intercept — `a = 0.588 m` for
`vel-accel-decel` over the usable window — which is what a ballistic-from-the-base model
cannot represent. The cube is released well forward of the base and then rolls, so a
constant offset dominates at low speed. Any symbolic replacement for `THROW_RANGE` should
use the affine form or a monotone interpolation, not `k*v^2`.

## What this does not answer

- **One standoff only** (1.35 m). Whether the dial's authority holds at other standoffs, and
  whether a *state-conditional* sampler could exploit that, is the sweep's job, not this
  probe's.
- **One bearing only** (`rotation=0.0`). The design doc's `RobotOnBinAxis` vs
  `BinWithinThrowRange` question needs a bearing axis and does not have one here.
- **Nothing about learning.** No `Method` was run. That the label is missable in both
  directions is necessary for a sampler to learn, not sufficient.
- **`solved` is `_check_goals()` after a fixed settle budget** (25 control steps, then 35
  more). Both readings are in the JSON; the tables quote the longer one.

## Reproducing

```text
scripts/with_kinder_env.sh python scripts/tossing3d_toss_speed_probe.py \
    --mode vel-accel-decel \
    --output docs/experiment-logs/2026-08-12-tossing3d-toss-speed-probe-vel-accel-decel.json \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --speeds-deg 60 100 140 180 220 260 300 340 380 420

python analysis/tossing3d_toss_speed_probe.py \
    --probe vel=docs/experiment-logs/2026-08-12-tossing3d-toss-speed-probe-vel.json \
    --probe vel-accel=docs/experiment-logs/2026-08-12-tossing3d-toss-speed-probe-vel-accel.json \
    --probe vel-accel-decel=docs/experiment-logs/2026-08-12-tossing3d-toss-speed-probe-vel-accel-decel.json \
    --output-png docs/experiment-logs/2026-08-12-tossing3d-toss-speed-probe.png
```

Each probe process ran inside
`systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -p OOMPolicy=continue`.
Three processes concurrently, against a machine already carrying 22 `hitl_pmp.cli` runs
from another agent and a load average near 26 — deliberately far below the core count,
since concurrency has been measured not to perturb results.

## Recommendation

Ship the parameter as an **effort scale on all three profile limits**, defaulting to
upstream's literals, and set its sampling bounds from the **60-220 deg/s** window rather
than from the whole reachable range. Do not ship the two-limit variant the design doc
names. Treat R5's doubled spread and the torque ceiling above 260 deg/s as known,
measured, and open.

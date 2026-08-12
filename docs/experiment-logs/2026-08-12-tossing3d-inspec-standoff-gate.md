# The toss dial at its low end: a standoff window, a sim artifact, and a constant that is not a hardware limit

**2026-08-12.** A 660-cell grid over the **low end** of the toss's speed dial
(60–83.34 deg/s) against 11 standoffs, on 10 fixed seeds.

> **Read this first — the framing this grid was run under was wrong.**
>
> This grid was commissioned to answer *"can a hardware-feasible release speed still solve
> Tossing3D?"*, under the belief that `_ARM_MAX_VEL[5] = 70 deg/s` was a hardware limit and
> that the shipped `max_vel = 140` therefore over-drove the real arm by 1.680x. **That
> belief is false**, and the section "What 83.34 deg/s actually is" below records what
> replaced it. The **measurements are unaffected** — every number here was measured, not
> derived from the assumption — but the *ceiling* they were run against is a
> kinder-baselines convention, not a physical bound. Read this as a characterisation of the
> dial's low end, which is what it is.

## What 83.34 deg/s actually is

`_ARM_MAX_VEL = [80,80,80,80,70,70,70] deg/s` is **kinder-baselines' own conservative
constant**, not the Kinova Gen3's mechanical limit. The evidence is the real robot's own
toss primitive, which the lab runs on its TidyBot
(`yixuanhuang98/tidybot_real`, `robot/kinova.py:120-124`, read directly rather than quoted):

```python
def toss(self, arm_heading=0):
    joint_angles = [arm_heading, 50, 180, 250, 0, 260, 90]  # windup
    self.move_angular(joint_angles)
    if self.check_joint_angles(joint_angles):
        self.movej_primitive.execute(
            [arm_heading, 20, 180, 325, 0, 25, 90],
            max_vel=140,
            max_accel=300,
            max_decel=200,
            gripper_release_ms=600,
        )
```

**Our sim toss is a port of this.** Reducing our constants mod 360 gives
`[0, 50, 180, 250, 0, 260, 90]` and `[0, 20, 180, 325, 0, 25, 90]` — **exactly** the real
configurations — with the same `(140, 300, 200)` triple. So those literals are the real
robot's demonstrated numbers, not sim inventions that violate hardware.

### Four divergences between the two, all verified here

1. **`max_vel` denominates differently, and our sim is the slower one.** The real primitive
   scales joints by `pos_diff_i / max(abs(pos_diff))` — an **L-infinity** norm, `125.0000
   deg`, joint 6 — so `max_vel=140` puts **joint 6 at 140 deg/s**. Ours runs the profile on
   `s_total = ||dq||_2 = 148.8288 deg`, so a path rate of 140 puts joint 6 at
   `140 x 0.839891 = 117.585 deg/s`. **Our sim is 16.01% slower at the binding joint than
   the primitive it ports.** To match it our path rate would have to be
   `140 / 0.839891 = 166.6882 deg/s` — *above* the shipped 140.
2. **`_ARM_MAX_VEL[5] = 70` is 2x exceeded by the real robot**, which runs that joint at
   140 deg/s in this very primitive. It is a conservative planning constant, not physics.
3. **Neither soft limit governs the real toss.** `set_joint_limits(speed_limits=(60,)*7)`
   and the Cartesian `twist_linear` apply to `ANGULAR_TRAJECTORY` / `CARTESIAN_TRAJECTORY`
   modes. The toss goes through `MoveJController`, which sets **`LOW_LEVEL_SERVOING`** and
   commands actuator positions in a **1 ms cyclic loop** under `SCHED_FIFO`, bypassing them
   by design. Kinova's published 40 cm/s Cartesian figure was never the governing
   constraint for this motion.
4. **Release trigger and control rate both differ.** Real releases on
   `gripper_release_ms=600`, **wall-clock**; ours on `fraction_covered >= 0.46`,
   **distance**. On the real profile (125 deg at 140/300/200: accel 0.4667 s, cruise
   0.3095 s, decel 0.7000 s, total 1.4762 s) 600 ms lands at distance fraction **0.4107**.
   And the real loop runs at **1 kHz** against our **10 Hz**.

**Stated at the strength it is warranted:** Yixuan Huang described this as *"the real-world
toss primitive we use on our tidybot"*, and was explicit that **the kinder-baselines toss
has not been run on hardware**. Nobody has told us that `toss()` has been executed with
these exact parameters and reliably lands objects. This is strong evidence that
`(140, 300, 200)` is hardware-demonstrated, **not** a measured success rate. Confirming it
would take a real-robot run reporting where the object lands.

## Method

11 standoffs (1.050–1.300, 0.025 steps) x 6 speeds (60, 65, 70, 75, 80, 83.3441 deg/s) x
10 fixed seeds = **660 cells, 0 errored**. `Pick` pinned at the oracle grasp so grasp
variance cannot be read as a speed effect; seeds shared across cells, so comparisons are
paired on scene. Both KINDER submodules verified at their pins (`3524010` / `4113237`) and
clean in the tree that ran. The probe's `in-spec` mode set `max_vel` to the requested speed
and `max_decel = max_accel = 178.5945 deg/s^2`.

## Results

![Speed x standoff over the dial's low end](2026-08-12-tossing3d-inspec-standoff-gate.png)

Seeds solved, out of 10 per cell:

| standoff | 60 | 65 | 70 | 75 | 80 | 83.34 | any |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.050 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | **60/60** |
| 1.075 | 4/10 | 10/10 | 8/10 | 10/10 | 10/10 | 10/10 | 52/60 |
| 1.100 | 0/10 | 9/10 | 3/10 | 10/10 | 10/10 | 10/10 | 42/60 |
| 1.125 | 0/10 | 4/10 | 0/10 | 4/10 | 10/10 | 10/10 | **28/60** |
| 1.150 | 0/10 | 0/10 | 0/10 | 0/10 | 4/10 | 10/10 | 14/60 |
| 1.175 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 5/10 | 5/60 |
| 1.200 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | **0/60** |
| 1.225 – 1.300 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/60 |

At these speeds the solvable standoffs are `[1.050, 1.175]`, and the two best — 1.050
(`60/60`) and 1.075 (`52/60`) — sit **below** `THROW_STANDOFF_BOUNDS`'s lower edge of
1.100. Upstream's own `TOSS_TARGET_DISTANCE_BOUNDS = (1.25, 1.45)` lies entirely inside the
region measured at `0/60`, so it cannot draw a solvable standoff anywhere in this speed
band.

### The dial is non-monotone here, and it is a simulation artifact

Paired on (standoff, seed), n = 110:

| step | mean range | paired t | p | Wilcoxon p |
| --- | --- | --- | --- | --- |
| 60 -> 65 | 0.9606 -> 1.0122 | -5.61 | 1.6e-07 | 1.1e-06 |
| **65 -> 70** | **1.0122 -> 0.9794 (-0.0329 m)** | **+4.51** | **1.6e-05** | **1.1e-06** |
| 70 -> 75 | 0.9794 -> 1.0037 | -2.65 | 9.3e-03 | 8.5e-03 |
| 75 -> 80 | 1.0037 -> 1.0422 | -4.82 | 4.6e-06 | 2.6e-07 |
| 80 -> 83.34 | 1.0422 -> 1.0711 | -3.70 | 3.4e-04 | 3.2e-04 |

The 65 -> 70 reversal is real, not seed noise. **The cause is our 10 Hz control rate.**
Release fires on the first control step past `fraction_covered >= 0.46`, once per
`_CONTROL_DT = 0.1 s`, so the *realised* release fraction sawtooths across the band
(0.4616, 0.4918, 0.4771, 0.5015, 0.4734, 0.4887). A higher realised fraction means a larger
Jacobian gain at release, fighting the higher commanded speed. Achieved release speed is
non-monotone for the same reason: 75 commanded achieves 79.9, 80 commanded achieves 72.4.

**On the real robot this artifact would be ~100x finer** — its loop is 1 kHz and its
release is wall-clock — so this is a fidelity defect in our simulation, not a property of
the toss. That makes "raise the sim's toss control rate" a real follow-up.

**Two things that are unmeasured and must not be assumed:**

- **Nobody has measured the dial finely at 140 deg/s.** PR #213's `R^2 = 0.99997` linear
  fit came from a **three-point** grid (60/100/140), which cannot see a sawtooth at all.
  This grid found reversals only because it stepped 5 deg/s. Whether 140 is monotone in its
  neighbourhood is **unknown**.
- **The profile gets coarser as speed rises, not finer.** Measured control steps in the
  swing: **32 at 60 deg/s, 25 at 83.34, 18 at 140, 14 at 240**. So there is no reason to
  expect the quantisation artifact to be milder at the operating point, and some reason to
  expect it worse. Also **unmeasured** — stated as a direction, not a result.

### Clips

Seven, captioned in-frame with standoff, commanded speed, achieved release speed, realised
release fraction, measured landing x and `InGoalRegion`. **Every one reproduces the grid
cell it illustrates to 0.000000 m, `7/7`** — a real check, since clips come from the public
`env.take_action` path and the grid from the hand-driven instrumented swing. Seed 1 carries
all but the last pair.

- [clean solve — standoff 1.050 @ 83.34 deg/s, cube rests at 2.0122](2026-08-12-tossing3d-inspec-gate-083deg-standoff1.050-seed1.mp4)
- **the sim artifact, made visible:**
  [65 deg/s at standoff 1.100 — lands in the bin at 1.9709](2026-08-12-tossing3d-inspec-gate-065deg-standoff1.100-seed1.mp4)
  versus
  [70 deg/s at the same standoff and seed — a *faster* command landing 0.24 m *shorter*, at 1.7300](2026-08-12-tossing3d-inspec-gate-070deg-standoff1.100-seed1.mp4).
  The captions carry both realised release fractions, 0.491 against 0.477, which is the
  whole explanation.
- [short-throw miss — 60 deg/s at standoff 1.150, rests at 1.7967](2026-08-12-tossing3d-inspec-gate-060deg-standoff1.150-seed1.mp4)
- [clean miss — standoff 1.200 @ 83.34 deg/s, still 0.08 m short at 1.7744](2026-08-12-tossing3d-inspec-gate-083deg-standoff1.200-seed1.mp4)
- the boundary, where the seed decides — standoff 1.175 @ 83.34 deg/s is `5/10`:
  [seed 1 misses at 1.7374](2026-08-12-tossing3d-inspec-gate-083deg-standoff1.175-seed1.mp4)
  and
  [seed 2 solves at 1.9818](2026-08-12-tossing3d-inspec-gate-083deg-standoff1.175-seed2.mp4).

### Three corrections to claims made earlier in this task

1. **`_CONTROL_DT` is live, not latent.** Previously reported as latent — `0.1 s` does agree
   with the env's default `control_frequency = 10 Hz`, and no committed measurement is
   corrupted. But it is what makes the dial non-monotone here.
2. **`g = 0.6108` is right after all.** An intermediate recommendation of ~0.622 came from
   evaluating the Jacobian gain at the nominal 0.46 fraction. Measured at the *achieved*
   release configurations across all 660 cells, `g` averages **0.610** (0.60408–0.61989,
   spread 3.09%), because PD lag pulls the achieved configuration back toward nominal.
3. **The base undershoot roughly holds** — 0.0175 m at standoff 1.050 falling to 0.0133 m
   at 1.300, against 0.0129 m at 1.35. An earlier single-cell reading of 0.0026 m was
   wrong; disregard it.

### Two things this does not settle

- **Barrier displacement below standoff 1.100.** `move_error` is `0/60` at every standoff
  including 1.050 and 1.075 with PR #103's collision checking live, so base planning
  *succeeds* there — but the probe records no barrier pose, so this cannot show the barrier
  stayed put.
- **Elevation drift refutes the design doc's section 4 inference.** That section expected a
  narrow window to drift "substantially less" than the 7.44 deg measured over 60–220 deg/s,
  and marked it unverified. Measured over 60–83.34 deg/s the swing is **7.212 deg** —
  essentially unchanged despite a ~7x narrower window. Determinism does survive: per-speed
  seed sd is 0.13–0.63 deg.

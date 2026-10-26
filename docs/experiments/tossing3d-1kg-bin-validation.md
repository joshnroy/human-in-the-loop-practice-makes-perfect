# Tossing3D: 1 kg bin, 100 g cube

> Historical validation record: these measurements used kindergarden `0dac89e`
> and kinder-baselines `427ad6c`. The PR port keeps current main's `490831f` /
> `032b426` pins and reports new validation separately. Launch preparation and
> temporary-directory handling have progressed since this record was written.

The current simulation defaults are a **1.0 kg movable bin** and **0.1 kg cubes**
in both layouts. Explicit masses in custom task configurations still take
precedence. The fixed-bin opt-out uses the same default masses. TMP, TMPDIR,
and TEMP remain redirected by `scripts/with_env.sh`.

The mass change exposed payload tracking errors in the pouring controller. Its
rim grasp now reaches 3 cm below the top, engaging more of the finger pads. Its
motion supplies joint velocity targets and compensates for the bin's gravity
force and moment about the rim grasp, using the existing PD interface and
unchanged actuator limits. The return path aligns the bin above its saved spot
before lowering, tracks the measured bin-to-hand relationship, and releases from
the measured placement pose. The final placement tolerance remains 2.5 cm and
5 degrees, with upright floor support and an actual subsequent cube pick checked
by the physics tests. The controller does not teleport objects or attach a weld.

All **eight physical recovery checks passed**: `exp22_A`, `exp22_C`, `exp22_D`,
and `exp22_F` in both layouts. Each checks physical lifting, a base quarter-turn,
sideways inversion, return placement, and an actual successful pick of the dumped
cube. The separate actual-throw/retrieval, full-episode, and far-bin witness tests
also passed (five checks).
The complete pour/fixed-bin/layout selection passed 36 checks in 180.65 seconds;
the two targeted test runs therefore passed 41 checks in total. Scoped Ruff,
formatting, mypy, and whitespace checks passed as well.

The new recording under `artifacts/tossing3d-pour-bin-1kg/rendering/` uses the
1 kg bin and completes in 676 control steps, with no skill error or human reset.
The bin returns upright with 0.02282 m position error, the hand is empty, and the
cube is on the floor with a valid pick plan. `pour-bin-physics.mp4` is at 2× playback.

## Toss and drag probe

`artifacts/tossing3d-pour-bin-1kg/stability_probe.py` records a bounded check using
actual pickup and toss controllers. It attempted 17 scenes/parameter combinations;
one initial pickup could not be planned, leaving **16 executed tosses**, including
**10 scored hits** and observed cube/bin contacts. All 16 ended with the bin upright
on the floor. Maximum tilt sampled at 10 Hz was **0.03734 degrees**, and maximum
final bin displacement was **1.74 mm**. These are sampled checks, not a guarantee
over every throw or a measurement of the training-time tip rate.

| Probe group | Scene seeds | Toss parameters (distance m, speed deg/s, release ms) |
| --- | --- | --- |
| Same-side starts | 0, 1, 2, 3, 4, 125 | 1.35, 130, 792; seed 0 had the initial pick refusal. |
| Barrier far-bin witnesses | 10126, 10127, 10128 | 2.5, 390, 460; all three scored. |
| Same-side timing/speed probes | 125 | Distance 1.35; speed/release pairs (115,400), (115,840), (140,720), (140,792), (200,600), (300,460), (420,400), (420,840). Several miss the bin; they are not all impact tests. |

The saved in-bin dragging case still moved the bin **0.17336 m** and did not grasp
the cube. It left the bin upright. A 1 kg bin is more stable under the tested toss
impacts, but it remains movable by gripper contact during a bad pick.

Raw outcomes are in `stability.json` and `drag.json` in the same artifact directory.
The earlier 3.8 kg checks and original 100 g-bin preview are historical results.

## Della comparison

Use the paired seeds, six arms, and cost settings in
[`tossing3d-heavy-bin-rerun.md`](tossing3d-heavy-bin-rerun.md), with **1 kg** as the
new bin mass. Its 3.8 kg lift failures describe the preceding configuration.
The solver-version mismatch still needs to be resolved before relaunching the
Curtis comparison. No Della jobs have been submitted for this change.

# PourCubeOutOfBin

Tossing3D now offers a parameterless robot recovery skill in both layouts. It is
applicable when the hand is empty, the cube is in an upright bin on the robot's side
of the barrier, and `PickPlannable(robot, cube)` is false. `PickNotPlannable` is the
explicit complement used by the STRIPS planner; it reads the same cached dry-run
verdict as `PickPlannable`.

The controller chooses a reachable rim and a clear direction for the base's
quarter-turn, approaches the rim with an open gripper, closes on the wall, and lifts.
It turns the base 90 degrees, rolls the bin sideways 180 degrees, waits upside down,
rolls upright, reverses the base turn, lowers the bin to its saved original pose,
opens the gripper, and withdraws. The motion matches the sideways-roll preview.

The live robot, bin, and cube move exclusively through motor commands and MuJoCo
contacts. Inverse kinematics runs in a separate `MjData`; the controller never
repositions the objects, welds the bin to the gripper, or requests a human reset.
It checks contacts on both fingers before lifting, bin height after lifting,
inversion, placement, uprightness, cube floor support, and the cube's actual pick
planner before reporting success. Failure is an ordinary failed skill with the
observed physical state preserved. Replacement tolerance is 2.5 cm and 5 degrees.

The simulation backend makes the bin movable by overriding `fixed` to false in a
temporary copy of the selected task configuration (an anonymous memory file on Linux).
Both layouts default to a 1 kg bin and 0.1 kg cubes; explicit masses in custom
task configurations take precedence. These masses also apply with a fixed bin.
All other task geometry and placement rules still come from the original task; no reference checkout is edited.
`KinderBackend(movable_bin=False)` retains the legacy fixed-bin dynamics and refuses
pouring. This changes default bin contact dynamics for other skills too: picks and
tosses can now move the bin. The real-robot backend does not offer this simulator
controller.

The skill uses action ID 4 with four unused zero parameter slots. Its symbolic
success moves the cube out of the bin, puts it on the floor on the robot's side,
and restores pick plannability and an empty hand. The same-side operator also adds
`OnFloor`. The belief-space practice model tracks a separate outcome-based belief
for pouring. Quantified delete effects can cause Fast Downward's translator to
emit conditional SAS effects; the default cost-optimal search uses admissible Hmax
for domains with those effects.

Validation covers four retained in-bin planning refusals (`exp22_A`, `exp22_C`,
`exp22_D`, `exp22_F`) in both layouts. Each case checks physical lifting and inversion,
return placement, and an actual successful cube pick after pouring. Offline tests
check applicability and a no-human plan from the refusal to holding the cube.
This describes the original pilot dependency revisions. On current main's newer
pick controller, A, C and D have a pick plan: tests now require pouring to refuse
those states without moving the world. F remains an unplannable pick and exercises
the full pouring recovery in both layouts, plus automatic EES, expectimax and
Curtis selection with outcome learning and charged execution ticks.
The earlier 3.8 kg trial failed during lifting. The current 1 kg configuration
adds payload gravity compensation and velocity tracking, and aligns the measured
bin pose above its original spot before lowering. See
[`tossing3d-1kg-bin-validation.md`](experiments/tossing3d-1kg-bin-validation.md) for
current validation and the proposed Della comparison.

Record a physics rollout, including the before/after states and outcome:

```bash
scripts/with_env.sh python scripts/tossing3d_pour_bin_demo.py \
  --output-dir artifacts/tossing3d-pour-bin-physics
```

The video is a controller rollout at 2× playback. The earlier preview scripts under
`artifacts/tossing3d-pour-bin*-preview` are kinematic illustrations, not validation.

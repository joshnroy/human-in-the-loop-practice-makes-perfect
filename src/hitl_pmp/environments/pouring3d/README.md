# Pouring3D proof of concept

Five 3 cm cubes start inside a rectangular handled pitcher. TidyBot pours them into
a shallow tray on the floor in front of its base. The autonomous reset picks cubes
individually from the tray and drops them into the pitcher. The human reset lifts
the tray, pours its contents back, and returns it to its floor home pose.

## Fidelity and scope

Cubes are free rigid bodies in MuJoCo. Both pouring actions transfer cubes through
gravity and contact dynamics; no cube locations or task-success flags are assigned
by a pouring controller. Containment checks transform all eight cube corners into
each vessel's local frame. Success requires all five contained cubes and settling.

The robot model is the existing KinDER TidyBot asset. Arm motion uses prescribed
joint positions and position-only numerical IK. Robot collision geometry is
disabled; gripper contact, wrist orientation, torque limits, and collision-free
paths are not validated. The pitcher and tray follow prescribed mocap trajectories.
Cube pickups use idealized, explicitly activated weld constraints; releases remove
the weld and let physics determine containment. The human is represented by the
tray motion, without a human body model. This is a contact-dynamics task prototype,
not a demonstration of an executable hardware controller.

Only the initial harness setup assigns cube positions. Motion interpolation for
containers occurs at the 2 ms physics timestep to avoid thin-wall tunneling.
Every skill has a bounded trajectory, an observed result, and a duration charged in
simulated seconds. These durations are design choices, not measured hardware times.

## Run

```bash
run-heavy scripts/with_env.sh python -m scripts.pouring3d_demo \
  --output-dir artifacts/pouring3d-poc-20261025 --seed 0
run-heavy scripts/with_env.sh pytest -q tests/environments/pouring3d/test_prototype.py
```

The demo produces separate skill videos, the full five-cube robot reset, a skills
overview, a setup image, failure examples, and a JSON manifest of observed states,
costs, outcomes, frame counts, and checksums. Videos are at normal simulated speed.
The seed controls the small initial cube-position jitter. Diagnostic trajectories
were tuned manually; they are not learned policies or held-out performance results.

## Project adapter

`PouringEnvironment` implements the core Environment interface. Its observation
contains qpos, qvel, mocap poses, equality activations, and handling state. The
single typed world object is a diagnostic encoding, not yet the relational object
schema needed for the practice planner. Solver/controller history is not represented;
Markov sufficiency has not been established.

Actions are seven-component vectors:

```text
[skill_id, cube_id, tilt_degrees, offset_x, offset_y, tilt_seconds, hold_seconds]
```

| ID | Skill |
| --- | --- |
| 0 | PickPitcher |
| 1 | Pour |
| 2 | PlacePitcherAtHome |
| 3 | PickCubeFromTray |
| 4 | DropCubeIntoPitcher |
| 5 | HumanPourBack |
| 6 | PickCubeFromFloor |
| 7 | Wait / inert macro action with a one-second accounting cost |

Cube IDs range from 0 to 4. Continuous parameters are used only for Pour. Picks from
inside the pitcher are not offered, so forward pick-and-place cannot bypass Pour.
The human action is explicit and increments the intervention count. It is a demo
action, not yet routed through the project's HumanOracle/planner reset interface.
Arbitrary `set_state` is rejected. `hard_reset` is privileged harness initialization.

### Continuous Pour parameters

| Parameter | Meaning | Bounds | Nominal demo value |
| --- | --- | --- | --- |
| `offset_x` | Pouring-position offset from tray center along world x | -0.4 to 0.4 m | 0 m |
| `offset_y` | Pouring-position offset from tray center along world y | -0.4 to 0.4 m | -0.055 m |
| `angle` | Maximum tilt about the fixed pouring axis | 0 to 150 degrees | 130 degrees |
| `tilt_seconds` | Duration of the upright-to-maximum-tilt motion | 0.5 to 4 s | 2 s |
| `hold_seconds` | Time held at maximum tilt | 0.2 to 4 s | 2 s |

The offsets are relative to the tray's position, in fixed world axes; this prototype
keeps the tray yaw fixed. The pitcher tilts about its negative x axis toward positive
y. Tilt duration controls speed through a fixed smooth trajectory profile. Grasp,
transport height (0.34 m above tray origin), transport duration, and return motion
are fixed controller rules rather than learned parameters. These bounds include
misses and retained-cube failures; they do not filter parameters by simulated success.

Reset completion means all five cubes inside the pitcher, both vessels upright at
their homes, hand empty, and cubes settled. It restores a set of eligible states,
not an exact arrangement of cubes. Human pour-back is unavailable when a cube is
on the floor; robot floor pick-and-drop is the demonstrated spill recovery.

## Before research use

1. Replace idealized attachments and position-only IK with feasible, observed
   grasps and full-pose motion controllers on the intended robot configuration.
2. Implement relational predicates and lifted skills, and route human requests
   through the costed HumanOracle interface. Offer the planner both reset choices.
3. Evaluate across held-out cube arrangements, reset methods, and parameter policies;
   characterize reset-mode dependence in Pour competence.
4. Demonstrate learning with a declared parameter domain and train/evaluation split.
   Neither successful nominal motion nor a small-tilt failure establishes learnability.
5. Measure robot/human durations and failure rates. If one reset dominates across
   all relevant states, the environment offers little evidence about reset selection.

Both reset cycles and retained-cube failures are checked on seed 0. That establishes
basic executable mechanics and accounting, not a reliability estimate or EES
Assumption 1. The prototype is not registered in the global CLI or integrated with
EES or the belief-space practice planner.

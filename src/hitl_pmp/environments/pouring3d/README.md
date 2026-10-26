# Native KinDER Pouring3D

`kinder/PitcherPour3D-o5-v0` is a native Dynamic3D TidyBot task, with the normal
lab2 room, robot model, collision geometry, torque controllers, and renderer.
Five free 3 cm cubes start inside a handled pitcher on the ground in front of the
robot. The task is to pour all five into the neighboring floor tray.

The native task lives in `reference/kindergarden/src/kinder/envs/dynamic3d/`;
this package supplies motor-driven macro controllers and the project adapter.
The earlier standalone scene has been replaced. No robot controller assigns live
qpos/qvel, disables contacts, adds attachments, or changes actuator limits.
IK uses a separate `MjData`. Only harness initialization places the starting cubes.

## Native task and props

| Object | Home x, y (m) | Outer dimensions (m) | Mass |
| --- | --- | --- | --- |
| TidyBot base | -0.16, -0.25 | Standard native asset | Standard native asset |
| Pitcher | 0.60, -0.25 | 0.18 × 0.18 × 0.21; 0.02 walls | 0.046 kg including handle |
| Tray | 0.65, 0.20 | 0.44 × 0.38 × 0.075; 0.01 walls | 0.30 kg |
| Five cubes | Inside pitcher, slight seeded jitter | 0.03 side length | 0.005 kg each |

These are explicit lightweight training-prop parameters selected for the stock
simulated gripper, not hardware-calibrated masses. The pitcher is a compound open
rectangular vessel with a solid, open-loop handle. The demonstrated robot grasp
pinches its rim. Both vessels and all cubes have free joints and real collisions.

## Task and reset contract

The goal requires all eight corners of each of the five cubes inside the tray's
interior, with linear speed below 0.025 m/s and angular speed below 0.25 rad/s.
Containment follows the vessel's actual orientation and uses 2 mm contact tolerance.

`ReadyToPour` requires all five cubes settled in the pitcher, empty robot hand, and
both vessels at home within 25 mm and 5 degrees of the initial orientation.
Robot reset picks individual cubes from the tray or exposed floor and drops them
into the pitcher. A cluster selector clears outer cubes before crowded interior
cubes; grasp orientations clear tray walls and align with cube faces. Grasp success
requires contact from both fingers and an observed physical lift.

Human reset pours the tray back into the pitcher and replaces it. The simulated
human applies bounded external forces (30 N per axis) and torques (2 Nm per axis)
to the free tray, updated at the physics timestep. The robot continues using native
motor commands. There is no human body model. This is an explicitly labeled human
interaction model, with no cube pose assignments or kinematic tray attachment.

A failed grasp, unreachable motion, retained cube, or miss is an observed failure.
Cubes remaining in the pitcher after a partial pour stay there; robot pick/drop can
recover individual strays. Human pour-back requires tray contents and no spills
outside either vessel. Reset restores eligible states, not an exact cube arrangement.
No cube pick from inside the pitcher is offered, so it cannot bypass Pour.

## Skills and predicates

| Skill | Initiation | Observed successful outcome |
| --- | --- | --- |
| PickPitcher | ReadyToPour | Both fingers contact pitcher; vessel lifted ≥20 cm |
| Pour | HoldingPitcher | All five cubes settled entirely inside tray |
| PlacePitcherAtHome | HoldingPitcher | Pitcher upright at home; hand empty |
| PickCubeFromTray(c) | HandEmpty, InTray(c) | Both fingers contact c; cube lifted ≥20 cm |
| PickCubeFromFloor(c) | HandEmpty, OnFloor(c) | Same physical grasp and lift |
| DropCubeIntoPitcher(c) | HoldingCube(c), pitcher home | Released c falls fully inside pitcher |
| HumanPourBack | HandEmpty, AnyInTray, NoSpills, vessels home | ReadyToPour |
| Wait | Always | One second of native simulation advances |

`predicates.py` exposes pure lookups into `PouringState.abstract_atoms`, computed
from native geometry and contacts at observation time:

- `HandEmpty(robot)`, `HoldingPitcher(robot,pitcher)`, `HoldingCube(robot,cube)`.
- `InPitcher(cube,pitcher)`, `InTray(cube,tray)`, `OnFloor(cube)`, `Settled(cube)`.
- `Upright(vessel)` (vertical within 5°), `AtHome(vessel)` (position and full orientation),
  `Empty(vessel)` (none of the named cubes fully contained).
- `AllInPitcher(pitcher)`, `AllInTray(tray)`, `AnyInTray(tray)`, `NoSpills()`,
  `ReadyToPour()`, `Goal()`.

These summarize observed outcomes; they do not assert that every geometrically valid
pick has a collision-free solution. Low-level execution checks report failures.

## Continuous Pour parameters

| Parameter | Meaning | Bounds | Nominal |
| --- | --- | --- | --- |
| offset_x | Wrist pivot offset from tray origin in world x | -0.4 to 0.4 m | +0.055 m |
| offset_y | Wrist pivot offset from tray origin in world y | -0.4 to 0.4 m | 0 m |
| angle | Maximum tilt around the fixed diagonal axis | 0 to 150° | 145° |
| tilt_seconds | Time for outward rotation; also used for return | 2 to 15 s | 10 s |
| hold_seconds | Dwell at maximum tilt | 0.2 to 8 s | 4 s |

The fixed tilt axis is normalized world `[-1, 0.45, 0]`, directing the pour toward
the corner away from the rim grasp. The wrist pivot is 0.45 m above the tray origin.
Base placement, grasp, transport trajectory, and axis are fixed controller rules.
The angle advances linearly in time; tilt duration sets angular speed. Bounds include
infeasible motions and failed transfers. Success is never used to filter parameters.

## Running and evidence

```bash
run-heavy scripts/with_env.sh python -m scripts.pouring3d_demo \
  --output-dir artifacts/pouring3d-native --seed 0 --reset both
run-heavy scripts/with_env.sh pytest -q tests/environments/pouring3d
scripts/with_env.sh pytest -q reference/kindergarden/tests/envs/dynamic3d/test_pitcher_pour3d.py
```

The default records one persistent sequence: pour → place → human pour-back → pour
→ place → five robot pick/drop pairs. `--reset none|human|robot` records shorter
sequences. `--simple-scene` suppresses the room background for diagnostic runs;
normal recordings keep it. `--no-video` runs exactly the same physics without rendering.

Videos come directly from `env.render()` after each native `env.step()`, at the
native 10 Hz control rate and normal simulated speed. The recorder emits a full
rollout, individual skill clips, setup/final frames, complete event snapshots,
per-frame simulation times, cube locations, state hashes, and source checksums.
It stops on a failed skill and records the failure rather than completing a montage
with a different state. No scene reset occurs between the demonstrated cycles.

`PouringEnvironment` exposes seven-component macros:
`[skill_id, cube_id, angle, offset_x, offset_y, tilt_seconds, hold_seconds]`, using
IDs 0–7 in the table order above except HumanPourBack=5, floor pick=6, Wait=7.
The observation contains typed relational objects, measured atoms, and diagnostic
physics/controller state. Arbitrary `set_state` is rejected. Every attempted skill
is charged its actual simulated duration; rejected initiation costs nothing. Human
calls also increment an intervention counter. Durations are scripted design choices.

This is an executable native task and proof of concept. Learned-policy evaluation,
held-out reliability, hardware transfer, and integration with the practice planner's
costed HumanOracle/lifted operator interfaces remain separate work. Macro availability
alone is not a complete practice-planner benchmark or evidence of learnability.

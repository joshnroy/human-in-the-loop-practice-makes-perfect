# Native KinDER Pouring3D

`kinder/PitcherPour3D-o5-v0` is a native Dynamic3D TidyBot task, with the normal
lab2 room, robot model, collision geometry, torque controllers, and renderer.
Five free 2 cm cubes start inside a handled pitcher on the ground in front of the
robot. The task is to pour exactly a requested count into the neighboring floor tray,
retaining the remaining cubes in the pitcher. The default request is all five.

The native task lives in `reference/kindergarden/src/kinder/envs/dynamic3d/`;
this package supplies motor-driven macro controllers and the project adapter.
The earlier standalone scene has been replaced. No robot controller assigns live
qpos/qvel, disables contacts, adds attachments, or changes actuator limits.
IK uses a separate `MjData`. Harness initialization and the explicitly simulated
human reset are the only operations that place cubes directly.

## Native task and props

| Object | Home x, y (m) | Outer dimensions (m) | Mass |
| --- | --- | --- | --- |
| TidyBot base | -0.16, -0.25 | Standard native asset | Standard native asset |
| Pitcher | 0.60, -0.25 | Approx. 0.150 × 0.112 × 0.178 including spout/handle; 0.003 walls | 0.120 kg including handle |
| Tray | 0.65, 0.20 | 0.44 × 0.38 × 0.075; 0.01 walls | 0.30 kg |
| Five cubes | Inside pitcher, slight seeded jitter | 0.02 side length | 0.003 kg each |

The pitcher approximates the [ACCUPOUR 32 oz / 1 quart pitcher, ASIN
B08WX3JWDM](https://www.amazon.com/dp/B08WX3JWDM). The selected listing supplies
overall dimensions (4.41 × 5.91 × 7.01 inches) and mass (0.12 kg). Taper, wall
thickness, spout profile, handle dimensions, and mass distribution are estimates;
this is not manufacturer CAD or hardware calibration. The native object docstring
also links the product. Cube/tray masses are explicit simulation design choices.

The hollow body uses 32 convex wall panels in each of two height bands, with a
flared spout at local +x and an open-bottom handle at -x. Sixteen coplanar box
strips approximate the circular floor and provide stable cube support contacts.
Containment follows the tapered polygon and spout instead of a bounding rectangle.
Five cubes start in a small pentagon, with ±0.5 mm seeded position jitter. The robot
pinches the upper handle across its width. Both finger pads must contact handle
geometry before and after lifting; rim contacts cannot satisfy that check. Both
vessels and all cubes have free joints and real collisions.

## Task and reset contract

Set `requested_count=k` (integer 1–5) on `PitcherPour3DEnv`,
`PouringSimulation.create`, or `PouringEnvironment`. The goal requires exactly k
cubes in the tray and exactly 5−k inside the pitcher, with all eight corners
contained, linear speed below 0.025 m/s and angular speed below 0.25 rad/s.
Undershoot, overshoot, any spill, or an unsettled cube fails. The native goal
classifier overrides the JSON's all-five default; its metadata describes the
requested split. The parent observation includes `requested_count_0` as a task
feature. The native flat physics vector is unchanged; the request is a fixed
environment constructor input, exposed by the scene and diagnostic snapshots.
Containment follows the vessel's actual orientation and uses 2 mm contact tolerance.

`ReadyToPour` requires all five cubes settled in the pitcher, empty robot hand, and
both vessels at home within 25 mm and 5 degrees of the initial orientation.
Robot reset picks individual cubes from the tray or exposed floor and drops them
into the pitcher. A cluster selector clears outer cubes before crowded interior
cubes; grasp orientations clear tray walls and align with cube faces. Grasp success
requires contact from both fingers and an observed physical lift.

The human's `reset_cubes` uses the same external intervention path as
SweepOffShelf3D and Tossing3D: `UnconditionalHumanOracle.execute_movables_reset`
calls `PouringEnvironment.reset_movables(destination="reset_cubes")`. That hook
respawns all five cubes inside the pitcher's current frame, clears their velocities,
and refreshes the observation. Robot/vessel poses, simulation time, and accumulated
robot costs are unchanged. No tray tipping or physics settling step is simulated.
The recorder holds the resulting image for one display second only.

Intervention pricing belongs to the caller/provider, separate from simulated
seconds. The demo accepts `--human-reset-practice-cost`, defaulting to the existing
v0 oracle's flat intervention cost of 1.0, and records that cost separately. It
records one intervention. A complete planner-facing lifted skill provider is still
separate work; the standard oracle/environment execution interface is implemented.

A failed grasp, unreachable motion, retained cube, or miss is an observed failure.
Cubes remaining in the pitcher after a partial pour stay there; robot pick/drop can
recover individual strays. Human reset can return any cubes outside the pitcher,
including exposed-floor spills. Both vessels must be home and
the hand empty. Human reset samples a small jittered arrangement in the pitcher;
robot reset retains the physical arrangement produced by its individual drops.
No cube pick from inside the pitcher is offered, so it cannot bypass Pour.

## Skills and predicates

| Skill | Initiation | Observed successful outcome |
| --- | --- | --- |
| PickPitcher | ReadyToPour | Both fingers contact handle; vessel lifted ≥20 cm |
| Pour | HoldingPitcher | Exactly requested count settled in tray; remainder settled in pitcher |
| PlacePitcherAtHome | HoldingPitcher | Pitcher upright at home; hand empty |
| PickCubeFromTray(c) | HandEmpty, InTray(c) | Both fingers contact c; cube lifted ≥20 cm |
| PickCubeFromFloor(c) | HandEmpty, OnFloor(c) | Same physical grasp and lift |
| DropCubeIntoPitcher(c) | HoldingCube(c), pitcher home | Released c falls fully inside pitcher |
| reset_cubes (human) | HandEmpty, some cube outside pitcher, vessels home | Instantaneous respawn; ReadyToPour |
| Wait | Always | One second of native simulation advances |

`predicates.py` exposes pure lookups into `PouringState.abstract_atoms`, computed
from native geometry and contacts at observation time:

- `HandEmpty(robot)`, `HoldingPitcher(robot,pitcher)`, `HoldingCube(robot,cube)`.
- `InPitcher(cube,pitcher)`, `InTray(cube,tray)`, `OnFloor(cube)`, `Settled(cube)`.
- `Upright(vessel)` (vertical within 5°), `AtHome(vessel)` (position and full orientation),
  `Empty(vessel)` (none of the named cubes fully contained).
- `AllInPitcher(pitcher)`, `AllInTray(tray)`, `AnyInTray(tray)`, `NoSpills()`,
  `ReadyToPour()`, `RequestedCountInTray()`, `RequestedRemainderInPitcher()`, `Goal()`.

These summarize observed outcomes; they do not assert that every geometrically valid
pick has a collision-free solution. Low-level execution checks report failures.

## Continuous Pour parameters

| Parameter | Meaning | Bounds | Nominal |
| --- | --- | --- | --- |
| offset_x | Wrist pivot offset from tray origin in world x | -0.4 to 0.4 m | 0 m |
| offset_y | Wrist pivot offset from tray origin in world y | -0.4 to 0.4 m | 0 m |
| angle | Maximum tilt toward the physical spout | 0 to 150° | 145° |
| tilt_seconds | Time for outward rotation; also used for return | 2 to 15 s | 10 s |
| hold_seconds | Dwell at maximum tilt | 0.2 to 8 s | 4 s |

The upright pitcher turns so its spout faces world +y, then tilts about world -x
toward that spout. The wrist pivot is 0.45 m above the tray origin.
Base placement, grasp, transport trajectory, and axis are fixed controller rules.
The angle advances linearly in time; tilt duration sets angular speed. Bounds include
infeasible motions and failed transfers. Success is never used to filter parameters.

These controls are an existing diagnostic interface, not five established learning
requirements. Centered aiming is the default. The prior rectangular-pitcher results
do not establish performance for this new geometry. Empty-all time optimization
remains a separate benchmark-design candidate. Requested-count pouring is implemented
as an exact split goal; parameter sensitivity alone does not establish learnability.

## Running and evidence

```bash
run-heavy scripts/with_env.sh python -m scripts.pouring3d_native_demo \
  --output-dir artifacts/pouring3d-native --seed 0 --reset both
run-heavy scripts/with_env.sh pytest -q tests/environments/pouring3d
scripts/with_env.sh pytest -q reference/kindergarden/tests/envs/dynamic3d/test_pitcher_pour3d.py
```

For the partial-pour proof of concept:

```bash
run-heavy scripts/with_env.sh python -m scripts.pouring3d_native_demo \
  --output-dir artifacts/pouring3d-requested-two --seed 0 --reset none \
  --requested-count 2 --angle 130 --tilt-seconds 10 --hold-seconds 0.2
```

This is a selected manually tuned example, not a learned or general count controller.
Exact-three metering remains unresolved: cubes can leave as a cluster. Requests 1–5
are valid goal specifications, not a guarantee that the controller can achieve each.
`--reset none` records pickup, pour, and placement. Reset variants return k
delivered cubes individually or invoke the existing human reset. Both aiming offsets stay centered in this diagnostic.

The default records one persistent sequence: pour → place → human pour-back → pour
→ place → five robot pick/drop pairs. `--reset none|human|robot` records shorter
sequences. `--simple-scene` suppresses the room background for diagnostic runs;
normal recordings keep it. `--no-video` runs exactly the same physics without rendering.

Videos come directly from `env.render()` after each native `env.step()`, at the
native 10 Hz control rate and normal simulated speed. The recorder emits a full
rollout, individual skill clips, setup/final frames, complete event snapshots,
per-frame simulation times, cube locations, state hashes, and source checksums.
It stops on a failed skill and records the failure. No full scene reset occurs
between cycles. The external human reset is the explicitly marked cube-respawn event; every
robot grasp, pour, pick, and drop continues from physical simulation state.

`PouringEnvironment` exposes seven-component macros:
`[skill_id, cube_id, angle, offset_x, offset_y, tilt_seconds, hold_seconds]`.
Robot IDs are 0–4, floor pick=6, Wait=7. Legacy ID 5 is reserved and rejected:
human reset is an external oracle call, never an action in the robot's action space.
The observation contains typed relational objects, measured atoms, and diagnostic
physics/controller state. Arbitrary `set_state` is rejected. Every attempted skill
is charged its actual simulated duration; rejected initiation costs nothing. Human
resets increment an intervention counter and advance no simulation time. Robot
durations are controller design choices; human practice cost is supplied separately.

This is an executable native task and proof of concept. Learned-policy evaluation,
held-out reliability, hardware transfer, and a complete planner-facing lifted skill
provider remain separate work. Macro availability
alone is not a complete practice-planner benchmark or evidence of learnability.

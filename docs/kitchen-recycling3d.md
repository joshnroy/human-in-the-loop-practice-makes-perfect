# Kitchen recycling: isolated Tossing3D simulation variant

> **Superseded in part (2026-11-02): the kitchen is now a re-flavour of Tossing3D.**
> `KitchenRecycling3DEnvironment` is a `Tossing3DEnvironment` on the kitchen scene, and
> its provider is `Tossing3DSkillProvider` itself. The planner sees exactly Tossing3D's
> skills, operators, predicates, samplers, bounds and costs:
> `PickCube`, `MoveToTossLocationAndToss`, `OpenGripper`, `PourCubeOutOfBin` and the two
> human resets, each priced at 5. There is one belief per skill and no near/far copies.
> The goal is `InBin` only, decided by KINDER's classifier on the bin's
> `blocks_goal_region`. What stays kitchen: the floor, the mats as the no-drive line
> (the barrier is the mat edge, x = 1.15), the bin target on the mats, the furniture, the
> kitchen pick controller (`RecyclingPickController`), and the human placements. A
> same-side reset puts the bin beside the robot on the kitchen floor and an opposite-side
> reset puts it back on its target; both put the cube on the kitchen floor. The drive zone
> reaches Tossing3D's toss planning through `Tossing3DState.drive_zone`. Only a single
> object is supported. The sections below that describe contexts, per-context
> calibration, settled-in-bin scoring, `NeedsRecovery`/`Pickable` and the three human
> skills describe the environment before this change.

This is **not** a change to the cube benchmark, its launch configurations, or its results.
Implementation starts from `d079f94b92b442d3eb5473081fae275ce163ac35` in a separate worktree.
Launch receipts, rather than this document, establish whether jobs have been submitted.
The existing Tossing3D/KINDER object classes, controller constants and task files are
not edited; the new environment has its own CLI registration.

## Story and task

During weekend training a person is home but busy. Every requested intervention costs
an interruption plus handling, including a separate cost for restoring the target bin
pose. During deployment the person leaves for the gym: the same observed world continues,
but human recovery, bin relocation and restoration are rejected **before state mutation**.
A missed throw stays where physics leaves it. The robot may attempt another reachable
pick; an unreachable miss is not silently reset, deleted or counted as successful.

The default goal is **both** the empty capped Gatorade bottle and empty food-can proxy
settled in the red bin, with the bin at its target pose. A green cube remains in the scene
as a calibration control, but is not a mandatory recycling goal. `required_objects` can
explicitly include it. The object catalog also accepts `kind="packaging"` box proxies;
those require their own dimensions, mass and calibration, not the can's statistics.

For the requested overnight experiment, **practice and evaluation both use exactly one
object**: the launch configuration contains one `objects` entry and the same sole
`required_objects` entry. Selection is based on pickup validation, not on which learned
method scores best. The other objects remain supported by the environment but are absent
from that experiment's physical scene and skill library. This tests single-object
recycling, not the default multi-object goal. A cube-only fallback is a calibration
control, not evidence of bottle or can performance.

Initial positions are seeded samples across the clear kitchen floor, separated from the
robot and other objects. Bottles start sideways with seeded random yaw. The old preview
configuration explicitly disables scattering only to reproduce its historical recording;
the overnight configuration must enable scattering. Training and held-out evaluation use
separate scene-seed streams with the same object and geometry specification.

## Physical facts versus simulation assumptions

Specifications were looked up on 2026-10-31. A product specification does not establish
the identity, empty mass, or condition of the actual item in the lab.

| Item | Evidence / configured geometry | Still unverified |
|---|---|---|
| Red bin | Sterilite **0647, 12 Qt Dishpan**, outside 38.1 × 32.385 × 16.51 cm; Classic Red is a listed color. [Manufacturer](https://www.sterilite.com/product/12-qt-dishpan/) | Actual bin identity, interior opening/taper, rim/handles, empty mass. The straight-wall 6 mm shell and 0.4 kg mass are explicit simulation assumptions. Published case shipping weight is **not** an individual empty-bin mass. |
| Capped empty 20 oz Gatorade bottle | Requested approximate envelope **7.3 × 7.3 × 19.5 cm**. A vending-equipment manufacturer's older specification lists 7.4 cm diameter and 19.3 cm height, consistent only at the approximate-envelope level. [Royal Vendors PDF](https://www.royalvendors.com/wp-content/uploads/2013/11/61pvs.pdf) | Current bottle SKU, actual diameter, empty capped mass, deformation/rigidity, friction and grasp pressure. The PDF's 0.64 kg is a filled-product weight, **not** the empty mass. The 35 g rigid cylindrical body/cap proxy is unverified. |
| Empty food can | Generic **7 × 7 × 10 cm**, 40 g rigid proxy. Actual commercial formats vary: 211×400 is about 6.83 × 10.16 cm. [Can supplier's format chart](https://wellscan.ca/can-size-chuck-size-reference-simple/) | Chosen can SKU, empty mass, precise diameter/height, open-end geometry, sharp edges and crush resistance. Do not treat this proxy as a measured can. |
| Flooring | **Reuse Tom's existing Sprout mats. No black-carpet purchase; no AIRHOP substitution.** The user allows arbitrary dimensions for this simulation, so the rectangle is configurable. | Exact purchased SKU and physical layout are not identified by a reliable source. Unrelated Sprout-branded products are not evidence of Tom's purchase. Survey footprint/reach before hardware operation. |

Mats are visual, noncolliding floor overlays. An explicit no-driving boundary and a
conservative 0.4 m robot swept-radius guard implement the restriction. The simulator
does **not** assert that the robot physically cannot cross foam mats. All initial and
human-recovered pickup objects must fit wholly on the kitchen floor. The deployment
bin is beyond the configured direct-placement reach from any allowed base position.

### Furnished preview

The restricted mat side now includes the existing MimicLabs lab3 desk meshes with four
Replica `frl_apartment_chair_04` chairs. The desk is scaled 0.6 in its horizontal axes
and 0.8 vertically to fit this illustrative layout; it is not a measured lab floorplan.
Furniture has conservative solid clearance-box collisions visible to the planner, in
both background-rendering modes. These boxes deliberately prohibit moving underneath
the desk or between chair legs. The camera looks from the opposite side so the bin is
on the left of the image; the bin's physical target and no-driving boundary are unchanged.
The room is a 6.2 × 6 m rectangle with four physical, axis-aligned walls instead of
the inherited chamfered corners. The TV/wall-unit and wall decoration are removed.
An illustrative stove with four unlit burners, oven front and flanking counters sits
on the robot's kitchen side. Stove/counter boxes participate in collision planning;
these are static scene fixtures, not functional cooking skills or measured appliances.
Original videos/results are preserved. Earlier calibration and rollout outcomes below
predate this furniture addition and are not new furnished-scene performance claims.
The autonomous-retrieval test places its reachable bin on the open kitchen side
at `(0.65, -1.4, 0)`. Furniture regressions check that every rendered mesh fits its
physical clearance box, every desk/chair box fits wholly on the mats, and each box
reaches the planner's observations. A room regression checks four physical rectangular
walls, absence of the TV geometry, kitchen-side stove placement and four burners.

## Shared interfaces and scoring

New code is contained in `src/hitl_pmp/environments/kitchen_recycling3d/`. No shared
planner or skill signatures change. The standalone adapter provides `pick(name=...)`,
`toss(name=...)`, `human(action=...)`, `begin_deployment()`, `observe()` and `step(action=...)`.
`RecyclingEpisode` and `ObjectObservation` are the simulator-independent interface for
future hardware adapters. The `kitchen_recycling3d` CLI registration supplies a distinct
problem, predicates, skill provider and motor-step protocol for the existing learning
algorithms. Deployment runs in a separate simulator with human skills removed and human
actions rejected independently at the environment boundary. This is **not a hardware
vision/controller adapter**. DP is not included in the requested overnight matrix.

Pick and OpenGripper are parameterless controllers; object/context grasp offsets are
calibration settings. Toss uses the shared learner's three parameters: standoff, speed,
and release time. Its azimuth is an object/context calibration setting. Skill identities
keep object/context statistics separate. Human recovery, near-bin relocation and target
restoration go through the existing charged human-request protocol, not robot opcodes.
There are no free periodic practice resets. Full experiment launch is gated on physical
pickup/recovery checks, all three method pilots, deployment rejection, and the frozen
source/dependency package; command-line parsing alone is not sufficient validation.

- Public identities distinguish bottle, can, cube and packaging. KINDER's legacy
  collision adapter discovers movables using `cube_*` names internally; the adapter
  maps those names to typed public identities. Those internal names do **not** make
  bottle perception or scoring use the cube's pose/shape.
- Observations include full object-specific position/quaternion, angular/linear motion
  and bilateral gripper contact. Simulation perception reads MuJoCo ground truth;
  there is no claim of a deployed RGB bottle/can detector.
- Sideways cylinder grasps follow the observed longitudinal axis. Descent has a
  per-object/context grasp offset, and held-object collision geometry uses the observed
  object orientation. The planning proxy is a conservative oriented bounding box;
  physical geometry is cylindrical for bottle/can and box-shaped for cube/packaging.
- A successful pick requires two distinct finger-side contacts, lift relative to the
  **start of that pick**, and a sustained confirmation interval. Height alone, a closed
  command, or controller termination is insufficient.
- In-bin scoring uses cylinder/body-cap support bounds in the **current bin frame**,
  not an upright cube box or a fixed world-coordinate target region. The horizontal
  footprint must fit the opening, the bottom cannot be below the bin floor, and the
  object's origin must lie below the rim. A tall bottle may protrude above an open rim;
  an object balanced on the rim is not counted. Stable low linear/angular velocity and
  no finger contact must persist for the object's configured settling interval.
- The terminal multi-object goal additionally requires the bin within 2.5 cm and
  0.05 rad of its target pose. These tolerances, like grasp/settling thresholds, need
  hardware validation; the simulation deliberately does not reset a shifted bin.
- `pick` also attempts objects in a reachable bin using the same bin-aware collision
  planner. No blanket ban on autonomous retrieval is introduced. The distant deployment
  bin is intentionally unreachable; failed/refused retrieval never summons a person.
- There is **no bin-dumping skill in this variant**, and no claim that Tossing3D's
  simulator-only pouring controller exists on hardware.

## Contexts and calibration

Statistics keys are `(object identity, near/far context, skill)`. Calibration proposals,
attempts and successes are never pooled across objects or contexts. `near` means the bin
center is less than 0.5 m beyond the kitchen's far boundary (including a bin temporarily
relocated within the kitchen); `far` is the default living-room target. The starting
context is retained in every practice event. Training relocation and restoration are
separate logged, costed actions. These contexts are not claimed equivalent to the old
same/opposite-side cube benchmark's competence groups.

`calibrate.py` runs a small **local simulation** grid independently for each object and
context. Each trial starts a new scene and records picks, tosses, observed endpoints,
parameters and human costs. A failure to find a successful toss remains explicitly
unresolved; a successful grasp can still supply a grasp-only proposal. Nothing sets
`hardware_validated=True`. The preview settings are smoke-calibrated on seed 0, not a
success-rate estimate or evidence of generalization. Do not use them as validated
hardware throws. Release times are milliseconds from swing start; speeds are degrees/s.

The local grid found successful pick-and-settle trials separately for all six
object/context pairs, and the chosen proposals were replayed with the final conservative
driving radius. A full multi-object rollout can still fail: clutter, earlier throws,
bin motion and object-object contacts alter subsequent states. The video reports those
outcomes without editing failures into successes.

## Single-object overnight configuration

`scripts/kitchen_recycling/experiment-config.json` selects `cube_control` only, in
both practice and evaluation. The new pickup controller passed 10/10 single-object
scattered starts and 5/5 pick/open/repick sequences without human recovery. All-three
object pickup regressions passed 23/23. The configured cube toss proposal succeeded
in 6/6 near/far trials across seeds 0, 1, 2. Near-bin retrieval succeeded in 2/3;
the remaining rim-adjacent landing was refused by the collision planner, remained
observed, and is not labeled physically impossible. Zero-motion pick refusals are
remembered despite the post-attempt settling ticks, enabling explicit human recovery
during training rather than silently teleporting or retrying an unchanged pose forever.
These finite simulation checks do not establish a learned-policy success rate.

The requested matrix is EES, expectimax, and Curtis, each with seeds 0, 1, 2. EES
uses grounded competence, all practice groundings, human-skill scoring enabled,
no skip-perfect ranking, and no goal pursuit. All arms use a 160,000 **counted-step**
horizon, held-out evaluation every 3,200 steps, 10 tasks per evaluation, and a
500-control-step deployment cap. Human interventions cost 12; robot motor steps cost
0.01. Costs are logged, not hard-capped at the historical DP-specific 2,000 budget.
There are no DP jobs in this matrix. Near/far statistics remain separate.

`--no-early-stop` keeps the measurement horizon even when a method chooses STOP;
it does not force robot actions or suppress STOP decisions. STOP/time bookkeeping
is explicit and must not be reported as executed robot motor steps. External STOP
files and errors still terminate a run. The final measurement is taken after the
last cycle's learning update, including when the horizon coincides with a measurement
boundary. Training and deployment use separate simulators; evaluation supplies no
learning feedback or free practice reset. Original cube runs/results are not overwritten.

Preparation scripts never submit by default. Source/dependency hashes, the selected
single-object config, recorded readiness checks and completed method pilots are required
before full authorization. Check submission receipts and scheduler status rather than
inferring launch from the existence of a manifest or snapshot.

## Local commands

Use the project environment with its pinned `reference/kindergarden` and
`reference/kinder-baselines/kinder-models` sources on `PYTHONPATH`, plus `src`.
KINDER `490831fa54a40c4b2b4570d45bb08e3dd63c24ef` and kinder-baselines
`032b42634a015c7f710667b3a791836fa839f6ff` were used. No asset download is required when
the existing shared MimicLabs assets are installed. Set `MUJOCO_GL=egl` on this workstation.

```bash
python -m hitl_pmp.environments.kitchen_recycling3d.calibrate \
  --output-dir /tmp/recycling-calibration --workers 3
python -m hitl_pmp.environments.kitchen_recycling3d \
  --config configs/kitchen_recycling3d_preview.json \
  --output-dir /tmp/recycling-preview \
  --object gatorade_20oz --object food_can
pytest -q tests/environments/kitchen_recycling3d
```

Output directories must be new. The preview writes `preview.mp4`, `scene.png` and
`report.json`. Frames are actual MuJoCo motor-controller steps at 10 fps, not a scripted
animation or state interpolation. The report includes settings, step counts, failures,
interventions, deployment rejection checks, final observations and remaining measurements.
It also records source-file SHA256 hashes. A released object is observed until confirmed
settled (inside or outside the bin), or until its configurable settling timeout; a timeout
is not a successful toss.

## Validation and recorded outcome

The final local check passed **81 tests**, covering the new variant and existing cube
fixed-bin, skill and problem regressions. Ruff formatting and lint checks passed.
Checks include sideways-cylinder geometry, full-object zone containment, sustained
two-finger grasp confirmation, settling and multi-object goals, human interruption
accounting, pre-mutation deployment rejection, and live autonomous retrieval of a cube
from a reachable bin. No original benchmark files or stored results were changed.

The seed-0 multi-object video deliberately retains a failure: the bottle is picked and
settles in the target bin, but the subsequent can pick fails collision-free grasp planning
after the can moves. Final completion is **1/2 recycling objects**, not a successful
multi-object deployment. Two training interventions cost 24 units total; all three
deployment human actions are rejected, with no automatic recovery. Independent local
calibration successes for each object/context do not establish reliable sequential
multi-object performance.

### New files

| File under this worktree | Responsibility |
|---|---|
| `src/hitl_pmp/environments/kitchen_recycling3d/domain.py` | Typed object/context configuration, geometry, scoring and intervention contract |
| `src/hitl_pmp/environments/kitchen_recycling3d/scene.py` | Isolated object proxies, red bin and noncolliding mat layout |
| `src/hitl_pmp/environments/kitchen_recycling3d/furniture.py` | Task-local lab desk/chair assets and matching physical/planning clearance boxes |
| `src/hitl_pmp/environments/kitchen_recycling3d/controllers.py` | Observed-pose grasp candidates and held-object collision orientation |
| `src/hitl_pmp/environments/kitchen_recycling3d/simulator.py` | Motor execution, ground-truth observations, driving guard and costed recovery |
| `src/hitl_pmp/environments/kitchen_recycling3d/calibrate.py` | Independent object/context simulation proposal search |
| `src/hitl_pmp/environments/kitchen_recycling3d/__main__.py` | Captioned video and auditable rollout report |
| `src/hitl_pmp/environments/kitchen_recycling3d/__init__.py` | Opt-in package exports |
| `configs/kitchen_recycling3d_preview.json` | Explicit unverified geometry and six simulation-calibrated proposals |
| `tests/environments/kitchen_recycling3d/{__init__,test_domain,test_simulator}.py` | Contract and live-physics regression checks |
| `docs/kitchen-recycling3d.md` | Evidence, assumptions, usage, limits and changed-file inventory |

## Before hardware

Measure the actual bin opening/taper/rim/handles and empty mass; bottle diameter, capped
empty mass, deformation and grasp force; chosen can/packaging dimensions, empty mass,
rigidity and safe edges; installed Sprout layout and robot swept footprint/reach.
Validate object-specific perception, contact/grasp confirmation, near/far release timing
and settled-in-bin thresholds independently. Hardware deployment remains blocked by
`hardware_blockers()` and by the absence of a hardware adapter. Physical mat crossing
difficulty must be demonstrated before claiming it as an environmental constraint.

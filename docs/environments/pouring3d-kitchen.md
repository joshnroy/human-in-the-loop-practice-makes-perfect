# Pouring3D kitchen variant

`pouring3d_kitchen` is opt-in. `pouring3d`, its assets, dependency pins, saved results,
and existing experiment jobs are unchanged. This work adds no kitchen-specific
learning algorithm and launches no training experiments.

## Task and lifecycle

Weekend practice uses **cold** dumplings and a **dry, unheated pan**.
Water is deployment-only scene geometry, not a planner predicate. The robot chooses
Pour1, Pour2 or Pour3 and learns each skill's ordinary three-dimensional sampler.
It may retrieve individual pieces using the robot, or request
`human_bulk_pour_back`. The latter costs a finite, strictly positive interruption
cost (default 5, in the learner's existing practice-cost units). The environment
also records the count and cost of direct human calls. This audit ledger does not
charge the learner a second time. Failed human requests still incur interruption
cost. The bulk-reset abstraction moves only the ingredients; it does not reset the
robot, jug, or pan.

Deployment has a fixed requested count. The planner must pour the observed correct
split into the uncovered pan, put down the jug, invoke `ActivateCooking`, then
`MoveAway`. `MealStarted(request)` is the deployment goal. No human action is
available through the provider, and direct `reset_movables` calls are rejected at
the environment boundary. Hot ingredient manipulation is not implemented.

Human removal of cooked food is a **between-task handoff**, declared only after
completed deployment. It is not an in-task rescue or a practice reset. The simulator
does not model cooking completion. An explicit test-harness initialization creates
an independent episode and logs that boundary; it is not a planner action.
`hard_reset`, `set_state`, and fresh training-task sampling reject active-task
resets. In-place practice uses the existing step-budget protocol; the cycle protocol
is deliberately rejected because its resets do not meet this variant's contract.

## Geometry and physical scope

The candidate pan is a [Lodge L8SK3 10.25-inch skillet](https://www.lodgecastiron.com/collections/skillets/products/round-cast-iron-classic-skillet).
The configured empty mass, 2.57 kg, is supplied by the task specification. It does
**not** establish any robot payload comparison. Pan dumping is excluded because
there is no implemented pan-grasp/dump skill.

The jug candidate is the approximately 1 L [Norpro 3037, listed in Norpro's catalog](https://www.norpro.com/wp-content/uploads/2023/08/2023-Norpro50thYrCatalogDigitalw1.pdf).
`KitchenGeometry` retains missing measured inner pan dimensions, jug dimensions,
handle clearance, and jug mass as `null`. Separately named `*_proxy_*` fields are
explicit simulation assumptions, not product measurements. The jug counter and the
pan's hob are both 0.74 m: the real setup stands the skillet and the jug on the same
29 in desk (0.7366 m, kept at 0.74 m), the skillet's part in place of a stove.
At the earlier 0.84 m the dumpling pickup was often out of the arm's reach. Recordings
made before these changes embed their own 0.84 m heights and replay at them. A
`--kitchen-geometry` JSON file can override these fields;
the complete configuration is embedded in recordings.

Deployment water is a visible, **non-colliding visual proxy**, not a solid surface or lid.
Fluid motion, buoyancy, added water inertia, and heat transfer are not simulated.
The pan and handled jug retain their native rigid-body contact geometry. Dumplings
are the existing three potsticker proxies. Contact/friction values are simulation
assumptions, not calibrated material measurements. Friction is set per surface
through MuJoCo's contact priority, under which a contact takes the coefficient of
its higher-priority geom:

| Geom | Priority | Coefficient | Governs |
| --- | ---: | ---: | --- |
| Skillet (`skillet_mu`) | 3 | 0.4 | every skillet contact: dumplings, desk, jug |
| Dumplings (`--surface-mu`) | 2 | 0.1 | dumpling-jug, dumpling-dumpling, worktops, floor |
| Jug | 1 | 0.7 | jug-counter |

The skillet's 0.4 is a judgement for frozen dough on dry seasoned cast iron
(0.3-0.5); the dumplings' 0.1 lets the pours meter pieces out of the jug. Raising the
skillet's priority also replaces its desk contact, previously max(0.6, 1.0) = 1.0, with
0.4. Gripper-pad contact is a separate explicit 0.6 pair and is unchanged. The
legacy `pouring3d` environment keeps its own 0.1 and priorities.
`test_kitchen_friction.py` reads these values from the solver's live contacts.

## Skill interface

The kitchen provider uses the existing grounding, PDDL, sampler, competence,
control-step accounting, human-cost, and recording interfaces. Food recovery keeps
the existing `Shrimp` identifiers for interface compatibility; the physical and
rendered objects are dumplings.

| Skill | Discrete objects / choice | Continuous learned parameters |
| --- | --- | --- |
| PickPitcher | Robot, jug, pan, ordered ingredient objects | None |
| Pour1 / Pour2 / Pour3 | Separate skills; matching count token, robot, jug, pan, ingredients | Tilt (degrees), sideways amplitude (m), feeding duration (s) |
| PlacePitcherAtHome | Robot, jug, count token | None |
| PickShrimpFromTray / Floor / Surface | Robot, ingredient; pan for tray pickup | None |
| DropShrimpIntoPitcher | Robot, ingredient, jug | None |
| Wait | None | None |
| human_bulk_pour_back | Robot, jug, pan, ordered ingredients; training only | None; explicit interruption cost |
| ActivateCooking | Robot, active count token, hob; deployment only | None |
| MoveAway | Robot, active count token, hob; deployment only | None |

Pouring uses the existing `joint_lateral` controller: measured arm/joint tracking,
prescribed tilt, horizontal wiggle perpendicular to the pour direction, independent
amplitude and duration. It does not predict future ingredient motion or stop at an
oracle count. Requested count selects the skill and scores the observed result.
Support motions retain native IK checks, add a sampled whole-base-path preflight
including the carried jug, and check an ingredient lift continuation before grasping. Carrying/pouring use
a larger IK clearance margin and wait on measured joint tracking before advancing
loaded transport waypoints. Ingredients and
partial outcomes are not snapped into desired bins after robot actions.

Before PickPitcher and before every ingredient pick, and only with an empty hand,
each continuous arm joint (1, 3, 5, 7) at least a full turn from its episode-initial
angle (`|q - q_neutral| >= 360 deg`, so a joint near +-180 deg never flips) is rotated
back by `k = round((q - q_neutral) / 360 deg)` whole turns along a
collision-checked joint path (same hand pose; joints 2, 4, 6 untouched; no home
posture). IK resolves each continuous joint to the equivalent nearest its previous
angle, so without this pick/place cycles ratchet joint 1 towards the +-688 deg IK
bound, where it pins and strands the jug. A loaded IK target within 20 deg of that
bound fails the skill with a message naming the joint. The motors track about
0.2 rad/s, so unwinding costs about 31 s (about 314 ticks at 10 Hz) per turn.

Existing predicates remain: HandEmpty, HoldingPitcher, HoldingShrimp, PitcherHome,
TrayHome, InPitcher, InTray, OnFloor, OnSurface, Settled, PayloadOrder, Count1/2/3,
ActiveRequest, NoSpills, NeedsReset, ReadyToPour, SplitCorrect, Goal. Added predicates:
HobOff, HobOn, ColdIngredients, RobotAway, MealStarted, PickPlannable. Containment,
settling and robot-away checks use native observed state. Hob state is an explicit
simulator userdata channel with a visible burner/control indicator. The simulated activation
skill is **not** a validated physical heat-control interface.

`PickPlannable(robot, ingredient)` is a precondition of all three ingredient picks and
of nothing else; in particular the human bulk pour-back does not require it. It is
**history-dependent**: false iff this environment instance has seen the dispatched pick
of that ingredient refuse to plan from the ingredient's current pose (within 5 mm and
5 deg) with the pan where it was, and true otherwise, including every state never tried
and every imagined planner state. A refusal is recorded only for a planning error
(the retry loop's "No feasible continuation..." / "No collision-free empty-arm
preparation path...", or an IK/clearance rejection before the fingers reach the
ingredient) with no finger contact during the skill and the ingredient unmoved; a
slip, lost grasp or tracking fault is an ordinary failed attempt. A refusal is
forgotten once the ingredient or pan moves. Why history: the pick's planning is
deterministic, so one refusal is exact evidence and costs one attempt, where asking
the planner by dry run cost a median 49 s per observation. A practice run keeps the
memory across sessions; a fresh evaluation environment starts empty. An ingredient the
robot cannot pick therefore leaves the costed human reset as the planner's route. See
`kitchen/pick_plannable.py`.

## Motion planning scene

Planned motions are searched in PyBullet and executed in MuJoCo, the Tossing3D split
(`kitchen/planning/`, laid out like `kinder_models.dynamic3d` for a later port).
`KitchenPyBulletSim` extends upstream's `PyBulletSim`: it reads the arm mount from
MuJoCo (upstream's hard-coded mount is 5 mm off for this model) and mirrors every
contact geom of the counter, stove, jug, skillet and dumplings as the same primitive
or the convex hull of the same mesh vertices. The robot stays upstream's Kinova Gen3
with Robotiq gripper. `test_kitchen_pybullet_scene.py` bounds the gap: link frames
agree to 0.1 mm; mirrored object distances to about 1 mm; and over 1200 sampled arm
configurations near every obstacle, wherever PyBullet clears the 1.5 cm planning
margin MuJoCo is not in contact. Out of contact PyBullet is at most about 1 cm more
optimistic than MuJoCo, all of it at the bracelet/gripper coupling, whose meshes differ.

**PickPitcher** plans its free-space motion: a base path to the pick pose
(`run_base_motion_planning`, the arm checked in 3-D along it) and an arm path to a hover
above the handle (`plan_checked_arm_motion`: BiRRT, remapped, densely re-checked, 1.5 cm
margin), both before the robot moves; a refusal executes nothing. The kitchen follows
every waypoint at 0.2 rad/s and 0.15 m/s. The descent, grasp and lift that follow are
contact motions and stay scripted. The hand-tuned aisle detour for a hand left over the
jug is gone; the planner routes around it.

**The carry to the pour** (inside Pour1/2/3) is planned whole before anything moves,
with the jug held rigidly in the planning model, its riding dumplings exempt from
collision with it, and its axis within 10 deg of vertical at every checked
configuration (provisional, until the spill angle is measured): the arm tucks the jug
into a carry pose 0.45 m ahead of the base and 1.10 m high; the base drives to the pour
pose with arm and jug checked in 3-D; the arm moves the jug to the pour start above the
pan. The goal configuration is the closest IKFast solution that is collision-free and
upright, not simply the closest. The tilt, wiggle and return of the pour stay the
scripted closed-loop controller. Measured in MuJoCo on 2026-11-02: closest approach
3.14 cm, jug tilt at most 11.5 deg (it hangs 3.6 deg in the compliant grasp), no
dumpling lost.

`mj_geomDistance` (MuJoCo 3.3.7) sometimes returns 0 for mesh pairs centimetres apart,
with witness points on neither geom; the parity test recomputes those from vertices.
The scripted controller's clearance checks use the same call, so some of their
rejections may be spurious.

## Validation and video reproduction

Offline and native boundary tests:

```sh
scripts/with_env.sh pytest -q tests/environments/pouring3d/test_kitchen.py \
  tests/environments/pouring3d/test_kitchen_native.py \
  tests/environments/pouring3d/test_practice_contract.py
```

Bounded native action checks, independent of any learner:

```sh
scripts/with_env.sh python -m scripts.kitchen_check --case deployment \
  --output artifacts/kitchen-validation/deployment
scripts/with_env.sh python -m scripts.kitchen_check --case practice \
  --output artifacts/kitchen-validation/practice
scripts/with_env.sh python -m scripts.render_kitchen_check \
  --source artifacts/kitchen-validation/deployment \
  --output artifacts/kitchen-validation/deployment.mp4
```

Run native checks under the workstation's memory-capped service policy. The checks
save actual MuJoCo poses, phase labels, observable hob states, action outcomes,
containment, and substep contact audit results. Videos replay those poses; they do
not rerun or animate an ideal trajectory. Transport is shown at 4x; pouring and ingredient contact/lift/release segments at
1x. Diagnostic fixture tests explicitly arrange partial pours/spills to verify
observation and reset semantics; they are not reported as successful robot pours.

A `--case recovery --initial-motion <motion.npz>` check uses an explicitly labeled,
independent post-pour pose fixture (with zero initial velocities). It validates
recovery from that state, not a continuous episode from the initial jug. Such
fixtures are harness initialization and never a free reset inside practice.

A `--case placement --initial-motion <motion.npz>` check similarly validates
placement from a recorded near-counter pose with a partially loaded jug. The low
release fallback requires an upright, slow jug within 9 mm of home and 20 mm above
support. Physical settling and the inherited 10 mm landing check still determine
success. This isolated fixture is not a rerun of the complete preceding pour.

The contact audit checks robot contacts against stations, pan and ground, and
unexpected robot–jug contacts, with explicit intended-grasp/contact exceptions.
It complements the native planner checks; zero audited contacts is not a claim of
hardware collision certification or exhaustive geometry verification.

## Remaining hardware checks

Measure the actual vessel interiors, handle clearance, jug mass and reachable
counter layout. Validate real jaw/handle fit, grasp retention and slip, ingredient
shape/friction and wet adhesion, native trajectory tracking and collision clearance,
placement, individual recovery, and observed containment. Existing idealized
attachment/proxy assumptions do not establish hardware feasibility.

Separately validate an actual appliance command/readback interface and its physical
operating procedure. A visible simulated hob state and a successful rigid-body
trajectory do not establish real heater operation, boiling-water behavior, or cooked
food readiness. No physical robot or heater has been operated by these checks.

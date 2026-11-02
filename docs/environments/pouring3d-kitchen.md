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
explicit simulation assumptions, not product measurements. The default jug counter
height is 0.84 m. The pan's hob is 0.74 m: the real setup stands the skillet directly
on a fixed-height desk (about 29 in, not measured) in place of a stove, and at the
earlier 0.84 m hob the dumpling pickup was often out of the arm's reach. Recordings
made before this change embed their own 0.84 m hob and replay at it. A
`--kitchen-geometry` JSON file can override these fields;
the complete configuration is embedded in recordings.

Deployment water is a visible, **non-colliding visual proxy**, not a solid surface or lid.
Fluid motion, buoyancy, added water inertia, and heat transfer are not simulated.
The pan and handled jug retain their native rigid-body contact geometry. Dumplings
are the existing three potsticker proxies. Contact/friction values are simulation
assumptions, not calibrated material measurements. The kitchen's default dumpling
`--surface-mu` is 0.4, a judgement for frozen dough on dry seasoned cast iron (0.3-0.5),
where the legacy `pouring3d` default stays 0.1. The dumpling geoms carry contact
priority, so this one value governs dumpling contact with the pan, the jug, the
worktops, the floor and each other. Gripper-pad contact is a separate explicit 0.6
pair and is unchanged.

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

# Scoop-and-pour onboarding assessment — 5 October 2026

> **Committed 2026-10-21 from the main checkout's uncommitted files.** Links to
> external sites and private Notion pages were removed (their text is kept), and links
> to paths outside this repository became plain paths. Most `artifacts/` evidence named
> here is not committed; it lives in the main checkout's untracked `artifacts/`
> directory, and the `/tmp` paths no longer exist.

Status: the user selected ScoopPour3D-o10 first and specified a single fixed
grasp location. No integration, learner training, held-out baseline, or comparative
experiment has been run. The remaining controller/observation contract,
intervention, and pilot below are proposals requiring approval.
No generated code-as-policy is proposed: controllers are fixed, reviewed native
action programs; PDDL selects skills; the existing learned samplers select numeric
parameters; EES and our belief-space method decide what to practice.

## Recommendation and evidence

The selected environment is **ScoopPour3D-o10**, subject to a controller-feasibility gate. It shares
MuJoCo, TidyBot, IK/base planning, and object-centric state conventions with
Tossing3D, and its native all-object goal fits the existing conjunctive PDDL goal
interface. This is an engineering recommendation, not demonstrated learnability.
Keep DynScoopPour2D-o10 as an alternative requiring its own approved scope.
Onboard one environment before comparing across both.

| Question | DynScoopPour2D | ScoopPour3D |
| --- | --- | --- |
| Native dynamics | PyMunk; hook graspable, particles not directly graspable | MuJoCo; TidyBot, scoop, source and target bins |
| Smallest public variant | 10 particles: 5 circles, 5 squares | 10 cubes |
| Native goal | At least 5/10 particles satisfy the settled-right predicate below | All 10 cube centers satisfy the target-region predicate below |
| Native interface verified here | 180 observation values, 5 action values | 695 observation values, 11 action values in the inspected scene |
| Existing scoop/pour skill provider | None found in this checkout or checked KINDER models | None found in this checkout or checked KINDER models |
| Reusable motion components | SE(2) BiRRT and neighboring hook controller, with caveats | TidyBot/PyBullet IK, base and arm planning; new tool geometry/control required |
| Main scientific unknown | Feasible tool trajectories, state ambiguity, learnable parameter surface | Feasible grasp/scoop/pour trajectories, grasp/load predicates, learnable parameter surface |
| Additional integration risk | Cardinality goal; visible-state restoration is not dynamically exact | Multi-object abstraction; scoop is not a Tossing3D cube |

The website describes four 2D counts (10/20/30/50) and two 3D counts (10/100).
The inspected 3D checkout additionally contains 20/30/40/50 and target-bin
regression tests. Website prose does not define the executable controller or goal.
Sources: 2D environment,
3D environment.

**Demonstrated in this assessment:** the three existing 2D reset/random-action
tests passed. Six 3D configuration tests passed. Two 3D runtime goal tests first
failed because registration changed `PYOPENGL_PLATFORM` to OSMesa after EGL had
been selected. An isolated CPU/OSMesa process then passed all eight selected 3D
tests, including target-bin movement and settling. Four intermediate-count tests
were deliberately deselected. These are 11 distinct passing tests, not evidence
of robot task completion. The 3D tests place objects using privileged state access.

Separate seed-9100 probes instantiated both o10 variants, rendered their initial
states, and executed one bounded action with valid observations and no task
success. The 2D action was zero; the 3D action had zero motion deltas and gripper
command 1 (close). The corrected probe supersedes an initial probe that called a
nonexistent 2D state accessor and failed before acting. No diagnostic state or
action has been supplied to a learner. No full-controller successful trajectory
has been demonstrated for either environment.

Evidence is in [the assessment artifact directory](scooppour-onboarding-2026-10-05/):
`native-tests.json`, `native-3d-tests-osmesa.json`, `runtime-probes.json`,
`provenance.json`, and the two initial-state PNGs. The runtime JSON records exact
object names, feature order, values, action bounds, and registration arguments.

## Existing work and conventions to reuse

Main checkout: `86a188fafb12fae78922e80c4581f21a8d706396`.
Executed KINDER: `8f600231da8ee898da1055144665a8c5f6c246f0`.
Executed kinder-baselines: `c18056d2f0530599630a9962d210bb2ef9e040d5`.
Both reference checkouts differ from the parent repository's gitlink pins.
Packed-4 was inspected read-only at `/tmp/hitl-packed4-method-integration`, HEAD
`292e17d8162c0ced2998d550b2db94da4decab0f`, **with substantial local edits**.
A commit alone therefore does not reproduce its inspected behavior. The evidence
manifest records status and diff hashes; future runs also need archived patches,
untracked source files, assets, environment versions, and content hashes.

Preserve the modified notebook, reference checkouts, existing scripts, artifacts,
and all other worktrees. Integration should use an isolated source/dependency
snapshot, not change or depend on mutable shared work. Do not transplant Packed-4
wholesale: its PDDL changes and method-specific hooks require review against main.

Reuse these seams:

- `core` Environment/Problem/Tasks/SkillProvider and fixed `Skill` contracts;
  `planning/pddl.py`, grounding, and Fast Downward;
- EES's `LearnedSkillSampler`, proposal/admission interface, consultation modes,
  and outcome accounting; our existing belief inference/search, with a new
  environment model instead of pretending ten particles are one Tossing3D cube;
- Packed-4's independent pick/place planning, full-surface uniform proposals,
  pre-release feasibility boundary, isolated evaluation, candidate logging, and
  state-only trajectory collection;
- Tossing3D's simulator/backend boundary, raw state logging and post-hoc renderer;
  `stats.json`, `config_snapshot.json`, `timing.json`, streamed auxiliary JSONL,
  and `scripts/run_sweep.py` for managed comparisons;
- repository plotting conventions, plus hollow markers for no-practice cycles.

Packed-4's older protocol says 9 candidates and 16 decisions; its inspected live
runner uses 100 candidates and defaults to 50 practice/10 evaluation decisions.
Its earlier 10/10 baseline was a ceiling under an earlier proposal/admission
contract. It is not evidence of learning under the revised full-surface contract.
Use code/config/provenance together, not an old experiment note as a new default.

## Executable task predicates and native action domains

**2D:** for particle i, native right-side success is
`x_i > 1.75 AND y_i < 0.75 AND vx_i² + vy_i² < 1e-4`.
Termination requires at least `ceil(0.5*N)` such particles. For o10 this is five,
not ten. It does not test angular velocity, hook use, or future recoverability.
Native reward is -1 per action. Changing the threshold to 100% changes the task
and needs approval. The current `Goal` abstraction is a conjunction: preserving
the native cardinality goal needs explicit compilation/planning support, not
choosing five privileged particles in advance.

Native 2D action order and support:
`(dx,dy,dtheta,darm,dgripper)` in
`[-.03,.03] × [-.03,.03] × [-pi/32,pi/32] × [-.08,.08] × [-.015,.015]`.
Workspace is `[0,3.5] × [0,3]`; robot arm is clamped to `[.2,.4]`.
The default frequencies are 100 physics Hz and 10 control Hz, but `step()` has
two nested ten-iteration physics loops. Thus its code advances 100 physics
substeps per native action; do not infer elapsed simulated time from “10 Hz”.
Treat this as an upstream timing issue to resolve explicitly, not silently fix.
Source: `dyn_scooppour2d.py::_get_reward_and_done` and `dynamic2d/base_env.py::step`.

**3D:** the o10 task conjoins `on(cube_i,target_bin_region)` for i=0..9.
That region is attached to the live green bin. Its specified local bounds are
`[-.185,.185] × [-.11,.11] × [.03,.055]` m, expanded by the region builder's
.01 m tolerance on each boundary. At identity orientation the effective box is
`[-.195,.195] × [-.12,.12] × [.02,.065]`, relative to the bin. The executable
checker uses the region's world-axis-aligned bounding box, inclusive bounds, and
cube centers. It does not require zero velocity, full cube containment, actual
support contact, scoop use, or an empty hand. Rotated-bin checks must match that
implementation rather than substitute a stricter oriented-box classifier.
Source: `dynamic3d/envs.py::_check_goals`, `objects/base.py::Region.check_in_region`,
and `tasks/ScoopPour3D/ScoopPour3D-o10.json`.

Native 3D action order is base x/y/yaw deltas, seven arm-joint deltas, gripper
command. The first ten supports are `[-.1,.1]`; gripper is `[0,1]`, where zero
commands open. These are joint-space actions, despite the website's Cartesian
description. Native reward is -1 per action. The current variant does not enable
the optional seven unbounded joint-velocity command slots.

Native reward, macro-skill effort, human cost, and planning wall time must be
logged separately. Neither benchmark supplies our human-intervention cost model.

## Skill inventory: what exists versus what must be designed

There are **zero implemented scoop/pour macro skills or learned parameter
selectors** in the inspected integration. Consequently there are no existing
macro preconditions, effects, costs, success predicates, or selector inputs to
report as verified. The following is a proposed complete initial 3D library,
not an implemented or validated controller contract.

| Proposed skill and object arguments | Intended continuous support | Preconditions; immediate success and symbolic effects | Proposed effort |
| --- | --- | --- | --- |
| GraspScoop(robot, scoop) | None: one fixed tool-relative grasp at the handle midpoint, with orientation following handle geometry | HandEmpty; success HoldingScoop, delete HandEmpty | 1 |
| Scoop(robot, scoop, source_bin, particle) | Entry x/y over the entire .45 × .30 m source footprint; depth [0,.07] m; stroke heading [-pi,pi]; stroke length [0,sqrt(.45²+.30²)] m; tool pitch [-pi,pi]; speed fraction [0,1] of native motion limits | HoldingScoop and InSource(particle); success HoldingScoop and Loaded(particle,scoop), remove InSource after lift | 1 |
| Pour(robot, scoop, target_bin, particle) | x/y over entire .45 × .30 m target footprint; lift fraction [0,1] of a geometry-derived vertical envelope; tilt axis and angle each [-pi,pi]; speed fraction [0,1] | HoldingScoop and Loaded(particle,scoop); success native InTarget(particle), NotLoaded(particle,scoop), and HoldingScoop; remove Loaded | 1 |
| OpenGripper(robot, scoop) | None | Closed/holding hand; success commanded-open hand with scoop no longer held; remove HoldingScoop. Dropped cargo effects are observed | 1 |
| HumanRestoreMovables(robot, scoop, source_bin, target_bin, all particles) | No learner-selected parameters; separate reset RNG | Available at every practice decision boundary; effects specified below | H, default proposal 5 |
| STOP (harness decision) | None | End cycle; no physical effect and no practice example | 0 |

All surface parameters are uniform over full observed extents, without edge
insets, grids, occupancy masks, or successful sites. In bin-local coordinates
the proposed x/y supports are `[-.225,.225] × [-.15,.15]` m. They deliberately
include poor placements. Geometry changes update the surface extent, not learned
success-based subregions. Bounds follow geometry/controller limits; none came
from a successful trajectory. Zero-speed proposals are retained as budgeted
no-motion failures if selected, rather than assigned successful replacement values.

**Not yet specified sufficiently to run:** the scoop-handle frame/grasp evidence,
the lift-envelope calculation, the contact motion phases, the Loaded classifier,
and tolerances for holding/load stability. A scoop is an open container; a
particle momentarily passing through its volume must not count as securely loaded.
Propose containment plus supporting contact and bounded relative motion at the
skill boundary; exact thresholds need geometry/contact inspection and review.
Tossing3D's Holding classifier and one-cube abstractor cannot be imported unchanged.
The broad seven-dimensional Scoop surface may be too sparse for a small pilot;
that is a hypothesis to test, not permission to tighten it around oracle successes.

Grasp sees no future Pour parameters/destination. Scoop and Pour draw independently;
Pour plans its own base/arm transport using its supplied values and actual grasp.
The handle midpoint is the concrete geometric choice for the user's fixed-grasp
instruction, not a location selected by a success sweep. If it is mechanically
infeasible, report the blocker instead of searching for a different grasp silently.
Grasp has no learned parameter selector; its observed reliability and cost still
matter to both planners. Fixed commanded grasp does not guarantee identical actual
grasp geometry after every execution.
The selected particle is a symbolic target, not a simulator-selected easy particle.
All other particles remain in the observations. Bulk contact may move several;
recompute all predicates after every execution. Optimistic PDDL effects describe
intended progress and never overwrite the real state. Failures may have physical
effects; model/log these rather than treat them as self-loops.

Immediate successful Scoop does not guarantee a safe transfer. Successful Pour
of one particle can tip a bin or strand others. OpenGripper can succeed while
dropping the only usable tool. Track native task success, skill success, and
continuation feasibility separately. A failed bounded planner establishes “no
plan found under budget”, not physical impossibility. Diagnostic continuation
search results must never become training labels or candidate masks.

If 2D is selected instead, write a separate reviewed GraspHook/Scoop/Pour/Open
contract. The neighboring `dyn_pushpullhook2d` controller is not drop-in: its base
class hardcodes `DynObstruction2DEnvConfig` bounds and falls back to direct
interpolation after motion-planning failure. The motion planner assumes no contact,
which conflicts with scooping. Separate contact-free transit from intended contact;
do not label a fallback as verified collision-free motion.

## Selector inputs and state sufficiency

Existing exact reference inputs:

- Tossing3D's learned toss row is `[1, standoff, speed, release]`; state-dependent
  stand-direction choice and feasible-band construction happen outside the row.
  PickCube and OpenGripper have zero learned parameters. The toss row relies on
  a controller-specific invariance argument; it is not a template for scooping.
- Packed-4 placement uses 36 values: bias; normalized candidate x/y; target-block
  half extents (3); and, for each of the three other blocks, normalized relative
  position (3), normalized half extents (3), relative quaternion (4). Neighbors are
  sorted by candidate distance with a row tie-break. It omits grasp transform,
  robot pose/velocities and held-block orientation; its adequacy is not proof for
  another controller. Its two Pick skills have no learned continuous parameters.
- The generic fallback is `[1] + concat(features of bound objects) + params`.
  Binding only robot/tool/one particle would omit the rest of the pile.

Proposed initial comparison inputs for **each** learned Scoop/Pour selector:
`[1] + flatten(all native objects in frozen name/feature order) + role indicators
+ selected skill's numeric parameters`. For this o10 scene the native state part
has 695 values. Role indicators are one-hot vectors over the same frozen object
list, in the skill's argument order. Separate per-skill models make skill identity
implicit. Raw rows and normalization transforms are logged; normalization fits
only training data. GraspScoop/OpenGripper/human/STOP have no learned parameter selector.
This is a concrete proposed observation baseline, not evidence of sufficiency.

### Simplification and EES Assumption 1 review

The user asked whether the particle argument can represent multiple particles,
which predicates are needed, and whether these skills satisfy EES Assumption 1.
This section records design alternatives, not additional approved assumptions.

Bulk execution already permits one Scoop or Pour to move multiple particles.
The smallest change to the existing fixed-arity EES/PDDL implementation is to keep
one named particle as an intended-effect witness: Scoop must include that particle
in its loaded payload, and Pour must transfer it. The controller may move others,
and every resulting particle predicate is reobserved. There is no physical
single-particle restriction. An explicit arbitrary-set argument is possible, but
needs a set representation and success/effect semantics; it is not a supported
variable-length object argument in the current `Skill`/`GroundSkill` interfaces.
An aggregate `HasLoad` effect alone cannot promise that any particular particle,
let alone all ten, has reached the target. Extra observed progress must not be
turned into a guaranteed unexecuted effect in the planner.

Proposed simplification for approval: fixed geometric motion templates with three
learned Scoop parameters `(entry_x, entry_y, depth)` and three learned Pour
parameters `(target_x, target_y, tilt_angle)`. Stroke direction/length, approach
orientation, transport clearance and speed would be deterministic controller
rules based on measured geometry and native motion limits. These rules must be
declared before learning; this is a restriction of the earlier parameter family,
so it needs approval. Keep the entire surface x/y domains. Validate that the
restricted controller family can solve the native task; do not extract successful
sites or parameter values from diagnostic trajectories to seed the sampler.
Transit stays inside each independently parameterized skill; no new Navigate or
Lift skill is needed unless native execution shows a separate decision is necessary.

Core proposed physical predicates:

| Predicate | Meaning at an observed skill boundary |
| --- | --- |
| HandEmpty(robot) | Gripper open and tool not held; not merely the negation of HoldingScoop |
| HoldingScoop(robot,scoop) | Actual retained grasp, with contact/relative-pose evidence; numeric tolerances unvalidated |
| ClosedEmpty(robot) | Gripper closed without holding the scoop, permitting OpenGripper recovery |
| InSource(particle,source_bin) | Particle in the source bin's physical interior; define geometry/tolerances separately from its initial sampling region |
| InScoop(particle,scoop) | Particle retained in the scoop, using the loaded-state classifier discussed above |
| InTarget(particle,target_bin) | Exact native target-region classifier, including its center/AABB/tolerance semantics |

Convenience tests `ScoopEmpty`, `HasLoad` and `SourceNonempty` derive from those
particle facts rather than adding hidden physical state. The original `Loaded`
name means `InScoop`; `NotLoaded` is its complement, explicitly represented if
needed for this implementation's positive add-effect success check. Success in
the source bin is not assumed disjoint from being in a scoop inside that bin;
remove `InSource` only when an actual lift makes it false. Fixed source/target
roles can be constants/static facts. All-ten `InTarget` remains the task goal.
Neither bin stability nor motion feasibility is implied by these six predicates;
adding initiation predicates for those conditions is a further design decision.

EES Assumption 1 requires, for a fixed grounded skill and its current parameter
policy, the same success probability throughout its initiation set. The policy
may depend on the observed state; that does not itself establish the invariance.
This differs from both having a Markov observation and choosing different skills'
parameters independently. The paper explicitly discusses grasp-to-placement
dependence as a possible violation. See Section II-B, printed page 3 of the
EES paper.

No proposed physical skill has been demonstrated to satisfy that assumption:

| Skill | Design-based risk, not an empirical finding |
| --- | --- |
| Fixed GraspScoop | Tool pose, robot configuration and obstructions can change success even at a fixed handle location |
| Scoop | Particle location, pile depletion/packing, source-bin pose and actual grasp change load probability |
| Pour | Payload size/distribution, grasp quality, target pose and approach accessibility change transfer probability |
| OpenGripper | Most plausible approximation if opening/release always works under its initiation condition; a jam or retained tool would break it |
| HumanRestoreMovables | Could have invariant own success if its declared effects were always achievable; current robot-conditioned placement can fail, so not established |
| STOP | Harness termination, not a learned physical skill |

This is especially relevant to a scalar competence per ground skill. Our model's
scalar competence estimates face the same context dependence; it is not a problem
unique to the EES baseline. Physical failure and later task infeasibility remain
separate from the immediate binary label. A fixed grasp reduces one source of
cross-skill variation but does not eliminate variable scoop payloads.

After approval, test the assumption diagnostically at frozen sampler checkpoints:
use identical policy/exploration mode and budgets across source configurations,
depletion levels, robot/tool poses and payload sizes/distributions. Keep these
executions outside training and evaluation tasks. Compare conditional skill-success
rates with uncertainty; finite tests can expose heterogeneity, not prove equality
over every possible initiation state. If the approximation is poor, present explicit
context partitions or a context-conditioned competence model for approval. Do not
silently filter initiation states using successful outcome simulations, force a
canonical pile with unlogged resets, or certify Assumption 1 from a predicate table.

The runtime JSON gives the exact 695-feature schema. Robot features include base
pose/velocity, seven arm positions/velocities, and gripper command/velocity.
Each movable has position, quaternion in **qw,qx,qy,qz** order, linear/angular
velocity, and bounding-box dimensions. Fixtures, drawers and collision geometry
are included. Do not accidentally use Packed-4's **qx,qy,qz,qw** convention here.

Known omissions/ambiguity: actual articulated finger configuration, some actuator
targets/internal controller state, contact histories/warm starts, and tool cavity
geometry beyond its bounding box. Fixed geometry/materials can be versioned
metadata; changing them would require observation/schema review. Identical visible
positions can have different outcomes because velocities, grasp transforms or
payload contacts differ. Counts/occupancy images alone also alias different piles.
The full native vector reduces these omissions but does not prove Markov state.

Before freezing the contract, test whether the native vector plus known controller
state suffices. Log the richer simulator state for audit even if it is not a learner
input. Any added observation channel, history feature, or fixed reset of hidden
controller state requires explicit approval and equal provision to both methods.
Simulator solver caches are not automatically privileged features for the learner.

### Controller design aimed at Assumption 1

The user asked whether a controller set can be constructed to work with
Assumption 1. The engineering target is a measured approximation on the unchanged
native task, not an unproved guarantee of identical success probability in all
continuous initiation states. No new state restriction, payload partition or
controller parameter restriction in this section has been approved or implemented.

A sufficient construction would bring every admissible start to the same
distribution of all outcome-relevant physical state, up to verified symmetries,
and give the contact controller the same parameter-policy distribution in that
normalized frame. This could make success independent of the original start.
Matching robot pose alone is insufficient: payload arrangement, obstacles,
materials, velocities and preparation failures can retain dependence on the start.
Different parameter distributions chosen by a state-conditioned sampler can also
break the invariance even when the mechanics have been normalized.

Proposed four-controller architecture:

| Controller | Proposed physical preparation and boundary contract | Remaining state dependence |
| --- | --- | --- |
| GraspScoop | Fixed handle-midpoint grasp; native IK/transit; lift to a declared tool-relative carry posture; verify grasp transform and low relative motion | Accessibility, contact quality, planning/execution failures |
| Scoop | Begin with empty scoop and verified fixed grasp; use bin-relative coordinates, fixed approach orientation and speed profile; execute selected full-domain entry/depth; lift and physically settle into a declared carry posture | Source-pile arrangement, depletion, wall proximity, bin motion and grasp error |
| Pour | Verify grasp and settled payload; use a target-relative approach posture and clearance rule; execute independent full-domain destination/tilt parameters; observe release and native target predicates | Payload count/arrangement, target occupancy, grasp error and settling/transport failures |
| OpenGripper | Bounded physical opening/release with observed empty-hand postcondition | Mechanical jams and tool retention |

Preparation consists entirely of executed native actions, with its time, actions,
spills and failed attempts counted. A successful preparation phase must not cause
earlier failures to disappear from the macro skill's competence denominator.
If preparation becomes a separate skill, it needs its own initiation condition,
outcome and competence audit; moving a failure to a different name is not evidence
that the complete chain satisfies the assumption. No teleportation, hidden
refilling, pile normalization via state assignment, or unbounded retry-until-success.

The low-level base/arm primitives support prescribed targets with bounded velocity
and acceleration; this establishes useful ingredients, not successful scoop
controllers. Inspecting `Scoop._create_xml_element` reveals that its fourth/right
wall construction is commented out, so the physical tool has an open x edge,
despite comments describing four walls. Payload settling must respect that open
edge. Use the actual compiled geometry, not the stale description, to define the
cavity and contact controller. No upstream geometry is changed.

If Pour success still differs with payload count, a candidate refinement is
`Pour[k]` for k=1..10 loaded particles, sharing the same controller but with an
explicit count-conditioned initiation set and competence estimate. It does not
discard particles or require the robot to load exactly k. Count partitioning alone
does not fix within-count pose/contact variation and increases data requirements;
it needs approval and equal treatment in both methods. Keep the symbolic target
particle as the intended-effect witness unless batch-effect semantics are separately
approved. For Scoop, payload count partitioning does not solve source-layout
dependence: assess named-target location, neighboring particles and depletion.

Validation should freeze each parameter policy separately at untrained and trained
checkpoints, then compare full-controller success across predeclared context strata
on isolated diagnostic states. Choose a practical equivalence margin and adequate
sample size before declaring the approximation acceptable; a nonsignificant
difference in a small sample is not equivalence. Include preparation/admission
statistics, task chaining and learning headroom. If residual differences are large,
report the failure and ask whether to refine the abstraction, use a context-dependent
competence model, or change the benchmark. Strict invariance and native arbitrary
pile configurations cannot currently be promised together.

## Simulation and learning-integrity boundary

| Use | Proposed permission and audit |
| --- | --- |
| Kinematics/transit feasibility | Separate planning twin, IK/joint limits/static collision checks. Log query count, reason, time and source-state hash. No live-world/RNG mutation |
| Candidate admission | Validate declared parameter support and contact-free approach only. Stop before grasp/load/release dynamics whose success is learned. Never simulate particle-transfer outcomes or future task completion to admit/rank candidates |
| Intended contact | A fixed reviewed controller phase executed only after selection. Any bin-floor contact allowance is explicit, shared, and logged; no blanket collision disabling |
| Ranking | Existing learned classifier predictions or declared exploration, from executed training examples only |
| Practice execution | Selected controller on the persistent world. Native reward, all actual effects, skill outcome and budgets logged exactly once |
| Evaluation | Fresh isolated processes/native seeded initializations; frozen learner; no update, human-help, cycle, or cost-learning hooks; preserve parent physics/model/RNG hashes |
| Feasibility/headroom diagnostics | Separate seeds, output roots, data loaders and RNGs. Oracle search may examine outcomes only here; no examples, scores, successful sites or narrow bounds exported to training |

Both methods receive the same skill library, observations, proposals, admission
checks, classifier architecture, exploration epsilon, fit budget, native-action
limits, and planning deadlines. The chosen macro skills and resulting states may
differ; accepted-pool sizes must be measured, not asserted equal. Log raw proposals
as well as admitted proposals: feasibility conditioning changes their distribution.
Rejections and timeouts are not executed failure examples. A dispatched controller
that fails is an actual attempt. No bootstrap/calibration examples enter training.

## Proposed human intervention — requires confirmation

**HumanRestoreMovables**, fixed effort `H=5` per invocation, configurable and
nonnegative. One robot macro attempt costs 1. This follows Tossing3D's default
effort scale; 5 is a convention, not measured human time or an identified crossover.
Also log native steps, wall time, physical settling and actual human calls.

Availability: every practice decision boundary, including initialization, partial
progress, a full scoop, and a held or dropped scoop. No assistance during evaluation.
A native-action budget ending mid-skill leaves the actual state for the next cycle;
it does not authorize an automatic intervention.

Effects: remove the scoop from the grasp if held, command the gripper open, restore
both bins to sampled native initial poses, return **all ten particles**, including
already transferred/loaded ones, to the native source-pile pose distribution, and
return the scoop to its native initial-pose distribution. Preserve particle identity,
shape, mass and material; do not resample object geometry. Preserve current robot
base/arm positions, explicitly zero their velocities and set hold targets to those
positions. Set relocated-object velocities to zero and log settling under the
native initialization procedure. Any settling drift is recorded, not repaired.
Learned parameters, beliefs, datasets, cycle clock and remaining practice allowance
persist. One paid call consumes one macro decision; intervention-internal settling
is separately counted and included in the intervention cost, not robot training.

Sampling uses a dedicated logged reset seed and collision checks against the
current robot. It cannot test whether the robot will successfully grasp/scoop after
the intervention. No fixed successful recovery poses. Bound sampling attempts;
if placement is impossible under the unchanged robot pose, log a failed paid call
and halt for inspection instead of moving the robot secretly. A request from a
partially successful state deliberately loses progress under this proposal.

A progress-preserving intervention that only recovers stranded particles is a
different experiment, not a fallback. Its definition would need to specify “stranded”,
whether loaded/scored particles move, bin repair, tool recovery and cost scaling.

## Smallest defensible staged experiment

**Gate A — integration feasibility, before any comparison.** Freeze approved
controller/observation/intervention contracts; test PDDL/native predicate parity,
independent parameters, admission stopping before contact outcomes, action limits,
costs, held-object intervention, end-cycle accounting and persistent state.
Demonstrate at least one complete robot-controller solution from a native diagnostic
initialization, then check feasibility across distinct diagnostic starts (seeds
9000–9009; seed 9100 is the onboarding probe). If needed, allow at most 100 complete
diagnostic rollouts or two hours, whichever is reached first. This is bounded
feasibility evidence, not a task-success ceiling or proof of impossibility.

Use disjoint diagnostic parameter trials to check for both successful and failed
executions after admission. Sample the declared full support. If everything is
pre-solved, nothing ever succeeds, or outcomes are indistinguishable from missing
state, stop for review. Do not inject noise, change dynamics, bias target locations,
shrink the domain, alter the goal, or inflate budgets to manufacture headroom.

**Gate B — untrained baseline.** Fix and save ten native task seeds **100–109**,
including exact initial physical states and hashes, before evaluating either
method. Same ten tasks at every checkpoint. No pretrained samples or diagnostic
examples. Evaluate checkpoint 0 for both methods; call it the untrained baseline,
not “random native actions”. No baseline result exists yet. These become repeatedly
used evaluation tasks; they are not an independent final confirmatory test set.

**Gate C — diagnostic pilot.** Proposed two arms: EES and our Model B/local-trend
belief method, one paired training seed 0, one human cost H=5. Retain existing
inference priors and sampler semantics, explicitly record their resolved values;
proposed cost tradeoff lambda=.001 follows the Packed-4 method default and requires
agreement. Maximum **five cycles**, unrestricted skill practice and STOP, no forced
complete-task cycles, no resets at success/cycle/evaluation boundaries.

| Budget | Proposed value | Origin and scaling |
| --- | --- | --- |
| Practice | 20 macro decisions/cycle, 5 cycles/arm | Small diagnostic allocation; 100 decisions/arm is not a learning-power claim |
| Evaluation | 10 fixed tasks at checkpoint 0 and after every cycle; 20 decisions/task | 60 task executions/arm; 20 allows roughly grasp + several scoop/pour attempts, but must pass Gate A |
| Native actions | 2,400 per practice cycle or evaluation task; max 600 per robot skill, clipped by remaining budget | Explicit provisional allowance, 5× Packed-4's 480 because tool transport/contact may need longer paths; not calibrated evidence. Gate A must validate it before baseline |
| Candidate pool | Target 100 admitted candidates; at most 10,000 raw proposals | Current EES/default Packed-4 target; EES 100 proposals per requested candidate. Entire intended support stays eligible |
| Motion planning | 5 s per native planning call | Packed-4 watchdog convention; expiry is a diagnostic, not “physically impossible” |
| Proposal construction | 15 s soft batch deadline, max 5 s final in-flight call | Earlier feasibility-only Packed-4 safeguard; deliberately smaller than current Packed-4 180 s. Same in both methods; log partial pools |
| High-level planning | 30 s per decision; our search also capped at 1,000 iterations | Safety/watchdog proposal; iteration cap follows Packed-4. EES keeps its actual PDDL planner. Log work and timeout; do not substitute another planner |
| Exploration and fitting | epsilon=.5; existing shared classifier; 10,000 maximum fitting iterations | Main EES defaults. Fit at cycle boundaries; zero-attempt cycles still get explicit records |

Native budgets are proposals, not authorization to truncate an otherwise proven
controller arbitrarily. If a full-controller solution does not fit, report its
actual action counts and seek a budget/design decision before baseline. Record
both discrete native-action count and actual simulator time; they are not portable
between environments. Scaling to N particles should follow measured successful
payload size and path length, not blindly multiply all budgets by N.

Maximum pilot load is 120 evaluation episodes plus 10 practice cycles across two
arms; the configured total native-action ceiling is 312,000. This excludes diagnostic
search and intervention-internal settling, which have separately reported budgets.
No learning superiority inference is supported by one seed. A baseline at ceiling,
unexplained zero performance, persistent starvation or missing observations triggers
review before spending the five cycles. Ordinary unsuccessful task executions remain
reported failures; bugs/timeouts are separate invalid-run reasons, never silent zeros
or silently dropped tasks. STOP alone is valid and does not prove mastery.

**Gate D — longer comparison, separately authorized after the pilot.** Use paired
training seeds (initial proposal five), then consider human-cost conditions such as
H=1/5/20 and an unavailable-intervention control. Choose the scientific comparison
before running it; the first H=5 pilot cannot establish cost sensitivity. Do not
reuse oracle-optimized task sets or change held-out layouts after seeing scores.

## Logging, reconstruction, and deliverables

Extend the existing sibling-log convention rather than enlarge byte-stable
`stats.json` with every diagnostic. Collection writes state/data only, no video
encoder or per-tick camera rendering. Save:

- initial full physical and visual states, scene/assets/camera/light definitions,
  schema and object ordering; native observations and higher-precision physical
  arrays at each native step, with internal substep state where contact/release
  inspection requires it;
- MuJoCo qpos/qvel/act/ctrl, time and relevant mocap/equality/controller state,
  robot and movable transforms, velocities, and grasp/contact evidence; for 2D,
  all robot-part/body transforms, shapes, velocities, constraints and held offsets;
- run/phase/task/cycle/decision/native-step IDs; before/after state hashes; raw
  actions, ground skills, supplied parameters, achieved outcomes, rewards, effort,
  native counts, planning times, interruptions, STOP and termination reasons;
- every candidate proposal with rejection reason, accepted pool in order, exact
  feature rows and scores at selection time, selected index/parameters, exploration
  mode, tie/fallback mode, proposal RNG and controller choices. Log exceptions and
  exhausted pools as events even when nothing executes;
- per-cycle attempted/successful counts, competence and learning-rate estimates
  read directly from each live model, evidence counts, uncertainty where available,
  and update/exclusion reasons. Include checkpoint 0 and every zero-attempt cycle;
- resolved config, process/backend/library versions, all source commits plus dirty
  patches/untracked source, asset hashes, fixed seeds and separate RNG states for
  training, planning, intervention, evaluation and diagnostics;
- interruption-safe chunked logs and separate learner/physics checkpoints if exact
  resume is supported. A visual trajectory alone is not a resumable experiment.

Current `competence_log.py` only writes competence for already-created models; it
does not meet the requested all-boundary learning-rate record. Do not manufacture
an EES latent learning-rate posterior: log its native extrapolation and an explicitly
named finite-horizon slope `(predicted_c_after_k - c_now)/k`, with k/units, if that is
the available quantity. Log our model's eta estimate as a distinct belief. For a
never-instantiated estimate log its prior/default provenance or “not instantiated”,
without creating model state merely to observe it. Measured success rates use an
explicit zero-attempt null, not 0% or 100%.

**Reconstruction gate:** in a fresh process, restore logged visual states without
re-executing actions or planning. Compare live versus reconstructed frames/poses
for initialization, free motion, held tool, loaded tool, pour contact, failed skill,
human intervention and cycle boundary. Verify camera/assets/object IDs and timestamps,
record numerical/pixel comparison tolerances and results. Test recovery of a truncated
log. No main run until these cases pass. Initial screenshots made during onboarding
do not satisfy this gate. PyMunk's warning about hidden state specifically means
action replay or visible-state restoration must not be called exact physics resume.

Post-run deliverables: evaluation solved-of-ten curves versus cycles, actual
practice attempts and cumulative cost; effort/human/native-action/planning-time
curves; per-skill empirical success, competence and learning-rate/extrapolation
plots; action/intervention timelines; chaptered practice and evaluation videos
with task/cycle/decision boundaries, failures and intervention transitions.
Filled markers indicate cycles with practice attempts; hollow markers indicate
none (per skill on skill plots, any skill on aggregate plots). Distinguish measured
outcomes, model beliefs, and diagnostic reconstructions in every plot/caption.
Post-hoc videos represent logged trajectories, with interpolation/presentation pauses
identified; diagnostic rollouts are separate chapters/artifacts, never measured
training footage. Preserve partial failed runs and identify gaps explicitly.

## Resources, monitoring, and conditional ETA

Measured host capacity: 24 logical CPUs and 63,772,028 KiB RAM (~60.8 GiB).
The requested half-machine ceiling is therefore 12 logical CPUs and ~30.4 GiB RAM
in aggregate, not per job. Use a tighter shared experiment cap of 8 CPUs/24 GiB,
initially at most two single-threaded workers, with all diagnostics, evaluators,
fitters and later encoders inside the same accounting. CPU rendering avoids GPU
sharing assumptions. Current probes used one CPU and an 8 GiB address-space limit,
serially; unrelated jobs were untouched. Sandbox process listings do not expose
all host jobs, so host-level aggregate monitoring must be verified before a sweep.

After launch, monitor at least every 15 minutes and report failures immediately:
process/resource state, last completed event, boundary/learner progress, rejection
rates, exceptions, invalid state and log health. No automatic retries, resets,
schema/physics/support changes, or budget increases. A failed worker halts dependent
work for diagnosis; preserve its partial artifacts. Future monitoring is not
currently scheduled because no experiment is authorized or running.

Engineering estimate, **not measured experiment throughput**: after scope approval,
4–8 focused hours to reach a 3D controller-feasibility checkpoint; approximately
1–3 working days for a validated integration, untrained baseline and small pilot
if native planning supports scoop contact and the approved observation/skill
contract proves adequate. New collision/tool geometry, a missing-state redesign,
or failed feasibility removes that estimate; report the blocker at Gate A instead
of promising a run date. The 2D alternative likely reduces simulator complexity
but adds goal-compilation and state-restoration work, so it is not a guaranteed
faster fallback.

After the pilot starts, estimate remaining runtime from actual per-skill,
proposal/planning, training and evaluation timings, accepted-pool rates and active
worker count, with median and conservative recent timing ranges. For the full
two-arm pilot, use `120*t_eval + 10*t_practice_cycle + fits + startup + logging`,
adjusted for measured concurrency, not reset speed. Today's ~.08 s 2D and ~1.85 s
3D reset/render/one-action probes do not predict controller or learning runtime.

Already confirmed: 3D-o10 first and a single fixed grasp location. Approval still
requested: the remaining skill/observation contract and any parameter simplification;
the full-movables intervention (including lost partial progress, held-tool effects and H=5); and
approve Gate A followed by baseline/pilot only once the exact skill/observation
contract and feasibility/reconstruction gates are reviewable. Unresolved skill
ranges/classifiers or observation additions require a further decision before
learning begins. A longer comparison is a later approval.

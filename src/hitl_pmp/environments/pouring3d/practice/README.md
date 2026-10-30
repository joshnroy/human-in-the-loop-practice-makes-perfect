# Three-dumpling pouring practice

The CLI default is now `--pour-payload potsticker --payload-count 3`: three rigid
frozen-potsticker proxies, a clear handle-grasped pitcher, and a heavy skillet on
separate work surfaces. The registered native scene is
`kinder/FrozenPotstickerPour3D-o3-v0`. `--pour-payload shrimp` explicitly selects the
historical payload; programmatic construction retains its legacy default.

Pour1, Pour2 and Pour3 have separate sampler and competence identities. They share
one timed controller with three learned parameters: tilt (87–95 degrees), horizontal
sideways peak amplitude (0–25 mm), and feeding duration (0–30 s). The aim point is
fixed at the pan center. Joint feedback tracks motion; food counts do not terminate
or adjust the pour. A fixed initial full-load calibration supplies gravity compensation.
The task's requested count is used by scoring, not by the timed control trajectory.
Robot observations include arm joint positions and velocities for the standard sampler.

Human reset respawns only the food; robot, pitcher, pan, simulator time and controller
steps are preserved. Robot recovery uses physical parallel-jaw contact grasps and
transport, with a bounded oblique-grasp attempt after a recoverable approach/contact failure. These
attempts consume ordinary control ticks. Symbolic `Shrimp`/`Tray` names remain as
compatibility identifiers for food/pan, and do not select shrimp geometry.

Recovery plans the arm branch through base approach and descent before moving,
then tracks that joint path rather than resolving onto a different IK branch at the
hover point. Joint settling is bounded and checked from measurements. Grasp retention
uses a 2 N minimum normal-force target per finger, increasing the existing gripper
position command within its actuator limits. The simulator supplies contact/force
feedback; transferring recovery to hardware requires an equivalent retention estimate.
These are fixed recovery-controller settings, not additional learned pour parameters.

Recording model version 6 stores the payload choice. Replays without that field keep
the historical shrimp geometry. Current readiness and validation are documented in
[the launch protocol](../../../../../docs/experiments/pouring3d-pilots.md).

## Historical shrimp implementation and evidence

The remaining evidence in this section predates the potsticker integration and must
not be interpreted as dumpling success rates.

`--env pouring3d` runs the native TidyBot two-station pouring scene through the regular
experiment runner. The pitcher is grasped by its handle. A task requests exactly
`k` of `N` separate frozen-shrimp rigid proxies in a stovetop skillet, with the rest retained
in the pitcher. Full success also requires settled pieces, no spills, and the
pitcher placed upright at home with an empty hand.

The default receiver is now an approximate **Ozark Trail 15-inch cast iron
skillet**: 0.381 m diameter, 0.52705 m overall length, 0.066548 m height and
4.74004 kg empty mass. The product link and source dimensions are in
[skillet.py](skillet.py). The hollow round collider includes tapered walls and
handles. Its interior profile, wall thickness and mass distribution are estimated.
It remains a free body that can slide or tip. No pan-lift skill is offered: its
empty mass exceeds the real Gen3's published 4 kg mid-range continuous rating.
This skill restriction does not assert that the simulator enforces that rating.

Existing symbolic names (`tray`, `InTray`, `PickShrimpFromTray`) denote the receiving
vessel, now the skillet. All counts, samplers, competences and pour controls retain
their existing meaning. Whole-capsule containment and pickup clearance use the
round cavity, not its rectangular bounding box.

The default `--pour-layout stations` places the pitcher on a 0.74 m prep table
and the pan on a separate 0.84 m unheated electric stovetop, 2.1 m apart. Both
supports are static collision fixtures; the vessels remain freely moving. Base
travel is executed within the existing pick, pour and placement skills. Raised
poses use an angled handle approach and collision-aware arm targets to respect
the native arm limits. The prep-table pitcher home is 5 cm nearer the front than
the initial raised prototype: the previous home required the gripper body to
intersect the tabletop before seating. The base remains fully supported.
Transport observes compliant handle motion and carries the vessel with the base
when backing away for placement, avoiding arm overextension. Tilt and upright
return wait for motor tracking. Placement aligns above the table,
lowers near the support, and checks stability before opening the fingers.

Raised stations enable handle contact dimension 4, activating the existing
0.005 m torsional friction coefficient for a finite finger contact patch.
Sliding friction and motor force limits are unchanged. This is an explicit,
uncalibrated contact-model change; legacy floor variants retain dimension 3.
Recording model version 5 identifies the repaired raised configuration.

The original seed-70 raised check poured exactly two but failed placement.
The final carry repair passes pickup and placement in 8/8 attempts: two
consecutive cycles per seed 70/71/72/73, with no robot or vessel reset between
cycles. Zero robot-support or robot-pan contacts occur over 3,260,200 physics
substeps. Five required payload resets pass; three zero-delivery attempts are
already ready. All 8/8 end ready, preserving robot/vessels/time across reset
requests. Maximum final pitcher position error is 0.34 mm and orientation error
0.083 degrees; scoring tolerances remain 25 mm / 5 degrees.

These motion checks do not establish reliable pouring: unchanged fixed recipes
score 4/8 full tasks (Pour1: 1/2, Pour2: 1/4, Pour3: 2/2). Four attempts exhaust
their feeding budget with too few pieces delivered; no skill throws an error.
The per-attempt raw data, source hashes, figure and native video are in
`artifacts/stations-transport-placement-validation-v3/` in the launch worktree.
Separate elevated pan/worktop shrimp-pickup fixtures previously failed; this
transport repair did not establish individual-shrimp recovery. Historical floor
results below do not establish raised-station motion reliability.

**Subsequent shrimp-recovery motion repair:** both originally failing pan and
prep-table fixtures now complete physical pick/drop and finish ready. The arm
prepares in the clear aisle before approaching; pan avoidance is active for
empty-hand recovery, and IK must satisfy arm/wrist clearance as well as pose
accuracy. Finger proximity is allowed for the supported-object pinch. A failed
grasp retreats through the actually executed approach using motors, instead of
requesting a new vertical IK branch from a stale commanded posture.

The two fixture audits cover 265,000 physics substeps; the only support/pan
contacts are fingertip-pad contacts while pinching (maximum penetration 0.0021 mm
in the pan and 0.0253 mm on the table). Regression tests reject other robot
contacts with these obstacles and check physical retreat after an injected grasp
failure. Physics, grasp-success checks and scoring tolerances are unchanged.
These are selected recovery witnesses, not a reliability estimate for arbitrary
landed shrimp poses. Raw data, source fingerprints and native videos are in
`artifacts/stations-recovery-repair-v8/`; the end-to-end check is recorded separately
in `artifacts/stations-recovery-integrated-v3/`.

The final yaw correction keeps the finger-closing axis horizontal for rotated
shrimp. Its eight raised-environment regressions pass, and rerunning the original
two fixtures still completes both pick/drop cycles. This does **not** yet solve
robot reset after an actual pour: the latest seed-70 full cycle pours exactly two
and places successfully, but both naturally landed pieces fail pickup. Allowing
three attempts per piece also fails all six attempts; the robot retreats and the
payload-only human reset succeeds. An experimental multi-grasp retry controller
did not solve the full sequence and was removed. Broader grasp retention remains
an overnight-launch blocker. Current evidence, videos and the explicit failure
counts are in `artifacts/stations-recovery-repair-final/` and
`artifacts/stations-recovery-integrated-v4/`.

**Parameter proof of existence on the repaired raised layout:** successful
native executions exist for all three requests, using the existing sampling bounds:

| Request | Tilt | Commanded peak amplitude | Max feeding | Starting state |
| --- | --- | --- | --- | --- |
| Pour1 | 91 degrees | 15 mm | 30 s | seed 70, cycle 2 |
| Pour2 | 90.5 degrees | 8 mm | 30 s | seed 70, cycle 1 |
| Pour3 | 92 degrees | 15 mm | 30 s | seed 73, cycle 2 |

Each selected execution passes the full exact-count task, physical placement and
payload-only reset, with zero audited robot-support/pan contacts. These are
existence witnesses, not success-rate estimates. The Pour1 setting fixes an
underpour on an identical repeat start where the old recipe delivered zero.
The bounded parameter search did not solve Pour2 on that particular second-cycle
start; its witness above uses the fresh start. Controller, dynamics, scoring and
learner defaults are unchanged by this search. Do not use this table as a per-seed
policy. Full raw outcomes, source hashes, a figure and videos are preserved in
`artifacts/stations-parameter-existence-report/` and its linked trial directories.

`OnSurface` and `PickShrimpFromSurface` handle shrimp spilled onto either worktop.
They remain spills for task scoring. Human reset still respawns payload only.

**Historical results below used the rectangular floor tray.** They are not validation
of the skillet. `--pour-layout floor --receiving-vessel tray` preserves that geometry for comparisons;
recordings log the receiver and model version. Older recordings without a receiver
field replay as the tray; recordings without a layout replay on the floor. Human reset still moves only payload.

This is an **experimental benchmark**, not a claim that all counts are reliable.
On fresh seeds 90–94, frozen per-target recipes scored 3/5, 4/5, 5/5 for targets
1–3 with three pieces, and 0/5, 4/5, 4/5, 5/5, 5/5 for targets 1–5 with five.
Exact-one failures usually released two pieces together. These native feasibility
results precede the practice adapter's rubber-finger contact correction; the
integrated seed-70 pick/pour/place trajectory matches its predecessor exactly
over 965 recorded joint states. Broader integrated reliability remains to measure.

## Skills and parameters

| Skill | Discrete arguments | Continuous parameters |
| --- | --- | --- |
| PickPitcher | robot, pitcher, tray, ordered payload bindings | none |
| Pour1 / Pour2 / Pour3 | robot, pitcher, tray, fixed count binding, ordered payload bindings | tilt degrees, peak amplitude metres, maximum feeding seconds |
| PlacePitcherAtHome | robot, pitcher, count | none |
| PickShrimpFromTray | robot, selected shrimp, tray | none |
| PickShrimpFromFloor | robot, selected shrimp | none |
| PickShrimpFromSurface | robot, selected shrimp | none |
| DropShrimpIntoPitcher | robot, held shrimp, pitcher | none |
| Wait | none | none; one second |
| reset_shrimp | external human intervention; all payload | none |

During practice, the robot chooses among `Pour1`, `Pour2`, and `Pour3`.
Each is a separate skill with its own learned parameter sampler and competence
belief; all call the same physical controller. There is no externally assigned
practice count schedule and no parameter transfer between these skills. The
five-piece variant analogously exposes `Pour1` through `Pour5`.

The count binding is fixed by a static `Count1(k)`, `Count2(k)`, etc. precondition.
It connects each skill's effects to the existing `SplitCorrect(k)`/`Goal(k)`
facts; it is not a selectable count within that skill. Invalid bindings are also
rejected by action encoding. All three continuous controls remain learnable.

At evaluation, the task fixes `Goal(request_k)`. Pouring another count cannot
satisfy that goal. `--requested-counts 1 2 3` balances evaluation tasks across
counts; `--requested-count 2` fixes evaluation to count two. Neither option
restricts which count the robot may choose to practice. `ActiveRequest(k)` records
the task request but does not gate the pour skills. `SplitCorrect(k)` and `Goal(k)`
are measured from the actual physical split, independently of the chosen action.
The ordered payload bindings expose all starting poses to the learned sampler;
`PayloadOrder` permits a single ordering rather than factorial equivalent choices.

Pour's native limits are 60–130 degrees, 0–25 mm amplitude and 0–60 seconds.
Default sampling bounds are 87–95 degrees, 0–25 mm and 0–30 seconds, appropriate
only to the explicit default low-friction hypothesis. Override all three bounds
when studying another contact hypothesis. They include failed settings and are
recorded in the resolved run configuration.

Aim is fixed at the tray center; wiggle is horizontal, perpendicular to pouring,
at 1 Hz. Amplitude is the commanded peak wrist displacement; compliant grasp
and motor tracking determine the actual pitcher displacement. The stations pour
forward (+x) and wiggle sideways (y); the historical
floor configuration pours +y and wiggles x. Raised tilt and return commands are
capped at 10 degrees/s and wait for motor tracking; the legacy floor return is
45 degrees/s. Distinct
mouth crossings are observed at 10 Hz from ideal simulator poses. Return begins
after the requested number crosses or the feeding budget expires. The budget
excludes transport, tilt and return. Feedback never selects the three parameters
and cannot retract an escaping second piece.

## Predicates and outcomes

`HandEmpty`, `HoldingPitcher`, `HoldingShrimp`, `InPitcher`, `InTray`, `OnFloor`, `OnSurface`,
`Settled`, `PitcherHome`, `TrayHome`, `PayloadOrder`, `ActiveRequest`,
`SplitCorrect`, `Goal`, `NeedsReset`, `NoSpills`, and `ReadyToPour` are measured
from the current state. Home predicates include upright orientation. Exact split
includes settled full-shape containment; `Goal(request)` additionally requires
home vessels and an empty hand. `ReadyToPour` means ready to start Pick–Pour–Place.

The planner's intended effects never update physics or overwrite measured atoms.
Pour's success effects include the retained handle grasp and correct split.
Per-piece destinations after pouring are unknown symbolically until observed.
Robot recovery goes only toward the pitcher, so picking cannot bypass pouring.

Robot reset physically pinches the shrimp, checks two finger contacts through
lift, and drops it into the pitcher. Near-wall and crowded grasps can fail. A
failed attempt keeps the actual state, including a displaced or spilled piece.
Human reset uses the standard `reset_movables`/HumanOracle route, respawning only
payload, with no tray animation or changes to robot/vessels/time. Human actions
are separately priced by the practice protocol. There is no free reset between
practice attempts. A full sampled-start reset is a distinct, explicitly charged
harness intervention when a method requests one.

## Run an experiment

From the repository root, with its native dependencies installed:

```bash
scripts/with_env.sh python -m scripts.run_sweep \
  --env pouring3d --methods pomdp --num-seeds 5 --max-workers 4 \
  --results-root results/pouring3d-balanced \
  --shared-args '--preset pouring3d-balanced --practice-step-budget 9000 --measurement-interval-steps 9000 --record-motion all'
```

The preset disables assigned-goal pursuit during practice. It balances nine
held-out evaluations across requests 1/2/3, while the robot chooses its practice
counts. For the EES comparison use `--methods ees` and
`--preset pouring3d-balanced-ees`; its random-start exploration is enabled, but
it can still spend its budget practicing Wait after an unsuccessful recovery.

The POMDP uses its ordinary per-skill competence and sampler rules. The generic
`SkillPlansPracticeDomain` configuration averages single Pick–PourN–Place
attempts over the declared evaluation counts. The planning algorithm, training
updates, and cost objective are unchanged; multiple learned samplers use the
same rules independently. There is no count-conditioned competence override.
The deployment-value surrogate does not forecast successful recovery from a
wrong split. Evaluation executes the real skills with its control-step budget.

Use the workstation's memory-capped launcher for real runs. Practice and
evaluation have independent simulators and seed streams. Evaluation cannot
respawn the ongoing practice scene. Human skills are unavailable in evaluation.
The example cost file is an explicit experimental assumption: each robot tick
costs one unit, and human reset costs one unit and 100 control-tick equivalents
of time. At the native 10 Hz rate that is ten seconds; it is not a measured human
duration. Change the cost model to match the experiment.

`recording.jsonl` stores full native joint/velocity states when motion is enabled;
the standard `analysis.render_recording` entry point reconstructs those states
without stepping physics. `scripts.pouring3d_practice_demo` additionally records
the skill/reset demonstration with support motion at 4x and the whole Pour at 1x.

For direct native use, call `PourBackend.register()` before
`gymnasium.make("kinder/FrozenShrimpPour3D-o3-v2", requested_count=2,
surface_mu=0.1)`. The o5 variant is also registered. The runner does this lazily.
Native v2 selects the two stations; v1 preserves the floor skillet; v0 preserves the rectangular floor tray.

## Physics and validation scope

Each piece is a 3 g rigid curled compound of seven capsules. Material coefficient
`surface_mu` is an uncalibrated sensitivity input; freezing itself is not
simulated. For the initial skillet check, shrimp–pan sliding friction stays at
the same explicit 0.1 hypothesis as shrimp–pitcher contact, isolating the geometry
change. This is not a measured cast-iron coefficient. The pan geom declares sliding friction 0.6; native maximum mixing with the
default support coefficient 1.0 gives an effective support contact coefficient
of 1.0. These are assumptions. Explicit shrimp–pan contacts use a 2 ms time
constant to prevent the legacy softer contact penetrating the 4.5 mm pan base. Rubber finger pairs use 0.6, matching the native lower pad's declared
coefficient, independently of plastic contact friction. No weld, adhesion,
ice-clumping, melting or food deformation is modeled. Practice model version 2 uses a 22 mm spawn ring for three pieces, removing the
original roughly 0.5 mm initial wall intersection. The five-piece scene uses
separated initial layers and gravity settling. Historical feasibility results
above predate this packing repair; they are not measurements of version 2.

Targeted checks cover discrete/continuous API separation, actual planner plans,
physical tray and floor pickup/drop, payload-only human reset invariants, and
interruption without state restoration. A short EES run checks the real learning
and evaluation pipeline. It does not establish learning improvement; report
that separately using held-out per-target results and reset-generated packings.

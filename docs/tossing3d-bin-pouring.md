# PourCubeOutOfBin

Tossing3D now offers a parameterless robot recovery skill in both layouts. It is
applicable when the hand is empty and the cube is in an upright bin on the robot's
side of the barrier. Picking may also be plannable: both actions are offered, along
with paid resets when enabled, and the method selects between them. The controller
also permits pouring a pickable cube. `PickNotPlannable` has been removed.

Preconditions encode the execution requirements of a skill, never which action is
preferred over another feasible action. These runs evaluate the algorithm's choice;
a rule such as "pour only if picking is impossible" would prescribe that choice.

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

The pouring-disabled ablation also works with the always-ask wrapper: unavailable
optional skills carry no practice belief. Prior seed slots are retained when a
skill is disabled, and pouring is appended after the existing skills so it does
not reseed their priors. Historical planner-value tests explicitly disable pouring
to retain the action library their pinned values describe. Pouring-enabled tests
continue to offer both pick and pour.

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
On current main's newer pick controller, A, C and D also have a pick plan. The
same physical tests now exercise pouring despite those feasible picks; F still has
no pick plan. Offline tests require both pick and pour to remain available in both
layouts, including the belief-space search's action enumeration alongside paid
resets. Automatic EES, expectimax and Curtis recovery checks use F, with outcome
learning and charged execution ticks.
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

## Action availability audit

The pouring-only-if-picking-fails rule was a preference encoded as feasibility.
It is removed from both STRIPS and dispatch, together with its otherwise unused
`PickNotPlannable` predicate. Applicability tests now require picking and pouring
simultaneously; a policy test must not require one particular choice when both
are feasible. The older pick-only retrieval test explicitly disables pouring.

The experiment has three parts: the environment, the pre-programmed skills
(including predicates), and the algorithm. The standard comparison holds the first
two fixed and varies the algorithm. RoboCode is the exception: it is not supplied
with the pre-programmed skill layer either. Preconditions describe skill semantics,
execution feasibility and the valid environment contract; they must not prescribe
which of two feasible skills the algorithm should choose.

The following are accepted parts of that setup, not unresolved audit findings:

| Rule | Assessment |
| --- | --- |
| `PickPlannable` on the pick itself | Execution feasibility: asks the same deterministic motion planner used at dispatch. It does not predict grasp success or disable retries after an execution failure. |
| `BinOnGround` on every robot skill | Environment-validity contract: the bin should remain upright during ordinary operation. A tipped bin is an invalid/problem state that can require human recovery. Keep the global gate, including on picks and empty-hand opening. |
| `ClosedEmpty` on `OpenGripper` | Skill semantics: this is the empty-hand gripper-opening skill, not a drop-cube skill. Keep its empty-hand precondition and current effects. |
| `OnGround` on picks | Retained benchmark restriction, including the explicit user requirement that a cube supported by the robot base must not be picked. Bin and rim retrieval have their own supported pick operators. |
| Starved parameter pool cache | Algorithm behavior: after a finite pool yields no accepted parameters, EES and belief-space search suppress that state/action pair for the practice session to avoid getting stuck on a failing skill. Keep this mechanism as part of the algorithm; it is not an environment or skill predicate. |

This audit covers the Tossing3D skill models, provider, pouring dispatch guards,
and the practice planners' action enumeration. It does not establish that every
feasible physical manipulation is represented. The accepted rules above remain
unchanged by the pouring correction; no further fixes to them are required by
this audit. The reset remains available alongside robot skills; it is not
reserved for cases where all robot alternatives fail. Always-ask arms are explicit
baseline policies, rather than environment preconditions.

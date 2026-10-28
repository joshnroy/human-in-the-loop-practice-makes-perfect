# Frozen-shrimp pouring practice

`--env pouring3d` runs the native TidyBot floor-pouring scene through the regular
experiment runner. The pitcher is grasped by its handle. A task requests exactly
`k` of `N` separate frozen-shrimp rigid proxies in the tray, with the rest retained
in the pitcher. Full success also requires settled pieces, no spills, and the
pitcher placed upright at home with an empty hand.

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
| Pour | robot, pitcher, tray, selected count, ordered payload bindings | tilt degrees, peak amplitude metres, maximum feeding seconds |
| PlacePitcherAtHome | robot, pitcher, count | none |
| PickShrimpFromTray | robot, selected shrimp, tray | none |
| PickShrimpFromFloor | robot, selected shrimp | none |
| DropShrimpIntoPitcher | robot, held shrimp, pitcher | none |
| Wait | none | none; one second |
| reset_shrimp | external human intervention; all payload | none |

During practice, the robot chooses the count by selecting a grounded `Pour(k)`
skill. All counts are available to the practice planner; there is no externally
assigned count schedule. A single learned Pour sampler receives the chosen count
as an input along with the physical state and predicts success for the three
continuous controls. Count is a discrete grounding argument, not a fourth
continuous control.

At evaluation, the task fixes `Goal(request_k)`. Pouring another count cannot
satisfy that goal. `--requested-counts 1 2 3` balances evaluation tasks across
counts; `--requested-count 2` fixes evaluation to count two. Neither option
restricts which count the robot may choose to practice. `ActiveRequest(k)` records
the task request but does not gate the Pour skill. `SplitCorrect(k)` and `Goal(k)`
are measured from the actual physical split, independently of the chosen action.
The ordered payload bindings expose all starting poses to the learned sampler;
`PayloadOrder` permits a single ordering rather than factorial equivalent choices.

Pour's native limits are 60–130 degrees, 0–25 mm amplitude and 0–60 seconds.
Default sampling bounds are 87–95 degrees, 0–25 mm and 0–30 seconds, appropriate
only to the explicit default low-friction hypothesis. Override all three bounds
when studying another contact hypothesis. They include failed settings and are
recorded in the resolved run configuration.

Aim is fixed at the tray center; wiggle is horizontal, perpendicular to pouring,
at 1 Hz. Tilt and physical return commands are 10 and 45 degrees/s. Distinct
mouth crossings are observed at 10 Hz from ideal simulator poses. Return begins
after the requested number crosses or the feeding budget expires. The budget
excludes transport, tilt and return. Feedback never selects the three parameters
and cannot retract an escaping second piece.

## Predicates and outcomes

`HandEmpty`, `HoldingPitcher`, `HoldingShrimp`, `InPitcher`, `InTray`, `OnFloor`,
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

The POMDP uses separate `Pour/1`, `Pour/2`, … competence beliefs and one shared
Pour parameter sampler. An example advances the attempted count's forecast
learning clock; the shared sampler trains on all counts. The value model averages
a single Pick–Pour–Place attempt over the declared evaluation counts. It does
not forecast cross-count transfer or successful recovery from a wrong split.
These are explicit planning approximations; evaluation executes the real skills
with the configured control-step budget.

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
`gymnasium.make("kinder/FrozenShrimpPour3D-o3-v0", requested_count=2,
surface_mu=0.1)`. The o5 variant is also registered. The runner does this lazily.

## Physics and validation scope

Each piece is a 3 g rigid curled compound of seven capsules. Material coefficient
`surface_mu` is an uncalibrated sensitivity input; freezing itself is not
simulated. Rubber finger pairs use 0.6, matching the native lower pad's declared
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

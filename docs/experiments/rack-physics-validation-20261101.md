# Rack physics repair and task-validity audit — 2026-11-01

This investigation supersedes overnight launch preparation. No learner or
overnight comparison is authorized by this validation. The original shelf-to-box
environment and all earlier result directories are preserved.

## Disposition

**Implemented and physically feasible in the tested rigid-body simulation;
not cleared as a nontrivial parameter-learning benchmark.** The obvious straight
0.03 m/s sweep solves the fixed deployment layout. A single straight 0.06 m/s
sweep solves that layout and all three tested human-return layouts. No learner
runs were launched, and no geometry or deployment distribution was changed to
manufacture difficulty.

| Required criterion | Current evidence | Assessment |
| --- | --- | --- |
| Stable contact physics and actual tool manipulation | Controlled passive checks, native 5/5 delivery, scraper pickup/return, two integrated individual food resets | Pass on tested cases; material assumptions remain uncalibrated. |
| Honest binary success and spills | All five unheld, supported, fully contained and settled; shared practice/evaluation check; counter/floor classification tests | Implemented. Stable overhang remains a failure pending an explicit scoring decision. |
| For each initial state, some successful parameters | Four measured layouts, two speeds each; each has a native full-cycle witness | 4/4 sampled states covered, not a guarantee over continuous states. |
| Nontrivial reason to practice sweep parameters | Straight/slow solves deployment; straight 0.06 m/s solves all four layouts | Not established; a fixed-policy baseline suffices on these samples. |
| Feasible autonomous and costed human resets | Two successful individual returns in one cycle; finite human batch/tool alternatives | Partially validated; not full five-food autonomous reset coverage. |
| Practice reaches relevant deployment states | Deployment initialization is fixed; human returns vary; robot-return coverage not measured | Distribution/coverage gate remains. Training-reset difficulty is not deployment difficulty. |
| No human assistance during deployment | Native rejection without state/cost mutation; symbolic availability restrictions | Validated. Autonomous food pick/place remains available. |
| Preserve original benchmark and algorithms | Additive environment/domain/CLI registration; original shelf implementation unchanged | Preserved; no core learner changes. |
| Hardware release and snag behavior | Rigid ellipsoids and assumed contact coefficients only | Not established; requires cooled cooked-food tests. |

## Historical fixes and what the port missed

The reference is the main checkout's
`scratch/prototypes/sweeptable-kitchen/src/hitl_pmp/environments/sweeptable3d/`:

- `balls.py:40`: six-dimensional contacts and 0.003 m rolling friction on food
  and supporting surfaces. The rack port initially retained three-dimensional
  contacts, which do not apply rolling resistance.
- `geometry.py:89` and `README-kitchen.md:89`: blade sliding friction 0.25 with
  contact priority 1, avoiding the higher-friction support/food override. The
  rack port had 0.4 without priority. The handle's grasp friction is independent.
- `contact.py`: support-force feedback and unloading near the edge. This has
  **not** been blindly enabled: a movable wire rack is not the old fixed board.
  Any downward-load controller needs separate rack-motion/clearance validation.

Those settings were simulation assumptions, not measured cooked-food materials.
The old task used capsules and a fixed, rimmed tray; this task uses ellipsoids,
a movable wire rack, and a movable flat slab. Previous success does not transfer
automatically.

MuJoCo's [contact documentation](https://mujoco.readthedocs.io/en/stable/computation/index.html)
describes the additional rotational friction constraints in six-dimensional
contacts. Merely specifying a rolling coefficient is insufficient. Moreover,
the inherited shelf environment gives `box_0` priority 1 at initialization:
changing only the food's contact dimension left **actual tray contacts at 3**.
The fix therefore configures both food and supporting rack/tray geometry.

## Controlled passive validation

The diagnostic places one food proxy over the actual tray with a specified
tilt, drop, and 0.03 m/s initial horizontal velocity. This is **privileged setup**,
not a robot delivery. Each case restores state and clears acceleration warm-start.
The run lasts five simulated seconds; the last recorded sample is at 4.9 s.

| Initial tilt / drop | Old final speed / angular speed | Repaired final speed / angular speed |
| --- | --- | --- |
| 0° / 1 mm | 0.00205 m/s / 0.163 rad/s | 0.000000087 m/s / 0.0000069 rad/s |
| 15° / 13 mm | 0.00153 m/s / 0.117 rad/s | 0.0000137 m/s / 0.00108 rad/s |
| 45° / 13 mm | 0.158 m/s / 12.6 rad/s | 0.0000281 m/s / 0.00201 rad/s |

Evidence: `artifacts/rack-passive-baseline-controlled.json` and
`artifacts/rack-rolling-restored-passive-v2.json`. Both retain the exact config and
actual contact dimensions; every repaired recorded food contact is dimension 6.
Three small native regression tests additionally use production geometry and the
inherited tray priority. This shows improved dissipative settling, **not** a
calibrated model of dough adhesion, release, deformation, or wire snagging.

`rack-rolling-restored-passive.json` is the intentionally retained unsuccessful
food-only patch: it compiled food dimension 6 but actual tray contact remained 3.

## Robot transfer diagnosis

| Revision / same straight 0.03 m/s sweep | Observation |
| --- | --- |
| Original | All five ultimately spilled to the floor. |
| Rolling resistance on food + supports | Three settled on tray; two overhung the rear edge. Pickup, stroke, and tool return completed. |
| Add historical lower-friction blade | Three on tray at sweep completion; this alone did not resolve the remaining two. |

The trace localizes the remaining error: the rear food is supported by the blade
near the tray edge, and the old lift simultaneously retreats 60 mm toward the rack.
The food consequently ends near x=1.355 m, overhanging the tray's x=1.3625 m rear
edge. It must not be scored as a completed delivery. The controller repair lifts
vertically without retreat, **then** waits for food settling. It retains the real
friction grasp, pose feedback, rack-displacement stop, IK, and collision guards.
Native validation of this revision is recorded separately below when complete.

The vertical-release-only run (`artifacts/rack-release-v4/report.json`) gave 4/5
immediately after the sweep, and 5/5 only after the long tool return. That is not
adequate evidence of immediate successful sweep effects. The final follow-through
therefore clears the entire blade past the receiving edge by one maximum food
diameter plus a 10 mm tracking margin (55 mm here, rather than the old fixed
25 mm). This is controller geometry, not a changed tray/food/initial distribution.
The unchanged reach and collision checks still apply.

**Native witness:** `artifacts/rack-followthrough-v5/report.json` is complete.
At 0° and 0.03 m/s, pickup completed in 438 control steps, sweep/release/settling
in 913, and scraper return in 1,556. All five food proxies were settled on the tray
both immediately after `SweepRack` and after `ReturnScraper`; the scraper was
physically supported at home. No human or food-pose writes performed this transfer.
The initial empty-hand `SweepRack` call in this diagnostic is a deliberate
precondition probe, not the physical transfer attempt.

The earlier faster 0.12 m/s revision's failure and failed cluttered reset are
retained in `artifacts/rack-physics-video-v3/`. It hit a fixture-clearance guard;
three food proxies ended on the tray. Picking a touching food pair was blocked
by carried-food/neighbor clearance, and explicit human recovery was used. This
is not a current-revision parameter comparison: the release path subsequently
changed. Do not mix revisions into one learning curve or success grid.

Practice predicates and task success now share `settled_on_tray`: actual support,
full containment, not held, finite speed ≤0.02 m/s and angular speed ≤0.5 rad/s.
`location()` still distinguishes raw physical location from settled delivery.
This is a current-state check, not a guarantee of future stability or a dwell timer.
No core learning/planning algorithm was changed.

## Criteria: do not conflate these questions

### Reset contact and placement follow-up

The isolated reset probe exposed two additional omissions. The inherited finger
pads have priority 1 and default contact dimension 3, overriding the food's
dimension 6. Changing closure from 0.8 to 0.75 or 0.85 did **not** prevent the
observed rotational slip (`rack-reset-grip75`, `rack-reset-grip85`). The task-local
backend now configures the **pads only** with dimension 4: torsional resistance
from the existing friction coefficients becomes active, with no welded grasp,
pose writes, altered finger geometry, or weakened bilateral-contact checks.
`gripper_contact_dimension=3` reproduces the earlier contact model; the setting is
consistent across skills, not toggled temporarily to make a pick pass.

With these contacts, the actual tray pickup succeeded in 455 control steps.
Placement then correctly stopped at a rack-wire clearance check because its old
hand/food offset was stale after transport. Placement now remeasures the carried
food's center and extent during bounded, guarded descent, instead of commanding
the stale target or relaxing the 3 mm support tolerance.

`artifacts/rack-measured-food-reset/report.json` records **successful physical
pickup (455 steps) and rack return (381 steps)**. All five finish on the rack.
Its real-time video is `artifacts/rack-measured-food-reset/two-views.mp4`. The
initial single food was explicitly placed on the tray for the test; the pickup
and return are robot actions. This is one feasible reset witness, not proof of
full-batch clutter recovery. The same new pad model retains successful 5/5 sweep
and scraper return in `artifacts/rack-soft-finger-transfer/report.json`.

The final full-cycle video, repeat coverage, and regression reports below use
both fixes together. Earlier reports lacking `gripper_contact_dimension` used
dimension 3; replaying their configs under newer defaults is **not** the same
physics. Use the recorded revision and explicitly set that field when comparing.

| Criterion | Evidence required | Audit at start of repair |
| --- | --- | --- |
| Credible physics | Stable passive contact and plausible complete robot manipulation | Passive settling repaired; full cycle still under validation. |
| Honest binary goal | All five actually settled on tray; partial delivery is failure | Implemented with shared practice/evaluation semantics and regression tests. |
| Solvable deployment states | For each sampled state, at least one parameter pair succeeds | Not yet established. One constant need not solve every state. |
| Useful parameter learning | Compare nominal fixed policy, best shared constant, and per-state parameter search | Not established; constant straight/slow is a serious baseline. |
| Meaningful recovery choice | Autonomous returns can work; human batch/tool help is explicit and costly | Individual reset witness exists; clutter failures and human fallback are retained. Revalidate after material changes. |
| Practice can reach relevant states | Actual reset actions cover the deployment/practice distribution | Human-return distribution exists; robot-return coverage remains unproven. |
| Autonomous deployment | Human actions rejected without state changes; legitimate pick/place alternatives retained | Contract and symbolic tests cover rejection; no alternative removed to force sweep learning. |
| Reproducibility / isolation | Configs, failures, limits and independent old benchmark preserved | Separate environment, append-only artifacts, bounded diagnostics. |
| Hardware validity | Real cooled cooked dumplings, measured rack/tool/material properties | Not established by simulation; supervised hardware tests remain essential. |

## Is parameter learning necessary?

The sweep parameters are **angle and speed**, not lateral offset. Food begins in a
fixed layout; changing initialization seeds alone does not vary it. Human batch
returns produce different compact layouts. The centered tray is wider than the
blade, and both nominal initial and human-return lateral extents fit behind a
zero-angle blade. Thus a fixed straight, slow sweep is geometrically plausible.
Robot-return placement candidates can span a wider region and require separate
physical coverage checks.

The previous kitchen report `artifacts/kitchen-final-v3-solve/report.json` in the
main checkout had a shared (-19°, 0.30 m/s) solution on all three tested seeds;
(-20°, 0.30) solved two and straight sweeps solved none. That demonstrates
parameter sensitivity, **not** necessity for a state-dependent parameter policy.

For this revision, compare on identical actual states:

1. The obvious fixed straight/slow baseline.
2. Every shared constant in the bounded tested parameter grid.
3. A per-state oracle that may choose a different tested pair for each state.

If a fixed nominal policy reliably succeeds, report that the task is weak for
skill-parameter learning; do not preserve bad physics or silently randomize an
uncontrollable bin to manufacture difficulty. Learning a non-obvious shared
constant can still be parameter optimization, but is a different claim from
requiring state-conditioned parameters. Finite-grid failure is not proof that no
continuous solution exists. Native learning remains on hold pending this audit.

## Videos and limits

### Current full physical sequence

`artifacts/rack-final-production-video/two-views.mp4` is the completed **463-second,
1×** wide-plus-close video. Its report records no source change during execution.
It uses the restored rolling resistance, soft-finger contacts, measured placement,
and corrected release/follow-through together:

| Action | Control steps | Observed outcome |
| --- | --- | --- |
| Pick scraper | 438 | Real friction grasp succeeds. |
| Sweep, release, settle | 910 | All five settled and fully contained on tray. |
| Return scraper | 1,540 | Tool home; all five remain delivered. |
| Pick / return food 0 | 396 / 380 | Individual physical return to rack succeeds. |
| Pick / return food 1 | 374 / 397 | Second individual physical return succeeds. |
| Human batch return / tool recovery | Explicit interventions | Costs 5 each; cumulative interruption cost 10. |

The human completion after two robot returns is a **scripted demonstration**,
not evidence that returning the remaining three autonomously is impossible.
There are no privileged food-placement writes in the robot delivery or individual
returns. Human batch return is explicitly a state-changing intervention, not a
robot trajectory. The final all-on-rack state is the reset, not the delivery goal.
`sweep-realtime.mp4` in that directory is a 25-second unaccelerated excerpt.

### Delivery containment must not disable recovery

The slow sweep on human-return seed 1 leaves one proxy stably tray-supported but
overhanging the rear edge by about **3.9 mm**. This is not a floor/counter spill.
The current strict full-containment goal continues to reject it; whether stable
support should instead count as delivery is an explicit pending task-definition
decision, not a physics repair.

The audit found a separate bug: deriving `FoodAvailable` solely from those strict
location labels made this overhanging food unavailable to pick. The backend and
pick controller now use actual unheld physical support to permit a grasp attempt,
independently of delivery credit. Regression tests preserve the failed goal while
allowing pick preconditions, and still reject unsupported food. This does not
guarantee reachable or collision-free grasps; the normal guards remain active.
The full video above predates this availability-only fix, not the physical fixes.
The final coverage matrix and recovery probes use the newer source digest and
must be reported separately.

`artifacts/rack-overhang-recovery-video/` is a deliberately retained **nonmatching
reproduction**: despite its directory name, it used zero readiness delay while the
matrix uses five seconds. It delivered all five, then food 3's reset attempt was
rejected by actual base/arm versus counter/leg clearance. It demonstrates a hard
reset, not the matrix's overhang failure. Its 303.8-second video must not be
captioned as a failed sweep. Initial human setup costs 5, subsequent batch/tool
recovery adds 10; total human cost is 15. A timing-matched reproduction is recorded
separately rather than relabeling this successful delivery as a failure.

**Timing-matched failure video:** `artifacts/rack-matched-seed1-video/two-views.mp4`
is complete (310.2 seconds, 1×, wide and close cameras, isolated dependencies,
no source change during execution). With the same five-second readiness setting,
human seed 1 and 0°/0.03 m/s, it reproduces **four fully contained foods and one
tray-supported overhang**, both after sweep and after tool return. There is no
floor/counter spill. The early zero-step rejected sweep is a labeled readiness
probe, not the physical sweep. Food 3's subsequent pick is now legitimately
available but is rejected before motion because the arm would intersect the
counter. Human batch/tool recovery is explicit and raises cumulative cost from
5 for setup to 15. The normal collision checks were not relaxed to make this pass.
`failure-realtime.mp4` is a 30-second excerpt showing the transfer and retained
overhang; consult the full video for the reset failure and human intervention.

### Historical fixed-policy check before the finger-contact repair

`artifacts/rack-fixed-policy-validation/summary.json` completed all three trials:
human batch-return seeds **0, 1, and 2 each succeeded at the same 0°, 0.03 m/s**.
Each used a fresh native environment, a costed human layout setup, robot pickup,
sweep, scraper return, and additional settling. All three had distinct measured
food layouts and no skill errors. Together with the original-layout witness,
this is **four sampled layouts solved by the obvious fixed policy**, not a learned
policy. `trial.json` files retain actual initial/final states, per-phase traces,
costs, controller budgets, and the frozen sweep-source digest.

This is positive feasibility evidence and negative evidence for the proposed
parameter-learning motivation. It does **not** establish all continuous reset
layouts are solvable, nor does it establish robot-only reset reachability.
There is no basis yet for claiming this task needs skill-parameter learning.
Recommended disposition: retain this as a physically grounded planning/recovery
investigation, but do not launch it as a demonstrated nontrivial parameter-learning
benchmark. A harder scene/distribution would be a separate, explicit design
decision, not a silent physics change.

The subsequent finger-contact repair changes grasp physics, including the held
scraper. These counts are **not** final-revision evidence and must not be pooled
with the final repeat below. The earlier success/failure evidence is preserved.

### Final fixed-policy repeat

`artifacts/rack-final-fixed-policy/summary.json` uses the final soft-finger contacts
and measured-food placement controller. At **0°, 0.03 m/s**, human-return seeds
**0 and 2 succeed; seed 1 fails (4/5 fully contained)**. All controllers finish,
the scraper returns home, and the failing food is physically supported by the
tray but overhangs its rear edge. The goal correctly remains false. The other
speed/angle checks retain this exact layout rather than rerandomizing until it
passes.

The same-revision grid is now complete in `artifacts/rack-final-matrix-a/` and
`artifacts/rack-final-matrix-b/`. Both have identical config and sweep-source
digest `6fb55761aa9a1f28fa481c2e0a7197277519910b2481c8db8622f4b774ca3914`.
Every repeated state matched its measured initial positions, orientations,
velocities, support, and locations; all eight trials completed without
errors and returned the scraper home. The aggregate is
`artifacts/rack-final-policy-comparison.json`.

| Measured layout | Straight 0.03 m/s | Straight 0.06 m/s |
| --- | --- | --- |
| Fixed deployment initialization | 5/5, tool home | 5/5, tool home |
| Human batch-return seed 0 | 5/5, tool home | 5/5, tool home |
| Human batch-return seed 1 | 4/5 fully contained; 3.9 mm overhang | 5/5, tool home |
| Human batch-return seed 2 | 5/5, tool home | 5/5, tool home |

Thus the per-state oracle covers 4/4, but so does the **same constant** 0°,
0.06 m/s. The slower baseline covers 3/4. These are deterministic finite witnesses,
not estimated generalization rates or a statistical treatment-effect claim. Speed
can affect containment, but this does not establish a need for state-conditioned
parameters or a substantial learning problem. Angle optimization is unnecessary
for these four witnesses; its global usefulness has not been disproved.

Separate earlier same-physics angle probes on human seed 1 are retained:
−10°/0.03 m/s delivers all five but `ReturnScraper` stops on a **6.1 mm hand
tracking guard**; +10°/0.03 m/s returns the tool but leaves one food supported
across rack/tray and not fully delivered. Do not describe the former as a
five-food transfer failure or pool these older tool-digest records into the grid.

Crucially, **deployment initialization is still the fixed original layout**;
different environment seeds do not create different deployment food arrangements.
That original layout is solved by the hardcoded straight/slow policy. Difficulty
on randomized *training reset* layouts is not evidence that deployment requires
parameter learning. Matching deployment to a richer, practice-reachable state
distribution would be a task-design change requiring an explicit decision; it
has not been silently introduced here.

### Completed contract checks

Final isolated-dependency regression: `artifacts/rack-isolated-regressions.txt`
reports **561 passed, 4 skipped**. The two additional tests were observed failing
before fixing the coverage factory's overly permissive shared-checkout fallback.
It now rejects dependencies outside this worktree before constructing a simulator.
Final `mypy src` passes across **247 source files**; targeted Ruff/format checks,
the import-layer contract, and `git diff --check` also pass.

The original matrix used clean shared reference checkouts at the worktree's exact
pins: KINDER `0dac89eb57aceab4f655b108411be5efefb466c1`, kinder-baselines
`427ad6cdfcc79efb489ebabc8fd39d77e104bb4e`. That was not the desired isolation,
even though the tracked source matched. These dependencies and cached scene assets
are now populated locally without altering either pin or shared source. The
factory-only change produces digest
`b051a09e2003c257e6c140a28e56d4c3adc41d01ad599d63f71cec9faa3da217`.

`artifacts/rack-isolated-deployment/` repeats fixed initialization at 0°/0.03 m/s;
`artifacts/rack-isolated-reset-layout/` repeats human seed 1 at 0°/0.06 m/s.
**Both complete successfully**, with initial and final poses, velocities,
locations, and support matching the original corresponding matrix trials within
the recorded 1e-8 absolute comparison tolerance. The other six cells were not
rerun after the factory-only change; do not claim an eight-cell isolated rerun.

`artifacts/rack-final-contracts.json` repeats all **21/21** native contracts after
the availability repair. `artifacts/rack-availability-regressions.txt` records
**559 passed, 4 skipped**, including the opt-in native cutoff test and selected
original shelf, CLI, and generic step-protocol regressions. Ruff and formatting
checks pass, mypy passes on 22 new environment/domain source files, and the import
layer contract remains kept. These are targeted checks, not the whole repository
suite or learner-performance validation.

`artifacts/rack-contracts-repaired.json` passes **21/21** checks on the repaired
model: both surface kinds, live geometry, food support and counter/floor spills,
timer/wait, movable rack/tray, training intervention costs, and deployment rejection
without state/cost changes. These privileged contract checks are not full robot
transfer witnesses. The opt-in native three-control-step cutoff test also passes
(`artifacts/rack-native-cutoff-repaired.txt`).

The real-time complete repaired scene video is
`artifacts/rack-validated-video/two-views.mp4` (321.7 seconds). Its 25-second
`sweep-closeup-realtime.mp4` excerpt is an unaccelerated selection of that video.
The full run contains successful 5/5 delivery and tool return, **two failed
autonomous food-reset attempts** (base/counter-leg clearance), then labeled,
costed human batch and tool recovery. It is not a fully autonomous reset-cycle
success. The final all-on-rack state is the human reset, not the delivery goal.

The old single-food reset witness must not be reused as proof for the new material
model: `artifacts/rack-single-reset-repaired/report.json` and the real-time
`artifacts/rack-reset-contact-video/` repeat show bilateral contact loss during
lift. Strict contact checks stop rather than claiming the food was returned.
Bounded gripper-command calibration is investigated separately; difficult clutter
and circulation cases continue to require human fallback.

The previous background movie was accelerated 5×. New renderer output defaults
to explicitly labeled **1×** playback, with wide and close views. Failures and
human intervention remain in reports/videos; human actions are labeled as such,
not portrayed as autonomous robot resets. These are KINDER's existing room assets,
not an accurate reconstruction of a real kitchen.

Hardware checks still needed: cooked-food release/snags and damage, rack height,
mass and displacement, scraper surface/grasp/clearance, tray graspability and
support, and a hygienic spill protocol. Raw ingredients and rigid proxies cannot
validate cooked-dumpling release. Elapsed-time readiness is optional and is not
temperature sensing or thermal simulation.

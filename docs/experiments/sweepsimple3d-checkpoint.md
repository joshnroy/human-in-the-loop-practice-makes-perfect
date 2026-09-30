# SweepSimple3D implementation checkpoint, 2026-09-29

Approved scope and source audit are in sweepsimple3d-integration-proposal.md.
No scientific runs have launched. Source remains a development worktree.

## Implemented and checked

- Native Simple simulator selection, complete tick/qpos recording, direct native
  floor-region goal checking, separately labeled start-support tolerance.
- Explicit fixed-sample human restoration; continuous-practice hard-reset guard;
  isolated evaluation simulator with approved 10-action horizon.
- Six new lifted skill schemas: learned PickFloorWiper and SweepCubeToGoal;
  fixed SweepCubeToStart,PlaceWiperAtStart,OpenGripper,ReturnRobotToStart.
  Each sweep grounds separately to five cubes. No grouped-reset macro.
- EES/ModelB CLI wiring and approved learnable supports. Native configured human
  costs and planner/inference settings remain those in the approved proposal.
- 10/10 focused tests pass, including positive native goal parity and human reset,
  forbidden practice reset, grounding and exact deployment-model coverage.
- 10/10 new source modules pass mypy. New files pass Ruff after formatting.

## Physical feasibility remains the gate

All probes initialize the same native seed0 and retain separate result/state/replay
files under scratchpad/sweepsimple3d/. These are development probes, not experiment
arms, calibration samples, independent seeds or an effect estimate.

- Initial inherited Drawer handle selector picked Simple's broad blade. That
  demonstrated physical lifting, not a usable handle grasp. The new Simple geometry
  binding selects the actual vertical handle from compiled native geometry.
- Mid-handle pickup and safe transport have succeeded, but forward sweeps have not
  yet achieved the native target. A full physical stroke initially moved the cube
  about 0.12 m, then a tilted/lifted blade passed over it.
- Grasp geometry and observed blade-contact feedback are being corrected in the
  new controller. Native masses, friction, collision geometry and goal are unchanged.
  Every approach now must admit both the approach and descent before execution.
- Failed side-grasp stow exposed incompatibility with the Drawer folded home pose;
  the new controller searches collision-checked upright tool-transport poses.
- No reverse-sweep feasibility result yet. No readiness claim or launch ETA until
  forward/reverse task coverage passes.

Historical forward-seed0-v3/result.json recorded controller return as success, but
the cube only moved 0.096 m and did not reach the native goal. That record is retained
unchanged and INVALID as a task-success measurement. Subsequent probes explicitly
check native target-region attainment and preserve failures.

## 17:42 EDT contact calibration checkpoint

Four seed-0 probes at approved sweep distances 0.40, 0.50, 0.60, 0.70 m
confirmed bilateral handle-pad contact after pickup (4/4). Native forward goal
attainment was 0/4. The first stance had no collision-free base route, the second
failed checked descent, and the latter two failed blade-contact correction.
These are repeated development probes of one scene, not four independent seeds.

The base tracker previously reset its arm hold target to the observed (compliant)
arm pose at every short segment. A new optional fixed arm target preserves that
setpoint across checked contact microsegments; two probes are measuring whether
this removes cumulative sag. Default Drawer tracking remains unchanged.

## 18:06 EDT physical diagnosis and contracts

- Focused Simple contracts: 8/8 passed. New negative control prevents closed-gripper
  proximity from being interpreted as a real handle grasp without bilateral pad contacts.
- Shared Drawer regressions: 175/176 passed. The failing native stock seed-11
  sequence leaves no wiper in hand before park_wiper. The identical failure was
  reproduced against archived pre-Simple revision a0ee5fbe, so it is preexisting.
- Repeated short strokes have moved the target approximately 0.4 m, but no complete
  native forward target has been attained; no reverse feasibility claim yet.
- Earlier inference that neighboring cubes caused the late floor-contact failure
  was not confirmed. Replaying final qpos and reading native contacts showed a
  sideways dropped wiper with floor contacts only. The old proximity-only held
  condition was therefore too permissive. It now requires bilateral handle/pad contact.
- Collision-free joint-space paths can still rotate the held tool enough to lose
  its grasp. New transport planning tries Cartesian lift/translate/lower routes and
  filters excessive tool tilt. Bounded tilt variants are development calibration,
  not changes to the learned skill sampler supports or native physics.
- From v25 onward each probe preserves source snapshots and SHA256 hashes alongside
  arguments, raw qpos/state logs, final physical contact names and failures.
- The OS launch readiness contract is acknowledged but remains NOT_READY. No frozen
  scientific revision or manifest, no experiment jobs, and no claimed launch ETA.

## Native geometry correction and reset audit

The baseline planning helper hard-codes arm mount z=0.4000 m; the compiled native
robot_gen3/base_link is at z=0.3948 m. Simple now computes the planning mount transform
from the native body and mobile-base poses and reuses it for hypothetical base poses.
Across 5/5 recorded arm configurations the native-palm to planning-EE transform is
now constant within approximately 0.02 mm. Previously it varied by millimeters.
The native simulator, bodies, actuators and object physics are unchanged.
Evidence: scratchpad/sweepsimple3d/native-mount-fk-parity.json.

Native body-name clarification: robot_base is the gripper palm, whose parent is
robot_bracelet_link. It is NOT the arm mounting base. Earlier contact observations
with that name establish palm/tool contact, not an arm-base offset of 0.55 m.

The reset contract now also checks native upright wiper orientation (vertical-axis
atol=1e-3, rtol=0), alongside the previously disclosed support tolerance and yaw.
All 13/13 declared starts pass; a fallen-wiper counterexample fails, and exact
human restoration passes. Native task goal classifiers are unchanged.
Evidence: scratchpad/sweepsimple3d/native-upright-start-audit.json (also copied to
results/sweepsimple3d-readiness/).

The floor grasp calibration now includes off-center blade grasps as well as handle
grasps. HoldingWiper requires bilateral physical pad contact with the wiper, rather
than proximity or a commanded closure. Every contact phase checks the actual grasp.
Contact segments may end with a checked retreat, lift and reposition while keeping
the same tool. These transitions are logged; no human/environment reset or hidden
regrasp is inserted. A skill succeeds only on native target-region attainment.

Controller feasibility is still NOT_READY. No complete native forward goal has been
reported, no reverse coverage claimed, no frozen scientific manifest or launch.

## 19:20 EDT physical transport checkpoint

Native compact transport now finishes with bilateral pad contact and no recorded
arm contact in horizontal-handle-v45, but checked floor approach/descent fails.
No forward native goal or reverse success is established. Explicit transported-tool
versus arm collision checks use 5 mm clearance to tolerate measured tracking drift;
native physics remains unchanged. The replay of selfcontact-handle-v44 confirmed
upper-arm/tool contact in both native MuJoCo and the calibrated PyBullet scene.

Simple contract tests pass 10/10. Ongoing unique probes clearance-handle-v46 and
clearance-close-v47 test approved sweep distances 0.55 and 0.45 m, with source
snapshots, exact arguments, raw state/qpos and 15-minute wall limits. These are
development feasibility probes, not launched experiment arms.

## 19:39 EDT planning fidelity diagnosis

Blade-grasp probes v48/v49/v50 reached the floor with bilateral grasp but failed
checked exit/lift planning; none achieved the native task goal. In v49 the inherited
chassis bounding-box proxy reported about1mm finger overlap absent from native
contacts. Simple now constructs planning chassis meshes directly from the compiled
native collision mesh vertices/poses and mirrors the adjacent arm-mount collision
exclusion. The original chassis box is removed from Simple collision bodies.

A remaining planning collision at v50 was traced to the generic .5rad finger
closure, while native driver/spring/follower joints were around .63rad. Closed-grasp
planning now reads those six native joint values rather than assuming .5rad. Probe
v51 measures this under unchanged native physics and approved stance support.
Complete native forward success remains0; reverse untested; scientific launch held.

## 19:49 EDT contact and resource checkpoint

11/11 Simple tests plus1/1 shared Drawer physical floor-pick regression pass;
10/10 new source modules pass mypy. Native chassis mesh world geometry and
hypothetical-base translation are tested against native mesh vertices, with no
physical state mutation.

v52 achieved repeated strokes and~0.095m selected-cube displacement before native
chassis planning rejected motion near a floor cube. v53 retained native blade
height feedback but lost bilateral grip under load; v54 at approved-15degree stance
was also blocked. Complete native task success remains0; reverse untested.
Testing lower-handle and near-center blade grasps (v55/v56) reduces the mechanical
lever arm without changing physics, learned parameter supports or the task.

The contact-step CLI option had been unused in earlier source: actual contact
segments were12mm. Current source uses that field explicitly (v53onward3mm).
Old probe snapshots/logs remain intact; argument metadata alone is insufficient
to infer physical step size in those older probes.

## 20:18 EDT native corridor geometry checkpoint

12/12 Simple tests pass, including physical pad-center versus planning geometry
at two native closed-finger articulations. The new native-limit IK filtering and
joint-path constraints allow narrow-end pushing in3/3 variants(v58-v60), with
~0.14m selected-cube motion and retained grasp, but native target success0/3.
A direct native2D collision audit verified the next3mm base step intersects
cube_3; pushing one cube leaves another in the chassis path.

Side poses(v62-v64) achieved native goal0/3 because the native cooking-counter
colliders occupy the proposed base positions, including distance.40m. No obstacle
is removed. The five initial cubes span~.276m laterally within the native.30m
blade, so current probes center a broadside blade over that local group while
keeping success tied to the selected cube. Lower-handle grasps reduce lever arm.
Trials v65/v66 and checked native-home transport variantv67 remain bounded
development probes; no experiment or reverse-readiness claim.

## 20:35 EDT approved-range feasibility grid

A30-case read-only geometry audit tests five approved distances(.40,.45,.55,.65,.70),
three approved heading offsets(-15,0,+15degrees), and both equivalent broadside
blade yaws from a recorded physical lower-handle grasp. All failures are retained in
scratchpad/sweepsimple3d/cluster-low-handle-v65/floor-goal-support-grid.json.
No clear native-limit floor goal was found at.40-.55m; at.65m/nominal heading,
80/100IK solutions pass, and at.70m96/100pass. This is geometric feasibility,
not physical task success. v71/v72 physically test these feasible settings.

Periodic stack traces confirm carry-pose search spends time in IK/Cartesian paths.
The numerical IK candidate now runs before IKFast, retaining the same native
joint limits, tight FK validation and collision checks; IKFast remains fallback.
No high-level planning-budget change. Latest complete v70 failed approach at
previously tested.55m, in~3min20s; full native task is still unvalidated.

### Continued floor-contact diagnosis

Native-start v74 did not reproduce the isolated v73 TypeError, but failed floor descent. Replay diagnostics v75–v79 are explicitly excluded from end-to-end readiness; the native joint-planner descent fallback did not make v79 feasible (0/1). A fresh higher-handle native trial v80 is running. Diagnostic replay v81 was stopped because its old source contained native shoulder/tool contact. Current Ruff and mypy pass; Simple tests 12/12 pass. No complete native task, reverse pass, frozen scientific revision, or launch yet.

### Cross-grasp and native contact geometry

Cross-grasp pickup/carry succeeded in native v84 (1/1, ~28 s). A 42-case geometric endpoint audit found 8/42 feasible cases within approved .40–.50 m and heading support. Replay v89/v90 moved the rear cube ~.25 m and two others ~.06–.08 m (2/2), then failed reapproach. Independent review and actual native geometry found the blade-height guard checked the highest corner of the whole blade (~12.5 mm), while the selected-cube contact edge was only6.7–7.1 mm, below its9.9 mm center. Local cube-footprint edge guard and held-tool validation for corrections/lifts are under test in replay v91. No complete native forward/reverse readiness pass or science launch.

### Native contact guard and measured-limit repairs

Independent review reproduced the v95 collision short circuit: recorded joint2=2.24001rad exceeds native2.24 by1e-5rad, but no prohibited collision pair exists. Replay precision is5dp. Strict planned limits remain; only tiny observed-pose overshoot bypasses that numeric bound test while actual geometry is still checked. The contact guard now uses the selected cube footprint and actual native top minus2mm, with lowering limited by native blade-floor clearance. Corrections previously targeted2mm absolute height, overcorrecting into the floor. v96 moves the rear cube~.32m, then loses grasp on reapproach. A bounded checked wrist rotation around the native support corner is under test in v97. Simple tests14/14 pass (including meaningful native geometry/soft-limit regressions); Ruff passes. Still no complete native task, reverse proof, or scientific launch.


## 2026-09-29 late checkpoint: first leg physically demonstrated

Diagnostic replay v105 (`scratchpad/sweepsimple3d/native-clearance-v105`) moved all 5/5 cubes from native sample0 into the north corridor (final y1.164–1.189m), retaining bilateral tool contact. This resumes a recorded valid pickup and is not a native-start readiness pass. No complete goal and no reverse/home measurement yet.

The earlier shallow Bullet wrist/chassis overlap was proxy padding: measured native separation+1.93mm vs Bullet−0.03mm. A narrow refinement consults exact native geometry only within the known proxy padding; genuine native penetration remains rejected. Regression verifies both clearance and a9.8mm penetration and preserves live native state.

At the westward turn, all184/184 hover IK endpoints collided with actual chassis geometry. Native representative wrist penetrations55.8–155.1mm establish a real stance issue. v106 uses behind-stroke stance for both route legs within approved distance/heading supports; no geometry, dynamics, goals or predicate relaxation. Checks now also validate carried arm/tool throughout unloading retreat and the exact arm target commanded during contact base motion. Per-tick grasp guards stop correction immediately after physical grasp loss.

Validation: Simple tests28/28; Ruff and mypy controller/shared motion pass. Shared Motion callback defaults toNone and leaves existing call behavior unchanged. Readiness contract read; noREADY record, frozen revision or science launch. Owner heartbeat records active probe.


## Native distance-query diagnosis (v108–v112)

Fresh native v108/v110 stopped at the same sixth contact stroke. Exact runtime instrumentation in v111 proved the native wrist/chassis refinement returned0.0 for a separated pair. Fresh MjData, full forward computation, and canonical base coordinates reproduce the same issue; no stale scratch-state hypothesis remains. Installed MuJoCo3.3.7 has native CCD enabled. Asking for a1cm positive distance produces0.0; asking for1µm yields positive capped clearance. Native collision detection reports no contact. A genuine−9.8099mm penetration remains negative for every tested cap. Evidence: `exact-pairs-v111/native-distance-thresholds.json`.

The helper now queries only the required collision threshold+1µm, treating its result as capped clearance, never an exact distance measurement. Native flags, shapes, physical simulation and zero-margin rejection criteria remain unchanged. v112 is a fresh native-start full-cycle probe.

Fixed reverse geometry also needed a north-side stance: the former west stance overlapped the native island66.5mm. At0.7m north stance,24/24 floor/hover IK endpoints, a direct descent interpolation and a34-waypoint base route are collision-free statically; this is not yet physical recovery success. Current fixed reverse distance0.7 remains in approved support.

Validation after the native threshold-query fix: Simple tests29/29 pass; Ruff and mypy pass. Exact v111 regression verifies real penetration stays rejected and native state/flags remain unchanged.


## West-turn transport checkpoint

Native v112 moved all 5/5 cubes into the north corridor, then the east-side base stance hit the right counter. A north-side blade-end push has verified base/arm/floor geometry at the same approved distance/heading. The controller uses this contact orientation for the west leg and allows at most36 internal short strokes; the experiment still counts one invocation as one skill action, with the unchanged20-action practice budget.

Native v114 and diagnostic replay v115 lost the handle during base turning. Restored evidence shows bilateral contact through tick62, release at63, and a fall onto the floor by67. Turning reached approximately0.75rad/s; arm tracking was good and gripper remained closed. The old drive continued123ticks after release.

Optional drive limits now bound per-tick translation/yaw and their change; all default toNone, preserving Drawer commands. Simple carried transport requests .01m/.005rad caps, .002m/.001rad per-tick changes, and checks bilateral grip after every tick. Focused motion tests4/4, Ruff/mypy pass. v116 retained the grasp through the previously failing transfer and reached the west stance; pushing and full native/recovery validation remain pending.

`--resume-final` permits explicit failed-state diagnostic restarts with disclosed velocity zeroing and false end-to-end-native-start provenance. These are never readiness trials. Live owner heartbeat is now external at `results/sweep-launch/environments/simple/owner-status.json` so future frozen source can remain clean. Generated replay evidence remains on disk under ignored `scratchpad/sweepsimple3d/`.

## 2026-09-30 leveling and implementation-reuse checkpoint

Bounded transport retains the handle, but the west-facing blade end remains above the native cube top. v118 isolates the current failure: 0 physical ticks execute because all three leveling candidates lack a checked path. The smaller 0.02-radian candidate reaches the native arm limit after an earlier partial adjustment. A static 15-candidate reach scan finds four collision-free IK endpoints only when the unloaded blade moves 2 cm toward the robot; all four occurred after the first eight distance-ranked candidates. Path-search budgeting now filters collision-rejected endpoints before consuming its eight attempts. Connected paths still fail, so this is not a complete recovery fix.

The alternate approved +15-degree stance fails carried-tool clearance (v123). v124 tests the opposite blade end at the previously transport-feasible stance. Both are explicitly diagnostic continuations, excluded from readiness. Native pickup remains 3/3 seeds and northward movement 5/5 cubes; full native goal, physical reverse sweep and wiper/robot return remain unverified. No scientific run or READY record exists. Calibration cube-context policy remains unapproved.

Simple directly subclasses Drawer Primitives, PlanningScene and Session and uses shared Motion. Shared changes are narrow extension hooks for the native floor handle geometry, measured finger articulation and optional bounded transport. Native floor predicates and goal checks stay distinct from Drawer counter/drawer conditions. No native KINDER or kinder-baselines source is changed in this branch. The draft stacks on `codex/sweep-learning-review` (a0ee5fbe); recovery PR396 is a separate dependency whose changes must not be deleted or silently substituted. Latest focused suite: 32/32 passing (Simple plus shared drive controls); Ruff passes. These software checks do not establish physical readiness.

## Overnight authorization and ground-path diagnosis

Josh explicitly approved rotating the five cube targets across the existing 24 calibration contexts (three seeds × eight), keeping the same trial count and Beta(1,1) posterior-mean formula. The deterministic rotation is `cube_index = (8 * seed_index + draw_index) % 5`. This supersedes the pending calibration-context note above. The live owner and installed AI continuation prompt retain this approval. Login lingering is enabled; installed launch/wakeup timers survive logout. Scientific readiness remains mandatory.

Draft PR397 is published on the own fork. Shared completion wrapper `scripts/run_sweep_verified_arm.py` runs one frozen arm once, reconciles complete harness and native event logs, and writes an atomic fsynced PASS record only after validation; its six failure-path tests pass. The parent relays commit23ed6b43 to the Drawer owner for reuse.

Native v125 used the existing higher handle-grasp candidate (-0.06 m versus -0.09 m). Pickup succeeded and contact strokes moved all five cubes north, but a later descent lost the handle at tick3159. The hand continued67 ticks before the old boundary check noticed. Recorded poses establish real separation, not a transient contact predicate. Audit then found the inherited room collider set omits the native ground plane. A joint-space descent can therefore drag the tool through the floor, whose real contact force strips the grasp. Simple now checks held-tool support height from compiled native box geometry in the existing collision predicate. It permits no deeper penetration than the observed native soft-contact state and preserves physical geometry and goal predicates. Approach/descent use the existing per-tick grip guard. v126 replays the recorded pickup to diagnose the changed paths; it is not end-to-end readiness.

## Contact tracking and execution termination

Ground-checked v128 kept the grasp but stopped after three non-progressing strokes (1485s); always unloading/reapproaching whenever a stale hold target crosses the floor is insufficient. The targeted correction first stalled at overly tight intermediate joint-waypoint tolerances: even a checked20mm upward target never progressed past its first waypoint. Restoring the shared0.03rad waypoint tolerance and adding an optional physical stop callback gives measured floor clearance: v135 raises the blade from0.035mm to6.55mm in4ticks, with grip retained (1/1 isolated diagnostic). The callback stops at5mm support height before the remaining joint target can lift the blade out of cube overlap. Existing Drawer behavior remains unchanged when no callback is supplied; its default-preservation regression passes.

Descent feasibility is now checked before expensive hover approach planning, followed by the original connected-path and observed-grasp checks. This avoids spending most of each trial on hover paths whose floor descent is impossible. The current fresh native full-cycle probe is v136. Full goal and reverse/home readiness remain unverified. The superseded v129/v133 diagnostic continuations were stopped explicitly with partial logs retained; v128 completed its own failure normally. Latest focused suite41/41 passes; no scientific run has launched.

### 2026-09-30 01:20 EDT — native gripper collision refinement

Fresh native v136 retained pickup and completed several contact strokes, then failed after 160.4 seconds / 2,755 physical ticks on a PyBullet right outer-knuckle/chassis proxy collision (0.66 mm). Native Robotiq links use driver/coupler/follower names, so the prior exact-name refinement returned unavailable. The native closed gripper at the recorded state is separated. The new fallback queries **all native gripper collision bodies**, conservatively, whenever a known URDF gripper link has no exact native name. It does not disable any native contact or ignore unknown links. Regression checks preserve the real 6.47 mm penetration caused by opening the gripper at this pose. Native planning regressions: 8/8 passed.

A diagnostic initially used the default open-finger planning configuration; it was corrected to use the actual recorded closed articulation before interpreting the result. The closed/open distinction is retained as the regression negative control.

Persistent fresh native v137 launched with the same development grasp −0.06 m, distance 0.40 m and heading −15 degrees. Full native goal and reverse recovery are still unverified; no experiment launch is authorized by the readiness gate yet. The development override has not replaced the frozen production grasp.

### 2026-09-30 02:10 EDT — native contact and bounded full-task checks

- v137 retained pickup but failed unloading after a real native gripper/chassis overlap. Mid/far stances v138/v139 and neutral v140 remained unsuccessful.
- Shared handle-center grasp v141 advanced12strokes, then the process segfaulted inside IKFast under the diagnostic watchdog; no result JSON. `crashed.json` and `service-crash.log` retain this separate infrastructure failure. v142 midpoint failed a later floor approach.
- v143 raised commanded blade clearance from1mm to5mm (diagnostic override only), revealing descent onto neighboring cubes. Widening broad-blade stand-off from25mm to50mm avoided that failure in v144, which then hit an ambiguous native forearm/chassis distance query after13strokes. At the exact recorded pose, native contact generation reports no contact; the enabled pair is not excluded. Zero-distance/zero-margin ambiguity now uses native `mj_forward` contact generation on private data. Positive margins and real penetration checks remain conservative;9/9 native collision regressions pass.
- v145 narrow blade-end approach was unreachable. Shared upper-handle grasp(+0.06m) v146 advanced14strokes then a different cube blocked the base. v147 midpoint advanced27strokes before3consecutive nonprogress strokes.
- Stalling diagnosis: once all cubes no longer fit, the old anchor centered on the selected cube but still started behind a neighbor at the blade edge. New safe-subset anchoring covers the target and nearby cubes centrally, avoids thin edge overlap, and preserves the original compact-group center.
- v148 checks a transparent nearer-cube order with nominal heading. v149 runs the same native task with up to10**counted** forward actions including initial pickup and failed calls; no reset, hidden pickup, or uncounted retries. Forward failures remain failures. A budget regression verifies8failed additional calls stop after10total actions; contact/budget checks3/3passed. This is only readiness-probe sequencing; neither method’s policy or approved experiment budget changes.

Full native all-five goal and reverse recovery remain unverified. No Simple scientific run has launched. Production grasp remains the original fixed value until physical verification supports freezing a replacement.

### 2026-09-30 — contact-feedback and radial-grasp diagnosis

The native shared grasp azimuth describes the finger closing axis, not the radial approach axis. A diagnostic -pi/2 offset with a +0.06 m handle grasp and a stand-ahead north stance reaches floor poses that the previous sideways approach could not. These are explicit probe overrides; production defaults remain unchanged pending feasibility.

v154/v155 stopped at a custom 5 mm palm-clearance margin despite an observed positive 4.794 mm gap. Matching the shared collision semantics removes that extra positive palm margin while retaining actual penetration rejection and the other arm clearances. The regression includes a real 3.696 mm palm-penetration negative control. Focused integration, collision, motion and accounting tests pass 47/47.

v156 (10 cm strokes) and v157 (20 cm strokes) complete with 0/2 native full-goal successes, 0/5 cubes in the goal each. Both pick up successfully. v156 loses bilateral handle contact after 14 strokes; v157 retains the handle while tipping approximately 35 degrees and safely rejects retreat clearance. Neither reaches the reverse/home checks. There are still no Simple scientific runs.

An independent recorded-state audit finds that v156's last loaded stroke moves the base 86.6 mm while moving the selected cube about 1 mm, with tool-to-palm drift reaching 37.6 mm / 0.146 rad. Of 257 height-correction records, 116 remain above the requested height and only 167 distinct physical ticks are represented. Joint convergence was incorrectly treated as physical correction success. Contact correction now uses a tighter final joint tolerance and explicitly checks measured blade height before recording success or advancing the base; failure ends and unloads the stroke. It does not change native goals, predicates or simulator physics. Actual contact forces were not logged, so force magnitudes are not inferred from state replay.

Fresh v158/v159 test shorter-lever radial handle grasps (0.0 / +0.03 m), preserving approved distance/heading supports and the same native start. These diagnostics started before the final physical-height check was added and retain that source provenance. Full native task, reverse recovery, calibration and method readiness remain pending.

### Follow-up: truthful contact feedback and early unloading

v160 physically verifies 155/155 height-correction events meet the actual blade target, at distinct physical ticks. It still fails the native task (0/5 goal cubes) through grasp loss on a base step. The feedback-reporting bug is fixed; the mechanical problem is not. v161's +0.03 m radial grasp completes pickup and repeated floor approaches but also fails (0/5), at a held-palm proxy overlap. Its static audit establishes clear endpoints and separate approach/descent paths, not a universal connected-path guarantee.

A new bounded controller response unloads the contact stroke when observed tool-to-palm drift exceeds 1 cm or 0.06 rad. This is below the 3.76 cm / 0.146 rad drift measured before v156's loss. v162 tests it; v163 additionally uses the existing narrow blade-end contact option to avoid loading several cubes at once. These are pending diagnostics, not claimed fixes. Focused suite remains 47/47 passing.

v159 exited with SIGSEGV exactly at the repeating traceback watchdog's 360-second dump, with interleaved native collision and watchdog traces. The prior v141 crash had a similar coincidence. Causation remains unresolved. Its crash marker and full journal are retained. Future probes keep fatal-error tracing but disable periodic asynchronous traceback dumps; external service/tick monitoring remains active. v160/v161 complete normally with this diagnostic-thread change. No failed evidence is replaced or treated as success.

### Native palm refinement, local blade starts, and first native cube goal

The exact-native palm helper (bb630bb1) checks only the zero-margin palm/tool constraint at the candidate arm, tool and base pose using scratch MjData. It preserves positive arm margins and real native penetration, does not mutate live state or collision flags, and does not infer separation from an ambiguous zero. Combined native collision tests pass 12/12. The full suite after contact-start regressions passes 52/52.

v162's 38 unloaded strokes all ended while actually touching cubes. First contact drift was only 0.04/0.12 mm in the first two strokes, growing to 10.49 mm; therefore the unloaded-settling hypothesis was rejected and no reference reanchoring was applied. v163 narrow blade-end contact moved cube0 about 0.63 m north, but the next native base target was blocked. Exact replay diagnosis found an unrelated cube about 0.60 m farther south was still pulling the sweep start backward because it shared the blade column. Start clearance now considers only nearby cubes within the actual blade footprint/rear reach. Nearby-cube clearance regressions are retained.

v166 diagnostic replay verifies the corrected corridor transition and places cube0 in the exact native goal (1/5). All other floor approaches then fail: a static audit finds zero raw floor IK for the narrow north orientations, despite clear hover endpoints and approach paths. Stowing cannot resolve those endpoints, so no speculative stow fallback was added. Broad north contact with yaw0 is statically reachable from this same grasp (96 native-valid floor candidates, 24/24 checked clear, checked hover descent), retaining the westward tool orientation.

v167 is the first fresh native-start trial to place a cube in the goal: pickup succeeds and cube4 reaches the native goal after21 checked strokes (1/5 so far). It continues the counted10-action task check. v168 physically picks with the near-blade radial grasp, but narrow north floor endpoints are unreachable; broad-contact comparisons will use the statically verified yaw0 geometry. Native ground goals include the upstream 5 cm region expansion; this is reused unchanged, not a local tolerance relaxation. The nominal x upper bound is0.875, while the native predicate boundary is0.925.

Neither full native five-cube success nor reverse/home readiness has passed. No Simple scientific run or READY record exists.

### Fixed recovery audit and parameter isolation

At v167's actual cube4 goal checkpoint, the old broad eastward reverse pose has zero raw floor IK solutions. Narrow eastward blade-end contact at the unchanged fixed distance0.70/heading0 has a clear base footprint, 96 native-valid floor candidates, 24/24 checked clear endpoints and a checked hover descent. A hypothetical post-east broad southward pose also passes the static checks. Horizontal goal/reset legs now share blade-end geometry. These are static feasibility checks only; physical reverse success remains unmeasured.

PlaceWiperAtStart previously inherited primitive.distance/heading_offset from the last learned pickup. Its dispatch now explicitly restores the fixed defaults0.70/0. A real-dispatch regression exercises two different pickups followed by home, verifies identical home parameters and one accounted action each, with no resets. Focused integration/collision/motion/accounting suite53/53 passes; focused mypy on controller/helper/environment passes. The diagnostic probe defaults remain unfrozen until full physical readiness.

### Native contact-topology diagnostic

Recorded-state audit rejects a handle-end-margin-only guard: the known v156 loss still had at least63 mm pad-contact margin, but the handle tip contacted both spring links before all contacts disappeared. The low-grasp v169 stall instead retained bilateral3+3 pad contacts with about71 mm margin and no non-pad contact. Three immutable recorded-state classification regressions pass; the complete focused suite is56/56.

A diagnostic `--native-contact-guard` now unloads on actual native handle contact with robot parts outside the intended pads, preserving bilateral-contact and native collision checks. Default behavior is unchanged pending physical validation. Fresh-native v171 and picked-state replay v173 reveal a real low-grasp limit: the long handle contacts the wrist (`robot_bracelet_link`) after tilting under cube load. They have not established full task success. The final-state replay v172 correctly rejects its source state immediately because bilateral contact is absent; no physical ticks execute, and it is not counted as a controller trial or readiness evidence.


### 2026-09-30: guarded stow and intermediate-grasp measurements

Native-contact low-grasp v171 and disclosed picked-state replay v173 both ended with 0/5 native goal cubes. Independent native-state audit locates subsequent grip loss during stationary-base arm stow: no non-pad contacts before loss, but accumulated relative grasp drift reaches 89 mm / 0.82 rad. The old stow follower continued 153 ticks after the first lost pad contact. Commit 0762820a adds per-tick bilateral checks and observed-grasp replanning after existing 1 cm / 0.06 rad thresholds, with an eight-attempt bound; 60/60 focused tests pass. This is a controller correction, not a changed success predicate or physics setting.

Fresh-native intermediate-grasp v174 (offset0) and v175 (offset−0.03) both finish 0/5 goals. v174 moves the first cube pair about9cm but later cannot find a checked compact carry pose; v175 still contacts the native wrist under load and later loses grip. All failed calls count against the 10-action diagnostic budget. v176 tests the previous upper grasp (+0.06) with native-contact guarding and corrected stow together. No full-task success, physical reverse-reset result, READY record, or Simple science batch exists yet.


### 2026-09-30: physical reverse diagnostics and carry-search corrections

Reverse-only replay v177 restores the final retained-grasp state of v167 (recorded rounded native qpos, explicitly zeroed velocities; excluded from end-to-end readiness). Fixed SweepCubeToStart moves cube4 east around the island, then fails the south leg after three nonprogress strokes. Native contact audit shows base travel about101mm versus blade35–37mm, with60–62mm grasp translation and0.23rad rotation; blade/cube contact exists only near stroke end. The cube nearly returns after unloading. The longer .20m-stroke comparison v179 also fails, reaching tilt1.10386rad beyond the unchanged1.1rad limit. These are0/2 disclosed diagnostic reverse variants, not independent seeds or full reset measurements. No home actions follow failed reverse skills.

Upper-grasp forward v176 and existing-angle1.8 v178 both fail0/5 native goals. Native spring-link contacts and eventual floor-descent grip loss remain. v178 additionally shows a small loaded observed joint-limit overshoot; no joint-limit relaxation was applied.

Commit acce689b fixes carry-search ordering: the existing checked joint fallback is tried per compact candidate, instead of exhaustively trying every Cartesian-only candidate first. Commit73875da0 fixes guarded-stow completion to compare against the validated path endpoint, because a Cartesian path may legitimately use another IK solution. Grasp checks, drift thresholds, native collision/tilt checks, and bounded retry count remain unchanged. The focused integration/accounting suite passes62/62. v180 measures the remaining existing1.2rad grasp approach; independent audit checks native gripper geometry/actuation conventions. Still no full native task pass, full robot-reset pass, READY record, or scientific launch.


### 2026-09-30: native pad insertion and new partial progress

Native close control is correctly255. Recorded native contact geometry places the old35mm insertion grasp at distal pad tips/width corners. Exact pad geometry suggests reducing insertion standoff to28mm (distal center) or20mm (combined patch center). Shared/production default remains35mm; bounded diagnostic hooke2a6705d changes no native physics. Actual-step contactforce logging verifies28mm eventually engages both pad boxes symmetrically; no forces are inferred from replay.20mm v181 loses grip during an unnecessary high carry. Opt-in checked pickup-pose retentionff2414df avoids that lift while preserving full carried-path checks and distinct compact fallback; v183 then succeeds pickup but loses grip in its first sweep.

Fresh-native v184 combines the existing lowhandle grasp−.09m with28mm insertion and checked pickup carry. It has reached2/5 exact upstream goal cubes: cube4 at tick4538 after28strokes, cube2 at4642. It continues the full counted10-action task and is not yet a full success. This is partial improvement over prior1/5, not readiness. Highhandle v182 remains an ongoing comparison. Focused guard/default/integration/accounting checks pass72/72.


## 2026-09-30 06:05 EDT — four-cube progress and transport diagnosis

The fresh-native selected-center trial v185 attained 4/5 goal cubes in 10/10
forward actions with the grasp retained. The contiguous-sequence trial v186 also
attained 4/5 in 10/10 actions, then lost the bilateral handle grasp on the final
transfer. Neither is a full-task pass; recovery/home was not entered.

Actual-step contact records show no non-pad tool collision during the failing
v186 transfer. The native planner took a long route around the already-goal
cubes; actual accumulated travel was 9.17 m with 6.43 rad yaw before loss.
The pad contact patch gradually reached a distal corner. No force/friction,
native goal, collision tolerance, or action-budget change is justified.

Commit da8c5e5e adds a native-planned upper-aisle alternative and chooses among
fully carried-collision-checked routes, retaining the original fallback. Static
reconstruction from the disclosed rounded recorded state gives 1.59 m/0.52 rad
versus 10.97 m/8.30 rad, with all waypoints collision-clear. This is diagnostic
static evidence only. Fresh physical trial v188 is running; v187 separately
checks deeper insertion with the same low handle grasp.

The obsolete higher-handle v182 completed naturally: 0/5 goal cubes, 10/10
forward actions, 51.8 minutes dominated by repeated identical-state exhausted
compact-pose searches. It was not interrupted. The exact-state failure cache
in subsequent revisions prevents redundant searches without erasing failures.

Strict production-controller probe mode now bypasses diagnostic overrides and
uses ordinary Environment.take_action for initial pickup/sweep as well as later
skills (218357fe). The initial-dispatch limitation is removed for this mode.
Controller defaults have not yet been promoted: diagnostic success cannot stand
in for a production-controller full-task/recovery measurement. No READY record,
calibration campaign, or Simple science job has been launched.


## 2026-09-30 06:41 EDT — full native task and face-aligned pickup

The lower/deeper diagnostic v187 attained all **5/5 native goal cubes in 9/10
allowed forward actions** from a fresh native seed-0 start. Its first reverse
skill subsequently lost the grasp; full robot recovery remains unverified.
The shorter-route v188 comparison attained 4/5 before contact nonprogress and
grip loss. Increased floor clearance alone did not cure reverse grasp migration
(v189), so this is not a proposal to relax ground checks or alter physics.

The actual contact audit identified a 33.8-degree mismatch between the prior
closing axis and a native square-handle face. Probe-only face-seeded yaw now
passes nominal pickup on **3/3 seeds (0, 1, 2)** with symmetric proximal and distal
pad contact patches and approximately 155 mm lift. Full native candidate v194
is running. Production defaults remain unpromoted until the full candidate is
validated; these diagnostic records are not calibration or science results.

Explicit OpenGripper/PickFloorWiper reset testing exposed a separate target
tracking defect: empty-hand homing displaced the tool about 4.4 cm after the
original target was captured. The Simple-specific observed-target refresh
(523d0ae8) fixes re-pick in v193 while preserving the shared Drawer default.
Reverse then retains the grip but stalls behind a nearby cube outside actual
blade contact. The accessible-cube-first v196 sequence is diagnostic only,
uses existing separately counted actions, and does not change either planner.

Simple-only protocol preflight now checks resolved goal-pursuit horizon zero
for both methods before simulator construction, including manifest overrides.
The draft readiness builder requires hashed passing evidence and emits the
installed gate's executable verified-wrapper commands and 4+4+16 staging.
It never publishes READY or launches by itself. No Simple scientific job has
launched. Focused integration/accounting tests: **127/127 pass**; Ruff passes.

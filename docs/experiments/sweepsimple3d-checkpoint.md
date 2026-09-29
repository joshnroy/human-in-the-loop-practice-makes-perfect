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

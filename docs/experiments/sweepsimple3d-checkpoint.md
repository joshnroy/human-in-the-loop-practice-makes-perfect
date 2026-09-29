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
- 6/6 focused tests pass, including positive native goal parity and human reset,
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

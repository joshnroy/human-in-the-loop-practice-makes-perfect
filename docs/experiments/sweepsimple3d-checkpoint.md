# SweepSimple3D implementation checkpoint,2026-09-29

Approved scope and source audit are in sweepsimple3d-integration-proposal.md.
No scientific runs have launched. Source remains a development worktree.

## Implemented and checked

- Native Simple simulator selection, complete tick/qpos recording, direct native
  floor-region goal checking, separately labeled start-support tolerance.
- Explicit fixed-sample human restoration; continuous-practice hard-reset guard;
  isolated evaluation simulator with approved10-action horizon.
- Six new lifted skill schemas: learned PickFloorWiper and SweepCubeToGoal;
  fixed SweepCubeToStart,PlaceWiperAtStart,OpenGripper,ReturnRobotToStart.
  Each sweep grounds separately to five cubes. No grouped-reset macro.
- EES/ModelB CLI wiring and approved learnable supports. Native configured human
  costs and planner/inference settings remain those in the approved proposal.
-6/6focused tests pass, including positive native goal parity and human reset,
  forbidden practice reset, grounding and exact deployment-model coverage.
 -10/10new source modules pass mypy. New files pass Ruff after formatting.

## Physical feasibility remains the gate

All probes initialize the same native seed0 and retain separate result/state/replay
files under scratchpad/sweepsimple3d/. These are development probes, not experiment
arms, calibration samples, independent seeds or an effect estimate.

- Initial inherited Drawer handle selector picked Simple's broad blade. That
  demonstrated physical lifting, not a usable handle grasp. The new Simple geometry
  binding selects the actual vertical handle from compiled native geometry.
- Mid-handle pickup and safe transport have succeeded, but forward sweeps have not
  yet achieved the native target. A full physical stroke initially moved the cube
  about0.12m, then a tilted/lifted blade passed over it.
- Grasp geometry and observed blade-contact feedback are being corrected in the
  new controller. Native masses, friction, collision geometry and goal are unchanged.
  Every approach now must admit both the approach and descent before execution.
- Failed side-grasp stow exposed incompatibility with the Drawer folded home pose;
  the new controller searches collision-checked upright tool-transport poses.
- No reverse-sweep feasibility result yet. No readiness claim or launch ETA until
  forward/reverse task coverage passes.

Historical forward-seed0-v3/result.json recorded controller return as success, but
the cube only moved0.096m and did not reach the native goal. That record is retained
unchanged and INVALID as a task-success measurement. Subsequent probes explicitly
check native target-region attainment and preserve failures.

# EES recovery from a recorded bin-rim landing

## Reproduction and scope

The unmodified standard transfer run's seed 3 stopped after action 19. Its last
physics tick is preserved in
[the regression fixture](../../tests/environments/tossing3d/fixtures/seed3_rim.json).
The cube was on the upright bin's rim, neither `OnGround` nor `InBin`.
Before this change the regression reproduced `InteractionComplete` instead of a
pickup. The benchmark continues unchanged in its separate worktree.

This change adds `OnBinRim` and `PickCubeFromRim` to the same-side skill provider.
The predicate tests oriented bounding extents against the actual bin wall geometry,
allows 2 mm support tolerance, rejects linear speeds above 2 cm/s and bins tilted
more than approximately 5.7 degrees. This is a conservative geometric eligibility
test, not a guarantee of grasp feasibility. It does not change `InBin` or test
success. The new operator reuses the existing bin-aware pickup controller.

## Physical evidence

[Recovery video](2026-08-27-tossing3d-rim/rim-recovery.mp4) ·
[Measured result](2026-08-27-tossing3d-rim/result.json)

Starting from the saved state, a fresh EES policy with a bin goal selects
`PickCubeFromRim` without an injected action sequence. The actual controller
executes in MuJoCo: `Holding` is true afterward, cube center height is 0.58868 m
(from 0.22448 m), and no controller error is reported. No human reset occurs.
The video includes two seconds of still frames at each end.

This is a state-replay recovery demonstration, not a new training run. It establishes
recovery for this recorded upright-rim case; it does not establish recovery from
all rim orientations or from overturned bins (including other stalled seeds).

```bash
scripts/with_env.sh python -m scripts.tossing3d_rim_demo \
  --snapshot tests/environments/tossing3d/fixtures/seed3_rim.json \
  --output-dir scratch/rim-recovery
```

## Tests

- Red: saved-state integration test raises `InteractionComplete` before the change.
- Green: EES selects rim pickup, physically grasps and lifts the recorded cube.
- Geometry checks cover three bin yaw angles; reject floor, airborne, moving,
  out-of-footprint, unsupported-center, and overturned-bin cases.
- During combined-suite verification, the existing metadata test helper leaked a
  fake `kinder` module into later tests. Added a failing cleanup regression and
  fixed its monkeypatch teardown; the metadata-plus-predicate sequence now passes
  all 21 tests. Offline logging tests also pass (17 passed, 1 skipped) when allowed
  to open their local socket outside the sandbox.
- The first combined run was interrupted after 71 failures, 1665 passes, 2 skips,
  1 expected failure and 3 errors from those verification issues. Final verification
  is split into simulator fidelity, remaining Tossing3D tests, and the rest of the
  suite to keep their resource use manageable.
- Tossing3D verification: 37 simulator-fidelity tests passed; the remaining domain
  group had 183 passed and 1 existing expected failure (220 passed total).
- Remaining suite: 1666 passed, 2 skipped. Across the three groups: **1886 passed,
  2 skipped, 1 existing expected failure**. Ruff, formatting, mypy (107 source
  files), import-layer checks and documentation links all passed.

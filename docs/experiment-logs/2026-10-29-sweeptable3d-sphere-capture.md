# SweepTable3D shallow-tray capture correction — 2026-10-29

## Diagnosis

The [previous full-five recovery run](2026-10-28-sweeptable3d-production-preflight.md)
returned all five balls using the robot, then scored the second sweep 4/5. The
blue ball's center was 8.6802 mm inside the nearest inner wall, with its bottom
on the tray floor. The old scorer required 10 mm clearance: the sphere's full
projected radius. At the actual rim height, its cross-sectional radius was only
8.6612 mm. Its upper hemisphere could overhang the shallow wall without collision.

A saved-state continuation held the robot still for 60 simulated seconds. All
five balls maintained tray-bottom contact for all 120,000 physics ticks; none
contacted the ground. Maximum ball displacement was below 0.001 mm. The blue
ball had 0.981 N bottom support and no wall contact. This is evidence of stable
capture in that case, not a disturbance-resistance or multi-seed guarantee.

The saved state contained positions, velocities and time, but not controls or
solver history. Robot hold controls were reconstructed and solver warm-start
cleared; this was not a bit-exact continuation. The report was written before
an optional oversized render failed. A separate supported-resolution render
succeeded. Evidence remains in the parent checkout under
`artifacts/sweeptable-fixes-20261028/retention-probe/report.json`.

## Correction

`BallScene.on_tray` now checks clearance using the sphere's widest horizontal
section overlapping the walls. A rim below the center uses
`sqrt(radius**2 - (center_z - rim_height)**2)`; a wall reaching the center uses
the full radius. Both axes must fit. Floor support, tray uprightness, velocity
limits, all-five success and irreversible ground failure retain their existing
checks. A rim must rise above the floor and overlap the sphere vertically.

No geometry, friction, controller, sweep parameter or learning algorithm changes
are part of this correction. Historical reports retain their original scores.

## Validation

The recorded blue-ball regression failed before the fix at three tray yaw angles
and passed afterward. Tests cover each signed wall, tall walls, corners,
wall-intersecting and outside balls, floor/airborne/rim-perched positions, and
missing or non-overlapping rims. The focused table suite passes 423 tests,
including 46 ball-scoring tests. Four native tests pass, including an explicitly
placed low-rim fixture checked through native predicates and the completed-sweep
success gate. The preflight harness passes 34 tests. Lint, formatting, types,
import boundaries and documentation links pass.

## Production controller rerun

The frozen-source seed-0 full-five preflight passes in 321.66 seconds:

- Initial physical pickup and sweep: 5/5 captured.
- Physical wiper return and individual robot resets: all five returned to table.
- Next physical pickup and sweep: 5/5 captured, including the near-rim blue ball.
- Final physical wiper return succeeds; no ground contacts or human resets.

Initialization has the same qpos hash as the previous 4/5 report. Both sweeps
retain angle -27 degrees and speed 0.15 m/s. The new successful path continues
through a final wiper return, so its final robot state is not identical to the
earlier stopped-at-failure run. The scorer is the only production-source change;
source stability is verified throughout the run.

Evidence: parent-checkout
`artifacts/sweeptable-fixes-20261029/sphere-capture-full5/report.json`, its
`full5-seed-0/result.json`, and recorded motion. Source-manifest digest:
`dcd6ffb9d50c4cbd40b12ae363aa4acf39eeb780fa3e67af3da9970cbb6787b0`.

This clears the previously failing seed-0 full-recovery case, not the entire
launch gate. Multi-seed physics coverage and separate learner validation remain
required. No overnight or Della experiments were launched.

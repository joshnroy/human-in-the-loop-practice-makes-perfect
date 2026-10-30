# Burner alignment and higher pouring: native validation

The revised scene centers the skillet on the front burner at world XY
(0.85, 0.85), rotates its long handle toward the human side (-Y, yaw -90
degrees), and gives the textured wrappers a slight yellow tint. Shape,
mass and contact parameters are unchanged. Both three- and ten-piece asset
layouts receive the visual placement change; only the three-piece environment
is validated here.

The tilted pouring pivot is 5 cm higher: its vertical drop from the upright
transport pose decreases from 0.16 to 0.11 m. The base stands 0.60 rather than
0.70 m behind the pan center to keep the elevated tilt arc reachable. These
are fixed motion settings, not additional learned parameters.

Two integration assumptions needed repair. Pan-home orientation is measured
relative to the initial pan pose rather than identity. If the first IK fit
violates clearance, the existing bounded alternate-configuration search also
runs for that failure, retaining its pose, clearance and continuity bounds.
Failed intermediate trials remain in the local evidence directory.

## Fresh-start physical results

Native controller revision: `09c6400897d27dd080a9703e81a4d62852be9301`.
Every recorded native source/asset hash matches this revision's runtime files.
All trials below use seed zero, start with three pieces in the pitcher, and
execute pickup, pour, placement and payload-only human reset. Duration is 12 s.

| Requested | Tilt (degrees) | Lateral amplitude (mm) | Delivered | Retained | Exact-count success | Ready after reset |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 89 | 8 | 1 | 2 | Yes | Yes |
| 2 | 89.125 | 8 | 1 | 2 | No | Yes |
| 3 | 90 | 15 | 3 | 0 | Yes | Yes |
| 2 | 89.1875 | 8 | 2 | 1 | Yes | Yes |
| 2 | 89.25 | 8 | 3 | 0 | No | Yes |
| 2 | 89.3125 | 8 | 3 | 0 | No | Yes |
| 2 | 89.375 | 8 | 3 | 0 | No | Yes |

There is one selected witness for each requested count, not a reliability
estimate. The nearby Pour2 settings demonstrate both successful and incorrect
parameters at the new height. Diagnostic recipes are not supplied to learner
samplers. No other seed was revalidated for this scene revision.

All seven cycles end ready, with zero audited forbidden contact steps across
2,949,400 physics steps. The audit covers robot contacts with supports and pan,
arm-ground contacts, and unintended pitcher contacts; it excludes robot
self-collision. Robot food recovery was not revalidated in this revision.

Local artifacts under `artifacts/scene-refresh/` include `geometry.json`,
`validation.json`, the three `v4-pour*/` trials, four `count2-tilt*/` trials,
native motion recordings, and the initial scene image. The Pour3 video shows
the complete practice cycle at 4x support-motion speed and 1x pouring speed.

## Checks and reproduction

The scene/predicate change passed 46 targeted tests. After the bounded IK
retry change, the 35 production/guard/launch tests were rerun and passed.
Selected mypy, lint, formatting and architecture checks passed.

Run each selected diagnostic in a fresh output directory, with workstation
memory/CPU limits as required:

```bash
scripts/with_env.sh python -m scripts.potsticker_readiness --root artifacts/pour1-check --seed 0 --target 1 --cycles 1
scripts/with_env.sh python -m scripts.potsticker_readiness --root artifacts/pour2-check --seed 0 --target 2 --cycles 1 --pour-tilt 89.1875
scripts/with_env.sh python -m scripts.potsticker_readiness --root artifacts/pour3-check --seed 0 --target 3 --cycles 1
```

The six ongoing learning runs remain frozen at `8823c4234` in the original
launch checkout. They use the previous scene. Their learning curves and the
previous scene's multi-seed witnesses must not be attributed to this revision.

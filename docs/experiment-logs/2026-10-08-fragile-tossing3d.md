# Fragile Tossing3D pilot — 2026-10-08

Status: all four production runs launched on October 8, 2026 around 16:40 EDT.
Della EES job **15238998** and expectimax job **15238999** are running. Both local
agentic systemd services are active; the original and new-wording variants each
have a $20 model cap. Independent 600-second monitors are running on both hosts.
Exact timestamps and service/job identities are in the launch receipts.
No final learning results are available yet.

| Method | Location | Seed | Human cost (each skill) | Damage cost | Mat |
|---|---|---:|---:|---:|---|
| EES | Della | 0 | 100 | 10 | 4 × 4 m |
| POMDP expectimax | Della | 0 | 100 | 10 | 4 × 4 m |
| Full-agentic original cost-sweep prompt | workstation | 0 | 100 | 10 | 4 × 4 m |
| Full-agentic new wording | workstation | 0 | 100 | 10 | 4 × 4 m |

Only these four runs are authorized for this pilot; earlier proposed larger sweeps
were superseded. Robot cost 1; human duration 1; human weight 1; lambda 3e-6.
Agentic: claude-opus-5-5, high effort, $20 cap EACH ($40 combined). Original here
includes the previously approved expanded hint; new wording adds the accumulated
future-cost paragraph. Both describe the fragile cube, protective mat, and damage.

85,000 counted practice steps; measurements every 1,700; 10 held-out tasks × 500
robot steps. Stop after three consecutive perfect evaluations. Expectimax: grid
25 × 16, depth 6, observation penalty 0; inherited cost filter uses 1,024 particles.
Damage is an added cost, not added experience. Wall impacts are unchanged.

## Validation

TDD failure logs are retained for missing environment, missing shared damage
accounting, and planner forecasts that initially omitted damage. Green checks:
8 contact tests; 99 accounting/planner/agentic tests; 38 bridge/CLI tests. Live
scene comparison showed identical initial and stepped physics; actual outside drop
charged 10, inside drop charged zero; human resets to both sides were exempt and
moved the non-colliding mat. Both structured 40-step smoke runs finished with
independent evaluations. Agentic restricted-container evaluation ran two steps
without model calls or transport errors. Smoke performance is not a learning result.

## Provenance and monitoring

Source commits, exact commands, costs and model settings are in
`artifacts/fragile-tossing-20261008/`. Runtime sources are frozen from commits;
shared mesh/texture assets are host-owned, read-only inputs. RoboCode source is
retained in the frozen runtime with a content hash in source-versions.json.
Every 600 seconds, independent host monitors save process state, evaluations,
reset counts and damage counts/costs. Monitoring never supplies evaluation
feedback to the coding agents and does not restart jobs automatically.

PR dependencies: KINDER #206 → HITL #417 → this results PR. Existing KINDER #204
and kinder-baselines #171 provide the previously used reset/controller APIs and
include latest main. No existing experiments were killed or overwritten.

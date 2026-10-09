# Fragile Tossing3D pilot — 2026-10-08

Status: all four production runs launched on October 8, 2026 around 16:40 EDT.
Both agentic runs and EES have finished; expectimax was still running in the
62,104-step audit snapshot below. Della jobs: EES **15238998**, expectimax
**15238999**. Each agentic run has a $20 model cap. Independent 600-second
monitors were installed on both hosts. Exact timestamps and service/job identities
are in the launch receipts. Results below refer to the original 0.1 kg bin.

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

## Post-launch review-only fixes

Type annotations and new-file lint were corrected after launch. These do not alter
experiment behavior; all four runs continue on the source commits recorded in
source-versions.json (HITL 4edfaa641, KINDER f564e8c, controllers c18056d).
KINDER CI additionally reports inherited lint/type failures in files outside the
new variant (placement_samplers, action-space return types, and existing modules).
The new environment module passes its focused type check. The integration's four
new annotation errors were corrected, and 53 affected tests passed again.


## Pilot finding: protected same-side practice works

The damage cost plus bin-attached mat produced the intended behavior in the
original-prompt agentic pilot: actual toss practice on the robot side, robot
retrieval of reachable misses and some bin landings, then transfer to far-side
held-out tasks. This is consistent with the intervention doing what we wanted;
one seed with the existing expanded hint does not isolate the causal effect of
the mat or damage cost from that hint.

Original agentic finished after 5,598 counted practice steps (5,582 robot steps),
with 12 same-side and 4 opposite-side human resets, zero charged damage contacts,
and three consecutive 10/10 scheduled evaluations (1,700, 3,400, 5,100 steps).
Final evaluation at 5,598 was also 10/10. Physical cost was 7,182. After the first
reset at step 927, seven logged throw attempts occurred before reset at 1,952.
The robot recovered the cube itself between attempts; retrieval occasionally
failed and still required human help.

The stitched original-prompt video is a recorded-state replay, not a new rollout:
`artifacts/fragile-tossing-20261008/original-video/cost-100/agentic-original-practice-5x.mp4`.
Duration 127.8 seconds; 5x motion; each of the 16 resets has a one-second pause
and destination label. The accompanying manifest reconciles 5,582 robot steps
and every reset with the final host counters.

### Expectimax reset comparison

At equal experience (5,598 practice steps), expectimax had **one same-side reset,
zero opposite-side resets**, and its latest evaluation was 0/10. Agentic had 16
resets and 10/10. Expectimax executed 26 toss-skill attempts between its resets at
steps 226 and 5,750. Thus it does reuse the cube; raw final reset totals conflate
strategy with the much longer time spent learning.

At the downloaded expectimax snapshot (62,104 steps): 70 resets, 68 same-side and
2 opposite-side. Of the 70 reset decisions, 51 had the cube in the same-side bin
with no PickPlannable atom, 13 elsewhere on the same side without PickPlannable,
and 2 had the cube across the barrier. Only human reset actions appeared among
root branches in those 66 decisions. Four other decisions offered another robot
action. The fixed retrieval controller's applicability is the main immediate
reset bottleneck; the agentic method writes its own controllers. Persistent 0/10
evaluation and 26 recorded motion-planning errors still need diagnosis before
attributing the performance gap to the expectimax algorithm.

Audit evidence: `artifacts/fragile-tossing-20261008/expectimax-audit/`, including
`reset-audit.json`, original decisions, step events, and status.

## Follow-up: heavy bin and smaller mat

The user observed the robot pushing the bin in the stitched video. The original
bin is only **0.1 kg**, and moving it moves the mat. Update KINDER draft PR #206
so this variant defaults to **100 kg**, scaling panel mass and inertia together.
Keep its free joint, existing friction/collision geometry, and both human reset
destinations. Ordinary Tossing3D remains unchanged. Frozen pilot sources and
already launched experiments retain the original mass; this is a subsequent
physics revision, not a correction to the existing results.

TDD: a physical push/reset regression failed at the original 0.1 kg mass, then
passed with the heavy bin. All nine fragile-environment tests passed. In the live
scene, a 50 N horizontal force over 400 physics steps moved the 100 kg bin only
2.48e-6 m. Live human resets to both sides also preserved the 100 kg mass and
mat/bin alignment. Increased mass resists pushing; it is not a mathematical guarantee of
immobility under arbitrary forces. No bin anchoring was added.

The current **4 x 4 m** mat was the explicitly approved generous starting size to
protect exploratory toss misses (2 m from bin center on each side), not a measured
minimum. We audited smaller squares against recorded near-floor toss endpoints,
using bin-local coordinates and a conservative enclosing radius for the cube:

| Mat side length | Original agentic endpoints covered | Expectimax endpoints covered |
|---|---:|---:|
| 1 m | 24/26 | 149/288 |
| 2 m | 25/26 | 206/288 |
| 2.5 m | 25/26 | 238/288 |
| 3 m | 25/26 | 263/288 |
| 4 m | 26/26 | 287/288 |

**Recommendation: test 3 x 3 m next** (44% less area). A 2 x 2 m mat is physically
compatible with successful bin landings, but would expose substantially more of
the structured controller's exploratory misses to damage cost. Feasibility does
not require every exploratory miss to be free of damage.

These are endpoint coverage checks, not exact collision counts, success forecasts,
or smaller-mat reruns. They omit intermediate bounces and use the old moving-bin
trajectories; heavier-bin dynamics can change those trajectories. No EES landing
coverage was measured here. Leave the default mat at 4 m pending the next size
choice. Evidence and reproducible analysis: `artifacts/fragile-tossing-20261008/`
`mat-size-audit.json` and `mat_size_audit.py`.


## Authorized rerun: 1 x 1 m mat and 100 kg bin (October 8, ~21:04 EDT)

User selected 1 x 1 m rather than the suggested 3 x 3 m and requested all four
experiments again. These are fresh seed-0 runs, not continuations of learned
agentic controllers. Preserve the previous 4 m/light-bin results separately.

| Method | Location | Job/service | Initial state |
|---|---|---|---|
| EES | Della | 15251746 | running |
| Expectimax | Della | 15251747 | running |
| Full-agentic original | workstation | hitl-fragile-mat1-heavy-original-h100-d10-seed0-20261008 | adapting |
| Full-agentic new wording | workstation | hitl-fragile-mat1-heavy-new-wording-h100-d10-seed0-20261008 | adapting |

Changed: mat side 4 -> 1 m; bin mass 0.1 -> 100 kg. Each generated agentic task
prompt explicitly describes the 1 m mat. Both practice and held-out evaluation
use the same new environment. Ordinary Tossing3D is unchanged.

Unchanged: human cost 100 per invocation; damage cost 10 per bare-ground contact
onset; robot step cost 1; human duration 1; 85,000 practice steps; measurements
every 1,700; 10 evaluation tasks with 500 robot steps each; early stop after three
consecutive 10/10 evaluations. Expectimax uses grid 25 x 16, depth 6, observation
penalty 0 and lambda 3e-6. Each agentic run uses Opus 5.5/high with its own $20 cap
($40 combined), and the same original/new-wording prompt distinction.

Validation: the new default/prompt regression failed on the prior 4 m default,
then all 13 focused tests passed. Physical drops at 0.4 m from mat center were
protected; at 0.6 m they incurred damage. Both structured 40-step smoke runs
succeeded. The sandbox evaluation smoke ran without transport/serialization
errors. Frozen runtime inspection confirmed 1 m mat extent and 100 kg bin mass.

Frozen commits: HITL eaca501fe; KINDER ec4d4ea; controllers c18056d2.
Commands, costs, launch receipts, runtime hashes, and smoke logs are retained in
`artifacts/fragile-tossing-mat1-heavy-20261008/`. Remote root:
`/scratch/gpfs/TSILVER/jr2860/experiments/fragile-tossing-mat1-heavy-seed0-20261008`.
Independent 600-second monitors run as workstation service
`hitl-fragile-mat1-heavy-monitor-10m-20261008` and Della tmux session
`fragile-mat1-heavy-monitor-10m-20261008`. No current results should yet be inferred
from startup health. Earlier jobs/results were not overwritten or cancelled.

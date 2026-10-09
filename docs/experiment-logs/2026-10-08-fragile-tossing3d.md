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


## Confirmed structured-controller dependency regression

The structured fragile results must not be interpreted as evidence against
expectimax or EES. Investigation of the persistent 0/10 found that both fragile
launch bundles froze controller commit c18056d2, which lacks the long-range
controller present in the preceding successful non-fragile human-cost sweep.
The old successful human-cost-100 expectimax run reached 10/10 repeatedly and
finished at 9/10. The first 4 m/light-bin fragile batch was already all-zero,
before the 1 m/heavy-bin changes.

The earlier successful runtime's composed toss controller allows simulation-only
max effort 3 (420 deg/s) and includes updated windup/held-object planning. The
fragile bundles carry the older max effort 1 (140 deg/s) implementation. The HITL
proposal still samples long-range speeds and standoffs. Requests such as 295.83
deg/s are therefore silently capped by the older controller. Recorded evaluation
throws fall short beyond the wall, after which `no_plan` is the expected terminal
reason because human recovery is unavailable in evaluation.

Matched diagnostic on held-out task seed 357381689, using the identical previously
trained snapshot (SHA256 94eec61751a57c585701d19b49cea206c8fc705d1e4ae81c8b7ba5ef51b7d647):

| Environment | Controller dependency | Result |
|---|---|---|
| Ordinary Tossing3D | frozen c18056d2 | fail, no_plan, 239 steps |
| Fragile, 1 m mat / 100 kg bin | frozen c18056d2 | fail, no_plan, 239 steps |
| Fragile, 1 m mat / 100 kg bin | previous successful runtime's controllers | success, 228 steps, zero damage |

Only the controller package path changed between the last two diagnostic runs;
there was no retraining or change to the saved policy, task, mat, or bin. This
establishes the controller-version regression on a real matched task, while not
isolating the speed cap from the other controller differences. The trained new
sampler was present (202 examples, 68 positives at snapshot 24), so an empty or
missing restored learner is not the explanation.

The launch validation checked execution, accounting, and short smoke rollouts,
but omitted a known-success long-range controller regression; that should have
been caught before launching. Restore the correct long-range dependency, add a
behavioral regression, and rerun affected EES/expectimax experiments. Both fragile
batches are affected; full-agentic writes its own controller and is not affected
by this fixed-controller mismatch. Production jobs were not altered during this
read-only investigation. Diagnostic evidence is under
`artifacts/fragile-tossing-mat1-heavy-20261008/failure-audit/`.


## Corrected structured reruns — October 8, ~23:56 EDT

User authorized rerunning the affected structured methods, reiterated monitoring
every 10 minutes, and explicitly prohibited further agentic runs. Exactly two
fresh seed-0 jobs were launched; no agentic jobs or model calls were launched.

| Method | Replacement Della job | Replaces | Initial status |
|---|---:|---:|---|
| EES | 15256890 | 15251746 | running |
| Expectimax | 15256891 | 15251747 | running |

Cancelled the two invalid jobs above, retaining their artifacts. The original
4 m batch is already finished; it was not relaunched because the current selected
environment is 1 m. The completed agentic runs are retained.

Frozen controller runtime is now **427ad6cdfcc79efb489ebabc8fd39d77e104bb4e**,
the exact long-range/settled-windup/attached-grasp implementation from the earlier
successful sweep (kinder-baselines PR #174). HITL eaca501fe and KINDER ec4d4ea are
unchanged. Runtime throwing source files match the previously successful bundle.

Same settings: 1 x 1 m mat, 100 kg bin, human cost 100, damage cost 10, robot cost
1, human duration 1; 85,000 practice steps, evaluation every 1,700 steps, 10 tasks
x 500 robot steps, stop after three consecutive 10/10 measurements. Expectimax
remains grid 25 x 16, depth 6, observation penalty 0, lambda 3e-6.

Prelaunch range guard failed against the old dependency as intended and passed
against the replacement. The previously successful held-out policy solved task
seed 357381689 in 228 robot steps with zero damage under the replacement's frozen
1 m/heavy-bin environment. This saved-policy diagnostic is not training data for
the fresh reruns. Both 40-step structured smoke runs passed. Swing, effort, and
attachment/handoff regressions: 26 passed. One stale test had incorrectly tested
the extended controller's speed range using the helper's default effort ceiling;
it now explicitly passes the controller's simulation ceiling (test-only commit
527fb91, pushed to existing PR #174). Runtime remains pinned to 427ad6c.

Both Della job scripts run the controller range guard before starting training;
their logs confirm it passed. The monitor runs independently in persistent Della
tmux session `fragile-mat1-fixed-monitor-10m-20261008`, every 600 seconds, recording
job state, progress, evaluation scores, resets, damage, and warnings. It does not
restart jobs or launch agentic runs.

Receipts, exact command arguments, source versions, validation evidence and
monitor/launch scripts are in `artifacts/fragile-tossing-mat1-fixed-20261008/`.
Remote root: `/scratch/gpfs/TSILVER/jr2860/experiments/fragile-tossing-mat1-fixed-seed0-20261008`.


## Corrected run endpoints — October 9

Both corrected Della runs completed cleanly at 85,000 practice steps, with all 51
evaluations complete and no monitor warnings. Neither triggered the three-perfect
evaluation early-stop rule. No additional agentic runs were launched.

| Method | Final eval | Best eval | Human resets total / same / opposite | Damage contacts | Physical cost | Finished EDT | Duration |
|---|---:|---:|---|---:|---:|---|---|
| EES | 8/10 | 10/10 | 314 / 0 / 314 | 290 | 118,986 | Oct 9 03:15 | 3h19m |
| Expectimax | 9/10 | 10/10 | 98 / 89 / 9 | 170 | 96,402 | Oct 9 04:31 | 4h35m |

The recovery from persistent 0/10 after restoring the controller is consistent
with the diagnosed dependency regression. These are single-seed results, and
best score is not the final score. The retained agentic endpoints use different
amounts of practice; raw totals are not equal-experience comparisons.

Full endpoint status and evaluation histories are preserved in
`artifacts/fragile-tossing-mat1-fixed-20261008/final-results.json`.

# SweepTable3D production integration and preflight — 2026-10-28

## Scope

Integrate the ball-and-shallow-tray investigation as the first-class
`--env sweeptable3d` environment. See the [contract and commands](../sweeptable3d.md).
This is a draft implementation and bounded local preflight, not an overnight
experiment or evidence of learned sweep competence.

The clean branch starts from main `70c3f63d202d2e6870e167d778615e61b077230d`.
Dependency pins are unchanged: KINDER `490831fa54a40c4b2b4570d45bb08e3dd63c24ef`
and kinder-baselines `032b42634a015c7f710667b3a791836fa839f6ff`.
No unrelated drawer/shelf investigation changes or large recordings are included.

## Test-first changes

Observed failing tests preceded implementations for sphere containment/settling,
contact-tolerant planning, angle/speed dispatch, single-attempt task semantics,
ground-failure precedence, CLI/learner/replay wiring, and preflight reporting.
Additional RED/GREEN regressions caught scene object order, rolling-contact setup
timing, production background selection, and CI omission of the new environment.
The generated learner commands also had a failing regression before adding the
required POMDP observation-weight flag, bounded search and cost/draw logging.

## Initial physical findings

Artifacts are retained locally under `artifacts/sweeptable-pr-20261028/` in the
parent checkout. `full5-final` and `human-final` are **pre-review** diagnostics;
the directory names do not mean that later recovery fixes were already present.
Both report `source_unchanged=true` during their runs, with source-manifest digest
`b4f8a721ee2e90a1a9acb061f4d5f8c14fe7c2e7b513a0ae5a5c2f24cf6fcd27`.

| Check | Observed result |
| --- | --- |
| Production initialization, seed 0 | Exact qpos hash matches the successful prototype; complete value below |
| Initial sweep, angle -27 degrees, speed 0.15 m/s, bin -0.18 m | All five balls settled inside the strict tray boundary; 768 control steps after 525-step pickup |
| Physical wiper return | Successful, 744 control steps |
| Full-five robot restoration | Failed on the first ball: three grasp candidates did not carry `cube_0`; 936 control steps charged |
| Deliberately bad +45 degree sweep | Aborted on hand tracking; the next sweep was blocked with zero motion |
| Human restoration after failed attempt | Wiper physically returned, human ball reset explicitly labeled, attempt latch cleared |
| Following sweep after human reset | Executed physically; `cube_2` hit the ground and the attempt failed |

The complete initial qpos hash is
`46adcaae203f15fd68530a0e2bacbb7539ceba3617d2959c317311b8e0ba25f5`.
Earlier `full5`, `full5-v2` and `human` artifacts used a different background or
initialization ordering and are not production validation. Background selection
changes physical colliders, not just appearance. Neither task thresholds nor
contact physics were loosened to make these gates pass.

## Initial learner entrypoint checks

Both methods ran with 3,000 practice steps, one evaluation, a 3,000-step evaluation
cap, a fixed -0.18 m tray, production background, human reset enabled, five sampler
training iterations and bounded POMDP search. Commands use the existing
`configs/sweepoffshelf3d_practice_costs.json` because the reset names match. Its
human cost/duration values are a smoke-test convention, not task-specific empirical
cost calibration.

| Method | Recorded completion | Learning updates | Human invocations | Evaluation |
| --- | --- | --- | --- | --- |
| EES | 3,000 practice steps, one completed evaluation | 2 | 0 | Failed after one sweep; `task_failed`, 994 steps |
| POMDP | 3,000 practice steps, one completed evaluation | 5 | 5 | Failed after one sweep; `task_failed`, 1,142 steps |

These runs exercised manipulation, charging, checkpoint/evaluation and terminal
failure handling. **Neither practiced SweepTable within this short budget**, so
they do not validate sweep sampler training or improvement. EES wrote complete
status/results and its bounded scope reported success, but its outer launcher
returned 143; POMDP returned zero. Retain this distinction rather than treating
all process exits as clean.

## Review findings

Review identified missing symbolic final-ball-reset reactivation, inherited shelf
collision geometry in the table planner, capsule obstacle-axis mismatch, and a
human wiper reset that could open a gripper holding a ball. These are integration
issues distinct from the task's intended sweep difficulty. They prompted follow-up
tests and new physical checks rather than attributing every failure to competence.

All four findings are fixed. Conditional return effects now make the next sweep
symbolically reachable after complete restoration; target encoding and traces
retain the selected ball despite the additional grounding slots. The planner uses
the live table collision geometry and correctly oriented capsule obstacles.
`reset_wiper` refuses a held ball before any physical motion, with matching
symbolic applicability. Physical ball-ball collisions remain enabled.

## Post-review static validation

- 654 focused domain, integration, evaluation, harness and CI-selection tests pass.
- The adjacent shelf regression run passes 45 tests with 2 existing expected failures.
- Ruff lint/format, mypy over 257 source files, import boundaries and documentation
  links pass. A wheel builds without downloading dependencies and contains the
  table scene JSON.
- Final heavy runs use source-manifest digest
  `6268465946bc6f75dbbdd9ff9de31e1939cd750b7f997bd3eb312fa8db4b14a0`.
  The manifest hashes working-tree content because these preflights precede the
  implementation commit. The exact manifest and raw recordings remain local.

## Post-review physical recovery

`full5-post-review` reran the production `full5` case with seed 0, background on,
900-second case limit and motion recording. The source remained unchanged.
Pickup (525 steps), all-five capture (768) and physical tool return (744) pass.
The first tray-to-table return still fails after three unsuccessful physical
grasp candidates, now charging 859 steps. The harness exits 1 and does not
substitute a human reset or advance to the later sweep.

This is a real, recorded recovery limitation, not a claim that every ball or
layout is unpickable. The previous two successful single-ball diagnostics are
not evidence that a full five-ball autonomous reset is reliable. Imperfect
robot resets can be part of the human-assisted practice problem; a robot-only
comparison must not assume reset reliability that has not been demonstrated.

## Post-review learner and replay checks

The frozen-source EES smoke completes 3,000 practice steps, three learning updates
and one evaluation in 233 seconds, exiting zero. Two physical returns of `cube_4`
from the tray succeed (511 and 531 charged steps), adding `BallsRestored` and
`OnTable`; another return is interrupted at the explicit practice budget.
The evaluation ends `task_failed` after one sweep at 1,119 control steps. This
run still does not practice SweepTable, so it is integration evidence, not a
sweep-learning result.

The final three native tests pass, including actual airborne-to-ground contact
latching and reset behavior. The shared renderer successfully replays the
post-review full-five diagnostic from saved states: `replay.mp4`, 585 frames,
58.5 seconds at 5x practice playback. It includes the successful stroke and failed
grasp; it is not a new simulation or an edited success-only demonstration.

The post-review POMDP run requests 5,000 practice steps but hits the external
1,200-second wall-time cap (exit 124). Its last persisted progress is 3,997
normalized practice steps, six learning updates and five human invocations;
no checkpoint evaluation completed and no SweepTable practice was recorded.
The last recorded robot action is a successful tray return. The final serialized
status still says `running` because the external timeout terminated the process;
the process/scope is no longer active. This is **not a passing POMDP preflight**.
The long interval without new progress needs profiling before selecting overnight
search budgets; this log does not assert a diagnosed root cause.

Source file hashes, HEAD and dependency pins match the saved manifest after all
post-review jobs finish. Staging files changed git-status metadata, not source
content. All local preflight jobs are stopped; no overnight jobs were submitted.

## Launch decision

The follow-up below supersedes individual blocker diagnoses here, not the
requirement to validate sweep learning before overnight comparisons.

Keep the environment PR draft and do not launch unrestricted overnight comparisons.
Before doing so:

1. Resolve/profile the final POMDP timeout and complete its bounded evaluation gate.
2. Exercise actual SweepTable practice, sampler draws, binary outcomes and training;
   generic learning-update counters are insufficient evidence for sweep learning.
3. Check solvability/parameter coverage across the intended ball-layout seeds and
   calibrate human costs and controller budgets for the comparison being run.
4. Retain human recovery for a pilot. Require a demonstrated full physical reset
   cycle before treating robot-only recovery as reliable.

The implementation, test-first regressions and reproducible failure evidence are
ready for review. The empirical launch gates are not all green.

## Follow-up: controller and bookkeeping fixes

These changes preserve the core algorithms: no objective, prior, sampling rule,
STOP rule, skill repertoire, task threshold or learner configuration is changed.
Follow-up artifacts are local under `artifacts/sweeptable-fixes-20261028/` in the
parent checkout.

### Timeout diagnosis and correction

The old POMDP log contains 22 completed search decisions totaling 13.003 seconds
of measured search time, but approximately 53.44 GB of uncompressed diagnostic
records. The large payload is primarily execution-cost tables whose keys contain
the complete Pydantic skill definition, repeated both in forecasts and belief
snapshots. It is not a 53 GB search tree. The precise final stalled stack was not
profiled, so the original timeout alone does not establish its entire time split.

The model also constructed 37,518 ground actions while only 30 satisfy the fixed
`CubeOrder` chain. The provider now explicitly declares that invariant, and the
grounder prunes only invariant-false combinations. The unfiltered and filtered
models produce identical ordered feasible actions and exact decisions/values
for both expectimax and determinized A* in four representative symbolic states.
Domains without declared invariants retain their old grounding behavior.

Cost diagnostics now use compact skill/object labels. Belief diagnostic snapshots
omit the duplicate forecast table, retaining those values in `estimated_costs`.
Internal cost keys, cost estimates and persisted belief models are unchanged.
Logging-enabled and logging-disabled planner decisions and beliefs agree in the
regression test. No full-model serialization occurs merely to build a belief
log payload when logging is disabled.

### Physical recovery

The table controller seats spheres farther inside the pads while preserving
1 mm support clearance and rejecting tip-only tilted candidates. Failed-grasp
history distinguishes full orientations and offsets relative to the target,
rather than excluding every vertical grasp after one miss. Capture is checked
before lifting and retention during lifting. These behaviors are table-specific;
the inherited shelf-cube depth and approach-axis retry defaults are preserved.
Planning still allows neighboring-ball contact, while physical collisions remain on.

The frozen-source seed-0 rerun reproduces the original initialization exactly.
Pickup, the initial 5/5 sweep and physical wiper return remain successful.
All 5/5 tray balls are then physically returned to the table, taking 1,127, 515,
460, 530 and 483 control steps respectively. There are no human resets or
unlabeled pose writes. The first ball was the prior failing recovery case.

The next pickup and sweep execute, but only 4/5 balls meet strict tray containment;
one is near the boundary. Therefore `controller-full5/report.json` still reports
`passed=false`, `physics_gate=incomplete_or_failed`, and `launch_ready=false`.
This is a demonstrated recovery cycle, not a passing repeated-delivery gate or
a reliability estimate across seeds. Source hashes remain unchanged throughout,
with manifest digest
`de7ce32e5dd4c15447c218aec04270a86fc30bd3018da0fcf7b5d8e553c3d8c2`.

### Learning-coverage diagnosis, not an algorithm workaround

EES's smoke explicitly disables initial goal pursuit, unlike the CLI default,
and only previously executed skills enter its practice-candidate set. This
matches the older shelf launcher, so the override is not established to be an
accident. It remains unchanged. A separate default-EES diagnostic would omit
that override; a new horizon of two is not the canonical default.

In the old POMDP run, SweepTable was a candidate at decisions 4, 5, 14 and 15.
At decision 4, STOP scored 0.102593 versus 0.094528 for the sampled sweep
continuation; at decision 14, the values were 0.135336 versus 0.120802.
Both searches exhausted their remaining one-action space in 4/20 permitted
iterations. Raising the iteration cap would not resolve those decisions.
Other selected paths use failure-dependent reset or pickup detours before a
favorable sampled sweep. Subsequent real success changes the state, causing
replanning. This supports a sampled-outcome explanation, not missing sweep
actions. No changes to that algorithm are made to force training coverage.

### Same-configuration POMDP preflight

`pomdp-fixed` completes 5,000 practice steps and one evaluation in 336.034 seconds
with exit zero, within the same 1,200-second cap. Its argument snapshot differs
from `pomdp-post-review` only in output directory. It records seven learning
updates and five human resets. The evaluation scores 0/1; successful completion
of the harness is not task success.

The new diagnostic stream contains 101 records totaling 2,984,005 uncompressed
bytes, with a largest record of 240,376 bytes. Its 24 measured searches total
0.737 seconds. These are single-run engineering measurements, not a statistical
claim about learning performance. The controller also changed, so physical costs
and later practice decisions need not match the earlier run; algorithm parity
is checked separately by the symbolic tests.

The unchanged planner selects SweepTable at practice step 4,466. The 5,000-step
cap interrupts that stroke after 534 control steps, so this run alone does not
demonstrate a complete practice sweep. A separate 6,000-step coverage diagnostic
extends only the practice/measurement budget, not planner settings or selection
logic. Both runs use the frozen source manifest recorded above.

The coverage diagnostic completes that practice attempt at step 5,081 with
`interrupted=false`: angle -14.013384 degrees, speed 0.298729 m/s, 615 charged
steps, and a real floor-contact failure for `cube_3`. `sampler_draws.jsonl`
records the binary failure and parameters. Cycle 6's refit includes one
SweepTable training example and one fitted failure, with no success examples.
This validates the complete execution-to-learning path, not a useful learned
classifier or improved success rate. No skill was forced into the chosen plan.
It finishes all 6,000 practice steps and one evaluation in 390.481 seconds,
exiting zero, with eight learning updates and six human resets. Evaluation is
0/1. The configuration comparison confirms that only output directory, practice
budget and matching measurement interval differ from the 5,000-step diagnostic.
The final source-file manifest still matches the frozen digest.

### Follow-up test evidence

Regression tests were observed failing before the corresponding changes.
The bookkeeping/planning checks pass 212/212 tests in the combined focused run;
the separately memory-capped unfiltered-versus-filtered planner parity test also
passes. Controller/domain invocations cover 475 unique passing tests: 385 domain
tests and 117 recovery/controller/shelf tests, with 27 overlapping tests. These
groups overlap the bookkeeping run and must not be added into a grand total.
Ruff, full-source mypy (257 files), import boundaries and documentation links pass.
Independent read-only review found no actionable issue in the bookkeeping diff.
The three native table-physics regressions also pass after the fixes.

The recovery and runtime defects are addressed for these bounded checks. EES's
experimental configuration, sweep success across seeds/parameters, human-cost
calibration and actual learning improvement are not established by these fixes.
Keep the PR draft; no overnight experiments were launched.

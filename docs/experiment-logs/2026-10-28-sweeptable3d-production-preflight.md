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

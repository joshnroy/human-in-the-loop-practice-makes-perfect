# Fragile Tossing3D without the human reset skill (seed 0)

**TL;DR.** Every completed arm of the fragile Tossing3D comparison was re-run from its own
frozen source with the human reset skill removed from practice (`--no-human-reset`). All
seven runs are complete. The reset-free hypothesis held for the three structured planners
and failed for three of the four agentic learners. EES, POMDP Planner / expectimax and
POMDP Planner / determinized A* stranded the cube across the wall within 590 counted steps
and ended with 1/10, 0/10 and 0/10. Full-Agentic / original (7,389 steps), Full-Agentic /
new wording (12,435 steps) and POMDP + Agentic Skills & States (15,121 steps) each reached
10/10 without a human. POMDP + Agentic Skills was partial: 0/10 at every measurement up
to 15,300 steps, then 5/10 at its 16,229-step endpoint only. The agentic runs paid in
raw-ground hits instead of resets (67-345 hits against 23-97 with the human).

## Question / goal

How much of each method's learning on fragile Tossing3D depends on the human reset skill?
The with-human comparison (1 m mat, movable 100 kg bin, human cost 100, damage cost 10,
seed 0, 85,000-step budget) gave every method a human who could put the cube back on the
robot's side for a cost. This entry measures the same methods with that skill withheld
during practice. Evaluation never offered a human in either condition, so the only change
is what the learner can do while practising.

## Background

The bin starts across the wall from the robot. The one trajectory that reaches the goal is
pick on the robot's side, then toss over the wall; the wall is a one-way door, because the
robot cannot cross it. In the with-human runs the human reset was what turned a stranded
cube back into practice material: EES called it 314 times, POMDP Planner / expectimax 98
times, the five A* seeds 300-376 times each, while the agentic learners needed 4-18 calls.

Every arm here is a re-run of a completed with-human arm from its own frozen source
(`hitl` eaca501fe, `kindergarden` ec4d4ead, `kinder-baselines` 427ad6cd for the three
Della arms; `provenance.json` in each artifact directory records the source and the one
flag that changed). The Full-Agentic arms additionally carry PR #421's patch, which removes
the human reset's language and requirement from the full-agentic protocol, and the two
hybrid arms carry PR #422's hybrid-skills patch, which does the same for hybrid runs. A
first POMDP + Agentic Skills attempt was aborted after five minutes because its frozen
task prompt still described both human reset skills and the human-cost hint although the
planner no longer offered them; it is preserved under
`artifacts/fragile-nohuman-lccc-20261013-aborted-prompt-still-described-human/ABORTED.json`
and the relaunch uses the prompt-gating change.

Method codes follow the planner-first order adopted on 2026-10-13 (planner, estimator,
states, skills): POMDP + Agentic Skills is CCCL and POMDP + Agentic Skills & States is
CCLL. The artifact directories keep their older names (`lccc`, `llcc`) and are not renamed.

## Hypothesis

With the bin starting across the wall, the first far-side toss strands the cube, so
reset-free practice should learn little. The structured planners should stop early; the
agentic learners might find a way to keep practising without ever committing the cube.

## Guidance given

Re-run each completed arm from its own frozen source with exactly one change,
`--no-human-reset`, the same seed, the same costs and (for the agentic arms) the same
$20 model budget. Do not touch the evaluation protocol. Report actual endpoints; never
extend a curve past the step at which the run stopped. Report counts as `x/y`.

## Methods

| Setting | Value |
| --- | --- |
| Environment | Fragile Tossing3D, barrier layout, 1 m square mat, movable 100 kg bin |
| Seed | 0 (the with-human A* comparator has five seeds, 0-4) |
| Costs | Robot control step 1; raw-ground contact 10; human reset 100 (never offered here) |
| Practice budget | 85,000 counted steps; measurement every 1,700 counted steps |
| Evaluation | 10 held-out episodes of at most 500 control steps, separate frozen process, no human |
| Early stop | Three consecutive complete 10/10 evaluations |
| Step protocol | A run ends when a practice cycle executes zero actions (`no_practice_action`) |
| Structured planners | EES; POMDP Planner with expectimax or determinized A*, depth 6, grid 25 x 16, local-trend beliefs, lambda 3e-6 |
| Agentic learners | POMDP + Agentic Skills; POMDP + Agentic Skills & States; Full-Agentic with the original and the new-wording prompt; Claude Opus 5.5, $20 budget each |
| Where | Della jobs 15486053 (EES), 15486054 (expectimax), 15486055 (A*), submitted 2026-10-13 10:01 EDT; the four agentic arms on the workstation, started 10:26-10:33 EDT |

With-human comparators are the completed runs already reported: EES and expectimax from
`experiments/fragile-tossing-mat1-fixed-seed0-20261008` on Della, A* from
`fragile-astar-grid-5seeds-20261010`, and the four agentic runs under
`artifacts/hybrid-expectimax-agentic-seed0-20261009-r3/`,
`artifacts/llcc-fragile-seed0-20261010-r2/` and
`artifacts/fragile-tossing-mat1-heavy-20261008/`. Their numbers below were re-read from
those run directories, not copied from the earlier write-ups.

Figures and the data behind them are built by `analysis/fragile_no_human_reset.py`
(tests in `tests/analysis/test_fragile_no_human_reset.py`): it reads each run's
`status.json`, `step_evaluations.json` or `measurements` plus `evaluations/*/results.json`,
and the practice event stream, checks that the damage events sum to the status total, and
writes `curves.json` beside this entry before plotting. `curves.json` was regenerated at
2026-10-13 14:03 EDT, after every run had finished.

## Results

### Final evaluation and practice spent, no human reset vs. with human reset

Robot actions are skill invocations for the planners (control steps are not comparable
for the full-agentic policies, which act at the control level). Human resets are 0 in
every no-human arm by construction.

| Method | No human reset: final eval | Counted steps used (of 85,000) | Robot actions | Raw-ground hits | Endpoint | With human reset: final eval / resets / hits / steps |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| EES | 1/10 | 590 | 6 (3 picks, 3 tosses) | 4 | `no_practice_action` | 8/10 / 314 / 290 / 85,000 |
| POMDP Planner / expectimax | 0/10 | 221 | 2 (1 pick, 1 toss) | 0 | `planner_stop`, then `no_practice_action` | 9/10 / 98 / 170 / 85,000 |
| POMDP Planner / determinized A* | 0/10 | 221 | 2 (1 pick, 1 toss) | 0 | `planner_stop`, then `no_practice_action` | 8.2 +/- 0.84 of 10 (n=5) / 330.4 +/- 29.4 / 218.2 +/- 11.6 / 85,000 |
| POMDP + Agentic Skills (CCCL) | 5/10 | 16,229 | 106 | 97 | `no_practice_action` | 10/10 / 4 / 97 / 15,654 |
| POMDP + Agentic Skills & States (CCLL) | 10/10 | 15,121 | 122 | 78 | `no_practice_action` | 9/10 / 11 / 38 / 33,575 |
| Full-Agentic / original | 10/10 | 7,389 | n/a | 67 | `agent_submission` | 10/10 / 18 / 23 / 11,340 |
| Full-Agentic / new wording | 10/10 | 12,435 | n/a | 345 | `agent_submission` | 10/10 / 7 / 86 / 7,467 |

The A* row's with-human column is the mean and sample standard deviation over seeds 0-4
(final evaluations 9, 8, 7, 8, 9 of 10; resets 310, 376, 330, 300, 336; hits 211, 212,
238, 211, 219). The no-human A* arm is seed 0 only.

Model spend of the no-human agentic arms: POMDP + Agentic Skills $8.02 (7 calls),
POMDP + Agentic Skills & States $15.26 adaptation (8 calls; evaluation classification is
a separate ledger), Full-Agentic / original $12.64, Full-Agentic / new wording $4.54.

![Held-out tasks solved per seed against counted practice steps, no human reset dashed and with human reset solid](2026-10-13-fragile-no-human-reset/2026-10-13-fragile-no-human-reset-performance.png)

![Cumulative raw-ground hits per seed against counted practice steps, same arms and styling](2026-10-13-fragile-no-human-reset/2026-10-13-fragile-no-human-reset-raw-ground-hits.png)

Dashed lines are the no-human arms, solid the with-human comparators, in each method's
colour; the five with-human A* seeds are the faint traces under the bold mean. Diamonds
are actual endpoints. The no-human structured arms are the three short dashed segments in
the bottom-left corner of the left panels, labelled individually because they are
otherwise invisible at the 85,000-step scale.

### Evaluation trajectories of the agentic arms, no human reset

| Method | Evaluations at 0, 1,700, 3,400, ... counted steps, then the endpoint |
| --- | --- |
| POMDP + Agentic Skills (CCCL) | 0/10 at each of 0 through 15,300 (10 measurements), then 5/10 at 16,229 |
| POMDP + Agentic Skills & States (CCLL) | 0/10 at each of 0 through 10,200 (7 measurements), 9/10 at 11,900, 10/10 at 13,600, 10/10 at 15,121 |
| Full-Agentic / original | 0, 0, 0, 0 of 10 at 0-5,100; 10/10 at 6,800; 10/10 at 7,389 |
| Full-Agentic / new wording | 0, 0, 0 of 10 at 0-3,400; 10/10 at 5,100; 9/10, 9/10 at 6,800, 8,500; 10/10 at 10,200, 11,900, 12,435 |

The POMDP + Agentic Skills 5/10 comes from the endpoint evaluation alone, after its last
skill revision; no earlier measurement solved any task, so there is no evidence that
level would hold.

### The robot strands itself

All three structured no-human runs end the same way, and the videos show it directly.

- **EES** picked and tossed three times; every action is labelled `random:` in
  `step_events.jsonl`, the `--reproduce-predicators-random-when-stranded` behaviour
  its arguments select. The first
  two tosses hit bare ground near the robot's side (impacts at counted steps 209, 210
  and 404) and the cube could be picked up again; the third toss (impact at step 582)
  left the cube beyond the wall. The next practice cycle executed zero actions and the
  run ended at 590 counted steps, 593 s of wall-clock.
- **POMDP Planner / expectimax** picked once and tossed once with parameters
  `[2.18, 382.65, 438.20]` (action vector `[1.0, 2.1809, 0.0, 382.6463, 438.1985]`). The
  toss registered no raw-ground contact, so the cube came to rest across the wall on the
  mat or in the bin; either way the robot could not reach it. The planner stopped the
  session (`planner_stop`), the next cycle had no action, and the run ended at 221 steps.
- **POMDP Planner / determinized A*** made the identical two decisions with identical
  parameters and identical step counts, because the same seed produced the same first
  decision and the solver difference never got a chance to matter. Its evaluation
  policy snapshots have the same SHA-256 as expectimax's at both measurements, so the
  two 0/10 results are the same evaluation run twice.

### Step-0 versus endpoint evaluation

The step-0 evaluation is the untrained policy. EES scored 1/10 at step 0 and 1/10 at its
endpoint; one learning update over six actions did not change what the evaluation policy
could do. Both POMDP Planner arms scored 1/10 at step 0 and 0/10 at the endpoint: the
single stranded toss updated the competence belief and the refit toss parameters, and the
one task the untrained policy had solved (task 0, in 226 control steps) was no longer
solved by the refit policy, which reported `no_plan` on all 10 tasks. With one seed
and one evaluation each, a 1/10 to 0/10 change is
within what the evaluation's own variability can produce; no inference is supported beyond
"no improvement". The four agentic arms all scored 0/10 at step 0.

### How the agentic arms practised without a human

The agentic learners did not strand themselves early because they did not commit the cube
across the wall while practising. Full-Agentic / new wording is the clearest case: its
cube crossed the wall exactly once, on the last throw at step 12,427, and 64/67 of its
recorded landings were at x < 1.0 m, on the robot's side; it calibrated its throw on
near-side landings and then submitted. That trade shows in the hit counts: 345 raw-ground
hits against 86 for the same prompt with the human. Full-Agentic / original used 67 hits
(23 with the human) and fewer steps than its with-human run (7,389 against 11,340).
POMDP + Agentic Skills & States reached 10/10 in fewer counted steps than its with-human
run (15,121 against 33,575, which ended on the model budget at 9/10). These are single
seeds; a second seed is needed before reading either step comparison as an effect.

The landing-side figures for Full-Agentic / new wording are the coordinator's verified
analysis of its `practice.jsonl`; they were not recomputed for this entry.

Videos (practice at 5x, then the endpoint evaluation at 1x; Della and workstation renders
from `artifacts/fragile-nohuman-20261013/videos/`, copied byte for byte):
[EES](2026-10-13-fragile-no-human-reset/ees-no-human-reset-practice-and-final-eval.mp4),
[POMDP Planner / expectimax](2026-10-13-fragile-no-human-reset/expectimax-no-human-reset-practice-and-final-eval.mp4),
[POMDP Planner / determinized A*](2026-10-13-fragile-no-human-reset/astar-no-human-reset-practice-and-final-eval.mp4),
[POMDP + Agentic Skills](2026-10-13-fragile-no-human-reset/hybrid-skills-no-human-reset-practice-and-final-eval.mp4),
[POMDP + Agentic Skills & States](2026-10-13-fragile-no-human-reset/hybrid-skills-states-no-human-reset-practice-and-final-eval.mp4),
[Full-Agentic / original](2026-10-13-fragile-no-human-reset/fullagentic-original-no-human-reset-practice-and-final-eval.mp4),
[Full-Agentic / new wording](2026-10-13-fragile-no-human-reset/fullagentic-new-wording-no-human-reset-practice-and-final-eval.mp4).

## Recommendation

The reset-free hypothesis held for the structured planners and failed for the agentic
learners. EES and both POMDP Planner solvers depend totally on the human reset in this
scene: without it they execute one or three tosses and have nothing admissible left, so
their with-human results are results about the reset budget as much as about the planner.
A planner that could choose short, near-side practice tosses on purpose, as the
Full-Agentic arms did, would have had 84,000 more steps to use; that is a model gap to
report, not a constraint to add. Three agentic arms reached 10/10 without a human, so
their with-human reset counts (7-18) measure convenience rather than necessity. Every arm
here is one seed: before claiming that the agentic learners need no human, run a second
seed of both conditions for Full-Agentic / new wording and POMDP + Agentic Skills & States,
and treat POMDP + Agentic Skills' 5/10 as an endpoint-only observation.

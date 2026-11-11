# Real TidyBot paper runs (MLP sampler, Curtis planner), records

Code: this branch's parent commit 0fd3d8b4b (`launch/real-20261109e`); robot code prpl-tidybot `ce25bae` on the
October package environment (`robot-october-environment-uv.lock`). Server start commands: `cmd-*.txt`.
Each run folder holds the checkpoint and snapshot pickles, evaluations, the config snapshot, `GROUND-TRUTH.md`
(Josh's in/out calls beside the robot's) and the robot's `skill_calls.jsonl`. Webcam video and the full robot
logs are not committed (GBs); they stay in the results folder on agni.

| seed | evaluation 0 | practice cycle 0 | evaluation 1 |
| --- | --- | --- | --- |
| 0 | 1/9 thrown | see GROUND-TRUTH.md | 8/9 thrown |
| 1 | 2/10 | 2/9 | 9/10 |
| 2 | 2/10 | 3/7 | 8/9 thrown |
| 3 | 4/10 | 3/8 | 8/10 |
| 4 | 1/10 | 3/9 | 9/10 |

Note added 2026-11-11: seeds 0-2 ran 0fd3d8b4b. Seeds 3 and 4 ran `launch/real-20261110a` @ b0dd2d27d, which is
0fd3d8b4b plus one harness commit (a scene look that sees no cube asks for it in view and looks again); the
planner is unchanged. "x/9 thrown" means one task was never thrown (failed pick) and is left out. Counts are
the run's own counter. Two throws are still disputed between the toss reply and the run counter: seed 2
evaluation 0 task 9 and seed 3 evaluation 1 task 5 (see each GROUND-TRUTH.md).

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

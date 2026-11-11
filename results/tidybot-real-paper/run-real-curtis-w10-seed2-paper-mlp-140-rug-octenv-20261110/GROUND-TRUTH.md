# Ground truth: paper seed 2 (MLP, fixed 1.40 m, box on rug, ce25bae + October env)

Started 2026-11-10 ~09:37 ET by Josh (y in tmux 0 `run`). Same code and servers as seed 1's evaluation 1
(launch/real-20261109e @ 0fd3d8b4b; skill server started 09:01 ET, NUC logs 2026-11-10/09-01-27).
Josh's calls are typed in chat as they happen; the robot's call is from the NUC skill_calls.jsonl.
Predicted throws (dry runs, no robot): ../dryrun-paper-seed2-20261110/ and ../../dryrun-paper-seed2-20261110-allmiss/.

## Evaluation 0 (before practice)
| task | time (ET) | speed / release | Josh | robot |
|---|---|---|---|---|
| 1 | 09:39 | 120/801 | miss | out |
| 2 | 09:41 | 117/722 | miss | out |
| 3 | 09:42 | 140/517 | hit | in |
| 4 | 09:43 | 101/712 | miss | out |
| 5 | 09:45 | 138/828 | miss | out |
| 6 | 09:46 | 139/846 | miss | out |
| 7 | 09:47 | 105/596 | miss (Josh corrected: his "hit" was for a later throw) | out |
| 8 | 09:48 | 128/722 | miss (see note) | out (cube seen on the floor at (-0.40, 0.97)) |
| 9 | 09:50 | 120/646 | hit (see note) | in (cube seen at (0.06, 1.29); run counter went 1/8 -> 2/9) |
| 10 | 09:51 | 126/846 | miss (Josh made no call; from the run counter) | out |

Note (09:52 ET): Josh typed one "hit" for throws 7-9 and placed it at 8 from memory. The robot's only hit in 7-9 is task 9. The toss reply's own in_bin said False for task 9; the run's count comes from the observe after the throw, which said True. Robot column for tasks 1-7 was taken from the toss reply and matches the run counter.
Update (09:53 ET): Josh: "or 9 whatever idk - see if it matches with robot calls". One hit by Josh in 7-9, one hit by the robot (task 9): counts agree; Josh did not independently confirm which throw.

Evaluation 0 total: 2/10 by the run's own counter (tasks 3 and 9).

## Practice cycle 0 (robot's calls only; Josh made no calls)
Reset order from pomdp_decisions.jsonl.zst: SAME, FAR, FAR, FAR, FAR, SAME (6 human resets). Box spot per throw is not recorded here.
| throw | time (ET) | speed / release | robot |
|---|---|---|---|
| 1 | 09:53 | 107/607 | out |
| 2 | 09:54 | 140/758 | out |
| 3 | 09:56 | 139/726 | out |
| 4 | 09:58 | 139/714 | out |
| 5 | 09:59 | 133/626 | in |
| 6 | 10:01 | 116/677 | in |
| 7 | 10:03 | 128/601 | in |

Practice total: 3/7 in (matches the sampler's labels in snapshots/after-update-0001.pickle).

## Evaluation 1 (after practice cycle 0)
Robot column: toss reply, cross-checked against the run's "x/y in the bin so far" counter.
| task | time (ET) | speed / release | Josh | robot |
|---|---|---|---|---|
| 1 | 10:06 | 130/598 | hit | in (run counter 1/1) |
| 2 | 10:07 | 131/581 | miss | out |
| 3 | 10:08 | never thrown | (no throw) | NEVER THROWN: the observe at 10:08:18 ET, right after Josh's y, returned no cube (cube: null), so no pick was planned; the run scored it a miss (counter 1/2 -> 1/3). Exclude from the throw count. |
| 4 | 10:09 | 139/564 | hit | in |
| 5 | 10:10 | 139/537 | hit | in |
| 6 | 10:11 | 129/599 | hit | in |
| 7 | 10:13 | 137/554 | hit (Josh: "robot is correct") | in |
| 8 | 10:14 | 133/589 | hit | in |
| 9 | 10:15 | 138/561 | hit | in |
| 10 | 10:17 | 140/582 | hit | in |
Evaluation 1 total: 8/9 thrown (task 3 never thrown, excluded; the run counter counts it as a miss).

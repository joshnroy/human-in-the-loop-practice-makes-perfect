# Ground truth: paper seed 3 (MLP, Curtis, fixed 1.40 m, box on the rug), 2026-11-10

Launched 10:21 ET from `launch/real-20261110a` @ b0dd2d27d: the code of seeds 0-2 (0fd3d8b4b) plus one
harness commit (a scene look that sees no cube asks for it in view and looks again, instead of a task
with no throw). Planner unchanged. Robot code ce25bae on the October environment.

Josh's calls are logged beside the robot's; where Josh made no call the robot's is used and marked.

## Evaluation 0 (before practice), started 10:25 ET

| task | time | speed/release | Josh | robot |
|---|---|---|---|---|
| 1 | 10:25 | 110 / 770 | (no call) | out |
| 2 | 10:27 | 127 / 779 | (no call) | out |
| 3 | 10:28 | 110 / 731 | hit, "barely though, bounced off the rim" (assigned to task 3 by timing; confirm) | in |
| 4 | 10:30 | 137 / 508 | hit, "narrowly", "rim again" | in |
| 5 | 10:31 | 102 / 684 | robot's call accepted ("that's fine") | out |
| 6 | 10:32 | 129 / 770 | robot's call accepted ("that's fine") | out |
| 7 | 10:33 | 111 / 663 | robot's call accepted ("that's fine") | in |
| 8 | 10:34 | 129 / 743 | robot's call accepted ("that's fine") | out |
| 9 | 10:36 | 123 / 564 | robot's call accepted ("that's fine") | in |
| 10 | 10:37 | 129 / 843 | (no call) | out |

Evaluation 0 total: 4/10 (all 10 thrown; robot's calls, Josh confirmed tasks 3, 4 and accepted 5-9).

## Practice cycle 0, started 10:38 ET

| throw | time | speed/release | box | Josh | robot |
|---|---|---|---|---|---|
| 1 | 10:38 | 130 / 658 | far | hit ("we got one hit so far") | in |
| - | 10:39 | human reset SAME, 16 s | | | |
| 2 | 10:40 | 106 / 640 | close | (no call) | out |
| 3 | 10:41 | 113 / 536 | close | (no call) | out |
| 4 | 10:42 | 125 / 801 | close | (no call) | out |
| 5 | 10:43 | 117 / 840 | close | (no call) | out |
| reset | 10:43 | FAR, 39 s | | | |
| 6 | 10:44 | 127 / 546 | far | (no call) | in |
| reset | 10:45 | SAME, 23 s | | | |
| 7 | 10:46 | 140 / 503 | close | (no call) | in |
| 8 | 10:47 | 114 / 616 | close | (no call) | out |

Practice cycle 0 total (robot's calls): 3/8 in (far 2/2, close 1/6), 3 human resets (SAME 16 s, FAR 39 s, SAME 23 s). Ended 10:48 ET; evaluation 1 prompt up at 10:48.

## Evaluation 1 (after practice cycle 0), started 10:50 ET

| task | time | speed/release | Josh | robot |
| --- | --- | --- | --- | --- |
| 1 | 10:51 | 139 / 632 | hit | in |
| 2 | 10:52 | 139 / 612 | (no call) | in |
| 3 | 10:53 | 140 / 505 | (no call) | in |
| 4 | 10:54 | 137 / 699 | (no call) | out |
| 5 | 10:55 | 132 / 595 | (no call) | out |
| 6 | 10:57 | 134 / 554 | (no call) | in |
| 7 | 10:58 | 134 / 591 | (no call) | in |
| 8 | 11:00 | 133 / 654 | (no call) | in |
| 9 | 11:01 | 137 / 523 | (no call) | out |
| 10 | 11:02 | 132 / 531 | (no call) | in |

Evaluation 1 total (robot's calls): 8/10, all 10 thrown. Ended 11:03 ET; the run then asked "restore practice: SAME" (start of practice cycle 1). The paper seed ends here.

Note (11:15 ET): the table's "robot" column is the toss reply. Task 5 (132 / 595) has toss reply `in_bin: false` ("cube not found after the toss") but the run's own counter scored it solved, which is why the table shows 7 "in" and the run total is 8/10. No call from Josh on task 5; which is right is open. Same disagreement on seed 2 evaluation 0 task 9 (reported by the eval-candidates video agent, not rechecked here).

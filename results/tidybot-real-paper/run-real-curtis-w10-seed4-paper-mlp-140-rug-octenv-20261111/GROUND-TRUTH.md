# Ground truth: paper seed 4 (MLP, Curtis, fixed 1.40 m, box on the rug), 2026-11-11

Launched 17:20 ET with `launch-paper-mlp-seed4.sh real` (BUDGET=100000) from `launch/real-20261110a` @ b0dd2d27d,
the same code as seed 3. `config_snapshot.json` differs from seed 3's only in the seed, the output folder and
the costs file. Robot code ce25bae on the October environment; servers restarted 17:10 ET (robot log
`logs/2026-11-11/17-10-01`). Checked before the first throw: the robot's `uv.lock` is byte-identical to the
recorded October lock file and its 134 installed packages match the October reference checkout.
No fake dry run was done for this seed (code unchanged since seed 3).

Josh made no in/out calls in chat during this run, so every result below is the robot's. The "robot" column
is the toss reply; in both evaluations it agrees with the run's own counter (`step_evaluations.json`) on every task.

## Evaluation 0 (before practice), prompt 17:21 ET

| task | time | speed/release | Josh | robot |
|---|---|---|---|---|
| 1 | 17:22 | 119 / 759 | (no call) | out |
| 2 | 17:23 | 110 / 823 | (no call) | out |
| 3 | 17:25 | 125 / 694 | (no call) | out |
| 4 | 17:26 | 130 / 721 | (no call) | out |
| 5 | 17:27 | 107 / 574 | (no call) | out |
| 6 | 17:29 | 124 / 755 | (no call) | out |
| 7 | 17:30 | 100 / 726 | (no call) | out |
| 8 | 17:31 | 129 / 552 | (no call) | in |
| 9 | 17:32 | 129 / 552 | (no call) | out |
| 10 | 17:34 | 123 / 764 | (no call) | out |

Evaluation 0 total (robot's calls): 1/10, all 10 thrown.

## Practice cycle 0, started 17:34 ET

| throw | time | speed/release | box | Josh | robot |
|---|---|---|---|---|---|
| 1 | 17:35 | 132 / 602 | far | (no call) | in |
| reset | 17:36 | SAME, 19 s | | | |
| 2 | 17:37 | 139 / 723 | close | (no call) | out |
| 3 | 17:38 | 125 / 574 | close | (no call) | out |
| 4 | 17:39 | 128 / 708 | close | (no call) | out |
| 5 | 17:40 | 118 / 726 | close | (no call) | out |
| 6 | 17:41 | 105 / 788 | close | (no call) | out |
| 7 | 17:42 | 124 / 578 | close | (no call) | in |
| 8 | 17:43 | 104 / 634 | close | (no call) | out |
| 9 | 17:44 | 131 / 549 | close | (no call) | in |

Practice cycle 0 total (robot's calls): 3/9 in (far 1/1, close 2/8), 1 human reset (SAME 19 s). Matches the
sampler's labels in `snapshots/after-update-0001.pickle`. Evaluation 1 prompt up at 17:44 ET.

## Evaluation 1 (after practice cycle 0), started 17:45 ET

| task | time | speed/release | Josh | robot |
|---|---|---|---|---|
| 1 | 17:46 | 129 / 641 | (no call) | in |
| 2 | 17:47 | 129 / 625 | (no call) | in |
| 3 | 17:48 | 122 / 611 | (no call) | in |
| 4 | 17:50 | 135 / 648 | (no call) | out |
| 5 | 17:51 | 115 / 605 | (no call) | in |
| 6 | 17:52 | 139 / 545 | (no call) | in |
| 7 | 17:54 | 132 / 623 | (no call) | in |
| 8 | 17:55 | 132 / 653 | (no call) | in |
| 9 | 17:56 | 118 / 617 | (no call) | in |
| 10 | 17:57 | 125 / 619 | (no call) | in |

Evaluation 1 total (robot's calls): 9/10, all 10 thrown. Ended 17:58 ET; the run then asked
"restore practice: SAME" (start of practice cycle 1) and Josh declined it. The paper seed ends here.
`status.json` says `phase: failed` for that reason only; both evaluations are complete.

Note: evaluation 0's seven-miss start was checked against seeds 0-3 while it ran. The nearest earlier throws
to each miss were themselves mostly misses, and the box was seen at (0.220, 1.337) against an earlier mean of
(0.221, 1.339), so nothing points at a setup error. No test was run on this; it is a description, not an inference.

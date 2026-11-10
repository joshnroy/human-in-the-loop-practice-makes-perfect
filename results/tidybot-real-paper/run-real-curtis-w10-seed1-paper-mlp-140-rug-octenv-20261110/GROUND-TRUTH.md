# Paper seed 1 (MLP), real TidyBot — ground truth

Started 2026-11-10 00:31 ET with `BUDGET=100000 ./launch-paper-mlp-seed1.sh real`. Same code and setup as seed 0
(launch/real-20261109e @ 0fd3d8b4b, robot ce25bae + October env, fixed 1.40 m stand, cardboard box on the rug
at (0.222, 1.346) yaw 0, 10 evaluation tasks, evaluation before practice and after every cycle).

Josh's calls are the ground truth; the robot's own call is noted where it differs.

## Evaluation 0 (before practice, untrained sampler, random picks): 2/10 in

Finished 2026-11-10 ~00:47 ET. All 10 tasks were thrown (none refused). Josh called tasks 7 and 9 as hits and gave
no call for the others; the robot's own calls are 2/10 in (tasks 7 and 9), agreeing with Josh on both, and the run's
`evaluations/0000/results.json` says 2/10 solved. Landing spot is the robot's estimate in the map frame; the box
centre is (0.222, 1.346). The throws match the fake dry run's evaluation 0 list 10/10.

| task | speed (deg/s) | release (ms) | Josh's call | robot's call | robot's landing spot (m) |
| --- | --- | --- | --- | --- | --- |
| 1 | 137 | 817 | (no call given) | out | (-0.08, 0.93) |
| 2 | 130 | 701 | (no call given) | out | (0.35, 0.75) |
| 3 | 130 | 762 | (no call given) | out | (-0.20, 0.71) |
| 4 | 110 | 653 | (no call given) | out | (0.00, 0.87) |
| 5 | 130 | 744 | (no call given) | out | (-0.33, 1.23) |
| 6 | 116 | 580 | (no call given) | out | (0.07, 1.06) |
| 7 | 136 | 506 | **hit (narrowly)** | in | (0.09, 1.43) |
| 8 | 112 | 751 | (no call given) | out | (-0.13, 0.42) |
| 9 | 128 | 688 | **hit** | in | (0.18, 1.39) |
| 10 | 114 | 767 | (no call given) | out | (-0.32, 1.15) |

Note: task 9 (128/688) went in well outside where seed 0's hits fell (132-139 deg/s, 504-541 ms).
NUC toss log: `~/prpl-tidybot-ce25bae-octenv/logs/2026-11-09/23-10-02/skill_calls.jsonl` (same skill server session as seed 0).

## Practice cycle 0 (in progress; robot's calls unless noted)

| # | time ET | box | speed/release | result |
|---|---|---|---|---|
| 1 | 00:46 | far | 109/546 | out |
| 2 | 00:47 | close | 113/626 | out |
| 3 | 00:48 | close | 127/729 | out |
| 4 | 00:49 | close | 108/622 | out |
| 5 | 00:50 | close | 119/763 | out |
| 6 | 00:51 | close | 120/612 | **in** (robot's call; Josh said "something scored" 00:52, no per-throw call) |

Throws 1-6 are the same parameters as the all-miss fake replay (dryrun-paper-seed1-20261110-allmiss).
Throw 6 is the first close-box hit from 1.40 m in either seed (seed 0: 0/6 close). From here the real
run no longer follows the all-miss replay.

Practice cycle 0, continued:

| # | time ET | box | speed/release | result |
|---|---|---|---|---|
| 7 | 00:53 | close | 134/644 | **in** (robot's call; Josh: "7 also scored", 00:54) |

Correction to the note above: throw 7 is still the all-miss replay's 7th throw (134/644), so a hit inside
a cycle does not change that cycle's remaining draws. Whether the match holds after the cycle-0 model
update is not known.
| 8 | 00:56 | close | 111/605 | out (robot's call) |

Graph: hits-so-far.png in this folder (regenerate with the scratchpad seed1_plot.py).
| 9 | 00:57 | close | 119/793 | out (robot's call) |

**Practice cycle 0 total: 2/9 in** (far 0/1, close 2/8). All 9 throws equal the all-miss replay's list.
Model update 1 happened 00:58 (snapshots/after-update-0001.pickle; training labels 0,0,0,0,0,1,1,0,0).
The model's inputs are (1.0, speed, release) only: it does not see which box it threw at.
Map: model-rating.png. At 00:58:30 the run is waiting at "evaluation after cycle 0: run it? [y/s]".
NUC session logs copied 00:59 to nuc_logs/23-10-02/ (931 MB, includes seed 0's swings).

Run stopped by Josh ~01:04 ET Nov 10 at the evaluation-1 prompt (evaluation 1 not run). Last webcam file webcam_2026-11-10_01-01-33.mp4.

## Resumed 2026-11-10 ~09:05 ET
Robot computer was powered off overnight; arm, base and skill servers restarted 09:01 with the same commands from ~/prpl-tidybot-ce25bae-octenv (new NUC log folder logs/2026-11-10/09-01-27). Launched `real --resume`. Plan (Josh 09:04): evaluation 1, practice cycle 1, evaluation 2, then seed 1 is complete. Seed 0 will NOT be resumed (Josh).

## Evaluation 1 (after cycle 0), 2026-11-10 from 09:13 ET, far box on the rug

| task | time | speed/release | Josh | robot |
| --- | --- | --- | --- | --- |
| 1 | 09:13 | 134/678 | miss | out |
| 2 | 09:15 | 139/619 | hit | in |
| 3 | 09:16 | 123/609 | hit | in |
| 4 | 09:17 | 124/562 | hit | in |
| 5 | 09:19 | 134/583 | hit | in |
| 6 | 09:20 | 139/630 | hit | in |
| 7 | 09:21 | 135/635 | hit | in |
| 8 | 09:23 | 138/552 | hit | in |
| 9 | 09:24 | 137/583 | hit | in |
| 10 | 09:25 | 139/519 | hit | in |

Evaluation 1 total: 9/10 in (Josh and the robot agree on all 10; all 10 thrown). Evaluation 0 was 2/10.

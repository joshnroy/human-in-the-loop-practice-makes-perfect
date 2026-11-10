# Jug gripper weld: paired pour-performance check

The gripper weld changes pour performance in the first matched pilot. With seed 0,
Pour2 and a 1 degree/s ramp, the table-weld baseline delivered exactly two dumplings;
the table-plus-gripper-weld candidate delivered all three. Both pickups and placements
succeeded, and both outcomes settled. This is a measured overshoot, not a setup failure.
The candidate remains an uncalibrated draft; the learning sweep uses the table-weld baseline.

**Update (16:42 EST):** the first 9 matched Della cases have identical success/failure
outcomes (8/9 in each variant), including success in the local pilot's failed setting.
See the Della checkpoint and cross-runtime qualification below before interpreting the local pilot.

## Frozen versions and protocol

- Baseline: `fb217820056873797918e23651c7e06633c3eeeb` (parked jug weld).
- Candidate: `c669e6e229c6a78c27cb5b2ed89c7f1dd5f7c8f1` (parked plus held jug weld).
- Both: kindergarden `c85dbbc50ddac6eb59e8b5e00d7cc942cdd691fe` and
  kinder-baselines `032b42634a015c7f710667b3a791836fa839f6ff`.
- [Probe harness](../../scripts/probe_kitchen_pour.py): public kitchen actions in fresh,
  isolated processes, identical seeds and requested counts. Imports resolve within each
  frozen checkout. No planner, cost, reset probability or pour-controller tuning changed.
- The controller keeps its existing 80-degree fast approach, 95-degree ceiling,
  40 mm sideways wiggle, 45 degree/s return and 30-second settlement window.
- Pour success uses the native exact split, retained count, settlement and controller-error
  checks. Setup success, pour success and post-pour placement are recorded separately.
- Grasp drift below is the change from immediately after pickup to immediately after Pour,
  including carry and preparation. It is an endpoint measurement, not peak transient drift.

## Local paired pilot (completed)

One seed and one setting cannot establish a success rate or select a calibrated recipe.
It does demonstrate that welding changes this setting's result.

| Measure | Table weld | Table + gripper weld |
| --- | ---: | ---: |
| Setup / pour / placement successful | yes / yes / yes | yes / no / yes |
| Delivered to pan / retained in jug | 2 / 1 | 3 / 0 |
| Spilled outside pan and jug | 0 | 0 |
| Settled | yes | yes |
| Commanded cutoff tilt | 91.90 degrees | 91.00 degrees |
| Endpoint grasp translation change | 5.690 mm | 0.200 mm |
| Endpoint grasp rotation change | 2.844 degrees | 0.969 degrees |
| Pour controller ticks | 930 | 884 |

Both variants triggered cutoff after detecting two dumplings outside the jug. The candidate
ended with a third in the pan; cutoff cannot recover a dumpling already departing.
Improved grasp retention therefore does not imply unchanged food motion or unchanged success.

![One matched pilot, with requested count and endpoint grasp drift](../figures/pouring-jug-gripper-pilot-20261110.png)

Raw evidence: [baseline](2026-11-10-pouring-jug-gripper-performance/local-baseline-pilot.json)
and [candidate](2026-11-10-pouring-jug-gripper-performance/local-candidate-pilot.json).
These local pilots are separate from the Della matrix and will not be counted twice.

## Della paired matrix (running)

The [fixed cases](2026-11-10-pouring-jug-gripper-performance/cases.json) comprise:

- 27 fresh starts per variant: seeds 0, 1 and 2; requested counts 1, 2 and 3;
  ramp rates 0.25, 1 and 5 degrees/s.
- Three additional robot-reset cases per variant: seed 0, each requested count, rate 1.
  Each first pours three, places the jug and returns each displaced dumpling with native
  robot pick/drop actions. A setup failure is reported separately from a pour failure.

Baseline array `17260868` and candidate array `17261239` each allow ten concurrent jobs,
one CPU and 6 GB per case. The 3-hour Slurm reservation selects the short QOS; each process
has a strict 55-minute timeout. Two entirely unstarted submissions were replaced because
Della assigned them to the two-job test QOS. Other experiment jobs were not changed.
See the [launch manifest](2026-11-10-pouring-jug-gripper-performance/manifest.json).

Only completed matched cases will enter comparisons. The analyzer checks every record's
main SHA, dependency pins, harness hash and case values; no missing outcome is imputed.
Full results and per-seed ramp plots will be appended here when available. No retuning or
promotion of the held-jug candidate has been performed.

## Della checkpoint: 2026-11-10 16:42 EST

**The local Pour2 overshoot did not reproduce in the same Della case.** At this checkpoint,
14/30 baseline cases and 9/30 candidate cases have completed, yielding
9/30 matched cases. Exact pour success is **8/9 baseline and 8/9 candidate**; every
matched case has the same success/failure outcome. All 9 setups and placements succeeded
per variant, all outcomes settled, and neither variant spilled outside the jug and pan.
All 9 fresh initial physical snapshots match exactly between variants.

For seed 0, Pour2 succeeds at 0.25 and 1 degree/s and overshoots to three at 5 degrees/s
in both Della variants. The local candidate's 1 degree/s overshoot therefore must not be
presented as a reproduced Della regression. The two local pilot records above remain
unchanged and are not pooled with the Della observations.

![Completed matched Della cases, including means with one seed](../figures/pouring-jug-gripper-della-checkpoint-1642-20261110.png)

The plot includes only completed matched cases. Marker shape identifies seed; horizontal
offsets keep the two variants visible. Lines connect measured settings rather than fitting
a continuous success curve. Counts give the actual available denominator, including one
seed. Missing settings are omitted. Robot-reset cases have not completed at this checkpoint.

[Checkpoint summary](2026-11-10-pouring-jug-gripper-performance/della-checkpoint-1642.json) includes the exact matched case IDs.
[Raw completed records](2026-11-10-pouring-jug-gripper-performance/della-records.jsonl)
preserve every field, with one case per JSON line; only the matched subset enters comparisons.
The candidate remains a draft, and the remaining cases continue to run.
This small partial matrix does not establish equivalence or a calibrated success rate.

### Cross-runtime qualification

The local and Della candidate Pour2/rate1 cases have identical source pins, harness hash,
initial physical state and pickup action summary (284 controller ticks). Yet the jug
position differs by about 18 micrometres along its largest differing axis after pickup,
and the food state has diverged by the start of the pour ramp. The result is cross-runtime
trajectory sensitivity; its cause has not been isolated.

Both runtimes report MuJoCo 3.3.7, NumPy 1.26.4 and SciPy 1.14.0. Python patch/compiler,
platform and numerical thread settings differ: Della caps the numerical libraries at one
thread; the local pilot inherited unset thread limits. KINDER distribution metadata also
differs, although both imports use the explicitly verified source checkout at the same pin.
These differences are possible contributors, not a demonstrated causal explanation.

Evidence: [local runtime](2026-11-10-pouring-jug-gripper-performance/local-runtime.json),
[Della runtime](2026-11-10-pouring-jug-gripper-performance/della-runtime.json), and
[cross-runtime comparison](2026-11-10-pouring-jug-gripper-performance/cross-runtime-audit.json).
The same-host Della matrix is the primary comparison for the proposed Della learning runs.

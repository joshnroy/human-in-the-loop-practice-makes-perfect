# Three-potsticker pouring after the entry-trajectory repair

## TL;DR

The repaired entry trajectory has **9/9 selected development seed/count existence
witnesses**. Seed zero additionally has **3/3 full-scene demonstrations**, from
pickup through pitcher placement. These are selected settings, not a fixed-policy
success rate. Repeated robot recovery and the six learning pilots remain launch gates.

![Current parameter witnesses and tilt sensitivity](2026-10-30-potsticker-repaired-entry.png)

## Question / goal

Check the current three-potsticker environment after correcting unintended contacts
on food transport and the next pitcher pickup. Preserve the
[earlier results](2026-10-30-potsticker-readiness.md); their selected recipes precede
the changed entry trajectory and must not be silently transferred to this revision.

## Methods

The joint-feedback pour uses tilt, horizontal sideways amplitude and feeding duration.
It has no food-count cutoff and no simulated lookahead. Seeds 1/2 use checkpoints
after physical pickup and transport; seed zero runs from the initial scene through
placement. No witness is injected into a learned sampler. The
[manifest](2026-10-30-potsticker-repaired-entry.json) records result paths, hashes and
source fingerprints. Recovery-only changes after preparation do not change the
fresh-scene entry branch: its initial hand-to-pitcher XY distance is 0.450 m, above
the 0.25 m threshold for the new post-recovery aisle reorientation.

## Results

| Development seed | Requested | Tilt (degrees) | Amplitude (mm) | Feeding (s) | Delivered |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 88.75 | 8 | 12 | 1 |
| 0 | 2 | 89.125 | 8 | 12 | 2 |
| 0 | 3 | 90 | 15 | 12 | 3 |
| 1 | 1 | 88.75 | 8 | 12 | 1 |
| 1 | 2 | 89.375 | 8 | 12 | 2 |
| 1 | 3 | 92 | 15 | 12 | 3 |
| 2 | 1 | 89.5 | 8 | 12 | 1 |
| 2 | 2 | 89.75 | 8 | 12 | 2 |
| 2 | 3 | 92 | 15 | 12 | 3 |

On the same seed-zero Pour2 prepared state, changing tilt from 89.125 to 88.75
degrees delivers one; changing feeding duration from 12 to zero seconds delivers
zero. Both are incorrect settings. Zero amplitude and 25 mm amplitude each still
deliver two at the successful tilt/duration. Thus this state demonstrates sensitivity
to tilt and duration, but does not establish that amplitude is necessary.

The v35 follow-up also tests zero and 25 mm amplitude at the selected Pour2 tilt
on seeds 1 and 2: **4/4 deliver two**. Across all three development states, those
extreme-amplitude probes succeed **6/6**. This is evidence of tolerance at these
selected tilts, not evidence that amplitude never affects the outcome elsewhere.

The v28 human-reset checks pass **6/6 ready-state returns**: two consecutive attempts
for each requested count, zero audited unintended contacts, and exact preservation
of non-payload positions, velocities, simulation time and controller ticks during
reset. Only **2/6 pours** are exact with those fixed recipes; these are different
metrics. Delivered pairs are 2/1 for target 1, 3/1 for target 2 and 3/1 for target 3.

The v30 robot run returned all three pieces, then failed at the next pitcher pickup:
the right gripper coupler contacted the pitcher during arm reorientation. The new
entry branch backs into the aisle before reorientation. The v34 uninterrupted
two-cycle check is testing that repair. The v33 individual retry diagnostic recovered
the remaining pan piece with zero audited contacts, but its starting pan was already
displaced; this is not a full ready-state recovery witness.

Current targeted integration/shared-method/planning tests: **200/200 passed**.
Typechecking passes for 269 source files. Full CI passed on the preceding pushed
revision; the final recovery changes require their own CI run.

## Recommendation

Freeze and launch a new six-run pilot batch only after inspecting the repeated-cycle
result. Preserve the aborted `cd40854a0` batch separately. Report first-pour exact
counts separately from full-task success after robot recovery, and inspect STOP,
Wait, interruptions, labels, resets and costs before launching the overnight study.

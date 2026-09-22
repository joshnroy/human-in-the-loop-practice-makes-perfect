# Calibrated long-range Tossing3D proposals

**Removed as a selectable proposal.** The `--toss-proposal` flag and the
`LongRangeTossProposal` sampler this document describes no longer exist: the
calibrated proposal supplied almost exclusively scoring throws, so an untrained
learner evaluated 10/10 before any practice and learning experiments built on it
were vacuous. The barrier layout's only toss candidate generator is now
`WideLongRangeTossProposal`, which keeps this calibration's standoff, yaw, and
release-time ridge (as `release_ridge`) but draws speed and release independently
over a band straddling that ridge. The same-side layout retains its existing
sampler. The calibration record and its physical-reliability evidence below are
retained unchanged; they describe the ridge, which is still the wide proposal's
center of success.

The farther receiver distribution requires a compatible launch position and
throw profile. Merely extending independent distance, speed, and release bounds
leaves most random throws short. The calibrated proposal samples joint speeds
from 390 to 420 degrees/second and release times around a line joining 460 ms at
390 degrees/second to 450 ms at 420 degrees/second, with ±2 ms timing variation.
Yaw offset is zero. The existing simulation effort ceiling remains unchanged.

The paired controller correction carries the observed cube grasp into hypothetical
base and arm poses during collision checking. It checks the straight joint path
that the controller actually executes, waits for observed windup convergence,
and times the swing from the observed arm configuration. Attached-cube checks
conservatively cover the whole swing, including the portion after release.

Development calibration used scene seeds 2026092200–2026092212. The selected
endpoint settings then succeeded in all 20 throws across the last ten of those
scenes. After freezing the continuous proposal, one independently drawn proposal
on each of 20 new scenes (2026092300–2026092319) succeeded 20/20, with no pickup
failures, controller errors, or timeouts. This validates the sampled proposal on
that finite pool; it is not a guarantee for every parameter combination or scene.

Calibration is offline controller development. Its trials and labels are not
added to the learner or counted as autonomous practice. Retraining must start
with empty learner data, preserve the initial evaluation, and retain autonomous
STOP. Any improvement present before practice is attributable to the controller
and proposal rather than classifier learning.

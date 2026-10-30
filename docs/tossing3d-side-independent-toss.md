# Side-independent Toss practice target

## Decision — 2026-10-29

Supersedes the earlier decision to retain an explicit side argument in Toss.
`MoveToTossLocationAndToss(robot, bin, cube, barrier)` has no side parameter.
The bin's current placement determines the successful landing side through
conditional effects. Side remains an observed scene property and an explicit
human-reset destination, not a distinct Toss practice target.

The motivation is to remove a modeling confound: a grounded-practice algorithm
should not have to move the same bin to another side merely to practice the
otherwise identical Toss controller. This change is shared by EES, expectimax,
and DP. It is not an EES algorithm correction or evidence of better performance.

## Preserved semantics

- Preconditions remain Holding(robot, cube) and BinOnGround(bin).
- Successful Toss retains its existing non-side effects and replaces CubeAtSide
  with the bin's side from the incoming state. The other side must not survive.
- Conditional effect-local constants are not extra action parameters or separate
  groundings. They are emitted as typed PDDL domain constants.
- Pick still requires robot/cube side agreement and feasible pick geometry.
- Reset actions still select cube and bin sides explicitly; expectimax still
  searches their downstream state-dependent consequences.
- Low-level controllers, sampling, costs, failure observations, and task goals
  are unchanged. Failure-effect observations remain conditioned on the incoming
  symbolic state, so opposite placements are not blindly interchanged.
- Pick, pouring, and reset signatures are intentionally unchanged. This decision
  removes the Toss split, not every side argument from the domain.

## Learning and comparison implications

The same robot/bin/cube/barrier tuple now has one Toss identity across placements.
Grounding-keyed competence, learning-rate predictions, exploration counts, and
targeted-practice bookkeeping therefore follow that single identity together.
This does not introduce cross-grounding target substitution or choose the nearest
setup as a new EES heuristic. EES's expanded-practice and perfect-skill settings
remain independent experimental axes. Parent pooling can still affect other
groundings or skills, but no longer distinguishes these two Toss placements.

Do not resume old grounding-keyed checkpoints or combine pre-change and
post-change runs into a single mean. Historical reports retain their original
encoding. Fresh runs of all methods need the same new source/environment.

## Relaunch gates

1. Validate one Toss grounding, both conditional landing sides, stale-side removal,
   same-side feasible picking, opposite-side pick rejection, and reset sequencing.
2. Validate actual Fast Downward planning and DP transitions, in addition to the
   expectimax transition model and EES practice bookkeeping.
3. Recollect DP calibration with the updated collector and source hashes; do not
   re-label old prior files or bypass source-compatibility checks. Calibration
   retains its declared deployment scene distribution, not a new side-balanced
   curriculum. OpenGripper coverage remains required.
4. Freeze a clean tested launch revision, verify Della preflight on that revision,
   then launch fresh seed-0 trials. No old seed-0 checkpoint continuation.

The previous Della retry, job 16446824, was cancelled before allocation (zero
runtime). Its evidence is retained; it produced no new validation or training data.

## Validation record

- New side-independent Toss regressions: 14 pass, including real Fast Downward
  planning, DP success transitions, expectimax stale-side replacement, both reset
  continuations, EES setup without bin relocation, and shared evidence/counts.
- Core and planning: 270 pass, including 32 new constant-binding regressions.
- Practice-makes-perfect methods: 684 pass, including Gurobi and reference checks.
- Belief methods: 488 pass. The unrelated 65 expensive shelf-symmetry cases were
  excluded from this completed run, not counted as passing.
- Targeted environment tests: 207 pass; oracle unit tests: 17 pass.
- DP calibration/fidelity scripts: 96 pass; other selected launch scripts: 71 pass.
- Changed-file lint/format checks pass. Broader type checking reports the same
  30 errors on the original collector and current collector, including installed
  simulator API mismatches; these are not waived as a clean type-check result.
- The live simulator walk passed the Toss dispatch point after the oracle's
  grounding was updated, but failed its later OpenGripper HandEmpty assertion.
  The frozen pre-change source reproduces that same recovery failure on this local
  runtime. This is not a passing full physics gate. Validate with the pinned
  experiment runtime before any full relaunch; do not weaken the recovery test.

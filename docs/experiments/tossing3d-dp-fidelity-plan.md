# Tossing3D DP fidelity audit and implementation

Status: DP preparation, validation and eventual submission are authorized independently
of EES, subject to readiness. The preparation helper submits nothing.
The active EES/environment work continues separately. Preserve historical results.

## Evidence boundary

On 2026-10-27, the user additionally approved a common OpenGripper controller that
moves the hand forward clear of the fixed robot base before opening, then applies
the unchanged settling check. This is shared by every method, not a DP-specific
adaptation. Its motion ticks are charged normally. Freeze and validate that behavior
before fresh closed-empty and holding-release calibration; earlier stationary-open
calibration cannot be reused.

Primary source: [Vats et al., v1](https://arxiv.org/html/2608.13415v1).
Sections 3–5 define per-skill competence, improvement and allocation over a supplied
library, followed by a curriculum. They do not prescribe our parent-name pooling or
alternative-grounding execution. Appendix A.3 describes their CMA-ES learner.
No official DP implementation has been verified in this audit.
Appendix A.5.1 explicitly enumerates 22 experimental skills: OpenDrawer × 3,
CloseDrawer × 3, Pick × 4, and Place × 4 items × 3 drawers. This experimental
library is object/destination-grounded, not merely four controller labels; it does
not establish a universal lifted-to-ground identity rule for every domain.

## Contract for this adaptation

| Topic | Current contract / action |
| --- | --- |
| Skill identity | Retain side-dependent grounded actions; parent pooling is an explicit transfer assumption, not established equivalence of the two Toss contexts. |
| Statistics | Move current competence, improvement predictions and allocation counts together in the parent-sharing arm. Do not add a mixed-rate arm to a new baseline comparison. |
| Eligibility | Untried deployment-MDP skills remain allocatable. An observed setting must not be described as excluding them. No EES-style discovery axis without a separately defined restriction. |
| Target execution | Default to the selected grounding. Preserve historical substitution only behind explicit `--dp-allow-grounding-substitution`; it is not implicitly enabled by parent/all. |
| Cost estimates | Exact-target execution uses that target's attempt price, not a cheaper sibling's price. Actual setup/reset costs remain charged. |
| Protocols | Episodes/once and cost/period are operational choices; do not label once as an unequivocal original-paper schedule. |
| Solver | Retain joint Gurobi allocation, independent verification, and the existing numerical tolerances. No silent fallback. |
| Environment | Same shared controller library, OpenGripper semantics, evaluation isolation and sampler as the new EES comparison. |

The user-approved inner learner, success objective and scored human-reset extension
remain. Do not add reset lookahead, clamp away negative improvement estimates to
force practice, filter parameterless skills, or adopt EES goal-pursuit/skip-perfect
axes merely because they exist in the inherited class.

## Proposed DP additions

No new run uses grounding substitution, including shared-statistics runs. The legacy
flag is for historical reproduction only and remains false in the new comparison.

| Arm | Statistics | Budget | Allocation | Full seeds |
| --- | --- | --- | --- | --- |
| DP-E-G | Grounded | 2,000 completed target rollouts | Once | 0, 1, 2 |
| DP-E-S | Parent-shared competence, rate and counts | 2,000 completed target rollouts | Once | 0, 1, 2 |
| DP-C-G | Grounded | 2,000 robot/human cost | Per period | 0, 1, 2 |
| DP-C-S | Parent-shared competence, rate and counts | 2,000 robot/human cost | Per period | 0, 1, 2 |

Four configurations add twelve full jobs. Pilots use seeds 0/1 and a 6,400-counted-step
ceiling, adding eight jobs. Both sizes evaluate every 3,200 counted steps on ten
held-out tasks with 500 control ticks per evaluation task.
Keep the common 160,000-step ceiling, identical environment/evaluation settings and
human scoring; no goal-pursuit, skip-perfect, observed-only eligibility or mixed-rate
axis. Episode/cost rows are protocol bundles, not an isolated test of either budget
units or reallocation schedule. The episode cap returns to the user's earlier 2,000
target-rollout choice rather than silently inheriting the overnight 350 cap.
These are maxima: exhausted allocations, method STOP or the outer ceiling can end
a run earlier. Record actual endpoints and spend; do not claim equal training cost.

Together with the twenty EES/expectimax/Curtis configurations, this would total
seventy-two full runs and forty-eight pilots, excluding prior-calibration jobs.
Keep seed-major submission without seed-completion barriers. DP recalibration is
not a prerequisite for launching the EES/planner pilots once their own gates pass.
Likewise, DP can proceed independently when its own calibration and integration gates pass.

## Preparation and validation helper

`scripts/tossing3d_dp_fidelity_launch.py` emits deterministic JSON, using the canonical
Tossing3D constructor and EES-priced cost configuration. It neither runs environments
nor invokes an optimizer, scheduler or submission command. Order is all four arms
for seed 0, then seed 1, then seed 2 (full only); no dependencies or barriers, with
maximum concurrency permitted by the scheduler. Names identify grounded/shared
statistics and episodes-once2000/cost-period2000. Shared maps to parent statistics.

All arms explicitly use exact targets, all-grounding eligibility including untried
deployment-reachable skills, goal horizon 0, init/interval 1, target-only exploration,
human scoring, no skip-perfect stop, no reset gate, no always-ask and no pouring.
Common masses are bin 1 kg/cube 0.1 kg; epsilon is 0.5, training iterations 10,000,
competence window/recency 5, with inherited candidate and sampler defaults checked.
The interaction limit is **20 skill-level actions**, not 20 control ticks. DP writes
`dp_allocations.jsonl` and sampler draws; the EES-only decision logger is not enabled.

Preparation requires a fitted prior plus both raw collection directories. It checks
current calibration source coverage and collection-driver hashes, exact robot-skill
coverage, pinned clean simulators, independent reserved fit/validation seeds, matching
environment/protocol and successful held-out validation. It re-fits the saved raw
probes and requires exact agreement with the supplied report. Both closed-empty and
holding-release OpenGripper contexts must be present under the current fitter. The
report must explicitly record `equal-context` scalar aggregation; this is a balanced
benchmark assumption, **not measured deployment context frequency**. Old priors or
old-driver collections cannot be relabelled fresh by re-fitting. Synthetic test
fixtures are not production calibration evidence.

Run from the final execution checkout, with these absolute paths set to actual
fresh evidence and output locations (these commands do not collect or submit):

```bash
PY=/home/josh/miniconda3/envs/hitl-pmp/bin/python
export PYTHONPATH=src:.
export DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD=1
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
# Set PRIOR, CURVES, OPEN_PROBES, RESULTS and MANIFEST to absolute paths.
"$PY" -m scripts.tossing3d_dp_fidelity_launch check --size pilot \
  --prior-calibration "$PRIOR" --calibration-root "$CURVES" \
  --open-root "$OPEN_PROBES" --results-root "$RESULTS"
"$PY" -m scripts.tossing3d_dp_fidelity_launch prepare --size pilot \
  --prior-calibration "$PRIOR" --calibration-root "$CURVES" \
  --open-root "$OPEN_PROBES" --results-root "$RESULTS" --out "$MANIFEST"
"$PY" -m scripts.tossing3d_dp_fidelity_launch validate --manifest "$MANIFEST"
```

Use `--size full` and a distinct manifest output for the twelve full jobs. Output
creation is exclusive: existing manifests are not overwritten. All paths are local
absolute paths, including the canonical cost file. Regenerate on the launch checkout;
do not copy a local manifest to another host and assume paths are portable. Validation
rebuilds the matrix against current sources/evidence and rejects changed commands,
ordering, duplicates, missing runs, provenance or source hashes. A recorded HEAD is
not itself a clean/deployed attestation. Final dependencies/assets, solver license,
numerical certification and pilot measurements remain separate launch gates. No
walltime prediction is a guarantee.

Focused validation (no simulation or solver execution; synthetic calibration probes,
real fitter, parser and DP factory; only simulator-revision lookup and environment
runner are mocked):

```bash
FD_EXEC_PATH=/home/josh/Documents/repos/research/downward \
  "$PY" -m pytest tests/scripts/test_tossing3d_dp_fidelity_launch.py -q \
  --basetemp="$PWD/.cache/pytest-dp-fidelity-launch"
```

## Parallel implementation plan

1. **Target contract:** isolate alternative grounding fulfillment; regress exact
   target identity and matching attempt prices, with legacy opt-in coverage.
2. **Eligibility and sharing:** prove that untried reachable deployment skills can
   receive allocation and that parent pooling does not silently change execution.
3. **Numerical certification:** reproduce the recorded overnight saturation cases
   against independent enumeration; validate the pending inequality formulation
   for nonnegative affine forecasts, grouped skills, negative slopes, and both
   optimization stages. Preserve pre-existing repair work and distinguish exact
   replay from reduced reconstructions.
4. **Integration:** CLI factory tests, DP regressions, common OpenGripper/PDDL tests,
   and deployment-snapshot preservation tests. Do not change the active EES matrix.
5. **Prelaunch audit:** check prior coverage and provenance after common environment
   changes; old OpenGripper probes cover closed-empty recovery, not held-cube drops.
   Recalibrate where needed. Freeze the new DP budget, seeds, protocols and source
   only after this review. The proposed matrix above is preparation, not a claim
   of source readiness or a submitted experiment.

## Historical interpretation

The overnight shared/all arms coupled transfer and alternative-grounding execution.
They are not clean sharing-only evidence. Preserve their flags and source revision;
reproducing that execution policy under the new code requires explicit opt-in.
Two cost/period seeds failed numerical certification; exclude them from completed-run
means while retaining the failure ledger. Audit shared evaluation snapshots before
treating historical success differences as method evidence.

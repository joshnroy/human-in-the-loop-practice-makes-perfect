# Sweep learning integration: approved scope, 2026-09-28

Josh approved these defaults through the parent session before leaving for 2–4 hours. This records authorization, not completed implementation or results.

1. Human and robot resets target the same declared valid start regions and orientations. Evaluate only seeds valid before the first action; log all exclusions, use the same accepted seed list in every arm, and never resample a rejected seed.
2. Learn OpenDrawer, PickWiper, and Sweep parameter samplers. Recovery controllers remain fixed, exposed as individual planner-selectable skills.
3. Preserve the faithful EXP22c EES configuration with the reset gate disabled. Report cost insensitivity if observed; do not alter the algorithm to force the hypothesis.
4. Pilot two methods (EES and ModelB) times three native human-cost levels plus no-human-reset: 8 arms with one shared valid seed. Calibrate native configured costs against robot sequences and freeze calibration before examining results. Wall seconds are separate diagnostics.
5. Practice 50 cycles, at most20 skill actions per cycle, continuously with no automatic/environment resets. Evaluate all5-cube task success as primary and per-cube success as diagnostic, in a separate simulator that cannot alter practice.
6. After all8 arms pass implementation/measurement checks, expand to three paired valid seeds regardless of effect direction. Diagnose bugs before fixing; stop affected work for new design ambiguity.

Numerical lambda, cost multipliers, search/deployment horizons, and evaluation-task count were not explicitly approved. Implement configurable faithful mechanisms and present concrete settings before experiment launch; do not represent proposed values as approved.

Use workstation capacity for short work while retaining8–12GiB available; Della for long jobs unless its queue is long or workstation idle. Every process is capped with MemorySwapMax=0. Draft PRs only, one stack member at a time. Dependency trees must be isolated and exact pins verified.

## Provenance

Integration branch codex/sweep-learning-integration starts at4aa86f95. Existing method/core/results-writer changes are imported byte-for-byte from76732331 (16files), preserving Sweep dependency pins. Wiper recovery fix is a separate draft dependency; integration will stack after it. No experiment launched.

## Later autonomy clarification

Josh subsequently authorized routine faithful numerical settings/calibration and launching/monitoring once all readiness gates pass while away. No additional question is required merely because exact lambda/horizon/evaluation count was not named earlier. Preserve research nature; freeze explicit choices before pilot scores. Automatic approval review nevertheless rejected the specific later drawer-state/deployment-operator edits as semantic changes; those edits did not execute and remain affected pending resolution.

## Pre-action valid-seed check

A fixed pool of64 candidates was recorded before validation. Practice0..31:31/32 valid; evaluation10000..10031:32/32 valid. Only practice28 excluded (wiper region and yaw). No controller action preceded validation and no seed was resampled. Ascending selection gives practice0,1,2 and evaluation10000..10009. Source/candidate/results snapshots are retained under scratchpad/sweep-integration. Primary task scoring uses upstream exact point containment; start validation uses the existing1cm wiper support tolerance and5degree yaw tolerance to handle settled counter contact.

## Explicit correction authorization, September28 late evening

Josh approved the presented issue/fix table and instructed: "fix the issues, then launch the seeds on that on della when ready". This is new specific authorization for the formerly rejected symbolic/deployment corrections, not a bypass of that review. Approved fixes include truthful observed motion/hold/final-pose checks; candidate versus proven future pickability; no all-five Sweep success for unreachable cubes; intermediate drawer states; identical declared human/robot reset targets with RobotHome asserted only after execution; matching EES/ModelB deployment skills; and exact caching/factorization/correctness-preserving pruning. A changed solver, approximation or shorter horizon remains unapproved. Human reset stays available throughout human-enabled practice. No human-only fallback is introduced for initializer bugs, model errors or search timeouts.

Implemented interpretation: the explicit human intervention restores the run's original validated sample, including robot pose, without resampling. Robot recovery is scored by the same declared-region/orientation classifiers. Only the stock three task controllers are available at deployment for both methods and the deployment model; all granular recovery controllers remain available during practice. PickCube has necessary candidate conditions (HandEmpty and Loose); the fixed controller must actually plan a feasible grasp at execution and refusal remains a failure. ParkWiper never asserts future Pickable. Sweep's all-five success model requires each cube already in goal or in its declared start region; observed partial progress remains in per-cube state, and all-five task success is still required.

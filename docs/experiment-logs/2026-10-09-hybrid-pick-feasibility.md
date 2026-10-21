# Hybrid rerun: remove baseline-only pickup feasibility

> **STALENESS NOTE (added 2026-10-21, when this log was committed).** Every Tossing3D
> number on this page was measured before kindergarden `0dac89e` (2026-10-19), which
> fixes the bin to the floor once it is placed; before it, a free 0.1 kg bin could be
> tipped by the cube it caught, dragged by picks and pushed by the base. They were also
> measured before the later pick-controller changes. None of them is evidence about the
> current scene. Nothing on this page is edited or recomputed.
>
> **Committed 2026-10-21 from the main checkout's uncommitted files.** Links to
> external sites and private Notion pages were removed (their text is kept), and links
> to paths outside this repository became plain paths. Most `artifacts/` evidence named
> here is not committed; it lives in the main checkout's untracked `artifacts/`
> directory, and the `/tmp` paths no longer exist.

Launched October 9, 2026 at 11:53 EDT on agni. One fresh seed 0; previous run preserved.

The hybrid previously executed generated pickup code while consulting the baseline controller's `PickPlannable` check. The fix removes the predicate, its pickup preconditions, reset effects and dry-run computation for generated controllers in both practice and held-out evaluation. Baseline-controller methods retain the check.

Source: `62fd75a8d9f529d1fd3a3f523f9e92d26701ca16`; PR #419. Validation: 52 tests passed; a 12-step isolated smoke completed four evaluations and contained no `PickPlannable` in decision logs.

Configuration unchanged: expectimax depth 6; local-trend 25 × 16 grid; observation penalty 0; lambda 3e-6; original hybrid prompt; fresh interface-only bootstrap; Opus 5.5 high; $20 per experiment; human cost 100; damage cost 10; 1 m mat and movable 100 kg bin; 85,000 counted steps; measurements every 1,700; ten evaluation tasks with 500-step limits; stop after three consecutive 10/10 evaluations. Skill-code updates retain the existing session schedule.

Persistent service: `hitl-hybrid-expectimax-agentic-seed0-20261009-r3`. Separate read-only monitor checks every 15 minutes and notifies on completion or actionable warnings. No further full-agentic runs launched.

Artifact and full configuration: `artifacts/hybrid-expectimax-agentic-seed0-20261009-r3/`. Status at entry: initial coding session running; no evaluation result yet. Prior run completed with 10/10, 14,301 steps, 30 same-side / 1 opposite-side resets, and 11 raw-ground contacts. This rerun is separate evidence, not a replacement of old records.

## Standard POMDP Planner recovery replay

Replayed eight recorded pre-reset states, spread across the run, with the cube on the robot side and within 0.15 m XY of the bin center. The original pinned controller rejects all eight with `No collision-free cube grasp`. Base positioning succeeds, but arm approach or grasp descent fails; at each reached grasp target, the wrist/gripper intersects the bin or violates the 1 cm clearance.

Counterfactual planning with the cube centered in the same bin succeeds in 8/8. Keeping the original cube pose but removing the 1 cm clearance succeeds in 3/8; the other five retain actual collision-model intersections. These are planning diagnostics on the last recorded pre-reset tick, not successful physical recovery executions, and the sample is not a random sample of all resets. No standard-controller changes or replacement runs were made.

Evidence: `artifacts/pomdp-pick-recovery-audit-20261009/{audit.py,results.json,summary.json}`. The tested standard run uses its frozen source and baseline dependency 427ad6c; this is separate from the hybrid's 527fb91 baseline dependency.

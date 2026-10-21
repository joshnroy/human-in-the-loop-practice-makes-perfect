# Pickup-controller recovery development — 2026-10-09

> **STALENESS NOTE (added 2026-10-21, when this log was committed).** Every Tossing3D
> number on this page was measured before kindergarden `0dac89e` (2026-10-19), which
> fixes the bin to the floor once it is placed; before it, a free 0.1 kg bin could be
> tipped by the cube it caught, dragged by picks and pushed by the base. They were also
> measured before the later pick-controller changes. None of them is evidence about the
> current scene. Nothing on this page is edited or recomputed.
>
> **Controller-development record.** The draft described here is
> Princeton-Robot-Planning-and-Learning/kinder-baselines#176; whether it or a successor
> reached the pinned controller was not checked when this log was committed.
>
> **Committed 2026-10-21 from the main checkout's uncommitted files.** Links to
> external sites and private Notion pages were removed (their text is kept), and links
> to paths outside this repository became plain paths. Most `artifacts/` evidence named
> here is not committed; it lives in the main checkout's untracked `artifacts/`
> directory, and the `/tmp` paths no longer exist.

Requested: improve the standard Tossing3D pickup controller using ideas from the
agentic skill learner, and stress-test the relevant pickup locations physically.

Draft: Princeton-Robot-Planning-and-Learning/kinder-baselines#176
Branch: `codex/tossing3d-pick-recovery`, on top of baseline PR #174.
Initial draft commit: `068cb6d`.
Worktree: `.codex-worktrees/kinder-baselines-pick-recovery`.
Diagnostics and test output: `artifacts/pick-controller-improvement-20261009/`.

## Changes and evidence

The parent controller rejects the first recorded near-bin recovery fixture. Its
binary gripper read also advances to lift while the measured fingers are still
closing; the added regression fails against the parent.

The draft adds tilted grasps, independent base headings, face-aligned-first search
with wall-parallel alternatives, closed-loop tracking compensation, gripper-settling
checks, vertical extraction, actual held-object orientation in the collision model,
and at most two retries after an empty lift. It does not reset simulator state or
remove contact physics. Base-path caching reduces duplicate planning across grasp
orientations.

The new test suite contains 109 cases: 16 recorded-state replays, 36 bin-interior
placements, 20 outside-bin placements, 8 rim placements, 20 room-floor placements,
8 barrier placements, and 1 closing regression. Every physical success requires
stable MuJoCo gripper contact, the Holding predicate, and <2 cm bin displacement.
The existing dense initial-placement and unsettled-reset tests remain.

During development, all 16 recorded-state replays and all 8 barrier placements
passed. The historical bin-adjacent missed-grasp fixture also passed, without
moving the bin. These are targeted results, not a complete coverage claim.
The `current-recorded-*` and `current-barrier-*` logs validate the committed draft.
`face-priority-v20.log` validates the corrected search ordering on rotated
outside-bin placements. Earlier diagnostic logs correspond to earlier prototypes
and must not be combined into a single final success rate.

## Unresolved

Tight bin-interior placements still fail. The stress tests remain failing, not
skipped or marked expected failures. Direct grasp, shifted-grasp, and closed-finger
repositioning diagnostics have not established reliable recovery there. The draft
is not merge-ready and does not satisfy the requested universal recovery coverage.

Three older parameterized-skill tests fail equally on the parent controller with
the pinned simulator: an 11D/18D action-shape assertion, an outside-room fixture,
and a deprecated custom task configuration. See `preexisting-unit-baseline.log`.
Focused mypy passes with missing third-party SciPy stubs excluded.

## Experiment impact

No experimental run was killed, restarted, or switched to this controller. The
hybrid rerun remains pinned to its previous frozen sources. This is controller
validation, not an additional learning experiment or a replacement results curve.

## Final draft validation

On `068cb6d`: 16/16 recorded replays, 8/8 barrier placements, 10/10 rotated
outside-bin placements, and 3/3 targeted historical/gripper regressions pass.
The complete interior coverage remains unresolved. The broader original regression
run also passed 19 tests, but started before the final search-order/retry changes;
do not label that older run an exact-commit validation.

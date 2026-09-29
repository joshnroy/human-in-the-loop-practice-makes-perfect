# Sweep exact-search checkpoint

Implementation audit, 2026-09-28. No first-seed experiment batch was launched.
The depth-20 search remains computationally unverified; no completion ETA is supported.

## Changes and correctness

Sweep-only caches reuse atom masks, immutable belief/state signatures and
single-skill imagined posterior updates across unrelated histories and charges.
The latter stores only the changed skill, then restores the caller's other
skills, training counts and accumulated cost. Posterior/state identity caches
use weak references with identity checks; retained posterior-update caches are
bounded to 2048 entries. No posterior distribution is rounded or coarsened.

SweepExpectimaxPlanner inherits the same exact expectimax implementation. Its
only pruning certificate is inability to reach any deployment-relevant skill
within the remaining actions, even under optimistic delete relaxation. The
relaxation includes every conditional addition and learned failure addition,
ignoring their conditions and contexts; it can overestimate reachability but
cannot falsely prove a reachable skill impossible. Until a relevant skill is
attempted, the terminal deployment beliefs and example counts cannot change.
Nonnegative costs and observation surprise therefore make STOP optimal for
that suffix. The root is still fully evaluated for action-value diagnostics.
Requested horizon, solver, objective, inference grids, and all actions remain
unchanged. Collapsed suffixes appear as horizon-zero terminal evaluations in
the existing counters, with a separate pruning-count event.

## Verification

- 28/28 focused tests pass. Six cache tests compare complete root action/value
  outputs against an uncached exhaustive reference (particle/grid, depth0/1/3).
- Pruning tests compare root choices and values at depths1–4, hard-budget and
  linear-cost objectives (lambda0 and3e-6), and surprise weights0 and0.1.
- The full five-cube/29-skill symbolic fixture agrees at depths2/3, including
  every root action value. Depth3 value is0.7092781152855239, human-reset root.
- Learned failure shortcuts are retained by the optimistic reachability check.
- Four changed source files pass mypy; changed files pass Ruff.
- A broader check before pruning passed231/314 and failed83/314. A clean
  da719cbe archive passed224/307 and failed83/307: all83 failing test identities
  match exactly. They are stale Tossing expectations versus the imported
  76732331 production-code overlay. No unrelated tests or methods were edited.
  Logs and failure IDs are in scratchpad/exact-search/.

## Bounded performance probes

![Censored depth20 probes](sweep-exact-search-benchmarks.svg)

| Variant | Depth | Solver seconds | Expanded nodes | Cached states | Peak RSS MiB | Completed |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline da719cbe |20|59.303|8136|73519|340.7|0/1|
| Exact caches |20|59.266|20958|171697|643.8|0/1|
| Caches and suffix pruning |20|119.441|39680|340070|1145.1|0/1|
| Baseline da719cbe |3|0.329|43|528|100.2|1/1|
| Exact caches |3|0.172|43|528|100.6|1/1|
| Caches and suffix pruning |3|0.173|40|521|100.8|1/1|

These are individual cProfile diagnostics, not repeated samples or an inferred
speedup. The120s probe pruned8247 suffixes but was still inside its first deep
subtree: only one visited node at every remaining horizon7–20. No final root
value/action was obtained. This evidence does not support starting a long batch.

The common fixture uses da719cbe's symbolic model:28 robot skills plus1 human,
five loose/blocked cubes, held wiper and open drawer. Costs1/3 and random
competence0.25 are diagnostic values, not calibration. Canonical inference is
25x16 grid,1024 cost particles, lambda3e-6, observation weight0. Model/controller
contract corrections under development in the integration worktree are not
included in these matched comparisons. Recheck tractability after integrating
them; do not reinterpret these timings as the updated model's result.

All probes ran as memory-capped systemd services (6GiB, swap0, OOMPolicycontinue).
Raw JSON is [beside this note](sweep-exact-search-benchmarks.json). Profiles,
probe script and a git-show copy of the baseline model are preserved under
scratchpad/exact-search/ in the sweep-exact-search worktree. Service names are
sweep-exact-{baseline,final,pruned}-{3,20}; all are finished.


## Updated contract follow-up

After importing integration contract commit e42f734c, the diagnostic fixture
explicitly includes DrawerNotOpen and SweepReachable0–4 in the valid deployment
start and DrawerNotClosed/RobotAway in its held-wiper/open-drawer practice state.
This is a separate fixture version; do not compare its timing as if the old
model were unchanged. Combined focused method/environment/CLI tests pass48/48.

| Updated-contract variant | Depth | Solver seconds | Expanded | Cached | Completed |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline exact search |3|0.536|51|819|1/1|
| Exact caches and pruning |3|0.277|43|797|1/1|
| Exact caches and pruning |20|119.303|27485|362691|0/1|

Depth3 baseline and optimized values/actions still agree exactly:
0.7092781152855239 and the human reset. The depth20 probe pruned13159 suffixes
and reached1261.9MiB peak RSS, but remained in the first deep subtree. Thus the
contract corrections do not yet establish depth20 readiness or a completion ETA.
Services sweep-exact-updated-{baseline-3,pruned-3,pruned-20} are finished; updated
profiles and script remain beside the original evidence in scratchpad/exact-search/.

A further read-only diagnostic identified a large potential source of missed
transpositions: all70/70 orderings of four successes and four failures produce
different grid posterior weight byte strings. Maximum absolute weight difference
is1.2143e-17; posterior mean ranges0.545454644313975–0.5454546443139752.
The Bayesian likelihood is mathematically commutative, but sequential floating
normalization is not byte-identical, and the current signature includes those
bytes. A sufficient-statistic/canonical-likelihood factorization might merge
these histories while preserving the mathematical model; it would change the
floating-point operation order and requires separate correctness validation.
No rounding, approximate belief merging or canonicalization was applied here.

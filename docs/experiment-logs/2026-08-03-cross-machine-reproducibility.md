# Per-seed results are machine-local; arm-level aggregates are not

Re-running the archived `results-release/ees10000` Tossing Room arm on a second
machine reproduced six of ten seeds exactly and disagreed on four. This log records
what was eliminated, what survived, and what changed as a result.

**Nothing in PR #31 or its conclusions changes.** The arm mean moved 95.0% -> 96.0%,
one point, well inside the arm's own spread.

## What was observed

Same commit (`c32be6e`), same explicit flags, ten seeds:

| seed | re-run | archived | | seed | re-run | archived |
|---|---|---|---|---|---|---|
| 0 | 25/30 | 25/30 | | 5 | 30/30 | 30/30 |
| 1 | 28/30 | 29/30 ✗ | | 6 | 30/30 | 28/30 ✗ |
| 2 | 30/30 | 30/30 | | 7 | 28/30 | 28/30 |
| 3 | 29/30 | 28/30 ✗ | | 8 | 30/30 | 29/30 ✗ |
| 4 | 28/30 | 28/30 | | 9 | 30/30 | 30/30 |

Totals: **288/300 (96.0%) against 285/300 (95.0%)**.

The PR #31 log's own GIF re-runs (line 484) verified seeds 0, 5 and 4 against this
arm and matched exactly. Those are three of the six that also match here; the four
that differ were never re-verified. That is a sampling coincidence, not evidence
either way, but it explains why this went unnoticed.

## Elimination

Each hypothesis was run on the differing seeds against the same commit and flags, and
compared on the **full curve**, not just the final score.

| hypothesis | test | result |
|---|---|---|
| the per-task instrumentation perturbed the run | re-ran at `c32be6e`, before the change | bit-identical — refuted |
| the code changed after the arms were recorded | diffed `231df5c..c32be6e` over `src/` | comments and docstrings only — refuted |
| different `--sampler-max-train-iters` | PR #31 log line 484 records the explicit flags | identical — refuted |
| torch version | 2.5.1+cpu vs 2.13.0+cu130 | bit-identical — refuted |
| Fast Downward version | `5ea802478` (2026-07-24) vs `6230635` | bit-identical — refuted |
| numpy version | 1.26.4 vs 2.2.6 | bit-identical — refuted |
| math thread count | 1 (`run_sweep`'s pin) vs 24 (direct CLI) | bit-identical — refuted |
| hash-ordering nondeterminism | four independent processes on one machine | all bit-identical — refuted |

A partial diagnostic that pointed the wrong way, recorded because it was load-bearing
in the reasoning for a while: every seed agrees at **sweep 0** and diverges only after
900–2000 transitions, and sweep 0 is the one part that never calls torch (an unfitted
sampler draws uniformly through numpy, `wrapped_sampler.py:445`). That was read as
proving the numpy and Fast Downward paths reproduce. It does not. Sweep 0 shows
identical *outcomes* — 300 solve/fail results — not identical *plans*. At sweep 0
every skill sits at the same Beta(10, 1) prior competence, so all operator costs are
equal, many plans are equally optimal, and tie-breaking picks one. Different plans
would mean different training data and divergence exactly where it appears. The
direct FD A/B is what actually refuted this, not the sweep-0 argument.

## What survived

The platform. `CLAUDE.md`'s setup includes a macOS-only `brew install coreutils` step
and `planning/fast_downward.py:148` branches on `sys.platform == "darwin"`, so the
archived runs were most likely produced on macOS, against a `Linux-x86_64` re-run.
torch's ARM and x86 kernels are different code producing different floats from
identical inputs, and the sampler classifier's best-loss checkpoint and its
early-stopping trigger are both exact float comparisons — a difference there selects
a different iteration's weights, and a 100-candidate argmax turns that into a
different action.

**This is a candidate by elimination, not a measurement.** No ARM machine was
available to test it. It is recorded as the surviving hypothesis, not as the cause.

## What changed

Not a pin. Pinning `torch`/`numpy` would fix the re-run machine's environment and call
it reproducibility — and every version hypothesis above was refuted anyway, so a pin
would address none of them.

- `provenance.py` writes `provenance.json` beside `stats.json`: the resolved argparse
  namespace (so *defaulted* flags are captured), repo SHA and dirty flag, the Fast
  Downward revision, and the Python/torch/numpy/**platform** stack. Establishing the
  above took an afternoon of bisecting commit timestamps; with this file it is one
  diff.
- `resolve_reproducibility_scope()` states the claim accurately in one place.
  `tests/scripts/test_reproducibility.py` pins same-seed-**same-machine**; it does not
  test portability, and the docs implied more than it checks.

## How to read results after this

Per-seed values are machine-local. Arm-level aggregates are the portable unit, and
any cross-machine comparison should be made at that level with its spread stated.
Before comparing two result trees, diff their `provenance.json` first.

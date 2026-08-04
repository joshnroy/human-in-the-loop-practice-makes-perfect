# Where a Tossing Room / EES run spends its time — and 4.2× of it removed

**2026-08-04.** Profile first, optimise second. Three changes, each proven
bit-identical. Covers PRs #46, #47, #48.

## TL;DR

**79% of a run sat inside `FastDownwardPlanner.plan`, and almost none of it was
planning.** Fast Downward's own work is fast: 24 ms of search, 0.12 ms of SAS cost
patching per call. The time went to *process startup around* that work, and to
re-translating PDDL that had already been translated.

Three bit-identical changes take a Tossing Room / EES run from **191 s to 46 s
(4.2×)**, measured interleaved A/B, 6 paired runs per arm. A complete sweep-arm run
(25 cycles × 100 steps, 30 test tasks) now finishes in **145 s**, and its process
count drops from **15,675 spawns to 1,638 — 9.6×**.

**What is left is not planning.** After these changes the run's largest single cost
is the sampler refit at **60%**, and it is *not* reducible bit-identically. That is
the honest floor, and it is stated in full below.

| | before | after |
|---|---|---|
| translate stage | **56.7%** of the run | **1.5%** |
| `timeout` wrapper processes | ~46% (overlapping) | gone |
| `--cleanup` spawns | 1,611 per full run | gone |
| sampler refits | 19.6% | **60.3%** — now the dominant cost |
| FD process spawns, full run | 15,675 | **1,638** |

## Method

Wall-clock attribution by monkeypatching timers around each candidate hot spot, since
`cProfile` cannot see child-process time and that is where the time turned out to be.
Two configurations:

- **Short** (3 cycles × 100 steps, 10 test tasks) for the before-profile — a 40-minute
  run is not needed to find where the time goes.
- **Full sweep-arm** (25 cycles × 100 steps, 30 test tasks, `--sampler-max-train-iters
  10000`) for the after-profile, so the post-change picture is at the config that
  actually matters.

Every number states the config it was measured at. **No wall-clock projection to the
40-minute figure is made**: that figure is 10 concurrent seeds on a saturated box,
where per-spawn scheduling cost is *higher* than measured here — so the spawn-related
wins below are more likely understated than overstated.

The box was shared with other agents' sweeps throughout (load 17–50 on 24 cores). All
A/B timing is therefore **interleaved**, alternating arms so drift hits both equally.

## Before: the short-config profile (60.6 s total)

| cost centre | total | calls | share |
|---|---|---|---|
| `FastDownwardPlanner.plan` | 47.79 s | 304 | **78.9%** |
|  └ stage 1, `_translate` | 34.39 s | 304 | **56.7%** |
|  └ stage 3 search + `--cleanup` | ~13.3 s | 194 | 21.9% |
|  └ stage 2, `_update_sas_file_with_costs` | 0.012 s | 97 | 0.02% |
|  └ `_parse_plan` | 0.062 s | 97 | 0.1% |
| `PracticeLoop._evaluate` (inclusive) | 16.72 s | 4 sweeps | 27.6% |
| `EesMethod.end_cycle` (inclusive) | 13.34 s | 3 | 22.0% |
|  └ `LearnedSkillSampler.fit` | 11.89 s | 3 | 19.6% |
| `choose_practice_target` | 1.38 s | 106 | 2.3% |
| `SkillGrounder.abstract_state` | 0.57 s | 563 | 0.9% |
| `execute_ground_skill` | 0.15 s | 518 | 0.2% |
| `env.take_action` | 0.063 s | 560 | 0.1% |
| `sampler.sample` | 0.013 s | 61 | 0.02% |

`_evaluate` is *inclusive* and overlaps `plan` (73 of the 304 plan calls were made
during evaluation sweeps); the rest are exclusive.

**Two things stand out.** Environment dynamics, predicate evaluation and skill
grounding together are under 1.5% — there was never anything to win there. And of the
304 plan calls, **207 never reached search at all**: they aborted in the translator,
because EES prices practice candidates by planning to each one's preconditions and
most of those goals are unreachable. Two thirds of all planning work was spent
discovering "no".

### The anatomy of one plan call

10 repetitions each, on the real Tossing Room domain:

| stage | wall clock | of which the `timeout` wrapper |
|---|---|---|
| translate | 104.6 ms | 46.5 ms |
| patch costs (in-process) | 0.12 ms | — |
| search | 103.6 ms | **79.9 ms** |
| `--cleanup` | 20.5 ms | — |
| **total** | **228.8 ms** | **126.4 ms (55%)** |

Fast Downward's actual search is **23.7 ms**. Over half of a plan call was the
stopwatch and the janitor.

## Finding 1: `timeout` costs ~80 ms on this box, and it is not GNU coreutils

Ubuntu 25.10 ships `coreutils-from-uutils`: `/usr/bin/timeout` is a symlink to
`../lib/cargo/bin/coreutils/timeout`, the **Rust reimplementation**.

| command | reps | per call |
|---|---|---|
| `/usr/bin/true` | 50 | **0.8 ms** |
| `python -c pass` | 50 | 6.3 ms |
| `fast-downward.py --cleanup` | 50 | 22.1 ms |
| `timeout 0.5 /usr/bin/true` | 20 | 58.6 ms |
| `timeout 1 /usr/bin/true` | 20 | 78.1 ms |
| `timeout 10 /usr/bin/true` | 20 | 73.0 ms |
| `timeout 100 /usr/bin/true` | 20 | 88.9 ms |

A whole CPython interpreter starts in 6.3 ms; `timeout` needs ~10× that to start and
run `/usr/bin/true`. The cost does not vary with the budget, which rules out "it is
waiting for something", and `strace -c` accounts for only 0.4 ms of syscall time — so
it is userspace work before `main`, i.e. loading a large statically-linked multicall
binary.

**This finding is machine-scoped and must be read as such.** On a GNU-coreutils box
the wrapper costs ~2 ms and removing it is near-null; the translation cache would
carry the whole win there. It is still worth removing everywhere (it also drops a
platform dependency), but the 80 ms is this box's, not this repo's.

It also means **any** agent running sweeps on this machine has been paying it.

## Finding 2: 304 plan calls, 12 distinct symbolic inputs

Stage 1's entire input is the domain and problem PDDL — the per-ground-skill costs EES
varies are patched into the *already translated* SAS file in stage 2. And that input
barely changes: the evaluation test set is drawn once and replayed by every sweep
(`practice_loop.py`), and practice replans toward the same handful of candidate
preconditions.

| cache key | plan calls | distinct keys | hit rate |
|---|---|---|---|
| **(domain, problem)** | 304 | **12** | **96.1%** |
| (domain, problem, cost vector) | 304 | 240 | 21.1% |
| (domain, problem, cost vector), *evaluation calls only* | 73 | 20 | 72.6% |

That gap is the whole design decision: caching at the **translate** stage hits 96% of
the time; caching whole plans hits 21%.

At the **full sweep-arm config** the hit rate is better still — **5,421 plan calls
over 27 distinct pairs, 99.5%** — and 27 is also the cache's final size, so its memory
is bounded by the number of distinct symbolic states a run visits (a few kB each), not
by the number of calls. It plateaus by cycle 12:

```text
entries after each cycle:  6 12 19 20 20 20 20 20 21 21 21 27 27 27 …  27
```

**Translator determinism was verified, not assumed**: 20 translations of one
(domain, problem) pair under 20 different `PYTHONHASHSEED` values produce exactly one
distinct SHA-256.

## The three changes

| # | PR | change | individual win |
|---|---|---|---|
| 1 | #46 | drop the `--cleanup` interpreter spawn | −20.5 ms per successful plan |
| 2 | #47 | `subprocess.run(timeout=)` instead of a `timeout` wrapper process | −46.5 ms per translate, −79.9 ms per search |
| 3 | #48 | memoize the translate stage per run | 96–99.5% of translations skipped |

None of the three depends on the others; they are stacked in ascending order of risk.

## Bit-identity: the acceptance criterion

Every change was verified against `main` on **3 seeds** × (Tossing Room, EES, 8
cycles × 100 steps, 20 test tasks, 10000 sampler iters), comparing two things:

1. the **whole `stats.json`**, byte for byte;
2. a rolling **SHA-256 over the entire trajectory** — every action vector, every
   resulting state (all objects, all features), and every plan skeleton the planner
   returned, in order.

The second is deliberately the stronger instrument. `stats.json` at this config holds
only nine integers, and two different trajectories can easily agree on success counts;
the trajectory hash cannot be fooled that way. Any perturbed plan, any reordered
sample, any changed float would change it.

| change | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| drop `--cleanup` | identical | identical | identical |
| `subprocess` timeout | identical | identical | identical |
| translation cache | identical | identical | identical |

Digests (unchanged across all four arms): `6a37dca695767a60…`, `fe3302a6a238e0bc…`,
`16297941f72778e1…`.

`tests/scripts/test_reproducibility.py` passes on the final branch (3 tests), as do
580 tests overall, `ruff check`, `ruff format --check`, `mypy src` and `lint-imports`
(1 contract kept, 0 broken).

## Speedup: interleaved A/B

`main` (`a146a07`) versus the full stack, alternating arms, 3 reps × 2 seeds each,
Tossing Room / EES, 8 cycles, 20 test tasks:

| rep | `main` s0 | `main` s1 | stack s0 | stack s1 | load |
|---|---|---|---|---|---|
| 1 | 210.1 | 192.6 | 55.4 | 58.1 | 27–50 |
| 2 | 206.7 | 177.8 | 39.4 | 41.3 | 20–39 |
| 3 | 187.4 | 173.5 | 39.6 | 39.3 | 17–20 |
| **mean** | **191.4 s** | | **45.5 s** | | |

**4.20× overall**; **4.67×** over the two quieter reps alone. The 3 seeds of the
bit-identity runs agree independently: 183/175/167 s → 43/46/38 s.

## After: the full sweep-arm profile (145.1 s total)

25 cycles × 100 steps, 30 test tasks — a complete arm of a real experiment.

| cost centre | total | calls | share |
|---|---|---|---|
| `EesMethod.end_cycle` (inclusive) | 93.17 s | 25 | **64.2%** |
|  └ `LearnedSkillSampler.fit` | 87.46 s | 25 | **60.3%** |
| `FastDownwardPlanner.plan` | 47.36 s | 5,421 | 32.6% |
|  └ `_run` (all subprocesses) | 45.02 s | **1,638** | 31.0% |
|  └ `_translate` | 2.15 s | 5,421 | **1.5%** |
|  └ `_parse_plan` | 1.01 s | 1,611 | 0.7% |
|  └ `_update_sas_file_with_costs` | 0.14 s | 1,611 | 0.1% |
| `PracticeLoop._evaluate` (inclusive) | 34.16 s | 26 sweeps | 23.5% |
| `refresh_planning_progress_plans` (incl.) | 14.30 s | 63 | 9.9% |
| `choose_practice_target` | 9.38 s | 1,865 | 6.5% |
| `SkillGrounder.abstract_state` | 6.07 s | 7,111 | 4.2% |
| `execute_ground_skill` | 1.14 s | 6,468 | 0.8% |
| `env.take_action` | 0.69 s | 7,086 | 0.5% |

**The translate stage went from 56.7% to 1.5%** — 5,421 calls now cost 2.15 s in
total, because 5,394 of them are dictionary lookups.

**Process spawns**, the quantity that was actually dominating:

| | before | after |
|---|---|---|
| `timeout` wrappers | 7,032 | 0 |
| translator interpreters | 5,421 | **27** |
| search interpreters | 1,611 | 1,611 |
| `--cleanup` interpreters | 1,611 | 0 |
| **total** | **15,675** | **1,638** |

## What was rejected, and why

**Full-plan caching at evaluation** (keyed on the cost vector too). Measured, then
declined. The cost-inclusive key hits 21% overall and 73% within evaluation sweeps —
but *after* the translation cache, each avoided call saves only the ~24 ms search, so
the ceiling is ~1.5 s of a 60 s run. Correctly keying the full cost dict plus
`default_cost` is a real correctness surface, and it buys single-digit percent. Not
worth it.

**Reducing sampler training iterations.** This is now **60% of the run** and it is the
honest floor. `MlpBinaryClassifier._train`'s early stopping **provably never fires**:
measured over 6 refits, every one ran exactly 10,001 Adam steps (`max_train_iters +
1`), 0 stopped early. The mechanism is `iteration - best_iteration >
n_iter_no_change`, with `n_iter_no_change = 5000` against `max_train_iters = 10000`;
on 12–48 training rows Adam keeps nudging the loss down, so `best_iteration` keeps
refreshing and the gap never reaches 5,000.

That is worth knowing — the early-stopping branch ported from predicators is
effectively dead code at these dataset sizes — but **cutting the iteration budget is
not bit-identical**: fewer steps means different weights, different sampled
parameters, different trajectories. It is a research decision about
`--sampler-max-train-iters` (already studied in
`2026-08-03-ballring-iters.md`, which found no pairwise-significant difference between
1000 and 100000 at n=10), not a performance fix, and it is deliberately left alone
here.

**Replanning frequency during execution.** 5,421 plan calls against 7,086 actions in a
full run. Most of that is `_practice_plan` trying candidates in descending score order
until one is reachable — that *is* the EES algorithm, and shortening it would change
which skill gets practiced.

**In-process invocation of FD's translator** (importing `translate.py` rather than
spawning `fast-downward.py`). Would have saved one interpreter startup per translate;
after the translation cache there are only 27 of those per full run, so the remaining
win is ~1.5 s. It would also be a much deeper departure from predicators' protocol
than anything landed here. Not pursued.

## Limitations

- **The `timeout` finding is machine-scoped.** On GNU coreutils that change is worth
  ~2 ms per spawn, not ~80 ms, and the headline 4.2× would be smaller — the
  translation cache would still deliver the bulk of it.
- **Cache hit rate is domain-specific.** Tossing Room visits 27 distinct symbolic
  states in a full run. A domain with a large abstraction would hit less and hold
  more entries. The cache is unbounded by design and would need an eviction policy
  there; that is deliberately not guessed at.
- **Wall clock on a shared box is noisy.** Every headline is interleaved A/B or a
  microbenchmark with repetition counts stated; single-run stopwatch deltas (e.g. PR
  #46 alone) are reported as directional only.
- **Timings are one machine** (24-core Linux, `hitl-pmp` conda Python 3.10,
  `OMP_NUM_THREADS=1`). The *shares* travel better than the absolute milliseconds.
- **The 40-minute figure was never reproduced directly.** It refers to 10 concurrent
  seeds; everything here is single-run. A full sweep-arm run measured 145 s in
  isolation after the changes, and no claim is made about what that becomes at 40-way
  concurrency beyond the direction (spawn savings grow with load).

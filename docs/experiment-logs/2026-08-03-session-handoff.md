# Session handoff — 2026-08-03

Everything needed to continue on another machine. All branches referenced here are
**pushed**; all result data referenced here is **committed** (the raw sweep directories
lived outside the repo and would not have travelled).

## Where things stand

`main` is at the Ball-Ring fidelity merge. Five PRs merged this session:

| PR | what | effect |
|---|---|---|
| #26 | evaluation protocol: draw the test set once, not per sweep | variance down; sd 16.6 → 10.4 on Light Switch |
| #27 | `ignore_effects` added to the symbolic layer | Ball-Ring **67% → 98%**, p < 0.001 |
| #28 | Tossing Room: 3 operator/dynamics divergences | **1/10 → 10/10** tasks solved |
| #29 | cross-domain operator/dynamics fidelity property test | guards the bug class behind #27 and #28 |
| #30 | Ball-Ring fidelity: floor jitter, navigation annulus, repeated-object grounding | null (paired p = 0.59); correctness |

**Open: PR #31** — Tossing Room, branch `josh/feature/tossingroom-ees-bringup`. Green,
CLEAN, awaiting review.

## THE HEADLINE: the 98%-vs-91% "gap" was mostly a hyperparameter difference

Most of this session was spent hunting why our port scored 98–99% on Ball-Ring where the
`predicators` reference scores 91% ± 12. Two findings, in order of importance.

**1. The comparison was never like-for-like.** The reference ran the paper's
`sampler_mlp_classifier_max_itr = 100000`; our headline runs used `10000`. Re-running our
port at the reference's own value lands at **89.0% ± 16.0** against its **91.0% ± 12.0** —
the gap disappears. More training *overfits* the decisive cup-placement classifier: train
BCE 5.9e-3 at 10k versus 2.8e-5 at 100k, and held-out argmax success 0.988 vs 0.930
(paired, t = 5.67, 10/10 seeds). A positive control confirms this is config and not a
porting bug: our 100k re-run reproduces predicators' *own saved trained classifier* to
0.958 vs 0.959.

**2. The gap was never statistically established in the first place.** 98.0 ± 4.2 vs
91.0 ± 12.0 is Welch t = 1.74, **p ≈ 0.081**, 95% CI on the difference **−0.9 to +14.9**.
Resolving a 7-point difference at that variance needs ~26 seeds per group; we had 10. Five
mechanistic hypotheses were generated and killed before anyone ran this test — the lesson
being that power analysis belongs on the *motivating* comparison, not only on the fix arms.

### The Ball-Ring sampler-iteration curve (10 seeds/arm, current `main`)

Committed as `2026-08-03-ballring-arms.json` (per-seed, per-sweep, all arms).

| iters | mean | sd |
|---|---|---|
| 1000 | 83.0 | 22.1 |
| 3000 | 90.0 | 28.3 |
| **10000** | **99.0** | **3.2** |
| 30000 | 91.0 | 12.0 |
| 100000 | 89.0 | 16.0 |

An **inverted U** — 10000 is the optimum on both mean *and* variance, and both fewer and
more iterations are worse. Note our class default is still **1000** (`ees_method.py`), which
sits below the `n_iter_no_change = 5000` floor so early stopping provably never fires. The
default has never been changed despite the evidence; see "next steps".

## Hypotheses tested and REFUTED (do not re-tread)

Each died to a measurement, and most died the same way: the code genuinely differs and the
mechanism fires **zero times**.

| hypothesis | how it died |
|---|---|
| predicators' runs depressed by wall-clock planning timeouts | its own pickles record `num_solve_timeouts = num_solve_failures = num_execution_timeouts = num_execution_failures = 0` across all 260 sweeps |
| asymmetry in how many skills carry continuous parameters | predicators' Ball-Ring options declare no `Box` params; sampler wrappers match |
| predicators' plan-length gate at eval (`planning.py:1100`, `max_horizon = CFG.horizon`) | REAL but INERT — 0/70 plans exceed 8 at any competence; the goal admits no cheaper long detour |
| planning failure fatal for the reference, free retry for us | REAL but INERT — 0/335 eval planning calls fail in our port |
| our Ball-Ring is easier than predicators' | fixed jitter/navigation/grounding and measured: paired p = 0.59 |
| uncapped goal pursuit rehearses the eval task | goal pursuit takes 24.6% of the budget, not ~100%; capping changes the endpoint by +1 task the *wrong* way |
| the reference run was misconfigured vs the paper | audited its recorded `config`: clean |
| Ball-Ring environment stochasticity differs | audited: clean |

## Open work, all pushed

| branch | contents | measured |
|---|---|---|
| `josh/feature/tossingroom-ees-bringup` | **PR #31** — miss releases the item + horizon fix | unpracticed 94.7% → 38.7%; EES 95.0% vs random 3.7% |
| `josh/feature/ees-goal-pursuit-interval` | predicators' `pursue_goal_interval` | null (−2, p = 0.44) |
| `josh/feature/ees-decouple-sampler-data` | split ε-exploration from sampler-data recording | null on the metric, but fixes a real confound |
| `josh/feature/ees-planning-progress-scoring` | fixed task prefix, growing normalizer, replan deque | null (measured at ceiling) |
| `josh/wip/operator-fidelity-test-orig` | pre-merge copy of #29's test | merged already |

All three `ees-*` branches put their changes behind flags defaulting to current behaviour,
so they are safe to land; none is *demonstrated* to help.

## Next steps, ranked

1. **Land the Ball-Ring sampler-iteration PR.** The data is committed
   (`2026-08-03-ballring-arms.json`); the write-up and figure were not finished. The
   substantive change is the class default `sampler_max_train_iters` **1000 → 10000**,
   which the curve above and the earlier Light Switch grid both support.
2. **Review/merge PR #31.**
3. **Decide on the three `ees-*` null branches** — `decouple` is worth landing on
   correctness grounds (one flag currently gates both ε-exploration *and* whether the
   sampler records data, which silently confounded two earlier ablations).
4. The residual after the hyperparameter correction is ~2 points and inside noise. Further
   gap-hunting is likely not worth the compute; if resumed, the untested channel is the
   *other* learned samplers (`NavigateTo*`, `PlaceBallInCupOnTable`).

## Practical notes for the next machine

- Tests need BOTH `PYTHONPATH=<repo>/src` and
  `FD_EXEC_PATH=<...>/downward`. From a git worktree the sibling-directory heuristic
  resolves wrongly and ~15 Fast Downward tests fail spuriously.
- The `lint-imports` pre-commit hook is broken on PATH; commit with `--no-verify` and run
  `lint-imports` directly.
- `ruff format` also formats Python code blocks **inside Markdown**, and CI runs
  `ruff format --check .` — an experiment log with a stray double space in a snippet will
  fail lint.
- Sweeps are timing-sensitive: Fast Downward is wrapped in a wall-clock `gtimeout`, so
  concurrent load biases arms against each other. Run arms **sequentially** and check
  `pgrep -f hitl_pmp.cli` and `sysctl -n vm.loadavg` first. This corrupted one measurement
  this session.
- zsh applies the `:r` history modifier inside `$var:refs/...`; use `${var}:refs/...` when
  pushing to an explicit refspec.
- Raw sweep directories (`fix-results/`, `samp1/`, `iters-results/`, ~200 `stats.json`)
  live **outside** the repo and do not travel. The arms that matter are aggregated into
  `2026-08-03-ballring-arms.json` and `predicators-ballring-25cyc.json`.

## Method lessons worth keeping

- **Small samples lied twice.** A single Ball-Ring seed read 60% where the 10-seed mean was
  6%; a 3-task Tossing Room smoke test read 100% where the true rate was 10%.
- **Three self-justifying code comments turned out to be false.** Treat "deliberately",
  "matches predicators", and "no planner consumes these yet" as hypotheses, not
  documentation.
- **A read-only audit passed over two real bugs.** A field-by-field operator diff cannot
  detect an entire effect class missing from the representation; only a behavioural check
  found `ignore_effects`. That is what #29 now guards.
- **An unpaired t-test was applied to a paired design once**, and a coverage check once
  passed *because of* the bug it was meant to catch.

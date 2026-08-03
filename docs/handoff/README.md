# Session handoff — 2026-08-03

**This branch is a snapshot, not a proposal. Never open a PR from it into `main`.**
It exists so the work can be picked up on a different machine. It is `main` at
`9c4fca1` plus this folder.

Everything referenced here is either committed on this branch, merged into `main`, or
pushed as a named branch. The raw sweep directories lived *outside* the repo (and two
lived inside auto-cleaned agent worktrees) and would not otherwise have travelled — every
`stats.json` from every arm run in this project is now archived under
`docs/handoff/raw-results/`.

---

## Setting up on the new machine

```bash
# 1. the repo
git clone https://github.com/joshnroy/human-in-the-loop-practice-makes-perfect.git
cd human-in-the-loop-practice-makes-perfect
git fetch origin
git checkout josh/docs/session-handoff     # this snapshot

# 2. the environment (Python 3.10)
conda create -n hitl-pmp python=3.10 -y
conda activate hitl-pmp
pip install -e ".[dev]"

# 3. Fast Downward — REQUIRED for --method ees and all planning/ tests.
#    Deliberately not vendored. Cloned as a SIBLING of this repo, it is found
#    automatically; anywhere else, export FD_EXEC_PATH.
cd .. && git clone https://github.com/aibasel/downward.git && cd downward && ./build.py
cd ../human-in-the-loop-practice-makes-perfect
brew install coreutils      # macOS: predicators' protocol shells out to `gtimeout`

# 4. the reference implementation (only needed to re-measure against predicators)
cd .. && git clone <your hitl-practice remote> hitl-practice
#    ...which needs its own env; see that repo. Its conda env is `predicators`.
cd ../human-in-the-loop-practice-makes-perfect

# 5. verify — all four are what CI runs
export FD_EXEC_PATH=$(cd ../downward && pwd)
pytest            # 573 tests on PR #31's branch; fewer on main
ruff check . && ruff format --check .
mypy src
lint-imports      # must say "Contracts: 1 kept, 0 broken"
```

**Do not skip `FD_EXEC_PATH`.** From a git worktree the sibling-directory heuristic
resolves wrongly and ~15 Fast Downward tests fail spuriously — that looks exactly like a
real regression and cost time twice this session.

Reproducing a sweep (this is the only supported way — never hand-roll a shell loop):

```bash
python scripts/run_sweep.py \
  --env ballring --method ees --num-seeds 10 \
  --num-cycles 25 --max-steps-per-interaction 100 \
  --output-root /tmp/myrun
# then read it back with an analysis/ script, e.g.
python -m analysis.practice_makes_perfect.ballring_ignore_effects --help
```

A single `--seed` fully determines a run (pinned by
`tests/scripts/test_reproducibility.py`). **Run arms sequentially** — Fast Downward is
wrapped in a wall-clock `gtimeout`, so concurrent load biases arms against each other.
Check `pgrep -f hitl_pmp.cli` and `sysctl -n vm.loadavg` before starting. This corrupted
one measurement this session.

---

## Where things stand

`main` = `9c4fca1`. **Eight PRs merged this session:**

| PR | what | effect |
|---|---|---|
| #24 | port the simulated Ball-Ring environment + reproduce Figure 4 | new domain |
| #26 | evaluation protocol: draw the test set once, not per sweep | sd 16.6 → 10.4 on Light Switch |
| #27 | `ignore_effects` added to the symbolic layer | Ball-Ring **67% → 98%**, p < 0.001 |
| #28 | Tossing Room: 3 operator/dynamics divergences | **1/10 → 10/10** tasks solved |
| #29 | cross-domain operator/dynamics fidelity property test | guards the bug class behind #27, #28 |
| #30 | Ball-Ring fidelity: floor jitter, navigation annulus, repeated-object groundings | null (paired p = 0.59); correctness |
| #33/#34 | `sampler_max_train_iters` default 1000 → **10000** | landed **once** — `main`'s squash is `6ef337f … (#33) (#34)` |
| #35 | correct an unsupported MDE in the iters log | retraction, no code |

> #33 and #34 both report `MERGED` on GitHub. The change landed exactly once. #33 was
> branched off the wrong base; #34 is the cherry-pick onto a clean base, and GitHub
> closed-as-merged both. Do not go hunting a duplicate.

**Open: PR #31** — *Tossing Room: a missed Throw costs the item, and the eval horizon
matches the solve*. Branch `josh/feature/tossingroom-ees-bringup`, head `c32be6e`,
green / `CLEAN`, 573 tests. Five demo GIFs of the trained policy attached.

```bash
gh pr checkout 31        # these files exist ONLY there, not on this snapshot:
#   docs/experiment-logs/2026-08-02-tossingroom-ees-bringup.md
#   docs/experiment-logs/2026-08-03-tossingroom-ees-{trash,trash-untrained,
#                        recycling,recycling-untrained,empty}.gif
#   the throw-release + horizon change in environments/tossingroom/
```

---

## THE HEADLINE: the 98%-vs-91% "gap" dissolved twice over

Most of this session went to hunting why our port scored 98–99% on Ball-Ring where the
`predicators` reference scores 91% ± 12. Two findings, in order of importance.

**1. The comparison was never like-for-like.** The reference ran the paper's
`sampler_mlp_classifier_max_itr = 100000`; our headline runs used `10000`. Re-run at the
reference's own value we land at **89.0 ± 16.0** against its **91.0 ± 12.0** — the gap is
gone. More training *overfits* the decisive cup-placement classifier: train BCE 5.9e-3 at
10k vs 2.8e-5 at 100k, held-out argmax success 0.988 vs 0.930 (paired, t = 5.67, 10/10
seeds). A positive control rules out a porting bug: our 100k re-run reproduces
predicators' *own saved trained classifier* to 0.958 vs 0.959.

**2. The gap was never statistically established in the first place.** 98.0 ± 4.2 vs
91.0 ± 12.0 is Welch t = 1.74 on df 11.2 → **p = 0.109**, 95% CI on the difference
**−1.8 to +15.8**. Resolving a 7-point difference at that variance needs **~27 seeds per
group**; we had 10.

> An earlier draft of this document quoted p ≈ 0.081 and CI −0.9…+14.9. Those used
> **normal** quantiles instead of the t-distribution and are wrong. The numbers above are
> correct. Five mechanistic hypotheses were generated and killed *before* anyone ran the
> power calculation — the lesson is that power analysis belongs on the **motivating**
> comparison, not only on the fix arms.

### The Ball-Ring sampler-iteration curve (10 seeds/arm, post-`ignore_effects`)

Archived at `docs/handoff/raw-results/samp1/`, and aggregated on `main` as
`docs/experiment-logs/2026-08-03-ballring-arms.json`.

| iters | mean | sd |
|---|---|---|
| 1000 | 83.0 | 22.1 |
| 3000 | 90.0 | 28.3 |
| 10000 | 99.0 | 3.2 |
| 30000 | 91.0 | 12.0 |
| 100000 | 89.0 | 16.0 |

**Do not describe this as an inverted U with 10000 "optimal"** — that claim was asserted
repeatedly in this session and is unsupported: **no pairwise difference in the grid
reaches p < 0.05** (3000 vs 10000 is p = 0.350). PR #35 exists specifically to retract it.
What *is* supported is the change #34 made: the old default of 1000 sat below the
`n_iter_no_change = 5000` floor, so early stopping could provably never fire. 10000 is a
defensible default, not a measured optimum.

Two further corrections worth carrying:

- **`iters3k`'s sd of 28.3 is one collapsed seed at 10%**, with nine at 90–100%. It is not
  spread; it is a single failure. Read the per-seed data, not the sd.
- **`iters10k` and `fix-results/envfix` are the same run.** 99.0 was at one point treated
  as two corroborating measurements. It is one.

---

## Hypotheses tested and REFUTED (do not re-tread)

Each died to a measurement, and most died the same way: the code genuinely differs and the
mechanism fires **zero times**.

| hypothesis | how it died |
|---|---|
| predicators' runs depressed by wall-clock planning timeouts | its own pickles record `num_solve_timeouts = num_solve_failures = num_execution_timeouts = num_execution_failures = 0` across all 260 sweeps |
| asymmetry in how many skills carry continuous parameters | predicators' Ball-Ring options declare no `Box` params; sampler wrappers match |
| predicators' plan-length gate at eval (`planning.py:1100`, `max_horizon = CFG.horizon`) | REAL but INERT — 0/70 plans exceed 8 at any competence; the goal admits no cheaper long detour |
| planning failure fatal for the reference, free retry for us | REAL but INERT — 0/335 eval planning calls fail in our port |
| our Ball-Ring is easier than predicators' | fixed jitter/navigation/grounding and measured: paired p = 0.59 (#30) |
| uncapped goal pursuit rehearses the eval task | goal pursuit takes 24.6% of the budget, not ~100%; capping moves the endpoint +1 task the *wrong* way |
| the reference run was misconfigured vs the paper | audited its recorded `config`: clean |
| Ball-Ring environment stochasticity differs | audited: clean |

---

## Tossing Room: the three-stage story

The archived arms make the progression legible. All three read from
`docs/handoff/raw-results/`, unpracticed = the evaluation at 0 transitions:

| stage | arms | unpracticed | trained |
|---|---|---|---|
| operator fixes only (#28) | `tossingroom/tossingroom-*` (2026-08-02 12:44–15:00) | 62.3% | 99.0% |
| \+ horizon fix and throw-release (**PR #31**) | `results-release/*` (2026-08-02 20:25) | **38.7%** | 95.0% |
| random-skills floor | `results-release/random` | 3.3% | 3.7% |

The point of PR #31: an *unpracticed* policy scored 94.7% purely by re-rolling a ~19%
throw inside a 16-step horizon. Correcting the horizon to `longest_shortest_solve() + 2`
(= 7) and making a missed throw *release* the item removes the free retry, and the
unpracticed rate falls to 38.7% — which is what a learning curve needs in order to have
anywhere to go.

**New from the GIF work, and sharper than the PR originally claimed:** for the RECYCLING
family a missed throw is **terminal at any horizon**, not merely expensive. The pile is in
room 3, the recycling bin in room 1, and `blocked_right_from = 2` makes room 3 unreachable
from room 1 — so once the item is gone Fast Downward correctly reports *no plan*. The
`recycling-untrained` GIF shows exactly this: throw ≈ 0.38 against a 0.988 target, then
three frames of `no-op (no plan)`.

**`--goal-type` pins TRAINING tasks too.** A single-family run is therefore a *different
experiment* from the shipped arms — at seed 0 it scores 2/30 with sweeps flipping between
30/30 and 2/30. Do not use it to reproduce or demo the arms; select a seed whose first
test task is the family you want instead.

---

## Open work, all pushed

| branch | contents | measured |
|---|---|---|
| `josh/feature/tossingroom-ees-bringup` | **PR #31**, head `c32be6e` | unpracticed 94.7% → 38.7%; EES 95.0% vs random 3.7% |
| `josh/feature/ees-decouple-sampler-data` | split ε-exploration from sampler-data recording | null on the metric, but fixes a real confound |
| `josh/feature/ees-goal-pursuit-interval` | predicators' `pursue_goal_interval` | null (−2, p = 0.44) |
| `josh/feature/ees-planning-progress-scoring` | fixed task prefix, growing normalizer, replan deque | null (measured at ceiling) |
| `josh/wip/operator-fidelity-test-orig` | pre-merge copy of #29's test | already merged |
| `josh/wip/fix-measure-base` | the shared base the four fix worktrees were built on | historical |

All three `ees-*` branches put their changes behind flags defaulting to current behaviour,
so they are safe to land; **none is demonstrated to help.**

## Next steps, ranked

1. **Review/merge PR #31.** It is the only open PR and it is green.
2. **Decide on the three `ees-*` null branches.** `decouple` is worth landing on
   correctness grounds independent of its null result: one flag currently gates both
   ε-exploration *and* whether the sampler records data, which silently confounded two
   earlier ablations.
3. **Housekeeping**, all still broken: ~19 scratch worktrees under `.claude/worktrees/`
   and `../{combos,fix-results,iters-results,samp1,*-wt,*-pr}`; the `lint-imports`
   pre-commit hook fails on PATH; `FD_EXEC_PATH` does not resolve from a worktree.
4. The residual after the hyperparameter correction is ~2 points and inside noise. Further
   gap-hunting is probably not worth the compute. If resumed, the untested channel is the
   *other* learned samplers (`NavigateTo*`, `PlaceBallInCupOnTable`).

---

## Archived raw results

`docs/handoff/raw-results/<root>/<arm>/<method>/<seed>/stats.json` — 350 files, every arm
ever run in this project. `docs/handoff/raw-results-index.json` is the same data
summarised per arm, machine-readable.

<details>
<summary>Every arm, summarised</summary>

| arm | seeds | sweeps | final transitions | unpracticed % | final % | sd | recorded |
|---|---|---|---|---|---|---|---|
| `fix-results/combo_decouple/ees` | 10 | 26 | 2500 | 0.0 | **96.0** | 7.0 | 2026-08-01 07:59 |
| `fix-results/combo_goal/ees` | 10 | 26 | 2500 | 0.0 | **96.0** | 7.0 | 2026-08-01 07:54 |
| `fix-results/combo_scoring/ees` | 10 | 26 | 2500 | 0.0 | **100.0** | 0.0 | 2026-08-01 08:06 |
| `fix-results/control/ees` | 10 | 26 | 2500 | 0.0 | **67.0** | 24.5 | 2026-08-01 07:15 |
| `fix-results/decouple/ees` | 10 | 26 | 2500 | 0.0 | **70.0** | 29.4 | 2026-08-01 07:38 |
| `fix-results/envfix/ees` | 10 | 26 | 2500 | 0.0 | **99.0** | 3.2 | 2026-08-01 09:34 |
| `fix-results/goal_interval/ees` | 10 | 26 | 2500 | 0.0 | **20.0** | 33.3 | 2026-08-01 07:29 |
| `fix-results/ignore_effects/ees` | 10 | 26 | 2500 | 0.0 | **98.0** | 4.2 | 2026-08-01 07:20 |
| `fix-results/scoring/ees` | 10 | 26 | 2500 | 0.0 | **78.0** | 34.3 | 2026-08-01 07:48 |
| `iters-results/ballring/1000/ees` | 10 | 26 | 2500 | 0.0 | **34.0** | 35.7 | 2026-07-31 18:38 |
| `iters-results/ballring/10000/ees` | 10 | 26 | 2500 | 0.0 | **67.0** | 24.5 | 2026-07-31 18:47 |
| `iters-results/ballring/100000/ees` | 10 | 26 | 2500 | 0.0 | **53.0** | 25.4 | 2026-07-31 19:03 |
| `iters-results/lightswitch/1000/ees` | 10 | 11 | 1500 | 0.0 | **100.0** | 0.0 | 2026-07-29 05:00 |
| `iters-results/lightswitch/10000/ees` | 10 | 11 | 1500 | 0.0 | **100.0** | 0.0 | 2026-07-29 05:03 |
| `iters-results/lightswitch/100000/ees` | 10 | 11 | 1500 | 0.0 | **99.0** | 3.2 | 2026-07-31 18:30 |
| `results-release/ees1000/ees` | 10 | 26 | 2500 | 38.7 | **93.3** | 14.8 | 2026-08-02 20:25 |
| `results-release/ees10000/ees` | 10 | 26 | 2500 | 38.7 | **95.0** | 5.0 | 2026-08-02 20:25 |
| `results-release/ees100000/ees` | 10 | 26 | 2500 | 38.7 | **94.0** | 6.0 | 2026-08-02 20:25 |
| `results-release/random/random-skills` | 10 | 26 | 2500 | 3.3 | **3.7** | 4.0 | 2026-08-02 20:25 |
| `repo-results/ees-double-observe/ees` | 10 | 11 | 1500 | 0.0 | **100.0** | 0.0 | 2026-07-21 18:30 |
| `repo-results/ees-grid100/ees` | 10 | 11 | 1500 | 0.0 | **98.0** | 6.3 | 2026-07-21 18:22 |
| `repo-results/ees/ees` | 10 | 11 | 1500 | 0.0 | **100.0** | 0.0 | 2026-07-21 18:30 |
| `repo-results/ees/random-skills` | 10 | 11 | 1500 | 0.0 | **0.0** | 0.0 | 2026-07-21 18:11 |
| `repo-results/ees/skill-oracle` | 10 | 1 | 0 | 100.0 | **100.0** | 0.0 | 2026-07-21 18:11 |
| `samp1/iters100k/ees` | 10 | 26 | 2500 | 0.0 | **89.0** | 16.0 | 2026-08-02 16:51 |
| `samp1/iters10k/ees` | 10 | 26 | 2500 | 0.0 | **99.0** | 3.2 | 2026-08-02 16:37 |
| `samp1/iters1k/ees` | 10 | 26 | 2500 | 0.0 | **83.0** | 22.1 | 2026-08-02 16:56 |
| `samp1/iters30k/ees` | 10 | 26 | 2500 | 0.0 | **91.0** | 12.0 | 2026-08-02 17:09 |
| `samp1/iters3k/ees` | 10 | 26 | 2500 | 0.0 | **90.0** | 28.3 | 2026-08-02 17:01 |
| `target-history/ees-baseline/ees` | 10 | 11 | 1500 | 0.0 | **100.0** | 0.0 | 2026-07-24 05:48 |
| `target-history/ees-target-history/ees` | 10 | 11 | 1500 | 0.0 | **100.0** | 0.0 | 2026-07-24 05:50 |
| `tossingroom/tossingroom-1000/ees` | 10 | 11 | 1500 | 62.3 | **99.0** | 3.2 | 2026-08-02 12:44 |
| `tossingroom/tossingroom-10000/ees` | 10 | 11 | 1500 | 62.3 | **99.0** | 3.2 | 2026-08-02 13:33 |
| `tossingroom/tossingroom-100000/ees` | 10 | 11 | 1500 | 62.3 | **98.3** | 4.2 | 2026-08-02 14:58 |
| `tossingroom/tossingroom-random/random-skills` | 10 | 11 | 1500 | 3.7 | **6.7** | 2.7 | 2026-08-02 15:00 |

</details>

| root | what it measured | where it lived |
|---|---|---|
| `repo-results/` | the earliest Light Switch EES runs + oracle/random baselines | untracked dir inside the repo |
| `target-history/` | the #23 `skip_perfect`/UCB all-attempts ablation | agent worktree (auto-cleaned) |
| `iters-results/` | the **pre-`ignore_effects`** two-domain iteration grid | `../iters-results` |
| `fix-results/` | the four parallel fix worktrees + their pairwise combos, vs `control` | `../fix-results` |
| `samp1/` | the **post-fix** 5-arm Ball-Ring iteration sweep (the table above) | `../samp1` |
| `tossingroom/` | Tossing Room after #28, before PR #31's horizon/throw fix | agent worktree |
| `results-release/` | Tossing Room release arms — **backs PR #31 and its GIFs** | agent worktree |

The repo-internal root is archived as `repo-results/`, not `results/`, because
`.gitignore:219` is a bare `results/` — which matches a directory of that name **at any
depth**, so `git add` on `docs/handoff/raw-results/results/` silently no-ops and you push
an empty archive. It did, until the staged file count was checked.

Two traps in reading these:

- `iters-results/ballring/10000` is **67.0 ± 24.5**, not 99.0. It predates
  `ignore_effects`, so every evaluation plan in it was structurally unexecutable. It is
  not comparable to `samp1/iters10k`.
- Only `stats.json` was archived. The `episode*.mp4` files (~4 MB per root) were not; the
  five that matter are committed as GIFs on PR #31.

---

## Method lessons worth keeping

- **Small samples lied twice.** A single Ball-Ring seed read 60% where the 10-seed mean was
  6%; a 3-task Tossing Room smoke test read 100% where the true rate was 10%.
- **Three self-justifying code comments turned out to be false.** Treat "deliberately",
  "matches predicators", and "no planner consumes these yet" as hypotheses, not
  documentation.
- **A read-only audit passed over two real bugs.** A field-by-field operator diff cannot
  detect an entire effect *class* missing from the representation; only a behavioural
  check found `ignore_effects`. That is what #29 now guards.
- **A green assertion that can never fire is worse than no assertion.** One test compared
  room objects with `is`, but `get_rooms()` rebuilds `Object`s per call with value-based
  equality — so it passed vacuously against unfixed code.
- **An unpaired t-test was applied to a paired design once**, and normal quantiles were
  used for a Welch test once. Same seeds across arms ⇒ paired.
- **Verify commits server-side, not in the working tree.** A `git commit` silently rejected
  by the pre-commit hook was "verified" with `grep` against the working tree; four agent
  branches were then built on the wrong base.
- **Check that the code you think you are running is the code you are running.** An escaped
  `PYTHONPATH` in a heredoc made two "different" confirmation arms byte-identical. Print
  `hitl_pmp.__file__` at the top of any comparison run.

## Other practical notes

- `ruff format` also formats Python code blocks **inside Markdown**, and CI runs
  `ruff format --check .` — a stray double space in a snippet inside an experiment log
  will fail lint.
- The `lint-imports` pre-commit hook is broken on PATH. Commit with `--no-verify` and run
  `lint-imports` directly.
- zsh applies the `:r` history modifier inside `$var:refs/...`; use `${var}:refs/...` when
  pushing to an explicit refspec.
- `grep -c` prints `0` *and* exits non-zero, so `grep -c ... || echo 0` emits `"0\n0"`.

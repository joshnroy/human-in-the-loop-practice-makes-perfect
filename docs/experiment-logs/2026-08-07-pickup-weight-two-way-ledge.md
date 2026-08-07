# Pre-registration: the pickup-weight fork under the two-way ledge

> **Environment renamed (2026-08-07).** The domain these runs were made on was
> called `tossingroomsplitpickupweight` at the time, and every command below names
> it that way. It is now registered as **`tossingroom`**, having taken over the name
> of a retired fork; the three superseded forks were deleted in the same stack. The
> domain itself is unchanged, so **every number below still reproduces** -- but the
> commands need `--env tossingroom` to run against current code. Nothing here has
> been edited, restated or recomputed.

**Status: pre-registration (commit `8cb89ec`) plus results, added in a later commit.**
Everything from "Question / goal" down to "Disclosure of prior observations" is the
pre-registration exactly as committed before any sweep ran, unedited. The results follow
it, under "Results".

## Question / goal

Three PRs have built a 2x2 and filled three of its cells. Two independent modifications
to reset-free practice on Tossing Room have each been measured, on different domains,
**never combined**:

- **unfreezing the sampler's task inputs** (`tossingroomsplitpickupweight`, PR #121/#122)
  — the item's weight is drawn at pickup rather than frozen into the task's initial
  state, so a reset-free run's sampler inputs keep varying without a `reset_to_task`.
- **removing irreversibility** (`--two-way-ledge`, PR #124/#125) — the one-way ledge
  becomes an ordinary corridor, so practice can no longer strand itself away from the
  only item source.

Final evaluation scores, 10 seeds, `x/300` (30 test tasks x 10 cycles):

| | frozen task inputs (`tossingroomsplit`) | weight drawn at pickup (`tossingroomsplitpickupweight`) |
|---|---|---|
| **one-way ledge** (irreversible) | `never` 85/300 · `scheduled` 151/300 | `never` 112/300 · `scheduled` 183/300 |
| **two-way ledge** (recoverable) | `never` 144/300 · `scheduled` 276/300 | **this experiment** |

This experiment runs the empty cell: `tossingroomsplitpickupweight` with
`--two-way-ledge`, arms `scheduled` and `never`, 10 fixed seeds.

## Background

PR #115 measured reset-free practice as worse than scheduled-reset practice on
`tossingroomsplit` (85/300 vs 151/300) and attributed it to **two entangled mechanisms**:

1. **Stranding.** The ledge blocks the rightward step out of room 2. The pile in room 3
   is the only item source, so rooms {0,1,2} are absorbing: a robot that walks left once
   can never practise any skill again for the rest of the run.
2. **Frozen sampler inputs.** `reset_to_task` is the only thing that installs a `Task`'s
   `initial_state`. Under `never` it is never called, so every per-task quantity the
   sampler reads as input stays at whatever the one `hard_reset` state set — the sampler
   trains on a single point of its input domain.

Each subsequent PR removed one mechanism and measured a partial gain on the `never` arm:
`+27` from unfreezing inputs (85 -> 112, one-way), `+59` from removing stranding
(85 -> 144, frozen inputs). **Neither closed the gap to `scheduled`.** This cell removes
both at once, and is the experiment that decides whether the two known mechanisms are
jointly sufficient to explain the reset-free penalty, or whether a third, so-far
unidentified mechanism is carrying it.

**The two-way ledge also makes the domain easier**, which is a property of the world and
not a method effect: EMPTY stops being an ordering task (solve 10 -> 9), the evaluation
horizon drops 12 -> 11, and RECYCLING stops being one-shot. A two-way number is therefore
never directly comparable to a one-way number. **Only the within-world gap between the two
reset policies is comparable across worlds**, which is why both arms are run here rather
than only `never`.

## Hypothesis

### The arithmetic of "additive", stated before looking

The two measured main effects on the `never` arm are `+27` (unfreezing) and `+59`
(two-way). A strictly additive model on the raw count scale predicts:

> **`never` = 85 + 27 + 59 = 171/300** under both mechanisms removed.

The same model on the `scheduled` arm predicts `151 + 32 + 125 = 308/300`, which is
**above the ceiling**. `scheduled` is already at 276/300 in the two-way frozen cell, so
the `scheduled` arm here is expected to be **ceiling-limited at roughly 280-300/300** and
carries little information about additivity. This is recorded now because it means:

> **"the effects are additive" and "the gap to `scheduled` closes" are NOT the same
> claim, and under the additive model they are in fact incompatible.** Additivity
> predicts `never` ~171/300 against a `scheduled` near ~290/300 — a gap of roughly 119,
> i.e. **wider** than the one-way frozen gap of 66. A reader expecting "additive =>
> the gap closes" should read this paragraph first.

Both questions are therefore pre-registered and reported **separately**.

### Primary prediction

**PARTIAL recovery, gap NOT closed.** `never` lands near 171/300 and remains
significantly below `scheduled`. Reasoning: the two mechanisms were each measured to
remove only part of the penalty, and nothing in either PR identified a mechanism that
would make their combination super-additive.

### Bimodality prediction (falsifiable, stated before looking)

`pickup-weight / never` in the one-way cell is **bimodal**: exactly 4/10 seeds finish at
16-21 and 6/10 at 5-7, matching the reported "6/10 seeds draw exactly one weight for the
whole run" seed-for-seed. That bimodality *is* stranding, visible in task outcomes — a
stranded run stops taking pickups, so its weight schedule stops advancing.

> **Prediction: under the two-way ledge the bimodality VANISHES.** No cluster of seeds at
> 5-7; the per-seed finals form one mode. On `tossingroomsplit` the flag took stranding
> from 74/100 to 0/100 cycles, so the mechanism generating the low mode should be gone.

The corresponding measured quantity is **`num_stranded_before_last_period`**, not raw
onset. PR #122's analysis established that onset cannot distinguish "stranded going into
the last period" from "took no pickup in the last period", and that on `scheduled` 5/10
seeds report a spurious onset at period 9.

> **Prediction: `num_stranded_before_last_period` = 0/10 seeds on both arms.**

## Decision rule

Let `never_2way_pw` be this cell's `never` score out of 300, with the MDE computed from
its own two denominators (below).

- **ADDITIVE** — `never_2way_pw` is within the MDE of the additive prediction 171/300
  (i.e. roughly 148-194/300), **and** stranding is 0/10 seeds, **and** the bimodality is
  gone. Reading: the two known mechanisms jointly account for the reset-free penalty on
  the count scale, and nothing further needs positing.
- **SOMETHING ELSE IS GOING ON** — `never_2way_pw` is **significantly below** 171/300
  (below ~148/300) while stranding is confirmed 0/10 and the sampler inputs are confirmed
  varying. Reading: with both known mechanisms removed the penalty persists, so the
  leading hypothesis (stranding was accidentally protective) is **not sufficient**, and a
  third mechanism none of #115/#121/#124 identified is carrying the residual.
- **SUPER-ADDITIVE / GAP CLOSES** — the paired test between the two arms of *this* cell
  returns p >= 0.05, i.e. no detectable difference between `scheduled` and `never`.
  Reading: the two mechanisms interact, and removing both is qualitatively different from
  removing either.
- **INDETERMINATE** — `never_2way_pw` sits above ~148/300 but the confidence interval
  spans both 171/300 and a materially different value, or the stranding/bimodality
  checks disagree with the score-level reading (e.g. stranding is 0/10 but the bimodality
  persists, which would mean the low mode was never stranding in the first place).

**The gap question is reported separately from the additivity question**, per the
arithmetic above. "Does removing both mechanisms close the gap to `scheduled`?" is
answered by the within-cell paired test, not by comparison to 171/300.

## Methods (planned)

**Settings are matched to the two source experiments by reading their committed
`config_snapshot.json`, not their prose.** This cell is the pickup-weight configuration
(PR #122's) with `two_way_ledge` set to `True` (PR #125's one differing key):

`--env tossingroomsplitpickupweight --method ees --num-cycles 10
--max-steps-per-interaction 150 --num-test-tasks 30 --exploration-epsilon 0.5
--two-way-ledge`, seeds 0-9, arms `--practice-reset-policy scheduled|never`.

Every other resolved value is the domain default and is asserted equal to PR #122's
snapshot key-by-key, with `two_way_ledge` and `practice_reset_policy` as the only
intended differences. That assertion is committed as a test.

`--two-way-ledge` does not currently exist on the pickup-weight fork — PR #124 scoped it
to `tossingroomsplit` only. Wiring it across is part of this task and is kept minimal and
separable: a mirror of #124's `ledge_blocks_rightward` helper plus the flag, with no
change to the fork's default behaviour (verified by `stats.json` byte-identity against a
banked default run).

Runs are driven by `scripts/run_sweep.py` with fixed seeds inside a memory-capped
`systemd-run --user --scope`. All 20 runs' `stats.json`, `config_snapshot.json` and
`timing.json` are committed.

## Analysis plan (fixed before results)

- **Counts as `x/300`**, never a bare percentage.
- **Per-comparison MDE from its own two denominators**, at
  `2.801585 * sqrt(p_bar * (1 - p_bar) * (1/n1 + 1/n2))`.
- **Paired tests.** The arms share a seed set, so the per-seed pairs are paired data; an
  unpaired test discards that structure. Exact sign-flip (permutation) test over the 10
  seed-level differences.
- **Interaction** against the existing three cells, on the `never` arm, reported as the
  observed value minus the additive prediction.
- **Stranding measured, not assumed**, via `num_stranded_before_last_period` read off
  #111's `practice_outcomes_per_cycle` by
  `analysis/practice_makes_perfect/pickup_weight_stranding.py`. Raw onset is not used.
- **Bimodality reported explicitly** as the per-seed finals, with the four-panel figure
  plotting faint per-seed lines under the bold mean so a mode split cannot be hidden by
  averaging.
- Existing cells are **re-extracted from their committed `stats.json` and checked against
  their published per-seed finals before a fourth panel is added**. If extraction does not
  reproduce the published vectors exactly, that is reported as a defect and the figure is
  not published.

## Disclosure of prior observations

Nothing from this cell has been run or inspected at the time of this commit — not a
probe, not a timing run. The wiring change is verified only against the fork's **default**
(one-way) behaviour, whose banked numbers are already published.

The two directionally-conflicting observations disclosed in PR #125's own
pre-registration remain the relevant priors and are not restated here.

---

# Results

**Everything above this line was committed before the sweep ran (`8cb89ec`) and has not
been edited.** No number published by PR #115, #122 or #125 is edited, restated or
recomputed anywhere below; the three existing cells are re-extracted only to check that
the extraction reproduces them, and it does, exactly.

## Headline

> **Removing both mechanisms closes the gap.** `never` 287/300 against `scheduled`
> 300/300 — a difference of 13/300 that sits entirely in one seed, with an exact paired
> sign-flip p-value of 1.0 (9/10 seeds have a difference of exactly zero). The
> pre-registered **SUPER-ADDITIVE / GAP CLOSES** branch of the decision rule is the one
> that fires.

The effects are **not** additive: additivity predicted `never` = 171/300 and the observed
value is 287/300, a deviation of `+116/300` against an MDE of `29.2/300` for that
contrast. That deviation is descriptive rather than a test, because 171/300 is a
prediction assembled from point estimates, not a measured arm.

## The completed 2x2

Final evaluation scores, 10 fixed seeds, `x/300` (30 test tasks x 10 seeds). The bottom
right cell is this experiment; the other three are re-extracted from committed data.

| variant | ledge | `scheduled` | `never` | within-world gap | MDE |
|---|---|---|---|---|---|
| `tossingroomsplit` | one-way | 151/300 | 85/300 | 66/300 | 33.5/300 |
| `tossingroomsplit` | two-way | 276/300 | 144/300 | 132/300 | 31.4/300 |
| `tossingroomsplitpickupweight` | one-way | 183/300 | 112/300 | 71/300 | 34.3/300 |
| **`tossingroomsplitpickupweight`** | **two-way** | **300/300** | **287/300** | **13/300** | **10.0/300** |

MDE is `2.801585 * sqrt(p_bar*(1-p_bar)*(1/n1 + 1/n2))`, computed per row from that row's
own two denominators.

**Extraction check.** All six previously-published per-seed vectors reproduce exactly
from the committed `stats.json`, and their sums reproduce the published `x/300`. The
fourth panel was only added after that check passed.

## The interaction, which is the surprising part

The two-way ledge does **opposite** things to the reset-free penalty in the two variants:

| variant | gap one-way | gap two-way | interaction |
|---|---|---|---|
| `tossingroomsplit` (frozen inputs) | 66/300 | 132/300 | **+66/300 — the gap WIDENS** |
| `tossingroomsplitpickupweight` (weight at pickup) | 71/300 | 13/300 | **-58/300 — the gap COLLAPSES** |

Three-way interaction: `-124/300`.

On the frozen-inputs domain, opening the ledge helped `scheduled` far more than `never`
(151 -> 276 against 85 -> 144), so the penalty got *bigger*. Only when the sampler's
inputs also keep varying does removing stranding let `never` catch up. **Neither
mechanism is the cause on its own; the penalty needs both present.** That is a stronger
statement than "the two fixes add up", and it is the opposite of what the pre-registration
predicted.

## Is the new cell bimodal? No.

Pre-registered prediction: the bimodality vanishes. **It does.**

| cell | per-seed finals | largest-gap split |
|---|---|---|
| pickup-weight / one-way / `never` | `[18, 16, 5, 6, 7, 6, 21, 20, 7, 6]` | 6/10 low `[5,6,6,6,7,7]`, 4/10 high `[16,18,20,21]`, gap 9 |
| pickup-weight / two-way / `never` | `[30, 30, 17, 30, 30, 30, 30, 30, 30, 30]` | 1/10 low `[17]`, 9/10 high `[30 x 9]`, gap 13 |

The one-way arm's 6/4 split at a gap of 9 is a genuine two-mode distribution. The two-way
arm is 9/10 seeds at the ceiling with a single low outlier — one tail, not a second mode.

## Stranding, measured rather than assumed

Read with `analysis/practice_makes_perfect/pickup_weight_stranding.py` off #111's
`practice_outcomes_per_cycle`, using **`num_stranded_before_last_period`**, not raw onset
(PR #122 established that onset cannot distinguish "stranded going into the last period"
from "took no pickup in the last period").

| arm | stranded seeds | ...before the last period | seeds drawing 1 weight | periods with pile access |
|---|---|---|---|---|
| two-way / `scheduled` | 0/10 | 0/10 | 0/10 | 10/10 every seed |
| two-way / `never` | 0/10 | 0/10 | 0/10 | 10/10 every seed |

Against 6/10 seeds drawing exactly one weight in the one-way pickup-weight cell. Both
mechanisms are confirmed removed by measurement, not by assumption: weight draws per seed
run 183-232 on the `never` arm.

## The one seed that is not at ceiling

Seed 2 of the `never` arm finishes 17/30 and carries the entire 13/300 difference. Its
per-family breakdown is **TRASH 1/14, RECYCLING 14/14, EMPTY 2/2**, and its practice
tallies are **1 trash pickup against 231 recycling pickups** for the whole run. So it is
a practice-*allocation* imbalance under reset-free exploration — the run essentially never
practised `ThrowTrash` — and specifically **not** stranding: it reached the pile in 10/10
periods and drew 232 weights.

Pooled per-family, `never` is TRASH 127/140, RECYCLING 140/140, EMPTY 20/20; `scheduled`
is 140/140, 140/140, 20/20.

## What this does NOT show

**The ceiling limits what can be concluded.** `scheduled` is at 300/300 — no headroom at
all. So this design cannot distinguish "reset-free practice is exactly as good as
scheduled-reset practice here" from "it is slightly worse, on a task set that has become
too easy to reveal it". The two-way pickup-weight world is the easiest of the four cells
by construction (EMPTY solve 9, horizon 11, RECYCLING repeatable), and both arms nearly
saturate it.

What is **not** a ceiling artifact is the `never` arm's own movement: 112/300 -> 287/300
between the one-way and two-way pickup-weight cells. That is a real gain on the arm that
had room to move.

A cross-world comparison of raw counts is never made here. The two-way world is an easier
domain, so only the within-world gaps are compared across worlds.

## The figure

The fourth panel is added to the **existing** cross-variant figure
(`analysis/practice_makes_perfect/reset_free_training_curves.py`, introduced on this stack
by #125) rather than to a second script of its own, so there is one cross-variant figure
rather than two near-identical ones. The layout becomes a 2x2 square, which is what the
four cells are: reading down a column holds the ledge fixed, across a row holds the
variant fixed.

```text
python analysis/practice_makes_perfect/reset_free_training_curves.py \
  --output docs/experiment-logs/2026-08-07-reset-free-four-variant-curves.png
```

It needs no aggregate: all 80 runs' `stats.json` are committed, so it regenerates from the
repository alone.

## Reproducing, and a rebase note

All 20 runs' `stats.json`, `config_snapshot.json` and `timing.json` are committed under
`docs/experiment-logs/2026-08-07-pickup-weight-two-way-ledge-runs/` (1,409,633 bytes, 60
files). Settings are asserted equal to PR #122's banked cells key-by-key, with
`two_way_ledge` and `output_dir` the only permitted differences, by
`tests/analysis/practice_makes_perfect/test_pickup_weight_two_way_ledge_runs.py`, which
also asserts the manipulation check on the committed data.

**These runs were re-run after this branch was rebased onto `main` @ `ebf3d92`, and the
reason is worth recording.** The sweep was first executed on the pre-rebase tree. `main`
had since gained #119, which changed `ees_method.py` and `wrapped_sampler.py`. That is
exactly the case CLAUDE.md says to check rather than assume, so it was checked by re-running
one seed on the post-rebase tree and comparing byte-for-byte:

* `evaluations` and `breakdowns` — **byte-identical**. Every number reported above is
  unaffected, and #119's claim that EES's behaviour is untouched holds here.
* `practice_outcomes_per_cycle` — **differs**, which is #119 doing exactly what it said it
  did: it reshaped that diagnostic to split the fallback pool.

So the results were never stale, but the committed artifacts would not have reproduced
byte-for-byte on the branch they sit on. The full 20 runs were therefore re-run on the
post-rebase tree and those are what is committed, so a reader re-running gets no diff.

### The default-off equivalence claim, stated precisely

The wiring commit (`--two-way-ledge` ported to this fork) claims a **byte-identical**
default (one-way) run against PR #122's banked `never/0/stats.json`, sha256
`bd8a632a2d97207eeded37b97339a7ccd40455b25cba018314dae03cf5f847b9`. **That was measured on
the pre-rebase tree and is true there, but it does not reproduce verbatim on this branch's
current tree, and the difference is not the wiring.** Re-measured on the post-rebase tree:

| field | fresh default-off run vs #122's banked `never/0` |
|---|---|
| `evaluations` | **identical** |
| `breakdowns` | **identical** |
| `num_practice_resets` | identical |
| `planning_attempts_per_cycle`, `planning_failures_per_cycle` | identical |
| `practice_outcomes_per_cycle` | **differs** — #119 reshaped this diagnostic |

So the claim that matters — **the flag changes nothing when it is off** — holds on both
trees and is what a reader should check. The literal whole-file sha256 holds only against a
pre-#119 tree, because #122's banked file predates #119. Quoting the sha256 without the
tree it was taken on would send a reader to a failing `cmp` and a false alarm.

### Shared-scratchpad collision: audited, committed data unaffected

While this experiment ran, another agent's `sweep.sh` and this one's collided in a shared
session scratchpad; that agent's copy was overwritten by this one's and **this experiment's
script was then executed a second time by that agent**, writing into
`pw2way/repro-oneway-never/` and `pw2way/scheduled/`. Two managers writing one results root
is the #123 collision, so it was audited rather than assumed harmless. Recorded because the
audit is evidence, not because the outcome was in doubt:

* **The committed runs come from a different tree entirely.** They were re-run after the
  rebase into `pw2way-v2/`, which the collision never touched. Byte-comparison:
  **20/20 committed `stats.json` identical to `pw2way-v2`, 0/20 identical to the collided
  `pw2way`.** The collided tree contributed nothing to anything committed.
* **Both collided directories audit clean anyway** — every `config_snapshot.json` under
  them carries this experiment's own `--env tossingroomsplitpickupweight`, `ees`, 10 cycles,
  150 steps, 30 test tasks, `--exploration-epsilon 0.5`; no foreign `--env`, no duplicated
  seed directory, no non-zero exit, and every file parses.
* **The duplicate execution was harmless by construction**: it ran *this* script, so it
  re-ran the same seeds under the same flags, and a fixed seed fully determines a run. The
  collided repro artifact still hashes to exactly the published
  `bd8a632a2d97207eeded37b97339a7ccd40455b25cba018314dae03cf5f847b9`.
* **The published numbers were re-derived from the committed files after the audit** and
  reproduce exactly: `scheduled` 300/300, `never` 287/300, per-seed
  `[30, 30, 17, 30, 30, 30, 30, 30, 30, 30]`, resets 10 and 0.

**Unrelated pre-existing breakage on the base — observed here, fixed elsewhere.** While
this experiment was running, `tests/analysis/practice_makes_perfect/test_tossing3d_practice_diagnosis.py`
had 2 failing tests **on `main` @ `ebf3d92` itself**, verified by checking `main` out
directly with this branch's changes stashed: #119 tightened `SkillPracticeTally`'s
validator and #127 added fixtures that violate it, and the two were merged separately so
neither PR's own CI saw the combination.

**This is now resolved and the note is kept only for the record.** It was independently
found and fixed in **PR #128** (`7a8daa1`, "Fix two #127 test fixtures that #119's tally
validator rejects"), which `main` picked up; this branch was subsequently rebased onto it
and the full suite is green here. Nothing on this branch ever touched either file, and no
result above depends on it.

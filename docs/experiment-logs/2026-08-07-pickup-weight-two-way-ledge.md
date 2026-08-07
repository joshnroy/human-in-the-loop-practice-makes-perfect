# Pre-registration: the pickup-weight fork under the two-way ledge

**Status: PRE-REGISTRATION. Written and committed before any sweep ran.** Every number
below is a prediction or a threshold. No result appears in this file at this commit; the
results section is added in a later commit on the same branch.

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

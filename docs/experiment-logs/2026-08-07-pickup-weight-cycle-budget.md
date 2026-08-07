# Ten times the practice budget: is the reset-free arm starved, or stranded?

Domain `tossingroom` (split throws, weight drawn at pickup), method `ees`, 10 fixed seeds
(0-9), 30 test tasks (14 TRASH / 14 RECYCLING / 2 EMPTY). A cube of **cycle budget**
(`1x` = 10 cycles, `10x` = 100) x **ledge** (one-way / two-way) x **reset policy**
(`scheduled` / `never`), 80 runs in total: the 40 at `1x` are the merged sweeps, read back
from their committed artifacts rather than re-run; the 40 at `10x` are new and are
committed under `2026-08-07-pickup-weight-cycle-budget-10x-runs/<cell>/<seed>/`.

**Headline: the starvation hypothesis is falsified, and the mechanism is sharper than
"starved".** Ten times the budget did not close the one-way reset-free gap; it **widened**
it, from `71` to `190` tasks pooled (`scheduled - never`, exact paired sign-flip
p = 0.001953, widened on 10/10 seeds). The reason is visible in one number: the one-way
reset-free arm logs **207 effective practice attempts at 10x — the identical 207 it logged
at 1x**. The extra ninety cycles bought it exactly zero additional practice.

> **Addendum, added 2026-08-07 after this log was merged as #166. Two things below are now
> qualified, and neither changes the headline.**
>
> 1. The two per-run figures in [Cost](#cost) — `279` s and `955` s — are labelled "mean"
>    but are the **medians**. The original sentence is left exactly as published; the note
>    beside it gives both statistics.
> 2. Chasing that label surfaced the larger point: **the one-way `never` cell is two
>    populations, not one**, split deterministically by seed, so *no* single-number summary
>    of it describes any run. See
>    [Addendum: the one-way `never` cell is a mixture](#addendum-the-one-way-never-cell-is-a-mixture-of-two-populations),
>    which qualifies every pooled one-way `never` figure in the Results tables — `112/300`,
>    `107/300`, `70/140`, `17/140`.

## Question / goal

The merged 1x A/B attributes the reset-free arm's deficit to *starvation* rather than to an
inability to learn. That is a testable claim with an obvious test: if the arm simply does
not get enough practice attempts, giving it ten times the cycles should buy back most of
the gap. If the gap survives at 10x, starvation is not a sufficient explanation.

## Background

`--practice-reset-policy scheduled` puts the environment back to a freshly-sampled train
task at the top of each practice period. `never` does not, so practice state runs
continuously across period boundaries — the real-robot condition, since a robot practising
in a lab is not teleported to a fresh start every few minutes.

Under the one-way ledge the merged 1x A/B measured `scheduled` **183/300** against `never`
**112/300**. Per family, 68 of the 71 tasks in that gap were TRASH; RECYCLING sat near the
floor in both arms (`25/140` against `22/140`, p = 0.9219 — a null result). The mechanism
proposed there was supply, not capability: the reset-free arm logged **207** effective
practice attempts against **1191**, with **85/100** cycles attempting not one. "Effective"
means a skill that needs the robot at the item pile (`Pickup*`, `Throw*`); a stranded robot
can still walk and press buttons for a whole period, so counting those would report a
starved arm as busy.

There is direct precedent for a budget increase rescuing an arm on this project: on an
earlier Tossing Room variant `ThrowRecycling` went from `11/56` to `901/982` at 10x, which
is what established starved-not-unable there.

**The one-way ledge is what makes stranding possible.** It blocks the *directed* edge from
room 2 to room 3, the only edge from rooms {0, 1, 2} into {3...6}, and the item pile — the
sole source of items — is in room 3. Rooms 0-2 are absorbing at any horizon, and no
recovery machinery exists. `--two-way-ledge` makes that edge traversable rightward too,
removing the domain's only irreversible action, which is why it is carried here as the
control condition.

## Hypothesis

**The reset-free arm is starved, not unable to learn, so the gap will largely close at
10x.** Concretely: the one-way `scheduled - never` gap shrinks substantially from its
1x value of 71 tasks, and the reset-free arm's effective-attempt count rises roughly in
proportion to the budget.

## Guidance given

- 10x the cycles on the same 2x2 the 1x result used, 10 fixed seeds per cell, 40 runs;
  change **only** the cycle count, reading the exact 1x conditions out of the committed
  `config_snapshot.json` files rather than reconstructing them from prose.
- Plot 1x against 10x directly, in one figure, with per-seed spread rather than only means.
- Keep the per-family TRASH / RECYCLING split, since the 1x gap was almost entirely TRASH
  and RECYCLING was a null result — whether RECYCLING moves at 10x is a separate question
  from whether the overall gap closes.
- Paired tests throughout (the arms share seeds); counts as `x/y` everywhere.

## Methods

**The only thing that differs between a `1x` cell and its `10x` twin is `--num-cycles`.**
The commands were built by reading the *resolved* argparse namespace out of the committed
`config_snapshot.json` files. The resulting 10x snapshots differ from the 1x ones in
exactly four keys: `num_cycles` (the manipulation), `env` (the domain was renamed
`tossingroomsplitpickupweight` -> `tossingroom` by #142, a rename #146 established is
byte-neutral), and `two_way_ledge` / `unsplit_skills` — two flags that did not exist when
the 1x runs were made, and whose defaults (`False`, `False`) are exactly the behaviour
those runs had.

Runs were driven by `scripts/run_sweep.py`, four sweeps of ten fixed seeds, inside a
memory-capped systemd unit.

### Checks, before any inference

**The 1x baseline is unchanged by the artifact regeneration.** #146 regenerated and #155
re-laid the pickup-weight artifacts this log reads its 1x column from. Re-deriving all four
1x cells after that rebase returns the published numbers to the digit — `183/300`,
`112/300`, `300/300`, `287/300`, effective attempts `1191` against `207`, starved cycles
`85/100`. Nothing in the 1x column moved.

**The 1x run is a strict prefix of its 10x twin, on 40/40 runs.** Same seed and same flags
but `--num-cycles`, so the first 1500 transitions should be the identical trajectory. They
are: the 10x arm's first 11 checkpoints reproduce the 1x arm's 11 checkpoints exactly, for
both the solved counts and the cumulative effective-attempt curve, in **10/10** seeds of
each of the four cells. This is the check that licenses putting the two budgets on one
axis at all.

**27/27 byte-identical on an accidental re-run.** Three cells lost their seed-9 run to an
external process kill and were re-run in full (there is no per-seed selection in
`run_sweep`). The 27 runs that completed twice produced byte-identical `stats.json` both
times, which is the reproducibility guarantee `--seed` is supposed to give.

**The rebase is a replay, not a re-run — checked by re-execution, not argued.** The 40 runs
were made at `2d69cf2`; the branch was then rebased forward. That delta touches `src/`,
including `ees_method.py` and `metrics.py`, which is exactly the case CLAUDE.md says makes a
rebase a *re-run* rather than a replay — so it was tested instead of reasoned about. The
changes add practice-*target* tallies: a new `stats.json` field plus counter calls placed
beside decisions the surrounding code had already made, with no branch, RNG draw or planner
call depending on them. Two seeds were therefore re-executed on the rebased code, one from
the cheap reset-free cell and one from the `scheduled` cell (which exercises far more of
`choose_practice_target`, where the counters were added). In both, **3/3** of the fields
this analysis reads — `breakdowns`, `num_practice_resets`, `practice_outcomes_per_cycle` —
are identical, and the only difference is the added
`practice_target_outcomes_per_cycle` key. The committed artifacts predate that key; no
number here depends on it. Later rebases moved only `analysis/` docstrings, a
`kinder_backend.py` docstring and `scripts/update_reference_repos.sh`, none of which any
Tossing Room run invokes.

**Manipulation.** `num_practice_resets` is measured out of each run's own `stats.json`
rather than restated from the flag, and is checked against the cell's budget: 100 in 20/20
10x `scheduled` runs, 0 in 20/20 10x `never` runs.

**Composition.** Every evaluation sweep of every run realises the domain's 14 TRASH /
14 RECYCLING / 2 EMPTY split; a goal misfiled between families would move tasks between
denominators invisibly, so it raises rather than being tolerated.

## Results

### The practice budget actually spent

| cell | effective attempts pooled | cycles attempting none |
| --- | --- | --- |
| 1x one-way `scheduled` | 1191 | 23/100 |
| 1x one-way `never` | 207 | 85/100 |
| 10x one-way `scheduled` | 4361 | 211/1000 |
| 10x one-way `never` | **207** | **985/1000** |
| 1x two-way `scheduled` | 4021 | 0/100 |
| 1x two-way `never` | 3982 | 0/100 |
| 10x two-way `scheduled` | 39212 | 0/1000 |
| 10x two-way `never` | 38992 | 0/1000 |

**This is the whole result.** Under the one-way ledge, the reset-free arm's effective
attempts are `207` at both budgets — not "fewer than the control", but *the same absolute
number*, because it takes its last effective attempt in cycle 1, 2 or 3 and then never
again. Per seed, the last effective practice attempt lands in cycle 1 for 6/10 seeds
(2 attempts each), cycle 2 for 3/10 (40 each) and cycle 3 for 1/10 (75). The remaining
97-or-so cycles of every seed are spent entirely walking. Under the two-way ledge, where
stranding is impossible, both arms scale with the budget as expected (roughly 10x the
attempts for 10x the cycles) and neither starves a single cycle.

### Final-checkpoint scores and the within-ledge gap

| ledge | budget | `scheduled` | `never` | gap | never worse on | exact paired sign-flip |
| --- | --- | --- | --- | --- | --- | --- |
| one-way | 1x | 183/300 | 112/300 | 71 | 8/10 seeds | p = 0.01172 |
| one-way | 10x | 297/300 | 107/300 | **190** | 10/10 seeds | p = 0.001953 |
| two-way | 1x | 300/300 | 287/300 | 13 | 1/10 seeds | p = 1 |
| two-way | 10x | 300/300 | 300/300 | 0 | 0/10 seeds | p = 1 |

The change in the one-way gap is `-119` pooled — it **widened** — and it widened on
**10/10** seeds (exact paired sign-flip p = 0.001953). The two-way gap moved from 13 to 0,
but that is **1/10** seeds moving with **9/10** tied, p = 1: a **null result** on the
change, and the minimum per-seed change this design had 80% power to detect is 3.64 tasks.
The two-way 10x cells are nonetheless a clean 300/300 in both arms.

> **Note added 2026-08-07, after merge.** The one-way `never` column above (`112/300` and
> `107/300`) pools two non-overlapping groups of seeds and is a **mixture, not a
> performance level** — see
> [Addendum: the one-way `never` cell is a mixture](#addendum-the-one-way-never-cell-is-a-mixture-of-two-populations).
> The gaps and their p-values are unaffected: every comparison here is paired within a seed.

### Per family, under the one-way ledge

| family | budget | `scheduled` | `never` | gap | exact paired sign-flip on the gap |
| --- | --- | --- | --- | --- | --- |
| TRASH | 1x | 138/140 | 70/140 | 68 | p = 0.01562 |
| TRASH | 10x | 139/140 | 70/140 | 69 | p = 0.02344 |
| RECYCLING | 1x | 25/140 | 22/140 | 3 | p = 0.9219 |
| RECYCLING | 10x | 138/140 | 17/140 | **121** | p = 0.001953 |
| EMPTY | 1x | 20/20 | 20/20 | 0 | p = 1 |
| EMPTY | 10x | 20/20 | 20/20 | 0 | p = 1 |

**TRASH is where the 1x gap lived, and it does not move at all**: 68 -> 69, a **null
result** on the change (p = 1, closed on 4/10 seeds, MDE 1.89 tasks per seed). Both arms
are already at their ceiling by 1x — `scheduled` at 138/140, `never` stuck at exactly
70/140 at both budgets.

**RECYCLING is where the extra budget went, and only for the arm that could use it.**
`scheduled` goes `25/140` -> `138/140`: the 1x RECYCLING null result was a *power* problem
after all, and the precedent shape (`11/56` -> `901/982`) reproduces — but for the
scheduled-reset arm. The reset-free arm goes `22/140` -> `17/140`, i.e. nowhere. So the
family that carried the 1x gap (TRASH) and the family that carries the 10x gap (RECYCLING)
are different families, which a pooled number alone would hide.

EMPTY is 20/20 in all four one-way cells at both budgets. It is 2 tasks per seed and its
denominator supports almost no inference; it is reported for completeness, not as a result.

### Addendum: the one-way `never` cell is a mixture of two populations

*Added 2026-08-07, after this log was merged as #166. Nothing above is edited. This
section adds a decomposition; it recomputes none of the published numbers, and the
headline — ten times the budget bought zero additional practice — is unaffected.*

**What prompted it.** The [Cost](#cost) paragraph quoted a per-run figure for the one-way
`never` cell that turned out to be a median labelled a mean. The two statistics diverge
there (`279` s against `418` s) and essentially nowhere else, and that asymmetry is not an
arithmetic curiosity: it is the signature of a cell whose runs fall into two separated
groups.

**The split, per seed.** Under the one-way ledge with no practice resets, `10/10` seeds
strand — but not at the same time, and *when* a seed strands is fixed by the seed. Six
seeds take their last effective practice attempt in cycle 1 and two attempts in total;
four take theirs in cycle 2 or 3, and 40 to 75 attempts. Those are the same six and the
same four at both budgets:

| stranding cycle | seeds | effective attempts, 1x | effective attempts, 10x | wall clock, 1x | wall clock, 10x |
| --- | --- | --- | --- | --- | --- |
| cycle 1 | 2, 3, 4, 5, 8, 9 (6/10) | 2 each | 2 each | 24.9-25.8 s | 254.4-285.8 s |
| cycle 2 | 0, 1, 6 (3/10) | 40 each | 40 each | 57.3-61.8 s | 595.4-658.8 s |
| cycle 3 | 7 (1/10) | 75 | 75 | 62.0 s | 670.4 s |

The per-seed change in effective attempts between the two budgets is **exactly zero on
10/10 seeds** (exact paired sign-flip p = 1, a **null result** in the strongest available
form — every difference is `0`, not merely small). So the published `207` is not an
aggregate that happens to repeat; it is `2, 2, 2, 2, 2, 2, 40, 40, 40, 75` reproducing
seed-for-seed across a tenfold change of budget.

**The wall-clock partition is the stranding partition.** Partitioning the 10x cell at its
widest wall-clock gap (309.6 s, between 285.8 s and 595.4 s) puts seeds `{0, 1, 6, 7}`
above it. Partitioning it instead by last effective practice cycle — a quantity read out
of `stats.json`, with `timing.json` never consulted — puts the same `{0, 1, 6, 7}` on the
late side. The two partitions agree on **10/10** seeds, two-sided Fisher exact
p = 0.00476. The 1x sweep gives the same partition at its own gap (31.5 s, between 25.8 s
and 57.3 s), also p = 0.00476.

**That p is the floor, not a measure of size.** With a 6/4 margin there are
`C(10, 4) = 210` labellings and only the single most extreme table clears the observed
probability, so `1/210 = 0.00476` is the smallest p this design can return. Perfect
agreement on ten seeds is a real observation and a small one.

**Concurrency does not explain it.** These runs shared a machine, so wall clock is not a
pure measure of work. At 1x the control is exact rather than statistical: all ten runs
launched **in the same second** at `--max-workers 10`, against an identical 1-minute load
average of `6.32` and `0` other `hitl_pmp.cli` processes — so no per-seed difference in
starting conditions exists there at all, and the same 4/10 still came out slow. At 10x,
where the runs were staggered, none of the six recorded concurrency covariates separates
the modes (exact permutation, smallest p = 0.15 for load average at start).

Two 1x covariates measured **at run end** do separate (`p = 0.005` for both the
machine-wide process count and the load average). That is reverse causation, not a
confound: a run still going when the fast six have finished necessarily observes a
different machine state at its end. The at-*start* covariates are the ones that could
cause anything, and those are identical.

**A residual this does not explain, stated as one.** The slow runs did not take more
steps. Total practice skill attempts are **14900 on 10/10 seeds** and online transitions
**15000 on 10/10 seeds** at 10x; only the *composition* differs. Nor does planner work
account for it: seed 6 logs **21475** planning attempts, *fewer* than every one of the six
fast seeds (`22692`-`22790`), yet runs 595.4 s against their 254-286 s. So effective
attempts identify the two groups cleanly, but the per-second mechanism converting them
into wall clock is **not established here**. Do not read the correlation as a cost model.

**What this qualifies.** Every pooled one-way `never` figure in the tables above —
`112/300` and `107/300` overall, `70/140` TRASH, `17/140` RECYCLING — is a **mixture of
two populations rather than a performance level**. At the final checkpoint the two modes
do not overlap on score in either budget: at 1x, `37/180` pooled over the six early-stranded
seeds against `75/120` over the four late-stranded ones, with the range `7/30` to `16/30`
empty; at 10x, `36/180` against `71/120`, with `12/30` to `17/30` empty. A mean over that
describes no seed. The *comparisons* the experiment draws are unaffected, because every
one of them is paired within a seed.

![The one-way reset-free cell is two populations](https://raw.githubusercontent.com/joshnroy/human-in-the-loop-practice-makes-perfect/386f53c222a04a815e50c780cddfd8979d9a9374/docs/experiment-logs/2026-08-07-pickup-weight-cycle-budget-wallclock-modes.png)

Per seed throughout, never an aggregate. **(a)** each seed's effective practice attempts at
1x joined to its own value at 10x — ten flat lines, the tenfold budget change buying
nothing on any seed. **(b)** wall clock against the cycle of the last effective attempt,
both budgets on one log axis, with each budget's widest gap drawn: the same 4/10 sit above
it in both. **(c)** final-checkpoint score per seed, with the empty band between the modes
shaded — this is why the pooled figure is a mixture. **(d)** the confound panel: wall clock
against the load each run actually started against, showing the 1x runs stacked at one
identical load and the 10x modes fully interleaved.

**The learning curves themselves, grouped by mode.** The split is not a late divergence:
the two groups separate at the **first** evaluation checkpoint after practice begins and
never re-converge, at either budget.

![Learning curves grouped by stranding mode](https://raw.githubusercontent.com/joshnroy/human-in-the-loop-practice-makes-perfect/386f53c222a04a815e50c780cddfd8979d9a9374/docs/experiment-logs/2026-08-07-pickup-weight-cycle-budget-mode-curves.png)

Faint per-seed traces under a bold per-group mean, one row per budget, and the two x axes
this project pairs for learning curves. **The two columns are the same curve rescaled**:
`150` online transitions per cycle exactly, on `40/40` runs, so "by transitions" carries no
information "by cycle" does not. The right column is drawn on a symlog axis for that
reason — the entire divergence is complete within three cycles of a hundred, which a
linear axis compresses into the leftmost few percent and hides.

**The early-stranded group does not measurably learn at all.** Its curve is flat for the
whole run at both budgets, and its final score is not an improvement on its score *before
any practice*: `50/180` at checkpoint 0 against `37/180` final at 1x and `36/180` at 10x.
The point estimate is slightly negative, but that is a **null result** in both budgets
(exact paired sign-flip p = 0.1250 at 1x and p = 0.3750 at 10x; 12 and 42 seeds
respectively would be needed for 80% power), so what is established is the *absence of
improvement*, not a decline. Across every checkpoint of every run, no early-stranded seed
ever exceeds `15/30`, while every late-stranded seed reaches at least `20/30`.

The late-stranded group does improve — `27/120` at checkpoint 0 to `75/120` at 1x and
`71/120` at 10x, rising on `4/4` seeds at both budgets. Its p = 0.1250 is the **floor** for
four seeds (`2/2**4`), not weak evidence: no four-seed comparison can return less.

**Watching it happen.** Two representative practice runs were recorded with
`--record-full-loop`, one from each group. The recorder is a pure observer — it draws from
no RNG, takes no action and decides nothing — so a recorded run takes the same actions as
an unrecorded one.

*Which seeds, and why.* By an explicit rule, not by eye:
`ResetFreeWallclockModes.representative_seed` takes **the seed whose final solved count is
closest to its group's median, ties broken by the lowest seed number.** Both groups have an
even number of seeds, so the median usually falls between two runs and the tie-break has to
be stated. That gives **seed 3** for stranded-in-cycle-1 (group finals `5, 6, 7, 6, 7, 6`,
median 6; seeds 3, 5 and 9 all sit on it) and **seed 0** for stranded-in-cycle-2-or-3
(group finals `18, 16, 21, 20`, median 19, which no run attains; seeds 0 and 7 are both one
away).

*These are not re-runs of the experiment.* Each recording is the same seed and the same
flags with `--num-cycles 3` instead of `10`, and each was checked to be a strict **prefix**
of its committed run rather than assumed to be: `evaluations`, `breakdowns` and
`practice_outcomes_per_cycle` are identical over the first 4 checkpoints for **2/2** seeds,
and the effective-attempt totals over 3 cycles (`2` and `40`) already equal the committed
10-cycle totals — which is the stranding restated. No committed number is recomputed here.

![Stranded against not stranded, same cycle and same step](https://raw.githubusercontent.com/joshnroy/human-in-the-loop-practice-makes-perfect/386f53c222a04a815e50c780cddfd8979d9a9374/docs/experiment-logs/2026-08-07-pickup-weight-stranding-contrast.png)

The same four moments of practice cycle 1 — steps 9, 49, 89 and 129 of 150 — from each
recording. The status bar the recorder draws carries the cycle and step, so the alignment
is checkable rather than asserted. Seed 3 is already west of the one-way ledge at step 9
and spends the rest of the period walking between rooms 0, 1 and 2; seed 0 is at the item
pile running `PickupTrash` at the same step, and is still working the pile-and-bin loop at
step 129. The red bar with the red `X` is the ledge: the only edge from rooms {0, 1, 2}
back into room 3, where the pile is.

The full recordings — every practice step, every evaluation episode and every reset, in
order — are committed beside this entry:

- [seed 3, stranded in cycle 1](https://raw.githubusercontent.com/joshnroy/human-in-the-loop-practice-makes-perfect/386f53c222a04a815e50c780cddfd8979d9a9374/docs/experiment-logs/2026-08-07-pickup-weight-stranded-seed3-cycle1.mp4) (2929 frames)
- [seed 0, stranded in cycle 2](https://raw.githubusercontent.com/joshnroy/human-in-the-loop-practice-makes-perfect/386f53c222a04a815e50c780cddfd8979d9a9374/docs/experiment-logs/2026-08-07-pickup-weight-stranded-seed0-cycle2.mp4) (2675 frames)

GitHub serves these as downloads rather than an inline player, which is why the montage
above exists.

Regenerate both figures and every number in this section with:

```bash
D=docs/experiment-logs
python -m analysis.practice_makes_perfect.reset_free_wallclock_modes \
  --budget "1x=$D/2026-08-07-pickup-weight-reset-free-runs/never/ees" \
  --budget "10x=$D/2026-08-07-pickup-weight-cycle-budget-10x-runs/oneway-never" \
  --output "$D/2026-08-07-pickup-weight-cycle-budget-wallclock-modes.png" \
  --curves-output "$D/2026-08-07-pickup-weight-cycle-budget-mode-curves.png"
```

And re-record either video with (seed 3 shown; seed 0 is the same but for `--seed`):

```bash
python -m hitl_pmp.cli --env tossingroom --method ees --seed 3 \
  --num-test-tasks 30 --num-rooms 7 --start-room 3 --recycling-bin-room 1 \
  --trash-bin-room 6 --blocked-right-from 2 --practice-reset-policy never \
  --num-cycles 3 --max-steps-per-interaction 150 \
  --output-dir /tmp/rec/seed3 \
  --record-full-loop "$D/2026-08-07-pickup-weight-stranded-seed3-cycle1.mp4"
```

### Figures

![Does ten times the budget close the reset-free gap?](2026-08-07-pickup-weight-cycle-budget-gap.png)

Per seed, the `scheduled - never` gap at 1x joined to the same seed's gap at 10x — one line
per seed, so a closing gap would be ten lines sloping toward zero. The one-way panels slope
the wrong way on 10/10 seeds. The two-way bottom row shows why per-seed plotting matters:
its entire mean movement is one seed going 13 -> 0 with nine tied at zero.

![1x against 10x learning curves](2026-08-07-pickup-weight-cycle-budget-curves.png)

All four (budget x policy) curves per panel on a shared online-transitions axis, bold
pooled mean over faint per-seed lines. **The 1x curves lie exactly underneath the 10x ones
over their first tenth** — that is the strict-prefix property above, not a plotting
artifact, and it is why only two lines per panel are visually distinguishable. The one-way
RECYCLING panel is the one to read: `scheduled` climbs from ~3 to 14 between 2000 and 6000
transitions while `never` stays flat for all 15000.

## Recommendation

**Stop describing the reset-free failure as starvation, and describe it as stranding.** The
two make opposite predictions about budget, and this experiment separates them cleanly: a
starved arm gets more practice when given more cycles, and this arm gets *precisely none*.
"More practice attempts would fix it" is now falsified for this domain; "it cannot obtain
practice attempts at all after cycle ~2" is what the data support.

Three concrete follow-ups, in the order I would run them:

1. **The reset-free arm needs a recovery mechanism, not a bigger budget.** This is the
   `humans/` gap: `Metrics.num_human_interventions()` reports nothing because intervention
   is not *representable* yet, and all three `Method.reset_environment` implementations
   return `False` without writing. A `HumanOracle` that can rescue a stranded robot at a
   cost is the experiment this result argues for, and the cost accounting is the point —
   the interesting number is how much human intervention buys back how much of the 190.
2. **Do not spend more compute on one-way reset-free cycle budgets.** The effective-attempt
   trace shows the marginal return is exactly zero after cycle 3, so any budget between 10x
   and infinity gives the same answer.
3. **The 10x scheduled result is independently worth banking**: RECYCLING `25/140` ->
   `138/140` means the merged 1x number understates what EES reaches on this domain, and
   any future baseline compared against the 1x RECYCLING figure is being compared against a
   budget-limited one.

## Reproducing this

Every `stats.json`, `config_snapshot.json` and `timing.json` for the 40 new runs is
committed under `2026-08-07-pickup-weight-cycle-budget-10x-runs/<cell>/<seed>/`, so every
figure and number above regenerates without re-running anything.

```bash
# The four 10x cells (seeds 0-9 are fixed, never drawn). --max-workers 3 because the box
# was carrying other agents' sweeps; concurrency does not affect results.
WORLD="--num-test-tasks 30 --num-rooms 7 --start-room 3 --recycling-bin-room 1 \
  --trash-bin-room 6 --blocked-right-from 2"
for cell in "oneway-scheduled|--practice-reset-policy scheduled" \
            "oneway-never|--practice-reset-policy never" \
            "twoway-scheduled|--practice-reset-policy scheduled --two-way-ledge" \
            "twoway-never|--practice-reset-policy never --two-way-ledge"; do
  python -m scripts.run_sweep --env tossingroom --methods ees --num-seeds 10 \
    --max-workers 3 --results-root "results/rf10x/${cell%%|*}" \
    --shared-args "$WORLD ${cell#*|}" \
    --method-args "ees=--num-cycles 100 --max-steps-per-interaction 150"
done

# Read the cube back. The 1x cells are the committed merged sweeps.
D=docs/experiment-logs
python -m analysis.practice_makes_perfect.reset_free_cycle_budget \
  --cell "1x:one-way:scheduled=$D/2026-08-07-pickup-weight-reset-free-runs/scheduled/ees" \
  --cell "1x:one-way:never=$D/2026-08-07-pickup-weight-reset-free-runs/never/ees" \
  --cell "1x:two-way:scheduled=$D/2026-08-07-pickup-weight-two-way-ledge-runs/scheduled/ees" \
  --cell "1x:two-way:never=$D/2026-08-07-pickup-weight-two-way-ledge-runs/never/ees" \
  --cell "10x:one-way:scheduled=$D/2026-08-07-pickup-weight-cycle-budget-10x-runs/oneway-scheduled" \
  --cell "10x:one-way:never=$D/2026-08-07-pickup-weight-cycle-budget-10x-runs/oneway-never" \
  --cell "10x:two-way:scheduled=$D/2026-08-07-pickup-weight-cycle-budget-10x-runs/twoway-scheduled" \
  --cell "10x:two-way:never=$D/2026-08-07-pickup-weight-cycle-budget-10x-runs/twoway-never" \
  --gap-output "$D/2026-08-07-pickup-weight-cycle-budget-gap.png" \
  --curves-output "$D/2026-08-07-pickup-weight-cycle-budget-curves.png"
```

### Cost

40 runs, 12 concurrent, **69 minutes** of wall clock against a pre-run estimate of 72-90
minutes at 8-10 workers. A further 75 minutes went to re-running three cells in full after
an external process kill took out their seed-9 run mid-flight; that recovery changed no
number (27/27 byte-identical) and is not part of the experiment's cost. Per-run mean was
279 s for one-way `never` — the stranded, and therefore cheap, cell — against 955 s for
one-way `scheduled`. The pre-run estimate assumed a uniform 12.97x scaling from the 1x
grid; the real scaling is strongly cell-dependent, because a stranded run does almost no
work no matter how many cycles it is given, which is itself the result.

> **Note added 2026-08-07, after merge: `279` and `955` above are the medians, not the
> means.** The sentence is left exactly as published. Recomputed from the committed
> `timing.json` files (`elapsed_seconds`, 10 runs per cell), the four 10x cells are:
>
> | cell | median | mean |
> | --- | --- | --- |
> | one-way `never` | 279.0 s | 417.7 s |
> | one-way `scheduled` | 954.8 s | 985.0 s |
> | two-way `never` | 1096.0 s | 1115.4 s |
> | two-way `scheduled` | 1114.9 s | 1118.9 s |
>
> This is a mislabelled statistic, not a wrong measurement: nothing else in this log reads
> these two figures, and the cost asymmetry the paragraph draws — a stranded run does far
> less work, so it costs far less — holds on either statistic (279 s against 955 s by
> median, 418 s against 985 s by mean).
>
> The two statistics diverge by 139 s in the one-way `never` cell and by at most 30 s in
> the other three, because that cell alone is **bimodal**: its per-seed wall clocks are
> 254.4, 256.9, 259.4, 267.5, 272.2, 285.8, 595.4, 655.9, 658.8 and 670.4 s — six runs and
> four runs, with nothing between 286 s and 595 s. That split is the stranding split, and
> it is what
> [Addendum: the one-way `never` cell is a mixture](#addendum-the-one-way-never-cell-is-a-mixture-of-two-populations)
> works through.

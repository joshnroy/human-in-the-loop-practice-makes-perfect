# The task-level crossover, at a budget whose raw data is committed: Tossing Room split throws at 60 cycles

**TL;DR.** *(written after the results; see Results below)*

## Question / goal

Two things, one run.

1. **Make the crossover region re-analysable.** PR #103 established that
   `ThrowRecycling`'s sampler is starved rather than broken, and that the task-level
   crossover is complete by roughly 4,000 online transitions. It committed **seven files
   and zero data** — no `stats.json`, no shards, no arms JSON — and the results root has
   since been cleaned up. A filesystem scan confirms nothing survives: every sweep
   directory on disk is `num_cycles=25`. So the finding cannot be re-analysed at all,
   which is what this run fixes.
2. **Draw the task-level training curve.** `TRASH` against `RECYCLING` over the whole
   budget, on one axes, with per-seed spread. The 250-cycle page has two figures and
   eleven tables, and **both figures are sampler-level**; its only task-level statements
   are a threshold list and an AUC in prose.

## Background

`tossingroomsplit` gives Tossing Room's two throws separate lifted skills (PR #70), so
`EesMethod` keys a `LearnedSkillSampler` per `skill_name` and each throw learns only from
its own attempts. The room then rations them unequally: trash is a retryable round trip
from the pile, while recycling sits behind a **one-way ledge**, and a throw always
releases the item — so reaching the recycling bin ends that period's chance of another
go. Recycling gets **0 or 1 attempts per practice period, never two**; PR #103 verified
that cap rather than assuming it (`0/2500` periods contained two).

At the old standard budget of **25 cycles x 100 steps = 2,500 transitions**, recycling's
endpoint was `70/140` against trash's `139/140`. That budget stops at the point recycling
is roughly halfway up its curve, which is why the original experiment read the asymmetry
as a property of the domain rather than of the budget.

**Why 60 cycles rather than 250.** From #103's own threshold crossings, recycling reaches
`126/140` at about 4,000 transitions. **60 cycles x 100 steps = 6,000 transitions** covers
that with margin for seeds slower than the pooled curve — and per-seed variance is
precisely what no surviving data can show. What this budget gives up is the final
`126/140 -> 140/140` tail, which happens somewhere between 4,000 and 25,000 and **cannot
be localised from committed data**.

## Hypothesis

**Registered before any count was computed**, with five of ten runs already written to
disk and no family number yet read from any of them.

> **The overlap should reproduce exactly.** `num_cycles` enters `PracticeLoop.run` only
> as the loop bound, so the first 60 cycles of a longer run are the whole of a 60-cycle
> run. This run therefore ought to reproduce #103's published counts everywhere the two
> overlap (0-6,000 transitions): the continuity point at 2,500 (`TRASH 139/140`,
> `RECYCLING 70/140`), and the four threshold crossings — `35/140` at 200 vs 1,400,
> `70/140` at 400 vs 2,500, `105/140` at 800 vs 3,500, `126/140` at 900 vs 4,000.
>
> **But it may legitimately not**, and that is the interesting outcome. #103's runs were
> collected at `db2589f`; this one is at `d647749`, and four merged PRs (#100, #102,
> #106, #111) touched the run path in between. If any of them moved an action choice or
> a random draw, the numbers differ for a real reason and the published ones are stale
> rather than wrong.
>
> **The crossover should complete inside the budget.** I expect `RECYCLING` to reach at
> least `126/140` by 6,000 transitions, and I do **not** expect the endpoint gap to close
> to zero — #103 needed far more than 6,000 for that.

## Guidance given

- **Commit the raw data — a first-class requirement, not a nicety**, since #103's
  omission is what forced this re-run. `stats.json`, `config_snapshot.json` and
  `timing.json` for all ten seeds. **No trace shards**: the point of #111 is that they
  are no longer needed.
- Fixed seeds 0-9 via `scripts/run_sweep.py`, never drawn, never a hand-rolled loop.
- **The box is shared.** Check the load, size `--max-workers` to what is free, and wrap
  the sweep in a memory-capped `systemd-run` scope — a kernel OOM takes down the whole
  session here.
- **One figure, and one Josh asked for verbatim:** *"one graph that's just the training
  curve of trash vs recycling over the whole timestep."* Per-seed spread, not just a
  mean. Counts as `x/y` in every label, caption and annotation.
- **Do not compare across budgets.** This run is 6,000 transitions; the published page is
  25,000. Its own Recommendation 4 warns against exactly that.
- **A disagreement with #103's published counts is a finding, not an error to smooth
  over.** Report it and add a marked staleness note; never edit, restate or recompute a
  published number.
- One hazard, found the same day: **`torch.manual_seed` pins initial weights but not
  reduction order**, so the ambient intra-op thread count is a second input to every
  sampler result. `run_sweep.py` pins `OMP_NUM_THREADS=1` on children while a bare CLI
  run inherits the machine default. Everything here therefore goes through
  `run_sweep.py`, and nothing is compared against a bare CLI run.

## Methods

*(filled in with the run's own recorded numbers below)*

## Results

*(pending)*

## Recommendation

*(pending)*

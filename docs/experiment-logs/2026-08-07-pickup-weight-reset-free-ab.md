# Reset-free practice when the training distribution varies at pickup: `scheduled` vs `never`

Domain `tossingroomsplitpickupweight`, method `ees`, 10 fixed seeds (0-9), 30 test tasks
(14 TRASH / 14 RECYCLING / 2 EMPTY), 10 cycles x 150 steps per interaction period. The
two arms are invoked with different `--practice-reset-policy` values and no other flag
difference.

**This is the same protocol as
[the 2026-08-06 reset-free A/B](./2026-08-06-reset-free-practice-ab.md), on a different
domain, and the numbers are not comparable across the two.** That experiment ran on
`tossingroomsplit`; this one runs on the new `tossingroomsplitpickupweight`, whose task
distribution genuinely differs. Nothing here may be pooled with those 80 banked runs.

## Pre-registration

**Written and committed before either sweep was run.** The commit that adds this section
contains no results; the numbers arrive in a later commit on the same branch.

### Background: what this follows from

The 2026-08-06 A/B measured `scheduled` **151/300** against `never` **85/300** on
`tossingroomsplit` (paired, p = 0.0078, `never` worse in 8/10 seeds and better in 0/10).
Its own published correction is that the comparison confounds **two** mechanisms, and
that no single-flag experiment on that domain can separate them:

1. **Stranding.** The one-way ledge blocks the *directed* edge 2 -> 3, the only edge from
   rooms {0, 1, 2} into {3...6}, and the pile -- the sole source of items -- is in room 3.
   Rooms 0-2 are absorbing at any horizon. A follow-up read out of run output found
   **9/10 `never` seeds strand permanently**, with onsets at period 1 for 6/10 seeds
   (2, 3, 4, 5, 8, 9), period 3 for 2/10 (6, 7), period 4 for 1/10 (seed 0), and never for
   seed 1. **0/9** stranded seeds recover: across 74 pooled post-onset periods the only
   skills executed in **74/74** are `MoveRoom` and `PressRecycling` -- zero pickups, zero
   throws, zero `PressTrash`. The apparent recovery visible under `scheduled` is
   `reset_to_task` rescuing the robot, not self-recovery. No recovery machinery exists:
   all three `Method.reset_environment` implementations return `False` without writing,
   and `TossingRoomSplitProblem` never sets `human`.
2. **A collapsed training distribution.** A `Task`'s continuous parameters live in its
   `initial_state`, and `reset_to_task` is the only thing that installs one. Under `never`
   it is never called, so every state feature no action writes stays frozen at its
   `hard_reset` value for the whole run -- and on that domain those features are the bin's
   `throw_distance` and the item's `weight`, which *are* the learned throw sampler's input
   row. Measured: 194 greedy throws at **1** distinct required-force target under `never`,
   against 86 distinct targets over 440 throws under `scheduled`.

The PR below this one in the stack removes mechanism 2 by construction. In
`tossingroomsplitpickupweight` the item weight is drawn **at pickup**, off a per-task
pre-sampled array, so it is written by an action the robot takes rather than by a reset;
and the bin distance is fixed, so it cannot be frozen at an unrepresentative value
because it never varies for anyone. A reset-free arm there therefore practises across the
weight distribution as long as it keeps picking things up.

### What is being manipulated

Exactly what the 2026-08-06 A/B manipulated: `scheduled` (the default) puts the
environment back to the freshly-sampled train task's initial state at the top of every
interaction period; `never` does not, so practice state runs continuously across period
boundaries and across the train task changing underneath it. A train task is still drawn
per period and still handed to `get_practice_policy`.

What is *different* is the domain underneath, and therefore what "no reset" costs. It no
longer costs the variation in the sampler's input row. It still costs the rescue.

### Prediction

**Direction: `never` still loses, and by a lot.** The mechanism removed is not the one
doing the damage. A stranded robot cannot reach the pile, so it takes no pickups; taking
no pickups is exactly the condition under which drawing weights at pickup changes nothing.
Weight-at-pickup can only vary the training distribution of an agent that is still
practising.

**This is the point of the experiment rather than a defect in it.** Josh's framing:
*"this is sort of intentional -- the point is that practice makes perfect gets stuck when
it gets stuck -- that's the whole point -- showing that it doesn't work in the reset free
setting."* A negative result here is the result.

**Magnitude: large, comparable to the 2026-08-06 gap.** I expect the `scheduled - never`
difference to be within roughly 10pp of that experiment's 22.0pp, and I will treat
anything under 10pp as no meaningful difference regardless of what a test says.

**Sharpened, falsifiable secondary prediction.** Under weight-at-pickup, a run's weight
draws are exactly its pickups. If stranding reproduces at the onsets measured on
`tossingroomsplit`, then **6/10 `never` seeds should draw exactly 1 weight for the whole
run** -- each of them the `PickupRecycling` that strands the robot -- with seed 1
unstranded and seeds 0, 6 and 7 stranding later. That is an n=1 unidentifiability
problem, not a sparse or a biased sample: trash and recycling weights are drawn from one
law (`Uniform[0.5, 1.5)`) and one stream, so no weight region is systematically excluded;
what stranding biases is the **item type** (pooled `never` pickups on `tossingroomsplit`:
298/307 trash against 9/307 recycling) and what it destroys is the **sample size**.

**What would falsify the framing.** If `never` closes to within 10pp of `scheduled` here,
then mechanism 2 -- not stranding -- was carrying the 2026-08-06 result, and that log's
causal reading needs revisiting. If the two arms' pickup counts come out comparable, then
stranding did not reproduce on this domain and neither prediction above is being tested.

**Per-family prediction.** As before: damage on RECYCLING first (its bin sits behind the
ledge), TRASH second, EMPTY least (a walk-and-press with no throw). EMPTY's denominator is
2/seed = 20/arm and cannot carry an inference on its own.

### Analysis plan, fixed in advance

- **Primary test: paired.** The arms share the seed set, so the 10 seeds are 10 paired
  observations. A paired t-test on the per-seed final-sweep success count, plus a Wilcoxon
  signed-rank test as the distribution-free companion. An unpaired test discards the
  pairing and understates significance.
- **Reported as counts.** `x/y` everywhere, per arm and per family, never a bare
  percentage.
- **MDE, derived per comparison from its own two denominators**, using this repo's
  constant `2.801585` (two-sided alpha = 0.05, 80% power):

  ```text
  MDE = 2.801585 * sqrt( p1(1 - p1)/n1 + p2(1 - p2)/n2 )
  ```

  evaluated at the observed arm rates. Each family gets its own MDE from its own
  `n1`/`n2`; the overall comparison gets its own.
- **A null result will be written as "null result", in full, and reported as such.**
- **Checks run before any inference**, all read out of each run's own `stats.json` rather
  than inferred from the flag: `num_practice_resets` (10 per `scheduled` run, 0 per
  `never` run), matched online transitions across arms, and the realised 14/14/2 test
  composition per seed.

## Results

*Not yet run. This section is filled in by a later commit on this branch.*

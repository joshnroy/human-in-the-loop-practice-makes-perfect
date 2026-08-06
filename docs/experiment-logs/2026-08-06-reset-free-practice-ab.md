# Reset-free practice on Tossing Room (split throws): `scheduled` vs `never`

Domain `tossingroomsplit`, method `ees`, 10 fixed seeds (0-9), 30 test tasks
(14 TRASH / 14 RECYCLING / 2 EMPTY), 10 cycles x 150 steps per interaction period.
The two arms differ in `--practice-reset-policy` and nothing else.

## Pre-registration

**Written and committed before either sweep was run.** The commit that adds this
section contains no results; the numbers arrive in a later commit on the same branch.

### What is being manipulated

`scheduled` (the default, and the only behaviour that existed before this stack) puts
the environment back to the freshly-sampled train task's initial state at the top of
every interaction period. `never` does not: practice state runs continuously across
period boundaries, and across the train task changing underneath it. Nothing else
differs -- a train task is still drawn per period and still handed to
`get_practice_policy`, so the train-task distribution is identical.

This comparison was not expressible before the two PRs below it in this stack.
`PracticeLoop._evaluate` used to run on the *same* `Problem` as practice, and every
evaluation episode opens with `reset_to_task`, a privileged state-write. A 30-task
sweep is therefore 30 resets handed to the practice environment for free, 11 times
over a 10-cycle run. A `never` arm run against that harness would have been reset 330
times and would have measured nothing.

### Prediction

**Direction: I expect `scheduled` to beat `never`.** `PracticeLoop`'s own docstring
has argued since it was written that the per-period reset is load-bearing rather than
tidiness -- an interaction period that resumes from wherever the last one ended begins
somewhere unearned, and on Light Switch that meant starting beside the light and
spending the whole budget on the toggle. Tossing Room's ledge makes the analogous
failure sharper and one-directional: the ledge is irreversible, so a practice period
that ends past it stays past it forever, and every subsequent period is stuck in the
region where the pile is unreachable and no throw can be practiced at all. Under
`never` I expect practice to strand itself early and stop generating throw experience,
so the learned samplers should be worse.

**Magnitude: large.** The scoping pass predicted 50-70pp against a measured 8.1pp MDE
at n=20. I will treat anything under 10pp as "no meaningful difference" regardless of
what a test says.

**Confidence: moderate, and one observation already points the other way.** A 3-cycle
mechanism probe on seed 0 alone -- run to confirm `num_practice_resets` really goes to
0, not to measure anything -- came out `never` 13/30 vs `scheduled` 8/30. That is a
single seed at a third of the training budget and supports no inference, but it is
disclosed here because it was seen before this prediction was written and it is
evidence against the direction predicted above.

**Per-family prediction.** If the stranding story is right, the damage should land on
RECYCLING first (its bin sits behind the ledge) and TRASH second, with EMPTY least
affected since it is a walk-and-press with no throw. EMPTY's denominator is 2/seed =
20/arm, which is too small to carry an inference on its own.

### Analysis plan, fixed in advance

- **Primary test: paired.** The arms share the seed set, so the 10 seeds are 10 paired
  observations. A paired t-test on the per-seed final-sweep success count, plus a
  Wilcoxon signed-rank test as the distribution-free companion. An unpaired test here
  would throw away the pairing and understate significance.
- **Reported as counts.** `x/y` everywhere, per arm and per family, never a bare
  percentage.
- **MDE, derived per comparison from its own two denominators**, using this repo's
  constant `2.801585` (two-sided alpha = 0.05, 80% power):

  ```text
  MDE = 2.801585 * sqrt( p1(1 - p1)/n1 + p2(1 - p2)/n2 )
  ```

  evaluated at the observed arm rates. Each family gets its own MDE from its own
  `n1`/`n2`; the overall comparison gets its own. The `20.19pp` figure that appears in
  older merged work belongs to a different comparison and is not reused here.
- **A null result will be written as "null result", in full, and reported as such.**

## Results

To be filled in by a later commit on this branch.

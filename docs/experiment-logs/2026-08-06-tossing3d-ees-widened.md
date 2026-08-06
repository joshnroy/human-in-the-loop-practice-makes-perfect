# EES on Tossing3D, with the throw standoff widened to the feasible range

Follows `2026-08-06-tossing3d-ees.md`, which ran EES on this domain under
`THROW_STANDOFF_BOUNDS = (1.20, 1.65)` and got a null result: 24/90 to 33/90 over ~55
online transitions, exact paired Wilcoxon n = 8 non-tied of 9, **p = 0.1328**, against a
99/100 skill-oracle ceiling. That log named the reason structural rather than
methodological — the thing to be learned here is a **constant**, and the old bounds were
barely wider than the band that solves, so a uniform draw was already right about as often
as not (155/330 pooled) and a learned sampler had almost no headroom over its own prior.

The preceding PR widened the bounds to the measured feasible range, `(0.45, 1.75)`. Same
physics, same skills, same state, same success criterion; only the interval the sampler
draws from changed. This log is the re-run.

## Pre-registration

Written **2026-08-06T19:59:06Z** and committed before any EES run under the widened bounds
had produced a `stats.json` — the single-seed timing run was still in flight. The commit
that adds this section contains no results, which is the point of committing it
separately.

### The design

10 fixed seeds, `--num-test-tasks 10`, `--num-cycles 20`,
`--max-steps-per-interaction 20` — matched to the previous run so the two protocols are
comparable. Arms: EES, and `skill-oracle` as the ceiling. **The comparison that matters is
EES against the newly measured uniform-draw baseline**, not against the previous run's
numbers, which were taken under a different prior and are not comparable.

### What was predicted, and why

1. **Pre-practice EES lands at the uniform baseline.** Before any learning the sampler
   *is* the uniform prior, so the first checkpoint should be indistinguishable from the
   measured uniform-draw rate. If it is not, something other than the sampler is driving
   the score.
2. **End-of-training beats the uniform baseline clearly.** Held with reasonable
   confidence: the baseline is now low enough that almost any localisation of the constant
   shows against it.
3. **But a *smaller* effect than "large and clearly significant", with sample size the
   binding constraint rather than the method.** The previous run measured ~55 online
   transitions per seed, which is **~18 `MoveToThrowPose` executions**;
   `exploration_epsilon = 0.5` (predicators' own default) keeps about half of those
   uniform-random; and the widened prior now hits the solving band only about one time in
   five. That is on the order of **3-4 positive labels per seed** from which to localise a
   constant inside a 1.30 m interval. Point prediction: end-of-training around
   **40-60/100 pooled**, not the ~99/100 ceiling.

   The one mechanism pulling the other way, noted at pre-registration time: at exploitation
   the wrapped sampler draws **100 candidates and takes the argmax** of the learned score,
   so even a poorly-fit classifier can convert into a good action. This is why prediction 2
   is held more confidently than prediction 3.
4. **Paired Wilcoxon over 10 seeds, end vs pre-practice: p < 0.05.** The widening lowers
   the floor enough that per-seed differences should be consistently positive even if
   individually modest — which is exactly what the old bounds denied the previous run (six
   seeds up, two down, one tied).
5. **The oracle ceiling stays ~99/100.** It is unaffected by the sampler, and the upper
   bound was chosen to keep `NearBin` false at the post-`Pick` pose on all 30 measured
   seeds. **If the ceiling drops, that is a defect in the choice of upper bound, not a
   result** — it would mean some seed's post-`Pick` pose is being admitted after all, and
   it should be treated as a bug to fix rather than a finding to report.

### What would make this wrong in the interesting direction

If end-of-training sits at the uniform baseline with a null result, that is the more
interesting outcome: it would mean the sampler cannot localise a constant even when the
signal is strong and the prior is mostly wrong. Two explanations would need separating
before believing it — too few positive labels (a budget problem, fixable with more cycles)
versus the sampler being unable to use them (a method problem). The per-seed count of
solved practice throws is what distinguishes them.

## Methods

<!-- filled in after the run -->

## Results

<!-- filled in after the run -->

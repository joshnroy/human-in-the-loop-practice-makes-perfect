# PREREGISTRATION (written before any result was computed)

Recorded here first, and left in the diff history deliberately, so the prediction cannot
be edited after the fact.

**Prediction.** The null does **not** persist. At the standard budget recycling
accumulates ~16 attempts and ~3.3 landings per seed; at 10x it should accumulate ~160 and
~33, and 33 positives is a materially different training set from 3.3 — the diagnosis's
"classifier interpolating <= 16 rows" regime is left behind. So I expect recycling's
informed rate to separate from its epsilon-random control, and I expect the separation to
be **real but far smaller than trash's +49pp** — of order +10 to +25pp at 25,000
transitions.

**Where.** Between **5,000 and 10,000** transitions. Target separation should cross the
0.1 tolerance much earlier than that (targets are ~uniform on [0.1, 0.9] and a uniform
force lands with probability ~0.2, so a seed should own two landings more than 0.1 apart
within its first ~5 landings, i.e. by ~2,500 transitions). Separation crossing tolerance
is therefore predicted to be **necessary but not sufficient**: the classifier needs both
the tilt to be visible *and* enough rows for the fit to find it, and the second is the
later of the two.

**Which variable it aligns with.** **Accumulated landings**, not transitions. The
classifier only ever sees positives, and the seeds differ several-fold in how fast they
accumulate them (the standard run's per-seed attempt ratio spanned 2.7x to 8.8x), so a
crossover pinned in transitions should be much more spread across seeds than one pinned
in landings.

**What would make me wrong in the more interesting direction.** If the null persists to
25,000 transitions, the saturated-classifier failure is not a budget problem at all, and
no amount of practice fixes it — which is a stronger claim than anything the standard run
could support, and it would mean a gate on positive count and target separation buys
nothing that more practice would not have bought anyway.

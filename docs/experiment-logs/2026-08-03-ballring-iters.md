# Sampler training budget on Ball-Ring: a null result, and a default that moves anyway

The Ball-Ring sampler-iteration sweep was run to answer "how many gradient steps per
sampler refit is right?". The point estimates trace a clean inverted U with a maximum at
10000. **That inverted U is not statistically established**, and this log exists mostly to
say so before the number gets quoted as a finding.

The class default `EesMethod.sampler_max_train_iters` still moves from 1000 to 10000 in
this change, but for a *structural* reason that needs no p-value. Those are two separate
claims and they should not be conflated.

![learning curves and endpoint vs training budget](./2026-08-03-ballring-iters.png)

## The structural reason the default moves

`MlpBinaryClassifier` early-stops after `n_iter_no_change = 5000` iterations without
improvement (predicators' `mlp_classifier_n_iter_no_change`). The old default of **1000**
sits below that floor, so the early-stopping branch could **never fire** — not "rarely",
never. Every refit ran exactly 1000 full-batch steps regardless of whether the loss had
stopped moving, and the mechanism ported from predicators was dead code in the default
configuration.

Any value above 5000 fixes that. 10000 is the argmax of the evidence we have among values
that do. That is the whole justification, and it is enough on its own.

One real consequence for reviewers: at 10000 early stopping *can* fire, so refit cost is
now **data-dependent** rather than a fixed 1000 steps. Wall-clock per cycle will vary
across seeds in a way it previously did not.

## The measurement, and why it does not establish an optimum

10 seeds per arm, Ball-Ring, fixed test set, 25 cycles, arms run **sequentially** on one
base (Fast Downward's timeout is wall-clock, so concurrent arms bias each other). Per-seed
per-sweep data is committed as `2026-08-03-ballring-arms.json`.

| iters | final mean % | sd | paired p vs 10000 | seeds to resolve at 80% power |
|---|---|---|---|---|
| 1000 | 83.0 | 22.1 | 0.057 | ~18 |
| 3000 | 90.0 | 28.3 | 0.350 | ~82 |
| **10000** | **99.0** | **3.2** | — | — |
| 30000 | 91.0 | 12.0 | 0.070 | ~20 |
| 100000 | 89.0 | 16.0 | 0.085 | ~22 |

Every arm shares the same 10 seeds, so these are **paired** two-sided t-tests. scipy is not
a project dependency, so the values are quoted as constants in `ballring_sampler_iters.py`
rather than recomputed by it; they come from:

```bash
pip install scipy  # not a project dependency; ad-hoc for this check only
python - <<'PY'
import json, statistics
from math import ceil
from scipy import stats
d = json.load(open("docs/experiment-logs/2026-08-03-ballring-arms.json"))
def endpoints(arm):
    out = {}
    for seed, rows in d[arm].items():
        last = max(rows, key=lambda row: row[0])
        out[seed] = 100 * last[1] / last[2]
    return out
E = {a: endpoints(a) for a in ("iters1k", "iters3k", "iters10k", "iters30k", "iters100k")}
seeds = sorted(E["iters10k"], key=int)
for arm in E:
    if arm == "iters10k":
        continue
    x = [E["iters10k"][s] for s in seeds]
    y = [E[arm][s] for s in seeds]
    diffs = [u - v for u, v in zip(x, y)]
    dz = statistics.mean(diffs) / statistics.stdev(diffs)
    n = ceil(((stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) / dz) ** 2) + 1
    print(arm, round(stats.ttest_rel(x, y).pvalue, 4), "n80 =", n)
PY
```

Not one comparison reaches p < 0.05, and the Bonferroni threshold for four comparisons
would be 0.0125. Every arm's endpoint also lies inside predicators' own ±1sd band
(91.0 ± 12.0). The correct summary is **"the point estimates order this way and nothing is
resolved"**, not "10000 is the optimum".

This is the same error the 2026-08-03 session handoff records as its main methodological
lesson — five mechanistic hypotheses were generated and killed to explain a 98-vs-91 "gap"
that a power analysis would have shown was never established (p ≈ 0.081). Applying that
lesson to this sweep is why the table above has a p-value column at all.

## What is separately solid

Two things from the same investigation do not depend on the sweep resolving:

- **Running at predicators' own 100000 reproduces predicators' own score** — 89.0 ± 16.0
  against its 91.0 ± 12.0. That is a positive control on the port's faithfulness, and it
  holds regardless of which arm is best.
- **More training genuinely overfits the decisive cup-placement classifier.** Train BCE
  falls from 5.9e-3 at 10000 to 2.8e-5 at 100000 while held-out argmax success falls from
  0.988 to 0.930 (paired, t = 5.67, 10/10 seeds). That is a direct measurement on the
  classifier, not an inference from endpoint scores, and it is why the endpoint curve
  bends down rather than plateauing.

The endpoint sweep is the underpowered part. The classifier measurement is not.

## What would settle it

~18–22 seeds per arm resolves 10000 against 1000, 30000, or 100000; 3000 would need ~82
and is probably not worth chasing. That is roughly a doubling of the sweep already run.
Given that the residual after the hyperparameter correction is ~2 points and inside noise,
this is recorded as an option rather than a recommendation.

## Reproducing

```bash
# one arm at a time -- check `pgrep -f hitl_pmp.cli` is empty first
python -m scripts.run_sweep --env ballring --methods ees --num-seeds 10 \
  --results-root results/iters-10k \
  --shared-args "--num-test-tasks 10" \
  --method-args "ees=--num-cycles 25 --max-steps-per-interaction 100 \
    --competence-window-size 2 --competence-recency-size 2 \
    --exploration-epsilon 0.5 --sampler-max-train-iters 10000"

python -m analysis.practice_makes_perfect.ballring_sampler_iters \
  --arms-json docs/experiment-logs/2026-08-03-ballring-arms.json \
  --output docs/experiment-logs/2026-08-03-ballring-iters.png
```

The analysis script reads the committed aggregate rather than a sweep directory: the raw
sweep directories for these arms lived outside the repo and did not survive the move
between machines, so the aggregate is the only remaining record.

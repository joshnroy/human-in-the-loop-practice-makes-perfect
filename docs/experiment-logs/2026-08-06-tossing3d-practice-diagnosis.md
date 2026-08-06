# Tossing3D: why does EES not learn? — pre-registration

**Status: PRE-REGISTRATION. No run has been made under the instrument. Nothing below is
a result.** Committed before any `stats.json` carrying `practice_outcomes_per_cycle`
exists for this domain, in the manner of the pre-registrations for PR #103 and PR #108.

## Question

PR #108 measured EES on Tossing3D at `19/100` pre-practice and `21/100` at end of
training over ~53 online transitions (exact paired Wilcoxon, n = 8 non-tied of 10,
`p = 0.8281`), against a `99/100` oracle ceiling and a `543/2700` uniform-draw baseline.
It declined to say why, writing that

> too few positive labels and a sampler that cannot use them are both consistent with the
> data, and `stats.json` records no practice outcomes, so the discriminating quantity does
> not exist in any run to date.

That quantity now exists. This asks it: **starvation or inability — or neither?**

## Background

Tossing3D has three lifted skills. `Pick` (`param_dim = 2`) draws from upstream's own
tight bounds; `Toss` has **`param_dim = 0`**; `MoveToThrowPose` (`param_dim = 1`) is the
standoff, and it is the domain's only meaningful learnable parameter. Because
`bin_init_region` is 1 mm wide the correct standoff is a **constant** — the same in every
episode. #105 widened `THROW_STANDOFF_BOUNDS` to the measured feasible `(0.45, 1.75)`, of
which `THROW_SOLVING_BAND = (1.15, 1.375)` reliably scores, so a uniform draw is right
roughly 1 time in 5 — measured through the CLI as `543/2700`.

#99's still-open recommendation is that this constant-target structure is itself the
problem: there is nothing for a state-conditioned sampler to condition *on*, which would
make Tossing3D structurally the identity arm of the representation A/B in PR #97. #105
widened the sampler's search range but left the target a constant, which makes it harder
to find without giving the sampler anything to condition on. If that reading is right,
"starved vs unable" may be the wrong dichotomy, and this pre-registers a third outcome
for exactly that reason.

## Hypothesis

**H3 — neither: the label the standoff sampler is trained on is not the label that
predicts task success.**

Stated as a mechanism, so it is falsifiable rather than a hedge. `MoveToThrowPose`'s
`add_effects` are `{NearBin(robot, bin)}`, and `NearBinClassifier`'s own docstring says
`NearBin` tests a standoff inside `THROW_STANDOFF_BOUNDS` — *the range the sampler draws
from*, not `THROW_SOLVING_BAND` — and that it "says the robot can throw from here, not
that it will score". EES labels a practice attempt by whether the skill's own add effects
held. So **every** standoff the sampler is able to draw should satisfy the label it is
graded on. Meanwhile the skill whose outcome does depend on the standoff — `Toss`, which
adds `InGoalRegion` — has `param_dim = 0` and therefore no sampler at all.

If that is what is happening, the consequence is mechanical rather than statistical.
`MlpBinaryClassifier.fit` takes its single-class shortcut when `np.all(y_data == 1)` and
sets `_single_class_prediction = 1.0`; every candidate then scores exactly 1.0;
`LearnedSkillSampler.sample`'s deviation-6 branch sees a maximum attained by every
candidate and returns a **uniform draw** with `was_informed = False`. Forever, at every
cycle, however much practice is bought. EES's standoff would then be a uniform draw over
the bounds by construction, which forces its evaluation score to the uniform-draw
baseline — `543/2700`, i.e. `20.1/100`, against #108's measured `19/100` and `21/100`.

This is derived from reading the code, and reading is not measurement. The run is what
decides it.

## Decision rule

Pooled over 10 seeds for `MoveToThrowPose`, from `practice_outcomes_per_cycle`: let `A`
be attempts, `S` successes, `I` informed attempts, `IS` informed successes. The reference
rate is the uniform-draw baseline, `543/2700 = 0.201`.

| conclusion | rule |
| --- | --- |
| **never asked** | `A == 0` — the skill was never practiced, and nothing else here applies |
| **starvation** | `S/A < 0.9` **and** `I/A <= 0.05` **and** `I` per window is rising across cycles — the classifier is accumulating labels and has not yet reached the point of expressing a belief |
| **inability** | `I/A >= 0.30` **and** `IS/I` within ±0.10 of `0.201` — the classifier is consulted in quantity, states a belief, and that belief is no better than the prior |
| **H3, neither** | `S/A >= 0.90` **and** `I/A <= 0.05` **and** `I` per window is flat at zero to the final cycle — the sampler's label is almost always positive, so it never has two classes to separate and no amount of practice changes that |
| **undecided** | anything else, including `S/A` in `[0.5, 0.9)` or `I/A` in `(0.05, 0.30)` |

`undecided` is a real outcome and will be reported as one, with the run that would settle
it, rather than rounded to whichever neighbouring cell is closest.

**The noise floor is checked before anything is interpreted.** Another agent has measured
Tossing3D as **not reproducible from `--seed`**: seed 0, run twice under identical
arguments and identical code, gave `3/10` and `2/10`, a same-seed swing of ≥10 pp on the
episode counts. Two consequences, both binding:

1. **No claim here may rest on a difference smaller than 10 pp in task success**, and none
   is designed to. Every cell above is a *categorical* claim about practice counts with a
   denominator in the hundreds — `I/A <= 0.05` against `I/A >= 0.30` is not a margin that
   a 1-episode-in-10 simulator wobble can cross.
2. **This run is not comparable per-seed with #108's.** Its seed *k* is not #108's seed
   *k*. Nothing below will be paired against #108's stored per-seed numbers; #108's
   aggregates are quoted as context only.

If the outcome lands in `undecided`, or if it depends on a task-success difference at all,
the honest report is that the diagnosis waits on the reproducibility fix — and that is
what will be written.

## Method (to be run, not yet run)

Protocol identical to #108's EES arm, so the two are comparable: `scripts/run_sweep.py`,
fixed seeds 0-9, `--num-test-tasks 10 --num-cycles 20 --max-steps-per-interaction 20
--max-workers 10`, KINDER venv, inside
`systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 -p OOMPolicy=continue`.

Code under test is #108's branch tip (which carries #99's `PddlWriter` sigil fix, without
which `--method ees` plans nothing here, and #105's widened bounds) plus the instrument
from PR #111. Analysis is `analysis/practice_makes_perfect/practice_diagnostics.py`,
unmodified.

One deviation from #108, stated in advance: it ran before #100/#102/#104/#106 merged and
carries a status note saying its EES counts were produced by code `main` has since
changed. This run is at that code, so its task-success numbers are a **fresh measurement**
rather than a reproduction, and are reported as such.

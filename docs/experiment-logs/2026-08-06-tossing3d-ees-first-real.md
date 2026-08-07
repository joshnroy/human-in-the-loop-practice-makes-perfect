# Tossing3D, EES vs uniform vs oracle: the first run where the sampler is consulted

**Everything in this file up to and including `Method (to be run, not yet run)` was
committed before any run of this experiment existed**, in the manner of the
pre-registrations for PR #103, PR #108 and PR #127. Results are appended below it, and
nothing above them is edited afterwards except this header.

## Question / goal

Does EES learn Tossing3D's throw standoff, now that the standoff sampler is graded on a
label that can actually fail?

## Background

Every previously published Tossing3D number — `24/90`, `33/90` (#99), `19/100`, `21/100`
(#108), the `543/2700` uniform-draw baseline (#105), and both arms of #127 — measured a
domain in which **the relevant sampler was never consulted**. Three defects, in the order
they were found, all now on `main` at `ebf3d92`:

- **#118 / #99** — `skills.py` declared `Variable(name="?robot")` while `PddlWriter` owns
  the `?` sigil, so every emitted PDDL domain carried `??robot` and Fast Downward's
  translator exited 31 while still parsing the domain. `0/6` symbolic states were
  plannable before the fix, `6/6` after. Without it `--method ees` plans nothing here at
  all.
- **#123** — `NearBinClassifier` accepted `low - tol <= dx <= high + tol` with `low`/`high`
  read straight off `THROW_STANDOFF_BOUNDS`, **the identical symbol `skills.py` samples
  the standoff from**. Every draw the sampler could make therefore satisfied the skill's
  own add effect by construction. Measured over 10 seeds in #127: `MoveToThrowPose` scored
  `175/175` labelled success on `0/175` informed draws, its measured success rate pinned
  at exactly `1.0`, so `skip_perfect` scored it `-inf` and `choose_practice_target` never
  selected it at all. The predicate is now `RobotAtSuccessfulThrowPose`, deriving its
  acceptance band per call from the live goal-region bounding box with only
  `THROW_RANGE = 1.275` calibrated; #123 also absorbed #105's widened bounds
  `(0.45, 1.75)`.
- **#119** — the practice tally previously pooled "this skill has no sampler"
  (`param_dim == 0`) together with "a sampler was consulted and could not discriminate".
  `SamplerConsultation` now separates `NO_SAMPLER` / `UNINFORMATIVE` / `INFORMED` /
  `EPSILON_RANDOM`. That undifferentiated pool is how the above stayed invisible across
  three experiments.

So there is no usable baseline in the literature of this repo. `24/90`, `33/90`, `19/100`
and `21/100` are all measurements of uniform sampling, and `543/2700` was measured under
bounds that #123 changed. **Every arm here is re-measured from scratch.**

The mechanism the fix turns on: `MoveToThrowPose`'s add effect can now fail, so its
training set has two classes, so `MlpBinaryClassifier.fit` no longer takes its
`np.all(y_data == 1)` single-class shortcut, so `LearnedSkillSampler.sample` no longer
finds its maximum attained by every candidate and no longer falls through to a uniform
draw with `was_informed = False`. And because the measured success rate is no longer
exactly `1.0`, `skip_perfect` no longer scores the skill `-inf`, so
`choose_practice_target` can select it.

## What a win here would and would not mean

`bin_init_region` is degenerate — the bin does not move between episodes — so **the
correct standoff is a constant**. A positive result here demonstrates **sampler learning
in the sense of finding and memorising a constant**. It does **not** demonstrate
representation learning, which would require the target to be a function of observable
state, and it does not address #99's separate standing concern that there is nothing here
for a state-conditioned sampler to condition *on*. Josh took this trade deliberately
("memorizing sampler is ok"). No result below may be written up as more than this.

## Hypotheses

**H1 (primary, structural).** With the label fixed, `MoveToThrowPose`'s sampler
discriminates: EES's `MoveToThrowPose` succeeds at a higher rate than the same skill under
uniform draws (`random-skills`), by more than the MDE on those two denominators.

**H2 (primary, within-run).** In EES's own runs, informed draws succeed more often than
epsilon-random draws of the same skill — the sampler's stated belief beats the uniform
prior it would otherwise fall back to.

**H3 (secondary, provisional).** EES's end-of-training task success exceeds
`random-skills`'. Marked provisional in advance for two independent reasons given under
"Noise floor" below, and **no conclusion here rests on it**.

**Ceiling check (not a hypothesis).** `skill-oracle` reaches approximately `100/100`,
confirming the domain is solvable at these settings and that any shortfall is the method's
and not the environment's.

## Pre-flight pass condition, checked before the full sweep

#123's own 3-seed / 20-cycle probe measured `MoveToThrowPose` at `54/121` succeeded,
`36/56` informed, and `121` attempts against `Pick`'s `60` — i.e. it went from never being
selected to being the most-practised skill in the domain. That is an unusually strong
prior, and it is cheap to check.

**Gate: a 3-seed, 20-cycle EES probe must show pooled `MoveToThrowPose`
`num_informed_attempts > 0`.** If it does not, something regressed between #123's probe
and the merge, the full sweep would be wasted, and the run stops there and reports the
regression as `x/y`.

Reported alongside the gate, but not gating: whether `MoveToThrowPose`'s attempt count
exceeds `Pick`'s, reproducing "most-practised skill in the domain".

**There is no positive control available in this experiment, and that is a real
limitation.** `Pick` was #127's control, but at `57/60` it now ties and falls back to
`0/0` informed — the same tie-and-fall-back state on a different skill that #123 fixed for
`MoveToThrowPose`. So "the sampler never made an informed draw" and "the instrument cannot
see informed draws in this build" are no longer separable by a within-run control. They
are instead separated by the `random-skills` arm, which exercises the same tally code, and
by the pre-flight gate above. Stated here rather than discovered later.

## Noise floor, and why H3 is provisional in advance

1. **Tossing3D has been measured as not reproducible from `--seed`** — seed 0 run twice
   under identical arguments and identical code gave `3/10` and `2/10`, a same-seed swing
   of at least 10 pp. #127 then observed its own arm A reproduce #108 exactly, per seed
   and in aggregate, which contradicts that. **The contradiction is unresolved and is not
   resolved here.** No claim below may rest on a task-success difference smaller than
   10 pp.
2. **The task-success axis is underpowered at this design** (see MDEs). H1 and H2 read
   counts of skill executions with denominators in the hundreds, which neither the seed
   defect nor the small evaluation set moves.

## Minimum detectable effects

Every MDE is derived from its own two denominators as
`2.801585 · √(p̄(1−p̄)(1/n₁ + 1/n₂))` — the two-sided 5%, 80%-power normal approximation,
where `2.801585 = z₀.₉₇₅ + z₀.₈₀`. Planning values below; **each is recomputed against the
realized denominators when results are in**, and the recomputed value is what is reported.

| comparison | n₁ | n₂ | planning p̄ | MDE |
| --- | --- | --- | --- | --- |
| H3: task success, EES vs `random-skills` (10 seeds × 10 test tasks each) | 100 | 100 | 0.25 | **17.2 pp** |
| H3: task success, EES vs `skill-oracle` | 100 | 100 | 0.60 | **19.4 pp** |
| H1: `MoveToThrowPose` success rate, EES vs `random-skills` (≈400 attempts/arm at #123's probe rate) | 400 | 400 | 0.30 | **9.1 pp** |
| H2: informed vs epsilon-random draws within EES (≈200 each) | 200 | 200 | 0.50 | **14.0 pp** |

The first row is the important one to read before any result is quoted: **at 100 vs 100
evaluation episodes, nothing under about `17/100` is detectable at all.** H3 is therefore
registered as descriptive unless the gap is large, and a non-significant H3 is a statement
about this design's power, not evidence of no effect.

## Amendment 1, made before any result was read

**Registered while the pre-flight probe was still running and before a single number from
it had been looked at.** Recorded here rather than folded silently into the rule above,
so the sequence stays auditable.

`RandomSkillsMethod.practice_outcomes()` is not overridden and therefore returns `{}` —
the concrete default on `Method`, whose docstring is explicit that `{}` means "this Method
does not measure practice at all", as distinct from a present zero-attempt entry. So the
`random-skills` arm records **no `MoveToThrowPose` tally at all**, and `U` as originally
defined — that arm's own `S/A` for the standoff — does not exist and cannot be computed.
This was found by reading `random_skills_method.py`, not by running it.

**`U` is therefore redefined as EES's own non-informed draws of the same skill**:
`num_random_attempts` (the epsilon-greedy coin flip) plus the uninformative fallback,
both of which are uniform draws over `THROW_STANDOFF_BOUNDS` by construction. Three
reasons this is a better reference than the one it replaces, not merely an available one:

1. It is **within-run and within-seed** — same code, same scene seeds, same cycle
   structure — so it removes the cross-arm confound the original rule carried.
2. It is the comparison H2 already registered, so the two primary hypotheses collapse into
   one test rather than one being dropped.
3. It is the same asymmetry #127 used as a signature of the deviation-6 path, so it is
   measured on a quantity this domain is already known to expose.

The rule below is otherwise unchanged: same thresholds, same MDE formula, same Fisher
test, same `undecided` cell. `random-skills` remains an arm and remains the **task-success**
baseline for H3 — that is what it was needed for and it still serves it. What it cannot do
is provide a standoff label rate.

**H1 is restated** as: EES's informed draws of `MoveToThrowPose` succeed at a higher rate
than its own non-informed draws of the same skill, by more than the MDE on those two
denominators. MDE at the anticipated denominators (≈187 informed against ≈217 non-informed,
scaling #123's probe to 10 seeds), planning `p̄ = 0.45`: **13.9 pp**.

## Decision rule

Applied to `MoveToThrowPose` pooled over 10 seeds, from `practice_outcomes_per_cycle`.
Let `A` be attempts, `S` successes, `I` informed attempts, `IS` informed successes, and
let `U` be the `random-skills` arm's own `S/A` for the same skill — the uniform-draw
reference, measured in this experiment rather than quoted from #105.

| conclusion | rule |
| --- | --- |
| **regressed** | `I == 0` — the pre-flight gate failed, or the full sweep contradicts it; report and stop |
| **learns the constant** | `I/A >= 0.20` **and** `S/A − U >= ` its MDE **and** Fisher's exact `p < 0.05` on that 2×2 |
| **consulted but no better than uniform** | `I/A >= 0.20` **and** `S/A − U <` its MDE — the sampler states a belief and the belief is not worth more than the prior |
| **starved** | `0 < I/A < 0.20` — the classifier discriminates only occasionally; report the per-cycle trend and the run that would settle it |
| **undecided** | anything else |

`undecided` is a real outcome and will be reported as one, with the run that would settle
it, rather than rounded to whichever neighbouring cell is nearest. A null or ambiguous
result reported plainly is the deliverable; an overstated one is a retraction later.

Tests: **Fisher's exact** (two-sided) for the unpaired 2×2 count comparisons, since the
counts are small and no normal approximation is wanted; **exact paired Wilcoxon
signed-rank** for per-seed pre-versus-post within an arm, since arms share the seed set
and unpaired treatment there would throw that structure away. Both are exact by
enumeration — scipy is not a dependency of this project, and
`PairedTests.wilcoxon_signed_rank` already exists in `analysis/`.

Pooling `x/y` across seeds and then testing it treats draws as exchangeable across seeds,
which they are not. So **every pooled count is reported beside its per-seed spread**, and
the figure plots per seed. At 2,500 transitions on Tossing Room the ten seeds spanned
`0/14` to `14/14`, and a pooled number there described no seed that was actually run.

## Method (to be run, not yet run)

Three arms, 10 fixed seeds each (`0-9`), every run driven by `scripts/run_sweep.py` —
never a hand-rolled loop, never a bare CLI invocation.

| arm | `--method` | why |
| --- | --- | --- |
| **EES** | `ees` | the arm of interest |
| **uniform** | `random-skills` | a **fresh** uniform baseline. `543/2700` and `155/330` were measured under the pre-#123 bounds and do not transfer |
| **ceiling** | `skill-oracle` | sanity check that the domain is solvable at these settings |

Shared: `--num-test-tasks 10`. EES and `random-skills` additionally get `--num-cycles 20
--max-steps-per-interaction 20`, which is #108's and #127's protocol, so the practice
budget is comparable to theirs. `skill-oracle` takes neither flag (it does not accept
them) and runs as a single evaluation sweep.

Run under the KINDER venv (`../kinder-venv`, never `hitl-pmp`), with `MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl`, `PYTHONPATH` at **this worktree's** `src/`, and every sweep inside
`systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 -p OOMPolicy=continue` —
a planner grounds a fresh PyBullet sim per sampling attempt, and this box's
`DefaultOOMPolicy=stop` means a kernel OOM takes down the whole session.

`--max-workers` is chosen at launch from the measured machine-wide load, targeting ~22
concurrent runs across every agent on the box, and the value chosen is recorded with its
reason. Concurrency has been separately measured not to perturb results, so yielding cores
costs wall clock only.

All three arms' raw data — `stats.json`, `config_snapshot.json` and `timing.json` for
every run — is committed. #103 committed none of its own and is now unre-analysable, which
cost a 20-minute re-run.

---

# Results

Everything below was produced after the pre-registration and Amendment 1 above were
committed (`15ef33f` and `2bfca24`).

## The pre-flight gate: passed

3 seeds, 20 cycles, `--max-steps-per-interaction 20`, `--num-test-tasks 10`. Raw data in
`2026-08-06-tossing3d-ees-first-real-data/preflight-probe/`.

| lifted skill | `param_dim` | attempts | succeeded | informed draws | informed succeeded | epsilon-random |
| --- | --- | --- | --- | --- | --- | --- |
| **`MoveToThrowPose`** (the standoff) | 1 | **142** | 49/142 | **65/142** | **32/65** | 57/142 |
| `Toss` (adds `InGoalRegion`) | 0 | 47 | 47/47 | 0/47 | — | 0/47 |
| `Pick` | 2 | 60 | 54/60 | 19/60 | 18/19 | 13/60 |

**The gate required `num_informed_attempts > 0` and got `65/142`.** The non-gating check
also reproduced: `MoveToThrowPose` was practised **142 times against `Pick`'s 60**, i.e.
it is the most-practised skill in the domain, exactly as #123's own probe found (`121`
against `60`). Both the counts and the ratio line up with that probe, at a different
number of evaluation episodes.

**One correction to the brief this experiment was commissioned under.** It stated that
`Pick` is no longer usable as a control because it "ties at `57/60` and falls back,
showing `0/0` informed". In these runs `Pick` scores `54/60` and makes **`19/60` informed
draws, of which `18/19` succeeded** — so it has *not* tied here, and a within-run positive
control does exist after all. That is reported as a departure from what was expected
rather than quietly relied upon; the verdict below still does not need it, because
Amendment 1's reference is internal to `MoveToThrowPose` itself. It does, however, mean
"the instrument cannot see informed draws" is independently excluded: two different
skills' samplers report them in the same `stats.json`.

Applying the decision rule to the probe alone — 3 seeds, so this is a preview and not the
result — gives **learns the constant**: informed draws landed `32/65` against the same
arm's uniform draws at `17/77`, a gap of `+27.2` pp against an MDE of `22.4` pp on those
denominators, Fisher exact `p = 0.0008057`.

Task success over the 3 probe seeds moved `10/30` pre-practice to `24/30` at end of
training (per seed `9/10`, `8/10`, `7/10`). Reported as context only; the 10-seed arm
below is what the pre-registration commits to, and the task-success axis is provisional
for the two reasons registered above.

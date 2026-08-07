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

## What was actually run

`scripts/run_sweep.py`, `--env tossing3d --methods ees random-skills skill-oracle
--num-seeds 10`, shared `--num-test-tasks 10`, with `--num-cycles 20
--max-steps-per-interaction 20` given to `ees` and `random-skills` only (`skill-oracle`
rejects both flags). KINDER venv, `MUJOCO_GL=egl`, inside `systemd-run --user --scope
-p MemoryMax=16G -p MemorySwapMax=0 -p OOMPolicy=continue`. **`30/30` runs succeeded.**

`--max-workers 8`, chosen from measured load rather than the default: the box already
carried 15 runs from another agent, and `8 + 15 = 23` sits at the ~22 machine-wide budget.
Recorded by `analysis/run_timing.py`: `2805.4 s` wall for the sweep, median `844.7 s` per
run. Concurrency has been separately measured not to perturb results, so this cost wall
clock only.

**Provenance was verified per run, not assumed from the path.** All 30 `config_snapshot.json`
files were checked to be `env=tossing3d` with the intended `num_cycles`,
`max_steps_per_interaction`, `num_test_tasks` and `task_config=coincident`. That check
earned its keep: the session scratchpad is **shared between agents**, a generic `sweep.sh`
there was overwritten by another agent's script of the same name and executed by mistake,
and the equally generic `probe/` directory turned out to contain a second agent's `never/`
and `scheduled/` arms alongside mine. Nothing foreign entered any number below — the
analysis only ever reads `ees/`, `random-skills/` and `skill-oracle/` — but the near miss
is the reason the check exists.

## The verdict: learns the constant

Pooled over 10 seeds, every cell an `x/y` count of skill executions.

| lifted skill | `param_dim` | attempts | succeeded | informed draws | informed succeeded | epsilon-random | unparameterized |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **`MoveToThrowPose`** (the standoff) | 1 | **481** | 165/481 | **206/481** | **117/206** | 188/481 | 0/481 |
| `Toss` (adds `InGoalRegion`) | 0 | 156 | 150/156 | 0/156 | — | 0/156 | 156/156 |
| `Pick` | 2 | 200 | 180/200 | 29/200 | 27/29 | 27/200 | 0/200 |

Against the amended decision rule:

- `I/A = 206/481 = 0.428 >= 0.20` — the sampler is consulted **and discriminates**, in
  quantity.
- Informed draws land **`117/206`**; the same arm's own uniform draws land **`48/275`**.
  A gap of **`+39.3` pp** against an **MDE of `12.3` pp** on those two realized
  denominators, and **Fisher exact `p = 1.968e-19`**.

That is the **learns the constant** cell, on all three clauses, with no cell adjacent.

**And it is not one seed's doing.** All `10/10` seeds have a higher informed rate than
their own uniform rate — visible in the left panel as ten non-crossing lines:

| seed | informed | uniform (same run) | attempts | pre-practice | end of training | `Toss` landed |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 10/21 | 6/21 | 42 | 1/10 | 9/10 | 16/16 |
| 1 | 10/18 | 6/26 | 44 | 5/10 | 8/10 | 16/16 |
| 2 | 12/26 | 5/30 | 56 | 4/10 | 7/10 | 15/15 |
| 3 | 12/16 | 4/27 | 43 | 2/10 | 10/10 | 15/15 |
| 4 | 12/18 | 6/30 | 48 | 4/10 | 8/10 | 16/17 |
| 5 | 11/17 | 4/19 | 36 | 2/10 | 7/10 | 13/13 |
| 6 | 13/20 | 4/26 | 46 | 3/10 | 7/10 | 13/14 |
| 7 | 14/25 | 5/25 | 50 | 5/10 | 9/10 | 16/19 |
| 8 | 10/23 | 4/44 | 67 | 5/10 | 6/10 | 13/14 |
| 9 | 13/22 | 4/27 | 49 | 6/10 | 9/10 | 17/17 |

**`MoveToThrowPose` is also now the most-practised skill in the domain** — `481` attempts
against `Pick`'s `200` and `Toss`'s `156`. Before #123 it was scored `-inf` by
`skip_perfect` and `choose_practice_target` never selected it at all. So the fix changed
both halves: the sampler can now learn, and EES now chooses to teach it.

![Tossing3D, three arms, post-fix](https://raw.githubusercontent.com/joshnroy/human-in-the-loop-practice-makes-perfect/2d2f683de42d332a75b7ce7fe314d66ed2f2547d/docs/experiment-logs/2026-08-06-tossing3d-ees-first-real.png)

> **The pin above is to `2d2f683`, an unmerged commit on this branch.** It is the commit
> that carries the figure, taken from `git rev-parse` rather than hand-expanded. It must
> be **re-pinned after merge**, since a squash-merge mints a new SHA and the branch commit
> stops being reachable once the branch is deleted.

## The fresh uniform baseline the old numbers cannot supply

`543/2700` and `155/330` were measured under the pre-#123 bounds and do not transfer, so
both are re-measured here:

- **Uniform standoff draws: `48/275`.** From EES's own non-informed draws (Amendment 1),
  because `random-skills` records no practice tally at all. Close to, but not the same
  quantity as, the old `543/2700` — that was a solve rate over `THROW_SOLVING_BAND`, this
  is the `RobotAtSuccessfulThrowPose` label rate over the widened `(0.45, 1.75)`.
- **Uniform task success: `24/100`** at end of training (`26/100` before), from the
  `random-skills` arm. It does not learn, which is what a uniform baseline should do.

## Task success (secondary and provisional, exactly as pre-registered)

| arm | pre-practice | end of training | per-seed change |
| --- | --- | --- | --- |
| **`ees`** | 37/100 | **80/100** | `+8, +3, +3, +8, +4, +5, +4, +4, +1, +3` |
| `random-skills` | 26/100 | 24/100 | `0, +4, -3, +1, -1, -1, +3, -1, -2, -2` |
| `skill-oracle` | 100/100 | 100/100 | one evaluation only — it never practices |

- **EES improves on `10/10` seeds, no seed unchanged or worse.** Exact paired Wilcoxon on
  the per-seed change, `n = 10` non-tied, **`p = 0.0020`**.
- EES `80/100` against `random-skills` `24/100`: `+56.0` pp, MDE `19.8` pp, Fisher exact
  `p = 1.199e-15`.
- EES `80/100` against the `skill-oracle` ceiling `100/100`: `−20.0` pp, MDE `11.9` pp,
  Fisher exact `p = 6.643e-07`. **EES does not reach the ceiling**, and that shortfall is
  itself larger than its MDE, so it is a real remaining gap rather than noise.

The pre-registration marked this axis provisional for two reasons and **both still stand**.
The effect happens to be far larger than the `19.8` pp MDE, so it survives the power
objection — but the same-seed reproducibility defect (a measured swing of at least 10 pp)
is untouched by this experiment, and a `+56` pp gap is not evidence that the defect is
gone. **No conclusion here rests on the task-success axis**; the verdict above reads
practice counts only, and `verdict` cannot read `evaluations` at all.

**The ceiling check passed:** `skill-oracle` is `100/100`, `10/10` per seed. So the domain
is solvable at these settings and EES's shortfall is the method's, not the environment's.

## What this does and does not demonstrate

**It demonstrates sampler learning in the sense of finding and memorising a constant.**
`bin_init_region` is degenerate — the bin does not move between episodes — so the correct
standoff is the same number in every task. A classifier that has learned "about 1.275 m"
and nothing else would produce exactly the numbers above.

**It does not demonstrate representation learning.** That would require the target to be
a function of observable state, and there is nothing here for a state-conditioned sampler
to condition *on*. #99's standing concern is untouched by this result, and a domain where
the bin moves is the experiment that would address it. Josh took this trade deliberately
("memorizing sampler is ok"); it is recorded here so the result is not later read as more
than it is.

Three further limits worth stating:

- **`Toss` still has `param_dim = 0`** and is `156/156` unparameterized. The skill whose
  add effect is the domain's actual success criterion still has no sampler; what improved
  is the standoff that feeds it.
- **The remaining `20/100` gap to the oracle is unexplained.** Nothing here measures where
  it goes.
- **`--num-cycles 20` was not varied.** Whether the curve has plateaued or would keep
  climbing is not answered; the middle panel suggests it flattens after roughly 40
  transitions, but that is a reading of a figure, not a measurement.

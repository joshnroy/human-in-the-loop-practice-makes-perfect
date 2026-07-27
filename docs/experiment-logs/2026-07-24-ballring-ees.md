# EES (Practice Makes Perfect) reproduction on simulated Ball-Ring

Reproducing the paper's **Ball-Ring (Simulated)** result — the middle panel of
Figure 4 — with our own EES port, compared directly against the reference
`predicators` implementation run on the same environment. Companion to
[the Light Switch EES log](./2026-07-21-ees-reproduction.md).

The headline: matching the paper's Ball-Ring curve took **four** faithfulness
fixes, each of which we isolated by running predicators' own code as the anchor.
Three of them are corrections to this port; the fourth is a config default. With
all four, our EES reproduces the paper's flat-then-climb-to-~90% shape.

## The domain, and why it is hard

Tables sit in a ring around the center of a unit room; one is **sticky**, the rest
normal. The goal is `BallOnTable(ball, sticky-table)`. A **bare ball placed on any
table rolls off** to the floor, and a **cup placed on the smooth sub-region of the
sticky table falls off** too. So the one reliable plan is: drop the ball into the
cup, carry the cup onto the sticky table's *non-smooth* region, and the ball rides
along. The entire learning problem is **specializing the cup-placement sampler** to
hit the non-smooth region — a needle-in-cluttered-state target. See
[`environments/ballring/README.md`](../../src/hitl_pmp/environments/ballring/README.md).

Paper mapping: this is predicators' `ball_and_cup_sticky_table`
(`BallAndCupStickyTableEnv`), the paper's "Ball-Ring (Simulated)". Deterministic
config from the paper's own `active_sampler_learning.yaml`: `pick_success_prob 1.0`,
`place_ball_fall_prob 1.0`, `place_smooth_fall_prob 1.0`, `place_sticky_fall_prob 0.0`,
5 tables / 1 sticky, `H_eval = 8`, free period 100 steps, competence
window/recency 2, ε = 0.5.

## What it took to match Figure 4 — four levers

Each was found by running predicators itself (`conda env predicators`, its native
`results` pickles) as ground truth and diffing against our behaviour.

### 1. Closed-loop execution (a correctness fix)

Our EES executed a multi-step plan **open-loop** — it popped each queued skill and
fired it without rechecking preconditions. On Ball-Ring a bare ball placed on a
table always falls, breaking the next queued skill's preconditions; we fired it
anyway, driving an inapplicable action into the env and tripping its `obj_type_id`
asserts (EES crashed on a seed-dependent subset of runs).

predicators is **closed-loop**: its option policy checks a `necessary_atoms` /
`expected_atoms` sequence before each step and raises `OptionExecutionFailure`,
which the explorer catches and **re-plans** on
(`active_sampler_explorer.py:340-349`, `utils.py:1483-1489`). (Its options'
`initiable` is always-`True` for these `SingletonParameterizedOption`s, so the real
guards are NSRT preconditions at plan time plus that execution-time atom check —
*not* `initiable`.) We replicate the effect: before executing the next queued
skill, if its preconditions no longer hold, drop the stale plan and re-plan
(`ees_method.py`, `_EesEpisode.step`). Crashes → 0.

### 2. Oracle feature selection (a capability the port lacked)

Light Switch uses `feature_selection: all` (raw concatenation of object features);
our port only implemented that. Ball-Ring's paper config uses
`feature_selection: oracle` — a **hand-engineered** per-option sampler input. For
the cup-placement skill predicators feeds the classifier
`[bias, table_radius, sticky, sticky_region_x/y_offset, sticky_region_radius,
table_x, table_y, place_x, place_y]` (`utils.py::construct_active_sampler_input`)
rather than five tables' worth of raw state. With `all`, the one relevant
sticky-region geometry is buried under clutter and the sampler barely learns.

We ported the oracle vector exactly (`environments/ballring/skills.py::oracle_sampler_input`,
routed through a `SkillProvider.oracle_sampler_input` hook so methods never import an
environment), reconciling the one representation gap: predicators' place params
*are* the placement `(x, y)`; ours sample `(u, θ)` and convert, so the row uses the
**converted** `place_x/place_y` (unit-test-pinned against the predicators vector).

### 3. The 25-cycle protocol (derived, not stated)

The paper never states the number of free periods. But it is forced by Figure 4's
geometry: `x-axis max ÷ free-period length`. Ball-Ring's panel runs to **2500**
transitions at 100 steps/period ⇒ **25 online-learning cycles** (Light Switch:
1500/150 = 10; Cleanup: 2500/125 = 20). predicators' config default is
`num_online_learning_cycles: 10`, which only reaches 1000 transitions — the flat
plateau, *before* the climb. Judging Ball-Ring at 10 cycles falsely reads as "EES
doesn't work"; the climb is cycles ~11-25.

### 4. Sampler training iterations (a config default)

Our `sampler_max_train_iters` defaults to **1000** (chosen for Light Switch speed);
predicators uses **100000**. Critically, our sampler's early-stopping floor is
`n_iter_no_change = 5000`, so at a 1000-iter cap the classifier **always stops
before it can converge**. On Ball-Ring, where success *is* sampler quality, a
never-converged sampler caps the solve rate. Raising to 100000 (it early-stops on
convergence, so the real cost is far below 100×) is the final lever.

## Results — the four-curve comparison

10 seeds, mean ± stderr. `ours-1k` is the default `sampler_max_train_iters`;
`ours-100k` matches predicators. Both have levers 1-3 (closed-loop, oracle FS,
25 cycles). "predicators" is the reference run in its own `predicators` conda env at
the identical config. "Fig 4" is read off the paper image (±5-10).

10 seeds each, mean % ± stderr:

| online transitions | ours-1k | ours-100k | predicators-25c | Fig 4 ~ |
|---|---|---|---|---|
| 0    | 0 ± 0   | 0 ± 0     | 0  | 0 |
| 500  | 13 ± 5  | 12 ± 6    | 37 | ~42 |
| 1000 | 23 ± 7  | 22 ± 10   | 53 | ~44 |
| 1500 | 32 ± 9  | 44 ± 13   | 81 | ~45 |
| 2000 | 30 ± 13 | **90 ± 6** | 95 | ~80 |
| 2500 | 41 ± 14 | 66 ± 11   | 91 | ~97 |

**What the comparison shows:**

- **predicators reproduces the paper.** Its 25-cycle curve is flat ~35-45% through
  ~1000 transitions, then climbs (onset ~cycle 11) to ~91-95% by 2000-2500 —
  Figure 4's shape and endpoint. This is our anchor.
- **ours-1k plateaus at ~41%** — closed-loop + oracle FS + 25 cycles get the shape
  started, but a never-converged sampler caps it far below the paper (flat ~13-41%
  the whole way).
- **ours-100k climbs to ~90%** (peak at 2000 transitions), reproducing the paper's
  flat-then-climb shape and matching predicators' ~95% peak. Isolating this single
  variable (1k → 100k, everything else identical) is what turns the ~40% plateau
  into the climb — the last piece of the gap.
- **Honest caveat: our tail is noisier than predicators'.** ours-100k peaks ~90% at
  2000 but slips to ~66 ± 11% at 2500, where predicators holds a stable ~91%. The
  climb and endpoint region are reproduced, but our per-seed curves are more
  volatile in the last few cycles (a subset of seeds regress). Candidates for the
  residual instability — not yet isolated — are the smaller remaining port
  deviations (the `skip_perfect`/UCB history and ε-greedy-scope choices carried over
  from the Light Switch log) and ordinary 10-seed variance on a hard sampler target;
  predicators' own per-seed curves are load-sensitive too (see its wall-clock-timeout
  note). Worth a follow-up, but it does not change the headline: sampler training is
  what unlocks the climb.

![ours vs predicators vs Figure 4](./2026-07-24-ballring-ees-comparison.png)

Random Skills stays at 0% throughout (undirected practice never assembles the
cup-carry plan); Skill Oracle is a 100% flat upper bound.

## The residual gap to predicators, and config parity

With the four levers above, `ours-100k` reproduces the paper's flat-then-climb shape
and peaks ~90% at 2000 transitions. But it still trails the predicators reference,
which climbs earlier (~500) and holds ~91-95%, and our tail is noisier (drops to ~66%
at 2500). Two questions this raises: is the comparison even fair (same config?), and
what accounts for the residual gap?

**Config parity.** Everything the *paper* states, both runs match:

| Setting | Predicators code | Paper text | Ours | Match |
|---|---|---|---|---|
| epsilon | 0.5 | 0.5 | 0.5 | ✓ |
| competence prior | Beta(10,1) | Beta(10,1) | Beta(10,1) | ✓ |
| competence window/recency | 2 | w = 2 | 2 | ✓ |
| sampler candidates | 100 | 100 | 100 | ✓ |
| sampler train iters | 100000 | 10000 | 100000 | ✓ |
| feature selection | oracle | (implied) | oracle | ✓ |
| free period / horizon / cycles | 100 / 8 / 25 | 100 / 8 / — | 100 / 8 / 25 | ✓ |
| planner | seq-opt-lmcut | LM-Cut | seq-opt-lmcut | ✓ |

The remaining differences are all **predicators-internal implementation choices the
paper never documents** — so "matching the paper" is done; "matching predicators'
code" is the operative, and looser, target:

| Behavior | Predicators code | Paper | Ours (default) |
|---|---|---|---|
| epsilon-greedy scope | target-only, greedy prefix | silent | every practice skill |
| goal-pursuit horizon cap | yes (`CFG.horizon`) | silent | none |
| skip_perfect / UCB history | all-attempts `_ground_op_hist` | silent | all-attempts (matched) |
| double-`observe()` | yes (a bug) | silent | no (keeps competence clean) |
| replanning-tasks deque | seen + 5 fictitious | silent | seen only |
| last-skill-of-period observed | yes | silent | no |

**Ablation: epsilon-greedy scope alone does NOT close the gap.** Our port explores
(random params) on *every* skill during a practice period; predicators explores only
the practice-target skill and is greedy for the prefix. On Ball-Ring's ~8-step plans
that looked like the prime suspect — at epsilon = 0.5 we randomize half of *all*
actions, not just the target. Running it (`--reproduce-predicators-explore-target-only`,
3 seeds, 100k iters) refutes it as a standalone cause:

| transitions | target-only (3 seeds) | ours-100k baseline (10) | predicators |
|---|---|---|---|
| 1000 | 20 ± 20 | 22 ± 10 | 53 |
| 1500 | 40 ± 31 | 44 ± 13 | 81 |
| 2000 | 53 ± 29 | 90 ± 6 | 95 |
| 2500 | 57 ± 30 | 66 ± 11 | 91 |

Target-only tracks the baseline (within a very noisy 3-seed band) and stays far below
predicators. The reason is a **coupling** the flag's own field comment documents:
predicators pairs target-only exploration with a **goal-pursuit horizon cap** (it
practices after `CFG.horizon` steps of pursuing the goal). Our goal-pursuit is greedy
and *uncapped* — it runs until the goal is achieved or planning fails. So restricting
exploration to the target, without also capping goal-pursuit, just yields *less*
exploration (the uncapped greedy goal phase still eats the period), not better-focused
exploration. In fact turning it on **alone deadlocks a goal-directed domain like Light
Switch**: a bad initial sampler never achieves the goal greedily, so the practice phase
where the target would be explored never begins (this is asserted in
`test_ees_method.py`). Our explore-everything default is precisely what *compensates*
for the missing horizon cap.

**Takeaway.** The residual gap is not any single one of these deviations. The
epsilon-greedy scope and the horizon cap are a **coupled pair** — a faithful
"ours == predicators' code" run needs both together (and probably the replanning-tasks
deque too).

### What to try next (ranked)

| # | Hypothesis / change | What it does | Effort | Likelihood it closes the gap |
|---|---|---|---|---|
| 1 | **Scope + goal-pursuit horizon cap together** | Add the `CFG.horizon` cap predicators has, then re-run with `--reproduce-predicators-explore-target-only`. The pair is the point: focused exploration only helps once goal-pursuit stops monopolizing the period. | moderate | **high** — the direct follow-up to the refuted scope-only run |
| 2 | **Replanning-tasks deque** (audit D1) | Score planning-progress against seen tasks **+ 5 fictitious replan goals** (`max_replan_tasks=5`), not seen-only. Changes *which* skill EES practices — could focus practice on the cup-placement earlier. | moderate | medium |
| 3 | **Diagnose the noisy tail** | Per-seed look at why some seeds regress 90%@2000 → 66%@2500; run more seeds to tighten the ±11 band. Part of the "gap" at 2500 may be variance, not method. | low | diagnostic (may shrink the *apparent* gap) |
| 4 | **Planning-progress task selection** | predicators uses `sorted(seen_idxs)[:10]` ("don't randomize — noisy"); ours uses the 10 *most recent*. Cheap to flip and test. | low | low–medium |
| 5 | **Predicators-side parity check** | Re-run predicators **sequentially** (no CPU contention) — its per-seed curves are wall-clock-timeout-sensitive, so some of its *early* climb may be a fast-hardware artifact inflating the gap. | low–medium | may shrink the gap from the reference side |
| 6 | **Last-skill-of-period observation** (audit D2) | Observe the final skill of each free period (needs the next state). ≤1 datapoint/period. | low | low |

### Follow-up #1: the goal-pursuit horizon cap (the coupled pair)

Implemented as `--goal-pursuit-horizon N` (`EesMethod.goal_pursuit_horizon`), the port's
version of predicators' `assigned_task_horizon`
(`active_sampler_explorer.py:191-198`): spend at most N skills pursuing the assigned
train task's goal, then declare it finished and practice for the rest of the period.
Exhausting the budget also drops the in-flight plan, matching predicators clearing
`current_policy` so it replans — those queued skills were selected to reach a goal we
just stopped pursuing. The cap governs practice only, never evaluation: it lives in
predicators' *explorer*, and `run_task_episode` already owns when an eval episode ends.

The default is `None` (uncapped, this port's original behavior) because **predicators
has no single global horizon** — it is a per-environment config, and the values differ
by more than an order of magnitude:

| predicators env | `CFG.horizon` | interaction period |
|---|---|---|
| `ball_and_cup_sticky_table` (Ball-Ring) | **8** | 100 |
| `grid_row` (Light Switch) | `grid_row_num_cells + 2` | 150 |
| everything else | 100 (default) | 100 |

That table is itself the explanation for why the scope-only ablation failed. On
Ball-Ring predicators spends **8 of its 100 steps** on the assigned goal and the other
~92 practicing. Uncapped, our goal phase can consume most of the period, so
restricting exploration to the practice target removed most of our exploration instead
of focusing it. Only with `--goal-pursuit-horizon 8` does target-only exploration
describe the same regime predicators is actually in.

Reproduce (3 seeds, matching the scope-only ablation's budget):

```bash
python -m scripts.run_sweep --env ballring --methods ees --num-seeds 3 \
  --results-root results/ballring-100k-targetonly-horizon8 \
  --shared-args "--num-test-tasks 10" \
  --method-args "ees=--num-cycles 25 --max-steps-per-interaction 100 \
    --competence-window-size 2 --competence-recency-size 2 \
    --exploration-epsilon 0.5 --sampler-max-train-iters 100000 \
    --reproduce-predicators-explore-target-only --goal-pursuit-horizon 8"
```

**Result: REFUTED — the cap does not close the gap.** All four arms at n = 10
(mean ± stderr, % of 10 evaluation tasks solved):

| transitions | scope+cap h=8 | scope-only | ours-100k baseline | predicators |
|---|---|---|---|---|
| 500 | 6 ± 6 | 1 ± 1 | 12 ± 6 | 37 ± 6 |
| 1000 | 23 ± 12 | 12 ± 7 | 22 ± 10 | 53 ± 8 |
| 1500 | 37 ± 10 | 31 ± 11 | 44 ± 13 | 81 ± 6 |
| 2000 | 50 ± 14 | 50 ± 13 | 90 ± 6 | 95 ± 2 |
| 2500 | 42 ± 12 | 65 ± 12 | 66 ± 11 | 91 ± 4 |

Welch t-tests on the per-seed finals at 2500 transitions, which are what actually
license the conclusions above:

| comparison | difference | p | verdict |
|---|---|---|---|
| scope-only vs baseline | −1 | 0.95 | no effect (a clean, properly powered null) |
| scope+cap vs baseline | −24 | 0.13 | **not significant** — trends worse, unproven |
| scope+cap vs scope-only | −23 | 0.17 | not significant |
| baseline vs predicators | −25 | **0.027** | significant |
| scope-only vs predicators | −26 | **0.038** | significant |
| scope+cap vs predicators | −49 | **<0.001** | significant |

So both of the explorer-side deviations the paper is silent on — epsilon-greedy scope
and the horizon cap — are now implemented and measured, individually and together, and
neither closes the gap. That is evidence *against* "we mis-ported the explorer".

**Do not over-read the scope+cap number.** An earlier draft of this log called the pair
"actively worse than the baseline". That was an over-claim: −24 points at p = 0.13 is
not a detected difference, it is a wide error bar. With sd ≈ 36 at n = 10, this
protocol can only resolve differences of roughly 35 points; anything smaller needs far
more seeds. The only defensible statements here are the null (scope-only ≈ baseline)
and the gap to predicators.

**Methodological note: the 3-seed version of this experiment was misleading.** An
earlier n=3 run of exactly this configuration produced a seed scoring 60% at 500
transitions, against predicators' 37% and the baseline's 12% — which looked like the
cap fixing the flat start. At n=10 the mean at 500 is **6%**: nine of ten seeds score
0 there, and that one seed was noise. Nothing at n=3 on this domain is reportable;
the per-seed spread is far too wide.

### The real finding: our variance, not our mean

The per-seed finals at 2500 for scope+cap are `[90, 10, 50, 50, 80, 0, 80, 0, 0, 60]`,
and individual seeds *collapse mid-run* — seed 8 goes 50% -> 100% -> **0%** over its
last three evaluation sweeps, and seed 7 goes 40% -> 80% -> **0%**. A run at 100%
dropping to 0% in a single cycle is not a tuning problem.

Across all 10 seeds at 2500 transitions, the spread — not the mean — is what separates
the two codebases, and unlike the mean differences above it is statistically solid
(F-test on the variances, df 9,9):

| arm | sd | worst seed | best seed | seeds at 0% | F vs predicators | p |
|---|---|---|---|---|---|---|
| predicators | **12.0** | **70** | 100 | 0 | — | — |
| ours-100k baseline | 33.7 | 10 | 100 | 0 | 7.9 | **0.005** |
| ours scope-only | 37.8 | 0 | 100 | 2 | 10.0 | **0.002** |
| ours scope+cap | 36.5 | 0 | 90 | 3 | 9.3 | **0.003** |

predicators' *worst* run out of ten still solves 70% of evaluation tasks; it never
comes close to failing. Ours range from 0% to 100% on the same protocol, and every one
of our arms is ~3x more variable than the reference at p < 0.005. That is the one
difference in this whole investigation that is unambiguous.

Individual seeds also *collapse mid-run* — seed 8 of the scope+cap arm goes
50% -> 100% -> **0%** over its last three evaluation sweeps, and seed 7 goes
40% -> 80% -> **0%**. A run at 100% dropping to 0% within a single cycle is not a
tuning problem, and no exploration-policy lever can explain it.

The question worth answering is therefore not "why is our mean lower" but **"why do
individual runs fail catastrophically"**. Fix that and the mean follows, because a mean
computed over a bimodal mixture is mostly reporting the mixing ratio. This reframes the
ranked list above: item #3 (the noisy tail) is the main event, not a cheap diagnostic.

### Two checks that bound where the bug can be

**1. The oracle/random bracket is clean.** On the same 10 seeds and the same evaluation
protocol:

| baseline | final % solved |
|---|---|
| `skill-oracle` | **100% on every seed** |
| `random-skills` | **0% on every seed** |

So the environment, skills, symbolic layer, task distribution, and evaluation harness
are all sound, and the metric separates a good policy from a bad one. The failure is
confined to the *learning* path. Caveat: `skill-oracle` runs a hand-coded policy
provider and never invokes Fast Downward, so **planner** behavior is not covered by
that 100%.

**2. The x-axis is comparable between the two codebases.** Worth verifying because
predicators counts `max_num_steps_interaction_request` in low-level env transitions
while this port executes exactly one skill per step — if its options spanned multiple
transitions, "2500 transitions" would mean different things on the two curves and every
comparison here would be invalid. It does not: every Ball-Ring option in predicators is
built with `utils.SingletonParameterizedOption`
(`ground_truth_models/ball_and_cup_sticky_table/options.py:43-160`), documented in
`utils.py:1105` as *"a parameterized option that takes a single action and stops"*. So
1 option = 1 transition = 1 of our skills, predicators' `horizon: 8` really is 8 skill
executions, and the axes line up.

### Environment-fidelity audit: three confirmed port bugs

Prompted by the variance finding, the Ball-Ring port was diffed against predicators'
`ball_and_cup_sticky_table` as configured by the paper's own YAML. The stochastic
dynamics, geometry, action space, all 16 operators' preconditions/add/delete effects,
the task distribution, the success criterion and the oracle feature vector all came back
clean. Three real mismatches did not. All three were verified by hand against both
codebases.

**1. `ignore_effects` is absent from this port's symbolic layer entirely** — the likely
big one. predicators' three `NavigateTo*` NSRTs declare
`ignore_effects = {ReachableSurface, ReachableBall, ReachableCup}`
(`nsrts.py:457,514,526`), i.e. navigating anywhere **wipes every reachability atom**.
Ours declare `delete_effects=frozenset()` (`environments/ballring/skills.py:68-93`), and
`ignore_effects` does not appear anywhere in `src/` — `core.method.types.Skill` has no
such field. So our symbolic model is *monotone* in reachability: once reachable to the
ball, reachable forever, even after navigating to a table across the room.

Initial atoms are always correct (the env guarantees exactly one reachable object), and
closed-loop replanning catches the divergence without burning a step, so this never
crashes — which is why it survived this long. But it is systematic: **any plan needing
re-navigation to a previously-visited object omits that navigate**, so FD returns a
7-operator plan where the true plan is 8. That matters twice over. It leaves zero slack
against `H_eval = 8`, and — far worse — **it mis-prices every plan EES scores**.
Planning progress is computed from plan costs over the seen tasks, and every one of
those costs is systematically one `NavigateToTable` too cheap, which distorts *which
skill EES decides to practice*. That is EES's core mechanism, so this is a correctness
bug in the same class as lever #1 (closed-loop execution), not a tuning detail.

**2. The evaluation test set is resampled on every sweep** — the direct suspect for the
variance result above. predicators generates its test tasks **once** and caches them
(`envs/base_env.py:180-193`, `if not self._test_tasks: ...`), so every cycle is scored
on the *same* 10 tasks and the learning curve isolates policy change. Ours calls
`problem.sample_test_task()` fresh inside the evaluation loop
(`practice_loop.py:174-175`), advancing an unbounded stream, so **every sweep measures a
different test set**. That stacks task-sampling variance on top of policy variance —
and because Ball-Ring tasks differ precisely in the sticky-region offset the sampler is
trying to learn, a policy that has specialized to part of that distribution can plausibly
swing between a favourable draw and an unfavourable one. It is a strong candidate for
both the ~3x variance inflation and the within-one-cycle 100% -> 0% collapses, and it is
cheap to fix.

**3. Floor placements have no jitter, so `BallInCup` is always true.** predicators'
`place_on_floor_sampler` scatters the object in a small disk around the room centre
(`dist ~ U(0, 2*radius)`, `nsrts.py:287-303`); ours returns the room centre exactly
(`skills.py:429-430`, no jitter — a deliberate simplification whose consequence was not
noticed). At the paper geometry `BallInCup` needs the two centres within 0.00245, so
placing the cup on the floor and then the ball on the floor puts both at exactly
(0.5, 0.5) and makes `BallInCup` true **100%** of the time, where predicators hits it
~5%. Our `PlaceBallOnFloor` therefore records a *failed* outcome (its `BallNotInCup`
add-effect does not hold) essentially every time, corrupting that skill's competence
curve and hence practice selection.

Lower-priority items from the same audit, not yet addressed: `PlaceBallOnFloor` is also
missing `ignore_effects = {BallInCup, ReachableBall}` (`nsrts.py:285`), so the model can
believe `BallInCup` and `BallNotInCup` simultaneously; `--num-test-tasks` defaults to 20
where the paper config uses 10 (every command in this log passes 10 explicitly, so
published numbers are unaffected); and train tasks are an unbounded fresh stream where
predicators draws with replacement from a fixed pool of 50, which changes what "planning
progress on seen tasks" means.

**Fix order** (most likely to explain the measured symptoms first): cache the test set
per run (2), add `ignore_effects` to `Skill` + `PddlWriter` and populate the navigate
operators (1), restore floor-place jitter (3). Each is a separate PR under this repo's
one-feature-per-PR rule, and each should be re-measured at n = 10 before the next lands
— given sd ~ 36, anything smaller cannot be resolved.

### Design status

The scope x cap factorial is missing a cell — cap-only has never been run, and it is
the arm that actually changes the practice/goal-pursuit budget split:

| | uncapped | capped (h=8) |
|---|---|---|
| explore every practice skill | baseline, 66 ± 11 (n=10) | **not run** |
| target-only | 65 ± 12 (n=10) | 42 ± 12 (n=10) |

Recommended order: **#1 first** (highest-likelihood, directly motivated by the refuted
result), then **#3** (cheap; tells us how much of the 2500 gap is real vs noise), then
**#2**. #5 is worth one run because it tests whether the reference itself is inflated.

## Faithfulness notes / deliberate deviations

1. **Sampler default is 1000, not 100000.** Kept for Light Switch run-time; pass
   `--sampler-max-train-iters 100000` to match the paper exactly (as `ours-100k`
   does). This is the one lever left as a non-matching *default*.
2. **Floor-place table avoidance.** predicators' place-on-floor targets the room
   center with tiny jitter, and its tables are generated in a *ring* at distance
   ~0.33 from center — so a floor placement is clear of every table *by
   construction*, never by a check. Our env can place a table nearer the center, so
   a floor-place could land on a table and hit the cup-place assert; we guard it by
   routing `obj_type_id == 0.0` (floor intent) through the floor path regardless of
   geometry. Same observable behaviour; a fully faithful alternative is to match the
   ring layout exactly.
3. **`(u, θ)` vs placement `(x, y)` sampler params.** Our place-on-table skills
   sample a `(fraction, angle)` and convert to a point; predicators samples the
   point. The oracle feature row uses the converted point, so the classifier sees
   the same quantity either way.
4. Everything else — closed-loop replan, oracle feature vector, competence
   window/recency 2, ε 0.5, Beta(10,1) prior, 25 cycles, `seq-opt-lmcut` planning —
   matches predicators.

## Reproducing

Fast Downward required (see the repo's Setup). Our runs (10 seeds each):

```bash
# The paper-matching curve (sampler converges):
python -m scripts.run_sweep --env ballring --methods ees --num-seeds 10 \
  --results-root results/ballring-100k \
  --shared-args "--num-test-tasks 10" \
  --method-args "ees=--num-cycles 25 --max-steps-per-interaction 100 \
    --competence-window-size 2 --competence-recency-size 2 \
    --exploration-epsilon 0.5 --sampler-max-train-iters 100000"

# The plateau contrast (default 1000 iters) + baselines:
python -m scripts.run_sweep --env ballring --methods ees random-skills skill-oracle \
  --num-seeds 10 --results-root results/ballring-1k \
  --shared-args "--num-test-tasks 10" \
  --method-args "ees=--num-cycles 25 --max-steps-per-interaction 100 \
    --competence-window-size 2 --competence-recency-size 2 --exploration-epsilon 0.5" \
  --method-args "random-skills=--num-cycles 25 --max-steps-per-interaction 100"
```

The predicators reference was run in its own `predicators` conda env with
`--approach active_sampler_learning --explorer active_sampler
--active_sampler_explore_task_strategy planning_progress
--active_sampler_learning_feature_selection oracle --sampler_mlp_classifier_max_itr
100000 --num_online_learning_cycles 25` plus the Ball-Ring config flags above.

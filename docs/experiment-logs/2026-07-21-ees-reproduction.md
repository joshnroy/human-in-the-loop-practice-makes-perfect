# EES (Practice Makes Perfect) reproduction on Light Switch

Porting the paper's own method — EES, Estimate/Extrapolate/Situate — from the
reference `predicators` implementation and reproducing its Light Switch result.
Companion to [the Random Skills baseline log](./2026-07-21-random-skills-baseline.md),
which established that undirected practice gets 0% on this environment.

## What EES is, mechanically

Three steps, each a distinct piece of the port:

| Step | What it does | Where it lives |
|------|--------------|----------------|
| **Estimate** | Beta-Bernoulli posterior competence per *ground* skill, from whether that skill's `add_effects` actually held after each execution | `competence_models.py` |
| **Extrapolate** | "How competent would this skill be after one more cycle of practice?" — current competence plus the best per-cycle improvement observed so far | `competence_models.py` |
| **Situate** | Substitute the extrapolated competence, re-price the *seen tasks'* cached plans, and practice whichever skill most reduces their total cost — planning to that skill's preconditions to reach somewhere it's executable | `ees_method.py` + `planning/fast_downward.py` |

The identity that makes this work: plan cost is `sum(-log(competence))`, so
minimizing it maximizes `prod(competence)` — the paper's `J_task`, the probability
a plan executes without replanning. That is exactly why a **cost-aware optimal**
planner is load-bearing and predicators' own built-in A* is not a substitute (it
ignores per-operator costs entirely). This port shells out to a real Fast Downward
with `seq-opt-lmcut`, patching per-ground-skill costs into the translated SAS file
— predicators' own three-stage protocol.

## Protocol

Taken from the paper's own experimental section wherever it states a number:

| Setting | Value | Source |
|---------|-------|--------|
| Light Switch grid size | 25 cells | paper ("in our main experiments") |
| Evaluation horizon `H_eval` | 27 (= cells + 2) | paper |
| Steps per free period | 150 | paper |
| Evaluation tasks per checkpoint | 10, held-out | paper |
| Seeds | 10 | paper |
| Exploration | epsilon-greedy, ε = 0.5 | paper |
| Competence prior | Beta(10, 1) | paper |
| Planning-progress tasks | 10 most recent | paper |
| Replan frequency | once per 100 scoring calls | paper |
| Online learning cycles | 10 | **predicators' default** — the paper never states its free-period count |

Command per run:

```bash
python -m hitl_pmp.cli --env lightswitch --method ees --grid-size 25 \
  --num-cycles 10 --max-steps-per-interaction 150 \
  --num-test-tasks 10 --seed <s> --output-dir <results>/ees/<s>
```

then `python -m analysis.practice_makes_perfect.ees --results-root <results> --output <png>`.

## Results

Mean fraction of the 10 held-out evaluation tasks solved, across **10 seeds**
(± standard error), at each checkpoint:

| Online transitions | EES | Random Skills | Skill Oracle |
|--------------------|-----|---------------|--------------|
| 0 (before practice) | 0.0% | 0.0% | 100% |
| 150   | **50.0%** ± 7.0 | 0.0% | 100% |
| 300   | **89.0%** ± 6.0 | 0.0% | 100% |
| 450   | **95.0%** ± 4.0 | 0.0% | 100% |
| 600   | **98.0%** ± 2.0 | 0.0% | 100% |
| 750   | **98.0%** ± 2.0 | 0.0% | 100% |
| 900   | **100.0%** ± 0.0 | 0.0% | 100% |
| 1050 – 1500 | **100.0%** ± 0.0 | 0.0% | 100% |

EES reaches the privileged oracle's success rate — from a standing start of 0% —
after roughly **900 online transitions** (6 free periods), and is already at 89%
after two. `skill-oracle` cheats with privileged ground-truth state and never
practices, so it is a flat upper bound rather than a curve; `random-skills`
collects the identical transition budget and never solves anything.

![EES vs baselines learning curve](./2026-07-21-ees-vs-baselines-light-switch.png)

An EES episode after training — it walks to the light and sets the dial in one
correct move, because the `TurnOnLight` sampler has been specialized away from its
uniform prior:

![EES trained episode](./2026-07-21-ees-trained-episode.gif)

## Comparison to the paper

**What can and cannot be compared.** The paper's Light Switch result is Figure 4,
which in the source available to us is *an image only* — the transcription carries
its caption and axes but no per-curve numbers, and the body text gives no
Light-Switch numbers either. So there are no published values to diff against.
What the paper does state in prose is compared below; no numeric curve comparison
is claimed, and none should be inferred from the chart above.

| Paper's claim (Light Switch) | This reproduction |
|------------------------------|-------------------|
| "EES is consistently the most sample efficient, achieving higher success rates after fewer online transitions than the baselines" | **Reproduced.** EES is the only practicing method that improves at all here: 0% → 100% in ~900 transitions, versus 0% for Random Skills over the same budget. |
| "Like the Random Skills baseline, MAPLE-Q fails to solve any evaluation tasks" (Random Skills ≈ 0%) | **Reproduced.** Random Skills scores exactly 0.0% at every checkpoint, all 10 seeds. (MAPLE-Q is not ported yet — it is pure deep RL and shares none of this scaffolding.) |
| "The main challenge in this environment is for the robot to specialize its parameter prior for the ToggleLight skill" | **Reproduced, and measured directly.** Probing the trained sampler over 200 fresh targets: mean |dlight − target| falls from **0.781** under the uniform prior to **0.028** learned, and the fraction of draws landing inside the 0.1 `light_on_tolerance` rises from **10% to 100%**. That specialization is the whole result — it is what turns a 0% policy into a 100% one. |
| JumpToLight "is impossible and never achieves its purported effect" | **Reproduced.** After training, EES's competence estimates are TurnOnLight 0.995, TurnOffLight 0.993, MoveRobot 0.917, and **JumpToLight 0.114** — it learned the impossible skill is impossible. Since plan cost is −log(competence), JumpToLight costs ~2.17 against ~0.005 for TurnOnLight, roughly a 400× penalty, so cost-aware planning routes around it rather than being trapped by it. |

## A bug this experiment caught

The first version of this port updated the competence model on **every** practice
attempt, including the ones where the epsilon-greedy branch deliberately chose a
*random* parameter. Success rate still hit 100%, so the headline curve looked fine —
but probing the trained model showed `TurnOnLight` competence at **0.575** while the
policy was solving 10/10 evaluation tasks, which is incoherent.

The cause: at the paper's ε = 0.5, half of all attempts are coin flips by
construction, so "competence" was measuring how often a coin flip works rather than
how good the skill is when the robot actually tries. predicators suppresses exactly
this update (`active_sampler_learning_approach.py` lines 442-443, keyed off the
`epsilon_bool` its sampler returns). After fixing it:

| Skill | Competence before fix | After fix |
|-------|----------------------|-----------|
| TurnOnLight | 0.575 | **0.995** |
| TurnOffLight | 0.897 | **0.993** |
| MoveRobot | 0.917 | 0.917 |
| JumpToLight (impossible) | 0.769 | **0.114** |

Note it is the *impossible* skill that moved most, and in the right direction. The
buggy version could not tell JumpToLight (0.769) from a mastered skill (0.575) —
it had them backwards — which is precisely the discrimination EES's whole mechanism
depends on, since those numbers become the planner's edge costs. The end-to-end
success curve alone would never have surfaced this; the per-skill probe did. Both
the pre- and post-fix numbers above come from the same seed-0 configuration, and the
10-seed sweep reported above was re-run from scratch on the fixed code.

## Faithfulness notes

Where this port deliberately differs from `predicators`, and why:

1. **One skill = one raw action.** Every Light Switch skill's `compute_action`
   completes in a single step, so there is no option-termination loop.
   predicators runs options over many low-level steps. Nothing in EES's logic
   depends on the difference here.
2. **The last skill of an interaction period is never scored.** Its outcome would
   need a subsequent state to check `add_effects` against. predicators observes at
   option termination instead. Costs at most one datapoint per period.
3. **predicators' double-count bug is not reproduced.** It calls `observe()` twice
   per non-exploratory attempt (`active_sampler_learning_approach.py` lines 407 and
   443); that is a bug, so this port observes once.
4. **`ToggleLight`'s prior is U(-1, 1), not U(0, 2π).** The paper's text says the
   latter; the reference *code* uses the former, and per this project's convention
   the codebase is ground truth where the two disagree. Inherited from the existing
   Light Switch port, not introduced here.
5. **Sampler training iterations default to 1000, not 100000.** The paper's config
   uses the larger value; the default here keeps a run to minutes. Raise
   `--sampler-max-train-iters` to match exactly.
6. **Only the "optimistic" competence model is ported**, because
   `CFG.skill_competence_model = "optimistic"` is what EES actually runs. The
   paper also describes an EM/latent-variable variant it does *not* use for its
   main results.

## Reproducing

Fast Downward is required and is not vendored — see CLAUDE.md's Setup section.
The full sweep is the script embedded above, run once per (method, seed).

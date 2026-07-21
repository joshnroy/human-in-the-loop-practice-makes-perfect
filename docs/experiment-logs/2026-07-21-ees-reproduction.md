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

RESULTS_TABLE_PLACEHOLDER

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
| "EES is consistently the most sample efficient, achieving higher success rates after fewer online transitions than the baselines" | REPRO_EES_PLACEHOLDER |
| "Like the Random Skills baseline, MAPLE-Q fails to solve any evaluation tasks" (Random Skills ≈ 0%) | REPRO_RANDOM_PLACEHOLDER |
| "The main challenge in this environment is for the robot to specialize its parameter prior for the ToggleLight skill" | REPRO_SAMPLER_PLACEHOLDER |
| JumpToLight "is impossible and never achieves its purported effect" | REPRO_JUMP_PLACEHOLDER |

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

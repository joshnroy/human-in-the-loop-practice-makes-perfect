# methods

This folder is where concrete `core.Method` implementations live — the agent/baseline
side of the codebase, mirroring `core.Problem` as described in `../core/README.md`. A
concrete `Method` is a real, constructor-injected pydantic instance now (not a
static-method container), with a required `env: Environment` field on the base
`Method` class (the one piece of context every `Method` concretely needs to act at
all) — `reset_environment`, `get_task_policy`, `generate_train_task`,
`execute_setup_command`, `execute_skill`, and `improve_skill_parameters` are all
ordinary instance methods (`self`, not `@staticmethod`).

## `oracle/` — privileged-knowledge baselines, wrapped as `core.Method`s

`SkillOracleMethod` (`oracle/skill_oracle_method.py`) wraps
`environments/lightswitch/skill_oracle_policy.py`'s `SkillOraclePolicy` as a real
`core.Method`, the first one actually wired through `PracticeLoop`, via
`oracle/cli.py`'s `SkillOracleCli` (`--method skill-oracle`) —
`environments/lightswitch/cli.py`'s `LightSwitchCli` itself no longer has its own
`run()` loop at all, since `--method` (not `--env` alone) is now how anything
actually runs. `SkillOraclePolicy` itself only knows Light Switch's own oracle logic
(same as `ActionOraclePolicy`); the `isinstance(self.env, LightSwitchEnvironment)`-
keyed dispatch that lets `SkillOracleMethod` stay domain-agnostic at its entrypoint
lives on `SkillOracleMethod` itself, not on the policy it wraps — `self.env` is a
real constructor-injected field inherited from the base `Method` class (the
`Environment` instance this particular `SkillOracleMethod` was built with), and that
dispatch is a *method* concern, not something a single domain's own policy file
should need to know about its siblings. `ActionOraclePolicy` isn't
wrapped/wired the same way (its raw-action-space oracle is exercised directly by its
own tests, not through the CLI) — only one oracle needed wiring to prove out the
pattern.

## `practice_makes_perfect/` — reproducing the original PMP/EES paper

Before this project's own novel baselines (below, still unimplemented), this subfolder
ports the original "Practice Makes Perfect" paper's own method (EES —
Estimate/Extrapolate/Situate) and every baseline it compares against in its own
evaluation (Fail Focus, Competence Gradient, Skill Diversity, Task-Relevant, Task
Repeat, Random Skills, MAPLE-Q), faithfully, as a reproduction exercise on Light
Switch — the only environment this codebase has so far. This is a pure repro of the
*reference paper's* results, not this project's own human-in-the-loop research
contribution; see `../../../CLAUDE.md` and `../planning/README.md` for why real Fast
Downward (not a hand-rolled substitute) is used for task planning here.

`RandomSkillsMethod` (`random_skills_method.py`, the first of the 8 paper approaches
to land) uniformly samples among the currently-applicable ground skills each step
(via `planning.grounding.SkillGrounder`) and executes one — no planning, no
competence model, no sampler learning, matching predicators' own
`RandomOptionsApproach`. Same split as `oracle/`:
`environments/lightswitch/random_skills_policy.py`'s `RandomSkillsPolicy` holds the
actual Light-Switch-specific grounding/sampling logic; `RandomSkillsMethod` itself
just dispatches on `isinstance(self.env, LightSwitchEnvironment)` and adds the one
thing no other `Method` needed yet — its own `seed`/RNG stream, since skill/param
sampling needs a source of randomness independent of task sampling. Wired in via
`cli.py`'s `RandomSkillsCli` (`--method random-skills`); `analysis/
practice_makes_perfect/random_skills.py` reads its `--output-dir` `stats.json`
output back in for reporting (never drives it directly — see the root `analysis/`
convention in `../../../CLAUDE.md`).

`EesMethod` (`ees_method.py`) is the paper's **own** method — the reproduction's
centrepiece — ported from predicators' `active_sampler_learning` approach plus its
`active_sampler` explorer under `active_sampler_explore_task_strategy=planning_progress`
(the exact combination `scripts/configs/active_sampler_learning.yaml` runs). Its three
named steps map onto three pieces here:

- **Estimate** — `competence_models.py`'s `OptimisticSkillCompetenceModel`, one per
  *ground* skill ever executed, updated with whether that skill's own `add_effects`
  actually held afterward. Beta-Bernoulli posterior mean under the paper's stated
  Beta(10, 1) prior.
- **Extrapolate** — that model's `predict_competence(num_additional_data=1)`: how
  competent would this skill be after another cycle's practice?
- **Situate** — the extrapolated competence is substituted into the cost dict and the
  *seen tasks'* cached plans are re-priced; the skill whose hypothetical improvement
  most reduces the cost of plans the robot actually needs wins, and EES then plans to
  that skill's preconditions in order to practice it somewhere it's executable.

Plan cost is `sum(-log(competence))`, so minimizing it maximizes `prod(competence)` —
the paper's `J_task`, the probability a plan runs without replanning. That identity is
why `../planning/fast_downward.py` (real Fast Downward, `seq-opt-lmcut`, per-ground-skill
costs patched into the translated SAS) is load-bearing rather than a convenience.
`wrapped_sampler.py` is the other learnable half: one `LearnedSkillSampler` per skill
name, a torch MLP binary classifier over `[1.0] + state_features + params` that scores
candidate parameters, refit from scratch each cycle, epsilon-greedy (0.5) while
practicing. Wired in via `cli.py`'s `EesCli` (`--method ees`); `analysis/
practice_makes_perfect/ees.py` renders the paper's own Figure 4 view (fraction of
evaluation tasks solved vs. online transitions) from the resulting `stats.json` files.

Practice happens through `core.Method`'s `get_practice_policy`/`end_cycle` pair (see
`../core/method/method.py`): exploration and data collection are confined to the
interaction period, so the held-out evaluation sweep never trains on itself. The
remaining six paper baselines (Fail Focus, Competence Gradient, Skill Diversity,
Task-Relevant, Task Repeat, MAPLE-Q) are still stacked follow-ups — each is a different
`score_ground_skill` on the same scaffolding, except MAPLE-Q which is pure deep RL.

The actual online-learning loop, `PracticeLoop`, lives at the top level
(`../../practice_loop.py`, alongside `../../cli.py`) rather than here, since it's the
one execution harness every `core.Method` runs through — oracles included (see
`oracle/` above) — not something specific to this paper reproduction. See its own
docstring for the exact reset semantics and how its optional `renderer` param works;
`problem`/`method`/`metrics` are real instances a caller constructs first (via its
own composition root, e.g. `../environments/lightswitch/cli.py`'s
`LightSwitchCli.run_method`) and simply passes in — there's no separate
`Problem.env`/`Problem.tasks`-style global wiring step to remember beforehand,
unlike the old ClassVar-singleton design. Driving `PracticeLoop` from a CLI
(constructing a fresh `Metrics()`, printing a success-rate summary, writing
`episode.mp4`) is `../../method_runner.py`'s `MethodRunner` — also top-level
and domain-/method-agnostic, called by every domain's own `<Domain>Cli.run_method`
(e.g. `../environments/lightswitch/cli.py`'s `LightSwitchCli.run_method`) so that
generic tail is written once rather than copy-pasted into each domain's `cli.py`.

`core.Metrics` (`../core/README.md`'s "`Metrics` is fully concrete" section) is what
`PracticeLoop` records evaluations into — used directly, no Light-Switch-specific
subclass, since nothing in this reproduction needs different behavior than the
generic default yet.

## This project's own planned baselines

Per the design doc's "Baselines" section, baselines form a progression, each expected to
fail in specific, documented ways that the eventual proposed method should not:

- A trivial baseline: a pre-set, fixed skill list plus TAMP-only planning, with no
  learning and no ability to acquire new skills. Expected to fail whenever the task
  distribution requires something outside its fixed skill set, or to get stuck behind an
  irreversible action it has no recovery skill for.
- `planning_to_practice.py` — extends the "Predicators"-style planning-to-practice
  baseline (see the sibling `hitl-practice` repo's TAMP conventions, e.g.
  `predicators/envs/`) with the ability to (re-)set the environment to any state via
  `Problem.execute_human_command`. Expected failure modes include burning excessive
  human-help cost if it resets more often than necessary, or thrashing when its skill
  library can't cover a gap.
- `pure_vla.py` — a baseline with no online learning at all: a stand-in for "a big
  pretrained, deep, model-free RL / VLA policy" that is simply rolled out as-is.
  Expected to fail on tasks outside its pretraining distribution, with no mechanism to
  recover via human help or skill improvement.
- `in_context_vla.py` — a VLA baseline that adapts within a single context window
  (in-context/few-shot) rather than via weight updates. Expected to fail once the
  needed adaptation exceeds what fits in context, or when it can't distinguish an
  irreversible mistake from a recoverable one.

## Files

- `__init__.py` — empty, marks this as a package.
- `cli.py` (per concrete method, once one exists) — same convention as
  `environments/<domain>/cli.py` (see
  [`../environments/README.md`](../environments/README.md)): a static-method
  container exposing `add_arguments(*, parser)`/`run(*, args)`, registered by name in
  `hitl_pmp/cli.py`'s registry, so a method's own hyperparameters become named CLI
  flags the same way an environment's do — no positional arguments, no
  method-specific knowledge in `hitl_pmp/cli.py` itself.

See [`../core/README.md`](../core/README.md) for the abstract `Method` interface
these will implement, and the design doc for this project's own baseline
progression and their documented expected failure modes.

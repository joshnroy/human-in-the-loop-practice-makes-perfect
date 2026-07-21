# methods

This folder is where concrete `core.Method` implementations live — the agent/baseline
side of the codebase, mirroring `core.Problem` as described in `../core/README.md`. A
`Method` implements `reset_environment`, `get_task_policy`, `generate_train_task`,
`execute_setup_command`, `execute_skill`, and `improve_skill_parameters`.

## `practice_makes_perfect/` — reproducing the original PMP/EES paper

Before this project's own novel baselines (below, still unimplemented), this subfolder
will port the original "Practice Makes Perfect" paper's own method (EES —
Estimate/Extrapolate/Situate) and every baseline it compares against in its own
evaluation (Fail Focus, Competence Gradient, Skill Diversity, Task-Relevant, Task
Repeat, Random Skills, MAPLE-Q), faithfully, as a reproduction exercise on Light
Switch — the only environment this codebase has so far. This is a pure repro of the
*reference paper's* results, not this project's own human-in-the-loop research
contribution; see `../../../CLAUDE.md` and `../planning/README.md` for why real Fast
Downward (not a hand-rolled substitute) is used for task planning here.

The actual online-learning loop, `PracticeLoop`, lives at the top level
(`../../practice_loop.py`, alongside `../../cli.py`) rather than here, since it's the
one execution harness every `core.Method` runs through — oracles included, once
they're wrapped as `Method`s — not something specific to this paper reproduction. See
its own docstring for the exact reset semantics, the `Problem.env`/`Problem.tasks`
wiring a caller must set up first, and how its optional `renderer` param works.
`SkillOracleMethod` (`environments/lightswitch/skill_oracle_policy.py`) is the
first `core.Method` actually wired through `PracticeLoop`, via
`environments/lightswitch/cli.py`'s `SkillOracleCli` (`--method skill-oracle`) —
`LightSwitchCli` itself no longer has its own `run()` loop at all, since
`--method` (not `--env` alone) is now how anything actually runs.
`ActionOraclePolicy` isn't wrapped/wired the same way (its raw-action-space
oracle is exercised directly by its own tests, not through the CLI) — only one
oracle needed wiring to prove out the pattern. `RandomSkillsMethod` (the first
of the 8 paper approaches) hasn't landed yet — tracked as a stacked follow-up,
along with the actual competence model, sampler learning, and Fast Downward
planning integration.

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

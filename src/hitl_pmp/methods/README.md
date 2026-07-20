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

This PR adds the first concrete baseline:

- `random_skills_method.py` — `RandomSkillsMethod`, the first concrete `Method` (of
  the 8 paper approaches) and the simplest: no planning anywhere, not even at
  evaluation time. Each step, uniformly samples one currently-applicable
  `GroundSkill` (via `planning.grounding.SkillGrounder`) and executes it with its
  own base sampler's params. Never pursues a task's actual goal, matching the
  paper's own near-0% Random Skills curve.

`core/metrics/metrics.py` also gains `record_evaluation`/`task_training_curve`, with
`environments/lightswitch/metrics.py`'s `LightSwitchMetrics` as the first concrete
implementation — but nothing yet drives an actual online-learning loop that calls
`record_evaluation`; that's `OnlineLearningRunner`, which lands in the next PR (the
first one to actually need both a `Method` and a `Metrics` together).

The competence model, sampler learning, real Fast Downward planning integration, and
each of the remaining 7 paper approaches (EES itself, Fail Focus, Competence
Gradient, Skill Diversity, Task-Relevant, Task Repeat, MAPLE-Q) land in further
stacked follow-up PRs, one per baseline.

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

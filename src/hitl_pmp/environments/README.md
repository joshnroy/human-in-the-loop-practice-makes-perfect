# environments

This is where **concrete** `Environment` implementations live — one subfolder per
domain, e.g. `environments/lightswitch/` or a future `environments/tossing_room/`.
See [`../core/README.md`](../core/README.md) for why `Environment` is the one
real-world/ground-truth instance for a domain (not a reusable dynamics function for
hypothetical planning), with no notion of tasks, humans, or reset cost.

## Convention for a domain subfolder

Each domain subfolder is expected to contain:

- `environment.py` — a concrete subclass of `core.Environment`: a real,
  constructor-injected instance now (e.g. `LightSwitchEnvironment(grid_size=10)`),
  not a static-method container — the domain's own dynamics (`take_action`,
  `get_valid_actions`, `get_current_state`/`set_state`/`hard_reset`) are ordinary
  instance methods (`self`, not `@staticmethod`) operating on that instance's own
  tracked `current_state`. Genuine per-run config (e.g. `LightSwitchEnvironment`'s
  `grid_size`/`canonical_light_target`) are real constructor fields; genuine
  structural constants that never vary between instances (`robot_type`/`light_type`/
  `action_space`/etc.) stay `ClassVar`, along with two specific tolerances
  (`light_on_tolerance`/`same_position_tolerance`) that stay `ClassVar` for a
  narrower reason — see `LightSwitchEnvironment`'s own docstring (module-level
  `Predicate` singletons in `predicates.py` read them via a late-bound class lookup,
  since `Predicate.holds`'s fixed `(state, objects)` signature has no per-instance
  slot to pass an `Environment` instance through). No tasks, no humans, no reset
  cost — just the physics/logic of the domain.
- `tasks.py` — a concrete subclass of `core.Tasks`: also a real, constructor-injected
  instance now, requiring the specific `Environment` instance it samples against as
  a constructor field (`core.Tasks.env`) — e.g. `LightSwitchTasks(env=env, seed=...)`
  needs that instance's own `grid_size` to place the light/cells correctly via
  `env.build_initial_state`. `sample_train_task`/`sample_test_task` (sampling initial
  states, goals, train/test splits) are ordinary instance methods; per-run RNG
  streams (`LightSwitchTasks.train_rng`/`test_rng`) are genuine instance state,
  derived from a `seed` constructor field rather than any shared global.
- `problem.py` — a concrete subclass of `core.Problem`: also a real,
  constructor-injected instance now, with `env`/`tasks` narrowed to this domain's
  own `Environment`/`Tasks` subclasses as required constructor fields (e.g.
  `LightSwitchProblem(env=env, tasks=tasks)`) rather than class-level assignment —
  `human` stays an optional `type[HumanOracle] | None` field (`HumanOracle` itself
  never got an instance, see `../core/README.md`), left unset for domains with no
  irreversible action. Implements `run_task_episode` (the one method `Problem`
  doesn't get for free as a passthrough) as an ordinary instance method. Its optional
  `renderer: type[core.Renderer] | None = None` param makes every episode optionally
  recordable through this same call — no separate rendering-only codepath.
- `predicates.py` — domain predicates, needed only if a planning-based `Method`
  requires symbolic `GroundAtom`s for this domain. Pure-RL-only domains can skip
  this file entirely.
- `skills.py` — optional: a static-method container (e.g. `LightSwitchSkills`)
  declaring this domain's `core.method.types.Skill` `ClassVar`s plus
  `sample_params(*, ground_skill, rng) -> np.ndarray` and `compute_action(*,
  ground_skill, params, state) -> Action`, the lifted → grounded → raw-`Action`
  pipeline described in [`../core/README.md`](../core/README.md). Only needed once a
  domain has skills a `Method`/policy can select, as opposed to acting directly in
  raw action space (e.g. `ActionOraclePolicy`, vs. `SkillOraclePolicy` which selects
  skills — see the Status section below).
- `cli.py` — optional: only needed if this domain should be runnable via the global
  `hitl_pmp/cli.py`. A static-method container (e.g. `LightSwitchCli`) exposing
  `add_arguments(*, parser)` (adds this domain's configurable values as named
  argparse flags — no positional arguments — defaults read live from the relevant
  classes/fields), `apply_config(*, args)` (now applies only the two ClassVars that
  didn't become constructor fields — `LightSwitchEnvironment.light_on_tolerance`/
  `.same_position_tolerance` — see that class's own docstring for exactly why those
  two specifically stay `ClassVar` rather than joining `grid_size`/
  `canonical_light_target` as constructor arguments), and
  `run_method(*, args, method, num_cycles, max_steps_per_interaction)` — this
  domain's own composition root, constructing the actual `LightSwitchEnvironment`/
  `LightSwitchTasks`/`LightSwitchProblem` instances from `args` and then
  `method(env=env)` with that same `env` instance, then delegates to
  `../method_runner.py`'s `MethodRunner` for the domain-agnostic rest: actually
  driving a `core.Method`
  through `practice_loop.py`'s `PracticeLoop`, printing a success-rate summary,
  and writing `episode.mp4` if `--output-dir` is set) — registered by name in
  `hitl_pmp/cli.py`'s `ENVIRONMENTS` dict, which has no domain-specific
  knowledge of its own. An environment is never run directly, though: a
  method-CLI (registered in `hitl_pmp/cli.py`'s `METHODS` dict instead, under
  `--method`, and living under `methods/<name>/cli.py` — not here, since it's
  method-specific glue, not environment-specific) is what calls `run_method`,
  supplying which `core.Method` to drive and its own `num_cycles`/
  `max_steps_per_interaction` (an oracle passes `0`/`0` since it never
  practices) — see `methods/oracle/cli.py`'s `SkillOracleCli` and
  [`../methods/README.md`](../methods/README.md). If `--output-dir` is set
  (global flag, `hitl_pmp/cli.py`) and the domain has a `renderer.py`, that
  demo `episode.mp4` gets written there. Run statistics/metrics tracking to
  that same flag is a separate, not-yet-built
  concern (see `core/metrics/metrics.py`).
- `renderer.py` — optional: only needed if this domain should be visually
  inspectable. A concrete subclass of `core.Renderer` (`render_frame(*, state, env,
  label=None) -> np.ndarray`) — still a static-method container itself (`Renderer`
  has no state of its own to hold between calls, so it never became a
  constructor-injected instance the way `Environment`/`Problem`/`Tasks`/`Method`
  did), but now takes the one `Environment` *instance* it needs to read per-instance
  config from (e.g. `LightSwitchRenderer` reading `env.grid_size` for its axis
  limits) as an explicit per-call argument rather than ever reaching for a global —
  pure rendering logic only, but should draw `label` onto the frame when given (e.g.
  as a title/caption) so a rendered episode shows which action/skill was just taken.
  Episode-loop frame capture lives inline in
  `problem.py`'s `run_task_episode` (via its optional `renderer` param, forwarding
  each step's `LabeledAction.label` straight through), and video-writing lives in
  the domain-agnostic `core.renderer.VideoWriter` — neither is this file's concern
  (see [`../core/README.md`](../core/README.md)).

## Precedent

The sibling repo `hitl-practice` (a fork of the "predicators" TAMP codebase, one
level up) organizes concrete environments similarly under `predicators/envs/`, and
vendors a PDDL translator under
`predicators/third_party/fast_downward_translator/` for planning-based domains.
This project intentionally does not extend that codebase (it's entangled and
difficult to extend), but the domain-subfolder convention above is worth citing as
precedent: `predicates.py` here plays the same symbolic-planning-support role that
predicate definitions play in `predicators/envs/`.

## Status

- `lightswitch/` — the paper's "Light Switch" environment, ported from the sibling
  `hitl-practice` repo's `GridRowEnv` (`predicators/envs/grid_row.py`), which is the
  paper's actual reference implementation. Where the paper's prose is imprecise or
  silent on an exact number, `GridRowEnv`'s code is ground truth — see the Notion
  page's "Details not in paper but in codebase" section. Has `environment.py`
  (including `get_cells()` — `Cell` objects for `skills.py`/`predicates.py`),
  `tasks.py`, `predicates.py` (`LightOn`, `RobotInCell`, `LightInCell`, `Adjacent`),
  `skills.py` (`LightSwitchSkills` — `MoveRobot`, `TurnOnLight`, `TurnOffLight`,
  `JumpToLight`, ported from `predicators/ground_truth_models/grid_row/options.py`;
  `JumpToLight` is deliberately a hardcoded no-op, the "impossible skill"), `problem.py`
  (`LightSwitchProblem` — no `human` is ever set, since this environment has no
  irreversible action and never needs `Problem.execute_human_command`),
  two privileged-knowledge oracle policies, establishing an upper bound before any
  learning `Method` exists (see the "Now What?" Problem Setting recipe) —
  `action_oracle_policy.py` (`ActionOraclePolicy`, operating at the raw-action
  level, no skill selection at all) and `skill_oracle_policy.py`
  (`SkillOraclePolicy`, identical behavior but routed entirely through
  `skills.py`'s `Skill`/`GroundSkill`/`compute_action` pipeline — the two exist
  side by side specifically to demonstrate that both are legitimate ways to
  produce a `Policy`, matching predicators' own skill-agnostic baseline
  interface). Both are Light-Switch-only, with no knowledge of any other
  domain; `SkillOraclePolicy` is wrapped as a real `core.Method`
  (`SkillOracleMethod`, in `../../methods/oracle/skill_oracle_method.py` — see
  [`../../methods/README.md`](../../methods/README.md), not here, since the
  `isinstance(self.env, LightSwitchEnvironment)`-keyed dispatch it needs
  (`self.env` a real constructor-injected field on `Method` itself, not a global)
  is a cross-domain concern), runnable via
  the global CLI's `--method` flag (below); `ActionOraclePolicy` isn't
  currently wrapped/wired the same way, so it's exercised directly by its own
  tests rather than through the CLI. `renderer.py`
  (`LightSwitchRenderer` — draws the robot and light on a 1D strip via
  matplotlib, plus whichever policy's `LabeledAction.label` as a second title
  line, e.g. `"MoveRobot(robot, cell0, cell99)"` or `"raw action [dx=...,
  dlight=...]"`), and `cli.py` (`LightSwitchCli` for this domain's own config
  flags and its shared `run_method` helper, called by
  `../../methods/oracle/cli.py`'s `SkillOracleCli`), runnable via
  `python -m hitl_pmp.cli --env lightswitch --method skill-oracle
  [--output-dir DIR]`.
- Every other domain subfolder: not started yet. The convention above describes the
  expected shape once one lands.

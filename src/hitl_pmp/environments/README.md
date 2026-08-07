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
- `tossingroomsplit/` — **Tossing Room with the two throws as separate lifted
  skills.** The world is byte-for-byte the same as `tossingroom/`'s (7 rooms, pile and
  start in room 3, recycling bin + its own emptying button in room 1, trash bin + its
  own button in room 6, one-way ledge out of room 2, a bin holding at most **one** item
  and refusing a throw when full) and the raw action space is unchanged; the entire
  difference is in the symbolic layer. `Throw` becomes `ThrowTrash` and `ThrowRecycling`,
  each binding its own item and bin *type*, which splits `Pickup` and `Press` too and
  drops both `BinAcceptsItem` and `ButtonForBin` (the types make them tautologies). Why
  the split matters: `EesMethod.sampler` keys its
  `LearnedSkillSampler` dict by skill **name**, so one name is one classifier — Tossing
  Room's single `Throw` pools both kinds' training rows and can transfer trash
  experience to recycling, and two names remove that channel while keeping the same
  architecture. The domain exists to measure what the removal costs each kind, given
  that the layout buys roughly a dozen trash attempts per practice period and at most
  one recycling attempt (the ledge closes behind the robot, and a missed throw spends
  the item). Runnable as `python -m hitl_pmp.cli --env tossingroomsplit --method ees`,
  on the identical flag set as `--env tossingroom` **plus one**: `--two-way-ledge`,
  which makes the ledge traversable rightward as well and so leaves the domain with no
  irreversible action at all. It is off by default, and a default run is byte-identical
  to one from before the flag existed. It is the positive control for the reset-free
  experiment: reset-free practice is worse than scheduled-reset practice here, and the
  attributed mechanism is that rooms 0–2 are absorbing while the pile — the only item
  source — sits in room 3, so a robot that walks left once can never practice again.
  Turning it on removes exactly that. **It also makes the domain easier**, in three ways
  that are not method effects: EMPTY stops being an ordering task and its shortest solve
  drops 10 → 9 (so the evaluation horizon drops 12 → 11), RECYCLING stops being
  one-attempt-per-period, and rooms 0–2 stop being absorbing. Never put a two-way number
  beside a one-way one without saying so. See its `skills.py` docstring for the full
  rationale and `environment.py`'s `two_way_ledge` field for this flag's.
- `tossingroomsplitidentity/` — **the same domain again under the degenerate identity
  throw representation**, and the counterpart arm to `tossingroomsplit/` above rather
  than a replacement for it. There, the required force is an unobserved affine function
  of two observable causes (a bin's per-task `throw_distance` and an item's per-task
  `weight`), so a sampler must learn a *relation*. Here the item carries `target_force`
  and a throw lands iff `|force - item.target_force| < throw_tolerance` — so the answer
  sits in each throw's own classifier row
  (`[1.0] + concat(state[obj] for obj in ground_skill.objects) + [force]`) at **index
  4**, and the optimal policy is the literal transformation `force* = x₄`: copy input
  index 4. That is what PRs #80/#81 removed, restored deliberately so the two
  representations can be compared at a matched protocol.

  **Exactly one delta, and it is enforced.** Layout, one-way ledge, capacity-1 bins,
  button wiring, the seven lifted skills and their arities, the 14 TRASH / 14 RECYCLING
  / 2 EMPTY test composition and the horizon of 12 are all identical, and
  `tests/environments/tossingroomsplitidentity/test_fork_equivalence.py` asserts each of
  them rather than leaving it to a reviewer's diff — the permitted differences are
  enumerated in one place there.

  **Difficulty is matched by construction, because the two arms draw the same tasks.**
  `Tasks` here draws the causal arm's two causes — from the same ranges, in the same
  order, with the same number of draws — and resolves them with the same five constants,
  then puts the *result* in the State as `target_force` and discards the causes. So the
  arms consume their RNG in lockstep: at a given seed they present the identical practice
  and test tasks with the identical required force for every throw, which makes them
  **paired** rather than merely comparable. A uniformly random force lands with
  probability exactly 0.20 on every task in both, and the best single *fixed* force scores
  the same in both (185/400 at seed 0).

  Two simpler draw ranges were rejected, both of which would have been a second delta: the
  pre-#80 `U[0.5, 1.0)`, whose top decile has its winning window clipped by the U(0, 1)
  force band (0.20 on only 8/10 of its range); and `U[0.1, 0.9)`, which fixes that but not
  the *marginal* — the causal arm's required force is a sum of two uniforms and so is
  triangular, and a flat target scores 120/400 for the best fixed force against 185/400.

  Runnable as `python -m hitl_pmp.cli --env tossingroomsplitidentity --method ees`, on the
  causal arm's flag set minus the five relation flags (`--reference-force`,
  `--reference-distance`, `--reference-weight`, `--distance-coefficient`,
  `--weight-coefficient`) and minus `--canonical-throw-distance`/`--canonical-item-weight`,
  plus `--canonical-target-force`. `--distance-low/high` and `--weight-low/high` are
  retained and mean exactly what they do in the causal arm. The five relation constants
  live on `Tasks` here and are deliberately **not** flags: they are what keeps the two
  arms paired, so changing one would silently unpair them.
- The remaining domain subfolders (`ballring/`, `tossingroom/`) are implemented but not
  written up in this Status section; their own module docstrings and the experiment
  logs under `docs/experiment-logs/` are the current record. The convention above
  describes the shape every one of them follows.

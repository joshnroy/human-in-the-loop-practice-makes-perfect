# environments

This is where **concrete** `Environment` implementations live — one subfolder per
domain, e.g. `environments/lightswitch/` or a future `environments/tossing_room/`.
See [`../core/README.md`](../core/README.md) for why `Environment` is the one
real-world/ground-truth instance for a domain (not a reusable dynamics function for
hypothetical planning), with no notion of tasks, humans, or reset cost.

## Convention for a domain subfolder

Each domain subfolder is expected to contain:

- `environment.py` — a concrete subclass of `core.Environment`: the domain's own
  dynamics (`take_action`, `get_valid_actions`, `get_current_state`/`set_state`/
  `hard_reset`), all operating on the one tracked `current_state`. No tasks, no
  humans, no reset cost — just the physics/logic of the domain.
- `tasks.py` — a concrete subclass of `core.Tasks`: `sample_train_task`/
  `sample_test_task` (sampling initial states, goals, train/test splits) specific to
  this domain.
- `problem.py` — a concrete subclass of `core.Problem` that sets `env`/`human`/`tasks`
  to this domain's `Environment`, a chosen `HumanOracle` from `../human_oracles/`, and
  this domain's `Tasks`, and implements `run_task_episode` (the one method `Problem`
  doesn't get for free as a passthrough). Its optional `renderer: type[core.Renderer]
  | None = None` param makes every episode optionally recordable through this same
  call — no separate rendering-only codepath.
- `predicates.py` — domain predicates, needed only if a planning-based `Method`
  requires symbolic `GroundAtom`s for this domain. Pure-RL-only domains can skip
  this file entirely.
- `skills.py` — optional: a static-method container (e.g. `LightSwitchSkills`)
  declaring this domain's `core.method.types.Skill` `ClassVar`s plus
  `sample_params(*, ground_skill, rng) -> np.ndarray` and `compute_action(*,
  ground_skill, params, state) -> Action`, the lifted → grounded → raw-`Action`
  pipeline described in [`../core/README.md`](../core/README.md). Only needed once a
  domain has skills a `Method` can select, as opposed to acting directly in raw
  action space (e.g. `OraclePolicy`).
- `cli.py` — optional: only needed if this domain should be runnable via the global
  `hitl_pmp/cli.py`. A static-method container (e.g. `LightSwitchCli`) exposing
  `add_arguments(*, parser)` (adds this domain's configurable values as named
  argparse flags — no positional arguments — defaults read live from the relevant
  classes) and `run(*, args)` (applies them, runs a chosen policy over sampled test
  tasks, returns/prints results) — registered by name in `hitl_pmp/cli.py`'s
  `ENVIRONMENTS` dict, which has no domain-specific knowledge of its own. `methods/`
  (once a concrete `Method` exists) is expected to follow the identical pattern —
  see [`../methods/README.md`](../methods/README.md). If `--output-dir` is set
  (global flag, `hitl_pmp/cli.py`) and the domain has a `renderer.py`, `run` is also
  expected to write an `episode.mp4` demo there. Run statistics/metrics tracking is
  a separate, not-yet-built concern (see `core/metrics/metrics.py`), not part of
  this flag.
- `renderer.py` — optional: only needed if this domain should be visually
  inspectable. A concrete subclass of `core.Renderer` (`render_frame(*, state) ->
  np.ndarray`) — pure rendering logic only. Episode-loop frame capture lives inline
  in `problem.py`'s `run_task_episode` (via its optional `renderer` param), and
  video-writing lives in the domain-agnostic `core.renderer.VideoWriter` — neither
  is this file's concern (see [`../core/README.md`](../core/README.md)).

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
  `oracle_policy.py` (`OraclePolicy` — a privileged-knowledge `Policy`, establishing
  an upper bound before any learning `Method` exists; see the "Now What?" Problem
  Setting recipe — operates at the raw-action level, not through `skills.py`),
  `renderer.py` (`LightSwitchRenderer` — draws the robot and light on a 1D strip via
  matplotlib), and `cli.py` (`LightSwitchCli`, runnable via
  `python -m hitl_pmp.cli --env lightswitch [--output-dir DIR]`).
- Every other domain subfolder: not started yet. The convention above describes the
  expected shape once one lands.

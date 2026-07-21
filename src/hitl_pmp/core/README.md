# core

This folder holds the **fixed abstract interfaces** for the project: `Problem`,
`Method`, `Metrics`, `Renderer` — plus `Environment`, `HumanOracle`, and `Tasks`,
which live *nested inside* `problem/` (see "What `Problem` actually is" below for
why). Concrete implementations live in sibling folders, not here.

```
core/
├── problem/
│   ├── problem.py            Problem — the composition root / facade
│   ├── environment/
│   │   ├── environment.py     Environment — pure dynamics
│   │   └── types.py            State, Object, Type, Action
│   ├── human/
│   │   ├── human.py            HumanOracle — the human-cost model
│   │   └── types.py            Cost, CommandStartStateDescription, CommandGoalDescription
│   └── tasks/
│       ├── tasks.py            Tasks — task/goal generation
│       └── types.py             Task, Goal, Predicate, GroundAtom
├── method/
│   ├── method.py               Method — the agent side
│   └── types.py                 LabeledAction, Policy, Rollout, Skill, GroundSkill, Variable, LiftedAtom, SetupCommand
├── metrics/
│   └── metrics.py               Metrics — the evaluation protocol
└── renderer/
    └── renderer.py              Renderer, VideoWriter
```

There is no `problem/types.py` — every type that used to live there now lives in
whichever of `environment/`, `human/`, or `tasks/` actually supports it.

## What `Problem` actually is

The project's design doc defines exactly **two** classes: `Problem` and `Method` (plus
`Metrics`). The doc's `Problem` bundles *everything* — task generation,
`send_command_to_human`, `reset_environment`, and the "standard MDP functions"
(`get_current_state`, `transition_function`, `get_valid_actions`, `get_initial_state`)
all sit as plain methods on one class. There is no separate `Environment` or
`HumanOracle` class in the doc at all.

`Environment` and `HumanOracle` (and now `Tasks`) as their own classes were introduced
during this codebase's design, not specified by the doc — motivated by wanting one
dynamics implementation to be reusable across differently-configured `HumanOracle`
pairings and task distributions, and to keep `Environment` Gym-compatible for
baselines that want SB3/RLlib. That reasoning still holds, which is why they stay
separate *classes* rather than being flattened back into one — but the doc is right
that they **belong to `Problem`**, not beside it: all three nest under `problem/`, not
siblings of it in `core/`.

To actually deliver on that — so that `Problem` reads like the doc's flat class, not
just a folder that happens to contain the other three — `Problem` is a **facade**:
`get_current_state`, `take_action`, `get_valid_actions`, `hard_reset`,
`sample_train_task`, `sample_test_task`, `calculate_cost_for_human_command`, and
`execute_human_command` are all concrete passthroughs to
`Problem.env`/`Problem.tasks`/`Problem.human` (a shared
private `_describe_command` helper builds the two `Command*Description` objects both
human-facing methods need, to avoid duplicating that construction). Notably,
`execute_human_command` doesn't return a cost at all — it hands `Problem.env` directly
to `Problem.human.execute_human_command`, which is responsible for updating it (e.g.
via `env.set_state`) to reflect whatever actually happened; querying the cost is
`calculate_cost_for_human_command`'s separate job. The **only** method that stays
abstract on `Problem` itself is `run_task_episode` — genuine orchestration logic
(loop calling the policy, taking actions, checking the goal) that no single part can
supply on its own, and which every concrete `Problem` must implement. It also takes
an optional `renderer: type[Renderer] | None = None` and returns `(succeeded,
frames)` — every episode, in the normal sweep or otherwise, is optionally
recordable through this one call, rather than a second rendering-only codepath that
would duplicate the same loop (see "`Renderer` is a pure function of `State`" below).
`Method` and
`Metrics` stay true top-level siblings of `problem/`, matching the doc's
`run(problem: Problem, method: Method) -> Metrics` treating them as independent peers.

## Conventions applied here

See the root [README's Conventions section](../../../README.md#conventions) for the
full rationale; the short version, as applied in this folder:

- **Data lives in the `types.py` of the module it supports, as pydantic `BaseModel`s**
  — no shared "bucket" file anywhere. `State`/`Object`/`Type`/`Action` support
  `Environment` (defining state/action space is Environment's job) →
  `problem/environment/types.py`. `Cost`/`CommandStartStateDescription`/
  `CommandGoalDescription` support `HumanOracle` (its two methods' signatures) →
  `problem/human/types.py` — `CommandStartStateDescription` currently just wraps a
  `State` (with a `TODO` to figure out what it should really contain, per the design
  doc's v3 human model needing NL/pictorial descriptions instead of raw states);
  `CommandGoalDescription` already wraps the same symbolic `Goal` that `Task.goal`
  uses. `Task`/`Goal`/`Predicate`/`GroundAtom` support `Tasks` (task/goal generation
  is `Tasks`' job) → `problem/tasks/types.py`. `Policy`/`Rollout`/`Skill`/
  `GroundSkill`/`SetupCommand` support `Method` → `method/types.py`. `dataclasses`/`attrs` are
  banned project-wide (ruff `TID251`). `Task`/`Goal` are intentionally **not**
  frozen/hashable (unlike `Object`/`Type`/`GroundAtom`/`Predicate`, which sit inside
  dict keys or a `frozenset`) — nothing puts a `Task` in a `set`/dict key position;
  `Tasks.sample_train_task`/`sample_test_task` each return a single `Task`, and
  `Task.initial_state` wraps a mutable numpy-backed `State` that couldn't honestly be
  hashed anyway.
- **No `__init__.py` re-exports anything** — every name has exactly one import path
  (e.g. `from hitl_pmp.core.problem.environment.types import State`), never a second
  shortcut through a package `__init__.py`.
- **Imports are absolute across subpackages, relative within one.** From
  `problem/problem.py`, `.environment.environment` and `.tasks.tasks` are same-tree
  relative imports; from `problem/human/human.py`, reaching
  `environment/` — a *sibling*, not a parent — requires the absolute
  `hitl_pmp.core.problem.environment.types`, since `..`-style parent-relative imports
  are banned (ruff `TID252`).
- **No `if TYPE_CHECKING:` guards** — also banned (`TID251`). `Problem.run_task_episode`
  needs `Method`'s `Policy`, and `Method.get_task_policy`/`generate_train_task` need
  `Tasks`' `Task`. This looks like a two-way dependency, but it isn't one at the file
  level: `problem.py` imports `Policy` from `method/types.py` (not `method.py`), and
  `method.py` imports `Task` from `problem/tasks/types.py` (not `problem.py`). Neither
  `types.py` imports the other's ABC file back, so there's no cycle — just import the
  target's `types.py` directly and skip the deferred-import trick entirely.
- **Behavior lives in the ABCs, as static-method containers — and every concrete
  business-logic helper underneath them follows the same rule.** None of
  `Environment`/`HumanOracle`/`Tasks`/`Problem`/`Method`/`Metrics` is ever
  instantiated — almost every method is `@staticmethod` (`Problem`'s facade methods
  are concrete but still static — they delegate, they don't need instance state of
  their own), and any state a concrete subclass needs (e.g. `Problem.env`,
  `Problem.human`, `Problem.tasks`) is a `ClassVar` set once on the class itself, Java
  static-class/singleton style, not constructor-assigned instance state. This isn't
  limited to the ABCs here: a domain's `Predicate.holds` classifier or a `Policy`
  function's real logic belongs on its own static-method class too (e.g.
  `environments/lightswitch/action_oracle_policy.py`'s `ActionOraclePolicy.get_action`),
  not a bare module-level function — the only exception is a short lambda adapter where an
  interface itself demands a positional callable (`Predicate.holds`, `Policy`), since
  the lambda carries no logic of its own that would otherwise be lost in a module.
  Every parameter (besides an unavoidable dunder like `__getitem__`) is keyword-only,
  enforced by ruff's `PLR0917` with `max-positional-args = 0`.
- **Files/classes are organized top-down**, most composite first — see
  `problem/tasks/types.py` (`Task` → `Goal` → `Predicate` → `GroundAtom`, in
  decreasing order of "what relies on what").

## Why `Environment`/`HumanOracle`/`Tasks` split the way they do

Gym/Gymnasium bakes in the assumption that `reset()` is free and automatic whenever an
episode ends. Our research problem breaks that assumption on purpose: a robot deployed
outside the factory can take **irreversible** actions, so ending an episode does not
imply a free reset — a human/oracle must sometimes intervene, at a cost, to move the
environment back to a usable state.

- **`Environment`** is *the real-world environment* (or the real/ground-truth
  simulator standing in for it) — there is exactly one of it, tracked via
  `current_state: ClassVar[State]`. It is **not** a reusable, stateless dynamics
  function that other code can call with a hypothetical state to explore "what if" —
  a `Method` that needs to plan carries its own model for that; it must not borrow
  `Environment` to do it. `take_action(*, action)` advances `current_state` by one
  action via the domain's own underlying dynamics and returns the new state;
  `get_valid_actions()` reads from `current_state` too — neither takes an explicit
  `state` argument, both operate on the one real state. `get_current_state()`/
  `set_state()` are concrete (shared across every `Environment`, not reimplemented
  per domain) — `set_state` is a *privileged external override* (used by a human, via
  `HumanOracle`/`Problem.execute_human_command`, to force a state — distinct from
  `take_action`'s normal forward dynamics). `hard_reset()` resets to the initial state
  distribution but is only ever called by the harness before a run starts, never by
  the agent or tied to a human cost. `action_space` is typed as `gymnasium.spaces.Space`
  (never the legacy `gym` package), not a plain numpy array — a `Space` is
  self-describing (bounds, shape, `sample()`, `contains()`), it's what `to_gym.py` will
  hand straight to SB3/RLlib with zero conversion, and it's left as the abstract
  `Space` rather than hardcoded to `Box` so a domain with a mixed
  discrete-skill/continuous-parameter action structure (e.g. Tossing Room) can pick
  `Discrete`, `MultiDiscrete`, or `Dict` instead.
- **`HumanOracle`** is the human/oracle cost model, independently swappable (the v0
  unconditional oracle up through a v3 natural-language, capability-aware oracle in the
  design doc) from whichever `Environment` it's paired with.
- **`Tasks`** is the task/goal distribution — `sample_train_task()`/`sample_test_task()`
  — also independently swappable from whichever `Environment`/`HumanOracle` it's paired
  with (a curriculum-learning `Tasks` and a random-sampling `Tasks` could sit on top of
  the same domain).
- **`Problem`** is the composition root/facade that binds one `Environment` + one
  `HumanOracle` + one `Tasks`. "No auto-reset" and "human-mediated reset has a cost"
  live here, via `execute_human_command`. Unlike `Environment`, a `Problem` is specific
  to one research question, not reusable across them.

## `Renderer` is a pure function of `State`, not a `Problem` component

`Renderer` (one abstract method, `render_frame(*, state, label=None) -> np.ndarray`)
sits as a top-level sibling of `problem/`, like `Method`/`Metrics` — not nested under
it like `Environment`/`HumanOracle`/`Tasks` are. Those three nest under `problem/`
because the design doc's `Problem` genuinely owns them (dynamics, human cost, task
generation are all `Problem`-scoped concepts). Rendering isn't: it's a pure, stateless
function of whatever `State` (and optional `label`) you hand it, useful standalone
(e.g. debugging a hand-built `State` with no `Problem` in scope at all) and with no
reset-cost/human-in-the-loop semantics of its own. `renderer.py` also holds one
non-abstract, domain-agnostic companion, not part of the `Renderer` interface itself
since it never varies per domain: `VideoWriter` (writes a frame sequence to a video
file via imageio's bundled ffmpeg, and `write_gif` converts an already-written video
to a gif — prefers imageio itself, pure Python, ffmpeg still doing the underlying
decode work just wrapped by a library instead of a raw subprocess call, falling back
to shelling out to `ffmpeg` directly, located via `imageio_ffmpeg`'s own bundled
copy rather than relying on `ffmpeg` being on `PATH`, only if that import/read ever
fails).

`label` is how a rendered episode shows which action/skill was just taken, without
`Problem`/`Method` needing a separate rendering-specific side channel:
`method/types.py`'s `LabeledAction` (`action` + `label`) is what every `Policy` now
returns instead of a bare `Action` — `Problem.run_task_episode` forwards
`labeled_action.label` straight into `renderer.render_frame`'s `label` param each
step (`None` on the very first frame, since no action has produced it yet). A raw
action-oracle labels itself with its own numbers
(`environments/lightswitch/action_oracle_policy.py`); a skill-based policy labels
itself with the `GroundSkill` it selected (`environments/lightswitch/
skill_oracle_policy.py`, e.g. `"MoveRobot(robot, cell0, cell99)"`) — `Renderer`
itself doesn't know or care which kind of policy produced the label it's asked to
draw.

There's deliberately no separate "run an episode and record it" utility here. An
earlier version had one (`EpisodeRenderer`), but it duplicated `Problem.run_task_episode`'s
own loop (check the goal, else `take_action`) — the same logic living in two places,
one of which would silently drift out of sync with the other over time. It would also
have introduced a real circular import: `EpisodeRenderer` needs `Problem` to drive the
episode, but if `Problem.run_task_episode`'s signature also needs `Renderer`, that's
`problem.py` → `renderer.py` → `problem.py`. Instead, `renderer.py` only ever imports
`State` (a leaf), and `Problem.run_task_episode` imports `Renderer` — one direction,
no cycle — and does its own inline frame capture when `renderer` is passed, so there
is exactly one place any episode ever runs, rendered or not.

## `Type` declares a feature schema, not just a name

`Type` carries `feature_names: tuple[str, ...]` and a `dim` property (`len(feature_names)`)
— e.g. `Type(name="block", feature_names=("x", "y", "z"))`. This is what lets `State`
actually enforce that an `Object`'s raw feature vector matches its declared type (a
`model_validator` on `State` rejects a vector whose length doesn't equal
`obj.type.dim`), and what lets `State.get(obj=..., feature_name="x")`/`.set(...)` look
up a feature by name instead of a raw, undocumented vector index.

`predicators` also gives `Type` a `parent` for subtype inheritance (e.g. a `"movable"`
type reusing all of `"base-object"`'s `feature_names` plus a few more, and predicates/
skills declared against a parent type automatically applying to every subtype). We
deliberately left it out: nothing in the codebase walks a parent chain yet
(no `is_instance()`), and no current domain needs a type hierarchy — Tossing Room's
`trash`/`recycling` don't need one; they can be flat types, or even a single `item`
type with an `is_recycling` feature. It's cheap to add back (a one-line field, matching
`predicators`' pattern exactly) once a second or third object type actually needs to
share a feature schema, or once a predicate/skill needs to generalize across a family
of related types — e.g. a real robot deployment with many movable-object subtypes
(mirroring Spot's `movable`/`container`/`dustpan`/`broom` in `predicators`), or once
PDDL generation wants a native `:types` hierarchy rather than a flat list.

This is a direct port of the `predicators` precedent
(`hitl-practice/predicators/structs.py`'s `Type`), and it's also the mechanism that
makes purely-symbolic domains (e.g. a grid-position `Type("obj", ("row", "column"))`)
and genuinely continuous ones (e.g. a real robot's
`Type("robot", ("gripper_open_percentage", "x", "y", "z", "qw", "qx", "qy", "qz"))`)
interchangeable under this same interface: neither `Type`/`Object`/`State` nor the
planner ever know or care whether a feature is discrete- or continuous-valued — that
distinction only exists inside a domain's own `Predicate.holds` classifiers.

## `Skill`/`GroundSkill` are a lifted/grounded pair, like `Predicate`/`GroundAtom`

`Skill` (name + `parameters` + `preconditions`/`add_effects`/`delete_effects` +
`param_dim`) is a lifted template — what a `Method` can select before being bound to
concrete objects; `GroundSkill` (`skill` + `objects`) binds one to a specific object
tuple, mirroring `GroundAtom`'s shape in `problem/tasks/types.py` exactly, and
exposes `.preconditions`/`.add_effects`/`.delete_effects` as *grounded* `GroundAtom`s
(substituting `objects` for `skill.parameters` positionally). Continuous parameters
are deliberately **not** part of `GroundSkill` — per `predicators`'
`_Option`/`_GroundNSRT.sample_option()` precedent, params are sampled fresh each
execution (a concrete `Method`'s job, inside `execute_skill`), so
`improve_skill_parameters` updates the *sampler*, not one already-consumed value.

`Skill`'s `preconditions`/`add_effects`/`delete_effects` are `LiftedAtom`s — a
`Predicate` applied to `Variable`s (a typed placeholder, e.g. `?robot`) rather than
concrete `Object`s, mirroring `predicators`' `STRIPSOperator`/`NSRT` symbolic half.
This was deliberately deferred until a real consumer existed; `methods/
practice_makes_perfect/` (reproducing the original PMP/EES paper) is that consumer —
it needs real STRIPS operators to task-plan over via Fast Downward (`planning/`).
`LiftedAtom.ground(*, substitution)` produces a `GroundAtom`, mirroring
`Predicate.__call__`. See `environments/lightswitch/skills.py` for a concrete
instantiation (`MoveRobot`, `TurnOnLight`, `TurnOffLight`, `JumpToLight`) and its
`sample_params`/`compute_action` static methods, which round out the lifted →
grounded → raw-`Action` pipeline these types describe.

```mermaid
flowchart LR
    skill["Skill<br/>(lifted template)<br/>name, parameters, preconditions/effects, param_dim"]
    ground["GroundSkill<br/>(bound to objects)<br/>skill, objects"]
    params["params: ndarray<br/>(sampled fresh each execution)"]
    action["Action<br/>(raw [dx, dlight])"]
    state["State"]

    skill -- "bind to concrete objects" --> ground
    ground -- "sample_params(rng)" --> params
    ground -- "compute_action(params, state)" --> action
    state -.-> action
    params -.-> action
    action -- "Environment.take_action" --> state

    subgraph example["Light Switch's four Skill instances (skills.py)"]
        direction TB
        moveRobot["MoveRobot(robot, cell, cell)<br/>param_dim=0"]
        turnOn["TurnOnLight(robot, cell, light)<br/>param_dim=1"]
        turnOff["TurnOffLight(robot, cell, light)<br/>param_dim=1"]
        jump["JumpToLight(robot, cell, cell, cell, light)<br/>param_dim=1, always a no-op"]
    end
    skill -.-> example
```

`compute_action` dispatches on `ground_skill.skill` by **value equality** (frozen
pydantic models), not object identity — any independently-constructed `Skill` with
matching `name`/`types`/`param_dim` is treated the same as the `LightSwitchSkills.*`
`ClassVar`s above.

## Files

- `problem/environment/` — `environment.py` (the `Environment` ABC) + `types.py`
  (`State`, `Object`, `Type`, `Action`). The most foundational subpackage: imports
  nothing else from `core/`.
- `problem/human/` — `human.py` (the `HumanOracle` ABC) + `types.py`
  (`Cost`, `CommandStartStateDescription`, `CommandGoalDescription`). Imports
  `State` from `../environment/types.py`.
- `problem/tasks/` — `tasks.py` (the `Tasks` ABC) + `types.py` (`Task`, `Goal`,
  `Predicate`, `GroundAtom`). Imports from `../environment/types.py`.
- `problem/problem.py` — `Problem`, the facade. Imports from `environment/`,
  `human/`, `tasks/`, and `method/types.py`.
- `method/` — `method.py` (the `Method` ABC) + `types.py` (`LabeledAction`, `Policy`,
  `Rollout`, `Skill`, `GroundSkill`, `SetupCommand`). Imports from
  `problem/environment/types.py` and `problem/tasks/types.py`.
- `metrics/` — `metrics.py`, the (mostly generic) evaluation protocol. No `types.py` —
  it has no supporting types of its own yet.
- `renderer/` — `renderer.py` (`Renderer`, `VideoWriter`). No `types.py` — frames are
  plain `np.ndarray`, no new pydantic type needed. Imports only `State` from
  `problem/environment/types.py` — nothing else in `core/`, so it stays a leaf
  `problem/problem.py` can safely depend on (see the dependency graph below).

## Module dependency graph

Most-foundational at the top, most-dependent at the bottom. This is a genuine DAG —
`problem.problem` depends on `method.types`, and `method.method` depends on
`problem.tasks.types`, but neither `types.py` imports the sibling ABC file back, so
there's no cycle and no `TYPE_CHECKING` needed anywhere:

```mermaid
graph TD
    env["problem/environment/<br/>Environment, State, Object, Type, Action"]
    ho["problem/human/<br/>HumanOracle, Cost"]
    tasktypes["problem/tasks/types.py<br/>Task, Goal, Predicate, GroundAtom"]
    tasks["problem/tasks/tasks.py<br/>Tasks"]
    mtypes["method/types.py<br/>LabeledAction, Policy, Rollout, Skill, GroundSkill, Variable, LiftedAtom, SetupCommand"]
    renderer["renderer/<br/>Renderer, VideoWriter"]
    problem["problem/problem.py<br/>Problem"]
    method["method/method.py<br/>Method"]
    metrics["metrics/<br/>Metrics"]

    ho --> env
    tasktypes --> env
    tasks --> tasktypes
    mtypes --> env
    renderer --> env
    problem --> env
    problem --> ho
    problem --> tasks
    problem --> mtypes
    problem --> renderer
    method --> env
    method --> mtypes
    method --> tasktypes
```

## Concrete implementations

Live in sibling folders: [`../environments/`](../environments/),
[`../humans/`](../humans/), [`../methods/`](../methods/),
[`../adapters/`](../adapters/), [`../planning/`](../planning/).

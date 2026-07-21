# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Design doc / paper notes live in Notion (ask Josh for access):
https://app.notion.com/p/joshnroy/Human-in-the-Loop-Practice-Makes-Perfect-37133470fbc580aab736c283e49ee5db?source=copy_link

## Setup

Uses the `hitl-pmp` conda environment (Python 3.10). Activate it before running any
command below — a persistent shell can silently fall back to `base`, which has
mismatched dependency versions (e.g. numpy stubs) and produces confusing failures:

```bash
conda activate hitl-pmp
pip install -e ".[dev]"
```

## Commands

```bash
pytest                        # run all tests
pytest tests/core/problem/test_problem.py::test_take_action_delegates_to_env  # single test
ruff check .                  # lint
ruff check --fix .            # lint, autofix
ruff format .                 # format
mypy src                      # typecheck (src only; tests/ has relaxed untyped-def rules)
coverage run --source=src/hitl_pmp -m pytest -q && coverage report -m  # coverage
pre-commit install            # optional: run lint/format/typecheck locally pre-commit
```

All three of lint/typecheck/test run in CI (`.github/workflows/ci.yml`) on every push/PR
to `main`. `main` only allows squash-merge (no merge commits, no rebase merge).

## Workflow: one independent feature per PR, stacked in dependency order

Multi-piece work (e.g. "port this paper baseline") gets decomposed into a list of
genuinely independent features *before* any branch is created, not discovered
mid-implementation. Two features are independent if neither imports/calls the
other; a feature that imports another is *dependent* on it, not independent, even
if they're conceptually part of the same effort.

- Write out the full dependency-ordered list up front (most-foundational first —
  a feature with zero dependencies on the others goes first; each later PR only
  depends on what's strictly below it in the list).
- Build and open **one PR at a time**, stacked on the previous one's branch,
  even when the whole set was scoped together. Don't bundle several independent
  features into one PR because they're related or were requested together — if a
  PR's diff spans more than one of the list's entries, split it before opening.
- If a later PR reveals that an earlier one's scope was wrong (e.g. a piece
  turns out to need infrastructure that didn't ship yet), fix the ordering going
  forward rather than quietly re-bundling — reopen/re-split as needed, and keep
  the running dependency list current so this doesn't recur.
- `analysis/` scripts are **post-run analysis only** — they read `--output-dir`
  output back in and produce plots/tables/reports; they never run a simulation
  or drive a `Method` themselves. That's `hitl_pmp/cli.py`'s job (`python -m
  hitl_pmp.cli --env ... --method ... --output-dir ...`). If an `analysis/`
  script is calling `Problem`/`Method`/`Environment` directly instead of
  invoking the CLI and reading its output, that's a sign that the CLI-side
  wiring it depends on shipped in a later PR than it should have.

## Architecture

`src/hitl_pmp/` is reusable library code; `tests/` mirrors it 1:1. `analysis/` (scripts/
notebooks producing results/figures) will import from `hitl_pmp`, never the reverse.

### The `core/` interfaces and the static-method-container pattern

`core/` holds six **fixed abstract interfaces**, none ever instantiated: `Problem`,
`Method`, `Renderer` (top-level), plus `Environment`, `HumanOracle`, `Tasks` (nested
*under* `core/problem/`, not siblings of it — see below for why). Every method is
`@staticmethod`; any state a concrete subclass needs (e.g. `Problem.env`) is a
`ClassVar` **set on the base class itself** (`Problem.env = ConcreteEnv`), Java
static-class/singleton style — not constructor-assigned instance state, and methods
reference the base class by name (`Problem.env`), never `cls`. The same
static-method-container rule extends to any concrete business logic underneath these
interfaces, however small (e.g.
`environments/lightswitch/action_oracle_policy.py`'s `ActionOraclePolicy.get_action`)
— never a bare module-level function, except a short lambda where an interface
demands a positional callable (`Predicate.holds`, `Policy`).

`Metrics` (`core/metrics/metrics.py`) sits alongside these but isn't actually
abstract: every method there is already a genuine, reusable default (nothing in this
codebase needs different behavior than "one task type, no real human-intervention
tracking" yet), so there's no forced-must-override method the way `Problem` still has
`run_task_episode`. Callers use `Metrics` directly, no per-domain subclass — a future
`Method`/environment that genuinely needs different behavior overrides just the
specific method that differs (ordinary subclassing, not contingent on the parent
being an ABC).

```
core/
├── problem/
│   ├── problem.py            Problem — composition root / facade
│   ├── environment/
│   │   ├── environment.py     Environment — the one real-world/ground-truth instance
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

`Problem.run_task_episode` takes an optional `renderer: type[Renderer] | None = None`
and returns `(succeeded, frames)` — every episode is optionally recordable through
this one call (no separate rendering-only codepath, which would duplicate the loop).

`src/hitl_pmp/cli.py` is the global CLI entrypoint (`python -m hitl_pmp.cli --env
<name> ...`, e.g. `--env lightswitch`); it dispatches to each registered
environment's own `environments/<name>/cli.py`, or — if `--method <name>` is given
instead — to a registered `core.Method`'s own `methods/<name>/cli.py`, which drives
it through `src/hitl_pmp/practice_loop.py`'s `PracticeLoop` (the one execution
harness every `Method` runs through; see the `core/` section above for why
`Metrics`, what it records evaluations into, is fully concrete). `METHODS` is empty
for now — nothing implements `core.Method` yet. All flags are named, no positional
arguments. `--output-dir DIR` (global), if the environment has a `renderer.py`,
additionally writes a demo `episode.mp4`. Writing run statistics (e.g. a `stats.json`
built from `Metrics`) to `--output-dir` is a separate, not-yet-built concern.

**Why `Environment`/`HumanOracle`/`Tasks` nest under `problem/`**: the design doc
defines only `Problem` and `Method` (plus `Metrics`) — the doc's `Problem` bundles task
generation, `send_command_to_human`, `reset_environment`, and the "standard MDP
functions" all as plain methods on one class. This codebase introduced `Environment`/
`HumanOracle`/`Tasks` as separate classes (motivated by wanting one dynamics
implementation reusable across different `HumanOracle`/task-distribution pairings, and
Gym-compatibility for RL baselines), but they still conceptually belong to `Problem`,
so they nest under it. To make `Problem` still read like the doc's flat class, it's a
**facade**: `get_current_state`, `take_action`, `get_valid_actions`, `hard_reset`,
`sample_train_task`, `sample_test_task`, `calculate_cost_for_human_command`, and
`execute_human_command` are all concrete one-line passthroughs to
`Problem.env`/`Problem.tasks`/`Problem.human`. The **only** abstract method on `Problem`
itself is `run_task_episode` — genuine orchestration no single part can supply. Full
rationale and a mermaid dependency graph: `src/hitl_pmp/core/README.md`.

**Why this breaks from Gym's `reset()`-is-free assumption**: a robot can take
**irreversible** actions, so ending an episode doesn't imply a free reset — a human
must sometimes intervene, at a cost. `Environment.take_action`/`get_valid_actions`
operate implicitly on the one tracked `current_state: ClassVar[State]` (no explicit
`state` param — this is not a reusable dynamics function for hypothetical "what-if"
planning; a `Method` that needs to plan carries its own model for that). `set_state` is
a privileged external override used by `HumanOracle` via
`Problem.execute_human_command`, distinct from `take_action`'s normal forward dynamics.
`hard_reset()` is harness-only (before a run starts), never called by the agent.
`HumanOracle.execute_human_command` takes `env: type[Environment]` directly and is
responsible for mutating it (e.g. `env.set_state(...)`) to reflect whatever actually
happened — it returns nothing; querying cost beforehand is
`calculate_cost_for_human_command`'s separate, side-effect-free job.

### Conventions (enforced by lint, not just documented)

- **Pydantic only** — `dataclasses`/`attrs` are banned via ruff `TID251`.
- **Keyword-only everywhere** — ruff `PLR0917` with `max-positional-args = 0`. Only
  `self`/`cls`, unavoidable dunders (`__getitem__`, silenced with `# noqa: PLR0917`),
  and third-party calls are exempt.
- **Lowercase filenames only** — ruff `N999`.
- **No `if TYPE_CHECKING:` guards** — banned via `TID251`. Where two subpackages each
  need a type the other owns, import the target's `types.py` directly (never its ABC
  file) — neither `types.py` imports the sibling ABC back, so there's no real cycle.
- **No `__init__.py` re-exports** — every name has exactly one import path (e.g.
  `from hitl_pmp.core.problem.environment.types import State`).
- **Imports absolute across subpackages, relative within one** — ruff `TID252` bans
  `..`-parent-relative imports.
- **Data lives in the `types.py` of the module it supports** as pydantic `BaseModel`s
  — never a shared "bucket" file. `Task`/`Goal` are deliberately *not*
  frozen/hashable (they wrap a mutable numpy-backed `State`); `Object`/`Type`/
  `GroundAtom`/`Predicate` *are* frozen (they sit in dict keys / a `frozenset`).
- **Files/classes organized top-down** — most composite/important first, e.g. in
  `tasks/types.py`: `Task` → `Goal` → `Predicate` → `GroundAtom`. No
  `model_rebuild()` calls needed: pydantic resolves forward references lazily on
  first real use.
- **`Type` declares a feature schema** (`feature_names: tuple[str, ...]`, `dim`
  property), not just a name — lets `State`'s validator reject a feature vector whose
  length doesn't match `obj.type.dim`, and lets `State.get(obj=, feature_name=)` look
  up by name instead of raw index. No `parent`/inheritance on `Type` — deliberately
  deferred until a domain actually needs a type hierarchy.

### Sibling folders (concrete implementations; each has its own README)

- `environments/` — concrete `Environment` + `Tasks` + `Problem` per domain, one
  subfolder each (e.g. `environments/lightswitch/`). A domain subfolder holds
  `environment.py`, `tasks.py`, `problem.py`, and optionally `predicates.py` (only if
  a planning-based `Method` needs symbolic `GroundAtom`s for that domain) and
  `skills.py` (only if a `Method` selects lifted `Skill`s rather than acting
  directly in raw action space — declares `Skill` `ClassVar`s plus
  `sample_params`/`compute_action` static methods; see `core/README.md`'s
  `Skill`/`GroundSkill` section and `environments/lightswitch/skills.py`).
- `humans/` — concrete `HumanOracle` implementations, the v0 (unconditional) →
  v3 (natural-language, capability-aware) axis from the design doc. Domain-agnostic:
  a `HumanOracle` knows nothing about any specific `Environment`'s dynamics.
- `methods/` — concrete `Method`/baseline implementations. `practice_makes_perfect/`
  reproduces the *original* PMP/EES paper's own method + every baseline it compares
  against (Fail Focus, Competence Gradient, Skill Diversity, Task-Relevant, Task
  Repeat, Random Skills, MAPLE-Q), on Light Switch — a faithfulness repro, not this
  project's own research contribution. This project's *own* planned baselines
  (trivial fixed-skill planner, `planning_to_practice.py`, `pure_vla.py`,
  `in_context_vla.py` — see the design doc's baseline progression and expected
  failure modes) stay unimplemented until that reproduction is done.
- `adapters/` — bidirectional `core.Environment` ↔ Gym/Gymnasium bridge:
  `from_gym.py` wraps a third-party `gym.Env` to satisfy `core.Environment`;
  `to_gym.py` wraps this project's `core.Environment` to expose the Gym interface for
  RL libraries like SB3/RLlib. Not mirror images — different directions, different
  jobs. Neither exists yet.
- `planning/` — bridges `Predicate`/`GroundAtom`/`Skill` to real Fast Downward (PDDL
  planning), for planning-based `Method`s only — needed for `methods/
  practice_makes_perfect/`, since EES's competence-cost-aware task planning has no
  built-in-planner substitute (predicators' own non-FD `astar` planner doesn't
  support per-operator costs at all). Pure deep-RL baselines (MAPLE-Q) never import
  it. Not vendored/bundled — see `planning/README.md` for the external install steps
  and why `_translate`/`_search` are `# pragma: no cover` (mirrors predicators' own
  precedent: CI can't be assumed to have Fast Downward installed).

### Sibling repo: `hitl-practice`

One directory up (`../hitl-practice`) is a fork of the `predicators` TAMP codebase that
the original "Practice Makes Perfect" paper was built on. This project deliberately does
**not** extend that codebase (it's entangled and hard to extend), but it's the reference
implementation for porting any paper environment/behavior faithfully — e.g. the paper's
"Light Switch" environment is `GridRowEnv` in `predicators/envs/grid_row.py`, with its
skills/NSRTs in `predicators/ground_truth_models/grid_row/`. When the paper's prose is
imprecise or silent on an exact number (tolerances, sampling ranges, defaults), treat
`hitl-practice`'s code as ground truth over the paper text, and check
`predicators/settings.py` for the actual default config values used.

# human-in-the-loop-practice-makes-perfect
Ask Josh for access to the notion page: https://app.notion.com/p/joshnroy/Human-in-the-Loop-Practice-Makes-Perfect-37133470fbc580aab736c283e49ee5db?source=copy_link

## Setup

Uses the `hitl-pmp` conda environment (Python 3.10).

```bash
conda activate hitl-pmp
pip install -e ".[dev]"
```

## Structure

- `src/hitl_pmp/` — core, reusable, tested library code
- `tests/` — unit tests for `hitl_pmp` (mirrors `src/hitl_pmp/`)
- `analysis/` — scripts/notebooks that use `hitl_pmp` to produce results and figures for the project

`analysis/` imports from `hitl_pmp`; `hitl_pmp` never depends on `analysis/`.

Subpackages under `src/hitl_pmp/` (see each folder's own README for details):

- [core/](src/hitl_pmp/core/README.md) — the fixed abstract interfaces: `Problem`, `Method`, `Metrics`, plus `Environment` and `HumanOracle` (nested under `problem/`, since the design doc's `Problem` is what actually owns them), each with its own supporting data types
- [environments/](src/hitl_pmp/environments/README.md) — concrete `Environment` + `Problem` implementations, one subfolder per domain
- [human_oracles/](src/hitl_pmp/human_oracles/README.md) — concrete `HumanOracle` implementations (the v0-v3 human-cost-model axis)
- [methods/](src/hitl_pmp/methods/README.md) — concrete `Method`/baseline implementations
- [adapters/](src/hitl_pmp/adapters/README.md) — the bidirectional Gym <-> `Environment` bridge
- [planning/](src/hitl_pmp/planning/README.md) — symbolic Predicate/GroundAtom -> PDDL -> Fast Downward bridge (optional, planning-based methods only)

## Conventions

- **All data/state lives in pydantic `BaseModel`s.** `dataclasses` and `attrs` are
  banned — enforced by ruff's `TID251` banned-api rule; importing either is a lint
  error, not just a style nit.
- **Every filename is `lower_case.py`, no exceptions.** Enforced by ruff's `N999`
  (`invalid-module-name`) — a capitalized or mixed-case filename is a lint error.
- **Each interface is a subpackage: an entrypoint file (named after the module) plus
  a `types.py`** for the data it supports — not a flat module, and never a shared
  bucket file. `problem/environment/environment.py` is `Environment`'s entrypoint;
  `problem/environment/types.py` holds `State`/`Object`/`Type`/`Action` because
  defining state/action space is Environment's job. Same pattern for
  `problem/human_oracle/` (`Cost` — `send_command` is what produces it),
  `problem/tasks/` (`Task`/`Goal`/`Predicate`/`GroundAtom` — task/goal generation is
  Tasks' job), and `method/` (`Policy`/`Rollout`/`Skill`/`SetupCommand`).
  `Environment`/`HumanOracle`/`Tasks` all nest *under* `problem/` rather than sitting
  beside it — see
  [`core/README.md`](src/hitl_pmp/core/README.md#what-problem-actually-is) for why. A
  `types.py` only splits into its own file once a type grows big enough to earn one.
  No `__init__.py` anywhere re-exports types from its submodules — every name has
  exactly one import path (e.g. `from hitl_pmp.core.problem.environment.types import
  State`), never a second shortcut through a package `__init__.py`. Imports across
  subpackages are absolute, never relative (`from ..x import y` is a `TID252` lint
  error) — see [`core/README.md`](src/hitl_pmp/core/README.md#module-dependency-graph)
  for the full dependency diagram. `if TYPE_CHECKING:` guards are also banned
  (`TID251`) — where two subpackages each need a type the other owns (`Problem` needs
  `Method`'s `Policy`, `Method` needs `Problem`'s `Task`), just import the target's
  `types.py` directly (never its ABC file) instead: neither `types.py` imports the
  other's ABC back, so there's no real cycle to defer around.
- **Behavior lives in static-method container classes, not OOP objects — and this
  applies to any business logic, not just the core ABCs.** `Environment`,
  `HumanOracle`, `Problem`, `Method`, `Metrics` are never instantiated — think Java's
  "static class" / singleton-by-class-name idiom, not encapsulated instance state.
  Every method is `@staticmethod` (`@abc.abstractmethod` on the interface, concrete
  overrides on subclasses); any state a concrete implementation needs (e.g.
  `Problem.env`, `Problem.human`, `Problem.tasks`) is a `ClassVar` set once on the
  class itself, not passed into a constructor. The same rule extends to any concrete
  business logic underneath those interfaces, however small — a `Predicate.holds`
  classifier or a `Policy` function's real logic still lives as a `@staticmethod` on
  its own class (e.g. `environments/lightswitch/predicates.py`'s `LightOnClassifier`,
  `environments/lightswitch/oracle_policy.py`'s `OraclePolicy`), never as a bare
  module-level function. Where an interface itself requires a positional callable
  (`Predicate.holds`, `Policy`), a short module-level lambda adapts the class's
  keyword-only method into that shape — the lambda is the only thing allowed to be a
  bare function, since it carries no logic of its own to find later. This is an
  organizational choice, not idiomatic OOP — the goal is functional-style code
  (explicit data in, explicit data out) organized into namespacing containers, so
  behavior is always reachable by class name (`OraclePolicy.get_action`) instead of
  scattered across module-level functions you have to grep for or chase through an
  inheritance tree to find.
- **Every parameter is keyword-only.** Enforced by ruff's `PLR0917`
  (`too-many-positional-arguments`) with `max-positional-args = 0` — any function we
  define with a positional-or-keyword parameter is a lint error. Exceptions: `self`/
  `cls`, dunder methods Python itself always calls positionally (e.g. `__getitem__`,
  silenced with `# noqa: PLR0917` right at the definition), and third-party library
  calls we don't control (e.g. `np.array([1, 2, 3])`) — the rule only inspects
  functions *we* define, so external APIs are naturally exempt.
- **Files/classes are organized top-down, not bottom-up.** The entrypoint file (e.g.
  `problem.py`) is what a reader should open first; its `types.py` is the supporting
  detail underneath, not interleaved above it. Within `types.py` itself, apply the
  same rule at the class level: the most important/composite type first, the types/
  helpers it depends on further down — the reverse of the usual leaves-before-use
  convention. Concretely: if `X` relies on `Y` (has a field typed `Y`, or otherwise
  needs `Y` to do its job), `Y` goes below `X`. See
  `src/hitl_pmp/core/problem/tasks/types.py` for the pattern: `Task` (relies on `Goal`),
  `Goal` (relies on `GroundAtom`), `Predicate` (relies on `GroundAtom` — it's what
  `Predicate.__call__` produces), `GroundAtom` last, since both `Goal` and `Predicate`
  rely on it. This works in Python without any explicit forward-reference bookkeeping:
  names inside function/method bodies, and (with `from __future__ import annotations`)
  inside type annotations, are only resolved when actually used — which happens after
  the whole module has finished importing, so pydantic resolves the forward references
  itself on first real use. No `Model.model_rebuild()` call is needed.

## Contributing

There are currently no tests in this repo beyond a single import smoke test, because there is no concrete/production code yet — `core/` contains only abstract interfaces (`Environment`, `HumanOracle`, `Tasks`, `Problem`, `Method`, `Metrics`). Standard guidance is to not unit test an ABC directly: it has no behavior of its own, so a test against it either tests nothing or duplicates once per subclass. Test concrete subclasses instead. (If an ABC ever grows concrete helper methods, the pattern is a minimal throwaway "dummy" subclass that implements just enough of the abstract surface to exercise that logic.)

Once concrete code lands under `environments/`, `methods/`, etc., it should get real tests:

- Each concrete `Environment`'s `take_action()` — determinism given a fixed seed/current_state/action, and that `get_valid_actions()` never yields an out-of-bounds action.
- Each concrete `Problem`'s `execute_human_command()` wiring.
- Regression tests for any fixed bug (add a test when you fix it, so it can't silently regress).
- Property-based tests via `hypothesis` for invariants once they apply — e.g. Predicate/GroundAtom set consistency, encode/decode round-trips, or `take_action()` determinism (use `@settings(derandomize=True)` or explicit seeding). This is the same approach the `predicators` bilevel-planning codebase takes with its GroundAtom/predicate vocabulary.

`pytest`, `ruff`, `mypy`, and `pre-commit` are already wired up (see Checks below and `.github/workflows/ci.yml`) so this only needs to slot in as concrete code arrives — no new tooling required.

## Checks

```bash
pytest              # tests
ruff check .        # lint
ruff format .       # format
mypy src            # typecheck
```

All three run in CI (`.github/workflows/ci.yml`) on every push/PR to `main`.
Optionally, run `pre-commit install` to run lint/format/typecheck locally before each commit.

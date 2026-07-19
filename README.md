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

- [core/](src/hitl_pmp/core/README.md) — the fixed abstract interfaces (`Environment`, `HumanOracle`, `Problem`, `Method`, `Metrics`) and shared structs
- [environments/](src/hitl_pmp/environments/README.md) — concrete `Environment` + `Problem` implementations, one subfolder per domain
- [human_oracles/](src/hitl_pmp/human_oracles/README.md) — concrete `HumanOracle` implementations (the v0-v3 human-cost-model axis)
- [methods/](src/hitl_pmp/methods/README.md) — concrete `Method`/baseline implementations
- [adapters/](src/hitl_pmp/adapters/README.md) — the bidirectional Gym <-> `Environment` bridge
- [planning/](src/hitl_pmp/planning/README.md) — symbolic Predicate/GroundAtom -> PDDL -> Fast Downward bridge (optional, planning-based methods only)

## Conventions

- **All data/state lives in pydantic `BaseModel`s.** `dataclasses` and `attrs` are
  banned — enforced by ruff's `TID251` banned-api rule; importing either is a lint
  error, not just a style nit.
- **Supporting data types live in the file whose ABC they support, not a shared
  bucket file.** E.g. `Task`/`Goal`/`GroundAtom`/`Predicate` exist only to support
  `Problem`, so they live in `problem.py`, not a project-wide `structs.py`. Only types
  genuinely used by 3+ files with no single owner (`Type`, `Object`, `State`, `Action`,
  `Cost`) stay in a shared `structs.py`. Where two files each need a type the other
  owns (e.g. `Problem` needs `Method`'s `Policy`, `Method` needs `Problem`'s `Task`),
  resolve it with `if TYPE_CHECKING:`-guarded imports rather than merging the files —
  this works whenever the need is type-hint-only, never at runtime.
- **Behavior lives in static-method container classes, not OOP objects.**
  `Environment`, `HumanOracle`, `Problem`, `Method`, `Metrics` are never instantiated —
  think Java's "static class" / singleton-by-class-name idiom, not encapsulated
  instance state. Every method is `@staticmethod` (`@abc.abstractmethod` on the
  interface, concrete overrides on subclasses); any state a concrete implementation
  needs (e.g. `Problem.env`, `Problem.human`) is a `ClassVar` set once on the class
  itself, not passed into a constructor. This is an organizational choice, not
  idiomatic OOP — the goal is functional-style code (explicit data in, explicit data
  out) organized into namespacing containers.
- **Every parameter is keyword-only.** Enforced by ruff's `PLR0917`
  (`too-many-positional-arguments`) with `max-positional-args = 0` — any function we
  define with a positional-or-keyword parameter is a lint error. Exceptions: `self`/
  `cls`, dunder methods Python itself always calls positionally (e.g. `__getitem__`,
  silenced with `# noqa: PLR0917` right at the definition), and third-party library
  calls we don't control (e.g. `np.array([1, 2, 3])`) — the rule only inspects
  functions *we* define, so external APIs are naturally exempt.
- **Files/classes are organized top-down, not bottom-up.** Define the most
  important/composite ("top-level") class first, with the types/helpers it depends on
  further down the file — the reverse of the usual leaves-before-use convention. This
  works in Python because names inside function/method bodies, and (with `from
  __future__ import annotations`) inside type annotations, are only resolved when
  actually used, which happens after the whole module has finished importing. See
  `src/hitl_pmp/core/problem.py` for the pattern (`Problem` first, then `Task`, `Goal`,
  `GroundAtom`, `Predicate` in decreasing order of compositeness) — pydantic models
  with forward references need an explicit `Model.model_rebuild()` call once every
  class in the file is defined, which is what the loop at the bottom of that file
  does.

## Contributing

There are currently no tests in this repo beyond a single import smoke test, because there is no concrete/production code yet — `core/` contains only abstract interfaces (`Environment`, `HumanOracle`, `Problem`, `Method`, `Metrics`). Standard guidance is to not unit test an ABC directly: it has no behavior of its own, so a test against it either tests nothing or duplicates once per subclass. Test concrete subclasses instead. (If an ABC ever grows concrete helper methods, the pattern is a minimal throwaway "dummy" subclass that implements just enough of the abstract surface to exercise that logic.)

Once concrete code lands under `environments/`, `methods/`, etc., it should get real tests:

- Each concrete `Environment`'s `simulate()` — determinism given a fixed seed/state/action, and that `valid_actions()` never yields an out-of-bounds action.
- Each concrete `Problem`'s `request_human_reset()` wiring.
- Regression tests for any fixed bug (add a test when you fix it, so it can't silently regress).
- Property-based tests via `hypothesis` for invariants once they apply — e.g. Predicate/GroundAtom set consistency, encode/decode round-trips, or `simulate()` determinism (use `@settings(derandomize=True)` or explicit seeding). This is the same approach the `predicators` bilevel-planning codebase takes with its GroundAtom/predicate vocabulary.

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

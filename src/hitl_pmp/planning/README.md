# planning

Bridges the symbolic `core.Skill`/`core.Predicate`/`core.GroundAtom` layer (see
`core/method/types.py`, `core/problem/tasks/types.py`) to Fast Downward, a classical
PDDL planner — needed to port Practice Makes Perfect (EES)'s task planning faithfully
(see `methods/README.md`): EES plans to reach the precondition of whichever skill it
wants to practice next, using `-log(competence)` as each skill's edge cost so the
minimum-cost plan is the maximum-likelihood-of-success plan.

This follows the precedent set by the sibling repo one level up
(`hitl-practice/predicators/planning.py`'s `_sesame_plan_with_fast_downward` +
`predicators/third_party/fast_downward_translator/`). We are not extending that
codebase (per the project design doc: it is entangled and difficult to extend), but
its planner-invocation protocol is worth reusing rather than reinventing — this
package's `FastDownwardPlanner` mirrors it function-for-function.

## Why Fast Downward, not a hand-rolled planner

predicators' own built-in `astar` task planner does **not** support per-operator
costs at all (grepped: `ground_op_costs` is only ever consumed by the Fast-Downward
code path) — EES's whole mechanism depends on cost-aware *optimal* search
(`seq-opt-lmcut`), so a real external planner is genuinely load-bearing here, not
an implementation convenience.

## Files

- `pddl.py` — `PddlWriter`, a static-method container with no external
  dependencies (pure string generation, fully unit-tested without needing Fast
  Downward installed):
  - `abstract_state(*, state, objects, predicates) -> frozenset[GroundAtom]` —
    brute-force evaluates every `Predicate` against every type-matching,
    distinct-object combination to produce the symbolic abstraction a PDDL
    problem's `:init` section needs.
  - `write_domain(*, domain_name, types, predicates, skills) -> str` — one PDDL
    `:action` block per `Skill`, translating its `parameters`/`preconditions`/
    `add_effects`/`delete_effects` (`core.method.types.Variable`/`LiftedAtom`)
    directly into PDDL syntax.
  - `write_problem(*, problem_name, domain_name, objects, init_atoms, goal_atoms)
    -> str`.
- `fast_downward.py` — `FastDownwardPlanner`, a static-method container:
  - `plan(*, state, goal, objects, types, predicates, skills, fd_exec_path,
    ground_skill_costs=None, default_cost=1.0, timeout=10.0) -> list[GroundSkill]`
    — the two-stage protocol predicators itself uses: (1) translate the written
    PDDL to a SAS+ file via Fast Downward's own translator (`--sas-file`), (2)
    optionally patch that file's per-operator costs in place (`_patch_sas_costs`,
    a pure text transform matching predicators' `_update_sas_file_with_costs`
    exactly — sets SAS's `metric` flag on, since FD otherwise ignores non-unit
    costs), (3) run search on the (possibly patched) SAS file alone, which skips
    re-translation, (4) parse the printed plan back into `GroundSkill`s via a
    lowercased name lookup (Fast Downward itself lowercases everything
    internally, confirmed against predicators' own `.lower()` calls when
    matching SAS operator names).
  - `_translate`/`_search` are the only two functions that actually shell out to
    the `fast-downward.py` binary — both are `# pragma: no cover`, mirroring
    predicators' own precedent for this exact situation (their equivalent
    functions carry the same exclusion, for the same reason: CI can't be assumed
    to have Fast Downward installed). Everything around them (PDDL generation,
    cost injection, plan parsing) is pure and fully unit-tested without needing a
    real Fast Downward install; `tests/planning/test_fast_downward.py` also has
    one genuine end-to-end integration test against the real binary, gated with
    `pytest.mark.skipif` so it runs (and was used to validate this package
    during development) wherever Fast Downward is installed, and skips
    everywhere else rather than failing.

## Setup

Fast Downward is an external, non-Python dependency — not vendored, not a pip
package:

```bash
git clone https://github.com/aibasel/downward.git
cd downward && ./build.py
```

Pass the resulting `downward/` directory as `fd_exec_path` to
`FastDownwardPlanner.plan`. On macOS, also `brew install coreutils` (for
`gtimeout` — matches predicators' own platform check, `gtimeout` on Darwin,
`timeout` elsewhere).

**Not currently installed in CI** — `.github/workflows/ci.yml` does not build Fast
Downward, so the real end-to-end integration test above is skipped there; only the
pure/mocked tests run. Building Fast Downward from source takes real CI minutes, so
whether to add it is left as a deliberate, visible follow-up decision rather than
something silently baked into this change.

## Optionality

Only needed by a planning-based `Method` that requires symbolic search over
`GroundAtom`s to produce a plan — e.g. the Practice Makes Perfect (EES) reproduction
and its paper baselines (`methods/`). Pure deep-RL baselines like MAPLE-Q, or any
`Method` that never grounds its policy in symbolic search, skip this package
entirely and never import from it.

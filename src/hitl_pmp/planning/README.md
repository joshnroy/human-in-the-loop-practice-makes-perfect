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
- `grounding.py` — `SkillGrounder`, a static-method container: backtracking
  search that finds every applicable `GroundSkill` given a set of true
  `GroundAtom`s, without brute-force enumerating all object combinations (which
  doesn't scale — see its own TODOs for where that would bite at large object
  counts).

`fast_downward.py` (the `FastDownwardPlanner` shelling out to a real Fast Downward
binary, mirroring predicators' `_sesame_plan_with_fast_downward` two-stage
translate/patch-costs/search protocol) is **not implemented yet** — it isn't needed
by any `Method` currently in this codebase (Random Skills never plans). It lands in
a future PR alongside the Practice Makes Perfect (EES) reproduction itself, which is
the first `Method` that actually needs cost-aware optimal search over `GroundSkill`s.

## Setup

Nothing to install yet — the files above are pure Python with no external
dependencies. Fast Downward itself (an external, non-Python binary) will be added
as a setup step once `fast_downward.py` lands.

## Optionality

Only needed by a planning-based `Method` that requires symbolic search over
`GroundAtom`s to produce a plan — e.g. the Practice Makes Perfect (EES) reproduction
and its paper baselines (`methods/`). Pure deep-RL baselines like MAPLE-Q, or any
`Method` that never grounds its policy in symbolic search, skip this package
entirely and never import from it.

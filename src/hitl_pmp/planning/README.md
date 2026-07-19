# planning

Shared infrastructure for bridging the symbolic `core.Predicate` /
`core.GroundAtom` layer (see `core/problem/tasks/types.py`) to classical planners such as
Fast Downward.

This follows the precedent set by the sibling repo one level up
(`hitl-practice/predicators/third_party/fast_downward_translator/`), a
vendored PDDL-to-SAS+ translator invoked from
`predicators/planning.py`'s `_sesame_plan_with_fast_downward`. We are not
extending that codebase (per the project design doc: it is entangled and
difficult to extend), but its planner-invocation conventions are worth
reusing rather than reinventing.

## Intended pipeline

Once implemented, this module is expected to provide the machinery for:

1. Take a numpy `State` (`dict[Object, np.ndarray]`).
2. Apply the domain's `Predicate`s to the state to classify which
   `GroundAtom`s hold, producing a symbolic abstraction of the state.
3. Combine the abstract state, the domain's typed `Object`s, and a `Goal`
   (a set of `GroundAtom`s) into a grounded PDDL problem instance.
4. Translate the grounded PDDL problem to SAS+.
5. Hand the SAS+ representation to the Fast Downward binary and interpret
   the returned plan.

## Optionality

This is genuinely optional infrastructure. It is only needed if/when a
planning-based `Method` (e.g. a "planning_to_practice" baseline living in
`../methods/`) requires symbolic search over `GroundAtom`s to produce a
plan. Pure deep-RL baselines, or any `Method` that never grounds its policy
in symbolic search, skip this package entirely and never import from it.

## Status

Nothing in this package is implemented yet. This README and the empty
`__init__.py` exist only to reserve the module's place in the architecture.

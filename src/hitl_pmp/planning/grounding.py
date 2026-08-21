import itertools

from hitl_pmp.core.method.types import GroundSkill, Skill, Variable
from hitl_pmp.core.problem.environment.types import Object, State
from hitl_pmp.core.problem.tasks.types import GroundAtom, Predicate


class SkillGrounder:
    """Finds every GroundSkill whose (fully-grounded) preconditions actually hold
    in a given symbolic state -- domain-agnostic, shared by every practice-time
    Method that needs "what can I do right now" (the active-sampler-family
    baselines' candidate-scoring loop, and Random Skills' uniform choice among
    them). A backtracking search over skill.parameters, pruning a candidate
    binding the moment any precondition it fully determines doesn't hold --
    this keeps runtime small even at Light Switch's scale (100 cells) despite
    never special-casing the domain, since most bindings get pruned within the
    first one or two parameters (e.g. RobotInCell pins "current_cell" to
    wherever the robot actually is, not all 100 cells). A static-method
    container, never instantiated, same as every other business-logic class in
    this project.

    Deliberately does NOT force distinct objects across a parameter/argument
    slot list, in either `abstract_state` or `applicable_ground_skills` --
    that's not a general STRIPS/PDDL rule, predicators' own `abstract()`
    documents "Duplicate arguments in predicates are allowed." and its
    `get_object_combinations` applies no distinctness filter, and the real
    FastDownwardPlanner this must stay consistent with doesn't apply one
    either (its generated PDDL has no explicit (not (= ?x ?y)) constraints).
    Same-type-twice predicates DO exist here -- Light Switch's
    `Adjacent(cell, cell)` and Tossing Room's `Adjacent(room, room)` /
    `CanMoveRoom(room, room)` -- so the filter was not dead code. It was
    merely *inert*: none of those relations ever holds reflexively
    (Adjacent(c, c) and CanMoveRoom(r, r) are both always false), so the
    repeated-object combinations it skipped were exactly the ones
    `predicate.holds` would have rejected anyway. Measured over 60 sampled
    initial states per domain, dropping the filter changes the abstraction of
    zero states across all three. Ball-Ring has no same-type-twice predicate
    at all. So nothing observable changes today -- but this is shared
    `planning/` code, and the first reflexive relation anyone adds (a
    `SameRoom`, an `Equal`) must abstract correctly rather than silently lose
    its diagonal."""

    @staticmethod
    def abstract_state(
        *, state: State, objects: tuple[Object, ...], predicates: tuple[Predicate, ...]
    ) -> frozenset[GroundAtom]:
        """Every GroundAtom that currently holds, across every predicate and every
        type-matching combination of objects -- the symbolic abstraction
        applicable_ground_skills' true_atoms needs. Brute-force over all
        combinations is fine at Light Switch's scale (at most a couple hundred
        objects).

        Repeated objects across a predicate's slots are *included*, matching
        predicators' `abstract()` ("Duplicate arguments in predicates are
        allowed.") -- see the class docstring.

        TODO(scale): this is O(product of per-slot candidate counts) per predicate
        -- quadratic for a 2-arity predicate over one large type (e.g. Adjacent
        over grid_size cells). Fine at grid_size=100 (~10k checks), but would not
        scale to a domain with thousands of objects of the same type; a smarter
        abstraction (e.g. only checking spatially-plausible pairs) would be needed
        there."""
        atoms: set[GroundAtom] = set()
        for predicate in predicates:
            candidates_per_slot = [
                [obj for obj in objects if obj.type == object_type]
                for object_type in predicate.types
            ]
            for combo in itertools.product(*candidates_per_slot):
                if predicate.holds(state, combo):
                    atoms.add(GroundAtom(predicate=predicate, objects=combo))
        return frozenset(atoms)

    @staticmethod
    def all_possible_ground_atoms(
        *, objects: tuple[Object, ...], predicates: tuple[Predicate, ...]
    ) -> frozenset[GroundAtom]:
        """Every GroundAtom that COULD exist, across every predicate and every
        type-matching combination of objects -- `abstract_state`'s unfiltered cousin,
        with no `predicate.holds` check at all. `abstract_state` is "what's true here";
        this is "what could ever be said" -- the universe a caller needs when it has to
        make a ground atom false rather than merely not assert it (a closed-world reset
        that deletes everything not in a target set, which cannot know in general which
        atoms hold at the point it executes, so it must delete every possible atom
        outside the target rather than only the ones some particular state happens to
        have true).

        Same repeated-object inclusion as `abstract_state` (see the class docstring),
        for the same reason: this must stay a strict superset of `abstract_state`'s
        output for identical `objects`/`predicates`, or the delete-effect set it is used
        to build could miss an atom `abstract_state` would have reported true."""
        atoms: set[GroundAtom] = set()
        for predicate in predicates:
            candidates_per_slot = [
                [obj for obj in objects if obj.type == object_type]
                for object_type in predicate.types
            ]
            for combo in itertools.product(*candidates_per_slot):
                atoms.add(GroundAtom(predicate=predicate, objects=combo))
        return frozenset(atoms)

    @staticmethod
    def applicable_ground_skills(
        *, skills: tuple[Skill, ...], objects: tuple[Object, ...], true_atoms: frozenset[GroundAtom]
    ) -> list[GroundSkill]:
        ground_skills: list[GroundSkill] = []
        for skill in skills:
            ground_skills.extend(
                SkillGrounder._applicable_groundings(
                    skill=skill, objects=objects, true_atoms=true_atoms
                )
            )
        return ground_skills

    @staticmethod
    def _applicable_groundings(
        *, skill: Skill, objects: tuple[Object, ...], true_atoms: frozenset[GroundAtom]
    ) -> list[GroundSkill]:
        # TODO(scale): pruning here relies on preconditions actually constraining
        # each parameter early (e.g. RobotInCell pins the very first variable it
        # applies to). A skill whose preconditions leave several parameters
        # under-constrained until late in `skill.parameters`' order would fall
        # back toward the full O(len(objects) ** num_unconstrained_params)
        # search this is meant to avoid -- fine for Light Switch's actual skills,
        # but not a general guarantee for an arbitrary domain's operators.
        solutions: list[dict[Variable, Object]] = []

        def backtrack(*, assignment: dict[Variable, Object]) -> None:
            if len(assignment) == len(skill.parameters):
                solutions.append(dict(assignment))
                return
            next_variable = skill.parameters[len(assignment)]
            for obj in objects:
                if obj.type != next_variable.type:
                    continue
                trial = {**assignment, next_variable: obj}
                if SkillGrounder._consistent(skill=skill, assignment=trial, true_atoms=true_atoms):
                    backtrack(assignment=trial)

        backtrack(assignment={})
        return [
            GroundSkill(skill=skill, objects=tuple(assignment[p] for p in skill.parameters))
            for assignment in solutions
        ]

    @staticmethod
    def _consistent(
        *, skill: Skill, assignment: dict[Variable, Object], true_atoms: frozenset[GroundAtom]
    ) -> bool:
        for precondition in skill.preconditions:
            if not all(variable in assignment for variable in precondition.variables):
                continue  # not fully determined by this partial assignment yet
            ground_objects = tuple(assignment[variable] for variable in precondition.variables)
            if (
                GroundAtom(predicate=precondition.predicate, objects=ground_objects)
                not in true_atoms
            ):
                return False
        return True

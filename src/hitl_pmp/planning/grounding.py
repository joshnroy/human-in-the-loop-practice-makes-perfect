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

    Deliberately does NOT force distinct objects across a skill's parameter
    slots -- that's not a general STRIPS/PDDL rule, and the real
    FastDownwardPlanner this must stay consistent with doesn't apply one
    either (its generated PDDL has no explicit (not (= ?x ?y)) constraints).
    For Light Switch specifically, preconditions like Adjacent already rule
    out same-cell bindings on their own (Adjacent(c, c) is never true), so
    nothing is lost by leaving distinctness unenforced here."""

    @staticmethod
    def abstract_state(
        *, state: State, objects: tuple[Object, ...], predicates: tuple[Predicate, ...]
    ) -> frozenset[GroundAtom]:
        """Every GroundAtom that currently holds, across every predicate and every
        type-matching combination of distinct objects -- the symbolic abstraction
        applicable_ground_skills' true_atoms needs. Brute-force over all
        combinations is fine at Light Switch's scale (at most a couple hundred
        objects).

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
                if len(set(combo)) != len(combo):
                    continue  # a ground atom never holds of a repeated object here
                if predicate.holds(state, combo):
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

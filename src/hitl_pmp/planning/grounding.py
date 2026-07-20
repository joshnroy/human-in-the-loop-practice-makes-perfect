from hitl_pmp.core.method.types import GroundSkill, Skill, Variable
from hitl_pmp.core.problem.environment.types import Object
from hitl_pmp.core.problem.tasks.types import GroundAtom


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

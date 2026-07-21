from hitl_pmp.core.method.types import LiftedAtom, Skill, Variable
from hitl_pmp.core.problem.environment.types import Object, Type
from hitl_pmp.core.problem.tasks.types import GroundAtom, Predicate


class PddlWriter:
    """Renders this project's symbolic layer (`Skill`/`Predicate`/`Type`/`Object`/
    `GroundAtom`) as PDDL domain and problem text, byte-for-byte matching the format
    predicators emits (`utils.create_pddl_domain`/`create_pddl_problem`,
    `Predicate.pddl_str`, `_Atom.pddl_str`, `STRIPSOperator.pddl_str` in the sibling
    hitl-practice repo). A static-method container, never instantiated, same as every
    other business-logic class in this project.

    Only `FastDownwardPlanner` (fast_downward.py) consumes this -- it writes the two
    strings to temp files and hands them to a real Fast Downward.

    Two deliberate deviations from predicators, both forced by type differences:

    1. **The "?" prefix.** predicators' `Variable.name` is required to already start
       with "?" (its `pddl_str` just interpolates `p.name`), while ours are plain
       identifiers (`"robot"`, `"current_cell"` -- see
       `environments/lightswitch/skills.py`). This writer adds the "?" at write time,
       in both `:parameters` and every variable reference inside a precondition or
       effect atom, so the emitted PDDL is identical to what predicators would emit
       for the same operators. Nothing about `Variable` itself changes.
    2. **Sorting key.** predicators sorts by its entities' `__lt__` (which compares
       their `str`, i.e. "name:type"); ours have no ordering, so everything here is
       sorted by `.name` (types, predicates, skills, objects) or by the emitted PDDL
       string (atoms, whose leading token is the predicate name anyway). That is this
       codebase's deterministic-ordering choice: the same inputs in any order always
       produce the identical file, which is what determinism is actually for.

    No type hierarchy is emitted, because `Type` has no `parent` (see
    `core/problem/environment/types.py`) -- predicators' `create_pddl_types_str`
    hierarchy branch has no analogue here, only its flat "case 1".

    No action costs appear in the PDDL. That is predicators' design, kept
    deliberately: costs are injected later by patching the *translated SAS file*
    (`FastDownwardPlanner`, stage 2), which is what allows a cost per *ground* skill
    rather than the one-cost-per-lifted-action a `(:functions (total-cost))` PDDL
    domain could express."""

    @staticmethod
    def domain_str(
        *,
        skills: tuple[Skill, ...],
        predicates: tuple[Predicate, ...],
        types: tuple[Type, ...],
        domain_name: str = "hitlpmp",
    ) -> str:
        types_str = " ".join(t.name for t in sorted(types, key=lambda t: t.name))
        preds_str = "\n    ".join(
            PddlWriter._predicate_declaration_str(predicate=predicate)
            for predicate in sorted(predicates, key=lambda p: p.name)
        )
        skills_str = "\n\n  ".join(
            PddlWriter._action_str(skill=skill) for skill in sorted(skills, key=lambda s: s.name)
        )
        return f"""(define (domain {domain_name})
  (:requirements :typing)
  (:types {types_str})

  (:predicates\n    {preds_str}
  )

  {skills_str}
)"""

    @staticmethod
    def problem_str(
        *,
        objects: tuple[Object, ...],
        init_atoms: frozenset[GroundAtom],
        goal: frozenset[GroundAtom],
        domain_name: str = "hitlpmp",
        problem_name: str = "hitlpmpproblem",
    ) -> str:
        objects_str = "\n    ".join(
            f"{obj.name} - {obj.type.name}" for obj in sorted(objects, key=lambda o: o.name)
        )
        init_str = "\n    ".join(PddlWriter._sorted_ground_atom_strs(atoms=init_atoms))
        goal_str = "\n    ".join(PddlWriter._sorted_ground_atom_strs(atoms=goal))
        return f"""(define (problem {problem_name}) (:domain {domain_name})
  (:objects\n    {objects_str}
  )
  (:init\n    {init_str}
  )
  (:goal (and {goal_str}))
)
"""

    @staticmethod
    def _predicate_declaration_str(*, predicate: Predicate) -> str:
        """A `(:predicates ...)` entry: typed, positionally-indexed slots (`?x0`,
        `?x1`, ...), exactly like predicators' `Predicate.pddl_str`. A 0-arity
        predicate declares no slots at all."""
        if not predicate.types:
            return f"({predicate.name})"
        slots_str = " ".join(f"?x{i} - {t.name}" for i, t in enumerate(predicate.types))
        return f"({predicate.name} {slots_str})"

    @staticmethod
    def _action_str(*, skill: Skill) -> str:
        params_str = " ".join(
            f"{PddlWriter._variable_str(variable=parameter)} - {parameter.type.name}"
            for parameter in skill.parameters
        )
        preconds_str = "\n        ".join(
            PddlWriter._sorted_lifted_atom_strs(atoms=skill.preconditions)
        )
        effects_str = "\n        ".join(
            PddlWriter._sorted_lifted_atom_strs(atoms=skill.add_effects)
        )
        if skill.delete_effects:
            if effects_str:
                effects_str += "\n        "
            effects_str += "\n        ".join(
                f"(not {atom_str})"
                for atom_str in PddlWriter._sorted_lifted_atom_strs(atoms=skill.delete_effects)
            )
        return f"""(:action {skill.name}
    :parameters ({params_str})
    :precondition (and {preconds_str})
    :effect (and {effects_str})
  )"""

    @staticmethod
    def _sorted_lifted_atom_strs(*, atoms: frozenset[LiftedAtom]) -> list[str]:
        return sorted(PddlWriter._lifted_atom_str(atom=atom) for atom in atoms)

    @staticmethod
    def _sorted_ground_atom_strs(*, atoms: frozenset[GroundAtom]) -> list[str]:
        return sorted(PddlWriter._ground_atom_str(atom=atom) for atom in atoms)

    @staticmethod
    def _lifted_atom_str(*, atom: LiftedAtom) -> str:
        if not atom.variables:
            return f"({atom.predicate.name})"
        variables_str = " ".join(
            PddlWriter._variable_str(variable=variable) for variable in atom.variables
        )
        return f"({atom.predicate.name} {variables_str})"

    @staticmethod
    def _ground_atom_str(*, atom: GroundAtom) -> str:
        if not atom.objects:
            return f"({atom.predicate.name})"
        objects_str = " ".join(obj.name for obj in atom.objects)
        return f"({atom.predicate.name} {objects_str})"

    @staticmethod
    def _variable_str(*, variable: Variable) -> str:
        """PDDL requires a leading "?" on every variable; our `Variable.name` does not
        carry one (unlike predicators', which requires it) -- so add it here, and only
        here, so the two places variables are written (`:parameters` and atom
        references) can never drift apart."""
        return f"?{variable.name}"

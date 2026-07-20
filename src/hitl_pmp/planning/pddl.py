import itertools

from hitl_pmp.core.method.types import LiftedAtom, Skill
from hitl_pmp.core.problem.environment.types import Object, State, Type
from hitl_pmp.core.problem.tasks.types import GroundAtom, Predicate


class PddlWriter:
    """Translates this codebase's symbolic types (Predicate/GroundAtom/Skill) into
    PDDL domain/problem text, following the same conventions as predicators'
    utils.create_pddl_domain/create_pddl_problem
    (hitl-practice/predicators/utils.py) -- a static-method container, never
    instantiated, same as every other business-logic class in this project."""

    @staticmethod
    def abstract_state(
        *, state: State, objects: tuple[Object, ...], predicates: tuple[Predicate, ...]
    ) -> frozenset[GroundAtom]:
        """Every GroundAtom that currently holds, across every predicate and every
        type-matching combination of distinct objects -- the symbolic abstraction
        a PDDL problem's :init section needs. Brute-force over all combinations is
        fine at Light Switch's scale (at most a couple hundred objects)."""
        atoms: set[GroundAtom] = set()
        for predicate in predicates:
            candidates_per_slot = [
                [obj for obj in objects if obj.type == object_type]
                for object_type in predicate.types
            ]
            for combo in itertools.product(*candidates_per_slot):
                if len(set(combo)) != len(combo):
                    continue  # PDDL predicates never hold of a repeated object here
                if predicate.holds(state, combo):
                    atoms.add(GroundAtom(predicate=predicate, objects=combo))
        return frozenset(atoms)

    @staticmethod
    def write_domain(
        *,
        domain_name: str,
        types: tuple[Type, ...],
        predicates: tuple[Predicate, ...],
        skills: tuple[Skill, ...],
    ) -> str:
        lines = [
            f"(define (domain {domain_name})",
            "  (:requirements :typing)",
            f"  (:types {' '.join(object_type.name for object_type in types)})",
            "  (:predicates",
        ]
        for predicate in predicates:
            lines.append(f"    {PddlWriter._predicate_signature(predicate=predicate)}")
        lines.append("  )")
        for skill in skills:
            lines.extend(PddlWriter._action_block(skill=skill))
        lines.append(")")
        return "\n".join(lines)

    @staticmethod
    def write_problem(
        *,
        problem_name: str,
        domain_name: str,
        objects: tuple[Object, ...],
        init_atoms: frozenset[GroundAtom],
        goal_atoms: frozenset[GroundAtom],
    ) -> str:
        lines = [
            f"(define (problem {problem_name})",
            f"  (:domain {domain_name})",
            "  (:objects",
        ]
        for obj in objects:
            lines.append(f"    {obj.name} - {obj.type.name}")
        lines.append("  )")
        lines.append("  (:init")
        for atom in sorted(init_atoms, key=lambda atom: PddlWriter._ground_atom_key(atom=atom)):
            lines.append(f"    {PddlWriter._ground_atom_literal(atom=atom)}")
        lines.append("  )")
        goal_literals = [
            PddlWriter._ground_atom_literal(atom=atom)
            for atom in sorted(goal_atoms, key=lambda atom: PddlWriter._ground_atom_key(atom=atom))
        ]
        lines.append(f"  (:goal (and {' '.join(goal_literals)}))")
        lines.append(")")
        return "\n".join(lines)

    @staticmethod
    def _predicate_signature(*, predicate: Predicate) -> str:
        tokens = [
            predicate.name,
            *(f"?x{i} - {object_type.name}" for i, object_type in enumerate(predicate.types)),
        ]
        return f"({' '.join(tokens)})"

    @staticmethod
    def _action_block(*, skill: Skill) -> list[str]:
        params = " ".join(
            f"?{variable.name} - {variable.type.name}" for variable in skill.parameters
        )
        atom_key = lambda atom: PddlWriter._lifted_atom_key(atom=atom)  # noqa: E731
        preconditions = [
            PddlWriter._lifted_atom_literal(atom=atom)
            for atom in sorted(skill.preconditions, key=atom_key)
        ]
        add_literals = [
            PddlWriter._lifted_atom_literal(atom=atom)
            for atom in sorted(skill.add_effects, key=atom_key)
        ]
        delete_literals = [
            f"(not {PddlWriter._lifted_atom_literal(atom=atom)})"
            for atom in sorted(skill.delete_effects, key=atom_key)
        ]
        return [
            f"  (:action {skill.name}",
            f"    :parameters ({params})",
            f"    :precondition (and {' '.join(preconditions)})",
            f"    :effect (and {' '.join(add_literals + delete_literals)})",
            "  )",
        ]

    @staticmethod
    def _ground_atom_literal(*, atom: GroundAtom) -> str:
        tokens = [atom.predicate.name, *(obj.name for obj in atom.objects)]
        return f"({' '.join(tokens)})"

    @staticmethod
    def _lifted_atom_literal(*, atom: LiftedAtom) -> str:
        tokens = [atom.predicate.name, *(f"?{variable.name}" for variable in atom.variables)]
        return f"({' '.join(tokens)})"

    @staticmethod
    def _ground_atom_key(*, atom: GroundAtom) -> tuple[str, ...]:
        return (atom.predicate.name, *(obj.name for obj in atom.objects))

    @staticmethod
    def _lifted_atom_key(*, atom: LiftedAtom) -> tuple[str, ...]:
        return (atom.predicate.name, *(variable.name for variable in atom.variables))

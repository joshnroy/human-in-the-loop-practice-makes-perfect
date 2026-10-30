from hitl_pmp.core.method.types import ConditionalAddEffect, LiftedAtom, Skill, Variable
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

    Three deliberate deviations from predicators, the first two forced by type
    differences:

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
    3. **Ignore-effect quantifier names.** An `ignore_effects` predicate becomes a
       universally-quantified delete, `(forall (?x0 - t0 ...) (not (P ?x0 ...)))`,
       exactly as predicators writes it -- except that predicators hardcodes the
       `?x{i}` names, which a `forall` would silently *shadow* if the action itself
       declared a parameter named `?x0` (wiping every atom of that predicate rather
       than only the intended ones). Here the prefix is lengthened until it cannot
       collide with any of that action's own parameter names; for every skill in this
       repo it stays plain `x`, so the emitted text is identical to predicators'. A
       0-arity ignore effect has nothing to quantify over, so it is written as the
       plain `(not (P))` it is equivalent to -- `(forall () ...)` is not legal PDDL.

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
        for type_ in types:
            PddlWriter._reject_sigil(name=type_.name, kind="Type")
        for skill in skills:
            PddlWriter._reject_sigil(name=skill.name, kind="Skill")
        for predicate in predicates:
            PddlWriter._reject_sigil(name=predicate.name, kind="Predicate")
        types_str = " ".join(t.name for t in sorted(types, key=lambda t: t.name))
        constants = PddlWriter.collect_constants(skills=skills)
        constants_str = ""
        if constants:
            declarations = "\n    ".join(f"{obj.name} - {obj.type.name}" for obj in constants)
            constants_str = f"\n  (:constants\n    {declarations}\n  )\n"
        preds_str = "\n    ".join(
            PddlWriter._predicate_declaration_str(predicate=predicate)
            for predicate in sorted(predicates, key=lambda p: p.name)
        )
        skills_str = "\n\n  ".join(
            PddlWriter._action_str(skill=skill) for skill in sorted(skills, key=lambda s: s.name)
        )
        requirements = ":typing"
        if any(
            skill.conditional_add_effects or skill.conditional_ignore_effects for skill in skills
        ):
            requirements += " :conditional-effects"
        return f"""(define (domain {domain_name})
  (:requirements {requirements})
  (:types {types_str})
{constants_str}
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
        constants: tuple[Object, ...] = (),
        domain_name: str = "hitlpmp",
        problem_name: str = "hitlpmpproblem",
    ) -> str:
        constants_by_name = {
            obj.name.lower(): obj for obj in PddlWriter._unique_constants(objects=constants)
        }
        for obj in objects:
            PddlWriter._reject_sigil(name=obj.name, kind="Object")
            PddlWriter._reject_sigil(name=obj.type.name, kind="Type")
            constant = constants_by_name.get(obj.name.lower())
            if constant is not None and constant != obj:
                raise ValueError(f"Object {obj.name!r} conflicts with a domain constant")
        objects_str = "\n    ".join(
            f"{obj.name} - {obj.type.name}"
            for obj in sorted(objects, key=lambda o: o.name)
            if obj.name.lower() not in constants_by_name
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
    def collect_constants(*, skills: tuple[Skill, ...]) -> tuple[Object, ...]:
        """Collect effect-local constants once for domain and problem rendering."""
        constants: list[Object] = []
        for skill in skills:
            parameter_names = {parameter.name.lower() for parameter in skill.parameters}
            for effect in skill.conditional_add_effects:
                for variable, obj in effect.constant_bindings:
                    PddlWriter._reject_sigil(name=variable.name, kind="Variable")
                    if variable.name.lower() in parameter_names:
                        raise ValueError("Constant bindings cannot shadow skill parameters")
                    constants.append(obj)
        return PddlWriter._unique_constants(objects=tuple(constants))

    @staticmethod
    def _unique_constants(*, objects: tuple[Object, ...]) -> tuple[Object, ...]:
        by_name: dict[str, Object] = {}
        for obj in objects:
            PddlWriter._reject_sigil(name=obj.name, kind="Object")
            PddlWriter._reject_sigil(name=obj.type.name, kind="Type")
            name = obj.name.lower()
            if name in by_name and by_name[name] != obj:
                raise ValueError(f"Conflicting domain constants named {obj.name!r}")
            by_name[name] = obj
        return tuple(sorted(by_name.values(), key=lambda obj: obj.name))

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
        if skill.ignore_effects:
            if effects_str:
                effects_str += "\n        "
            prefix = PddlWriter._quantified_variable_prefix(skill=skill)
            effects_str += "\n        ".join(
                PddlWriter._ignore_effect_str(predicate=predicate, prefix=prefix)
                for predicate in sorted(skill.ignore_effects, key=lambda p: p.name)
            )
        if skill.conditional_add_effects:
            if effects_str:
                effects_str += "\n        "
            effects_str += "\n        ".join(
                sorted(
                    PddlWriter._conditional_add_effect_str(effect=effect)
                    for effect in skill.conditional_add_effects
                )
            )
        if skill.conditional_ignore_effects:
            if effects_str:
                effects_str += "\n        "
            prefix = PddlWriter._quantified_variable_prefix(skill=skill)
            effects_str += "\n        ".join(
                sorted(
                    PddlWriter._ignore_effect_str(
                        predicate=predicate, prefix=prefix, conditions=effect.conditions
                    )
                    for effect in skill.conditional_ignore_effects
                    for predicate in effect.ignore_effects
                )
            )
        return f"""(:action {skill.name}
    :parameters ({params_str})
    :precondition (and {preconds_str})
    :effect (and {effects_str})
  )"""

    @staticmethod
    def _conditional_add_effect_str(*, effect: ConditionalAddEffect) -> str:
        conditions = " ".join(
            PddlWriter._sorted_lifted_atom_strs(
                atoms=effect.conditions, constant_bindings=effect.constant_bindings
            )
        )
        additions = " ".join(
            PddlWriter._sorted_lifted_atom_strs(
                atoms=effect.add_effects, constant_bindings=effect.constant_bindings
            )
        )
        return f"(when (and {conditions}) (and {additions}))"

    @staticmethod
    def _ignore_effect_str(
        *, predicate: Predicate, prefix: str, conditions: frozenset[LiftedAtom] | None = None
    ) -> str:
        """One `ignore_effects` predicate as a universally-quantified delete: "after
        this action, no atom of this predicate holds, whatever its objects". Mirrors
        predicators' `STRIPSOperator.pddl_str`, apart from the quantifier naming and
        the 0-arity case (see this class's docstring, deviation 3)."""
        refs_str = " ".join(f"?{prefix}{i}" for i in range(len(predicate.types)))
        atom_str = f"({predicate.name} {refs_str})" if refs_str else f"({predicate.name})"
        effect_str = f"(not {atom_str})"
        if conditions is not None:
            guard = " ".join(PddlWriter._sorted_lifted_atom_strs(atoms=conditions))
            effect_str = f"(when (and {guard}) {effect_str})"
        if not predicate.types:
            return effect_str
        slots_str = " ".join(f"?{prefix}{i} - {t.name}" for i, t in enumerate(predicate.types))
        return f"(forall ({slots_str}) {effect_str})"

    @staticmethod
    def _quantified_variable_prefix(*, skill: Skill) -> str:
        """A base name for an ignore effect's `forall` variables that none of this
        action's own parameters can shadow -- `"x"` (matching predicators) unless the
        action really does declare a parameter named `x0`/`x1`/..., in which case the
        prefix grows (`xx`, `xxx`, ...) until it is unique. It grows by repeating a
        letter rather than prepending punctuation because PDDL names must *begin* with
        a letter."""
        reserved = {parameter.name for parameter in skill.parameters}
        prefix = "x"
        while any(name.startswith(prefix) and name[len(prefix) :].isdigit() for name in reserved):
            prefix = f"{prefix}x"
        return prefix

    @staticmethod
    def _sorted_lifted_atom_strs(
        *, atoms: frozenset[LiftedAtom], constant_bindings: tuple[tuple[Variable, Object], ...] = ()
    ) -> list[str]:
        return sorted(
            PddlWriter._lifted_atom_str(atom=atom, constant_bindings=constant_bindings)
            for atom in atoms
        )

    @staticmethod
    def _sorted_ground_atom_strs(*, atoms: frozenset[GroundAtom]) -> list[str]:
        return sorted(PddlWriter._ground_atom_str(atom=atom) for atom in atoms)

    @staticmethod
    def _lifted_atom_str(
        *, atom: LiftedAtom, constant_bindings: tuple[tuple[Variable, Object], ...] = ()
    ) -> str:
        if not atom.variables:
            return f"({atom.predicate.name})"
        constants = dict(constant_bindings)
        variables_str = " ".join(
            constants[variable].name
            if variable in constants
            else PddlWriter._variable_str(variable=variable)
            for variable in atom.variables
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
        PddlWriter._reject_sigil(name=variable.name, kind="Variable")
        return f"?{variable.name}"

    @staticmethod
    def _reject_sigil(*, name: str, kind: str) -> None:
        """Refuse any name that already carries PDDL's "?", at every site this writer
        interpolates one raw.

        The alternative is a **silent** wrong answer, and it has happened: a `Variable`
        declared "?robot" renders "??robot", Fast Downward's translator splits that into
        two tokens and exits 31, `FastDownwardPlanner` turns the non-zero exit into
        `PlanningFailure`, and `EesMethod._next_plan` catches that and returns a no-op.
        `environments/tossing3d/skills.py` declared all five of its variables that way,
        and `--env tossing3d --method ees` still exited 0 and wrote a full `stats.json`
        in which the method had never taken a single action.

        Checked for objects, predicates, types and skills too, not just variables: a "?"
        in any of those names produces the identical failure, and closing the class costs
        one call site each. Nothing raises today -- this is a tripwire, not a
        migration."""
        if name.startswith("?"):
            raise ValueError(
                f"{kind} name {name!r} already carries the '?' this writer adds; declare "
                "it plain (e.g. 'robot'), or the emitted PDDL is '??robot' and Fast "
                "Downward's translator rejects the whole domain."
            )

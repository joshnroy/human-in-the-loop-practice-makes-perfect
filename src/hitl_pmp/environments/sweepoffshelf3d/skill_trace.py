"""`SkillTraces.build`: one robot skill attempt as a compact, loggable `SkillTrace`.

A skill succeeds only when every effect its operator promises holds afterwards (the
all-or-nothing rule in `skills.py`), so the outcome is judged against the operator, not
only against whether the controller raised: a controller that returns cleanly while a
promised effect is missing is `effects_missing`, never `success`. A failure in which the
robot took no control step is a `plan_failure` -- every candidate was rejected before
anything moved -- and one after it moved is an `execution_failure`.

Decision logs already run to hundreds of GB, so a trace is bounded: at most `MAX_NOTES`
candidate notes (the overflow is counted, not kept) and `MAX_REASON_CHARS` per reason.
"""

from collections.abc import Iterable

import numpy as np

from hitl_pmp.core.method.types import GroundSkill, Skill
from hitl_pmp.core.problem.environment.types import Action
from hitl_pmp.core.problem.tasks.types import GroundAtom

from .environment import NOOP_ID, SKILL_IDS, SweepOffShelf3DEnvironment
from .objects import SweepOffShelf3DObjects as O
from .skills import SweepOffShelf3DSkills as S
from .types import AbstractAtom, SkillNote, SkillTrace

MAX_NOTES = 48
MAX_REASON_CHARS = 240
INTERRUPTED_PREFIX = "interrupted:"


class SkillTraces:
    """A static-method container, never instantiated."""

    @staticmethod
    def last(*, env: SweepOffShelf3DEnvironment) -> SkillTrace | None:
        """The trace of `env`'s last robot skill; None before the first one."""
        attempt = env.last_skill_attempt()
        if attempt is None:
            return None
        action, before, after, notes = attempt
        return SkillTraces.build(
            action=action,
            atoms_before=before,
            atoms_after=after,
            error=env.last_skill_error(),
            controller_steps=env.last_controller_steps(),
            notes=notes,
        )

    @staticmethod
    def build(
        *,
        action: Action,
        atoms_before: frozenset[AbstractAtom],
        atoms_after: frozenset[AbstractAtom],
        error: str | None,
        controller_steps: int,
        notes: list[SkillNote],
    ) -> SkillTrace:
        values = np.asarray(action, dtype=float).reshape(-1)
        skill_id = int(round(values[0])) if values.size else NOOP_ID
        ground = SkillTraces.ground(action=values)
        missing: list[str] = []
        if ground is not None:
            missing = SkillTraces.missing_effects(
                ground=ground, before=atoms_before, after=atoms_after
            )
        if skill_id == NOOP_ID:
            name = "noop"
        elif ground is None:
            name = f"unknown:{skill_id}"
        else:
            name = ground.skill.name
        params = [round(float(v), 6) for v in values[2:4]] if name == S.SWEEP_SHELF.name else []
        return SkillTrace(
            skill=name,
            objects=[] if ground is None else [obj.name for obj in ground.objects],
            params=params,
            outcome=SkillTraces.outcome(
                error=error, controller_steps=controller_steps, missing=missing
            ),
            error=None if error is None else error[:MAX_REASON_CHARS],
            controller_steps=controller_steps,
            missing_effects=missing,
            added=SkillTraces.names(atoms=atoms_after - atoms_before),
            removed=SkillTraces.names(atoms=atoms_before - atoms_after),
            notes=[SkillTraces.clip(note=note) for note in notes[:MAX_NOTES]],
            notes_dropped=max(0, len(notes) - MAX_NOTES),
        )

    @staticmethod
    def outcome(*, error: str | None, controller_steps: int, missing: list[str]) -> str:
        if error is not None and error.startswith(INTERRUPTED_PREFIX):
            return "interrupted"
        if error is not None:
            return "plan_failure" if controller_steps == 0 else "execution_failure"
        return "effects_missing" if missing else "success"

    @staticmethod
    def ground(*, action: np.ndarray) -> GroundSkill | None:
        """The operator `action` runs, bound to the objects its slots name; None for a
        no-op or an id no operator carries."""
        if action.size < 2:
            return None
        names = {index: name for name, index in SKILL_IDS.items()}
        name = names.get(int(round(action[0])))
        skills: dict[str, Skill] = {skill.name: skill for skill in S.robot_skills()}
        if name is None or name not in skills:
            return None
        skill = skills[name]
        cube = O.cubes[int(np.clip(round(action[1]), 0, len(O.cubes) - 1))]
        cube_slots = [p for p in skill.parameters if p.type == O.cube_type]
        by_type = {O.robot_type: O.robot, O.wiper_type: O.wiper, O.box_type: O.box}
        objects = []
        next_cube = iter(O.cubes)
        for parameter in skill.parameters:
            if parameter.type == O.cube_type:
                objects.append(cube if len(cube_slots) == 1 else next(next_cube))
            else:
                objects.append(by_type[parameter.type])
        return GroundSkill(skill=skill, objects=tuple(objects))

    @staticmethod
    def missing_effects(
        *, ground: GroundSkill, before: frozenset[AbstractAtom], after: frozenset[AbstractAtom]
    ) -> list[str]:
        promised = {SkillTraces.key(atom=atom) for atom in ground.add_effects}
        for effect in ground.conditional_add_effects:
            if {SkillTraces.key(atom=atom) for atom in effect.conditions} <= before:
                promised |= {SkillTraces.key(atom=atom) for atom in effect.add_effects}
        removed = {SkillTraces.key(atom=atom) for atom in ground.delete_effects} - promised
        absent = SkillTraces.names(atoms=promised - after)
        present = SkillTraces.names(atoms=removed & after)
        return absent + [f"not {name}" for name in present]

    @staticmethod
    def key(*, atom: GroundAtom) -> AbstractAtom:
        return (atom.predicate.name, tuple(o.name for o in atom.objects))

    @staticmethod
    def names(*, atoms: Iterable[AbstractAtom]) -> list[str]:
        return sorted(f"{name}({','.join(objects)})" for name, objects in atoms)

    @staticmethod
    def clip(*, note: SkillNote) -> SkillNote:
        params = None if note.params is None else [round(float(v), 4) for v in note.params]
        return SkillNote(
            stage=note.stage[:64],
            kind=note.kind[:32],
            reason=note.reason[:MAX_REASON_CHARS],
            params=params,
        )

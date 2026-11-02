"""Table-specific action grounding and effect verdicts using shared trace formatting."""

import numpy as np

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.environment.types import Action
from hitl_pmp.environments.sweepoffshelf3d.skill_trace import MAX_NOTES, MAX_REASON_CHARS
from hitl_pmp.environments.sweepoffshelf3d.skill_trace import SkillTraces as SharedSkillTraces
from hitl_pmp.environments.sweepoffshelf3d.types import AbstractAtom, SkillNote, SkillTrace

from .environment import SweepTable3DEnvironment
from .objects import SweepTable3DObjects as O
from .skills import NOOP_ID, SKILL_IDS
from .skills import SweepTable3DSkills as S


class SkillTraces:
    """Use table object types and operators with domain-neutral formatting helpers."""

    @staticmethod
    def last(*, env: SweepTable3DEnvironment) -> SkillTrace | None:
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
        missing = (
            []
            if ground is None
            else SharedSkillTraces.missing_effects(
                ground=ground, before=atoms_before, after=atoms_after
            )
        )
        name = (
            "noop"
            if skill_id == NOOP_ID
            else (f"unknown:{skill_id}" if ground is None else ground.skill.name)
        )
        return SkillTrace(
            skill=name,
            objects=[] if ground is None else [obj.name for obj in ground.objects],
            params=[round(float(value), 6) for value in values[2:4]]
            if name == S.SWEEP_TABLE.name
            else [],
            outcome=SharedSkillTraces.outcome(
                error=error, controller_steps=controller_steps, missing=missing
            ),
            error=None if error is None else error[:MAX_REASON_CHARS],
            controller_steps=controller_steps,
            missing_effects=missing,
            added=SharedSkillTraces.names(atoms=atoms_after - atoms_before),
            removed=SharedSkillTraces.names(atoms=atoms_before - atoms_after),
            notes=[
                SharedSkillTraces.clip(note=note) for note in SharedSkillTraces.kept(notes=notes)
            ],
            notes_dropped=max(0, len(notes) - MAX_NOTES),
        )

    @staticmethod
    def ground(*, action: np.ndarray) -> GroundSkill | None:
        if action.size < 2:
            return None
        names = {index: name for name, index in SKILL_IDS.items()}
        skills = {skill.name: skill for skill in S.robot_skills()}
        name = names.get(int(round(action[0])))
        if name not in skills:
            return None
        skill = skills[name]
        cube = O.cubes[int(np.clip(round(action[1]), 0, len(O.cubes) - 1))]
        bindings = {
            "robot": O.robot,
            "wiper": O.wiper,
            "box": O.box,
            "cube": cube,
            **{f"cube{index}": ball for index, ball in enumerate(O.balls)},
        }
        return GroundSkill(
            skill=skill,
            objects=tuple(bindings[parameter.name] for parameter in skill.parameters),
        )

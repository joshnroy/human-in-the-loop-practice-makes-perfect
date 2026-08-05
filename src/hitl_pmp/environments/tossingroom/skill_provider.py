import numpy as np

from hitl_pmp.core.method.skill_provider import OraclePolicyProvider, SkillProvider
from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Skill
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.problem.tasks.types import Goal, Predicate

from .environment import TossingRoomEnvironment
from .predicates import (
    ADJACENT,
    BIN_ACCEPTS_ITEM,
    BIN_EMPTY,
    BIN_IN_ROOM,
    BUTTON_FOR_BIN,
    BUTTON_IN_ROOM,
    CAN_MOVE_ROOM,
    HAND_EMPTY,
    HOLDING,
    ITEM_IN_BIN,
    PILE_IN_ROOM,
    ROBOT_IN_ROOM,
)
from .skill_oracle_policy import SkillOraclePolicy
from .skills import TossingRoomSkills


class TossingRoomSkillProvider(SkillProvider):
    """The Tossing Room domain's `SkillProvider`: exposes `TossingRoomSkills` and this
    domain's predicates/types/objects to a domain-agnostic Method, delegating
    `sample_params`/`compute_action` straight to `TossingRoomSkills`. Mirrors
    `LightSwitchSkillProvider`; only the injection seam is new."""

    env: TossingRoomEnvironment

    def skills(self) -> tuple[Skill, ...]:
        return (
            TossingRoomSkills.PICKUP,
            TossingRoomSkills.MOVE_ROOM,
            TossingRoomSkills.THROW,
            TossingRoomSkills.PRESS,
        )

    def predicates(self) -> tuple[Predicate, ...]:
        return (
            ROBOT_IN_ROOM,
            HAND_EMPTY,
            HOLDING,
            ADJACENT,
            ITEM_IN_BIN,
            BIN_EMPTY,
            BIN_IN_ROOM,
            BUTTON_IN_ROOM,
            BUTTON_FOR_BIN,
            PILE_IN_ROOM,
            BIN_ACCEPTS_ITEM,
            CAN_MOVE_ROOM,
        )

    def types(self) -> tuple[Type, ...]:
        return (
            TossingRoomEnvironment.robot_type,
            TossingRoomEnvironment.room_type,
            TossingRoomEnvironment.bin_type,
            TossingRoomEnvironment.button_type,
            TossingRoomEnvironment.item_type,
            TossingRoomEnvironment.pile_type,
        )

    def objects(self) -> tuple[Object, ...]:
        env = self.env
        return (
            env.robot,
            env.recycling_bin,
            env.trash_bin,
            env.trash_button,
            env.recycling_button,
            env.pile,
            env.trash,
            env.recycling,
            *env.get_rooms(),
        )

    def sample_params(self, *, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        return TossingRoomSkills.sample_params(ground_skill=ground_skill, rng=rng)

    def compute_action(
        self, *, ground_skill: GroundSkill, params: np.ndarray, state: State
    ) -> Action:
        return TossingRoomSkills.compute_action(
            ground_skill=ground_skill, params=params, state=state
        )


class TossingRoomOracle(OraclePolicyProvider):
    """Tossing Room's privileged solver, driving `SkillOracleMethod` as the upper-bound
    baseline. Goal-DEPENDENT: from ground-truth state alone it cannot tell a
    throw-recycling task from a throw-trash one, so `get_labeled_action` consults the
    task goal (threaded down by `SkillOracleMethod.get_task_policy`) to pick which
    item/bin/room to head for -- unlike the goal-agnostic Light Switch / Ball-Ring
    oracles."""

    env: TossingRoomEnvironment

    def get_labeled_action(self, *, state: State, goal: Goal) -> LabeledAction:
        return SkillOraclePolicy.get_labeled_action(state=state, env=self.env, goal=goal)

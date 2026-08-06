import numpy as np

from hitl_pmp.core.method.skill_provider import OraclePolicyProvider, SkillProvider
from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Skill
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.problem.tasks.types import Goal, Predicate

from .environment import TossingRoomSplitIdentityEnvironment
from .predicates import (
    ADJACENT,
    CAN_MOVE_ROOM,
    HAND_EMPTY,
    HOLDING_RECYCLING,
    HOLDING_TRASH,
    PILE_IN_ROOM,
    RECYCLING_BIN_EMPTY,
    RECYCLING_BIN_IN_ROOM,
    RECYCLING_BUTTON_IN_ROOM,
    RECYCLING_IN_BIN,
    ROBOT_IN_ROOM,
    TRASH_BIN_EMPTY,
    TRASH_BIN_IN_ROOM,
    TRASH_BUTTON_IN_ROOM,
    TRASH_IN_BIN,
)
from .skill_oracle_policy import SkillOraclePolicy
from .skills import TossingRoomSplitIdentitySkills


class TossingRoomSplitIdentitySkillProvider(SkillProvider):
    """This domain's `SkillProvider`: exposes `TossingRoomSplitIdentitySkills` and this domain's
    predicates/types/objects to a domain-agnostic Method.

    Seven lifted skills where Tossing Room has four -- `Pickup`, `Throw` and `Press` each
    split per kind. Only the throw split matters to learning (`Pickup` and both presses
    are `param_dim=0`, so none gets a sampler); see `skills.py`."""

    env: TossingRoomSplitIdentityEnvironment

    def skills(self) -> tuple[Skill, ...]:
        return (
            TossingRoomSplitIdentitySkills.PICKUP_TRASH,
            TossingRoomSplitIdentitySkills.PICKUP_RECYCLING,
            TossingRoomSplitIdentitySkills.MOVE_ROOM,
            TossingRoomSplitIdentitySkills.THROW_TRASH,
            TossingRoomSplitIdentitySkills.THROW_RECYCLING,
            TossingRoomSplitIdentitySkills.PRESS_TRASH,
            TossingRoomSplitIdentitySkills.PRESS_RECYCLING,
        )

    def predicates(self) -> tuple[Predicate, ...]:
        return (
            ROBOT_IN_ROOM,
            HAND_EMPTY,
            HOLDING_TRASH,
            HOLDING_RECYCLING,
            ADJACENT,
            TRASH_IN_BIN,
            RECYCLING_IN_BIN,
            TRASH_BIN_EMPTY,
            RECYCLING_BIN_EMPTY,
            TRASH_BIN_IN_ROOM,
            RECYCLING_BIN_IN_ROOM,
            TRASH_BUTTON_IN_ROOM,
            RECYCLING_BUTTON_IN_ROOM,
            PILE_IN_ROOM,
            CAN_MOVE_ROOM,
        )

    def types(self) -> tuple[Type, ...]:
        return (
            TossingRoomSplitIdentityEnvironment.robot_type,
            TossingRoomSplitIdentityEnvironment.room_type,
            TossingRoomSplitIdentityEnvironment.trash_bin_type,
            TossingRoomSplitIdentityEnvironment.recycling_bin_type,
            TossingRoomSplitIdentityEnvironment.trash_button_type,
            TossingRoomSplitIdentityEnvironment.recycling_button_type,
            TossingRoomSplitIdentityEnvironment.trash_type,
            TossingRoomSplitIdentityEnvironment.recycling_type,
            TossingRoomSplitIdentityEnvironment.pile_type,
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
        return TossingRoomSplitIdentitySkills.sample_params(ground_skill=ground_skill, rng=rng)

    def compute_action(
        self, *, ground_skill: GroundSkill, params: np.ndarray, state: State
    ) -> Action:
        return TossingRoomSplitIdentitySkills.compute_action(
            ground_skill=ground_skill, params=params, state=state
        )


class TossingRoomSplitIdentityOracle(OraclePolicyProvider):
    """This domain's privileged solver, driving `SkillOracleMethod` as the upper-bound
    baseline. Goal-DEPENDENT: the goal picks which throw *skill* to use, not merely
    which objects to bind -- see `skill_oracle_policy.py`."""

    env: TossingRoomSplitIdentityEnvironment

    def get_labeled_action(self, *, state: State, goal: Goal) -> LabeledAction:
        return SkillOraclePolicy.get_labeled_action(state=state, env=self.env, goal=goal)

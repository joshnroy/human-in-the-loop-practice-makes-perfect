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
    HOLDING_RECYCLING,
    HOLDING_TRASH,
    ITEM_IN_BIN,
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
from .skills import TossingRoomSkills, TossingRoomUnsplitSkills


class TossingRoomSkillProvider(SkillProvider):
    """This domain's `SkillProvider`: exposes its lifted skills and its
    predicates/types/objects to a domain-agnostic Method.

    **Which symbolic layer it exposes is `self.env.unsplit_skills`'s call**, and this is
    the one place that choice is made for a Method. By default, seven lifted skills:
    `Pickup`, `Throw` and `Press` each split per kind. Under `--unsplit-skills`, the four
    unsplit ones, with a single `Throw` whose `?item` ranges over both kinds. Only the
    throw split matters to learning (`Pickup` and the presses are `param_dim=0`, so none
    gets a sampler); the rest of the collapse is forced by the type system rather than
    chosen. See `skills.py`.

    Every method below dispatches on that one field rather than on a subclass: the two
    arms are the same domain under two decompositions, and a second provider class would
    invite the two to drift in something other than the decomposition."""

    env: TossingRoomEnvironment

    def skills(self) -> tuple[Skill, ...]:
        if self.env.unsplit_skills:
            return (
                TossingRoomUnsplitSkills.PICKUP,
                TossingRoomUnsplitSkills.MOVE_ROOM,
                TossingRoomUnsplitSkills.THROW,
                TossingRoomUnsplitSkills.PRESS,
            )
        return (
            TossingRoomSkills.PICKUP_TRASH,
            TossingRoomSkills.PICKUP_RECYCLING,
            TossingRoomSkills.MOVE_ROOM,
            TossingRoomSkills.THROW_TRASH,
            TossingRoomSkills.THROW_RECYCLING,
            TossingRoomSkills.PRESS_TRASH,
            TossingRoomSkills.PRESS_RECYCLING,
        )

    def predicates(self) -> tuple[Predicate, ...]:
        if self.env.unsplit_skills:
            return (
                ROBOT_IN_ROOM,
                HAND_EMPTY,
                HOLDING,
                ADJACENT,
                ITEM_IN_BIN,
                BIN_EMPTY,
                BIN_IN_ROOM,
                BUTTON_IN_ROOM,
                # Not tautologies on this arm: with one item type and one bin type a
                # mismatched pairing is well-typed, so these are what exclude it.
                BIN_ACCEPTS_ITEM,
                BUTTON_FOR_BIN,
                PILE_IN_ROOM,
                CAN_MOVE_ROOM,
            )
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
        if self.env.unsplit_skills:
            return (
                TossingRoomEnvironment.robot_type,
                TossingRoomEnvironment.room_type,
                TossingRoomEnvironment.bin_type,
                TossingRoomEnvironment.button_type,
                TossingRoomEnvironment.item_type,
                TossingRoomEnvironment.pile_type,
            )
        return (
            TossingRoomEnvironment.robot_type,
            TossingRoomEnvironment.room_type,
            TossingRoomEnvironment.trash_bin_type,
            TossingRoomEnvironment.recycling_bin_type,
            TossingRoomEnvironment.trash_button_type,
            TossingRoomEnvironment.recycling_button_type,
            TossingRoomEnvironment.trash_type,
            TossingRoomEnvironment.recycling_type,
            TossingRoomEnvironment.pile_type,
        )

    def objects(self) -> tuple[Object, ...]:
        """Resolved through the environment's own `*_for_kind` accessors, so the objects a
        Method grounds over carry whichever typing this instance's `State` is keyed by --
        naming `env.trash_bin` here would hand the unsplit arm a key its own states do not
        contain."""
        env = self.env
        return (
            env.robot,
            env.bin_for_kind(kind=env.RECYCLING_KIND),
            env.bin_for_kind(kind=env.TRASH_KIND),
            env.button_for_kind(kind=env.TRASH_KIND),
            env.button_for_kind(kind=env.RECYCLING_KIND),
            env.pile,
            env.item_for_kind(kind=env.TRASH_KIND),
            env.item_for_kind(kind=env.RECYCLING_KIND),
            *env.get_rooms(),
        )

    def sample_params(self, *, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        if self.env.unsplit_skills:
            return TossingRoomUnsplitSkills.sample_params(ground_skill=ground_skill, rng=rng)
        return TossingRoomSkills.sample_params(ground_skill=ground_skill, rng=rng)

    def compute_action(
        self, *, ground_skill: GroundSkill, params: np.ndarray, state: State
    ) -> Action:
        if self.env.unsplit_skills:
            return TossingRoomUnsplitSkills.compute_action(
                ground_skill=ground_skill, params=params, state=state
            )
        return TossingRoomSkills.compute_action(
            ground_skill=ground_skill, params=params, state=state
        )


class TossingRoomOracle(OraclePolicyProvider):
    """This domain's privileged solver, driving `SkillOracleMethod` as the upper-bound
    baseline. Goal-DEPENDENT: the goal picks which throw *skill* to use, not merely
    which objects to bind -- see `skill_oracle_policy.py`."""

    env: TossingRoomEnvironment

    def get_labeled_action(self, *, state: State, goal: Goal) -> LabeledAction:
        return SkillOraclePolicy.get_labeled_action(state=state, env=self.env, goal=goal)

"""The two injection seams a domain-agnostic `Method` needs from Tossing3D."""

import numpy as np

from hitl_pmp.core.method.skill_provider import OraclePolicyProvider, SkillProvider
from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Skill
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.problem.tasks.types import Goal, Predicate

from .environment import Tossing3DEnvironment
from .predicates import HAND_EMPTY, HOLDING, IN_BIN, ON_GROUND, REACHABLE
from .skill_oracle_policy import ORACLE_THROW_STANDOFF, SkillOraclePolicy
from .skills import Tossing3DSkills


class Tossing3DSkillProvider(SkillProvider):
    """Tossing3D's `SkillProvider`, mirroring `TossingRoomSkillProvider`.

    `objects()` is a fixed four: upstream's task JSON names exactly one cube, one bin and
    one barrier, plus the robot. There is no configuration that changes the cast -- `o2`
    would add a second cube, and this domain does not support it (see the README).

    It was five until the goal region stopped being a symbolic object. The scored box is
    still in the `State`, carried on the bin (see `predicates.py`'s module docstring); it
    is simply not something a planner binds a variable to, because no skill can act on it.
    """

    env: Tossing3DEnvironment

    def skills(self) -> tuple[Skill, ...]:
        return (Tossing3DSkills.PICK_CUBE, Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS)

    def predicates(self) -> tuple[Predicate, ...]:
        return (IN_BIN, HAND_EMPTY, HOLDING, ON_GROUND, REACHABLE)

    def types(self) -> tuple[Type, ...]:
        return (
            Tossing3DEnvironment.robot_type,
            Tossing3DEnvironment.cube_type,
            Tossing3DEnvironment.bin_type,
            Tossing3DEnvironment.barrier_type,
        )

    def objects(self) -> tuple[Object, ...]:
        env = self.env
        return (env.robot, env.cube, env.bin, env.barrier)

    def sample_params(self, *, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        return Tossing3DSkills.sample_params(ground_skill=ground_skill, rng=rng)

    def compute_action(
        self, *, ground_skill: GroundSkill, params: np.ndarray, state: State
    ) -> Action:
        return Tossing3DSkills.compute_action(ground_skill=ground_skill, params=params, state=state)


class Tossing3DOracle(OraclePolicyProvider):
    """Tossing3D's privileged solver, driving `SkillOracleMethod`.

    Goal-agnostic (one goal family; see `skill_oracle_policy.py`). `throw_standoff` is a
    constructor field rather than a constant read off the policy module because the
    standoff that solves depends on which scene is loaded -- 1.35 on the coincident
    config, 1.55 on stock -- and the CLI has to be able to say which.
    """

    env: Tossing3DEnvironment
    throw_standoff: float = ORACLE_THROW_STANDOFF

    def get_labeled_action(self, *, state: State, goal: Goal) -> LabeledAction:
        return SkillOraclePolicy.get_labeled_action(
            state=state, env=self.env, goal=goal, throw_standoff=self.throw_standoff
        )

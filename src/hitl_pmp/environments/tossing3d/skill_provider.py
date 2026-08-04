import numpy as np

from hitl_pmp.core.method.skill_provider import OraclePolicyProvider, SkillProvider
from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Skill
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.problem.tasks.types import Goal, Predicate

from .environment import Tossing3DEnvironment
from .predicates import AT_THROW_POSE, HAND_EMPTY, HOLDING, IN_GOAL_REGION, REACHABLE
from .skill_oracle_policy import SkillOraclePolicy
from .skills import Tossing3DSkills


class Tossing3DSkillProvider(SkillProvider):
    """The Tossing3D domain's `SkillProvider`: exposes `Tossing3DSkills` and this
    domain's predicates/types/objects to a domain-agnostic Method. Mirrors Tossing
    Room's; the only difference is that `sample_params` forwards the environment
    instance, because this domain's sampling bounds are per-instance configuration
    (the swing prior is a CLI flag) rather than module constants."""

    env: Tossing3DEnvironment

    def skills(self) -> tuple[Skill, ...]:
        return (
            Tossing3DSkills.PICK,
            Tossing3DSkills.MOVE_TO_THROW_POSE,
            Tossing3DSkills.TOSS,
        )

    def predicates(self) -> tuple[Predicate, ...]:
        return (IN_GOAL_REGION, HAND_EMPTY, HOLDING, REACHABLE, AT_THROW_POSE)

    def types(self) -> tuple[Type, ...]:
        return (
            Tossing3DEnvironment.robot_type,
            Tossing3DEnvironment.cube_type,
            Tossing3DEnvironment.bin_type,
            Tossing3DEnvironment.barrier_type,
            Tossing3DEnvironment.region_type,
        )

    def objects(self) -> tuple[Object, ...]:
        # `scene` is deliberately absent: it carries the KINDER reset seed, which no
        # predicate reads and nothing plans over, so grounding skills across it would
        # only widen the planner's search for nothing.
        env = self.env
        return (env.robot, env.cube, env.bin_object, env.barrier, env.goal_region)

    def sample_params(self, *, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        return Tossing3DSkills.sample_params(ground_skill=ground_skill, rng=rng, env=self.env)

    def compute_action(
        self, *, ground_skill: GroundSkill, params: np.ndarray, state: State
    ) -> Action:
        return Tossing3DSkills.compute_action(ground_skill=ground_skill, params=params, state=state)


class Tossing3DOracle(OraclePolicyProvider):
    """Tossing3D's privileged solver, driving `SkillOracleMethod` as the upper-bound
    baseline. Goal-agnostic (the domain has one goal), so `goal` is unused."""

    env: Tossing3DEnvironment

    def get_labeled_action(self, *, state: State, goal: Goal) -> LabeledAction:
        del goal
        return SkillOraclePolicy.get_labeled_action(state=state, env=self.env)

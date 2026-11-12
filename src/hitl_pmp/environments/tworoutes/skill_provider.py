import numpy as np

from hitl_pmp.core.method.skill_provider import OraclePolicyProvider, SkillProvider
from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Skill
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.problem.tasks.types import Goal, Predicate

from .environment import ROUTE_B, TwoRoutesEnvironment
from .predicates import TwoRoutesPredicates
from .skills import TwoRoutesSkills


class TwoRoutesSkillProvider(SkillProvider):
    """No skill has a parameter: what is learned is scripted in the environment."""

    env: TwoRoutesEnvironment
    offer_human_reset: bool = True

    def skills(self) -> tuple[Skill, ...]:
        return TwoRoutesSkills.all()

    def predicates(self) -> tuple[Predicate, ...]:
        return TwoRoutesPredicates.all()

    def types(self) -> tuple[Type, ...]:
        return (TwoRoutesEnvironment.robot_type,)

    def objects(self) -> tuple[Object, ...]:
        return (self.env.robot,)

    def ground(self, *, name: str) -> GroundSkill:
        skill = next(s for s in (*self.skills(), TwoRoutesSkills.human_reset()) if s.name == name)
        return GroundSkill(skill=skill, objects=(self.env.robot,))

    def sample_params(self, *, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        del ground_skill, rng
        return np.zeros(0)

    def compute_action(
        self, *, ground_skill: GroundSkill, params: np.ndarray, state: State
    ) -> Action:
        del params, state
        return self.env.action_for(skill=ground_skill.skill.name)

    def human_cube_bin_reset_skill(self) -> GroundSkill | None:
        if not self.offer_human_reset:
            return None
        return self.ground(name=TwoRoutesSkills.human_reset().name)

    def movables_reset_destination(self, *, ground_skill: GroundSkill) -> str | None:
        return ground_skill.skill.name


class TwoRoutesOracle(OraclePolicyProvider):
    """Route B, the better-rewarded route, once: an episode is one attempt."""

    env: TwoRoutesEnvironment

    def get_labeled_action(self, *, state: State, goal: Goal) -> LabeledAction:
        del goal
        if not state.get(obj=self.env.robot, feature_name="ready"):
            return LabeledAction(action=self.env.noop_action(), label="no-op")
        return LabeledAction(action=self.env.action_for(skill=ROUTE_B), label=ROUTE_B)

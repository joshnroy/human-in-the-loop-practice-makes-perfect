import numpy as np

from hitl_pmp.core.method.skill_provider import OraclePolicyProvider, SkillProvider
from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Skill
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.problem.tasks.types import Goal, Predicate

from .environment import LightSwitchEnvironment
from .predicates import ADJACENT, LIGHT_IN_CELL, LIGHT_OFF, LIGHT_ON, ROBOT_IN_CELL
from .skill_oracle_policy import SkillOraclePolicy
from .skills import LightSwitchSkills


class LightSwitchSkillProvider(SkillProvider):
    """The Light Switch domain's `SkillProvider`: exposes `LightSwitchSkills` and
    this domain's predicates/types/objects to a domain-agnostic Method
    (EesMethod/RandomSkillsMethod). Delegates `sample_params`/`compute_action`
    straight to `LightSwitchSkills` (unchanged), so behavior is identical to the
    previous hardcoded wiring; only the injection seam is new."""

    env: LightSwitchEnvironment

    def skills(self) -> tuple[Skill, ...]:
        return (
            LightSwitchSkills.MOVE_ROBOT,
            LightSwitchSkills.TURN_ON_LIGHT,
            LightSwitchSkills.TURN_OFF_LIGHT,
            LightSwitchSkills.JUMP_TO_LIGHT,
        )

    def predicates(self) -> tuple[Predicate, ...]:
        return (LIGHT_ON, LIGHT_OFF, ROBOT_IN_CELL, LIGHT_IN_CELL, ADJACENT)

    def types(self) -> tuple[Type, ...]:
        return (
            LightSwitchEnvironment.robot_type,
            LightSwitchEnvironment.light_type,
            LightSwitchEnvironment.cell_type,
        )

    def objects(self) -> tuple[Object, ...]:
        return (self.env.robot, self.env.light, *self.env.get_cells())

    def sample_params(self, *, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        return LightSwitchSkills.sample_params(ground_skill=ground_skill, rng=rng)

    def compute_action(
        self, *, ground_skill: GroundSkill, params: np.ndarray, state: State
    ) -> Action:
        return LightSwitchSkills.compute_action(
            ground_skill=ground_skill, params=params, state=state
        )


class LightSwitchOracle(OraclePolicyProvider):
    """The Light Switch privileged solver, driving `SkillOracleMethod`. Wraps the
    existing `SkillOraclePolicy` (unchanged), binding it to this domain's env
    instance."""

    env: LightSwitchEnvironment

    def get_labeled_action(self, *, state: State, goal: Goal) -> LabeledAction:
        del goal  # Light Switch always drives toward the light from privileged state
        return SkillOraclePolicy.get_labeled_action(state=state, env=self.env)

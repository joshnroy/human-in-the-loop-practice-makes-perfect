"""Replace the sampler, retaining structured practice/deployment planning."""

from collections.abc import Callable
from typing import Any

import numpy as np
from pydantic import Field

from hitl_pmp.core.method.types import GroundSkill, LabeledAction
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.methods.belief_space.tossing3d_constants import TOSS_SKILL
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod


def skill_action(*, ground_skill: GroundSkill) -> LabeledAction:
    names = {
        "PickCube": Tossing3DEnvironment.pick_cube_id,
        "MoveToTossLocationAndToss": Tossing3DEnvironment.move_to_toss_location_and_toss_id,
        "OpenGripper": Tossing3DEnvironment.open_gripper_id,
    }
    name = ground_skill.skill.name
    return LabeledAction(
        action=np.array([names[name], 0, 0, 0, 0], dtype=float),
        label=f"{name}({', '.join(obj.name for obj in ground_skill.objects)})",
    )


class HybridDeploymentMethod(EesMethod):
    def execute_ground_skill(self, *, ground_skill: GroundSkill, state: Any, explore: bool) -> Any:
        del state, explore
        return skill_action(ground_skill=ground_skill), None


class HybridMethod(Tossing3DPomdpMethod):
    exploration_epsilon: float = Field(default=0, ge=0, le=0)
    revise_skills: Callable[[], None] = Field(default=lambda: None, exclude=True)

    def model_post_init(self, __context: object) -> None:
        super().model_post_init(__context)
        # The binary classifier needs both classes before an informed prediction.
        # Generated code has no such gate; retain Bayesian skill beliefs only.
        self._pomdp_state = self._pomdp_state.model_copy(update={"sampler_training": {}})

    def execute_ground_skill(self, *, ground_skill: GroundSkill, state: Any, explore: bool) -> Any:
        del state, explore
        return skill_action(ground_skill=ground_skill), None

    def observe_outcome(
        self, *, ground_skill: GroundSkill, success: bool, was_random_exploration: bool = False
    ) -> None:
        super().observe_outcome(
            ground_skill=ground_skill,
            success=success,
            was_random_exploration=was_random_exploration,
        )
        if ground_skill.skill.name == TOSS_SKILL:
            self._pomdp_state = self._pomdp_model.observe_training_example(
                state=self._pomdp_state, skill_name=TOSS_SKILL, success=success
            )

    def fit_samplers(self) -> None:
        # Called after the pending outcome is flushed by the existing end_cycle.
        self.revise_skills()

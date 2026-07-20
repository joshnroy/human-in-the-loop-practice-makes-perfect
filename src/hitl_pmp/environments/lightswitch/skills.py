from typing import ClassVar

import numpy as np

from hitl_pmp.core.method.types import GroundSkill, Skill
from hitl_pmp.core.problem.environment.types import Action, State

from .environment import LightSwitchEnvironment


class LightSwitchSkills:
    """Lifted skill templates for Light Switch, ported from predicators'
    ground_truth_models/grid_row/options.py. A static-method container, never
    instantiated, same as every other business-logic class in this project."""

    MOVE_ROBOT: ClassVar[Skill] = Skill(
        name="MoveRobot",
        types=(
            LightSwitchEnvironment.robot_type,
            LightSwitchEnvironment.cell_type,
            LightSwitchEnvironment.cell_type,
        ),
        param_dim=0,
    )
    TURN_ON_LIGHT: ClassVar[Skill] = Skill(
        name="TurnOnLight",
        types=(
            LightSwitchEnvironment.robot_type,
            LightSwitchEnvironment.cell_type,
            LightSwitchEnvironment.light_type,
        ),
        param_dim=1,
    )
    TURN_OFF_LIGHT: ClassVar[Skill] = Skill(
        name="TurnOffLight",
        types=(
            LightSwitchEnvironment.robot_type,
            LightSwitchEnvironment.cell_type,
            LightSwitchEnvironment.light_type,
        ),
        param_dim=1,
    )
    # theta is ignored by compute_action below (matches predicators' own hardcoded
    # no-op) -- param_dim=1 anyway, purely to mirror the reference implementation's
    # signature; nothing here ever reads the sampled value.
    JUMP_TO_LIGHT: ClassVar[Skill] = Skill(
        name="JumpToLight",
        types=(
            LightSwitchEnvironment.robot_type,
            LightSwitchEnvironment.cell_type,
            LightSwitchEnvironment.cell_type,
            LightSwitchEnvironment.cell_type,
            LightSwitchEnvironment.light_type,
        ),
        param_dim=1,
    )

    @staticmethod
    def sample_params(*, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        """Uniform(-1, 1) per continuous dim, matching predicators' light_sampler --
        every param_dim>0 skill here shares this one sampling distribution,
        regardless of what the sampled value ends up meaning to compute_action."""
        return rng.uniform(-1.0, 1.0, size=ground_skill.skill.param_dim)

    @staticmethod
    def compute_action(*, ground_skill: GroundSkill, params: np.ndarray, state: State) -> Action:
        """The lifted "option policy" layer: reads state (and, for the toggle
        skills, the sampled param) to produce one raw [dx, dlight] action --
        mirrors predicators' _create_move_robot_policy/_toggle_light_policy/
        _jump_to_light_policy."""
        skills = LightSwitchSkills
        skill = ground_skill.skill

        if skill == skills.MOVE_ROBOT:
            robot, _current_cell, target_cell = ground_skill.objects
            dx = state.get(obj=target_cell, feature_name="x") - state.get(
                obj=robot, feature_name="x"
            )
            return np.array([dx, 0.0])

        if skill in (skills.TURN_ON_LIGHT, skills.TURN_OFF_LIGHT):
            dlight = float(params[0])
            return np.array([0.0, dlight])

        if skill == skills.JUMP_TO_LIGHT:
            # The "impossible" skill: predicators hardcodes a no-op regardless of
            # state or params, so a Method can never actually complete a task via
            # this skill's own policy.
            return np.array([0.0, 0.0])

        raise ValueError(f"Unknown skill: {skill.name}")

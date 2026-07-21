import numpy as np

from hitl_pmp.core.method.types import GroundSkill, LabeledAction
from hitl_pmp.core.problem.environment.types import State

from .environment import LightSwitchEnvironment
from .skills import LightSwitchSkills


class SkillOraclePolicy:
    """Cheats with privileged ground-truth state -- routes every action through
    skills.py's lifted -> grounded -> compute_action pipeline (unlike
    ActionOraclePolicy, which acts directly in raw action space). Light-Switch-
    only, same as ActionOraclePolicy -- the cross-environment dispatch lives one
    layer up, in methods/oracle/skill_oracle_method.py's SkillOracleMethod, since
    that's the layer allowed to reason about which environment is wired (see
    core/README.md's Problem facade section). A static-method container, never
    instantiated, same as every other business-logic class in this project."""

    @staticmethod
    def get_labeled_action(*, state: State) -> LabeledAction:
        """MoveRobot to the light's cell, then TurnOnLight by exactly the
        remaining gap -- same two-action solve as ActionOraclePolicy; the point
        of this method is to exercise Skill/GroundSkill end-to-end with a real
        caller, not to behave differently."""
        env = LightSwitchEnvironment
        robot, light = env.robot, env.light
        robot_x = state.get(obj=robot, feature_name="x")
        light_x = state.get(obj=light, feature_name="x")

        cells = env.get_cells()
        light_cell = cells[-1]  # matches build_initial_state's light placement

        if abs(robot_x - light_x) >= env.same_position_tolerance:
            # MoveRobot's compute_action only reads the target cell (and robot's own
            # position) -- current_cell is part of the skill's signature purely for
            # fidelity with predicators' MoveRobot(robot, current_cell, target_cell),
            # so any nearest-cell estimate is fine here.
            current_cell = min(
                cells, key=lambda cell: abs(state.get(obj=cell, feature_name="x") - robot_x)
            )
            ground_skill = GroundSkill(
                skill=LightSwitchSkills.MOVE_ROBOT, objects=(robot, current_cell, light_cell)
            )
            params: np.ndarray = np.zeros(0)
        else:
            level = state.get(obj=light, feature_name="level")
            target = state.get(obj=light, feature_name="target")
            ground_skill = GroundSkill(
                skill=LightSwitchSkills.TURN_ON_LIGHT, objects=(robot, light_cell, light)
            )
            params = np.array([target - level])

        action = LightSwitchSkills.compute_action(
            ground_skill=ground_skill, params=params, state=state
        )
        objects_desc = ", ".join(obj.name for obj in ground_skill.objects)
        label = f"{ground_skill.skill.name}({objects_desc})"
        if params.size > 0:
            # Show every entry in params, not just params[0] -- params has no name
            # of its own at the GroundSkill/compute_action level (it's a bare
            # ndarray, deliberately skill-agnostic), so this must stay generic over
            # param_dim rather than assuming a single scalar, or a future skill with
            # param_dim > 1 would silently lose everything past the first entry.
            rounded_params = [round(float(p), 2) for p in params]
            label += f", params={rounded_params}"
        return LabeledAction(action=action, label=label)

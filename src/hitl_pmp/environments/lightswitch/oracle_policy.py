import numpy as np

from hitl_pmp.core.method.types import Policy
from hitl_pmp.core.problem.environment.types import Action, State

from .environment import LightSwitchEnvironment


class OraclePolicy:
    """Cheats with privileged ground-truth state: move straight to the light, then
    dial it exactly to target -- always solves in two actions, since take_action only
    applies dlight based on where the robot already was at the start of that action
    (matches GridRowEnv.simulate's ordering, see environment.py). A static-method
    container, never instantiated, same as every other business-logic class in this
    project."""

    @staticmethod
    def get_action(*, state: State) -> Action:
        robot = LightSwitchEnvironment.robot
        light = LightSwitchEnvironment.light
        robot_x = state.get(obj=robot, feature_name="x")
        light_x = state.get(obj=light, feature_name="x")
        if abs(robot_x - light_x) >= LightSwitchEnvironment.same_position_tolerance:
            return np.array([light_x - robot_x, 0.0])
        level = state.get(obj=light, feature_name="level")
        target = state.get(obj=light, feature_name="target")
        return np.array([0.0, target - level])


# Policy is a positional Callable[[State], Action] per its interface contract (core/
# method/types.py) -- this lambda just adapts that into a call to OraclePolicy's
# keyword-only get_action, same pattern as predicates.py's LIGHT_ON.
ORACLE_POLICY: Policy = lambda state: OraclePolicy.get_action(state=state)  # noqa: E731

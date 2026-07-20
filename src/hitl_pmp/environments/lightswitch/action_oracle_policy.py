import numpy as np

from hitl_pmp.core.method.types import LabeledAction, Policy
from hitl_pmp.core.problem.environment.types import Action, State

from .environment import LightSwitchEnvironment


class ActionOraclePolicy:
    """Cheats with privileged ground-truth state, acting directly in raw action
    space -- no skill selection anywhere in the loop. This is a legitimate design
    point, not a gap: predicators' own abstract baseline interface is skill-
    agnostic, and it ships genuine raw-action baselines with no skill layer at all
    (random_actions_approach.py, gnn_action_policy_approach.py). Always solves in
    two actions, since take_action only applies dlight based on where the robot
    already was at the start of that action (matches GridRowEnv.simulate's
    ordering, see environment.py). A static-method container, never instantiated,
    same as every other business-logic class in this project."""

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

    @staticmethod
    def get_labeled_action(*, state: State) -> LabeledAction:
        action = ActionOraclePolicy.get_action(state=state)
        return LabeledAction(
            action=action, label=f"raw action [dx={action[0]:.2f}, dlight={action[1]:.2f}]"
        )


# Policy is a positional Callable[[State], LabeledAction] per its interface contract
# (core/method/types.py) -- this lambda just adapts that into a call to
# ActionOraclePolicy's keyword-only get_labeled_action, same pattern as
# predicates.py's LIGHT_ON.
ACTION_ORACLE_POLICY: Policy = lambda state: ActionOraclePolicy.get_labeled_action(  # noqa: E731
    state=state
)

import numpy as np

from hitl_pmp.core.method.types import Policy
from hitl_pmp.core.problem.environment.types import Action, State
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks


def _compute_oracle_action(*, state: State) -> Action:
    """Move straight to the light, then dial it exactly to target -- two actions,
    since take_action only applies dlight based on where the robot already was at
    the start of that action (matches GridRowEnv.simulate's ordering)."""
    robot = LightSwitchEnvironment.robot
    light = LightSwitchEnvironment.light
    robot_x = state.get(obj=robot, feature_name="x")
    light_x = state.get(obj=light, feature_name="x")
    if abs(robot_x - light_x) >= LightSwitchEnvironment.same_position_tolerance:
        return np.array([light_x - robot_x, 0.0])
    level = state.get(obj=light, feature_name="level")
    target = state.get(obj=light, feature_name="target")
    return np.array([0.0, target - level])


# Policy is a positional Callable[[State], Action] per its interface contract -- this
# lambda just adapts that into a call to the keyword-only helper above.
_oracle_policy: Policy = lambda state: _compute_oracle_action(state=state)  # noqa: E731


def test_oracle_policy_solves_a_sampled_train_task() -> None:
    task = LightSwitchTasks.sample_train_task()
    LightSwitchEnvironment.set_state(state=task.initial_state)

    state = LightSwitchEnvironment.get_current_state()
    assert task.goal.is_satisfied(state=state) is False

    for _ in range(2):
        state = LightSwitchEnvironment.take_action(action=_oracle_policy(state))

    assert task.goal.is_satisfied(state=state) is True


def test_oracle_policy_solves_a_sampled_test_task() -> None:
    task = LightSwitchTasks.sample_test_task()
    LightSwitchEnvironment.set_state(state=task.initial_state)

    state = LightSwitchEnvironment.get_current_state()
    for _ in range(2):
        state = LightSwitchEnvironment.take_action(action=_oracle_policy(state))

    assert task.goal.is_satisfied(state=state) is True


def test_jump_action_never_reaches_the_light_in_one_step() -> None:
    """The paper's JumpToLight skill is always impossible -- there is no single raw
    [dx, dlight] action that both moves the robot to the light's cell and dials the
    light on in one take_action call, since dlight is applied based on the
    pre-action position. This is the mechanism the paper's "impossible skill"
    exploits, at the raw environment level."""
    task = LightSwitchTasks.sample_train_task()
    LightSwitchEnvironment.set_state(state=task.initial_state)

    state = LightSwitchEnvironment.get_current_state()
    light = LightSwitchEnvironment.light
    light_x = state.get(obj=light, feature_name="x")
    robot = LightSwitchEnvironment.robot
    robot_x = state.get(obj=robot, feature_name="x")
    target = state.get(obj=light, feature_name="target")

    jump_action = np.array([light_x - robot_x, target])
    next_state = LightSwitchEnvironment.take_action(action=jump_action)

    assert task.goal.is_satisfied(state=next_state) is False

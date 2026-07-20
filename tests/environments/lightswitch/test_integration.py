import numpy as np
import pytest

from hitl_pmp.core.method.types import Policy
from hitl_pmp.environments.lightswitch.action_oracle_policy import ACTION_ORACLE_POLICY
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.skill_oracle_policy import SKILL_ORACLE_POLICY
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks

_ORACLE_POLICIES: dict[str, Policy] = {
    "action-oracle": ACTION_ORACLE_POLICY,
    "skill-oracle": SKILL_ORACLE_POLICY,
}


@pytest.mark.parametrize("policy", _ORACLE_POLICIES.values(), ids=_ORACLE_POLICIES.keys())
def test_oracle_policy_solves_a_sampled_train_task(*, policy: Policy) -> None:
    task = LightSwitchTasks.sample_train_task()
    LightSwitchEnvironment.set_state(state=task.initial_state)

    state = LightSwitchEnvironment.get_current_state()
    assert task.goal.is_satisfied(state=state) is False

    for _ in range(2):
        state = LightSwitchEnvironment.take_action(action=policy(state).action)

    assert task.goal.is_satisfied(state=state) is True


@pytest.mark.parametrize("policy", _ORACLE_POLICIES.values(), ids=_ORACLE_POLICIES.keys())
def test_oracle_policy_solves_a_sampled_test_task(*, policy: Policy) -> None:
    task = LightSwitchTasks.sample_test_task()
    LightSwitchEnvironment.set_state(state=task.initial_state)

    state = LightSwitchEnvironment.get_current_state()
    for _ in range(2):
        state = LightSwitchEnvironment.take_action(action=policy(state).action)

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

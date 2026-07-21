from collections.abc import Iterator

import pytest

from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, State
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.skill_oracle_policy import (
    SkillOracleMethod,
    SkillOraclePolicy,
)
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks


class _OtherEnv(Environment):
    """A stand-in for "some environment SkillOraclePolicy has no branch for yet" --
    only needs to exist as a distinct type, its methods are never actually called."""

    @staticmethod
    def take_action(*, action: Action) -> State:
        raise NotImplementedError

    @staticmethod
    def get_valid_actions() -> list[Action]:
        raise NotImplementedError

    @staticmethod
    def hard_reset() -> None:
        raise NotImplementedError


@pytest.fixture(autouse=True)
def _wire_problem_env() -> Iterator[None]:
    original_env = getattr(Problem, "env", None)
    Problem.env = LightSwitchEnvironment
    try:
        yield
    finally:
        if original_env is not None:
            Problem.env = original_env


def test_get_lightswitch_labeled_action_moves_straight_to_the_light_first() -> None:
    state = LightSwitchEnvironment.build_initial_state(light_level=0.0, light_target=0.7)
    labeled = SkillOraclePolicy.get_lightswitch_labeled_action(state=state)
    robot_x = state.get(obj=LightSwitchEnvironment.robot, feature_name="x")
    light_x = state.get(obj=LightSwitchEnvironment.light, feature_name="x")
    assert labeled.action.tolist() == [light_x - robot_x, 0.0]
    assert labeled.label.startswith("MoveRobot(")
    # MoveRobot has param_dim=0 -- no continuous parameters to show.
    assert "params=" not in labeled.label


def test_get_lightswitch_labeled_action_dials_exactly_to_target_once_at_the_light() -> None:
    light_x = float(LightSwitchEnvironment.grid_size - 0.5)
    state = LightSwitchEnvironment.build_initial_state(light_level=0.2, light_target=0.9)
    state.set(obj=LightSwitchEnvironment.robot, feature_name="x", feature_val=light_x)

    labeled = SkillOraclePolicy.get_lightswitch_labeled_action(state=state)
    assert labeled.action[0] == 0.0
    assert labeled.action[1] == 0.9 - 0.2
    assert labeled.label.startswith("TurnOnLight(")
    # TurnOnLight has param_dim=1 -- the actual dlight value sent to compute_action
    # should be visible in the label, not just the skill name and objects.
    assert "params=[0.7]" in labeled.label


def test_get_labeled_action_dispatches_to_lightswitch_when_that_env_is_wired() -> None:
    state = LightSwitchEnvironment.build_initial_state(light_level=0.0, light_target=0.7)
    dispatched = SkillOraclePolicy.get_labeled_action(state=state)
    direct = SkillOraclePolicy.get_lightswitch_labeled_action(state=state)
    assert dispatched.action.tolist() == direct.action.tolist()
    assert dispatched.label == direct.label


def test_get_labeled_action_raises_for_an_unrecognized_env() -> None:
    Problem.env = _OtherEnv
    state = LightSwitchEnvironment.build_initial_state(light_level=0.0, light_target=0.7)
    with pytest.raises(NotImplementedError):
        SkillOraclePolicy.get_labeled_action(state=state)


def test_solves_a_sampled_task_in_exactly_two_actions() -> None:
    task = LightSwitchTasks.sample_train_task()
    LightSwitchEnvironment.set_state(state=task.initial_state)
    policy = SkillOracleMethod.get_task_policy(task=task)

    state = LightSwitchEnvironment.get_current_state()
    assert task.goal.is_satisfied(state=state) is False

    state = LightSwitchEnvironment.take_action(action=policy(state).action)
    assert task.goal.is_satisfied(state=state) is False

    state = LightSwitchEnvironment.take_action(action=policy(state).action)
    assert task.goal.is_satisfied(state=state) is True


def test_reset_environment_directly_sets_state_and_returns_true() -> None:
    start_state = LightSwitchEnvironment.build_initial_state(light_level=0.3, light_target=0.8)
    assert SkillOracleMethod.reset_environment(start_state=start_state) is True
    assert LightSwitchEnvironment.get_current_state() is start_state


def test_generate_train_task_is_unreachable() -> None:
    with pytest.raises(NotImplementedError):
        SkillOracleMethod.generate_train_task(tbd_inputs=None)


def test_execute_setup_command_is_unreachable() -> None:
    with pytest.raises(NotImplementedError):
        SkillOracleMethod.execute_setup_command(setup_command=None)  # type: ignore[arg-type]


def test_execute_skill_is_unreachable() -> None:
    with pytest.raises(NotImplementedError):
        SkillOracleMethod.execute_skill(skill=None)  # type: ignore[arg-type]


def test_improve_skill_parameters_is_unreachable() -> None:
    with pytest.raises(NotImplementedError):
        SkillOracleMethod.improve_skill_parameters(skill=None, rollout=None)  # type: ignore[arg-type]

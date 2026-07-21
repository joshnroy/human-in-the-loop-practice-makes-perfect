import pytest

from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, State
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.skill_oracle_policy import SkillOraclePolicy
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks
from hitl_pmp.methods.oracle.skill_oracle_method import SkillOracleMethod


class _OtherEnv(Environment):
    """A stand-in for "some environment SkillOracleMethod has no branch for yet" --
    only needs to exist as a distinct type, its methods are never actually called."""

    def take_action(self, *, action: Action) -> State:
        raise NotImplementedError

    def get_valid_actions(self) -> list[Action]:
        raise NotImplementedError

    def hard_reset(self) -> None:
        raise NotImplementedError


def test_get_labeled_action_dispatches_to_lightswitch_when_that_env_is_wired() -> None:
    env = LightSwitchEnvironment()
    method = SkillOracleMethod(env=env)
    state = env.build_initial_state(light_level=0.0, light_target=0.7)
    dispatched = method.get_labeled_action(state=state)
    direct = SkillOraclePolicy.get_labeled_action(state=state, env=env)
    assert dispatched.action.tolist() == direct.action.tolist()
    assert dispatched.label == direct.label


def test_get_labeled_action_raises_for_an_unrecognized_env() -> None:
    method = SkillOracleMethod(env=_OtherEnv())
    state = LightSwitchEnvironment().build_initial_state(light_level=0.0, light_target=0.7)
    with pytest.raises(NotImplementedError):
        method.get_labeled_action(state=state)


def test_solves_a_sampled_task_in_exactly_two_actions() -> None:
    env = LightSwitchEnvironment()
    tasks = LightSwitchTasks(env=env)
    task = tasks.sample_train_task()
    env.set_state(state=task.initial_state)
    method = SkillOracleMethod(env=env)
    policy = method.get_task_policy(task=task)

    state = env.get_current_state()
    assert task.goal.is_satisfied(state=state) is False

    state = env.take_action(action=policy(state).action)
    assert task.goal.is_satisfied(state=state) is False

    state = env.take_action(action=policy(state).action)
    assert task.goal.is_satisfied(state=state) is True


def test_reset_environment_directly_sets_state_and_returns_true() -> None:
    env = LightSwitchEnvironment()
    method = SkillOracleMethod(env=env)
    start_state = env.build_initial_state(light_level=0.3, light_target=0.8)
    assert method.reset_environment(start_state=start_state) is True
    assert env.get_current_state() is start_state


def test_generate_train_task_is_unreachable() -> None:
    method = SkillOracleMethod(env=LightSwitchEnvironment())
    with pytest.raises(NotImplementedError):
        method.generate_train_task(tbd_inputs=None)


def test_execute_setup_command_is_unreachable() -> None:
    method = SkillOracleMethod(env=LightSwitchEnvironment())
    with pytest.raises(NotImplementedError):
        method.execute_setup_command(setup_command=None)  # type: ignore[arg-type]


def test_execute_skill_is_unreachable() -> None:
    method = SkillOracleMethod(env=LightSwitchEnvironment())
    with pytest.raises(NotImplementedError):
        method.execute_skill(skill=None)  # type: ignore[arg-type]


def test_improve_skill_parameters_is_unreachable() -> None:
    method = SkillOracleMethod(env=LightSwitchEnvironment())
    with pytest.raises(NotImplementedError):
        method.improve_skill_parameters(skill=None, rollout=None)  # type: ignore[arg-type]

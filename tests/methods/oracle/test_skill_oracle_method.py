import pytest

from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.skill_oracle_policy import SkillOraclePolicy
from hitl_pmp.environments.lightswitch.skill_provider import LightSwitchOracle
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks
from hitl_pmp.methods.oracle.skill_oracle_method import SkillOracleMethod


def _method(*, env: LightSwitchEnvironment) -> SkillOracleMethod:
    return SkillOracleMethod(env=env, oracle=LightSwitchOracle(env=env))


def test_task_policy_delegates_to_the_injected_oracle() -> None:
    """The domain-agnostic SkillOracleMethod just drives whatever OraclePolicyProvider
    it is handed -- here Light Switch's, which wraps SkillOraclePolicy."""
    env = LightSwitchEnvironment()
    method = _method(env=env)
    state = env.build_initial_state(light_level=0.0, light_target=0.7)
    dispatched = method.get_task_policy(task=LightSwitchTasks(env=env).sample_train_task())(state)
    direct = SkillOraclePolicy.get_labeled_action(state=state, env=env)
    assert dispatched.action.tolist() == direct.action.tolist()
    assert dispatched.label == direct.label


def test_solves_a_sampled_task_in_exactly_two_actions() -> None:
    env = LightSwitchEnvironment()
    tasks = LightSwitchTasks(env=env)
    task = tasks.sample_train_task()
    env.set_state(state=task.initial_state)
    method = _method(env=env)
    policy = method.get_task_policy(task=task)

    state = env.get_current_state()
    assert task.goal.is_satisfied(state=state) is False
    state = env.take_action(action=policy(state).action)
    assert task.goal.is_satisfied(state=state) is False
    state = env.take_action(action=policy(state).action)
    assert task.goal.is_satisfied(state=state) is True


def test_reset_environment_reports_failure_and_leaves_the_environment_alone() -> None:
    """This method has no way to self-navigate, so it reports failure -- and does not
    reach for the privileged `set_state` that would fake one."""
    env = LightSwitchEnvironment()
    method = _method(env=env)
    stranded = env.build_initial_state(light_level=0.1, light_target=0.9)
    env.set_state(state=stranded)
    start_state = env.build_initial_state(light_level=0.3, light_target=0.8)
    assert method.reset_environment(start_state=start_state) is False
    assert env.get_current_state() is stranded


def test_generate_train_task_is_unreachable() -> None:
    with pytest.raises(NotImplementedError):
        _method(env=LightSwitchEnvironment()).generate_train_task(tbd_inputs=None)


def test_execute_setup_command_is_unreachable() -> None:
    with pytest.raises(NotImplementedError):
        _method(env=LightSwitchEnvironment()).execute_setup_command(setup_command=None)  # type: ignore[arg-type]


def test_execute_skill_is_unreachable() -> None:
    with pytest.raises(NotImplementedError):
        _method(env=LightSwitchEnvironment()).execute_skill(skill=None)  # type: ignore[arg-type]


def test_improve_skill_parameters_is_unreachable() -> None:
    with pytest.raises(NotImplementedError):
        _method(env=LightSwitchEnvironment()).improve_skill_parameters(skill=None, rollout=None)  # type: ignore[arg-type]

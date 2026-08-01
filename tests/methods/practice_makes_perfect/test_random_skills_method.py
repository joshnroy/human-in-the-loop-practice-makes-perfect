import pytest

from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.skill_provider import LightSwitchSkillProvider
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks
from hitl_pmp.methods.practice_makes_perfect.random_skills_method import RandomSkillsMethod


def _method(*, env: LightSwitchEnvironment, seed: int = 0) -> RandomSkillsMethod:
    return RandomSkillsMethod(env=env, skill_provider=LightSwitchSkillProvider(env=env), seed=seed)


def test_get_labeled_action_returns_an_applicable_skill_from_the_provider() -> None:
    """Fully domain-agnostic now: the method grounds/executes via the injected
    SkillProvider, so the emitted label names one of Light Switch's own skills."""
    env = LightSwitchEnvironment()
    method = _method(env=env)
    state = env.build_initial_state(light_level=0.0, light_target=0.7)
    label = method.get_labeled_action(state=state).label
    assert any(
        label.startswith(name)
        for name in ("MoveRobot", "TurnOnLight", "TurnOffLight", "JumpToLight")
    )


def test_solves_a_sampled_task_eventually_given_enough_steps() -> None:
    """Not a guaranteed-two-action solve like the oracle -- a genuine uniform random
    walk over applicable ground skills takes many more steps than the paper's own
    per-episode horizon to reliably reach the goal. Uses a small grid and a large
    step budget purely to confirm the wired-up policy is genuinely making progress."""
    env = LightSwitchEnvironment(grid_size=5)
    tasks = LightSwitchTasks(env=env, seed=0)
    task = tasks.sample_train_task()
    env.set_state(state=task.initial_state)
    method = _method(env=env, seed=0)
    policy = method.get_task_policy(task=task)

    state = env.get_current_state()
    for _ in range(2000):
        if task.goal.is_satisfied(state=state):
            break
        state = env.take_action(action=policy(state).action)
    assert task.goal.is_satisfied(state=state) is True


def test_reset_environment_directly_sets_state_and_returns_true() -> None:
    env = LightSwitchEnvironment()
    method = _method(env=env)
    start_state = env.build_initial_state(light_level=0.3, light_target=0.8)
    assert method.reset_environment(start_state=start_state) is True
    assert env.get_current_state() is start_state


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


def test_same_seed_produces_identical_action_sequences() -> None:
    env_a = LightSwitchEnvironment()
    env_b = LightSwitchEnvironment()
    state = env_a.build_initial_state(light_level=0.0, light_target=0.7)
    method_a = _method(env=env_a, seed=42)
    method_b = _method(env=env_b, seed=42)

    for _ in range(10):
        labeled_a = method_a.get_labeled_action(state=state)
        labeled_b = method_b.get_labeled_action(state=state)
        assert labeled_a.label == labeled_b.label
        assert labeled_a.action.tolist() == labeled_b.action.tolist()


def test_different_seeds_can_produce_different_action_sequences() -> None:
    env_a = LightSwitchEnvironment()
    env_b = LightSwitchEnvironment()
    light_x = float(env_a.grid_size - 0.5)
    state = env_a.build_initial_state(light_level=0.0, light_target=0.7)
    state.set(obj=LightSwitchEnvironment.robot, feature_name="x", feature_val=light_x)
    method_a = _method(env=env_a, seed=1)
    method_b = _method(env=env_b, seed=2)

    labels_a = [method_a.get_labeled_action(state=state).label for _ in range(20)]
    labels_b = [method_b.get_labeled_action(state=state).label for _ in range(20)]
    assert labels_a != labels_b

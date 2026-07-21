import numpy as np
import pytest

from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, State
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.random_skills_policy import RandomSkillsPolicy
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks
from hitl_pmp.methods.practice_makes_perfect.random_skills_method import RandomSkillsMethod


class _OtherEnv(Environment):
    """A stand-in for "some environment RandomSkillsMethod has no branch for yet" --
    only needs to exist as a distinct type, its methods are never actually called."""

    def take_action(self, *, action: Action) -> State:
        raise NotImplementedError

    def get_valid_actions(self) -> list[Action]:
        raise NotImplementedError

    def hard_reset(self) -> None:
        raise NotImplementedError


def test_get_labeled_action_dispatches_to_lightswitch_when_that_env_is_wired() -> None:
    env = LightSwitchEnvironment()
    method = RandomSkillsMethod(env=env, seed=0)
    state = env.build_initial_state(light_level=0.0, light_target=0.7)
    dispatched = method.get_labeled_action(state=state)
    direct = RandomSkillsPolicy.get_labeled_action(
        state=state, env=env, rng=np.random.default_rng(0)
    )
    assert dispatched.action.tolist() == direct.action.tolist()
    assert dispatched.label == direct.label


def test_get_labeled_action_raises_for_an_unrecognized_env() -> None:
    method = RandomSkillsMethod(env=_OtherEnv())
    state = LightSwitchEnvironment().build_initial_state(light_level=0.0, light_target=0.7)
    with pytest.raises(NotImplementedError):
        method.get_labeled_action(state=state)


def test_solves_a_sampled_task_eventually_given_enough_steps() -> None:
    """Not a guaranteed-two-action solve like the oracle (this baseline doesn't
    cheat) -- a genuine uniform random walk over applicable ground skills takes
    many more steps than the paper's own per-episode horizon (grid_size + 2) to
    reliably reach the goal, since MoveRobot dominates the applicable set almost
    everywhere and only occasionally lands on a turn skill once colocated with
    the light (see test_random_skills_policy.py's identical reasoning). Uses a
    small grid_size and a much larger step budget than any real evaluation
    episode would, purely to confirm the wired-up policy is genuinely making
    progress -- not stuck looping on an always-inapplicable or always-no-op
    skill -- not to claim this baseline solves within a normal episode's
    horizon (its whole point, empirically, is that it usually doesn't)."""
    env = LightSwitchEnvironment(grid_size=5)
    tasks = LightSwitchTasks(env=env, seed=0)
    task = tasks.sample_train_task()
    env.set_state(state=task.initial_state)
    method = RandomSkillsMethod(env=env, seed=0)
    policy = method.get_task_policy(task=task)

    state = env.get_current_state()
    for _ in range(2000):
        if task.goal.is_satisfied(state=state):
            break
        state = env.take_action(action=policy(state).action)
    assert task.goal.is_satisfied(state=state) is True


def test_reset_environment_directly_sets_state_and_returns_true() -> None:
    env = LightSwitchEnvironment()
    method = RandomSkillsMethod(env=env)
    start_state = env.build_initial_state(light_level=0.3, light_target=0.8)
    assert method.reset_environment(start_state=start_state) is True
    assert env.get_current_state() is start_state


def test_generate_train_task_is_unreachable() -> None:
    method = RandomSkillsMethod(env=LightSwitchEnvironment())
    with pytest.raises(NotImplementedError):
        method.generate_train_task(tbd_inputs=None)


def test_execute_setup_command_is_unreachable() -> None:
    method = RandomSkillsMethod(env=LightSwitchEnvironment())
    with pytest.raises(NotImplementedError):
        method.execute_setup_command(setup_command=None)  # type: ignore[arg-type]


def test_execute_skill_is_unreachable() -> None:
    method = RandomSkillsMethod(env=LightSwitchEnvironment())
    with pytest.raises(NotImplementedError):
        method.execute_skill(skill=None)  # type: ignore[arg-type]


def test_improve_skill_parameters_is_unreachable() -> None:
    method = RandomSkillsMethod(env=LightSwitchEnvironment())
    with pytest.raises(NotImplementedError):
        method.improve_skill_parameters(skill=None, rollout=None)  # type: ignore[arg-type]


def test_same_seed_produces_identical_action_sequences() -> None:
    env_a = LightSwitchEnvironment()
    env_b = LightSwitchEnvironment()
    state = env_a.build_initial_state(light_level=0.0, light_target=0.7)
    method_a = RandomSkillsMethod(env=env_a, seed=42)
    method_b = RandomSkillsMethod(env=env_b, seed=42)

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
    method_a = RandomSkillsMethod(env=env_a, seed=1)
    method_b = RandomSkillsMethod(env=env_b, seed=2)

    labels_a = [method_a.get_labeled_action(state=state).label for _ in range(20)]
    labels_b = [method_b.get_labeled_action(state=state).label for _ in range(20)]
    assert labels_a != labels_b

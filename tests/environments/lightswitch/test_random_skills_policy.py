import numpy as np
import pytest

from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.random_skills_policy import RandomSkillsPolicy
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks


def test_get_labeled_action_returns_a_move_or_turn_on_skill_from_the_start_state() -> None:
    """From a fresh initial state (robot at cell0, light at the last cell, off),
    only MoveRobot toward cell1 (or, at grid_size=1, TurnOnLight) is applicable --
    JumpToLight needs the robot 2 cells from the light, TurnOnLight/TurnOffLight
    need the robot colocated with the light."""
    env = LightSwitchEnvironment(grid_size=5)
    state = env.build_initial_state(light_level=0.0, light_target=0.7)
    labeled = RandomSkillsPolicy.get_labeled_action(
        state=state, env=env, rng=np.random.default_rng(0)
    )
    assert labeled.label.startswith("MoveRobot(")


def test_get_labeled_action_never_picks_a_turn_skill_when_not_colocated_with_the_light() -> None:
    """TurnOnLight/TurnOffLight's precondition genuinely requires
    RobotInCell(robot, light's cell) -- unlike MoveRobot (whose precondition
    holds for any current position) or JumpToLight (whose start/landing cells
    aren't forced distinct, so it's applicable more often than its name
    suggests -- see SkillGrounder's own docstring on not enforcing parameter
    distinctness), a turn skill should never be groundable while the robot
    isn't at the light."""
    env = LightSwitchEnvironment(grid_size=5)
    state = env.build_initial_state(light_level=0.0, light_target=0.7)  # robot starts at cell0

    for seed in range(20):
        labeled = RandomSkillsPolicy.get_labeled_action(
            state=state, env=env, rng=np.random.default_rng(seed)
        )
        assert not labeled.label.startswith("TurnOnLight(")
        assert not labeled.label.startswith("TurnOffLight(")


def test_get_labeled_action_is_deterministic_given_the_same_rng_state() -> None:
    env = LightSwitchEnvironment(grid_size=5)
    state = env.build_initial_state(light_level=0.0, light_target=0.7)
    first = RandomSkillsPolicy.get_labeled_action(
        state=state, env=env, rng=np.random.default_rng(7)
    )
    second = RandomSkillsPolicy.get_labeled_action(
        state=state, env=env, rng=np.random.default_rng(7)
    )
    assert first.label == second.label
    assert first.action.tolist() == second.action.tolist()


def test_get_labeled_action_varies_the_chosen_skill_across_many_seeds() -> None:
    """With the robot colocated with the light (light off, so only TurnOnLight
    of the two toggle skills is actually applicable -- TurnOffLight's own
    precondition needs LightOn), MoveRobot/TurnOnLight/JumpToLight are all
    simultaneously applicable (SkillGrounder doesn't force a skill's own
    parameters to bind to distinct objects, so e.g. JumpToLight's start/landing
    cells can coincide) -- over enough seeds, uniform sampling should pick more
    than just one of them (this would fail if get_labeled_action always picked
    the first applicable ground skill instead of actually sampling)."""
    env = LightSwitchEnvironment(grid_size=5)
    light_x = float(env.grid_size - 0.5)
    state = env.build_initial_state(light_level=0.0, light_target=0.7)
    state.set(obj=LightSwitchEnvironment.robot, feature_name="x", feature_val=light_x)

    labels = {
        RandomSkillsPolicy.get_labeled_action(
            state=state, env=env, rng=np.random.default_rng(seed)
        ).label.split("(")[0]
        for seed in range(30)
    }
    assert len(labels) > 1
    assert "TurnOnLight" in labels


def test_get_labeled_action_params_are_within_the_skills_declared_range() -> None:
    env = LightSwitchEnvironment(grid_size=5)
    light_x = float(env.grid_size - 0.5)
    state = env.build_initial_state(light_level=0.5, light_target=0.7)
    state.set(obj=LightSwitchEnvironment.robot, feature_name="x", feature_val=light_x)

    for seed in range(10):
        labeled = RandomSkillsPolicy.get_labeled_action(
            state=state, env=env, rng=np.random.default_rng(seed)
        )
        # TurnOnLight/TurnOffLight's raw action is [0.0, dlight] with
        # dlight = sampled param in [-1, 1] (LightSwitchSkills.sample_params).
        assert -1.0 <= labeled.action[1] <= 1.0


def test_solves_a_sampled_task_eventually_given_enough_steps() -> None:
    """Not a guaranteed-two-action solve like the oracle (this baseline doesn't
    cheat) -- a genuine uniform random walk over applicable ground skills takes
    many more steps than the paper's own per-episode horizon (grid_size + 2) to
    reliably reach the goal (empirically ~60-500 steps at grid_size=5, seed
    dependent), since MoveRobot dominates the applicable set almost everywhere
    and only occasionally lands on a turn skill once colocated with the light.
    This test uses a much larger step budget than any real evaluation episode
    would, purely to confirm the policy is genuinely making progress -- not
    stuck looping on an always-inapplicable or always-no-op skill -- not to
    claim this baseline solves within a normal episode's horizon (its whole
    point, empirically, is that it usually doesn't)."""
    env = LightSwitchEnvironment(grid_size=5)
    tasks = LightSwitchTasks(env=env, seed=0)
    task = tasks.sample_train_task()
    env.set_state(state=task.initial_state)
    rng = np.random.default_rng(0)

    state = env.get_current_state()
    for _ in range(2000):
        if task.goal.is_satisfied(state=state):
            break
        labeled = RandomSkillsPolicy.get_labeled_action(state=state, env=env, rng=rng)
        state = env.take_action(action=labeled.action)
    assert task.goal.is_satisfied(state=state) is True


def test_get_labeled_action_raises_when_no_ground_skill_is_applicable() -> None:
    """The robot sitting off-grid at a non-cell-aligned position (impossible via
    normal skill-driven dynamics, but a direct test of the failure path) has no
    ground skill whose preconditions hold anywhere -- confirms this fails loudly
    with a clear message rather than a cryptic numpy IndexError."""
    env = LightSwitchEnvironment(grid_size=5)
    state = env.build_initial_state(light_level=0.0, light_target=0.7)
    state.set(obj=LightSwitchEnvironment.robot, feature_name="x", feature_val=1.23)
    with pytest.raises(AssertionError, match="No applicable ground skills"):
        RandomSkillsPolicy.get_labeled_action(state=state, env=env, rng=np.random.default_rng(0))

from collections.abc import Iterator

import pytest

from hitl_pmp.core.method.types import GroundSkill, Rollout
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.predicates import (
    ADJACENT,
    LIGHT_IN_CELL,
    LIGHT_OFF,
    LIGHT_ON,
    ROBOT_IN_CELL,
)
from hitl_pmp.environments.lightswitch.skills import LightSwitchSkills
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks
from hitl_pmp.methods.practice_makes_perfect.random_skills_method import RandomSkillsMethod


@pytest.fixture(autouse=True)
def _wire_random_skills_method() -> Iterator[None]:
    env = LightSwitchEnvironment
    original_grid_size = env.grid_size
    try:
        env.grid_size = 5
        RandomSkillsMethod.env = env
        RandomSkillsMethod.tasks = LightSwitchTasks
        RandomSkillsMethod.predicates = (
            ADJACENT,
            LIGHT_IN_CELL,
            LIGHT_OFF,
            LIGHT_ON,
            ROBOT_IN_CELL,
        )
        RandomSkillsMethod.skills = (
            LightSwitchSkills.MOVE_ROBOT,
            LightSwitchSkills.TURN_ON_LIGHT,
            LightSwitchSkills.TURN_OFF_LIGHT,
            LightSwitchSkills.JUMP_TO_LIGHT,
        )
        RandomSkillsMethod.objects = (env.robot, env.light, *env.get_cells())
        RandomSkillsMethod.compute_action = LightSwitchSkills.compute_action
        RandomSkillsMethod.sample_params = LightSwitchSkills.sample_params
        RandomSkillsMethod.reset_state(seed=0)
        yield
    finally:
        env.grid_size = original_grid_size


def test_reset_environment_sets_the_given_state_and_returns_true() -> None:
    state = LightSwitchEnvironment.build_initial_state(light_level=0.0, light_target=0.5)
    assert RandomSkillsMethod.reset_environment(start_state=state) is True
    assert LightSwitchEnvironment.get_current_state() is state


def test_generate_train_task_delegates_to_the_domains_tasks() -> None:
    task = RandomSkillsMethod.generate_train_task(tbd_inputs=None)
    assert task.goal.atoms  # a real LightOn goal, not an empty placeholder


def test_get_task_policy_ignores_the_task_and_always_acts() -> None:
    state = LightSwitchEnvironment.build_initial_state(light_level=0.0, light_target=0.5)
    task = RandomSkillsMethod.generate_train_task(tbd_inputs=None)
    policy = RandomSkillsMethod.get_task_policy(task=task)
    labeled = policy(state)
    assert labeled.action.shape == (2,)
    assert labeled.label  # some skill got selected and described


def test_get_task_policy_selects_only_currently_applicable_skills() -> None:
    """From the initial state (robot at cell0, far from the light), only
    MoveRobot is ever applicable -- TurnOnLight/TurnOffLight/JumpToLight all
    require being at or near the light. Confirmed across many draws so this
    isn't a single lucky sample."""
    state = LightSwitchEnvironment.build_initial_state(light_level=0.0, light_target=0.5)
    task = RandomSkillsMethod.generate_train_task(tbd_inputs=None)
    policy = RandomSkillsMethod.get_task_policy(task=task)
    for _ in range(20):
        labeled = policy(state)
        assert labeled.label.startswith("MoveRobot(")


def test_execute_skill_applies_the_action_and_returns_a_two_state_rollout() -> None:
    LightSwitchEnvironment.set_state(
        state=LightSwitchEnvironment.build_initial_state(light_level=0.0, light_target=0.5)
    )
    cells = LightSwitchEnvironment.get_cells()
    ground_skill = GroundSkill(
        skill=LightSwitchSkills.MOVE_ROBOT,
        objects=(LightSwitchEnvironment.robot, cells[0], cells[1]),
    )
    rollout = RandomSkillsMethod.execute_skill(skill=ground_skill)
    assert isinstance(rollout, Rollout)
    assert len(rollout.states) == 2
    assert len(rollout.actions) == 1
    expected_x = rollout.states[0].get(obj=cells[1], feature_name="x")
    assert rollout.states[1].get(obj=LightSwitchEnvironment.robot, feature_name="x") == expected_x
    # Also confirms execute_skill actually advanced the real environment, not
    # just a local copy.
    assert (
        LightSwitchEnvironment.get_current_state().get(
            obj=LightSwitchEnvironment.robot, feature_name="x"
        )
        == expected_x
    )


def test_improve_skill_parameters_is_a_genuine_noop() -> None:
    cells = LightSwitchEnvironment.get_cells()
    ground_skill = GroundSkill(
        skill=LightSwitchSkills.MOVE_ROBOT,
        objects=(LightSwitchEnvironment.robot, cells[0], cells[1]),
    )
    state = LightSwitchEnvironment.build_initial_state(light_level=0.0, light_target=0.5)
    rollout = Rollout(states=[state], actions=[])
    # Should not raise and should not touch any shared state.
    assert RandomSkillsMethod.improve_skill_parameters(skill=ground_skill, rollout=rollout) is None


def test_execute_setup_command_is_unreachable() -> None:
    with pytest.raises(NotImplementedError):
        RandomSkillsMethod.execute_setup_command(
            setup_command=None  # type: ignore[arg-type]
        )


def test_reset_state_reseeds_the_rng_deterministically() -> None:
    RandomSkillsMethod.reset_state(seed=42)
    first_draw = RandomSkillsMethod.rng.integers(1000)
    RandomSkillsMethod.reset_state(seed=42)
    second_draw = RandomSkillsMethod.rng.integers(1000)
    assert first_draw == second_draw

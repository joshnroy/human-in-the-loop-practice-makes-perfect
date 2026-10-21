"""End-to-end reset semantics, cost accounting, and competence isolation."""

import itertools

import numpy as np
import pytest

pytest.importorskip("kinder")

from kinder.envs.dynamic3d.tablesetting3d import TableSetting3DEnv

from hitl_pmp.environments.tablesetting3d.environment import TableSetting3DEnvironment
from hitl_pmp.methods.table_setting_reset import TableSettingResetSelector


@pytest.fixture
def env():
    instance = TableSetting3DEnv(human_competence=0)
    instance.reset(seed=0)
    yield instance
    instance.close()


def test_robot_can_set_and_reset_every_plate_without_human(*, env):
    assert not env.goal_reached()
    assert env.plates_at_home().all()
    assert env.plate_mass <= env.robot_payload < 2 * env.plate_mass
    assert 4 * env.plate_mass <= env.human_payload
    for i in range(4):
        before = env.plates.copy()
        obs, reward, _, truncated, info = env.step(i)
        assert env.observation_space.contains(obs)
        assert np.all(before[np.arange(4) != i] == env.plates[np.arange(4) != i])
        assert reward == -info["cost"] < 0
        assert not truncated
        assert sum(k["duration"] for k in env.last_trajectory) == pytest.approx(info["cost"])
    assert env.goal_reached()
    predicted = env.robot_reset_cost()
    elapsed = env.elapsed_seconds
    for i in range(4):
        env.step(4 + i)
    assert env.elapsed_seconds - elapsed == pytest.approx(predicted)
    assert env.plates_at_home().all()
    assert env.human_interventions == 0


@pytest.mark.parametrize("competence", [0.0, 1.0])
def test_human_success_and_spill_preserve_robot_and_are_recoverable(*, env, competence):
    env.human_competence = competence
    for i in range(4):
        env.step(i)
    robot = env.robot_xy.copy()
    recovery_cost = env.robot_reset_cost(spilled=True)
    _, _, _, _, info = env.step(8)
    assert info["human_success"] is bool(competence)
    assert info["reset_complete"] is bool(competence)
    np.testing.assert_array_equal(robot, env.robot_xy)
    assert env.human_interventions == 1
    if not competence:
        np.testing.assert_array_equal(env.plates, env.spill_positions)
        start = env.elapsed_seconds
        for i in range(4):
            env.step(4 + i)
        assert env.elapsed_seconds - start == pytest.approx(recovery_cost)
    assert env.plates_at_home().all()
    for i in range(4):
        env.step(i)
    assert env.goal_reached()


def test_competence_alone_changes_reset_choice(*, env):
    for i in range(4):
        env.step(i)
    high = env.reset_costs(competence_estimate=0.95)
    low = env.reset_costs(competence_estimate=0.05)
    assert high["human"] < high["robot"]
    assert low["human"] > low["robot"]
    assert high["robot"] == low["robot"]
    human = env.human_reset_cost()
    env.human_competence = 1.0
    assert env.human_reset_cost() == human
    elapsed = env.elapsed_seconds
    np.testing.assert_array_equal(env.reset_costs(competence_estimate=0.5)["robot"], high["robot"])
    assert env.elapsed_seconds == elapsed


def test_partial_reset_does_not_move_already_home_plates(*, env):
    env.step(0)
    home = env.plates[1:].copy()
    env.step(8)
    np.testing.assert_array_equal(env.plates[1:], home)
    env.step(4)
    assert env.plates_at_home().all()


def test_assigned_settings_and_height_matter(*, env):
    env.plates = env.settings[::-1].copy()
    assert not env.goal_reached()
    env.plates = env.settings.copy()
    env.plates[:, 2] = 0.015
    assert not env.goal_reached()


def test_no_free_repeat_intervention(*, env):
    _, reward, _, _, info = env.step(8)
    assert reward < 0
    assert info["event"] == "unavailable"
    assert "human_success" not in info
    assert env.human_interventions == 0


def test_seed_replays_human_outcomes_without_forecast_consuming_rng():
    instances = [TableSetting3DEnv(human_competence=0.5) for _ in range(2)]
    try:
        for instance in instances:
            instance.reset(seed=18)
        for _ in range(10):
            results = []
            for index, instance in enumerate(instances):
                for i in range(4):
                    instance.step(i)
                if index == 0:
                    instance.reset_costs(competence_estimate=0.4)
                    instance.robot_reset_cost(spilled=True)
                results.append(instance.step(8)[4]["human_success"])
                for i in np.flatnonzero(~instance.plates_at_home()):
                    instance.step(int(i) + 4)
            assert results[0] == results[1]
    finally:
        for instance in instances:
            instance.close()


@pytest.mark.parametrize("value", [-1, 1.1, np.nan, np.inf])
def test_invalid_competence(value):  # noqa: PLR0917
    with pytest.raises(ValueError):
        TableSetting3DEnv(human_competence=value)


def test_adapter_roundtrip_and_invalid_actions():
    env = TableSetting3DEnvironment(seed=0)
    try:
        env.hard_reset()
        initial = env.get_current_state().model_copy(deep=True)
        env.take_action(action=np.array([0]))
        env.set_state(state=initial)
        assert env.backend.plates_at_home().all()
        for value in (0.5, -1, 10, np.nan):
            with pytest.raises(ValueError):
                env.take_action(action=np.array([value]))
    finally:
        env.close()


def test_learned_selector_uses_observed_outcomes(*, env):
    for i in range(4):
        env.step(i)
    kwargs = dict(
        robot_cost=env.robot_reset_cost(),
        human_attempt_cost=env.human_reset_cost(),
        spill_recovery_cost=env.robot_reset_cost(spilled=True),
    )
    reliable = TableSettingResetSelector()
    unreliable = TableSettingResetSelector()
    for _ in range(3):
        reliable.observe(success=True)
        unreliable.observe(success=False)
    assert reliable.choose(**kwargs) == "human"
    assert unreliable.choose(**kwargs) == "robot"


def test_every_mixed_plate_state_has_correct_autonomous_recovery_cost(*, env):
    """All 81 combinations of home, table, and spill positions remain recoverable."""
    positions = (env.homes.copy(), env.settings.copy(), env.spill_positions.copy())
    for locations in itertools.product(range(3), repeat=4):
        env.reset(seed=0)
        env.robot_xy[:] = [2.5, -0.4]
        env.plates = np.array([positions[location][i] for i, location in enumerate(locations)])
        predicted = env.robot_reset_cost()
        before = env.elapsed_seconds
        for i in np.flatnonzero(~env.plates_at_home()):
            _, _, _, _, info = env.step(int(i) + 4)
            assert sum(frame["duration"] for frame in env.last_trajectory) == pytest.approx(
                info["cost"]
            ), locations
        assert env.plates_at_home().all(), locations
        assert env.elapsed_seconds - before == pytest.approx(predicted), locations
        assert env.human_interventions == 0


def test_restore_before_hard_reset_is_seeded_and_clears_stale_outcomes():
    donor = TableSetting3DEnvironment(seed=19)
    restored = TableSetting3DEnvironment(seed=19)
    try:
        donor.hard_reset()
        for i in range(4):
            donor.take_action(action=np.array([i]))
        table_set = donor.get_current_state().model_copy(deep=True)
        restored.set_state(state=table_set)
        assert (
            restored.backend.np_random.bit_generator.state
            == donor.backend.np_random.bit_generator.state
        )
        for env in (donor, restored):
            env.take_action(action=np.array([8]))
        assert restored.last_info["human_success"] == donor.last_info["human_success"]
        restored.set_state(state=table_set)
        assert "human_success" not in restored.last_info
        assert restored.backend.human_interventions == 1
    finally:
        donor.close()
        restored.close()


@pytest.mark.parametrize("bad_cost", [-1, np.nan, np.inf])
def test_selector_rejects_invalid_costs(*, bad_cost):
    with pytest.raises(ValueError):
        TableSettingResetSelector().choose(
            robot_cost=bad_cost, human_attempt_cost=1, spill_recovery_cost=1
        )

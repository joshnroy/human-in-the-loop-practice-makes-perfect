"""Harness isolation, step metering, snapshot restoration and scattered scene contracts."""

import numpy as np
import pytest

from hitl_pmp.core.control_steps import ControlStepLimitReached
from hitl_pmp.environments.kitchen_recycling3d.domain import KitchenConfig, classify_zone
from hitl_pmp.environments.kitchen_recycling3d.environment import KitchenRecyclingEnvironment
from hitl_pmp.environments.kitchen_recycling3d.scene import initial_positions


@pytest.fixture
def environment():
    env = KitchenRecyclingEnvironment(config=KitchenConfig(scatter_objects=False))
    env.hard_reset()
    try:
        yield env
    finally:
        env.close()


def test_scatter_is_reproducible_separated_and_spans_kitchen():
    config = KitchenConfig()
    samples = []
    for seed in range(30):
        positions = np.array(initial_positions(config=config, seed=seed))
        np.testing.assert_array_equal(positions, initial_positions(config=config, seed=seed))
        assert len(set(positions[:, 0])) == 3
        assert np.all(np.linalg.norm(positions, axis=1) > config.robot_radius + 0.15)
        for index, position in enumerate(positions):
            for other in positions[index + 1 :]:
                assert np.linalg.norm(position - other) > 0.45
        samples.extend(positions)
    samples = np.array(samples)
    assert np.ptp(samples[:, 0]) > 1.3
    assert np.ptp(samples[:, 1]) > 3.5


def test_live_scatter_matches_seeded_positions_and_boundaries():
    env = KitchenRecyclingEnvironment(seed=13)
    try:
        env.hard_reset()
        intended = initial_positions(config=env.config, seed=13)
        for spec, position in zip(env.config.objects, intended, strict=True):
            observation = env.simulator().episode.observations[spec.name]
            assert classify_zone(config=env.config, observation=observation) == "kitchen"
            np.testing.assert_allclose(observation.position[:2], position, atol=0.01)
    finally:
        env.close()


@pytest.mark.parametrize("name", ["cube_control", "gatorade_20oz", "food_can"])
def test_single_object_scene_contains_only_requested_object(*, name):
    catalog = KitchenConfig()
    config = KitchenConfig(objects=(catalog.object(name=name),), required_objects=(name,))
    env = KitchenRecyclingEnvironment(config=config, seed=17, phase="deployment")
    try:
        env.hard_reset()
        simulator = env.simulator()
        assert set(simulator.names) == {name}
        assert set(simulator.episode.observations) == {name}
        assert [item.name for item in env.items] == [name]
        assert (
            classify_zone(config=config, observation=simulator.episode.observations[name])
            == "kitchen"
        )
        assert config.required_objects == (name,)
        with pytest.raises(PermissionError):
            env.reset_movables(destination=f"recover:{name}")
    finally:
        env.close()


def test_snapshot_restores_physics_and_episode_without_reset(*, environment):
    sim = environment.simulator()
    before = environment.get_current_state().model_copy(deep=True)
    snapshot = environment.snapshot()
    sim.idle(seconds=0.3)
    environment.reset_movables(destination="relocate_bin_near")
    assert sim.episode.human_cost == 12
    restored = environment.restore(snapshot=snapshot)
    assert sim.episode.human_cost == 0
    assert sim.data.time == snapshot["simulator"]["episode"]._time
    for obj in before.data:
        np.testing.assert_array_equal(restored.data[obj], before.data[obj])
    np.testing.assert_array_equal(
        environment.snapshot()["simulator"]["physics"], snapshot["simulator"]["physics"]
    )


def test_budget_interrupts_before_next_motor_step(*, environment):
    calls = []

    def before():
        if len(calls) >= 2:
            raise ControlStepLimitReached()

    def after():
        calls.append(environment.simulator().steps)

    environment.set_control_step_observers(before=before, after=after)
    start = environment.simulator().steps
    with pytest.raises(ControlStepLimitReached):
        environment.simulator().idle(seconds=1)
    assert environment.simulator().steps == start + 2
    assert len(calls) == 2


def test_human_cost_not_double_metered_as_robot_steps(*, environment):
    calls = []
    environment.set_control_step_observers(
        before=lambda: calls.append("before"), after=lambda: calls.append("after")
    )
    assert environment.reset_movables(destination="recover:food_can")
    assert environment.reset_movables(destination="relocate_bin_near")
    assert environment.reset_movables(destination="restore_bin_target")
    assert environment.simulator().episode.human_cost == 36
    assert calls == []


def test_deployment_rejects_all_human_paths_without_mutation():
    env = KitchenRecyclingEnvironment(phase="deployment")
    try:
        env.hard_reset()
        snapshot = env.snapshot()
        for destination in ("recover:food_can", "relocate_bin_near", "restore_bin_target"):
            with pytest.raises(PermissionError):
                env.reset_movables(destination=destination)
        for opcode in (3, 4, 5):
            with pytest.raises(PermissionError):
                env.take_action(action=np.array([opcode, 0, 0, 0, 0, 0]))
        np.testing.assert_array_equal(
            env.snapshot()["simulator"]["physics"], snapshot["simulator"]["physics"]
        )
        assert env.simulator().episode.human_cost == 0
    finally:
        env.close()


def test_evaluation_instance_cannot_reset_training(*, environment):
    before = environment.snapshot()
    evaluation = KitchenRecyclingEnvironment(phase="deployment", seed=51)
    try:
        evaluation.hard_reset()
        evaluation.reset_to_seed(seed=52)
    finally:
        evaluation.close()
    np.testing.assert_array_equal(
        environment.snapshot()["simulator"]["physics"], before["simulator"]["physics"]
    )


def test_observation_identity_changes_only_when_world_changes(*, environment):
    from hitl_pmp.environments.kitchen_recycling3d.problem import KitchenRecyclingProblem
    from hitl_pmp.environments.kitchen_recycling3d.tasks import KitchenRecyclingTasks
    from hitl_pmp.humans.oracle import UnconditionalHumanOracle
    from hitl_pmp.practice_loop import PracticeLoop, PracticeResetPolicy

    problem = KitchenRecyclingProblem(
        env=environment,
        tasks=KitchenRecyclingTasks(env=environment),
        human=UnconditionalHumanOracle,
    )
    before = environment.get_current_state()
    task = PracticeLoop._sample_practice_task(
        problem=problem, practice_reset_policy=PracticeResetPolicy.NEVER
    )
    assert task.initial_state is before
    assert environment.get_current_state() is before
    environment.simulator().idle(seconds=0.1)
    advanced = environment.get_current_state()
    assert advanced is not before
    assert environment.get_current_state() is advanced
    environment.reset_to_seed(seed=environment.seed)
    assert environment.get_current_state() is not advanced


def test_refused_pick_marks_blocked_despite_post_attempt_settling(*, environment, monkeypatch):
    simulator = environment.simulator()

    def refuse(*, name, calibration):
        del name, calibration
        simulator.idle(seconds=0.3, closed=True)
        return {"steps": 0, "error": "No collision-free grasp", "success": False}

    monkeypatch.setattr(simulator, "pick", refuse)
    before = simulator.steps
    state = environment.take_action(action=np.array([0, 0, 0.045, 0, 0, 0]))
    assert simulator.steps > before
    assert state.get(obj=environment.items[0], feature_name="blocked") == 1
    assert environment.reset_movables(destination="recover:cube_control")
    assert (
        environment.get_current_state().get(obj=environment.items[0], feature_name="blocked") == 0
    )

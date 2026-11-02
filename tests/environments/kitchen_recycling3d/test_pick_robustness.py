"""Physical pick regressions: success requires observed lift and bilateral contact."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.kitchen_recycling3d.domain import KitchenConfig

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None or importlib.util.find_spec("kinder_models") is None,
    reason="Optional Tossing3D simulation dependencies are not installed",
)


def preview_config():
    root = Path(__file__).resolve().parents[3]
    return KitchenConfig.model_validate_json(
        (root / "configs/kitchen_recycling3d_preview.json").read_text()
    )


def assert_physical_pick(*, simulator, name):
    before = simulator.episode.observations[name]
    result = simulator.pick(name=name)
    after = simulator.episode.observations[name]
    assert result["error"] is None, (name, result, before, after)
    assert result["success"], (result, before, after)
    assert result["steps"] > 0
    assert name in simulator.episode.holding
    assert after.finger_contacts == 2
    assert after.position[2] > before.position[2] + 0.07
    assert simulator.episode.human_cost == 0


def test_can_after_bottle_toss_without_base_contact(*, monkeypatch):
    from hitl_pmp.environments.kitchen_recycling3d.controllers import RecyclingPickController
    from hitl_pmp.environments.kitchen_recycling3d.simulator import RecyclingSimulator

    original_observe = RecyclingPickController.observe
    base_positions = []

    def observe(*args):
        controller, state = args
        original_observe(controller, state)
        if controller.spec.name == "food_can" and controller.current_phase.name == "BASE_MOTION":
            target = controller.objects[1]
            base_positions.append([state.get(target, axis) for axis in "xyz"])

    monkeypatch.setattr(RecyclingPickController, "observe", observe)
    simulator = RecyclingSimulator(config=preview_config(), pretty=False, seed=0)
    try:
        simulator.begin_deployment()
        assert_physical_pick(simulator=simulator, name="gatorade_20oz")
        result = simulator.toss(name="gatorade_20oz")
        assert result["error"] is None, result
        before = simulator.episode.observations["food_can"].position
        assert_physical_pick(simulator=simulator, name="food_can")
        assert base_positions
        assert np.max(np.linalg.norm(np.asarray(base_positions) - before, axis=1)) < 0.005
    finally:
        simulator.close()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_all_three_picked_in_scattered_episode(*, seed):
    from hitl_pmp.environments.kitchen_recycling3d.simulator import RecyclingSimulator

    config = preview_config().model_copy(update={"scatter_objects": True})
    simulator = RecyclingSimulator(config=config, pretty=False, seed=seed)
    try:
        simulator.begin_deployment()
        for name in ("gatorade_20oz", "food_can", "cube_control"):
            assert_physical_pick(simulator=simulator, name=name)
            result = simulator.toss(name=name)
            assert result["error"] is None, result
            assert not simulator.episode.holding
    finally:
        simulator.close()


@pytest.mark.parametrize("name", ["cube_control", "gatorade_20oz", "food_can"])
@pytest.mark.parametrize("seed", range(5))
def test_scattered_floor_pick(*, name, seed):
    from hitl_pmp.environments.kitchen_recycling3d.simulator import RecyclingSimulator

    config = preview_config().model_copy(update={"scatter_objects": True})
    simulator = RecyclingSimulator(config=config, pretty=False, seed=seed)
    try:
        simulator.begin_deployment()
        assert_physical_pick(simulator=simulator, name=name)
    finally:
        simulator.close()


def single_cube_config():
    config = preview_config()
    spec = config.object(name="cube_control")
    return KitchenConfig(
        objects=(spec,),
        required_objects=(spec.name,),
        scatter_objects=True,
        calibrations={
            key: value
            for key, value in config.calibrations.items()
            if key.startswith(spec.name + "/")
        },
    )


@pytest.mark.parametrize("seed", range(10))
def test_single_cube_scattered_floor_pick(*, seed):
    from hitl_pmp.environments.kitchen_recycling3d.simulator import RecyclingSimulator

    config = single_cube_config()
    simulator = RecyclingSimulator(config=config, pretty=False, seed=seed)
    try:
        simulator.begin_deployment()
        assert_physical_pick(simulator=simulator, name="cube_control")
    finally:
        simulator.close()


@pytest.mark.parametrize("seed", range(5))
def test_single_cube_release_and_repick(*, seed):
    from hitl_pmp.environments.kitchen_recycling3d.simulator import RecyclingSimulator

    simulator = RecyclingSimulator(config=single_cube_config(), pretty=False, seed=seed)
    try:
        simulator.begin_deployment()
        assert_physical_pick(simulator=simulator, name="cube_control")
        result = simulator.open_gripper()
        assert result["error"] is None, result
        assert result["success"], result
        assert not simulator.episode.holding
        assert simulator.episode.observations["cube_control"].finger_contacts == 0
        assert_physical_pick(simulator=simulator, name="cube_control")
    finally:
        simulator.close()


@pytest.mark.parametrize("name", ["gatorade_20oz", "food_can"])
@pytest.mark.parametrize("sideways", [False, True])
def test_upright_and_sideways_cylinders(*, name, sideways):
    from hitl_pmp.environments.kitchen_recycling3d.simulator import RecyclingSimulator

    simulator = RecyclingSimulator(config=preview_config(), pretty=False, seed=0)
    try:
        state = simulator.backend._require_state().copy()
        target = state.get_object_from_name(simulator.names[name])
        spec = simulator.config.object(name=name)
        quaternion = (
            Rotation.from_euler("z", 0.4) * Rotation.from_euler("y", np.pi / 2 if sideways else 0)
        ).as_quat()
        for key, value in zip(("qx", "qy", "qz", "qw"), quaternion, strict=True):
            state.set(target, key, float(value))
        state.set(target, "z", spec.dimensions[0 if sideways else 2] / 2 + 0.002)
        simulator.scene.set_state(state)
        simulator.backend._state = state
        simulator.idle(seconds=0.8)
        simulator.begin_deployment()
        assert_physical_pick(simulator=simulator, name=name)
    finally:
        simulator.close()

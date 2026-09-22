"""Partial resets retain the task's center-region support and physical safeguards."""

import copy
import importlib.util
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend
from hitl_pmp.environments.tossing3d.layout import Tossing3DLayout

needs_kinder = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None or importlib.util.find_spec("kinder_models") is None,
    reason="KINDER is an optional extra",
)


def _scene(*, predicate: str = "on", yaw: tuple[float, float] = (0.0, 0.0)):
    region = {"target": "ground", "ranges": [[2.6, -2.3, 3.42, 2.3]], "yaw_ranges": [list(yaw)]}
    return SimpleNamespace(
        task_config={
            "convex_placement_room_body": "tossing_room",
            "initial_state": [[predicate, "bin_0", "spawn"]],
            "regions": {"spawn": region},
            "objects": {"bin": {"bin_0": {"length": 0.3, "width": 0.3, "height": 0.2}}},
        },
        _objects_dict={
            "bin_0": SimpleNamespace(
                REGISTERED_NAME="bin",
                get_bounding_box_from_config=Mock(return_value=[-0.15, -0.15, 0, 0.15, 0.15, 0.2]),
            )
        },
    )


def test_on_region_override_does_not_mutate_the_task() -> None:
    scene = _scene()
    original = copy.deepcopy(scene.task_config)
    names, overrides = KinderBackend._movables_reset_regions(
        object_centric=scene, object_names=("bin_0",)
    )
    assert names["bin_0"] != "spawn"
    assert scene.task_config == original
    np.testing.assert_allclose(
        overrides[names["bin_0"]]["ranges"], [[2.445, -2.455, 3.575, 2.455]], atol=1e-15, rtol=0
    )
    overrides[names["bin_0"]]["yaw_ranges"][0][0] = 99
    assert scene.task_config == original


def test_in_region_keeps_upstream_containment_semantics() -> None:
    scene = _scene(predicate="in")
    names, overrides = KinderBackend._movables_reset_regions(
        object_centric=scene, object_names=("bin_0",)
    )
    assert names == {"bin_0": "spawn"}
    assert overrides == {}
    scene._objects_dict["bin_0"].get_bounding_box_from_config.assert_not_called()


def test_shared_spawn_region_gets_object_specific_overrides() -> None:
    scene = _scene()
    scene.task_config["initial_state"].append(["on", "small_bin", "spawn"])
    scene.task_config["objects"]["bin"]["small_bin"] = {"length": 0.1, "width": 0.1}
    scene._objects_dict["small_bin"] = SimpleNamespace(
        REGISTERED_NAME="bin",
        get_bounding_box_from_config=Mock(return_value=[-0.05, -0.05, 0, 0.05, 0.05, 0.1]),
    )
    names, overrides = KinderBackend._movables_reset_regions(
        object_centric=scene, object_names=("bin_0", "small_bin")
    )
    assert len(set(names.values())) == 2
    assert overrides[names["bin_0"]]["ranges"] != overrides[names["small_bin"]]["ranges"]


@needs_kinder
@pytest.mark.parametrize("yaw", [(0.0, 0.0), (45.0, 45.0), (-45.0, 45.0)])
@pytest.mark.parametrize(
    "center", [[2.6001, -2.2999], [3.4199, 2.2999], [3.343319892883301, -0.977752149105072]]
)
def test_upstream_sampler_restores_center_support_including_formerly_excluded_edges(
    *, yaw, center
) -> None:
    from kinder.envs.dynamic3d.placement_samplers import sample_feasible_ground_positions

    scene = _scene(yaw=yaw)
    # Boundary positions and a real test-bin position that the old reset excluded.
    center = np.array(center)
    delta = 1e-6
    region = scene.task_config["regions"]["spawn"]
    region["ranges"] = [[*(center - delta), *(center + delta)]]
    names, overrides = KinderBackend._movables_reset_regions(
        object_centric=scene, object_names=("bin_0",)
    )
    pose = sample_feasible_ground_positions(
        scene.task_config["objects"], np.random.default_rng(0), names, overrides, [], []
    )["bin"]["bin_0"]
    np.testing.assert_allclose(pose["position"][:2], center, atol=delta, rtol=0)
    assert np.deg2rad(yaw[0]) <= pose["yaw"] <= np.deg2rad(yaw[1])


@needs_kinder
def test_center_support_still_respects_room_and_obstacles() -> None:
    from kinder.envs.dynamic3d.placement_samplers import sample_feasible_ground_positions

    scene = _scene()
    names, overrides = KinderBackend._movables_reset_regions(
        object_centric=scene, object_names=("bin_0",)
    )
    for planes, obstacles in (
        ([(np.array([-1.0, 0.0]), -2.0)], []),
        ([], [[2.0, -3.0, 0.0, 4.0, 3.0, 1.0]]),
    ):
        with pytest.raises(RuntimeError, match="No feasible ground placement"):
            sample_feasible_ground_positions(
                scene.task_config["objects"],
                np.random.default_rng(0),
                names,
                overrides,
                planes,
                obstacles,
            )


@needs_kinder
@pytest.mark.parametrize("layout", [Tossing3DLayout.BARRIER, Tossing3DLayout.SAME_SIDE])
def test_live_partial_reset_preserves_robot_and_goal_attachment(*, layout) -> None:
    env = Tossing3DEnvironment(layout=layout)
    try:
        env.hard_reset()
        env.take_action(action=np.array([0, 0, 0, 0, 0], dtype=float))
        before = env.backend().snapshot()
        robot = before.get_object_from_name(env.backend().robot_name)
        robot_before = before[robot].copy()
        backend = env.backend()
        scene = backend._object_centric()
        config_before = copy.deepcopy(scene.task_config)
        barrier_before = before[before.get_object_from_name("cuboid_barrier")].copy()
        assert env.reset_movables()
        after = backend.snapshot()
        np.testing.assert_allclose(after[robot], robot_before, atol=1e-7, rtol=0)
        np.testing.assert_allclose(
            after[after.get_object_from_name("cuboid_barrier")], barrier_before, atol=1e-7, rtol=0
        )
        assert scene.task_config == config_before
        bin_ = after.get_object_from_name("bin_0")
        bbox = backend.goal_region_bbox()
        assert (bbox[0] + bbox[3]) / 2 == pytest.approx(after.get(bin_, "x"), abs=1e-6)
        assert (bbox[1] + bbox[4]) / 2 == pytest.approx(after.get(bin_, "y"), abs=1e-6)
    finally:
        env.close()

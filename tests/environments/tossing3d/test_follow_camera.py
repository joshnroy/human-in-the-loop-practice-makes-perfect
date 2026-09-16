"""Presentation tracking moves only the camera and restores it after rendering."""

from types import SimpleNamespace

import numpy as np
import pytest

from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend, interaction_camera_position


@pytest.mark.parametrize("fail", [False, True])
def test_camera_follows_base_without_accumulating_motion(fail):
    positions = np.array([[1.0, -1.8, 5.5]])
    original = positions.copy()
    model = SimpleNamespace(mj_model=SimpleNamespace(cam_pos=positions), camera_name2id=lambda _: 0)
    sim = SimpleNamespace(model=model, forward=lambda: None)
    scene = SimpleNamespace(
        _robot_env=SimpleNamespace(sim=sim, camera_width=640, camera_height=480),
        task_config={"cameras": {"task_view": {"lookat": [1, 0, 0]}}},
    )
    base = {"pos_base_x": 3.0, "pos_base_y": 2.0}
    objects = {
        "robot": base,
        "cube_0": {"x": 4.0, "y": 2.0, "z": 3.0},
        "bin_0": {"x": 6.0, "y": 2.0, "z": 0.0},
    }
    expected = interaction_camera_position(
        points=np.array([[3, 2, .7], [4, 2, 3], [6, 2, 0]]),
        original_position=original[0], original_target=np.array([1, 0, 0]),
        vertical_fov=45, aspect=640/480,
    )

    def render():
        np.testing.assert_allclose(positions[0], expected)
        if fail:
            raise RuntimeError("render failed")
        return np.zeros((2, 2, 3), dtype=np.uint8)

    backend = KinderBackend()
    backend._robot_name = "robot"
    backend._state = SimpleNamespace(
        get_object_from_name=lambda name: name, get=lambda name, key: objects[name][key]
    )
    backend._raw_env = SimpleNamespace(
        unwrapped=SimpleNamespace(_object_centric_env=scene), render=render
    )
    for _ in range(2):
        if fail:
            with pytest.raises(RuntimeError, match="render failed"):
                backend.render(follow_robot=True)
        else:
            assert backend.render(follow_robot=True).shape == (2, 2, 3)
        np.testing.assert_array_equal(positions, original)


@pytest.mark.parametrize("aspect", [0.75, 4/3, 2.0])
def test_fit_contains_padded_bounds_during_high_and_far_tosses(aspect):
    points = np.array([[0, 0, .7], [8, -5, 6], [10, 0, 0]])
    position = interaction_camera_position(
        points=points, original_position=np.array([1, -1.8, 5.5]),
        original_target=np.array([1, 0, 0]), vertical_fov=65, aspect=aspect,
    )
    lower, upper = points.min(axis=0)-1, points.max(axis=0)+1
    center = (lower+upper)/2
    radius = np.linalg.norm(upper-lower)/2
    angular_radius = np.arcsin(radius/np.linalg.norm(position-center))
    vertical = np.deg2rad(65)/2
    assert angular_radius < vertical
    assert angular_radius < np.arctan(np.tan(vertical)*aspect)

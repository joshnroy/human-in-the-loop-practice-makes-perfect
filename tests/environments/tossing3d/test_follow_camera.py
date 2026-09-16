"""Presentation tracking moves only the camera and restores it after rendering."""

from types import SimpleNamespace

import numpy as np
import pytest

from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend


@pytest.mark.parametrize("fail", [False, True])
def test_camera_follows_base_without_accumulating_motion(fail):
    positions = np.array([[1.0, -1.8, 5.5]])
    original = positions.copy()
    model = SimpleNamespace(cam_pos=positions, camera_name2id=lambda _: 0)
    sim = SimpleNamespace(model=model, forward=lambda: None)
    scene = SimpleNamespace(
        _robot_env=SimpleNamespace(sim=sim),
        task_config={"cameras": {"task_view": {"lookat": [1, 0, 0]}}},
    )
    base = {"pos_base_x": 3.0, "pos_base_y": 2.0}

    def render():
        np.testing.assert_allclose(positions[0], [3, 0.2, 5.5])
        if fail:
            raise RuntimeError("render failed")
        return np.zeros((2, 2, 3), dtype=np.uint8)

    backend = KinderBackend()
    backend._robot_name = "robot"
    backend._state = SimpleNamespace(
        get_object_from_name=lambda _: "robot", get=lambda _, key: base[key]
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

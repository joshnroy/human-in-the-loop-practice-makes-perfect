"""The shared no-op pickup hook preserves valid empty approach paths."""

from types import SimpleNamespace

import numpy as np
import pytest

from hitl_pmp.environments.sweep_drawer3d.primitives import Primitives


class FollowReached(Exception):
    """Stop the fake controller only after its empty approach reaches motion."""


def test_empty_replanned_approach_reaches_motion_with_observed_start(*, monkeypatch) -> None:
    from pybullet_helpers import geometry
    from pybullet_helpers.ikfast import utils

    monkeypatch.setattr(geometry, "set_pose", lambda *_: None)
    monkeypatch.setattr(utils, "ikfast_closest_inverse_kinematics", lambda *_, **__: [np.zeros(13)])
    actual = np.arange(7, dtype=float) / 10
    captured = []

    def hook(*, start, path):
        captured.append(start.copy())
        return Primitives.wiper_pickup_descent_clear(SimpleNamespace(), start=start, path=path)

    def follow(*, path, grip):
        assert path == []
        assert grip == 0
        raise FollowReached

    scene = SimpleNamespace(
        wiper_body=1, cid=0,
        plan_base=lambda **_: [(0., 0., 0.)], sync=lambda **_: None,
        bodies=lambda: set(), robot=SimpleNamespace(set_joints=lambda _: None),
        planning_fingers=lambda **_: np.zeros(13), in_collision=lambda **_: False,
        linear_path=lambda **_: [np.zeros(7)], plan_arm=lambda **_: [np.zeros(7)],
    )
    session = SimpleNamespace(
        mj_data=SimpleNamespace(geom_xpos=np.array([[0., 0., .19]]),
                                geom_xmat=np.eye(3).reshape(1, 9)),
        position=lambda **_: np.zeros(3), quaternion=lambda **_: (0., 0., 0., 1.),
        arm=lambda: actual.copy(),
    )
    primitive = SimpleNamespace(
        session=session, scene=scene,
        wiper_handle_geometry=lambda: (0, 2), wiper_grasp_yaw=lambda **_: 0.,
        wiper_approach_angles=lambda: (1.2,), wiper_grasp_offsets=lambda: (-.09,),
        wiper_grasp_point=lambda **kwargs: kwargs["center"], wiper_grasp_standoff=lambda: .035,
        stances=lambda **_: [(0., 0., 0.)], wiper_pickup_descent_clear=hook,
        wiper_pick_targets_after_navigation=lambda **kwargs: (kwargs["hover"], kwargs["target"]),
        resolve_at_actual_base=lambda **_: ([], [np.zeros(7)]),
        motion=SimpleNamespace(go_home=lambda **_: None, set_gripper=lambda **_: None,
                               drive=lambda **_: True, follow=follow),
    )
    with pytest.raises(FollowReached):
        Primitives.recover_wiper(primitive)
    assert len(captured) == 2
    np.testing.assert_array_equal(captured[-1], actual)

"""Blocked IK endpoints must not starve a later collision-free floor descent."""

from types import SimpleNamespace

import numpy as np

from hitl_pmp.environments.sweep_simple3d.controllers import FloorPlanningScene


def test_blocked_endpoints_do_not_consume_path_search_budget(*, monkeypatch) -> None:
    # The diagnostic failure had valid endpoints ranked after eight blocked ones.
    solutions = [np.array([i, 0, 0, 0, 0, 0, 0], dtype=float) for i in range(13)]
    monkeypatch.setattr(
        "pybullet_helpers.ikfast.utils.ikfast_closest_inverse_kinematics",
        lambda *args, **kwargs: solutions,
    )
    searched = []

    def plan(**kwargs):
        searched.append(kwargs["goal"][0])
        return [kwargs["goal"]]

    scene = SimpleNamespace(
        linear_path=lambda **kwargs: None,
        sync=lambda: None,
        robot=None,
        within_arm_limits=lambda **kwargs: True,
        planning_fingers=lambda **kwargs: kwargs["arm"],
        in_collision=lambda **kwargs: kwargs["joints"][0] < 12,
        native_joint_path=plan,
        held_path_clear=lambda **kwargs: True,
        wiper_body=1,
        max_tool_tilt=0.2,
    )
    path = FloorPlanningScene.floor_descent(
        scene, start=np.zeros(7), target=None, bodies=set(), held_tf=None
    )
    assert searched == [12]
    np.testing.assert_array_equal(path[-1], solutions[-1])

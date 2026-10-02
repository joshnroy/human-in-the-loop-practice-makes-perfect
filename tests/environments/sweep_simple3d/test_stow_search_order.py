"""A feasible early carry candidate gets the existing checked joint fallback."""

from types import SimpleNamespace

import mujoco
import numpy as np
from pybullet_helpers.geometry import Pose
from pybullet_helpers.ikfast import utils

from hitl_pmp.environments.sweep_simple3d.controllers import FloorPrimitives


def test_stow_tries_checked_fallback_before_enumerating_more_candidates(*, monkeypatch) -> None:
    calls = []
    goal = np.arange(7, dtype=float) / 10
    bodies = {3, 4}

    def plan_arm(**kwargs):  # noqa: ANN003, ANN202 -- fake scene callback
        calls.append(kwargs)
        if len(calls) == 1:  # HOME is unavailable for this physical attachment.
            return None
        return [goal] if kwargs["allow_joint_fallback"] else None

    def inverse_kinematics(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202 -- library callback
        assert len(calls) == 1, "Search must accept the first checked fallback"
        return [goal]

    session = SimpleNamespace(
        base=lambda: np.zeros(3),
        position=lambda **kwargs: np.array([0.0, 0.0, 0.2]),
        quaternion=lambda **kwargs: (0.0, 0.0, 0.0, 1.0),
        yaw=lambda **kwargs: 0.0,
    )
    session.mj_model = mujoco.MjModel.from_xml_string("<mujoco/>")
    session.mj_data = mujoco.MjData(session.mj_model)
    scene = SimpleNamespace(
        fk=lambda **_: Pose((0.0, 0.0, 0.0)),
        ee_now=lambda: Pose((0.0, 0.0, 0.3)),
        plan_arm=plan_arm,
        bodies=lambda: bodies,
        robot=object(),
        wiper_body=7,
        within_arm_limits=lambda **kwargs: True,
    )
    scene.sync = lambda: None
    scene.max_tool_tilt = 1.1
    primitive = FloorPrimitives.model_construct(session=session, scene=scene, motion=None)
    monkeypatch.setattr(utils, "ikfast_closest_inverse_kinematics", inverse_kinematics)
    np.testing.assert_array_equal(primitive.wiper_stow_goal(), goal)
    assert len(calls) == 2
    assert calls[0]["allow_joint_fallback"] is False
    assert calls[1]["allow_joint_fallback"] is True
    assert all(call["bodies"] is bodies and call["held"] == 7 for call in calls)
    assert calls[0]["held_tf"] == calls[1]["held_tf"]


def test_eligible_home_stays_first_without_compact_search(*, monkeypatch) -> None:
    from hitl_pmp.environments.sweep_drawer3d.types import SweepDrawerScene

    calls = []
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    session = SimpleNamespace(
        mj_model=model,
        mj_data=mujoco.MjData(model),
        base=lambda: np.zeros(3),
        position=lambda **_: np.zeros(3),
        quaternion=lambda **_: (0.0, 0.0, 0.0, 1.0),
    )

    def plan(**kwargs):  # noqa: ANN003, ANN202 -- fake planning callback
        calls.append(kwargs)
        return [np.array(SweepDrawerScene.HOME)]

    def unexpected(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202 -- library callback
        raise AssertionError("Eligible HOME should avoid compact IK search")

    scene = SimpleNamespace(
        sync=lambda: None,
        ee_now=lambda: Pose((0.0, 0.0, 0.0)),
        fk=lambda **_: Pose((0.0, 0.0, 0.0)),
        bodies=lambda: set(),
        max_tool_tilt=1.1,
        plan_arm=plan,
        wiper_body=1,
    )
    primitive = FloorPrimitives.model_construct(session=session, scene=scene, motion=None)
    monkeypatch.setattr(utils, "ikfast_closest_inverse_kinematics", unexpected)
    np.testing.assert_array_equal(primitive.wiper_stow_goal(), SweepDrawerScene.HOME)
    assert len(calls) == 1
    assert calls[0]["allow_joint_fallback"] is False

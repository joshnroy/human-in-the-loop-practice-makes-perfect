"""Placement checks a complete carried path and actual settled home conditions."""

from types import SimpleNamespace

import numpy as np
import pytest
from pybullet_helpers.geometry import Pose
from pybullet_helpers.ikfast import utils

from hitl_pmp.environments.sweep_simple3d.controllers import ExecutionError, FloorPrimitives
from hitl_pmp.environments.sweep_simple3d.regions import SimpleRegions


@pytest.mark.parametrize("floor_clear", [True, False])
def test_place_prefers_floor_and_refreshes_carried_descent(
    *, monkeypatch: pytest.MonkeyPatch, floor_clear: bool
) -> None:
    calls = []
    moved = False

    def follow(**kwargs):
        nonlocal moved
        calls.append(("follow", kwargs))
        moved = True
        return True

    def ik(robot, *, world_from_target):  # noqa: PLR0917
        del robot
        calls.append(("ik", world_from_target.position[2]))
        # Invalid candidates must not consume the twelve-candidate budget.
        return [np.full(7, 9.0)] * 13 + [np.zeros(7)]

    def descent(**kwargs):
        calls.append(("descent", kwargs))
        if not moved and kwargs["target"].position[2] < 0.1 and not floor_clear:
            return None
        return [np.zeros(7), np.zeros(7)]

    monkeypatch.setattr(utils, "ikfast_closest_inverse_kinematics", ik)
    monkeypatch.setattr(
        SimpleRegions,
        "validate",
        lambda **kwargs: SimpleNamespace(
            checks={
                "wiper_0:region": True,
                "wiper_0:upright": True,
                "wiper_0:yaw": True,
                "cube_0:region": False,
            }
        ),
    )
    scene = SimpleNamespace(
        robot=object(),
        wiper_body=5,
        sync=lambda: None,
        ee_now=lambda: Pose.identity(),
        bodies=lambda: {3},
        within_arm_limits=lambda **kwargs: bool(np.max(kwargs["arm"]) < 3),
        plan_arm=lambda **kwargs: [np.zeros(7), np.zeros(7)],
        floor_descent=descent,
        linear_path=lambda **kwargs: [np.zeros(7)],
    )
    primitive = SimpleNamespace(
        stow_wiper=lambda: None,
        stances=lambda **kwargs: [(0.0, 0.0, 0.0)],
        place_transport_stance=lambda **kwargs: (0.0, 0.0, 0.0),
        place_target_orientation=lambda: (0.0, 0.0, 0.0, 1.0),
        transport_wiper=lambda **kwargs: None,
        require_handle=lambda **kwargs: None,
        scene=scene,
        session=SimpleNamespace(
            initial_pose=lambda **kwargs: ((1.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            position=lambda **kwargs: np.array([0.01 if moved else 0.0, 0.0, 0.0]),
            quaternion=lambda **kwargs: (0.0, 0.0, 0.0, 1.0),
            arm=lambda: np.zeros(7),
        ),
        motion=SimpleNamespace(follow=follow, set_gripper=lambda **kwargs: None),
    )
    assert "observed settled" in FloorPrimitives.place_wiper_at_start(primitive)
    heights = [value for kind, value in calls if kind == "ik"]
    assert heights == pytest.approx([0.151] if floor_clear else [0.151, 0.45])
    descents = [value for kind, value in calls if kind == "descent"]
    assert descents[-1]["held_tf"].position[0] == pytest.approx(0.01)
    assert descents[-1]["target"].position[0] == pytest.approx(0.99)
    assert all("max_ticks" not in value for kind, value in calls if kind == "follow")
    monkeypatch.setattr(
        SimpleRegions,
        "validate",
        lambda **kwargs: SimpleNamespace(
            checks={"wiper_0:region": False, "wiper_0:upright": True, "wiper_0:yaw": True}
        ),
    )
    with pytest.raises(ExecutionError, match="did not settle"):
        FloorPrimitives.place_wiper_at_start(primitive)

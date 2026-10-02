"""Only completely exhausted, unchanged native carry searches are cached."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from pybullet_helpers.geometry import Pose
from pybullet_helpers.ikfast import utils

from hitl_pmp.environments.sweep_simple3d.controllers import FloorPrimitives
from hitl_pmp.environments.sweep_simple3d.physical.motion import ExecutionError


@pytest.mark.parametrize("change", ["qpos", "geometry", "tilt", "exception", "success"])
def test_exact_exhaustion_cache(*, monkeypatch, change: str) -> None:
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1"/>'
        "</body></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    calls = []
    ik_calls = []
    mode = ["exception" if change == "exception" else "fail"]

    def plan(**kwargs):  # noqa: ANN003, ANN202 -- fake planning callback
        calls.append(kwargs)
        if mode[0] == "exception":
            raise RuntimeError("interrupted search")
        return [np.zeros(7)] if mode[0] == "success" else None

    session = SimpleNamespace(
        mj_model=model,
        mj_data=data,
        base=lambda: np.zeros(3),
        position=lambda **kwargs: np.zeros(3),
        quaternion=lambda **kwargs: (0.0, 0.0, 0.0, 1.0),
        yaw=lambda **kwargs: 0.0,
    )
    scene = SimpleNamespace(
        fk=lambda **_: Pose((0.0, 0.0, 0.0)),
        sync=lambda: None,
        ee_now=lambda: Pose((0.0, 0.0, 0.0)),
        plan_arm=plan,
        bodies=lambda: {1},
        robot=object(),
        wiper_body=2,
        within_arm_limits=lambda **kwargs: True,
        max_tool_tilt=1.1,
    )
    primitive = FloorPrimitives.model_construct(session=session, scene=scene, motion=None)

    def ik(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202 -- library callback
        ik_calls.append(kwargs)
        return []

    monkeypatch.setattr(utils, "ikfast_closest_inverse_kinematics", ik)
    if change == "exception":
        for _ in range(2):
            with pytest.raises(RuntimeError, match="interrupted"):
                primitive.wiper_stow_goal()
        assert len(calls) == 2
        return
    # Each call still raises the ordinary execution failure for its action.
    failures = 0
    for _ in range(2):
        with pytest.raises(ExecutionError, match="No collision-free compact"):
            primitive.wiper_stow_goal()
        failures += 1
    assert failures == 2
    assert len(calls) == 1
    assert len(ik_calls) == 120  # One complete 2 x 3 x 4 x 5 search, not two.
    if change == "qpos":
        data.qpos[0] += 1e-12
    elif change == "geometry":
        model.geom_size[0, 0] += 1e-12
    else:
        scene.max_tool_tilt += 1e-12
    if change == "success":
        mode[0] = "success"
        primitive.wiper_stow_goal()
        primitive.wiper_stow_goal()
        assert len(calls) == 3  # Success is not cached.
    else:
        with pytest.raises(ExecutionError):
            primitive.wiper_stow_goal()
        assert len(calls) == 2
        assert len(ik_calls) == 240

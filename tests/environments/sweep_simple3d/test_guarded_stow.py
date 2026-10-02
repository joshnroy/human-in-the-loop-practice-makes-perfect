"""Stow distinguishes drift interruption from reaching the arm goal."""

from types import SimpleNamespace

import numpy as np
import pytest
from pybullet_helpers.geometry import Pose

from hitl_pmp.environments.sweep_drawer3d.motion import ExecutionError
from hitl_pmp.environments.sweep_simple3d.controllers import FloorPrimitives


@pytest.mark.parametrize("mode", ["replan", "alternate_ik", "budget", "stall", "grip_loss"])
def test_stow_replans_live_grasp_and_preserves_failures(*, monkeypatch, mode: str) -> None:
    state = SimpleNamespace(arm=np.zeros(7), tool=np.zeros(3), lost=False, ticks=0)
    attachments = []
    guards = []

    def require_handle(self, *, phase: str) -> None:  # noqa: ANN001, PLR0917 -- bound-method callback
        guards.append(phase)
        if state.lost:
            raise ExecutionError("test physical grip lost")

    def plan_arm(**kwargs):  # noqa: ANN003, ANN202 -- fake planning callback
        attachments.append(kwargs["held_tf"])
        return [np.full(7, 0.4 if mode == "alternate_ik" else 1.0)]

    def follow(**kwargs) -> bool:  # noqa: ANN003 -- fake motion callback
        state.ticks += 1
        if mode == "alternate_ik":
            state.arm[:] = kwargs["path"][-1]
            kwargs["tick_guard"]()
            assert not kwargs["stop_condition"]()
            return True
        if mode == "grip_loss":
            state.lost = True
            kwargs["tick_guard"]()
            pytest.fail("per-tick guard must interrupt immediately")
        if mode != "stall":
            state.arm += 0.1
        state.tool[0] += 0.02
        kwargs["tick_guard"]()
        if mode == "replan" and state.ticks == 2:
            state.arm[:] = 1.0
            return True
        assert kwargs["stop_condition"]()
        return True  # Motion.follow also returns True for drift interruption.

    session = SimpleNamespace(arm=lambda: state.arm.copy(),
                              position=lambda **kwargs: state.tool.copy(),
                              quaternion=lambda **kwargs: (0.0, 0.0, 0.0, 1.0))
    scene = SimpleNamespace(ee_now=lambda: Pose((0.0, 0.0, 0.0)),
                            plan_arm=plan_arm, bodies=set, wiper_body=1)
    primitive = FloorPrimitives.model_construct(
        session=session, scene=scene, motion=SimpleNamespace(follow=follow)
    )
    monkeypatch.setattr(FloorPrimitives, "require_handle", require_handle)
    monkeypatch.setattr(FloorPrimitives, "wiper_stow_goal", lambda self: np.ones(7))
    if mode == "alternate_ik":
        primitive.stow_wiper()
        assert state.ticks == 1
        np.testing.assert_array_equal(state.arm, np.full(7, 0.4))
    elif mode == "replan":
        primitive.stow_wiper()
        assert state.ticks == 2
        assert attachments[0].position[0] == 0.0
        assert attachments[1].position[0] == pytest.approx(0.02)
        np.testing.assert_array_equal(state.arm, np.ones(7))
    else:
        message = {"budget": "eight grasp replanning", "stall": "no arm progress",
                   "grip_loss": "test physical grip lost"}[mode]
        with pytest.raises(ExecutionError, match=message):
            primitive.stow_wiper()
        assert state.ticks == (8 if mode == "budget" else 1)
    assert "upright transport" in guards

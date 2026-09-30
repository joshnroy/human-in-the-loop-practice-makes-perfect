"""Retaining pickup posture is opt-in and cannot short-circuit compact stow."""

from types import SimpleNamespace

import numpy as np
import pytest
from pybullet_helpers.geometry import Pose

from hitl_pmp.environments.sweep_drawer3d.motion import ExecutionError
from hitl_pmp.environments.sweep_drawer3d.primitives import Primitives
from hitl_pmp.environments.sweep_simple3d.controllers import FloorPrimitives


@pytest.mark.parametrize("case", ["default", "clear", "collision", "low", "lost"])
def test_pickup_carry_admission_and_fallback(*, monkeypatch, case: str) -> None:
    current, compact = np.zeros(7), np.ones(7)
    checked = []
    stows = []
    bodies = {4, 9}

    def clear_path(**kwargs) -> bool:  # noqa: ANN003 -- fake planning callback
        checked.append(kwargs)
        return case != "collision"

    def require_handle(self, *, phase: str) -> None:  # noqa: ANN001, PLR0917 -- bound callback
        if case == "lost":
            raise ExecutionError("physical grasp lost")

    def stow(self) -> np.ndarray:  # noqa: ANN001, PLR0917 -- bound callback
        stows.append(True)
        return compact

    scene = SimpleNamespace(sync=lambda: None, ee_now=lambda: Pose((0.0, 0.0, 0.0)),
                            bodies=lambda: bodies, held_path_clear=clear_path,
                            wiper_body=3, max_tool_tilt=1.1)
    session = SimpleNamespace(arm=lambda: current,
                              position=lambda **kwargs: np.array([0.0, 0.0, 0.15]),
                              quaternion=lambda **kwargs: (0.0, 0.0, 0.0, 1.0))
    primitive = FloorPrimitives.model_construct(
        session=session, scene=scene, motion=None, floor_clearance=0.005
    )
    if case != "default":
        primitive.retain_pickup_carry_pose = True
    monkeypatch.setattr(FloorPrimitives, "require_handle", require_handle)
    monkeypatch.setattr(FloorPrimitives, "wiper_stow_goal", stow)
    monkeypatch.setattr(FloorPrimitives, "blade_minimum_height",
                        lambda self: 0.001 if case == "low" else 0.15)
    if case == "lost":
        with pytest.raises(ExecutionError, match="physical grasp lost"):
            primitive.wiper_pickup_carry_goal()
        assert not stows
        return
    result = primitive.wiper_pickup_carry_goal()
    np.testing.assert_array_equal(result, current if case == "clear" else compact)
    assert len(stows) == (0 if case == "clear" else 1)
    if checked:
        assert checked[0]["bodies"] is bodies
        assert checked[0]["held"] == 3
        assert checked[0]["allowed_tilt"] == 1.1
        np.testing.assert_array_equal(checked[0]["path"][0], current)
    else:
        assert case in ("default", "low")


def test_shared_default_delegates_and_explicit_stow_ignores_pickup_hook(*, monkeypatch) -> None:
    compact = np.zeros(7)
    monkeypatch.setattr(Primitives, "wiper_stow_goal", lambda self: compact)
    assert Primitives.model_construct().wiper_pickup_carry_goal() is compact
    requested = []

    def compact_goal(self) -> np.ndarray:  # noqa: ANN001, PLR0917 -- bound callback
        requested.append(True)
        return compact

    def unexpected_pickup(self) -> np.ndarray:  # noqa: ANN001, PLR0917 -- bound callback
        pytest.fail("Explicit transport stow must not use pickup-only retention")

    monkeypatch.setattr(FloorPrimitives, "wiper_stow_goal", compact_goal)
    monkeypatch.setattr(FloorPrimitives, "wiper_pickup_carry_goal", unexpected_pickup)
    monkeypatch.setattr(FloorPrimitives, "require_handle", lambda self, **kwargs: None)
    primitive = FloorPrimitives.model_construct(
        session=SimpleNamespace(arm=lambda: compact), retain_pickup_carry_pose=True
    )
    primitive.stow_wiper()
    assert requested == [True]

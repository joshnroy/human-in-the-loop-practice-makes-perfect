"""Fixed placement can leave a blocked bearing without bypassing native routes."""

from types import SimpleNamespace

import numpy as np
import pytest
from pybullet_helpers.geometry import Pose

from hitl_pmp.environments.sweep_simple3d.controllers import ExecutionError, FloorPrimitives


@pytest.mark.parametrize("original_status", ["clear", "no_route", "held_collision"])
def test_place_stance_preserves_order_and_checks_carried_routes(*, original_status: str) -> None:
    original = (1.12, 1.44, -0.90)
    north = (1.55, 0.89 + 0.7, -np.pi / 2)
    attempts = []
    checked = []
    current_base = None

    def sync(*, base=None):
        nonlocal current_base
        current_base = base

    def routes(*, target):
        attempts.append(target)
        if target == original and original_status == "no_route":
            return []
        return [("native", [target])]

    def collision(**kwargs):
        assert kwargs["held"] == 5 and kwargs["bodies"] == {7}
        checked.append(current_base)
        return current_base == original and original_status == "held_collision"

    scene = SimpleNamespace(
        ee_now=lambda: Pose.identity(),
        sync=sync,
        planning_fingers=lambda **kwargs: kwargs["arm"],
        bodies=lambda: {7},
        wiper_body=5,
        in_collision=collision,
    )
    primitive = SimpleNamespace(
        distance=0.7,
        scene=scene,
        stances=lambda **kwargs: [original],
        transport_base_candidates=routes,
        session=SimpleNamespace(
            arm=lambda: np.zeros(7),
            position=lambda **kwargs: np.zeros(3),
            quaternion=lambda **kwargs: (0.0, 0.0, 0.0, 1.0),
        ),
    )
    result = FloorPrimitives.place_transport_stance(primitive, position=np.array([1.55, 0.89, 0.0]))
    assert result == pytest.approx(original if original_status == "clear" else north)
    assert attempts == ([original] if original_status == "clear" else [original, north])
    assert checked[-1] == result
    assert current_base is None  # Hypothetical planning pose is restored.
    scene.in_collision = lambda **kwargs: True
    with pytest.raises(ExecutionError, match="No native carried base route"):
        FloorPrimitives.place_transport_stance(primitive, position=np.array([1.55, 0.89, 0.0]))

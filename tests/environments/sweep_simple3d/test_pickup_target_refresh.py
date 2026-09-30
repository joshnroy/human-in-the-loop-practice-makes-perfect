"""Moved tools are targeted at their observed pose after navigation."""

from types import SimpleNamespace

import numpy as np
from pybullet_helpers.geometry import Pose

from hitl_pmp.environments.sweep_drawer3d.primitives import Primitives
from hitl_pmp.environments.sweep_simple3d.controllers import FloorPrimitives


def test_shared_default_preserves_native_targets() -> None:
    primitive = Primitives.model_construct()
    hover, target = Pose((1, 2, 3)), Pose((4, 5, 6))
    result = primitive.wiper_pick_targets_after_navigation(
        hover=hover, target=target, along=-.09, approach=np.array([1., 0., 0.])
    )
    assert result[0] is hover and result[1] is target


def test_simple_refresh_uses_live_translated_rotated_handle(*, monkeypatch) -> None:
    # Handle axis has changed as well as position since the plan was made.
    rotation = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    data = SimpleNamespace(geom_xpos=np.array([[1.1, 1.15, .19]]),
                           geom_xmat=np.array([rotation.reshape(-1)]))
    records = []
    session = SimpleNamespace(
        mj_data=data, ticks=17, _write=lambda *, record: records.append(record)
    )
    primitive = FloorPrimitives.model_construct(session=session, diagnostic_grasp_standoff=.020)
    monkeypatch.setattr(FloorPrimitives, "wiper_handle_geometry", lambda self: (0, 2))
    old = Pose((1.05, 1.19, .10))
    hover, target = primitive.wiper_pick_targets_after_navigation(
        hover=Pose((0., 0., 0.)), target=old, along=-.09, approach=np.array([1., 0., 0.])
    )
    np.testing.assert_allclose(target.position, [1.08, 1.24, .19])
    np.testing.assert_allclose(hover.position, [1., 1.24, .19])
    assert target.orientation == old.orientation
    assert records[0]["kind"] == "pickup_target_refreshed"
    np.testing.assert_allclose(data.geom_xpos[0], [1.1, 1.15, .19])

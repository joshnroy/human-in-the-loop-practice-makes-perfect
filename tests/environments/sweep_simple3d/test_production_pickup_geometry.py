"""Production floor pickup uses native handle geometry without probe monkeypatches."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from hitl_pmp.environments.sweep_simple3d.controllers import FloorPrimitives


@pytest.mark.parametrize("yaw", [-2.1, 0.0, 1.3])
def test_grasp_yaw_tracks_native_handle_face(*, yaw: float) -> None:
    c, s = np.cos(yaw), np.sin(yaw)
    matrix = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    session = SimpleNamespace(mj_data=SimpleNamespace(geom_xmat=np.array([
        np.eye(3).ravel(), matrix.ravel(),
    ])))
    primitive = FloorPrimitives.model_construct(session=session)
    with patch.object(FloorPrimitives, "wiper_handle_geometry", return_value=(1, 2)):
        assert primitive.wiper_grasp_yaw(axis=np.array([0.0, 0.0, 1.0])) == pytest.approx(yaw)


def test_production_floor_profile_keeps_reverse_mode_unpromoted() -> None:
    primitive = FloorPrimitives.model_construct()
    assert primitive.floor_clearance == .005
    assert primitive.stand_ahead and primitive.native_contact_guard
    assert primitive.retain_pickup_carry_pose
    assert primitive.wiper_grasp_offsets() == (-.12,)
    assert primitive.wiper_grasp_standoff() == .020
    assert primitive.narrow_contact is False
    assert primitive.distance == .70

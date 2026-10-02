"""Placement chooses inside the native start yaw set without changing acceptance."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.sweep_drawer3d.start_regions import SweepRegions
from hitl_pmp.environments.sweep_simple3d.controllers import FloorPrimitives
from hitl_pmp.environments.sweep_simple3d.session import SweepSimpleSession


def test_actual_native_start_interval_center_and_predicates_are_unchanged() -> None:
    session = SweepSimpleSession(seed=1)
    try:
        core = session.env.unwrapped._object_centric_env
        before = deepcopy(core.task_config)
        qpos = session.mj_data.qpos.copy()
        primitive = SimpleNamespace(session=session)
        orientation = FloorPrimitives.place_target_orientation(primitive)
        rotation = Rotation.from_quat(orientation)
        yaw = rotation.as_euler("xyz")[2]
        region = next(region for _, name, region in before["initial_state"] if name == "wiper_0")
        bounds = before["regions"][region]["yaw_ranges"]
        assert bounds == [[-45, 45]]
        assert yaw == pytest.approx(0.)
        np.testing.assert_allclose(rotation.as_matrix()[:, 2], [0., 0., 1.])
        assert SweepRegions.yaw_matches(yaw=yaw, ranges=bounds)
        assert not SweepRegions.yaw_matches(yaw=np.radians(46.62895575), ranges=bounds)
        assert core.task_config == before
        np.testing.assert_array_equal(session.mj_data.qpos, qpos)
    finally:
        session.close()


@pytest.mark.parametrize("bounds", [[[10., 30.]], [[170., 190.]], [[-170., -100.]]])
def test_target_reads_declared_bounds_in_degrees(*, bounds: list[list[float]]) -> None:
    core = SimpleNamespace(task_config={"initial_state": [["in", "wiper_0", "home"]],
                                       "regions": {"home": {"yaw_ranges": bounds}}})
    primitive = SimpleNamespace(session=SimpleNamespace(
        env=SimpleNamespace(unwrapped=SimpleNamespace(_object_centric_env=core))))
    rotation = Rotation.from_quat(FloorPrimitives.place_target_orientation(primitive))
    expected = Rotation.from_euler("z", np.radians(sum(bounds[0]) / 2))
    assert (rotation.inv() * expected).magnitude() < 1e-12
    assert SweepRegions.yaw_matches(yaw=rotation.as_euler("xyz")[2], ranges=bounds)

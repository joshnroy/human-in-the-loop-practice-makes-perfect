"""The optional no-drive zone a scene may put on Tossing3D's toss planning.

Absent, nothing changes; present, a stand outside it is refused before planning and the
far band's floor follows the zone's edge rather than the barrier-measured line.
"""

import numpy as np

from hitl_pmp.environments.tossing3d.parameter_feasibility import TossParameterFeasibility
from hitl_pmp.environments.tossing3d.toss_direction import TossStandoffBand
from hitl_pmp.environments.tossing3d.types import TossFeasibilityGeometry
from hitl_pmp.environments.tossing3d.wide_long_range_proposal import (
    FAR_STAND_X_LIMIT_M,
    WideLongRangeTossProposal,
)

ZONE = (-1.06, -2.06, 0.71, 2.06)


def geometry(*, bin_x: float, zone: tuple[float, float, float, float] | None):
    return TossFeasibilityGeometry(
        robot_pose=(0.0, 0.0, 0.0),
        robot_size=(0.55, 0.55),
        bin_pose=(bin_x, 0.0, 0.0),
        obstacles=(),
        sampling_x_bounds=(-5.0, 5.0),
        sampling_y_bounds=(-5.0, 5.0),
        drive_zone=zone,
    )


def test_without_a_zone_the_gate_accepts_a_stand_past_the_kitchen_line():
    params = np.array([1.25, 0.0])
    assert (
        TossParameterFeasibility.rejection_reason_from_geometry(
            geometry=geometry(bin_x=2.05, zone=None), params=params
        )
        is None
    )


def test_a_zone_refuses_a_stand_outside_it_and_accepts_one_inside():
    zoned = geometry(bin_x=2.05, zone=ZONE)
    outside = TossParameterFeasibility.rejection_reason_from_geometry(
        geometry=zoned, params=np.array([1.25, 0.0])
    )
    inside = TossParameterFeasibility.rejection_reason_from_geometry(
        geometry=zoned, params=np.array([1.5, 0.0])
    )
    assert outside == "outside_drive_zone"
    assert inside is None


def test_far_standoff_floor_defaults_to_the_measured_barrier_line():
    assert WideLongRangeTossProposal.far_standoff_bounds(bin_x=3.0)[0] == 3.0 - FAR_STAND_X_LIMIT_M
    assert WideLongRangeTossProposal.far_standoff_bounds(bin_x=3.0, stand_x_limit=0.71)[0] == (
        3.0 - 0.71
    )


def test_far_band_floor_follows_the_zone_edge():
    TossStandoffBand.clear_cache()
    low, high = TossStandoffBand.far_bounds(geometry=geometry(bin_x=2.05, zone=ZONE))
    assert np.isclose(low, 2.05 - ZONE[2])
    assert high == 2.6

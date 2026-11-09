import math

import pytest

from hitl_pmp.environments.tossing3d.real.frame import (
    BIN_SPOTS,
    CUBE_SPOT,
    ROOM_FRAME,
    RealRoomFrame,
)
from hitl_pmp.environments.tossing3d.sides import Tossing3DSide
from hitl_pmp.environments.tossing3d.wide_long_range_proposal import REAL_TOSS_PARAMETERS


def test_map_to_domain_and_back_is_the_identity() -> None:
    for x, y, yaw in ((0.24, 1.40, 0.3), (-0.30, -0.78, -2.0), (1.7, -1.5, 3.1)):
        dx, dy = ROOM_FRAME.to_domain(x=x, y=y)
        back = ROOM_FRAME.to_map(x=dx, y=dy)
        assert back == pytest.approx((x, y), abs=1e-12)
        turned = ROOM_FRAME.yaw_to_map(yaw=ROOM_FRAME.yaw_to_domain(yaw=yaw))
        assert math.isclose(math.remainder(turned - yaw, math.tau), 0.0, abs_tol=1e-12)


def test_the_barrier_centre_line_is_domain_x_zero_and_the_robot_side_is_negative() -> None:
    # The tape's robot-side edge is y = 0.349 + 0.0267 x; the robot works on map -y.
    for x in (-1.5, 0.0, 1.7):
        edge = (x, 0.349 + 0.0267 * x)
        assert ROOM_FRAME.to_domain(x=edge[0], y=edge[1])[0] == pytest.approx(-0.025, abs=2e-3)
    assert ROOM_FRAME.to_domain(x=0.0, y=-1.0)[0] < 0.0
    assert ROOM_FRAME.to_domain(x=0.0, y=1.0)[0] > 0.0


def test_a_map_direction_into_the_far_side_is_domain_plus_x() -> None:
    away_from_robot = math.atan2(-ROOM_FRAME.robot_side[1], -ROOM_FRAME.robot_side[0])
    assert ROOM_FRAME.yaw_to_domain(yaw=away_from_robot) == pytest.approx(0.0, abs=1e-12)


def test_the_two_bin_spots_land_on_their_named_sides_in_the_domain_frame() -> None:
    robot_spot = BIN_SPOTS[Tossing3DSide.ROBOT]
    far_spot = BIN_SPOTS[Tossing3DSide.OPPOSITE]
    assert robot_spot.center == (-0.30, -1.00)
    assert far_spot.center == (0.245, 1.27)
    assert ROOM_FRAME.to_domain(x=robot_spot.center[0], y=robot_spot.center[1])[0] < 0.0
    assert ROOM_FRAME.to_domain(x=far_spot.center[0], y=far_spot.center[1])[0] > 0.0
    assert ROOM_FRAME.to_domain(x=CUBE_SPOT[0], y=CUBE_SPOT[1])[0] < 0.0


def test_a_frame_needs_a_unit_robot_side_direction() -> None:
    with pytest.raises(ValueError, match="unit"):
        RealRoomFrame(barrier_center=(0.0, 0.0), robot_side=(0.0, -2.0))


def test_the_final_layout_throws_from_one_distance_at_both_bins() -> None:
    """2026-11-05 (Josh): the robot throws from 1.20 m only, and searches over speed and
    release alone. From the closest stand the old far spot allowed (1.37 m), even the
    fastest throw reached only the near bin's near rim. The far bin moved to
    (0.245, 1.07): at 1.19 the base turning into its stand came 4.2 cm from the tape."""
    low, high = REAL_TOSS_PARAMETERS.standoff_bounds
    assert low == pytest.approx(1.40, abs=0.006) and high == pytest.approx(1.40, abs=0.006)
    assert REAL_TOSS_PARAMETERS.speed_bounds == (100.0, 140.0)
    assert REAL_TOSS_PARAMETERS.release_ms_bounds == (500.0, 850.0)
    assert BIN_SPOTS[Tossing3DSide.OPPOSITE].center == pytest.approx((0.245, 1.27))

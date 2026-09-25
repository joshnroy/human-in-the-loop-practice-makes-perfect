"""The robot-side receiver region is a measured block of the south-wall band.

Measured 2026-09-24 with the real base motion planner and live picks. Every bin
centre on a 0.15 m grid over x in [-0.35, 0.40], y in [-1.90, -1.30] was placeable,
picked the cube from its spawn strip (3 seeds x 4 robot poses, robot clear of the
bin) and had a plannable stand direction at every standoff in [1.25, 2.60]. The
edges come from a cube lying in the bin 0.072 m off centre, which the wider band
x in [-0.35, 0.85], y in [-2.80, -1.30] refused to pick toward the south wall
(y <= -2.10), the barrier (x >= 0.40) and the south-west corner (y <= -2.00 at
x = -0.33): 98/160 picked. Each edge keeps 0.10 m to the first refusal; the west edge
is kindergarden's placement limit (x > -0.338), the north edge the probed one.
"""

from hitl_pmp.environments.tossing3d.sides import BIN_RESET_REGION_BY_SIDE, Tossing3DSide


def test_robot_side_receiver_region_is_the_measured_south_band_block() -> None:
    region = BIN_RESET_REGION_BY_SIDE[Tossing3DSide.ROBOT]
    assert region.ranges == ((-0.33, -1.9, 0.3, -1.3),)
    assert region.yaw_ranges == ((180, 180),)


def test_opposite_side_receiver_region_mirrors_the_graded_task_region() -> None:
    """The reset-side far region mirrors the installed task's graded receiver
    spawn range (kindergarden's [1.48, 3.42] union of the original near support
    and the #191 far extension), so practice resets and evaluation sample the
    same distance ramp."""
    assert BIN_RESET_REGION_BY_SIDE[Tossing3DSide.OPPOSITE].ranges == ((1.48, -2.3, 3.42, 2.3),)

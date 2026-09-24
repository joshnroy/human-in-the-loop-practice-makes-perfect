"""The robot-side receiver region must keep marginal cubes graspable.

Measured 2026-09-22 at the live pick controller with the trap-2 stuck arrangement
(cube 0.073 m off bin center, face gap 0.052 m): every probed bin position with
x <= -1.0 -- including mid-room y, far from any wall plane -- REFUSED the grasp
(20/20, "No collision-free cube grasp", 0 steps), so the old robot-side range
x in [-2.3, -1.48] sat entirely inside the refusing zone. The eastward-extent
mapping then found the maximal grasp-safe RECTANGLE: every probed point inside
x in [-0.9, 0.2], y in [-1.0, 1.5] accepted and lifted (14/14 across the two
probe rounds), while y = -1.5 refused at x = 0.6 and y = +-1.9 refused at
x = -0.5 (0/2). The y = 0 corridor stays graspable east to at least x = 1.1 but
the region is north-south asymmetric beyond the rectangle, so only the rectangle
is encoded.
"""

from hitl_pmp.environments.tossing3d.sides import BIN_RESET_REGION_BY_SIDE, Tossing3DSide


def test_robot_side_receiver_region_is_the_measured_maximal_grasp_safe_rectangle() -> None:
    ranges = BIN_RESET_REGION_BY_SIDE[Tossing3DSide.ROBOT].ranges
    assert len(ranges) == 1
    x_min, y_min, x_max, y_max = ranges[0]
    # West edge at the measured -1.0 refuse / -0.9 accept boundary.
    assert x_min == -0.9
    # East edge inside the grasp-safe span and clear of the cube spawn strip
    # (x in [0.54, 0.71]) and the barrier at x = 1.25.
    assert x_max == 0.2
    # South edge pulled in to y = -1.0: y = -1.5 refused at x = 0.6.
    assert (y_min, y_max) == (-1.0, 1.5)


def test_opposite_side_receiver_region_mirrors_the_graded_task_region() -> None:
    """The reset-side far region mirrors the installed task's graded receiver
    spawn range (kindergarden's [1.48, 3.42] union of the original near support
    and the #191 far extension), so practice resets and evaluation sample the
    same distance ramp."""
    assert BIN_RESET_REGION_BY_SIDE[Tossing3DSide.OPPOSITE].ranges == ((1.48, -2.3, 3.42, 2.3),)

"""A reset bin never overlaps the robot and never sits near the cube's spawn region.

Both rules are occupancy in the reset sampler: the bin's centre ranges are the
region minus the forbidden zones, so the placement stays uniform over the valid
free space instead of being rejected after the fact. A check after every reset
raises if either rule is ever violated anyway.

The 0.20 m footprint gap to `blocks_init_region` is the fine pick sweep of
2026-09-24: bin centres every 0.05 m over x in [-0.10, 0.20], y in [-0.6, 0.6],
3 seeds x 3 robot poses. With the robot clear of the bin, every pick at a gap of
>= 0.20 m succeeded (998/998), while failures reached a 0.182 m gap. Three robot
poses are not every approach, so this is a measured floor, not a proof.
"""

import importlib.util

import numpy as np
import pytest

from hitl_pmp.environments.tossing3d.bin_placement import (
    BinPlacementRules,
    BinPlacementViolationError,
)

needs_kinder = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None or importlib.util.find_spec("kinder_models") is None,
    reason="KINDER is an optional extra",
)

SPAWN = ((0.5, -0.25, 0.75, 0.25),)  # Tossing3D-o1.json's blocks_init_region
BIN_HALF = (0.15, 0.15)
OLD_ROBOT_SIDE_RECTANGLE = (-0.9, -1.0, 0.2, 1.5)
RESET_ROBOT_AABB = (-0.32, -0.32, 0.23, 0.23)  # the scene's reset pose, ~(-0.045, -0.043)


def _contains(*, ranges, point) -> bool:
    x, y = point
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in ranges)


def _area(*, ranges) -> float:
    return sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in ranges)


def test_the_cube_spawn_gap_is_the_measured_floor() -> None:
    assert BinPlacementRules.CUBE_SPAWN_MIN_GAP_M == 0.20


def test_subtracting_a_hole_leaves_disjoint_rectangles_covering_the_rest() -> None:
    region = ((0.0, 0.0, 4.0, 3.0),)
    hole = (1.0, 1.0, 2.0, 2.0)
    pieces = BinPlacementRules.subtract(ranges=region, holes=(hole,))
    assert _area(ranges=pieces) == pytest.approx(12.0 - 1.0)
    rng = np.random.default_rng(0)
    for x, y in rng.uniform((0, 0), (4, 3), size=(2000, 2)):
        inside_hole = 1.0 < x < 2.0 and 1.0 < y < 2.0
        hits = sum(x0 < x < x1 and y0 < y < y1 for x0, y0, x1, y1 in pieces)
        assert hits == (0 if inside_hole else 1)


def test_a_hole_outside_the_region_changes_nothing_and_a_covering_hole_empties_it() -> None:
    region = ((0.0, 0.0, 1.0, 1.0),)
    assert BinPlacementRules.subtract(ranges=region, holes=((5.0, 5.0, 6.0, 6.0),)) == region
    assert BinPlacementRules.subtract(ranges=region, holes=((-1.0, -1.0, 2.0, 2.0),)) == ()


def test_the_known_refused_bin_beside_the_cube_is_no_longer_placeable() -> None:
    """Bin (0.15, 0.25) refused the grasp on 10/10 seeds in the 180/270 probe. Even on
    the old rectangle, the rules alone now forbid it."""
    free = BinPlacementRules.free_centre_ranges(
        ranges=(OLD_ROBOT_SIDE_RECTANGLE,),
        robot_aabb=(5.0, 5.0, 5.1, 5.1),
        cube_spawn=SPAWN,
        bin_half=BIN_HALF,
        clearance=0.005,
    )
    assert not _contains(ranges=free, point=(0.15, 0.25))
    assert _contains(ranges=free, point=(-0.5, 0.25))


def test_a_bin_inside_the_robots_footprint_is_no_longer_placeable() -> None:
    """The robot parked at (-0.2, -1.2) after a toss: KINDER's sampler used to place the
    bin inside it (359/359 such placements refused the pick)."""
    robot = BinPlacementRules.aabb(center=(-0.2, -1.2), size=(0.55, 0.55), yaw=np.pi / 2)
    free = BinPlacementRules.free_centre_ranges(
        ranges=((-0.33, -1.9, 0.3, -1.3),),
        robot_aabb=robot,
        cube_spawn=SPAWN,
        bin_half=BIN_HALF,
        clearance=0.005,
    )
    for point in ((-0.2, -1.35), (-0.2, -1.6), (0.05, -1.45)):
        assert not _contains(ranges=free, point=point)
    assert _contains(ranges=free, point=(-0.2, -1.7))
    assert _contains(ranges=free, point=(0.27, -1.35))


def test_no_robot_side_toss_stand_leaves_the_block_without_a_free_placement() -> None:
    """With practice resets robot-side only, the robot stands where a toss at a block
    bin left it: `bin - d (cos, sin)(pi + direction)`. None of those stands, for any
    corner bin, direction or standoff, covers the whole block. (A robot parked where a
    far-side toss stood can, which is why practice must not reset to the far side.)"""
    block = (-0.33, -1.9, 0.3, -1.3)
    for bx, by in (
        (block[0], block[1]),
        (block[0], block[3]),
        (block[2], block[1]),
        (block[2], block[3]),
        (0.0, -1.6),
    ):
        for direction in (0, 90, 180, 270):
            yaw = np.pi + np.radians(direction)
            for d in np.arange(1.25, 2.6001, 0.05):
                stand = (bx - d * np.cos(yaw), by - d * np.sin(yaw))
                robot = BinPlacementRules.aabb(center=stand, size=(0.55, 0.55), yaw=yaw)
                free = BinPlacementRules.free_centre_ranges(
                    ranges=(block,),
                    robot_aabb=robot,
                    cube_spawn=SPAWN,
                    bin_half=BIN_HALF,
                    clearance=0.005,
                )
                assert free, (bx, by, direction, d)
    far_stand = BinPlacementRules.aabb(center=(0.0, -1.6), size=(0.55, 0.55), yaw=0.0)
    assert not BinPlacementRules.free_centre_ranges(
        ranges=(block,),
        robot_aabb=far_stand,
        cube_spawn=SPAWN,
        bin_half=BIN_HALF,
        clearance=0.005,
    )


def test_robot_aabb_covers_the_rotated_footprint() -> None:
    x0, y0, x1, y1 = BinPlacementRules.aabb(center=(0.0, 0.0), size=(0.55, 0.55), yaw=np.pi / 4)
    half = 0.55 / np.sqrt(2)
    assert (x0, y0, x1, y1) == pytest.approx((-half, -half, half, half))


def test_post_reset_check_raises_on_a_bin_overlapping_the_robot() -> None:
    with pytest.raises(BinPlacementViolationError, match="overlaps the robot"):
        BinPlacementRules.check(
            bin_aabb=(-0.35, -1.5, -0.05, -1.2),
            robot_aabb=(-0.475, -1.475, 0.075, -0.925),
            cube_spawn=SPAWN,
            context="constructed",
        )


def test_post_reset_check_raises_on_a_bin_too_close_to_the_cube_spawn_region() -> None:
    with pytest.raises(BinPlacementViolationError, match="cube spawn") as caught:
        BinPlacementRules.check(
            bin_aabb=(0.05, 0.1, 0.35, 0.4),  # 0.15 m from the spawn region
            robot_aabb=(5.0, 5.0, 5.1, 5.1),
            cube_spawn=SPAWN,
            context="constructed",
        )
    assert "constructed" in str(caught.value)


def test_post_reset_check_accepts_a_valid_bin() -> None:
    BinPlacementRules.check(
        bin_aabb=(0.35, -2.0, 0.65, -1.7),
        robot_aabb=(-0.3, -0.3, 0.25, 0.25),
        cube_spawn=SPAWN,
        context="constructed",
    )


# --- live resets --------------------------------------------------------------------

ROBOT_POSES = {
    "reset": None,
    # Standing north of a block bin after tossing into it.
    "north_stand": (0.0, -0.3, float(-np.pi / 2)),
    # Standing west of a near-barrier far bin, just above the block.
    "far_stand": (0.1, -1.1, 0.0),
}


def _park_robot(*, env, pose) -> None:
    if pose is None:
        return
    from kinder.envs.dynamic3d.object_types import MujocoTidyBotRobotObjectType

    backend = env.backend()
    snapshot = backend.snapshot()
    (robot,) = snapshot.get_objects(MujocoTidyBotRobotObjectType)
    for feature, value in zip(("pos_base_x", "pos_base_y", "pos_base_rot"), pose, strict=True):
        snapshot.set(robot, feature, value)
    backend.restore(snapshot=snapshot)


@needs_kinder
@pytest.mark.parametrize("pose_name", sorted(ROBOT_POSES))
def test_a_robot_side_reset_lands_in_the_band_clear_of_robot_and_cube(*, pose_name: str) -> None:
    from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
    from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend
    from hitl_pmp.environments.tossing3d.sides import BIN_RESET_REGION_BY_SIDE, Tossing3DSide

    band = BIN_RESET_REGION_BY_SIDE[Tossing3DSide.ROBOT].ranges[0]
    env = Tossing3DEnvironment()
    try:
        env.reset_to_seed(seed=2026092401)
        _park_robot(env=env, pose=ROBOT_POSES[pose_name])
        for _ in range(6):
            assert env.reset_movables(destination="robot_side")
            geometry = KinderBackend.toss_feasibility_geometry(snapshot=env.backend().snapshot())
            bx, by, _ = geometry.bin_pose
            assert band[0] - 1e-6 <= bx <= band[2] + 1e-6
            assert band[1] - 1e-6 <= by <= band[3] + 1e-6
            robot = BinPlacementRules.aabb(
                center=geometry.robot_pose[:2], size=geometry.robot_size, yaw=geometry.robot_pose[2]
            )
            bin_box = BinPlacementRules.aabb(
                center=(bx, by), size=(0.3, 0.3), yaw=geometry.bin_pose[2]
            )
            BinPlacementRules.check(
                bin_aabb=bin_box, robot_aabb=robot, cube_spawn=SPAWN, context=pose_name
            )
    finally:
        env.close()


@needs_kinder
@pytest.mark.parametrize("pose_name", sorted(ROBOT_POSES))
@pytest.mark.parametrize("seed", [2026092401, 2026092403])
def test_a_band_reset_is_followed_by_a_successful_pick(*, pose_name: str, seed: int) -> None:
    from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
    from hitl_pmp.environments.tossing3d.predicates import HOLDING

    env = Tossing3DEnvironment()
    try:
        env.reset_to_seed(seed=seed)
        _park_robot(env=env, pose=ROBOT_POSES[pose_name])
        assert env.reset_movables(destination="robot_side")
        action = np.zeros(5, dtype=float)
        action[0] = env.pick_cube_id
        picked = env.take_action(action=action)
        assert HOLDING.holds(picked, (env.robot, env.cube)), env.last_skill_error()
    finally:
        env.close()


def _bin_scene():
    from types import SimpleNamespace
    from unittest.mock import Mock

    return SimpleNamespace(
        task_config={"objects": {"bin": {"bin_0": {}}}},
        _objects_dict={
            "bin_0": SimpleNamespace(
                REGISTERED_NAME="bin",
                get_bounding_box_from_config=Mock(return_value=[-0.15, -0.15, 0, 0.15, 0.15, 0.2]),
            )
        },
    )


def test_the_reset_region_handed_to_kinder_is_the_free_space_with_its_yaw() -> None:
    from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend

    region = {"target": "ground", "ranges": [[-0.33, -1.9, 0.3, -1.3]], "yaw_ranges": [[180, 180]]}
    robot = BinPlacementRules.aabb(center=(0.1, -1.1), size=(0.55, 0.55), yaw=0.0)
    cleared = KinderBackend()._bin_region_clear_of_robot_and_cube(  # noqa: SLF001
        object_centric=_bin_scene(), region=region, robot_aabb=robot, cube_spawn=SPAWN
    )
    assert cleared["target"] == "ground"
    assert len(cleared["yaw_ranges"]) == len(cleared["ranges"]) >= 1
    assert all(yaw == [180, 180] for yaw in cleared["yaw_ranges"])
    assert not _contains(ranges=cleared["ranges"], point=(0.1, -1.4))
    assert _contains(ranges=cleared["ranges"], point=(0.1, -1.8))


def test_a_robot_covering_the_whole_region_raises_instead_of_placing() -> None:
    from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend

    region = {"target": "ground", "ranges": [[-0.33, -1.9, 0.3, -1.3]], "yaw_ranges": [[180, 180]]}
    robot = BinPlacementRules.aabb(center=(0.0, -1.6), size=(0.55, 0.55), yaw=0.0)
    with pytest.raises(BinPlacementViolationError, match="no bin placement"):
        KinderBackend()._bin_region_clear_of_robot_and_cube(  # noqa: SLF001
            object_centric=_bin_scene(), region=region, robot_aabb=robot, cube_spawn=SPAWN
        )

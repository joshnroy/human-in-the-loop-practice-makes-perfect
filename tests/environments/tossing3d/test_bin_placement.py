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
    """After a toss at a block bin the robot stands at `bin - d (cos, sin)(pi +
    direction)`. None of those stands, for any corner bin, direction or standoff, covers
    the whole block. (A robot parked where a far-side toss stood can, which is why
    `reset_cube_and_bin` moves a blocking robot clear first -- see
    `test_either_reset_after_a_far_side_toss_stand_is_placeable`.)"""
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


def test_the_scene_start_and_spawn_strip_picks_never_empty_the_block() -> None:
    """The other places a practising robot stands when a reset can come: the scene's
    reset pose, and the pick stand 0.55 m (`PickCubeController.TARGET_DISTANCE`) from a
    cube anywhere in its spawn region, at any approach angle."""
    block = (-0.33, -1.9, 0.3, -1.3)
    stands = [(-0.045, -0.043, 0.08)]
    for cx in np.linspace(0.5, 0.75, 6):
        for cy in np.linspace(-0.25, 0.25, 6):
            for angle in np.radians(np.arange(0, 360, 15)):
                stands.append((cx - 0.55 * np.cos(angle), cy - 0.55 * np.sin(angle), angle))
    for x, y, yaw in stands:
        robot = BinPlacementRules.aabb(center=(x, y), size=(0.55, 0.55), yaw=yaw)
        assert BinPlacementRules.free_centre_ranges(
            ranges=(block,), robot_aabb=robot, cube_spawn=SPAWN, bin_half=BIN_HALF, clearance=0.005
        ), (x, y, yaw)


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


@needs_kinder
@pytest.mark.parametrize("seed", [2026092401, 2026092403])
def test_a_far_side_practice_reset_is_placeable_and_tossable(*, seed: int) -> None:
    """The far-side reset destination the planner may now choose: the bin lands in
    the far region beyond the barrier, the cube is picked from its spawn strip, and a
    toss drawn from the far bin's own standoff band (#365, corner-aware) runs to
    completion. Scoring is not asserted -- that is competence, not feasibility."""
    from hitl_pmp.core.method.types import GroundSkill
    from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
    from hitl_pmp.environments.tossing3d.predicates import BIN_AT_SIDE, HOLDING
    from hitl_pmp.environments.tossing3d.sides import (
        BIN_RESET_REGION_BY_SIDE,
        Tossing3DSide,
        Tossing3DSides,
    )
    from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills
    from hitl_pmp.environments.tossing3d.toss import Tossing3DToss

    far = BIN_RESET_REGION_BY_SIDE[Tossing3DSide.OPPOSITE].ranges[0]
    env = Tossing3DEnvironment()
    try:
        env.reset_to_seed(seed=seed)
        assert env.reset_movables(destination="opposite_side")
        placed = env.get_current_state()
        assert BIN_AT_SIDE.holds(placed, (env.bin, env.barrier, Tossing3DSides.opposite))
        bx = placed.get(obj=env.bin, feature_name="x")
        by = placed.get(obj=env.bin, feature_name="y")
        assert far[0] - 1e-3 <= bx <= far[2] + 1e-3
        assert far[1] - 1e-3 <= by <= far[3] + 1e-3
        action = np.zeros(5, dtype=float)
        action[0] = env.pick_cube_id
        picked = env.take_action(action=action)
        assert HOLDING.holds(picked, (env.robot, env.cube)), env.last_skill_error()
        toss = GroundSkill(
            skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
            objects=(env.robot, env.bin, env.cube, env.barrier, Tossing3DSides.opposite),
        )
        low, high = Tossing3DToss.standoff_bounds(ground_skill=toss, state=picked)
        assert low < high
        rng = np.random.default_rng(seed)
        for _ in range(50):
            params = Tossing3DToss.sample_params_at_state(rng=rng, ground_skill=toss, state=picked)
            if Tossing3DToss.rejection_reason(state=picked, params=params) is None:
                break
        else:
            pytest.fail("no far-band standoff had a plannable stand direction")
        assert low - 1e-9 <= params[0] <= high + 1e-9
        env.take_action(action=Tossing3DToss.compute_action(params=params, state=picked))
        assert env.last_skill_error() is None, params
        assert sum(env.last_controller_steps()) > 0
    finally:
        env.close()


# The robot pose logged when a 50-cycle run crashed at cycle 45: its footprint AABB was
# (-0.363, -1.786, 0.411, -1.012), a 0.55 m square turned 45 degrees and centred inside
# the robot-side block, so no bin centre in the block was clear of it.
LOGGED_BLOCKING_POSE = (0.024, -1.399, float(np.pi / 4))
# Where a far-side toss can leave the robot: west of a near-barrier far bin, facing
# east, centred on the block (`test_no_robot_side_toss_stand_...`'s `far_stand`).
FAR_TOSS_STAND = (0.0, -1.6, 0.0)


def _robot_pose(*, env) -> tuple[float, float, float]:
    """Read from the live simulator: `_park_robot` restores the backend directly, so
    the environment's adopted state does not see the parked pose."""
    from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend

    geometry = KinderBackend.toss_feasibility_geometry(snapshot=env.backend().snapshot())
    return tuple(float(value) for value in geometry.robot_pose)


def _assert_placed_clear(*, env, destination: str) -> None:
    from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend
    from hitl_pmp.environments.tossing3d.sides import BIN_RESET_REGION_BY_SIDE, Tossing3DSide

    region = BIN_RESET_REGION_BY_SIDE[Tossing3DSide(destination)].ranges[0]
    geometry = KinderBackend.toss_feasibility_geometry(snapshot=env.backend().snapshot())
    bx, by, _ = geometry.bin_pose
    assert region[0] - 1e-3 <= bx <= region[2] + 1e-3
    assert region[1] - 1e-3 <= by <= region[3] + 1e-3
    robot = BinPlacementRules.aabb(
        center=geometry.robot_pose[:2], size=geometry.robot_size, yaw=geometry.robot_pose[2]
    )
    BinPlacementRules.check(
        bin_aabb=BinPlacementRules.aabb(center=(bx, by), size=(0.3, 0.3), yaw=geometry.bin_pose[2]),
        robot_aabb=robot,
        cube_spawn=SPAWN,
        context=destination,
    )


@needs_kinder
def test_a_reset_with_the_robot_standing_in_the_block_moves_the_robot_clear_first() -> None:
    """The logged crash: the robot stood where it covered the whole robot-side block, so
    `reset_cube_and_bin` found no clear bin centre and raised. A person resetting the
    scene has the robot move out of the way first; the reset now sends the robot back to
    its scene-start pose (clear of the block, measured) and then places the bin."""
    from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment

    env = Tossing3DEnvironment()
    try:
        env.reset_to_seed(seed=2026092401)
        start = _robot_pose(env=env)
        _park_robot(env=env, pose=LOGGED_BLOCKING_POSE)
        assert env.reset_movables(destination="robot_side")
        _assert_placed_clear(env=env, destination="robot_side")
        assert _robot_pose(env=env) == pytest.approx(start, abs=1e-3)
        # The moved robot is a working robot, not a teleported ghost: it picks from there.
        from hitl_pmp.environments.tossing3d.predicates import HOLDING

        action = np.zeros(5, dtype=float)
        action[0] = env.pick_cube_id
        picked = env.take_action(action=action)
        assert HOLDING.holds(picked, (env.robot, env.cube)), env.last_skill_error()
    finally:
        env.close()


@needs_kinder
@pytest.mark.parametrize("destination", ["robot_side", "opposite_side"])
def test_either_reset_after_a_far_side_toss_stand_is_placeable(*, destination: str) -> None:
    """The sequence reopened by letting practice reset far-side: a far-side reset, then
    the robot standing where a far-side toss leaves it, then the planner asks for
    either reset. Both must place the bin clear of the robot rather than raise."""
    from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment

    env = Tossing3DEnvironment()
    try:
        env.reset_to_seed(seed=2026092403)
        assert env.reset_movables(destination="opposite_side")
        _park_robot(env=env, pose=FAR_TOSS_STAND)
        assert env.reset_movables(destination=destination)
        _assert_placed_clear(env=env, destination=destination)
    finally:
        env.close()


@needs_kinder
def test_a_reset_that_has_room_leaves_the_robot_where_it_stands() -> None:
    """Moving the robot is a fallback for a genuinely blocked region, never a default:
    a reset with a free placement leaves the robot exactly where it was."""
    from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment

    env = Tossing3DEnvironment()
    try:
        env.reset_to_seed(seed=2026092401)
        _park_robot(env=env, pose=ROBOT_POSES["north_stand"])
        before = _robot_pose(env=env)
        assert env.reset_movables(destination="robot_side")
        assert _robot_pose(env=env) == pytest.approx(before, abs=1e-6)
    finally:
        env.close()

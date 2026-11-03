"""The kitchen's tapered bin, scored by its whole interior, and the flat mat-edge sides."""

import importlib.util

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.kitchen_recycling3d.bin_geometry import TaperedBin, TaperedBinCollision
from hitl_pmp.environments.kitchen_recycling3d.domain import KitchenConfig
from hitl_pmp.environments.kitchen_recycling3d.environment import KitchenRecycling3DEnvironment
from hitl_pmp.environments.kitchen_recycling3d.scene import scene_config
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import (
    BIN_ON_OPPOSITE_SIDE,
    BIN_ON_ROBOT_SIDE,
    CUBE_ON_OPPOSITE_SIDE,
    CUBE_ON_ROBOT_SIDE,
)
from hitl_pmp.environments.tossing3d.types import Tossing3DObservation

from .test_kitchen import config

CUBE_HALF = 0.025
RESTING_Z = 0.005 + CUBE_HALF


def bin_() -> TaperedBin:
    return TaperedBin.of(config=config())


def test_the_bin_is_the_real_rubbermaid_bin():
    geometry = bin_()
    assert (geometry.length, geometry.width, geometry.height) == (0.365, 0.26, 0.384)
    assert (geometry.base_length, geometry.base_width, geometry.wall) == (0.29, 0.19, 0.005)


def test_a_cube_twelve_centimetres_off_centre_inside_the_walls_scores():
    assert bin_().contains_local(point=(0.12, 0.0, RESTING_Z))
    assert bin_().contains_local(point=(0.0, 0.08, RESTING_Z))
    assert bin_().contains_local(point=(0.12, 0.08, RESTING_Z))


def test_cubes_against_each_sloped_wall_score():
    geometry = bin_()
    for z in (RESTING_Z, 0.15, 0.3):
        for axis in (0, 1):
            for sign in (1, -1):
                point = [0.0, 0.0, z]
                point[axis] = sign * (geometry.inner_half(axis=axis, z=z) - CUBE_HALF)
                assert geometry.contains_local(point=point)


def test_a_cube_resting_outside_the_bin_never_scores():
    geometry = bin_()
    # Beside the narrow base, under the overhanging rim: inside the rim's box.
    beside = geometry.base_length / 2 + CUBE_HALF + 0.001
    assert beside < geometry.inner_half(axis=0, z=geometry.height)
    assert not geometry.contains_local(point=(beside, 0.0, CUBE_HALF))
    assert not geometry.contains_local(point=(0.0, geometry.base_width / 2 + 0.026, CUBE_HALF))
    # Above the rim, and in the wall itself.
    assert not geometry.contains_local(point=(0.0, 0.0, geometry.height + 0.01))
    assert not geometry.contains_local(point=(geometry.outer_half(axis=0, z=0.2), 0.0, 0.2))


def test_a_yawed_bin_scores_in_its_own_frame():
    geometry = bin_()
    quarter = (0.0, 0.0, 0.7071067811865476, 0.7071067811865476)
    # The bin's x now points along world y.
    point = (2.05, 0.12, RESTING_Z)
    assert geometry.contains(point=point, bin_position=(2.05, 0, 0), bin_quaternion_xyzw=quarter)
    assert not geometry.contains(
        point=(2.05 + 0.12, 0, RESTING_Z),
        bin_position=(2.05, 0, 0),
        bin_quaternion_xyzw=quarter,
    )


def test_the_scored_region_is_sized_from_the_bin_not_ten_centimetres():
    scene = scene_config(config=config(), seed=0)
    ranges = scene["regions"]["blocks_goal_region"]["ranges"][0]
    assert ranges == pytest.approx(list(bin_().interior_box()))
    assert ranges[3] > 0.17


def test_the_panels_close_the_walls_and_stay_inside_the_rim_footprint():
    geometry = bin_()
    panels = geometry.panels()
    assert len(panels) == 1 + 4 * TaperedBin.STRIPS
    assert sum(not panel.wall for panel in panels) == 1


def test_the_scene_builds_the_tapered_bin():
    scene = scene_config(config=config(), seed=0)
    (bin_config,) = scene["objects"]["kitchen_tapered_bin"].values()
    assert (bin_config["base_length"], bin_config["base_width"]) == (0.29, 0.19)
    assert "bin" not in scene["objects"]


def observation(*, cube_x: float, bin_x: float = 2.05) -> Tossing3DObservation:
    return Tossing3DObservation(
        features={
            "robot": {
                "pos_base_x": 0.0,
                "pos_base_y": 0.0,
                "pos_base_rot": 0.0,
                "pos_gripper": 0.0,
            },
            "cube_0": {
                "x": cube_x,
                "y": 0.0,
                "z": CUBE_HALF,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
                "bb_x": 0.05,
                "bb_y": 0.05,
                "bb_z": 0.05,
            },
            "bin_0": {"x": bin_x, "y": 0.0, "z": 0.0},
            "cuboid_barrier": {"x": 1.15, "y": 0.0, "z": 0.0005, "bb_x": 0.016},
        },
        goal_region=(0.0,) * 6,
        solved=False,
    )


def kitchen_state(*, cube_x: float, bin_x: float = 2.05):
    env = KitchenRecycling3DEnvironment(kitchen_config=config())
    built = env.build_state(
        observation=observation(cube_x=cube_x, bin_x=bin_x), seed=0, steps_taken=0
    )
    return env, built


def sides(*, env, state, obj, robot, opposite):
    return (
        robot.holds(state, (obj, env.barrier)),
        opposite.holds(state, (obj, env.barrier)),
    )


@pytest.mark.parametrize("cube_x", [1.126, 1.13, 1.14, 1.15, 1.16, 1.175, 1.18])
def test_a_cube_straddling_the_mat_edge_is_on_the_mat_side(*, cube_x):
    env, state = kitchen_state(cube_x=cube_x)
    assert sides(
        env=env, state=state, obj=env.cube, robot=CUBE_ON_ROBOT_SIDE, opposite=CUBE_ON_OPPOSITE_SIDE
    ) == (False, True)


def test_a_cube_entirely_on_the_kitchen_floor_is_on_the_robot_side():
    env, state = kitchen_state(cube_x=1.15 - CUBE_HALF)
    assert sides(
        env=env, state=state, obj=env.cube, robot=CUBE_ON_ROBOT_SIDE, opposite=CUBE_ON_OPPOSITE_SIDE
    ) == (True, False)


@pytest.mark.parametrize(
    ("bin_x", "expected"),
    [
        (0.65, (True, False)),
        (1.15 - 0.1825, (True, False)),
        (1.0, (False, True)),
        (2.05, (False, True)),
    ],
)
def test_the_bin_is_on_the_robot_side_only_with_its_whole_footprint_off_the_mat(*, bin_x, expected):
    env, state = kitchen_state(cube_x=0.5, bin_x=bin_x)
    assert (
        sides(
            env=env,
            state=state,
            obj=env.bin,
            robot=BIN_ON_ROBOT_SIDE,
            opposite=BIN_ON_OPPOSITE_SIDE,
        )
        == expected
    )


def test_tossing3d_keeps_its_open_halfspace_sides():
    """The same straddling cube in Tossing3D is still on neither side."""
    env = Tossing3DEnvironment()
    state = env.build_state(observation=observation(cube_x=1.14), seed=0, steps_taken=0)
    assert state.flat_barrier is None
    assert not CUBE_ON_ROBOT_SIDE.holds(state, (env.cube, env.barrier))
    assert not CUBE_ON_OPPOSITE_SIDE.holds(state, (env.cube, env.barrier))


def test_kinders_own_bin_is_still_registered_as_bin():
    pytest.importorskip("kinder")
    from kinder.envs.dynamic3d.objects.base import REGISTERED_OBJECTS
    from kinder.envs.dynamic3d.objects.primitive_objects import Bin

    from hitl_pmp.environments.kitchen_recycling3d.scene import register_simulation_objects

    register_simulation_objects()
    assert REGISTERED_OBJECTS["bin"] is Bin
    assert REGISTERED_OBJECTS["kitchen_tapered_bin"] is not Bin
    assert issubclass(REGISTERED_OBJECTS["kitchen_tapered_bin"], Bin)


def test_a_config_whose_base_exceeds_its_rim_is_refused():
    with pytest.raises(ValueError, match="base"):
        TaperedBin.of(config=KitchenConfig(bin_base=(0.4, 0.19)))


needs_pybullet = pytest.mark.skipif(
    importlib.util.find_spec("pybullet") is None, reason="needs PyBullet"
)


@pytest.fixture
def client():
    import pybullet as p

    cid = p.connect(p.DIRECT)
    yield cid
    p.disconnect(physicsClientId=cid)


def shape_count(*, body: int, client: int) -> int:
    import pybullet as p

    links = range(-1, p.getNumJoints(body, physicsClientId=client))
    return sum(len(p.getCollisionShapeData(body, link, physicsClientId=client)) for link in links)


@needs_pybullet
@pytest.mark.parametrize("count", [1, 5, 16, 17, 40])
def test_a_planning_body_keeps_every_panel_past_pybullets_sixteen_shape_compound(*, client, count):
    """PyBullet keeps at most 16 shapes of one compound and drops the rest silently."""
    panels = bin_().panels()
    panels = (panels * (count // len(panels) + 1))[:count]
    body = TaperedBinCollision.body(
        client=client, panels=panels, position=(0, 0, 0), orientation=(0, 0, 0, 1)
    )
    assert shape_count(body=body, client=client) == count


@needs_pybullet
@pytest.mark.parametrize(("position", "yaw"), [((0.0, 0.0, 0.0), 0.0), ((2.05, 0.4, 0.0), 0.7)])
def test_a_probe_just_outside_every_wall_is_close_to_the_planning_bin_at_every_height(
    *, client, position, yaw
):
    """The -y wall's top strip (z 0.288-0.384) was the 17th panel, and missing."""
    import pybullet as p

    geometry = bin_()
    quaternion = Rotation.from_euler("z", yaw).as_quat()
    body = TaperedBinCollision.body(
        client=client,
        panels=geometry.panels(),
        position=(0, 0, 0),
        orientation=(0, 0, 0, 1),
    )
    # Built at the origin, then moved: the bin's pose is set this way after a reset.
    p.resetBasePositionAndOrientation(body, position, quaternion, physicsClientId=client)
    sphere = p.createCollisionShape(p.GEOM_SPHERE, radius=0.005, physicsClientId=client)
    probe = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=sphere, physicsClientId=client)
    rotation = Rotation.from_quat(quaternion)
    for axis in (0, 1):
        for sign in (1.0, -1.0):
            for z in (0.10, 0.25, 0.33, 0.38):
                local = [0.0, 0.0, z]
                local[axis] = sign * (geometry.outer_half(axis=axis, z=z) + 0.02)
                world = rotation.apply(local) + np.asarray(position)
                p.resetBasePositionAndOrientation(
                    probe, world, (0, 0, 0, 1), physicsClientId=client
                )
                points = p.getClosestPoints(probe, body, 0.5, physicsClientId=client)
                clearance = min((point[8] for point in points), default=np.inf)
                assert clearance <= 0.025, (axis, sign, z, clearance)

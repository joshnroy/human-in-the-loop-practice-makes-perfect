"""Ball physics and shallow-tray support contracts."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.sweeptable3d.balls import BallScene


def test_packaged_scene_has_balls_and_quarter_diameter_rim():
    scene = json.loads(Path(BallScene.scene_path()).read_text())
    balls = scene["objects"]["table_ball"]
    tray = scene["objects"]["bin"]["box_0"]
    assert len(balls) == 5
    assert {ball["size"] for ball in balls.values()} == {0.01}
    assert tray["height"] - tray["wall_thickness"] == pytest.approx(0.005)
    assert list(scene["objects"]) == ["table_wiper", "bin", "table_ball"]


@pytest.mark.parametrize(
    "center,upright,expected",
    [
        ([0, 0, 0.02], 1, True),
        ([0, 0, 0.05], 1, False),
        ([0, 0, 0.015], 1, False),
        ([0.11, 0, 0.02], 1, False),
        ([0, 0, 0.02], 0.9, False),
    ],
)
def test_supported_ball_may_protrude_above_rim(*, center, upright, expected):
    assert (
        BallScene.on_tray(
            local_center=np.array(center),
            radius=0.01,
            half_x=0.115,
            half_y=0.14,
            floor=0.01,
            rim_height=0.015,
            upright=upright,
        )
        is expected
    )


@pytest.mark.parametrize("yaw", [0, 45, 180])
def test_recorded_blue_ball_is_captured_by_shallow_tray(*, yaw):
    rotation = Rotation.from_euler("z", yaw, degrees=True)
    tray_position = np.array([1.2470812954886388, -0.18, 0.0])
    local_center = np.array([-0.10631978331844343, -0.001313687752590165, 0.0199983977439831])
    tray = SimpleNamespace(
        length=0.25,
        width=0.30,
        wall_thickness=0.01,
        height=0.015,
        get_pose=lambda: (tray_position, rotation.as_quat()[[3, 0, 1, 2]]),
    )
    ball = SimpleNamespace(
        size=[0.01],
        get_pose=lambda: (tray_position + rotation.apply(local_center), [1, 0, 0, 0]),
    )
    core = SimpleNamespace(get_object=lambda name: tray if name == "box_0" else ball)
    assert BallScene.in_tray(core=core, name="cube_1")


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("rim_height", [0.015, 0.02, 0.03])
@pytest.mark.parametrize("clearance", [-0.00001, 0.00001])
def test_wall_clearance_uses_sphere_section_at_rim(*, axis, sign, rim_height, clearance):
    half_extents = [0.115, 0.14]
    section_radius = np.sqrt(0.01**2 - max(0, 0.02 - rim_height) ** 2)
    center = np.array([0.0, 0.0, 0.02])
    center[axis] = sign * (half_extents[axis] - section_radius - clearance)
    assert BallScene.on_tray(
        local_center=center,
        radius=0.01,
        half_x=half_extents[0],
        half_y=half_extents[1],
        floor=0.01,
        rim_height=rim_height,
        upright=1,
    ) is (clearance > 0)


@pytest.mark.parametrize(
    "center,rim_height,expected",
    [
        ([0.1063, 0.1313, 0.02], 0.015, True),
        ([0.1064, 0.1313, 0.02], 0.015, False),
        ([0.1063, 0.1314, 0.02], 0.015, False),
        ([0.116, 0, 0.02], 0.015, False),
        ([0, 0.141, 0.02], 0.015, False),
        ([0.12, 0, 0.025], 0.015, False),
        ([0, 0, 0.01], 0.015, False),
        ([0, 0, 0.03], 0.015, False),
        ([0, 0, 0.02], 0.01, False),
        ([0, 0, 0.02], 0.009, False),
        ([0, 0, 0.0208], 0.0105, False),
    ],
)
def test_sphere_capture_requires_floor_support_and_wall_clearance(*, center, rim_height, expected):
    assert (
        BallScene.on_tray(
            local_center=np.array(center),
            radius=0.01,
            half_x=0.115,
            half_y=0.14,
            floor=0.01,
            rim_height=rim_height,
            upright=1,
        )
        is expected
    )


def test_settling_rejects_moving_ball():
    model = SimpleNamespace(
        body=lambda name: SimpleNamespace(id=0),
        body_jntadr=np.array([0]),
        jnt_dofadr=np.array([0]),
    )
    data = SimpleNamespace(qvel=np.zeros(6))
    assert BallScene.velocities_settled(model=model, data=data, names=["cube_0"])
    data.qvel[0] = 0.05
    assert not BallScene.velocities_settled(model=model, data=data, names=["cube_0"])
    data.qvel[:] = 0
    data.qvel[4] = 1
    assert not BallScene.velocities_settled(model=model, data=data, names=["cube_0"])


def test_open_table_board_heights_and_capsule_identification():
    from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import ShelfSkills

    core = SimpleNamespace(_shelf_geoms=lambda: [0, 1], shelf_top_z=lambda **kwargs: 0.28)
    assert ShelfSkills.board_heights(core=core) == {1: 0.28, 2: 1.28, 3: 0.28}
    model = SimpleNamespace(
        geom_type=np.array([6, 3]), geom_size=np.array([[0.0075, 0.0075, 0.04], [0.01, 0.09, 0]])
    )
    assert ShelfSkills.wiper_geometry(model=model, geoms=[0, 1]) == (1, 0)

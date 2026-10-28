"""Ball physics and shallow-tray support contracts."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

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
            upright=upright,
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

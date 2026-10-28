"""Whole-shrimp scoring must follow the curved collider through rotated poses."""

import xml.etree.ElementTree as ET
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.pouring3d.shrimp_native import CookedShrimp, ShrimpPourScene


@pytest.fixture
def shrimp_scene(monkeypatch):  # noqa: PLR0917
    root = ET.fromstring(
        '<mujoco><worldbody><body name="pitcher"/><body name="tray"/></worldbody></mujoco>'
    )
    shrimp = CookedShrimp(name="shrimp_0", options={"size": [0.017, 0.018, 0.0055], "mass": 0.003})
    root.find("worldbody").append(shrimp.xml_element)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    scene = object.__new__(ShrimpPourScene)
    pitcher = SimpleNamespace(length=0.11, width=0.114, height=0.14, wall_thickness=0.003)
    tray = SimpleNamespace(length=0.44, width=0.38, height=0.075, wall_thickness=0.01)
    monkeypatch.setattr(scene, "_model_and_data", lambda: (model, data))
    monkeypatch.setattr(scene, "get_object", lambda name: {"pitcher": pitcher, "tray": tray}[name])
    return scene, model, data


def test_curved_piece_resting_on_pitcher_floor_is_contained(shrimp_scene):  # noqa: PLR0917
    scene, model, data = shrimp_scene
    data.qpos[:3] = [0, 0, 0.0075]
    quat = Rotation.from_rotvec([-0.085, 0.028, -0.028]).as_quat()
    data.qpos[3:7] = quat[[3, 0, 1, 2]]
    mujoco.mj_forward(model, data)
    assert scene.cube_in_vessel("shrimp_0", "pitcher")
    assert scene.cube_corners("shrimp_0")[:, 2].min() >= 0.001


def test_center_inside_tray_does_not_count_protruding_shrimp(shrimp_scene):  # noqa: PLR0917
    scene, model, data = shrimp_scene
    data.qpos[:3] = [0.212, 0, 0.02]
    data.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    assert not scene.cube_in_vessel("shrimp_0", "tray")
    data.qpos[0] = 0.19
    mujoco.mj_forward(model, data)
    assert scene.cube_in_vessel("shrimp_0", "tray")


def test_curved_collision_body_keeps_open_curl(shrimp_scene):  # noqa: PLR0917
    _, model, _ = shrimp_scene
    body = model.body("shrimp_0").id
    geoms = range(model.body_geomadr[body], model.body_geomadr[body] + model.body_geomnum[body])
    assert model.body_mass[body] == pytest.approx(0.003)
    assert model.body_geomnum[body] == 7
    assert all(model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE for g in geoms)
    # The geometric curl center remains outside every capsule collider.
    center = np.array([-0.003, 0, 0])
    for start, end, radius in CookedShrimp.segments():
        direction = end - start
        t = np.clip(np.dot(center - start, direction) / np.dot(direction, direction), 0, 1)
        assert np.linalg.norm(center - start - t * direction) > radius

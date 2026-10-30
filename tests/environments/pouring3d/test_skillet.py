"""The receiving skillet is hollow, movable, and scored by its actual cavity."""

import importlib.util
import xml.etree.ElementTree as ET

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="KinDER required"
)


def test_skillet_mass_free_joint_and_hollow_cavity():
    import mujoco

    from hitl_pmp.environments.pouring3d.practice.skillet import CastIronSkillet

    pan = CastIronSkillet(name="tray", options=CastIronSkillet.specification())
    root = ET.Element("mujoco")
    asset = ET.SubElement(root, "asset")
    asset.extend(pan.get_assets())
    ET.SubElement(root, "worldbody").append(pan.xml_element)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert model.nq == 7 and model.nv == 6
    assert model.body_mass[model.body("tray").id] == pytest.approx(10.45 * 0.45359237)
    # A downward ray through the center must hit the floor, not a convex lid.
    geom = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(
        model, data, np.array([0.0, 0.0, 0.3]), np.array([0.0, 0.0, -1.0]), None, 1, -1, geom
    )
    assert distance == pytest.approx(0.3 - pan.wall_thickness, abs=1e-6)
    assert pan.contains_capsules(endpoints=np.array([[0, 0, 0.012]]), radii=np.array([0.005]))
    # Inside the old rectangular footprint but outside the round pan.
    assert not pan.contains_capsules(
        endpoints=np.array([[0.14, 0.14, 0.012]]), radii=np.array([0.005])
    )
    assert not pan.contains_capsules(
        endpoints=np.array([[0, 0, pan.height]]), radii=np.array([0.005])
    )


def test_native_default_skillet_and_legacy_tray_are_distinct():
    from hitl_pmp.environments.pouring3d.practice.environment import FrozenPourEnvironment
    from hitl_pmp.environments.pouring3d.practice.skillet import CastIronSkillet

    for receiver in ("skillet", "tray"):
        env = FrozenPourEnvironment(layout="floor", receiving_vessel=receiver, scene_bg=False)
        try:
            env.hard_reset()
            c = env.controller()
            assert isinstance(c.scene.get_object("tray"), CastIronSkillet) == (
                receiver == "skillet"
            )
            assert c.ready()
            before = c.body_pose(name="tray")[0].copy()
            c.pause(phase="pan stability", seconds=2)
            assert np.linalg.norm(c.body_pose(name="tray")[0] - before) < 0.002
            assert c.vessel_home(name="tray")
        finally:
            env.close()


def test_falling_shrimp_stays_above_thin_pan_floor():
    import mujoco
    from scipy.spatial.transform import Rotation

    from hitl_pmp.environments.pouring3d.practice.environment import FrozenPourEnvironment
    from hitl_pmp.environments.pouring3d.shrimp_native import CookedShrimp

    env = FrozenPourEnvironment(layout="floor", scene_bg=False)
    try:
        env.hard_reset()
        c = env.controller()
        pan, _ = c.body_pose(name="tray")
        for i, name in enumerate(c.scene.cube_names):
            obj = c.scene.get_object(name)
            quat = Rotation.from_euler("xyz", [0.5 * i, 0.7 * i, 0.3 * i]).as_quat()
            obj.set_pose(pan + [0.07 * (i - 1), 0.03, 0.35], quat[[3, 0, 1, 2]])
            obj.set_velocity([0, 0, 0], [0, 0, 0])
        mujoco.mj_forward(c.model, c.data)
        minimum = 1.0
        endpoints = np.array([p for a, b, _ in CookedShrimp.segments() for p in (a, b)])
        radii = np.repeat([r for _, _, r in CookedShrimp.segments()], 2)
        for _ in range(60):
            c.motor_step(target_q=c.last_q, grip=0)
            for name in c.scene.cube_names:
                origin, rotation = c.body_pose(name=name)
                bottom = np.min(rotation.apply(endpoints)[:, 2] + origin[2] - radii)
                minimum = min(minimum, float(bottom))
        assert minimum >= 0.0025, minimum
        assert c.scene.cube_locations() == ["tray"] * 3
        assert c.scene.cubes_settled()
        assert c.vessel_home(name="tray")
    finally:
        env.close()

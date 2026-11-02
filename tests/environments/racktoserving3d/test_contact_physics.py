"""Small native settling regressions using the production food and tray geometry."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.geometry import install_geometry
from hitl_pmp.environments.racktoserving3d.scene import DUMPLING_TYPE, TRAY_TYPE, build_scene


@pytest.mark.parametrize("tilt", [0, 15, 45])
def test_production_food_settles_on_priority_tray(*, tilt: float) -> None:
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("kinder")
    from kinder.envs.dynamic3d.objects.base import REGISTERED_OBJECTS

    install_geometry()
    config = RackServingConfig()
    scene = build_scene(config=config)
    root = ET.Element("mujoco")
    ET.SubElement(root, "option", timestep="0.0005")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "geom", type="plane", size="1 1 0.1")
    for kind, name in ((TRAY_TYPE, "box_0"), (DUMPLING_TYPE, "cube_0")):
        instance = REGISTERED_OBJECTS[kind](name=name, options=scene["objects"][kind][name])
        body = instance.xml_element
        if name == "box_0":
            geom = body.find("geom")
            geom.set("priority", "1")
            geom.set("solref", "0.004 1")
            geom.set("solimp", "0.99 0.999 0.001 0.5 2")
        else:
            rotation = Rotation.from_euler("y", tilt, degrees=True)
            radii = (
                np.array([config.dumpling_depth, config.dumpling_width, config.dumpling_height]) / 2
            )
            extent = np.linalg.norm(rotation.as_matrix()[2] * radii)
            body.set("pos", f"0.10 0 {config.tray_thickness + extent + 0.013}")
            body.set("quat", " ".join(map(str, rotation.as_quat()[[3, 0, 1, 2]])))
        world.append(body)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    food_body = model.body("cube_0").id
    velocity_index = model.jnt_dofadr[model.body_jntadr[food_body]]
    food_geom = model.geom("cube_0_dumpling").id
    data.qvel[velocity_index] = -0.03
    observed_dimensions = set()
    for _ in range(10000):
        mujoco.mj_step(model, data)
        observed_dimensions.update(
            int(contact.dim)
            for contact in data.contact[: data.ncon]
            if food_geom in (contact.geom1, contact.geom2)
        )
    assert observed_dimensions == {6}
    assert np.linalg.norm(data.qvel[velocity_index : velocity_index + 3]) < 0.001
    assert np.linalg.norm(data.qvel[velocity_index + 3 : velocity_index + 6]) < 0.05
    assert abs(data.xpos[food_body, 0]) < config.tray_depth / 2 - config.dumpling_depth / 2
    assert data.xpos[food_body, 2] > config.tray_thickness

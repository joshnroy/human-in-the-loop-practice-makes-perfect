"""The planning image ships the arm the simulator's own pick plans with."""

import importlib.util
import xml.etree.ElementTree as ET

import pytest

from scripts.build_agentic_planning_image import PlanningImageBuilder

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pybullet_helpers") is None,
    reason="the robot assets come from pybullet_helpers",
)


def test_the_default_urdf_is_the_vision_module_wrist() -> None:
    """The default URDF collides the bracelet's vision module and has no RealSense box,
    matching `KinderBackend._pick_collision_sim` (which `planning_spec` says it does)."""
    root = ET.parse(PlanningImageBuilder.default_urdf()).getroot()
    (bracelet,) = [link for link in root.iter("link") if link.get("name") == "bracelet_link"]
    meshes = [
        mesh.get("filename", "") for c in bracelet.iter("collision") for mesh in c.iter("mesh")
    ]
    assert meshes and all(m.endswith("bracelet_with_vision_link.STL") for m in meshes)
    assert not [box for box in root.iter("box") if box.get("size") == "0.025 0.09 0.035"]

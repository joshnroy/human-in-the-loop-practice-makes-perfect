"""Native benchmark variants with surface-specific rubber-finger contacts.

The frozen proxy's priority-2 plastic contact coefficient would otherwise override
the Robotiq rubber pads. Explicit finger pairs use the native lower pad coefficient
0.6, an uncalibrated assumption. Payload/pitcher, tray and payload/payload contacts
remain unchanged; no adhesive grasp, weld or forced object motion is introduced.
"""

import xml.etree.ElementTree as ET
from typing import Any

from hitl_pmp.environments.pouring3d.five_frozen import FiveFrozenShrimpEnv, FiveFrozenShrimpScene
from hitl_pmp.environments.pouring3d.frozen_native import FrozenShrimpEnv, FrozenShrimpScene


class FingerContacts:
    @staticmethod
    def apply(*, xml: str, names: tuple[str, ...], robot: str) -> str:
        root = ET.fromstring(xml)
        contact = root.find("contact")
        if contact is None:
            contact = ET.SubElement(root, "contact")
        for name in names:
            for segment in range(7):
                for side in ("left", "right"):
                    for pad in (1, 2):
                        ET.SubElement(
                            contact,
                            "pair",
                            geom1=f"{name}_segment_{segment}",
                            geom2=f"{robot}_{side}_pad{pad}",
                            condim="3",
                            friction="0.6 0.6 0 0 0",
                            solref="0.008 1",
                            solimp="0.95 0.99 0.001 0.5 2",
                        )
        return ET.tostring(root, encoding="unicode")


class PracticeThreeScene(FrozenShrimpScene):
    def _create_scene_xml(self) -> str:
        return FingerContacts.apply(
            xml=super()._create_scene_xml(), names=self.cube_names, robot=self.robot_name
        )


class PracticeFiveScene(FiveFrozenShrimpScene):
    def _create_scene_xml(self) -> str:
        return FingerContacts.apply(
            xml=super()._create_scene_xml(), names=self.cube_names, robot=self.robot_name
        )


class PracticeThreeEnv(FrozenShrimpEnv):
    def _create_object_centric_env(self, *args: Any, **kwargs: Any) -> PracticeThreeScene:
        return PracticeThreeScene(*args, **kwargs)


class PracticeFiveEnv(FiveFrozenShrimpEnv):
    def _create_object_centric_env(self, *args: Any, **kwargs: Any) -> PracticeFiveScene:
        return PracticeFiveScene(*args, **kwargs)

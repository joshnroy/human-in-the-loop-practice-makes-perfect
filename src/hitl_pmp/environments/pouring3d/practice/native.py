"""Native benchmark variants with surface-specific rubber-finger contacts.

The frozen proxy's priority-2 plastic contact coefficient would otherwise override
the Robotiq rubber pads. Explicit finger pairs use the native lower pad coefficient
0.6, an uncalibrated assumption. Payload/pitcher, tray and payload/payload contacts
remain unchanged; no adhesive grasp, weld or forced object motion is introduced.
"""

import xml.etree.ElementTree as ET
from typing import Any, cast

import mujoco
import numpy as np
from kinder.envs.dynamic3d.objects.primitive_objects import Bin
from scipy.spatial.transform import Rotation

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
    def respawn_payload_in_pitcher(self) -> None:
        # Version 2 uses the audited 22 mm bottom ring. The historical three-piece
        # prototype keeps its original 24 mm layout for reproducibility.
        pitcher = cast(Bin, self.get_object("pitcher"))
        origin = np.asarray(pitcher.get_position(), dtype=float)
        quat = np.asarray(pitcher.get_orientation(), dtype=float)
        rotation = Rotation.from_quat(quat[[1, 2, 3, 0]])
        for name, offset in zip(
            self.cube_names, FiveFrozenShrimpScene.reset_offsets()[:3], strict=True
        ):
            local = offset.copy()
            local[:2] += self.np_random.uniform(-0.0005, 0.0005, 2)
            local[2] += pitcher.wall_thickness
            shrimp = self.get_object(name)
            shrimp.set_pose(origin + rotation.apply(local), quat)
            shrimp.set_velocity([0.0] * 3, [0.0] * 3)
        model, data = self._model_and_data()
        mujoco.mj_forward(model, data)

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

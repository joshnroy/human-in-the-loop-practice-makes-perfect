"""Native benchmark variants with surface-specific rubber-finger contacts.

The frozen proxy's priority-2 plastic contact coefficient would otherwise override
the Robotiq rubber pads. Explicit finger pairs use the native lower pad coefficient
0.6, an uncalibrated assumption. Explicit skillet pairs use stiffer impact
contacts with the same chosen sliding coefficient. No adhesive grasp, weld or
forced object motion is introduced.
"""

import xml.etree.ElementTree as ET
from typing import Any, cast

import mujoco
import numpy as np
from kinder.envs.dynamic3d.objects.primitive_objects import Bin
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.pouring3d.five_frozen import FiveFrozenShrimpEnv, FiveFrozenShrimpScene
from hitl_pmp.environments.pouring3d.frozen_native import FrozenShrimpEnv, FrozenShrimpScene
from hitl_pmp.environments.pouring3d.shrimp_native import CookedShrimp

from .skillet import CastIronSkillet
from .stations import PourStovetop  # noqa: F401 -- register native fixture


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


class HandleContactPatch:
    @staticmethod
    def apply(*, xml: str, enabled: bool) -> str:
        if not enabled:
            return xml
        root = ET.fromstring(xml)
        for name in ("pitcher_handle_upper", "pitcher_handle_grip"):
            geom = root.find(f".//geom[@name='{name}']")
            assert geom is not None
            # The existing 0.005 m torsional coefficient is inactive at condim=3.
            # A finite rubber contact patch can resist twisting about its normal.
            geom.set("condim", "4")
        return ET.tostring(root, encoding="unicode")


class SkilletContainment:
    @staticmethod
    def contains(*, scene: Any, shrimp: str) -> bool:
        model, data = scene._model_and_data()
        body, pan_body = model.body(shrimp).id, model.body("tray").id
        parts = CookedShrimp.segments()
        endpoints = np.array([p for start, end, _ in parts for p in (start, end)])
        radii = np.repeat([r for _, _, r in parts], 2)
        local = (
            endpoints @ data.xmat[body].reshape(3, 3).T + data.xpos[body] - data.xpos[pan_body]
        ) @ data.xmat[pan_body].reshape(3, 3)
        return bool(scene.get_object("tray").contains_capsules(endpoints=local, radii=radii))


class SkilletContacts:
    @staticmethod
    def apply(*, xml: str, names: tuple[str, ...], mu: float, skillet: bool) -> str:
        if not skillet:
            return xml
        root = ET.fromstring(xml)
        contact = root.find("contact")
        assert contact is not None
        pan = root.find(".//body[@name='tray']")
        assert pan is not None
        # The legacy 8 ms soft contact lets falling pieces cross a 4.5 mm base.
        # Keep sliding friction explicit but resolve the rigid pan impact faster.
        for name in names:
            for segment in range(7):
                for geom in pan.findall("geom"):
                    ET.SubElement(
                        contact,
                        "pair",
                        geom1=f"{name}_segment_{segment}",
                        geom2=geom.attrib["name"],
                        condim="3",
                        friction=f"{mu} {mu} 0 0 0",
                        solref="0.002 1",
                        solimp="0.99 0.999 0.0001 0.5 2",
                    )
        return ET.tostring(root, encoding="unicode")


class SupportSurfaceLocations:
    @staticmethod
    def classify(*, scene: Any, locations: list[str]) -> list[str]:
        if "prep_table" not in scene._fixtures_dict:
            return locations
        model, data = scene._model_and_data()
        for i, (name, location) in enumerate(zip(scene.cube_names, locations, strict=True)):
            if location != "air":
                continue
            points = scene.cube_corners(name)
            for support in ("prep_table_top", "stovetop_top"):
                geom = model.geom(support).id
                local = (points - data.geom_xpos[geom]) @ data.geom_xmat[geom].reshape(3, 3)
                bottom = float(local[:, 2].min() - model.geom_size[geom, 2])
                if -0.002 <= bottom <= 0.005 and np.all(
                    np.abs(local[:, :2]) <= model.geom_size[geom, :2] + 0.002
                ):
                    locations[i] = "surface"
                    break
        return locations


class PracticeThreeScene(FrozenShrimpScene):
    def cube_locations(self) -> list[str]:
        return SupportSurfaceLocations.classify(scene=self, locations=super().cube_locations())

    def cube_in_vessel(self, cube: str, vessel: str) -> bool:  # noqa: PLR0917
        if vessel == "tray" and isinstance(self.get_object(vessel), CastIronSkillet):
            return SkilletContainment.contains(scene=self, shrimp=cube)
        return bool(super().cube_in_vessel(cube, vessel))

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
        xml = FingerContacts.apply(
            xml=HandleContactPatch.apply(
                xml=super()._create_scene_xml(), enabled="prep_table" in self._fixtures_dict
            ),
            names=self.cube_names,
            robot=self.robot_name,
        )
        return SkilletContacts.apply(
            xml=xml,
            names=self.cube_names,
            mu=self.surface_mu,
            skillet=isinstance(self.get_object("tray"), CastIronSkillet),
        )


class PracticeFiveScene(FiveFrozenShrimpScene):
    def cube_locations(self) -> list[str]:
        return SupportSurfaceLocations.classify(scene=self, locations=super().cube_locations())

    def cube_in_vessel(self, cube: str, vessel: str) -> bool:  # noqa: PLR0917
        if vessel == "tray" and isinstance(self.get_object(vessel), CastIronSkillet):
            return SkilletContainment.contains(scene=self, shrimp=cube)
        return bool(super().cube_in_vessel(cube, vessel))

    def _create_scene_xml(self) -> str:
        xml = FingerContacts.apply(
            xml=HandleContactPatch.apply(
                xml=super()._create_scene_xml(), enabled="prep_table" in self._fixtures_dict
            ),
            names=self.cube_names,
            robot=self.robot_name,
        )
        return SkilletContacts.apply(
            xml=xml,
            names=self.cube_names,
            mu=self.surface_mu,
            skillet=isinstance(self.get_object("tray"), CastIronSkillet),
        )


class PracticeThreeEnv(FrozenShrimpEnv):
    def _create_object_centric_env(self, *args: Any, **kwargs: Any) -> PracticeThreeScene:
        return PracticeThreeScene(*args, **kwargs)


class PracticeFiveEnv(FiveFrozenShrimpEnv):
    def _create_object_centric_env(self, *args: Any, **kwargs: Any) -> PracticeFiveScene:
        return PracticeFiveScene(*args, **kwargs)

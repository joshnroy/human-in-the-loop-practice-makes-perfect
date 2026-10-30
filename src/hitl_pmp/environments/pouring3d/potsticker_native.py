"""Frozen potsticker scene with three or ten physically separate pieces.

Shape reference: https://www.traderjoes.com/home/discover/stories/potsticker-soups-on
55 x 25 x 27 mm and 20 g are estimates, not measured product specifications.
The accepted preview's pouch, folds, wrapper and seam are all collision geometry.
Friction is an explicit sensitivity assumption; no ice adhesion or thawing model.
"""

import xml.etree.ElementTree as ET
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import mujoco
import numpy as np
from kinder.envs.dynamic3d.objects.base import register_object
from kinder.envs.dynamic3d.objects.primitive_objects import Cuboid
from kinder.envs.dynamic3d.pitcher_pour3d import (
    ObjectCentricPitcherPour3DEnv,
    PitcherPour3DEnv,
    PouringPitcher,
)
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .practice.native import HandleContactPatch, SupportSurfaceLocations
from .practice.skillet import CastIronSkillet

ASSETS = Path(__file__).parent / "assets"


@lru_cache(maxsize=1)
def shape_xml() -> ET.Element:
    return ET.parse(ASSETS / "potsticker-shape.xml").getroot()


@lru_cache(maxsize=1)
def support_vertices() -> np.ndarray:
    """Mesh vertices plus conservative 0.65 mm seam-capsule bounding boxes."""
    root = shape_xml()
    points = [
        np.fromstring(m.attrib["vertex"], sep=" ").reshape(-1, 3)
        for m in root.findall("asset/mesh")
    ]
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    for geom in root.findall("worldbody/body/geom"):
        if geom.get("type") == "capsule":
            endpoints = np.fromstring(geom.attrib["fromto"], sep=" ").reshape(2, 3)
            points.extend(endpoint + signs * float(geom.attrib["size"]) for endpoint in endpoints)
    vertices = np.concatenate(points)
    return vertices[ConvexHull(vertices).vertices]


@register_object
class FrozenPotsticker(Cuboid):
    def _create_xml_element(self) -> ET.Element:
        body = deepcopy(shape_xml().find("worldbody/body"))
        assert body is not None
        body.set("name", self.name)
        # KinDER compiles before its seeded pose initializer runs. Keep the ten
        # uninitialized compound bodies apart during that initial forward pass.
        body.set("pos", f"3 {int(self.name.rsplit('_', 1)[1]) * 0.1} 1")
        ET.SubElement(body, "freejoint", name=self.joint_name)
        for i, geom in enumerate(body.findall("geom")):
            geom.set("name", f"{self.name}_part_{i}")
            geom.set("mass", str(float(geom.attrib["mass"]) * self.mass / 0.020))
        return body

    def get_assets(self) -> list[ET.Element]:
        if self.name != "potsticker_0":
            return []
        node = shape_xml().find("asset")
        assert node is not None
        assets = deepcopy(list(node))
        for node in assets:
            if node.tag == "texture":
                node.set("file", str(ASSETS / "potsticker-wrapper.png"))
        return assets


class PotstickerScene(ObjectCentricPitcherPour3DEnv):
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple:
        _, info = super().reset(seed=seed, options=options)
        # Stacked pieces occasionally need more than the base scene's one second.
        # Wait with native dynamics; do not zero residual velocity to manufacture rest.
        for _ in range(30):
            if self.cubes_settled():
                break
            assert self.action_space.shape is not None
            self._robot_env.step(np.zeros(self.action_space.shape))
        self._current_state = self._get_object_centric_state()
        return self._get_current_state(), info

    def __init__(
        self,
        *args: Any,
        payload_count: int = 3,
        requested_count: int = 1,
        surface_mu: float = 0.1,
        **kwargs: Any,
    ) -> None:
        if type(payload_count) is not int or payload_count not in (3, 10):
            raise ValueError("Potsticker payload count must be 3 or 10")
        if type(requested_count) is not int or not 1 <= requested_count <= payload_count:
            raise ValueError("Requested count must be within payload count")
        if not np.isfinite(surface_mu) or surface_mu < 0:
            raise ValueError("Invalid surface friction")
        self.cube_names = tuple(f"potsticker_{i}" for i in range(payload_count))
        self.surface_mu = surface_mu
        # The legacy base constructor restricts its own five-cube task to 1..5.
        # This variant validates against its actual payload before using the base.
        super().__init__(
            *args, requested_count=min(requested_count, 5), cube_material=None, **kwargs
        )
        self.requested_count = requested_count

    def _create_scene_xml(self) -> str:
        root = ET.fromstring(
            HandleContactPatch.apply(xml=super()._create_scene_xml(), enabled=True)
        )
        contact = root.find("contact")
        if contact is None:
            contact = ET.SubElement(root, "contact")
        for name in self.cube_names:
            body = root.find(f".//body[@name='{name}']")
            assert body is not None
            for geom in body.findall("geom"):
                geom.set("friction", f"{self.surface_mu} 0 0")
                for side in ("left", "right"):
                    for pad in (1, 2):
                        ET.SubElement(
                            contact,
                            "pair",
                            geom1=geom.attrib["name"],
                            geom2=f"{self.robot_name}_{side}_pad{pad}",
                            condim="3",
                            friction=".6 .6 0 0 0",
                            solref=".008 1",
                        )
        return ET.tostring(root, encoding="unicode")

    def respawn_payload_in_pitcher(self) -> None:
        pitcher = cast(PouringPitcher, self.get_object("pitcher"))
        origin = np.asarray(pitcher.get_position())
        quat = np.asarray(pitcher.get_orientation())
        rotation = Rotation.from_quat(quat[[1, 2, 3, 0]])
        for i, name in enumerate(self.cube_names):
            layer, side = divmod(i, 2)
            yaw = (layer % 2) * np.pi / 2 + self.np_random.uniform(-0.035, 0.035)
            local_rotation = Rotation.from_euler("z", yaw)
            offset = local_rotation.apply([0, (side - 0.5) * 0.030, 0])
            offset[:2] += self.np_random.uniform(-0.0005, 0.0005, 2)
            offset[2] = pitcher.wall_thickness + 0.001 + layer * 0.029
            q = (rotation * local_rotation).as_quat()
            obj = self.get_object(name)
            obj.set_pose(origin + rotation.apply(offset), q[[3, 0, 1, 2]])
            obj.set_velocity([0.0] * 3, [0.0] * 3)
        model, data = self._model_and_data()
        mujoco.mj_forward(model, data)

    def cube_corners(self, name: str) -> np.ndarray:  # noqa: PLR0917 -- native scene protocol
        model, data = self._model_and_data()
        body = model.body(name).id
        return support_vertices() @ data.xmat[body].reshape(3, 3).T + data.xpos[body]

    def cube_in_vessel(self, cube: str, vessel: str) -> bool:  # noqa: PLR0917 -- native protocol
        model, data = self._model_and_data()
        body = model.body(vessel).id
        local = (self.cube_corners(cube) - data.xpos[body]) @ data.xmat[body].reshape(3, 3)
        geometry = self.get_object(vessel)
        if isinstance(geometry, CastIronSkillet):
            return bool(geometry.contains_capsules(endpoints=local, radii=np.zeros(len(local))))
        return bool(cast(PouringPitcher, geometry).contains_points(local))

    def cube_locations(self) -> list[str]:
        return SupportSurfaceLocations.classify(scene=self, locations=super().cube_locations())

    def _check_goals(self) -> bool:
        locations = self.cube_locations()
        return (
            locations.count("tray") == self.requested_count
            and locations.count("pitcher") == len(self.cube_names) - self.requested_count
            and self.cubes_settled()
        )

    def physical_snapshot(self) -> dict[str, Any]:
        result = super().physical_snapshot()
        result.update(
            payload="frozen potsticker rigid proxy",
            payload_count=len(self.cube_names),
            surface_mu=self.surface_mu,
        )
        return result


class PotstickerPourEnv(PitcherPour3DEnv):
    def _create_object_centric_env(self, *args: Any, **kwargs: Any) -> PotstickerScene:
        return PotstickerScene(*args, **kwargs)

    def _create_variant_specific_description(self) -> str:
        scene = cast(PotstickerScene, self._object_centric_env)
        return (
            f"Pour exactly {scene.requested_count} of {len(scene.cube_names)} frozen potstickers."
        )


def make_potsticker_env(
    *, count: int, requested: int, seed: int = 0, scene_bg: bool = True, surface_mu: float = 0.1
) -> Any:
    env = PotstickerPourEnv(
        task_config_path=str(ASSETS / f"potsticker-pour{count}-stations.json"),
        payload_count=count,
        requested_count=requested,
        surface_mu=surface_mu,
        scene_bg=scene_bg,
        render_mode="rgb_array",
    )
    env.reset(seed=seed)
    return env

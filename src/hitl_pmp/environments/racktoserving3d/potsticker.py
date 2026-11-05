"""Pouring3D's frozen potsticker as the rack's food: its pouch, folds, wrapper and seam.

By default (``collision="full"``) every part is collision geometry, exactly as in
Pouring3D. The shape is turned a quarter turn about z and recentred, so the 55 mm long
axis lies along body y and the body origin is the hull's geometric centre: the frame
the rack's ellipsoid had (long axis across the rack, ``face 0`` grasps across the
short axis, poses are geometric centres). Mass, friction and contact dimension stay
the rack's own configuration; the asset's contact ``priority`` is dropped so the
rack's surfaces combine with the food as they did with the ellipsoid.

Every other ``collision`` keeps all 41 parts as visual-only geometry, so the piece
still looks like a potsticker, and collides through a few simpler hidden geoms
(group 3) instead: 41 colliding parts per piece made contact-rich skills slow. The
body's mass and inertia are pinned to the full model's, so only contact changes.
"""

import xml.etree.ElementTree as ET
from copy import deepcopy
from functools import lru_cache
from typing import Any, ClassVar, Literal

import numpy as np
from scipy.spatial import ConvexHull

from hitl_pmp.environments.pouring3d.potsticker_shape import ASSETS, shape_xml
from hitl_pmp.environments.pouring3d.potsticker_shape import (
    support_vertices as source_support_vertices,
)

PREFIX = "rack_potsticker_"
CollisionModel = Literal["full", "hull", "hull_crest", "primitives", "pouch_hull", "ellipsoid"]
# Hull vertices kept by the decimated pouch; MuJoCo's convex collision and PyBullet's
# mirror both cost more per vertex, and 48 keep the hull within about a millimetre.
POUCH_HULL_VERTICES = 48
# Crest capsules replacing the 16 seam capsules and 8 folds.
CREST_SEGMENTS = 4
CREST_RADIUS_M = 0.0015
HIDDEN_GROUP = "3"


class RackPotsticker:
    TURN: ClassVar[np.ndarray] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    SOURCE_MASS: ClassVar[float] = 0.020

    @staticmethod
    @lru_cache(maxsize=1)
    def centre() -> np.ndarray:
        turned = source_support_vertices() @ RackPotsticker.TURN.T
        return (turned.min(axis=0) + turned.max(axis=0)) / 2

    @staticmethod
    def place(*, points: np.ndarray) -> np.ndarray:
        """Source-frame points in the rack food's body frame."""
        return points @ RackPotsticker.TURN.T - RackPotsticker.centre()

    @staticmethod
    @lru_cache(maxsize=1)
    def support_vertices() -> np.ndarray:
        return RackPotsticker.place(points=source_support_vertices())

    @staticmethod
    def extents() -> np.ndarray:
        """(depth, width, height): world x, y, z extents at zero yaw."""
        return np.ptp(RackPotsticker.support_vertices(), axis=0)

    @staticmethod
    def body(  # noqa: PLR0913 -- one argument per configured food property
        *,
        name: str,
        joint: str,
        mass: float,
        friction: str,
        condim: int,
        index: int,
        collision: CollisionModel = "full",
    ) -> ET.Element:
        body = deepcopy(shape_xml().find("worldbody/body"))
        assert body is not None
        body.set("name", name)
        # KINDER compiles before its pose initializer runs; keep the compound bodies apart.
        body.set("pos", f"3 {index * 0.1} 1")
        ET.SubElement(body, "freejoint", name=joint)
        for i, geom in enumerate(body.findall("geom")):
            geom.set("name", f"{name}_part_{i}")
            geom.set("mass", str(float(geom.attrib["mass"]) * mass / RackPotsticker.SOURCE_MASS))
            geom.set("friction", friction)
            geom.set("condim", str(condim))
            geom.attrib.pop("priority", None)
            geom.set("material", PREFIX + geom.attrib["material"])
            if geom.get("type") == "mesh":
                geom.set("mesh", PREFIX + geom.attrib["mesh"])
            if geom.get("type") == "capsule":
                ends = np.fromstring(geom.attrib["fromto"], sep=" ").reshape(2, 3)
                geom.set(
                    "fromto",
                    " ".join(
                        f"{value:.10g}" for value in RackPotsticker.place(points=ends).ravel()
                    ),
                )
        if collision == "full":
            return body
        return RackPotsticker._simplify(
            body=body, name=name, mass=mass, friction=friction, condim=condim, collision=collision
        )

    @staticmethod
    def _simplify(  # noqa: PLR0913 -- the full body plus the food properties it keeps
        *,
        body: ET.Element,
        name: str,
        mass: float,
        friction: str,
        condim: int,
        collision: CollisionModel,
    ) -> ET.Element:
        contact = shape_xml().find("worldbody/body/geom")
        assert contact is not None
        for i, geom in enumerate(body.findall("geom")):
            geom.set("name", f"{name}_visual_{i}")
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            geom.set("mass", "0")
            for key in ("friction", "condim", "solref", "solimp"):
                geom.attrib.pop(key, None)
        position, quaternion, inertia = RackPotsticker.full_inertia()
        scale = mass / RackPotsticker.SOURCE_MASS
        inertial = ET.Element(
            "inertial",
            pos=_Mjcf.values(values=position),
            quat=_Mjcf.values(values=quaternion),
            mass=str(mass),
            diaginertia=_Mjcf.values(values=inertia * scale),
        )
        body.insert(0, inertial)
        for i, spec in enumerate(RackPotsticker.collision_geoms(collision=collision)):
            geom = ET.SubElement(
                body,
                "geom",
                name=f"{name}_part_{i}",
                group=HIDDEN_GROUP,
                rgba="0.9 0.2 0.2 1",
                mass="0",
                friction=friction,
                condim=str(condim),
                solref=contact.attrib["solref"],
                solimp=contact.attrib["solimp"],
            )
            for key, value in spec.items():
                geom.set(
                    key,
                    value if key in {"type", "mesh"} else _Mjcf.values(values=np.asarray(value)),
                )
        return body

    @staticmethod
    def collision_geoms(*, collision: CollisionModel) -> list[dict[str, Any]]:
        """MJCF attributes of each simplified collision geom, in the body frame."""
        half = RackPotsticker.extents() / 2
        bottom, top = -half[2], half[2]
        if collision == "ellipsoid":
            return [{"type": "ellipsoid", "size": half}]
        if collision == "hull":
            return [{"type": "mesh", "mesh": PREFIX + "collision_hull"}]
        if collision == "pouch_hull":
            return [{"type": "mesh", "mesh": PREFIX + "collision_pouch"}]
        if collision == "hull_crest":
            crest = RackPotsticker.crest()
            return [{"type": "mesh", "mesh": PREFIX + "collision_pouch"}] + [
                {"type": "capsule", "fromto": np.concatenate(pair), "size": [CREST_RADIUS_M]}
                for pair in zip(crest[:-1], crest[1:], strict=True)
            ]
        if collision == "primitives":
            # A flat-bottomed slab, the rounded belly along the long axis, and the crest.
            belly = 0.0095
            belly_z = bottom + 0.0138
            crest_radius = 0.0025
            return [
                {
                    "type": "box",
                    "pos": [0, 0, bottom + 0.005],
                    "size": [0.0075, half[1] - 0.0105, 0.005],
                },
                {
                    "type": "capsule",
                    "fromto": [0, -(half[1] - belly), belly_z, 0, half[1] - belly, belly_z],
                    "size": [belly],
                },
                {
                    "type": "capsule",
                    "fromto": [0, -0.012, top - crest_radius, 0, 0.012, top - crest_radius],
                    "size": [crest_radius],
                },
            ]
        raise ValueError(f"No simplified collision model {collision!r}")

    @staticmethod
    def collision_support(*, collision: CollisionModel) -> np.ndarray:
        """Points whose hull bounds the simplified collision geometry."""
        if collision == "full":
            return RackPotsticker.support_vertices()
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        points = []
        for spec in RackPotsticker.collision_geoms(collision=collision):
            kind = spec["type"]
            if kind == "mesh":
                points.append(RackPotsticker.collision_meshes()[spec["mesh"]])
            elif kind in {"ellipsoid", "box"}:
                points.append(np.asarray(spec.get("pos", np.zeros(3))) + signs * spec["size"])
            elif kind == "capsule":
                ends = np.asarray(spec["fromto"], dtype=float).reshape(2, 3)
                points.extend(end + signs * spec["size"][0] for end in ends)
        return np.concatenate(points)

    @staticmethod
    @lru_cache(maxsize=1)
    def crest() -> np.ndarray:
        """The seam's chain points, thinned to `CREST_SEGMENTS` capsules whose tops stay
        at the seam's top."""
        seams = [
            g for g in shape_xml().findall("worldbody/body/geom") if g.get("type") == "capsule"
        ]
        ends = [np.fromstring(g.attrib["fromto"], sep=" ").reshape(2, 3) for g in seams]
        chain = RackPotsticker.place(points=np.array([ends[0][0], *(pair[1] for pair in ends)]))
        picks = np.linspace(0, len(chain) - 1, CREST_SEGMENTS + 1).round().astype(int)
        drop = CREST_RADIUS_M - float(seams[0].attrib["size"])
        return chain[picks] - np.array([0, 0, drop])

    @staticmethod
    @lru_cache(maxsize=1)
    def collision_meshes() -> dict[str, np.ndarray]:
        pouch = shape_xml().find("asset/mesh[@name='pouch']")
        assert pouch is not None
        points = RackPotsticker.place(
            points=np.fromstring(pouch.attrib["vertex"], sep=" ").reshape(-1, 3)
        )
        return {
            PREFIX + "collision_hull": RackPotsticker.support_vertices(),
            PREFIX + "collision_pouch": _Mjcf.farthest_points(
                points=points[ConvexHull(points).vertices], count=POUCH_HULL_VERTICES
            ),
        }

    @staticmethod
    @lru_cache(maxsize=1)
    def full_inertia() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The full model's centre of mass, principal frame and inertia at SOURCE_MASS."""
        import mujoco

        root = ET.Element("mujoco")
        asset = ET.SubElement(root, "asset")
        for element in RackPotsticker.assets():
            if element.tag == "texture":
                continue
            element.attrib.pop("texture", None)
            asset.append(element)
        world = ET.SubElement(root, "worldbody")
        world.append(
            RackPotsticker.body(
                name="probe",
                joint="probe",
                mass=RackPotsticker.SOURCE_MASS,
                friction="1 0 0",
                condim=3,
                index=0,
            )
        )
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        return (
            np.array(model.body_ipos[1]),
            np.array(model.body_iquat[1]),
            np.array(model.body_inertia[1]),
        )

    @staticmethod
    def assets(*, collision: CollisionModel = "full") -> list[ET.Element]:
        node = shape_xml().find("asset")
        assert node is not None
        assets = deepcopy(list(node))
        for asset in assets:
            asset.set("name", PREFIX + asset.attrib["name"])
            if asset.tag == "texture":
                asset.set("file", str(ASSETS / "potsticker-wrapper.png"))
            if asset.tag == "material":
                asset.set("texture", PREFIX + asset.attrib["texture"])
            if asset.tag == "mesh":
                vertices = np.fromstring(asset.attrib["vertex"], sep=" ").reshape(-1, 3)
                asset.set(
                    "vertex",
                    " ".join(
                        f"{value:.10g}" for value in RackPotsticker.place(points=vertices).ravel()
                    ),
                )
        if collision in {"hull", "pouch_hull", "hull_crest"}:
            for mesh, vertices in RackPotsticker.collision_meshes().items():
                assets.append(
                    ET.Element("mesh", name=mesh, vertex=_Mjcf.values(values=vertices.ravel()))
                )
        return assets


class _Mjcf:
    """A static-method container, never instantiated."""

    @staticmethod
    def values(*, values: np.ndarray) -> str:
        return " ".join(f"{value:.10g}" for value in np.ravel(values))

    @staticmethod
    def farthest_points(*, points: np.ndarray, count: int) -> np.ndarray:
        """`count` of `points`, each the farthest from those already chosen, starting from
        the six axis extremes so the decimated hull keeps the full extents."""
        chosen = sorted({
            int(i)
            for axis in range(3)
            for i in (np.argmin(points[:, axis]), np.argmax(points[:, axis]))
        })
        distance = np.min(np.linalg.norm(points[:, None] - points[chosen][None], axis=2), axis=1)
        while len(chosen) < count:
            index = int(np.argmax(distance))
            chosen.append(index)
            distance = np.minimum(distance, np.linalg.norm(points - points[index], axis=1))
        return points[chosen]

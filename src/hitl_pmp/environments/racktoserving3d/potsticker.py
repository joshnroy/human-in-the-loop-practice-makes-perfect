"""Pouring3D's frozen potsticker as the rack's food: its pouch, folds, wrapper and seam.

Every part is collision geometry, exactly as in Pouring3D. The shape is turned a
quarter turn about z and recentred, so the 55 mm long axis lies along body y and
the body origin is the hull's geometric centre: the frame the rack's ellipsoid had
(long axis across the rack, ``face 0`` grasps across the short axis, poses are
geometric centres). Mass, friction and contact dimension stay the rack's own
configuration; the asset's contact ``priority`` is dropped so the rack's surfaces
combine with the food as they did with the ellipsoid.
"""

import xml.etree.ElementTree as ET
from copy import deepcopy
from functools import lru_cache
from typing import ClassVar

import numpy as np

from hitl_pmp.environments.pouring3d.potsticker_shape import ASSETS, shape_xml
from hitl_pmp.environments.pouring3d.potsticker_shape import (
    support_vertices as source_support_vertices,
)

PREFIX = "rack_potsticker_"


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
    def body(
        *, name: str, joint: str, mass: float, friction: str, condim: int, index: int
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
        return body

    @staticmethod
    def assets() -> list[ET.Element]:
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
        return assets

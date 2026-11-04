"""The frozen potsticker's shape, free of any simulator import.

55 x 25 x 27 mm and 20 g are estimates, not measured product specifications.
Pouring3D's scene and RackToServing3D's food both build their bodies from it.
"""

import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

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

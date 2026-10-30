"""Approximate Ozark Trail 15-inch cast iron skillet, with a real hollow collider.

Product: https://www.walmart.com/ip/2142825020
Listed: 15 in diameter, 20.75 in overall length, 2.62 in height, 10.45 lb.
Interior taper, 4.5 mm walls, handles and mass distribution are estimates, not CAD.
No thermal behavior is modeled. The pan has a free joint; weight does not weld it
to the floor. The practice skill set offers individual shrimp recovery, not pan lift.
"""

import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
from kinder.envs.dynamic3d.objects.base import register_object
from kinder.envs.dynamic3d.objects.primitive_objects import Bin


@register_object
class CastIronSkillet(Bin):
    segments = 32

    @staticmethod
    def specification() -> dict[str, Any]:
        return {
            "length": 0.381,
            "width": 0.381,
            "height": 0.066548,
            "wall_thickness": 0.0045,
            "mass": 10.45 * 0.45359237,
            "bottom_radius": 0.15,
            "handle_extension": 0.11605,
            "helper_extension": 0.03,
            "rgba": [0.075, 0.075, 0.08, 1],
            "solref": [0.004, 1],
            "solimp": [0.99, 0.999, 0.001],
        }

    def ring(self, *, height: float, inner: bool) -> np.ndarray:
        angles = np.arange(self.segments) * 2 * np.pi / self.segments
        radius = float(self.options["bottom_radius"])
        radius += (self.length / 2 - radius) * height / self.height
        radius -= self.wall_thickness if inner else 0
        return np.column_stack((
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.full(self.segments, height),
        ))

    def _create_xml_element(self) -> ET.Element:
        body = ET.Element("body", name=self.name)
        ET.SubElement(body, "freejoint", name=self.joint_name)
        self._assets: list[ET.Element] = []
        rings = [
            self.ring(height=z, inner=inner) for z in (0.0, self.height) for inner in (False, True)
        ]
        for index in range(self.segments):
            name = f"{self.name}_wall_{index}"
            vertices = np.concatenate([
                ring[[index, (index + 1) % self.segments]] for ring in rings
            ])
            self._assets.append(
                ET.Element("mesh", name=name, vertex=" ".join(map(str, vertices.ravel())))
            )
            ET.SubElement(
                body,
                "geom",
                name=name,
                type="mesh",
                mesh=name,
                mass=str(self.mass * 0.4 / self.segments),
            )
        # Coplanar boxes avoid capsule/convex-mesh chatter on a flat floor,
        # matching the pitcher's floor construction. Strip edges approximate a disk.
        radius = float(self.options["bottom_radius"])
        for index in range(self.segments):
            x = -radius + (index + 0.5) * 2 * radius / self.segments
            half_width = np.sqrt(radius**2 - x**2)
            ET.SubElement(
                body,
                "geom",
                name=f"{self.name}_bottom_{index}",
                type="box",
                size=f"{radius / self.segments + 1e-6} {half_width} {self.wall_thickness / 2}",
                pos=f"{x} 0 {self.wall_thickness / 2}",
                mass=str(self.mass * 0.5 / self.segments),
            )
        for side, key in ((1, "handle_extension"), (-1, "helper_extension")):
            radius = 0.005
            inner_x = side * (self.length / 2 - 0.005)
            outer_x = side * (self.length / 2 + float(self.options[key]) - radius)
            points = [
                (inner_x, -0.022, self.height - 0.008),
                (outer_x, -0.022, self.height - 0.008),
                (outer_x, 0.022, self.height - 0.008),
                (inner_x, 0.022, self.height - 0.008),
            ]
            for index, (start, end) in enumerate(zip(points[:-1], points[1:], strict=True)):
                ET.SubElement(
                    body,
                    "geom",
                    name=f"{self.name}_handle_{side}_{index}",
                    type="capsule",
                    fromto=" ".join(map(str, (*start, *end))),
                    size=str(radius),
                    mass=str(self.mass * 0.1 / 6),
                )
        for geom in body.iter("geom"):
            geom.set("rgba", self.rgba)
            geom.set("friction", "0.6 0.005 0.0001")
            geom.set("solref", "0.004 1")
            geom.set("solimp", "0.99 0.999 0.001 0.5 2")
        return body

    def get_assets(self) -> list[ET.Element]:
        return self._assets

    def cavity_planes(self) -> tuple[np.ndarray, np.ndarray]:
        bottom = self.ring(height=0, inner=True)
        top = self.ring(height=self.height, inner=True)
        normals = np.cross(np.roll(bottom, -1, axis=0) - bottom, top - bottom)
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        return normals, np.sum(normals * bottom, axis=1)

    def contains_capsules(self, *, endpoints: np.ndarray, radii: np.ndarray) -> bool:
        tolerance = 0.002
        normals, offsets = self.cavity_planes()
        return bool(
            np.min(endpoints[:, 2] - radii) >= self.wall_thickness - tolerance
            and np.max(endpoints[:, 2] + radii) <= self.height + tolerance
            and np.all(endpoints @ normals.T + radii[:, None] <= offsets + tolerance)
        )

    def get_bounding_box_dimensions(self) -> tuple[float, float, float]:
        return (
            self.length
            + float(self.options["handle_extension"])
            + float(self.options["helper_extension"]),
            self.width,
            self.height,
        )

    @staticmethod
    def get_bounding_box_from_config(  # noqa: PLR0917
        pos: np.ndarray, object_config: dict[str, Any]
    ) -> list[float]:
        bounds = Bin.get_bounding_box_from_config(pos, object_config)
        bounds[0] -= float(object_config["helper_extension"])
        bounds[3] += float(object_config["handle_extension"])
        return bounds

"""Task-local capsule geometry, registered without modifying KINDER's wiper."""

import xml.etree.ElementTree as ET
from typing import Any


class CapsuleWiperGeometry:
    @staticmethod
    def body(*, wiper: Any) -> ET.Element:
        radius = wiper.head_height / 2
        half_length = wiper.head_length / 2 - radius
        if radius <= 0 or half_length <= 0 or not wiper.upright:
            raise ValueError("Table wiper requires a positive upright capsule head")
        body = ET.Element("body", name=wiper.name)
        ET.SubElement(body, "freejoint", name=wiper.joint_name)
        width = wiper.handle_width / 2
        ET.SubElement(
            body,
            "geom",
            type="box",
            size=f"{width} {width} {wiper.handle_height / 2}",
            pos=f"0 0 {wiper.handle_height / 2 + wiper.head_height}",
            rgba=wiper.handle_rgba,
            mass=str(wiper.mass / 2),
        )
        ET.SubElement(
            body,
            "geom",
            type="capsule",
            size=f"{radius} {half_length}",
            pos=f"0 0 {radius}",
            quat="0.70710678 0 0.70710678 0",
            rgba=wiper.head_rgba,
            mass=str(wiper.mass / 2),
        )
        return body

    @staticmethod
    def install() -> None:
        from kinder.envs.dynamic3d.objects.base import REGISTERED_OBJECTS, register_object
        from kinder.envs.dynamic3d.objects.primitive_objects import Wiper

        if "table_wiper" in REGISTERED_OBJECTS:
            return

        class CapsuleWiper(Wiper):
            def _create_xml_element(self) -> ET.Element:
                return CapsuleWiperGeometry.body(wiper=self)

        register_object(name="table_wiper")(CapsuleWiper)

"""Native static workstations; the skillet and pitcher remain free bodies."""

import xml.etree.ElementTree as ET

from kinder.envs.dynamic3d.objects.base import register_fixture
from kinder.envs.dynamic3d.objects.fixtures import Table


@register_fixture
class PourStovetop(Table):
    """An unheated electric hob with a flat physical supporting surface."""

    def _create_xml_element(self) -> ET.Element:
        body = super()._create_xml_element()
        h = self.table_height
        ET.SubElement(
            body,
            "geom",
            name=f"{self.name}_cabinet",
            type="box",
            pos=f"0 0 {(h - 0.06) / 2}",
            size=f"0.33 0.38 {(h - 0.06) / 2}",
            rgba="0.55 0.57 0.59 1",
        )
        ET.SubElement(
            body,
            "geom",
            name=f"{self.name}_door",
            type="box",
            pos=f"-0.334 0 {h * 0.46}",
            size="0.004 0.30 0.22",
            rgba="0.07 0.08 0.09 1",
        )
        # Burner markings have no collision geometry above the flat hob.
        for i, (x, y) in enumerate(((-0.15, -0.20), (-0.15, 0.20), (0.15, -0.20), (0.15, 0.20))):
            ET.SubElement(
                body,
                "geom",
                name=f"{self.name}_burner_{i}",
                type="cylinder",
                pos=f"{x} {y} {h + 0.0001}",
                size="0.10 0.0001",
                rgba="0.30 0.31 0.32 1",
                contype="0",
                conaffinity="0",
                group="1",
            )
        for i, y in enumerate((-0.24, -0.08, 0.08, 0.24)):
            ET.SubElement(
                body,
                "geom",
                name=f"{self.name}_knob_{i}",
                type="cylinder",
                pos=f"-0.354 {y} {h - 0.045}",
                size="0.021 0.012",
                quat=".70710678 0 .70710678 0",
                rgba="0.12 0.12 0.13 1",
            )
        return body

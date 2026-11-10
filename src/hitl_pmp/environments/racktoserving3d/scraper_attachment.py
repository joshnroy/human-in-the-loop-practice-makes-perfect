"""Optional parked scraper constraint; contacts with food and fixtures stay physical."""

import xml.etree.ElementTree as ET
from typing import Any

HOME_WELD = "scraper_home_weld"
SCRAPER = "wiper_0"


class ScraperAttachment:
    """Constraint transitions belong to completed grasps, returns and human resets."""

    @staticmethod
    def install(*, core: Any) -> None:
        original = core._create_scene_xml  # noqa: SLF001

        def scene_xml() -> str:
            return ScraperAttachment.scene_xml(xml=original())

        core._create_scene_xml = scene_xml  # noqa: SLF001

    @staticmethod
    def scene_xml(*, xml: str) -> str:
        root = ET.fromstring(xml)
        equality = root.find("equality")
        if equality is None:
            equality = ET.SubElement(root, "equality")
        ET.SubElement(
            equality, "weld", name=HOME_WELD, body1="world", body2=SCRAPER, active="false"
        )
        return ET.tostring(root, encoding="unicode")

    @staticmethod
    def home_id(*, model: Any) -> int:
        return int(model.equality(HOME_WELD).id)

    @staticmethod
    def anchor_home(*, model: Any, data: Any) -> None:
        weld, body = ScraperAttachment.home_id(model=model), model.body(SCRAPER).id
        model.eq_data[weld, 3:6] = data.xpos[body]
        model.eq_data[weld, 6:10] = data.xquat[body]
        data.eq_active[weld] = 1

    @staticmethod
    def release_home(*, model: Any, data: Any) -> None:
        data.eq_active[ScraperAttachment.home_id(model=model)] = 0

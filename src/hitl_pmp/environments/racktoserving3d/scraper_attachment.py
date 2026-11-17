"""Optional parked/held scraper constraints; food and fixture contacts stay physical."""

import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

from hitl_pmp.environments.pouring3d.kitchen.grasp_weld import GraspWeld

from .welded_food import HAND

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
        GraspWeld.add_to_scene(root=root, hand=HAND, names=(SCRAPER,))
        return ET.tostring(root, encoding="unicode")

    @staticmethod
    def home_id(*, model: Any) -> int:
        return int(model.equality(HOME_WELD).id)

    @staticmethod
    def anchor_home(*, model: Any, data: Any) -> None:
        GraspWeld.release(model=model, data=data, names=(SCRAPER,))
        weld, body = ScraperAttachment.home_id(model=model), model.body(SCRAPER).id
        model.eq_data[weld, 3:6] = data.xpos[body]
        model.eq_data[weld, 6:10] = data.xquat[body]
        data.eq_active[weld] = 1

    @staticmethod
    def release_home(*, model: Any, data: Any) -> None:
        data.eq_active[ScraperAttachment.home_id(model=model)] = 0

    @staticmethod
    def attach_grasp(*, model: Any, data: Any) -> None:
        ScraperAttachment.release_home(model=model, data=data)
        GraspWeld.attach(model=model, data=data, hand=HAND, name=SCRAPER)

    @staticmethod
    def grasped(*, model: Any, data: Any) -> bool:
        return GraspWeld.attached(model=model, data=data, names=(SCRAPER,)) == SCRAPER

    @staticmethod
    def release_grasp(*, model: Any, data: Any) -> None:
        GraspWeld.release(model=model, data=data, names=(SCRAPER,))

    @staticmethod
    def state(*, model: Any, data: Any) -> list[float]:
        """Both activity flags and relative transforms, for exact recorded constraints."""
        ids = (ScraperAttachment.home_id(model=model), GraspWeld.weld(model=model, name=SCRAPER))
        return [
            float(value)
            for weld in ids
            for value in np.r_[data.eq_active[weld], model.eq_data[weld]]
        ]

    @staticmethod
    def restore(*, model: Any, data: Any, state: list[float]) -> None:
        ids = (ScraperAttachment.home_id(model=model), GraspWeld.weld(model=model, name=SCRAPER))
        values = np.asarray(state, dtype=float).reshape(2, 12)
        if not np.isfinite(values).all() or not np.isin(values[:, 0], [0, 1]).all():
            raise ValueError("scraper constraints require finite transforms and binary activity")
        if values[:, 0].sum() > 1:
            raise ValueError("scraper home and grasp constraints cannot both be active")
        for weld, row in zip(ids, values, strict=True):
            data.eq_active[weld] = int(row[0])
            model.eq_data[weld] = row[1:]

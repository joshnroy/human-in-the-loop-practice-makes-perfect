"""Task-local spheres, rolling contacts, and settled shallow-tray containment."""

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


class BallScene:
    @staticmethod
    def scene_path() -> Path:
        return Path(__file__).with_name("scene.json")

    @staticmethod
    def install() -> None:
        from kinder.envs.dynamic3d.objects.base import REGISTERED_OBJECTS, register_object
        from kinder.envs.dynamic3d.objects.primitive_objects import Cube

        if "table_ball" in REGISTERED_OBJECTS:
            return

        class TableBall(Cube):
            def _create_xml_element(self) -> ET.Element:
                body = super()._create_xml_element()
                geom = body.find("geom")
                if geom is None:
                    raise ValueError("Ball primitive needs a collision geom")
                geom.set("type", "sphere")
                geom.set("size", str(float(self.size[0])))
                return body

        register_object(name="table_ball")(TableBall)

    @staticmethod
    def configure(*, core: Any) -> None:
        model = core._robot_env.sim.model.mj_model
        for geom in range(model.ngeom):
            body = model.body(int(model.geom_bodyid[geom])).name or ""
            name = model.geom(geom).name or ""
            if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            if (
                body.startswith("cube_")
                or body == "box_0"
                or name.startswith(("floor", "cupboard_1_shelf"))
            ):
                model.geom_condim[geom] = 6
        core.cube_in_box = lambda name: BallScene.in_tray(core=core, name=name)

    @staticmethod
    def on_tray(
        *,
        local_center: np.ndarray,
        radius: float,
        half_x: float,
        half_y: float,
        floor: float,
        upright: float,
    ) -> bool:
        return bool(
            upright >= np.cos(np.deg2rad(5))
            and abs(local_center[0]) + radius <= half_x
            and abs(local_center[1]) + radius <= half_y
            and floor - 0.0001 <= local_center[2] - radius <= floor + 0.001
        )

    @staticmethod
    def in_tray(*, core: Any, name: str) -> bool:
        tray = core.get_object("box_0")
        ball = core.get_object(name)
        tray_position, tray_quaternion = tray.get_pose()
        rotation = Rotation.from_quat(np.asarray(tray_quaternion)[[1, 2, 3, 0]])
        position, _ = ball.get_pose()
        return BallScene.on_tray(
            local_center=rotation.inv().apply(np.asarray(position) - tray_position),
            radius=float(ball.size[0]),
            half_x=tray.length / 2 - tray.wall_thickness,
            half_y=tray.width / 2 - tray.wall_thickness,
            floor=tray.wall_thickness,
            upright=float(rotation.apply([0, 0, 1])[2]),
        )

    @staticmethod
    def velocities_settled(*, model: Any, data: Any, names: Sequence[str]) -> bool:
        for name in names:
            joint = model.body_jntadr[model.body(name).id]
            address = model.jnt_dofadr[joint]
            velocity = data.qvel[address : address + 6]
            if not np.isfinite(velocity).all():
                return False
            if np.linalg.norm(velocity[:3]) > 0.01 or np.linalg.norm(velocity[3:]) > 0.5:
                return False
        return True

    @staticmethod
    def settled(*, core: Any, names: Sequence[str]) -> bool:
        sim = core._robot_env.sim
        return all(BallScene.in_tray(core=core, name=name) for name in names) and (
            BallScene.velocities_settled(
                model=sim.model.mj_model, data=sim.data.mj_data, names=names
            )
        )

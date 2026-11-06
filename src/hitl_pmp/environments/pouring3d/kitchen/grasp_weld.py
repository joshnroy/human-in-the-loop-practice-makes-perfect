"""A dumpling held rigidly by a MuJoCo weld equality from the hand, frozen at the pose
it is grasped in: the kitchen's grasp (`KitchenScene`), shared with RackToServing3D's
welded food handling."""

import xml.etree.ElementTree as ET
from typing import Any

import numpy as np


class GraspWeld:
    """A static-method container, never instantiated."""

    @staticmethod
    def add_to_scene(*, root: ET.Element, hand: str, names: tuple[str, ...]) -> None:
        """One inactive `grasp_weld_<name>` from `hand` to each named body."""
        equality = root.find("equality")
        if equality is None:
            equality = ET.SubElement(root, "equality")
        for name in names:
            ET.SubElement(
                equality,
                "weld",
                name=f"grasp_weld_{name}",
                body1=hand,
                body2=name,
                active="false",
            )

    @staticmethod
    def weld(*, model: Any, name: str) -> int:
        import mujoco

        return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"grasp_weld_{name}"))

    @staticmethod
    def attach(*, model: Any, data: Any, hand: str, name: str) -> None:
        """Engage `name`'s weld at the piece's current pose relative to `hand`."""
        import mujoco

        hand_id, piece = model.body(hand).id, model.body(name).id
        hand_rotation = data.xmat[hand_id].reshape(3, 3)
        weld = GraspWeld.weld(model=model, name=name)
        model.eq_data[weld, 3:6] = hand_rotation.T @ (data.xpos[piece] - data.xpos[hand_id])
        relative = np.zeros(4)
        hand_inverse = np.zeros(4)
        mujoco.mju_negQuat(hand_inverse, data.xquat[hand_id])
        mujoco.mju_mulQuat(relative, hand_inverse, data.xquat[piece])
        model.eq_data[weld, 6:10] = relative
        data.eq_active[weld] = 1

    @staticmethod
    def attached(*, model: Any, data: Any, names: tuple[str, ...]) -> str | None:
        return next(
            (name for name in names if data.eq_active[GraspWeld.weld(model=model, name=name)]),
            None,
        )

    @staticmethod
    def release(*, model: Any, data: Any, names: tuple[str, ...]) -> None:
        for name in names:
            data.eq_active[GraspWeld.weld(model=model, name=name)] = 0

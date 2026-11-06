"""Welded food handling: grasp quality is not what RackToServing3D studies (Josh,
2026-11-06: "claim pick and place always works and study the actual real question").

The robot still plans and drives its base and arm, checked against the fixtures (the
counter and its legs, the rack, the plate, the scraper) and never against food. Robot
bodies and food never collide in physics (`<contact><exclude>`), so the base can run
over a dumpling on the floor and the hand never shoves a neighbour; food still collides
with the world, the scraper and other food, so sweeps stay physical. A pick closes the
fingers over the dumpling, snaps it upright into the hand and engages the kitchen's
grasp weld (`GraspWeld`); a placement snaps it upright onto its spot, at rest, and
releases the weld.
"""

import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

from hitl_pmp.environments.pouring3d.kitchen.grasp_weld import GraspWeld

HAND = "robot_bracelet_link"
ROBOT_PREFIX = "robot_"
# The held dumpling's centre along the pinch site's approach axis, as the physical
# grasp holds it (measured: 33 mm).
HELD_CENTRE_M = 0.033
# The finger command that closes the pads to a dumpling's width (the physical grasp
# stops at a driver angle of 0.61 of the 2F-85's 0.8 rad).
CLOSED_GRIP = 0.77
# A set-down dumpling's lowest point this far above its support before it settles.
SET_DOWN_GAP_M = 0.0005


class WeldedFood:
    """A static-method container, never instantiated."""

    @staticmethod
    def install(*, core: Any, names: tuple[str, ...]) -> None:
        """Rebuild every reset's scene with a grasp weld per food and no robot-food
        contact. KINDER compiles a fresh scene each reset from this method's XML."""
        original = core._create_scene_xml  # noqa: SLF001

        def scene_xml() -> str:
            return WeldedFood.scene_xml(xml=original(), names=names)

        core._create_scene_xml = scene_xml  # noqa: SLF001

    @staticmethod
    def scene_xml(*, xml: str, names: tuple[str, ...]) -> str:
        root = ET.fromstring(xml)
        GraspWeld.add_to_scene(root=root, hand=HAND, names=names)
        return ET.tostring(root, encoding="unicode")

    @staticmethod
    def separate_robot_and_food(*, model: Any, names: tuple[str, ...]) -> None:
        """No robot geom and food geom ever collide; every other pair collides exactly
        as before. Food moves to its own bit (2); each other non-robot geom gains bit 2
        on whichever side bit 1 already let food in. The robot is an `<include>` the
        scene XML never sees, so this filters the compiled model, once per reset."""
        bodies = [model.body(index).name for index in range(model.nbody)]
        robot = np.array([bodies[body].startswith(ROBOT_PREFIX) for body in model.geom_bodyid])
        food_bodies = {model.body(name).id for name in names}
        food = np.isin(model.geom_bodyid, list(food_bodies))
        contype, conaffinity = model.geom_contype, model.geom_conaffinity
        collidable = (contype != 0) | (conaffinity != 0)
        if np.any((contype[food & collidable] != 1) | (conaffinity[food & collidable] != 1)):
            raise RuntimeError("welded food expects every food geom on contact bit 1 alone")
        if np.any(((contype | conaffinity)[robot] & 2) != 0):
            raise RuntimeError("welded food expects no robot geom on contact bit 2")
        others = collidable & ~robot & ~food
        new_type = contype[others] | np.where(conaffinity[others] & 1, 2, 0)
        new_affinity = conaffinity[others] | np.where(contype[others] & 1, 2, 0)
        contype[others], conaffinity[others] = new_type, new_affinity
        contype[food & collidable] = conaffinity[food & collidable] = 2

    @staticmethod
    def attached(*, backend: Any) -> str | None:
        return GraspWeld.attached(
            model=backend.model(), data=backend.data(), names=tuple(backend.cube_names)
        )

    @staticmethod
    def release(*, backend: Any) -> None:
        GraspWeld.release(
            model=backend.model(), data=backend.data(), names=tuple(backend.cube_names)
        )

    @staticmethod
    def leaned(*, heading: float, lean_deg: float) -> np.ndarray:
        """The pinch-site rotation for a base facing `heading`: pointing down, leaned
        `lean_deg` from vertical away from the base, its x axis (across the pads)
        horizontal and square to the heading."""
        lean = np.deg2rad(lean_deg)
        forward = np.array([np.cos(heading), np.sin(heading), 0.0])
        z = np.cos(lean) * np.array([0.0, 0.0, -1.0]) + np.sin(lean) * forward
        x = np.array([-np.sin(heading), np.cos(heading), 0.0])
        return np.column_stack([x, np.cross(z, x), z])

    @staticmethod
    def snap_into_hand(*, backend: Any, shelf: Any, name: str) -> None:
        """Move `name` upright into the hand, at rest, its short axis across the pads,
        and engage its weld."""
        site = shelf.site_pose()
        across = site[:3, 0]
        rotation = WeldedFood.yawed(yaw=float(np.arctan2(across[1], across[0])))
        WeldedFood._set_pose(
            backend=backend,
            name=name,
            position=site[:3, 3] + HELD_CENTRE_M * site[:3, 2],
            rotation=rotation,
        )
        model, data = backend.model(), backend.data()
        GraspWeld.attach(model=model, data=data, hand=HAND, name=name)

    @staticmethod
    def set_down(*, backend: Any, name: str, xy: np.ndarray, top_z: float) -> None:
        """Release `name`'s weld and place it upright and at rest on a support whose top
        is `top_z`, centred at `xy`, keeping its yaw."""
        model, data = backend.model(), backend.data()
        GraspWeld.release(model=model, data=data, names=(name,))
        rotation = data.xmat[model.body(name).id].reshape(3, 3)
        yaw = float(np.arctan2(rotation[1, 0] - rotation[0, 1], rotation[0, 0] + rotation[1, 1]))
        upright = WeldedFood.yawed(yaw=yaw)
        centre = np.array([float(xy[0]), float(xy[1]), top_z + 0.05])
        WeldedFood._set_pose(backend=backend, name=name, position=centre, rotation=upright)
        bottom = float(backend.corners(name=name)[:, 2].min())
        centre[2] -= bottom - (top_z + SET_DOWN_GAP_M)
        WeldedFood._set_pose(backend=backend, name=name, position=centre, rotation=upright)

    @staticmethod
    def yawed(*, yaw: float) -> np.ndarray:
        return np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ])

    @staticmethod
    def _set_pose(*, backend: Any, name: str, position: np.ndarray, rotation: np.ndarray) -> None:
        import mujoco

        model, data = backend.model(), backend.data()
        joint = model.joint(f"{name}_joint").id
        address, dof = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, np.asarray(rotation, dtype=float).reshape(9))
        data.qpos[address : address + 7] = np.r_[position, quaternion]
        data.qvel[dof : dof + 6] = 0.0
        mujoco.mj_forward(model, data)

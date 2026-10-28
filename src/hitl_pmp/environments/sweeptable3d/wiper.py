"""Physical capsule-wiper pickup and return on the existing table simulation."""

from itertools import product

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import (
    ShelfSkills,
    SkillFailed,
    WiperLost,
)

BODY_REST = (
    Rotation.from_euler("y", -90, degrees=True).as_matrix()
    @ Rotation.from_euler("z", 90, degrees=True).as_matrix()
)
OVERHANG = 0.059
PRE_GRASP = 0.10
LIFT = 0.12
PLACE_CLEARANCE = 0.002


class TableWiper:
    """Attach to one live ShelfSkills without resetting it or replacing its recorder.

    Only initialize_rest writes an object pose, for scene setup or a human reset.
    Pick and place command the robot exclusively through ShelfSkills control steps.
    """

    def __init__(self, *, shelf: ShelfSkills, rest_y: float = -0.42) -> None:
        self.shelf = shelf
        self.core = shelf.core
        self.model = shelf.model
        self.data = shelf.data
        self.rest_y = float(rest_y)
        self.base = np.array([0.60, self.rest_y, 0.0])
        self.top = float(shelf.board[1])
        self.front = float(shelf.front_x)
        heads = [geom for geom in shelf.wiper_geoms if int(self.model.geom_type[geom]) == 3]
        handles = [geom for geom in shelf.wiper_geoms if int(self.model.geom_type[geom]) == 6]
        if len(heads) != 1 or len(handles) != 1:
            raise SkillFailed("TableWiper requires one capsule head and one box handle")
        self.head_geom, self.handle_geom = heads[0], handles[0]
        shelf.head_geom, shelf.handle_geom = self.head_geom, self.handle_geom
        self.radius = float(self.model.geom_size[self.head_geom, 0])
        handle_rotation = self.geom_rotation(geom=self.handle_geom)
        handle_axis = int(np.argmax(self.model.geom_size[self.handle_geom]))
        self.handle_axis = handle_rotation[:, handle_axis].copy()
        offset = self.model.geom_pos[self.handle_geom] - self.model.geom_pos[self.head_geom]
        if float(offset @ self.handle_axis) < 0:
            self.handle_axis *= -1
        self.end_in_body = (
            self.model.geom_pos[self.handle_geom]
            + self.model.geom_size[self.handle_geom, handle_axis] * self.handle_axis
        )
        self.point = self.end_in_body - 0.01 * self.handle_axis
        self.hand_in_body = self.coax_hand_rotation(body_rotation=np.eye(3))
        self.wiper_dof = int(self.model.jnt_dofadr[self.model.body_jntadr[shelf.wiper_body]])

    def rest_pose(self) -> np.ndarray:
        """Level, outward-facing rest pose with measured support on the tabletop."""
        pose = np.eye(4)
        pose[:3, :3] = BODY_REST
        end_offset = BODY_REST @ self.end_in_body
        lowest = min(
            self.capsule_bottom(pose=pose),
            float(self.handle_corners(pose=pose)[:, 2].min()),
        )
        pose[:3, 3] = [
            self.front - OVERHANG - end_offset[0],
            self.rest_y,
            self.top - lowest,
        ]
        return pose

    def initialize_rest(self) -> None:
        """Set up the rest pose; privileged setup/human reset only, never a skill."""
        pose = self.rest_pose()
        quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
        obj = self.core.get_object("wiper_0")
        obj.set_pose(pose[:3, 3], quaternion)
        obj.set_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.shelf.in_hand = None
        self.shelf.sim.forward()
        self.shelf.hold(count=20, label="wiper_initialize_rest")

    def at_home(self) -> bool:
        """Read live support, orientation, overhang, velocity and robot clearance."""
        pose = self.wiper_pose()
        if not np.isfinite(pose).all():
            return False
        handle = pose[:3, :3] @ self.handle_axis
        blade = pose[:3, :3] @ self.geom_rotation(geom=self.head_geom)[:, 2]
        end = self.handle_end(pose=pose)
        overhang = self.front - end[0]
        level_limit = np.sin(np.deg2rad(5.0))
        if abs(handle[2]) > level_limit or abs(blade[2]) > level_limit:
            return False
        if handle[0] > -np.cos(np.deg2rad(5.0)):
            return False
        if not 0.0537 <= overhang <= 0.065:
            return False
        if abs(pose[1, 3] - self.rest_y) > 0.02:
            return False
        bottom = min(
            self.capsule_bottom(pose=pose),
            self.handle_table_bottom(pose=pose),
        )
        if abs(bottom - self.top) > 0.003:
            return False
        velocity = self.data.qvel[self.wiper_dof : self.wiper_dof + 6]
        if not np.isfinite(velocity).all():
            return False
        if np.linalg.norm(velocity[:3]) > 0.01 or np.linalg.norm(velocity[3:]) > 0.1:
            return False
        if self.shelf.in_hand is not None:
            return False
        supported = False
        wiper_geoms = set(self.shelf.wiper_geoms)
        for contact in self.data.contact[: self.data.ncon]:
            for mine, other in (
                (int(contact.geom1), int(contact.geom2)),
                (int(contact.geom2), int(contact.geom1)),
            ):
                if mine not in wiper_geoms:
                    continue
                if other in self.shelf.robot_geoms:
                    return False
                if (
                    other in self.shelf.cupboard_geoms
                    and contact.dist <= 0.001
                    and abs(contact.pos[2] - self.top) <= 0.003
                    and abs(contact.frame[2]) >= 0.9
                ):
                    supported = True
        return supported

    def pick(self) -> None:
        """Approach the live handle end-on, verify the grasp, lift and tilt 45 degrees."""
        shelf = self.shelf
        if shelf.in_hand is not None or shelf.pads_hold_wiper():
            raise SkillFailed("PickWiper requires an empty gripper")
        try:
            shelf.grip = 0.0
            shelf.move_base(target=self.base, label="pick_wiper_base")
            pose = self.wiper_pose()
            rotation = pose[:3, :3] @ self.hand_in_body
            grasp = self.grasp_world(pose=pose)
            pre = grasp - PRE_GRASP * rotation[:, 2]
            self.transit(
                position=pre + [0, 0, 0.15],
                rotation=rotation,
                label="pick_wiper_pre_high",
                holding=False,
            )
            self.line(position=pre, rotation=rotation, label="pick_wiper_pre", holding=False)
            self.line(
                position=grasp,
                rotation=rotation,
                label="pick_wiper_approach",
                holding=False,
                step=0.005,
                settle=60,
            )
            live_grasp = self.grasp_world()
            site = shelf.site_pose()
            live_rotation = self.wiper_pose()[:3, :3] @ self.hand_in_body
            error = float(np.linalg.norm(site[:3, 3] - live_grasp))
            angle = self.rotation_error(actual=site[:3, :3], target=live_rotation)
            if error >= 0.005 or angle >= 5.0:
                raise SkillFailed(
                    f"PickWiper pre-close grasp error {error * 1000:.1f} mm / {angle:.1f} deg"
                )
            shelf.grip = 1.0
            shelf.hold(count=40, label="pick_wiper_close")
            if not shelf.pads_hold_wiper():
                raise SkillFailed("PickWiper: both pads must hold the handle before lifting")
            shelf.in_hand = np.linalg.inv(shelf.site_pose()) @ self.wiper_pose()
            lifted = grasp + [0, 0, LIFT]
            self.line(
                position=lifted,
                rotation=rotation,
                label="pick_wiper_lift",
                holding=True,
            )
            shelf.in_hand = np.linalg.inv(shelf.site_pose()) @ self.wiper_pose()
            sweep_rotation = self.sweep_body_rotation(yaw_deg=0)
            site_rotation = sweep_rotation @ shelf.in_hand[:3, :3].T
            predicted = np.eye(4)
            predicted[:3, :3] = sweep_rotation
            predicted[:3, 3] = lifted + site_rotation @ shelf.in_hand[:3, 3]
            lowest = min(
                self.capsule_bottom(pose=predicted),
                float(self.handle_corners(pose=predicted)[:, 2].min()),
            )
            clearance = max(0.0, self.top + 0.03 - lowest)
            if clearance > 0:
                lifted[2] += clearance
                self.line(
                    position=lifted,
                    rotation=shelf.site_pose()[:3, :3],
                    label="pick_wiper_tilt_clearance",
                    holding=True,
                )
                shelf.in_hand = np.linalg.inv(shelf.site_pose()) @ self.wiper_pose()
            self.line(
                position=lifted,
                rotation=sweep_rotation @ shelf.in_hand[:3, :3].T,
                label="pick_wiper_tilt",
                holding=True,
                steps_per=4,
            )
            shelf.hold(count=10, label="pick_wiper_hold")
            self.require_held(label="pick_wiper_hold")
            angle = self.rotation_error(
                actual=self.wiper_pose()[:3, :3],
                target=sweep_rotation,
            )
            if angle > 5.0:
                raise SkillFailed(f"PickWiper: lifted wiper orientation error {angle:.1f} deg")
        except WiperLost as exc:
            raise SkillFailed(f"PickWiper: {exc}") from exc

    def place(self) -> None:
        """Return using the measured grasp, release, retreat and wait for physical rest."""
        shelf = self.shelf
        self.require_held(label="place_wiper_start")
        shelf.in_hand = np.linalg.inv(shelf.site_pose()) @ self.wiper_pose()
        target_site = self.rest_pose() @ np.linalg.inv(shelf.in_hand)
        rotation, grasp = target_site[:3, :3], target_site[:3, 3]
        try:
            here = shelf.site_pose()
            carry_z = max(float(here[2, 3]), float(grasp[2] + LIFT))
            self.line(
                position=np.array([here[0, 3], here[1, 3], carry_z]),
                rotation=here[:3, :3],
                label="place_wiper_raise",
                holding=True,
            )
            shelf.move_base(target=self.base, label="place_wiper_base")
            above = np.array([grasp[0], grasp[1], carry_z])
            self.transit(
                position=above,
                rotation=self.sweep_body_rotation(yaw_deg=0) @ shelf.in_hand[:3, :3].T,
                label="place_wiper_carry",
                holding=True,
            )
            self.line(
                position=above,
                rotation=rotation,
                label="place_wiper_untilt",
                holding=True,
                steps_per=4,
            )
            self.line(
                position=grasp + [0, 0, PLACE_CLEARANCE],
                rotation=rotation,
                label="place_wiper_lower",
                holding=True,
                step=0.005,
            )
            with shelf.wiper_unwatched():
                shelf.grip = 0.0
                shelf.hold(count=30, label="place_wiper_release")
                shelf.in_hand = None
                pre = grasp - PRE_GRASP * rotation[:, 2]
                self.line(
                    position=pre + [0, 0, PLACE_CLEARANCE],
                    rotation=rotation,
                    label="place_wiper_retreat",
                    holding=False,
                    step=0.005,
                )
                self.line(
                    position=pre + [0, 0, 0.15],
                    rotation=rotation,
                    label="place_wiper_clear",
                    holding=False,
                    step=0.02,
                )
                consecutive = 0
                for _ in range(100):
                    shelf.hold(count=1, label="place_wiper_settle")
                    consecutive = consecutive + 1 if self.at_home() else 0
                    if consecutive >= 10:
                        return
            raise SkillFailed("PlaceWiper: wiper did not come to rest at home with gripper clear")
        except WiperLost as exc:
            raise SkillFailed(f"PlaceWiper: {exc}") from exc

    @staticmethod
    def sweep_body_rotation(*, yaw_deg: float) -> np.ndarray:
        """World body rotation; callers using travel yaw pass its negative here."""
        return (
            Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
            @ Rotation.from_euler("y", -45, degrees=True).as_matrix()
            @ Rotation.from_euler("z", 90, degrees=True).as_matrix()
        )

    def wiper_pose(self) -> np.ndarray:
        return self.shelf.wiper_pose()

    def grasp_world(self, *, pose: np.ndarray | None = None) -> np.ndarray:
        pose = self.wiper_pose() if pose is None else pose
        return pose[:3, :3] @ self.point + pose[:3, 3]

    def handle_end(self, *, pose: np.ndarray | None = None) -> np.ndarray:
        pose = self.wiper_pose() if pose is None else pose
        return pose[:3, :3] @ self.end_in_body + pose[:3, 3]

    def handle_corners(self, *, pose: np.ndarray | None = None) -> np.ndarray:
        pose = self.wiper_pose() if pose is None else pose
        signs = np.array(list(product((-1, 1), repeat=3)))
        local = (signs * self.model.geom_size[self.handle_geom]) @ self.geom_rotation(
            geom=self.handle_geom
        ).T + self.model.geom_pos[self.handle_geom]
        return local @ pose[:3, :3].T + pose[:3, 3]

    def capsule_bottom(self, *, pose: np.ndarray | None = None) -> float:
        pose = self.wiper_pose() if pose is None else pose
        center = pose[:3, :3] @ self.model.geom_pos[self.head_geom] + pose[:3, 3]
        axis = pose[:3, :3] @ self.geom_rotation(geom=self.head_geom)[:, 2]
        return float(
            center[2] - self.model.geom_size[self.head_geom, 1] * abs(axis[2]) - self.radius
        )

    def handle_table_bottom(self, *, pose: np.ndarray | None = None) -> float:
        """Clip the handle at the table edge: its overhang can lie below the board."""
        corners = self.handle_corners(pose=pose)
        heights = [float(corner[2]) for corner in corners if corner[0] >= self.front]
        for index, corner in enumerate(corners):
            for edge in (1, 2, 4):
                other_index = index ^ edge
                if index >= other_index:
                    continue
                other = corners[other_index]
                if (corner[0] - self.front) * (other[0] - self.front) < 0:
                    fraction = (self.front - corner[0]) / (other[0] - corner[0])
                    heights.append(float(corner[2] + fraction * (other[2] - corner[2])))
        return min(heights, default=float("inf"))

    def geom_rotation(self, *, geom: int) -> np.ndarray:
        return Rotation.from_quat(self.model.geom_quat[geom][[1, 2, 3, 0]]).as_matrix()

    @staticmethod
    def coax_hand_rotation(*, body_rotation: np.ndarray) -> np.ndarray:
        handle = body_rotation[:, 2]
        approach = -handle
        closing = np.cross(handle, body_rotation[:, 0])
        closing /= np.linalg.norm(closing)
        return np.column_stack([closing, np.cross(approach, closing), approach])

    @staticmethod
    def rotation_error(*, actual: np.ndarray, target: np.ndarray) -> float:
        return float(np.rad2deg(Rotation.from_matrix(actual.T @ target).magnitude()))

    def require_held(self, *, label: str) -> None:
        if self.shelf.in_hand is None or not self.shelf.pads_hold_wiper():
            self.shelf.in_hand = None
            raise SkillFailed(f"{label}: wiper is not held by both pads")

    def ik(
        self,
        *,
        position: np.ndarray,
        rotation: np.ndarray,
        holding: bool,
        seed: np.ndarray | None = None,
        tolerance: float = 0.008,
    ) -> np.ndarray:
        shelf = self.shelf
        arm, penetration = shelf.ik(
            position=position,
            rotation=rotation,
            base=shelf.base_target,
            holding=holding,
            attempts=12,
            seed=seed,
        )
        pose = shelf.site_at(base=shelf.base_target, arm=arm)
        error = float(np.linalg.norm(pose[:3, 3] - position))
        angle = self.rotation_error(actual=pose[:3, :3], target=rotation)
        if not np.isfinite(arm).all() or not np.isfinite([penetration, error, angle]).all():
            raise SkillFailed("Wiper waypoint IK returned a nonfinite solution")
        if error > tolerance or angle > 2.0:
            raise SkillFailed(f"Wiper waypoint IK error {error * 1000:.1f} mm / {angle:.1f} deg")
        return arm

    def line(
        self,
        *,
        position: np.ndarray,
        rotation: np.ndarray,
        label: str,
        holding: bool,
        step: float = 0.01,
        steps_per: int = 3,
        settle: int = 40,
    ) -> None:
        """Interpolate Cartesian position and rotation on one guarded IK branch."""
        shelf = self.shelf
        start = shelf.site_pose()
        distance = float(np.linalg.norm(position - start[:3, 3]))
        angle = self.rotation_error(actual=start[:3, :3], target=rotation)
        count = max(1, int(np.ceil(max(distance / step, angle / 3.0))))
        slerp = Slerp([0, 1], Rotation.from_matrix([start[:3, :3], rotation]))
        arm = shelf.data.qpos[shelf.qa].copy()
        for fraction in np.linspace(0, 1, count + 1)[1:]:
            waypoint = (1 - fraction) * start[:3, 3] + fraction * position
            previous = arm
            arm = self.ik(
                position=waypoint,
                rotation=slerp(fraction).as_matrix(),
                holding=holding,
                seed=previous,
            )
            jump = float(np.max(np.abs((arm - previous + np.pi) % (2 * np.pi) - np.pi)))
            if jump > 0.6:
                raise SkillFailed(f"{label}: IK branch jump {jump:.2f} rad")
            for _ in range(steps_per):
                shelf.step(arm=arm, base=shelf.base_target, label=label)
        self.settle_hand(
            arm=arm,
            position=position,
            rotation=rotation,
            label=label,
            limit=settle,
        )
        if holding:
            self.require_held(label=label)

    def transit(
        self,
        *,
        position: np.ndarray,
        rotation: np.ndarray,
        label: str,
        holding: bool,
    ) -> None:
        """Joint-space transit in open space before the end-on Cartesian approach."""
        arm = self.ik(position=position, rotation=rotation, holding=holding, tolerance=0.005)
        self.settle_hand(arm=arm, position=position, rotation=rotation, label=label, limit=300)
        if holding:
            self.require_held(label=label)

    def settle_hand(
        self,
        *,
        arm: np.ndarray,
        position: np.ndarray,
        rotation: np.ndarray,
        label: str,
        limit: int,
    ) -> None:
        error, angle = float("inf"), float("inf")
        for _ in range(limit):
            self.shelf.step(arm=arm, base=self.shelf.base_target, label=label)
            pose = self.shelf.site_pose()
            error = float(np.linalg.norm(pose[:3, 3] - position))
            angle = self.rotation_error(actual=pose[:3, :3], target=rotation)
            if error < 0.001 and angle < 0.5:
                return
        if not np.isfinite([error, angle]).all() or error > 0.005 or angle > 2.0:
            raise SkillFailed(
                f"{label}: hand tracking error {error * 1000:.1f} mm / {angle:.1f} deg"
            )

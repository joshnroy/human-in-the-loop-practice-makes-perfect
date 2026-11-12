"""Physical scraper pickup, transport and return for RackToServing3D.

TableWiper is ported from scratch/prototypes/sweeptable-kitchen/src/hitl_pmp/
environments/sweeptable3d/wiper.py. RackScraper adds counter-home configuration
and full-box clearance without modifying the source benchmark. Optional home
attachment anchors the parked scraper; only setup, human reset and a completed
return may write its home pose. Physical handling remains available.
"""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from itertools import product
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import (
    ShelfSkills,
    SkillFailed,
    WiperLost,
)

from .scraper_attachment import ScraperAttachment
from .servo import JointStep
from .transit import (
    ArmTransit,
    geom_half_extents,
    geoms_half_extents,
    segment_samples,
)

BODY_REST = (
    Rotation.from_euler("y", -90, degrees=True).as_matrix()
    @ Rotation.from_euler("z", 90, degrees=True).as_matrix()
)
OVERHANG = 0.059
PRE_GRASP = 0.10
LIFT = 0.12
PLACE_CLEARANCE = 0.002
SCRAPER_RETURN_STANDOFF = 0.05
# Largest commanded change per control step along a planned path: under the per-step
# clip (0.1 rad, `base_step_limit`), so the robot follows each checked straight segment
# rather than an axis-clipped one.
PLANNED_JOINT_STEP = 0.05
PLANNED_BASE_FRACTION = 0.8
PLANNED_JOINT_LAG = 0.03
# A start no plan can leave (`BlockedStart`) is left along a checked line this
# long (straight up first), or a held tool backs off ESCAPE_BACK_M and rises ESCAPE_UP_M;
# failing every line, the base backs away ESCAPE_BACK_M.
ESCAPE_BACK_M = 0.10
ESCAPE_UP_M = 0.06
ESCAPE_STEP_M = 0.005
PLANNED_CATCH_UP_STEPS = 20


@contextmanager
def wiper_unwatched(*, shelf: ShelfSkills) -> Generator[None, None, None]:
    """Port the kitchen release guard for older ShelfSkills, including nesting."""
    watching = shelf.watch_wiper
    shelf.watch_wiper = False
    try:
        yield
    finally:
        shelf.watch_wiper, shelf.wiper_unheld_steps = watching, 0


class RackShelfSkills(ShelfSkills):
    """Shelf servo aware of an optional rigid scraper grasp."""

    scraper_attachment: str = "physical"

    def wiper_unwatched(self) -> Any:
        return wiper_unwatched(shelf=self)

    def pads_hold_wiper(self) -> bool:
        if self.scraper_attachment == "grasp_weld" and ScraperAttachment.grasped(
            model=self.model, data=self.data
        ):
            return True
        return super().pads_hold_wiper()


class BlockedStart:
    """Leaving a start no plan can begin from, shared by every PyBullet-planned skill.

    PyBullet's mirror keeps a 15 mm margin from everything, so a configuration MuJoCo
    shows clear (a hand left just above the rack, a held dumpling resting on its
    wires) can be one no plan begins from, and the skill refuses at 0 steps on every
    call. `tool` is the motion that executes: a `RackScraper` or the food's motion.
    """

    @staticmethod
    def plan(
        *,
        tool: Any,
        make_plan: Callable[[], Any],
        held_name: str | None,
        holding: bool,
        label: str,
        distance: float = ESCAPE_UP_M,
        back_first: bool = False,
        prepare: Callable[[], None] | None = None,
    ) -> Any:
        """`make_plan()`, or, if it is refused from a start the planner's own checks
        reject (`ScraperMotionPlanner.start_blocked`, `held_name` included), the plan
        made once more after `escape`. From a start that is not blocked a refusal
        moves nothing. `prepare` runs first when an escape is needed (an empty food
        hand opens)."""
        from .planning import PlanningRefused, ScraperMotionPlanner

        try:
            return make_plan()
        except (PlanningRefused, SkillFailed) as exc:
            # Only a planner refusal (raised, or wrapped by the food skills) can be a
            # blocked start; any other failure (no free placement) stands as it is.
            refused = isinstance(exc, PlanningRefused) or isinstance(exc.__cause__, PlanningRefused)
            reason = str(exc)
            if not refused:
                raise
            if not ScraperMotionPlanner.start_blocked(tool=tool, held_name=held_name):
                raise SkillFailed(reason) from exc
        if prepare is not None:
            prepare()
        BlockedStart.escape(
            tool=tool,
            label=label,
            holding=holding,
            distance=distance,
            back_first=back_first,
            reason=reason,
        )
        try:
            return make_plan()
        except (PlanningRefused, SkillFailed) as again:
            raise SkillFailed(f"after escaping the start: {again}") from again

    @staticmethod
    def escape(
        *,
        tool: Any,
        label: str,
        holding: bool,
        distance: float,
        back_first: bool,
        reason: str,
    ) -> None:
        """Move the hand, and whatever it holds, out of a blocked start along the first
        line `line_plannable` accepts (every waypoint's IK and joint segment checked by
        the execution guard before the first step), keeping the hand's rotation: with
        `back_first`, back off the base's heading ESCAPE_BACK_M and up ESCAPE_UP_M (a
        held tool's old escape); then `distance` along each of `directions`. Failing
        every line, back the base away, checked by the MuJoCo guard. An escape that
        cannot finish never starts."""
        from .planning import ScraperMotionPlanner

        shelf = tool.shelf
        here = shelf.site_pose()
        rotation = here[:3, :3]
        base = shelf.data.qpos[shelf.ba].copy()
        heading = np.array([np.cos(base[2]), np.sin(base[2]), 0.0])
        offsets = [distance * direction for direction in BlockedStart.directions()]
        if back_first:
            offsets.insert(0, -ESCAPE_BACK_M * heading + [0.0, 0.0, ESCAPE_UP_M])
        for offset in offsets:
            target = here[:3, 3] + offset
            if not tool.line_plannable(
                position=target, rotation=rotation, holding=holding, step=ESCAPE_STEP_M
            ):
                continue
            tool.line(
                position=target,
                rotation=rotation,
                label=label,
                holding=holding,
                step=ESCAPE_STEP_M,
                steps_per=5,
            )
            return
        back = base - ESCAPE_BACK_M * heading
        arm = shelf.data.qpos[shelf.qa].copy()
        ok = ScraperMotionPlanner.mujoco_guard(tool=tool, holding=holding)
        if not all(ok(base=pose, arm=arm) for pose in segment_samples(start=base, target=back)):
            raise SkillFailed(f"{reason}; escape refused: no checked line out, nor back-away")
        tool.follow_base_path(path=[base, back], label=f"{label}_back")

    @staticmethod
    def directions() -> list[np.ndarray]:
        """Straight up, then up and away along each of eight headings, then level."""
        headings = [
            np.array([np.cos(angle), np.sin(angle), 0.0])
            for angle in np.deg2rad(np.arange(0.0, 360.0, 45.0))
        ]
        tilted = [(heading + [0.0, 0.0, 1.0]) / np.sqrt(2.0) for heading in headings]
        return [np.array([0.0, 0.0, 1.0]), *tilted, *headings]


class TableWiper:
    """Attach to one live ShelfSkills without resetting it or replacing its recorder.

    Pick and place drive the robot through ShelfSkills control steps. Optional
    attachment can snap a completed return to its verified seat.
    """

    def __init__(
        self,
        *,
        shelf: ShelfSkills,
        rest_y: float = -0.42,
        home_top: float | None = None,
        home_front: float | None = None,
        base: np.ndarray | None = None,
    ) -> None:
        self.attachment = "physical"
        self.shelf = shelf
        self.core = shelf.core
        self.model = shelf.model
        self.data = shelf.data
        self.rest_y = float(rest_y)
        self.base = np.array([0.60, self.rest_y, 0.0] if base is None else base, dtype=float)
        self.top = float(shelf.board[1])
        self.front = float(shelf.front_x)
        self.home_top = self.top if home_top is None else float(home_top)
        self.home_front = self.front if home_front is None else float(home_front)
        if self.base.shape != (3,) or not np.isfinite(self.base).all():
            raise ValueError("Tool base must be a finite (x, y, heading) vector")
        if not np.isfinite([self.rest_y, self.home_top, self.home_front]).all():
            raise ValueError("Tool home coordinates must be finite")
        heads = [geom for geom in shelf.wiper_geoms if int(self.model.geom_type[geom]) == 3]
        handles = [geom for geom in shelf.wiper_geoms if int(self.model.geom_type[geom]) == 6]
        self.is_bench_scraper = not heads and len(handles) == 2
        if self.is_bench_scraper:
            heads = [max(handles, key=lambda geom: self.model.geom_size[geom, 2])]
            handles = [geom for geom in handles if geom != heads[0]]
        if len(heads) != 1 or len(handles) != 1:
            raise SkillFailed("TableWiper requires a capsule or box blade and one box handle")
        self.head_geom, self.handle_geom = heads[0], handles[0]
        shelf.head_geom, shelf.handle_geom = self.head_geom, self.handle_geom
        self.radius = float(self.model.geom_size[self.head_geom, 0])
        handle_rotation = self.geom_rotation(geom=self.handle_geom)
        handle_axis = int(np.argmax(self.model.geom_size[self.handle_geom]))
        if self.is_bench_scraper:
            handle_axis = 2
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
        if self.is_bench_scraper:
            lowest = self.capsule_bottom(pose=pose)
        pose[:3, 3] = [
            self.home_front - OVERHANG - end_offset[0],
            self.rest_y,
            self.home_top - lowest,
        ]
        return pose

    def initialize_rest(self) -> None:
        """Set up the rest pose; privileged setup/human reset only, never a skill."""
        self.release_attachments()
        pose = self.rest_pose()
        quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
        obj = self.core.get_object("wiper_0")
        obj.set_pose(pose[:3, 3], quaternion)
        obj.set_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.shelf.in_hand = None
        self.shelf.grip = 0.0
        self.shelf.sim.forward()
        self.anchor_home()
        self.shelf.hold(count=20, label="wiper_initialize_rest")

    def release_attachments(self) -> None:
        if self.attachment != "physical":
            ScraperAttachment.release_home(model=self.model, data=self.data)
            ScraperAttachment.release_grasp(model=self.model, data=self.data)

    def anchor_home(self) -> None:
        if self.attachment != "physical":
            ScraperAttachment.anchor_home(model=self.model, data=self.data)

    def accept_grasp(self) -> None:
        """Only called after bilateral handle contact has verified the physical grasp."""
        self.release_attachments()
        if self.attachment == "grasp_weld":
            ScraperAttachment.attach_grasp(model=self.model, data=self.data)

    def park_after_seating(self) -> None:
        """Anchor only a completed return with support and a pose close to its seat."""
        if self.attachment == "physical":
            return
        pose, home = self.wiper_pose(), self.rest_pose()
        if (
            not self.table_support()
            or np.linalg.norm(pose[:3, 3] - home[:3, 3]) > 0.005
            or self.rotation_error(actual=pose[:3, :3], target=home[:3, :3]) > 5.0
        ):
            raise SkillFailed("PlaceWiper: scraper has not reached its home seat")
        self.release_attachments()
        quaternion = Rotation.from_matrix(home[:3, :3]).as_quat()[[3, 0, 1, 2]]
        obj = self.core.get_object("wiper_0")
        obj.set_pose(home[:3, 3], quaternion)
        obj.set_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.shelf.sim.forward()
        self.anchor_home()

    def at_home(self) -> bool:
        """Read live support, orientation, overhang, velocity and robot clearance."""
        pose = self.wiper_pose()
        if not np.isfinite(pose).all():
            return False
        handle = pose[:3, :3] @ self.handle_axis
        blade_axis = 0 if self.is_bench_scraper else 2
        blade = pose[:3, :3] @ self.geom_rotation(geom=self.head_geom)[:, blade_axis]
        end = self.handle_end(pose=pose)
        overhang = self.home_front - end[0]
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
        if abs(bottom - self.home_top) > 0.003:
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
                    and abs(contact.pos[2] - self.home_top) <= 0.003
                    and abs(contact.frame[2]) >= 0.9
                ):
                    supported = True
        return supported

    def pick(self) -> None:
        """Plan the base and the arm to a hover in front of the handle (PyBullet) and
        follow both, then approach the live handle end-on, verify the grasp, lift and
        tilt 45 degrees as scripted contact motions. A refused plan moves nothing."""
        from .planning import ScraperMotionPlanner

        shelf = self.shelf
        if shelf.in_hand is not None or shelf.pads_hold_wiper():
            raise SkillFailed("PickWiper requires an empty gripper")
        try:
            pose = self.wiper_pose()
            rotation = pose[:3, :3] @ self.hand_in_body
            grasp = self.grasp_world(pose=pose)
            pre = grasp - PRE_GRASP * rotation[:, 2]

            # A food pick that failed can leave the empty hand just above the rack,
            # within the planner's margin of the food on it, so no base path can start.
            plan = BlockedStart.plan(
                tool=self,
                make_plan=lambda: ScraperMotionPlanner.plan_pick(
                    tool=self,
                    grasp=grasp,
                    pre=pre,
                    rotation=rotation,
                    preferred=self.base,
                    after=self.lift_waypoints(pose=pose),
                ),
                held_name=None,
                holding=False,
                label="pick_wiper_escape",
            )
            shelf.grip = 0.0
            self.follow_base_path(path=plan.base_path, label="pick_wiper_base")
            self.follow_arm_path(path=plan.arm_after_base, label="pick_wiper_pre")
            self.settle_hand(
                arm=plan.arm_after_base[-1],
                position=pre,
                rotation=rotation,
                label="pick_wiper_pre",
                limit=300,
            )
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
            if not self.pads_hold_grip():
                raise SkillFailed("PickWiper: both pads must hold the handle before lifting")
            self.accept_grasp()
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
            clearance = max(0.0, max(self.top, self.home_top) + 0.03 - lowest)
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

    def lift_waypoints(self, *, pose: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """The hand poses the scripted lift and tilt pass through from the grasp of a
        scraper at `pose`, with the nominal grasp, so a stance can be checked for them
        before anything moves."""
        site = np.eye(4)
        site[:3, :3], site[:3, 3] = self.hand_in_body, self.point
        in_hand = np.linalg.inv(site)
        rotation = pose[:3, :3] @ self.hand_in_body
        lifted = self.grasp_world(pose=pose) + [0, 0, LIFT]
        sweep_rotation = self.sweep_body_rotation(yaw_deg=0)
        site_rotation = sweep_rotation @ in_hand[:3, :3].T
        predicted = np.eye(4)
        predicted[:3, :3] = sweep_rotation
        predicted[:3, 3] = lifted + site_rotation @ in_hand[:3, 3]
        lowest = min(
            self.capsule_bottom(pose=predicted),
            float(self.handle_corners(pose=predicted)[:, 2].min()),
        )
        raised = lifted + [0, 0, max(0.0, max(self.top, self.home_top) + 0.03 - lowest)]
        return [(lifted, rotation), (raised, rotation), (raised, site_rotation)]

    def place(self) -> None:
        """Return using the measured grasp and wait for physical rest after release.

        Planned before anything moves (PyBullet, the scraper carried rigidly): the arm
        raised if the base cannot move with it as it is, the base to a stance from which
        the seat is reachable with every joint clear of its limit, and the arm to a hover
        above the seat. The lowering, seat, release and retreat stay scripted. Untilting
        can shift the physical grasp, so it is remeasured before lowering rather than
        commanding a stale tool pose. A refused plan moves nothing.
        """
        from .planning import ScraperMotionPlanner

        shelf = self.shelf
        self.require_held(label="place_wiper_start")
        shelf.in_hand = np.linalg.inv(shelf.site_pose()) @ self.wiper_pose()
        target_site = self.rest_pose() @ np.linalg.inv(shelf.in_hand)
        rotation, grasp = target_site[:3, :3], target_site[:3, 3]
        preferred = self.base.copy()
        if self.is_bench_scraper:
            preferred[0] -= SCRAPER_RETURN_STANDOFF

        # A sweep that failed mid-stroke leaves the blade on the rack. Back the held
        # tool away from the base's heading and up first, as the scripted return did.
        plan_and_hover = BlockedStart.plan(
            tool=self,
            make_plan=lambda: ScraperMotionPlanner.plan_return(
                tool=self,
                seat=grasp + [0, 0, PLACE_CLEARANCE],
                rotation=rotation,
                preferred=preferred,
                carry_height=float(grasp[2] + LIFT),
            ),
            held_name="wiper_0",
            holding=True,
            label="place_wiper_escape",
            back_first=True,
        )
        plan, hover = plan_and_hover
        try:
            self.follow_arm_path(path=plan.arm_before_base, label="place_wiper_raise")
            self.require_held(label="place_wiper_raise")
            self.follow_base_path(path=plan.base_path, label="place_wiper_base")
            self.follow_arm_path(path=plan.arm_after_base, label="place_wiper_carry")
            self.settle_hand(
                arm=plan.arm_after_base[-1],
                position=hover,
                rotation=rotation,
                label="place_wiper_carry",
                limit=300,
            )
            self.require_held(label="place_wiper_carry")
            if self.is_bench_scraper:
                self.require_held(label="place_wiper_remeasure")
                shelf.in_hand = np.linalg.inv(shelf.site_pose()) @ self.wiper_pose()
                target_site = self.rest_pose() @ np.linalg.inv(shelf.in_hand)
                rotation, grasp = target_site[:3, :3], target_site[:3, 3]
            self.line(
                position=grasp + [0, 0, PLACE_CLEARANCE],
                rotation=rotation,
                label="place_wiper_lower",
                holding=True,
                step=0.005,
            )
            if self.is_bench_scraper:
                self.seat_scraper(position=grasp + [0, 0, PLACE_CLEARANCE], rotation=rotation)
            self.park_after_seating()
            with wiper_unwatched(shelf=shelf):
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

    def table_support(self) -> bool:
        """Require a physical upward support force at the board surface."""
        shelf = self.shelf
        force = np.zeros(6)
        for index, contact in enumerate(shelf.data.contact[: shelf.data.ncon]):
            pair = {int(contact.geom1), int(contact.geom2)}
            if (
                pair & set(shelf.wiper_geoms)
                and pair & shelf.cupboard_geoms
                and contact.dist <= 0
                and abs(contact.pos[2] - self.home_top) <= 0.003
                and abs(contact.frame[2]) >= 0.9
            ):
                shelf.mujoco.mj_contactForce(shelf.model, shelf.data, index, force)
                if force[0] > 0.01:
                    return True
        return False

    def seat_scraper(self, *, position: np.ndarray, rotation: np.ndarray) -> None:
        """Seek board support over at most six millimeters, without releasing early."""
        del position, rotation
        for attempt in range(7):
            self.require_held(label="place_wiper_seat")
            if self.table_support():
                return
            if attempt == 6:
                break
            self.shelf.in_hand = np.linalg.inv(self.shelf.site_pose()) @ self.wiper_pose()
            desired_body = self.rest_pose().copy()
            desired_body[2, 3] += PLACE_CLEARANCE - 0.001 * (attempt + 1)
            target = desired_body @ np.linalg.inv(self.shelf.in_hand)
            self.line(
                position=target[:3, 3],
                rotation=target[:3, :3],
                label="place_wiper_seat",
                holding=True,
                step=0.001,
            )
        raise SkillFailed("PlaceWiper: no board support; retaining scraper")

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

    @contextmanager
    def compensate_payload(self) -> Generator[None, None, None]:
        """Compensate the held scraper's weight through bounded robot joint torques."""
        if not self.is_bench_scraper:
            yield
            return
        shelf = self.shelf
        robot = shelf.core._robot_env
        original = robot._compute_arm_torques
        jacobian = np.zeros((3, shelf.model.nv))
        hand_body = int(shelf.model.site_bodyid[shelf.site])
        payload = float(shelf.model.body_mass[shelf.wiper_body])

        def compensated(*args: Any, **kwargs: Any) -> np.ndarray:
            torques = np.array(original(*args, **kwargs), copy=True)
            if shelf.in_hand is not None and shelf.pads_hold_wiper():
                shelf.mujoco.mj_jac(
                    shelf.model,
                    shelf.data,
                    jacobian,
                    None,
                    shelf.data.xipos[shelf.wiper_body],
                    hand_body,
                )
                torques += jacobian[:, shelf.va].T @ (-payload * shelf.model.opt.gravity)
            return np.clip(torques, -robot.ARM_TORQUE_LIMITS, robot.ARM_TORQUE_LIMITS)

        robot._compute_arm_torques = compensated
        try:
            yield
        finally:
            robot._compute_arm_torques = original

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
        """Lowest physical head point, including the flat blade in the kitchen scene."""
        pose = self.wiper_pose() if pose is None else pose
        center = pose[:3, :3] @ self.model.geom_pos[self.head_geom] + pose[:3, 3]
        if self.is_bench_scraper:
            rotation = pose[:3, :3] @ self.geom_rotation(geom=self.head_geom)
            return float(center[2] - np.abs(rotation[2]) @ self.model.geom_size[self.head_geom])
        axis = pose[:3, :3] @ self.geom_rotation(geom=self.head_geom)[:, 2]
        return float(
            center[2] - self.model.geom_size[self.head_geom, 1] * abs(axis[2]) - self.radius
        )

    def blade_edge(self, *, pose: np.ndarray | None = None) -> np.ndarray:
        """Endpoints of the flat scraper's lower working edge in world coordinates."""
        if not self.is_bench_scraper:
            raise ValueError("A capsule has no flat blade edge")
        pose = self.wiper_pose() if pose is None else pose
        rotation = pose[:3, :3] @ self.geom_rotation(geom=self.head_geom)
        size = self.model.geom_size[self.head_geom]
        center = pose[:3, :3] @ self.model.geom_pos[self.head_geom] + pose[:3, 3]
        local = np.array([
            [-size[0], -np.copysign(size[1], rotation[2, 1]), -size[2]],
            [size[0], -np.copysign(size[1], rotation[2, 1]), -size[2]],
        ])
        return local @ rotation.T + center

    def handle_table_bottom(self, *, pose: np.ndarray | None = None) -> float:
        """Clip the handle at the table edge: its overhang can lie below the board."""
        corners = self.handle_corners(pose=pose)
        heights = [float(corner[2]) for corner in corners if corner[0] >= self.home_front]
        for index, corner in enumerate(corners):
            for edge in (1, 2, 4):
                other_index = index ^ edge
                if index >= other_index:
                    continue
                other = corners[other_index]
                if (corner[0] - self.home_front) * (other[0] - self.home_front) < 0:
                    fraction = (self.home_front - corner[0]) / (other[0] - corner[0])
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

    def pads_hold_grip(self) -> bool:
        return self.shelf.pads_hold_wiper()

    def require_held(self, *, label: str) -> None:
        if self.shelf.in_hand is None or not self.pads_hold_grip():
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
        rest_above: float = 0.0,
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
            try:
                arm = self.ik(
                    position=waypoint,
                    rotation=slerp(fraction).as_matrix(),
                    holding=holding,
                    seed=previous,
                )
            except SkillFailed as exc:
                raise SkillFailed(f"{label}: Cartesian fraction {fraction:.3f}: {exc}") from exc
            jump = float(np.max(np.abs((arm - previous + np.pi) % (2 * np.pi) - np.pi)))
            if jump > 0.6:
                raise SkillFailed(f"{label}: IK branch jump {jump:.2f} rad")
            for _ in range(steps_per):
                self.step(arm=arm, label=label)
        self.settle_hand(
            arm=arm,
            position=position,
            rotation=rotation,
            label=label,
            limit=settle,
            rest_above=rest_above,
        )
        if holding:
            self.require_held(label=label)

    def line_plannable(
        self, *, position: np.ndarray, rotation: np.ndarray, holding: bool, step: float = 0.01
    ) -> bool:
        """Whether ``line`` to this pose finds every waypoint's IK, checked as ``ik``
        checks it, on one branch; nothing moves."""
        shelf = self.shelf
        start = shelf.site_pose()
        distance = float(np.linalg.norm(position - start[:3, 3]))
        angle = self.rotation_error(actual=start[:3, :3], target=rotation)
        count = max(1, int(np.ceil(max(distance / step, angle / 3.0))))
        slerp = Slerp([0, 1], Rotation.from_matrix([start[:3, :3], rotation]))
        arm = shelf.data.qpos[shelf.qa].copy()
        for fraction in np.linspace(0, 1, count + 1)[1:]:
            previous = arm
            try:
                arm = self.ik(
                    position=(1 - fraction) * start[:3, 3] + fraction * position,
                    rotation=slerp(fraction).as_matrix(),
                    holding=holding,
                    seed=previous,
                )
            except SkillFailed:
                return False
            if float(np.max(np.abs((arm - previous + np.pi) % (2 * np.pi) - np.pi))) > 0.6:
                return False
        return True

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

    def step(self, *, arm: np.ndarray, label: str) -> None:
        self.shelf.step(arm=arm, base=self.shelf.base_target, label=label)

    def move_base(self, *, target: np.ndarray, label: str) -> None:
        self.shelf.move_base(target=target, label=label)

    def follow_base_path(self, *, path: list[np.ndarray], label: str) -> None:
        """Drive the base through every planned waypoint, the arm held where it is."""
        shelf = self.shelf
        arm = shelf.data.qpos[shelf.qa].copy()
        limit = PLANNED_BASE_FRACTION * shelf.base_step_limit
        for start, end in zip(path[:-1], path[1:], strict=True):
            delta = np.asarray(end, dtype=float) - start
            count = max(1, int(np.ceil(np.max(np.abs(delta)) / limit)))
            for index in range(1, count + 1):
                shelf.base_target = start + delta * (index / count)
                self.step(arm=arm, label=label)
        shelf.base_target = np.asarray(path[-1], dtype=float).copy()
        for _ in range(150):
            if np.max(np.abs(shelf.data.qpos[shelf.ba] - shelf.base_target)) < 0.003:
                return
            self.step(arm=arm, label=label)
        raise SkillFailed(f"Base did not track the planned path during {label}")

    def follow_arm_path(self, *, path: list[np.ndarray], label: str) -> None:
        """Drive the arm along each straight joint segment of a checked plan, the short
        way round as the planner checked it."""
        shelf = self.shelf
        current = shelf.data.qpos[shelf.qa].copy()
        for waypoint in path[1:]:
            delta = (np.asarray(waypoint, dtype=float) - current + np.pi) % (2 * np.pi) - np.pi
            count = max(1, int(np.ceil(np.max(np.abs(delta)) / PLANNED_JOINT_STEP)))
            for index in range(1, count + 1):
                point = current + delta * (index / count)
                # Advance only once the arm has caught up: loaded, it lags the command
                # by over 0.1 rad, which takes it off the checked segment.
                for _ in range(PLANNED_CATCH_UP_STEPS):
                    self.step(arm=point, label=label)
                    lag = (point - shelf.data.qpos[shelf.qa] + np.pi) % (2 * np.pi) - np.pi
                    if np.max(np.abs(lag)) < PLANNED_JOINT_LAG:
                        break
            current = current + delta
        for _ in range(200):
            error = (current - shelf.data.qpos[shelf.qa] + np.pi) % (2 * np.pi) - np.pi
            if np.max(np.abs(error)) < 0.01:
                return
            self.step(arm=current, label=label)
        raise SkillFailed(f"Arm did not track the planned path during {label}")

    def settle_hand(
        self,
        *,
        arm: np.ndarray,
        position: np.ndarray,
        rotation: np.ndarray,
        label: str,
        limit: int,
        rest_above: float = 0.0,
    ) -> None:
        """Hold ``arm`` until the hand is on its target, or fail on a tracking error.

        A hand up to ``rest_above`` over its target, on it horizontally and in
        rotation, is not a tracking error: what it holds is resting on something.
        """
        error, angle = float("inf"), float("inf")
        offset = np.full(3, np.inf)
        for _ in range(limit):
            self.step(arm=arm, label=label)
            pose = self.shelf.site_pose()
            offset = pose[:3, 3] - position
            error = float(np.linalg.norm(offset))
            angle = self.rotation_error(actual=pose[:3, :3], target=rotation)
            if error < 0.001 and angle < 0.5:
                return
        resting = 0 < offset[2] <= rest_above and np.linalg.norm(offset[:2]) <= 0.005
        if resting and np.isfinite([error, angle]).all() and angle <= 2.0:
            return
        if not np.isfinite([error, angle]).all() or error > 0.005 or angle > 2.0:
            raise SkillFailed(
                f"{label}: hand tracking error {error * 1000:.1f} mm / {angle:.1f} deg"
            )


SUPPORT_ALLOWANCE = 0.003
# A box overlap this shallow is contact, not penetration: the arm tracks to millimetres,
# and a blade leaving the rack's front edge sags below its top while its back face is
# still microns inside the front face.
GRAZE = 1e-4


def box_overlap(
    *,
    center: np.ndarray,
    rotation: np.ndarray,
    size: np.ndarray,
    other_center: np.ndarray,
    other_rotation: np.ndarray,
    other_size: np.ndarray,
) -> float:
    """Signed separating-axis overlap of two complete oriented boxes, in metres."""
    axes = [*rotation.T, *other_rotation.T]
    axes.extend(np.cross(first, second) for first in rotation.T for second in other_rotation.T)
    delta = other_center - center
    overlap = float("inf")
    for axis in axes:
        length = float(np.linalg.norm(axis))
        if length < 1e-10:
            continue
        axis = axis / length
        projected = (
            np.abs(rotation.T @ axis) @ size
            + np.abs(other_rotation.T @ axis) @ other_size
            - abs(delta @ axis)
        )
        overlap = min(overlap, float(projected))
        if overlap < 0:
            return overlap
    return overlap


class RackScraper(TableWiper):
    """Physical broad scraper with a counter home independent of the rack top.

    Configuration is duck typed: counter_height, counter_front_x and tool_rest_y
    default to 0.9, 0.98 and -0.34 respectively. Explicit home_top, home_front,
    rest_y and base take precedence. top/front retain the rack's sweep geometry;
    rest_pose is the setup callback; optional attachment snaps only setup, human
    recovery and a verified return to this pose.
    The backend owns the actual two-box blade/full-width-handle geometry.
    """

    def __init__(
        self,
        *,
        shelf: ShelfSkills,
        config: Any = None,
        home_top: float | None = None,
        home_front: float | None = None,
        rest_y: float | None = None,
        base: np.ndarray | None = None,
    ) -> None:
        home_top = getattr(config, "counter_height", 0.9) if home_top is None else home_top
        home_front = getattr(config, "counter_front_x", 0.98) if home_front is None else home_front
        rest_y = getattr(config, "tool_rest_y", -0.34) if rest_y is None else rest_y
        super().__init__(
            shelf=shelf, rest_y=rest_y, home_top=home_top, home_front=home_front, base=base
        )
        self.attachment = getattr(config, "scraper_attachment", "physical")
        if isinstance(shelf, RackShelfSkills):
            shelf.scraper_attachment = self.attachment
        elif self.attachment == "grasp_weld":
            raise ValueError("grasp_weld requires RackShelfSkills for held-tool monitoring")
        if not self.is_bench_scraper:
            raise SkillFailed("RackScraper requires a physical box blade and full-width handle")
        self.point = self.end_in_body + 0.010 * self.handle_axis
        blade = self.model.geom_size[self.head_geom]
        handle = self.model.geom_size[self.handle_geom]
        width = float(getattr(config, "scraper_width", 2 * blade[0]))
        if (
            not np.isfinite(np.r_[blade, handle]).all()
            or np.any(blade <= 0)
            or np.any(handle <= 0)
            or not np.isclose(2 * blade[0], width)
            or not np.isclose(handle[0], blade[0])
            or not 0 < 2 * blade[1] <= 0.005
            or blade[1] >= handle[1]
        ):
            raise SkillFailed(
                "RackScraper requires the configured thin blade and full-width handle"
            )

    def pads_hold_grip(self) -> bool:
        """Require bilateral handle contact until a verified grasp is welded."""
        if self.attachment == "grasp_weld" and ScraperAttachment.grasped(
            model=self.model, data=self.data
        ):
            return True
        shelf = self.shelf
        touched = [False] * len(shelf.pad_sides)
        for contact in shelf.data.contact[: shelf.data.ncon]:
            if contact.dist > 0:
                continue
            for pad, other in (
                (int(contact.geom1), int(contact.geom2)),
                (int(contact.geom2), int(contact.geom1)),
            ):
                if other == self.handle_geom:
                    for index, pads in enumerate(shelf.pad_sides):
                        touched[index] |= pad in pads
        return len(touched) >= 2 and all(touched)

    def tool_clear(self, *, pose: np.ndarray, data: Any = None, inflate: float = 0.0) -> bool:
        """Check both complete boxes against current rack, counter and tray geometry.

        Only shallow upward support at the counter home or rack sweep height is
        permitted. Vertical faces and rack members are never removed from the
        collider set. Calculations use transforms, not freejoint pose writes.
        """
        if not np.isfinite(pose).all():
            return False
        data = self.data if data is None else data
        shelf, model = self.shelf, self.model
        obstacles = list(shelf.cupboard_geoms | shelf.box_geoms)
        ids = np.array(obstacles, dtype=int)
        if ids.size:
            # All obstacles' world-axis reach at once, for a broad-phase reject that
            # only ever keeps a superset of what the per-obstacle reject below keeps.
            other_reach = np.einsum(
                "nij,nj->ni",
                np.abs(np.asarray(data.geom_xmat)[ids].reshape(-1, 3, 3)),
                geoms_half_extents(model=model, geoms=ids),
            )
            other_centers = np.asarray(data.geom_xpos)[ids]
        for geom in (self.head_geom, self.handle_geom):
            center = pose[:3, :3] @ model.geom_pos[geom] + pose[:3, 3]
            rotation = pose[:3, :3] @ self.geom_rotation(geom=geom)
            # A planner inflates the boxes to keep a margin for tracking error.
            size = model.geom_size[geom] + inflate
            bottom = float(center[2] - np.abs(rotation[2]) @ size)
            if ids.size:
                reach = np.abs(rotation) @ size + other_reach
                slack = 1e-9 * np.maximum(1.0, np.abs(reach))
                apart = np.any(np.abs(other_centers - center) > reach + slack, axis=1)
                candidates = [
                    o for o, far in zip(obstacles, apart.tolist(), strict=True) if not far
                ]
            else:
                candidates = []
            for obstacle in candidates:
                if not (model.geom_contype[obstacle] or model.geom_conaffinity[obstacle]):
                    continue
                kind = int(model.geom_type[obstacle])
                other_center = data.geom_xpos[obstacle]
                other_rotation = data.geom_xmat[obstacle].reshape(3, 3)
                other_size = geom_half_extents(model=model, geom=obstacle)
                if np.any(
                    np.abs(other_center - center)
                    > np.abs(rotation) @ size + np.abs(other_rotation) @ other_size
                ):
                    continue
                overlap = box_overlap(
                    center=center,
                    rotation=rotation,
                    size=size,
                    other_center=other_center,
                    other_rotation=other_rotation,
                    other_size=other_size,
                )
                if overlap <= GRAZE:
                    continue
                support_top = float(other_center[2] + np.abs(other_rotation[2]) @ other_size)
                support = (
                    obstacle in shelf.cupboard_geoms
                    and (
                        (kind == 6 and abs(other_rotation[2, 2]) >= np.cos(np.deg2rad(2)))
                        or (kind in (3, 5) and abs(other_rotation[2, 2]) <= np.sin(np.deg2rad(2)))
                        or kind == 2
                    )
                    and min(abs(support_top - self.home_top), abs(support_top - self.top)) <= 0.003
                    and bottom >= support_top - SUPPORT_ALLOWANCE
                )
                # The final sweep slides food over the plate rim on purpose; let the
                # blade ride it as shallowly as it may ride the rack, never from the side.
                rides_plate = (
                    obstacle in shelf.box_geoms and bottom >= support_top - SUPPORT_ALLOWANCE
                )
                if not (support or rides_plate):
                    return False
        return True

    def check_configuration(
        self, *, base: np.ndarray, arm: np.ndarray, holding: bool, inflate: float = 0.0
    ) -> None:
        """Reject robot collision and complete carried geometry in scratch kinematics."""
        shelf = self.shelf
        if shelf.ik_data is shelf.data:
            raise SkillFailed("Scraper clearance requires separate scratch data")
        if not np.isfinite(np.r_[base, arm]).all():
            raise SkillFailed("Scraper clearance requires finite joint targets")
        pose = shelf.site_at(base=base, arm=arm)
        collision = shelf.collision_cost(d=shelf.ik_data, holding=holding)
        if not np.isfinite(collision) or collision > 0:
            pairs = []
            contacts = getattr(shelf.ik_data, "contact", ())
            for contact in contacts[: getattr(shelf.ik_data, "ncon", 0)]:
                if contact.dist <= 0 and (
                    contact.geom1 in shelf.robot_geoms or contact.geom2 in shelf.robot_geoms
                ):
                    names = [
                        shelf.model.geom(geom).name
                        or f"{shelf.model.body(shelf.model.geom_bodyid[geom]).name}/geom_{geom}"
                        for geom in (contact.geom1, contact.geom2)
                    ]
                    pairs.append((*names, round(float(contact.dist), 5)))
            raise SkillFailed(f"Scraper motion would collide, cost {collision:.5f}: {pairs[:8]}")
        if holding:
            if shelf.in_hand is None:
                raise SkillFailed("Scraper clearance requires a measured grasp")
            if not self.tool_clear(pose=pose @ shelf.in_hand, data=shelf.ik_data, inflate=inflate):
                raise SkillFailed("Scraper blade or handle would collide with fixture geometry")

    def check_arm_segment(
        self,
        *,
        start: np.ndarray,
        target: np.ndarray,
        base: np.ndarray,
        holding: bool,
        axis_clipped: bool = True,
    ) -> None:
        """Check the wrapped, straight joint path `JointStep` drives the arm along.

        The axis-clipped path is still checked for a move being planned although the
        servo no longer follows it: a target only reachable past it is refused, and
        that refusal is what sends a blade resting on the plate rim to retreat before
        it lifts. Dropping the pass made that lift plannable from the stretched rear
        stance, where it then failed on hand tracking (6.0 mm).

        A transit re-checking a segment of a route it has already planned passes
        ``axis_clipped=False``: the arm starts up to 0.02 rad off the planned waypoint,
        and from there the path it will not take can graze what the planned one
        cleared (a Sweep refused 30 steps in, one blade corner 0.07 mm into the rack).
        """
        delta = (target - start + np.pi) % (2 * np.pi) - np.pi
        count = max(1, int(np.ceil(np.max(np.abs(delta)) / 0.025)))
        for fraction in np.linspace(0, 1, count + 1):
            try:
                self.check_configuration(base=base, arm=start + fraction * delta, holding=holding)
            except SkillFailed as exc:
                raise SkillFailed(f"straight joint segment fraction {fraction:.3f}: {exc}") from exc
        if not axis_clipped:
            return
        for distance in np.arange(0, np.max(np.abs(delta)), 0.025):
            try:
                self.check_configuration(
                    base=base,
                    arm=start + np.sign(delta) * np.minimum(np.abs(delta), distance),
                    holding=holding,
                )
            except SkillFailed as exc:
                raise SkillFailed(
                    f"axis-clipped joint segment at {distance:.3f} rad: {exc}"
                ) from exc

    def ik(
        self,
        *,
        position: np.ndarray,
        rotation: np.ndarray,
        holding: bool,
        seed: np.ndarray | None = None,
        tolerance: float = 0.008,
    ) -> np.ndarray:
        arm = super().ik(
            position=position, rotation=rotation, holding=holding, seed=seed, tolerance=tolerance
        )
        start = self.shelf.data.qpos[self.shelf.qa].copy() if seed is None else seed
        self.check_arm_segment(
            start=start, target=arm, base=self.shelf.base_target, holding=holding
        )
        return arm

    def step(self, *, arm: np.ndarray, label: str) -> None:
        """Recheck measured and next commanded configurations against mutable fixtures."""
        shelf = self.shelf
        holding = shelf.in_hand is not None
        if holding:
            self.require_held(label=label)
        current_arm = shelf.data.qpos[shelf.qa].copy()
        current_base = shelf.data.qpos[shelf.ba].copy()
        next_arm = JointStep.toward(current=current_arm, target=arm)
        next_base = current_base + np.clip(
            shelf.base_target - current_base, -shelf.base_step_limit, shelf.base_step_limit
        )
        for fraction in np.linspace(0, 1, 5):
            self.check_configuration(
                base=(1 - fraction) * current_base + fraction * next_base,
                arm=(1 - fraction) * current_arm + fraction * next_arm,
                holding=holding,
            )
        super().step(arm=arm, label=label)

    def transit(
        self,
        *,
        position: np.ndarray,
        rotation: np.ndarray,
        label: str,
        holding: bool,
    ) -> None:
        """Use bounded joint detours only for transit, never for approach or lowering."""
        shelf = self.shelf
        if holding:
            self.require_held(label=label)
        try:
            target = super().ik(
                position=position, rotation=rotation, holding=holding, tolerance=0.005
            )
            planner = ArmTransit(tool=self, base=shelf.base_target, holding=holding)
            route = planner.plan(start=shelf.data.qpos[shelf.qa].copy(), target=target)
            if route is None:
                raise SkillFailed(
                    f"no safe arm transit within planning budget ({planner.edge_checks} edges): "
                    f"{planner.last_failure}"
                )
            steps = 0
            for waypoint in route[:-1]:
                self.check_arm_segment(
                    start=shelf.data.qpos[shelf.qa].copy(),
                    target=waypoint,
                    base=shelf.base_target,
                    holding=holding,
                    axis_clipped=False,
                )
                for _ in range(80):
                    error = planner.delta(start=shelf.data.qpos[shelf.qa], target=waypoint)
                    if np.max(np.abs(error)) < 0.02:
                        break
                    if steps >= 360:
                        raise SkillFailed("arm transit exhausted its 360-step detour budget")
                    self.step(arm=waypoint, label=label)
                    steps += 1
                else:
                    raise SkillFailed("arm transit waypoint failed to converge")
            self.check_arm_segment(
                start=shelf.data.qpos[shelf.qa].copy(),
                target=target,
                base=shelf.base_target,
                holding=holding,
                axis_clipped=False,
            )
            self.settle_hand(
                arm=target, position=position, rotation=rotation, label=label, limit=300
            )
            if holding:
                self.require_held(label=label)
        except SkillFailed as exc:
            raise SkillFailed(f"{label}: {exc}") from exc

    def move_base(self, *, target: np.ndarray, label: str) -> None:
        """Preflight direct and axis-separated routes with full held-tool geometry."""
        shelf = self.shelf
        arm = shelf.arm_reference.copy()
        start = shelf.data.qpos[shelf.ba].copy()
        target = np.asarray(target, dtype=float).copy()
        holding = shelf.in_hand is not None
        outside = min(start[0], target[0]) - 0.10
        candidates = [
            [target],
            [np.array([start[0], target[1], start[2]]), target],
            [np.array([target[0], start[1], start[2]]), target],
            [
                np.array([outside, start[1], start[2]]),
                np.array([outside, target[1], start[2]]),
                target,
            ],
        ]
        route = None
        failure = "no candidate route"
        for candidate in candidates:
            try:
                for begin, end in zip([start, *candidate[:-1]], candidate, strict=True):
                    for base in segment_samples(start=begin, target=end):
                        self.check_configuration(base=base, arm=arm, holding=holding)
            except SkillFailed as exc:
                failure = str(exc)
                continue
            route = candidate
            break
        if route is None:
            raise SkillFailed(f"{label}: no safe base transit: {failure}")
        for waypoint in route:
            current = shelf.data.qpos[shelf.ba].copy()
            for base in segment_samples(start=current, target=waypoint):
                self.check_configuration(base=base, arm=arm, holding=holding)
            shelf.base_target = waypoint.copy()
            distance = np.max(np.abs(waypoint - current))
            # SweepServo ramps a held scraper's heading across each control period,
            # so a turn advances about 0.8 base_step_limit per step, not 1.0.
            limit = max(150, int(np.ceil(2 * distance / shelf.base_step_limit)) + 50)
            for _ in range(limit):
                self.step(arm=arm, label=label)
                if np.max(np.abs(shelf.data.qpos[shelf.ba] - waypoint)) < 0.003:
                    break
            else:
                raise SkillFailed(f"Base failed to converge during {label}")

"""Individual food manipulation using robot commands, never object pose writes.

Integrate with ``RackFoodController(controllers=current_controllers).pick(name=...)``
and ``.place(name=..., destination='rack_0' or 'box_0')``. Failures retain the
physical state for observation or the existing explicit human recovery skills.
The scraper's ``holding`` flag denotes a scraper, not food; this adapter keeps
it false and checks the measured food payload separately.

The default rigid 25-mm-high ellipsoid uses a 31-mm pinch-site offset at 45
degrees and a force-limited close (``food_grip_force``). Native finger contacts
and the observed food extent govern holding and placement; rack-wire and fixture
collision failures are retained rather than weakening clearance guards.
"""

from collections.abc import Generator
from contextlib import contextmanager
from functools import cached_property
from time import monotonic
from typing import Any, ClassVar

import numpy as np
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import Frames, SkillFailed

from .semantics import CircularSupport
from .tool import RackScraper
from .transit import ArmTransit, geom_half_extents, segment_samples

# How far behind a stance, along its own heading, a counter circulation may enter it.
APPROACH_BACKOFFS = (0.0, 0.1, 0.2, 0.3)
# Hand yaws about world z a placement may set the held food down with, from the grasp's.
PLACEMENT_TURNS_DEG = (0.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0, 180.0)
# Pad normal force per unit of the 2F-85 finger actuator's tendon force, measured in
# this scene: at the model's own +5 limit each pad presses a held dumpling with ~46 N.
FINGER_NEWTONS_PER_ACTUATOR_UNIT = 46.0 / 5.0
FINGER_ACTUATOR_SUFFIX = "fingers_actuator"
# How far food may have moved since its grasp was planned and still be closed on.
REGRASP_POSITION_TOLERANCE = 0.002
REGRASP_ANGLE_TOLERANCE_DEG = 3.0
REGRASP_ATTEMPTS = 2
# How much a pad-fixture contact present when a lift starts may shrink (measured-state
# jitter) and still count as the same contact the lift is leaving.
DEPARTURE_TOLERANCE = 1e-4


class _FoodMotion(RackScraper):
    """Reuse guarded scraper routes, adding food collisions and live grasp checks."""

    def __init__(self, *, food: "RackFoodController") -> None:
        self.food = food
        self.departing: dict[tuple[int, int], float] = {}
        super().__init__(shelf=food.shelf, config=food.backend.config)

    def check_configuration(
        self, *, base: np.ndarray, arm: np.ndarray, holding: bool, inflate: float = 0.0
    ) -> None:
        del inflate  # the food check carries no scraper
        try:
            super().check_configuration(base=base, arm=arm, holding=False)
        except SkillFailed:
            if not self.departing or self._undeparted_cost() > 0:
                raise
        shelf, backend = self.shelf, self.food.backend
        scratch = shelf.ik_data
        carried = self.food.carried
        foods = {
            geom
            for name in backend.cube_names
            if name != carried
            for geom in backend.body_geoms(name=name)
        }
        for contact in scratch.contact[: scratch.ncon]:
            if contact.dist > 0:
                continue
            for robot, other in (
                (int(contact.geom1), int(contact.geom2)),
                (int(contact.geom2), int(contact.geom1)),
            ):
                if (
                    robot in shelf.robot_geoms
                    and other in foods
                    and not (robot in shelf.pad_geoms and other in self.food.target_geoms)
                ):
                    raise SkillFailed("Food motion would collide with ungrasped food")
        if carried is None:
            return
        live_hand = shelf.site_pose()
        target_hand = shelf.site_at(base=base, arm=arm)
        transform = target_hand @ np.linalg.inv(live_hand)
        obstacles = shelf.cupboard_geoms | shelf.box_geoms | set(shelf.wiper_geoms) | foods
        for geom in backend.body_geoms(name=carried):
            center = transform[:3, :3] @ shelf.data.geom_xpos[geom] + transform[:3, 3]
            rotation = transform[:3, :3] @ shelf.data.geom_xmat[geom].reshape(3, 3)
            size = shelf.model.geom_size[geom]
            bottom = center[2] - (
                np.linalg.norm(rotation[2] * size)
                if int(shelf.model.geom_type[geom]) == 4
                else np.abs(rotation[2]) @ size
            )
            for obstacle in obstacles:
                other_center = scratch.geom_xpos[obstacle]
                other_rotation = scratch.geom_xmat[obstacle].reshape(3, 3)
                other_size = geom_half_extents(model=shelf.model, geom=obstacle)
                if np.any(
                    np.abs(center - other_center)
                    > np.abs(rotation) @ size + np.abs(other_rotation) @ other_size
                ):
                    continue
                overlap = self._penetration(
                    scratch=scratch, geom=geom, obstacle=obstacle, center=center, rotation=rotation
                )
                top = other_center[2] + np.abs(other_rotation[2]) @ other_size
                support = obstacle in shelf.cupboard_geoms | shelf.box_geoms
                if overlap > 0 and not (support and bottom >= top - 0.003):
                    raise SkillFailed(
                        f"Carried food would collide with {shelf.model.geom(obstacle).name}: "
                        f"overlap={overlap:.4f}, bottom={bottom:.4f}, top={top:.4f}"
                    )

    def _penetration(
        self,
        *,
        scratch: Any,
        geom: int,
        obstacle: int,
        center: np.ndarray,
        rotation: np.ndarray,
    ) -> float:
        """Depth of the carried geom, moved to its planned pose, into `obstacle`.

        MuJoCo's own narrow phase on the real shapes: an ellipsoid dumpling is not its
        bounding box, so two neighbours with a visible gap do not count as colliding.
        """
        import mujoco

        saved_pos, saved_mat = scratch.geom_xpos[geom].copy(), scratch.geom_xmat[geom].copy()
        try:
            scratch.geom_xpos[geom] = center
            scratch.geom_xmat[geom] = rotation.reshape(9)
            distance = mujoco.mj_geomDistance(self.shelf.model, scratch, geom, obstacle, 0.01, None)
        finally:
            scratch.geom_xpos[geom], scratch.geom_xmat[geom] = saved_pos, saved_mat
        return max(0.0, -float(distance))

    def pad_fixture_contacts(self) -> dict[tuple[int, int], float]:
        """Closest distance of each (pad, fixture) pair among scratch data's contacts."""
        shelf = self.shelf
        scratch = shelf.ik_data
        fixed = shelf.cupboard_geoms | shelf.box_geoms
        pairs: dict[tuple[int, int], float] = {}
        for contact in scratch.contact[: scratch.ncon]:
            for robot, other in (
                (int(contact.geom1), int(contact.geom2)),
                (int(contact.geom2), int(contact.geom1)),
            ):
                if robot in shelf.pad_geoms and other in fixed:
                    pairs[robot, other] = min(pairs.get((robot, other), np.inf), contact.dist)
        return pairs

    def _undeparted_cost(self) -> float:
        """``collision_cost`` of scratch data's contacts, less the departing ones.

        A departing contact is a (pad, fixture) pair recorded when the lift began
        whose distance has not shrunk by more than ``DEPARTURE_TOLERANCE``. Every
        other term counts exactly as ``ShelfSkills.collision_cost`` counts it.
        """
        shelf = self.shelf
        scratch = shelf.ik_data
        fixed = shelf.cupboard_geoms | shelf.box_geoms
        wiper = set(shelf.wiper_geoms)
        robot = shelf.robot_geoms
        cost = 0.0
        for contact in scratch.contact[: scratch.ncon]:
            a, b = int(contact.geom1), int(contact.geom2)
            depth = max(0.0, -contact.dist)
            pair = (a, b) if a in robot else (b, a)
            start = self.departing.get(pair)
            if start is not None and contact.dist >= start - DEPARTURE_TOLERANCE:
                continue
            if a in robot and b in robot:
                cost += max(0.0, depth - 0.001)
            elif (a in robot and b in fixed) or (b in robot and a in fixed):
                cost += depth + 0.001
            elif (a in robot and b in wiper and a not in shelf.pad_geoms) or (
                b in robot and a in wiper and b not in shelf.pad_geoms
            ):
                cost += depth
        return cost

    @contextmanager
    def departing_from_contacts(self) -> Generator[None, None, None]:
        """Within this scope, pad-fixture contacts present now may persist, not deepen.

        A closed grasp on the wire rack can leave a pad resting on a wire. The lift
        moves away from that contact, but the unconditioned check rejects the lift's
        own starting configuration.
        """
        shelf = self.shelf
        shelf.site_at(base=shelf.data.qpos[shelf.ba].copy(), arm=shelf.data.qpos[shelf.qa].copy())
        shelf.mujoco.mj_collision(shelf.model, shelf.ik_data)
        self.departing = self.pad_fixture_contacts()
        try:
            yield
        finally:
            self.departing = {}

    def step(self, *, arm: np.ndarray, label: str) -> None:
        if self.food.carried is not None:
            self.food._require_held(name=self.food.carried)
        super().step(arm=arm, label=label)
        if self.food.carried is not None:
            self.food._require_held(name=self.food.carried)

    def check_arm_clearance(self, *, base: np.ndarray, arm: np.ndarray) -> None:
        """Reserve five millimeters for tracking error on non-gripper robot links."""
        shelf = self.shelf
        model, scratch = shelf.model, shelf.ik_data
        shelf.site_at(base=base, arm=arm)
        margin = 0.005
        for robot in shelf.robot_geoms - shelf.gripper_geoms:
            if not (model.geom_contype[robot] or model.geom_conaffinity[robot]):
                continue
            for obstacle in shelf.cupboard_geoms | shelf.box_geoms:
                if not (model.geom_contype[obstacle] or model.geom_conaffinity[obstacle]):
                    continue
                bound = model.geom_rbound[robot] + model.geom_rbound[obstacle] + margin
                if np.linalg.norm(scratch.geom_xpos[robot] - scratch.geom_xpos[obstacle]) > bound:
                    continue
                separation = shelf.mujoco.mj_geomDistance(
                    model,
                    scratch,
                    robot,
                    obstacle,
                    margin,
                    None,
                )
                if not np.isfinite(separation) or separation < margin:
                    raise SkillFailed("Food waypoint lacks robot/fixture tracking clearance")

    def transit_arm(self, *, arm: np.ndarray, label: str) -> None:
        """Execute the checked joint branch, not another IK solution at the same hand pose."""
        shelf = self.shelf
        base = shelf.base_target.copy()
        pose = shelf.site_at(base=base, arm=arm)
        planner = ArmTransit(tool=self, base=base, holding=False)
        route = planner.plan(start=shelf.data.qpos[shelf.qa].copy(), target=arm)
        if route is None:
            raise SkillFailed(f"{label}: no checked joint transit: {planner.last_failure}")
        steps = 0
        for waypoint in route:
            self.check_arm_segment(
                start=shelf.data.qpos[shelf.qa].copy(),
                target=waypoint,
                base=base,
                holding=False,
            )
            for _ in range(120):
                error = planner.delta(start=shelf.data.qpos[shelf.qa], target=waypoint)
                if np.max(np.abs(error)) < 0.01:
                    break
                if steps >= 360:
                    raise SkillFailed(f"{label}: joint transit exhausted its step budget")
                self.step(arm=waypoint, label=label)
                steps += 1
            else:
                raise SkillFailed(f"{label}: checked joint branch did not converge")
        self.settle_hand(
            arm=arm,
            position=pose[:3, 3],
            rotation=pose[:3, :3],
            label=label,
            limit=40,
        )

    def move_base(self, *, target: np.ndarray, label: str) -> None:
        """Try checked circulation only when direct-route planning failed without motion."""
        try:
            super().move_base(target=target, label=label)
            return
        except SkillFailed as exc:
            if not str(exc).startswith(f"{label}: no safe base transit:"):
                raise
            failure = str(exc)
        shelf = self.shelf
        start = shelf.data.qpos[shelf.ba].copy()
        arm = shelf.arm_reference.copy()
        for route in self.circulation_routes(start=start, target=target):
            try:
                for begin, end in zip([start, *route[:-1]], route, strict=True):
                    for base in segment_samples(start=begin, target=end):
                        self.check_configuration(base=base, arm=arm, holding=False)
            except SkillFailed as exc:
                failure = str(exc)
                continue
            for waypoint in route:
                super().move_base(target=waypoint, label=label)
            return
        raise SkillFailed(f"{label}: no safe food circulation: {failure}")

    def circulation_routes(
        self, *, start: np.ndarray, target: np.ndarray
    ) -> list[list[np.ndarray]]:
        """Routes around either side of the counter that leave and enter stances head-on.

        The base backs out from its start along its own heading, drives sideways past
        the counter, turns to the stance heading, then drives to an entry point behind
        the stance along that heading and straight in. Each back-off is one of
        ``APPROACH_BACKOFFS``, smallest total first. With none, a leg slides along the
        counter's edge at the stance's own distance from it, which clips a corner leg
        whenever the stance is yawed off the edge's normal.
        """
        counter = self.food.backend.support(name="counter")
        leaving = np.array([np.cos(start[2]), np.sin(start[2])])
        arriving = np.array([np.cos(target[2]), np.sin(target[2])])
        pairs = sorted(
            ((out, back) for out in APPROACH_BACKOFFS for back in APPROACH_BACKOFFS),
            key=lambda pair: (pair[0] + pair[1], pair[0]),
        )
        routes = []
        for out, back in pairs:
            exit_ = np.r_[start[:2] - out * leaving, start[2]]
            entry = np.r_[target[:2] - back * arriving, target[2]]
            for outside in (counter.y_min - 0.8, counter.y_max + 0.8):
                route = [
                    *([exit_] if out > 0 else []),
                    np.array([exit_[0], outside, start[2]]),
                    np.array([entry[0], outside, start[2]]),
                    np.array([entry[0], outside, target[2]]),
                    entry,
                    *([target] if back > 0 else []),
                ]
                routes.append(route)
        return routes

    def plan_base(self, *, target: np.ndarray, arm: np.ndarray) -> list[np.ndarray]:
        """Preflight a staged arm's complete base route before selecting its stance."""
        shelf = self.shelf
        start = shelf.data.qpos[shelf.ba].copy()
        outside_x = min(start[0], target[0]) - 0.10
        candidates = [
            [target],
            [np.array([start[0], target[1], start[2]]), target],
            [np.array([target[0], start[1], start[2]]), target],
            [
                np.array([outside_x, start[1], start[2]]),
                np.array([outside_x, target[1], start[2]]),
                target,
            ],
        ]
        candidates.extend(self.circulation_routes(start=start, target=target))
        failure = "no candidate base route"
        for route in candidates:
            try:
                for begin, end in zip([start, *route[:-1]], route, strict=True):
                    for base in segment_samples(start=begin, target=end):
                        self.check_configuration(base=base, arm=arm, holding=False)
            except SkillFailed as exc:
                failure = str(exc)
                continue
            return route
        raise SkillFailed(f"No safe food base route: {failure}")


class RackFoodController:
    """Bounded angled grasp search on the existing full-height counter scene."""

    # How many draws of the shared rack layout a put-back tries, in order, before it
    # fails: the first reachable one is used.
    RACK_PUT_BACK_DRAWS: ClassVar[int] = 15

    def __init__(self, *, controllers: Any) -> None:
        self.controllers = controllers
        self.backend = controllers.backend
        self.shelf = controllers.shelf
        self.carried: str | None = None
        self.target_geoms: set[int] = set()

    @cached_property
    def motion(self) -> _FoodMotion:
        return _FoodMotion(food=self)

    def _validate_name(self, *, name: str) -> None:
        if name not in self.backend.cube_names:
            raise SkillFailed(f"Unknown food: {name}")

    def _pose(self, *, name: str) -> np.ndarray:
        position, quaternion = self.backend.core().get_object(name).get_pose()
        pose = Frames.pose(
            rotation=Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix(),
            position=position,
        )
        if not np.isfinite(pose).all():
            raise SkillFailed(f"Nonfinite observed food pose: {name}")
        return pose

    def _require_held(self, *, name: str) -> None:
        if not self.backend.held(name=name):
            self.controllers.held_food = None
            self.shelf.held_cube = None
            self.carried = None
            self._limit_grip(newtons=None)
            raise SkillFailed(f"Food {name} is not held by bilateral pad contact")

    def restore_gripper_force(self) -> None:
        """End a food grasp's force limit when the food leaves the hand by other means."""
        self._limit_grip(newtons=None)

    def _limit_grip(self, *, newtons: float | None) -> None:
        """Command the fingers' closing force, as the real 2F-85 takes a force setting.

        Position-commanded closing on a rigid ellipsoid keeps driving the fingers in
        at the actuator's full force; a force-limited close stops at the target
        normal force. ``None`` restores the gripper model's own limit, which every
        non-food grasp (the scraper) uses. Only the closing side is changed.
        """
        model = self.shelf.model
        actuator = next(
            index
            for index in range(model.nu)
            if model.actuator(index).name.endswith(FINGER_ACTUATOR_SUFFIX)
        )
        default = getattr(self.shelf, "finger_force_default", None)
        if default is None:
            default = float(model.actuator_forcerange[actuator, 1])
            self.shelf.finger_force_default = default
        model.actuator_forcerange[actuator, 1] = (
            default if newtons is None else newtons / FINGER_NEWTONS_PER_ACTUATOR_UNIT
        )

    def _checked_arm(
        self, *, pose: np.ndarray, base: np.ndarray, seed: np.ndarray | None = None
    ) -> np.ndarray:
        arm, cost = self.shelf.ik(
            position=pose[:3, 3],
            rotation=pose[:3, :3],
            base=base,
            holding=False,
            attempts=4,
            seed=seed,
        )
        actual = self.shelf.site_at(base=base, arm=arm)
        error = np.linalg.norm(actual[:3, 3] - pose[:3, 3])
        angle = self.motion.rotation_error(actual=actual[:3, :3], target=pose[:3, :3])
        if not np.isfinite(np.r_[arm, cost, error, angle]).all() or error > 0.003 or angle > 2:
            raise SkillFailed("Food waypoint IK is inaccurate")
        limited = np.asarray(self.shelf.model.jnt_limited[self.shelf.joints], dtype=bool)
        ranges = self.shelf.model.jnt_range[self.shelf.joints]
        if np.any(arm[limited] <= ranges[limited, 0] + 0.01) or np.any(
            arm[limited] >= ranges[limited, 1] - 0.01
        ):
            raise SkillFailed("Food waypoint IK is too close to a joint limit")
        self.motion.check_configuration(base=base, arm=arm, holding=False)
        if self.carried is not None:
            self.motion.check_arm_clearance(base=base, arm=arm)
        return arm

    def _base_candidates(self, *, centre: np.ndarray, approach: np.ndarray) -> list[np.ndarray]:
        forward = np.asarray(approach[:2], dtype=float)
        length = np.linalg.norm(forward)
        if not np.isfinite(length) or length < 1e-6:
            raise SkillFailed("Food grasp requires a finite angled approach")
        forward = forward / length
        sideways = np.array([-forward[1], forward[0]])
        return [
            np.r_[
                centre[:2] - distance * forward + offset * sideways,
                np.arctan2(forward[1], forward[0]),
            ]
            for distance in (0.50, 0.45, 0.55, 0.60)
            for offset in (0.0, -0.12, 0.12)
        ]

    @staticmethod
    def _edge_distance(
        *, centre: np.ndarray, approach: np.ndarray, lower: np.ndarray, upper: np.ndarray
    ) -> float:
        """How far the base side of `approach` runs from `centre` to the counter edge.

        A vertical approach has no base side; it ranks last rather than failing.
        """
        length = float(np.linalg.norm(approach[:2]))
        if length < 1e-6:
            return float("inf")
        direction = -approach[:2] / length
        return min(
            max(0.0, (boundary - centre[axis]) / direction[axis])
            for axis, boundary in enumerate(np.where(direction > 0, upper, lower))
            if abs(direction[axis]) > 1e-6
        )

    def _placement_turns(self, *, rotation: np.ndarray, centre: np.ndarray) -> list[np.ndarray]:
        """The held hand yawed about world z, nearest counter edge first.

        The grasp fixes the hand's lean and the food's pose in the hand, not the
        direction the robot must stand to set it down: food picked from behind the
        counter would otherwise have to be placed from behind it too.
        """
        counter = self.backend.support(name="counter")
        lower = np.array([counter.x_min, counter.y_min])
        upper = np.array([counter.x_max, counter.y_max])
        turns = [
            Rotation.from_euler("z", degrees, degrees=True).as_matrix()
            for degrees in PLACEMENT_TURNS_DEG
        ]
        return sorted(
            turns,
            key=lambda turn: round(
                self._edge_distance(
                    centre=centre, approach=turn @ rotation[:, 2], lower=lower, upper=upper
                ),
                6,
            ),
        )

    def _grasp_candidates(self, *, name: str) -> list[np.ndarray]:
        centre = self._pose(name=name)[:3, 3]
        counter = self.backend.support(name="counter")
        lower = np.array([counter.x_min, counter.y_min])
        upper = np.array([counter.x_max, counter.y_max])
        candidates = []
        for lean in (45.0, 30.0, 60.0):
            # Face 0 closes across the dumpling's short axis, onto its fat sides. A pinch on
            # the tips (face 1) lets the food swing about the pinch line as it lifts.
            for face, sign in ((0, -1.0), (0, 1.0)):
                grasp = self.shelf.cube_grasp(
                    cube=name,
                    centre=centre,
                    lean_deg=lean,
                    face=face,
                    sign=sign,
                )
                standoff = 0.033 if lean == 30.0 else 0.031
                grasp[:3, 3] = centre - standoff * grasp[:3, 2]
                distance = self._edge_distance(
                    centre=centre, approach=grasp[:3, 2], lower=lower, upper=upper
                )
                flipped = grasp.copy()
                flipped[:3, :2] *= -1
                candidates.extend([(distance, grasp), (distance, flipped)])
        candidates.sort(key=lambda candidate: round(candidate[0], 6))
        return [pose for _, pose in candidates]

    def _plan_pick(self, *, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        centre = self._pose(name=name)[:3, 3]
        self.target_geoms = self.backend.body_geoms(name=name)
        deadline = monotonic() + 20.0
        failure = "no candidate"
        for grasp in self._grasp_candidates(name=name):
            pre = self.shelf.cube_pre(grasp=grasp)
            for base in self._base_candidates(centre=centre, approach=grasp[:3, 2]):
                if monotonic() >= deadline:
                    raise SkillFailed(f"No collision-free food grasp within budget: {failure}")
                try:
                    staged = self._check_descent(grasp=grasp, pre=pre, base=base)
                    self.motion.plan_base(target=base, arm=staged)
                except SkillFailed as exc:
                    failure = str(exc)
                    continue
                return grasp, pre, base
        raise SkillFailed(f"No collision-free food grasp: {failure}")

    def _check_descent(self, *, grasp: np.ndarray, pre: np.ndarray, base: np.ndarray) -> np.ndarray:
        """Check the pre-grasp and the straight descent at `base`; return the staged arm."""
        staged = arm = self._checked_arm(pose=pre, base=base)
        for pose in Frames.line(start=pre, end=grasp, step=0.01):
            following = self._checked_arm(pose=pose, base=base, seed=arm)
            self.motion.check_arm_segment(start=arm, target=following, base=base, holding=False)
            arm = following
        return staged

    def _replan_grasp(
        self, *, name: str, grasp: np.ndarray, base: np.ndarray, planned: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """A new grasp and pre-grasp if the food moved since `planned`, else None.

        The open pads can push the food during the descent. A pinch planned for its
        old yaw then closes off the food's axes and the food rotates out while lifting.
        The replanned grasp keeps the stance and is the reachable candidate for the
        observed pose with the hand rotation nearest the current one.
        """
        now = self._pose(name=name)
        moved = float(np.linalg.norm(now[:3, 3] - planned[:3, 3]))
        turned = RackScraper.rotation_error(actual=planned[:3, :3], target=now[:3, :3])
        if moved <= REGRASP_POSITION_TOLERANCE and turned <= REGRASP_ANGLE_TOLERANCE_DEG:
            return None
        candidates = sorted(
            self._grasp_candidates(name=name),
            key=lambda candidate: RackScraper.rotation_error(
                actual=grasp[:3, :3], target=candidate[:3, :3]
            ),
        )
        failure = "no candidate"
        for candidate in candidates:
            pre = self.shelf.cube_pre(grasp=candidate)
            try:
                self._check_descent(grasp=candidate, pre=pre, base=base)
            except SkillFailed as exc:
                failure = str(exc)
                continue
            return candidate, pre
        raise SkillFailed(f"Food moved before closing; no regrasp from this stance: {failure}")

    def _approach(self, *, pose: np.ndarray, base: np.ndarray, label: str) -> None:
        arm = self._checked_arm(pose=pose, base=base)
        if self.carried is not None:
            self.motion.transit_arm(arm=arm, label=f"{label}_arm_clearance")
        else:
            staging = self.shelf.site_at(base=self.shelf.data.qpos[self.shelf.ba].copy(), arm=arm)
            self.motion.transit(
                position=staging[:3, 3],
                rotation=staging[:3, :3],
                label=f"{label}_arm_clearance",
                holding=False,
            )
        self.motion.move_base(target=base, label=f"{label}_base")
        self.motion.line(
            position=pose[:3, 3],
            rotation=pose[:3, :3],
            label=f"{label}_pre",
            holding=False,
        )

    def pick(self, *, name: str) -> None:
        """Pick one observed food; verify contact before and throughout lifting."""
        self._validate_name(name=name)
        backend, shelf = self.backend, self.shelf
        backend.sync_clock()
        if shelf.in_hand is not None or any(
            backend.held(name=other) for other in ("wiper_0", *backend.cube_names)
        ):
            raise SkillFailed("PickFood requires an empty hand")
        if not backend.food_available(name=name):
            raise SkillFailed("Food has no observed pickable support")
        self.carried = None
        with self.controllers.feedback():
            grasp, pre, base = self._plan_pick(name=name)
            planned = self._pose(name=name)
            self._limit_grip(newtons=None)
            shelf.grip = 0.0
            shelf.hold(count=10, label="food_open")
            self._approach(pose=pre, base=base, label="food")
            self.motion.line(
                position=grasp[:3, 3],
                rotation=grasp[:3, :3],
                label="food_approach",
                holding=False,
                step=0.005,
            )
            for _ in range(REGRASP_ATTEMPTS):
                replanned = self._replan_grasp(name=name, grasp=grasp, base=base, planned=planned)
                if replanned is None:
                    break
                planned = self._pose(name=name)
                for label, pose in (
                    ("food_regrasp_retreat", pre),
                    ("food_regrasp_pre", replanned[1]),
                    ("food_regrasp", replanned[0]),
                ):
                    self.motion.line(
                        position=pose[:3, 3],
                        rotation=pose[:3, :3],
                        label=label,
                        holding=False,
                        step=0.005,
                    )
                grasp, pre = replanned
            self._limit_grip(newtons=backend.config.food_grip_force)
            shelf.grip = backend.config.food_grip_command
            shelf.hold(count=30, label="food_close")
            self._require_held(name=name)
            self.controllers.held_food = shelf.held_cube = self.carried = name
            with self.motion.departing_from_contacts():
                self.motion.line(
                    position=pre[:3, 3],
                    rotation=pre[:3, :3],
                    label="food_lift",
                    holding=False,
                    step=0.005,
                    steps_per=5,
                )
            self._require_held(name=name)

    @staticmethod
    def _box_xy(*, lower: np.ndarray, upper: np.ndarray) -> list[tuple[float, float]]:
        """The four horizontal corners of an axis-aligned box."""
        return [(float(x), float(y)) for x in (lower[0], upper[0]) for y in (lower[1], upper[1])]

    def _placement_candidates(self, *, name: str, destination: str) -> list[np.ndarray]:
        region = self.backend.support(name=destination)
        corners = self.backend.corners(name=name)
        centre = self._pose(name=name)[:3, 3]
        lower, upper = corners.min(axis=0) - centre, corners.max(axis=0) - centre
        if destination == "rack_0":
            # The same layout distribution as every reset and the person's return.
            return [
                np.array([x, y, region.top_z - lower[2] + 0.003])
                for x, y in self.backend.rack_put_back_draws(
                    name=name, count=self.RACK_PUT_BACK_DRAWS
                )
            ]
        xs = (region.x_min - lower[0] + 0.02, region.x_max - upper[0] - 0.02)
        ys = (region.y_min - lower[1] + 0.02, region.y_max - upper[1] - 0.02)
        if xs[0] > xs[1] or ys[0] > ys[1]:
            raise SkillFailed("Food footprint does not fit destination")
        round_target = isinstance(region, CircularSupport)
        candidates = []
        for target_x in np.linspace(*xs, 7 if round_target else 3):
            for target_y in np.linspace(*ys, 7 if round_target else 5):
                target = np.array([target_x, target_y, region.top_z - lower[2] + 0.003])
                if round_target and not region.contains_xy(
                    points=self._box_xy(lower=target + lower, upper=target + upper), margin=0.02
                ):
                    continue
                occupied = False
                for other in self.backend.cube_names:
                    if other == name:
                        continue
                    obstacle = self.backend.corners(name=other)
                    occupied |= bool(
                        np.all(
                            (target + upper + 0.005 >= obstacle.min(axis=0))
                            & (target + lower - 0.005 <= obstacle.max(axis=0))
                        )
                    )
                if not occupied:
                    candidates.append(target)
        if not candidates:
            raise SkillFailed("No unoccupied food placement at destination; retaining grasp")
        return candidates

    def place(self, *, name: str, destination: str) -> None:
        """Return to a free live support footprint or fail without releasing early."""
        self._validate_name(name=name)
        if destination not in ("rack_0", "box_0"):
            raise SkillFailed(f"Unknown food destination: {destination}")
        self._require_held(name=name)
        shelf = self.shelf
        self.carried = name
        with self.controllers.feedback():
            here = shelf.site_pose()
            offset = here[:3, 3] - self._pose(name=name)[:3, 3]
            deadline = monotonic() + 20.0
            failure = "no reachable placement"
            selected = None
            for centre, turn in (
                (spot, turn)
                for spot in self._placement_candidates(name=name, destination=destination)
                for turn in self._placement_turns(rotation=here[:3, :3], centre=spot)
            ):
                down = Frames.pose(rotation=turn @ here[:3, :3], position=centre + turn @ offset)
                pre = self.shelf.cube_pre(grasp=down)
                for base in self._base_candidates(centre=centre, approach=down[:3, 2]):
                    if monotonic() >= deadline:
                        raise SkillFailed(f"No safe food placement within budget: {failure}")
                    try:
                        arm = self._checked_arm(pose=pre, base=base)
                        self.motion.plan_base(target=base, arm=arm)
                        for pose in Frames.line(start=pre, end=down, step=0.01):
                            following = self._checked_arm(pose=pose, base=base, seed=arm)
                            self.motion.check_arm_segment(
                                start=arm,
                                target=following,
                                base=base,
                                holding=False,
                            )
                            arm = following
                    except SkillFailed as exc:
                        failure = str(exc)
                        continue
                    selected = down, pre, base, centre
                    break
                if selected is not None:
                    break
            if selected is None:
                raise SkillFailed(f"No safe food placement: {failure}")
            down, pre, base, centre = selected
            self._approach(pose=pre, base=base, label="food_place")
            self._lower_food(
                name=name,
                destination=destination,
                target_xy=centre[:2],
                rotation=down[:3, :3],
            )
            self._require_held(name=name)
            self.carried = None
            shelf.grip = 0.0
            self._limit_grip(newtons=None)
            shelf.hold(count=20, label="food_release")
            if self.backend.held(name=name):
                raise SkillFailed("Food remained held after release")
            self.controllers.held_food = shelf.held_cube = None
            self.target_geoms = self.backend.body_geoms(name=name)
            self.motion.line(
                position=pre[:3, 3],
                rotation=pre[:3, :3],
                label="food_retreat",
                holding=False,
                step=0.005,
            )
            shelf.hold(count=20, label="food_rest")
            expected = "OnRack" if destination == "rack_0" else "OnTray"
            if self.backend.location(name=name) != expected:
                raise SkillFailed("Food did not settle at the requested destination")

    def _lower_food(
        self, *, name: str, destination: str, target_xy: np.ndarray, rotation: np.ndarray
    ) -> None:
        """Reobserve food pose and extent after transit and during each guarded descent."""
        for _ in range(40):
            self._require_held(name=name)
            centre = self._pose(name=name)[:3, 3]
            corners = self.backend.corners(name=name)
            region = self.backend.support(name=destination)
            target = np.r_[target_xy, region.top_z + centre[2] - corners[:, 2].min() + 0.003]
            error = target - centre
            bounds = corners[:, :2] + error[:2]
            if not region.contains_xy(
                points=self._box_xy(lower=bounds.min(axis=0), upper=bounds.max(axis=0))
            ):
                raise SkillFailed("Observed food no longer fits the planned support footprint")
            distance = np.linalg.norm(error)
            if distance < 0.0015:
                return
            hand = self.shelf.site_pose()
            self.motion.line(
                position=hand[:3, 3] + error * min(1.0, 0.02 / distance),
                rotation=rotation,
                label="food_lower",
                holding=False,
                step=0.005,
                steps_per=5,
            )
        raise SkillFailed("Measured food lowering failed to converge; retaining grasp")

"""Individual food manipulation using robot commands, never object pose writes.

Integrate with ``RackFoodController(controllers=current_controllers).pick(name=...)``
and ``.place(name=..., destination='rack_0' or 'box_0')``. Failures retain the
physical state for observation or the existing explicit human recovery skills.
The scraper's ``holding`` flag denotes a scraper, not food; this adapter keeps
it false and checks the measured food payload separately.

The rigid 27-mm-high frozen potsticker (`potsticker.py`) uses a 31-mm pinch-site offset at 45
degrees and a force-limited close (``food_grip_force``). Native finger contacts
and the observed food extent govern holding and placement; rack-wire and fixture
collision failures are retained rather than weakening clearance guards.
"""

from collections.abc import Generator
from contextlib import contextmanager
from functools import cached_property
from typing import Any, ClassVar

import numpy as np
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import Frames, SkillFailed

from .semantics import CircularSupport
from .tool import RackScraper
from .transit import geom_half_extents, geoms_half_extents

# How far behind a stance, along its own heading, a counter circulation may enter it.
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
# How far a hand starting with a pad against a fixture first moves to leave it, and how
# finely that move is checked: the guard checks each step at five fractions.
DEPART_M = 0.005
DEPART_SAMPLES = 21
DEPART_STEPS = 15
# Planned food reaches: stances behind the food along the approach, within the arm's
# reach of the hover from its mount, and how long the search may take.
STANCE_DISTANCES_M = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
STANCE_OFFSETS_M = (0.0, -0.12, 0.12, -0.24, 0.24)
STANCE_TURNS_RAD = (0.0, 0.3, -0.3)
ARM_MOUNT_AHEAD_M = 0.12
ARM_REACH_M = 0.85
PLAN_DEADLINE_S = 60.0
# Holding food, the tuck before the base moves raises the hand by this.
CARRY_RAISE_M = 0.10
# A held-food skill starting in contact first rises this far before it plans.
PLACE_ESCAPE_M = 0.04
# A pick starting with the open hand down at the plate rises this far before it plans
# again: the dumpling's height, 27 mm, plus the planner's 15 mm margin, plus play.
PICK_ESCAPE_M = 0.06
# A food lost from the hand is waited for, the arm still, for at most this many control
# steps, until no food moves faster than this or touches the robot.
SETTLE_LOST_FOOD_STEPS = 60
SETTLED_FOOD_M_S = 0.02


class CarriedGeom:
    @staticmethod
    def bounds(
        *, model: Any, geom: int, center: np.ndarray, rotation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Broad-phase box centre and half extents, and the lowest point, of a geom at a pose.

        A potsticker mesh part's box is its AABB, which need not be centred on the geom
        frame, and its lowest point comes from its own vertices, not that box.
        """
        kind = int(model.geom_type[geom])
        if kind == 7:
            mesh = model.geom_dataid[geom]
            start = model.mesh_vertadr[mesh]
            vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh]]
            return (
                center + rotation @ model.geom_aabb[geom][:3],
                np.asarray(model.geom_aabb[geom][3:]),
                float(center[2] + np.min(vertices @ rotation[2])),
            )
        if kind == 4:
            size = np.asarray(model.geom_size[geom])
            return center, size, float(center[2] - np.linalg.norm(rotation[2] * size))
        size = geom_half_extents(model=model, geom=geom)
        return center, size, float(center[2] - np.abs(rotation[2]) @ size)


class _FoodMotion(RackScraper):
    """Reuse guarded scraper routes, adding food collisions and live grasp checks."""

    _measuring_departure: bool = False
    departing_food: dict[tuple[int, int], float] = {}  # noqa: RUF012 -- reassigned per scope, never mutated in place outside one

    def __init__(self, *, food: "RackFoodController") -> None:
        self.food = food
        self.departing: dict[tuple[int, int], float] = {}
        # Carried-food overlaps (carried geom, obstacle) present when a departing scope
        # began; like `departing`, each may persist but not deepen.
        self.departing_food = {}
        # (robot geom, ungrasped food geom) contacts present when a departing scope
        # began, by distance; like `departing`, each may persist but not deepen.
        self.departing_touch: dict[tuple[int, int], float] = {}
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
        foods = self._food_geoms(carried=carried)
        if self._touches_ungrasped_food(scratch=scratch, foods=foods):
            raise SkillFailed("Food motion would collide with ungrasped food")
        if carried is None:
            return
        live_hand = shelf.site_pose()
        target_hand = shelf.site_at(base=base, arm=arm)
        transform = target_hand @ np.linalg.inv(live_hand)
        obstacles = list(shelf.cupboard_geoms | shelf.box_geoms | set(shelf.wiper_geoms) | foods)
        # Every obstacle's world box at once: a potsticker carries 41 parts, and the
        # per-pair broad phase in Python dominated placement planning.
        other_centers = scratch.geom_xpos[obstacles]
        other_rotations = scratch.geom_xmat[obstacles].reshape(-1, 3, 3)
        other_reach = np.einsum(
            "nij,nj->ni",
            np.abs(other_rotations),
            geoms_half_extents(model=shelf.model, geoms=np.asarray(obstacles)),
        )
        for geom in backend.body_geoms(name=carried):
            center = transform[:3, :3] @ shelf.data.geom_xpos[geom] + transform[:3, 3]
            rotation = transform[:3, :3] @ shelf.data.geom_xmat[geom].reshape(3, 3)
            box_center, size, bottom = CarriedGeom.bounds(
                model=shelf.model, geom=geom, center=center, rotation=rotation
            )
            near = np.all(
                np.abs(box_center - other_centers) <= np.abs(rotation) @ size + other_reach, axis=1
            )
            for index in np.flatnonzero(near):
                obstacle = obstacles[index]
                other_center = other_centers[index]
                other_rotation = other_rotations[index]
                other_size = geom_half_extents(model=shelf.model, geom=obstacle)
                overlap = self._penetration(
                    scratch=scratch, geom=geom, obstacle=obstacle, center=center, rotation=rotation
                )
                top = other_center[2] + np.abs(other_rotation[2]) @ other_size
                support = obstacle in shelf.cupboard_geoms | shelf.box_geoms
                if overlap > 0 and not (support and bottom >= top - 0.003):
                    if self._measuring_departure:
                        self.departing_food[geom, obstacle] = overlap
                        continue
                    allowed = self.departing_food.get((geom, obstacle))
                    if allowed is not None and overlap <= allowed + DEPARTURE_TOLERANCE:
                        continue
                    raise SkillFailed(
                        f"Carried food would collide with {shelf.model.geom(obstacle).name}: "
                        f"overlap={overlap:.4f}, bottom={bottom:.4f}, top={top:.4f}"
                    )

    def _food_geoms(self, *, carried: str | None) -> set[int]:
        """Every food's contact geoms but the carried one's. A controller lives for one
        skill call, under one model, so each body's geoms are read once."""
        backend = self.food.backend
        cache: dict[str, set[int]] = self.__dict__.setdefault("_body_geoms", {})
        for name in backend.cube_names:
            if name not in cache:
                cache[name] = backend.body_geoms(name=name)
        # Built in the same insertion order as before, so iteration order is too.
        return {geom for name in backend.cube_names if name != carried for geom in cache[name]}

    def _touches_ungrasped_food(self, *, scratch: Any, foods: set[int]) -> bool:
        """Whether a non-positive-distance contact in `scratch` puts a robot geom on an
        ungrasped food (a pad on the target food excepted), in either geom order."""
        ncon = int(scratch.ncon)
        if ncon == 0 or not foods:
            return False
        shelf = self.shelf
        count = int(shelf.model.ngeom)

        def mask(*, geoms: Any) -> np.ndarray:
            out = np.zeros(count, dtype=bool)
            out[np.fromiter(geoms, dtype=int, count=len(geoms))] = True
            return out

        robot, food, pad = (mask(geoms=g) for g in (shelf.robot_geoms, foods, shelf.pad_geoms))
        target = mask(geoms=self.food.target_geoms)
        a = np.asarray(scratch.contact.geom1[:ncon])
        b = np.asarray(scratch.contact.geom2[:ncon])
        dist = np.asarray(scratch.contact.dist[:ncon])
        touching = ~(dist > 0)
        hit = (robot[a] & food[b] & ~(pad[a] & target[b])) | (
            robot[b] & food[a] & ~(pad[b] & target[a])
        )
        hits = np.flatnonzero(touching & hit)
        if not self.departing_touch:
            return bool(hits.size)
        for index in hits:
            pair = (
                (int(a[index]), int(b[index]))
                if robot[a[index]]
                else (int(b[index]), int(a[index]))
            )
            start = self.departing_touch.get(pair)
            if start is None or dist[index] < start - DEPARTURE_TOLERANCE:
                return True
        return False

    def food_touches(self) -> dict[tuple[int, int], float]:
        """Closest distance of each touching (robot geom, food geom) pair among scratch
        data's contacts, the target food's pads excepted as `_touches_ungrasped_food`
        excepts them."""
        shelf = self.shelf
        scratch = shelf.ik_data
        foods = self._food_geoms(carried=self.food.carried)
        target = self.food.target_geoms
        pairs: dict[tuple[int, int], float] = {}
        for contact in scratch.contact[: scratch.ncon]:
            if contact.dist > 0:
                continue
            for robot, other in (
                (int(contact.geom1), int(contact.geom2)),
                (int(contact.geom2), int(contact.geom1)),
            ):
                if robot not in shelf.robot_geoms or other not in foods:
                    continue
                if robot in shelf.pad_geoms and other in target:
                    continue
                pairs[robot, other] = min(pairs.get((robot, other), np.inf), contact.dist)
        return pairs

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
        own starting configuration. Holding food, the food's own overlaps with
        fixtures at the start are measured and allowed the same way. So is the hand
        resting against an ungrasped dumpling: a placement beside one already served
        releases with a pad touching it, and its retreat starts there.
        """
        shelf = self.shelf
        shelf.site_at(base=shelf.data.qpos[shelf.ba].copy(), arm=shelf.data.qpos[shelf.qa].copy())
        shelf.mujoco.mj_collision(shelf.model, shelf.ik_data)
        self.departing = self.pad_fixture_contacts()
        self.departing_touch = self.food_touches()
        self.departing_food = {}
        if self.food.carried is not None:
            self._measuring_departure = True
            try:
                self.check_configuration(
                    base=shelf.data.qpos[shelf.ba].copy(),
                    arm=shelf.data.qpos[shelf.qa].copy(),
                    holding=False,
                )
            except SkillFailed:
                pass
            finally:
                self._measuring_departure = False
        try:
            yield
        finally:
            self.departing, self.departing_food, self.departing_touch = {}, {}, {}

    def depart(self, *, label: str) -> None:
        """Leave the departing pad-fixture contacts along a short line on which none deepens.

        A pad resting against the welded plate's thin, tilted rim box is not left by
        just any short move: straight up, or back along the approach, first presses
        the pad's corner up to 0.24 mm deeper before clearing the rim, and the guard
        rejects the line. Each candidate direction, straight up first, is checked as
        the execution guard checks a step (joint-interpolated), and the first that
        never deepens a starting contact and ends clear of every one is taken. With no
        such contact this moves nothing.
        """
        if not self.departing:
            return
        shelf = self.shelf
        here = shelf.site_pose()
        base = shelf.data.qpos[shelf.ba].copy()
        arm = shelf.data.qpos[shelf.qa].copy()
        for direction in self._departure_directions():
            target = here[:3, 3] + DEPART_M * direction
            checked = self._departs(base=base, arm=arm, position=target, rotation=here[:3, :3])
            if checked is not None:
                # Command the very arm checked, not a fresh Cartesian line: a line's own
                # IK waypoints are other configurations, and one touched the rim.
                for _ in range(DEPART_STEPS):
                    self.step(arm=checked, label=label)
                self.settle_hand(
                    arm=checked, position=target, rotation=here[:3, :3], label=label, limit=40
                )
                return
        names = sorted(
            f"{shelf.model.geom(fixture).name}/{shelf.model.geom(pad).name}"
            for pad, fixture in self.departing
        )
        raise SkillFailed(f"{label}: no departure from {names} leaves them without deepening")

    @staticmethod
    def _departure_directions() -> list[np.ndarray]:
        """Straight up, then up and away along each of eight headings, then level."""
        headings = [
            np.array([np.cos(angle), np.sin(angle), 0.0])
            for angle in np.deg2rad(np.arange(0.0, 360.0, 45.0))
        ]
        tilted = [(heading + [0.0, 0.0, 1.0]) / np.sqrt(2.0) for heading in headings]
        return [np.array([0.0, 0.0, 1.0]), *tilted, *headings]

    def _departs(
        self, *, base: np.ndarray, arm: np.ndarray, position: np.ndarray, rotation: np.ndarray
    ) -> np.ndarray | None:
        """The arm at `position` if the joint-interpolated move there keeps every
        starting pad-fixture contact no deeper than it began, adds none, and ends
        clear; else None."""
        shelf = self.shelf
        target, _ = shelf.ik(
            position=position, rotation=rotation, base=base, holding=False, attempts=4, seed=arm
        )
        if not np.isfinite(target).all():
            return None
        reached = shelf.site_at(base=base, arm=target)
        if np.linalg.norm(reached[:3, 3] - position) > 0.001:
            return None
        delta = (target - arm + np.pi) % (2 * np.pi) - np.pi
        for fraction in np.linspace(0.0, 1.0, DEPART_SAMPLES):
            shelf.site_at(base=base, arm=arm + fraction * delta)
            shelf.mujoco.mj_collision(shelf.model, shelf.ik_data)
            contacts = self.pad_fixture_contacts()
            for pair, distance in contacts.items():
                start = self.departing.get(pair)
                if start is None or distance < start - DEPARTURE_TOLERANCE:
                    return None
            if fraction == 1.0 and contacts:
                return None
        return arm + delta

    def step(self, *, arm: np.ndarray, label: str) -> None:
        if self.food.carried is not None:
            self.food._require_held(name=self.food.carried)
        super().step(arm=arm, label=label)
        if self.food.carried is not None:
            self.food._require_held(name=self.food.carried)

    def settle_lost_food(self) -> None:
        """Hold the arm still, the hand as it is, until no food is moving or touching
        the robot, for at most ``SETTLE_LOST_FOOD_STEPS`` control steps.

        Physics advances only inside a skill. A PickFood that lost its dumpling
        during the lift used to return the moment it noticed, the dumpling falling
        beside the closed hand 15 cm above the counter, and every later plan saw it
        hang there: Della c6ed1a6 EES seed 0 refused PickScraper at 0 steps 10 times
        in a row, "no stance reaches the scraper". The hand is not opened: a failed
        food skill never releases. With no food moving or touching the robot, as when
        a skill refuses before it moves, this takes no step.
        """
        shelf = self.shelf
        arm = shelf.data.qpos[shelf.qa].copy()
        for _ in range(SETTLE_LOST_FOOD_STEPS):
            if not self.food_unsettled():
                return
            shelf.step(arm=arm, label="food_lost_settle")

    def food_unsettled(self) -> bool:
        """Whether any food moves faster than ``SETTLED_FOOD_M_S`` or touches the robot."""
        backend, shelf = self.food.backend, self.shelf
        model, data = shelf.model, shelf.data
        foods: set[int] = set()
        for name in backend.cube_names:
            dof = model.jnt_dofadr[model.joint(f"{name}_joint").id]
            if float(np.linalg.norm(data.qvel[dof : dof + 3])) > SETTLED_FOOD_M_S:
                return True
            foods |= backend.body_geoms(name=name)
        for contact in data.contact[: data.ncon]:
            a, b = int(contact.geom1), int(contact.geom2)
            if contact.dist <= 0 and (
                (a in shelf.robot_geoms and b in foods) or (b in shelf.robot_geoms and a in foods)
            ):
                return True
        return False

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
            self.motion.settle_lost_food()
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

    def _stances(
        self, *, centre: np.ndarray, approach: np.ndarray, hover: np.ndarray
    ) -> list[np.ndarray]:
        """Bases behind `centre` along the approach, within reach of the hover: a stance
        under the counter is refused by the planner, not by this list."""
        forward = np.asarray(approach[:2], dtype=float)
        length = np.linalg.norm(forward)
        if not np.isfinite(length) or length < 1e-6:
            raise SkillFailed("Food grasp requires a finite angled approach")
        forward = forward / length
        sideways = np.array([-forward[1], forward[0]])
        heading = float(np.arctan2(forward[1], forward[0]))
        out = []
        for distance in STANCE_DISTANCES_M:
            for offset in STANCE_OFFSETS_M:
                for turn in STANCE_TURNS_RAD:
                    stance = np.r_[
                        centre[:2] - distance * forward + offset * sideways, heading + turn
                    ]
                    mount = stance[:2] + ARM_MOUNT_AHEAD_M * np.array([
                        np.cos(stance[2]),
                        np.sin(stance[2]),
                    ])
                    if np.linalg.norm(hover[:2] - mount) <= ARM_REACH_M:
                        out.append(stance)
        out.sort(
            key=lambda stance: float(
                abs(np.linalg.norm(stance[:2] - centre[:2]) - 0.55)
                + 0.3 * abs((stance[2] - heading + np.pi) % (2 * np.pi) - np.pi)
            )
        )
        return out

    def _plan_pick(self, *, name: str) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
        """Planned base and arm to a pre-grasp from which the straight descent is clear."""
        from .planning import PlanningRefused, ReachCandidate, ScraperMotionPlanner

        centre = self._pose(name=name)[:3, 3]
        self.target_geoms = self.backend.body_geoms(name=name)

        def descent(*, grasp: np.ndarray, pre: np.ndarray) -> Any:
            def contact(stance: np.ndarray, hover: np.ndarray) -> str | None:  # noqa: PLR0917 -- ReachCandidate callback
                try:
                    self._check_descent(grasp=grasp, pre=pre, base=stance, seed=hover)
                except SkillFailed as exc:
                    return str(exc)[:70]
                return None

            return contact

        def candidates() -> Generator[Any, None, None]:
            for grasp in self._grasp_candidates(name=name):
                pre = self.shelf.cube_pre(grasp=grasp)
                contact = descent(grasp=grasp, pre=pre)
                for stance in self._stances(centre=centre, approach=grasp[:3, 2], hover=pre[:3, 3]):
                    yield ReachCandidate(
                        stance=stance,
                        position=pre[:3, 3],
                        rotation=pre[:3, :3],
                        contact=contact,
                        payload=(grasp, pre),
                    )

        try:
            plan, chosen = ScraperMotionPlanner.plan_reach(
                tool=self.motion,
                candidates=candidates(),
                held_name=None,
                carry_height=0.0,
                deadline_s=PLAN_DEADLINE_S,
                what="the food pre-grasp",
            )
        except PlanningRefused as exc:
            raise SkillFailed(f"No collision-free food grasp: {exc}") from exc
        grasp, pre = chosen.payload
        return plan, grasp, pre, plan.stance

    def _plan_pick_from_here(self, *, name: str) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
        """`_plan_pick`, first leaving a start no plan can begin from.

        A placement whose retreat failed leaves the open hand down at the plate, within
        the planner's margin of it and of the food on it, so every reach is refused.
        Rise straight up, open, out of that start (a line checked as it goes, as the
        placement's own escape), then plan once more. From a start that is not blocked
        a refusal moves nothing.
        """
        from .planning import ScraperMotionPlanner

        try:
            return self._plan_pick(name=name)
        except SkillFailed as exc:
            if not ScraperMotionPlanner.start_blocked(tool=self.motion, held_name=None):
                raise
            reason = str(exc)
        shelf = self.shelf
        if shelf.grip > 0.0:
            shelf.grip = 0.0
            shelf.hold(count=25, label="food_pick_escape_open")
        here = shelf.site_pose()
        try:
            self.motion.line(
                position=here[:3, 3] + [0.0, 0.0, PICK_ESCAPE_M],
                rotation=here[:3, :3],
                label="food_pick_escape",
                holding=False,
                step=0.005,
                steps_per=5,
            )
        except SkillFailed as exc:
            raise SkillFailed(f"{reason}; escape refused: {exc}") from exc
        try:
            return self._plan_pick(name=name)
        except SkillFailed as again:
            raise SkillFailed(f"after escaping the start: {again}") from again

    def _follow(self, *, plan: Any, pose: np.ndarray, label: str) -> None:
        """Execute a planned reach: tuck, base, arm to `pose`."""
        motion = self.motion
        motion.follow_arm_path(path=plan.arm_before_base, label=f"{label}_tuck")
        motion.follow_base_path(path=plan.base_path, label=f"{label}_base")
        motion.follow_arm_path(path=plan.arm_after_base, label=f"{label}_pre")
        motion.settle_hand(
            arm=plan.arm_after_base[-1],
            position=pose[:3, 3],
            rotation=pose[:3, :3],
            label=f"{label}_pre",
            limit=300,
        )

    def _check_descent(
        self,
        *,
        grasp: np.ndarray,
        pre: np.ndarray,
        base: np.ndarray,
        seed: np.ndarray | None = None,
    ) -> np.ndarray:
        """Check the pre-grasp and the straight descent at `base`; return the staged arm."""
        staged = arm = self._checked_arm(pose=pre, base=base, seed=seed)
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
            with self.motion.departing_from_contacts():
                plan, grasp, pre, base = self._plan_pick_from_here(name=name)
                planned = self._pose(name=name)
                self._limit_grip(newtons=None)
                shelf.grip = 0.0
                shelf.hold(count=10, label="food_open")
                self._follow(plan=plan, pose=pre, label="food")
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
                # A grasp closed beside the plate's rim can leave a pad touching it.
                self.motion.depart(label="food_lift_depart")
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

    def _plan_place(
        self, *, name: str, destination: str
    ) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
        """Planned carry to a pre-placement from which the straight descent is clear."""
        from .planning import PlanningRefused, ReachCandidate, ScraperMotionPlanner

        shelf = self.shelf
        here = shelf.site_pose()
        offset = here[:3, 3] - self._pose(name=name)[:3, 3]

        def descent(*, down: np.ndarray, pre: np.ndarray) -> Any:
            def contact(stance: np.ndarray, hover: np.ndarray) -> str | None:  # noqa: PLR0917 -- ReachCandidate callback
                try:
                    arm = self._checked_arm(pose=pre, base=stance, seed=hover)
                    for pose in Frames.line(start=pre, end=down, step=0.01):
                        following = self._checked_arm(pose=pose, base=stance, seed=arm)
                        self.motion.check_arm_segment(
                            start=arm, target=following, base=stance, holding=False
                        )
                        arm = following
                except SkillFailed as exc:
                    return str(exc)[:70]
                return None

            return contact

        spots = self._placement_candidates(name=name, destination=destination)

        def candidates() -> Generator[Any, None, None]:
            for spot in spots:
                for turn in self._placement_turns(rotation=here[:3, :3], centre=spot):
                    down = Frames.pose(rotation=turn @ here[:3, :3], position=spot + turn @ offset)
                    pre = shelf.cube_pre(grasp=down)
                    contact = descent(down=down, pre=pre)
                    for stance in self._stances(
                        centre=spot, approach=down[:3, 2], hover=pre[:3, 3]
                    ):
                        yield ReachCandidate(
                            stance=stance,
                            position=pre[:3, 3],
                            rotation=pre[:3, :3],
                            contact=contact,
                            payload=(down, pre, spot),
                        )

        try:
            plan, chosen = ScraperMotionPlanner.plan_reach(
                tool=self.motion,
                candidates=candidates(),
                held_name=name,
                carry_height=float(here[2, 3] + CARRY_RAISE_M),
                deadline_s=PLAN_DEADLINE_S,
                what="the food pre-placement",
            )
        except PlanningRefused as exc:
            raise SkillFailed(f"No safe food placement: {exc}") from exc
        down, pre, centre = chosen.payload
        return plan, down, pre, centre

    def place(self, *, name: str, destination: str) -> None:
        """Return to a free live support footprint or fail without releasing early."""
        self._validate_name(name=name)
        if destination not in ("rack_0", "box_0"):
            raise SkillFailed(f"Unknown food destination: {destination}")
        self._require_held(name=name)
        shelf = self.shelf
        self.carried = name
        with self.controllers.feedback():
            # A skill can start with a pad already resting on a fixture (a lift that
            # failed with the food still held leaves one on the plate's rim). That
            # contact may persist, not deepen, until the hand is over the target.
            with self.motion.departing_from_contacts():
                if self.motion.departing or self.motion.departing_food:
                    # Leave the start contact first, then rise straight up, as the lift
                    # does: the planner keeps a margin from everything, so it cannot
                    # start there.
                    self.motion.depart(label="food_place_depart")
                    here = shelf.site_pose()
                    self.motion.line(
                        position=here[:3, 3] + [0.0, 0.0, PLACE_ESCAPE_M],
                        rotation=here[:3, :3],
                        label="food_place_escape",
                        holding=False,
                        step=0.005,
                        steps_per=5,
                    )
                plan, down, pre, centre = self._plan_place(name=name, destination=destination)
                self._follow(plan=plan, pose=pre, label="food_place")
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
            # Set down beside a served dumpling, the open hand can rest against it.
            with self.motion.departing_from_contacts():
                self.motion.depart(label="food_retreat_depart")
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

"""Checked empty-hand base transit for scraper pickup.

Ported from scratch/prototypes/sweeptable-kitchen/src/hitl_pmp/environments/
sweeptable3d/navigation.py. Segment sampling is local, adapted from that
prototype's kitchen_recovery.segment_samples, with no recovery-module dependency.
"""

from time import monotonic
from typing import Any

import numpy as np

from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed

TRANSIT_MARGIN = 0.01
TRANSFER_STEP = 0.002
TRANSFER_TURN = np.deg2rad(0.5)


def geom_half_extents(*, model: Any, geom: int) -> np.ndarray:
    """Conservative local bounds for boxes, spheres, rounded wires and flat cylinders.

    A cylinder has flat caps, so its exact box is (r, r, half-length); treating it
    as a capsule would put a radius-tall phantom block above a wide plate.
    """
    kind = int(model.geom_type[geom])
    size = model.geom_size[geom]
    if kind == 6:
        return np.array(size, copy=True)
    if kind == 2:
        return np.full(3, size[0])
    if kind == 3:
        return np.array([size[0], size[0], size[1] + size[0]])
    if kind == 5:
        return np.array([size[0], size[0], size[1]])
    return np.full(3, model.geom_rbound[geom])


def geoms_half_extents(*, model: Any, geoms: np.ndarray) -> np.ndarray:
    """`geom_half_extents` for many geoms at once, value for value."""
    kind = np.asarray(model.geom_type)[geoms].astype(int)[:, None]
    size = np.asarray(model.geom_size, dtype=float)[geoms]
    radius = size[:, :1]
    rbound = np.asarray(model.geom_rbound, dtype=float)[geoms][:, None]
    return np.select(
        [kind == 6, kind == 2, kind == 3, kind == 5],
        [
            size,
            np.repeat(radius, 3, axis=1),
            np.column_stack([radius, radius, size[:, 1] + size[:, 0]]),
            np.column_stack([radius, radius, size[:, 1]]),
        ],
        default=np.repeat(rbound, 3, axis=1),
    )


def segment_samples(*, start: np.ndarray, target: np.ndarray) -> list[np.ndarray]:
    """Sample straight and axis-clipped paths, including unwrapped headings."""
    start, target = np.asarray(start, dtype=float), np.asarray(target, dtype=float)
    if start.shape != (3,) or target.shape != (3,):
        raise ValueError("Base endpoints must be (x, y, heading) vectors")
    if not np.isfinite(np.r_[start, target]).all():
        raise ValueError("Base endpoints must be finite")
    delta = target - start
    limits = np.array([TRANSFER_STEP, TRANSFER_STEP, TRANSFER_TURN])
    count = max(1, int(np.ceil(np.max(np.abs(delta) / limits))))
    straight = [start + fraction * delta for fraction in np.linspace(0, 1, count + 1)]
    clipped = [
        start + np.sign(delta) * np.minimum(np.abs(delta), index * TRANSFER_STEP)
        for index in range(int(np.ceil(np.max(np.abs(delta)) / TRANSFER_STEP)) + 1)
    ]
    return [*straight, *clipped, target.copy()]


# A start at most this far past a joint stop is driven back inside it before routing,
# to this far inside.
LIMIT_RECOVERY_RAD = 0.1
LIMIT_RECOVERY_MARGIN = 0.01


class ArmTransit:
    """Bounded bidirectional joint routing using the tool's complete collision guard.

    Planning only touches scratch kinematics. Goal-tree edges are checked in
    execution order because axis-clipped joint paths are not reversible. Both
    trees use wrapped angles, but no limited joint may cross its physical stop.
    """

    def __init__(
        self,
        *,
        tool: Any,
        base: np.ndarray,
        holding: bool,
        max_iterations: int = 400,
        max_edge_checks: int = 1200,
        time_limit_s: float = 15.0,
    ) -> None:
        if max_iterations < 1 or max_edge_checks < 1:
            raise ValueError("Arm transit requires positive planning budgets")
        if not np.isfinite(time_limit_s) or time_limit_s <= 0:
            raise ValueError("Arm transit requires a finite positive time limit")
        self.tool = tool
        self.base = np.asarray(base, dtype=float).copy()
        self.holding = holding
        self.max_iterations = max_iterations
        self.max_edge_checks = max_edge_checks
        self.time_limit_s = time_limit_s
        self.edge_checks = 0
        self.last_failure = ""
        self.deadline = 0.0
        shelf = tool.shelf
        self.limited = np.asarray(shelf.model.jnt_limited[shelf.joints], dtype=bool)
        self.ranges = np.asarray(shelf.model.jnt_range[shelf.joints], dtype=float).copy()

    @staticmethod
    def delta(*, start: np.ndarray, target: np.ndarray) -> np.ndarray:
        return (target - start + np.pi) % (2 * np.pi) - np.pi

    def exhausted(self) -> bool:
        return self.edge_checks >= self.max_edge_checks or monotonic() >= self.deadline

    def in_limits(self, *, arm: np.ndarray) -> bool:
        return bool(
            np.isfinite(arm).all()
            and np.all(arm[self.limited] >= self.ranges[self.limited, 0])
            and np.all(arm[self.limited] <= self.ranges[self.limited, 1])
        )

    def back_inside(self, *, arm: np.ndarray) -> np.ndarray | None:
        """`arm` with each limited joint just past its stop moved `LIMIT_RECOVERY_MARGIN`
        inside it; None if `arm` is within its limits or too far past to recover."""
        if not np.isfinite(arm).all() or self.in_limits(arm=arm):
            return None
        low = self.ranges[:, 0] + LIMIT_RECOVERY_MARGIN
        high = self.ranges[:, 1] - LIMIT_RECOVERY_MARGIN
        inside = np.where(self.limited, np.clip(arm, low, high), arm)
        if np.max(np.abs(inside - arm)) > LIMIT_RECOVERY_RAD:
            return None
        return inside

    def edge_clear(self, *, start: np.ndarray, target: np.ndarray) -> bool:
        if self.exhausted():
            return False
        self.edge_checks += 1
        endpoint = start + self.delta(start=start, target=target)
        if not self.in_limits(arm=start) or not self.in_limits(arm=endpoint):
            self.last_failure = "wrapped joint segment crosses a physical joint limit"
            return False
        try:
            self.tool.check_arm_segment(
                start=start, target=target, base=self.base, holding=self.holding
            )
        except SkillFailed as exc:
            self.last_failure = str(exc)
            return False
        return True

    def extend(
        self,
        *,
        nodes: list[np.ndarray],
        parents: list[int],
        target: np.ndarray,
        reverse: bool,
    ) -> tuple[int | None, bool]:
        distances = self.delta(start=np.asarray(nodes), target=target)
        nearest = int(np.argmin(np.linalg.norm(distances, axis=1)))
        delta = distances[nearest]
        distance = float(np.max(np.abs(delta)))
        if distance < 1e-9:
            return nearest, True
        candidate = nodes[nearest] + delta * min(1.0, 0.25 / distance)
        start, end = (candidate, nodes[nearest]) if reverse else (nodes[nearest], candidate)
        if not self.edge_clear(start=start, target=end):
            return None, False
        nodes.append(candidate)
        parents.append(nearest)
        return len(nodes) - 1, distance <= 0.25

    @staticmethod
    def trace(*, nodes: list[np.ndarray], parents: list[int], index: int) -> list[np.ndarray]:
        route = []
        while index >= 0:
            route.append(nodes[index].copy())
            index = parents[index]
        return route

    def shorten(self, *, route: list[np.ndarray]) -> list[np.ndarray]:
        """Try a bounded number of shortcuts, preserving directed collision checks."""
        index, attempts = 0, 0
        while index < len(route) - 2 and attempts < 24 and not self.exhausted():
            for end in range(len(route) - 1, index + 1, -1):
                if attempts >= 24 or self.exhausted():
                    break
                attempts += 1
                if self.edge_clear(start=route[index], target=route[end]):
                    route[index + 1 : end] = []
                    break
            index += 1
        return route[1:]

    def plan(self, *, start: np.ndarray, target: np.ndarray) -> list[np.ndarray] | None:
        """Return waypoints excluding start, or fail without moving the robot."""
        start, target = np.asarray(start, dtype=float), np.asarray(target, dtype=float)
        if start.shape != self.limited.shape or target.shape != start.shape:
            raise ValueError("Arm transit joint vectors must match the robot joints")
        self.edge_checks = 0
        self.last_failure = ""
        self.deadline = monotonic() + self.time_limit_s
        inside = self.back_inside(arm=start)
        if inside is not None:
            # Contact can push a joint a little past its stop; the controller then holds
            # it there. Drive it back inside first, along a checked edge, then plan.
            try:
                self.tool.check_arm_segment(
                    start=start, target=inside, base=self.base, holding=self.holding
                )
            except SkillFailed as exc:
                raise SkillFailed(f"Arm transit start past a joint stop is blocked: {exc}") from exc
            rest = self.plan(start=inside, target=target)
            return None if rest is None else [inside, *rest]
        for name, arm in (("start", start), ("goal", target)):
            if not self.in_limits(arm=arm):
                raise SkillFailed(f"Arm transit {name} violates joint limits")
            try:
                self.tool.check_configuration(base=self.base, arm=arm, holding=self.holding)
            except SkillFailed as exc:
                raise SkillFailed(f"Arm transit {name} configuration blocked: {exc}") from exc
        if self.edge_clear(start=start, target=target):
            return [target.copy()]
        home = np.asarray(self.tool.shelf.home_arm, dtype=float)
        if self.edge_clear(start=start, target=home) and self.edge_clear(start=home, target=target):
            return [home.copy(), target.copy()]
        lower = np.where(self.limited, self.ranges[:, 0], start - np.pi)
        upper = np.where(self.limited, self.ranges[:, 1], start + np.pi)
        nearby = start + self.delta(start=start, target=target)
        local_lower = np.maximum(lower, np.minimum(start, nearby) - 0.8)
        local_upper = np.minimum(upper, np.maximum(start, nearby) + 0.8)
        trees = [[start.copy()], [target.copy()]]
        parents = [[-1], [-1]]
        rng = np.random.default_rng(0)
        for iteration in range(self.max_iterations):
            if self.exhausted():
                break
            active, other = iteration % 2, 1 - iteration % 2
            bias = rng.random()
            sample = (
                trees[other][0]
                if bias < 0.2
                else rng.uniform(local_lower, local_upper)
                if bias < 0.8
                else rng.uniform(lower, upper)
            )
            index, _ = self.extend(
                nodes=trees[active], parents=parents[active], target=sample, reverse=active == 1
            )
            if index is None:
                continue
            for _ in range(64):
                if self.exhausted():
                    break
                connected, reached = self.extend(
                    nodes=trees[other],
                    parents=parents[other],
                    target=trees[active][index],
                    reverse=other == 1,
                )
                if connected is None:
                    break
                if reached:
                    indices = [index, connected] if active == 0 else [connected, index]
                    outgoing = self.trace(nodes=trees[0], parents=parents[0], index=indices[0])
                    incoming = self.trace(nodes=trees[1], parents=parents[1], index=indices[1])
                    return self.shorten(route=[*reversed(outgoing), *incoming[1:]])
        return None


class EmptyHandTransit:
    """Keep the measured arm fixed and route around live fixture geometry.

    All planning uses scratch kinematics. Execution uses the existing guarded
    base controller; neither scene poses nor collision masks are changed.
    Preserve the requested unwrapped heading: subsequent skills command literal
    zero, so ending at 2*pi would make them rotate a full turn while carrying.
    """

    def __init__(self, *, shelf: Any) -> None:
        self.shelf = shelf
        self.refresh_geometry()

    def refresh_geometry(self) -> None:
        """Read mutable collider membership and collision enablement afresh."""
        shelf = self.shelf
        model = shelf.model
        enabled = (np.asarray(model.geom_contype) != 0) | (np.asarray(model.geom_conaffinity) != 0)
        self.robot = {geom for geom in shelf.robot_geoms if enabled[geom]}
        self.fixtures = {geom for geom in shelf.cupboard_geoms | shelf.box_geoms if enabled[geom]}
        is_robot = np.zeros(int(model.ngeom), dtype=bool)
        is_robot[list(self.robot)] = True
        self.obstacles = set(
            np.flatnonzero(enabled & ~is_robot & (np.asarray(model.geom_type) != 0)).tolist()
        )
        # Index arrays in the sets' own iteration order, so pairs are visited in the
        # order the per-pair loop over `self.robot` x `self.obstacles` visits them.
        self.robot_index = np.fromiter(self.robot, dtype=int, count=len(self.robot))
        self.obstacle_index = np.fromiter(self.obstacles, dtype=int, count=len(self.obstacles))

    def clear(self, *, base: np.ndarray, arm: np.ndarray) -> bool:
        shelf = self.shelf
        self.refresh_geometry()
        model, scratch = shelf.model, shelf.ik_data
        if scratch is shelf.data:
            raise SkillFailed("Empty-hand transit requires separate scratch data")
        shelf.site_at(base=base, arm=arm)
        if shelf.collision_cost(d=scratch, holding=False) > 0:
            return False
        return not any(
            shelf.mujoco.mj_geomDistance(model, scratch, robot, obstacle, TRANSIT_MARGIN, None)
            < TRANSIT_MARGIN
            for robot, obstacle in self.near_pairs(model=model, scratch=scratch)
        )

    def near_pairs(self, *, model: Any, scratch: Any) -> list[tuple[int, int]]:
        """Robot/obstacle pairs whose bounding spheres come within the margin.

        The bounding-sphere reject is evaluated for all pairs at once, in the
        per-pair loop's order. A vectorised norm can differ from the scalar one in
        the last bit, so any pair within a hair of its bound is re-decided with the
        scalar expression: the exact checks run on exactly the same pairs.
        """
        robots, obstacles = self.robot_index, self.obstacle_index
        if robots.size == 0 or obstacles.size == 0:
            return []
        xpos, rbound = scratch.geom_xpos, model.geom_rbound
        separation = np.linalg.norm(xpos[robots][:, None, :] - xpos[obstacles][None, :, :], axis=-1)
        bound = rbound[robots][:, None] + rbound[obstacles][None, :] + TRANSIT_MARGIN
        slack = 1e-9 * np.maximum(1.0, bound)
        pairs = []
        for row, column in zip(*np.nonzero(~(separation > bound + slack)), strict=True):
            robot, obstacle = int(robots[row]), int(obstacles[column])
            if separation[row, column] > bound[row, column] - slack[row, column] and (
                np.linalg.norm(xpos[robot] - xpos[obstacle]) > bound[row, column]
            ):
                continue
            pairs.append((robot, obstacle))
        return pairs

    def segment_clear(self, *, start: np.ndarray, target: np.ndarray, arm: np.ndarray) -> bool:
        return all(
            self.clear(base=base, arm=arm) for base in segment_samples(start=start, target=target)
        )

    def plan(self, *, target: np.ndarray, arm: np.ndarray) -> list[np.ndarray] | None:
        shelf = self.shelf
        start = shelf.data.qpos[shelf.ba].copy()
        target = np.asarray(target, dtype=float).copy()
        if not np.isfinite(np.r_[start, target, arm]).all():
            raise SkillFailed("Empty-hand transit requires finite joint targets")
        if not self.clear(base=start, arm=arm) or not self.clear(base=target, arm=arm):
            return None
        if self.segment_clear(start=start, target=target, arm=arm):
            return [target]
        model, data = shelf.model, shelf.data
        lower, upper = [], []
        for geom in self.fixtures:
            extent = np.abs(data.geom_xmat[geom].reshape(3, 3)) @ geom_half_extents(
                model=model, geom=geom
            )
            lower.append(data.geom_xpos[geom, :2] - extent[:2])
            upper.append(data.geom_xpos[geom, :2] + extent[:2])
        if not lower or not self.robot:
            return None
        radius = (
            max(
                np.linalg.norm(data.geom_xpos[geom, :2] - start[:2]) + model.geom_rbound[geom]
                for geom in self.robot
            )
            + 0.04
        )
        bounds_low = np.min(lower, axis=0) - radius
        bounds_high = np.max(upper, axis=0) + radius
        candidates = []
        for extra in (0.0, 0.15):
            front = min(bounds_low[0] - extra, target[0])
            for side in (bounds_low[1] - extra, bounds_high[1] + extra):
                candidates.append([
                    np.array([start[0], side, start[2]]),
                    np.array([front, side, start[2]]),
                    np.array([front, side, target[2]]),
                    np.array([front, target[1], target[2]]),
                    target,
                ])
        candidates.sort(
            key=lambda route: sum(
                np.linalg.norm(end[:2] - begin[:2])
                for begin, end in zip([start, *route[:-1]], route, strict=True)
            )
        )
        for route in candidates:
            if all(
                self.segment_clear(start=begin, target=end, arm=arm)
                for begin, end in zip([start, *route[:-1]], route, strict=True)
            ):
                return route
        return None

    def drive(self, *, target: np.ndarray, label: str) -> None:
        shelf = self.shelf
        if shelf.in_hand is not None or shelf.pads_hold_wiper():
            raise SkillFailed(f"{label}: empty-hand transit requires an empty gripper")
        arm = shelf.data.qpos[shelf.qa].copy()
        route = self.plan(target=target, arm=arm)
        if route is None:
            raise SkillFailed(f"{label}: no collision-free base route")
        for waypoint in route:
            start = shelf.data.qpos[shelf.ba].copy()
            arm = shelf.data.qpos[shelf.qa].copy()
            if not self.segment_clear(start=start, target=waypoint, arm=arm):
                raise SkillFailed(f"{label}: segment no longer clear")
            shelf.arm_reference = arm.copy()
            shelf.move_base(target=waypoint, label=label)

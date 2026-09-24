"""Where the robot stands to throw: chosen by the controller, not drawn by the sampler.

The toss used to carry a yaw parameter -- the stand direction about the bin -- drawn
uniformly over +-pi/2. It is gone from the learned parameters. The robot always stands
exactly `standoff` from the bin, facing it, and the throw itself is the same whichever
side it stands on, so the direction is not something a sampler should have to learn:
it is a reachability question with a deterministic answer given the state.

## The rule

The four candidate directions are the bin-relative right angles `TOSS_DIRECTIONS_DEG`,
in the controller's own convention (`get_target_robot_pose_from_parameters`: the stand
is `bin - standoff * (cos, sin)(bin_yaw + direction)`). For one `(state, standoff)`:

1. Rank the four by **clearance**, most first, lower angle first on an exact tie.
2. Walk that order. A direction the cheap geometry gate (`TossParameterFeasibility`)
   proves blocked is skipped without planning. Otherwise the real base motion planner
   (`base_plan_failure`) is asked, and the first plannable direction wins.
3. If none is plannable, raise `NoFeasibleTossDirectionError` with every direction's
   reason. Deliberately not a rejection: silently dropping a standoff reshapes the
   candidate pool's standoff distribution, and that must be loud.

**Clearance** is the Euclidean distance from the stand point (the base centre the
controller targets) to the nearest overhead collider rectangle the base planner itself
collides against (`KinderBackend.toss_feasibility_geometry`: room walls, the barrier,
fixed scene colliders), excluding the bin. The bin is excluded because it is the
target: every direction stands exactly `standoff` from its centre, so it can only ever
contribute a tie, and at short standoffs it would mask the walls entirely.

## The invariant

A bin on the far side of the barrier from the robot must be thrown at from west of the
bin (`stand_x < bin_x`). Feasibility plus the selection rule already imply it -- every
non-west stand of a far bin is across the barrier -- and `select` checks it anyway, so a
violation raises `TossDirectionInvariantError` rather than going by silently.
"""

import math
import time
from collections import OrderedDict
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict

from .kinder_backend import KinderBackend
from .parameter_feasibility import TossParameterFeasibility
from .types import PlanarCollisionBox, TossFeasibilityGeometry

TOSS_DIRECTIONS_DEG = (0, 90, 180, 270)

# The one approximation in the rule: the real planner is asked about the standoff
# rounded to this resolution, and its answer is cached per state on
# (direction, rounded standoff). Planner feasibility depends only on the state, the
# direction and the standoff -- never speed or release -- and every candidate in a pool
# shares one state, so this turns ~100 planner calls per pool into at most one per
# centimetre of standoff. Asking about the rounded value, rather than the first
# candidate's exact one, keeps each cached answer a pure function of its key.
PLANNER_STANDOFF_RESOLUTION_M = 0.01

# Bounded so a long run cannot grow it without limit; an evicted entry is recomputed,
# and the planner is seeded, so eviction changes cost, never the answer.
_PLAN_CACHE_LIMIT = 8192


class TossDirectionChoice(BaseModel):
    """The selected stand direction for one `(state, standoff)`."""

    model_config = ConfigDict(frozen=True)

    direction_deg: int
    rotation: float
    stand_xy: tuple[float, float]
    clearance_m: float


class NoFeasibleTossDirectionError(RuntimeError):
    """No direction of the four is plannable for this standoff.

    Not a subclass of `NoFeasibleParametersError`, on purpose: that one is the
    empty-pool signal the practice loop replans around, and this must stop the run.
    """

    def __init__(
        self,
        *,
        standoff: float,
        bin_pose: tuple[float, float, float],
        robot_pose: tuple[float, float, float],
        reasons: dict[int, str],
    ) -> None:
        self.standoff = standoff
        self.bin_pose = bin_pose
        self.robot_pose = robot_pose
        self.reasons = dict(reasons)
        per_direction = "; ".join(
            f"{direction}deg: {reason}" for direction, reason in sorted(reasons.items())
        )
        super().__init__(
            f"no feasible toss direction: standoff={standoff}, "
            f"bin_pose={TossDirectionSelector.rounded(values=bin_pose)}, "
            f"robot_pose={TossDirectionSelector.rounded(values=robot_pose)}; {per_direction}"
        )


class TossDirectionInvariantError(RuntimeError):
    """A far-side bin was given a stand that is not west of it."""


class TossDirectionSelector:
    """The direction rule above. A static-method container, never instantiated."""

    # Keyed on the full planner geometry, so two states that differ in anything the
    # base planner reads never share an entry.
    _plan_cache: ClassVar[OrderedDict[tuple[TossFeasibilityGeometry, int, float], str | None]] = (
        OrderedDict()
    )
    planner_calls: ClassVar[int] = 0
    planner_seconds: ClassVar[float] = 0.0

    @staticmethod
    def base_plan_failure(*, snapshot: Any, distance: float, rotation: float) -> str | None:
        """The one swappable feasibility check: `None` if the real base motion planner
        finds a path to this stand, else a reason. Replace this function to change what
        "plannable" means; nothing else in the rule depends on how it is decided."""
        return KinderBackend.toss_base_plan_failure(
            snapshot=snapshot, distance=distance, rotation=rotation
        )

    @staticmethod
    def select_for_state(*, state: Any, standoff: float) -> TossDirectionChoice:
        """`select` on a live `Tossing3DState`; raises if the state carries no geometry,
        since then no direction can be chosen at all."""
        snapshot = getattr(state, "object_centric", None)
        geometry = (
            None if snapshot is None else KinderBackend.toss_feasibility_geometry(snapshot=snapshot)
        )
        if geometry is None:
            raise ValueError(
                "cannot choose a toss direction: the state carries no base-planner "
                "geometry (not a simulator-backed Tossing3DState)"
            )
        return TossDirectionSelector.select(geometry=geometry, snapshot=snapshot, standoff=standoff)

    @staticmethod
    def select(
        *, geometry: TossFeasibilityGeometry, snapshot: Any, standoff: float
    ) -> TossDirectionChoice:
        ranked = sorted(
            TOSS_DIRECTIONS_DEG,
            # Rounded to the micrometre: the controller's float32 cast of 90 vs 270
            # degrees moves mirror-image stands ~1e-7 m apart, which must still tie
            # so the lower angle wins deterministically.
            key=lambda direction: (
                -round(
                    TossDirectionSelector.clearance(
                        geometry=geometry,
                        stand_xy=TossDirectionSelector.stand_point(
                            geometry=geometry, standoff=standoff, direction_deg=direction
                        ),
                    ),
                    6,
                ),
                direction,
            ),
        )
        reasons: dict[int, str] = {}
        for direction in ranked:
            rotation = TossDirectionSelector.rotation(direction_deg=direction)
            reason = TossParameterFeasibility.rejection_reason_from_geometry(
                geometry=geometry, params=np.array([standoff, rotation])
            )
            if reason is None:
                reason = TossDirectionSelector._cached_plan_failure(
                    geometry=geometry, snapshot=snapshot, standoff=standoff, direction=direction
                )
            if reason is not None:
                reasons[direction] = reason
                continue
            stand_xy = TossDirectionSelector.stand_point(
                geometry=geometry, standoff=standoff, direction_deg=direction
            )
            TossDirectionSelector._check_far_bins_stand_west(geometry=geometry, stand_xy=stand_xy)
            return TossDirectionChoice(
                direction_deg=direction,
                rotation=rotation,
                stand_xy=stand_xy,
                clearance_m=TossDirectionSelector.clearance(geometry=geometry, stand_xy=stand_xy),
            )
        raise NoFeasibleTossDirectionError(
            standoff=standoff,
            bin_pose=geometry.bin_pose,
            robot_pose=geometry.robot_pose,
            reasons=reasons,
        )

    @staticmethod
    def rotation(*, direction_deg: int) -> float:
        return math.radians(direction_deg)

    @staticmethod
    def stand_point(
        *, geometry: TossFeasibilityGeometry, standoff: float, direction_deg: int
    ) -> tuple[float, float]:
        """The controller's own target formula, with its float32 parameter cast."""
        distance, rotation = (
            float(value)
            for value in np.asarray(
                [standoff, TossDirectionSelector.rotation(direction_deg=direction_deg)],
                dtype=np.float32,
            )
        )
        yaw = geometry.bin_pose[2] + rotation
        return (
            float(geometry.bin_pose[0] - distance * math.cos(yaw)),
            float(geometry.bin_pose[1] - distance * math.sin(yaw)),
        )

    @staticmethod
    def clearance(*, geometry: TossFeasibilityGeometry, stand_xy: tuple[float, float]) -> float:
        distances = [
            TossDirectionSelector._point_to_box(point=stand_xy, box=obstacle)
            for obstacle in geometry.obstacles
            if obstacle.name != KinderBackend.bin_name
        ]
        return min(distances, default=math.inf)

    @staticmethod
    def clear_plan_cache() -> None:
        TossDirectionSelector._plan_cache.clear()

    @staticmethod
    def rounded(*, values: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(round(float(value), 3) for value in values)

    @staticmethod
    def _cached_plan_failure(
        *, geometry: TossFeasibilityGeometry, snapshot: Any, standoff: float, direction: int
    ) -> str | None:
        rounded_standoff = round(
            round(standoff / PLANNER_STANDOFF_RESOLUTION_M) * PLANNER_STANDOFF_RESOLUTION_M, 6
        )
        key = (geometry, direction, rounded_standoff)
        cache = TossDirectionSelector._plan_cache
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        started = time.perf_counter()
        reason = TossDirectionSelector.base_plan_failure(
            snapshot=snapshot,
            distance=rounded_standoff,
            rotation=TossDirectionSelector.rotation(direction_deg=direction),
        )
        TossDirectionSelector.planner_calls += 1
        TossDirectionSelector.planner_seconds += time.perf_counter() - started
        cache[key] = reason
        if len(cache) > _PLAN_CACHE_LIMIT:
            cache.popitem(last=False)
        return reason

    @staticmethod
    def _check_far_bins_stand_west(
        *, geometry: TossFeasibilityGeometry, stand_xy: tuple[float, float]
    ) -> None:
        barriers = [
            obstacle
            for obstacle in geometry.obstacles
            if KinderBackend.barrier_name in obstacle.name
        ]
        if not barriers:
            return
        barrier_x = barriers[0].center[0]
        bin_x = geometry.bin_pose[0]
        robot_x = geometry.robot_pose[0]
        if (bin_x - barrier_x) * (robot_x - barrier_x) >= 0.0:
            return
        if not stand_xy[0] < bin_x:
            raise TossDirectionInvariantError(
                f"far-side bin at x={bin_x:.3f} was given a stand at "
                f"{TossDirectionSelector.rounded(values=stand_xy)}, not west of the bin"
            )

    @staticmethod
    def _point_to_box(*, point: tuple[float, float], box: PlanarCollisionBox) -> float:
        dx = point[0] - box.center[0]
        dy = point[1] - box.center[1]
        cosine, sine = math.cos(box.yaw), math.sin(box.yaw)
        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy
        outside_x = max(abs(local_x) - box.width / 2, 0.0)
        outside_y = max(abs(local_y) - box.height / 2, 0.0)
        return math.hypot(outside_x, outside_y)

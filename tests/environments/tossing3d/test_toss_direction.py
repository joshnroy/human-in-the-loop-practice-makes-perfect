"""The controller-chosen toss direction: most clearance among the plannable four.

Offline: the geometry is a hand-built copy of the live scene's overhead colliders
(measured at seed 125 -- barrier at x = 1.3, room walls at x = -2.0 / 4.2 and
y = +-3.0, two 45-degree corner colliders), and the real base planner is replaced by a
recording stub, so every test here is about the selection rule rather than RRT.
"""

import math

import numpy as np
import pytest

from hitl_pmp.environments.tossing3d.toss_direction import (
    PLANNER_STANDOFF_RESOLUTION_M,
    TOSS_DIRECTIONS_DEG,
    NoFeasibleTossDirectionError,
    TossDirectionInvariantError,
    TossDirectionSelector,
)
from hitl_pmp.environments.tossing3d.types import PlanarCollisionBox, TossFeasibilityGeometry

_WALLS = (
    PlanarCollisionBox(
        name="collider:tossing_room:3", center=(-1.25, 2.25), width=2.12, height=0.02, yaw=0.785
    ),
    PlanarCollisionBox(
        name="collider:tossing_room:4", center=(-1.25, -2.25), width=2.12, height=0.02, yaw=-0.785
    ),
    PlanarCollisionBox(
        name="collider:tossing_room:5", center=(1.85, 3.0), width=4.7, height=0.02, yaw=0.0
    ),
    PlanarCollisionBox(
        name="collider:tossing_room:6", center=(1.85, -3.0), width=4.7, height=0.02, yaw=0.0
    ),
    PlanarCollisionBox(
        name="collider:tossing_room:7", center=(-2.0, 0.0), width=3.0, height=0.02, yaw=math.pi / 2
    ),
    PlanarCollisionBox(
        name="collider:tossing_room:8", center=(4.2, 0.0), width=6.0, height=0.02, yaw=-math.pi / 2
    ),
)


def _geometry(
    *,
    bin_xy: tuple[float, float],
    bin_yaw: float,
    barrier_height: float = 10.0,
) -> TossFeasibilityGeometry:
    barrier = PlanarCollisionBox(
        name="collider:cuboid_barrier:9",
        center=(1.3, 0.0),
        width=0.06,
        height=barrier_height,
        yaw=0.0,
    )
    bin_box = PlanarCollisionBox(name="bin_0", center=bin_xy, width=0.3, height=0.3, yaw=bin_yaw)
    return TossFeasibilityGeometry(
        robot_pose=(0.0, 0.0, 0.0),
        robot_size=(0.55, 0.55),
        bin_pose=(*bin_xy, bin_yaw),
        obstacles=(bin_box, barrier, *_WALLS),
        sampling_x_bounds=(-2.5, 2.5),
        sampling_y_bounds=(-2.5, 2.5),
    )


class _Planner:
    """Records every (direction, standoff) the selector hands the real planner."""

    def __init__(self, *, failing: frozenset[int] = frozenset()) -> None:
        self.failing = failing
        self.calls: list[tuple[int, float]] = []

    def __call__(self, *, snapshot: object, distance: float, rotation: float) -> str | None:
        del snapshot
        direction = round(math.degrees(rotation)) % 360
        self.calls.append((direction, distance))
        return "base_motion_plan_failed" if direction in self.failing else None


@pytest.fixture
def planner(*, monkeypatch: pytest.MonkeyPatch):
    def install(*, failing: frozenset[int] = frozenset()) -> _Planner:
        stub = _Planner(failing=failing)
        monkeypatch.setattr(TossDirectionSelector, "base_plan_failure", stub)
        return stub

    TossDirectionSelector.clear_plan_cache()
    yield install
    TossDirectionSelector.clear_plan_cache()


def test_the_four_directions_are_the_bin_relative_right_angles() -> None:
    assert TOSS_DIRECTIONS_DEG == (0, 90, 180, 270)
    assert PLANNER_STANDOFF_RESOLUTION_M == 0.01


def test_a_far_bin_is_thrown_at_from_the_west_and_only_that_direction_is_planned(
    *, planner
) -> None:
    """Every non-west stand of a far bin is across the full-width barrier, which the
    cheap geometry gate proves, so the real planner runs once."""
    stub = planner()
    geometry = _geometry(bin_xy=(2.1, 0.0), bin_yaw=0.0)
    choice = TossDirectionSelector.select(geometry=geometry, snapshot=None, standoff=1.9)
    assert choice.direction_deg == 0
    assert choice.stand_xy == pytest.approx((0.2, 0.0), abs=1e-6)
    assert choice.stand_xy[0] < geometry.bin_pose[0]
    assert [direction for direction, _ in stub.calls] == [0]


def test_the_most_clearance_plannable_direction_wins_and_planning_stops_there(*, planner) -> None:
    """A robot-side bin (yaw pi, as the grasp-safe reset region places them) north of
    the room's centre line: east is into the barrier, west is into the wall, so north
    (90) and south (270) survive the gate; south stands farther from every collider."""
    stub = planner()
    geometry = _geometry(bin_xy=(-0.35, 0.3), bin_yaw=math.pi)
    choice = TossDirectionSelector.select(geometry=geometry, snapshot=None, standoff=1.5)
    clearances = {
        direction: TossDirectionSelector.clearance(
            geometry=geometry,
            stand_xy=TossDirectionSelector.stand_point(
                geometry=geometry, standoff=1.5, direction_deg=direction
            ),
        )
        for direction in (90, 270)
    }
    assert clearances[270] > clearances[90]
    assert choice.direction_deg == 270
    assert choice.clearance_m == pytest.approx(clearances[270])
    assert [direction for direction, _ in stub.calls] == [270]


def test_a_planner_failure_falls_through_to_the_next_most_clearance_direction(*, planner) -> None:
    stub = planner(failing=frozenset({270}))
    geometry = _geometry(bin_xy=(-0.35, 0.3), bin_yaw=math.pi)
    choice = TossDirectionSelector.select(geometry=geometry, snapshot=None, standoff=1.5)
    assert choice.direction_deg == 90
    assert [direction for direction, _ in stub.calls] == [270, 90]


def test_selection_is_deterministic_given_state_and_standoff(*, planner) -> None:
    planner()
    geometry = _geometry(bin_xy=(-0.35, 0.0), bin_yaw=math.pi)
    first = TossDirectionSelector.select(geometry=geometry, snapshot=None, standoff=1.5)
    TossDirectionSelector.clear_plan_cache()
    second = TossDirectionSelector.select(geometry=geometry, snapshot=None, standoff=1.5)
    assert first == second
    # A symmetric pair (bin on the centre line) ties exactly; the lower angle wins.
    assert first.direction_deg == 90


def test_a_second_candidate_at_the_same_rounded_standoff_makes_no_planner_call(*, planner) -> None:
    stub = planner()
    geometry = _geometry(bin_xy=(2.1, 0.0), bin_yaw=0.0)
    TossDirectionSelector.select(geometry=geometry, snapshot=None, standoff=1.901)
    TossDirectionSelector.select(geometry=geometry, snapshot=None, standoff=1.904)
    assert len(stub.calls) == 1
    # The planner is asked about the rounded standoff, so its cached answer is exactly
    # a function of the key it is stored under.
    assert stub.calls[0][1] == pytest.approx(1.90)
    TossDirectionSelector.select(geometry=geometry, snapshot=None, standoff=1.93)
    assert len(stub.calls) == 2


def test_the_selector_raises_when_no_direction_is_feasible(*, planner) -> None:
    planner(failing=frozenset(TOSS_DIRECTIONS_DEG))
    geometry = _geometry(bin_xy=(-0.35, 0.3), bin_yaw=math.pi)
    with pytest.raises(NoFeasibleTossDirectionError):
        TossDirectionSelector.select(geometry=geometry, snapshot=None, standoff=1.5)


def test_the_no_feasible_direction_error_carries_every_directions_reason(*, planner) -> None:
    planner(failing=frozenset(TOSS_DIRECTIONS_DEG))
    geometry = _geometry(bin_xy=(-0.35, 0.3), bin_yaw=math.pi)
    with pytest.raises(NoFeasibleTossDirectionError) as caught:
        TossDirectionSelector.select(geometry=geometry, snapshot=None, standoff=1.5)
    error = caught.value
    assert set(error.reasons) == set(TOSS_DIRECTIONS_DEG)
    assert error.reasons[0].startswith("target_collision:collider:cuboid_barrier")
    assert error.reasons[180].startswith("target_collision:collider:tossing_room:7")
    assert error.reasons[90] == "base_motion_plan_failed"
    assert error.reasons[270] == "base_motion_plan_failed"
    message = str(error)
    for fragment in (
        "standoff=1.5",
        "bin_pose=(-0.35, 0.3",
        "robot_pose=(0.0, 0.0",
        "0deg",
        "90deg",
        "180deg",
        "270deg",
        "base_motion_plan_failed",
        "collider:cuboid_barrier",
    ):
        assert fragment in message, fragment


def test_the_no_feasible_direction_error_is_not_the_empty_pool_signal() -> None:
    """Proposal checking turns the selector's error into a rejection; anywhere else --
    executing an accepted candidate -- it is an impossible state, and the empty-pool
    replan, which catches `NoFeasibleParametersError`, must not swallow it."""
    from hitl_pmp.core.method.method import NoFeasibleParametersError

    assert not issubclass(NoFeasibleTossDirectionError, NoFeasibleParametersError)


def test_an_opposite_side_stand_that_is_not_west_of_the_bin_raises(*, planner) -> None:
    """A short barrier can be driven around, so the geometry gate proves nothing and
    the most-clearance direction for this far bin is north -- level with the bin, not
    west of it. The invariant check turns that into an error at selection time."""
    planner()
    geometry = _geometry(bin_xy=(2.5, 0.0), bin_yaw=0.0, barrier_height=1.0)
    with pytest.raises(TossDirectionInvariantError, match="west"):
        TossDirectionSelector.select(geometry=geometry, snapshot=None, standoff=1.25)


@pytest.mark.parametrize("bin_y", [-2.2, -1.0, 0.0, 1.0, 2.2])
@pytest.mark.parametrize("bin_x", [1.48, 2.0, 2.6, 3.42])
@pytest.mark.parametrize("standoff", [1.25, 1.9, 2.6])
def test_far_bins_always_stand_west(
    *, planner, bin_x: float, bin_y: float, standoff: float
) -> None:
    planner()
    geometry = _geometry(bin_xy=(bin_x, bin_y), bin_yaw=0.0)
    try:
        choice = TossDirectionSelector.select(geometry=geometry, snapshot=None, standoff=standoff)
    except NoFeasibleTossDirectionError:
        return
    assert choice.stand_xy[0] < bin_x


def test_clearance_is_the_distance_to_the_nearest_collider_excluding_the_bin() -> None:
    geometry = _geometry(bin_xy=(2.1, 0.0), bin_yaw=0.0)
    # Nearest collider to (0.2, 0) is the barrier's west face at x = 1.27.
    assert TossDirectionSelector.clearance(geometry=geometry, stand_xy=(0.2, 0.0)) == pytest.approx(
        1.07, abs=1e-9
    )
    # Standing right beside the bin measures the barrier's east face (x = 1.33), not
    # the bin 0.05 m away.
    assert TossDirectionSelector.clearance(geometry=geometry, stand_xy=(1.9, 0.0)) == pytest.approx(
        0.57, abs=1e-9
    )


def test_stand_point_matches_the_controllers_target_formula() -> None:
    geometry = _geometry(bin_xy=(-0.35, 0.3), bin_yaw=math.pi)
    for direction in TOSS_DIRECTIONS_DEG:
        yaw = math.pi + math.radians(direction)
        expected = (-0.35 - 1.5 * math.cos(yaw), 0.3 - 1.5 * math.sin(yaw))
        assert TossDirectionSelector.stand_point(
            geometry=geometry, standoff=1.5, direction_deg=direction
        ) == pytest.approx(expected, abs=1e-6)
    assert np.isclose(TossDirectionSelector.rotation(direction_deg=270), 1.5 * math.pi)

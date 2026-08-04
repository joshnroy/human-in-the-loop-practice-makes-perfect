import numpy as np

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import (
    AT_THROW_POSE,
    HAND_EMPTY,
    HOLDING,
    IN_GOAL_REGION,
    REACHABLE,
)

from .conftest import BARRIER_X, BIN_X, GOAL_REGION, build_state, throw_pose_base

_ENV = Tossing3DEnvironment


def _holds(*, predicate, state, objects) -> bool:
    return predicate.holds(state, objects)


def test_in_goal_region_accepts_a_cube_inside_kinders_box() -> None:
    state = build_state(cube=(2.0, 0.0, 0.025))
    assert _holds(predicate=IN_GOAL_REGION, state=state, objects=(_ENV.cube, _ENV.goal_region))


def test_in_goal_region_rejects_a_cube_tossed_to_the_far_end_of_the_bin() -> None:
    """A full-power swing lands at x ~ 2.22, past the region's 2.15 edge, so it misses
    KINDER's own goal. Note the region and the bin *overlap*: the bin's footprint spans
    x in [2.08, 2.38], so being in the bin is not itself disqualifying -- see
    `test_in_goal_region_accepts_a_cube_resting_in_the_near_end_of_the_bin`."""
    state = build_state(cube=(2.218, -0.001, 0.044))
    assert not _holds(predicate=IN_GOAL_REGION, state=state, objects=(_ENV.cube, _ENV.goal_region))


def test_in_goal_region_accepts_a_cube_resting_in_the_near_end_of_the_bin() -> None:
    """Pins the overlap the docs used to deny. The goal region reaches x=2.15 and the
    bin's near edge is x=2.08, so a cube on the bin floor (z=0.044) anywhere in between
    satisfies KINDER's goal."""
    state = build_state(cube=(2.12, 0.0, 0.044))
    assert _holds(predicate=IN_GOAL_REGION, state=state, objects=(_ENV.cube, _ENV.goal_region))


def test_in_goal_region_covers_the_inflation_shells_the_raw_json_range_omits() -> None:
    """The regression test for the bug this domain shipped with.

    KINDER inflates the task JSON's range by `ground_placement_threshold` (0.05 m) per
    side before testing containment; this domain originally scored against the raw range
    and so counted every landing in the resulting 5 cm shells as a miss. Each position
    below is inside the true box and outside the raw JSON range, i.e. exactly the set
    that used to be mis-scored.

    This is offline arithmetic against whatever box `conftest` supplies -- it cannot
    catch a *wrong box*. `test_kinder_fidelity.py::test_goal_region_bounds_match_kinders_own_region`
    is what does that.
    """
    json_range = (1.90, -0.10, 0.0, 2.10, 0.10, 0.10)
    for position in (
        (1.87, 0.0, 0.05),  # -x shell
        (2.12, 0.0, 0.05),  # +x shell
        (2.0, -0.12, 0.05),  # -y shell
        (2.0, 0.12, 0.05),  # +y shell
        (2.0, 0.0, 0.13),  # +z shell
    ):
        outside_json = not all(
            json_range[axis] <= position[axis] <= json_range[axis + 3] for axis in range(3)
        )
        assert outside_json, f"{position} is not in a shell; this test has drifted"
        state = build_state(cube=position)
        assert _holds(predicate=IN_GOAL_REGION, state=state, objects=(_ENV.cube, _ENV.goal_region))


def test_in_goal_region_rejects_a_cube_that_fell_short() -> None:
    state = build_state(cube=(1.66, 0.013, 0.025))
    assert not _holds(predicate=IN_GOAL_REGION, state=state, objects=(_ENV.cube, _ENV.goal_region))


def test_in_goal_region_is_inclusive_of_its_own_boundary() -> None:
    for x in (GOAL_REGION[0], GOAL_REGION[3]):
        state = build_state(cube=(x, 0.0, 0.05))
        assert _holds(predicate=IN_GOAL_REGION, state=state, objects=(_ENV.cube, _ENV.goal_region))


def test_in_goal_region_checks_every_axis_not_just_x() -> None:
    off_in_y = build_state(cube=(2.0, 0.5, 0.025))
    off_in_z = build_state(cube=(2.0, 0.0, 0.5))
    assert not _holds(
        predicate=IN_GOAL_REGION, state=off_in_y, objects=(_ENV.cube, _ENV.goal_region)
    )
    assert not _holds(
        predicate=IN_GOAL_REGION, state=off_in_z, objects=(_ENV.cube, _ENV.goal_region)
    )


def test_hand_empty_and_holding_are_complementary() -> None:
    empty = build_state(holding=0.0)
    full = build_state(holding=1.0, cube=(0.6, 0.0, 0.587))
    assert _holds(predicate=HAND_EMPTY, state=empty, objects=(_ENV.robot,))
    assert not _holds(predicate=HOLDING, state=empty, objects=(_ENV.robot, _ENV.cube))
    assert _holds(predicate=HOLDING, state=full, objects=(_ENV.robot, _ENV.cube))
    assert not _holds(predicate=HAND_EMPTY, state=full, objects=(_ENV.robot,))


def test_reachable_is_false_once_the_cube_is_past_the_barrier() -> None:
    """The domain's irreversibility. A cube in the goal region is a SOLVED task whose
    cube is nonetheless gone for good -- both sides of the barrier matter."""
    on_this_side = build_state(cube=(BARRIER_X - 0.2, 0.0, 0.025))
    in_the_region = build_state(cube=(2.0, 0.0, 0.025))
    in_the_bin = build_state(cube=(2.22, 0.0, 0.044))
    assert _holds(predicate=REACHABLE, state=on_this_side, objects=(_ENV.cube, _ENV.barrier))
    assert not _holds(predicate=REACHABLE, state=in_the_region, objects=(_ENV.cube, _ENV.barrier))
    assert not _holds(predicate=REACHABLE, state=in_the_bin, objects=(_ENV.cube, _ENV.barrier))


def test_at_throw_pose_holds_at_the_standoff_distance_from_the_bin() -> None:
    state = build_state(base=throw_pose_base())
    assert _holds(predicate=AT_THROW_POSE, state=state, objects=(_ENV.robot, _ENV.bin_object))


def test_at_throw_pose_is_false_at_the_start_pose() -> None:
    state = build_state(base=(0.0, 0.0, 0.0))
    assert not _holds(predicate=AT_THROW_POSE, state=state, objects=(_ENV.robot, _ENV.bin_object))


def test_at_throw_pose_measures_distance_not_x_alone() -> None:
    """Standing `throw_standoff` metres away in x but far off in y is not the throw
    pose -- the check is a radius around the bin, matching what move_to_target does."""
    state = build_state(base=(BIN_X - _ENV.throw_standoff, 1.0, 0.0))
    assert not _holds(predicate=AT_THROW_POSE, state=state, objects=(_ENV.robot, _ENV.bin_object))


def test_at_throw_pose_tolerates_the_base_planners_waypoint_slack() -> None:
    inside = _ENV.throw_pose_tolerance / 2
    state = build_state(base=(BIN_X - _ENV.throw_standoff + inside, 0.0, 0.0))
    assert _holds(predicate=AT_THROW_POSE, state=state, objects=(_ENV.robot, _ENV.bin_object))
    outside = _ENV.throw_pose_tolerance * 2
    state = build_state(base=(BIN_X - _ENV.throw_standoff + outside, 0.0, 0.0))
    assert not _holds(predicate=AT_THROW_POSE, state=state, objects=(_ENV.robot, _ENV.bin_object))


def test_every_classifier_reads_only_the_state() -> None:
    """Guards the property the whole CI-testability of this domain rests on: no
    predicate may consult the environment instance, only the State it is handed."""
    state = build_state()
    mutated = state.model_copy(deep=True)
    mutated.set(obj=_ENV.goal_region, feature_name="x_min", feature_val=-np.inf)
    mutated.set(obj=_ENV.goal_region, feature_name="x_max", feature_val=np.inf)
    mutated.set(obj=_ENV.goal_region, feature_name="y_min", feature_val=-np.inf)
    mutated.set(obj=_ENV.goal_region, feature_name="y_max", feature_val=np.inf)
    mutated.set(obj=_ENV.goal_region, feature_name="z_min", feature_val=-np.inf)
    mutated.set(obj=_ENV.goal_region, feature_name="z_max", feature_val=np.inf)
    assert not _holds(predicate=IN_GOAL_REGION, state=state, objects=(_ENV.cube, _ENV.goal_region))
    assert _holds(predicate=IN_GOAL_REGION, state=mutated, objects=(_ENV.cube, _ENV.goal_region))

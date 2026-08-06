"""Offline tests for Tossing3D's six predicates.

`InGoalRegion` gets the most attention, including deliberate boundary probes, because it
*is* the success criterion and because a wrong goal box has already shipped once in this
project's history. `test_kinder_fidelity.py` checks it against KINDER's own
`_check_goals()` whenever the simulator is installed; these run everywhere.
"""

import numpy as np
import pytest

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import (
    GRASP_THRESHOLD,
    HAND_EMPTY,
    HANDEMPTY_TOL,
    HOLDING,
    IN_GOAL_REGION,
    NEAR_BIN,
    NEAR_BIN_TOLERANCE,
    ON_GROUND,
    REACHABLE,
    THROW_STANDOFF_BOUNDS,
    HandEmptyClassifier,
    HoldingClassifier,
    InGoalRegionClassifier,
    NearBinClassifier,
    OnGroundClassifier,
    ReachableClassifier,
)

from .observations import BARRIER_X, COINCIDENT_BIN_X, CUBE_START_X, GOAL_REGION_BBOX, state

_ENV = Tossing3DEnvironment()


def _in_goal_region(*, x: float, y: float = 0.0, z: float = 0.0444) -> bool:
    return InGoalRegionClassifier.holds(
        state=state(cube_x=x, cube_y=y, cube_z=z),
        cube=_ENV.cube,
        goal_region=_ENV.goal_region,
    )


def test_in_goal_region_accepts_the_measured_coincident_landing() -> None:
    """x = 1.9902, z = 0.0444 is where the oracle's cube comes to rest on the coincident
    config at standoff 1.35 -- inside the bin, and inside the goal box. `_check_goals()`
    says True there, so this must too."""
    assert _in_goal_region(x=1.9902, y=0.0105, z=0.0444)


def test_in_goal_region_rejects_the_measured_stock_landing() -> None:
    """The same throw on stock rests at x = 2.2197, past the goal box's 2.15 far edge,
    and `_check_goals()` says False. Landing in the bin is a scored failure there."""
    assert not _in_goal_region(x=2.2197, y=0.0103, z=0.0444)


@pytest.mark.parametrize("x", [1.85, 2.15])
def test_in_goal_region_is_inclusive_at_the_boundary(*, x: float) -> None:
    """Upstream's `Region.check_in_region` uses `>=`/`<=`, so this does too. A strict
    comparison would disagree with `_check_goals()` on exactly the measure-zero set that
    is hardest to notice and easiest to argue about."""
    assert _in_goal_region(x=x)


@pytest.mark.parametrize("x", [1.8499, 2.1501])
def test_in_goal_region_rejects_just_outside_the_boundary(*, x: float) -> None:
    assert not _in_goal_region(x=x)


def test_in_goal_region_tests_all_three_axes_not_just_x() -> None:
    """A toss controls x, so x is where attention goes -- but a cube that flew sideways
    or is still in the air is not in the region either, and upstream checks all three."""
    assert not _in_goal_region(x=2.0, y=0.5)
    assert not _in_goal_region(x=2.0, z=0.4)


def test_in_goal_region_follows_the_box_it_is_given_rather_than_a_constant() -> None:
    """The box is per-episode state, not a literal: if upstream ever moves
    `blocks_goal_region`, the predicate must move with it."""
    shifted = list(GOAL_REGION_BBOX)
    shifted[0], shifted[3] = 3.0, 3.3
    assert not InGoalRegionClassifier.holds(
        state=state(cube_x=2.0, goal_region=tuple(shifted)),
        cube=_ENV.cube,
        goal_region=_ENV.goal_region,
    )
    assert InGoalRegionClassifier.holds(
        state=state(cube_x=3.1, goal_region=tuple(shifted)),
        cube=_ENV.cube,
        goal_region=_ENV.goal_region,
    )


def test_hand_empty_holds_only_at_an_open_gripper() -> None:
    assert HandEmptyClassifier.holds(state=state(gripper=0.0), robot=_ENV.robot)
    assert HandEmptyClassifier.holds(state=state(gripper=HANDEMPTY_TOL / 2), robot=_ENV.robot)
    assert not HandEmptyClassifier.holds(state=state(gripper=0.5), robot=_ENV.robot)


def test_holding_needs_both_a_closed_gripper_and_a_lifted_cube() -> None:
    """Upstream's own two conjuncts. Either alone is not enough: a closed gripper with the
    cube still on the floor is a failed grasp, and a cube in the air with an open gripper
    is a cube in flight."""
    assert HoldingClassifier.holds(
        state=state(gripper=GRASP_THRESHOLD + 0.5, cube_z=0.4), robot=_ENV.robot, cube=_ENV.cube
    )
    assert not HoldingClassifier.holds(
        state=state(gripper=GRASP_THRESHOLD + 0.5, cube_z=0.025),
        robot=_ENV.robot,
        cube=_ENV.cube,
    )
    assert not HoldingClassifier.holds(
        state=state(gripper=0.0, cube_z=0.4), robot=_ENV.robot, cube=_ENV.cube
    )


def test_on_ground_holds_for_a_cube_resting_flat_on_the_floor() -> None:
    """z - bb_z/2 == 0 is the floor: the cube's centre sits one half-extent up."""
    assert OnGroundClassifier.holds(state=state(cube_z=0.025), cube=_ENV.cube)


def test_on_ground_rejects_a_cube_in_the_air() -> None:
    assert not OnGroundClassifier.holds(state=state(cube_z=0.4), cube=_ENV.cube)


def test_on_ground_rejects_a_cube_resting_on_a_corner() -> None:
    """Upstream's flatness conjunct, and it is load-bearing rather than decorative:
    `pick_shelf` builds its grasp pose from the object's orientation."""
    assert not OnGroundClassifier.holds(state=state(cube_z=0.025, cube_qx=0.7), cube=_ENV.cube)
    assert not OnGroundClassifier.holds(state=state(cube_z=0.025, cube_qy=0.7), cube=_ENV.cube)


def test_reachable_is_a_one_way_door_across_the_barrier() -> None:
    """This is the domain's whole point in one predicate: the base cannot cross the
    barrier, so a cube past it can never be picked up again."""
    assert ReachableClassifier.holds(
        state=state(cube_x=CUBE_START_X), cube=_ENV.cube, barrier=_ENV.barrier
    )
    assert not ReachableClassifier.holds(
        state=state(cube_x=1.99), cube=_ENV.cube, barrier=_ENV.barrier
    )


def test_reachable_reads_the_barriers_live_x_rather_than_a_constant() -> None:
    """The barrier's pose is sampled from `barrier_init_region` per episode, so a literal
    would be right for one seed and quietly wrong for the next."""
    beyond_default = state(cube_x=BARRIER_X + 0.5)
    assert not ReachableClassifier.holds(state=beyond_default, cube=_ENV.cube, barrier=_ENV.barrier)
    beyond_default.set(obj=_ENV.barrier, feature_name="x", feature_val=BARRIER_X + 1.0)
    assert ReachableClassifier.holds(state=beyond_default, cube=_ENV.cube, barrier=_ENV.barrier)


@pytest.mark.parametrize("standoff", [THROW_STANDOFF_BOUNDS[0], 1.35, THROW_STANDOFF_BOUNDS[1]])
def test_near_bin_holds_across_the_whole_sampled_standoff_range(*, standoff: float) -> None:
    """Every standoff the sampler can draw must satisfy the add effect of the skill that
    draws it, including both endpoints."""
    assert NearBinClassifier.holds(
        state=state(base_x=COINCIDENT_BIN_X - standoff), robot=_ENV.robot, target=_ENV.bin
    )


def test_near_bin_tolerates_exactly_what_the_controller_tolerates() -> None:
    """`move_to_target` stops once the base is within `WAYPOINT_TOL` of its own planned
    waypoint, so a pose that far off a band endpoint still has to count."""
    just_inside = COINCIDENT_BIN_X - (THROW_STANDOFF_BOUNDS[1] + NEAR_BIN_TOLERANCE * 0.9)
    assert NearBinClassifier.holds(
        state=state(base_x=just_inside), robot=_ENV.robot, target=_ENV.bin
    )
    just_outside = COINCIDENT_BIN_X - (THROW_STANDOFF_BOUNDS[1] + NEAR_BIN_TOLERANCE * 1.1)
    assert not NearBinClassifier.holds(
        state=state(base_x=just_outside), robot=_ENV.robot, target=_ENV.bin
    )


def test_near_bin_rejects_standing_on_top_of_the_bin() -> None:
    """The lower bound is not decoration. A base at the bin's own position has nowhere to
    throw from, and an operator model that called that a throw pose would let a planner
    skip the walk entirely."""
    assert not NearBinClassifier.holds(
        state=state(base_x=COINCIDENT_BIN_X), robot=_ENV.robot, target=_ENV.bin
    )


def test_near_bin_rejects_the_far_side_of_the_room() -> None:
    """Derived from the upper bound rather than written as a literal, so that widening the
    bounds again cannot quietly turn "the far side of the room" into a legal throw pose."""
    far_side = COINCIDENT_BIN_X - (THROW_STANDOFF_BOUNDS[1] + 0.2)
    assert not NearBinClassifier.holds(
        state=state(base_x=far_side), robot=_ENV.robot, target=_ENV.bin
    )


def test_near_bin_rejects_the_worst_measured_pose_that_pick_leaves_the_base_in() -> None:
    """The constraint that sets the upper bound, pinned offline against a real pose.

    `Pick` drives the base to the *cube*, and the widened band would admit that pose if it
    reached far enough -- whereupon the oracle skips `MoveToThrowPose` and throws from
    wherever it stands, which is the failure `NearBinClassifier`'s docstring records. Over
    30 scene seeds exactly one leaves the base inside the 0.04 m lateral tolerance: seed
    14, at 1.8592 m from the bin and 0.0074 m off its axis. The lateral conjunct cannot
    save that one, so the standoff conjunct has to, and it does only while the upper bound
    stays below 1.8592 - NEAR_BIN_TOLERANCE."""
    assert not NearBinClassifier.holds(
        state=state(base_x=COINCIDENT_BIN_X - 1.8592, base_y=0.0074),
        robot=_ENV.robot,
        target=_ENV.bin,
    )


def test_near_bin_rejects_a_base_off_the_bins_axis() -> None:
    """The conjunct a plain distance test does not have, and the one that matters.

    `pick_shelf` drives the base to the *cube*, off to one side. Measured on the canonical
    scene, that leaves the base 1.76 m from the bin and 0.37 m off its axis -- a distance
    a loose band happily admits. `NearBin` was briefly written that way, and the
    consequence was that the oracle skipped `MoveToThrowPose` entirely and threw from a
    pose facing 40 degrees off: the cube landed at (0.9969, -0.7196) and the episode
    failed. See `NearBinClassifier`'s own docstring."""
    assert not NearBinClassifier.holds(
        state=state(base_x=0.279, base_y=0.366), robot=_ENV.robot, target=_ENV.bin
    )


def test_near_bin_rejects_a_diagonal_approach_at_the_right_distance() -> None:
    """The general form of the case above: the right distance is not enough, because
    `MoveToThrowPose` pins `rot = 0` and therefore always ends on the bin's axis."""
    offset = 1.35 / np.sqrt(2)
    assert not NearBinClassifier.holds(
        state=state(base_x=COINCIDENT_BIN_X - offset, base_y=offset),
        robot=_ENV.robot,
        target=_ENV.bin,
    )


def test_every_predicate_declares_the_types_it_is_actually_applied_to() -> None:
    """A `Predicate`'s `types` is what `SkillGrounder` enumerates over, so a wrong entry
    silently grounds a predicate onto objects it was never written for."""
    expected = {
        IN_GOAL_REGION: (
            Tossing3DEnvironment.cube_type,
            Tossing3DEnvironment.goal_region_type,
        ),
        HAND_EMPTY: (Tossing3DEnvironment.robot_type,),
        HOLDING: (Tossing3DEnvironment.robot_type, Tossing3DEnvironment.cube_type),
        ON_GROUND: (Tossing3DEnvironment.cube_type,),
        REACHABLE: (Tossing3DEnvironment.cube_type, Tossing3DEnvironment.barrier_type),
        NEAR_BIN: (Tossing3DEnvironment.robot_type, Tossing3DEnvironment.bin_type),
    }
    for predicate, types in expected.items():
        assert predicate.types == types, predicate.name


def test_the_lambda_adapters_pass_objects_through_in_declaration_order() -> None:
    """`Predicate.holds` is a positional `(state, objects)` callable, and each predicate
    here adapts that to a keyword-only classifier. A transposed pair would typecheck and
    then silently ask whether the barrier is left of the cube."""
    airborne = state(gripper=0.9, cube_z=0.4)
    assert HOLDING.holds(airborne, (_ENV.robot, _ENV.cube))
    assert REACHABLE.holds(state(cube_x=CUBE_START_X), (_ENV.cube, _ENV.barrier))
    assert IN_GOAL_REGION.holds(state(cube_x=1.99), (_ENV.cube, _ENV.goal_region))

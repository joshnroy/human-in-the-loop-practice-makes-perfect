"""Offline tests for Tossing3D's six predicates.

`InBin` gets the most attention, including deliberate boundary probes, because it
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
    IN_BIN,
    ON_GROUND,
    REACHABLE,
    ROBOT_AT_SUCCESSFUL_THROW_POSE,
    THROW_OVERSHOOT_MARGIN,
    THROW_POSE_LATERAL_TOLERANCE,
    THROW_SHORTFALL_MARGIN,
    THROW_STANDOFF_BOUNDS,
    HandEmptyClassifier,
    HoldingClassifier,
    InBinClassifier,
    OnGroundClassifier,
    ReachableClassifier,
    RobotAtSuccessfulThrowPoseClassifier,
)
from hitl_pmp.environments.tossing3d.skill_oracle_policy import ORACLE_THROW_STANDOFF

from .observations import BARRIER_X, BIN_X, CUBE_START_X, GOAL_REGION_BBOX, state

_ENV = Tossing3DEnvironment()


def _in_bin(*, x: float, y: float = 0.0, z: float = 0.0444) -> bool:
    return InBinClassifier.holds(
        state=state(cube_x=x, cube_y=y, cube_z=z), cube=_ENV.cube, target=_ENV.bin
    )


def test_in_bin_accepts_the_measured_landing() -> None:
    """x = 1.9902, z = 0.0444 is where the oracle's cube comes to rest on the shipped
    scene at standoff 1.35 -- inside the bin, and inside the goal box.
    `_check_goals()` says True there, so this must too."""
    assert _in_bin(x=1.9902, y=0.0105, z=0.0444)


def test_in_bin_rejects_a_landing_past_the_far_edge() -> None:
    """x = 2.2197 is past the goal box's 2.15 far edge, so this must be False. It is a
    measured point rather than an invented one: it is where this same throw came to rest
    on the scene KINDER shipped before `kindergarden` PR #126, whose bin sat 23 cm too
    far out -- a cube landing in that bin scored a failure, and `_check_goals()` agreed
    with this predicate that it was outside the region."""
    assert not _in_bin(x=2.2197, y=0.0103, z=0.0444)


@pytest.mark.parametrize("x", [1.85, 2.15])
def test_in_bin_is_inclusive_at_the_boundary(*, x: float) -> None:
    """Upstream's `Region.check_in_region` uses `>=`/`<=`, so this does too. A strict
    comparison would disagree with `_check_goals()` on exactly the measure-zero set that
    is hardest to notice and easiest to argue about."""
    assert _in_bin(x=x)


@pytest.mark.parametrize("x", [1.8499, 2.1501])
def test_in_bin_rejects_just_outside_the_boundary(*, x: float) -> None:
    assert not _in_bin(x=x)


def test_in_bin_tests_all_three_axes_not_just_x() -> None:
    """A toss controls x, so x is where attention goes -- but a cube that flew sideways
    or is still in the air is not in the region either, and upstream checks all three."""
    assert not _in_bin(x=2.0, y=0.5)
    assert not _in_bin(x=2.0, z=0.4)


def test_in_bin_follows_the_box_it_is_given_rather_than_a_constant() -> None:
    """The box is per-episode state, not a literal: if upstream ever moves
    `blocks_goal_region`, the predicate must move with it."""
    shifted = list(GOAL_REGION_BBOX)
    shifted[0], shifted[3] = 3.0, 3.3
    assert not InBinClassifier.holds(
        state=state(cube_x=2.0, goal_region=tuple(shifted)), cube=_ENV.cube, target=_ENV.bin
    )
    assert InBinClassifier.holds(
        state=state(cube_x=3.1, goal_region=tuple(shifted)), cube=_ENV.cube, target=_ENV.bin
    )


def test_in_bin_reads_the_scored_box_not_the_bins_pose() -> None:
    """**The guard the bin-is-the-goal-region simplification makes necessary.**

    The box now rides on the bin object, which makes "derive it from the bin's own x plus a
    half-extent" look like a tidy simplification. It is not: KINDER scores against
    `blocks_goal_region`, whose bbox is the task JSON's range inflated by
    `ground_placement_threshold`, and a box re-derived from the bin's pose would disagree
    with `_check_goals()` -- the exact defect that has already shipped once here.

    Moving the bin's pose while holding the box still must therefore not move the verdict.
    That the two *coincide* in the shipped scene is the domain's assumption; that the
    predicate *reads* the box rather than the pose is what keeps it correct anyway."""
    far = state(cube_x=2.0, bin_x=BIN_X + 5.0)
    assert InBinClassifier.holds(state=far, cube=_ENV.cube, target=_ENV.bin)


def test_in_bin_scores_a_bin_off_the_box_the_way_kinder_does() -> None:
    """The honest consequence of the assumption, pinned so nobody "fixes" it later.

    Where the bin and the scored region come apart, the bin object's own `x` falls outside
    its own `x_max` and `InBin` is true exactly when the cube is *not* in the bin. The
    **name** is wrong on such a scene; the **arithmetic** is not, and the arithmetic is
    what `_check_goals()` agreement depends on. A future change that made `InBin` follow
    the bin's pose would flip both of these and look like a bugfix.

    That geometry is no longer one this domain can load -- #237 retired the task-config
    enum, so the scene is whatever the installed KINDER registers, and there the two
    coincide. The observation here is hand-built, so the case stays expressible, and it is
    the one the assumption is weakest on: it is exactly the scene that shipped for months
    before `kindergarden` PR #126, whose measured numbers are the ones used below."""
    # The bin's measured x before PR #126 moved it back onto the box that scores. Local
    # rather than in `observations.py`: no scene puts the bin here any more, so this is
    # this test's own historical datum rather than the domain's geometry.
    bin_off_the_box = 2.2305

    in_the_bin = state(cube_x=2.2197, cube_z=0.0444, bin_x=bin_off_the_box)
    assert not InBinClassifier.holds(state=in_the_bin, cube=_ENV.cube, target=_ENV.bin)
    on_the_scored_box = state(cube_x=2.0, cube_z=0.0444, bin_x=bin_off_the_box)
    assert InBinClassifier.holds(state=on_the_scored_box, cube=_ENV.cube, target=_ENV.bin)


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


def _at_throw_pose(*, standoff: float, base_y: float = 0.0, **kwargs) -> bool:
    """Does the predicate hold for a base placed `standoff` metres in front of the bin?"""
    bin_x = kwargs.pop("bin_x", BIN_X)
    return RobotAtSuccessfulThrowPoseClassifier.holds(
        state=state(base_x=bin_x - standoff, base_y=base_y, bin_x=bin_x, **kwargs),
        robot=_ENV.robot,
        target=_ENV.bin,
    )


def _accepted_band(**kwargs) -> tuple[float, float]:
    """The interval of standoffs the predicate accepts, found by scanning at 1 mm."""
    accepted = [
        standoff / 1000
        for standoff in range(0, 3000)
        if _at_throw_pose(standoff=standoff / 1000, **kwargs)
    ]
    return (min(accepted), max(accepted))


def test_the_accepted_band_is_strictly_narrower_than_the_range_the_sampler_draws_from() -> None:
    """**The property whose absence is the whole reason this predicate was rewritten.**

    `MoveToThrowPose`'s only add effect is this predicate, and EES trains that skill's
    success classifier on exactly that label. While the predicate accepted every standoff
    in `THROW_STANDOFF_BOUNDS` -- the interval the sampler draws from -- the label was
    constant-true, the classifier saw one class, and every draw fell back to uniform. If
    this ever fails again, the sampler has silently stopped being able to learn."""
    rng = np.random.default_rng(0)
    draws = [float(rng.uniform(*THROW_STANDOFF_BOUNDS)) for _ in range(200)]
    accepted = [d for d in draws if _at_throw_pose(standoff=d)]
    assert accepted, "no sampled standoff is accepted: the skill could never succeed"
    assert len(accepted) < len(draws), (
        "every sampled standoff is accepted: the add effect is constant-true and no "
        "sampler can learn from it"
    )


def test_the_accepted_band_moves_when_the_bin_moves() -> None:
    """**The test that catches someone reintroducing a measured literal.**

    The band is derived from live scene geometry plus two fixed margins, so shifting the
    bin shifts the whole band by exactly the same amount. A predicate that instead
    hard-coded `[1.15, 1.375]` -- the *value* the derivation happens to produce on the
    scene as shipped -- would hold the band still and be silently wrong the moment the bin
    moves. That is not hypothetical: kindergarden#126 has since moved it, which is exactly
    the event this test was written against."""
    shift = 0.25
    here = _accepted_band()
    there = _accepted_band(bin_x=BIN_X + shift)
    assert there[0] == pytest.approx(here[0] + shift, abs=2e-3)
    assert there[1] == pytest.approx(here[1] + shift, abs=2e-3)


def test_the_accepted_band_moves_when_the_goal_region_moves() -> None:
    """The other half of "derived, not pinned": the band tracks the goal box too, so a
    task JSON that retargets the throw needs no change here."""
    x_min, y_min, z_min, x_max, y_max, z_max = GOAL_REGION_BBOX
    shifted = (x_min - 0.30, y_min, z_min, x_max - 0.30, y_max, z_max)
    here = _accepted_band()
    there = _accepted_band(goal_region=shifted)
    assert there[0] == pytest.approx(here[0] + 0.30, abs=2e-3)
    assert there[1] == pytest.approx(here[1] + 0.30, abs=2e-3)


def test_the_accepted_band_matches_the_measured_five_of_five_core() -> None:
    """**The discriminating test.** PR #105's finer sweep (5 scene seeds, 0.025 m
    resolution, bin on the goal region) found a 5/5 core of `[1.150, 1.375]`, narrower than the
    geometric band `[1.125, 1.425]` on both ends -- 2/5 and 3/5 partial-solving at the old
    edges, not the 3/3-or-nothing the coarser 48-episode grid implied. The predicate now
    trims the goal box by `THROW_OVERSHOOT_MARGIN`/`THROW_SHORTFALL_MARGIN` to land exactly
    on that measured core.

    Landing-space and standoff-space run in *opposite* directions
    (`landing_x = base_x + THROW_RANGE`, `base_x = bin_x - standoff`, so a larger standoff
    gives a *smaller* landing_x), so it is easy to trim the wrong edge of the box and get a
    band that is still 0.225 m wide but shifted the wrong way -- e.g. `[1.175, 1.400]`,
    which would reject the reliable 1.150 endpoint and accept the partial-solving 1.400
    one. This test catches that inversion; the width-only check below does not."""
    assert _accepted_band() == pytest.approx((1.15, 1.375), abs=2e-3)


def test_the_accepted_band_is_narrower_than_the_goal_regions_x_extent() -> None:
    """The tightened band is the box's own width (0.300 m) minus both margins -- 0.225 m --
    not the full extent any more. This is the arithmetic check that the derivation is
    still a derivation and not a fit: it holds for any box at any bin position, unlike the
    test above, which pins the one measured core."""
    low, high = _accepted_band()
    full_extent = GOAL_REGION_BBOX[3] - GOAL_REGION_BBOX[0]
    assert high - low == pytest.approx(
        full_extent - (THROW_OVERSHOOT_MARGIN + THROW_SHORTFALL_MARGIN), abs=2e-3
    )


def test_the_oracle_standoff_is_inside_the_accepted_band() -> None:
    """`SkillOraclePolicy` throws from 1.35 and scores 99/100. If tightening the predicate
    ever excluded that standoff the oracle would stop planning a throw at all, and every
    EES number on this domain is measured against that ceiling. This is one of the two
    reasons a further tightening to `[1.150, 1.325]` (proposed by PR #196, investigated and
    rejected in `docs/experiment-logs/2026-08-10-tossing3d-throw-band-retightening-sweep.md`)
    was not landed: 1.35 would fail this assertion."""
    assert _at_throw_pose(standoff=ORACLE_THROW_STANDOFF)


@pytest.mark.parametrize("standoff", [1.16, 1.25, 1.35, 1.37])
def test_the_band_accepts_every_standoff_measured_to_solve(*, standoff: float) -> None:
    """PR #105's finer sweep (5 scene seeds, 0.025 m resolution) solved 5/5 at every point
    from 1.150 through 1.375. These four sit safely inside that 5/5 core -- a millimetre or
    more clear of both edges, so this test is not sensitive to `BIN_X`'s own
    0.1 mm offset from a round 2.0 -- while `test_the_accepted_band_matches_the_measured_
    five_of_five_core` pins the edges themselves via a fine scan. A predicate rejecting one
    of these four would be calling a pose a failure that demonstrably scores on every seed
    tested, not just most of them."""
    assert _at_throw_pose(standoff=standoff)


@pytest.mark.parametrize(
    "standoff", [0.45, 0.80, 1.00, 1.10, 1.125, 1.40, 1.425, 1.45, 1.55, 1.65, 1.75]
)
def test_the_band_rejects_every_standoff_measured_not_to_solve(*, standoff: float) -> None:
    """`1.10`/`1.45` and beyond are the coarse 48-episode grid's 0/3 points. `1.125` and
    `1.425` -- the old geometric band's own edges -- and `1.40` are PR #105's finer-grained
    2/5, 2/5 and 3/5: not zero, but not the 5/5 this predicate now requires either.

    **This predicate is independent of `THROW_STANDOFF_BOUNDS`** -- it takes a raw
    standoff and says nothing about whether the sampler could ever draw it -- so every
    value below still belongs here regardless of where the sampler's own range sits.
    That said, the sampler's range moved: `0.45` and `0.80` predate the barrier-collision
    tightening and are no longer standoffs `MoveToThrowPose` can draw at all (the
    sampler's floor is now 1.10 m; see `predicates.THROW_STANDOFF_BOUNDS`). `1.10`
    happens to equal the *new* lower bound and correctly still rejects here -- the
    predicate's own accepted band starts at 1.15, not 1.10 -- so it needs no removal.
    `1.75` remains the upper bound both before and after that change."""
    assert not _at_throw_pose(standoff=standoff)


def test_the_predicate_rejects_standing_on_top_of_the_bin() -> None:
    """A base at the bin's own position throws the cube clean over it -- 1.275 m past a
    box whose far edge is 0.15 m away."""
    assert not _at_throw_pose(standoff=0.0)


def test_the_predicate_rejects_the_far_side_of_the_room() -> None:
    assert not _at_throw_pose(standoff=THROW_STANDOFF_BOUNDS[1] + 0.2)


def test_the_predicate_rejects_the_worst_measured_pose_that_pick_leaves_the_base_in() -> None:
    """`Pick` drives the base to the *cube*, and if the predicate held there the oracle
    would skip `MoveToThrowPose` and throw from wherever it stood. Over 30 scene seeds
    exactly one leaves the base inside the 0.04 m lateral tolerance: seed 14, at 1.8592 m
    from the bin and 0.0074 m off its axis. The lateral conjunct cannot save that one, so
    the standoff conjunct has to -- and now does so on its own terms, because a throw from
    1.8592 m back lands at x = 1.42, well short of the box."""
    assert not _at_throw_pose(standoff=1.8592, base_y=0.0074)


def test_the_predicate_rejects_a_base_off_the_bins_axis() -> None:
    """The conjunct a plain distance test does not have, and the one that was measured.

    `pick_shelf` drives the base to the *cube*, off to one side. On the canonical scene
    that leaves the base 1.76 m from the bin and 0.37 m off its axis -- a distance a loose
    band happily admits. The predicate was briefly written that way, and the oracle
    skipped `MoveToThrowPose` and threw facing 40 degrees off: the cube landed at
    (0.9969, -0.7196) and the episode failed."""
    assert not RobotAtSuccessfulThrowPoseClassifier.holds(
        state=state(base_x=0.279, base_y=0.366),
        robot=_ENV.robot,
        target=_ENV.bin,
    )


def test_the_predicate_rejects_a_diagonal_approach_at_the_right_distance() -> None:
    """The general form of the case above: the right distance is not enough, because
    `MoveToThrowPose` pins `rot = 0` and therefore always ends on the bin's axis."""
    offset = 1.35 / np.sqrt(2)
    assert not RobotAtSuccessfulThrowPoseClassifier.holds(
        state=state(base_x=BIN_X - offset, base_y=offset),
        robot=_ENV.robot,
        target=_ENV.bin,
    )


def test_the_lateral_conjunct_still_tolerates_exactly_what_the_controller_tolerates() -> None:
    """`move_to_target` stops once the base is within `WAYPOINT_TOL` of its own planned
    waypoint, so a pose that far off the axis still has to count."""
    assert _at_throw_pose(standoff=ORACLE_THROW_STANDOFF, base_y=THROW_POSE_LATERAL_TOLERANCE * 0.9)
    assert not _at_throw_pose(
        standoff=ORACLE_THROW_STANDOFF, base_y=THROW_POSE_LATERAL_TOLERANCE * 1.1
    )


def test_every_predicate_declares_the_types_it_is_actually_applied_to() -> None:
    """A `Predicate`'s `types` is what `SkillGrounder` enumerates over, so a wrong entry
    silently grounds a predicate onto objects it was never written for."""
    expected = {
        IN_BIN: (
            Tossing3DEnvironment.cube_type,
            Tossing3DEnvironment.bin_type,
        ),
        HAND_EMPTY: (Tossing3DEnvironment.robot_type,),
        HOLDING: (Tossing3DEnvironment.robot_type, Tossing3DEnvironment.cube_type),
        ON_GROUND: (Tossing3DEnvironment.cube_type,),
        REACHABLE: (Tossing3DEnvironment.cube_type, Tossing3DEnvironment.barrier_type),
        ROBOT_AT_SUCCESSFUL_THROW_POSE: (
            Tossing3DEnvironment.robot_type,
            Tossing3DEnvironment.bin_type,
        ),
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
    assert IN_BIN.holds(state(cube_x=1.99), (_ENV.cube, _ENV.bin))

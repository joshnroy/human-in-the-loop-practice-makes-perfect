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
    ACCEPTED_THROW_STANDOFF_BOUNDS,
    GRASP_THRESHOLD,
    HAND_EMPTY,
    HANDEMPTY_TOL,
    HOLDING,
    IN_BIN,
    ON_GROUND,
    REACHABLE,
    ROBOT_AT_SUCCESSFUL_THROW_POSE,
    THROW_POSE_TOLERANCE,
    THROW_RANGE,
    THROW_RANGE_MAX,
    THROW_RANGE_MIN,
    THROW_STANDOFF_BOUNDS,
    WAYPOINT_TOLERANCE,
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


def test_the_accepted_base_position_follows_the_bin_while_the_standoff_band_does_not() -> None:
    """Upstream's band is on `dx = bin_x - pos_base_x`, a distance measured *from the
    bin*, so the two spaces behave differently and it is worth pinning which is which.

    In **standoff** space the band is invariant: 1.09 to 1.400 from the bin, wherever the
    bin is. In **world** space the accepted base positions therefore slide by exactly the
    bin's own shift. The previous derived band moved in *both* spaces, because it was a
    function of the scored box rather than of the bin, so this is a real change of
    behaviour and not a restatement."""
    # Large enough that every standoff below lands clear of the band once the bin moves.
    shift = 0.40
    assert _accepted_band(bin_x=BIN_X + shift) == pytest.approx(_accepted_band(), abs=2e-3)
    for standoff in (1.10, ORACLE_THROW_STANDOFF, 1.39):
        base_x = BIN_X - standoff
        assert RobotAtSuccessfulThrowPoseClassifier.holds(
            state=state(base_x=base_x), robot=_ENV.robot, target=_ENV.bin
        )
        assert not RobotAtSuccessfulThrowPoseClassifier.holds(
            state=state(base_x=base_x, bin_x=BIN_X + shift), robot=_ENV.robot, target=_ENV.bin
        )


def test_the_accepted_band_no_longer_follows_the_goal_region() -> None:
    """**A capability deliberately given up, asserted so nobody assumes otherwise.**

    hitl's previous band was *derived* from the live scored box, so retargeting the
    throw in the task JSON needed no change here. kinder-baselines' band is a literal on
    `dx`, and its own comment concedes the cost: "A literal, not a live read of the
    region, so a scene whose bin or goal region moves needs remeasuring."

    Moving the goal region 0.30 m therefore leaves the band exactly where it was. That
    is the trade accepted when this predicate converged onto upstream's definition, and
    it is asserted rather than merely noted, so a reader who expects the old behaviour
    finds this test instead of a silent wrong answer."""
    x_min, y_min, z_min, x_max, y_max, z_max = GOAL_REGION_BBOX
    shifted = (x_min - 0.30, y_min, z_min, x_max - 0.30, y_max, z_max)
    assert _accepted_band(goal_region=shifted) == pytest.approx(_accepted_band(), abs=2e-3)


def test_the_accepted_band_is_upstreams_measured_band_with_a_widened_far_edge() -> None:
    """The band is upstream's literal, not a derivation, and its far edge is this
    repo's own 1.400 rather than kinder-baselines' 1.375.

    **The near edge is upstream's unchanged.** It sits below the sampler's 1.10 m floor,
    so inside `THROW_STANDOFF_BOUNDS` this is a one-sided threshold at the far edge.

    See `RobotAtSuccessfulThrowPoseClassifier`'s docstring for why the far edge moved
    and for the question that was left open when it did."""
    assert ACCEPTED_THROW_STANDOFF_BOUNDS == (1.09, 1.400)
    low, high = _accepted_band()
    assert (low, high) == pytest.approx(ACCEPTED_THROW_STANDOFF_BOUNDS, abs=2e-3)


def test_the_far_edge_clears_the_worst_measured_overshoot_at_the_oracle_standoff() -> None:
    """**The reason the far edge is 1.400 and not upstream's 1.375.**

    `move_to_target` stops long of its commanded standoff by a per-seed constant
    measured at 0.7-27.7 mm over 5 scene seeds, so the oracle's commanded 1.35 lands at
    an *achieved* `dx` of up to 1.3777. Upstream's 1.375 edge falls inside that spread,
    which would make this predicate false at a pose the oracle has just correctly moved
    to -- and `SkillOraclePolicy` branches on it, so the oracle would re-issue
    `MoveToThrowPose` forever instead of throwing.

    1.3777 is the worst achieved value measured; this pins the margin that protects it.
    That the pose genuinely scores was measured rather than assumed -- 40 oracle rollouts
    over 8 scene seeds put the first miss at an achieved `dx` of 1.4464."""
    worst_achieved_dx_at_the_oracle_standoff = 1.3777
    assert ACCEPTED_THROW_STANDOFF_BOUNDS[1] > worst_achieved_dx_at_the_oracle_standoff
    assert _at_throw_pose(standoff=worst_achieved_dx_at_the_oracle_standoff)


def test_the_far_edge_stays_inside_the_measured_scoring_range() -> None:
    """The other side of the same measurement: the band must not promise a throw that
    misses. The first achieved `dx` measured to miss is 1.4464, so a far edge at or above
    it would accept a pose no toss parameterisation rescues.

    The gap is deliberate -- 1.400 is conservative by 46 mm against the measured edge, so
    this predicate calls some genuinely-scoring poses a failure. Safe direction for a
    precondition; widening it belongs in its own change with its own sweep."""
    first_achieved_dx_measured_to_miss = 1.4464
    assert ACCEPTED_THROW_STANDOFF_BOUNDS[1] < first_achieved_dx_measured_to_miss
    assert not _at_throw_pose(standoff=first_achieved_dx_measured_to_miss)


def test_the_five_of_five_core_still_lies_inside_the_accepted_band() -> None:
    """PR #105's measured 5/5 core -- `[1.150, 1.375]`, 5 scene seeds at 0.025 m
    resolution -- lies inside the band, as it did before the convergence."""
    low, high = _accepted_band()
    assert low <= 1.150
    assert high >= 1.375


def test_a_base_facing_away_from_the_bin_is_not_at_a_throw_pose() -> None:
    """**The conjunct this predicate gained from kinder-baselines.**

    `MoveToThrowPose` pins `rot = 0`, so a base it placed always faces the bin and this
    conjunct passes -- but `pos_base_rot` is a free state variable, and after `Pick` it
    was measured across 25 real states at 0.39-1.39 rad, with heading errors of
    0.29-0.92 rad against a tolerance of 0.08. So the conjunct is a genuine function of
    the state rather than a tautology, even though it was measured never to be the
    conjunct that decides an observed state (`0/75`)."""
    assert _at_throw_pose(standoff=ORACLE_THROW_STANDOFF, base_rot=0.0)
    assert not _at_throw_pose(standoff=ORACLE_THROW_STANDOFF, base_rot=np.pi / 2)
    assert not _at_throw_pose(standoff=ORACLE_THROW_STANDOFF, base_rot=np.pi)


def test_the_heading_conjunct_uses_the_shortest_signed_angle() -> None:
    """A base facing `+x` reported as `-pi` rather than `+pi` is the same heading, so
    the wrap has to go the short way round. A naive subtraction makes this 2*pi off."""
    assert not _at_throw_pose(standoff=ORACLE_THROW_STANDOFF, base_rot=-np.pi)
    assert _at_throw_pose(standoff=ORACLE_THROW_STANDOFF, base_rot=-1e-9)


def test_the_range_interval_brackets_the_calibrated_single_throw_range() -> None:
    """`THROW_RANGE` is the impact range at the oracle's own `(140 deg/s, 720 ms)`
    pair -- the same quantity the two endpoints are measured in -- so it has to fall
    inside the interval the parameter box spans. Nothing forced that."""
    assert THROW_RANGE_MIN < THROW_RANGE < THROW_RANGE_MAX


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


@pytest.mark.parametrize("standoff", [1.425, 1.45, 1.55, 1.65, 1.75])
def test_the_band_rejects_every_standoff_no_toss_parameterisation_reaches(
    *, standoff: float
) -> None:
    """`1.45` and beyond are the coarse 48-episode grid's 0/3 points, and this stack's
    standoff grid agrees over the parameter box: `0/150` at each of 1.45 through 1.75,
    so `0/1050` beyond 1.400. `1.425` is the geometric band's far edge, PR #105's 2/5.

    **This predicate is independent of `THROW_STANDOFF_BOUNDS`** -- it takes a raw
    standoff and says nothing about what the sampler can draw. Rejecting `1.75`, the
    sampler's upper bound, is what keeps the add effect two-class."""
    assert not _at_throw_pose(standoff=standoff)


@pytest.mark.parametrize("standoff", [1.10, 1.125, 1.40])
def test_the_band_accepts_short_standoffs_some_toss_parameterisation_reaches(
    *, standoff: float
) -> None:
    """Over the toss parameter box this stack's standoff grid solves `37/150` at 1.10
    and `3/150` at 1.40, so some parameterisation scores from all three.

    All three were measured not to solve at a single fixed 140 deg/s throw -- `1.10` the
    coarse grid's `0/3`, `1.125` and `1.40` PR #105's `2/5` and `3/5`. Those numbers
    stand exactly as published; they are evidence about *one* throw."""
    assert _at_throw_pose(standoff=standoff)


@pytest.mark.parametrize("standoff", [0.45, 0.80, 1.00])
def test_the_band_rejects_standoffs_below_the_samplers_floor(*, standoff: float) -> None:
    """**This flipped when the predicate converged onto upstream's band, and the flip is
    a narrowing rather than a correction.**

    The previous band was a union over the toss parameter box, so it accepted down to
    0.209 m on the model's claim that *some* parameterisation reaches the box from that
    close. Nothing was ever thrown from there -- the sampler's floor is 1.10 m
    (`THROW_STANDOFF_BOUNDS`) and every grid in this stack starts there -- so the old
    acceptance was unmeasured model output, and its replacement by upstream's measured
    1.09 m near edge loses no evidence.

    Both edges sit below the sampler's floor either way, so inside the range the sampler
    can actually draw this predicate is a one-sided threshold at the far edge, exactly as
    it was before."""
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


def test_the_lateral_conjunct_is_upstreams_doubled_waypoint_tolerance() -> None:
    """`move_to_target` stops once the base is within `WAYPOINT_TOLERANCE` of its own
    planned waypoint; kinder-baselines allows twice that, on the grounds that its own
    sampler already spends half of it off-axis. Converging on upstream's number widens
    this conjunct from 0.04 to 0.08."""
    assert THROW_POSE_TOLERANCE == 2 * WAYPOINT_TOLERANCE
    assert _at_throw_pose(standoff=ORACLE_THROW_STANDOFF, base_y=THROW_POSE_TOLERANCE * 0.9)
    assert not _at_throw_pose(standoff=ORACLE_THROW_STANDOFF, base_y=THROW_POSE_TOLERANCE * 1.1)


def test_widening_the_lateral_conjunct_does_not_readmit_the_recorded_failure() -> None:
    """**The check that had to pass before the tolerance could be widened at all.**

    The 0.04 m tolerance was measured against a real failure: the oracle once threw from
    a pose facing 40 degrees off the bin and the cube landed at (0.9969, -0.7196).
    Doubling a tolerance that was set by a recorded failure is exactly the kind of change
    that quietly re-admits it, so the pose is asserted directly rather than argued about.

    It is rejected twice over -- `|dy| = 0.366` against a 0.08 tolerance, and
    `dx = 1.721` against a band topping out at 1.400 -- so the widening has 4.6x of
    margin on the conjunct it touches."""
    assert not RobotAtSuccessfulThrowPoseClassifier.holds(
        state=state(base_x=0.279, base_y=0.366),
        robot=_ENV.robot,
        target=_ENV.bin,
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

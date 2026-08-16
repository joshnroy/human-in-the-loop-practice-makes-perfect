"""Offline tests for Tossing3D's five predicates.

`InBin` gets the most attention, including deliberate boundary probes, because it
*is* the success criterion and because a wrong goal box has already shipped once in this
project's history. `test_kinder_fidelity.py` checks it against KINDER's own
`_check_goals()` whenever the simulator is installed; these run everywhere.

**It was six.** `RobotAtSuccessfulThrowPose` named the pose between `MoveToThrowPose` and
`Toss`, and upstream composed those two controllers into one -- so there is no longer a
state between them for a predicate to describe. Its whole calibration block went with it
(`THROW_RANGE`, the overshoot/shortfall margins, `THROW_STANDOFF_BOUNDS`,
`THROW_POSE_LATERAL_TOLERANCE`), and so did the roughly two dozen band tests that lived
in the second half of this file. Those numbers are not wrong; they describe a
decomposition this domain no longer runs.
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
    ON_GROUND_TOL,
    REACHABLE,
    CubeSymmetry,
    HandEmptyClassifier,
    HoldingClassifier,
    InBinClassifier,
    OnGroundClassifier,
    ReachableClassifier,
)

from .observations import BARRIER_X, BIN_X, CUBE_START_X, GOAL_REGION_BBOX, state

_ENV = Tossing3DEnvironment()


def _axis_angle(*, axis: tuple[float, float, float], angle: float) -> tuple[float, ...]:
    """One rotation as MuJoCo's `(qx, qy, qz, qw)`, so the cases below read as poses."""
    unit = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    return (*(unit * np.sin(angle / 2)), float(np.cos(angle / 2)))


def _compose(*, first: tuple[float, ...], then: tuple[float, ...]) -> tuple[float, ...]:
    """`then * first` in `(x, y, z, w)` order -- `first` applied, then `then`."""
    x1, y1, z1, w1 = first
    x2, y2, z2, w2 = then
    return (
        w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1,
        w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1,
        w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1,
        w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1,
    )


# The six face-down rests, as the rotation that puts each face on the floor. Upright and
# upside-down are the ones a hand-written `qx ~ 0 and qy ~ 0` test also accepts; the four
# side faces are exactly the ones it rejects, and exactly the ones a thrown cube lands on.
_FACE_DOWN_ROTATIONS = {
    "upright": _axis_angle(axis=(0.0, 0.0, 1.0), angle=0.0),
    "upside-down": _axis_angle(axis=(1.0, 0.0, 0.0), angle=np.pi),
    "+x face": _axis_angle(axis=(0.0, 1.0, 0.0), angle=np.pi / 2),
    "-x face": _axis_angle(axis=(0.0, 1.0, 0.0), angle=-np.pi / 2),
    "+y face": _axis_angle(axis=(1.0, 0.0, 0.0), angle=np.pi / 2),
    "-y face": _axis_angle(axis=(1.0, 0.0, 0.0), angle=-np.pi / 2),
}


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


@pytest.mark.parametrize("face", sorted(_FACE_DOWN_ROTATIONS))
def test_on_ground_accepts_a_cube_resting_on_any_one_of_its_six_faces(*, face: str) -> None:
    """**The change the composed toss forced, stated as `6/6` faces rather than `1/6`.**

    `MoveToTossLocationAndToss` *adds* `OnGround`, and a thrown cube lands on whichever
    face it likes. The test this classifier used to run -- `qx` and `qy` both near zero --
    is "resting on the face it started on", and measured against these six rotations it
    accepts `1/6`: only `upright`. Even `upside-down` fails it, since that rotation is
    `qx = 1`. Under it the add effect would read false on most scoring throws and EES
    would label a throw that scored a failure.

    Written face by face rather than as one loop so a regression names which faces broke.
    """
    qx, qy, qz, qw = _FACE_DOWN_ROTATIONS[face]
    assert OnGroundClassifier.holds(
        state=state(cube_z=0.025, cube_qx=qx, cube_qy=qy, cube_qz=qz, cube_qw=qw),
        cube=_ENV.cube,
    )


@pytest.mark.parametrize("face", sorted(_FACE_DOWN_ROTATIONS))
@pytest.mark.parametrize("yaw_degrees", [17.0, 45.0, 123.0, -88.0])
def test_on_ground_ignores_yaw_entirely_once_a_face_is_down(
    *, face: str, yaw_degrees: float
) -> None:
    """A cube on a face is four-fold symmetric about the vertical, so its yaw carries no
    information about whether it is resting. Deliberately awkward yaws rather than
    multiples of 90 degrees, which the symmetry group would map back onto the face
    rotations above and so could not distinguish a real derivation from a lookup."""
    rotation = _compose(
        first=_FACE_DOWN_ROTATIONS[face],
        then=_axis_angle(axis=(0.0, 0.0, 1.0), angle=np.deg2rad(yaw_degrees)),
    )
    qx, qy, qz, qw = rotation
    assert OnGroundClassifier.holds(
        state=state(cube_z=0.025, cube_qx=qx, cube_qy=qy, cube_qz=qz, cube_qw=qw),
        cube=_ENV.cube,
    )


def test_on_ground_rejects_a_cube_balanced_on_a_corner() -> None:
    """A body diagonal pointing straight up is the furthest a cube gets from any face.

    The rotation is derived rather than typed: it is the one taking the body diagonal
    `(1, 1, 1)/sqrt(3)` onto `+z`, so this is a genuine corner balance and not a
    quaternion chosen for being far from the identity."""
    diagonal = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    rotation = _axis_angle(
        axis=tuple(np.cross(diagonal, [0.0, 0.0, 1.0])), angle=float(np.arccos(diagonal[2]))
    )
    qx, qy, qz, qw = rotation
    # Not merely outside the tolerance -- as far outside as the geometry allows.
    assert CubeSymmetry.tilt_from_upright(rotation=rotation) > 4 * ON_GROUND_TOL
    assert not OnGroundClassifier.holds(
        state=state(cube_z=0.025, cube_qx=qx, cube_qy=qy, cube_qz=qz, cube_qw=qw),
        cube=_ENV.cube,
    )


def test_on_ground_rejects_a_cube_balanced_on_an_edge() -> None:
    """The intermediate case, and the one a face-interchangeable test could plausibly
    have swallowed: 45 degrees about `x` puts an edge down, which is a rest position no
    grasp derived from a face can use."""
    rotation = _axis_angle(axis=(1.0, 0.0, 0.0), angle=np.pi / 4)
    qx, qy, qz, qw = rotation
    assert not OnGroundClassifier.holds(
        state=state(cube_z=0.025, cube_qx=qx, cube_qy=qy, cube_qz=qz, cube_qw=qw),
        cube=_ENV.cube,
    )


def test_on_ground_rejects_a_cube_that_is_face_down_but_still_in_the_air() -> None:
    """The two conjuncts are independent, so the height one has to be checked at a
    rotation the flatness one accepts -- otherwise a classifier that had dropped the
    height test entirely would still pass every case above."""
    qx, qy, qz, qw = _FACE_DOWN_ROTATIONS["+x face"]
    assert not OnGroundClassifier.holds(
        state=state(cube_z=0.4, cube_qx=qx, cube_qy=qy, cube_qz=qz, cube_qw=qw),
        cube=_ENV.cube,
    )


def test_a_non_cube_keeps_the_strict_face_it_started_on_test() -> None:
    """Only a cube's faces are interchangeable. The branch is taken on the measured
    bounding box rather than on the object's name, so an object whose extents differ must
    still be rejected on its side -- which is exactly where the cube branch accepts."""
    qx, qy, qz, qw = _FACE_DOWN_ROTATIONS["+x face"]
    oblong = state(cube_z=0.025, cube_qx=qx, cube_qy=qy, cube_qz=qz, cube_qw=qw)
    oblong.set(obj=_ENV.cube, feature_name="bb_x", feature_val=0.12)
    assert not OnGroundClassifier.holds(state=oblong, cube=_ENV.cube)


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


# **The `RobotAtSuccessfulThrowPose` band tests lived here, and are deleted rather than
# ported.** Roughly two dozen of them pinned the interval of standoffs the predicate
# accepted: that it was strictly narrower than `THROW_STANDOFF_BOUNDS` (so EES's success
# classifier saw two classes), that it tracked the bin and the goal region rather than a
# literal, that its far edge sat on 1.400, that PR #105's `5/5` core `[1.150, 1.375]` fell
# inside it, and that it rejected the pose `Pick` leaves the base in. None of them is
# wrong; all of them describe a state between two controllers that upstream has since
# composed into one, so there is no pose left for any predicate to accept or reject. The
# numbers stay readable in git history and in the experiment logs that cite them.


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

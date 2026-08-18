"""The six Tossing3D predicates are kinder-baselines', not ours.

This file is the contract of the swap: each hitl `Predicate` must agree with the
kinder-baselines classifier it is now backed by, *by construction* rather than by
a threshold copied across and kept in step by hand.

Everything here is offline. `ObjectCentricState` is constructible with no MuJoCo,
and four of kinder-baselines' six classifiers are `@staticmethod`s, so they are
callable with no abstractor instance and hence no simulator process. The other two
(`Holding`'s forward-kinematics conjunct, `MovableInGoalRegion`'s ground fixture)
genuinely need a simulator and are covered in `test_kinder_fidelity.py`.
"""

import importlib.util

import pytest

from hitl_pmp.environments.tossing3d.skill_oracle_policy import ORACLE_THROW_STANDOFF

from .object_centric import BARRIER_X, BIN_X, object_centric_state

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="requires the tossing3d extra"
)


def _kb():
    """Upstream's abstractor and its two constants, imported on use.

    Lazy for the same reason `object_centric.py`'s imports are: pytest imports every test
    module at collection time, and a module-scope `import kinder_models` here would put
    `mujoco` in `sys.modules` before the several tests asserting the *absence* of a
    simulator import ever run.
    """
    from kinder_models.dynamic3d.tossing.state_abstractions import (
        THROW_POSE_TOLERANCE,
        THROW_STANDOFF_BOUNDS,
        Tossing3DStateAbstractor,
    )

    return Tossing3DStateAbstractor, THROW_STANDOFF_BOUNDS, THROW_POSE_TOLERANCE


def _at_throw_pose(*, standoff: float, base_y: float = 0.0, base_rot: float = 0.0) -> bool:
    abstractor, _, _ = _kb()
    state, objects = object_centric_state(base_x=BIN_X - standoff, base_y=base_y, base_rot=base_rot)
    return abstractor._check_at_throw_pose(  # noqa: SLF001
        state, objects["robot"], objects["bin_0"]
    )


def test_the_accepted_standoff_band_is_kinder_baselines_own() -> None:
    """The band is kinder-baselines' `THROW_STANDOFF_BOUNDS`, not a number of ours.

    Swept at millimetre resolution so the edges are pinned from both sides rather
    than asserted at two points. This is the conjunct the audit found materially
    disagreed with our own predicate (478/1805 of a grid), so it is the one that
    actually moves behaviour.
    """
    accepted = [d / 1000.0 for d in range(1000, 1500) if _at_throw_pose(standoff=d / 1000.0)]
    assert accepted, "no standoff accepted at all"
    _, kb_bounds, _ = _kb()
    low, high = kb_bounds
    assert (min(accepted), max(accepted)) == (low, high)
    assert len(accepted) == 286, f"{len(accepted)}/500 millimetre standoffs accepted"


def test_the_band_rejects_past_its_upper_edge() -> None:
    """1.400 was our own predicate's one-sided threshold; kinder-baselines' stops at
    1.375, and that difference is the whole behavioural content of this swap."""
    assert not _at_throw_pose(standoff=1.400)
    assert _at_throw_pose(standoff=1.375)


def test_the_lateral_tolerance_is_kinder_baselines_own_not_ours() -> None:
    """Ours was `WAYPOINT_TOLERANCE` (0.04); kinder-baselines' is twice that."""
    _, _, tolerance = _kb()
    assert tolerance == pytest.approx(0.08)
    assert _at_throw_pose(standoff=1.2, base_y=0.07)
    assert not _at_throw_pose(standoff=1.2, base_y=0.09)


def test_the_heading_conjunct_exists_and_ours_had_none() -> None:
    """A conjunct our own predicate did not have at all: the base must *face* the bin,
    not merely sit on its axis."""
    assert _at_throw_pose(standoff=1.2, base_rot=0.0)
    assert not _at_throw_pose(standoff=1.2, base_rot=0.5)


def _abstractor_static(*, name: str):
    abstractor, _, _ = _kb()
    return getattr(abstractor, name)


def test_hand_empty_holds_only_at_an_open_gripper() -> None:
    """Upstream reads the gripper *command*, not finger pose, so this and `Holding` are
    deliberately not complementary."""
    check = _abstractor_static(name="_check_gripper_open")
    for gripper, expected in ((0.0, True), (0.0005, True), (0.05, False), (0.9, False)):
        state, objects = object_centric_state(gripper=gripper)
        assert check(state, objects["robot"]) is expected, gripper


def test_on_ground_holds_for_a_cube_resting_flat_on_the_floor() -> None:
    check = _abstractor_static(name="_check_on_ground")
    state, objects = object_centric_state(cube_z=0.025)
    assert check(state, objects["cube_0"])


def test_on_ground_rejects_a_cube_in_the_air() -> None:
    check = _abstractor_static(name="_check_on_ground")
    state, objects = object_centric_state(cube_z=0.4)
    assert not check(state, objects["cube_0"])


def test_on_ground_rejects_a_cube_resting_on_a_corner() -> None:
    """Flatness is upstream's own conjunct and is load-bearing rather than decorative:
    `pick_shelf` builds its grasp pose from the object's orientation, so a cube that came
    to rest on a corner is not a cube this grasp is modelled on."""
    check = _abstractor_static(name="_check_on_ground")
    state, objects = object_centric_state(cube_z=0.025, cube_qx=0.5)
    assert not check(state, objects["cube_0"])
    state, objects = object_centric_state(cube_z=0.025, cube_qy=0.5)
    assert not check(state, objects["cube_0"])


def test_movable_is_down_x_is_a_one_way_door_across_the_barrier() -> None:
    """The irreversibility this whole domain exists to exhibit: the barrier is a 5 m
    immovable wall, so a cube past it can never be picked up again."""
    check = _abstractor_static(name="_check_is_down_x")
    near, objects = object_centric_state(cube_x=BARRIER_X - 0.1)
    assert check(near, objects["cube_0"], objects["cuboid_barrier"])
    far, objects = object_centric_state(cube_x=BARRIER_X + 0.1)
    assert not check(far, objects["cube_0"], objects["cuboid_barrier"])


def test_movable_is_down_x_reads_the_barriers_live_x_rather_than_a_constant() -> None:
    """The barrier's pose is sampled from `barrier_init_region` per episode, so a literal
    would be right for one seed and quietly wrong for the next."""
    check = _abstractor_static(name="_check_is_down_x")
    state, objects = object_centric_state(cube_x=1.5, barrier_x=1.8)
    assert check(state, objects["cube_0"], objects["cuboid_barrier"])
    state, objects = object_centric_state(cube_x=1.5, barrier_x=1.2)
    assert not check(state, objects["cube_0"], objects["cuboid_barrier"])


def test_the_oracle_standoff_sits_inside_the_band_but_near_its_upper_edge() -> None:
    """Why the oracle needed a convergence check as well as this predicate.

    1.35 is accepted, but only 25 mm below the band's upper edge -- and `move_to_target`
    overshoots its commanded standoff by up to 27.7 mm, so the *achieved* pose can fall
    outside a band the *commanded* one is comfortably inside.
    """
    assert _at_throw_pose(standoff=ORACLE_THROW_STANDOFF)
    _, kb_bounds, _ = _kb()
    assert kb_bounds[1] - ORACLE_THROW_STANDOFF == pytest.approx(0.025)
    assert not _at_throw_pose(standoff=ORACLE_THROW_STANDOFF + 0.0277)


def test_the_standoff_conjunct_rejects_most_but_not_all_post_pick_poses() -> None:
    """The hazard the standoff conjunct exists for, and the honest limit of it.

    Over 30 scene seeds the post-`Pick` base sits 1.364-1.971 m from the bin. If the
    predicate accepted those, the oracle -- and any planner reading it -- would believe
    it was already at a throw pose and skip `MoveToThrowPose` entirely.

    Upstream's band excludes the bulk of that interval, **but not its near end**: 1.364
    falls inside `(1.09, 1.375)`. So the standoff conjunct alone is not what makes a
    post-`Pick` pose unacceptable, and this test says so rather than asserting a
    convenient falsehood. What actually excludes those poses is the lateral and heading
    conjuncts -- `Pick` drives the base to the *cube*, which is off to one side -- and
    that is checked directly below.
    """
    for standoff in (1.6, 1.859, 1.971):
        assert not _at_throw_pose(standoff=standoff), standoff
    assert _at_throw_pose(standoff=1.364), (
        "upstream's band accepts the closest post-Pick standoff; if this ever flips, the "
        "note in this test's docstring is stale"
    )


def test_a_post_pick_pose_is_excluded_by_facing_rather_than_by_distance() -> None:
    """`Pick` drives the base to the cube, which sits off the bin's axis, so the pose it
    leaves behind fails the lateral and heading conjuncts even at an accepted standoff.

    This is the defect class `tests/environments/test_operator_dynamics_fidelity.py`
    exists for: an over-permissive operator model once let the oracle throw from a pose
    facing 40 degrees away from the bin, landing the cube at `(0.9969, -0.7196)`.
    """
    assert not _at_throw_pose(standoff=1.364, base_y=0.3)
    assert not _at_throw_pose(standoff=1.364, base_rot=0.7)


def test_the_band_rejects_standing_on_top_of_the_bin() -> None:
    assert not _at_throw_pose(standoff=0.0)

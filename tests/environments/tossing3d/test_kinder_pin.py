"""What the `reference/kinder-baselines` pin has to expose for this domain to work.

A gitlink is a silent dependency: bumping it changes what every Tossing3D run executes,
and nothing in this repo's diff shows the code that moved. These assertions fail at any
pin without the toss release parameters, which is what stops a later
`git submodule update` from quietly walking the pin backwards.

Gated on `find_spec` for the *import* package names (`kinder`, `kinder_models`), not the
distribution name `kindergarden`; CI never installs the optional extra. Checks only
upstream's API surface -- no simulator, no controller execution.
"""

import importlib.util
import inspect

import numpy as np
import pytest

needs_kinder = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None or importlib.util.find_spec("kinder_models") is None,
    reason="KINDER is an optional extra (`kindergarden` + `kinder_models`); CI never installs it",
)

pytestmark = needs_kinder

# The swing's hand-tuned "throw hard" limits, in deg/s and deg/s^2 -- upstream's own
# literals, demonstrated on the real TidyBot (`yixuanhuang98/tidybot_real`,
# `robot/kinova.py:120-124`).
UPSTREAM_TOSS_LIMITS_DEG = (140.0, 300.0, 200.0)


def test_toss_profile_limits_exists_and_defaults_to_upstreams_literals() -> None:
    """The default triple must be what every committed Tossing3D number was measured at,
    so that passing no release speed leaves those numbers valid.
    """
    from kinder_models.dynamic3d.tossing.toss_swing import (
        TOSS_MAX_VELOCITY,
        toss_profile_limits,
    )

    assert np.rad2deg(TOSS_MAX_VELOCITY) == pytest.approx(140.0)
    assert np.rad2deg(toss_profile_limits()) == pytest.approx(UPSTREAM_TOSS_LIMITS_DEG)


def test_toss_profile_limits_scales_all_three_by_one_effort() -> None:
    """All three limits move together, so the profile is the default replayed on a
    stretched clock -- an *effort*, not a speed cap. Scaling `max_vel` alone would push
    the release point into the acceleration phase.
    """
    from kinder_models.dynamic3d.tossing.toss_swing import (
        TOSS_MAX_VELOCITY,
        toss_profile_limits,
    )

    default_vel, default_accel, default_decel = toss_profile_limits()
    for factor in (0.4286, 0.5953, 1.0):
        vel, accel, decel = toss_profile_limits(TOSS_MAX_VELOCITY * factor)
        assert vel == pytest.approx(default_vel * factor)
        assert accel / vel == pytest.approx(default_accel / default_vel)
        assert decel / vel == pytest.approx(default_decel / default_vel)


def test_toss_profile_limits_clamps_effort_at_the_default() -> None:
    """`TOSS_MAX_VELOCITY` is a genuine ceiling: a release speed above it is clamped
    rather than scaling the profile past the default.

    `TOSS_SPEED_BOUNDS` caps at `TOSS_MAX_VELOCITY`, so this probes above our own range
    deliberately -- from inside our bounds the clamp is unreachable and so invisible.
    """
    from kinder_models.dynamic3d.tossing.toss_swing import (
        TOSS_MAX_VELOCITY,
        toss_profile_limits,
    )

    assert toss_profile_limits(np.deg2rad(240.0)) == pytest.approx(toss_profile_limits())
    assert toss_profile_limits(np.deg2rad(240.0))[0] == pytest.approx(TOSS_MAX_VELOCITY)


def test_the_release_speed_our_sampler_can_draw_is_never_clamped() -> None:
    """The clamp must sit at or above our own upper bound, so no draw the EES sampler
    makes is silently rewritten. `TOSS_SPEED_BOUNDS`' top edge is the boundary case: it
    is exactly `TOSS_MAX_VELOCITY`, so it must pass through unscaled rather than
    tripping the clamp.
    """
    from kinder_models.dynamic3d.tossing.toss_swing import toss_profile_limits

    from hitl_pmp.environments.tossing3d.skills import TOSS_SPEED_BOUNDS

    for deg in np.linspace(TOSS_SPEED_BOUNDS[0], TOSS_SPEED_BOUNDS[1], 37):
        assert np.rad2deg(toss_profile_limits(np.deg2rad(deg))[0]) == pytest.approx(deg)


# --- The two controllers this domain drives, and the four bounds it samples from -------
#
# `skills.py` restates all four continuous bounds as module constants rather than reading
# them off the controller class, so that the whole symbolic layer -- sampler included --
# runs on a machine with no KINDER. That restatement is exactly the kind of duplication
# that goes stale silently: a one-sided upstream retune would leave hitl and the kb-side
# bilevel planner drawing from different boxes while every offline test still passed.
# These are the tests that make such a drift loud.


def test_the_two_controllers_this_domain_drives_are_both_exposed() -> None:
    """`KinderBackend.run_pick_cube` and `run_move_to_toss_location_and_toss` look these
    up by string key out of `create_lifted_controllers`, and a missing key surfaces as a
    `ValueError` deep inside a rollout rather than at import. Checked by name only, so
    this costs no PyBullet client."""
    import inspect

    from kinder_models.dynamic3d.tossing.parameterized_skills import create_lifted_controllers

    source = inspect.getsource(create_lifted_controllers)
    assert '"pick_cube"' in source
    assert '"move_to_toss_location_and_toss"' in source


def test_the_pick_ignores_the_parameters_its_own_sampler_draws() -> None:
    """`param_dim=0` on this side is only honest if upstream's controller agrees, and at
    the bumped pin the honest reason changed.

    It used to be that `sample_parameters` returned `tuple()` -- nothing to draw, so
    nothing to learn. At this pin it returns a real `[distance, rotation]` pair and
    rejection-tests the resulting base pose against other cubes. What makes `param_dim=0`
    still correct is one line further in: `PickCubeController.reset` opens with
    `del params` above the comment *"This is an entirely hardcoded controller"*, so the
    drawn pair is discarded and cannot affect the motion.

    That distinction is worth a test rather than a comment, because the two failure modes
    look identical from outside: a pick that ignores its parameters and a pick that has
    none both execute the same way, but only the first would silently start mattering if
    upstream ever consumed them. Asserted on the source rather than by executing a
    controller, so no PyBullet client is opened."""
    import inspect

    from kinder_models.dynamic3d.tossing.parameterized_skills import PickCubeController

    reset_source = inspect.getsource(PickCubeController.reset)
    assert "del params" in reset_source, (
        "PickCubeController.reset no longer discards its parameters, so this domain's "
        "param_dim=0 for pick_cube has stopped being true"
    )


def test_the_distance_bounds_are_upstreams_own() -> None:
    """The standoff the composed toss drives to, in metres from the bin."""
    from kinder_models.dynamic3d.tossing.parameterized_skills import (
        MoveToTossLocationAndTossController,
    )

    from hitl_pmp.environments.tossing3d.skills import TOSS_DISTANCE_BOUNDS

    assert (
        pytest.approx(MoveToTossLocationAndTossController.TARGET_DISTANCE_BOUNDS)
        == TOSS_DISTANCE_BOUNDS
    )


def test_the_rotation_bounds_are_upstreams_own() -> None:
    """Both sides derive this from `WAYPOINT_TOLERANCE` and the largest standoff rather
    than writing the number out, so this catches a drift in either input as well as in
    the arithmetic."""
    from kinder_models.dynamic3d.tossing.parameterized_skills import (
        MoveToTossLocationAndTossController,
    )

    from hitl_pmp.environments.tossing3d.skills import MAX_TOSS_ROTATION, TOSS_ROTATION_BOUNDS

    assert (
        pytest.approx(MoveToTossLocationAndTossController.MAX_TARGET_ROTATION) == MAX_TOSS_ROTATION
    )
    assert (
        pytest.approx(MoveToTossLocationAndTossController.TARGET_ROTATION_BOUNDS)
        == TOSS_ROTATION_BOUNDS
    )


def test_the_speed_bounds_are_upstreams_own_read_through_the_degree_conversion() -> None:
    """**The one bound that is not stated in upstream's units.** This domain carries the
    dial in joint-path deg/s because every measured toss number in its docs is in deg/s;
    upstream's `SPEED_BOUNDS` are rad/s. So the equality has to be asserted *through* the
    conversion, and a missing conversion on either side is a 57x error that would sail
    past a same-units comparison."""
    from kinder_models.dynamic3d.tossing.parameterized_skills import (
        MoveToTossLocationAndTossController,
    )

    from hitl_pmp.environments.tossing3d.skills import TOSS_SPEED_BOUNDS

    assert np.deg2rad(TOSS_SPEED_BOUNDS) == pytest.approx(
        MoveToTossLocationAndTossController.SPEED_BOUNDS
    )


def test_the_release_millisecond_bounds_are_upstreams_own() -> None:
    """Absolute milliseconds rather than a swing fraction, because that is what the real
    TidyBot's `movej_primitive.execute()` takes -- so this is one of the few dials whose
    units are the robot's rather than the simulator's."""
    from kinder_models.dynamic3d.tossing.parameterized_skills import (
        MoveToTossLocationAndTossController,
    )

    from hitl_pmp.environments.tossing3d.skills import TOSS_RELEASE_MS_BOUNDS

    assert (
        pytest.approx(MoveToTossLocationAndTossController.RELEASE_MS_BOUNDS)
        == TOSS_RELEASE_MS_BOUNDS
    )


def test_the_waypoint_tolerance_is_upstreams_own() -> None:
    """`skills.py` computes `MAX_TOSS_ROTATION` from its own copy of this constant, so a
    stale copy would make the rotation-bounds test above fail for a reason that reads as
    an upstream retune. Pinned separately so the two failures are distinguishable."""
    from kinder_models.dynamic3d.utils import WAYPOINT_TOLERANCE as UPSTREAM_WAYPOINT_TOLERANCE

    from hitl_pmp.environments.tossing3d.skills import WAYPOINT_TOLERANCE

    assert pytest.approx(UPSTREAM_WAYPOINT_TOLERANCE) == WAYPOINT_TOLERANCE


def test_every_release_ms_the_sampler_can_draw_still_opens_the_gripper() -> None:
    """The bounds' upper edge is set by the *shortest* swing, not the longest.
    `gripper_release_ms` is deliberately unclamped in `plan_toss_swing`: past the end of
    the swing the gripper never opens and the cube is never thrown.

    Recomputed by planning real swings across the whole speed range rather than compared
    against a copied number, so it fails if a pin bump changes the swing's timing.

    **Nominal, and labelled as such.** The composed controller plans its swing from the
    arm configuration the motion planner actually reached, which is a few milliseconds
    from the nominal windup-to-release difference used here -- upstream's own note says to
    re-derive a release millisecond by running the swing, never by recomputing from the
    two configurations. This is therefore a guard against a *large* change in the swing's
    duration, not a millisecond-accurate claim.
    """
    from kinder_models.dynamic3d.tossing.toss_swing import (
        TOSS_RELEASE_ARM_CONFIGURATION,
        TOSS_SLICES_PER_CONTROL_STEP,
        TOSS_WINDUP_ARM_CONFIGURATION,
        plan_toss_swing,
    )

    from hitl_pmp.environments.tossing3d.skills import TOSS_RELEASE_MS_BOUNDS, TOSS_SPEED_BOUNDS

    shortest_ms = min(
        (len(swing.trajectory) - 1) * TOSS_SLICES_PER_CONTROL_STEP
        for swing in (
            plan_toss_swing(
                [TOSS_RELEASE_ARM_CONFIGURATION],
                TOSS_WINDUP_ARM_CONFIGURATION,
                release_speed=float(np.deg2rad(deg)),
            )
            for deg in np.linspace(TOSS_SPEED_BOUNDS[0], TOSS_SPEED_BOUNDS[1], 37)
        )
    )
    assert TOSS_RELEASE_MS_BOUNDS[1] <= shortest_ms


def test_the_toss_schedule_is_exactly_as_wide_as_kinder_demands() -> None:
    """The coupling between the two pins, which is otherwise silent until a throw runs.

    `MujocoEnv.step` requires a control schedule to cover the period *exactly*; the
    toss's mid-step gripper release is the only schedule this domain emits, and it is
    `TOSS_SLICES_PER_CONTROL_STEP` rows wide. Both sides are re-derived from their own
    constants rather than the shared literal, so bumping one pin without the other fails
    here instead of deep inside a rollout.
    """
    from kinder.envs.dynamic3d.mujoco_utils import (
        CONTROL_SCHEDULE_TIMESTEP,
        SIMULATION_TIMESTEP,
    )
    from kinder_models.dynamic3d.tossing.toss_swing import (
        TOSS_SLICES_PER_CONTROL_STEP,
    )

    # Private upstream, and deliberately imported anyway: it is the constant the
    # schedule width is derived from, and re-deriving it here would be the duplication
    # this test exists to catch.
    from kinder_models.dynamic3d.utils import _CONTROL_TIMESTEP  # noqa: PLC2701

    num_sim_steps = int(_CONTROL_TIMESTEP / SIMULATION_TIMESTEP)
    ticks_per_row = int(round(CONTROL_SCHEDULE_TIMESTEP / SIMULATION_TIMESTEP))
    assert num_sim_steps // ticks_per_row == TOSS_SLICES_PER_CONTROL_STEP


# These two used to read `TossController.reset`'s signature, because that is where this
# package's `run_toss` handed the two dials in. The composed controller does not use
# `TossController` at all -- it calls `plan_toss_swing` directly -- so `TossController`
# is now upstream API this domain never touches, and pinning it would imply otherwise.
# Retargeted onto `plan_toss_swing`, which is the live path.


def test_the_swing_planner_accepts_a_release_speed() -> None:
    """The parameter has to reach the thing that plans the swing, not just the helper
    that scales the profile. Signature-only, so nothing is planned and no PyBullet client
    is opened."""
    from kinder_models.dynamic3d.tossing.toss_swing import TOSS_MAX_VELOCITY, plan_toss_swing

    parameters = inspect.signature(plan_toss_swing).parameters
    assert "release_speed" in parameters
    assert parameters["release_speed"].default == TOSS_MAX_VELOCITY


def test_the_swing_planner_accepts_a_gripper_release_millisecond() -> None:
    """The second dial. Signature-only, for the same reason as the sibling test above.

    720 is upstream's own default and is **not** the oracle's 792 -- see
    `test_skill_oracle_policy.py` for where that came from."""
    from kinder_models.dynamic3d.tossing.toss_swing import (
        TOSS_DEFAULT_GRIPPER_RELEASE_MILLISECONDS,
        plan_toss_swing,
    )

    parameters = inspect.signature(plan_toss_swing).parameters
    assert "gripper_release_ms" in parameters
    assert parameters["gripper_release_ms"].default == TOSS_DEFAULT_GRIPPER_RELEASE_MILLISECONDS
    assert TOSS_DEFAULT_GRIPPER_RELEASE_MILLISECONDS == 720


def test_the_release_fraction_trigger_is_gone_rather_than_kept_alongside() -> None:
    """There must be exactly one way to say when the gripper opens. With
    `_release_fraction` still present, a `gripper_release_ms` we pass could be silently
    overridden by a distance-fraction test that fired first.
    """
    from kinder_models.dynamic3d.tossing import parameterized_skills

    source = inspect.getsource(parameterized_skills)
    assert "_release_fraction" not in source
    assert "gripper_release_ms" in source

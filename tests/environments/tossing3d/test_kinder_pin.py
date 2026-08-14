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
    from kinder_models.dynamic3d.tossing.parameterized_skills import (
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
    from kinder_models.dynamic3d.tossing.parameterized_skills import (
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
    from kinder_models.dynamic3d.tossing.parameterized_skills import (
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
    from kinder_models.dynamic3d.tossing.parameterized_skills import toss_profile_limits

    from hitl_pmp.environments.tossing3d.predicates import TOSS_SPEED_BOUNDS

    for deg in np.linspace(TOSS_SPEED_BOUNDS[0], TOSS_SPEED_BOUNDS[1], 37):
        assert np.rad2deg(toss_profile_limits(np.deg2rad(deg))[0]) == pytest.approx(deg)


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
    from kinder_models.dynamic3d.tossing.parameterized_skills import (
        TOSS_SLICES_PER_CONTROL_STEP,
    )
    from kinder_models.dynamic3d.utils import CONTROL_TIMESTEP

    num_sim_steps = int(CONTROL_TIMESTEP / SIMULATION_TIMESTEP)
    ticks_per_row = int(round(CONTROL_SCHEDULE_TIMESTEP / SIMULATION_TIMESTEP))
    assert num_sim_steps // ticks_per_row == TOSS_SLICES_PER_CONTROL_STEP


def test_toss_controller_reset_accepts_a_release_speed() -> None:
    """The parameter has to reach the controller, not just the helper. Signature-only:
    constructing a `TossController` opens a PyBullet client.
    """
    from kinder_models.dynamic3d.tossing.parameterized_skills import (
        TOSS_MAX_VELOCITY,
        TossController,
    )

    parameters = inspect.signature(TossController.reset).parameters
    assert "release_speed" in parameters
    assert parameters["release_speed"].default == TOSS_MAX_VELOCITY


def test_toss_controller_reset_accepts_a_gripper_release_millisecond() -> None:
    """The second dial. Signature-only, for the same reason as the sibling test above."""
    from kinder_models.dynamic3d.tossing.parameterized_skills import (
        TOSS_DEFAULT_GRIPPER_RELEASE_MILLISECONDS,
        TossController,
    )

    parameters = inspect.signature(TossController.reset).parameters
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


def test_the_restated_throw_pose_constants_still_equal_upstreams() -> None:
    """`predicates.py` restates upstream's throw-pose tolerances rather than importing
    them, because importing anything from `kinder_models` there would make the whole
    `--env tossing3d` registration require MuJoCo and take ~900 lines of offline predicate
    and skill tests off CI with it.

    Restating buys that at the cost of a silent-drift risk, and this is what pays for it:
    wherever KINDER *is* installed, a tolerance upstream retunes fails here instead of
    quietly leaving the two definitions disagreeing.

    The band's **far edge is deliberately not** asserted equal -- this repo ships 1.400
    where upstream ships 1.375, so that its own oracle standoff clears
    `move_to_target`'s stopping error. See `RobotAtSuccessfulThrowPoseClassifier`'s
    docstring. The near edge is upstream's unchanged, and is asserted.
    """
    from kinder_models.dynamic3d.tossing.state_abstractions import (
        THROW_POSE_TOLERANCE as UPSTREAM_THROW_POSE_TOLERANCE,
    )
    from kinder_models.dynamic3d.tossing.state_abstractions import (
        THROW_STANDOFF_BOUNDS as UPSTREAM_ACCEPTED_BAND,
    )
    from kinder_models.dynamic3d.utils import WAYPOINT_TOLERANCE as UPSTREAM_WAYPOINT_TOLERANCE

    from hitl_pmp.environments.tossing3d.predicates import (
        ACCEPTED_THROW_STANDOFF_BOUNDS,
        THROW_POSE_TOLERANCE,
        WAYPOINT_TOLERANCE,
    )

    assert WAYPOINT_TOLERANCE == UPSTREAM_WAYPOINT_TOLERANCE
    assert THROW_POSE_TOLERANCE == UPSTREAM_THROW_POSE_TOLERANCE
    assert ACCEPTED_THROW_STANDOFF_BOUNDS[0] == UPSTREAM_ACCEPTED_BAND[0]
    assert ACCEPTED_THROW_STANDOFF_BOUNDS[1] > UPSTREAM_ACCEPTED_BAND[1]


def test_the_restated_signed_angle_distance_matches_upstreams() -> None:
    """The heading conjunct's wrap, checked against `prpl_utils`' own implementation over
    the full circle rather than at a handful of points -- a naive difference agrees
    everywhere except across the branch cut, which is exactly where it matters."""
    from prpl_utils.utils import get_signed_angle_distance

    from hitl_pmp.environments.tossing3d.predicates import (
        RobotAtSuccessfulThrowPoseClassifier as Classifier,
    )

    angles = np.linspace(-np.pi, np.pi, 73)
    for target in angles:
        for source in angles:
            assert Classifier._signed_angle_distance(  # noqa: SLF001
                target=float(target), source=float(source)
            ) == pytest.approx(get_signed_angle_distance(float(target), float(source)), abs=1e-12)

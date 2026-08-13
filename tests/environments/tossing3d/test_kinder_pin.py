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
        TOSS_MAX_VEL,
        toss_profile_limits,
    )

    assert np.rad2deg(TOSS_MAX_VEL) == pytest.approx(140.0)
    assert np.rad2deg(toss_profile_limits()) == pytest.approx(UPSTREAM_TOSS_LIMITS_DEG)


def test_toss_profile_limits_scales_all_three_by_one_effort() -> None:
    """All three limits move together, so the profile is the default replayed on a
    stretched clock -- an *effort*, not a speed cap. Scaling `max_vel` alone would push
    the release point into the acceleration phase.
    """
    from kinder_models.dynamic3d.tossing.parameterized_skills import (
        TOSS_MAX_VEL,
        toss_profile_limits,
    )

    default_vel, default_accel, default_decel = toss_profile_limits()
    for factor in (0.4286, 0.5953, 1.0, 1.7143):
        vel, accel, decel = toss_profile_limits(TOSS_MAX_VEL * factor)
        assert vel == pytest.approx(default_vel * factor)
        assert accel / vel == pytest.approx(default_accel / default_vel)
        assert decel / vel == pytest.approx(default_decel / default_vel)


def test_toss_profile_limits_does_not_clamp_above_the_default() -> None:
    """No clamp, deliberately: `_ARM_MAX_VEL[5] = 70` deg/s is kinder-baselines' own
    conservative constant, not a hardware limit -- the real TidyBot primitive runs that
    joint at 140 deg/s.

    `TOSS_SPEED_BOUNDS` caps at `TOSS_MAX_VEL`, so this checks upstream rather than our
    range. A clamp introduced at or below the default would be invisible from inside our
    own bounds, since every draw would hit it identically.
    """
    from kinder_models.dynamic3d.tossing.parameterized_skills import (
        TOSS_MAX_VEL,
        toss_profile_limits,
    )

    fast = toss_profile_limits(np.deg2rad(240.0))
    assert np.rad2deg(fast[0]) == pytest.approx(240.0)
    assert fast[0] > TOSS_MAX_VEL


def test_toss_controller_reset_accepts_a_release_speed() -> None:
    """The parameter has to reach the controller, not just the helper. Signature-only:
    constructing a `TossController` opens a PyBullet client.
    """
    from kinder_models.dynamic3d.tossing.parameterized_skills import (
        TOSS_MAX_VEL,
        TossController,
    )

    parameters = inspect.signature(TossController.reset).parameters
    assert "release_speed" in parameters
    assert parameters["release_speed"].default == TOSS_MAX_VEL


def test_toss_controller_reset_accepts_a_gripper_release_millisecond() -> None:
    """The second dial. Signature-only, for the same reason as the sibling test above."""
    from kinder_models.dynamic3d.tossing.parameterized_skills import (
        TOSS_DEFAULT_GRIPPER_RELEASE_MS,
        TossController,
    )

    parameters = inspect.signature(TossController.reset).parameters
    assert "gripper_release_ms" in parameters
    assert parameters["gripper_release_ms"].default == TOSS_DEFAULT_GRIPPER_RELEASE_MS
    assert TOSS_DEFAULT_GRIPPER_RELEASE_MS == 720


def test_the_release_fraction_trigger_is_gone_rather_than_kept_alongside() -> None:
    """There must be exactly one way to say when the gripper opens. With
    `_release_fraction` still present, a `gripper_release_ms` we pass could be silently
    overridden by a distance-fraction test that fired first.
    """
    from kinder_models.dynamic3d.tossing import parameterized_skills

    source = inspect.getsource(parameterized_skills)
    assert "_release_fraction" not in source
    assert "gripper_release_ms" in source

"""What the `reference/kinder-baselines` pin has to expose for this domain to work.

A gitlink is a silent dependency: bumping it changes what every Tossing3D run executes,
and nothing in this repo's diff shows the code that moved. So the capabilities this
domain relies on get asserted here, against the *pinned* checkout, rather than being
assumed from a SHA in a table.

The pin moved `3524010` -> `1b564a1` (`joshnroy/kinder-baselines` PR #8) to make the
toss's release speed a parameter. Before that bump `toss_profile_limits` did not exist
and `TossController.reset` took no `release_speed`: the swing's `(140, 300, 200)` deg/s
limits were literals inline in `reset`. Every one of these assertions therefore fails at
the old pin, which is the point -- it is what stops a later `git submodule update` from
quietly walking the pin backwards under a domain that now needs the parameter.

**Every test here skips cleanly without KINDER**, gated on `find_spec` for the *import*
package names (`kinder`, `kinder_models`) rather than the distribution name
`kindergarden`. CI never installs the optional extra.

This file deliberately checks only upstream's own API surface -- no simulator, no
controller execution -- so it is fast and cannot fail for a reason unrelated to the pin.
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

# The swing's hand-tuned "throw hard" limits, in deg/s and deg/s^2. These are upstream's
# own literals, demonstrated on the real TidyBot (`yixuanhuang98/tidybot_real`,
# `robot/kinova.py:120-124`), and they were inline in `TossController.reset` until PR #8
# turned the first of them into a parameter defaulting to exactly this value.
UPSTREAM_TOSS_LIMITS_DEG = (140.0, 300.0, 200.0)


def test_toss_profile_limits_exists_and_defaults_to_upstreams_literals() -> None:
    """The default triple must be byte-for-byte what every earlier result was measured at.

    This is the compatibility claim the pin bump rests on: a caller that passes no
    release speed gets the motion the old inline literals produced, so no committed
    Tossing3D number is invalidated by the parameter merely existing.
    """
    from kinder_models.dynamic3d.tossing.parameterized_skills import (
        TOSS_MAX_VEL,
        toss_profile_limits,
    )

    assert np.rad2deg(TOSS_MAX_VEL) == pytest.approx(140.0)
    assert np.rad2deg(toss_profile_limits()) == pytest.approx(UPSTREAM_TOSS_LIMITS_DEG)


def test_toss_profile_limits_scales_all_three_by_one_effort() -> None:
    """All three limits move together, so the profile is the default replayed on a
    stretched clock.

    This is what makes the parameter an *effort* rather than a speed cap, and it is
    load-bearing for us: `TossController` releases at a fixed fraction of the path's
    *distance*, so scaling `max_vel` alone would push the release point into the
    acceleration phase and the commanded release speed would stop tracking the
    parameter. Asserted as a ratio invariant rather than at one sample.
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
    """No clamp, deliberately.

    `_ARM_MAX_VEL[5] = 70` deg/s is kinder-baselines' own conservative constant, not a
    hardware limit -- the real TidyBot primitive this ports runs that joint at 140 deg/s.
    So a release speed above upstream's own `TOSS_MAX_VEL` is a legitimate request of the
    controller, and this asserts the controller honours it.

    **Our sampler no longer draws above 140**: `TOSS_SPEED_BOUNDS` caps at exactly
    `TOSS_MAX_VEL`, because 140 is what the real primitive commands. That makes this test
    a check on upstream rather than on our range -- and it is still worth keeping, for two
    reasons. A clamp introduced at *or below* the default would be invisible from inside
    our own bounds, since every draw we make would hit it identically. And the cap is a
    policy choice about what the robot should be asked to do, not a statement that the
    controller cannot go faster; if it is ever revisited, this is what says whether the
    mechanism underneath still responds.
    """
    from kinder_models.dynamic3d.tossing.parameterized_skills import (
        TOSS_MAX_VEL,
        toss_profile_limits,
    )

    fast = toss_profile_limits(np.deg2rad(240.0))
    assert np.rad2deg(fast[0]) == pytest.approx(240.0)
    assert fast[0] > TOSS_MAX_VEL


def test_toss_controller_reset_accepts_a_release_speed() -> None:
    """The parameter has to reach the controller, not just the helper.

    Signature-only: constructing a `TossController` opens a PyBullet client, which this
    file has no business doing.
    """
    from kinder_models.dynamic3d.tossing.parameterized_skills import (
        TOSS_MAX_VEL,
        TossController,
    )

    parameters = inspect.signature(TossController.reset).parameters
    assert "release_speed" in parameters
    assert parameters["release_speed"].default == TOSS_MAX_VEL


def test_toss_controller_reset_accepts_a_gripper_release_millisecond() -> None:
    """The second dial, added by `joshnroy/kinder-baselines` PR #12.

    Signature-only, for the same reason as the sibling test above.
    """
    from kinder_models.dynamic3d.tossing.parameterized_skills import (
        TOSS_DEFAULT_GRIPPER_RELEASE_MS,
        TossController,
    )

    parameters = inspect.signature(TossController.reset).parameters
    assert "gripper_release_ms" in parameters
    assert parameters["gripper_release_ms"].default == TOSS_DEFAULT_GRIPPER_RELEASE_MS
    assert TOSS_DEFAULT_GRIPPER_RELEASE_MS == 720


def test_the_release_fraction_trigger_is_gone_rather_than_kept_alongside() -> None:
    """kb#12 **deletes** `_release_fraction` rather than leaving it beside the new dial.

    This matters to us, not just to upstream tidiness: two ways to say when the gripper
    opens would mean a `gripper_release_ms` we passed could be silently overridden by a
    distance-fraction test that fired first, and our dial would appear connected while
    doing nothing at some speeds and something at others. Asserted against the pinned
    source so a pin walking backwards is loud.
    """
    from kinder_models.dynamic3d.tossing import parameterized_skills

    source = inspect.getsource(parameterized_skills)
    assert "_release_fraction" not in source
    assert "gripper_release_ms" in source

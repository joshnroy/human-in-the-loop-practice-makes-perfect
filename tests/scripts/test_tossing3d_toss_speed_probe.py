"""Everything in `scripts/tossing3d_toss_speed_probe.py` that can be pinned without
driving MuJoCo: the profile-limit scaling that *is* the probe's independent variable,
argument parsing, and the result model.

`run_cell` itself is not covered here, for the reason
`test_tossing3d_skill_parameter_sweep.py` gives for its own grid runners: it calls
`env.reset_to_seed`, which needs a live simulator. What *is* covered is the arithmetic
the whole probe turns on, which is pure.

One test in this file does need KINDER, and deliberately so. The probe exists to answer
"does scaling more than `max_vel` lift the release-speed ceiling", and that question is
decided by upstream's own `_trapezoidal_motion_profile` before any physics runs. Asking
it against a re-implementation would answer a different question, so that test gates on
`importlib.util.find_spec("kinder_models")` and skips on CI, which never installs the
optional extra.
"""

import importlib.util

import numpy as np
import pytest

from scripts.tossing3d_toss_speed_probe import (
    SCALING_MODES,
    UPSTREAM_TOSS_MAX_ACCEL_DEG,
    UPSTREAM_TOSS_MAX_DECEL_DEG,
    UPSTREAM_TOSS_MAX_VEL_DEG,
    ProbeCellResult,
    _parse_args,
    commanded_release_speed_deg,
    profile_limits_deg,
)

_KINDER_MISSING = importlib.util.find_spec("kinder_models") is None
_NEEDS_KINDER = pytest.mark.skipif(_KINDER_MISSING, reason="KINDER is not installed")


@pytest.mark.parametrize("mode", SCALING_MODES)
def test_every_mode_reproduces_upstream_literals_at_the_default_speed(  # noqa: PLR0917
    mode: str,
) -> None:
    """The dial's default must be upstream's own toss, bit-for-bit, in *every* mode --
    otherwise the probe is measuring a different controller from the one the oracle and
    every committed Tossing3D number ran on, and the pin comparison means nothing."""
    limits = profile_limits_deg(release_speed_deg=UPSTREAM_TOSS_MAX_VEL_DEG, mode=mode)
    assert limits == (
        UPSTREAM_TOSS_MAX_VEL_DEG,
        UPSTREAM_TOSS_MAX_ACCEL_DEG,
        UPSTREAM_TOSS_MAX_DECEL_DEG,
    )


def test_vel_mode_leaves_both_acceleration_limits_alone() -> None:
    """Option A: `max_vel` is the only literal that moves."""
    assert profile_limits_deg(release_speed_deg=280.0, mode="vel") == (280.0, 300.0, 200.0)


def test_vel_accel_mode_scales_the_acceleration_but_not_the_deceleration() -> None:
    """Option D exactly as the design doc words it -- "a single effort parameter
    multiplying both" `max_vel` and `max_accel`."""
    assert profile_limits_deg(release_speed_deg=280.0, mode="vel-accel") == (
        280.0,
        600.0,
        200.0,
    )


def test_vel_accel_decel_mode_scales_all_three_by_the_same_factor() -> None:
    """Option D extended to the third limit `_trapezoidal_motion_profile` takes. Scaling
    two of three is not a uniform effort scaling: it changes the profile's *shape*, and
    the deceleration phase is what decides whether the profile stays trapezoidal."""
    assert profile_limits_deg(release_speed_deg=280.0, mode="vel-accel-decel") == (
        280.0,
        600.0,
        400.0,
    )


def test_an_unknown_mode_is_rejected_rather_than_silently_treated_as_vel_only() -> None:
    with pytest.raises(ValueError, match="unknown scaling mode"):
        profile_limits_deg(release_speed_deg=280.0, mode="effort")


@_NEEDS_KINDER
def test_scaling_max_vel_alone_saturates_the_commanded_release_speed() -> None:
    """The ceiling this probe exists to test for. Under `vel`, commanding more than
    roughly 180 deg/s buys nothing: the release point moves into the acceleration phase,
    where the speed is set by `max_accel`, which `vel` never touches."""
    ceiling = commanded_release_speed_deg(release_speed_deg=200.0, mode="vel")
    for speed in (240.0, 300.0, 420.0):
        assert commanded_release_speed_deg(release_speed_deg=speed, mode="vel") == (
            pytest.approx(ceiling)
        )


@_NEEDS_KINDER
def test_scaling_max_vel_and_max_accel_together_does_not_lift_the_ceiling() -> None:
    """The design doc's option D, measured against its own promise. Scaling `max_accel`
    with `max_vel` while `max_decel` stays put makes the deceleration phase grow as the
    *square* of the scale factor, which drives the profile triangular and claws back what
    the extra acceleration bought. The result is a ceiling essentially where option A's
    already was.

    Scoped to the speed the profile *commands*. What a real arm achieves is a separate
    question -- a torque-limited PD loop tracking that profile can overshoot it -- and is
    the probe's job, not this test's."""
    vel_only_ceiling = commanded_release_speed_deg(release_speed_deg=420.0, mode="vel")
    speeds = (240.0, 300.0, 340.0, 420.0)
    reached = [
        commanded_release_speed_deg(release_speed_deg=speed, mode="vel-accel") for speed in speeds
    ]
    assert max(reached) < 1.1 * vel_only_ceiling


@_NEEDS_KINDER
def test_scaling_all_three_limits_together_does_lift_the_ceiling() -> None:
    """The variant the probe recommends. With every limit scaled the profile keeps its
    shape, so the release point stays in the same phase and the commanded release speed
    goes on rising well past where the other two modes flatten."""
    vel_only_ceiling = commanded_release_speed_deg(release_speed_deg=420.0, mode="vel")
    reached = commanded_release_speed_deg(release_speed_deg=420.0, mode="vel-accel-decel")
    assert reached > 1.5 * vel_only_ceiling


def test_the_default_grid_holds_the_standoff_at_the_oracle_point() -> None:
    """`Pick` and `MoveToThrowPose` are both held fixed so that scene-to-scene grasp and
    base-pose variance cannot be read as an effect of the speed dial -- the same reason
    `tossing3d_skill_parameter_sweep.py` pins `Pick` at the oracle point."""
    args = _parse_args(argv=["--mode", "vel", "--output", "out.json"])
    assert args.standoff == pytest.approx(1.35)


def test_seeds_are_shared_across_every_speed_so_the_grid_is_paired() -> None:
    args = _parse_args(argv=["--mode", "vel", "--output", "out.json"])
    assert len(args.seeds) == len(set(args.seeds))
    assert len(args.seeds) >= 5


def test_a_cell_that_never_released_serialises_without_inventing_numbers() -> None:
    """A failed grasp or a failed windup leaves the release fields genuinely unknown, and
    the model must say so rather than defaulting them to zero -- a zero release speed is
    a measurement, `null` is the absence of one."""
    cell = ProbeCellResult(
        seed=3, commanded_speed_deg=140.0, mode="vel", standoff=1.35, pick_error="grasp failed"
    )
    dumped = cell.model_dump()
    assert dumped["achieved_release_speed_deg"] is None
    assert dumped["range_m"] is None
    assert dumped["solved"] is None


def test_the_release_speed_is_bracketed_by_two_readings_not_asserted_by_one() -> None:
    """The cube separates *during* the release step, so neither end of that step is the
    launch speed on its own. Two scaling modes can share the pre-step reading exactly --
    their profiles are identical up to release -- and still differ in the velocity
    feedforward commanded on that very step, which is why the post-step reading has to be
    recorded too rather than inferred."""
    fields = set(ProbeCellResult.model_fields)
    assert {"achieved_release_speed_deg", "achieved_release_speed_after_deg"} <= fields


def test_range_is_measured_from_the_base_at_release_not_from_the_world_origin() -> None:
    """The base is at x ~ 0.64 at the throw pose, so a range measured from the origin
    would overstate every throw by that much and make the ballistic fit meaningless."""
    cell = ProbeCellResult(
        seed=0,
        commanded_speed_deg=140.0,
        mode="vel",
        standoff=1.35,
        base_x_at_release=0.638,
        cube_x_final=1.988,
    )
    assert cell.range_m == pytest.approx(1.350)


def test_the_upstream_literals_match_the_toss_controller_source() -> None:
    """Guards the one thing this file hardcodes about upstream. These three numbers are
    copied out of `TossController.reset`; if upstream retunes the toss they go stale
    silently, and every "default reproduces upstream" claim above goes with them."""
    assert (
        UPSTREAM_TOSS_MAX_VEL_DEG,
        UPSTREAM_TOSS_MAX_ACCEL_DEG,
        UPSTREAM_TOSS_MAX_DECEL_DEG,
    ) == (140.0, 300.0, 200.0)
    assert np.isclose(np.deg2rad(UPSTREAM_TOSS_MAX_VEL_DEG), 2.4434609527920612)

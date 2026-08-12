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
    ALL_MODES,
    IN_SPEC_MODE,
    SCALING_MODES,
    UPSTREAM_TOSS_MAX_ACCEL_DEG,
    UPSTREAM_TOSS_MAX_DECEL_DEG,
    UPSTREAM_TOSS_MAX_VEL_DEG,
    ProbeCellResult,
    _parse_args,
    clip_caption,
    commanded_release_speed_deg,
    profile_limits_deg,
    toss_profile_ceilings_deg,
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


def test_in_spec_is_offered_as_a_mode_but_is_not_one_of_the_scaling_modes() -> None:
    """`in-spec` does not scale upstream's literals -- it replaces them with limits
    derived from KINDER's own declared per-joint ceilings. Keeping it out of
    `SCALING_MODES` is what lets
    `test_every_mode_reproduces_upstream_literals_at_the_default_speed` stay true: at
    140 deg/s `in-spec` deliberately does *not* reproduce them, because 140 is 1.68x
    over the ceiling and gets clamped."""
    assert IN_SPEC_MODE not in SCALING_MODES
    assert set(ALL_MODES) == set(SCALING_MODES) | {IN_SPEC_MODE}


@_NEEDS_KINDER
def test_the_in_spec_ceilings_are_derived_from_kinders_declared_arm_limits() -> None:
    """The whole point of deriving rather than hardcoding: the ceiling must track
    `_ARM_MAX_VEL`/`_ARM_MAX_ACCEL`, so that changing kinder's declared limits changes
    this and nothing has to remember to follow. Joint 6 binds, carrying 0.8399 of the
    toss direction against the smallest velocity limit in the arm."""
    from kinder_models.dynamic3d.tossing.parameterized_skills import (
        TOSS_RELEASE_ARM_CONF,
        TOSS_WINDUP_ARM_CONF,
    )
    from kinder_models.dynamic3d.utils import _ARM_MAX_ACCEL, _ARM_MAX_VEL  # noqa: PLC2701

    direction = TOSS_RELEASE_ARM_CONF - TOSS_WINDUP_ARM_CONF
    direction = direction / float(np.linalg.norm(direction))
    moving = np.abs(direction) > 1e-6
    expected_vel = float(np.rad2deg(np.min(_ARM_MAX_VEL[moving] / np.abs(direction)[moving])))
    expected_accel = float(np.rad2deg(np.min(_ARM_MAX_ACCEL[moving] / np.abs(direction)[moving])))

    vel_ceiling, accel_ceiling = toss_profile_ceilings_deg()

    assert vel_ceiling == pytest.approx(expected_vel)
    assert accel_ceiling == pytest.approx(expected_accel)
    # Joint 6 (index 5) is the binding one, and it binds on velocity.
    assert int(np.argmin(_ARM_MAX_VEL[moving] / np.abs(direction)[moving])) == 2


@_NEEDS_KINDER
def test_in_spec_mode_clamps_a_request_above_the_ceiling_instead_of_honouring_it() -> None:
    """Upstream's shipped 140 deg/s is 1.68x the derived ceiling. Under `in-spec` the
    dial saturates there rather than reproducing it, which is exactly the difference
    between a hardware-feasible throw and the one that ships today."""
    vel_ceiling, _ = toss_profile_ceilings_deg()

    at_default = profile_limits_deg(release_speed_deg=UPSTREAM_TOSS_MAX_VEL_DEG, mode=IN_SPEC_MODE)
    way_over = profile_limits_deg(release_speed_deg=420.0, mode=IN_SPEC_MODE)

    assert at_default[0] == pytest.approx(vel_ceiling)
    assert way_over[0] == pytest.approx(vel_ceiling)
    assert vel_ceiling < UPSTREAM_TOSS_MAX_VEL_DEG


@_NEEDS_KINDER
def test_in_spec_mode_decelerates_as_hard_as_it_accelerates() -> None:
    """`_compute_per_joint_profile`, which `MoveArmToConfController` already uses, passes
    `max_decel = max_accel`. Adopting its convention repairs the shipped profile's 1.120x
    deceleration asymmetry as a side effect rather than as a separate change."""
    _, accel_ceiling = toss_profile_ceilings_deg()

    for requested in (60.0, 75.0, 140.0):
        _, accel, decel = profile_limits_deg(release_speed_deg=requested, mode=IN_SPEC_MODE)
        assert accel == pytest.approx(accel_ceiling)
        assert decel == pytest.approx(accel_ceiling)


@_NEEDS_KINDER
def test_in_spec_mode_keeps_every_joint_inside_its_declared_limits() -> None:
    """The hardware-feasibility claim itself, checked per joint rather than on the path
    scalar. Each joint moves `|toss_dir_j|` of the path rate, so a path rate that is in
    spec on the binding joint must be in spec on all of them."""
    from kinder_models.dynamic3d.tossing.parameterized_skills import (
        TOSS_RELEASE_ARM_CONF,
        TOSS_WINDUP_ARM_CONF,
    )
    from kinder_models.dynamic3d.utils import _ARM_MAX_ACCEL, _ARM_MAX_VEL  # noqa: PLC2701

    direction = TOSS_RELEASE_ARM_CONF - TOSS_WINDUP_ARM_CONF
    direction = np.abs(direction / float(np.linalg.norm(direction)))

    for requested in (60.0, 70.0, 83.0, 140.0, 420.0):
        vel, accel, decel = profile_limits_deg(release_speed_deg=requested, mode=IN_SPEC_MODE)
        per_joint_vel = np.deg2rad(vel) * direction
        per_joint_accel = np.deg2rad(max(accel, decel)) * direction
        assert np.all(per_joint_vel <= _ARM_MAX_VEL + 1e-9)
        assert np.all(per_joint_accel <= _ARM_MAX_ACCEL + 1e-9)


def test_a_clip_caption_names_the_parameters_that_produced_the_throw() -> None:
    """A clip is evidence only if a viewer can read the parameters off the frame. The
    renderer's second caption line already carries the measured landing x and the
    `InGoalRegion` verdict from the state being drawn, so this line carries the inputs:
    which standoff, which commanded speed, and -- for the non-monotone pair -- the
    achieved release speed and realised release fraction that explain why a *faster*
    command can land *shorter*."""
    caption = clip_caption(
        mode=IN_SPEC_MODE,
        speed_deg=70.0,
        seed=3,
        standoff=1.100,
        achieved_release_speed_deg=72.6,
        release_fraction=0.4771,
    )
    assert "in-spec" in caption
    assert "70" in caption
    assert "seed 3" in caption
    assert "1.100" in caption
    assert "72.6" in caption
    assert "0.477" in caption


def test_a_clip_caption_omits_the_measured_fields_when_they_are_unknown() -> None:
    """`record_cell` drives the public `take_action` route and never observes a release
    instant, so those two fields are genuinely absent unless the grid supplies them.
    Printing `None` or a fabricated `0.0` on a video frame would be worse than omitting
    them."""
    caption = clip_caption(mode=IN_SPEC_MODE, speed_deg=60.0, seed=0, standoff=1.05)
    assert "None" not in caption
    assert "rel" not in caption


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

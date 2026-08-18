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

from .kinder_symbols import MovedKinderSymbol, RenamedKinderSymbol

needs_kinder = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None or importlib.util.find_spec("kinder_models") is None,
    reason="KINDER is an optional extra (`kindergarden` + `kinder_models`); CI never installs it",
)

pytestmark = needs_kinder

# The swing's hand-tuned "throw hard" limits, in deg/s and deg/s^2 -- upstream's own
# literals, demonstrated on the real TidyBot (`yixuanhuang98/tidybot_real`,
# `robot/kinova.py:120-124`).
UPSTREAM_TOSS_LIMITS_DEG = (140.0, 300.0, 200.0)

# Where the swing lives, newest-first. `toss_swing.py` is a split-out of
# `parameterized_skills.py`, and it is only a partial one: `parameterized_skills`
# re-exports most of the moved names for its own use, but not `toss_profile_limits` or
# `TOSS_SLICES_PER_CONTROL_STEP`. Those two are therefore the only ones resolved through
# `MovedKinderSymbol`; everything else imports directly and resolves on both lines.
TOSS_SWING_MODULES = (
    "kinder_models.dynamic3d.tossing.toss_swing",
    "kinder_models.dynamic3d.tossing.parameterized_skills",
)


def _toss_profile_limits():
    """Upstream's `(max_vel, max_accel, max_decel)` helper, wherever it currently lives."""
    return MovedKinderSymbol.resolve(modules=TOSS_SWING_MODULES, names=("toss_profile_limits",))


def test_toss_profile_limits_exists_and_defaults_to_upstreams_literals() -> None:
    """The default triple must be what every committed Tossing3D number was measured at,
    so that passing no release speed leaves those numbers valid.
    """
    from kinder_models.dynamic3d.tossing.parameterized_skills import TOSS_MAX_VELOCITY

    toss_profile_limits = _toss_profile_limits()

    assert np.rad2deg(TOSS_MAX_VELOCITY) == pytest.approx(140.0)
    assert np.rad2deg(toss_profile_limits()) == pytest.approx(UPSTREAM_TOSS_LIMITS_DEG)


def test_toss_profile_limits_scales_all_three_by_one_effort() -> None:
    """All three limits move together, so the profile is the default replayed on a
    stretched clock -- an *effort*, not a speed cap. Scaling `max_vel` alone would push
    the release point into the acceleration phase.
    """
    from kinder_models.dynamic3d.tossing.parameterized_skills import TOSS_MAX_VELOCITY

    toss_profile_limits = _toss_profile_limits()

    default_vel, default_accel, default_decel = toss_profile_limits()
    for factor in (0.4286, 0.5953, 1.0):
        vel, accel, decel = toss_profile_limits(TOSS_MAX_VELOCITY * factor)
        assert vel == pytest.approx(default_vel * factor)
        assert accel / vel == pytest.approx(default_accel / default_vel)
        assert decel / vel == pytest.approx(default_decel / default_vel)


def test_toss_profile_limits_clamps_effort_at_the_default() -> None:
    """`TOSS_MAX_VELOCITY` is a genuine ceiling: a release speed above it is clamped
    rather than scaling the profile past the default.

    The sampler's own band tops out at `TOSS_MAX_VELOCITY`, so this probes above the
    reachable range deliberately -- from inside the band the clamp is unreachable and so
    invisible.
    """
    from kinder_models.dynamic3d.tossing.parameterized_skills import TOSS_MAX_VELOCITY

    toss_profile_limits = _toss_profile_limits()

    assert toss_profile_limits(np.deg2rad(240.0)) == pytest.approx(toss_profile_limits())
    assert toss_profile_limits(np.deg2rad(240.0))[0] == pytest.approx(TOSS_MAX_VELOCITY)


def test_no_release_speed_the_sampler_can_draw_is_ever_silently_clamped() -> None:
    """The clamp must sit at or above the top of the band that is actually drawn from, so
    no sampled speed is rewritten on its way into the profile.

    **This test changed hands rather than changing meaning.** It used to read
    `TOSS_SPEED_BOUNDS` out of this repo's own `predicates.py`. That constant is gone --
    the fused `move_to_toss_location_and_toss` controller carries its own `SPEED_BOUNDS`
    and its own `sample_parameters` draws from them, which is the whole point of the
    bridge (a second copy of upstream's number is a number that can drift). The property
    worth protecting is unchanged, so it is re-homed onto the constant that now decides
    it instead of being deleted with the one that used to.

    The top edge is the boundary case and is asserted exactly: it *is* `TOSS_MAX_VELOCITY`,
    so it must pass through unscaled rather than tripping the clamp.
    """
    from kinder_models.dynamic3d.tossing.parameterized_skills import (
        TOSS_MAX_VELOCITY,
        MoveToTossLocationAndTossController,
    )

    toss_profile_limits = _toss_profile_limits()
    low, high = MoveToTossLocationAndTossController.SPEED_BOUNDS

    assert high == pytest.approx(TOSS_MAX_VELOCITY), (
        "the sampler's upper bound has come off the clamp point, so the top of its own "
        "range is now silently rewritten on the way into the profile"
    )
    for speed in np.linspace(low, high, 37):
        assert toss_profile_limits(speed)[0] == pytest.approx(speed)


def test_the_sampler_draws_all_four_dials_from_upstreams_own_bounds() -> None:
    """The bounds this repo used to declare, now read off the controller that owns them.

    hitl carried `TOSS_RELEASE_MS_BOUNDS = (300, 1400)` against a controller band of
    `(700, 840)` -- a window about nine times too wide, so the large majority of its draws
    could not score, and nothing detected it because both numbers were internally
    consistent. There is one number now. This pins that the sampler really does draw four
    parameters, since `param_dim` is declared in `skills.py` and
    `LiftedParameterizedController.params_space` is `None` on every Tossing3D controller,
    so there is nothing to read the arity off at runtime.
    """
    from kinder_models.dynamic3d.tossing.parameterized_skills import (
        MoveToTossLocationAndTossController,
    )

    controller = MoveToTossLocationAndTossController
    drawn = controller.sample_parameters(controller, None, np.random.default_rng(0))

    assert drawn.shape == (4,)
    for value, bounds in zip(
        drawn[2:],
        (controller.SPEED_BOUNDS, controller.RELEASE_MS_BOUNDS),
        strict=True,
    ):
        assert bounds[0] <= value <= bounds[1]


def test_the_base_plan_ignores_the_held_object_without_being_told_to() -> None:
    """**The one behaviour a deleted test used to protect, re-homed to its new owner.**

    `KinderBackend.run_move_to_throw_pose` used to pass
    `disable_collision_objects=["cube_0"]` by hand, and a test here spied on it. That
    method is gone with the three-skill decomposition, but the hazard is not: upstream
    PR #103 turned base-motion collision-checking *on* (`run_base_motion_planning` had
    hardcoded `obstacle_geoms` empty), so the cube the robot is carrying is an obstacle to
    the robot's own base plan unless something excludes it. Every plan fails if nothing
    does.

    Nothing in this repo does any more -- `_execute` passes no `reset_kwargs` at all -- so
    this is now purely a claim about the pin, which is exactly what this file is for.
    Three parts, because the claim needs all three: the keyword defaults to `None`, `None`
    means "the object at index 1", and index 1 is `?held`.
    """
    import inspect

    from kinder_models.dynamic3d.tossing.parameterized_skills import (
        MoveToTossLocationAndTossController,
        create_lifted_controllers,
    )

    reset_parameters = inspect.signature(MoveToTossLocationAndTossController.reset).parameters
    assert reset_parameters["disable_collision_objects"].default is None

    source = inspect.getsource(MoveToTossLocationAndTossController._plan_base_motion)  # noqa: SLF001
    assert "if disable_collision_objects is None:" in source
    assert "disable_collision_objects = [self.objects[1].name]" in source

    # No simulator: `create_lifted_controllers` `del`s its first two arguments and merely
    # stores `pybullet_sim`, so the declarations are readable without building a scene.
    lifted = create_lifted_controllers(None, None, pybullet_sim=None)
    variables = lifted["move_to_toss_location_and_toss"].variables
    assert [variable.name for variable in variables] == ["?robot", "?held", "?barrier"], (
        "the controller's parameters have been reordered, so `self.objects[1]` is no "
        "longer the held object and the base plan now excludes the wrong thing"
    )


def test_the_gripper_release_millisecond_is_rounded_rather_than_truncated() -> None:
    """Upstream `divmod`s an int, and `int()` truncates toward zero, so `722.9` meaning
    722 would be a systematic bias toward releasing early -- and the arm is near peak
    speed at release, where 1 ms is millimetres of landing position rather than a rounding
    detail.

    This used to be `KinderBackend.run_toss`'s job and had its own test there. The dial is
    drawn by the controller's own sampler now and rounded inside its own `reset`, so the
    check belongs here: it is upstream's behaviour that this repo depends on.
    """
    import inspect

    from kinder_models.dynamic3d.tossing.parameterized_skills import (
        MoveToTossLocationAndTossController,
    )

    source = inspect.getsource(MoveToTossLocationAndTossController.reset)
    assert "int(round(float(current_params[3])))" in source, (
        "the release millisecond is no longer rounded on its way into the controller; "
        "if this moved rather than went away, follow it -- truncation biases every throw "
        "toward an early release"
    )


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
    from kinder_models.dynamic3d import utils

    slices_per_control_step = MovedKinderSymbol.resolve(
        modules=TOSS_SWING_MODULES, names=("TOSS_SLICES_PER_CONTROL_STEP",)
    )
    control_timestep = RenamedKinderSymbol.resolve(
        module=utils, names=("CONTROL_TIMESTEP", "_CONTROL_TIMESTEP")
    )
    assert isinstance(control_timestep, float)

    num_sim_steps = int(control_timestep / SIMULATION_TIMESTEP)
    ticks_per_row = int(round(CONTROL_SCHEDULE_TIMESTEP / SIMULATION_TIMESTEP))
    assert num_sim_steps // ticks_per_row == slices_per_control_step


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

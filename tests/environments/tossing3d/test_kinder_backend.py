"""Offline tests for `KinderBackend`'s GL backend, sub-step recording switch and
controller seam.

Everything here runs without MuJoCo, which is the point: the switch, the drain and the
argument marshalling are plain Python, and CI is exactly where a wiring regression would
otherwise go unnoticed. Whether the wrapper really collects one frame per physics tick,
and whether a controller really does what it says, are questions only a simulator can
answer and live in `test_kinder_fidelity.py`.

**The `run_move_to_throw_pose` and `run_toss` tests are deleted rather than ported**,
because the seams they guarded no longer exist:

- `run_move_to_throw_pose` threaded `disable_collision_objects=["cube_0"]` through to
  upstream's `move_to_target`, so the robot's own held cube did not become an obstacle to
  its own base plan. `run_controller` no longer takes that keyword at all -- the composed
  controller's own default is the held cube's name, so passing it from here would be this
  package inventing a controller parameter.
- `run_toss` rounded `gripper_release_ms` to an int before handing it on, because the old
  `TossController.reset` truncated toward zero. Upstream's `plan_toss_swing` does its own
  `int(round(...))`, so the millisecond stays a float across this seam.
- `run_toss` drove two controllers (`move_arm_to_conf` then `toss`) and skipped the swing
  when the windup failed. There is one controller now and no windup for this package to
  sequence.

All three properties are re-asserted below in the form the new seam has, as absences
where they used to be presences.
"""

import importlib.util
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from hitl_pmp.environments.tossing3d import kinder_backend
from hitl_pmp.environments.tossing3d.kinder_backend import ControllerRun, KinderBackend

# --- the GL backend --------------------------------------------------------------
#
# `register_all_environments()` rewrites MUJOCO_GL to `osmesa` when DISPLAY is unset, and
# under a backend that is selected but not installed `import mujoco` raises into a handler
# that swallows every exception -- so the Dynamic3D category vanishes in silence. Undoing
# that rewrite is why this function exists. It used to undo it by hardcoding `egl`, which
# also silently overrode an operator who had *asked* for osmesa: on a headless CI runner
# with no EGL driver every Tossing3D test then failed with
# `AttributeError: 'NoneType' object has no attribute 'eglQueryString'`, and no workflow
# env could reach it. The snapshot is what separates the two cases -- it is read before
# KINDER can write anything, so what it holds is a request rather than a rewrite.


def test_an_explicitly_requested_backend_is_reasserted_rather_than_overridden() -> None:
    """The reversal: a deliberately-requested backend now survives, where it used to be
    replaced by `egl` on every call."""
    environ = {"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": ""}
    resolved = KinderBackend.configure_headless_rendering(
        environ=environ, backend="osmesa", platform="osmesa"
    )
    assert resolved["MUJOCO_GL"] == "osmesa"
    assert resolved["PYOPENGL_PLATFORM"] == "osmesa"
    assert environ["MUJOCO_GL"] == "osmesa"


def test_the_platform_follows_the_backend_when_only_a_backend_is_requested() -> None:
    """CI passes `PYOPENGL_PLATFORM=""` deliberately, so the pair must stay consistent
    without the caller restating it."""
    resolved = KinderBackend.configure_headless_rendering(environ={}, backend="osmesa")
    assert resolved["PYOPENGL_PLATFORM"] == "osmesa"


def test_nothing_requested_still_means_egl() -> None:
    """The workstation path, unchanged: no request, so the default stands."""
    resolved = KinderBackend.configure_headless_rendering(
        environ={}, backend=kinder_backend.DEFAULT_GL_BACKEND
    )
    assert resolved["MUJOCO_GL"] == "egl"
    assert resolved["PYOPENGL_PLATFORM"] == "egl"
    assert resolved["DISPLAY"] == kinder_backend.FALLBACK_DISPLAY


def test_kinders_own_rewrite_is_still_undone() -> None:
    """The property the hardcoded `egl` was protecting, kept: an `osmesa` that arrived
    from upstream's rewrite rather than from a request is still replaced."""
    environ = {"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa", "DISPLAY": ":0"}
    resolved = KinderBackend.configure_headless_rendering(environ=environ, backend="egl")
    assert resolved["MUJOCO_GL"] == "egl"
    assert resolved["PYOPENGL_PLATFORM"] == "egl"


def test_a_backend_written_after_import_is_ignored(*, monkeypatch: pytest.MonkeyPatch) -> None:
    """The distinction itself. A value that appears in the environment *after* import is
    exactly what `register_all_environments()` produces, so it must not be mistaken for a
    request -- whatever this machine happens to have asked for.

    `monkeypatch` rather than a hand-rolled `os.environ` edit: an earlier draft popped the
    variable in a `finally` instead of restoring it, which left the *process* with no
    `MUJOCO_GL` and made every later Tossing3D test in the run render through the wrong
    backend. It passed locally, where the ambient value is `egl` either way, and failed
    only on CI.
    """
    late = "osmesa" if kinder_backend.REQUESTED_GL_BACKEND != "osmesa" else "egl"
    monkeypatch.setenv("MUJOCO_GL", late)
    resolved = KinderBackend.configure_headless_rendering(environ={})
    assert resolved["MUJOCO_GL"] == kinder_backend.REQUESTED_GL_BACKEND != late


def test_the_request_is_snapshotted_at_import(*, monkeypatch: pytest.MonkeyPatch) -> None:
    """Where the line is drawn, pinned directly: the snapshot is taken when the module is
    imported, which is necessarily before anything of ours can call into KINDER.

    A **private copy** of the module, not `importlib.reload`. Reloading rebinds the
    snapshot that every other test in the process shares, so it is only as correct as the
    environment at reload time -- and one earlier test leaving `MUJOCO_GL` unset was enough
    to pin the whole run to the wrong backend. Loading a second copy asks the same question
    and cannot answer it for anybody else.

    The probe's name is **package-qualified**, which is load-bearing rather than tidy:
    `kinder_backend` imports `from .types import AbstractAtom`, and a relative import is
    resolved against `__package__`, which `module_from_spec` derives from the spec's own
    name. Under a bare `"_gl_snapshot_probe"` there is no parent package and the copy
    fails to execute at all. The name still differs from the real module's, so the copy is
    never registered in `sys.modules` and no other test sees it.
    """
    before = kinder_backend.REQUESTED_GL_BACKEND
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)
    spec = importlib.util.spec_from_file_location(
        f"{kinder_backend.__package__}._gl_snapshot_probe", kinder_backend.__file__
    )
    assert spec is not None and spec.loader is not None
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)

    assert fresh.REQUESTED_GL_BACKEND == "osmesa"
    assert fresh.REQUESTED_GL_PLATFORM == "osmesa"
    assert before == kinder_backend.REQUESTED_GL_BACKEND


# --- sub-step recording ------------------------------------------------------------


def test_substep_recording_is_off_by_default() -> None:
    """One skill here is hundreds of MuJoCo ticks, and recording renders every one of
    them. A training run that never asked for a video must not pay for one, so the
    default is the cheap behaviour and the demo path opts in."""
    assert KinderBackend().record_substeps is False


def test_toggling_substep_recording_imports_no_simulator() -> None:
    """`KinderBackend` is lazy by construction and the switch must not break that: it is
    flipped by `Tossing3DProblem.run_task_episode`, which a test may reach without ever
    intending to build a scene.

    **This one keeps a subprocess rather than moving to `no_kinder_import`**, and the
    reason is worth stating because the other three did move.

    An in-process `sys.modules` check stopped meaning anything here: the predicates are
    upstream's now, so `test_kb_predicate_parity.py` genuinely imports `kinder_models` --
    and therefore MuJoCo -- inside its own tests, and it sorts before this file. A
    same-interpreter assertion was reading *another module's* import and failing for a
    reason that has nothing to do with `KinderBackend`.

    A fresh interpreter fixes that and is **strictly stronger than the fixture**: the
    fixture is behavioural, so it catches only code that reaches `KinderBackend.api()` at
    runtime, while this catches a module-scope `import kinder` anywhere in the chain too --
    the gap `conftest.py` names as its own. The cost is one interpreter spawn, which is why
    it is not the mechanism for all four.
    """
    backend = KinderBackend()
    backend.set_substep_recording(enabled=True)
    assert backend.record_substeps is True

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend\n"
            "KinderBackend().set_substep_recording(enabled=True)\n"
            "print('mujoco' in sys.modules)\n",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert probe.stdout.strip() == "False", probe.stdout + probe.stderr


def test_draining_before_any_scene_exists_is_empty_rather_than_an_error() -> None:
    """`run_task_episode` drains unconditionally once recording is on -- including the
    clearing drain immediately after a reset. A backend that has never built a scene has
    no frames, and that is an ordinary answer rather than a failure."""
    backend = KinderBackend()
    backend.set_substep_recording(enabled=True)

    assert backend.drain_substep_frames() == []


def test_draining_while_recording_is_off_is_empty_even_with_a_scene() -> None:
    """The drain is keyed on the switch, not on whether a wrapper happens to be in
    place, so turning recording off mid-run cannot leave a half-collected clip behind."""
    backend = KinderBackend()

    assert backend.record_substeps is False
    assert backend.drain_substep_frames() == []


def _spy_run_controller(*, monkeypatch: pytest.MonkeyPatch) -> tuple[KinderBackend, list[dict]]:
    """A `KinderBackend` whose `run_controller` records its arguments instead of running.

    Offline: nothing here builds a scene, which is the point -- what these tests are
    about is which arguments reach `run_controller`, not simulator behaviour.
    """
    calls: list[dict[str, Any]] = []

    # `self` must stay positional -- `monkeypatch.setattr` installs this as an unbound
    # class method, and Python's bound-method call convention always passes the instance
    # positionally, the same reasoning `core/README.md` gives for `__getitem__`.
    def spy(  # noqa: PLR0917
        self: KinderBackend, *, module: str, key: str, **kwargs: Any
    ) -> ControllerRun:
        calls.append({"module": module, "key": key, **kwargs})
        return ControllerRun(steps=1, terminated=True)

    monkeypatch.setattr(KinderBackend, "run_controller", spy)

    backend = KinderBackend()
    # `robot_name` is a property that reads a scene-derived private attribute and raises
    # before any `reset()`; the run methods only need a name to pass through, not a real
    # one, so a `PrivateAttr` write stands in for a scene without a simulator.
    backend._robot_name = "robot_test"  # noqa: SLF001
    return backend, calls


def test_the_pick_drives_upstreams_parameterless_pick_cube(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`params=None` rather than `np.zeros(0)`, because upstream's `sample_parameters`
    returns `tuple()` and its `reset` immediately `del`s the argument. An empty array
    would be this package inventing a shape upstream never asked for.

    Also pins the object triple: upstream's `pick_cube` is grounded on
    `(robot, cube, barrier)`, and the barrier -- which the controller itself ignores -- is
    third rather than second."""
    backend, calls = _spy_run_controller(monkeypatch=monkeypatch)

    backend.run_pick_cube()

    assert len(calls) == 1
    assert calls[0]["module"] == "tossing"
    assert calls[0]["key"] == "pick_cube"
    assert calls[0]["object_names"] == ("robot_test", backend.cube_name, backend.barrier_name)
    assert calls[0]["params"] is None
    assert calls[0]["limit"] == KinderBackend.pick_step_limit


def test_the_pick_passes_no_collision_exclusions_because_passing_one_is_a_type_error(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`disable_collision_objects` exists on `MoveToTargetGroundController.reset` and on
    the composed toss, not on `PickCubeController.reset`. It is also no longer a
    `run_controller` keyword at all, so this is pinned as an absence at both ends."""
    backend, calls = _spy_run_controller(monkeypatch=monkeypatch)

    backend.run_pick_cube()

    assert "disable_collision_objects" not in calls[0]


def test_the_composed_toss_drives_one_controller_rather_than_three(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The migration's headline at this seam. It used to be `move_to_target` ->
    `move_arm_to_conf` -> `toss`, sequenced here; the composition is upstream's now, so
    the phase boundaries are not visible from this package and neither is a per-phase
    step count.

    The object 3-tuple is upstream's own `(?robot, ?held, ?barrier)`, read off
    `create_lifted_controllers` at the pinned commit. **It used to be a 4-tuple here, with
    the bin second**, and that was wrong in a way no spy could catch: this test replaces
    `run_controller`, so the extra object never reached
    `LiftedParameterizedController.ground`, whose `zip(objects, self.variables,
    strict=True)` rejects it. Every test that drives the composed toss failed on it, and
    this one passed.

    Two things ride on the order, which is why it is asserted rather than assumed. The
    controller drives to a pose relative to `bin_0` by name, so the bin does not need to be
    passed at all; and upstream's `_plan_base_motion` defaults
    `disable_collision_objects` to `[self.objects[1].name]` -- the *held* object, on the
    grounds that "the robot's own cargo would otherwise reject every base plan".

    **That second one never actually fired, and the distinction is worth keeping.** With
    the bin in slot 1 the default *would have* disabled collisions against the bin while
    leaving the held cube as an obstacle to the robot's own base plan -- but `ground()`
    runs before `reset()`, so the arity check rejected the 4-tuple before
    `_plan_base_motion` was ever reached. A second, latent defect masked by the louder
    one, which would have surfaced the moment the first was fixed by padding the tuple
    rather than by correcting the order."""
    backend, calls = _spy_run_controller(monkeypatch=monkeypatch)

    backend.run_move_to_toss_location_and_toss(
        distance=1.35, rotation=0.0, release_speed_deg_s=140.0, gripper_release_ms=792.0
    )

    assert [call["key"] for call in calls] == ["move_to_toss_location_and_toss"]
    assert calls[0]["module"] == "tossing"
    assert calls[0]["object_names"] == (
        "robot_test",
        backend.cube_name,
        backend.barrier_name,
    )
    assert calls[0]["limit"] == KinderBackend.toss_step_limit


def test_the_composed_toss_converts_the_release_speed_to_radians_exactly_once(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This domain carries the dial in joint-path deg/s and upstream's `SPEED_BOUNDS` are
    rad/s, so exactly one site converts. A second or missing conversion is a silent 57x
    error either way, and it is silent because both readings are plausible numbers.

    The other three slots must pass through untouched, which is what makes this a
    conversion test rather than a "some number changed" test."""
    backend, calls = _spy_run_controller(monkeypatch=monkeypatch)

    backend.run_move_to_toss_location_and_toss(
        distance=1.31, rotation=-0.007, release_speed_deg_s=128.5, gripper_release_ms=733.0
    )

    assert calls[0]["params"] == pytest.approx([1.31, -0.007, np.deg2rad(128.5), 733.0])


def test_the_composed_toss_leaves_the_gripper_millisecond_a_float(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The old `TossController.reset` truncated toward zero, so this package rounded to a
    whole millisecond before handing it over -- `722.9` would otherwise have meant 722, a
    systematic bias toward releasing early. `plan_toss_swing` does its own
    `int(round(...))`, so rounding again here would be a second rounding of an
    already-rounded value."""
    backend, calls = _spy_run_controller(monkeypatch=monkeypatch)

    backend.run_move_to_toss_location_and_toss(
        distance=1.35, rotation=0.0, release_speed_deg_s=140.0, gripper_release_ms=722.9
    )

    assert calls[0]["params"][3] == pytest.approx(722.9)


def test_the_composed_toss_passes_no_collision_exclusions_of_its_own(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upstream's own default for `disable_collision_objects` is the held cube's name, so
    the robot's cargo does not reject its own base plan. Passing `["cube_0"]` from here
    would duplicate that default and go stale the moment upstream changed it -- and it is
    no longer a `run_controller` keyword in any case."""
    backend, calls = _spy_run_controller(monkeypatch=monkeypatch)

    backend.run_move_to_toss_location_and_toss(
        distance=1.35, rotation=0.0, release_speed_deg_s=140.0, gripper_release_ms=792.0
    )

    assert "disable_collision_objects" not in calls[0]


# --- run_controller's own error handling ----------------------------------------------
#
# `bilevel_planning` ships with `kinder_models` rather than with this repo, so these gate
# the same way every other KINDER-backed test does. They execute no simulator: the
# controller is a stub, and the only thing borrowed from upstream is the exception class,
# which has to be the real one because its BASE CLASS is the whole point.

needs_bilevel_planning = pytest.mark.skipif(
    importlib.util.find_spec("bilevel_planning") is None,
    reason="`bilevel_planning` ships with kinder_models, an optional extra",
)


class _StubController:
    """A grounded controller that fails the way upstream's samplers fail."""

    def __init__(self, *, error: BaseException) -> None:
        self._error = error

    def reset(self, x, params) -> None:  # noqa: PLR0917  (upstream's positional signature)
        del x, params
        raise self._error


class _StubLifted:
    def __init__(self, *, controller: _StubController) -> None:
        self._controller = controller

    def ground(self, objects) -> _StubController:  # noqa: PLR0917
        del objects
        return self._controller


class _StubKinderState:
    def get_object_from_name(self, name: str) -> str:  # noqa: PLR0917
        return name


def _backend_whose_controller_raises(
    *, monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> KinderBackend:
    """A backend wired to a stub controller that raises `error` out of `reset`."""
    from types import ModuleType, SimpleNamespace

    from hitl_pmp.environments.tossing3d.kinder_backend import KinderApi

    lifted = {
        "move_to_toss_location_and_toss": _StubLifted(controller=_StubController(error=error))
    }

    def factory(action_space) -> dict:  # noqa: PLR0917
        del action_space
        return lifted

    api = KinderApi(
        kinder=ModuleType("stub_kinder"),
        robot_type=object,
        tossing_controllers=factory,
        shelf_controllers=factory,
        render_collection=object,
    )
    monkeypatch.setattr(KinderBackend, "api", lambda self: api)  # noqa: PLR0917

    backend = KinderBackend()
    backend._robot_name = "robot_test"  # noqa: SLF001
    backend._state = _StubKinderState()  # noqa: SLF001
    backend._env = SimpleNamespace(action_space=None)  # noqa: SLF001
    return backend


@needs_bilevel_planning
def test_a_trajectory_sampling_failure_is_reported_rather_than_allowed_to_escape(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**`TrajectorySamplingFailure` is not an `Exception` subclass**, so the `except
    Exception` this handler used to be written as missed every sampling failure and let
    it out of `take_action` -- which must be total over its action space, since a
    `Method` may emit any vector in the Box.

    The class is imported from upstream rather than restated, because the property under
    test is precisely its base class: a locally-defined stand-alone `BaseException` would
    pass this test while proving nothing about what upstream actually raises."""
    from bilevel_planning.trajectory_samplers.trajectory_sampler import (
        TrajectorySamplingFailure,
    )

    # The premise, asserted rather than assumed: if upstream ever makes this an ordinary
    # Exception then the reasoning above stops applying, and this should say so loudly
    # rather than continuing to pass for a reason that no longer holds.
    assert not issubclass(TrajectorySamplingFailure, Exception)

    backend = _backend_whose_controller_raises(
        monkeypatch=monkeypatch, error=TrajectorySamplingFailure("no sample worked")
    )

    run = backend.run_move_to_toss_location_and_toss(
        distance=1.35, rotation=0.0, release_speed_deg_s=140.0, gripper_release_ms=792.0
    )

    assert run.error is not None
    assert "TrajectorySamplingFailure" in run.error
    assert "no sample worked" in run.error
    assert run.terminated is False
    assert run.steps == 0


def test_a_planner_assertion_is_reported_the_same_way(*, monkeypatch: pytest.MonkeyPatch) -> None:
    """The other kind, and the reason the handler cannot simply be narrowed to
    `TrajectorySamplingFailure`: KINDER's own motion planners still `assert plan is not
    None`, so an unreachable grasp arrives as an `AssertionError`. Both kinds are ordinary
    outcomes of a skill whose parameters do not work out."""
    backend = _backend_whose_controller_raises(
        monkeypatch=monkeypatch, error=AssertionError("plan is not None")
    )

    run = backend.run_move_to_toss_location_and_toss(
        distance=1.35, rotation=0.0, release_speed_deg_s=140.0, gripper_release_ms=792.0
    )

    assert run.error is not None
    assert "AssertionError" in run.error


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt(), SystemExit()])
def test_an_interrupt_still_escapes_rather_than_being_swallowed_as_a_failed_skill(
    *, monkeypatch: pytest.MonkeyPatch, interrupt: BaseException
) -> None:
    """`BaseException` is wide enough to catch Ctrl-C and a `sys.exit`, and reporting
    either as "the skill did not work out" would make a long sweep impossible to stop.
    The two are re-raised explicitly, which is the cost of catching that widely."""
    backend = _backend_whose_controller_raises(monkeypatch=monkeypatch, error=interrupt)

    with pytest.raises(type(interrupt)):
        backend.run_move_to_toss_location_and_toss(
            distance=1.35, rotation=0.0, release_speed_deg_s=140.0, gripper_release_ms=792.0
        )


def test_the_laziness_guard_itself_trips_on_a_real_import(*, no_kinder_import) -> None:
    """A guard that can only ever pass proves nothing, and this one replaced an assertion
    that had exactly that failure mode -- `"mujoco" not in sys.modules` is trivially true
    for a process that never touches KINDER, whatever the code under test does.

    `KinderBackend.api()` is the single door: all five KINDER imports in this package sit
    inside it. Calling it is therefore the one thing the guard must catch."""
    del no_kinder_import

    with pytest.raises(AssertionError, match="supposed to stay lazy"):
        KinderBackend().api()

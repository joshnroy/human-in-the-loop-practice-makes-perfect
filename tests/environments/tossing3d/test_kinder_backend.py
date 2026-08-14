"""Offline tests for `KinderBackend`'s sub-step recording switch and GL backend.

Everything here runs without MuJoCo, which is the point: the switch and the drain are
plain Python, and CI -- which never installs the optional extra -- is exactly where a
wiring regression would otherwise go unnoticed. Whether the wrapper really collects one
frame per physics tick is a question only a simulator can answer, and lives in
`test_kinder_fidelity.py`.
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

    **In a subprocess, because an in-process `sys.modules` check stopped meaning this.**
    The predicates are upstream's now, so `test_kb_predicate_parity.py` genuinely imports
    `kinder_models` -- and therefore MuJoCo -- inside its own tests. It sorts before this
    file, so a same-interpreter assertion here was reading *another module's* import and
    failing for a reason that has nothing to do with `KinderBackend`. A fresh interpreter
    asks the question this test is actually about.
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


def test_move_to_throw_pose_disables_collision_against_the_held_cube(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`run_move_to_throw_pose`'s own docstring claims
    `disable_collision_objects=["cube_0"]` is passed to upstream's `move_to_target` --
    the robot is holding the cube at this point, so treating it as a base-motion obstacle
    makes every plan fail. It wasn't actually threaded through: `run_controller` has a
    real `disable_collision_objects` parameter, and `run_move_to_throw_pose` never
    supplied it, leaving it `None`.

    This is entirely offline: no simulator needed, since the bug is about which kwarg
    reaches `run_controller`, not about simulator behaviour.

    That upstream fix has now landed and the pin has bumped, so this is no longer the
    dormant guard it was written as. `reference/kinder-baselines` used to be pinned at
    `11eace5`, where `run_base_motion_planning` hardcoded `obstacle_geoms` empty
    (Princeton-Robot-Planning-and-Learning/kinder-baselines#102) and this kwarg therefore
    had zero observable effect. The pin is now `1b564a1`, which contains upstream PR #103
    -- the fix for that issue -- so collision-checking against scene obstacles is **on**.
    The held cube is a `MujocoObjectType` like any other, and `run_base_motion_planning`
    filters `disable_collision_objects` out of the obstacle set *before* looking up each
    remaining object's geom, so passing `["cube_0"]` is what keeps the robot's own held
    cube from becoming an unexcluded obstacle to its own base-motion plan.
    """
    calls: list[dict[str, Any]] = []

    # `self` must stay positional -- `monkeypatch.setattr` installs this as an unbound
    # class method, and Python's bound-method call convention always passes the instance
    # positionally, the same reasoning `core/README.md` gives for `__getitem__`.
    def spy_run_controller(  # noqa: PLR0917
        self: KinderBackend, *, module: str, key: str, **kwargs: Any
    ) -> ControllerRun:
        calls.append({"module": module, "key": key, **kwargs})
        return ControllerRun(steps=1, terminated=True)

    monkeypatch.setattr(KinderBackend, "run_controller", spy_run_controller)

    backend = KinderBackend()
    # `robot_name` is a property that reads a scene-derived private attribute and raises
    # before any `reset()`; `run_move_to_throw_pose` only needs a name to pass through,
    # not a real one, so a `PrivateAttr` write stands in for a scene without a simulator.
    backend._robot_name = "robot_test"  # noqa: SLF001

    backend.run_move_to_throw_pose(standoff=1.35, rotation=0.0)

    assert len(calls) == 1
    assert calls[0]["disable_collision_objects"] == [backend.cube_name]


def test_run_toss_converts_the_release_speed_to_radians_exactly_once(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This domain carries the dial in joint-path deg/s and `TossController.reset` takes
    rad/s, so exactly one site converts. A second or missing conversion is a silent 57x
    error either way. Offline: it spies on `run_controller`.
    """
    calls: list[dict[str, Any]] = []

    def spy_run_controller(  # noqa: PLR0917  (see the sibling test for why `self` is positional)
        self: KinderBackend, *, module: str, key: str, **kwargs: Any
    ) -> ControllerRun:
        calls.append({"module": module, "key": key, **kwargs})
        return ControllerRun(steps=1, terminated=True)

    monkeypatch.setattr(KinderBackend, "run_controller", spy_run_controller)

    backend = KinderBackend()
    backend._robot_name = "robot_test"  # noqa: SLF001

    backend.run_toss(release_speed_deg_s=140.0, gripper_release_ms=720.0)

    assert [call["key"] for call in calls] == ["move_arm_to_conf", "toss"]
    # `move_arm_to_conf.reset` declares no release speed; passing one is a TypeError.
    assert "release_speed" not in calls[0]
    assert calls[1]["release_speed"] == pytest.approx(np.deg2rad(140.0))


def test_run_toss_rounds_the_gripper_release_ms_to_an_int_rather_than_truncating(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upstream `divmod`s `int(gripper_release_ms)`, and `int()` truncates toward zero, so
    a float handed straight through would make `722.9` mean 722 -- a systematic bias toward
    releasing early. Offline: it spies on `run_controller`.
    """
    calls: list[dict[str, Any]] = []

    def spy_run_controller(  # noqa: PLR0917  (see the sibling test for why `self` is positional)
        self: KinderBackend, *, module: str, key: str, **kwargs: Any
    ) -> ControllerRun:
        calls.append({"module": module, "key": key, **kwargs})
        return ControllerRun(steps=1, terminated=True)

    monkeypatch.setattr(KinderBackend, "run_controller", spy_run_controller)

    backend = KinderBackend()
    backend._robot_name = "robot_test"  # noqa: SLF001

    backend.run_toss(release_speed_deg_s=140.0, gripper_release_ms=722.9)

    assert "gripper_release_ms" not in calls[0]
    scheduled = calls[1]["gripper_release_ms"]
    assert isinstance(scheduled, int)
    assert scheduled == 723


def test_run_toss_skips_the_swing_when_the_windup_fails_whatever_the_speed(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tossing from an unknown arm pose is not what upstream measured, at any speed."""
    calls: list[str] = []

    def spy_run_controller(  # noqa: PLR0917
        self: KinderBackend, *, module: str, key: str, **kwargs: Any
    ) -> ControllerRun:
        del module, kwargs
        calls.append(key)
        return ControllerRun(steps=0, terminated=False, error="planning failed")

    monkeypatch.setattr(KinderBackend, "run_controller", spy_run_controller)

    backend = KinderBackend()
    backend._robot_name = "robot_test"  # noqa: SLF001

    windup, swing = backend.run_toss(release_speed_deg_s=140.0, gripper_release_ms=520.0)

    assert calls == ["move_arm_to_conf"]
    assert windup.terminated is False
    assert swing.error == "windup did not terminate"

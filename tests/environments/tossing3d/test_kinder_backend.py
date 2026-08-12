"""Offline tests for `KinderBackend`'s sub-step recording switch.

Everything here runs without MuJoCo, which is the point: the switch and the drain are
plain Python, and CI -- which never installs the optional extra -- is exactly where a
wiring regression would otherwise go unnoticed. Whether the wrapper really collects one
frame per physics tick is a question only a simulator can answer, and lives in
`test_kinder_fidelity.py`.
"""

import sys
from typing import Any

import numpy as np
import pytest

from hitl_pmp.environments.tossing3d.kinder_backend import ControllerRun, KinderBackend


def test_substep_recording_is_off_by_default() -> None:
    """One skill here is hundreds of MuJoCo ticks, and recording renders every one of
    them. A training run that never asked for a video must not pay for one, so the
    default is the cheap behaviour and the demo path opts in."""
    assert KinderBackend().record_substeps is False


def test_toggling_substep_recording_imports_no_simulator() -> None:
    """`KinderBackend` is lazy by construction and the switch must not break that: it is
    flipped by `Tossing3DProblem.run_task_episode`, which a test may reach without ever
    intending to build a scene."""
    backend = KinderBackend()
    backend.set_substep_recording(enabled=True)

    assert backend.record_substeps is True
    assert "mujoco" not in sys.modules


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
    """**The one place degrees become radians.**

    This domain carries the dial in joint-path deg/s, because that is the unit every
    measurement of it is written in (#213's fit, #221's grid, upstream's own `140`
    literal, and the real TidyBot primitive). Upstream's `TossController.reset` takes
    rad/s. So exactly one site converts, and this test is what pins it: a second
    conversion anywhere upstream of here would drive the arm at 1/57th of the commanded
    speed, and a *missing* one would drive it at 57x -- neither of which raises, and both
    of which would silently invalidate every number measured afterwards.

    Offline: it spies on `run_controller`, so no simulator and no controller is involved.
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

    backend.run_toss(release_speed_deg_s=140.0)

    assert [call["key"] for call in calls] == ["move_arm_to_conf", "toss"]
    # The windup is `move_arm_to_conf`, whose `reset` takes no release speed at all --
    # passing one is a TypeError, so it must not be forwarded there.
    assert "release_speed" not in calls[0]
    assert calls[1]["release_speed"] == pytest.approx(np.deg2rad(140.0))


def test_run_toss_skips_the_swing_when_the_windup_fails_whatever_the_speed(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adding a parameter must not change the windup-failed short circuit: tossing from
    an unknown arm pose is not the thing upstream measured, at any release speed."""
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

    windup, swing = backend.run_toss(release_speed_deg_s=240.0)

    assert calls == ["move_arm_to_conf"]
    assert windup.terminated is False
    assert swing.error == "windup did not terminate"

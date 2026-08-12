"""Offline tests for `KinderBackend`'s sub-step recording switch.

Everything here runs without MuJoCo, which is the point: the switch and the drain are
plain Python, and CI -- which never installs the optional extra -- is exactly where a
wiring regression would otherwise go unnoticed. Whether the wrapper really collects one
frame per physics tick is a question only a simulator can answer, and lives in
`test_kinder_fidelity.py`.
"""

import sys
from typing import Any

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
    had zero observable effect. The pin is now `3524010`, which contains upstream PR #103
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

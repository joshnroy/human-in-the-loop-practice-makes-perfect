"""Offline tests for `KinderBackend`'s sub-step recording switch.

Everything here runs without MuJoCo, which is the point: the switch and the drain are
plain Python, and CI -- which never installs the optional extra -- is exactly where a
wiring regression would otherwise go unnoticed. Whether the wrapper really collects one
frame per physics tick is a question only a simulator can answer, and lives in
`test_kinder_fidelity.py`.
"""

import sys

from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend


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

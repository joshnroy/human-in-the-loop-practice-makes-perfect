"""Offline tests for `KinderBackend`'s recording switch and its controller dispatch.

Everything here runs without MuJoCo, and that is a deliberate exception to this package's
otherwise simulator-only rule. The rest of the domain went simulator-backed because its
predicates genuinely need a `PyBulletSim` to answer at all (see `conftest.py`). Nothing in
this file is like that: the recording switch is a boolean and a wrapper swap, and
`run_skill` is a dispatch that forwards keyword arguments. Both are plain Python, and a
wiring regression in either would be invisible to a simulator test, which would see only
the end result.

What a simulator has to answer instead -- whether the wrapper really collects one frame
per physics tick -- lives in `test_kinder_fidelity.py`.

## What used to be here and is not

Four tests spied on `run_controller` and the three skill-specific wrappers built on it
(`run_pick`, `run_move_to_throw_pose`, `run_toss`). All four are gone, because all four
methods are gone: `KinderBackend` now exposes one generic `run_skill`, and the arguments
those tests pinned are no longer this repo's to get right.

* the deg/s -> rad/s conversion and the gripper-release rounding were ours while this repo
  declared its own toss dials; the fused controller's own sampler draws both now, in the
  controller's own units, so there is no conversion site left to test;
* the windup/swing sequencing was ours while `Toss` was two upstream controllers driven
  back to back; it is one controller now;
* `disable_collision_objects=["cube_0"]` was ours to thread through; upstream's
  `_plan_base_motion` now defaults it to the held object.

The last one is a real behaviour that still has to hold, so it did not simply evaporate --
it moved to `test_kinder_pin.py`, which is where "upstream still does this for us" belongs.
"""

import sys
from typing import Any

import numpy as np
import pytest

from hitl_pmp.adapters.kinder.types import ControllerRun
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


class _SpyControllers:
    """Stands in for `KinderControllers`, recording the one `run` call it receives.

    Not a `KinderControllers` subclass: constructing one needs a `KinderStateTranslator`,
    which needs a live KINDER state, which is the simulator this file exists to avoid.
    `KinderBackend.controllers()` is monkeypatched to hand this back instead, so nothing
    type-checks it at runtime.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(self, **kwargs: Any) -> ControllerRun:
        self.calls.append(kwargs)
        return ControllerRun(steps=7, terminated=True)


def _spying_backend(*, monkeypatch: pytest.MonkeyPatch) -> tuple[KinderBackend, _SpyControllers]:
    spy = _SpyControllers()
    monkeypatch.setattr(KinderBackend, "controllers", lambda self: spy)
    return KinderBackend(), spy


def test_run_skill_forwards_its_key_objects_params_and_limit_verbatim(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`run_skill` is the one seam between this domain and the generic bridge, so it is
    where a transposed argument would land. Every value is distinct here, so swapping any
    two of them fails rather than coincidentally passing."""
    backend, spy = _spying_backend(monkeypatch=monkeypatch)
    params = np.array([1.1, 2.2, 3.3, 4.4])

    run = backend.run_skill(
        key="move_to_toss_location_and_toss",
        object_names=("robot_test", "cube_0", "cuboid_barrier"),
        params=params,
        state="the-core-state",
        limit=123,
    )

    (call,) = spy.calls
    assert call["key"] == "move_to_toss_location_and_toss"
    assert call["object_names"] == ("robot_test", "cube_0", "cuboid_barrier")
    assert call["params"] is params
    assert call["state"] == "the-core-state"
    assert call["limit"] == 123
    assert run.steps == 7


def test_run_skill_hands_the_bridge_a_step_that_advances_the_simulator(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`KinderControllers.run` owns the loop but not the simulator, so the step callable
    is what actually moves the world -- and it must also keep `KinderBackend`'s own cached
    state in step, or `kinder_state()` would report the pose the skill started from for
    the rest of the episode.
    """
    backend, spy = _spying_backend(monkeypatch=monkeypatch)

    # Both signatures are positional because gymnasium's are: `env.step(action)` and
    # `observation_space.devectorize(observation)` are third-party contracts, which is the
    # documented exemption from this repo's keyword-only rule.
    class _FakeSpace:
        @staticmethod
        def devectorize(observation: Any) -> str:  # noqa: PLR0917
            return f"devectorized:{observation}"

    class _FakeEnv:
        observation_space = _FakeSpace()

        @staticmethod
        def step(action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:  # noqa: PLR0917
            return f"observation-for-{action.tolist()}", 0.0, False, False, {}

    backend._env = _FakeEnv()  # noqa: SLF001

    backend.run_skill(
        key="pick_cube",
        object_names=("robot_test", "cube_0", "cuboid_barrier"),
        params=np.array([0.5, 0.0]),
        state="the-core-state",
        limit=1,
    )

    (call,) = spy.calls
    reached = call["step"](np.array([9.0]))

    assert reached == "devectorized:observation-for-[9.0]"
    assert backend.kinder_state() == reached

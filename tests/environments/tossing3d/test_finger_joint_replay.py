"""Replays of recorded Tossing3D states restore the gripper fingers.

A KINDER `ObjectCentricState` carries one gripper position and one gripper velocity,
and upstream's `set_state` writes them to the gripper actuator's control and to the
two driver joints' velocities. The eight finger joints' own positions and velocities
are written by nothing, so a restore keeps whatever the previous rollout left in them:
open after one replay, closed around nothing after another. Two replays of one recorded
state then diverged from the first control step.

The recording now carries those sixteen numbers beside the objects, and the replay
restore writes them back. In-run restores do not: changing them would change the
dynamics every run is measured under.
"""

import argparse
import importlib.util
import json
import logging
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.environments.tossing3d.cli import Tossing3DCli
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend
from hitl_pmp.environments.tossing3d.layout import Tossing3DLayout
from hitl_pmp.environments.tossing3d.recording import (
    Tossing3DRecording,
    Tossing3DReplayRenderer,
)
from hitl_pmp.recording.run_record.reader import RunRecordReader
from hitl_pmp.recording.run_record.types import ActionEntry, Phase, TickEntry

HAS_KINDER = (
    importlib.util.find_spec("kinder") is not None
    and importlib.util.find_spec("kinder_models") is not None
)
pytestmark = pytest.mark.skipif(not HAS_KINDER, reason="needs the KINDER simulator")

SCENE = {
    "layout": Tossing3DLayout.BARRIER,
    "variant": "o1",
    "scene_bg": False,
    "canonical_seed": 125,
    "seed": 0,
    "test_env_seed_offset": 10000,
}
KEY = KinderBackend.finger_joints_key
FINGER_JOINTS = (
    "robot_right_driver_joint",
    "robot_right_coupler_joint",
    "robot_right_spring_link_joint",
    "robot_right_follower_joint",
    "robot_left_driver_joint",
    "robot_left_coupler_joint",
    "robot_left_spring_link_joint",
    "robot_left_follower_joint",
)
OLD_STATE = Path(__file__).parent / "fixtures/pick_dragged_bin_state.json"


def _started_env() -> Tossing3DEnvironment:
    problem = Tossing3DCli.build_problem(args=argparse.Namespace(**SCENE))
    problem.hard_reset()
    problem.reset_to_task(task=problem.sample_train_task())
    return problem.env


def _pick(*, env: Tossing3DEnvironment) -> np.ndarray:
    return np.array([float(env.pick_cube_id), 0.0, 0.0, 0.0, 0.0])


def test_every_recorded_state_carries_the_finger_joints(*, tmp_path: Path) -> None:
    """Dispatch states (a segment's start, a skill's settled end) and per-tick states
    alike, and named in the header so a reader can tell qpos from qvel."""
    path = tmp_path / "recording.jsonl"
    args = argparse.Namespace(**SCENE)
    problem = Tossing3DCli.build_problem(args=args)
    env = problem.env
    writer = Tossing3DRecording.writer(env=env, args=args, method="test", output_path=path)
    env.attach_state_log_writer(writer=writer)
    try:
        problem.hard_reset()
        problem.reset_to_task(task=problem.sample_train_task())
        writer.begin_segment(phase=Phase.PRACTICE)
        problem.take_action(action=_pick(env=env))
        writer.robot_action(
            label="PickCube(robot, cube_0, barrier_0)",
            start_step=0,
            end_step=1,
            success=env.last_skill_error() is None,
        )
        live_fingers = env.kinder_backend().finger_joint_plain()[KEY]
    finally:
        writer.close()
        env.close()

    header = RunRecordReader.header(path=path)
    assert header.feature_names[KEY] == [f"qpos:{j}" for j in FINGER_JOINTS] + [
        f"qvel:{j}" for j in FINGER_JOINTS
    ]
    entries = list(RunRecordReader.entries(path=path))
    ticks = [e for e in entries if isinstance(e, TickEntry)]
    robot = next(e for e in entries if isinstance(e, ActionEntry))
    assert robot.num_ticks > 50
    assert all(len(t.state[KEY]) == 2 * len(FINGER_JOINTS) for t in ticks)
    first, last = ticks[0].state[KEY], ticks[-1].state[KEY]
    assert last == live_fingers
    # The pick closes the gripper: the recorded finger positions move.
    assert max(abs(a - b) for a, b in zip(first[:8], last[:8], strict=True)) > 0.1


def test_reading_the_finger_joints_changes_nothing_in_the_simulator() -> None:
    """Recording must not perturb a run: the read is a copy of MuJoCo's arrays."""
    import mujoco

    env = _started_env()
    try:
        env.take_action(action=_pick(env=env))
        sim = env.kinder_backend()._object_centric()._robot_env.sim  # noqa: SLF001
        model, data = sim.model.mj_model, sim.data.mj_data
        spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        before = np.empty(mujoco.mj_stateSize(model, spec))
        mujoco.mj_getState(model, data, before, spec)
        Tossing3DRecording.state(env=env)
        env.kinder_backend().finger_joint_plain()
        after = np.empty_like(before)
        mujoco.mj_getState(model, data, after, spec)
        assert before.tobytes() == after.tobytes()
    finally:
        env.close()


def test_replaying_one_recorded_state_twice_gives_bit_identical_traces() -> None:
    """The finding this exists for, with a pick as the intervening skill that leaves
    the fingers closed: the second replay starts from a recorded state whose fingers
    are open, in a simulator whose fingers are closed."""
    env = _started_env()
    backend = env.kinder_backend()
    trace: list[str] = []

    def observe() -> None:
        state = backend.snapshot_to_plain(snapshot=backend.live_state())
        state.update(backend.finger_joint_plain())
        trace.append(json.dumps(state))

    backend.set_control_step_observers(before=None, after=observe)
    try:
        recorded = Tossing3DRecording.state(env=env)
        traces = []
        for _ in range(2):
            live = backend.finger_joint_plain()[KEY]
            if traces:
                # Not vacuous: the first replay left the fingers somewhere else.
                assert max(abs(a - b) for a, b in zip(live, recorded[KEY], strict=True)) > 0.1
            env.restore_plain_snapshot(plain=recorded, dtype=np.float64, replay=True)
            assert backend.finger_joint_plain()[KEY] == recorded[KEY]
            trace.clear()
            env.take_action(action=_pick(env=env))
            traces.append(list(trace))
        assert len(traces[0]) > 50
        assert traces[0] == traces[1]
    finally:
        env.close()


def test_an_old_recording_without_finger_joints_still_replays(
    *, caplog: pytest.LogCaptureFixture
) -> None:
    """A state recorded before this change restores as it always did -- the fingers
    are left alone -- and says so once, rather than once per frame."""
    plain = json.loads(OLD_STATE.read_text())
    assert KEY not in plain
    renderer = Tossing3DReplayRenderer(scene=SCENE, scene_bg=False)
    try:
        backend = renderer._env().kinder_backend()  # noqa: SLF001
        fingers = backend.finger_joint_plain()[KEY]
        with caplog.at_level(logging.WARNING):
            for _ in range(3):
                frame = renderer.show(state=plain)
        assert frame.size > 0
        # Positions untouched; upstream's own restore still writes the two driver
        # joints' velocities from the state's gripper velocity, as it always has.
        assert backend.finger_joint_plain()[KEY][:8] == fingers[:8]
        shown = renderer.current_state()
        assert shown["cube_0"] == pytest.approx(plain["cube_0"])
        warnings = [r for r in caplog.records if "finger" in r.getMessage()]
        assert len(warnings) == 1
    finally:
        renderer.close()


def test_an_in_run_restore_leaves_the_finger_joints_alone() -> None:
    """The default restore is unchanged: only a replay writes the fingers back."""
    env = _started_env()
    backend = env.kinder_backend()
    try:
        recorded = Tossing3DRecording.state(env=env)
        env.take_action(action=_pick(env=env))
        closed = backend.finger_joint_plain()[KEY]
        env.restore_plain_snapshot(plain=recorded, dtype=np.float64)
        assert backend.finger_joint_plain()[KEY][:8] == closed[:8]
    finally:
        env.close()

"""Native round-trip recovery and interruption invariants for the registered domain."""

import importlib.util

import numpy as np
import pytest

from hitl_pmp.environments.pouring3d.practice.environment import FrozenPourEnvironment
from hitl_pmp.environments.pouring3d.practice.skill_provider import FrozenPourSkillProvider

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="KinDER required"
)


def test_native_pour_robot_reset_and_human_invariants():
    env = FrozenPourEnvironment(
        layout="floor", canonical_seed=70, requested_count=1, scene_bg=False
    )
    provider = FrozenPourSkillProvider(env=env)

    def run(*, name, shrimp=0, params=()):
        state = env.get_current_state()
        ground = provider.ground(name=name, shrimp=shrimp, request=2)
        env.take_action(
            action=provider.compute_action(
                ground_skill=ground, params=np.array(params), state=state
            )
        )
        assert env.last_info["success"], env.last_info

    try:
        run(name="PickPitcher")
        run(name="Pour2", params=(90.5, 0.008, 30))
        assert env.last_info["requested_count"] == 2
        run(name="PlacePitcherAtHome")
        # Practice selected Pour(2), despite the task distribution's reference
        # goal being one. This is skill success, but cannot pass a count-one test.
        from hitl_pmp.environments.pouring3d.practice.tasks import FrozenPourTasks

        state = env.get_current_state()
        tasks = FrozenPourTasks(env=env)
        assert tasks.goal(requested=2).is_satisfied(state=state)
        assert not tasks.goal(requested=1).is_satisfied(state=state)
        assert env.requested_count == env.controller().scene.requested_count == 1
        c = env.controller()
        selected = c.scene.cube_locations().index("tray")
        run(name="PickShrimpFromTray", shrimp=selected)
        run(name="DropShrimpIntoPitcher", shrimp=selected)
        assert c.scene.cube_locations()[selected] == "pitcher"
        before = c.data.qpos.copy()
        time = float(c.data.time)
        steps = c.steps
        assert env.reset_movables(destination="reset_shrimp")
        # Only shrimp free joints may change; human reset does not move the robot/vessels.
        changed = set()
        for name in c.scene.cube_names:
            joint = c.model.body_jntadr[c.model.body(name).id]
            address = c.model.jnt_qposadr[joint]
            changed.update(range(address, address + 7))
        preserved = [i for i in range(c.model.nq) if i not in changed]
        np.testing.assert_array_equal(before[preserved], c.data.qpos[preserved])
        assert c.data.time == time and c.steps == steps
        assert c.ready()
    finally:
        env.close()


def test_step_interrupt_adopts_actual_state_without_reset():
    from hitl_pmp.core.control_steps import ControlStepLimitReached
    from hitl_pmp.environments.pouring3d.practice.types import PourObjects

    env = FrozenPourEnvironment(layout="floor", scene_bg=False)
    provider = FrozenPourSkillProvider(env=env)
    observed = []

    def before():
        if len(observed) == 3:
            raise ControlStepLimitReached("test budget")

    try:
        env.hard_reset()
        c = env.controller()
        env.set_control_step_observers(
            before=before, after=lambda: observed.append(c.data.qpos.copy())
        )
        action = provider.encode(
            ground_skill=provider.ground(name="PickPitcher"), params=np.zeros(0)
        )
        with pytest.raises(ControlStepLimitReached):
            env.take_action(action=action)
        actual = c.data.qpos.copy()
        state = env.adopt_interrupted_skill(action=action, reason="test budget")
        np.testing.assert_array_equal(actual, c.data.qpos)
        assert c.steps == 3 and len(observed) == 3
        assert state.get(obj=PourObjects.scene, feature_name="steps_taken") == 1
        assert "interrupted" in env.last_info["error"]
        with pytest.raises(ValueError):
            env.set_state(state=state)
    finally:
        env.close()


def test_native_floor_recovery():
    import mujoco

    env = FrozenPourEnvironment(layout="floor", scene_bg=False)
    provider = FrozenPourSkillProvider(env=env)
    try:
        env.hard_reset()
        c = env.controller()
        # Privileged test fixture: a settled spilled piece on open ground.
        shrimp = c.scene.get_object("shrimp_0")
        shrimp.set_pose([0.35, 0.2, 0.0055], [1, 0, 0, 0])
        shrimp.set_velocity([0, 0, 0], [0, 0, 0])
        mujoco.mj_forward(c.model, c.data)
        env.current_state = env.observe()
        for name in ("PickShrimpFromFloor", "DropShrimpIntoPitcher"):
            ground = provider.ground(name=name, shrimp=0)
            env.take_action(
                action=provider.compute_action(
                    ground_skill=ground, params=np.zeros(0), state=env.get_current_state()
                )
            )
            assert env.last_info["success"], env.last_info
        assert c.ready()
    finally:
        env.close()


@pytest.mark.parametrize("layout", ["floor", "stations"])
def test_motion_recording_groups_ticks_with_skill_and_replays_exact_state(*, tmp_path, layout):
    import argparse
    import json

    from hitl_pmp.environments.pouring3d.practice.problem import FrozenPourProblem
    from hitl_pmp.environments.pouring3d.practice.recording import FrozenPourReplayRenderer
    from hitl_pmp.environments.pouring3d.practice.step_protocol import FrozenPourStepProtocol
    from hitl_pmp.environments.pouring3d.practice.tasks import FrozenPourTasks
    from hitl_pmp.recording.run_record.types import Phase

    env = FrozenPourEnvironment(layout=layout, scene_bg=False)
    problem = FrozenPourProblem(env=env, tasks=FrozenPourTasks(env=env))
    path = tmp_path / "recording.jsonl"
    writer = FrozenPourStepProtocol.attach_recording(
        args=argparse.Namespace(record_motion="all", method="test", seed=0),
        problem=problem,
        output_path=path,
    )
    replay = FrozenPourReplayRenderer(
        scene={
            "payload_count": 3,
            "requested_count": 2,
            "surface_mu": 0.1,
            "receiving_vessel": "skillet",
            "layout": layout,
            "motion": True,
        },
        scene_bg=False,
        width=320,
        height=240,
    )
    try:
        writer.begin_segment(phase=Phase.PRACTICE)
        env.take_action(action=env.noop_action())
        writer.robot_action(label="Wait()", start_step=0, end_step=10, success=True)
        writer.close()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        action_index = next(i for i, row in enumerate(rows) if row["kind"] == "action")
        ticks = [row for row in rows[action_index + 1 :] if row["kind"] == "tick"]
        assert len(ticks) >= 10
        assert all(row["seq"] == rows[action_index]["seq"] for row in ticks)
        live = FrozenPourStepProtocol.live_state(env=env)
        assert ticks[-1]["state"] == live
        assert replay.show(state=live).shape == (240, 320, 3)
        assert replay.current_state() == live
    finally:
        writer.close()
        replay.close()
        env.close()

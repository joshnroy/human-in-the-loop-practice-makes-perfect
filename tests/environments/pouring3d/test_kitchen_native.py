"""Native observation/reset boundaries with explicitly arranged diagnostic fixtures."""

import numpy as np
import pytest

pytest.importorskip("kinder")

from hitl_pmp.environments.pouring3d.kitchen.environment import HUMAN_RESET, KitchenEnvironment


def test_the_kitchen_pour_action_carries_only_a_ramp_rate():
    from hitl_pmp.environments.pouring3d.feedback_controller import FeedbackPourParameters

    env = KitchenEnvironment(requested_count=1)
    try:
        c = env.controller()
        for action, match in (
            # The open-loop (tilt, amplitude, duration) action is no longer a kitchen Pour.
            ([1, 0, 1, 86.0, 0.06, 30.0], "slots 4 and 5"),
            ([1, 0, 1, 1.0, 0.04, 0.0], "slots 4 and 5"),
            ([1, 0, 1, 0.1, 0.0, 0.0], "ramp_rate_deg_s"),
        ):
            steps = c.steps
            env.take_action(action=np.array(action, dtype=float))
            assert env.last_info["success"] is False
            assert match in env.last_info["error"]
            assert c.steps == steps
        with pytest.raises(ValueError, match="closed-loop"):
            c.run_joint_pour(parameters=FeedbackPourParameters(), requested_count=1)
    finally:
        env.close()


def test_loaded_jug_release_requires_low_upright_slow_pose_above_home():
    import mujoco

    env = KitchenEnvironment()
    try:
        c = env.controller()
        jug = c.scene.get_object("pitcher")
        c.held = "pitcher"
        # The recorded loaded-jug seating edge case: about 7 mm off center,
        # 9.5 mm above support. Reject independently unsafe deviations.
        cases = [
            ([0.0073, 0, 0.0095], [1, 0, 0, 0], [0] * 6, True),
            ([0.010, 0, 0.0095], [1, 0, 0, 0], [0] * 6, False),
            ([0, 0, 0.021], [1, 0, 0, 0], [0] * 6, False),
            ([0, 0, -0.001], [1, 0, 0, 0], [0] * 6, False),
            ([0, 0, 0.01], [np.cos(0.04), np.sin(0.04), 0, 0], [0] * 6, False),
            ([0, 0, 0.01], [1, 0, 0, 0], [0.06, 0, 0, 0, 0, 0], False),
            ([0, 0, 0.01], [1, 0, 0, 0], [0, 0, 0, 0.11, 0, 0], False),
        ]
        for offset, quaternion, velocity, allowed in cases:
            jug.set_pose(c.home + offset, quaternion)
            jug.set_velocity(velocity[:3], velocity[3:])
            mujoco.mj_forward(c.model, c.data)
            assert c.can_release_pitcher_over_home() == allowed
        c.held = None
        assert not c.can_release_pitcher_over_home()
    finally:
        env.close()


def test_partial_spill_failure_and_costed_human_reset_preserve_robot_and_vessels():
    import mujoco

    env = KitchenEnvironment(requested_count=3, human_interruption_cost=7)
    try:
        c = env.controller()
        initial_state = env.observe()
        piece = c.scene.get_object(c.scene.cube_names[0])
        pan = c.data.xpos[c.model.body("tray").id].copy()
        piece.set_pose(pan + [0, 0, 0.009], [1, 0, 0, 0])
        piece.set_velocity([0.0] * 3, [0.0] * 3)
        mujoco.mj_forward(c.model, c.data)
        # A split is judged settled by position over a window: watch the placed piece
        # come to rest (it tumbles about 2 cm first) before reading Split1.
        for _ in range(12):
            c.pause(phase="observe rest", seconds=1.0)
            if c.outcome_settled():
                break
        atoms = env.observe().abstract_atoms
        assert ("InTray", ("potsticker_0", "tray")) in atoms
        assert ("Split1", ()) in atoms
        assert ("Split3", ()) not in atoms
        piece.set_pose([0.35, 0.2, 0.002], [1, 0, 0, 0])
        mujoco.mj_forward(c.model, c.data)
        assert ("OnFloor", ("potsticker_0",)) in env.observe().abstract_atoms
        before = c.data.qpos.copy()
        env.take_action(action=np.array([1, 0, 3, 1.0, 0, 0]))
        assert not env.last_info["success"]
        np.testing.assert_array_equal(before, c.data.qpos)
        assert c.scene.cube_locations()[0] == "floor"
        assert env.reset_movables(destination=HUMAN_RESET)
        assert env.human_interventions == 1
        assert env.interruption_cost_total == 7
        assert c.scene.cube_locations() == ["pitcher"] * 3
        # Bulk food reset leaves the robot and both vessels untouched.
        for name in ("pitcher", "tray"):
            body = c.model.body(name).id
            address = c.model.jnt_qposadr[c.model.body_jntadr[body]]
            np.testing.assert_array_equal(
                before[address : address + 7], c.data.qpos[address : address + 7]
            )
        np.testing.assert_array_equal(before[c.arm_q], c.data.qpos[c.arm_q])
        np.testing.assert_array_equal(before[c.base_q], c.data.qpos[c.base_q])
        with pytest.raises(ValueError, match="No free"):
            env.hard_reset()
        with pytest.raises(ValueError, match="No free"):
            env.set_state(state=initial_state)
        c.scene.set_cooking(enabled=True)
        with pytest.raises(ValueError, match="cold"):
            env.reset_movables(destination=HUMAN_RESET)
    finally:
        env.close()


def test_deployment_rejects_a_human_reset_and_a_premature_handoff_without_mutating_world():
    env = KitchenEnvironment(deployment=True, requested_count=3)
    try:
        c = env.controller()
        before = c.data.qpos.copy()
        with pytest.raises(ValueError, match="forbidden"):
            env.reset_movables(destination=HUMAN_RESET)
        np.testing.assert_array_equal(before, c.data.qpos)
        assert env.human_interventions == 0
        assert env.interruption_cost_total == 0
        with pytest.raises(ValueError, match="Handoff"):
            env.finish_human_handoff()
    finally:
        env.close()


def test_blocked_mobile_path_is_rejected_before_any_physics_step(*, monkeypatch):
    import mujoco

    env = KitchenEnvironment()
    try:
        c = env.controller()
        before, time = c.data.qpos.copy(), float(c.data.time)
        # Force a real obstacle-pair broadphase candidate and a blocked narrowphase.
        a, b = c.obstacle_pairs[0]
        c.obstacle_pairs = [(a, b)]
        c.model.geom_rbound[a] = 100.0
        monkeypatch.setattr(mujoco, "mj_geomDistance", lambda *args: -0.001)
        with pytest.raises(ValueError, match="Base path"):
            c.move_base(target=c.base_home + [-0.2, 0, 0], phase="test blocked path", seconds=1)
        np.testing.assert_array_equal(before, c.data.qpos)
        assert c.data.time == time
    finally:
        env.close()


@pytest.mark.parametrize("deployment", [False, True])
def test_recording_roundtrips_observed_hob_and_exact_native_poses(*, tmp_path, deployment):
    import argparse
    import json

    from hitl_pmp.environments.pouring3d.kitchen.recording import KitchenReplayRenderer
    from hitl_pmp.environments.pouring3d.kitchen.step_protocol import KitchenStepProtocol
    from hitl_pmp.environments.pouring3d.kitchen.tasks import KitchenProblem, KitchenTasks
    from hitl_pmp.recording.run_record.types import Phase

    env = KitchenEnvironment(deployment=deployment, scene_bg=False)
    replay, writer = None, None
    try:
        c = env.controller()
        # A diagnostic appliance fixture, not a claimed successful cooking action.
        c.scene.set_cooking(enabled=deployment)
        observed = env.observe()
        assert all(name != "WaterPresent" for name, _ in observed.abstract_atoms)
        from hitl_pmp.environments.pouring3d.kitchen.types import KitchenObjects

        assert observed.data[KitchenObjects.hob].shape == (2,)
        problem = KitchenProblem(env=env, tasks=KitchenTasks(env=env))
        path = tmp_path / "recording.jsonl"
        writer = KitchenStepProtocol.attach_recording(
            args=argparse.Namespace(record_motion="all", method="test", seed=0),
            problem=problem,
            output_path=path,
        )
        writer.begin_segment(phase=Phase.EVALUATION)
        state = KitchenStepProtocol.live_state(env=env)
        writer.record_tick(state=state)
        writer.close()
        header = json.loads(path.read_text().splitlines()[0])
        assert header["environment"] == "pouring3d_kitchen"
        assert header["scene"]["geometry"]["measured_jug_dimensions_m"] is None
        replay = KitchenReplayRenderer(scene=header["scene"], scene_bg=False, width=320, height=240)
        assert replay.show(state=state).shape == (240, 320, 3)
        assert replay.current_state() == state
        assert c.scene.water_present() == deployment
        assert replay._controller.scene.water_present() == deployment
        if deployment:
            water = c.model.geom("kitchen_water_proxy").id
            assert c.model.geom_contype[water] == 0
            assert all(b != water for _, b in c.pan_pairs)
    finally:
        if replay is not None:
            replay.close()
        if writer is not None:
            writer.close()
        env.close()


def test_recovery_branch_must_allow_lift_before_executing_grasp(*, monkeypatch):
    from scipy.spatial.transform import Rotation

    from hitl_pmp.environments.pouring3d.kitchen.native import KitchenController
    from hitl_pmp.environments.pouring3d.potsticker_controller import PotstickerController

    env = KitchenEnvironment()
    try:
        c = env.controller()
        q, base, phase = c.last_q.copy(), c.base_target.copy(), c.phase
        posture = c.posture
        before = c.data.qpos.copy()
        c.payload_approach_plan = [(base.copy(), q.copy(), "test grasp", 1.0)]
        monkeypatch.setattr(
            PotstickerController, "payload_approach_feasible", lambda self, **kwargs: True
        )

        def blocked(self, **kwargs):  # noqa: PLR0917
            self.posture = 99.0
            raise ValueError("No lift continuation")

        monkeypatch.setattr(KitchenController, "inverse_kinematics", blocked)
        assert not c.payload_approach_feasible(
            q=q, preparation=np.zeros(3), grasp=np.zeros(3), rotation=Rotation.identity(), base=base
        )
        assert not c.payload_approach_plan
        np.testing.assert_array_equal(c.last_q, q)
        np.testing.assert_array_equal(c.base_target, base)
        np.testing.assert_array_equal(c.data.qpos, before)
        assert c.phase == phase
        assert c.posture == posture
    finally:
        env.close()

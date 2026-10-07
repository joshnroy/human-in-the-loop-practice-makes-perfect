"""The generated controller gets bounded actuators and observations, not an oracle."""

import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from gymnasium.spaces import Box
from PIL import Image
from relational_structs import Object, ObjectCentricState, Type

from hitl_pmp.agentic_runtime.observations import ObservationEvidence
from hitl_pmp.environments.tossing3d.agentic_bridge import (
    AgenticTossing3DEnvironment,
    Tossing3DAgenticBridge,
)
from hitl_pmp.environments.tossing3d.kinder_backend import ControllerRun, KinderBackend


@pytest.fixture
def bridge(*, monkeypatch: pytest.MonkeyPatch) -> Tossing3DAgenticBridge:
    backend = KinderBackend()
    robot = SimpleNamespace(
        act_delta=True,
        control_frequency=10.0,
        qpos={"base": [0.0] * 3, "arm": [0.0] * 7, "gripper": [0.0]},
        qvel={"base": [0.0] * 3, "arm": [0.0] * 7, "gripper": [0.0]},
        ctrl={"gripper": [0.0]},
        sim=SimpleNamespace(
            model=SimpleNamespace(
                mj_model=SimpleNamespace(site=Mock(return_value=SimpleNamespace(id=0)))
            ),
            data=SimpleNamespace(
                mj_data=SimpleNamespace(
                    time=7.0,
                    site_xpos=np.array([[0.2, -0.3, 0.7]]),
                    site_xmat=np.array([np.eye(3).ravel()]),
                )
            ),
        ),
    )
    inner = SimpleNamespace(
        _robot_env=robot,
        render_all_cameras=Mock(
            return_value={
                "task_view_image": np.full((6, 8, 3), 123, dtype=np.uint8),
                "tidybot_wrist_image": np.full((6, 8, 3), 124, dtype=np.uint8),
            }
        ),
    )
    backend._raw_env = SimpleNamespace(unwrapped=SimpleNamespace(_object_centric_env=inner))
    state = Mock()
    actor = SimpleNamespace(name="tidybot")
    state.get_objects.return_value = [actor]
    state.type_features = {}
    state.get.side_effect = lambda obj, name: 0.25
    backend._state = state
    backend._robot_name = "tidybot"
    backend._env = Mock()
    backend._env.action_space = Box(
        np.array([-0.1] * 10 + [0.0] + [-np.inf] * 7),
        np.array([0.1] * 10 + [1.0] + [np.inf] * 7),
        dtype=np.float64,
    )
    backend._env.step.return_value = ("new-state", 1.0, True, False, {"solved": True})
    backend._env.observation_space.devectorize.return_value = state
    env = AgenticTossing3DEnvironment()
    env._backend = backend

    def tool_pose(  # noqa: PLR0917 -- stands in for a bound bridge method
        self: Tossing3DAgenticBridge, *, proprioception: dict[str, float]
    ) -> dict[str, float]:
        del self, proprioception
        robot.sim.model.mj_model.site("tidybot_pinch_site")
        return {
            **{
                f"pos_tool_{axis}": float(value)
                for axis, value in zip("xyz", robot.sim.data.mj_data.site_xpos[0], strict=True)
            },
            "quat_tool_w": 1.0,
            "quat_tool_x": 0.0,
            "quat_tool_y": 0.0,
            "quat_tool_z": 0.0,
        }

    monkeypatch.setattr(Tossing3DAgenticBridge, "_tool_pose", tool_pose)
    return Tossing3DAgenticBridge(env=env, step_limit=2)


class TestAgenticBridge:
    def test_only_camera_and_robot_proprioception_cross_boundary(
        self, *, bridge: Tossing3DAgenticBridge
    ) -> None:
        pixels = np.full((6, 8, 3), 123, dtype=np.uint8)
        observation = bridge.observe()
        assert set(observation) == {
            "observation_mode",
            "images",
            "image_views",
            "proprioception",
            "control_step",
            "simulation_time_s",
            "action_spec",
        }
        assert len(observation["images"]) == 2
        assert observation["simulation_time_s"] == 7.0
        assert observation["observation_mode"] == "rgb"
        assert observation["image_views"] == ["task_overview", "robot_wrist"]
        assert observation["action_spec"]["shape"] == [18]
        assert len(observation["proprioception"]) == 23
        assert not any("tool" in key for key in observation["proprioception"])
        bridge.env.backend()._object_centric()._robot_env.sim.model.mj_model.site.assert_not_called()
        assert "pos_arm_joint7" in observation["proprioception"]
        assert not {"reward", "solved", "atoms", "objects", "goal_region"} & observation.keys()
        encoded = observation["images"][0].split(",", 1)[1]
        decoded = np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))))
        assert np.array_equal(decoded, pixels)

    def test_object_state_is_labeled_numeric_and_does_not_render_or_advance(
        self, *, bridge: Tossing3DAgenticBridge
    ) -> None:
        cube_type = Type("mujoco_movable_object")
        cube = Object("cube", cube_type)
        state = ObjectCentricState(
            {cube: np.array([0.25, -0.5, 0.1, 0.04])},
            {cube_type: ["x", "y", "z", "bb_x"]},
        )
        backend = bridge.env.backend()
        backend._state = state
        bridge.observation_mode = "object_state"
        observation = bridge.observe()
        assert observation["objects"] == [
            {
                "name": "cube",
                "type": "mujoco_movable_object",
                "features": {"x": 0.25, "y": -0.5, "z": 0.1, "bb_x": 0.04},
            }
        ]
        assert observation["observation_mode"] == "object_state"
        assert not {"images", "reward", "success", "atoms", "goal_region"} & observation.keys()
        assert observation["state_spec"]["bounding_box"] == "full object-frame dimensions"
        assert "forward kinematics" in observation["state_spec"]["tool_pose"]
        expected_tool = {
            "pos_tool_x": 0.2,
            "pos_tool_y": -0.3,
            "pos_tool_z": 0.7,
            "quat_tool_w": 1.0,
            "quat_tool_x": 0.0,
            "quat_tool_y": 0.0,
            "quat_tool_z": 0.0,
        }
        for key, value in expected_tool.items():
            assert observation["proprioception"][key] == value
        assert np.isfinite(list(observation["proprioception"].values())).all()
        backend._object_centric().render_all_cameras.assert_not_called()
        backend._env.step.assert_not_called()
        backend._env.reset.assert_not_called()

    def test_distinct_execution_and_observer_bridges_share_physical_time(
        self, *, bridge: Tossing3DAgenticBridge, tmp_path: Path
    ) -> None:
        backend = bridge.env.backend()
        cube_type = Type("mujoco_movable_object")
        cube = Object("cube", cube_type)
        state = ObjectCentricState({cube: np.array([0.25])}, {cube_type: ["x"]})
        backend._state = state
        backend._env.observation_space.devectorize.return_value = state
        bridge.observation_mode = "object_state"
        executor_bridge = Tossing3DAgenticBridge(env=bridge.env, observation_mode="object_state")
        before = bridge.observe()
        clock = backend._object_centric()._robot_env.sim.data.mj_data

        def step(action: object) -> tuple:  # noqa: PLR0917 -- native gym step signature
            del action
            clock.time += 0.1
            return "updated-state", 0.0, False, False, {}

        backend._env.step.side_effect = step
        final = executor_bridge.step(action=[0.0] * 18)
        current = bridge.observe()
        assert before["simulation_time_s"] < final["simulation_time_s"]
        assert current["simulation_time_s"] == final["simulation_time_s"] == clock.time
        assert current["control_step"] == 0 and final["control_step"] == 1
        assert "bridge-local" in current["state_spec"]["control_step"].casefold()
        path = tmp_path / "trajectory.jsonl"
        path.write_text(json.dumps({"observation": final}) + "\n")
        evidence = ObservationEvidence.with_history(
            observation=current, trajectory_path=str(path), before=before
        )
        assert (
            evidence["recent_observations"][-1]["simulation_time_s"] == current["simulation_time_s"]
        )
        assert backend._env.step.call_count == 1
        backend._env.reset.assert_not_called()

    @pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), -0.1])
    def test_observation_rejects_invalid_physical_time(
        self, *, bridge: Tossing3DAgenticBridge, timestamp: float
    ) -> None:
        bridge.env.backend()._object_centric()._robot_env.sim.data.mj_data.time = timestamp
        with pytest.raises(ValueError, match="simulation time"):
            bridge.observe()
        bridge.env.backend()._env.step.assert_not_called()

    @pytest.mark.parametrize("feature,value", [("x", float("nan")), ("success", 1.0)])
    def test_object_state_refuses_nonphysical_or_nonfinite_features(
        self, *, bridge: Tossing3DAgenticBridge, feature: str, value: float
    ) -> None:
        cube_type = Type("mujoco_movable_object")
        cube = Object("cube", cube_type)
        bridge.env.backend()._state = ObjectCentricState(
            {cube: np.array([value])}, {cube_type: [feature]}
        )
        bridge.observation_mode = "object_state"
        with pytest.raises(ValueError):
            bridge.observe()
        bridge.env.backend()._env.step.assert_not_called()

    def test_object_robot_uses_measured_finger_position_not_gripper_command(
        self, *, bridge: Tossing3DAgenticBridge
    ) -> None:
        robot_type = Type("mujoco_tidybot_robot")
        robot_object = Object("tidybot", robot_type)
        backend = bridge.env.backend()
        backend._state = ObjectCentricState(
            {robot_object: np.array([1.0])}, {robot_type: ["pos_gripper"]}
        )
        robot = backend._object_centric()._robot_env
        robot.qpos["gripper"][0] = 0.3
        robot.ctrl["gripper"][0] = 255.0
        bridge.observation_mode = "object_state"
        observation = bridge.observe()
        features = observation["objects"][0]["features"]
        assert features["pos_gripper"] == 0.3
        assert features["gripper_command"] == 1.0
        assert features == observation["proprioception"]

    def test_object_tool_pose_rejects_nonfinite_robot_fk(
        self, *, bridge: Tossing3DAgenticBridge
    ) -> None:
        bridge.observation_mode = "object_state"
        robot = bridge.env.backend()._object_centric()._robot_env
        robot.sim.data.mj_data.site_xpos[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            bridge.observe()
        bridge.env.backend()._env.step.assert_not_called()

    @pytest.mark.parametrize(
        "action",
        [
            [0.0] * 17,
            [float("nan")] * 18,
            [0.0] * 11 + [13.0] * 7,
            [0.0] * 10 + [1.1] + [0.0] * 7,
            {"reset": True},
            {"values": [0.0] * 18, "reset": True},
            {"schedule": [[0.0] * 18] * 99},
        ],
    )
    def test_rejects_invalid_actions_before_simulator_step(
        self, *, bridge: Tossing3DAgenticBridge, action: object
    ) -> None:
        with pytest.raises((ValueError, TypeError)):
            bridge.step(action=action)
        bridge.env.backend()._env.step.assert_not_called()

    def test_step_discards_native_success_and_preserves_state(
        self, *, bridge: Tossing3DAgenticBridge
    ) -> None:
        observation = bridge.step(action=[0.0] * 18)
        assert observation["control_step"] == 1
        assert "solved" not in observation and "reward" not in observation
        backend = bridge.env.backend()
        backend._env.observation_space.devectorize.assert_called_once_with("new-state")
        assert bridge.control_steps == 1
        bridge.step(action={"schedule": [[0.0] * 18] * 100})
        with pytest.raises(RuntimeError, match="step budget"):
            bridge.step(action=[0.0] * 18)
        assert backend._env.step.call_count == 2

    def test_spec_is_finite_joint_space_not_parameterized_skill(
        self, *, bridge: Tossing3DAgenticBridge
    ) -> None:
        spec = bridge.action_spec()
        assert spec["shape"] == [18]
        assert np.isfinite(spec["low"]).all() and np.isfinite(spec["high"]).all()
        assert spec["schedule_rows"] == 100
        assert spec["gripper"] == {"index": 10, "open": 0.0, "closed": 1.0}
        assert spec["position_mode"] == "delta"

    def test_registered_policy_is_one_whole_environment_action(self) -> None:
        env = AgenticTossing3DEnvironment()
        execute = Mock(return_value=ControllerRun(steps=7, terminated=True))
        env.register_policy(skill_id=100, name="generated_pick", executor=execute)
        result = env._execute(action=np.array([100, 1.0, 2.0, 3.0, 4.0]))
        assert result[0].steps == 7
        assert execute.call_count == 1
        assert np.array_equal(execute.call_args.kwargs["params"], [1, 2, 3, 4])
        assert env._skill_label(action=np.array([100])) == ("generated_pick", ())
        with pytest.raises(ValueError, match="reserved"):
            env.register_policy(skill_id=0, name="wrong", executor=execute)
        with pytest.raises(ValueError, match="already"):
            env.register_policy(skill_id=100, name="other", executor=execute)

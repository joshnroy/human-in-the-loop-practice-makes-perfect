"""A generated controller can reconstruct real tool FK from the exported robot model."""

import importlib.util
import json
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.tossing3d.agentic_bridge import (
    AgenticTossing3DEnvironment,
    Tossing3DAgenticBridge,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="requires the tossing3d extra"
)


class TestAgenticRobotSpec:
    def test_compiled_chain_matches_actual_tool_without_advancing_physics(self) -> None:
        env = AgenticTossing3DEnvironment(scene_bg=False)
        try:
            env.reset_to_seed(seed=125)
            bridge = Tossing3DAgenticBridge(env=env, observation_mode="object_state")
            backend = env.backend()
            robot = backend._object_centric()._robot_env
            model = robot.sim.model.mj_model
            data = robot.sim.data.mj_data
            qpos, qvel, time = data.qpos.copy(), data.qvel.copy(), float(data.time)
            with (
                patch.object(backend._env, "step", side_effect=AssertionError("step")),
                patch.object(backend._env, "reset", side_effect=AssertionError("reset")),
                patch.object(
                    backend._object_centric(),
                    "render_all_cameras",
                    side_effect=AssertionError("render"),
                ),
            ):
                observation = bridge.observe()
                spec = bridge.robot_spec()
            json.dumps({"observation": observation, "robot_spec": spec}, allow_nan=False)
            assert observation["simulation_time_s"] == time
            assert {
                backend.cube_name,
                backend.bin_name,
                backend.barrier_name,
                backend.robot_name,
            } <= {obj["name"] for obj in observation["objects"]}
            chain = spec["kinematic_chain"]
            keys = {joint["position_key"] for link in chain["links"] for joint in link["joints"]}
            assert keys == {
                "pos_base_x",
                "pos_base_y",
                "pos_base_rot",
                *(f"pos_arm_joint{i}" for i in range(1, 8)),
            }
            kinematics = {key: value for key, value in spec.items() if key != "planning_scene"}
            assert "cube" not in json.dumps(kinematics) and "barrier" not in json.dumps(kinematics)
            from kinder_models.dynamic3d.utils import ROBOT_ARM_POSE_TO_BASE

            geometry = spec["planning_scene"]
            assert geometry["robot"]["arm_mount_position_m"] == list(
                ROBOT_ARM_POSE_TO_BASE.position
            )
            assert geometry["boxes"] == [backend.cube_name]
            bin_object = backend._object_centric().get_object(backend.bin_name)
            assert geometry["bins"][backend.bin_name] == {
                key: float(getattr(bin_object, key))
                for key in ("length", "width", "height", "wall_thickness")
            }
            assert not {"poses", "goal", "success", "controller"} & geometry.keys()
            predicted = self.forward_kinematics(
                chain=chain, proprioception=observation["proprioception"]
            )
            tool_id = int(model.site(chain["tool"]["name"]).id)
            proprioception = observation["proprioception"]
            observed_tool_position = [proprioception[f"pos_tool_{axis}"] for axis in "xyz"]
            observed_tool_quaternion = [proprioception[f"quat_tool_{axis}"] for axis in "xyzw"]
            np.testing.assert_allclose(observed_tool_position, data.site_xpos[tool_id], atol=1e-9)
            np.testing.assert_allclose(
                Rotation.from_quat(observed_tool_quaternion).as_matrix(),
                predicted[:3, :3],
                atol=1e-9,
            )
            np.testing.assert_allclose(predicted[:3, 3], data.site_xpos[tool_id], atol=1e-9)
            np.testing.assert_allclose(
                predicted[:3, :3], data.site_xmat[tool_id].reshape(3, 3), atol=1e-9
            )
            np.testing.assert_array_equal(data.qpos, qpos)
            np.testing.assert_array_equal(data.qvel, qvel)
            assert float(data.time) == time and bridge.control_steps == 0
        finally:
            env.close()

    @staticmethod
    def forward_kinematics(
        *, chain: dict[str, Any], proprioception: dict[str, float]
    ) -> np.ndarray:
        transform = np.eye(4)
        for link in [*chain["links"], {**chain["tool"], "joints": []}]:
            fixed = np.eye(4)
            quaternion = link["quaternion_wxyz"]
            fixed[:3, :3] = Rotation.from_quat([*quaternion[1:], quaternion[0]]).as_matrix()
            fixed[:3, 3] = link["position_m"]
            transform = transform @ fixed
            for joint in link["joints"]:
                amount = proprioception[joint["position_key"]] - joint["reference_position"]
                dynamic = np.eye(4)
                axis, pivot = np.asarray(joint["axis"]), np.asarray(joint["position_m"])
                if joint["type"] == "slide":
                    dynamic[:3, 3] = amount * axis
                else:
                    dynamic[:3, :3] = Rotation.from_rotvec(amount * axis).as_matrix()
                    dynamic[:3, 3] = pivot - dynamic[:3, :3] @ pivot
                transform = transform @ dynamic
        return transform

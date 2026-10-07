"""Actual isolated PyBullet geometry, with no live-world or native skill API."""

import importlib.util
import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hitl_pmp.agentic_runtime.planning_scene import PlanningScene

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pybullet") is None
    or importlib.util.find_spec("pybullet_helpers") is None,
    reason="requires local PyBullet and the approved robot assets",
)


@pytest.fixture
def inputs(*, tmp_path: Path) -> dict[str, Any]:
    module = importlib.util.find_spec("pybullet_helpers")
    assert module is not None and module.origin is not None
    asset_root = Path(module.origin).parent / "assets/urdf/kortex_description"
    tree = ET.parse(asset_root / "gen3_7dof.urdf")
    for mesh in tree.getroot().findall(".//mesh"):
        relative = mesh.attrib["filename"].removeprefix("package://kortex_description/")
        mesh.set("filename", str(asset_root / relative))
    urdf = tmp_path / "robot.urdf"
    tree.write(urdf)
    proprioception = {
        **{
            f"pos_arm_joint{i}": value
            for i, value in enumerate([-4.3, -1.6, -4.8, -1.8, -1.4, -1.1, 1.6], start=1)
        },
        "pos_base_x": 0.0,
        "pos_base_y": 0.0,
        "pos_base_rot": 0.0,
        "pos_gripper": 0.0,
        "gripper_command": 0.0,
    }
    return {
        "observation": {
            "observation_mode": "object_state",
            "proprioception": proprioception,
            "simulation_time_s": 0.0,
            "action_spec": {"control_frequency_hz": 10},
            "objects": [
                {
                    "name": "cube",
                    "type": "mujoco_movable_object",
                    "features": {
                        "x": 2.0,
                        "y": 0.0,
                        "z": 0.06,
                        "qx": 0,
                        "qy": 0,
                        "qz": 0,
                        "qw": 1,
                        "bb_x": 0.04,
                        "bb_y": 0.04,
                        "bb_z": 0.04,
                    },
                },
                {
                    "name": "bin",
                    "type": "mujoco_movable_object",
                    "features": {
                        "x": 2.0,
                        "y": 0.0,
                        "z": 0.0,
                        "qx": 0,
                        "qy": 0,
                        "qz": 0,
                        "qw": 1,
                        "bb_x": 0.3,
                        "bb_y": 0.3,
                        "bb_z": 0.2,
                    },
                },
            ],
        },
        "specification": {
            "schema_version": 1,
            "robot": {
                "urdf": str(urdf),
                "arm_mount_position_m": [0.12, 0, 0.4],
                "arm_mount_quaternion_xyzw": [0, 0, 0, 1],
                "tool_link": "tool_frame",
                "arm_joint_names": [f"joint_{i}" for i in range(1, 8)],
                "finger_limits_rad": [0, 0.8],
                "finger_joints": [
                    {"name": name, "multiplier": multiplier}
                    for name, multiplier in (
                        ("finger_joint", 1),
                        ("right_outer_knuckle_joint", 1),
                        ("left_inner_knuckle_joint", 1),
                        ("right_inner_knuckle_joint", 1),
                        ("left_inner_finger_joint", -1),
                        ("right_inner_finger_joint", -1),
                    )
                ],
            },
            "boxes": ["cube"],
            "bins": {"bin": {"length": 0.3, "width": 0.3, "height": 0.2, "wall_thickness": 0.01}},
        },
    }


def test_hollow_bin_interior_remains_free_but_floor_blocks(*, inputs: dict[str, Any]) -> None:
    import pybullet as p

    with PlanningScene(**inputs) as scene:
        assert not p.getClosestPoints(
            scene.bodies["cube"], scene.bodies["bin"], 0, physicsClientId=scene.client_id
        )
        p.resetBasePositionAndOrientation(
            scene.bodies["cube"], [2, 0, 0.005], [0, 0, 0, 1], physicsClientId=scene.client_id
        )
        assert p.getClosestPoints(
            scene.bodies["cube"], scene.bodies["bin"], 0, physicsClientId=scene.client_id
        )


def test_robot_fk_tracks_base_and_ik_is_a_configuration_proposal(*, inputs: dict[str, Any]) -> None:
    with PlanningScene(**inputs) as scene:
        initial = np.array(scene.tool_pose()["position_m"])
        scene.set_configuration(base=[0.2, -0.1, 0], arm=scene.arm, gripper=0)
        translated = scene.tool_pose()
        np.testing.assert_allclose(
            np.array(translated["position_m"]) - initial, [0.2, -0.1, 0], atol=1e-6
        )
        joints = scene.inverse_kinematics(
            position_m=translated["position_m"], quaternion_xyzw=translated["quaternion_xyzw"]
        )
        assert len(joints) == 7 and np.isfinite(joints).all()
        scene.set_configuration(base=scene.base, arm=joints, gripper=0)
        np.testing.assert_allclose(
            scene.tool_pose()["position_m"], translated["position_m"], atol=1e-4
        )


def test_preview_has_caller_budget_restores_configuration_and_keeps_objects_fixed(
    *, inputs: dict[str, Any], tmp_path: Path
) -> None:
    import pybullet as p

    observations = []

    def policy(observation: Any, memory: Any, parameters: Any) -> dict[str, Any]:  # noqa: PLR0917
        observations.append(observation)
        action = [0.0] * 18
        action[0] = 0.01
        return {"action": action, "memory": memory, "done": False}

    with PlanningScene(**inputs) as scene:
        before = list(scene.arm)
        cube_before = p.getBasePositionAndOrientation(
            scene.bodies["cube"], physicsClientId=scene.client_id
        )
        result = scene.render_policy(
            policy=policy, parameters={}, max_steps=2, output_dir=tmp_path / "preview"
        )
        assert len(observations) == len(result["frames"]) == 2
        assert all(item["planning_only"] for item in observations)
        assert observations[1]["proprioception"]["pos_base_x"] == pytest.approx(0.01)
        assert scene.base == [0, 0, 0] and scene.arm == before
        assert (
            p.getBasePositionAndOrientation(scene.bodies["cube"], physicsClientId=scene.client_id)
            == cube_before
        )
        assert result["planning_only"] is True and "success" not in result
        first = Path(result["frames"][0]).read_bytes()
        assert first[:8] == b"\x89PNG\r\n\x1a\n"
        assert struct.unpack(">II", first[16:24]) == (640, 480)
        assert json.loads((tmp_path / "preview/preview.json").read_text())["planning_only"]


def test_preview_rejects_dynamic_schedule_and_restores_copy(
    *, inputs: dict[str, Any], tmp_path: Path
) -> None:
    with PlanningScene(**inputs) as scene:
        with pytest.raises(ValueError, match="18 finite"):
            scene.render_policy(
                policy=lambda *args: {"action": {"schedule": [[0] * 18]}},
                parameters={},
                max_steps=2,
                output_dir=tmp_path,
            )
        assert scene.base == [0, 0, 0]


def test_close_is_idempotent_and_does_not_disconnect_another_client(
    *, inputs: dict[str, Any]
) -> None:
    import pybullet as p

    first = PlanningScene(**inputs)
    first.close()
    with PlanningScene(**inputs) as second:
        first.close()
        assert p.isConnected(second.client_id)
    assert not p.isConnected(second.client_id)


def test_rgb_mode_never_creates_a_planning_world(*, inputs: dict[str, Any]) -> None:
    inputs["observation"]["observation_mode"] = "rgb"
    with pytest.raises(ValueError, match="object-state"):
        PlanningScene(**inputs)

"""Standalone, snapshot-only PyBullet kinematics for an isolated coding sandbox.

This file is copied into the planning image without importing the host package.
It deliberately has no simulator server, task oracle, grasp routine, or controller.
"""

import copy
import importlib
import json
import math
import struct
import weakref
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


class PlanningScene:
    """Kinematic arm and primitive geometry reconstructed from a supplied snapshot.

    Cube and bin poses remain fixed during previews. FK, IK and contact distances
    concern this approximate planning geometry, never the actual task world.
    """

    def __init__(self, *, observation: dict[str, Any], specification: dict[str, Any]) -> None:
        if observation.get("observation_mode") != "object_state":
            raise ValueError("Planning geometry requires an explicit object-state snapshot")
        self._observation = json.loads(json.dumps(observation, allow_nan=False))
        self.specification = json.loads(json.dumps(specification, allow_nan=False))
        if specification.get("schema_version") != 1:
            raise ValueError("Unsupported planning-scene specification")
        self._p = importlib.import_module("pybullet")
        self.client_id = self._p.connect(self._p.DIRECT)
        if self.client_id < 0:
            raise RuntimeError("Cannot create isolated PyBullet planning client")
        self._finalizer = weakref.finalize(self, self._p.disconnect, self.client_id)
        self.bodies: dict[str, int] = {}
        try:
            robot = specification["robot"]
            self.robot_id = self._p.loadURDF(
                robot["urdf"],
                useFixedBase=True,
                flags=self._p.URDF_USE_SELF_COLLISION,
                physicsClientId=self.client_id,
            )
            joints = {
                info[1].decode(): (index, info)
                for index in range(
                    self._p.getNumJoints(self.robot_id, physicsClientId=self.client_id)
                )
                for info in [
                    self._p.getJointInfo(self.robot_id, index, physicsClientId=self.client_id)
                ]
            }
            self.arm_joint_indices = [joints[name][0] for name in robot["arm_joint_names"]]
            self._finger_joints = [
                (joints[item["name"]][0], float(item["multiplier"]))
                for item in robot["finger_joints"]
            ]
            tool = robot["tool_link"]
            self.tool_link_index = next(
                index for index, info in joints.values() if info[12].decode() == tool
            )
            self._movable_indices = [
                index
                for index, info in sorted(joints.values(), key=lambda item: item[1][3])
                if info[3] >= 0
            ]
            bins = specification.get("bins", {})
            for item in observation["objects"]:
                name, features = item["name"], item["features"]
                if name in bins:
                    self.bodies[name] = self._bin(features=features, dimensions=bins[name])
                elif item["type"] == "mujoco_static_collider" or name in specification["boxes"]:
                    self.bodies[name] = self._box(features=features)
            robot_state = observation["proprioception"]
            self.set_configuration(
                base=[robot_state[f"pos_base_{axis}"] for axis in ("x", "y", "rot")],
                arm=[robot_state[f"pos_arm_joint{i}"] for i in range(1, 8)],
                gripper=robot_state["pos_gripper"],
            )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Release only this local client, once, including construction failures."""
        self._finalizer()

    def __enter__(self) -> "PlanningScene":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def set_configuration(self, *, base: list[float], arm: list[float], gripper: float) -> None:
        """Set a proposed configuration in the copy, without advancing any physics."""
        self._finite(values=base, count=3)
        self._finite(values=arm, count=7)
        self._finite(values=[gripper], count=1)
        robot = self.specification["robot"]
        yaw = self._p.getQuaternionFromEuler([0, 0, base[2]])
        position, orientation = self._p.multiplyTransforms(
            [base[0], base[1], 0],
            yaw,
            robot["arm_mount_position_m"],
            robot["arm_mount_quaternion_xyzw"],
        )
        self._p.resetBasePositionAndOrientation(
            self.robot_id, position, orientation, physicsClientId=self.client_id
        )
        for index, value in zip(self.arm_joint_indices, arm, strict=True):
            self._p.resetJointState(self.robot_id, index, value, physicsClientId=self.client_id)
        for index, multiplier in self._finger_joints:
            self._p.resetJointState(
                self.robot_id, index, gripper * multiplier, physicsClientId=self.client_id
            )
        self.base, self.arm, self.gripper = list(base), list(arm), float(gripper)

    def tool_pose(self) -> dict[str, list[float]]:
        """Return the URDF tool_link pose; it is not a live measured pinch-site pose."""
        pose = self._p.getLinkState(
            self.robot_id,
            self.tool_link_index,
            computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        return {"position_m": list(pose[4]), "quaternion_xyzw": list(pose[5])}

    def inverse_kinematics(
        self,
        *,
        position_m: list[float],
        quaternion_xyzw: list[float] | None = None,
        max_iterations: int = 100,
    ) -> list[float]:
        """Numerical IK proposal only; callers must check residuals and collisions."""
        self._finite(values=position_m, count=3)
        if max_iterations <= 0:
            raise ValueError("IK iterations must be positive")
        kwargs: dict[str, Any] = {}
        if quaternion_xyzw is not None:
            self._finite(values=quaternion_xyzw, count=4)
            kwargs["targetOrientation"] = quaternion_xyzw
        values = self._p.calculateInverseKinematics(
            self.robot_id,
            self.tool_link_index,
            targetPosition=position_m,
            maxNumIterations=max_iterations,
            physicsClientId=self.client_id,
            **kwargs,
        )
        proposed = dict(zip(self._movable_indices, values, strict=True))
        return [float(proposed[index]) for index in self.arm_joint_indices]

    def collision_distances(self, *, distance_m: float = 0.0) -> list[dict[str, Any]]:
        """Raw arm-to-object signed distances; intended contacts are not labeled failures."""
        self._finite(values=[distance_m], count=1)
        if distance_m < 0:
            raise ValueError("Query distance must be nonnegative")
        contacts = []
        for name, body in self.bodies.items():
            for contact in self._p.getClosestPoints(
                self.robot_id, body, distance_m, physicsClientId=self.client_id
            ):
                contacts.append({
                    "object": name,
                    "robot_link_index": int(contact[3]),
                    "distance_m": float(contact[8]),
                })
        return contacts

    def render_state(
        self,
        *,
        path: str | Path,
        width: int = 640,
        height: int = 480,
        target_m: list[float] | None = None,
        distance_m: float = 2.0,
        yaw_degrees: float = 45,
        pitch_degrees: float = -35,
    ) -> str:
        """Save a local planning-scene PNG with PyBullet's CPU renderer."""
        if width <= 0 or height <= 0 or distance_m <= 0:
            raise ValueError("Render dimensions and camera distance must be positive")
        target = target_m if target_m is not None else [self.base[0], self.base[1], 0.4]
        self._finite(values=target, count=3)
        view = self._p.computeViewMatrixFromYawPitchRoll(
            target, distance_m, yaw_degrees, pitch_degrees, 0, 2
        )
        projection = self._p.computeProjectionMatrixFOV(60, width / height, 0.02, 20)
        result = self._p.getCameraImage(
            width,
            height,
            viewMatrix=view,
            projectionMatrix=projection,
            renderer=self._p.ER_TINY_RENDERER,
            physicsClientId=self.client_id,
        )
        pixels = np.asarray(result[2], dtype=np.uint8).reshape(height, width, 4)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._write_png(path=destination, pixels=pixels)
        return str(destination)

    def render_policy(
        self,
        *,
        policy: Callable[..., dict[str, Any]],
        parameters: dict[str, Any],
        max_steps: int,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        """Preview position commands kinematically, with no grasp/toss outcome claims.

        The caller supplies the sole step budget. Velocity targets are recorded but
        not simulated. Millisecond schedules require dynamics and are rejected.
        """
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("A positive caller-owned max_steps is required")
        initial = (list(self.base), list(self.arm), self.gripper)
        memory: dict[str, Any] = {}
        frames, trajectory = [], []
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        try:
            for step in range(max_steps):
                observation = self._preview_observation(step=step)
                output = policy(copy.deepcopy(observation), memory, copy.deepcopy(parameters))
                if not isinstance(output, dict):
                    raise ValueError("Policy preview requires a dictionary output")
                if output.get("done", False):
                    break
                action: Any = output.get("action")
                self._finite(values=action, count=18)
                if not 0 <= action[10] <= 1:
                    raise ValueError("Gripper command must lie in [0, 1]")
                limits = self.specification["robot"]["finger_limits_rad"]
                self.set_configuration(
                    base=[q + delta for q, delta in zip(self.base, action[:3], strict=True)],
                    arm=[q + delta for q, delta in zip(self.arm, action[3:10], strict=True)],
                    gripper=limits[0] + action[10] * (limits[1] - limits[0]),
                )
                memory = output.get("memory", memory)
                trajectory.append({"step": step, "action": action, "tool_pose": self.tool_pose()})
                frames.append(self.render_state(path=destination / f"frame-{step:05d}.png"))
        finally:
            self.set_configuration(base=initial[0], arm=initial[1], gripper=initial[2])
        result = {
            "planning_only": True,
            "model": "kinematic position targets; fixed objects; velocity targets ignored",
            "frames": frames,
            "trajectory": trajectory,
        }
        (destination / "preview.json").write_text(json.dumps(result, indent=2))
        return result

    def _preview_observation(self, *, step: int) -> dict[str, Any]:
        observation = copy.deepcopy(self._observation)
        proprioception = observation["proprioception"]
        for axis, value in zip(("x", "y", "rot"), self.base, strict=True):
            proprioception[f"pos_base_{axis}"] = value
        for index, value in enumerate(self.arm, start=1):
            proprioception[f"pos_arm_joint{index}"] = value
        proprioception["pos_gripper"] = self.gripper
        limits = self.specification["robot"]["finger_limits_rad"]
        proprioception["gripper_command"] = (self.gripper - limits[0]) / (limits[1] - limits[0])
        pose = self.tool_pose()
        for axis, value in zip("xyz", pose["position_m"], strict=True):
            proprioception[f"pos_tool_{axis}"] = value
        for axis, value in zip("xyzw", pose["quaternion_xyzw"], strict=True):
            proprioception[f"quat_tool_{axis}"] = value
        for key in proprioception:
            if key.startswith("vel_"):
                proprioception[key] = 0.0
        for item in observation["objects"]:
            if item["type"] == "mujoco_tidybot_robot":
                item["features"] = dict(proprioception)
            else:
                for key in ("vx", "vy", "vz", "wx", "wy", "wz"):
                    if key in item["features"]:
                        item["features"][key] = 0.0
        observation["control_step"] = step
        frequency = float(observation["action_spec"]["control_frequency_hz"])
        observation["simulation_time_s"] = (
            float(observation.get("simulation_time_s", 0)) + step / frequency
        )
        observation["planning_only"] = True
        return observation

    def _box(self, *, features: dict[str, float]) -> int:
        return self._geometry(
            features=features,
            half_extents=[[features[f"bb_{axis}"] / 2 for axis in "xyz"]],
            positions=[[0, 0, 0]],
        )

    def _bin(self, *, features: dict[str, float], dimensions: dict[str, float]) -> int:
        length, width, height, thickness = [
            dimensions[key] for key in ("length", "width", "height", "wall_thickness")
        ]
        if not 0 < thickness < min(length / 2, width / 2, height):
            raise ValueError("Invalid hollow-bin geometry")
        z = (height + thickness) / 2
        half_height = (height - thickness) / 2
        return self._geometry(
            features=features,
            half_extents=[
                [length / 2, width / 2, thickness / 2],
                [thickness / 2, width / 2, half_height],
                [thickness / 2, width / 2, half_height],
                [length / 2 - thickness, thickness / 2, half_height],
                [length / 2 - thickness, thickness / 2, half_height],
            ],
            positions=[
                [0, 0, thickness / 2],
                [(length - thickness) / 2, 0, z],
                [-(length - thickness) / 2, 0, z],
                [0, (width - thickness) / 2, z],
                [0, -(width - thickness) / 2, z],
            ],
        )

    def _geometry(
        self,
        *,
        features: dict[str, float],
        half_extents: list[list[float]],
        positions: list[list[float]],
    ) -> int:
        shape = self._p.createCollisionShapeArray(
            shapeTypes=[self._p.GEOM_BOX] * len(positions),
            halfExtents=half_extents,
            collisionFramePositions=positions,
            physicsClientId=self.client_id,
        )
        visual = self._p.createVisualShapeArray(
            shapeTypes=[self._p.GEOM_BOX] * len(positions),
            halfExtents=half_extents,
            visualFramePositions=positions,
            rgbaColors=[[0.55, 0.6, 0.7, 1]] * len(positions),
            physicsClientId=self.client_id,
        )
        return int(
            self._p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=shape,
                baseVisualShapeIndex=visual,
                basePosition=[features[axis] for axis in "xyz"],
                baseOrientation=[features[f"q{axis}"] for axis in "xyzw"],
                physicsClientId=self.client_id,
            )
        )

    @staticmethod
    def _finite(*, values: Any, count: int) -> None:
        if (
            not isinstance(values, (list, tuple))
            or len(values) != count
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in values
            )
        ):
            raise ValueError(f"Expected {count} finite numeric values")

    @staticmethod
    def _write_png(*, path: Path, pixels: np.ndarray) -> None:
        height, width, _ = pixels.shape
        raw = b"".join(b"\x00" + row.tobytes() for row in pixels)
        chunks = [
            (b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)),
            (b"IDAT", zlib.compress(raw)),
            (b"IEND", b""),
        ]
        result = b"\x89PNG\r\n\x1a\n"
        for kind, data in chunks:
            result += struct.pack(">I", len(data)) + kind + data
            result += struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        path.write_bytes(result)

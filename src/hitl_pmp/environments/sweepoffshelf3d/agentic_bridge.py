"""Restricted host interface for a generated SweepOffShelf3D joint-space controller.

The Full-Agentic counterpart of `tossing3d/agentic_bridge.py`, with the same shape so
the LLM-facing contract is analogous: numeric object-state observations, robot
proprioception with a forward-kinematics tool pose, and raw actuator commands. It
exposes no robot skills, predicates, goal labels or scene resets; the human's two
skills reach the scene only through `SweepOffShelf3DEnvironment.reset_movables`.

The action is the scene's native one, exactly as every other arm drives it: KINDER's
11-D delta command (`TidyBot3DConfig(use_arm_velocities=False)`, set by the backend),
so unlike Tossing3D's 18-D command it carries no arm-velocity targets and admits no
millisecond schedule.
"""

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from scipy.spatial.transform import Rotation

from .environment import SweepOffShelf3DEnvironment

ACTION_WIDTH = 11
PHYSICAL_FEATURES = frozenset({
    "x",
    "y",
    "z",
    "qw",
    "qx",
    "qy",
    "qz",
    "vx",
    "vy",
    "vz",
    "wx",
    "wy",
    "wz",
    "bb_x",
    "bb_y",
    "bb_z",
})


class SweepOffShelf3DAgenticBridge(BaseModel):
    """One policy execution's bounded observation/actuator interface.

    Construct a fresh bridge for a fresh control-step budget; doing so does not reset
    the simulator.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    env: SweepOffShelf3DEnvironment
    observation_mode: Literal["object_state"] = "object_state"
    step_limit: int = Field(default=1000, gt=0)
    _control_steps: int = PrivateAttr(default=0)

    @property
    def control_steps(self) -> int:
        """Number of dispatched control periods, including a failed step attempt."""
        return self._control_steps

    def observe(self) -> dict[str, Any]:
        """Return measured physical state, excluding native task labels."""
        robot = self._robot()
        simulation_time = float(robot.sim.data.mj_data.time)
        if not np.isfinite(simulation_time) or simulation_time < 0:
            raise ValueError("simulation time must be finite and nonnegative")
        proprioception: dict[str, float] = {}
        for prefix, readings in (("pos", robot.qpos), ("vel", robot.qvel)):
            for index, axis in enumerate(("x", "y", "rot")):
                proprioception[f"{prefix}_base_{axis}"] = float(readings["base"][index])
            for index in range(7):
                proprioception[f"{prefix}_arm_joint{index + 1}"] = float(readings["arm"][index])
            proprioception[f"{prefix}_gripper"] = float(readings["gripper"][0])
        # As in Tossing3D: the command is kept apart from the measured finger position,
        # so a closed command is never an implicit grasp label.
        proprioception["gripper_command"] = float(robot.ctrl["gripper"][0]) / 255.0
        proprioception.update(self._tool_pose(proprioception=proprioception))
        if not np.isfinite(list(proprioception.values())).all():
            raise ValueError("robot proprioception must be finite")
        return {
            "observation_mode": self.observation_mode,
            "proprioception": proprioception,
            "control_step": self._control_steps,
            "simulation_time_s": simulation_time,
            "action_spec": self.action_spec(),
            "objects": self._objects(proprioception=proprioception),
            "state_spec": {
                "position_frame": "world",
                "length_unit": "m",
                "angle_unit": "rad",
                "quaternion_order": "wxyz",
                "linear_velocity_unit": "m/s",
                "linear_velocity_frame": "world",
                "angular_velocity_unit": "rad/s",
                "angular_velocity_frame": "object-local",
                "bounding_box": "full object-frame dimensions",
                "simulation_time_s": (
                    "Physical simulation clock in seconds, shared by every bridge to this "
                    "world and monotonic between world initializations; use to order samples"
                ),
                "control_step": (
                    "Bridge-local dispatched control-period count for option budgeting; "
                    "not a global timeline and may remain zero for an observation-only bridge"
                ),
                "tool_pose": (
                    "pos_tool_x/y/z and quat_tool_w/x/y/z describe the robot pinch site "
                    "in the world frame, from calibrated robot forward kinematics and "
                    "measured base/joint positions; not contact or grasp labels"
                ),
                "robot_gripper": (
                    "pos_gripper is measured driver-joint radians; "
                    "gripper_command is normalized actuator command"
                ),
                "static_objects": (
                    "Objects without velocity features are fixed scene geometry "
                    "(the cupboard and its boards)"
                ),
            },
        }

    def robot_spec(self) -> dict[str, Any]:
        """Export calibrated kinematics; no planning model, controllers or goal labels."""
        model = self.env.backend().model()
        name = self.robot_name()
        tool_id = int(model.site(f"{name}_pinch_site").id)
        tool_body = int(model.site_bodyid[tool_id])
        body_ids = []
        body_id = tool_body
        while body_id:
            body_ids.append(body_id)
            body_id = int(model.body_parentid[body_id])
        position_keys = {
            f"{name}_joint_{suffix}": f"pos_base_{axis}"
            for suffix, axis in (("x", "x"), ("y", "y"), ("th", "rot"))
        }
        position_keys.update({f"{name}_joint_{i}": f"pos_arm_joint{i}" for i in range(1, 8)})
        links = []
        found_joint_names = set()
        for body_id in reversed(body_ids):
            body_name = str(model.body(body_id).name)
            if not body_name.startswith(f"{name}_"):
                raise ValueError("robot kinematic chain contains a non-robot body")
            parent_id = int(model.body_parentid[body_id])
            joints = []
            start = int(model.body_jntadr[body_id])
            for joint_id in range(start, start + int(model.body_jntnum[body_id])):
                joint_name = str(model.joint(joint_id).name)
                if joint_name not in position_keys:
                    raise ValueError(f"unsupported robot-chain joint: {joint_name}")
                joint_type = {2: "slide", 3: "hinge"}.get(int(model.jnt_type[joint_id]))
                if joint_type is None:
                    raise ValueError("robot chain requires scalar hinge or slide joints")
                position_key = position_keys[joint_name]
                joints.append({
                    "name": joint_name,
                    "type": joint_type,
                    "axis": model.jnt_axis[joint_id].tolist(),
                    "position_m": model.jnt_pos[joint_id].tolist(),
                    "reference_position": float(model.qpos0[model.jnt_qposadr[joint_id]]),
                    "limits": (
                        model.jnt_range[joint_id].tolist() if model.jnt_limited[joint_id] else None
                    ),
                    "position_key": position_key,
                    "velocity_key": position_key.replace("pos_", "vel_", 1),
                })
                found_joint_names.add(joint_name)
            links.append({
                "name": body_name,
                "parent": str(model.body(parent_id).name) if parent_id else "world",
                "position_m": model.body_pos[body_id].tolist(),
                "quaternion_wxyz": model.body_quat[body_id].tolist(),
                "joints": joints,
            })
        if found_joint_names != position_keys.keys():
            raise ValueError("robot specification is missing controlled base or arm joints")
        gripper_id = int(model.joint(f"{name}_right_driver_joint").id)
        return {
            "schema_version": 1,
            "robot_name": name,
            "length_unit": "m",
            "angle_unit": "rad",
            "quaternion_order": "wxyz",
            "kinematic_chain": {
                "root_frame": "world",
                "transform_order": (
                    "For each link, apply its fixed parent-relative position/quaternion, "
                    "then its joints in listed order. A hinge rotates about its local "
                    "axis through position_m by q-reference_position; a slide translates "
                    "along its local axis by q-reference_position. Apply the tool's "
                    "fixed parent-relative transform last."
                ),
                "links": links,
                "tool": {
                    "name": str(model.site(tool_id).name),
                    "parent": str(model.body(tool_body).name),
                    "position_m": model.site_pos[tool_id].tolist(),
                    "quaternion_wxyz": model.site_quat[tool_id].tolist(),
                },
            },
            "gripper": {
                "position_key": "pos_gripper",
                "velocity_key": "vel_gripper",
                "command_key": "gripper_command",
                "limits": model.jnt_range[gripper_id].tolist(),
                "command_open": 0.0,
                "command_closed": 1.0,
            },
            "action_spec": self.action_spec(),
        }

    def action_spec(self) -> dict[str, Any]:
        """Describe the native 11-D delta interface and its finite bounds."""
        robot = self._robot()
        space = self.env.backend().env().action_space
        if space.shape != (ACTION_WIDTH,) or not robot.act_delta:
            raise ValueError("agentic SweepOffShelf3D requires the native 11-D delta action")
        low = np.asarray(space.low, dtype=float).copy()
        high = np.asarray(space.high, dtype=float).copy()
        if not np.isfinite(low).all() or not np.isfinite(high).all():
            raise ValueError("native action bounds must be finite")
        return {
            "shape": [ACTION_WIDTH],
            "low": low.tolist(),
            "high": high.tolist(),
            "position_mode": "delta",
            "base": {
                "indices": [0, 1, 2],
                "coordinates": "world x/y/yaw",
                "units": ["m", "m", "rad"],
            },
            "arm_position": {"indices": list(range(3, 10)), "units": "rad"},
            "gripper": {"index": 10, "open": 0.0, "closed": 1.0},
            "control_frequency_hz": float(robot.control_frequency),
            "step_limit": self.step_limit,
        }

    def step(self, *, action: Any) -> dict[str, Any]:
        """Validate one numeric command, then advance physics by one control period."""
        if self._control_steps >= self.step_limit:
            raise RuntimeError("generated policy exhausted its control-step budget")
        spec = self.action_spec()
        if isinstance(action, dict):
            if set(action) != {"values"}:
                raise ValueError("action must contain only values")
            action = action["values"]
        array = np.asarray(action, dtype=float)
        if array.shape != (ACTION_WIDTH,):
            raise ValueError("action must be an 11-vector")
        if not np.isfinite(array).all():
            raise ValueError("actuator commands must be finite")
        if np.any(array < spec["low"]) or np.any(array > spec["high"]):
            raise ValueError("actuator command exceeds the published bounds")
        backend = self.env.backend()
        self._control_steps += 1
        # Discard reward and termination: the controller must judge outcomes itself.
        backend.env().step(array)
        # The backend's own record of the last gripper command, as its controllers keep it.
        backend.grip = float(array[-1])
        return self.observe()

    def robot_name(self) -> str:
        return self.env.robot.name

    def _tool_pose(self, *, proprioception: dict[str, float]) -> dict[str, float]:
        chain = self.robot_spec()["kinematic_chain"]
        position, rotation = np.zeros(3), np.eye(3)
        for link in [*chain["links"], {**chain["tool"], "joints": []}]:
            position += rotation @ np.asarray(link["position_m"])
            quaternion = link["quaternion_wxyz"]
            rotation = rotation @ Rotation.from_quat([*quaternion[1:], quaternion[0]]).as_matrix()
            for joint in link["joints"]:
                amount = proprioception[joint["position_key"]] - joint["reference_position"]
                axis, pivot = np.asarray(joint["axis"]), np.asarray(joint["position_m"])
                if joint["type"] == "slide":
                    position += rotation @ (amount * axis)
                else:
                    turn = Rotation.from_rotvec(amount * axis).as_matrix()
                    position += rotation @ (pivot - turn @ pivot)
                    rotation = rotation @ turn
        quaternion_xyzw = Rotation.from_matrix(rotation).as_quat()
        return {
            **{
                f"pos_tool_{axis}": float(value)
                for axis, value in zip("xyz", position, strict=True)
            },
            **{
                f"quat_tool_{axis}": float(value)
                for axis, value in zip("xyzw", quaternion_xyzw, strict=True)
            },
        }

    def _objects(self, *, proprioception: dict[str, float]) -> list[dict[str, Any]]:
        state = self._state()
        objects = []
        for obj in state:
            if obj.name == self.robot_name():
                features = dict(proprioception)
            else:
                labels = state.type_features[obj.type]
                if not labels or not set(labels) <= PHYSICAL_FEATURES:
                    raise ValueError(f"unsupported physical object features for {obj.name}")
                features = {}
                for label in labels:
                    value = state.get(obj, label)
                    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value):
                        raise ValueError(f"non-numeric physical feature: {obj.name}.{label}")
                    features[label] = float(value)
            objects.append({"name": obj.name, "type": obj.type.name, "features": features})
        if not objects:
            raise ValueError("object-state observation contains no objects")
        return objects

    def _state(self) -> Any:
        # Read fresh from the scene: a human reset moves objects without stepping.
        return self.env.backend().core()._get_object_centric_state()  # noqa: SLF001

    def _robot(self) -> Any:
        # This host-only handle is never returned by any relay operation.
        return self.env.backend().core()._robot_env  # noqa: SLF001

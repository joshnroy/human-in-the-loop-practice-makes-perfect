"""Restricted host interface for generated Tossing3D joint-space controllers.

Generated Python runs in the isolated runtime. This module executes only numeric
actuator commands and supplies explicit visual or numeric observations. The existing
environment remains responsible for one-option bookkeeping and paid human resets.
"""

import base64
import io
from typing import Any, ClassVar, Protocol

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from scipy.spatial.transform import Rotation

from hitl_pmp.agentic_runtime.observations import ObservationMode
from hitl_pmp.core.problem.environment.types import Action

from .environment import Tossing3DEnvironment
from .kinder_backend import ControllerRun


class AgenticTossing3DEnvironment(Tossing3DEnvironment):
    """Dispatch an isolated generated policy as one ordinary environment action.

    Register callbacks separately on practice and evaluation worlds. Callbacks execute
    the sandbox runtime; they must never load generated Python into the host process.
    A learner may update the callback's policy file between sessions without changing
    the action ID, so the existing action budget and recording path stay intact.
    """

    generated_skill_id_start: ClassVar[int] = 100
    _policy_executors: dict[int, "GeneratedPolicyExecutor"] = PrivateAttr(default_factory=dict)
    _policy_names: dict[int, str] = PrivateAttr(default_factory=dict)

    def register_policy(
        self, *, skill_id: int, name: str, executor: "GeneratedPolicyExecutor"
    ) -> None:
        """Install a trusted host callback, never a source-code string."""
        if skill_id < self.generated_skill_id_start:
            raise ValueError("skill IDs below 100 are reserved for native environment actions")
        if skill_id in self._policy_executors:
            raise ValueError(f"skill ID {skill_id} is already registered")
        if not name.strip():
            raise ValueError("a generated policy needs a nonempty name")
        self._policy_executors[skill_id] = executor
        self._policy_names[skill_id] = name

    def _execute(self, *, action: Action) -> list[ControllerRun]:
        if np.asarray(action).shape != (5,) or not np.all(np.isfinite(action)):
            return [ControllerRun(steps=0, terminated=False, error="invalid policy action")]
        skill_id = int(round(float(action[0])))
        if skill_id in self._policy_executors:
            return [self._policy_executors[skill_id](params=np.asarray(action[1:]).copy())]
        return super()._execute(action=action)

    def _skill_label(self, *, action: Action) -> tuple[str, tuple[str, ...]]:
        skill_id = int(round(float(action[0])))
        if skill_id in self._policy_names:
            return self._policy_names[skill_id], ()
        return super()._skill_label(action=action)


class Tossing3DAgenticBridge(BaseModel):
    """One policy execution's bounded observation/actuator interface.

    Construct a fresh bridge for each option to give it a fresh control-step budget;
    doing so does not reset the simulator. Numeric object poses are opt-in; neither
    mode exports native success, goal labels, predicates, or scene-reset operations.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    env: Tossing3DEnvironment
    observation_mode: ObservationMode = "rgb"
    step_limit: int = Field(default=1000, gt=0)
    arm_velocity_limit: float = Field(default=12.0, gt=0, allow_inf_nan=False)
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
        # KINDER's pos_gripper feature is a command, not a measured finger position.
        # Keep that command separately so a closed command is never an implicit grasp label.
        proprioception["gripper_command"] = float(robot.ctrl["gripper"][0]) / 255.0
        if self.observation_mode == "object_state":
            proprioception.update(self._tool_pose(proprioception=proprioception))
        if not np.isfinite(list(proprioception.values())).all():
            raise ValueError("robot proprioception must be finite")
        observation: dict[str, Any] = {
            "observation_mode": self.observation_mode,
            "proprioception": proprioception,
            "control_step": self._control_steps,
            "simulation_time_s": simulation_time,
            "action_spec": self.action_spec(),
        }
        if self.observation_mode == "object_state":
            observation["objects"] = self._objects(proprioception=proprioception)
            observation["state_spec"] = {
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
            }
            return observation
        backend = self.env.backend()
        frames = backend._object_centric().render_all_cameras()  # noqa: SLF001
        images = []
        for camera in (backend.camera, f"{backend.robot_name}_wrist"):
            key = f"{camera}_image"
            if key not in frames:
                raise RuntimeError(f"Required observation camera is unavailable: {camera}")
            pixels = np.asarray(frames[key], dtype=np.uint8)
            buffer = io.BytesIO()
            Image.fromarray(pixels).save(buffer, format="PNG")
            images.append(
                "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
            )
        observation.update(images=images, image_views=["task_overview", "robot_wrist"])
        return observation

    def robot_spec(self, *, include_planning: bool = True) -> dict[str, Any]:
        """Export calibrated kinematics and optional static planning-model geometry."""
        robot = self._robot()
        model = robot.sim.model.mj_model
        name = self.env.backend().robot_name
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
        specification = {
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
        if include_planning and self.observation_mode == "object_state":
            specification["planning_scene"] = self.planning_spec()
        return specification

    def planning_spec(self) -> dict[str, Any]:
        """Export approved PyBullet geometry, without poses, controllers, or goal labels."""
        from kinder_models.dynamic3d.utils import ROBOT_ARM_POSE_TO_BASE

        backend = self.env.backend()
        bin_object = backend._object_centric().get_object(backend.bin_name)  # noqa: SLF001
        # Match the arm-only collision model used by _pick_collision_sim; this
        # does not expose that controller, its grasp transforms, or its planner.
        return {
            "schema_version": 1,
            "model": "kinematic arm and observed primitives; no task dynamics",
            "robot": {
                "urdf": "/opt/hitl-planning/robot/gen3_7dof.urdf",
                "arm_mount_position_m": list(ROBOT_ARM_POSE_TO_BASE.position),
                "arm_mount_quaternion_xyzw": list(ROBOT_ARM_POSE_TO_BASE.orientation),
                "arm_joint_names": [f"joint_{i}" for i in range(1, 8)],
                "tool_link": "tool_frame",
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
                "finger_limits_rad": [0.0, 0.8],
            },
            "boxes": [backend.cube_name],
            "bins": {
                backend.bin_name: {
                    key: float(getattr(bin_object, key))
                    for key in ("length", "width", "height", "wall_thickness")
                }
            },
        }

    def _tool_pose(self, *, proprioception: dict[str, float]) -> dict[str, float]:
        # Cached simulator site transforms can lag the last integration tick. Use
        # the exported robot-only calibration with the same readings as the observation.
        chain = self.robot_spec(include_planning=False)["kinematic_chain"]
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
        backend = self.env.backend()
        state = backend._require_state()  # noqa: SLF001 -- public numeric observation schema
        physical_features = {
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
        }
        objects = []
        for obj in state:
            if obj.name == backend.robot_name:
                features = dict(proprioception)
            else:
                labels = state.type_features[obj.type]
                if not labels or not set(labels) <= physical_features:
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

    def action_spec(self) -> dict[str, Any]:
        """Describe the native 18D joint-space interface and explicit finite bounds."""
        backend = self.env.backend()
        robot = self._robot()
        space = backend._env.action_space  # noqa: SLF001 -- numeric native boundary only
        if space.shape != (18,) or not robot.act_delta:
            raise ValueError("agentic Tossing3D requires the native 18D delta-action mode")
        low, high = (
            np.asarray(space.low, dtype=float).copy(),
            np.asarray(space.high, dtype=float).copy(),
        )
        low[11:18], high[11:18] = -self.arm_velocity_limit, self.arm_velocity_limit
        if not np.isfinite(low).all() or not np.isfinite(high).all():
            raise ValueError("native position action bounds must be finite")
        frequency = float(robot.control_frequency)
        # Pinned KINDER schedules one row per millisecond across a control period.
        schedule_rows = int(round(1.0 / frequency / 0.001))
        return {
            "shape": [18],
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
            "arm_velocity": {"indices": list(range(11, 18)), "units": "rad/s"},
            "control_frequency_hz": frequency,
            "schedule_rows": schedule_rows,
            "schedule_timestep_s": 0.001,
            "step_limit": self.step_limit,
        }

    def step(self, *, action: Any) -> dict[str, Any]:
        """Validate one numeric action or millisecond schedule, then advance physics."""
        if self._control_steps >= self.step_limit:
            raise RuntimeError("generated policy exhausted its control-step budget")
        spec = self.action_spec()
        if isinstance(action, dict):
            if set(action) not in ({"values"}, {"schedule"}):
                raise ValueError("action must contain only values or only schedule")
            action = action.get("values", action.get("schedule"))
        array = np.asarray(action, dtype=float)
        if array.shape not in ((18,), (spec["schedule_rows"], 18)):
            raise ValueError("action must be an 18-vector or one complete millisecond schedule")
        if not np.isfinite(array).all():
            raise ValueError("actuator commands must be finite")
        if np.any(array < spec["low"]) or np.any(array > spec["high"]):
            raise ValueError("actuator command exceeds the published bounds")
        backend = self.env.backend()
        self._control_steps += 1
        # Discard reward and termination: the VLM must evaluate actual skill outcomes.
        observation, _, _, _, _ = backend._env.step(array)  # noqa: SLF001
        backend._state = backend._env.observation_space.devectorize(observation)  # noqa: SLF001
        return self.observe()

    def _robot(self) -> Any:
        # This host-only handle is never returned by any relay operation.
        return self.env.backend()._object_centric()._robot_env  # noqa: SLF001


class GeneratedPolicyExecutor(Protocol):
    """Trusted callback that drives an isolated runtime for one whole option."""

    def __call__(self, *, params: np.ndarray) -> ControllerRun: ...

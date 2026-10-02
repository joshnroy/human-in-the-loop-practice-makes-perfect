"""Scratch native geometry queries for the zero-margin palm/tool constraint."""

from collections.abc import Sequence

import mujoco
import numpy as np
from pybullet_helpers.geometry import Pose


class NativePalmClearance:
    """Query actual palm geometry, excluding intentionally contacting fingers.

    Results are capped at one micrometer: positive proves separation, negative
    proves penetration, and zero is unresolved/contact and must not clear a proxy
    rejection. This helper does not justify any positive clearance margin.
    """

    def __init__(self, *, model: mujoco.MjModel, live_data: mujoco.MjData) -> None:
        self.model = model
        self.live_data = live_data
        self.data = mujoco.MjData(model)
        self.palm_geoms = self._geoms(name="robot_base")
        self.tool_geoms = self._geoms(name="wiper_0")
        pad_bodies = {"robot_left_pad", "robot_right_pad"}
        self.nonpad_tool_pairs = []
        for robot in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[robot])
            if name is None or not name.startswith("robot_") or name in pad_bodies:
                continue
            for tool in self.tool_geoms:
                if (model.geom_contype[robot] & model.geom_conaffinity[tool]) or (
                    model.geom_contype[tool] & model.geom_conaffinity[robot]
                ):
                    self.nonpad_tool_pairs.append((robot, tool, name))
        names = [f"robot_joint_{i}" for i in range(1, 8)] + [
            f"robot_{side}_{part}_joint"
            for part in ("driver", "spring_link", "follower")
            for side in ("left", "right")
        ]
        self.joint_addresses = [self._address(name=name) for name in names]
        self.base_addresses = [
            self._address(name=f"robot_joint_{axis}") for axis in ("x", "y", "th")
        ]
        self.tool_address = self._address(name="wiper_0_joint")

    def _address(self, *, name: str) -> int:
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            raise ValueError(f"Missing native joint: {name}")
        return int(self.model.jnt_qposadr[joint])

    def _geoms(self, *, name: str) -> list[int]:
        body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        geoms = [
            g
            for g in range(self.model.ngeom)
            if body >= 0
            and self.model.geom_bodyid[g] == body
            and (self.model.geom_contype[g] or self.model.geom_conaffinity[g])
        ]
        if not geoms:
            raise ValueError(f"Missing native collision geometry: {name}")
        return geoms

    def distance(
        self, *, joints: Sequence[float], tool_pose: Pose, base: Sequence[float] | None = None
    ) -> float:
        """Evaluate 13 planning joints and a WORLD tool pose (quaternion xyzw).

        Optional base is world x/y/yaw. Otherwise use the live native base.
        All other coordinates are copied anew from live data on every query.
        Neither native model flags nor live state are changed, and no physics
        stepping occurs. A large positive distance cap is deliberately avoided
        because pinned MuJoCo can return spurious zero for separated meshes.
        """
        self._set_candidate(joints=joints, tool_pose=tool_pose, base=base)
        return min(
            float(mujoco.mj_geomDistance(self.model, self.data, palm, tool, 1e-6, None))
            for palm in self.palm_geoms
            for tool in self.tool_geoms
        )

    def nonpad_tool_contacts(
        self,
        *,
        joints: Sequence[float],
        tool_pose: Pose,
        base: Sequence[float] | None = None,
    ) -> list[dict[str, object]]:
        """Return zero/negative native robot/tool pairs, permitting only actual pads.

        Native collision masks select physical pairs; visual meshes cannot block
        pickup. Scratch geometry and the same tiny distance cap preserve live state.
        """
        self._set_candidate(joints=joints, tool_pose=tool_pose, base=base)
        contacts = []
        for robot, tool, name in self.nonpad_tool_pairs:
            distance = float(mujoco.mj_geomDistance(self.model, self.data, robot, tool, 1e-6, None))
            if distance <= 0.0:
                contacts.append({
                    "robot_geom": robot,
                    "tool_geom": tool,
                    "robot_body": name,
                    "distance": distance,
                })
        return contacts

    def _set_candidate(
        self,
        *,
        joints: Sequence[float],
        tool_pose: Pose,
        base: Sequence[float] | None,
    ) -> None:
        values = np.asarray(joints, dtype=float)
        pose = np.asarray((*tool_pose.position, *tool_pose.orientation), dtype=float)
        if values.shape != (13,) or not np.isfinite(values).all():
            raise ValueError("Expected 13 finite planning joints")
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError("Expected a finite world tool pose")
        quat = pose[3:]
        norm = np.linalg.norm(quat)
        if norm < 1e-12:
            raise ValueError("Tool quaternion must be nonzero")
        self.data.qpos[:] = self.live_data.qpos
        self.data.qpos[self.joint_addresses] = values
        if base is not None:
            base_values = np.asarray(base, dtype=float)
            if base_values.shape != (3,) or not np.isfinite(base_values).all():
                raise ValueError("Expected finite base x/y/yaw")
            self.data.qpos[self.base_addresses] = base_values
        adr = self.tool_address
        self.data.qpos[adr : adr + 3] = pose[:3]
        self.data.qpos[adr + 3 : adr + 7] = (quat / norm)[[3, 0, 1, 2]]
        mujoco.mj_kinematics(self.model, self.data)

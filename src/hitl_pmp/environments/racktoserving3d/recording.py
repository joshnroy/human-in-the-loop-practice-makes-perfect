"""RackToServing3D's replay renderer: a storyboard, or full motion when recorded.

The rack is built on SweepOffShelf3D's machinery, so is its recording. Without
`--record-motion` each recorded state is the action-boundary `features()`: the robot's
base pose and gripper command, each dumpling's full pose, the scraper's position and
yaw, and the rack's and plate's live support bounds -- a storyboard, one frame per
action, arm at home. With it, every `motion_stride()`-th control step is recorded as
`RackServingBackend.motion_state()`: every robot joint and every free body's pose
(five dumplings, the scraper, the plate and the rack), played one tick per frame at
`MOTION_FPS`, which is full motion at real time.

The drawing scene is rebuilt from the variant config the header carries, in the lab2
room whatever the run used, seen from a free camera that frames the rack, the plate
and the robot together.
"""

from typing import Any, ClassVar

import numpy as np
from pydantic import PrivateAttr

from hitl_pmp.environments.sweepoffshelf3d.recording import SweepOffShelf3DReplayRenderer
from hitl_pmp.recording.run_record.types import PlainState

from .backend import RackServingBackend
from .config import RackServingConfig
from .objects import RackServing3DObjects as Objects

MOVABLES = (*(food.name for food in Objects.food), "wiper_0", "box_0", "rack_0")


class RackServingReplayRenderer(SweepOffShelf3DReplayRenderer):
    """Poses a drawing rack scene from recorded states -- no physics is stepped."""

    # A free camera beyond the counter's near-left corner, high enough that the robot
    # stays in frame at every stance a skill drives it to (base x in [0, 2.3], y in
    # [-1.3, 0] in a recorded run): (lookat x, y, z, distance, azimuth, elevation deg).
    # The shelf's `task_view` looks along the counter and leaves the robot out of frame.
    CAMERA: ClassVar[tuple[float, ...]] = (1.2, -0.2, 0.7, 3.1, -110.0, -38.0)

    _backend: RackServingBackend | None = PrivateAttr(default=None)  # type: ignore[assignment]
    _camera: Any = PrivateAttr(default=None)
    _home_centres: dict[str, np.ndarray] = PrivateAttr(default_factory=dict)

    def show(self, *, state: PlainState) -> np.ndarray:
        import mujoco

        backend = self._built()
        model, data = backend.model(), backend.data()
        if "robot_joints" in state:
            self._pose_motion(state=state)
        else:
            self._pose_boundary(state=state)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        if self._renderer is None:
            model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), self.width)
            model.vis.global_.offheight = max(int(model.vis.global_.offheight), self.height)
            self._renderer = mujoco.Renderer(model, height=self.height, width=self.width)
            camera = mujoco.MjvCamera()
            *lookat, camera.distance, camera.azimuth, camera.elevation = self.CAMERA
            camera.lookat[:] = lookat
            self._camera = camera
        self._renderer.update_scene(data, camera=self._camera)
        return np.asarray(self._renderer.render(), dtype=np.uint8).copy()

    def _pose_motion(self, *, state: PlainState) -> None:
        """Every robot joint and every free body's pose, exactly as recorded."""
        backend = self._built()
        model, data = backend.model(), backend.data()
        names = (self.motion() or {}).get("robot_joints", RackServingBackend.ROBOT_JOINTS)
        for joint, value in zip(names, state["robot_joints"], strict=True):
            data.qpos[model.jnt_qposadr[model.joint(joint).id]] = float(value)
        backend.grip = float(state[Objects.robot.name][3])
        for name in MOVABLES:
            address = model.jnt_qposadr[model.joint(f"{name}_joint").id]
            data.qpos[address : address + 7] = np.asarray(state[name][-7:], dtype=float)

    def _pose_boundary(self, *, state: PlainState) -> None:
        """A storyboard state: base and gripper, each dumpling's full pose, the scraper
        by position and yaw, the rack and plate shifted by their bounds' centres."""
        backend = self._built()
        model, data = backend.model(), backend.data()
        x, y, theta, grip = (float(v) for v in state[Objects.robot.name])
        for axis, value in (("x", x), ("y", y), ("th", theta)):
            data.qpos[model.jnt_qposadr[model.joint(f"robot_joint_{axis}").id]] = value
        for side in ("left", "right"):
            for joint, closed in SweepOffShelf3DReplayRenderer._GRIPPER.items():
                address = model.jnt_qposadr[model.joint(f"robot_{side}_{joint}").id]
                data.qpos[address] = closed * grip
        backend.grip = grip
        for food in Objects.food:
            address = model.jnt_qposadr[model.joint(f"{food.name}_joint").id]
            data.qpos[address : address + 7] = np.asarray(state[food.name][:7], dtype=float)
        wiper = np.asarray(state["wiper_0"], dtype=float)
        self._place(name="wiper_0", position=wiper[:3], turn=float(wiper[3]) - self._home_wiper_yaw)
        for name in ("rack_0", "box_0"):
            x0, y0, x1, y1, _ = (float(v) for v in state[name])
            shift = np.array([(x0 + x1) / 2, (y0 + y1) / 2]) - self._home_centres[name]
            home_position, _ = self._home[name]
            self._place(name=name, position=home_position + np.array([*shift, 0.0]))

    def _built(self) -> RackServingBackend:  # type: ignore[override]
        if self._backend is None:
            scene_bg = self.scene.get("scene_bg", True) if self.scene_bg is None else self.scene_bg
            config = RackServingConfig.model_validate({
                name: value
                for name, value in self.scene.items()
                if name in RackServingConfig.model_fields
            })
            backend = RackServingBackend(config=config, scene_bg=bool(scene_bg))
            backend.reset(seed=int(self.scene.get("canonical_seed", 0)))
            core = backend.core()
            for name in MOVABLES:
                position, quaternion = core.get_object(name).get_pose()
                self._home[name] = (
                    np.asarray(position, dtype=float).copy(),
                    np.asarray(quaternion, dtype=float).copy(),
                )
            features = backend.features()
            for name in ("rack_0", "box_0"):
                x0, y0, x1, y1, _ = features[name]
                self._home_centres[name] = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
            self._home_wiper_yaw = float(features["wiper_0"][3])
            self._backend = backend
        return self._backend

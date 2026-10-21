"""SweepOffShelf3D's replay renderer: a storyboard, or full motion when recorded.

By default the shelf's step-protocol recording holds one state per action boundary: the
robot's base pose and gripper command, each cube's position, the wiper's position and
yaw, and the box's interior. It has no arm joints and no per-control-step ticks, so the
replay is a storyboard -- one frame per action, the arm drawn at its home pose.

A recording made with `--record-motion` says so in its header (`scene.motion`) and holds
every `motion_stride()`-th control step of each skill, each state also carrying every
robot joint and each movable's full pose (`SweepOffShelf3DBackend.motion_state`). Played
one tick per frame at `MOTION_FPS`, it is full motion at real time.
"""

from typing import Any, ClassVar

import numpy as np
from pydantic import PrivateAttr

from hitl_pmp.recording.replay.renderer import ReplayRenderer
from hitl_pmp.recording.run_record.types import PlainState

from .backend import SweepOffShelf3DBackend
from .objects import SweepOffShelf3DObjects as O


class SweepOffShelf3DReplayRenderer(ReplayRenderer):
    """Poses a drawing scene from recorded shelf states -- no physics is stepped.

    `scene_bg` overrides the recorded value: presentation videos use the lab2 room
    whatever the run used. Built on first use, so choosing it costs nothing."""

    storyboard: ClassVar[bool] = True
    # Frames per second of full-motion playback; the recording's stride is chosen from
    # it, so one recorded tick is one frame at real time.
    MOTION_FPS: ClassVar[int] = 10
    # The Robotiq linkage at a gripper command g in [0, 1], read off the live sim
    # (closed: driver 0.787, spring link 0.785, follower -0.766, coupler 0).
    _GRIPPER: ClassVar[dict[str, float]] = {
        "driver_joint": 0.787,
        "spring_link_joint": 0.785,
        "follower_joint": -0.766,
    }

    scene: dict[str, Any]
    scene_bg: bool | None = True
    # Frames per second of the storyboard: each action's frame shows 1 / fps seconds,
    # and a human reset holds for the replayer's pause.
    frames_per_second: int = 2
    width: int = 960
    height: int = 720

    _backend: SweepOffShelf3DBackend | None = PrivateAttr(default=None)
    _home: dict[str, tuple[np.ndarray, np.ndarray]] = PrivateAttr(default_factory=dict)
    _home_box_center: np.ndarray = PrivateAttr(default_factory=lambda: np.zeros(2))
    _home_wiper_yaw: float = PrivateAttr(default=0.0)
    _renderer: Any = PrivateAttr(default=None)

    @staticmethod
    def motion_stride() -> int:
        """Control steps per recorded tick, so that ticks arrive at `MOTION_FPS`."""
        return max(
            1,
            round(
                SweepOffShelf3DBackend.CONTROL_FREQUENCY / SweepOffShelf3DReplayRenderer.MOTION_FPS
            ),
        )

    def motion(self) -> dict[str, Any] | None:
        return self.scene.get("motion")

    def is_storyboard(self) -> bool:
        return self.motion() is None

    def fps(self) -> int:
        motion = self.motion()
        return self.frames_per_second if motion is None else int(motion["fps"])

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
            # The compiled model's offscreen buffer is the env's 640x480; the
            # presentation canvas draws the scene at 960x720.
            model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), self.width)
            model.vis.global_.offheight = max(int(model.vis.global_.offheight), self.height)
            self._renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        self._renderer.update_scene(data, camera="task_view")
        return np.asarray(self._renderer.render(), dtype=np.uint8).copy()

    def _pose_motion(self, *, state: PlainState) -> None:
        """Every robot joint and every movable's free joint, exactly as recorded."""
        backend = self._built()
        model, data = backend.model(), backend.data()
        joints = self.motion() or {}
        names = joints.get("robot_joints", SweepOffShelf3DBackend.ROBOT_JOINTS)
        for joint, value in zip(names, state["robot_joints"], strict=True):
            data.qpos[model.jnt_qposadr[model.joint(joint).id]] = float(value)
        backend.grip = float(state[O.robot.name][3])
        for name in (*backend.cube_names, backend.wiper_name, backend.box_name):
            address = model.jnt_qposadr[model.joint(f"{name}_joint").id]
            data.qpos[address : address + 7] = np.asarray(state[name][-7:], dtype=float)

    def _pose_boundary(self, *, state: PlainState) -> None:
        backend = self._built()
        model, data = backend.model(), backend.data()
        x, y, theta, grip = (float(v) for v in state[O.robot.name])
        for axis, value in (("x", x), ("y", y), ("th", theta)):
            data.qpos[model.jnt_qposadr[model.joint(f"robot_joint_{axis}").id]] = value
        for side in ("left", "right"):
            for joint, closed in SweepOffShelf3DReplayRenderer._GRIPPER.items():
                address = model.jnt_qposadr[model.joint(f"robot_{side}_{joint}").id]
                data.qpos[address] = closed * grip
        backend.grip = grip
        for name in backend.cube_names:
            self._place(name=name, position=np.asarray(state[name], dtype=float))
        wiper = np.asarray(state[backend.wiper_name], dtype=float)
        self._place(
            name=backend.wiper_name,
            position=wiper[:3],
            turn=float(wiper[3]) - self._home_wiper_yaw,
        )
        x0, y0, x1, y1, _ = (float(v) for v in state[backend.box_name])
        shift = np.array([(x0 + x1) / 2, (y0 + y1) / 2]) - self._home_box_center
        home_position, _ = self._home[backend.box_name]
        self._place(name=backend.box_name, position=home_position + np.array([*shift, 0.0]))

    def current_state(self) -> PlainState:
        """Read back from the posed scene itself, as the live environment observes it."""
        backend = self._built()
        if self.motion() is not None:
            return backend.motion_state()
        return {name: [float(v) for v in values] for name, values in backend.features().items()}

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._backend is not None:
            self._backend.close()
            self._backend = None

    def _built(self) -> SweepOffShelf3DBackend:
        if self._backend is None:
            scene_bg = self.scene.get("scene_bg", True) if self.scene_bg is None else self.scene_bg
            backend = SweepOffShelf3DBackend(scene_bg=bool(scene_bg))
            backend.reset(seed=int(self.scene.get("canonical_seed", 0)))
            core = backend.core()
            for name in (*backend.cube_names, backend.wiper_name, backend.box_name):
                position, quaternion = core.get_object(name).get_pose()
                self._home[name] = (
                    np.asarray(position, dtype=float).copy(),
                    np.asarray(quaternion, dtype=float).copy(),
                )
            x0, y0, x1, y1, _ = backend.features()[backend.box_name]
            self._home_box_center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
            self._home_wiper_yaw = float(backend.features()[backend.wiper_name][3])
            self._backend = backend
        return self._backend

    def _place(self, *, name: str, position: np.ndarray, turn: float = 0.0) -> None:
        """Put `name` at `position`, at its home orientation turned by `turn` about z."""
        assert self._backend is not None
        _, home = self._home[name]
        half = turn / 2
        w0, x0, y0, z0 = home
        cz, sz = np.cos(half), np.sin(half)
        quaternion = [
            cz * w0 - sz * z0,
            cz * x0 - sz * y0,
            cz * y0 + sz * x0,
            cz * z0 + sz * w0,
        ]
        obj = self._backend.core().get_object(name)
        obj.set_pose(list(map(float, position)), quaternion)
        obj.set_velocity([0.0] * 3, [0.0] * 3)

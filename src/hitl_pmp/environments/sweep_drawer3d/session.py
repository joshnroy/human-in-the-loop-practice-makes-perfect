"""A live KINDER SweepIntoDrawer3D-o5 episode: stepping, state queries and the tick log.

The only module here that imports KINDER at call time (never at import time), so the
package imports, typechecks and tests without it -- the same rule as
environments/tossing3d/kinder_backend.py.
"""

import json
import os
import time
from pathlib import Path
from typing import IO, Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr
from scipy.spatial.transform import Rotation

from .types import CubeLocation, ResetStep, SweepDrawerScene


class KinderImports:
    """Imports KINDER with the rendering-backend ordering CLAUDE.md documents."""

    @staticmethod
    def load() -> Any:
        backend = os.environ.get("MUJOCO_GL", "egl"), os.environ.get("PYOPENGL_PLATFORM", "egl")
        import kinder
        import kinder.envs.dynamic3d.envs  # noqa: F401  (the MODULE: it pulls in mujoco)

        kinder.register_all_environments()
        os.environ["MUJOCO_GL"], os.environ["PYOPENGL_PLATFORM"] = backend
        return kinder


class CubePlaces:
    """Places a cube can rest that move when the drawer does."""

    HANDLE_TOP: ClassVar[float] = 0.328
    # how far a cube may sit off the ledge's or the bar's middle and still rest on it
    SLACK: ClassVar[float] = 0.012

    @staticmethod
    def rides_drawer_front(*, position: np.ndarray, drawer_pos: float) -> bool:
        """On top of the task drawer's face -- a 2 cm ledge between the countertop's edge
        and the handle -- or on the handle's bar. Slide the drawer and such a cube goes
        with it, which is how it is brought clear of the countertop's edge."""
        s = SweepDrawerScene
        x, y, z = (float(v) for v in position)
        if abs(y) > s.DRAWER_HALF_WIDTH:
            return False
        rest = s.CUBE_HALF
        on_face = (
            abs(z - (s.DRAWER_WALL_TOP + rest)) < CubePlaces.SLACK
            and s.COUNTER_EDGE_X + drawer_pos - CubePlaces.SLACK
            < x
            < s.DRAWER_FACE_X + drawer_pos + CubePlaces.SLACK
        )
        on_handle = (
            abs(z - (CubePlaces.HANDLE_TOP + rest)) < CubePlaces.SLACK
            and s.DRAWER_FACE_X + drawer_pos - CubePlaces.SLACK
            < x
            < s.DRAWER_HANDLE_FRONT_X + drawer_pos + CubePlaces.SLACK
        )
        return on_face or on_handle


class SweepDrawerSession(BaseModel):
    """One seeded episode of SweepIntoDrawer3D-o5 with a per-tick state log."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    seed: int
    log_path: Path | None = None
    # Every joint position at every tick, for drawing the run afterwards. The state log
    # cannot serve: it records the gripper's command, not where the fingers are, and
    # only the one drawer the task opens.
    replay_path: Path | None = None

    _env: Any = PrivateAttr(default=None)
    _state: Any = PrivateAttr(default=None)
    _ticks: int = PrivateAttr(default=0)
    _log: IO[str] | None = PrivateAttr(default=None)
    _replay: IO[str] | None = PrivateAttr(default=None)
    _steps: list[ResetStep] = PrivateAttr(default_factory=list)
    _current: dict[str, Any] | None = PrivateAttr(default=None)
    _initial: dict[str, tuple[np.ndarray, tuple[float, float, float, float]]] = PrivateAttr(
        default_factory=dict
    )

    def model_post_init(self, __context: Any) -> None:  # noqa: PLR0917
        kinder = KinderImports.load()
        self._env = kinder.make(
            SweepDrawerScene.ENV_ID, render_mode="rgb_array", allow_state_access=True
        )
        obs, _ = self._env.reset(seed=self.seed)
        self._state = self._env.observation_space.devectorize(obs)
        self._initial = {
            n: (self.position(name=n), self.quaternion(name=n))
            for n in (SweepDrawerScene.WIPER, *SweepDrawerScene.CUBES)
        }
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = self.log_path.open("w")
            logged = self.logged_objects()
            self._write(
                record={
                    "kind": "header",
                    "env": SweepDrawerScene.ENV_ID,
                    "seed": self.seed,
                    "logged_objects": logged,
                    "features": {
                        n: list(self._state.type_features[self._obj(name=n).type]) for n in logged
                    },
                }
            )
        if self.replay_path is not None:
            self.replay_path.parent.mkdir(parents=True, exist_ok=True)
            self._replay = self.replay_path.open("w")
            self._replay.write(json.dumps(self.replay_header()) + "\n")
        self._tick()

    # ------------------------------------------------------------------ plumbing
    @property
    def env(self) -> Any:
        return self._env

    @property
    def state(self) -> Any:
        return self._state

    @property
    def ticks(self) -> int:
        return self._ticks

    @property
    def steps(self) -> list[ResetStep]:
        return self._steps

    @property
    def mj_model(self) -> Any:
        return self._env.unwrapped._object_centric_env._robot_env.sim.model.mj_model

    @property
    def mj_data(self) -> Any:
        return self._env.unwrapped._object_centric_env._robot_env.sim.data.mj_data

    def initial_pose(self, *, name: str) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        """Where `name` was at episode start: the wiper's parking spot, for one."""
        return self._initial[name]

    def restore(self, *, state: Any) -> None:
        """Overwrite the simulator from an object-centric state (tests and replay only)."""
        self._env.unwrapped._object_centric_env.set_state(state)
        self.step(action=np.zeros(11))

    def logged_objects(self) -> list[str]:
        s = SweepDrawerScene
        return [s.ROBOT, s.WIPER, s.DRAWER, *s.CUBES]

    def close(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None
        if self._replay is not None:
            self._replay.close()
            self._replay = None
        self._env.close()

    def replay_header(self) -> dict[str, Any]:
        """Which joint each logged position belongs to, so a replay can refuse a scene
        whose joints have moved under the same names."""
        import mujoco

        m = self.mj_model
        sizes = {
            int(mujoco.mjtJoint.mjJNT_FREE): 7,
            int(mujoco.mjtJoint.mjJNT_BALL): 4,
            int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
            int(mujoco.mjtJoint.mjJNT_HINGE): 1,
        }
        joints = [
            {
                "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}",
                "adr": int(m.jnt_qposadr[j]),
                "n": sizes[int(m.jnt_type[j])],
            }
            for j in range(m.njnt)
        ]
        return {
            "kind": "header",
            "env": SweepDrawerScene.ENV_ID,
            "seed": self.seed,
            "nq": int(m.nq),
            "joints": joints,
        }

    def _write(self, *, record: dict[str, Any]) -> None:
        if self._log is not None:
            self._log.write(json.dumps(record) + "\n")

    def _tick(self) -> None:
        name = None if self._current is None else self._current["name"]
        if self._replay is not None:
            qpos = [round(float(v), 5) for v in self.mj_data.qpos]
            record = {"kind": "tick", "t": self._ticks, "step": name, "qpos": qpos}
            self._replay.write(json.dumps(record) + "\n")
        if self._log is None:
            return
        snapshot = {
            n: [round(float(v), 5) for v in self._state[self._obj(name=n)]]
            for n in self.logged_objects()
        }
        self._write(record={"kind": "tick", "t": self._ticks, "step": name, "state": snapshot})

    # ------------------------------------------------------------------ stepping
    def step(self, *, action: np.ndarray, clip: bool = True) -> Any:
        """One 10 Hz control step. Our own primitives stay inside the declared action
        space; kinder-models' sweep3D skills emit arm deltas outside it and depend on
        that, so they are passed through unclipped (`clip=False`)."""
        a = np.asarray(action, dtype=np.float32)
        if clip:
            a = np.clip(a, self._env.action_space.low, self._env.action_space.high)
        obs, *_ = self._env.step(a)
        self._state = self._env.observation_space.devectorize(obs)
        self._ticks += 1
        self._tick()
        return self._state

    def begin(self, *, name: str, kind: str, phase: str) -> None:
        self._current = {
            "name": name,
            "kind": kind,
            "phase": phase,
            "t0": time.perf_counter(),
            "tick0": self._ticks,
        }

    def end(self, *, success: bool, note: str = "") -> ResetStep:
        cur = self._current
        assert cur is not None, "end() without begin()"
        record = ResetStep(
            index=len(self._steps),
            name=cur["name"],
            phase=cur["phase"],
            kind=cur["kind"],
            success=success,
            ticks=self._ticks - cur["tick0"],
            wall_s=round(time.perf_counter() - cur["t0"], 2),
            locations=self.locations(),
            drawer_pos=round(self.drawer_pos(), 4),
            note=note,
        )
        self._steps.append(record)
        self._current = None
        return record

    # ------------------------------------------------------------------ state queries
    def _obj(self, *, name: str) -> Any:
        return self._state.get_object_from_name(name)

    def position(self, *, name: str) -> np.ndarray:
        o = self._obj(name=name)
        return np.array([self._state.get(o, k) for k in ("x", "y", "z")], dtype=float)

    def quaternion(self, *, name: str) -> tuple[float, float, float, float]:
        """(x, y, z, w), the PyBullet / scipy convention."""
        o = self._obj(name=name)
        qx, qy, qz, qw = (float(self._state.get(o, k)) for k in ("qx", "qy", "qz", "qw"))
        return (qx, qy, qz, qw)

    def yaw(self, *, name: str) -> float:
        return float(Rotation.from_quat(self.quaternion(name=name)).as_euler("zyx")[0])

    def arm(self) -> np.ndarray:
        r = self._obj(name=SweepDrawerScene.ROBOT)
        return np.array([self._state.get(r, f"pos_arm_joint{i}") for i in range(1, 8)], dtype=float)

    def base(self) -> tuple[float, float, float]:
        r = self._obj(name=SweepDrawerScene.ROBOT)
        return tuple(
            float(self._state.get(r, k)) for k in ("pos_base_x", "pos_base_y", "pos_base_rot")
        )  # type: ignore[return-value]

    def gripper(self) -> float:
        """The gripper COMMAND as the state reports it (0 open .. 1 closed). It tracks the
        command immediately, not the fingers -- use finger_closure() to see the fingers."""
        return float(self._state.get(self._obj(name=SweepDrawerScene.ROBOT), "pos_gripper"))

    def finger_closure(self) -> float:
        """Mean driver-joint angle of the two Robotiq fingers (0 open .. ~0.8 closed)."""
        import mujoco

        m, d = self.mj_model, self.mj_data
        vals = []
        for side in ("left", "right"):
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot_{side}_driver_joint")
            vals.append(float(d.qpos[m.jnt_qposadr[j]]))
        return float(np.mean(vals))

    def drawer_pos(self) -> float:
        return float(self._state.get(self._obj(name=SweepDrawerScene.DRAWER), "pos"))

    def locations(self) -> dict[str, CubeLocation]:
        return {c: self.location(cube=c) for c in SweepDrawerScene.CUBES}

    def location(self, *, cube: str) -> CubeLocation:
        """drawer / counter / floor / other, from the cube's pose and the drawer opening."""
        s = SweepDrawerScene
        x, y, z = self.position(name=cube)
        dp = self.drawer_pos()
        if z < 0.08:
            return "floor"
        in_drawer_x = 0.13 + dp < x < s.DRAWER_FRONT_INNER_X + dp + 0.01
        if 0.2 < z < 0.43 and abs(y) < s.DRAWER_SIDE_INNER_Y + 0.002 and in_drawer_x:
            return "drawer"
        if 0.45 < z < 0.52 and 0.125 < x < s.COUNTER_EDGE_X and abs(y) < 1.0:
            return "counter"
        return "other"

    def rides_drawer_front(self, *, cube: str) -> bool:
        return CubePlaces.rides_drawer_front(
            position=self.position(name=cube), drawer_pos=self.drawer_pos()
        )

    def in_pile(self, *, cube: str) -> bool:
        """Resting on the countertop inside the task's `blocks_init_region`."""
        s = SweepDrawerScene
        x, y, z = (float(v) for v in self.position(name=cube))
        return (
            s.PILE_X[0] <= x <= s.PILE_X[1] and s.PILE_Y[0] <= y <= s.PILE_Y[1] and 0.45 < z < 0.5
        )

    def elevated_movables(self) -> list[str]:
        """Movables that are not base obstacles: those above the chassis (countertop,
        drawer) and those low enough to pass under it. The compiled chassis collides only
        from z = 0.085 up (it has no wheel geometry), so a 2 cm cube or the 1.5 cm-thick
        wiper on the floor passes beneath it in this model; a real base's wheels would not
        -- recorded as a sim-to-real gap, not hidden."""
        names = (SweepDrawerScene.WIPER, *SweepDrawerScene.CUBES)
        return [
            n for n in names if self.position(name=n)[2] > 0.15 or self.position(name=n)[2] < 0.05
        ]

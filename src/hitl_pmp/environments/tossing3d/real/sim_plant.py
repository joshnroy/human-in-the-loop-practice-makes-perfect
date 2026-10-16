"""The KINDER simulation as the real robot's plant, for a dry run of the real loop.

`SimPlantTossingRobot` implements `TossingRobot` -- the real robot side's contract -- by
driving the existing KINDER Tossing3D simulation, so the whole practice loop can run
through `RealTossingBackend`, the real oracle's checks and the real CLI with nothing
real attached. Every pick and toss outcome is the simulator's physics for the same
parameters: there is no outcome model here.

## The real room, laid into the sim's

The two rooms differ by a rigid motion. The real robot works east of a barrier at
x = 0 with the bin to the west; the sim robot works west of its barrier at x ~ 1.3
with the bin to the east. A rotation by pi about z, then a shift onto the sim's
barrier, maps one onto the other with handedness kept:

    x_sim = barrier_x_sim - x_real,  y_sim = -y_real,  yaw_sim = yaw_real + pi

and is its own inverse in x and y. Everything crossing the contract -- poses in
`observe()`, placement targets, the toss `direction` -- goes through it, so the backend
above sees real-frame numbers only.

## What the plant does not reproduce, stated

- **The parameters run through the sim controller outside its own declared ranges.**
  The real set (standoff 0.9-1.45 m, 100-140 deg/s, 500-850 ms) overlaps the sim's
  (1.25-2.6 m, 115-420 deg/s, 400-840 ms) only in part. Upstream's controller declares
  its bounds but does not enforce them in `reset`, so a real draw executes as given;
  whether the sim's physics at, say, 0.9 m and 100 deg/s resemble the real swing is
  not established here. The dry run proves wiring, not transfer.
- **The barrier.** The sim's barrier spans its room; the real one is a finite
  rectangle. `observe()` reports the configured real rectangle, as the robot side
  would; the sim's own barrier is what the plant's base actually cannot cross.
- **The bin's size** is the sim's (0.30 x 0.30 x 0.20 m) and its fit is exact.
- **Hidden cubes** do not happen: the simulator always knows where the cube is.
"""

import math
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr

from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend
from hitl_pmp.environments.tossing3d.types import TossParameterSet
from hitl_pmp.environments.tossing3d.wide_long_range_proposal import REAL_TOSS_PARAMETERS

from .types import BarrierRect, PlacementRequest

_ROBOT = "robot"
_CUBE = "cube_0"
_BIN = "bin_0"
_BARRIER = "cuboid_barrier"


class SimPlantTossingRobot(BaseModel):
    """`TossingRobot` over a live KINDER Tossing3D scene. See the module docstring."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    barrier: BarrierRect = BarrierRect()
    parameters: TossParameterSet = REAL_TOSS_PARAMETERS
    scene_seed: int = 125
    scene_bg: bool = False

    bin_length: ClassVar[float] = 0.30
    bin_width: ClassVar[float] = 0.30
    bin_height: ClassVar[float] = 0.20
    cube_half: ClassVar[float] = 0.025

    _backend: KinderBackend | None = PrivateAttr(default=None)
    _sim_barrier_x: float = PrivateAttr(default=1.3)
    _start: tuple[float, float, float] | None = PrivateAttr(default=None)

    # --- the contract -------------------------------------------------------------------

    def observe(self) -> dict[str, Any]:
        backend = self._sim()
        features = backend.observe().features
        atoms = backend.abstract_atoms()
        robot_name = backend.robot_name
        robot = features[robot_name]
        cube = features[_CUBE]
        bin_ = features[_BIN]
        holding = ("Holding", ("robot", _CUBE)) in atoms
        bin_x, bin_y, bin_yaw = self._to_real(
            x=bin_["x"], y=bin_["y"], yaw=self._yaw(features=bin_)
        )
        return {
            "robot": list(
                self._to_real(
                    x=robot["pos_base_x"], y=robot["pos_base_y"], yaw=robot["pos_base_rot"]
                )
            ),
            "cube": list(self._to_real(x=cube["x"], y=cube["y"], yaw=self._yaw(features=cube))),
            "cube_upright": ("OnGround", (_CUBE,)) in atoms,
            "bin": {
                "x": bin_x,
                "y": bin_y,
                "yaw": bin_yaw,
                "length": self.bin_length,
                "width": self.bin_width,
                "height": self.bin_height,
            },
            "holding": holding,
            "gripper": robot["pos_gripper"],
            "gripper_command": robot["pos_gripper"],
            "cube_in_bin": ("MovableInGoalRegion", (_CUBE,)) in atoms,
            "barrier": self.barrier.model_dump(),
            "barrier_x": self.barrier.x,
        }

    def can_pick(self) -> dict[str, Any]:
        failure = self._sim().pick_cube_plan_failure()
        return {"feasible": failure is None, "note": failure}

    def pick(self) -> dict[str, Any]:
        backend = self._sim()
        run = backend.run_pick_cube()
        held = ("Holding", ("robot", _CUBE)) in backend.abstract_atoms()
        gripper = backend.observe().features[backend.robot_name]["pos_gripper"]
        return {"held": held, "gripper": gripper, "note": run.error}

    def open_gripper(self) -> None:
        self._sim().run_open_gripper()

    def toss(
        self, *, standoff: float, speed: float, release_ms: float, direction: float
    ) -> dict[str, Any]:
        violation = self.parameters.violation(params=(standoff, speed, release_ms))
        if violation is not None:
            raise ValueError(f"outside the real bounds: {violation}")
        backend = self._sim()
        if ("Holding", ("robot", _CUBE)) not in backend.abstract_atoms():
            return {"tossed": False, "landed": None, "in_bin": False, "note": "not holding"}
        bin_ = backend.observe().features[_BIN]
        sim_bin_yaw = self._yaw(features=bin_)
        # Real azimuth bin -> stand, rotated by pi, is the sim's; the controller's stand
        # is at azimuth sim_bin_yaw + rotation + pi, so rotation = direction - bin yaw.
        rotation = math.atan2(math.sin(direction - sim_bin_yaw), math.cos(direction - sim_bin_yaw))
        run = backend.run_move_to_toss_location_and_toss(
            distance=standoff,
            rotation=rotation,
            release_speed_deg_s=speed,
            gripper_release_ms=release_ms,
        )
        if run.steps == 0:
            return {"tossed": False, "landed": None, "in_bin": False, "note": run.error}
        cube = backend.observe().features[_CUBE]
        x, y, _ = self._to_real(x=cube["x"], y=cube["y"], yaw=0.0)
        return {
            "tossed": True,
            "landed": [x, y],
            "in_bin": ("MovableInGoalRegion", (_CUBE,)) in backend.abstract_atoms(),
            "note": run.error,
        }

    def retract(self) -> None:
        """The sim's toss controller ends where it ends; there is no arm to stow."""

    def go_home(self) -> None:
        """Back to the scene's own start pose, gripper open."""
        self._sim()
        assert self._start is not None
        self._edit(robot_pose=self._start)

    def stop(self) -> None:
        """Nothing runs between calls in the sim, so there is nothing to stop."""

    # --- the person's hands -------------------------------------------------------------

    def place(self, *, request: PlacementRequest) -> None:
        """Put the cube and the bin at the request's (real-frame) targets, exactly."""
        cube = request.cube_target
        bin_ = request.bin_target
        self._edit(
            cube_xy=None if cube is None else self._to_sim_xy(x=cube[0], y=cube[1]),
            bin_xy=None if bin_ is None else self._to_sim_xy(x=bin_[0], y=bin_[1]),
        )

    def in_bin(self) -> bool:
        return ("MovableInGoalRegion", (_CUBE,)) in self._sim().abstract_atoms()

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()

    # --- helpers ------------------------------------------------------------------------

    def _sim(self) -> KinderBackend:
        if self._backend is None:
            backend = KinderBackend(scene_bg=self.scene_bg)
            observation = backend.reset(seed=self.scene_seed)
            robot = observation.features[backend.robot_name]
            self._start = (robot["pos_base_x"], robot["pos_base_y"], robot["pos_base_rot"])
            self._sim_barrier_x = observation.features[_BARRIER]["x"]
            self._backend = backend
        return self._backend

    def _edit(
        self,
        *,
        cube_xy: tuple[float, float] | None = None,
        bin_xy: tuple[float, float] | None = None,
        robot_pose: tuple[float, float, float] | None = None,
    ) -> None:
        """Rewrite object poses in the live scene, through the backend's own restore."""
        backend = self._sim()
        snapshot = backend.live_state()
        plain = backend.snapshot_to_plain(snapshot=snapshot)
        names = {obj.name: list(snapshot.type_features[obj.type]) for obj in snapshot}

        def write(*, name: str, **values: float) -> None:
            for feature, value in values.items():
                if feature in names[name]:
                    plain[name][names[name].index(feature)] = value

        zeros = {f: 0.0 for f in ("vx", "vy", "vz", "wx", "wy", "wz")}
        if cube_xy is not None:
            write(
                name=_CUBE,
                x=cube_xy[0],
                y=cube_xy[1],
                z=self.cube_half,
                qx=0.0,
                qy=0.0,
                qz=0.0,
                qw=1.0,
                **zeros,
            )
        if bin_xy is not None:
            write(name=_BIN, x=bin_xy[0], y=bin_xy[1], **zeros)
        if robot_pose is not None:
            write(
                name=backend.robot_name,
                pos_base_x=robot_pose[0],
                pos_base_y=robot_pose[1],
                pos_base_rot=robot_pose[2],
                pos_gripper=0.0,
                vel_base_x=0.0,
                vel_base_y=0.0,
                vel_base_rot=0.0,
            )
        backend.restore(snapshot=backend.plain_to_snapshot(plain=plain, dtype=np.float64))

    def _to_real(self, *, x: float, y: float, yaw: float) -> tuple[float, float, float]:
        angle = yaw - math.pi
        return (
            self._sim_barrier_x - x,
            -y,
            math.atan2(math.sin(angle), math.cos(angle)),
        )

    def _to_sim_xy(self, *, x: float, y: float) -> tuple[float, float]:
        return (self._sim_barrier_x - x, -y)

    @staticmethod
    def _yaw(*, features: dict[str, float]) -> float:
        qx, qy, qz, qw = (features[key] for key in ("qx", "qy", "qz", "qw"))
        return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


class SimPlantPlacer(BaseModel):
    """The scripted person for the dry run: honours every request exactly, and looks
    into the bin by asking the simulator."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    robot: SimPlantTossingRobot

    def place(self, *, request: PlacementRequest) -> None:
        self.robot.place(request=request)

    def inspect_cube_in_bin(self, *, owner: str) -> bool:
        del owner
        return self.robot.in_bin()

    def continue_after_unverified_reset(
        self, *, owner: str, unmet: tuple[str, ...], attempts: int
    ) -> bool:
        """The dry run carries on, as an unattended run must."""
        del owner, unmet, attempts
        return True

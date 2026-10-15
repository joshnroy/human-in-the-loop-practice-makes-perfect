"""`FakeTossingRobot`: a deterministic, configurable stand-in for the real client.

It implements `TossingRobot` with the contract's exact reply shapes and its one hard
rule -- `toss` raises `ValueError` outside the real bounds and nothing moves -- over a
toy world of three poses. It does no physics. Outcomes are whatever its two hooks say
(`pick_outcome`, `toss_outcome`), defaulting to simple fixed rules, so a test states
exactly the world it wants. `place` is the human's hands: the scripted placer
(`real/placement.py`) calls it to realize a reset.
"""

import math
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, PrivateAttr

from hitl_pmp.environments.tossing3d.types import TossParameterSet
from hitl_pmp.environments.tossing3d.wide_long_range_proposal import REAL_TOSS_PARAMETERS

from .types import BarrierRect, BinFit, PlacementRequest, RobotPose


class FakeTossOutcome(BaseModel):
    """What one fake toss does: where the cube ends up, and whether it scored."""

    model_config = ConfigDict(frozen=True)

    landed: tuple[float, float]
    in_bin: bool
    upright: bool = True
    # False: the bin's walls hide the landed cube, so `observe()` reports none and the
    # toss reply says `landed: None, in_bin: False` -- as the real robot side does.
    visible: bool = True


class FakeTossingRobot(BaseModel):
    """A toy world behind the real contract. Every field is a knob a test can set."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # `None` is the skill server's barrier disabled.
    barrier: BarrierRect | None = BarrierRect()
    home: RobotPose = RobotPose(x=1.2, y=0.0, yaw=math.pi)
    cube_start: RobotPose | None = RobotPose(x=0.8, y=0.0, yaw=0.0)
    bin_start: BinFit | None = BinFit(x=-0.8, y=0.0, yaw=0.0, length=0.30, width=0.30, height=0.20)
    parameters: TossParameterSet = REAL_TOSS_PARAMETERS
    # (state of the world) -> whether the pick lifts the cube. Default: always.
    pick_outcome: Callable[["FakeTossingRobot"], bool] | None = None
    # (standoff, speed, release_ms, direction, world) -> outcome. Default: scores when
    # speed and release both lie in the middle third of their real ranges.
    toss_outcome: (
        Callable[[float, float, float, float, "FakeTossingRobot"], FakeTossOutcome] | None
    ) = None
    # Whether `can_pick` reports a plan. Default: a visible cube on the robot's side.
    can_pick_outcome: Callable[["FakeTossingRobot"], bool] | None = None
    # Called with the skill's name before any motion skill; may raise, standing in for
    # the client's `SkillStopped`/`SkillBusy`.
    interrupt: Callable[[str], None] | None = None

    _robot: RobotPose = PrivateAttr()
    _cube: RobotPose | None = PrivateAttr(default=None)
    _cube_upright: bool = PrivateAttr(default=True)
    _bin: BinFit | None = PrivateAttr(default=None)
    _holding: bool = PrivateAttr(default=False)
    _gripper_command: float = PrivateAttr(default=0.0)
    _cube_in_bin: bool = PrivateAttr(default=False)
    _cube_hidden: bool = PrivateAttr(default=False)
    _truly_in_bin: bool = PrivateAttr(default=False)
    _calls: list[str] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: object) -> None:
        self._robot = self.home
        self._cube = self.cube_start
        self._bin = self.bin_start

    def calls(self) -> list[str]:
        return list(self._calls)

    def place(
        self,
        *,
        cube: RobotPose | None,
        bin_: BinFit | None,
        robot: RobotPose | None = None,
    ) -> None:
        """The human's hands: put the cube and bin exactly here, gripper empty."""
        self._calls.append("place")
        self._cube_hidden = False
        self._cube = cube
        self._cube_upright = True
        self._bin = bin_
        self._holding = False
        self._gripper_command = 0.0
        self._cube_in_bin = (
            cube is not None and bin_ is not None and self._inside_bin(x=cube.x, y=cube.y)
        )
        if robot is not None:
            self._robot = robot

    def observe(self) -> dict[str, Any]:
        self._calls.append("observe")
        cube = None if self._cube_hidden else self._cube
        bin_ = self._bin
        barrier = self.barrier
        return {
            "robot": [self._robot.x, self._robot.y, self._robot.yaw],
            "cube": None if cube is None else [cube.x, cube.y, cube.yaw],
            "cube_upright": self._cube_upright and cube is not None and not self._holding,
            "bin": None if bin_ is None else bin_.model_dump(),
            "holding": self._holding,
            "gripper": 0.8 if self._holding else self._gripper_command,
            "gripper_command": self._gripper_command,
            "cube_in_bin": self._cube_in_bin,
            "barrier": None if barrier is None else barrier.model_dump(),
            "barrier_x": None if barrier is None else barrier.x,
        }

    def can_pick(self) -> dict[str, Any]:
        self._calls.append("can_pick")
        if self.can_pick_outcome is not None:
            feasible = self.can_pick_outcome(self)
        else:
            cube = self._cube
            feasible = (
                cube is not None
                and not self._holding
                and (cube.x - self._barrier_x()) * (self._robot.x - self._barrier_x()) > 0
            )
        return {"feasible": bool(feasible), "note": None if feasible else "no plan"}

    def pick(self) -> dict[str, Any]:
        self._interrupt(skill="pick")
        self._calls.append("pick")
        self._gripper_command = 1.0
        cube = self._cube
        held = cube is not None and (self.pick_outcome(self) if self.pick_outcome else True)
        if held:
            assert cube is not None
            self._robot = RobotPose(x=cube.x + 0.55, y=cube.y, yaw=math.pi)
            self._holding = True
            self._cube_in_bin = False
        return {"held": held, "gripper": 0.8 if held else 1.0, "note": None if held else "missed"}

    def open_gripper(self) -> None:
        self._interrupt(skill="open_gripper")
        self._calls.append("open_gripper")
        self._gripper_command = 0.0
        if self._holding:
            self._holding = False

    def toss(
        self, *, standoff: float, speed: float, release_ms: float, direction: float
    ) -> dict[str, Any]:
        for name, value, (low, high) in (
            ("standoff", standoff, self.parameters.standoff_bounds),
            ("speed", speed, self.parameters.speed_bounds),
            ("release_ms", release_ms, self.parameters.release_ms_bounds),
        ):
            if not low <= value <= high:
                raise ValueError(f"{name}={value} outside the real bounds [{low}, {high}]")
        self._interrupt(skill="toss")
        self._calls.append(f"toss:{standoff:.3f}:{speed:.1f}:{release_ms:.0f}:{direction:.4f}")
        if not self._holding:
            return {"tossed": False, "landed": None, "in_bin": False, "note": "not holding"}
        if self._bin is None:
            return {"tossed": False, "landed": None, "in_bin": False, "note": "no bin found"}
        bin_ = self._bin
        self._robot = RobotPose(
            x=bin_.x + standoff * math.cos(direction),
            y=bin_.y + standoff * math.sin(direction),
            yaw=direction + math.pi,
        )
        outcome = (
            self.toss_outcome(standoff, speed, release_ms, direction, self)
            if self.toss_outcome is not None
            else self._default_outcome(speed=speed, release_ms=release_ms)
        )
        self._holding = False
        self._gripper_command = 0.0
        self._cube = RobotPose(x=outcome.landed[0], y=outcome.landed[1], yaw=0.0)
        self._cube_upright = outcome.upright
        self._cube_in_bin = outcome.in_bin and outcome.visible
        self._cube_hidden = not outcome.visible
        self._truly_in_bin = outcome.in_bin
        return {
            "tossed": True,
            "landed": list(outcome.landed) if outcome.visible else None,
            "in_bin": outcome.in_bin and outcome.visible,
            "note": None,
        }

    def retract(self) -> None:
        self._calls.append("retract")

    def go_home(self) -> None:
        self._calls.append("go_home")
        self._robot = self.home

    def stop(self) -> None:
        self._calls.append("stop")

    def truly_in_bin(self) -> bool:
        """Ground truth a person would see, hidden cube or not."""
        return self._truly_in_bin if self._cube_hidden else self._cube_in_bin

    def _interrupt(self, *, skill: str) -> None:
        if self.interrupt is not None:
            self.interrupt(skill)

    def _barrier_x(self) -> float:
        return 0.0 if self.barrier is None else self.barrier.x

    def _default_outcome(self, *, speed: float, release_ms: float) -> FakeTossOutcome:
        assert self._bin is not None
        (s_low, s_high), (r_low, r_high) = (
            self.parameters.speed_bounds,
            self.parameters.release_ms_bounds,
        )
        scores = (
            s_low + (s_high - s_low) / 3 <= speed <= s_high - (s_high - s_low) / 3
            and r_low + (r_high - r_low) / 3 <= release_ms <= r_high - (r_high - r_low) / 3
        )
        if scores:
            return FakeTossOutcome(landed=(self._bin.x, self._bin.y), in_bin=True)
        # A miss lands short of the bin, on the bin's side of the barrier.
        return FakeTossOutcome(landed=(self._bin.x, self._bin.y + 0.5), in_bin=False)

    def _inside_bin(self, *, x: float, y: float) -> bool:
        assert self._bin is not None
        bin_ = self._bin
        cosine, sine = math.cos(bin_.yaw), math.sin(bin_.yaw)
        local_x = cosine * (x - bin_.x) + sine * (y - bin_.y)
        local_y = -sine * (x - bin_.x) + cosine * (y - bin_.y)
        return abs(local_x) < bin_.length / 2 - 0.05 and abs(local_y) < bin_.width / 2 - 0.05


class FakeRobotPlacer(BaseModel):
    """A `RealScenePlacer` whose "human" teleports a `FakeTossingRobot`'s world.

    A target is honoured exactly; a restore puts back the requested poses (the cube
    upright on the floor, since a person cannot put it back in the gripper either).
    `requests` records every request, for tests to assert on.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    robot: FakeTossingRobot
    _requests: list[PlacementRequest] = PrivateAttr(default_factory=list)
    _inspections: int = PrivateAttr(default=0)

    def requests(self) -> list[PlacementRequest]:
        return list(self._requests)

    def inspect_cube_in_bin(self, *, owner: str) -> bool:
        """The person looks into the bin: the fake's ground truth."""
        del owner
        self._inspections += 1
        return self.robot.truly_in_bin()

    def inspections(self) -> int:
        return self._inspections

    def place(self, *, request: PlacementRequest) -> None:
        self._requests.append(request)
        start = self.robot.bin_start
        bin_ = (
            None
            if request.bin_target is None or start is None
            else start.model_copy(update={"x": request.bin_target[0], "y": request.bin_target[1]})
        )
        cube = (
            None
            if request.cube_target is None
            else RobotPose(x=request.cube_target[0], y=request.cube_target[1], yaw=0.0)
        )
        self.robot.place(cube=cube, bin_=bin_)

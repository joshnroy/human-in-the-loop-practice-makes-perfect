"""The real TidyBot's skill-level contract, and the scene description built from it.

The robot side exposes a skill-level RPC client. This package never imports it: it
codes against `TossingRobot` below, a structural `Protocol` matching that client's
contract exactly, so the client, `FakeTossingRobot` and the simulator-as-plant robot are
interchangeable. Replies arrive as plain dicts; every one is validated into a model here
before anything above reads it, so a contract drift fails at the boundary, by name.

Units are the contract's: map frame, metres, radians; toss speed in joint-path deg/s
and release in milliseconds from the start of the swing (the sim's own units).
"""

import math
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


@runtime_checkable
class TossingRobot(Protocol):
    """The robot side's skill-level client.

    `toss` raises `ValueError` outside the real bounds, and nothing moves; it returns
    `tossed: False` with a note, moving nothing, when not holding or no bin is found.
    A motion skill interrupted by `stop()` raises the client's `SkillStopped`, and one
    started while another runs raises `SkillBusy`; both are `RuntimeError`s, and both
    become a failed action here, never a silent retry. `observe()` takes seconds (it
    localizes the cube and the bin), so the backend observes once per change.
    `direction` is the azimuth from the bin centre to the robot's stand, map frame,
    radians -- chosen on this side (`real/backend.py`), never by the robot.
    """

    def observe(self) -> dict[str, Any]: ...

    def can_pick(self) -> dict[str, Any]: ...

    def pick(self) -> dict[str, Any]: ...

    def open_gripper(self) -> Any: ...

    def toss(
        self, *, standoff: float, speed: float, release_ms: float, direction: float
    ) -> dict[str, Any]: ...

    def retract(self) -> Any: ...

    def go_home(self) -> Any: ...

    def stop(self) -> Any: ...


class RobotPose(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    x: float
    y: float
    yaw: float

    @staticmethod
    def parse(*, value: Any) -> "RobotPose | None":
        """The contract's `[x, y, yaw] | None`."""
        if value is None:
            return None
        x, y, yaw = (float(v) for v in value)
        return RobotPose(x=x, y=y, yaw=yaw)


class BinFit(BaseModel):
    """The robot side's fitted bin: an upright box, so its presence is `BinOnGround`."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    x: float
    y: float
    yaw: float
    length: float = Field(gt=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class BarrierRect(BaseModel):
    """The barrier: a finite rectangle on the floor, long side along its local y.

    Default is the virtual barrier until a physical one is placed at a measured line:
    centre (0, 0), long side along map y (y in [-0.9, 0.9]), 0.05 m thick.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    x: float = 0.0
    y: float = 0.0
    length: float = Field(default=1.8, gt=0)
    thickness: float = Field(default=0.05, gt=0)
    yaw: float = 0.0

    def x_extent(self) -> float:
        """The rectangle's extent along map x, which is what the sim's side atoms read
        as the barrier's `bb_x` (exactly its thickness at the default yaw 0)."""
        return abs(math.cos(self.yaw)) * self.thickness + abs(math.sin(self.yaw)) * self.length


class RobotObservation(BaseModel):
    """One `observe()` reply, validated."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    robot: RobotPose
    cube: RobotPose | None
    cube_upright: bool
    bin: BinFit | None
    holding: bool
    gripper: float
    gripper_command: float
    cube_in_bin: bool
    barrier: BarrierRect
    barrier_x: float

    @staticmethod
    def parse(*, reply: dict[str, Any]) -> "RobotObservation":
        robot = RobotPose.parse(value=reply["robot"])
        assert robot is not None, "observe() returned no robot pose"
        observation = RobotObservation(
            robot=robot,
            cube=RobotPose.parse(value=reply["cube"]),
            cube_upright=bool(reply["cube_upright"]),
            bin=None if reply["bin"] is None else BinFit(**reply["bin"]),
            holding=bool(reply["holding"]),
            gripper=float(reply["gripper"]),
            gripper_command=float(reply["gripper_command"]),
            cube_in_bin=bool(reply["cube_in_bin"]),
            barrier=BarrierRect(**RobotObservation._barrier(reply=reply)),
            barrier_x=float(reply["barrier_x"]),
        )
        if not math.isclose(observation.barrier_x, observation.barrier.x, abs_tol=1e-9):
            raise ValueError(
                f"observe() reported barrier_x={observation.barrier_x} but a barrier "
                f"centred at x={observation.barrier.x}; the contract says they are equal"
            )
        return observation

    @staticmethod
    def _barrier(*, reply: dict[str, Any]) -> dict[str, Any]:
        """The contract reports `barrier`/`barrier_x` as None when the skill server's
        barrier is disabled. This domain's side atoms -- and so its whole operator model
        -- are defined relative to the barrier, so that is refused here, by name."""
        if reply["barrier"] is None or reply["barrier_x"] is None:
            raise BarrierDisabledError(
                "observe() reports no barrier: the real Tossing3D domain's side atoms are "
                "defined relative to one. Enable the barrier in the skill server's config."
            )
        barrier: dict[str, Any] = reply["barrier"]
        return barrier


class BarrierDisabledError(ValueError):
    """`observe()` reported the barrier disabled; this domain needs it."""


class CanPickReply(BaseModel):
    model_config = ConfigDict(frozen=True)

    feasible: bool
    note: str | None = None


class PickReply(BaseModel):
    model_config = ConfigDict(frozen=True)

    held: bool
    gripper: float
    note: str | None = None


class TossReply(BaseModel):
    model_config = ConfigDict(frozen=True)

    tossed: bool
    landed: tuple[float, float] | None = None
    in_bin: bool
    note: str | None = None


class PlanarRegion(BaseModel):
    """An axis-aligned map-frame rectangle a human is asked to place an object in."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def centre(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)


class RealSceneLayout(BaseModel):
    """Everything about the real room this package assumes rather than observes.

    The robot works on the +x side of the barrier and never crosses it; the bin may be
    anywhere, and is localized by the robot side on every `observe()`, as is the barrier
    rectangle itself (`RobotObservation.barrier`), so neither is configured here. The regions are
    where a human is asked to put things on a reset -- placeholders until measured in
    the room, every one overridable.

    `scoring_margin_m` shrinks the bin's interior to the box carried on the bin as its
    scored region, for display and logging only: `InBin` is the robot side's own
    `cube_in_bin`, which applies the same rule.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    drivable_x: tuple[float, float] = (-1.5, 1.7)
    drivable_y: tuple[float, float] = (-1.5, 1.5)
    # TidyBot++ base footprint, metres, along its own x (forward) then y.
    robot_size: tuple[float, float] = (0.548, 0.508)
    cube_half_size_m: float = 0.025
    bin_wall_thickness_m: float = 0.02
    scoring_margin_m: float = 0.03
    cube_reset_region: PlanarRegion = PlanarRegion(x_min=0.6, y_min=-0.4, x_max=1.0, y_max=0.4)
    robot_side_bin_region: PlanarRegion = PlanarRegion(x_min=0.8, y_min=0.7, x_max=1.3, y_max=1.1)
    opposite_side_bin_region: PlanarRegion = PlanarRegion(
        x_min=-1.1, y_min=-0.6, x_max=-0.6, y_max=0.6
    )


class RealSceneSnapshot(BaseModel):
    """What the real backend carries on a `Tossing3DState` as `object_centric`.

    Its type is what tells the shared toss pipeline it is looking at the real room
    (`real/geometry.py`): the observation it was built from and the layout it is in,
    which together are the whole toss-feasibility geometry.
    """

    model_config = ConfigDict(frozen=True)

    observation: RobotObservation
    layout: RealSceneLayout
    # The `can_pick()` dry run's verdict for this observation, or `None` where the
    # `PickPlannable` gate decides without asking (held, or across the barrier).
    can_pick: bool | None = None


class PlacementRequest(BaseModel):
    """One request for a human to arrange the scene, then confirm.

    `kind` is why: `"scene"` sets up a task's initial scene, `"reset"` is the human
    reset a method asked for, `"restore"` puts back the scene one environment left
    when the other (practice vs evaluation) used the robot in between.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["scene", "reset", "restore"]
    owner: str
    cube_target: tuple[float, float] | None
    cube_region: PlanarRegion | None = None
    bin_target: tuple[float, float] | None
    bin_region: PlanarRegion | None = None
    # A restore of a scene in which the robot was holding the cube: a human cannot put
    # a cube back into a closed gripper, so this is surfaced rather than pretended.
    was_holding: bool = False
    seed: int | None = None
    destination: str | None = None
    # Why a previous attempt at this placement did not take, to show the person.
    notes: tuple[str, ...] = ()


@runtime_checkable
class RealScenePlacer(Protocol):
    """Whoever arranges the scene: a person at a prompt, or a script in a test."""

    def place(self, *, request: PlacementRequest) -> None: ...

    def inspect_cube_in_bin(self, *, owner: str) -> bool:
        """Right after a toss the bin's walls can hide the cube, so the robot side
        reports no cube and `cube_in_bin` False. That is unknown, not a miss: a person
        looks and says whether the cube is in the bin."""
        ...

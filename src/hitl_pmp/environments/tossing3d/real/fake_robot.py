"""A robot with no physics behind the skill server's contract, and a person who places
things exactly where asked: for tests and `--robot fake` dry runs.

It answers in the map frame with replies shaped like the robot side's. Its rules are a
stand-in, not a model of the room: a cube is pickable on the robot's side of the tape,
a stand is refused unless the whole base stays on the robot's side, and a toss lands in
the bin when its speed suits its standoff (`FakeTossingRobot.lands_in_bin`), and short
of the bin, toward the robot, otherwise. A dump empties the bin onto `_POUR_OFFSET`
beside it when the rim grip holds (`dump_grips`).
"""

import math
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from hitl_pmp.environments.tossing3d.types import TossParameterSet

from .backend import PlacementRequest, TerminalPerson
from .frame import ROOM_FRAME

# The base's half-depth plus the robot side's clearance, from the tape's centre line.
_STAND_MARGIN_M = 0.38
# A cube this far into the robot's side, or farther, can be picked.
_PICK_MARGIN_M = 0.15
_SHORT_LANDING_M = 0.45
# The blue recycling bin's outer rim and height.
_BIN = {"length": 0.365, "width": 0.26, "height": 0.384}
# Where a dump drops the contents, from the bin's centre (map frame): the 2026-11-02 real
# dumps at the robot-side spot poured at about this offset.
_POUR_OFFSET = (0.65, 0.75)
# The robot side's own `real_toss_bounds`, wider than this package's real set.
_SERVER_TOSS_BOUNDS = TossParameterSet(
    standoff_bounds=(0.9, 1.45), speed_bounds=(100.0, 140.0), release_ms_bounds=(500.0, 850.0)
)


# What each skill takes on the robot, in seconds, so a dry run's budget runs out as a
# real one's does: pick and toss are the 2026-11-06 means (#763), the dump the middle
# of its 52-66 s, and opening the gripper the sim's 39 ticks (never timed on the robot).
FAKE_SKILL_SECONDS: Final[dict[str, float]] = {
    "pick": 36.4,
    "toss": 28.3,
    "dump_bin": 59.0,
    "open_gripper": 3.9,
}


class FakeTossingRobot(BaseModel):
    """`TossingRobot` with no robot. `calls` records every call, in order; `seconds` is
    its own clock, advanced by each skill's `FAKE_SKILL_SECONDS` (`now`)."""

    robot: tuple[float, float, float] = (1.326, -0.493, -3.132)
    cube: tuple[float, float, float] | None = None
    bin: tuple[float, float, float] | None = None
    holding: bool = False
    gripper_command: float = 0.0
    barrier_enabled: bool = True
    dump_grips: bool = True
    calls: list[tuple[str, tuple[Any, ...]]] = Field(default_factory=list)
    seconds: float = 0.0

    def now(self) -> float:
        return self.seconds

    def put(
        self, *, cube: tuple[float, float], bin_center: tuple[float, float], bin_yaw: float
    ) -> None:
        """What a person does: the cube on the floor, the bin upright."""
        self.cube = (cube[0], cube[1], 0.0)
        self.bin = (bin_center[0], bin_center[1], bin_yaw)
        self.holding = False

    def observe(self) -> dict[str, Any]:
        self.calls.append(("observe", ()))
        cube = None if self.holding or self.cube is None else list(self.cube)
        bin_ = (
            None
            if self.bin is None
            else {"x": self.bin[0], "y": self.bin[1], "yaw": self.bin[2], **_BIN}
        )
        return {
            "robot": list(self.robot),
            "cube": cube,
            "bin": bin_,
            "holding": self.holding,
            "gripper": 0.49 if self.holding else self.gripper_command,
            "gripper_command": self.gripper_command,
            "cube_upright": cube is not None,
            "cube_in_bin": cube is not None and self._in_bin(xy=(cube[0], cube[1])),
            "barrier": None
            if not self.barrier_enabled
            else {
                "x": ROOM_FRAME.barrier_center[0],
                "y": ROOM_FRAME.barrier_center[1],
                "length": 6.0,
                "thickness": 0.05,
                "yaw": math.atan2(-ROOM_FRAME.robot_side[0], ROOM_FRAME.robot_side[1]),
                "robot_side": list(ROOM_FRAME.robot_side),
            },
            "barrier_x": ROOM_FRAME.barrier_center[0] if self.barrier_enabled else None,
            "cube_inside_bin": cube is not None and self._in_bin(xy=(cube[0], cube[1]), margin=0.0),
            "toss_direction": None if self.bin is None else self.toss_direction(),
        }

    def toss_direction(self) -> float:
        """The robot side's fixed azimuth: from +x of a bin on the robot's side, from -y
        of one across the tape."""
        assert self.bin is not None
        return 0.0 if ROOM_FRAME.to_domain(x=self.bin[0], y=self.bin[1])[0] < 0.0 else -math.pi / 2

    def can_pick(self) -> dict[str, Any]:
        self.calls.append(("can_pick", ()))
        check = self._pick_check()
        return {**check, "refused": not check["feasible"], "reason": check["note"]}

    def _pick_check(self) -> dict[str, Any]:
        if self.holding:
            return {"feasible": False, "note": "already holding the cube"}
        if self.cube is None:
            return {"feasible": False, "note": "no cube found"}
        if ROOM_FRAME.to_domain(x=self.cube[0], y=self.cube[1])[0] > -_PICK_MARGIN_M:
            return {"feasible": False, "note": "cube is not on the robot's side"}
        return {"feasible": True, "note": None}

    def pick(self) -> dict[str, Any]:
        self.seconds += FAKE_SKILL_SECONDS["pick"]
        self.calls.append(("pick", ()))
        check = self._pick_check()
        if not check["feasible"]:
            return {
                "held": False,
                "gripper": 0.0,
                "note": check["note"],
                "refused": True,
                "reason": check["note"],
                "cube": None,
                "attempts": [],
            }
        assert self.cube is not None
        found = {"x": self.cube[0], "y": self.cube[1], "yaw_deg": 0.0, "in_bin": False}
        self.robot = (self.cube[0] + 0.55, self.cube[1], math.pi)
        self.holding, self.gripper_command, self.cube = True, 1.0, None
        return {
            "held": True,
            "gripper": 0.49,
            "note": None,
            "refused": False,
            "reason": None,
            "cube": found,
            "attempts": [],
        }

    def toss(
        self, *, standoff: float, speed: float, release_ms: float, direction: float
    ) -> dict[str, Any]:
        self.seconds += FAKE_SKILL_SECONDS["toss"]
        self.calls.append(("toss", (standoff, speed, release_ms, direction)))
        refusal = _SERVER_TOSS_BOUNDS.violation(params=(standoff, speed, release_ms))
        if refusal is not None:
            raise ValueError(refusal)
        if not self.holding:
            return {"tossed": False, "landed": None, "in_bin": False, "note": "not holding"}
        if self.bin is None:
            return {"tossed": False, "landed": None, "in_bin": False, "note": "no bin found"}
        stand = (
            self.bin[0] + standoff * math.cos(direction),
            self.bin[1] + standoff * math.sin(direction),
        )
        if ROOM_FRAME.to_domain(x=stand[0], y=stand[1])[0] > -_STAND_MARGIN_M:
            raise ValueError(f"toss pose {stand} is across the barrier")
        self.robot = (stand[0], stand[1], direction + math.pi)
        if self.lands_in_bin(standoff=standoff, speed=speed, release_ms=release_ms):
            landed = (self.bin[0], self.bin[1])
        else:
            landed = (
                self.bin[0] + _SHORT_LANDING_M * math.cos(direction),
                self.bin[1] + _SHORT_LANDING_M * math.sin(direction),
            )
        self.cube = (landed[0], landed[1], 0.0)
        self.holding, self.gripper_command = False, 0.0
        return {
            "tossed": True,
            "landed": list(landed),
            "in_bin": self._in_bin(xy=landed),
            "note": None,
            "inside_bin": self._in_bin(xy=landed, margin=0.0),
            "pose": list(self.robot),
            "direction": direction,
        }

    @staticmethod
    def lands_in_bin(*, standoff: float, speed: float, release_ms: float) -> bool:
        """A band through the parameter box: the speed that suits a standoff rises with
        it, and only mid-range releases score."""
        suited = 100.0 + 40.0 * (standoff - 0.9) / 0.55
        return abs(speed - suited) <= 10.0 and 550.0 <= release_ms <= 800.0

    def dump_bin(self) -> dict[str, Any]:
        self.seconds += FAKE_SKILL_SECONDS["dump_bin"]
        self.calls.append(("dump_bin", ()))
        reply: dict[str, Any] = {
            "dumped": False,
            "refused": False,
            "reason": None,
            "note": None,
            "stages": [],
            "gripper": None,
        }
        why = None
        if self.holding:
            why = "holding the cube: open the gripper first"
        elif self.bin is None:
            why = "no bin found"
        elif ROOM_FRAME.to_domain(x=self.bin[0], y=self.bin[1])[0] > -_STAND_MARGIN_M:
            why = "the bin is across the barrier"
        if why is not None:
            return reply | {"refused": True, "reason": why, "note": why}
        assert self.bin is not None
        pour = (self.bin[0] + _POUR_OFFSET[0], self.bin[1] + _POUR_OFFSET[1])
        reply |= {"pour_point": list(pour)}
        if not self.dump_grips:
            return reply | {"note": "no grip on the bin's rim (gripper read 1.00)", "gripper": 1.0}
        if self.cube is not None and self._in_bin(xy=(self.cube[0], self.cube[1]), margin=0.0):
            self.cube = (pour[0], pour[1], 0.0)
        return reply | {"dumped": True, "gripper": 0.93}

    def open_gripper(self) -> None:
        self.seconds += FAKE_SKILL_SECONDS["open_gripper"]
        self.calls.append(("open_gripper", ()))
        if self.holding:
            heading = self.robot[2]
            self.cube = (
                self.robot[0] + 0.5 * math.cos(heading),
                self.robot[1] + 0.5 * math.sin(heading),
                0.0,
            )
        self.holding, self.gripper_command = False, 0.0

    def retract(self) -> None:
        self.calls.append(("retract", ()))

    def go_home(self) -> None:
        self.calls.append(("go_home", ()))

    def stop(self) -> None:
        self.calls.append(("stop", ()))

    def _in_bin(self, *, xy: tuple[float, float], margin: float = 0.05) -> bool:
        if self.bin is None:
            return False
        dx, dy = xy[0] - self.bin[0], xy[1] - self.bin[1]
        cosine, sine = math.cos(self.bin[2]), math.sin(self.bin[2])
        along, across = dx * cosine + dy * sine, -dx * sine + dy * cosine
        return (
            abs(along) <= _BIN["length"] / 2 - margin and abs(across) <= _BIN["width"] / 2 - margin
        )


class FakePerson(TerminalPerson):
    """`TerminalPerson` -- shown the same request, through the same prompt -- whose
    y is typed at once and who puts everything exactly where asked."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    robot: FakeTossingRobot

    def model_post_init(self, __context: object) -> None:
        self.read = lambda prompt: self._press_enter(prompt=prompt)

    def _press_enter(self, *, prompt: str) -> str:
        self.write(f"{prompt}y")
        return "y"

    def placed(self, *, request: PlacementRequest) -> None:
        cube = request.bin.center if request.cube_in_bin else request.cube
        self.robot.put(cube=cube, bin_center=request.bin.center, bin_yaw=request.bin.yaw)

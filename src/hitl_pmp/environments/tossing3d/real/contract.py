"""The robot side's skill server, as this package calls it: the one module to update
when its API changes.

Built against prpl-tidybot's skill server API of 2026-10-20 (v2, prpl-tidybot 35d785c on branch
josh/skill-server-one-path; `hitl-pmp-results/skill-server-api-20261020.md`), without
importing it: that package cannot be installed in this environment. The RPC is
`multiprocessing.managers.BaseManager` with one shared object, `Skills`, on port 50002.
Everything is in the robot's map frame (metres, radians); `real/frame.py` converts.

    observe() -> {robot [x, y, yaw], cube [x, y, yaw] | None,
                  bin {x, y, yaw, length, width, height} | None, holding, gripper,
                  gripper_command, cube_upright, cube_in_bin (scored), cube_inside_bin,
                  barrier {x, y, length, thickness, yaw, robot_side}, barrier_x,
                  toss_direction (the fixed map azimuth for the bin's spot) | None}
    can_pick() -> {feasible, note, refused, reason}     a dry run; moves nothing
    pick() -> {held, gripper, note, refused, reason, cube, bin, stand, attempts}
        refused: nothing moved; held False and not refused: a real miss
    toss(standoff, speed, release_ms, direction) -> {tossed, landed, in_bin, note,
        inside_bin, pose, direction, bin, swing}
        direction is the map-frame azimuth bin -> robot; raises ValueError and moves
        nothing outside the robot's bounds or for a pose it does not allow
    open_gripper(), retract(), go_home(), stop()

The authkey is never committed: it is read from `$TIDYBOT_RPC_AUTHKEY` or a file.
"""

import os
from multiprocessing.managers import BaseManager
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, PrivateAttr

DEFAULT_PORT = 50002
AUTHKEY_ENV_VAR = "TIDYBOT_RPC_AUTHKEY"


class BinReply(BaseModel):
    """`BinFit.as_dict()`: centre, the long side's yaw, and outer dimensions."""

    x: float
    y: float
    yaw: float
    length: float
    width: float
    height: float


class BarrierReply(BaseModel):
    """`Barrier.as_dict()`: the tape's centre, extent, and the direction into the robot's
    side."""

    x: float
    y: float
    length: float
    thickness: float
    yaw: float
    robot_side: tuple[float, float]


class ObserveReply(BaseModel):
    model_config = ConfigDict(frozen=True)

    robot: tuple[float, float, float]
    cube: tuple[float, float, float] | None
    bin: BinReply | None
    holding: bool
    gripper: float
    gripper_command: float
    cube_upright: bool
    cube_in_bin: bool
    barrier: BarrierReply | None
    barrier_x: float | None
    cube_inside_bin: bool
    toss_direction: float | None


class CanPickReply(BaseModel):
    feasible: bool
    note: str | None = None
    refused: bool = False
    reason: str | None = None


class PickReply(BaseModel):
    held: bool
    gripper: float
    note: str | None = None
    # True: nothing moved -- the pick could not or may not start.
    refused: bool
    reason: str | None = None
    cube: dict[str, Any] | None = None
    bin: BinReply | None = None
    stand: tuple[float, float, float] | None = None
    # A per-grasp log, for logging only: not a stable schema.
    attempts: list[dict[str, Any]] = []


class TossReply(BaseModel):
    tossed: bool
    landed: tuple[float, float] | None
    in_bin: bool
    note: str | None = None
    inside_bin: bool = False
    pose: tuple[float, float, float] | None = None
    direction: float | None = None
    bin: BinReply | None = None
    swing: dict[str, Any] | None = None


@runtime_checkable
class TossingRobot(Protocol):
    """What the real backend calls: the skill server's client, or the fake."""

    def observe(self) -> dict[str, Any]: ...

    def can_pick(self) -> dict[str, Any]: ...

    def pick(self) -> dict[str, Any]: ...

    def toss(
        self, *, standoff: float, speed: float, release_ms: float, direction: float
    ) -> dict[str, Any]: ...

    def open_gripper(self) -> None: ...

    def retract(self) -> None: ...

    def go_home(self) -> None: ...

    def stop(self) -> None: ...


class _SkillClientManager(BaseManager):
    """Client-side manager. Its own class because `register` writes a class-wide
    registry."""


_SkillClientManager.register("Skills")


class SkillServerClient(BaseModel):
    """`TossingRobot` over the skill server's RPC. A server-side `ValueError` (a refused
    toss) is re-raised here as a `ValueError`, as `BaseManager` proxies do."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    host: str
    port: int = DEFAULT_PORT
    authkey: bytes

    _skills: Any = PrivateAttr(default=None)

    @staticmethod
    def read_authkey(*, path: Path | None) -> bytes:
        """The RPC authkey from `path` if given, else `$TIDYBOT_RPC_AUTHKEY`."""
        if path is not None:
            return Path(path).read_text().strip().encode()
        value = os.environ.get(AUTHKEY_ENV_VAR)
        if not value:
            raise ValueError(
                f"no skill-server authkey: set {AUTHKEY_ENV_VAR} or pass --robot-authkey-file"
            )
        return value.encode()

    def _connect(self) -> Any:
        manager = _SkillClientManager(address=(self.host, self.port), authkey=self.authkey)
        manager.connect()
        return manager.Skills()  # type: ignore[attr-defined]

    def _call(self, *, name: str, args: tuple[float, ...] = ()) -> Any:
        if self._skills is None:
            self._skills = self._connect()
        try:
            return getattr(self._skills, name)(*args)
        except KeyboardInterrupt:
            self.stop()
            raise

    def observe(self) -> dict[str, Any]:
        return dict(self._call(name="observe"))

    def can_pick(self) -> dict[str, Any]:
        return dict(self._call(name="can_pick"))

    def pick(self) -> dict[str, Any]:
        return dict(self._call(name="pick"))

    def toss(
        self, *, standoff: float, speed: float, release_ms: float, direction: float
    ) -> dict[str, Any]:
        return dict(
            self._call(
                name="toss",
                args=(float(standoff), float(speed), float(release_ms), float(direction)),
            )
        )

    def open_gripper(self) -> None:
        self._call(name="open_gripper")

    def retract(self) -> None:
        self._call(name="retract")

    def go_home(self) -> None:
        self._call(name="go_home")

    def stop(self) -> None:
        """Halt arm and base now, on a fresh connection (the first may be busy)."""
        self._connect().stop()

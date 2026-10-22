"""The room as the dashboard draws it, in the map frame: the bin spots, the tape, the
last observation and the recent toss landings. Shared by the plain map and the camera
overlay."""

from typing import Final

from pydantic import BaseModel

from .sources import Observation

# The bin's footprint (length along its yaw, width), as the skill server reports it.
BIN_SIZE_M: Final = (0.39, 0.29)


class BinSpotView(BaseModel):
    name: str
    x: float
    y: float
    yaw: float
    length: float
    width: float


class Landing(BaseModel):
    index: int
    x: float
    y: float
    in_bin: bool | None
    inside_bin: bool | None
    spot: str | None


class Room(BaseModel):
    bin_spots: list[BinSpotView]
    tape_intercept: float
    tape_slope: float
    tape_thickness: float
    drivable_box: tuple[float, float, float, float]
    landings: list[Landing]
    observation: Observation | None
    observed_bin: BinSpotView | None

"""The live ceiling camera: the newest stitched frame, and the map drawn onto it.

The frame and the calibration come from `ceiling_helper.py`, which runs in
prpl-tidybot's own environment (hitl-pmp does not depend on `prpl_tidybot`) as a
subprocess of the dashboard and writes into a directory this module reads.

## Projection

The stitched frame is the cameras' *undistorted* frames stacked top to bottom in
calibration order, so camera `i` owns rows `[i * h, (i + 1) * h)` and a map point
projects through camera `i`'s pinhole model with no distortion, exactly as
prpl-tidybot's `CeilingCamera.project` does (`cv.projectPoints` with zero
coefficients). A point is drawn in every camera whose frame it falls inside, so where
the two views overlap it appears twice, as it does in the picture.

Heights: everything on the floor (the tape, landings, the cube, the robot's base pose)
is drawn at z = 0, and every bin outline -- the two spots and the bin as last observed
-- at its rim, z = 0.21 m, which is the edge the camera sees.
"""

import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Final

import numpy as np
from pydantic import BaseModel, PrivateAttr

from .room import BIN_SIZE_M, Room

BIN_RIM_HEIGHT_M: Final = 0.21
CUBE_SIDE_M: Final = 0.05
# The tape is drawn over this map-x span (the room's drivable box).
TAPE_X_SPAN: Final = (-1.846, 2.046)
_TAPE_SEGMENTS: Final = 48
# A frame older than this is not shown; the page falls back to the plain map.
STALE_FRAME_SECONDS: Final = 30.0


class CameraModel(BaseModel):
    serial: str
    width: int
    height: int
    camera_matrix: list[list[float]]
    rvec: list[float]
    tvec: list[float]

    def rotation(self) -> np.ndarray:
        """Rodrigues: the rotation vector as a matrix."""
        vector = np.asarray(self.rvec, dtype=np.float64)
        angle = float(np.linalg.norm(vector))
        if angle < 1e-12:
            return np.eye(3)
        k = vector / angle
        cross = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        return np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * cross @ cross


class CeilingCalibration(BaseModel):
    cameras: list[CameraModel]

    @staticmethod
    def load(*, path: Path) -> "CeilingCalibration":
        return CeilingCalibration.model_validate(json.loads(path.read_text()))

    @property
    def width(self) -> int:
        return max(camera.width for camera in self.cameras)

    @property
    def height(self) -> int:
        return sum(camera.height for camera in self.cameras)

    def project(
        self, *, points: list[tuple[float, float, float]]
    ) -> list[list[tuple[int, tuple[float, float]]]]:
        """For each map point, `(camera, (u, v))` in stitched-frame pixels for every
        camera whose frame it falls inside."""
        world = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        seen: list[list[tuple[int, tuple[float, float]]]] = [[] for _ in range(len(world))]
        offset = 0
        for index, camera in enumerate(self.cameras):
            in_camera = world @ camera.rotation().T + np.asarray(camera.tvec)
            matrix = np.asarray(camera.camera_matrix)
            for number, (x, y, z) in enumerate(in_camera):
                if z <= 1e-6:
                    continue
                u = matrix[0, 0] * x / z + matrix[0, 1] * y / z + matrix[0, 2]
                v = matrix[1, 1] * y / z + matrix[1, 2]
                if 0 <= u < camera.width and 0 <= v < camera.height:
                    seen[number].append((index, (float(u), float(v + offset))))
            offset += camera.height
        return seen


class OverlayShape(BaseModel):
    kind: str
    label: str
    camera: int
    # Stitched-frame pixels; a polygon when `closed`, else a polyline (or one point).
    points: list[tuple[float, float]]
    closed: bool = False
    in_bin: bool | None = None


class Overlay(BaseModel):
    width: int
    height: int
    shapes: list[OverlayShape]

    @staticmethod
    def build(*, calibration: CeilingCalibration, room: Room) -> "Overlay":
        shapes: list[OverlayShape] = []

        def add(
            *,
            kind: str,
            label: str,
            points: list[tuple[float, float, float]],
            closed: bool = False,
            in_bin: bool | None = None,
        ) -> None:
            """A shape is drawn in each camera that sees all of its points."""
            projected = calibration.project(points=points)
            cameras = set.intersection(*({camera for camera, _ in seen} for seen in projected))
            for camera in sorted(cameras):
                pixels = [dict(seen)[camera] for seen in projected]
                shapes.append(
                    OverlayShape(
                        kind=kind,
                        label=label,
                        camera=camera,
                        points=pixels,
                        closed=closed,
                        in_bin=in_bin,
                    )
                )

        # The tape: both edges, in short segments so a camera draws what it sees.
        step = (TAPE_X_SPAN[1] - TAPE_X_SPAN[0]) / _TAPE_SEGMENTS
        thickness = room.tape_thickness * math.hypot(1.0, room.tape_slope)
        for edge, intercept in (
            ("robot-side edge", room.tape_intercept),
            ("far edge", room.tape_intercept + thickness),
        ):
            for number in range(_TAPE_SEGMENTS):
                x0 = TAPE_X_SPAN[0] + number * step
                add(
                    kind="tape",
                    label=f"tape, {edge}",
                    points=[(x, intercept + room.tape_slope * x, 0.0) for x in (x0, x0 + step)],
                )
        for spot in room.bin_spots:
            add(
                kind="spot",
                label=f"{spot.name} spot (rim)",
                points=Overlay._rectangle(
                    x=spot.x, y=spot.y, yaw=spot.yaw, size=BIN_SIZE_M, z=BIN_RIM_HEIGHT_M
                ),
                closed=True,
            )
        if room.observed_bin is not None:
            seen_bin = room.observed_bin
            add(
                kind="bin",
                label="bin, last observed (rim)",
                points=Overlay._rectangle(
                    x=seen_bin.x,
                    y=seen_bin.y,
                    yaw=seen_bin.yaw,
                    size=(seen_bin.length, seen_bin.width),
                    z=BIN_RIM_HEIGHT_M,
                ),
                closed=True,
            )
        observation = room.observation
        if observation is not None and observation.cube is not None:
            add(
                kind="cube",
                label="cube, last observed",
                points=Overlay._rectangle(
                    x=observation.cube[0],
                    y=observation.cube[1],
                    yaw=0.0,
                    size=(CUBE_SIDE_M, CUBE_SIDE_M),
                    z=0.0,
                ),
                closed=True,
            )
        if observation is not None and observation.robot is not None:
            x, y, yaw = observation.robot
            add(
                kind="robot",
                label="robot base, last observed",
                points=[
                    (x + 0.25 * math.cos(yaw + turn), y + 0.25 * math.sin(yaw + turn), 0.0)
                    for turn in (0.0, 2.5, -2.5)
                ],
                closed=True,
            )
        for landing in room.landings:
            add(
                kind="landing",
                label=f"toss {landing.index} landing",
                points=[(landing.x, landing.y, 0.0)],
                in_bin=landing.in_bin,
            )
        return Overlay(width=calibration.width, height=calibration.height, shapes=shapes)

    @staticmethod
    def _rectangle(
        *, x: float, y: float, yaw: float, size: tuple[float, float], z: float
    ) -> list[tuple[float, float, float]]:
        c, s = math.cos(yaw), math.sin(yaw)
        half_l, half_w = size[0] / 2, size[1] / 2
        corners = [(half_l, half_w), (-half_l, half_w), (-half_l, -half_w), (half_l, -half_w)]
        return [(x + u * c - v * s, y + u * s + v * c, z) for u, v in corners]


class CameraState(BaseModel):
    available: bool
    frame_time: float | None
    age_seconds: float | None
    grab_seconds: float | None
    error: str | None
    overlay: Overlay | None


class CameraFeed(BaseModel):
    """The helper subprocess and the directory it writes."""

    directory: Path
    python: Path | None = None
    pythonpath: str | None = None
    host: str = "192.168.0.131"
    period: float = 2.0

    _process: subprocess.Popen[bytes] | None = PrivateAttr(default=None)
    _calibration: CeilingCalibration | None = PrivateAttr(default=None)

    @staticmethod
    def scratch_directory() -> Path:
        return Path(tempfile.mkdtemp(prefix="real-dashboard-camera-"))

    def start(self) -> None:
        if self.python is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        environment = {"PYTHONPATH": self.pythonpath} if self.pythonpath else {}
        self._process = subprocess.Popen(  # noqa: S603 -- our own helper, our own flags
            [
                str(self.python),
                str(Path(__file__).with_name("ceiling_helper.py")),
                "--out-dir",
                str(self.directory),
                "--host",
                self.host,
                "--period",
                str(self.period),
            ],
            env={**_Environment.base(), **environment},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def stop(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()

    def frame(self) -> bytes | None:
        path = self.directory / "frame.jpg"
        try:
            return path.read_bytes()
        except FileNotFoundError:
            return None

    def state(self, *, room: Room) -> CameraState:
        meta = _Environment.json(path=self.directory / "frame.json") or {}
        if self._calibration is None and (self.directory / "calibration.json").exists():
            try:
                self._calibration = CeilingCalibration.load(
                    path=self.directory / "calibration.json"
                )
            except ValueError:
                self._calibration = None
        frame_time = meta.get("time") if "error" not in meta else None
        if frame_time is None and (self.directory / "frame.jpg").exists():
            frame_time = (self.directory / "frame.jpg").stat().st_mtime
        age = time.time() - frame_time if isinstance(frame_time, int | float) else None
        error = meta.get("error")
        if self.python is None and not (self.directory / "frame.jpg").exists():
            error = "no camera configured (--camera-python)"
        elif self._process is not None and self._process.poll() is not None:
            error = error or f"camera helper exited ({self._process.returncode})"
        available = (
            self._calibration is not None
            and age is not None
            and age < STALE_FRAME_SECONDS
            and (self.directory / "frame.jpg").exists()
        )
        return CameraState(
            available=available,
            frame_time=frame_time if isinstance(frame_time, int | float) else None,
            age_seconds=age,
            grab_seconds=meta.get("grab_seconds"),
            error=error,
            overlay=Overlay.build(calibration=self._calibration, room=room)
            if self._calibration is not None
            else None,
        )


class _Environment:
    @staticmethod
    def base() -> dict[str, str]:
        """The helper's environment: ours minus anything naming our own interpreter's
        packages, so prpl-tidybot's venv resolves its own."""
        drop = {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX"}
        return {key: value for key, value in os.environ.items() if key not in drop}

    @staticmethod
    def json(*, path: Path) -> dict[str, Any] | None:
        try:
            loaded = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        return loaded if isinstance(loaded, dict) else None

"""Render a SweepIntoDrawer3D-o5 cycle from its per-tick replay log.

    scripts/with_env.sh python scripts/render_sweep_drawer_reset.py \\
        --run-dir out/6 --output out/seed6.mp4

Reads `<run-dir>/replay_log.jsonl` (written by `measure_sweep_drawer_reset.py
--replay-log`) and `<run-dir>/cycle.json`. Nothing is simulated: every frame is the
logged joint positions put back into the compiled scene and drawn, from a three-quarter
view and a close-up of the drawer, with the step that was running captioned beneath.
"""

import argparse
import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict

from hitl_pmp.environments.sweep_drawer3d.session import SweepDrawerSession
from hitl_pmp.environments.sweep_drawer3d.types import SweepDrawerScene


class View(BaseModel):
    """A free camera: where it looks, from how far, from which side, from how high."""

    lookat: tuple[float, float, float]
    distance: float
    azimuth: float
    elevation: float
    title: str


class Layout:
    """Two panels side by side over a caption bar; every dimension even, as NVENC needs."""

    PANEL: ClassVar[tuple[int, int]] = (960, 720)
    BAR: ClassVar[int] = 64
    BACKGROUND: ClassVar[tuple[int, int, int]] = (16, 16, 16)
    FOREGROUND: ClassVar[tuple[int, int, int]] = (238, 238, 238)
    MUTED: ClassVar[tuple[int, int, int]] = (160, 160, 160)
    VIEWS: ClassVar[tuple[View, ...]] = (
        View(
            lookat=(1.0, -0.05, 0.35),
            distance=2.3,
            azimuth=35.0,
            elevation=-28.0,
            title="three-quarter view",
        ),
        View(
            lookat=(1.0, 0.05, 0.22),
            distance=0.85,
            azimuth=90.0,
            elevation=-52.0,
            title="drawer and floor, close up",
        ),
    )

    @staticmethod
    def size() -> tuple[int, int]:
        """(width, height) of a whole frame."""
        return 2 * Layout.PANEL[0], Layout.PANEL[1] + Layout.BAR

    @staticmethod
    def font(*, size: int) -> Any:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()

    @staticmethod
    def compose(*, panels: list[np.ndarray], lines: tuple[str, str]) -> np.ndarray:
        width, height = Layout.size()
        frame = Image.new("RGB", (width, height), Layout.BACKGROUND)
        for k, panel in enumerate(panels):
            frame.paste(Image.fromarray(panel), (k * Layout.PANEL[0], 0))
        draw = ImageDraw.Draw(frame)
        for k, view in enumerate(Layout.VIEWS):
            draw.text(
                (k * Layout.PANEL[0] + 12, 10),
                view.title,
                fill=Layout.FOREGROUND,
                font=Layout.font(size=18),
            )
        top = Layout.PANEL[1]
        draw.text((14, top + 8), lines[0], fill=Layout.FOREGROUND, font=Layout.font(size=22))
        draw.text((14, top + 38), lines[1], fill=Layout.MUTED, font=Layout.font(size=17))
        return np.asarray(frame, dtype=np.uint8)


class Caption:
    """What is written under a frame. Every number is read from the log, none restated."""

    PHASES: ClassVar[dict[str, str]] = {
        "attempt_1": "practice attempt 1 (stock skills)",
        "reset": "robot self-reset",
        "attempt_2": "practice attempt 2 (stock skills)",
    }

    @staticmethod
    def step_at(*, cycle: dict) -> dict[int, dict]:
        """Tick -> the step that was running during it."""
        out: dict[int, dict] = {}
        tick = 0
        for step in cycle["steps"]:
            for t in range(tick + 1, tick + step["ticks"] + 1):
                out[t] = step
            tick += step["ticks"]
        return out

    @staticmethod
    def lines(
        *, tick: int, step: dict | None, drawer: float, piled: int, speed: float
    ) -> tuple[str, str]:
        if step is None:
            head = "start"
        else:
            phase = Caption.PHASES.get(step["phase"], step["phase"])
            head = f"{phase}:  {step['name'].replace('_', ' ')}"
        cubes = len(SweepDrawerScene.CUBES)
        tail = (
            f"simulated time {tick / 10:6.1f} s   |   drawer open {max(drawer, 0.0) * 100:4.1f} cm"
            f"   |   cubes in the pile region {piled}/{cubes}   |   played at {speed:g}x"
        )
        return head, tail


class ReplayLog:
    """The per-tick joint positions written by `SweepDrawerSession`."""

    @staticmethod
    def header(*, path: Path) -> dict:
        with path.open() as f:
            head = json.loads(f.readline())
        if head.get("kind") != "header":
            raise ValueError(f"{path} does not start with a header record")
        return head

    @staticmethod
    def ticks(*, path: Path, stride: int) -> Iterator[dict]:
        with path.open() as f:
            f.readline()
            for line in f:
                record = json.loads(line)
                if record["t"] % stride == 0:
                    yield record

    @staticmethod
    def check(*, header: dict, model: Any) -> None:
        """Refuse to draw a log onto a scene it was not recorded from: a moved
        `reference/kindergarden` pin can reorder the joints under the same names."""
        import mujoco

        if header["nq"] != model.nq:
            raise ValueError(f"log has nq={header['nq']}, the compiled scene has {model.nq}")
        for joint in header["joints"]:
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint["name"])
            if j < 0 or int(model.jnt_qposadr[j]) != joint["adr"]:
                raise ValueError(f"joint {joint['name']} is not where the log recorded it")


class ResetVideo(BaseModel):
    """Draws one run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_dir: Path
    output: Path
    stride: int = 1
    fps: int = 40
    ffmpeg: str = "ffmpeg"
    codec: str = "h264_nvenc"
    first_tick: int = 0
    last_tick: int | None = None

    def speed(self) -> float:
        """Playback speed against simulated time, at the scene's 10 Hz control rate."""
        return self.fps * self.stride / 10.0

    def quality(self) -> list[str]:
        """Rate control in the codec's own flags: NVENC and libx264 share none of them."""
        if self.codec.endswith("_nvenc"):
            return ["-preset", "p5", "-rc", "vbr", "-cq", "24", "-b:v", "0"]
        return ["-preset", "medium", "-crf", "22"]

    def encoder(self) -> list[str]:
        width, height = Layout.size()
        return [
            self.ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-c:v",
            self.codec,
            *self.quality(),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.output),
        ]

    def render(self) -> int:
        import mujoco

        cycle = json.loads((self.run_dir / "cycle.json").read_text())
        log = self.run_dir / "replay_log.jsonl"
        header = ReplayLog.header(path=log)
        session = SweepDrawerSession(seed=int(header["seed"]))
        model, data = session.mj_model, session.mj_data
        ReplayLog.check(header=header, model=model)
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, Layout.PANEL[0])
        model.vis.global_.offheight = max(model.vis.global_.offheight, Layout.PANEL[1])
        renderer = mujoco.Renderer(model, height=Layout.PANEL[1], width=Layout.PANEL[0])
        cameras = []
        for view in Layout.VIEWS:
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = view.lookat
            camera.distance, camera.azimuth, camera.elevation = (
                view.distance,
                view.azimuth,
                view.elevation,
            )
            cameras.append(camera)
        drawer = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, SweepDrawerScene.DRAWER + "_joint"
        )
        cube_adr = [
            int(
                model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, c + "_joint")]
            )
            for c in SweepDrawerScene.CUBES
        ]
        steps = Caption.step_at(cycle=cycle)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        frames = 0
        with subprocess.Popen(self.encoder(), stdin=subprocess.PIPE) as pipe:
            assert pipe.stdin is not None
            for record in ReplayLog.ticks(path=log, stride=self.stride):
                if record["t"] < self.first_tick:
                    continue
                if self.last_tick is not None and record["t"] > self.last_tick:
                    break
                data.qpos[:] = record["qpos"]
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                panels = []
                for camera in cameras:
                    renderer.update_scene(data, camera=camera)
                    panels.append(renderer.render().copy())
                piled = sum(
                    ResetVideo.in_pile(position=data.qpos[adr : adr + 3]) for adr in cube_adr
                )
                lines = Caption.lines(
                    tick=record["t"],
                    step=steps.get(record["t"]),
                    drawer=float(data.qpos[model.jnt_qposadr[drawer]]),
                    piled=piled,
                    speed=self.speed(),
                )
                pipe.stdin.write(Layout.compose(panels=panels, lines=lines).tobytes())
                frames += 1
            pipe.stdin.close()
            code = pipe.wait()
        renderer.close()
        session.close()
        if code != 0:
            raise RuntimeError(f"ffmpeg exited with {code}")
        return frames

    @staticmethod
    def in_pile(*, position: np.ndarray) -> bool:
        s = SweepDrawerScene
        x, y, z = (float(v) for v in position)
        return (
            s.PILE_X[0] <= x <= s.PILE_X[1] and s.PILE_Y[0] <= y <= s.PILE_Y[1] and 0.45 < z < 0.5
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=40)
    parser.add_argument("--first-tick", type=int, default=0)
    parser.add_argument("--last-tick", type=int, default=None)
    parser.add_argument("--ffmpeg", default="ffmpeg", help="an ffmpeg built with --codec")
    parser.add_argument("--codec", default="h264_nvenc")
    args = parser.parse_args()
    video = ResetVideo(
        run_dir=args.run_dir,
        output=args.output,
        stride=args.stride,
        fps=args.fps,
        first_tick=args.first_tick,
        last_tick=args.last_tick,
        ffmpeg=args.ffmpeg,
        codec=args.codec,
    )
    frames = video.render()
    print(f"{frames} frames at {video.fps} fps, {video.speed():g}x -> {video.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

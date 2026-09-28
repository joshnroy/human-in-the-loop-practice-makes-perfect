"""Render a SweepIntoDrawer3D-o5 cycle from its per-tick replay log.

    scripts/with_env.sh python scripts/render_sweep_drawer_reset.py \\
        --run-dir out/6 --output out/seed6.mp4

Reads `<run-dir>/replay_log.jsonl` (written by `measure_sweep_drawer_reset.py
--replay-log`) and `<run-dir>/cycle.json`. Nothing is simulated: every frame is the
logged joint positions put into the scene and drawn, from a three-quarter view and a
close-up of the drawer, with the step that was running captioned beneath.

The scene drawn is the task's own with its MimicLabs room around it (`scene_bg=True`).
The measurement runs without the room, so the log is carried over joint by joint, by
name, and refused if a joint it recorded is missing or a different size.
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

from hitl_pmp.environments.sweep_drawer3d.session import KinderImports
from hitl_pmp.environments.sweep_drawer3d.types import SweepDrawerScene


class View(BaseModel):
    """A free camera: where it looks, from how far, from which side, from how high."""

    lookat: tuple[float, float, float]
    distance: float
    azimuth: float
    elevation: float
    title: str
    width: int


class Layout:
    """Two panels side by side over a caption bar; every dimension even, as NVENC needs."""

    HEIGHT: ClassVar[int] = 720
    BAR: ClassVar[int] = 64
    BACKGROUND: ClassVar[tuple[int, int, int]] = (16, 16, 16)
    FOREGROUND: ClassVar[tuple[int, int, int]] = (238, 238, 238)
    MUTED: ClassVar[tuple[int, int, int]] = (160, 160, 160)
    VIEWS: ClassVar[tuple[View, ...]] = (
        View(
            lookat=(0.95, 0.0, 0.35),
            distance=2.5,
            azimuth=140.0,
            elevation=-28.0,
            title="three-quarter view",
            width=1152,
        ),
        View(
            lookat=(1.06, 0.05, 0.22),
            distance=0.85,
            azimuth=60.0,
            elevation=-50.0,
            title="the drawer, close up",
            width=768,
        ),
    )

    @staticmethod
    def size() -> tuple[int, int]:
        """(width, height) of a whole frame."""
        return sum(v.width for v in Layout.VIEWS), Layout.HEIGHT + Layout.BAR

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
        draw = ImageDraw.Draw(frame)
        left = 0
        for view, panel in zip(Layout.VIEWS, panels, strict=True):
            frame.paste(Image.fromarray(panel), (left, 0))
            font = Layout.font(size=17)
            box = draw.textbbox((left + 18, 14), view.title, font=font)
            # the panel behind may be any brightness: the title brings its own ground
            draw.rectangle((box[0] - 8, box[1] - 6, box[2] + 8, box[3] + 6), fill=Layout.BACKGROUND)
            draw.text((left + 18, 14), view.title, fill=Layout.FOREGROUND, font=font)
            left += view.width
        top = Layout.HEIGHT
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
        *, seed: int, tick: int, step: dict | None, drawer: float, piled: int, speed: float
    ) -> tuple[str, str]:
        if step is None:
            head = "start"
        else:
            phase = Caption.PHASES.get(step["phase"], step["phase"])
            head = f"{phase}:  {step['name'].replace('_', ' ')}"
        cubes = len(SweepDrawerScene.CUBES)
        tail = (
            f"seed {seed}   |   simulated time {tick / 10:6.1f} s"
            f"   |   drawer open {max(drawer, 0.0) * 100:4.1f} cm"
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
    def mapping(*, header: dict, model: Any) -> list[tuple[int, int, int]]:
        """(address in the log, address in the scene, size) for every logged joint.

        By name, not by position: the scene drawn has a room the measured one lacks, and a
        moved `reference/kindergarden` pin can reorder joints under the same names.
        """
        import mujoco

        sizes = {
            int(mujoco.mjtJoint.mjJNT_FREE): 7,
            int(mujoco.mjtJoint.mjJNT_BALL): 4,
            int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
            int(mujoco.mjtJoint.mjJNT_HINGE): 1,
        }
        out = []
        for joint in header["joints"]:
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint["name"])
            if j < 0:
                raise ValueError(f"the scene has no joint {joint['name']}, which the log recorded")
            if sizes[int(model.jnt_type[j])] != joint["n"]:
                raise ValueError(f"joint {joint['name']} is a different kind in the scene")
            out.append((int(joint["adr"]), int(model.jnt_qposadr[j]), int(joint["n"])))
        return out


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
    room: bool = True

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

    def scene(self, *, seed: int) -> tuple[Any, Any, Any]:
        """(environment, model, data) of the scene to draw into."""
        kinder = KinderImports.load()
        env = kinder.make(
            SweepDrawerScene.ENV_ID,
            render_mode="rgb_array",
            allow_state_access=True,
            scene_bg=True if self.room else None,
        )
        env.reset(seed=seed)
        sim = env.unwrapped._object_centric_env._robot_env.sim
        return env, sim.model.mj_model, sim.data.mj_data

    def render(self) -> int:
        import mujoco

        cycle = json.loads((self.run_dir / "cycle.json").read_text())
        log = self.run_dir / "replay_log.jsonl"
        header = ReplayLog.header(path=log)
        seed = int(header["seed"])
        env, model, data = self.scene(seed=seed)
        mapping = ReplayLog.mapping(header=header, model=model)
        widest = max(v.width for v in Layout.VIEWS)
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, widest)
        model.vis.global_.offheight = max(model.vis.global_.offheight, Layout.HEIGHT)
        renderers, cameras = [], []
        for view in Layout.VIEWS:
            renderers.append(mujoco.Renderer(model, height=Layout.HEIGHT, width=view.width))
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = view.lookat
            camera.distance, camera.azimuth, camera.elevation = (
                view.distance,
                view.azimuth,
                view.elevation,
            )
            cameras.append(camera)
        joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, SweepDrawerScene.DRAWER + "_joint"
        )
        drawer = int(model.jnt_qposadr[joint])
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
                for src, dst, n in mapping:
                    data.qpos[dst : dst + n] = record["qpos"][src : src + n]
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                panels = []
                for renderer, camera in zip(renderers, cameras, strict=True):
                    renderer.update_scene(data, camera=camera)
                    panels.append(renderer.render().copy())
                piled = sum(
                    ResetVideo.in_pile(position=data.qpos[adr : adr + 3]) for adr in cube_adr
                )
                lines = Caption.lines(
                    seed=seed,
                    tick=record["t"],
                    step=steps.get(record["t"]),
                    drawer=float(data.qpos[drawer]),
                    piled=piled,
                    speed=self.speed(),
                )
                pipe.stdin.write(Layout.compose(panels=panels, lines=lines).tobytes())
                frames += 1
            pipe.stdin.close()
            code = pipe.wait()
        for renderer in renderers:
            renderer.close()
        env.close()
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
    parser.add_argument(
        "--no-room", action="store_true", help="draw the scene without its MimicLabs room"
    )
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
        room=not args.no_room,
    )
    frames = video.render()
    print(f"{frames} frames at {video.fps} fps, {video.speed():g}x -> {video.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

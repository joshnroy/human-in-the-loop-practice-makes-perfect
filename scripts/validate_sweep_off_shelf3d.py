"""TEMPORARY scripted solvability check of SweepOffShelf3D: captioned videos in lab2.

Not a planner, not a `Method`, not a `HumanOracle`. Episodes are sequences of the
skills in `sweep_off_shelf3d_skills.py`; this file only records them (two cameras side
by side, a caption, a contact sheet) and reports what the physics did. A failure is a
finding it reports, never something it patches. Episodes:

1. `pick_wiper`: `pick_wiper`, then drive and turn the base while holding it.
2. `sweep_cubes_into_box`: `pick_wiper`, `sweep_shelf`, `return_wiper`; `x/5` cubes
   in the box by the env's own `cube_in_box`.
3. `cube_out_of_box_to_shelf`: setup drops `cube_0` into the box (the only teleport);
   the robot picks it out, sets it down on the floor, `pick_cube_from_floor`, and
   places it back on shelf index 1.
4. `cube_shelf_to_box`: `pick_cube_from_shelf` on the cube nearest the front, then
   `place_cube_in_box`.

`--height`/`--speed` set `sweep_shelf`'s parameters; with `--no-video` the skills run
headless, which is how a height x speed grid is run (one process per cell).

Argument parsing and the caption/sheet helpers test without a simulator.
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict

from scripts.sweep_off_shelf3d_skills import CUBES, ShelfSkills, SkillFailed, SweepSettings

# The task's front_view, and a free camera added here (not in the task) that looks
# into the cube slot from the front left, so the head, the cubes and the front edge show.
CAMERAS = ("front_view", "slot_view")
SLOT_LOOKAT = (1.45, 0.0, 0.29)
SLOT_DISTANCE = 0.85
SLOT_AZIMUTH = 40.0
SLOT_ELEVATION = -24.0
PANEL_SIZE = (640, 480)
FRAME_SIZE = (PANEL_SIZE[0] * len(CAMERAS), PANEL_SIZE[1])
EPISODES = (
    "pick_wiper",
    "sweep_cubes_into_box",
    "cube_out_of_box_to_shelf",
    "cube_shelf_to_box",
)
DEFAULT_HEIGHT = 0.0
DEFAULT_SPEED = 0.05


class Phase(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    start_tick: int
    end_tick: int
    in_box: int
    on_floor: int
    on_shelf: int
    error: str | None = None


class EpisodeResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    episode: str
    seed: int
    ticks: int
    phases: tuple[Phase, ...]
    in_box: int
    on_floor: int
    on_shelf: int
    success: bool
    height: float | None
    speed: float | None
    wiper_slip_deg: float
    robot_cupboard_contact_steps: int
    arm_cupboard_contact_steps: int
    robot_box_contact_steps: int
    notes: tuple[str, ...]
    grasps: tuple[dict[str, Any], ...]
    video: Path | None
    contact_sheet: Path | None


class ValidationPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    output_dir: Path
    seed: int = 0
    scene: str = "mimiclabs-lab2"
    episodes: tuple[str, ...] = EPISODES
    fps: int = 30
    tick_limit: int = 12000
    height: float = DEFAULT_HEIGHT
    speed: float = DEFAULT_SPEED
    suffix: str = ""
    video: bool = True


class Captions:
    @staticmethod
    def font(*, size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
        for candidate in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ):
            if Path(candidate).is_file():
                return ImageFont.truetype(candidate, size)
        return ImageFont.load_default()

    @staticmethod
    def stack(*, panels: list[np.ndarray]) -> np.ndarray:
        return np.concatenate([np.asarray(p, dtype=np.uint8) for p in panels], axis=1)

    @staticmethod
    def annotate(*, frame: np.ndarray, header: str, banner: str | None) -> np.ndarray:
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rectangle([0, 0, image.width, 26], fill=(0, 0, 0, 170))
        draw.text((8, 5), header, fill=(255, 255, 255, 255), font=Captions.font(size=15))
        if banner:
            draw.rectangle(
                [0, image.height - 44, image.width, image.height], fill=(190, 30, 30, 200)
            )
            draw.text(
                (8, image.height - 36),
                banner,
                fill=(255, 255, 255, 255),
                font=Captions.font(size=20),
            )
        return np.asarray(image)

    @staticmethod
    def contact_sheet(
        *, keyframes: list[tuple[str, np.ndarray]], path: Path, columns: int = 2
    ) -> Path:
        rows = (len(keyframes) + columns - 1) // columns
        width, height = FRAME_SIZE
        sheet = Image.new("RGB", (columns * width, rows * (height + 30)), (20, 20, 20))
        draw = ImageDraw.Draw(sheet)
        font = Captions.font(size=18)
        for index, (caption, frame) in enumerate(keyframes):
            x = (index % columns) * width
            y = (index // columns) * (height + 30)
            sheet.paste(Image.fromarray(frame), (x, y + 30))
            draw.text((x + 8, y + 6), caption, fill=(255, 255, 255), font=font)
        sheet.save(path)
        return path


class VideoWriter:
    """Stream RGB frames to ffmpeg; NVENC when the binary has it, else libx264."""

    def __init__(self, *, path: Path, fps: int) -> None:
        import imageio_ffmpeg

        binary = imageio_ffmpeg.get_ffmpeg_exe()
        encoders = subprocess.run(
            [binary, "-hide_banner", "-encoders"], capture_output=True, text=True, check=False
        ).stdout
        self.encoder = "h264_nvenc" if "h264_nvenc" in encoders else "libx264"
        width, height = FRAME_SIZE
        self.process = subprocess.Popen(
            [
                binary,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-c:v",
                self.encoder,
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )
        self.frames = 0

    def write(self, *, frame: np.ndarray) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        self.frames += 1

    def close(self) -> None:
        assert self.process.stdin is not None
        if self.process.stdin.closed:
            return
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError(f"ffmpeg ({self.encoder}) exited with {self.process.returncode}")


class VideoRecorder:
    """Renders every control step into a captioned two-camera video and keeps
    keyframes for a contact sheet; the skills know nothing about it."""

    def __init__(self, *, path: Path, fps: int, episode: str, seed: int) -> None:
        self.path = path
        self.fps = fps
        self.episode = episode
        self.seed = seed
        self.writer = VideoWriter(path=path, fps=fps)
        self.keyframes: list[tuple[str, np.ndarray]] = []
        self.banner: str | None = None
        self.banner_until = -1
        self.renderer: Any = None
        self.camera: Any = None

    def attach(self, *, skills: ShelfSkills) -> None:
        mujoco = skills.mujoco
        self.renderer = mujoco.Renderer(skills.model, height=PANEL_SIZE[1], width=PANEL_SIZE[0])
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = SLOT_LOOKAT
        self.camera.distance = SLOT_DISTANCE
        self.camera.azimuth = SLOT_AZIMUTH
        self.camera.elevation = SLOT_ELEVATION

    def header(self, *, skills: ShelfSkills) -> str:
        c = skills.counts()
        return (
            f"SweepOffShelf3D seed {self.seed} | {self.episode} | step {skills.ticks} | "
            f"{skills.phase} | in box {c['in_box']}/5 | floor {c['on_floor']} | "
            f"shelf {c['on_shelf']}"
            + (f" | grasp turned {skills.slip_deg:.0f} deg" if skills.in_hand is not None else "")
        )

    def frame(self, *, skills: ShelfSkills) -> np.ndarray:
        panels = []
        for camera in CAMERAS:
            if camera == "slot_view":
                self.renderer.update_scene(skills.data, camera=self.camera)
                panels.append(self.renderer.render())
                continue
            frame = skills.sim.render(
                width=PANEL_SIZE[0], height=PANEL_SIZE[1], camera_name=camera, mode="offscreen"
            )
            panels.append(np.asarray(frame[0] if isinstance(frame, tuple) else frame))
        banner = self.banner if skills.ticks <= self.banner_until else None
        return Captions.annotate(
            frame=Captions.stack(panels=panels), header=self.header(skills=skills), banner=banner
        )

    def on_step(self, *, skills: ShelfSkills, label: str) -> None:
        del label
        self.writer.write(frame=self.frame(skills=skills))

    def mark(self, *, skills: ShelfSkills, label: str) -> None:
        frame = self.frame(skills=skills)
        self.writer.write(frame=frame)
        self.keyframes.append((f"step {skills.ticks}: {label}", frame))

    def show_banner(self, *, text: str, ticks: int) -> None:
        self.banner = text
        self.banner_until = ticks + self.fps

    def finish(self, *, sheet: Path) -> Path:
        self.writer.close()
        if self.renderer is not None:
            self.renderer.close()
        keyframes = self.keyframes
        if len(keyframes) > 8:
            picks = np.linspace(0, len(keyframes) - 1, 8).round().astype(int)
            keyframes = [keyframes[i] for i in picks]
        return Captions.contact_sheet(keyframes=keyframes, path=sheet)


class Episode:
    """One episode: a sequence of skills, each a phase that records its outcome."""

    def __init__(self, *, plan: ValidationPlan, episode: str) -> None:
        self.plan = plan
        self.episode = episode
        self.stem = f"sweep_off_shelf3d_{episode}_seed{plan.seed}{plan.suffix}"
        plan.output_dir.mkdir(parents=True, exist_ok=True)
        self.recorder = (
            VideoRecorder(
                path=plan.output_dir / f"{self.stem}.mp4",
                fps=plan.fps,
                episode=episode,
                seed=plan.seed,
            )
            if plan.video
            else None
        )
        self.skills = ShelfSkills(
            seed=plan.seed, scene=plan.scene, tick_limit=plan.tick_limit, recorder=self.recorder
        )
        if self.recorder is not None:
            self.recorder.attach(skills=self.skills)
        self.phases: list[Phase] = []
        self.picked = ""

    def phase(self, *, name: str, action: Any) -> bool:
        skills = self.skills
        skills.phase = name
        start = skills.ticks
        error = None
        try:
            action()
        except SkillFailed as exc:
            error = repr(exc)
        if error and self.recorder is not None:
            self.recorder.show_banner(text=f"{name} FAILED", ticks=skills.ticks)
        skills.mark(label=f"{name} ended" + (" with error" if error else ""))
        record = Phase(
            name=name, start_tick=start, end_tick=skills.ticks, error=error, **skills.counts()
        )
        self.phases.append(record)
        print(json.dumps(record.model_dump()), flush=True)
        return error is None

    def carry_wiper(self) -> None:
        skills = self.skills
        here = skills.data.qpos[skills.ba].copy()
        for target in (
            here + [-0.30, 0.0, 0.0],
            here + [-0.30, 0.0, 0.5],
            here + [-0.30, 0.0, 0.0],
            here,
        ):
            skills.move_base(target=target, label="carry_wiper")
        skills.hold(count=20, label="hold_after_carry")
        if skills.wiper_lift() < 0.05:
            raise SkillFailed("carry_wiper: the wiper dropped")

    def pick_a_shelf_cube(self) -> None:
        """`pick_cube_from_shelf` on the cubes nearest the front first. A cube with no
        collision-free route raises before the robot moves, so the next is tried."""
        skills = self.skills
        cubes = sorted(
            skills.shelf_cubes(),
            key=lambda c: skills.cube_position(cube=c)[0] + abs(skills.cube_position(cube=c)[1]),
        )
        for cube in cubes:
            try:
                skills.pick_cube_from_shelf(cube=cube)
            except SkillFailed as exc:
                if "no collision-free route" in str(exc):
                    skills.notes.append(f"{cube}: {exc}")
                    continue
                raise
            self.picked = cube
            return
        raise SkillFailed("no cube on the shelf has a collision-free route")

    def run(self) -> EpisodeResult:
        skills = self.skills
        skills.mark(label="reset")
        plan = self.plan
        success = False
        if self.episode == "pick_wiper":
            success = self.phase(name="pick_wiper", action=skills.pick_wiper) and self.phase(
                name="carry_wiper", action=self.carry_wiper
            )
        elif self.episode == "sweep_cubes_into_box":
            if self.phase(name="pick_wiper", action=skills.pick_wiper):
                self.phase(
                    name=f"sweep_shelf(h={plan.height}, v={plan.speed})",
                    action=lambda: skills.sweep_shelf(height=plan.height, speed=plan.speed),
                )
                if skills.in_hand is not None:
                    self.phase(name="return_wiper", action=skills.return_wiper)
            if skills.ticks + 30 <= plan.tick_limit:
                skills.hold(count=30, label="settle")
            success = skills.counts()["in_box"] == len(CUBES)
        elif self.episode == "cube_out_of_box_to_shelf":
            cube = "cube_0"
            success = (
                self.phase(
                    name="setup_drop_cube_into_box",
                    action=lambda: skills.drop_cube_into_box(cube=cube),
                )
                and self.phase(
                    name="pick_cube_from_box", action=lambda: skills.pick_cube_from_box(cube=cube)
                )
                and self.phase(
                    name="set_cube_down_on_floor",
                    action=lambda: skills.set_cube_down_on_floor(cube=cube),
                )
                and self.phase(
                    name="pick_cube_from_floor",
                    action=lambda: skills.pick_cube_from_floor(cube=cube),
                )
                and self.phase(
                    name="place_cube_on_shelf", action=lambda: skills.place_cube_on_shelf(cube=cube)
                )
            )
        elif self.episode == "cube_shelf_to_box":
            success = self.phase(
                name="pick_cube_from_shelf", action=self.pick_a_shelf_cube
            ) and self.phase(
                name="place_cube_in_box",
                action=lambda: skills.place_cube_in_box(cube=self.picked),
            )
        else:
            raise ValueError(f"unknown episode {self.episode!r}")
        skills.phase = "end"
        skills.mark(label="end")
        sheet = None
        if self.recorder is not None:
            sheet = self.recorder.finish(sheet=plan.output_dir / f"{self.stem}_contact_sheet.png")
        sweep = self.episode == "sweep_cubes_into_box"
        return EpisodeResult(
            episode=self.episode,
            seed=plan.seed,
            ticks=skills.ticks,
            phases=tuple(self.phases),
            success=success,
            height=plan.height if sweep else None,
            speed=plan.speed if sweep else None,
            wiper_slip_deg=skills.slip_deg,
            robot_cupboard_contact_steps=skills.cupboard_contact_steps,
            arm_cupboard_contact_steps=skills.arm_cupboard_contact_steps,
            robot_box_contact_steps=skills.box_contact_steps,
            notes=tuple(skills.notes),
            grasps=tuple(skills.grasps_taken),
            video=None if self.recorder is None else self.recorder.path,
            contact_sheet=sheet,
            **skills.counts(),
        )

    def close(self) -> None:
        self.skills.close()


class Validator:
    @staticmethod
    def run(*, plan: ValidationPlan) -> list[EpisodeResult]:
        results = []
        for name in plan.episodes:
            episode = Episode(plan=plan, episode=name)
            try:
                results.append(episode.run())
            finally:
                episode.close()
        out = plan.output_dir / f"sweep_off_shelf3d_seed{plan.seed}{plan.suffix}_validation.json"
        out.write_text(json.dumps([r.model_dump(mode="json") for r in results], indent=2) + "\n")
        return results

    @staticmethod
    def parse_args(*, argv: list[str] | None = None) -> ValidationPlan:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument(
            "--scene", default="mimiclabs-lab2", choices=["mimiclabs-lab2", "simple"]
        )
        parser.add_argument("--episodes", nargs="+", choices=EPISODES, default=list(EPISODES))
        parser.add_argument("--fps", type=int, default=30)
        parser.add_argument("--tick-limit", type=int, default=12000)
        parser.add_argument("--height", type=float, default=DEFAULT_HEIGHT)
        parser.add_argument("--speed", type=float, default=DEFAULT_SPEED)
        parser.add_argument("--suffix", default="")
        parser.add_argument("--no-video", action="store_true")
        args = parser.parse_args(argv)
        SweepSettings(height=args.height, speed=args.speed)
        return ValidationPlan(
            output_dir=args.output_dir,
            seed=args.seed,
            scene=args.scene,
            episodes=tuple(args.episodes),
            fps=args.fps,
            tick_limit=args.tick_limit,
            height=args.height,
            speed=args.speed,
            suffix=args.suffix,
            video=not args.no_video,
        )

    @staticmethod
    def main() -> None:
        plan = Validator.parse_args()
        for result in Validator.run(plan=plan):
            print(json.dumps(result.model_dump(mode="json"), indent=2), flush=True)


if __name__ == "__main__":
    Validator.main()

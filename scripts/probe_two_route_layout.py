"""Count the dumplings each bulk skill lands on the plate from a start layout.

Per env seed: reset, PickScraper, then one Sweep(d) or GatherAndSweep(s, delta, d),
and report how many dumplings are settled on the plate: `count` as the skill returns
(what practice and evaluation observe), and `count_after_1s` after one more second of
physics. One JSON line per seed. `--video DIR` also writes `<skill>-seed<seed>.mp4`
per seed (15 Hz capture at 30 fps, so 2x), in the lab2 scene. Probe only; never
launches learning.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
from hitl_pmp.environments.racktoserving3d.skills import SKILL_IDS

VIEW = 480
CAPTURE_HZ = 15


class TwoRouteProbeVideo:
    """Two fixed views of the live scene, captured from the backend's tick observer."""

    def __init__(self, *, backend: Any, path: Path, title: str) -> None:
        import imageio.v2 as imageio
        import mujoco

        self.backend, self.title = backend, title
        model = backend.model()
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, VIEW)
        model.vis.global_.offheight = max(model.vis.global_.offheight, VIEW)
        self.cameras = []
        for lookat, distance, azimuth, elevation in (
            ([1.45, 0.0, 0.95], 2.1, 35, -38),
            ([1.40, 0.0, 0.92], 0.80, 180, -89.5),
        ):
            camera = mujoco.MjvCamera()
            camera.lookat[:] = lookat
            camera.distance, camera.azimuth, camera.elevation = distance, azimuth, elevation
            self.cameras.append(camera)
        self.renderer = mujoco.Renderer(model, height=VIEW, width=VIEW)
        self.writer = imageio.get_writer(
            str(path),
            fps=2 * CAPTURE_HZ,
            codec="libx264",
            quality=7,
            ffmpeg_params=["-threads", "1"],
        )
        self.last = -1.0
        self.label = "start"

    def frame(self, *, repeat: int = 1, force: bool = False) -> None:
        from PIL import Image, ImageDraw, ImageFont

        data = self.backend.data()
        if not force and data.time - self.last < 1 / CAPTURE_HZ - 1e-9:
            return
        self.last = float(data.time)
        views = []
        for camera in self.cameras:
            self.renderer.update_scene(data, camera=camera)
            views.append(self.renderer.render().copy())
        canvas = Image.new("RGB", (2 * VIEW, VIEW + 28), "#111827")
        canvas.paste(Image.fromarray(np.concatenate(views, axis=1)), (0, 28))
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
        ImageDraw.Draw(canvas).text(
            (8, 5),
            f"{self.title} | {self.label} | plate {self.backend.plate_count()}/5 | 2x",
            font=font,
            fill="white",
        )
        pixels = np.asarray(canvas)
        for _ in range(repeat):
            self.writer.append_data(pixels)

    def close(self) -> None:
        self.writer.close()
        self.renderer.close()


def main() -> None:
    import mujoco

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", choices=("Sweep", "GatherAndSweep"), required=True)
    parser.add_argument("--params", type=float, nargs=3, default=(0.0, 0.0, 0.08))
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--start-layout", default="spaced")
    parser.add_argument("--pitch", type=float, default=RackServingConfig().spaced_pitch)
    parser.add_argument("--long-axis", default=RackServingConfig().spaced_long_axis)
    parser.add_argument("--video", type=Path, default=None)
    args = parser.parse_args()
    config = RackServingConfig(
        start_layout=args.start_layout,
        spaced_pitch=args.pitch,
        spaced_long_axis=args.long_axis,
        surface_kind="wire_solid",
        scraper_attachment="grasp_weld",
        bulk_transfer_only=True,
    )
    env = RackServingEnvironment(
        config=config, scene_bg=args.video is not None, skill_step_limit=12000
    )
    try:
        for seed in args.seeds:
            env.reset_to_seed(seed=seed)
            backend = env.backend()
            start = [backend.food_centre(name=name) for name in backend.cube_names]
            row: dict[str, object] = {
                "seed": seed,
                "skill": args.skill,
                "params": list(args.params),
                "start_kind": backend.start_kind(),
                "start_y": [round(y, 4) for _, y in start],
            }
            video = None
            if args.video is not None:
                args.video.mkdir(parents=True, exist_ok=True)
                shown = args.params[2:] if args.skill == "Sweep" else args.params
                video = TwoRouteProbeVideo(
                    backend=backend,
                    path=args.video / f"{args.skill}-seed{seed}.mp4",
                    title=f"{args.skill}({', '.join(f'{v:g}' for v in shown)}), spaced start,"
                    f" env seed {seed}",
                )
                video.frame(repeat=15, force=True)
                backend.set_tick_observer(observer=video.frame)
            try:
                if video is not None:
                    video.label = "PickScraper"
                env.take_action(action=np.array([SKILL_IDS["PickScraper"], 0, 0.0, 0.0, 0.0]))
                before = backend.control_steps()
                if video is not None:
                    video.label = args.skill
                env.take_action(action=np.array([SKILL_IDS[args.skill], 0, *args.params]))
                row["steps"] = backend.control_steps() - before
                row["skill_error"] = env.last_skill_error()
                served = [name for name in backend.cube_names if backend.settled_on_tray(name=name)]
                row["served"] = served
                row["locations"] = [backend.location(name=name) for name in backend.cube_names]
                row["count"] = len(served)
                model, data = backend.model(), backend.data()
                for _ in range(int(1.0 / model.opt.timestep)):
                    mujoco.mj_step(model, data)
                    if video is not None:
                        video.frame()
                row["count_after_1s"] = backend.plate_count()
                row["locations_after_1s"] = [
                    backend.location(name=name) for name in backend.cube_names
                ]
                if video is not None:
                    video.label = "done"
                    video.frame(repeat=45, force=True)
            except Exception as error:  # noqa: BLE001 - a probe reports every failure
                row["error"] = f"{type(error).__name__}: {error}"
            finally:
                backend.set_tick_observer(observer=None)
                if video is not None:
                    video.close()
            print(json.dumps(row), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()

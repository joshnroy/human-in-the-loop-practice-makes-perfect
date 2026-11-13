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

VIEW = 416
CAPTURE_HZ = 15


class TwoRouteProbeVideo:
    """Four fixed views of the live scene, captured from the backend's tick observer:
    oblique, top close-up, and two low views of the rack's front edge and the plate's
    near rim (from the plate side, and from the side), where stranded dumplings rest."""

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
            ([1.345, 0.0, 0.925], 0.42, 0, -14),
            ([1.345, 0.0, 0.925], 0.50, 90, -8),
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
        self.frames = 0
        self.marks: dict[str, float] = {}

    def mark(self, *, label: str) -> None:
        """Start a labelled phase and remember the clip time it starts at."""
        self.label = label
        self.marks[label] = round(self.frames / (2 * CAPTURE_HZ), 1)

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
        canvas = Image.new("RGB", (len(views) * VIEW, VIEW + 32), "#111827")
        canvas.paste(Image.fromarray(np.concatenate(views, axis=1)), (0, 32))
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
        ImageDraw.Draw(canvas).text(
            (8, 5),
            f"{self.title} | {self.label} | plate {self.backend.plate_count()}/5 | 2x | "
            f"clip t={self.frames / (2 * CAPTURE_HZ):.1f}s | views: oblique, top, "
            "plate-side low, side low",
            font=font,
            fill="white",
        )
        pixels = np.asarray(canvas)
        for _ in range(repeat):
            self.writer.append_data(pixels)
        self.frames += repeat

    def close(self) -> None:
        self.writer.close()
        self.renderer.close()


class TwoRouteProbeSafety:
    """Whether the blade stayed clear of the plate: its lowest point while over the
    plate's footprint, and every tick the scraper touched the plate."""

    def __init__(self, *, backend: Any, tool: Any, also: Any = None) -> None:
        self.backend, self.tool, self.also = backend, tool, also
        self.model, self.data = backend.model(), backend.data()
        self.scraper = self.model.body("wiper_0").id
        self.plate = self.model.body("box_0").id
        self.near_x = backend.serving_near_edge_x()
        self.lowest = float("inf")
        self.contact_ticks = 0
        self.deepest = 0.0

    def tick(self) -> None:
        ends = self.tool.blade_edge()
        if float(ends[:, 0].mean()) <= self.near_x:
            self.lowest = min(self.lowest, float(ends[:, 2].min()))
        touched = False
        for contact in self.data.contact[: self.data.ncon]:
            bodies = {
                int(self.model.geom_bodyid[contact.geom1]),
                int(self.model.geom_bodyid[contact.geom2]),
            }
            if bodies == {self.scraper, self.plate} and contact.dist <= 0:
                touched = True
                self.deepest = max(self.deepest, float(-contact.dist))
        self.contact_ticks += touched
        if self.also is not None:
            self.also()

    def report(self) -> dict[str, Any]:
        return {
            "blade_lowest_z_over_plate": None
            if self.lowest == float("inf")
            else round(self.lowest, 4),
            "rim_top_z": round(float(self.backend.config.corelle_rim_top_z), 4),
            "scraper_plate_contact_ticks": self.contact_ticks,
            "deepest_scraper_plate_penetration_m": round(self.deepest, 5),
        }


class TwoRouteProbeFoods:
    """Where each dumpling ended: pose, every body it touches, and the surface under it."""

    @staticmethod
    def describe(*, backend: Any) -> list[dict[str, Any]]:
        model, data = backend.model(), backend.data()
        well = backend.corelle_support()
        rim_radius = backend.config.corelle_diameter / 2
        rack = backend.support(name="rack_0")
        rows = []
        for name in backend.cube_names:
            body = model.body(name).id
            touching: set[str] = set()
            for contact in data.contact[: data.ncon]:
                if contact.dist > 0.001:
                    continue
                bodies = {
                    int(model.geom_bodyid[contact.geom1]),
                    int(model.geom_bodyid[contact.geom2]),
                }
                if body in bodies:
                    touching |= {model.body(other).name or "world" for other in bodies - {body}}
            x, y, z = (float(v) for v in data.xpos[body])
            radius = float(np.hypot(x - well.center_x, y - well.center_y))
            location = backend.location(name=name)
            if location == "OnTray":
                surface = "plate well"
            elif "box_0" in touching:
                surface = "plate rim" if radius > well.radius else "plate, not resting in the well"
            elif location == "SpilledCounter":
                surface = "counter beside the rack" if x > rack.x_min else "counter"
            elif location == "OnRack":
                surface = "rack"
            else:
                surface = "none of these"
            rows.append({
                "name": name,
                "xyz": [round(x, 3), round(y, 3), round(z, 3)],
                "plate_centre_distance": round(radius, 3),
                "well_radius": round(float(well.radius), 3),
                "rim_radius": round(float(rim_radius), 3),
                "upright_cosine": round(float(data.xmat[body].reshape(3, 3)[2, 2]), 2),
                "location": location,
                "settled_on_plate": bool(backend.settled_on_tray(name=name)),
                "touching": sorted(touching),
                "below": sorted(backend.supporting_bodies(name=name)),
                "surface": surface,
            })
        return rows


def main() -> None:
    import mujoco

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skill",
        choices=("Sweep", "GatherAndSweep", "PickScraperCycle"),
        required=True,
        help="PickScraperCycle: PickScraper then the person's tool recovery, three times.",
    )
    parser.add_argument("--banner", default="", help="--video only: text leading the banner")
    parser.add_argument("--clip-name", default=None, help="--video only: file stem")
    parser.add_argument("--params", type=float, nargs=3, default=(0.0, 0.0, 0.08))
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--start-layout", default="spaced")
    parser.add_argument("--pitch", type=float, default=RackServingConfig().spaced_pitch)
    parser.add_argument("--long-axis", default=RackServingConfig().spaced_long_axis)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--side-offset", type=float, default=RackServingConfig().side_offset)
    parser.add_argument("--blade-height-feedback-limit", type=float, default=0.0)
    parser.add_argument(
        "--scraper-tilt-degrees", type=float, default=RackServingConfig().scraper_tilt_degrees
    )
    args = parser.parse_args()
    config = RackServingConfig(
        start_layout=args.start_layout,
        spaced_pitch=args.pitch,
        spaced_long_axis=args.long_axis,
        side_offset=args.side_offset,
        surface_kind="wire_solid",
        scraper_attachment="grasp_weld",
        bulk_transfer_only=True,
        blade_height_feedback_limit=args.blade_height_feedback_limit,
        scraper_tilt_degrees=args.scraper_tilt_degrees,
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
                    path=args.video / f"{args.clip_name or args.skill}-seed{seed}.mp4",
                    title=(f"{args.banner} | " if args.banner else "")
                    + f"{args.skill}({', '.join(f'{v:g}' for v in shown)}), env seed {seed}",
                )
                video.frame(repeat=15, force=True)
                backend.set_tick_observer(observer=video.frame)
            try:
                if video is not None:
                    video.mark(label="PickScraper")
                env.take_action(action=np.array([SKILL_IDS["PickScraper"], 0, 0.0, 0.0, 0.0]))
                before = backend.control_steps()
                if args.skill == "PickScraperCycle":
                    errors = [env.last_skill_error()]
                    for cycle in range(2, 4):
                        if video is not None:
                            video.mark(label=f"human tool recovery {cycle - 1}")
                            video.frame(repeat=20, force=True)
                        env.reset_movables(destination="human_tool_recovery")
                        if video is not None:
                            video.frame(repeat=20, force=True)
                            video.mark(label=f"PickScraper {cycle}")
                        env.take_action(
                            action=np.array([SKILL_IDS["PickScraper"], 0, 0.0, 0.0, 0.0])
                        )
                        errors.append(env.last_skill_error())
                    row["pick_errors"] = errors
                    if video is not None:
                        video.mark(label="done")
                        video.frame(repeat=45, force=True)
                        row["clip_marks"] = video.marks
                    print(json.dumps(row), flush=True)
                    continue
                if video is not None:
                    video.mark(label=args.skill)
                safety = TwoRouteProbeSafety(
                    backend=backend,
                    tool=env.rack_controllers().tool,
                    also=video.frame if video is not None else None,
                )
                backend.set_tick_observer(observer=safety.tick)
                env.take_action(action=np.array([SKILL_IDS[args.skill], 0, *args.params]))
                backend.set_tick_observer(observer=video.frame if video is not None else None)
                row["blade_safety"] = safety.report()
                row["steps"] = backend.control_steps() - before
                row["skill_error"] = env.last_skill_error()
                served = [name for name in backend.cube_names if backend.settled_on_tray(name=name)]
                row["served"] = served
                row["locations"] = [backend.location(name=name) for name in backend.cube_names]
                row["count"] = len(served)
                row["foods"] = TwoRouteProbeFoods.describe(backend=backend)
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
                    video.mark(label="done")
                    video.frame(repeat=60, force=True)
                    row["clip_marks"] = video.marks
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

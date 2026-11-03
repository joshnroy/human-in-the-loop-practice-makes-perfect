"""Drive the arm through PickScraper then the shipped GatherAndSweep(s, delta, d) on one layout.

Layouts come from the same RNG as the driven-blade grid (rng 20261103, five foods
uniform on the rack with 3 cm margins, yaw +/-45 deg, centres >= 5 cm apart);
``--layout -1`` keeps the environment's own seed-0 layout. Writes report.json (per
stroke reach and blade-end error from the controller's metrics, served count) and,
unless --no-video, a 2x full-motion video: 15 Hz capture written at 30 fps.
Probe only; never launches learning.
"""

import argparse
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.controllers import StrokeStances
from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
from hitl_pmp.environments.racktoserving3d.skills import SKILL_IDS

VIEW = 432
CAPTURE_HZ = 15


class Layouts:
    @staticmethod
    def centres(*, layout: int, rack, count: int) -> tuple[list[np.ndarray], list[float]]:
        rng = np.random.default_rng(20261103)
        centres: list[np.ndarray] = []
        yaws: list[float] = []
        for _ in range(layout + 1):
            centres = []
            while len(centres) < count:
                p = np.array([
                    rng.uniform(rack.x_min + 0.03, rack.x_max - 0.03),
                    rng.uniform(rack.y_min + 0.03, rack.y_max - 0.03),
                ])
                if all(np.linalg.norm(p - q) >= 0.05 for q in centres):
                    centres.append(p)
            yaws = [rng.uniform(-45, 45) for _ in centres]
        return centres, yaws


class ClearanceDiagnosis:
    @staticmethod
    def explain(*, tool, pose, data) -> dict:
        """Which fixture geoms a rejected scraper pose overlaps, and by how much."""
        from hitl_pmp.environments.racktoserving3d.tool import box_overlap
        from hitl_pmp.environments.racktoserving3d.transit import geom_half_extents

        data = tool.data if data is None else data
        shelf, model = tool.shelf, tool.model
        hits = []
        for geom in (tool.head_geom, tool.handle_geom):
            center = pose[:3, :3] @ model.geom_pos[geom] + pose[:3, 3]
            rotation = pose[:3, :3] @ tool.geom_rotation(geom=geom)
            size = model.geom_size[geom]
            bottom = float(center[2] - np.abs(rotation[2]) @ size)
            for obstacle in shelf.cupboard_geoms | shelf.box_geoms:
                other_center = data.geom_xpos[obstacle]
                other_rotation = data.geom_xmat[obstacle].reshape(3, 3)
                other_size = geom_half_extents(model=model, geom=obstacle)
                overlap = box_overlap(
                    center=center,
                    rotation=rotation,
                    size=size,
                    other_center=other_center,
                    other_rotation=other_rotation,
                    other_size=other_size,
                )
                if overlap > 1e-9:
                    hits.append({
                        "tool_geom": "head" if geom == tool.head_geom else "handle",
                        "obstacle": model.geom(obstacle).name
                        or f"{model.body(model.geom_bodyid[obstacle]).name}/{obstacle}",
                        "overlap": float(overlap),
                        "tool_bottom": bottom,
                        "obstacle_top": float(
                            other_center[2] + np.abs(other_rotation[2]) @ other_size
                        ),
                    })
        return {
            "blade_edge": tool.blade_edge(pose=pose).mean(axis=0).round(4).tolist(),
            "hits": hits,
        }


def main():
    import mujoco

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", type=int, default=-1)
    parser.add_argument("--s", type=float, required=True)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--d", type=float, required=True)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--side-exit", choices=("front", "rear_lane"), default=None)
    parser.add_argument(
        "--side-offset", type=float, default=None, help="override StrokeStances.side_offset"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output directory")
    args.output.mkdir(parents=True)
    config = RackServingConfig(cooling_seconds=0)
    env = RackServingEnvironment(config=config, scene_bg=True, skill_step_limit=8000)
    report = {"args": {k: str(v) for k, v in vars(args).items()}, "phases": []}
    renderer = writer = None
    label = "initial"
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=1) + "\n")

    try:
        env.reset_to_seed(seed=0)
        backend = env.backend()
        model, data = backend.model(), backend.data()
        if args.layout >= 0:
            rack = backend.support(name="rack_0")
            centres, yaws = Layouts.centres(
                layout=args.layout, rack=rack, count=len(backend.cube_names)
            )
            for name, p, yaw in zip(backend.cube_names, centres, yaws, strict=True):
                joint = model.body_jntadr[model.body(name).id]
                qa, va = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
                q = Rotation.from_euler("z", yaw, degrees=True).as_quat()
                z = rack.top_z + config.dumpling_height / 2 + 0.002
                data.qpos[qa : qa + 7] = [p[0], p[1], z, q[3], *q[:3]]
                data.qvel[va : va + 6] = 0
            mujoco.mj_forward(model, data)
            for _ in range(int(0.5 / model.opt.timestep)):
                mujoco.mj_step(model, data)
        cameras = []
        for lookat, distance, azimuth, elevation in (
            ([1.45, 0.0, 0.95], 2.1, 35, -38),
            ([1.55, 0.0, 0.9], 2.6, 180, -89.5),
            ([1.40, 0.0, 0.92], 0.80, 180, -89.5),
        ):
            camera = mujoco.MjvCamera()
            camera.lookat[:] = lookat
            camera.distance, camera.azimuth, camera.elevation = distance, azimuth, elevation
            cameras.append(camera)
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, VIEW)
        model.vis.global_.offheight = max(model.vis.global_.offheight, VIEW)
        if not args.no_video:
            renderer = mujoco.Renderer(model, height=VIEW, width=VIEW)
            writer = imageio.get_writer(
                str(args.output / "gather-and-sweep.mp4"),
                fps=2 * CAPTURE_HZ,
                codec="libx264",
                quality=7,
                ffmpeg_params=["-threads", "1"],
            )
        last_frame = [-1.0]

        def frame(*, repeat=1, force=False):
            if writer is None:
                return
            if not force and data.time - last_frame[0] < 1 / CAPTURE_HZ - 1e-9:
                return
            last_frame[0] = float(data.time)
            views = []
            for camera in cameras:
                renderer.update_scene(data, camera=camera)
                views.append(renderer.render().copy())
            canvas = Image.new("RGB", (3 * VIEW, VIEW + 48), "#111827")
            canvas.paste(Image.fromarray(np.concatenate(views, axis=1)), (0, 48))
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (8, 4),
                f"GatherAndSweep(s={args.s:.3f}, delta={args.delta:+.3f}, d={args.d:.3f})"
                f" | layout {args.layout} | real arm, 2x speed",
                font=font,
                fill="white",
            )
            locations = [backend.location(name=name) for name in backend.cube_names]
            served = sum(backend.settled_on_tray(name=name) for name in backend.cube_names)
            draw.text(
                (8, 26),
                f"{label[:70]} | t={data.time:.1f}s | rack {locations.count('OnRack')}/5"
                f" | plate {served}/5 | oblique | top | top close-up",
                font=font,
                fill="#fbbf24",
            )
            pixels = np.asarray(canvas)
            for _ in range(repeat):
                writer.append_data(pixels)

        def on_tick():
            nonlocal label
            stroke_labels = [row["label"] for row in controllers.metrics.get("strokes", [])]
            if label.startswith("GatherAndSweep"):
                label = f"GatherAndSweep: {len(stroke_labels)}/5 strokes done"
            ticks[0] += 1
            if ticks[0] % 5 == 0 and motions and motions[-1].endswith("_push"):
                trace.append({
                    "tick": ticks[0],
                    "motion": motions[-1],
                    "blade_edge": controllers.tool.blade_edge().mean(axis=0).round(4).tolist(),
                    "base": data.qpos[controllers.shelf.ba].round(4).tolist(),
                })
            frame()

        ticks = [0]
        trace: list[dict] = []
        report["push_trace"] = trace

        controllers = env.rack_controllers()
        overrides = {
            key: value
            for key, value in (("side_offset", args.side_offset), ("side_exit", args.side_exit))
            if value is not None
        }
        if overrides:
            controllers.stances = StrokeStances(**overrides)
        motions: list[str] = []
        for method in ("step", "line", "transit", "move_base"):
            original = getattr(controllers.tool, method)

            def traced(*, _original=original, _method=method, **kwargs):
                motions.append(f"{_method}:{kwargs.get('label')}")
                del motions[:-30]
                return _original(**kwargs)

            setattr(controllers.tool, method, traced)
        report["last_motions"] = motions
        report["clearance_failures"] = clearance_failures = []
        tool_clear = controllers.tool.tool_clear

        def traced_clear(*, pose, data=None):
            clear = tool_clear(pose=pose, data=data)
            if not clear:
                clearance_failures.append(
                    ClearanceDiagnosis.explain(tool=controllers.tool, pose=pose, data=data)
                )
                del clearance_failures[:-3]
            return clear

        controllers.tool.tool_clear = traced_clear
        frame(repeat=10, force=True)
        backend.set_tick_observer(observer=on_tick)

        def foods():
            return {
                name: {
                    "position": np
                    .asarray(backend.core().get_object(name).get_pose()[0])
                    .round(4)
                    .tolist(),
                    "location": backend.location(name=name),
                }
                for name in backend.cube_names
            }

        report["initial_foods"] = foods()
        for skill in ("PickScraper", "GatherAndSweep"):
            label = skill
            began, sim0 = time.monotonic(), float(data.time)
            env.take_action(
                action=np.array([SKILL_IDS[skill], 0, args.s, args.delta, args.d], dtype=float)
            )
            error = env.last_skill_error()
            row = {
                "skill": skill,
                "error": error,
                "wall_seconds": round(time.monotonic() - began, 1),
                "sim_seconds": round(float(data.time) - sim0, 2),
                "foods": foods(),
                "base": data.qpos[controllers.shelf.ba].round(4).tolist(),
                "held_scraper": backend.held(name="wiper_0"),
            }
            if skill == "GatherAndSweep":
                row["metrics"] = controllers.metrics
            report["phases"].append(row)
            save()
            print(json.dumps({k: v for k, v in row.items() if k != "foods"}), flush=True)
            if error:
                label = f"{skill} FAILED: {error}"
                frame(repeat=15, force=True)
                break
        for _ in range(int(1.0 / model.opt.timestep)):
            mujoco.mj_step(model, data)
        served = int(sum(backend.settled_on_tray(name=name) for name in backend.cube_names))
        report["served"] = served
        report["fully_served"] = served == len(backend.cube_names)
        report["final_foods"] = foods()
        label = f"done: {served}/5 served"
        frame(repeat=30, force=True)
    finally:
        save()
        env.backend().set_tick_observer(observer=None)
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        env.close()


if __name__ == "__main__":
    main()

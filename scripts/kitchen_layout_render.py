"""Render a `kitchen_layout_benchmark.py` run: every tick, at a fixed speed-up, labelled.

The main view frames the jug's table and the stove together; an inset follows the
skill (the pan while pouring, the gripper otherwise). `--still` writes one overhead
frame of the episode-initial layout instead of a video.
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hitl_pmp.environments.pouring3d.kitchen.config import KitchenGeometry
from hitl_pmp.environments.pouring3d.kitchen.native import KitchenController

WIDTH, HEIGHT = 1280, 720
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


class KitchenLayoutRender:
    @staticmethod
    def camera(*, lookat: np.ndarray, distance: float, azimuth: float, elevation: float):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat
        cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
        return cam

    @staticmethod
    def build(*, report: dict) -> KitchenController:
        c = KitchenController.build(
            geometry=KitchenGeometry.model_validate(report["geometry"]),
            deployment=False,
            requested=1,
            seed=report["seed"],
            scene_bg=True,
            surface_mu=0.1,
        )
        c.model.vis.global_.offwidth, c.model.vis.global_.offheight = WIDTH, HEIGHT
        return c

    @staticmethod
    def work_area(*, c: KitchenController) -> np.ndarray:
        jug = c.data.xpos[c.model.body("pitcher").id]
        pan = c.data.xpos[c.model.body("tray").id]
        return np.array([(jug[0] + pan[0]) / 2 - 0.25, (jug[1] + pan[1]) / 2, 0.75])

    @staticmethod
    def still(*, source: Path, output: Path, label: str) -> None:
        report = json.loads((source / "result.json").read_text())
        c = KitchenLayoutRender.build(report=report)
        renderer = mujoco.Renderer(c.model, height=HEIGHT, width=WIDTH)
        try:
            centre = KitchenLayoutRender.work_area(c=c)
            cam = KitchenLayoutRender.camera(
                lookat=centre + [0.25, 0, -0.75], distance=3.4, azimuth=180, elevation=-89
            )
            renderer.update_scene(c.data, camera=cam)
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, WIDTH, 40), fill=(20, 25, 30))
            draw.text((14, 8), label, fill="white", font=ImageFont.truetype(FONT, 22))
            frame.save(output)
        finally:
            renderer.close()
            c.env.close()

    @staticmethod
    def video(
        *, source: Path, output: Path, speed: int, label: str, azimuth: float, frames: int
    ) -> None:
        report = json.loads((source / "result.json").read_text())
        motion = np.load(source / "motion.npz")
        fps = round(speed * report["control_frequency"])
        c = KitchenLayoutRender.build(report=report)
        renderer = mujoco.Renderer(c.model, height=HEIGHT, width=WIDTH)
        inset = mujoco.Renderer(c.model, height=270, width=360)
        font = ImageFont.truetype(FONT, 22)
        small = ImageFont.truetype(FONT, 18)
        tray = c.model.body("tray").id
        writer = imageio.get_writer(output, fps=fps, codec="libx264", quality=8, macro_block_size=1)
        rows = report["rows"]
        try:
            centre = KitchenLayoutRender.work_area(c=c)
            main = KitchenLayoutRender.camera(
                lookat=centre, distance=2.9, azimuth=azimuth, elevation=-38
            )
            ordinal: dict[str, int] = {}
            labels = []
            previous = None
            for skill in motion["skills"]:
                if skill != previous:
                    ordinal[str(skill)] = len(labels)
                    labels.append(str(skill))
                    previous = skill
            boundary = -1
            previous = None
            for i, qpos in enumerate(motion["qpos"][: frames or None]):
                skill, phase = str(motion["skills"][i]), str(motion["phases"][i])
                if skill != previous:
                    boundary += 1
                    previous = skill
                c.data.qpos[:] = qpos
                c.data.qvel[:] = 0
                mujoco.mj_forward(c.model, c.data)
                renderer.update_scene(c.data, camera=main)
                frame = Image.fromarray(renderer.render())
                focus = (
                    c.data.xpos[tray] + [0, 0, 0.13]
                    if skill.startswith("Pour")
                    else c.data.site_xpos[c.site].copy()
                )
                inset.update_scene(
                    c.data,
                    camera=KitchenLayoutRender.camera(
                        lookat=focus, distance=0.8, azimuth=150, elevation=-30
                    ),
                )
                frame.paste(Image.fromarray(inset.render()), (WIDTH - 372, HEIGHT - 282))
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, WIDTH, 74), fill=(20, 25, 30))
                row = rows[boundary] if boundary < len(rows) else None
                status = "" if row is None else ("  ok" if row["success"] else "  FAILED")
                draw.text(
                    (14, 6),
                    f"{label} | seed {report['seed']} | {speed}x | skill {boundary + 1}/"
                    f"{len(labels)}: {skill}{status}",
                    fill="white",
                    font=font,
                )
                draw.text(
                    (14, 40),
                    f"tick {i} | {phase.replace('shrimp', 'dumpling')}",
                    fill=(200, 210, 220),
                    font=small,
                )
                writer.append_data(np.asarray(frame))
        finally:
            writer.close()
            renderer.close()
            inset.close()
            c.env.close()
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speed", type=int, default=4)
    parser.add_argument("--label", default="Kitchen pouring, jug table beside the stove")
    parser.add_argument("--still", action="store_true")
    parser.add_argument("--azimuth", type=float, default=-35.0)
    parser.add_argument("--frames", type=int, default=0, help="Render only the first N ticks")
    args = parser.parse_args()
    if args.still:
        KitchenLayoutRender.still(source=args.source, output=args.output, label=args.label)
    else:
        KitchenLayoutRender.video(
            source=args.source,
            output=args.output,
            speed=args.speed,
            label=args.label,
            azimuth=args.azimuth,
            frames=args.frames,
        )

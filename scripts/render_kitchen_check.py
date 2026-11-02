"""Render saved native poses; 4x transport, 1x pour, explicit simulated-hob label."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hitl_pmp.environments.pouring3d.kitchen.config import KitchenGeometry
from hitl_pmp.environments.pouring3d.kitchen.native import KitchenController


def render(*, source: Path, output: Path) -> None:
    report = json.loads((source / "result.json").read_text())
    motion = np.load(source / "motion.npz")
    c = KitchenController.build(
        geometry=KitchenGeometry.model_validate(report["geometry"]),
        deployment=report.get("deployment", report["case"] == "deployment"),
        requested=report["requested"],
        seed=0,
        scene_bg=True,
        surface_mu=0.1,
    )
    c.model.vis.global_.offwidth, c.model.vis.global_.offheight = 960, 720
    renderer = mujoco.Renderer(c.model, height=720, width=960)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    writer = imageio.get_writer(output, fps=10, codec="libx264", quality=8, macro_block_size=1)
    try:
        for i, qpos in enumerate(motion["qpos"]):
            phase = str(motion["phases"][i])
            pouring = any(
                word in phase.lower()
                for word in (
                    "tilt",
                    "wiggle",
                    "return upright",
                    "pour hold",
                    "joint lateral",
                    "joint-feedback",
                )
            )
            if phase == "cold practice reset complete":
                phase = f"After HUMAN bulk pour-back (cost {report['interruption_cost_total']:g})"
            contact = any(
                word in phase
                for word in (
                    "pinch shrimp",
                    "retain shrimp",
                    "secure shrimp",
                    "lift shrimp",
                    "release shrimp",
                    "withdraw empty gripper",
                )
            )
            speed = 1 if pouring or contact else 4
            if i % speed and i != len(motion["qpos"]) - 1:
                continue
            c.data.qpos[:] = qpos
            c.data.qvel[:] = 0
            c.data.time = float(motion["time"][i])
            on = bool(motion["hob"][i])
            c.scene.set_cooking(enabled=on)
            mujoco.mj_forward(c.model, c.data)
            renderer.update_scene(c.data, camera="task_view")
            frame = Image.fromarray(renderer.render())
            recovery = report["case"] == "recovery"
            if pouring or "upright" in phase or "potstickers" in phase or recovery:
                camera = mujoco.MjvCamera()
                camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                focus = "potsticker_0" if recovery else "tray"
                camera.lookat[:] = c.data.xpos[c.model.body(focus).id] + [
                    0,
                    0,
                    0.06 if recovery else 0.13,
                ]
                camera.distance, camera.azimuth, camera.elevation = (
                    0.65 if recovery else 0.8,
                    135,
                    -25,
                )
                renderer.update_scene(c.data, camera=camera)
                closeup = Image.fromarray(renderer.render()).resize((400, 300))
                frame.paste(closeup, (550, 410))
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 960, 92), fill=(20, 25, 30))
            title = f"Kitchen | {report['case']} | target {report['requested']} | {speed}x"
            draw.text((14, 7), title, fill="white", font=font)
            draw.text((14, 35), phase.replace("shrimp", "dumpling"), fill="white", font=font)
            water_label = "water is visual only" if c.scene.water_present() else "dry practice pan"
            draw.text(
                (14, 63),
                f"SIMULATED hob: {'ON' if on else 'OFF'} | {water_label}",
                fill=(255, 200, 90),
                font=font,
            )
            writer.append_data(np.asarray(frame))
            if i == 0:
                frame.save(source / "initial.png")
            if i == len(motion["qpos"]) - 1:
                frame.save(source / "final.png")
                for _ in range(20):
                    writer.append_data(np.asarray(frame))
    finally:
        writer.close()
        renderer.close()
        c.env.close()
    print(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    render(**vars(p.parse_args()))

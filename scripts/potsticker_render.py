"""Render recorded native potsticker trajectories; no simulation edits."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from hitl_pmp.environments.pouring3d.potsticker_native import make_potsticker_env


def render(*, folder: Path) -> None:
    result = json.loads((folder / "result.json").read_text())
    env = make_potsticker_env(
        count=result["count"], requested=result.get("requested", 1), seed=result.get("seed", 0)
    )
    model, _ = env.unwrapped._object_centric_env._model_and_data()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.distance = 0.75
    camera.azimuth = 135
    camera.elevation = -30
    camera.lookat[:] = [0.75, 1.05, 1.00]
    motion = np.load(folder / "motion.npz")
    rows = result.get("rows", [])
    recovery_piece = (
        int(rows[0]["action"][1])
        if len(rows) == 2 and rows[0].get("skill", "").startswith("PickShrimp")
        else None
    )
    policy = result.get("mode", "preparation")
    if result.get("anticipation_seconds", 0):
        policy += f" + {result['anticipation_seconds']}s lookahead"
    if result.get("return_speed_deg_s", 10) != 10:
        policy += f" + {result['return_speed_deg_s']}deg/s return"
    try:
        with imageio.get_writer(
            folder / "native.mp4",
            fps=10,
            codec="libx264",
            quality=8,
            ffmpeg_params=["-threads", "2"],
        ) as writer:
            for i, (q, phase) in enumerate(zip(motion["qpos"], motion["phases"], strict=True)):
                pouring = any(
                    x in str(phase)
                    for x in ("tilt", "wiggle", "return upright", "settle potstickers")
                )
                if not pouring and i % 4 and i != len(motion["qpos"]) - 1:
                    continue
                data.qpos[:] = q
                # Recorded poses need geometry/camera transforms, not another
                # contact solve. This leaves the saved native trajectory intact.
                mujoco.mj_kinematics(model, data)
                mujoco.mj_camlight(model, data)
                renderer.update_scene(data, camera="task_view")
                wide = renderer.render().copy()
                if recovery_piece is not None:
                    camera.lookat[:] = data.xpos[model.body(f"potsticker_{recovery_piece}").id] + [
                        0,
                        0,
                        0.04,
                    ]
                    camera.distance = 0.60
                renderer.update_scene(data, camera=camera)
                image = Image.fromarray(np.concatenate([wide, renderer.render().copy()], axis=1))
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 1280, 32), fill="black")
                text = (
                    f"{result['count']} potstickers | target {result.get('requested', '-')} | "
                    f"{policy} | {phase} | {'1x' if pouring else '4x'}"
                )
                draw.text((10, 10), text, fill="white")
                writer.append_data(np.asarray(image))
            image.save(folder / "final.png")
            for _ in range(15):
                writer.append_data(np.asarray(image))
    finally:
        renderer.close()
        env.close()
    print(folder / "native.mp4", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    render(folder=parser.parse_args().folder)

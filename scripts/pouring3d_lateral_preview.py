"""Render only the horizontal wiggle after native pickup and fixed-tilt setup."""

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any, cast

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from hitl_pmp.environments.pouring3d.lateral_wiggle import (
    LateralPouringController,
    LateralPourParameters,
)
from hitl_pmp.environments.pouring3d.simulation import PouringSimulation
from scripts.pouring3d_target_demo import RequestedCountDemo
from scripts.run_sweep import SweepRun, SweepRunner


class LateralPreview:
    @staticmethod
    def worker(*, root: Path) -> None:
        import mujoco
        from kinder.envs.dynamic3d.pitcher_pour3d import PitcherPour3DEnv

        root.mkdir(parents=True, exist_ok=True)
        native = Path(inspect.getfile(PitcherPour3DEnv))
        task = native.parent / "tasks/PitcherPour3D/PitcherPour3D-o5-winco.json"
        sim = PouringSimulation.create(
            seed=0,
            cube_material="foam",
            requested_count=2,
            scene_bg=True,
            task_config_path=str(task),
        )
        c = LateralPouringController.create(env=sim.env)
        assert isinstance(c, LateralPouringController)
        sim.controller = c
        # Size only the offscreen framebuffer; physics and assets stay native.
        c.model.vis.global_.offwidth = 960
        c.model.vis.global_.offheight = 720
        renderer = mujoco.Renderer(c.model, width=960, height=720)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = 0.50
        camera.azimuth = 90
        camera.elevation = -20
        frames = 0
        parameters = LateralPourParameters()
        try:
            with cast(
                Any,
                imageio.get_writer(
                    root / "sideways-wiggle-only.mp4",
                    fps=10,
                    codec="libx264",
                    quality=8,
                    ffmpeg_params=["-threads", "2", "-movflags", "+faststart"],
                ),
            ) as writer:

                def record(**kwargs: Any) -> None:
                    nonlocal frames
                    controller = kwargs["controller"]
                    if controller.phase != "horizontal sideways wiggle":
                        return
                    if frames == 0:
                        position, _ = controller.body_pose(name="pitcher")
                        camera.lookat[:] = position
                    renderer.update_scene(controller.data, camera=camera)
                    canvas = Image.new("RGB", (960, 784), (20, 25, 33))
                    canvas.paste(Image.fromarray(renderer.render()), (0, 64))
                    draw = ImageDraw.Draw(canvas)
                    draw.text(
                        (14, 5),
                        "Sideways wiggle | fixed tilt 103 deg | 1x speed",
                        font=RequestedCountDemo.font(size=25),
                        fill="white",
                    )
                    draw.text(
                        (14, 36),
                        "Horizontal, across the pour direction | +/-15 mm at 1 Hz",
                        font=RequestedCountDemo.font(size=19),
                        fill="#c5d8ed",
                    )
                    writer.append_data(np.asarray(canvas))
                    frames += 1

                picked = sim.execute(skill="PickPitcher")
                assert picked["success"]
                c.callback = record
                event = c.run_lateral_pour(parameters=parameters)
                c.callback = None
            assert event["error"] is None and c.held == "pitcher"
            trace = c.lateral_trace
            assert len(trace) == frames == 120
            commands = np.array([r["commanded_position"] for r in trace])
            assert np.ptp(commands[:, 1]) == np.ptp(commands[:, 2]) == 0
            actual = np.array([r["actual_hand_position"] for r in trace])
            assert np.ptp(actual[:, 0]) > 0.008
            changes = np.diff(actual[:, 0])
            assert (changes > 0.001).sum() > 10 and (changes < -0.001).sum() > 10
            sources = [
                native,
                task,
                Path(inspect.getfile(LateralPouringController)),
                Path(__file__).resolve(),
            ]
            result = {
                "parameters": parameters.model_dump(),
                "event": event,
                "frames": frames,
                "fps": 10,
                "actual_hand_range_m": np.ptp(actual, axis=0).tolist(),
                "source_sha256": {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
                },
            }
            (root / "result.json").write_text(json.dumps(result, indent=2))
            print(
                json.dumps({"frames": frames, "actual_hand_range_m": result["actual_hand_range_m"]})
            )
        finally:
            c.callback = None
            renderer.close()
            sim.close()

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--root", type=Path, required=True)
        parser.add_argument("--worker", action="store_true")
        args = parser.parse_args()
        if args.worker:
            LateralPreview.worker(root=args.root)
        else:
            repo = Path(__file__).resolve().parents[1]
            results = SweepRunner.execute(
                runs=[
                    SweepRun(
                        method="sideways-preview",
                        seed=0,
                        output_dir=args.root,
                        command=[
                            "run-heavy",
                            str(repo / "scripts/with_env.sh"),
                            sys.executable,
                            "-m",
                            "scripts.pouring3d_lateral_preview",
                            "--root",
                            str(args.root),
                            "--worker",
                        ],
                    )
                ],
                max_workers=1,
            )
            if not all(r.succeeded for r in results):
                raise SystemExit(1)


if __name__ == "__main__":
    LateralPreview.main()

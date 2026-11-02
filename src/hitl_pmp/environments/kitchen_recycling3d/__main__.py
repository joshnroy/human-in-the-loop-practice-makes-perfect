"""Local, captioned motor-controller preview; never submits cluster experiments."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .domain import KitchenConfig
from .simulator import RecyclingSimulator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--object", action="append", dest="objects")
    parser.add_argument("--no-background", action="store_true")
    arguments = parser.parse_args()
    sources = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in Path(__file__).parent.glob("*.py")
    }
    config = (
        KitchenConfig.model_validate_json(arguments.config.read_text())
        if arguments.config
        else KitchenConfig()
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=False)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
    small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    writer = imageio.get_writer(
        arguments.output_dir / "preview.mp4", fps=10, codec="libx264", macro_block_size=1
    )
    captured = 0

    def record(*, frame: Any, caption: Any, episode: Any) -> None:
        nonlocal captured
        canvas = Image.new("RGB", (frame.shape[1], frame.shape[0] + 130), "#111827")
        canvas.paste(Image.fromarray(frame), (0, 60))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (16, 8),
            "Kitchen recycling | MuJoCo preview, not hardware",
            font=font,
            fill="white",
        )
        draw.text(
            (16, 35),
            "Sprout mats: no-drive zone, not a physical barrier",
            font=small_font,
            fill="#fbbf24",
        )
        bottom = frame.shape[0] + 65
        draw.text((16, bottom), caption[:72], font=small_font, fill="white")
        solved = len(set(episode.config.required_objects) & episode.settled)
        required = len(episode.config.required_objects)
        draw.text(
            (16, bottom + 25),
            f"{episode.phase.upper()} | settled {solved}/{required}"
            f" recycling objects | human cost {episode.human_cost:g}",
            font=small_font,
            fill="#a7f3d0",
        )
        if captured == 0:
            canvas.save(arguments.output_dir / "scene.png")
        writer.append_data(np.asarray(canvas))
        captured += 1

    simulator = None
    results = []
    try:
        simulator = RecyclingSimulator(
            config=config,
            seed=arguments.seed,
            frame_callback=record,
            pretty=not arguments.no_background,
        )
        simulator.caption = "WEEKEND TRAINING: human home but busy; each interruption is costed"
        simulator.idle(seconds=1.0)
        simulator.human(action="relocate_bin", destination=(0.65, 1.4, 0))
        simulator.human(action="restore_bin")
        simulator.begin_deployment()
        simulator.idle(seconds=1.0)
        for name in arguments.objects or [spec.name for spec in config.objects]:
            result = simulator.pick(name=name)
            results.append({"object": name, "skill": "pick", **result})
            print(json.dumps(results[-1]), flush=True)
            if result["success"]:
                result = simulator.toss(name=name)
                results.append({"object": name, "skill": "toss", **result})
                print(json.dumps(results[-1]), flush=True)
            simulator.caption = (
                f"{name}: {'success' if result['success'] else 'failure'} | miss stays observed"
            )
            simulator.idle(seconds=1.2, closed=bool(simulator.episode.holding))
            if simulator.episode.holding:
                break
        simulator.caption = "DEPLOYMENT: human recovery rejected; no automatic scene reset"
        rejected = []
        for action in ("recover_object", "relocate_bin", "restore_bin"):
            try:
                simulator.human(
                    action=action,
                    name=config.objects[0].name,
                    destination=(0.5, 0, config.objects[0].dimensions[2] / 2 + 0.002),
                )
            except PermissionError:
                rejected.append(action)
        simulator.idle(seconds=1.5, closed=bool(simulator.episode.holding))
        report = {
            "config": config.model_dump(),
            "source_sha256": sources,
            "seed": arguments.seed,
            "results": results,
            "episode": simulator.episode.model_dump(mode="json"),
            "control_steps": simulator.steps,
            "rejected_deployment_actions": rejected,
            "hardware_blockers": config.hardware_blockers(),
            "rendering": "Actual MuJoCo controller rollout; no scripted successful trajectories",
            "calibration": "Object/context-specific proposals, NOT validated hardware parameters",
        }
        (arguments.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        writer.close()
        if simulator is not None:
            simulator.close()


if __name__ == "__main__":
    main()

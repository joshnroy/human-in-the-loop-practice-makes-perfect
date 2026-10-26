"""Render individual pouring skills and persistent robot/human reset cycles."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hitl_pmp.environments.pouring3d.simulation import PouringSimulation


class PouringDemo:
    @staticmethod
    def annotate(*, sim: PouringSimulation, title: str) -> np.ndarray:
        frame = sim.render()
        canvas = Image.new("RGB", (960, 752), "#15252f")
        canvas.paste(Image.fromarray(frame), (0, 64))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        draw.text((20, 8), title, fill="white", font=font)
        locations = sim.locations()
        counts = " | ".join(
            f"{name}: {locations.count(name)}/5" for name in ["pitcher", "tray", "floor"]
        )
        draw.text(
            (20, 37), counts + f" | holding: {sim.held or 'nothing'}", fill="#cee2e8", font=small
        )
        draw.text(
            (20, 714),
            "MuJoCo cube contacts | scripted robot IK + idealized grasps | 1x simulated time",
            fill="#cee2e8",
            font=small,
        )
        return np.asarray(canvas)

    @staticmethod
    def record(
        *,
        sim: PouringSimulation,
        directory: Path,
        filename: str,
        skill: str,
        cube: int | None = None,
        parameters: dict[str, float] | None = None,
        title: str | None = None,
    ) -> dict[str, Any]:
        path = directory / filename
        label = title or skill
        writer = imageio.get_writer(path, fps=25, codec="libx264", quality=7, macro_block_size=16)
        count = 0

        def append(**kwargs: Any) -> None:
            nonlocal count
            writer.append_data(PouringDemo.annotate(sim=kwargs["sim"], title=label))
            count += 1

        sim.frame_stride = 20
        try:
            for _ in range(12):
                append(sim=sim)
            sim.frame_callback = append
            event = sim.execute(skill=skill, cube=cube, parameters=parameters)
            sim.frame_callback = None
            for _ in range(20):
                append(sim=sim)
        finally:
            sim.frame_callback = None
            writer.close()
        result = {
            **event,
            "video": filename,
            "frames": count,
            "fps": 25,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        print(f"{filename}: success={event['success']}, {event['after']['locations']}", flush=True)
        return result

    @staticmethod
    def concatenate(*, directory: Path, files: list[str], output: str) -> None:
        with imageio.get_writer(directory / output, fps=25, codec="libx264", quality=7) as writer:
            for filename in files:
                with imageio.get_reader(directory / filename) as reader:
                    for frame in reader:
                        writer.append_data(frame)

    @staticmethod
    def run(*, directory: Path, seed: int) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        sim = PouringSimulation.create(seed=seed)
        events = []
        try:
            imageio.imwrite(
                directory / "setup.png",
                PouringDemo.annotate(sim=sim, title="Pouring3D | five-cube floor setup"),
            )
            imageio.imwrite(directory / "setup-overhead.png", sim.render(overhead=True))
            (directory / "scene.xml").write_text(sim.source_xml)
            for skill, filename in [
                ("PickPitcher", "01-pick-pitcher.mp4"),
                ("Pour", "02-pour.mp4"),
                ("PlacePitcherAtHome", "03-place-pitcher.mp4"),
            ]:
                events.append(
                    PouringDemo.record(sim=sim, directory=directory, filename=filename, skill=skill)
                )
            human = PouringDemo.record(
                sim=sim,
                directory=directory,
                filename="04-human-pour-back.mp4",
                skill="HumanPourBack",
                title="Human reset | lift tray, pour back, replace tray",
            )
            events.append(human)
            if not sim.ready():
                raise RuntimeError(f"Human reset did not restore readiness: {sim.locations()}")
            sim.execute(skill="PickPitcher")
            sim.execute(skill="Pour")
            sim.execute(skill="PlacePitcherAtHome")
            robot_files = []
            for cube in range(5):
                for skill, prefix in [
                    ("PickCubeFromTray", "pick"),
                    ("DropCubeIntoPitcher", "drop"),
                ]:
                    name = f"robot-{prefix}-cube{cube + 1}.mp4"
                    events.append(
                        PouringDemo.record(
                            sim=sim,
                            directory=directory,
                            filename=name,
                            skill=skill,
                            cube=cube,
                            title=f"Robot reset | {prefix} cube {cube + 1} of 5",
                        )
                    )
                    robot_files.append(name)
            if not sim.ready():
                raise RuntimeError(f"Robot reset did not restore readiness: {sim.locations()}")
            PouringDemo.concatenate(
                directory=directory, files=robot_files, output="05-robot-reset-five-cubes.mp4"
            )
            sim.execute(skill="PickPitcher")
            failed = PouringDemo.record(
                sim=sim,
                directory=directory,
                filename="06-small-tilt-failure.mp4",
                skill="Pour",
                parameters={"angle": 10.0},
                title="Pour failure example | 10-degree tilt retains the cubes",
            )
            events.append(failed)
            sim.execute(skill="PlacePitcherAtHome")
            sim.execute(skill="PickPitcher")
            spill = PouringDemo.record(
                sim=sim,
                directory=directory,
                filename="07-missed-tray.mp4",
                skill="Pour",
                parameters={"offset_y": 0.35},
                title="Pour failure example | offset misses the tray",
            )
            events.append(spill)
            sim.execute(skill="PlacePitcherAtHome")
            spilled = [i for i, location in enumerate(sim.locations()) if location == "floor"]
            if spilled:
                cube = spilled[0]
                events.append(
                    PouringDemo.record(
                        sim=sim,
                        directory=directory,
                        filename="08-pick-cube-from-floor.mp4",
                        skill="PickCubeFromFloor",
                        cube=cube,
                        title="Recovery | pick a spilled cube from the floor",
                    )
                )
                events.append(
                    PouringDemo.record(
                        sim=sim,
                        directory=directory,
                        filename="09-drop-recovered-cube.mp4",
                        skill="DropCubeIntoPitcher",
                        cube=cube,
                        title="Recovery | drop the recovered cube into the pitcher",
                    )
                )
            PouringDemo.concatenate(
                directory=directory,
                files=[
                    "01-pick-pitcher.mp4",
                    "02-pour.mp4",
                    "03-place-pitcher.mp4",
                    "04-human-pour-back.mp4",
                    "robot-pick-cube1.mp4",
                    "robot-drop-cube1.mp4",
                ],
                output="skills-overview.mp4",
            )
            manifest = {
                "environment": "Pouring3D proof of concept",
                "seed": seed,
                "fidelity": (
                    "dynamic cube contacts; mocap vessels; kinematic robot; idealized grasps"
                ),
                "learned_policy": False,
                "planner_integrated": False,
                "events": events,
                "max_position_ik_error_m": sim.max_ik_error,
                "elapsed_simulated_seconds": sim.elapsed_seconds,
                "human_interventions": sim.human_interventions,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "simulation_sha256": hashlib.sha256(
                    Path(
                        __import__(PouringSimulation.__module__, fromlist=["__file__"]).__file__
                    ).read_bytes()
                ).hexdigest(),
                "scene_sha256": hashlib.sha256(sim.source_xml.encode()).hexdigest(),
            }
            (directory / "results.json").write_text(json.dumps(manifest, indent=2))
        finally:
            sim.close()

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--seed", type=int, default=0)
        args = parser.parse_args()
        PouringDemo.run(directory=args.output_dir, seed=args.seed)


if __name__ == "__main__":
    PouringDemo.main()

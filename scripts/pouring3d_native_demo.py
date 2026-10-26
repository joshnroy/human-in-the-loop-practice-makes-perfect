"""Record native KinDER motor execution through the environment's own renderer."""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any

import imageio.v2 as imageio

from hitl_pmp.environments.pouring3d.native_controller import (
    NativePouringController,
    PourParameters,
)


class NativePouringDemo:
    @staticmethod
    def run(
        *, directory: Path, seed: int, video: bool, scene_bg: bool, reset: str = "both"
    ) -> dict[str, Any]:
        from kinder.envs.dynamic3d.pitcher_pour3d import PitcherPour3DEnv

        directory.mkdir(parents=True, exist_ok=True)
        env = PitcherPour3DEnv(scene_bg=scene_bg, render_mode="rgb_array")
        writer = None
        skill_writer = None
        frames: list[dict[str, Any]] = []
        try:
            observation, _ = env.reset(seed=seed)
            controller = NativePouringController.create(env=env)
            before = controller.scene.physical_snapshot()
            (directory / "initial-state.json").write_text(json.dumps(before, indent=2))
            if video:
                writer = imageio.get_writer(
                    directory / "native-rollout.mp4",
                    fps=controller.frequency,
                    codec="libx264",
                    quality=8,
                )
                imageio.imwrite(directory / "native-setup.png", env.render())

            def record(**kwargs: Any) -> None:
                c = kwargs["controller"]
                entry = {
                    "frame": len(frames),
                    "phase": c.phase,
                    "hand": c.hand_pose()[0].tolist(),
                    "pitcher": c.body_pose(name="pitcher")[0].tolist(),
                    "tray": c.body_pose(name="tray")[0].tolist(),
                    "tray_quat": c.body_pose(name="tray")[1].as_quat().tolist(),
                    "cubes": [c.body_pose(name=n)[0].tolist() for n in c.scene.cube_names],
                    "pitcher_quat": c.body_pose(name="pitcher")[1].as_quat().tolist(),
                    "contacts": sorted(c.finger_contacts(name="pitcher")),
                    "time": float(c.data.time),
                    "locations": c.scene.cube_locations(),
                    "qpos_sha256": hashlib.sha256(c.data.qpos.tobytes()).hexdigest(),
                }
                if writer is not None:
                    frame = env.render()
                    writer.append_data(frame)
                    if skill_writer is not None:
                        skill_writer.append_data(frame)
                frames.append(entry)

            controller.callback = record
            record(controller=controller)
            prefix = [("PickPitcher", 0), ("Pour", 0), ("PlacePitcherAtHome", 0)]
            sequence = list(prefix)
            if reset in {"human", "both"}:
                sequence.append(("HumanPourBack", 0))
            if reset == "both":
                sequence.extend(prefix)
            if reset in {"robot", "both"}:
                sequence.extend(
                    (skill, -1)
                    for cube in range(5)
                    for skill in ["PickCubeFromTray", "DropCubeIntoPitcher"]
                )
            selected_cube = 0
            for index, (skill, cube) in enumerate(sequence):
                if skill == "PickCubeFromTray" and cube == -1:
                    selected_cube = controller.select_reset_cube()
                if cube == -1:
                    cube = selected_cube
                if video:
                    (directory / "skills").mkdir(exist_ok=True)
                    skill_writer = imageio.get_writer(
                        directory / "skills" / f"{index:02}-{skill}-{cube}.mp4",
                        fps=controller.frequency,
                        codec="libx264",
                        quality=8,
                    )
                    skill_writer.append_data(env.render())
                event = controller.run_skill(
                    skill=skill, cube=cube, parameters=PourParameters() if skill == "Pour" else None
                )
                if skill_writer is not None:
                    skill_writer.close()
                    skill_writer = None
                if not event["success"]:
                    break
            final = controller.scene.physical_snapshot()
            if writer is not None:
                imageio.imwrite(directory / "native-final.png", env.render())
                writer.close()
                writer = None
            manifest = {
                "environment": "kinder/PitcherPour3D-o5-v0",
                "seed": seed,
                "scene_bg": scene_bg,
                "native_module": inspect.getfile(PitcherPour3DEnv),
                "native_source_sha256": hashlib.sha256(
                    Path(inspect.getfile(PitcherPour3DEnv)).read_bytes()
                ).hexdigest(),
                "controller_source_sha256": hashlib.sha256(
                    Path(inspect.getfile(NativePouringController)).read_bytes()
                ).hexdigest(),
                "initial_observation_dim": len(observation),
                "initial": before,
                "final": final,
                "events": controller.events,
                "control_steps": controller.steps,
                "frames": len(frames),
                "fps": controller.frequency,
                "video": video,
                "goal": final["goal"],
                "reset": reset,
                "completed": bool(
                    (final["goal"] if reset == "none" else controller.ready())
                    and len(controller.events) == len(sequence)
                    and all(e["success"] for e in controller.events)
                ),
                "render_path": "env.render() after every native env.step()",
                "physics": (
                    "Native TidyBot PD motor control; free vessels and cubes; "
                    "contacts enabled; no object attachments"
                ),
                "mujoco_gl": os.environ.get("MUJOCO_GL"),
            }
            if video:
                manifest["video_sha256"] = hashlib.sha256(
                    (directory / "native-rollout.mp4").read_bytes()
                ).hexdigest()
            (directory / "native-results.json").write_text(json.dumps(manifest, indent=2))
            (directory / "native-frames.json").write_text(json.dumps(frames, indent=2))
            print(
                json.dumps({
                    k: manifest[k] for k in ["completed", "goal", "control_steps", "frames"]
                }),
                flush=True,
            )
            return manifest
        finally:
            if skill_writer is not None:
                skill_writer.close()
            if writer is not None:
                writer.close()
            env.close()

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--reset", choices=["none", "human", "robot", "both"], default="both")
        parser.add_argument("--no-video", action="store_true")
        parser.add_argument("--simple-scene", action="store_true")
        args = parser.parse_args()
        NativePouringDemo.run(
            directory=args.output_dir,
            seed=args.seed,
            video=not args.no_video,
            reset=args.reset,
            scene_bg=not args.simple_scene,
        )


if __name__ == "__main__":
    NativePouringDemo.main()

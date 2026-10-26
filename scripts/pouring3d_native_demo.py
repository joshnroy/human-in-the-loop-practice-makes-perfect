"""Record native KinDER motor execution through the environment's own renderer."""

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any

import imageio.v2 as imageio

from hitl_pmp.environments.pouring3d.environment import PouringEnvironment
from hitl_pmp.environments.pouring3d.native_controller import (
    NativePouringController,
    PourParameters,
)
from hitl_pmp.humans.oracle import UnconditionalHumanOracle


class NativePouringDemo:
    @staticmethod
    def run(
        *,
        directory: Path,
        seed: int,
        video: bool,
        scene_bg: bool,
        reset: str = "both",
        human_reset_practice_cost: float = UnconditionalHumanOracle.intervention_cost,
    ) -> dict[str, Any]:
        from kinder.envs.dynamic3d.pitcher_pour3d import PitcherPour3DEnv

        if not math.isfinite(human_reset_practice_cost) or human_reset_practice_cost < 0:
            raise ValueError("Human practice cost must be finite and nonnegative")
        directory.mkdir(parents=True, exist_ok=True)
        adapter = PouringEnvironment(seed=seed, scene_bg=scene_bg)
        adapter.hard_reset()
        env, controller = adapter.sim.env, adapter.sim.controller
        writer = None
        skill_writer = None
        frames: list[dict[str, Any]] = []
        try:
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
                    "handle_contacts": sorted(c.finger_contacts(name="pitcher", handle_only=True)),
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
                sequence.append(("reset_cubes", 0))
            if reset == "both":
                sequence.extend(prefix)
            if reset in {"robot", "both"}:
                sequence.extend(
                    (skill, -1)
                    for cube in range(5)
                    for skill in ["PickCubeFromTray", "DropCubeIntoPitcher"]
                )
            selected_cube = 0
            events = []
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
                if skill == "reset_cubes":
                    UnconditionalHumanOracle.execute_movables_reset(env=adapter, destination=skill)
                    event = dict(adapter.last_info, practice_cost=human_reset_practice_cost)
                    controller.phase = "human reset_cubes: instantaneous external intervention"
                    # Recording pause only: the oracle hook does not step physics.
                    for _ in range(round(controller.frequency)):
                        record(controller=controller)
                else:
                    event = adapter.sim.execute(
                        skill=skill,
                        cube=cube,
                        parameters=PourParameters().model_dump() if skill == "Pour" else None,
                    )
                events.append(event)
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
                "task_config_sha256": hashlib.sha256(
                    (
                        Path(inspect.getfile(PitcherPour3DEnv)).parent
                        / "tasks/PitcherPour3D/PitcherPour3D-o5.json"
                    ).read_bytes()
                ).hexdigest(),
                "controller_source_sha256": hashlib.sha256(
                    Path(inspect.getfile(NativePouringController)).read_bytes()
                ).hexdigest(),
                "initial_observation_dim": int(env.observation_space.shape[0]),
                "initial": before,
                "final": final,
                "events": events,
                "control_steps": controller.steps,
                "frames": len(frames),
                "fps": controller.frequency,
                "video": video,
                "goal": final["goal"],
                "reset": reset,
                "completed": bool(
                    (final["goal"] if reset == "none" else controller.ready())
                    and len(events) == len(sequence)
                    and all(e["success"] for e in events)
                ),
                "render_path": (
                    "env.render() after native steps; 1 s display hold after human reset"
                ),
                "human_reset_interface": (
                    "UnconditionalHumanOracle.execute_movables_reset -> Environment.reset_movables"
                ),
                "human_reset_practice_cost": human_reset_practice_cost,
                "human_reset_display_hold_seconds": 1.0 if reset in {"human", "both"} else 0.0,
                "physics": (
                    "Native TidyBot PD motor control; free vessels and cubes; "
                    "contacts enabled; no object attachments. reset_cubes is "
                    "an explicitly simulated human intervention that respawns "
                    "cube poses and clears cube velocities; vessels stay in place."
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
        parser.add_argument(
            "--human-reset-practice-cost",
            type=float,
            default=UnconditionalHumanOracle.intervention_cost,
        )
        args = parser.parse_args()
        NativePouringDemo.run(
            directory=args.output_dir,
            seed=args.seed,
            video=not args.no_video,
            reset=args.reset,
            scene_bg=not args.simple_scene,
            human_reset_practice_cost=args.human_reset_practice_cost,
        )


if __name__ == "__main__":
    NativePouringDemo.main()

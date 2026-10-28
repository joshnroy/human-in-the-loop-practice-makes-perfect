"""Run the registered practice skills and render their recorded native motion."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from scripts.run_sweep import SweepRun, SweepRunner


class PracticeDemo:
    @staticmethod
    def worker(*, root: Path, seed: int, render: bool) -> None:
        from hitl_pmp.environments.pouring3d.practice.environment import FrozenPourEnvironment
        from hitl_pmp.environments.pouring3d.practice.skill_provider import FrozenPourSkillProvider

        root.mkdir(parents=True, exist_ok=True)
        if (root / "result.json").exists():
            raise FileExistsError("Do not overwrite a completed trial")
        env = FrozenPourEnvironment(canonical_seed=seed, scene_bg=True)
        provider = FrozenPourSkillProvider(env=env)
        frames: list[np.ndarray] = []
        phases: list[str] = []
        rows = []
        active = "initial"

        def capture(**_: Any) -> None:
            frames.append(env.controller().data.qpos.copy())
            phases.append(active)

        def run(*, name: str, shrimp: int = 0, params: tuple[float, ...] = ()) -> bool:
            nonlocal active
            active = name
            state = env.get_current_state()
            skill = provider.ground(name=name, shrimp=shrimp)
            env.take_action(
                action=provider.compute_action(
                    ground_skill=skill, params=np.array(params), state=state
                )
            )
            row = {k: v for k, v in env.last_info.items() if k not in {"before", "after"}}
            rows.append(row)
            print(json.dumps(row), flush=True)
            return bool(row["success"])

        try:
            env.hard_reset()
            env.set_motion_sink(sink=capture)
            capture()
            if run(name="PickPitcher"):
                run(name="Pour", params=(90.5, 0.008, 30))
                run(name="PlacePitcherAtHome")
                c = env.controller()
                for _ in range(2 * env.payload_count):
                    candidates = [
                        (i, loc)
                        for i, loc in enumerate(c.scene.cube_locations())
                        if loc in {"tray", "floor"}
                    ]
                    if not candidates:
                        break
                    index, location = min(
                        candidates, key=lambda pair: (pair[1] != "floor", pair[0])
                    )
                    if c.held is None:
                        if not run(
                            name="PickShrimpFromTray"
                            if location == "tray"
                            else "PickShrimpFromFloor",
                            shrimp=index,
                        ):
                            continue
                        if not run(name="DropShrimpIntoPitcher", shrimp=index):
                            break
                # Human reset is a separate intervention, invoked only if still needed.
                if "tray" in c.scene.cube_locations() or "floor" in c.scene.cube_locations():
                    before = c.data.qpos.copy()
                    success = env.reset_movables(destination="reset_shrimp")
                    rows.append({**env.last_info, "success": success})
                    active = "HumanResetToPitcher"
                    if success:
                        capture()
                    np.save(root / "before-human.npy", before)
            c = env.controller()
            result = {
                "seed": seed,
                "steps": c.steps,
                "rows": rows,
                "ready": c.ready(),
                "locations": c.scene.cube_locations(),
            }
            (root / "result.json").write_text(json.dumps(result, indent=2))
            np.savez_compressed(root / "motion.npz", qpos=np.array(frames), skills=np.array(phases))
            if render:
                PracticeDemo.render(root=root, controller=c, frames=frames, phases=phases)
        finally:
            env.close()

    @staticmethod
    def render(*, root: Path, controller: Any, frames: list[np.ndarray], phases: list[str]) -> None:
        import imageio.v2 as imageio
        import mujoco
        from PIL import Image, ImageDraw

        c = controller
        drawing = mujoco.MjData(c.model)
        renderer = mujoco.Renderer(c.model, height=480, width=640)
        try:
            with imageio.get_writer(
                root / "skills.mp4", fps=10, codec="libx264", quality=8
            ) as writer:
                for index, (qpos, phase) in enumerate(zip(frames, phases, strict=True)):
                    if phase != "Pour" and index % 4 and index != len(frames) - 1:
                        continue
                    drawing.qpos[:] = qpos
                    mujoco.mj_forward(c.model, drawing)
                    renderer.update_scene(drawing, camera="task_view")
                    frame = Image.fromarray(renderer.render().copy())
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, 640, 38), fill="black")
                    draw.text(
                        (12, 12),
                        f"{phase} | {'1x' if phase == 'Pour' else '4x'} | actual native motion",
                        fill="white",
                    )
                    writer.append_data(np.asarray(frame))
                    if index == len(frames) - 1:
                        frame.save(root / "final.png")
        finally:
            renderer.close()

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--seed", type=int, default=70)
        parser.add_argument("--worker", action="store_true")
        parser.add_argument("--no-render", action="store_true")
        args = parser.parse_args()
        if args.worker:
            PracticeDemo.worker(root=args.output_dir, seed=args.seed, render=not args.no_render)
            return
        command = [
            sys.executable,
            "-m",
            "scripts.pouring3d_practice_demo",
            "--worker",
            "--output-dir",
            str(args.output_dir),
            "--seed",
            str(args.seed),
        ]
        if args.no_render:
            command.append("--no-render")
        outcomes = SweepRunner.execute(
            runs=[
                SweepRun(
                    method="practice-skills",
                    seed=args.seed,
                    output_dir=args.output_dir,
                    command=command,
                )
            ],
            max_workers=1,
        )
        if not all(outcome.succeeded for outcome in outcomes):
            raise SystemExit(1)


if __name__ == "__main__":
    PracticeDemo.main()

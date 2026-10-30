"""Fixed-recipe skillet checks with native motion, reset audits, and honest outcomes."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from scripts.pouring3d_practice_demo import PracticeDemo
from scripts.run_sweep import SweepRun, SweepRunner


class SkilletCheck:
    @staticmethod
    def trial(
        *,
        root: Path,
        seed: int,
        target: int,
        weak: bool,
        layout: str = "stations",
        reset_mode: str = "robot",
    ) -> None:
        from hitl_pmp.environments.pouring3d.practice.environment import FrozenPourEnvironment
        from hitl_pmp.environments.pouring3d.practice.skill_provider import FrozenPourSkillProvider
        from hitl_pmp.environments.pouring3d.practice.tasks import FrozenPourTasks

        root.mkdir(parents=True, exist_ok=True)
        if (root / "result.json").exists():
            raise FileExistsError(root)
        env = FrozenPourEnvironment(
            canonical_seed=seed, requested_count=target, scene_bg=True, layout=layout
        )
        provider = FrozenPourSkillProvider(env=env)
        frames, phases, rows, times, motion_phases = [], [], [], [], []
        active = "Initial"

        def capture(**_: Any) -> None:
            frames.append(env.controller().data.qpos.copy())
            phases.append(active)
            motion_phases.append(env.controller().phase)
            times.append(float(env.controller().data.time))

        def run(*, name: str, shrimp: int = 0, params: tuple = ()) -> bool:
            nonlocal active
            active = name
            skill = provider.ground(name=name, shrimp=shrimp)
            env.take_action(
                action=provider.compute_action(
                    ground_skill=skill, params=np.array(params), state=env.get_current_state()
                )
            )
            rows.append(dict(env.last_info))
            return bool(env.last_info["success"])

        try:
            env.hard_reset()
            c = env.controller()
            # Audit each physics substep, not just the rendered 10 Hz frames.
            robot = np.array([
                (c.model.body(int(b)).name or "").startswith("robot") for b in c.model.geom_bodyid
            ])
            support = np.array([
                c.model.body(int(b)).name in {"prep_table", "stovetop"} for b in c.model.geom_bodyid
            ])
            pan = np.array([c.model.body(int(b)).name == "tray" for b in c.model.geom_bodyid])
            pitcher_wall = np.array([
                c.model.body(int(b)).name == "pitcher"
                and not (c.model.geom(i).name or "").startswith("pitcher_handle_")
                for i, b in enumerate(c.model.geom_bodyid)
            ])
            audit: dict[str, Any] = {
                "physics_steps": 0,
                "robot_support_contact_steps": 0,
                "max_robot_support_penetration_mm": 0.0,
                "robot_pan_contact_steps": 0,
                "max_robot_pan_penetration_mm": 0.0,
                "robot_pitcher_wall_contact_steps": 0,
                "max_robot_pitcher_wall_penetration_mm": 0.0,
                "worst_phase": None,
            }
            sim = c.scene._robot_env.sim
            original_step = sim.step

            def audited_step() -> None:
                original_step()
                audit["physics_steps"] += 1
                pairs = c.data.contact.geom
                mask = (robot[pairs[:, 0]] & support[pairs[:, 1]]) | (
                    support[pairs[:, 0]] & robot[pairs[:, 1]]
                )
                for category, obstacle in (("pan", pan), ("pitcher_wall", pitcher_wall)):
                    selected = (robot[pairs[:, 0]] & obstacle[pairs[:, 1]]) | (
                        obstacle[pairs[:, 0]] & robot[pairs[:, 1]]
                    )
                    depths = c.data.contact.dist[selected]
                    if len(depths) and depths.min() < 0:
                        audit[f"robot_{category}_contact_steps"] += 1
                        audit[f"max_robot_{category}_penetration_mm"] = max(
                            audit[f"max_robot_{category}_penetration_mm"],
                            float(-depths.min() * 1000),
                        )
                distances = c.data.contact.dist[mask]
                if len(distances) and distances.min() < 0:
                    audit["robot_support_contact_steps"] += 1
                    penetration = float(-distances.min() * 1000)
                    if penetration > audit["max_robot_support_penetration_mm"]:
                        audit["max_robot_support_penetration_mm"] = penetration
                        audit["worst_phase"] = c.phase

            sim.step = audited_step
            env.set_motion_sink(sink=capture)
            capture()
            initial_ready = c.ready()
            initial_pan = c.data.qpos[
                c.model.jnt_qposadr[c.model.body_jntadr[c.model.body("tray").id]] :
            ][:7].copy()
            params = (
                (60.0, 0.0, 0.0)
                if weak
                else ((92.0, 0.015, 30.0) if target == 3 else (90.5, 0.008, 30.0))
            )
            picked = run(name="PickPitcher")
            poured = run(name=f"Pour{target}", params=params) if picked else False
            delivered = c.scene.cube_locations().count("tray")
            after_pour = c.scene.physical_snapshot()
            placed = run(name="PlacePitcherAtHome") if c.held == "pitcher" else False
            full = (
                FrozenPourTasks(env=env)
                .goal(requested=target)
                .is_satisfied(state=env.get_current_state())
            )
            pour_end = len(frames)
            # Recovery is checked after scoring; it cannot improve this trial's score.
            recovery = []
            for index, location in enumerate(c.scene.cube_locations()):
                if (
                    reset_mode == "human"
                    or location not in {"tray", "floor", "surface"}
                    or c.held is not None
                ):
                    continue
                grasp = run(
                    name={
                        "tray": "PickShrimpFromTray",
                        "floor": "PickShrimpFromFloor",
                        "surface": "PickShrimpFromSurface",
                    }[location],
                    shrimp=index,
                )
                drop = run(name="DropShrimpIntoPitcher", shrimp=index) if grasp else False
                recovery.append({"shrimp": index, "source": location, "grasp": grasp, "drop": drop})
            robot_ready = c.ready()
            # Audit payload-only reset even when physical recovery completed.
            # Moving one piece is an explicitly recorded test fixture, not robot motion.
            reset_fixture_used = c.held is None and robot_ready
            if reset_fixture_used:
                import mujoco

                pan, rotation = c.body_pose(name="tray")
                obj = c.scene.get_object(c.scene.cube_names[0])
                obj.set_pose(pan + rotation.apply([0, 0, 0.010]), [1, 0, 0, 0])
                obj.set_velocity([0, 0, 0], [0, 0, 0])
                mujoco.mj_forward(c.model, c.data)
            before = c.data.qpos.copy()
            velocities = c.data.qvel.copy()
            time, steps = float(c.data.time), c.steps
            reset = env.reset_movables(destination="reset_shrimp")
            if reset and not reset_fixture_used:
                active = "HumanResetToPitcher"
                capture()
            changed_q, changed_v = set(), set()
            for name in c.scene.cube_names:
                joint = c.model.body_jntadr[c.model.body(name).id]
                q, v = c.model.jnt_qposadr[joint], c.model.jnt_dofadr[joint]
                changed_q.update(range(q, q + 7))
                changed_v.update(range(v, v + 6))
            keep_q = [i for i in range(c.model.nq) if i not in changed_q]
            keep_v = [i for i in range(c.model.nv) if i not in changed_v]
            preserved = bool(
                np.array_equal(before[keep_q], c.data.qpos[keep_q])
                and np.array_equal(velocities[keep_v], c.data.qvel[keep_v])
                and time == c.data.time
                and steps == c.steps
            )
            result = {
                "seed": seed,
                "physics_audit": audit,
                "layout": layout,
                "reset_fixture_used": reset_fixture_used,
                "reset_mode": reset_mode,
                "final_ready": c.ready(),
                "target": target,
                "weak": weak,
                "parameters": params,
                "initial_ready": initial_ready,
                "picked": picked,
                "pour_success": poured,
                "delivered": delivered,
                "placed": placed,
                "full_success": full,
                "after_pour": after_pour,
                "robot_recovery": recovery,
                "robot_ready": robot_ready,
                "human_reset": reset,
                "reset_preserved_robot_vessels_time": preserved,
                "pan_initial_qpos": initial_pan.tolist(),
                "rows": rows,
                "pour_end_frame": pour_end,
            }
            (root / "result.json").write_text(json.dumps(result, indent=2))
            np.savez_compressed(
                root / "motion.npz",
                qpos=np.array(frames),
                skills=np.array(phases),
                times=np.array(times),
                motion_phases=np.array(motion_phases),
            )
            print(
                json.dumps({k: v for k, v in result.items() if k not in {"rows", "after_pour"}}),
                flush=True,
            )
        finally:
            env.close()

    @staticmethod
    def render(*, root: Path) -> None:
        from hitl_pmp.environments.pouring3d.practice.environment import FrozenPourEnvironment

        result = json.loads((root / "result.json").read_text())
        env = FrozenPourEnvironment(scene_bg=True, layout=result.get("layout", "floor"))
        try:
            env.hard_reset()
            with np.load(root / "motion.npz") as motion:
                PracticeDemo.render(
                    root=root,
                    controller=env.controller(),
                    frames=list(motion["qpos"]),
                    phases=list(motion["skills"]),
                )
        finally:
            env.close()

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--seed", type=int, default=90)
        parser.add_argument("--target", type=int, choices=(1, 2, 3), default=1)
        parser.add_argument("--weak", action="store_true")
        parser.add_argument("--reset-mode", choices=("robot", "human"), default="robot")
        parser.add_argument("--layout", choices=("floor", "stations"), default="stations")
        parser.add_argument("--worker", action="store_true")
        parser.add_argument("--render", action="store_true")
        args = parser.parse_args()
        if args.render:
            SkilletCheck.render(root=args.output_dir)
        elif args.worker:
            SkilletCheck.trial(
                root=args.output_dir,
                seed=args.seed,
                target=args.target,
                weak=args.weak,
                layout=args.layout,
                reset_mode=args.reset_mode,
            )
        else:
            args.output_dir.mkdir(parents=True, exist_ok=False)
            repo = Path(__file__).resolve().parents[1]
            files = [
                Path(__file__),
                *sorted((repo / "src/hitl_pmp/environments/pouring3d").rglob("*.py")),
                *sorted((repo / "src/hitl_pmp/environments/pouring3d/assets").glob("*.json")),
            ]
            (args.output_dir / "protocol.json").write_text(
                json.dumps(
                    {
                        "commit": subprocess.check_output(
                            ["git", "rev-parse", "HEAD"], text=True
                        ).strip(),
                        "source_sha256": {
                            str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in files
                        },
                        "receiver": "Ozark Trail 15-inch skillet",
                        "layout": args.layout,
                        "seeds": list(range(90, 95)),
                        "fixed_recipes": True,
                        "surface_mu": 0.1,
                        "weak_control": "60 degrees, no wiggle, zero feed; seed 90 per target",
                    },
                    indent=2,
                )
            )
            runs = []
            for weak in (False, True):
                for target in (1, 2, 3):
                    for seed in [90] if weak else range(90, 95):
                        label = f"{'weak' if weak else 'recipe'}-{target}"
                        out = args.output_dir / label / str(seed)
                        command = [
                            sys.executable,
                            "-m",
                            "scripts.pouring3d_skillet_check",
                            "--worker",
                            "--reset-mode",
                            args.reset_mode,
                            "--layout",
                            args.layout,
                            "--output-dir",
                            str(out),
                            "--seed",
                            str(seed),
                            "--target",
                            str(target),
                        ] + (["--weak"] if weak else [])
                        runs.append(
                            SweepRun(method=label, seed=seed, output_dir=out, command=command)
                        )
            outcomes = SweepRunner.execute(runs=runs, max_workers=4)
            if not all(outcome.succeeded for outcome in outcomes):
                raise SystemExit(1)


if __name__ == "__main__":
    SkilletCheck.main()

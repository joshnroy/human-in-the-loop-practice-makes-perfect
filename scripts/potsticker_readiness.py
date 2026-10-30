"""Integrated native practice/recovery checks, with per-substep contact auditing."""

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from hitl_pmp.environments.pouring3d.practice.environment import FrozenPourEnvironment
from hitl_pmp.environments.pouring3d.practice.skill_provider import FrozenPourSkillProvider
from scripts.potsticker_feasibility import hashes, json_default, restore, save_checkpoint
from scripts.run_sweep import SweepRun, SweepRunner


def trial(
    *,
    root,
    seed,
    target,
    checkpoint=None,
    index=0,
    fixture=None,
    cycles=0,
    robot_reset=False,
    grasp_yaw=0.0,
    grasp_depth=0.032,
    grasp_angle=30.0,
    grasp_open=0.0,
    floor_carry_angle=30.0,
    grasp_mode="mass_pca",
    full_close=False,
    recovery_order="0,1,2",
):
    root.mkdir(parents=True, exist_ok=True)
    if (root / "result.json").exists():
        raise FileExistsError(root)
    env = FrozenPourEnvironment(
        payload_kind="potsticker", canonical_seed=seed, requested_count=target
    )
    provider = FrozenPourSkillProvider(env=env)
    frames, phases, times, rows = [], [], [], []
    source_hashes = hashes()
    start = time.monotonic()
    audit = {
        "obstacles": ["prep_table", "stovetop", "tray", "ground_planes_except_mobile_base"],
        "physics_steps": 0,
        "forbidden_contact_steps": 0,
        "pad_contact_steps": 0,
        "max_penetration_mm": 0.0,
        "worst_phase": None,
    }
    error = None
    try:
        c = env.controller()
        c.recovery_yaw_offset = grasp_yaw
        c.recovery_finger_offset = grasp_depth
        c.recovery_pitch_degrees = grasp_angle
        c.recovery_open_grip = grasp_open
        c.recovery_floor_carry_angle = floor_carry_angle
        c.recovery_geometry = grasp_mode
        c.recovery_contact_close = not full_close
        c.diagnostic_deadline = start + 2400
        if checkpoint:
            restore(c=c, folder=checkpoint)
        if fixture:
            obj = c.scene.get_object(c.scene.cube_names[index])
            point = [0.68, -0.85, 0.741] if fixture == "surface" else [0.35, 0.2, 0.001]
            obj.set_pose(point, [1, 0, 0, 0])
            obj.set_velocity([0.0] * 3, [0.0] * 3)
            mujoco.mj_forward(c.model, c.data)
            c.pause(phase="settle recovery fixture", seconds=2)
            assert c.scene.cube_locations()[index] == fixture
        env.current_state = env.observe()
        robot = np.array([
            (c.model.body(int(b)).name or "").startswith("robot") for b in c.model.geom_bodyid
        ])
        pads = np.array([
            (c.model.body(int(b)).name or "").endswith("_pad") for b in c.model.geom_bodyid
        ])
        obstacle = np.array([
            c.model.body(int(b)).name in {"prep_table", "stovetop", "tray"}
            for b in c.model.geom_bodyid
        ])
        ground = c.model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
        arm = robot & np.array([
            c.model.body(int(b)).name != "robot_base_link" for b in c.model.geom_bodyid
        ])
        sim = c.scene._robot_env.sim
        original_step = sim.step

        def physics_step():
            original_step()
            audit["physics_steps"] += 1
            pairs = c.data.contact.geom
            mask = (
                (robot[pairs[:, 0]] & obstacle[pairs[:, 1]])
                | (obstacle[pairs[:, 0]] & robot[pairs[:, 1]])
                | (arm[pairs[:, 0]] & ground[pairs[:, 1]])
                | (ground[pairs[:, 0]] & arm[pairs[:, 1]])
            ) & (c.data.contact.dist < 0)
            allowed = (pads[pairs[:, 0]] | pads[pairs[:, 1]]) & ("shrimp" in c.phase)
            if np.any(mask & allowed):
                audit["pad_contact_steps"] += 1
            if np.any(mask & ~allowed):
                audit["forbidden_contact_steps"] += 1
                depth = float(-c.data.contact.dist[mask & ~allowed].min() * 1000)
                if depth > audit["max_penetration_mm"]:
                    audit["max_penetration_mm"] = depth
                    audit["worst_phase"] = c.phase

        sim.step = physics_step

        def capture(**_):
            frames.append(c.data.qpos.copy())
            phases.append(c.phase)
            times.append(float(c.data.time))
            if len(frames) % 50 == 0:
                (root / "progress.json").write_text(
                    json.dumps({
                        "phase": c.phase,
                        "frames": len(frames),
                        "locations": c.scene.cube_locations(),
                        "rows": len(rows),
                    })
                )

        env.set_motion_sink(sink=capture)
        capture()

        def run(*, name, index=0, params=()):
            g = provider.ground(name=name, shrimp=index, request=target)
            env.take_action(
                action=provider.compute_action(
                    ground_skill=g, params=np.array(params), state=env.get_current_state()
                )
            )
            rows.append(dict(env.last_info))
            if not env.last_info["success"]:
                raise ValueError(
                    f"{name}: {env.last_info.get('error')} / {env.last_info.get('locations')}"
                )

        def recover(*, selected):
            location = c.scene.cube_locations()[selected]
            run(
                name={
                    "tray": "PickShrimpFromTray",
                    "surface": "PickShrimpFromSurface",
                    "floor": "PickShrimpFromFloor",
                }[location],
                index=selected,
            )
            run(name="DropShrimpIntoPitcher", index=selected)

        if cycles:
            recipe = {1: (89, 0.008, 12), 2: (89.5, 0.008, 12), 3: (90, 0.015, 12)}[target]
            for _cycle in range(cycles):
                run(name="PickPitcher")
                g = provider.ground(name=f"Pour{target}", request=target)
                env.take_action(
                    action=provider.compute_action(
                        ground_skill=g, params=np.array(recipe), state=env.get_current_state()
                    )
                )
                rows.append(dict(env.last_info))
                if env.last_info.get("error"):
                    raise ValueError(str(env.last_info["error"]))
                run(name="PlacePitcherAtHome")
                if robot_reset:
                    for selected in map(int, recovery_order.split(",")):
                        if c.scene.cube_locations()[selected] != "pitcher":
                            recover(selected=selected)
                elif c.scene.cube_locations() != ["pitcher"] * 3:
                    q = c.data.qpos.copy()
                    v = c.data.qvel.copy()
                    clock = float(c.data.time)
                    steps = c.steps
                    env.reset_movables(destination="reset_shrimp")
                    rows.append(dict(env.last_info))
                    keepq = np.ones(c.model.nq, dtype=bool)
                    keepv = np.ones(c.model.nv, dtype=bool)
                    for name in c.scene.cube_names:
                        joint = c.model.body_jntadr[c.model.body(name).id]
                        a = c.model.jnt_qposadr[joint]
                        b = c.model.jnt_dofadr[joint]
                        keepq[a : a + 7] = False
                        keepv[b : b + 6] = False
                    assert np.array_equal(q[keepq], c.data.qpos[keepq]) and np.array_equal(
                        v[keepv], c.data.qvel[keepv]
                    )
                    assert c.data.time == clock and c.steps == steps
                if not c.ready():
                    run(name="Wait")
                if not c.ready():
                    raise ValueError("Cycle did not return to ready")
        else:
            recover(selected=index)
        save_checkpoint(c=c, folder=root)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if env._controller is not None:
            c = env.controller()
            result = dict(
                count=3,
                requested=target,
                seed=seed,
                mode="joint_lateral",
                error=error,
                success=error is None and audit["forbidden_contact_steps"] == 0,
                rows=rows,
                audit=audit,
                after=c.scene.physical_snapshot(),
                ready=c.ready(),
                seconds_wall=time.monotonic() - start,
                source_sha256=source_hashes,
                grasp_settings={
                    "yaw": grasp_yaw,
                    "depth": grasp_depth,
                    "angle": grasp_angle,
                    "open": grasp_open,
                    "floor_carry_angle": floor_carry_angle,
                    "grasp_mode": grasp_mode,
                    "full_close": full_close,
                },
            )
            (root / "result.json").write_text(json.dumps(result, indent=2, default=json_default))
            np.savez_compressed(root / "motion.npz", qpos=frames, phases=phases, times=times)
            print(
                json.dumps({
                    k: result[k] for k in ("success", "error", "ready", "seconds_wall", "audit")
                }),
                flush=True,
            )
        env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target", type=int, default=1)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--fixture", choices=("surface", "floor"))
    p.add_argument("--cycles", type=int, default=0)
    p.add_argument("--robot-reset", action="store_true")
    p.add_argument("--grasp-yaw", type=float, default=0.0)
    p.add_argument("--grasp-depth", type=float, default=0.032)
    p.add_argument("--grasp-angle", type=float, default=30.0)
    p.add_argument("--grasp-open", type=float, default=0.0)
    p.add_argument("--floor-carry-angle", type=float, default=30.0)
    p.add_argument("--grasp-mode", choices=("mass_pca", "bounds_local"), default="mass_pca")
    p.add_argument("--full-close", action="store_true")
    p.add_argument("--recovery-order", default="0,1,2")
    p.add_argument("--launch", type=Path)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    if a.launch:
        runs = []
        for i, j in enumerate(json.loads(a.launch.read_text())):
            command = [sys.executable, "-m", "scripts.potsticker_readiness"]
            for k, v in j.items():
                command += ["--" + k.replace("_", "-")]
                if v is not True:
                    command.append(str(v))
            runs.append(
                SweepRun(
                    method=f"check{i}",
                    seed=j.get("seed", 0),
                    output_dir=Path(j["root"]),
                    command=command,
                )
            )
        outcomes = SweepRunner.execute(runs=runs, max_workers=a.workers)
        if not all(o.succeeded for o in outcomes):
            raise SystemExit(1)
    else:
        trial(
            root=a.root,
            seed=a.seed,
            target=a.target,
            checkpoint=a.checkpoint,
            index=a.index,
            fixture=a.fixture,
            cycles=a.cycles,
            robot_reset=a.robot_reset,
            grasp_yaw=a.grasp_yaw,
            grasp_depth=a.grasp_depth,
            grasp_angle=a.grasp_angle,
            grasp_open=a.grasp_open,
            floor_carry_angle=a.floor_carry_angle,
            grasp_mode=a.grasp_mode,
            full_close=a.full_close,
            recovery_order=a.recovery_order,
        )


if __name__ == "__main__":
    main()

"""Bounded native checks and exact-pose evidence; never launches learning jobs."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from hitl_pmp.environments.pouring3d.kitchen.environment import HUMAN_RESET, KitchenEnvironment
from hitl_pmp.environments.pouring3d.kitchen.skill_provider import KitchenSkillProvider


def check(
    *,
    output: Path,
    case: str,
    ramp_rate: float,
    requested: int,
    initial_motion: Path | None = None,
) -> bool:
    output.mkdir(parents=True, exist_ok=False)
    env = KitchenEnvironment(deployment=case == "deployment", requested_count=requested)
    provider = KitchenSkillProvider(env=env)
    frames, phases, times, hob = [], [], [], []
    rows: list[dict[str, Any]] = []
    audit: dict[str, Any] = {
        "physics_steps": 0,
        "forbidden_contact_steps": 0,
        "max_penetration_mm": 0.0,
        "worst_pair": None,
        "worst_phase": None,
    }
    start = time.monotonic()
    error = None
    try:
        c = env.controller()
        c.diagnostic_deadline = start + 1800
        names = [c.model.body(int(b)).name or "" for b in c.model.geom_bodyid]
        robot = np.array([n.startswith("robot") for n in names])
        pads = np.array([n.endswith("_pad") for n in names])
        obstacle = np.array([n in {"prep_table", "stovetop", "tray"} for n in names])
        arm = robot & np.array([n != "robot_base_link" for n in names])
        ground = c.model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
        pitcher = np.array([n == "pitcher" for n in names])
        sim = c.scene._robot_env.sim
        original_step = sim.step

        def step() -> None:
            original_step()
            audit["physics_steps"] += 1
            pairs, distances = c.data.contact.geom, c.data.contact.dist
            mask = (
                (robot[pairs[:, 0]] & obstacle[pairs[:, 1]])
                | (obstacle[pairs[:, 0]] & robot[pairs[:, 1]])
                | (arm[pairs[:, 0]] & ground[pairs[:, 1]])
                | (ground[pairs[:, 0]] & arm[pairs[:, 1]])
            ) & (distances < 0)
            allowed = (pads[pairs[:, 0]] | pads[pairs[:, 1]]) & ("shrimp" in c.phase)
            jug = (
                (robot[pairs[:, 0]] & pitcher[pairs[:, 1]])
                | (pitcher[pairs[:, 0]] & robot[pairs[:, 1]])
            ) & (distances < 0)
            if c.held == "pitcher" or c.phase in {
                "descend to pitcher handle",
                "close fingers on pitcher handle",
                "lift pitcher by handle",
                "release pitcher",
                "withdraw gripper",
            }:
                jug[:] = False
            forbidden = (mask & ~allowed) | jug
            if np.any(forbidden):
                audit["forbidden_contact_steps"] += 1
                idx = np.flatnonzero(forbidden)[np.argmin(distances[forbidden])]
                depth = -float(distances[idx]) * 1000
                if depth > audit["max_penetration_mm"]:
                    audit.update(
                        max_penetration_mm=depth,
                        worst_pair=[names[g] for g in pairs[idx]],
                        worst_phase=c.phase,
                    )

        sim.step = step

        def capture(**_: Any) -> None:
            frames.append(c.data.qpos.copy())
            phases.append(c.phase)
            times.append(float(c.data.time))
            hob.append(float(c.scene.cooking_observed()))
            if len(frames) % 50 == 0:
                (output / "progress.json").write_text(
                    json.dumps({
                        "phase": c.phase,
                        "frames": len(frames),
                        "locations": c.scene.cube_locations(),
                        "rows": rows,
                        "audit": audit,
                    })
                )

        if initial_motion is not None:
            if case not in {"recovery", "placement"}:
                raise ValueError(
                    "Recorded-pose fixture is only allowed for an explicit recovery/placement check"
                )
            recorded = np.load(initial_motion)
            index = (
                -1
                if case == "placement"
                else np.flatnonzero(recorded["phases"] == "clear workspace for reset")[-1]
            )
            c.data.qpos[:] = recorded["qpos"][index]
            c.data.qvel[:] = 0
            mujoco.mj_forward(c.model, c.data)
            c.last_q = c.data.qpos[c.arm_q].copy()
            c.base_target = c.data.qpos[c.base_q].copy()
            c.phase = "explicit recorded-pose fixture"
            if case == "placement":
                c.held, c.grip = "pitcher", 1.0
                vessel, orientation = c.body_pose(name="pitcher")
                hand, hand_rotation = c.hand_pose()
                c.hand_offset = orientation.inv().apply(hand - vessel)
                c.rotation_offset = orientation.inv() * hand_rotation
            env.current_state = env.observe()
        env.set_motion_sink(sink=capture)
        capture()

        def run(*, name: str, index: int = 0, params: tuple[float, ...] = ()) -> bool:
            g = provider.ground(name=name, request=requested, shrimp=index)
            env.take_action(action=provider.encode(ground_skill=g, params=np.array(params)))
            rows.append(dict(env.last_info))
            capture()
            return bool(env.last_info["success"])

        if case not in {"recovery", "placement"}:
            if not run(name="PickPitcher"):
                raise ValueError("Jug pickup failed")
            run(name=f"Pour{requested}", params=(ramp_rate,))
            if not run(name="PlacePitcherAtHome"):
                raise ValueError("Jug placement failed")
        elif initial_motion is None:
            raise ValueError("Recovery requires an explicit recorded-pose fixture")
        if case == "placement":
            locations = c.scene.cube_locations()
            if not run(name="PlacePitcherAtHome") or c.scene.cube_locations() != locations:
                raise ValueError("Loaded placement failed or changed ingredient containment")
        elif case == "failure":
            locations = c.scene.cube_locations()
            # The closed-loop Pour fails by overshooting (a fast ramp), not by under-pouring.
            if (
                rows[1]["success"]
                or locations.count("tray") == requested
                or any(location not in {"pitcher", "tray"} for location in locations)
            ):
                raise ValueError("Expected an observed wrong count with every ingredient contained")
            rows.append({"check": "wrong count preserved without reset", "success": True})
        elif case == "cycle":
            if rows[1]["success"] or c.scene.cube_locations().count("tray") == requested:
                raise ValueError("Cycle must start with a failed pour")
            for index, location in enumerate(c.scene.cube_locations()):
                if location == "pitcher":
                    continue
                skill = {
                    "tray": "PickShrimpFromTray",
                    "surface": "PickShrimpFromSurface",
                    "floor": "PickShrimpFromFloor",
                }[location]
                if not run(name=skill, index=index) or not run(
                    name="DropShrimpIntoPitcher", index=index
                ):
                    raise ValueError("Continuous individual recovery failed")
            if c.scene.cube_locations() != ["pitcher"] * 3:
                raise ValueError("Robot recovery did not restore all ingredients")
            if not run(name="PickPitcher"):
                raise ValueError("Post-recovery jug pickup failed")
            if not run(name=f"Pour{requested}", params=(1.0,)):
                raise ValueError("Retry failed exact-count goal")
            if not run(name="PlacePitcherAtHome"):
                raise ValueError("Post-retry jug placement failed")
            rows.append({"check": "continuous robot recovery and retry", "success": True})
        elif case == "deployment":
            before = c.data.qpos.copy()
            try:
                env.reset_movables(destination=HUMAN_RESET)
                raise AssertionError("Deployment accepted human action")
            except ValueError:
                np.testing.assert_array_equal(before, c.data.qpos)
                rows.append({"check": "deployment human rejection", "success": True})
            c.pause(phase="deployment complete", seconds=2)
        else:
            # Recover one real poured ingredient through the robot action interface.
            locations = c.scene.cube_locations()
            selected = next((i for i, loc in enumerate(locations) if loc != "pitcher"), None)
            if selected is not None:
                name = {
                    "tray": "PickShrimpFromTray",
                    "surface": "PickShrimpFromSurface",
                    "floor": "PickShrimpFromFloor",
                }[locations[selected]]
                if not run(name=name, index=selected) or not run(
                    name="DropShrimpIntoPitcher", index=selected
                ):
                    raise ValueError("Individual cold recovery failed")
            before = c.data.qpos.copy()
            env.reset_movables(destination=HUMAN_RESET)
            rows.append(dict(env.last_info))
            capture()
            # True means attempted; whether it worked is `last_info["success"]`.
            if not env.last_info["success"]:
                raise ValueError("Human bulk reset failed")
            c.pause(phase="cold practice reset complete", seconds=2)
        try:
            env.hard_reset()
            raise AssertionError("Free in-task reset accepted")
        except ValueError:
            rows.append({"check": "hidden free reset rejection", "success": True})
    except Exception as exc:
        error = repr(exc)
    finally:
        if frames:
            np.savez_compressed(
                output / "motion.npz",
                qpos=np.array(frames),
                phases=np.array(phases),
                time=np.array(times),
                hob=np.array(hob),
            )
        report = {
            "case": case,
            "deployment": env.deployment,
            "water_present": env.controller().scene.water_present(),
            "initial_motion_fixture": str(initial_motion) if initial_motion else None,
            "parameters": None
            if case in {"recovery", "placement"}
            else {"ramp_rate_deg_s": ramp_rate},
            "requested": requested,
            "error": error,
            "rows": rows,
            "audit": audit,
            "geometry": env.geometry.model_dump(),
            "locations": env.controller().scene.cube_locations(),
            "atoms": sorted(env.observe().abstract_atoms),
            "interruption_cost_total": env.interruption_cost_total,
            "wall_seconds": time.monotonic() - start,
            "passed": error is None and audit["forbidden_contact_steps"] == 0,
        }
        (output / "result.json").write_text(json.dumps(report, indent=2, default=str))
        print(json.dumps(report, default=str), flush=True)
        env.close()
    return bool(report["passed"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case",
        choices=("deployment", "practice", "failure", "recovery", "placement", "cycle"),
        default="deployment",
    )
    parser.add_argument("--initial-motion", type=Path)
    parser.add_argument(
        "--ramp-rate", type=float, default=1.0, help="Pour tilt ramp rate, deg/s (0.25-5)"
    )
    parser.add_argument("--requested", type=int, default=3)
    args = parser.parse_args()
    raise SystemExit(0 if check(**vars(args)) else 1)

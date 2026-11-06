"""Time one fixed practice cycle in a kitchen layout: ticks, wall time and base motion.

Runs PickPitcher -> Pour1 -> PlacePitcherAtHome -> (pick + drop every poured dumpling)
-> PickPitcher -> Pour3 -> PlacePitcherAtHome on one seed and one `KitchenGeometry`,
recording per skill the controller ticks, the wall-clock seconds and the ticks in which
the mobile base moved. The base is identified by its own joints (`robot_joint_x`,
`_y`, `_th`), read through `jnt_dofadr`, never by a fixed `qvel` slice. Saves every
tick's `qpos` for `kitchen_layout_render.py`. Never launches learning.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from hitl_pmp.environments.pouring3d.kitchen.config import KitchenGeometry
from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment
from hitl_pmp.environments.pouring3d.kitchen.skill_provider import KitchenSkillProvider

PICKS = {
    "tray": "PickShrimpFromTray",
    "surface": "PickShrimpFromSurface",
    "floor": "PickShrimpFromFloor",
}
# A base joint moves in a tick when its speed exceeds these (m/s, rad/s).
BASE_SPEED_M_S = 0.005
BASE_TURN_RAD_S = 0.01


class KitchenLayoutBenchmark:
    @staticmethod
    def run(*, output: Path, seed: int, geometry_path: Path | None, ramp_rate: float) -> dict:
        output.mkdir(parents=True, exist_ok=False)
        geometry = (
            KitchenGeometry.model_validate_json(geometry_path.read_text())
            if geometry_path
            else KitchenGeometry()
        )
        env = KitchenEnvironment(canonical_seed=seed, requested_count=1, geometry=geometry)
        provider = KitchenSkillProvider(env=env)
        c = env.controller()
        model = c.model
        robot = c.scene.robot_name
        base_joints = [model.joint(f"{robot}_joint_{a}").id for a in ("x", "y", "th")]
        base_dof = np.array([model.jnt_dofadr[j] for j in base_joints])
        ticks: dict[str, list[Any]] = {k: [] for k in ("qpos", "phase", "skill", "time", "hob")}
        base_v: list[np.ndarray] = []
        current = {"skill": "initial"}

        def after() -> None:
            ticks["qpos"].append(c.data.qpos.copy())
            ticks["phase"].append(c.phase)
            ticks["skill"].append(current["skill"])
            ticks["time"].append(float(c.data.time))
            ticks["hob"].append(float(c.scene.cooking_observed()))
            base_v.append(c.data.qvel[base_dof].copy())

        env.set_control_step_observers(before=None, after=after)
        rows: list[dict[str, Any]] = []

        def run(*, name: str, index: int = 0, params: tuple[float, ...] = ()) -> bool:
            g = provider.ground(name=name, shrimp=index)
            current["skill"] = name
            first = len(base_v)
            start = time.monotonic()
            env.take_action(action=provider.encode(ground_skill=g, params=np.array(params)))
            wall = time.monotonic() - start
            v = np.array(base_v[first:]).reshape(-1, 3)
            q = np.array([x[c.base_q] for x in ticks["qpos"][first:]]).reshape(-1, 3)
            moving = (np.linalg.norm(v[:, :2], axis=1) > BASE_SPEED_M_S) | (
                np.abs(v[:, 2]) > BASE_TURN_RAD_S
            )
            phases = ticks["phase"][first:]
            row = {
                "skill": name,
                "index": index,
                "success": bool(env.last_info["success"]),
                "error": env.last_info.get("error"),
                "ticks": int(env.last_info["controller_steps"]),
                "wall_seconds": wall,
                "sim_seconds": float(env.last_info["seconds"]),
                "base_moving_ticks": int(moving.sum()),
                "base_path_m": float(np.linalg.norm(np.diff(q[:, :2], axis=0), axis=1).sum())
                if len(q) > 1
                else 0.0,
                "base_turn_rad": float(np.abs(np.diff(q[:, 2])).sum()) if len(q) > 1 else 0.0,
                "base_moving_phases": sorted({p for p, m in zip(phases, moving, strict=True) if m}),
                "locations": env.last_info["locations"],
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            return row["success"]

        error = None
        start = time.monotonic()
        try:
            for name, params in (("PickPitcher", ()), ("Pour1", (ramp_rate,))):
                if not run(name=name, params=params) and name == "PickPitcher":
                    raise ValueError("PickPitcher failed")
            if not run(name="PlacePitcherAtHome"):
                raise ValueError("PlacePitcherAtHome failed")
            for index, location in enumerate(c.scene.cube_locations()):
                if location == "pitcher":
                    continue
                if not run(name=PICKS[location], index=index):
                    raise ValueError(f"{PICKS[location]} failed")
                if not run(name="DropShrimpIntoPitcher", index=index):
                    raise ValueError("DropShrimpIntoPitcher failed")
            if not run(name="PickPitcher"):
                raise ValueError("Second PickPitcher failed")
            run(name="Pour3", params=(ramp_rate,))
            run(name="PlacePitcherAtHome")
        except Exception as exc:  # noqa: BLE001 -- the report records any failure
            error = repr(exc)
        finally:
            np.savez_compressed(
                output / "motion.npz",
                qpos=np.array(ticks["qpos"]),
                phases=np.array(ticks["phase"]),
                skills=np.array(ticks["skill"]),
                time=np.array(ticks["time"]),
                hob=np.array(ticks["hob"]),
            )
            report = {
                "seed": seed,
                "ramp_rate": ramp_rate,
                "geometry": geometry.model_dump(),
                "base_dof": base_dof.tolist(),
                "control_frequency": float(c.frequency),
                "error": error,
                "rows": rows,
                "total_ticks": sum(r["ticks"] for r in rows),
                "total_wall_seconds": time.monotonic() - start,
                "final_locations": c.scene.cube_locations(),
            }
            (output / "result.json").write_text(json.dumps(report, indent=2, default=str))
            env.close()
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--kitchen-geometry", type=Path, dest="geometry_path")
    parser.add_argument("--ramp-rate", type=float, default=1.0)
    report = KitchenLayoutBenchmark.run(**vars(parser.parse_args()))
    raise SystemExit(0 if report["error"] is None else 1)

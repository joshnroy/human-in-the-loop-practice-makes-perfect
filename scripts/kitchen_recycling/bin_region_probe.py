"""Certify kitchen bin centres for the fixed 1.45 m toss, cell by cell, in the live sim.

Every cell is a bin centre on a grid over a candidate block of one side. A cell is
certified when, in order and each with the real controllers:

1. **placeable**: the bin's footprint lies wholly on its side (the mats for the far side,
   the kitchen floor for the robot side), keeps `KitchenHumanPlacement.MARGIN` from every
   static collider and the robot's start footprint, and is still within 2 cm of where it
   was put after the human-placement settle;
2. robot side only, **dumpable**: with the cube dropped in the bin, `PourCubeOutOfBin`
   runs without error and leaves the cube out of the bin, on the ground, and the bin
   upright on the ground;
3. **pickable**: the kitchen pick then holds the cube (far side: from a fixed floor spot;
   robot side: wherever the dump left it, beside the bin);
4. **plannable**: a stand exactly 1.45 m from the bin passes `Tossing3DToss`'s own
   rejection check (the standoff band, then `TossDirectionSelector` asking the real base
   motion planner) in some direction;
5. **scores**: a toss at 1.45 m with any of the given (speed, release) pairs lands in
   the bin (`InBin`). Pairs are tried in order, each after a fresh placement, dump and
   pick, stopping at the first that scores.

`--search` instead throws a speed x release grid from one cell, to find the known-good
pairs the probe uses. Results are one JSON line per cell (or per throw for `--search`).

Run on agni through `run-heavy`; each worker holds one kitchen scene (~2.2 GiB).
"""

import argparse
import json
import multiprocessing
import time
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

CONFIG = Path(__file__).resolve().parent / "experiment-config.json"
SETTLE_TOLERANCE_M = 0.02
FAR_CUBE_SPOT = (0.3, -0.6)
STANDOFF_M = 1.45


class BinRegionProbe:
    """One kitchen scene per worker process, built on first use."""

    _env: ClassVar[Any] = None

    @staticmethod
    def env() -> Any:
        from hitl_pmp.environments.kitchen_recycling3d.domain import KitchenConfig
        from hitl_pmp.environments.kitchen_recycling3d.environment import (
            KitchenRecycling3DEnvironment,
        )

        if BinRegionProbe._env is None:
            BinRegionProbe._env = KitchenRecycling3DEnvironment(
                kitchen_config=KitchenConfig.model_validate_json(CONFIG.read_text()),
                scene_bg=False,
                visual_furniture=False,
            )
        return BinRegionProbe._env

    @staticmethod
    def placement_refusal(*, env: Any, side: str, xy: tuple[float, float]) -> str | None:
        """Why a bin centred at `xy` is not a human placement on `side`, or None."""
        from hitl_pmp.environments.kitchen_recycling3d.placement import KitchenHumanPlacement

        config = env.kitchen_config
        half = np.array(config.bin_outside[:2]) / 2
        bounds = config.mat_bounds if side == "opposite_side" else config.kitchen_bounds
        centre = np.asarray(xy, dtype=float)
        if np.any(centre - half < np.array(bounds[:2])) or np.any(
            centre + half > np.array(bounds[2:])
        ):
            return "footprint off its side"
        margin = KitchenHumanPlacement.MARGIN
        state = env.kinder_backend().live_state()
        for obj in state:
            if obj.type.name != "mujoco_static_collider":
                continue
            row = np.array([float(state.get(obj, key)) for key in ("x", "y", "bb_x", "bb_y")])
            if np.all(np.abs(centre - row[:2]) < half + row[2:] / 2 + margin):
                return f"overlaps collider {obj.name}"
        outside = np.maximum(np.abs(centre) - half, 0.0)
        if float(np.linalg.norm(outside)) < config.robot_radius + margin:
            return "overlaps the robot start"
        return None

    @staticmethod
    def place(*, env: Any, bin_xy: tuple[float, float], cube_xyz: tuple[float, ...]) -> float:
        """Reset the scene, put the bin and cube down as a human would, and return how far
        the bin moved while settling."""
        env.reset_to_seed(seed=0)
        backend = env.kinder_backend()
        backend._write_placement(  # noqa: SLF001
            state=backend._require_state().copy(),  # noqa: SLF001
            placements=(("bin_0", (*bin_xy, 0.0)), ("cube_0", cube_xyz)),
        )
        env._adopt(state=env._observed_state(seed=0, steps_taken=0))  # noqa: SLF001
        state = env.get_current_state()
        settled = (
            state.get(obj=env.bin, feature_name="x"),
            state.get(obj=env.bin, feature_name="y"),
        )
        return float(np.hypot(settled[0] - bin_xy[0], settled[1] - bin_xy[1]))

    @staticmethod
    def toss_ground_skill(*, env: Any) -> Any:
        from hitl_pmp.core.method.types import GroundSkill
        from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills

        return GroundSkill(
            skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
            objects=(env.robot, env.bin, env.cube, env.barrier),
        )

    @staticmethod
    def throw(*, env: Any, speed: float, release: float) -> dict[str, Any]:
        """One toss at 1.45 m from the current (holding) state."""
        from hitl_pmp.environments.tossing3d.predicates import IN_BIN
        from hitl_pmp.environments.tossing3d.toss import Tossing3DToss

        params = np.array([STANDOFF_M, speed, release])
        state = env.get_current_state()
        refusal = Tossing3DToss.rejection_reason(
            ground_skill=BinRegionProbe.toss_ground_skill(env=env),
            state=state,
            params=params,
            parameters=Tossing3DToss.parameters_of(env=env),
        )
        if refusal is not None:
            return {"plannable": False, "refusal": refusal}
        choice = Tossing3DToss.choose_direction(state=state, params=params)
        after = env.take_action(action=Tossing3DToss.compute_action(params=params, state=state))
        return {
            "plannable": True,
            "direction_deg": choice.direction_deg,
            "stand_xy": [round(v, 3) for v in choice.stand_xy],
            "skill_error": env.last_skill_error(),
            "scored": bool(IN_BIN.holds(after, (env.cube, env.bin))),
            "cube_xy": [
                round(float(after.get(obj=env.cube, feature_name=f)), 3) for f in ("x", "y")
            ],
        }

    @staticmethod
    def prepare(*, env: Any, side: str, xy: tuple[float, float]) -> dict[str, Any]:
        """Place, (robot side) dump, and pick; the row so far, with `ready` when holding."""
        from hitl_pmp.environments.tossing3d.predicates import (
            BIN_ON_GROUND,
            HOLDING,
            IN_BIN,
            ON_GROUND,
        )

        row: dict[str, Any] = {}
        cube_z = env.kitchen_config.objects[0].dimensions[2] / 2 + 0.008
        cube = (*FAR_CUBE_SPOT, cube_z) if side == "opposite_side" else (*xy, cube_z)
        moved = BinRegionProbe.place(env=env, bin_xy=xy, cube_xyz=cube)
        row["settle_moved_m"] = round(moved, 4)
        row["placeable"] = moved <= SETTLE_TOLERANCE_M
        if not row["placeable"]:
            return {**row, "reason": "bin moved while settling"}
        if side == "robot_side":
            if not IN_BIN.holds(env.get_current_state(), (env.cube, env.bin)):
                return {**row, "reason": "cube dropped in the bin did not read InBin"}
            after = env.take_action(action=np.array([env.pour_bin_id, 0, 0, 0, 0]))
            row["dump_error"] = env.last_skill_error()
            row["dumped"] = bool(
                row["dump_error"] is None
                and not IN_BIN.holds(after, (env.cube, env.bin))
                and ON_GROUND.holds(after, (env.cube,))
                and BIN_ON_GROUND.holds(after, (env.bin,))
            )
            if not row["dumped"]:
                return {**row, "reason": "dump failed"}
        after = env.take_action(action=np.array([env.pick_cube_id, 0, 0, 0, 0]))
        row["pick_error"] = env.last_skill_error()
        row["picked"] = bool(HOLDING.holds(after, (env.robot, env.cube)))
        if not row["picked"]:
            return {**row, "reason": "pick failed"}
        return {**row, "ready": True}

    @staticmethod
    def cell(*, side: str, xy: tuple[float, float], pairs: list[tuple[float, float]]) -> dict:
        """Certify one cell. Every throw gets its own placement, dump and pick (a
        restored snapshot does not restore the gripper's fingers), and the pairs stop at
        the first that scores."""
        env = BinRegionProbe.env()
        start = time.monotonic()
        row: dict[str, Any] = {"side": side, "x": round(xy[0], 3), "y": round(xy[1], 3)}
        env.reset_to_seed(seed=0)
        refusal = BinRegionProbe.placement_refusal(env=env, side=side, xy=xy)
        row["placeable"] = refusal is None
        if refusal is not None:
            row["reason"] = refusal
            return {**row, "certified": False, "seconds": round(time.monotonic() - start, 1)}
        row["throws"] = []
        for speed, release in pairs:
            prepared = BinRegionProbe.prepare(env=env, side=side, xy=xy)
            row.update({k: v for k, v in prepared.items() if k != "ready"})
            if not prepared.get("ready"):
                return {**row, "certified": False, "seconds": round(time.monotonic() - start, 1)}
            row["throws"].append({
                "speed": speed,
                "release": release,
                **BinRegionProbe.throw(env=env, speed=speed, release=release),
            })
            if row["throws"][-1].get("scored"):
                break
        row["plannable"] = any(t["plannable"] for t in row["throws"])
        row["scored"] = sum(bool(t.get("scored")) for t in row["throws"])
        row["certified"] = row["plannable"] and row["scored"] > 0
        if not row["certified"]:
            row["reason"] = "no plannable stand" if not row["plannable"] else "no throw scored"
        row["seconds"] = round(time.monotonic() - start, 1)
        return row

    @staticmethod
    def search(*, side: str, xy: tuple[float, float], pairs: list[tuple[float, float]]) -> dict:
        """Throw every pair from one cell (one pick, restored between throws)."""
        return BinRegionProbe.cell(side=side, xy=xy, pairs=pairs)


def run_job(job: dict) -> dict:  # noqa: PLR0917 -- Pool.imap passes it positionally
    """Pool entry point (a positional callable, as `multiprocessing` demands)."""
    try:
        return BinRegionProbe.cell(
            side=job["side"], xy=tuple(job["xy"]), pairs=[tuple(p) for p in job["pairs"]]
        )
    except BaseException as error:  # noqa: BLE001 -- TrajectorySamplingFailure is not an Exception
        return {
            "side": job["side"],
            "x": job["xy"][0],
            "y": job["xy"][1],
            "certified": False,
            "reason": f"crashed: {type(error).__name__}: {error}"[:300],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("opposite_side", "robot_side"), required=True)
    parser.add_argument("--x-range", type=float, nargs=2, required=True)
    parser.add_argument("--y-range", type=float, nargs=2, required=True)
    parser.add_argument("--step", type=float, default=0.2)
    parser.add_argument(
        "--pairs", default="150:720", help="Comma-separated speed:release pairs to throw."
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Each grid cell gets one job per pair (for a speed x release search).",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pairs = [tuple(float(v) for v in pair.split(":")) for pair in args.pairs.split(",")]
    xs = np.round(np.arange(args.x_range[0], args.x_range[1] + 1e-9, args.step), 3)
    ys = np.round(np.arange(args.y_range[0], args.y_range[1] + 1e-9, args.step), 3)
    cells = [(float(x), float(y)) for x in xs for y in ys]
    jobs = (
        [{"side": args.side, "xy": c, "pairs": [p]} for c in cells for p in pairs]
        if args.search
        else [{"side": args.side, "xy": c, "pairs": pairs} for c in cells]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.output.open("a", encoding="utf-8") as stream,
        multiprocessing.get_context("spawn").Pool(args.workers, maxtasksperchild=8) as pool,
    ):
        for row in pool.imap_unordered(run_job, jobs):
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(json.dumps({k: row.get(k) for k in ("x", "y", "certified", "reason")}))


if __name__ == "__main__":
    main()

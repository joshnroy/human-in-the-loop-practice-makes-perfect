"""Measure the SweepIntoDrawer3D-o5 robot self-reset: attempt -> reset -> second attempt.

    scripts/with_env.sh python scripts/measure_sweep_drawer_reset.py --seed 3 --output-dir out/

One seed per process (fixed seeds, as run_sweep.py does): writes
`<output-dir>/<seed>/cycle.json` (every step, the reset outcome, both attempts' cube
locations) and `<output-dir>/<seed>/state_log.jsonl` (the object-centric state at every
tick). With `--replay-log` it also writes `replay_log.jsonl`, every joint position at
every tick, which `render_sweep_drawer_reset.py` draws. Aggregate with
`--summarize <output-dir>`.
"""

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

from hitl_pmp.environments.sweep_drawer3d.self_reset import SweepDrawerSelfReset
from hitl_pmp.environments.sweep_drawer3d.session import SweepDrawerSession
from hitl_pmp.environments.sweep_drawer3d.stock_skills import StockSweepSkills
from hitl_pmp.environments.sweep_drawer3d.types import SweepDrawerScene


class ResetMeasurement:
    """One seed's full cycle, and the aggregation over seeds."""

    # A cube the sweep never moved out of the pile region needs nothing from the reset;
    # counting it as "retrieved from the counter" would flatter the reset.
    UNDISTURBED = "pile (undisturbed)"

    @staticmethod
    def cycle(
        *, seed: int, out: Path, second_attempt: bool = True, replay_log: bool = False
    ) -> dict:
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        session = SweepDrawerSession(
            seed=seed,
            log_path=out / "state_log.jsonl",
            replay_path=out / "replay_log.jsonl" if replay_log else None,
        )
        StockSweepSkills.attempt(session=session, phase="attempt_1")
        after_1 = session.locations()
        piled_1 = {c: session.in_pile(cube=c) for c in SweepDrawerScene.CUBES}
        drawer_1 = session.drawer_pos()
        reset = SweepDrawerSelfReset(session=session)
        outcome = reset.run()
        record: dict = {
            "seed": seed,
            "after_attempt_1": after_1,
            "in_pile_after_attempt_1": piled_1,
            "drawer_after_attempt_1": round(drawer_1, 4),
            "reset": outcome.model_dump()
            | {"success": outcome.success, "n_in_pile": outcome.n_in_pile},
        }
        for cube, retrieval in outcome.retrievals.items():
            record["reset"]["retrievals"][cube] |= {
                "pathway": retrieval.pathway,
                "rescued_by": list(retrieval.rescued_by),
            }
        if second_attempt and outcome.success:
            StockSweepSkills.attempt(session=session, phase="attempt_2")
            record["after_attempt_2"] = session.locations()
            record["drawer_after_attempt_2"] = round(session.drawer_pos(), 4)
        record["steps"] = [s.model_dump() for s in session.steps]
        record["total_ticks"] = session.ticks
        record["total_wall_s"] = round(time.perf_counter() - t0, 1)
        session.close()
        (out / "cycle.json").write_text(json.dumps(record, indent=1))
        return record

    @staticmethod
    def wiper_home(cycle: dict) -> bool:  # noqa: PLR0917
        r = cycle["reset"]
        return bool(r["wiper_on_counter"] and r["wiper_xy_error"] < 0.02)

    @staticmethod
    def origin(*, cycle: dict, cube: str) -> str:
        """Where the first sweep left a cube, with the pile split off from the counter."""
        piled = cycle.get("in_pile_after_attempt_1", {})
        if piled.get(cube):
            return ResetMeasurement.UNDISTURBED
        return str(cycle["after_attempt_1"][cube])

    @staticmethod
    def spread(*, values: list[float]) -> dict:
        if not values:
            return {"n": 0}
        return {
            "n": len(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }

    @staticmethod
    def same_outcomes(cycle: dict) -> bool:  # noqa: PLR0917
        """Whether the second attempt's skills succeeded and failed as the first's did."""
        first = [s["success"] for s in cycle["steps"] if s["phase"] == "attempt_1"]
        second = [s["success"] for s in cycle["steps"] if s["phase"] == "attempt_2"]
        return first == second

    @staticmethod
    def summarize(*, root: Path) -> dict:
        paths = sorted(root.glob("*/cycle.json"), key=lambda p: int(p.parent.name))
        cycles = [json.loads(p.read_text()) for p in paths]
        started: Counter[str] = Counter()
        retrieved: Counter[str] = Counter()
        blocked: Counter[str] = Counter()
        blocked_retrieved: Counter[str] = Counter()
        pathways: Counter[str] = Counter()
        rescued: Counter[str] = Counter()
        for c in cycles:
            retrievals = c["reset"].get("retrievals", {})
            for cube in c["after_attempt_1"]:
                where = ResetMeasurement.origin(cycle=c, cube=cube)
                started[where] += 1
                if c["reset"]["in_pile"][cube]:
                    retrieved[where] += 1
                got = retrievals.get(cube)
                if got is None:
                    continue
                pathways[f"{where}: {got['pathway']}"] += 1
                if got["blocked"]:
                    blocked_retrieved[where] += 1
                for strategy in set(got["rescued_by"]):
                    rescued[strategy] += 1
            for cube, where in c["reset"]["locations"].items():
                origin = ResetMeasurement.origin(cycle=c, cube=cube)
                if not c["reset"]["in_pile"][cube] and origin != ResetMeasurement.UNDISTURBED:
                    blocked[f"left at {where}, from {origin}"] += 1
        attempted: Counter[str] = Counter()
        succeeded: Counter[str] = Counter()
        for c in cycles:
            for s in c["steps"]:
                if s["phase"] != "reset":
                    continue
                kind = s["name"].split("_cube_")[0]
                attempted[kind] += 1
                succeeded[kind] += bool(s["success"])
        full = [c for c in cycles if c["reset"]["success"]]
        second = [c for c in full if "after_attempt_2" in c]
        return {
            "seeds": [c["seed"] for c in cycles],
            "full_reset": f"{len(full)}/{len(cycles)}",
            "seeds_not_fully_reset": [c["seed"] for c in cycles if not c["reset"]["success"]],
            "cubes_back_in_pile_by_origin": {
                k: f"{retrieved[k]}/{started[k]}" for k in sorted(started)
            },
            "cubes_moved_by_the_sweep_back_in_pile": (
                f"{sum(v for k, v in retrieved.items() if k != ResetMeasurement.UNDISTURBED)}"
                f"/{sum(v for k, v in started.items() if k != ResetMeasurement.UNDISTURBED)}"
            ),
            "cubes_not_retrieved": dict(blocked),
            "retrieved_cubes_that_had_no_grasp_where_they_lay": dict(blocked_retrieved),
            "rescued_by_strategy": dict(rescued),
            "pathways": dict(sorted(pathways.items())),
            "reset_actions_succeeded_of_attempted": {
                k: f"{succeeded[k]}/{attempted[k]}" for k in sorted(attempted)
            },
            "drawer_closed": f"{sum(c['reset']['drawer_pos'] < 0.01 for c in cycles)}"
            f"/{len(cycles)}",
            "wiper_home": f"{sum(ResetMeasurement.wiper_home(c) for c in cycles)}/{len(cycles)}",
            "robot_actions_per_reset": ResetMeasurement.spread(
                values=[c["reset"]["robot_actions"] for c in cycles]
            ),
            "reset_wall_s": ResetMeasurement.spread(values=[c["reset"]["wall_s"] for c in cycles]),
            "reset_sim_s": ResetMeasurement.spread(
                values=[c["reset"]["ticks"] / 10 for c in cycles]
            ),
            "second_attempt_ran": f"{len(second)}/{len(full)}",
            "second_attempt_same_skill_outcomes": (
                f"{sum(ResetMeasurement.same_outcomes(c) for c in second)}/{len(second)}"
            ),
            "attempt_1_into_drawer": [
                sum(v == "drawer" for v in c["after_attempt_1"].values()) for c in cycles
            ],
            "attempt_2_into_drawer": [
                sum(v == "drawer" for v in c["after_attempt_2"].values()) for c in second
            ],
            "n_cubes": len(SweepDrawerScene.CUBES),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--no-second-attempt", action="store_true")
    parser.add_argument(
        "--replay-log",
        action="store_true",
        help="also write replay_log.jsonl, every joint position at every tick, for rendering",
    )
    args = parser.parse_args()
    if args.summarize:
        print(json.dumps(ResetMeasurement.summarize(root=args.output_dir), indent=1))
        return 0
    if args.seed is None:
        parser.error("--seed is required unless --summarize")
    rec = ResetMeasurement.cycle(
        seed=args.seed,
        out=args.output_dir / str(args.seed),
        second_attempt=not args.no_second_attempt,
        replay_log=args.replay_log,
    )
    print(json.dumps({k: rec[k] for k in ("seed", "after_attempt_1", "reset")}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

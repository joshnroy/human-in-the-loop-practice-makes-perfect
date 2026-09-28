"""Measure the SweepIntoDrawer3D-o5 robot self-reset: attempt -> reset -> second attempt.

    scripts/with_env.sh python scripts/measure_sweep_drawer_reset.py --seed 3 --output-dir out/

One seed per process (fixed seeds, as run_sweep.py does): writes
`<output-dir>/<seed>/cycle.json` (every step, the reset outcome, both attempts' cube
locations) and `<output-dir>/<seed>/state_log.jsonl` (every tick, for replay rendering).
Aggregate with `--summarize <output-dir>`.
"""

import argparse
import json
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

    @staticmethod
    def cycle(*, seed: int, out: Path, second_attempt: bool = True) -> dict:
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        session = SweepDrawerSession(seed=seed, log_path=out / "state_log.jsonl")
        StockSweepSkills.attempt(session=session, phase="attempt_1")
        after_1 = session.locations()
        drawer_1 = session.drawer_pos()
        reset = SweepDrawerSelfReset(session=session)
        outcome = reset.run()
        record: dict = {
            "seed": seed,
            "after_attempt_1": after_1,
            "drawer_after_attempt_1": round(drawer_1, 4),
            "reset": outcome.model_dump()
            | {"success": outcome.success, "n_in_pile": outcome.n_in_pile},
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
    def summarize(*, root: Path) -> dict:
        cycles = [json.loads(p.read_text()) for p in sorted(root.glob("*/cycle.json"))]
        started = Counter()
        retrieved = Counter()
        for c in cycles:
            for cube, where in c["after_attempt_1"].items():
                started[where] += 1
                if c["reset"]["in_pile"][cube]:
                    retrieved[where] += 1
        full = [c for c in cycles if c["reset"]["success"]]
        second = [c for c in full if "after_attempt_2" in c]
        return {
            "seeds": [c["seed"] for c in cycles],
            "full_reset": f"{len(full)}/{len(cycles)}",
            "cubes_back_in_pile_by_origin": {
                k: f"{retrieved[k]}/{started[k]}" for k in sorted(started)
            },
            "drawer_closed": f"{sum(c['reset']['drawer_pos'] < 0.01 for c in cycles)}"
            f"/{len(cycles)}",
            "wiper_home": f"{sum(ResetMeasurement.wiper_home(c) for c in cycles)}/{len(cycles)}",
            "robot_actions_per_reset": [c["reset"]["robot_actions"] for c in cycles],
            "reset_wall_s": [c["reset"]["wall_s"] for c in cycles],
            "reset_ticks": [c["reset"]["ticks"] for c in cycles],
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
    )
    print(json.dumps({k: rec[k] for k in ("seed", "after_attempt_1", "reset")}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

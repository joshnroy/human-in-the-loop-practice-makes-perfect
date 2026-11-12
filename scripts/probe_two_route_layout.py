"""Count the dumplings each bulk skill lands on the plate from a start layout.

Per env seed: reset, PickScraper, then one Sweep(d) or GatherAndSweep(s, delta, d),
and report how many dumplings are settled on the plate. One JSON line per seed.
Probe only; never launches learning.
"""

import argparse
import json

import numpy as np

from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
from hitl_pmp.environments.racktoserving3d.skills import SKILL_IDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", choices=("Sweep", "GatherAndSweep"), required=True)
    parser.add_argument("--params", type=float, nargs=3, default=(0.0, 0.0, 0.08))
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--start-layout", default="spaced")
    parser.add_argument("--pitch", type=float, default=RackServingConfig().spaced_pitch)
    parser.add_argument("--long-axis", default=RackServingConfig().spaced_long_axis)
    args = parser.parse_args()
    config = RackServingConfig(
        start_layout=args.start_layout,
        spaced_pitch=args.pitch,
        spaced_long_axis=args.long_axis,
        surface_kind="wire_solid",
        scraper_attachment="grasp_weld",
        bulk_transfer_only=True,
    )
    env = RackServingEnvironment(config=config, scene_bg=False, skill_step_limit=12000)
    try:
        for seed in args.seeds:
            env.reset_to_seed(seed=seed)
            backend = env.backend()
            start = [backend.food_centre(name=name) for name in backend.cube_names]
            row: dict[str, object] = {
                "seed": seed,
                "skill": args.skill,
                "params": list(args.params),
                "start_kind": backend.start_kind(),
                "start_y": [round(y, 4) for _, y in start],
            }
            try:
                env.take_action(action=np.array([SKILL_IDS["PickScraper"], 0, 0.0, 0.0, 0.0]))
                before = backend.control_steps()
                env.take_action(action=np.array([SKILL_IDS[args.skill], 0, *args.params]))
                row["steps"] = backend.control_steps() - before
                served = [name for name in backend.cube_names if backend.settled_on_tray(name=name)]
                row["served"] = served
                row["locations"] = [backend.location(name=name) for name in backend.cube_names]
                row["count"] = len(served)
            except Exception as error:  # noqa: BLE001 - a probe reports every failure
                row["error"] = f"{type(error).__name__}: {error}"
            print(json.dumps(row), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()

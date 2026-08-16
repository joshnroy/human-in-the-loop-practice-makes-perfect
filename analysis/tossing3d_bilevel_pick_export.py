"""Condense the pick-standoff sweeps into one committable JSON. Post-run only.

The raw per-seed records carry every trajectory-sampling attempt (~3 MB); this keeps
the counts, the per-seed outcomes and the cost, which is what the log cites.
"""

import argparse
import glob
import json
import os
from collections import Counter
from typing import Any

ARMS = {
    "zero_parameter_shipped_spt5": "base_s5",
    "sampled_spt5": "sampled_s5",
    "rng_shift_control_spt5": "rngctl_s5",
    "fixed_0.50_0.00_spt5": "fixed_d050_r000",
    "fixed_0.55_-0.35_spt5": "fixed_d055_rm035",
    "fixed_0.58_+0.20_spt5": "fixed_d058_rp020",
    "zero_parameter_shipped_spt25": "base_s25",
    "sampled_spt25": "sampled_s25",
    "rng_shift_control_spt25": "rngctl_s25",
}


def main() -> None:  # noqa: PLR0917
    """Write the condensed summary of every arm to --output."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    out: dict[str, Any] = {
        "environment": "kinder/Tossing3D-o1-v0",
        "seeds": [100, 139],
        "max_abstract_plans": 1,
        "planning_timeout_seconds": 1800,
        "max_skill_horizon": 400,
        "kindergarden_head": "f3c05a253a9785173431e75617f7eec5177017a2",
        "kinder_baselines_head": "7c898caf19a8b98441a9b3f13d0ac9a9374e04c0",
        "arms": {},
    }
    for name, directory in ARMS.items():
        paths = glob.glob(os.path.join(args.results_root, directory, "seed_*.json"))
        records = []
        for path in paths:
            with open(path, encoding="utf-8") as handle:
                records.append(json.load(handle))
        records.sort(key=lambda r: r["seed"])
        counts = Counter(r["outcome"] for r in records)
        attempts = [a for r in records for a in r.get("attempts", [])]
        picks = [a for a in attempts if a["skill"] == "pick_cube"]
        out["arms"][name] = {
            "n_seeds": len(records),
            "scored": counts["scored"],
            "planned_not_scored": counts["planned_not_scored"],
            "plan_not_found": counts["plan_not_found"],
            "plan_not_found_seeds": [
                r["seed"] for r in records if r["outcome"] == "plan_not_found"
            ],
            "planned_not_scored_seeds": [
                r["seed"] for r in records if r["outcome"] == "planned_not_scored"
            ],
            "total_wall_seconds": round(sum(r["total_seconds"] for r in records), 1),
            "sampler_calls": len(attempts),
            "pick_sampler_calls": len(picks),
            "pick_draws_rejected": sum(1 for a in picks if not a["reached_target_abstract_state"]),
            "per_seed_outcome": {str(r["seed"]): r["outcome"] for r in records},
        }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

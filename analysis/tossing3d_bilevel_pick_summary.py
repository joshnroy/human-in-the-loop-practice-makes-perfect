"""Summarise a tossing3d_bilevel_pick_probe sweep. Post-run analysis only."""

import argparse
import glob
import json
import os
from collections import Counter
from typing import Any


def load(results_dir: str) -> list[dict[str, Any]]:  # noqa: PLR0917
    """Read every seed_*.json in a sweep directory, seed-ordered."""
    records = []
    for path in sorted(glob.glob(os.path.join(results_dir, "seed_*.json"))):
        with open(path, encoding="utf-8") as f:
            records.append(json.load(f))
    return sorted(records, key=lambda r: r["seed"])


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:  # noqa: PLR0917
    """Outcome counts, per-outcome seed lists, and sampler cost."""
    counts = Counter(r["outcome"] for r in records)
    total = len(records)
    by_outcome = {
        outcome: [r["seed"] for r in records if r["outcome"] == outcome]
        for outcome in sorted(counts)
    }
    attempts = [a for r in records for a in r.get("attempts", [])]
    pick_attempts = [a for a in attempts if a["skill"] == "pick_cube"]
    toss_attempts = [a for a in attempts if a["skill"] != "pick_cube"]
    return {
        "n": total,
        "counts": dict(counts),
        "by_outcome": by_outcome,
        "total_wall_seconds": sum(r["total_seconds"] for r in records),
        "sampler_calls": len(attempts),
        "pick_sampler_calls": len(pick_attempts),
        "toss_sampler_calls": len(toss_attempts),
        "pick_rejections": sum(1 for a in pick_attempts if not a["reached_target_abstract_state"]),
        "toss_rejections": sum(1 for a in toss_attempts if not a["reached_target_abstract_state"]),
        "pick_sampler_seconds": sum(a["seconds"] for a in pick_attempts),
        "toss_sampler_seconds": sum(a["seconds"] for a in toss_attempts),
    }


def main() -> None:
    """Print one sweep's summary, or a per-seed diff against a baseline sweep."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--baseline-dir", default=None)
    args = parser.parse_args()

    records = load(args.results_dir)
    summary = summarise(records)
    print(json.dumps(summary, indent=2))

    if args.baseline_dir is None:
        return

    baseline = {r["seed"]: r for r in load(args.baseline_dir)}
    print("\nper-seed deltas (baseline -> this):")
    gained, lost, same = [], [], []
    for record in records:
        before = baseline[record["seed"]]["outcome"]
        after = record["outcome"]
        if before == after:
            same.append(record["seed"])
            continue
        line = f"  seed {record['seed']}: {before} -> {after}"
        if after == "scored":
            gained.append(line)
        else:
            lost.append(line)
    print(f"gained ({len(gained)}):")
    print("\n".join(gained) or "  none")
    print(f"lost ({len(lost)}):")
    print("\n".join(lost) or "  none")
    print(f"unchanged: {len(same)}/{len(records)}")


if __name__ == "__main__":
    main()

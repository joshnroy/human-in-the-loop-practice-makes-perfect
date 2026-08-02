"""Post-run analysis for the Tossing Room evaluation horizon: how much of an unpracticed
EES's score is free retries of the one stochastic skill.

Reads only already-produced output (CLAUDE.md's analysis/ convention -- never runs a
simulation): the JSON `scripts/tossingroom_horizon_sweep.py` writes, one entry per
episode recording whether it solved, how many actions it took, and which skill each
action was.

A plot would be the wrong output. The claim is not a trend to eyeball, it is an
arithmetic identity: the non-`Throw` skill counts must come out **identical** at every
horizon (the same integers, not merely similar), because the extra steps a longer
horizon grants are spent re-throwing and on nothing else. A table shows that; a curve
hides it. So this prints markdown, ready to paste into the experiment log.

Every horizon is derived from the single rollout set the sweep collected -- see that
script's docstring for why prefix-truncation is exact rather than an approximation.
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

# TossingRoomEnvironment's discrete skill ids. Hard-coded rather than imported: analysis/
# reads run output, and the ids are part of the JSON's on-disk format, not of whatever
# the library happens to define today.
_SKILL_NAMES = {0: "Pickup", 1: "MoveRoom", 2: "Throw", 3: "Press"}
_THROW = 2


class TossingRoomHorizonTable:
    """A static-method container, never instantiated."""

    @staticmethod
    def row(*, seeds: list[dict], num_test_tasks: int, horizon: int) -> dict:
        """One horizon's statistics, derived by truncating each recorded trajectory.

        Success at `horizon` is exactly `solved and steps <= horizon`; the actions that
        would have been taken are the first `min(steps, horizon)` of the record."""
        per_seed_percent: list[float] = []
        throws_per_episode: list[int] = []
        skill_counts: Counter = Counter()
        for seed_run in seeds:
            solved = 0
            for episode in seed_run["episodes"]:
                prefix = episode["skill_ids"][: min(episode["steps"], horizon)]
                solved += int(episode["solved"] and episode["steps"] <= horizon)
                throws_per_episode.append(sum(1 for skill in prefix if skill == _THROW))
                skill_counts.update(prefix)
            per_seed_percent.append(100.0 * solved / num_test_tasks)
        return {
            "horizon": horizon,
            "mean_percent": statistics.mean(per_seed_percent),
            "sd_percent": statistics.stdev(per_seed_percent) if len(per_seed_percent) > 1 else 0.0,
            "worst_percent": min(per_seed_percent),
            "mean_throws": statistics.mean(throws_per_episode),
            "max_throws": max(throws_per_episode),
            "skill_counts": {
                _SKILL_NAMES[key]: value for key, value in sorted(skill_counts.items())
            },
        }

    @staticmethod
    def per_throw_hit_rate(*, seeds: list[dict]) -> float:
        """Successes per throw issued, over the episodes that need a throw at all.

        The domain's chance rate, measured rather than assumed: a random force is
        Uniform(0, 1) against a target in Uniform(0.5, 1.0) with tolerance 0.1, which
        argues for ~0.19. This is what the runs actually realised, and it is the number
        the retry arithmetic in the log is built on."""
        episodes = [
            episode
            for seed_run in seeds
            for episode in seed_run["episodes"]
            if _THROW in episode["skill_ids"]
        ]
        throws = sum(sum(1 for s in e["skill_ids"] if s == _THROW) for e in episodes)
        return sum(e["solved"] for e in episodes) / throws if throws else 0.0

    @staticmethod
    def print_report(*, data: dict, horizons: list[int]) -> None:
        seeds = data["seeds"]
        num_test_tasks = data["num_test_tasks"]
        rows = [
            TossingRoomHorizonTable.row(seeds=seeds, num_test_tasks=num_test_tasks, horizon=horizon)
            for horizon in horizons
        ]
        print(
            f"{len(seeds)} seeds x {num_test_tasks} test tasks "
            f"= {len(seeds) * num_test_tasks} episodes"
        )
        hit_rate = TossingRoomHorizonTable.per_throw_hit_rate(seeds=seeds)
        print(f"measured per-throw hit rate: {hit_rate:.3f}")
        print()
        print(
            "| horizon | solved (mean over seeds) | sd across seeds | worst seed "
            "| `Throw`/episode | max in one episode |"
        )
        print("|---|---|---|---|---|---|")
        for row in rows:
            print(
                f"| {row['horizon']} | {row['mean_percent']:.1f}% | {row['sd_percent']:.1f} "
                f"| {row['worst_percent']:.0f}% | {row['mean_throws']:.2f} | {row['max_throws']} |"
            )
        print()
        print("skill counts by horizon (only Throw should move):")
        for row in rows:
            print(f"  H={row['horizon']:>3}: {row['skill_counts']}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[16, 12, 9, 8, 7, 6, 5],
        help="Horizons to derive. Each must be <= the sweep's own max_horizon.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data = json.loads(args.traces.read_text())
    too_long = [h for h in args.horizons if h > data["max_horizon"]]
    if too_long:
        raise ValueError(
            f"horizons {too_long} exceed the sweep's max_horizon {data['max_horizon']}; "
            "a longer horizon cannot be derived from a shorter rollout, only re-run"
        )
    TossingRoomHorizonTable.print_report(data=data, horizons=args.horizons)


if __name__ == "__main__":
    main()

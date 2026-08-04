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

**Pass one JSON per horizon**, each rolled out at that horizon. Deriving shorter
horizons by truncating a single long rollout is *not* valid here and was measured not to
be: `EesMethod` shares one RNG stream across a sweep, so a longer rollout issues more
`Throw` actions, consumes more draws, and hands every episode after the first a
different sampled force. See `scripts/tossingroom_horizon_sweep.py`'s docstring for the
seed-for-seed measurement that establishes it.

`--traces` is therefore repeatable, and each file contributes the single row for the
horizon it was rolled out at. Truncation is still used *within* a file, but only ever
down to that file's own horizon, where it is a no-op on the trajectories and merely
recovers the per-skill counts.
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
        """One horizon's statistics, over a sweep rolled out at that same horizon.

        `horizon` is the sweep's own `max_horizon`, so the truncation below is a no-op
        on the trajectories and merely recovers the per-skill counts. Calling this with
        a *shorter* horizon than the sweep was rolled out at yields an unbiased estimate
        of a different run, not this run -- see the module docstring, and note that
        nothing in `main` offers a way to do it."""
        per_seed_percent: list[float] = []
        per_seed_solved: list[int] = []
        throws_per_episode: list[int] = []
        skill_counts: Counter = Counter()
        for seed_run in seeds:
            solved = 0
            for episode in seed_run["episodes"]:
                prefix = episode["skill_ids"][: min(episode["steps"], horizon)]
                solved += int(episode["solved"] and episode["steps"] <= horizon)
                throws_per_episode.append(sum(1 for skill in prefix if skill == _THROW))
                skill_counts.update(prefix)
            per_seed_solved.append(solved)
            per_seed_percent.append(100.0 * solved / num_test_tasks)
        return {
            "horizon": horizon,
            # The counts, so the table can report a rate as the episodes behind it
            # rather than as a percentage a reader has to multiply back out. These are
            # summed per-episode `solved` flags -- the record itself, not a derivation.
            "solved": sum(per_seed_solved),
            "episodes": len(per_seed_solved) * num_test_tasks,
            "worst_solved": min(per_seed_solved),
            "num_test_tasks": num_test_tasks,
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
    def print_report(*, datasets: list[dict]) -> None:
        """One row per input file, at that file's own rolled-out horizon."""
        rows = []
        for data in datasets:
            rows.append(
                TossingRoomHorizonTable.row(
                    seeds=data["seeds"],
                    num_test_tasks=data["num_test_tasks"],
                    horizon=data["max_horizon"],
                )
            )
        first = datasets[0]
        print(
            f"{len(first['seeds'])} seeds x {first['num_test_tasks']} test tasks "
            f"= {len(first['seeds']) * first['num_test_tasks']} episodes per horizon"
        )
        for data in datasets:
            hit_rate = TossingRoomHorizonTable.per_throw_hit_rate(seeds=data["seeds"])
            print(f"  H={data['max_horizon']:>3}: measured per-throw hit rate {hit_rate:.3f}")
        print()
        print(
            "| horizon | solved | % | sd across seeds | worst seed "
            "| `Throw`/episode | max in one episode |"
        )
        print("|---|---|---|---|---|---|---|")
        for row in rows:
            solved = f"{row['solved']}/{row['episodes']}"
            worst = f"{row['worst_solved']}/{row['num_test_tasks']}"
            print(
                f"| {row['horizon']} | {solved} | {row['mean_percent']:.1f}% "
                f"| {row['sd_percent']:.1f} | {worst} | {row['mean_throws']:.2f} "
                f"| {row['max_throws']} |"
            )
        print()
        print("skill counts by horizon:")
        for row in rows:
            print(f"  H={row['horizon']:>3}: {row['skill_counts']}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traces",
        type=Path,
        action="append",
        required=True,
        help=(
            "Repeatable: one sweep JSON per horizon, each rolled out AT that horizon. "
            "There is deliberately no flag for deriving extra horizons from one file -- "
            "see this module's docstring."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    datasets = [json.loads(path.read_text()) for path in args.traces]
    datasets.sort(key=lambda data: -data["max_horizon"])
    shapes = {(len(data["seeds"]), data["num_test_tasks"]) for data in datasets}
    if len(shapes) > 1:
        raise ValueError(
            f"the sweeps disagree on (seeds, test tasks): {sorted(shapes)}; rows from "
            "different protocols must not be put in one table"
        )
    TossingRoomHorizonTable.print_report(datasets=datasets)


if __name__ == "__main__":
    main()

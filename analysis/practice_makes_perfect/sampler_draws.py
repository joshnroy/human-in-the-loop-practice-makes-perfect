"""Post-run reader for the per-draw sampler record `--record-sampler-draws` writes.

Reads `--results-root DIR` laid out as `DIR/<method>/<seed>/sampler_draws.jsonl` --
`scripts/run_sweep.py`'s own layout -- and answers the two questions integer counters
cannot: *which* parameter the sampler chose on each draw, and how that choice moved
over the run.

Never runs a simulation; see CLAUDE.md's `analysis/` convention. It imports
`SamplerDraw` from `hitl_pmp.sampler_draws` rather than redeclaring the schema, the
same way `analysis/run_timing.py` imports `RunTiming` from its writer: one definition
of the record beats a second copy that can drift out of step.

## The transitions join, and why it lives here

A draw carries the `cycle` it was made in, not an online-transition count -- a `Method`
does not know the harness's transition count, and threading one through every `Method`
to serve an analysis script would be the tail wagging the dog. `stats.json` already
records `(num_online_transitions, num_solved, num_total)` per evaluation sweep, and
sweep `i` is the sweep that measures cycle `i`, so joining the two puts draws on the
same x-axis every learning curve uses. `transitions_by_cycle` is that join, and it is
the only place it happens.

The join is a *left edge*: `transitions_by_cycle[c]` is the transition count standing
at the sweep after cycle `c`, so a draw made during cycle `c` happened at or before it.
Placing draws at the sweep that measured them is what makes a trajectory line up with
the success curve plotted above it, which is the whole point of sharing an axis.

## Reporting

`print_report` writes `x/y` for every rate, never a bare percentage -- the pools here
are exactly the ones whose denominators differ by an order of magnitude (`117/206`
informed against `48/275` uniform in PR #133), which is precisely where a percentage
misleads.
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from hitl_pmp.sampler_draws import SAMPLER_DRAWS_FILENAME, SamplerDraw


class SamplerDrawAnalysis:
    """A static-method container, never instantiated, same as every other
    business-logic class in this project."""

    @staticmethod
    def load_run(*, run_dir: Path) -> list[SamplerDraw]:
        """One run's draws, in the order they were made.

        Order is the file's own, which is chronological: the recorder appends as each
        outcome becomes known. A blank trailing line is tolerated because the file is
        flushed per draw and a killed run can leave one -- that robustness is the
        reason the format is JSONL at all, so the reader has to honour it.
        """
        path = run_dir / SAMPLER_DRAWS_FILENAME
        if not path.exists():
            return []
        return [
            SamplerDraw.model_validate_json(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]

    @staticmethod
    def load(*, results_root: Path) -> dict[tuple[str, int], list[SamplerDraw]]:
        """Every run under a results root, keyed `(method, seed)`.

        Runs with no draws file are omitted rather than recorded as empty: a
        `random-skills` arm consults no sampler and never writes one, and an arm of
        empty lists would invite an analysis to report `0/0` for it as though that
        were a measurement of the same thing.
        """
        loaded: dict[tuple[str, int], list[SamplerDraw]] = {}
        for path in sorted(results_root.glob(f"*/*/{SAMPLER_DRAWS_FILENAME}")):
            run_dir = path.parent
            draws = SamplerDrawAnalysis.load_run(run_dir=run_dir)
            if draws:
                loaded[(run_dir.parent.name, int(run_dir.name))] = draws
        return loaded

    @staticmethod
    def transitions_by_cycle(*, run_dir: Path) -> list[int]:
        """Online transitions standing at each evaluation sweep, index = cycle.

        Read from `stats.json`'s `evaluations`, which is a list of
        `(num_online_transitions, num_solved, num_total)` triples, one per sweep, with
        sweep 0 taken before any practice. See this module's docstring on why the join
        lives here rather than in the recorder.
        """
        stats = json.loads((run_dir / "stats.json").read_text())
        return [int(evaluation[0]) for evaluation in stats["evaluations"]]

    @staticmethod
    def pool_counts(*, draws: list[SamplerDraw]) -> dict[str, dict[str, tuple[int, int]]]:
        """`{skill: {consultation: (successes, attempts)}}` -- successes over attempts,
        kept as a pair so every caller reports `x/y` and none can accidentally print a
        rate whose denominator has gone missing."""
        counts: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for draw in draws:
            entry = counts[draw.skill][draw.consultation]
            entry[0] += int(draw.success)
            entry[1] += 1
        return {
            skill: {pool: (hit, total) for pool, (hit, total) in pools.items()}
            for skill, pools in counts.items()
        }

    @staticmethod
    def parameter_trajectory(
        *, draws: list[SamplerDraw], skill: str, index: int = 0
    ) -> list[tuple[int, float, bool, str]]:
        """One skill's chosen parameter over the run, as
        `(cycle, value, success, consultation)`.

        `index` selects which continuous parameter, for a skill with more than one.
        Tossing3D's `MoveToThrowPose` has exactly one -- the throw standoff -- which is
        the trajectory PR #133 could not plot.
        """
        return [
            (draw.cycle, draw.params[index], draw.success, draw.consultation)
            for draw in draws
            if draw.skill == skill and len(draw.params) > index
        ]

    @staticmethod
    def achieved(*, draws: list[SamplerDraw], skill: str, feature: str) -> list[tuple[int, float]]:
        """One post-action feature over the run, as `(cycle, value)`.

        `feature` is the recorder's `"<object>.<feature>"` key. Draws whose record does
        not carry it are skipped rather than defaulted: a missing key means the skill
        did not bind that object, which is a different statement from a zero.
        """
        return [
            (draw.cycle, draw.achieved[feature])
            for draw in draws
            if draw.skill == skill and feature in draw.achieved
        ]

    @staticmethod
    def print_report(*, results_root: Path) -> None:
        loaded = SamplerDrawAnalysis.load(results_root=results_root)
        if not loaded:
            print(f"No {SAMPLER_DRAWS_FILENAME} found under {results_root}")
            return
        by_method: dict[str, list[SamplerDraw]] = defaultdict(list)
        for (method, _seed), draws in loaded.items():
            by_method[method].extend(draws)
        print(f"{len(loaded)} run(s) under {results_root}")
        for method, draws in sorted(by_method.items()):
            seeds = sorted(seed for (name, seed) in loaded if name == method)
            print(f"\n{method}: {len(draws)} draws over {len(seeds)} seeds {seeds}")
            for skill, pools in sorted(SamplerDrawAnalysis.pool_counts(draws=draws).items()):
                parts = [f"{pool} {hit}/{total}" for pool, (hit, total) in sorted(pools.items())]
                print(f"  {skill}: " + ", ".join(parts))
                trajectory = SamplerDrawAnalysis.parameter_trajectory(draws=draws, skill=skill)
                if trajectory:
                    values = [value for _cycle, value, _ok, _pool in trajectory]
                    print(
                        f"    param[0] over {len(values)} draws: "
                        f"min {min(values):.4f}, median {statistics.median(values):.4f}, "
                        f"max {max(values):.4f}"
                    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    SamplerDrawAnalysis.print_report(results_root=args.results_root)


if __name__ == "__main__":
    main()

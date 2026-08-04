"""Post-run analysis for the Tossing Room EES bring-up: does the learned sampler
actually move `Throw`'s force onto each task's own `target_force`?

Reads only already-produced output (CLAUDE.md's analysis/ convention -- never runs a
simulation): one JSON per arm as written by `scripts/tossingroom_throw_traces.py`.

A success-rate curve alone cannot answer this domain's question, which is why this
figure exists next to `tossingroom_comparison.py`. `Throw` is the only stochastic
skill. Both panels here are horizon-independent:

* **left** -- median |chosen force - target_force| over the greedy (non-epsilon)
  throws of each evaluation sweep, against the `throw_tolerance` band a throw has to
  land inside. This is the learned quantity itself, with no retry accounting in it.
  It carries the whole result.
* **right** -- greedy throw *attempts* per evaluation episode. This panel used to be
  the horizon-robust half of the argument, on the reasoning that a policy which has
  learned the force needs one throw while a guessing policy needs however many the
  horizon allows -- it fell 2.57 -> 1.28 as the left panel entered the band.

  **Since a missed `Throw` releases the item, it is pinned at exactly 1.00 for the
  whole run**, pre-practice included: there is no second greedy throw to count, whatever
  the policy knows. It is kept because a dead-flat line at the floor is itself the
  cleanest available picture of "the retry channel is closed" -- but it is now a
  constant, so no trend should be read into it and none can be. The quantity that
  replaced its diagnostic role is the no-op fraction, which `print_table` reports.
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# Assigned in fixed order and never cycled, so an arm keeps its colour when another
# arm is added or dropped. Linestyle repeats identity as a second channel.
_ARM_STYLES: tuple[tuple[str, str], ...] = (
    ("tab:blue", "-"),
    ("tab:orange", "--"),
    ("tab:green", "-."),
    ("tab:purple", ":"),
)
# TossingRoomEnvironment.throw_tolerance's default. Hard-coded rather than imported:
# analysis/ reads run output, and a run that used a different tolerance would carry
# it in its own JSON -- see `_tolerance`.
_DEFAULT_TOLERANCE = 0.1


class TossingRoomThrowConvergence:
    """A static-method container, never instantiated."""

    @staticmethod
    def sweep_series(*, arm: dict, key: str) -> dict[int, tuple[float, float]]:
        """transitions -> (mean over seeds, stderr) of one per-sweep statistic."""
        by_transitions: dict[int, list[float]] = {}
        for seed_run in arm["seeds"]:
            for sweep in seed_run["sweeps"]:
                value = TossingRoomThrowConvergence._statistic(sweep=sweep, key=key)
                if value is not None:
                    by_transitions.setdefault(int(sweep["transitions"]), []).append(value)
        out: dict[int, tuple[float, float]] = {}
        for transitions, values in sorted(by_transitions.items()):
            stderr = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
            out[transitions] = (statistics.mean(values), stderr)
        return out

    @staticmethod
    def _statistic(*, sweep: dict, key: str) -> float | None:
        errors = sweep["greedy_throw_errors"]
        if key == "median_error":
            return statistics.median(errors) if errors else None
        if key == "attempts_per_episode":
            # Episodes needing a throw at all -- the button-press goal family has
            # param_dim 0 and would otherwise dilute this toward zero.
            throw_episodes = sweep["skill_counts"].get("Pickup", 0)
            return len(errors) / throw_episodes if throw_episodes else None
        if key == "frac_within_tolerance":
            return None if not errors else sum(e < _DEFAULT_TOLERANCE for e in errors) / len(errors)
        if key == "noop_fraction":
            return sweep["num_noop_actions"] / sweep["num_actions"] if sweep["num_actions"] else 0.0
        raise ValueError(f"unknown statistic {key!r}")

    @staticmethod
    def _plot(*, ax, series: dict[int, tuple[float, float]], label: str, **kwargs) -> None:
        xs = sorted(series)
        means = [series[x][0] for x in xs]
        errs = [series[x][1] for x in xs]
        (line,) = ax.plot(xs, means, label=label, linewidth=2, **kwargs)
        ax.fill_between(
            xs,
            [m - e for m, e in zip(means, errs, strict=True)],
            [m + e for m, e in zip(means, errs, strict=True)],
            color=line.get_color(),
            alpha=0.15,
            linewidth=0,
        )

    @staticmethod
    def render(*, arms: list[dict], output: Path, tolerance: float) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for index, arm in enumerate(arms):
            color, linestyle = _ARM_STYLES[index % len(_ARM_STYLES)]
            TossingRoomThrowConvergence._plot(
                ax=axes[0],
                series=TossingRoomThrowConvergence.sweep_series(arm=arm, key="median_error"),
                label=arm["label"],
                color=color,
                linestyle=linestyle,
            )
            TossingRoomThrowConvergence._plot(
                ax=axes[1],
                series=TossingRoomThrowConvergence.sweep_series(
                    arm=arm, key="attempts_per_episode"
                ),
                label=arm["label"],
                color=color,
                linestyle=linestyle,
            )
        axes[0].axhspan(0, tolerance, color="tab:green", alpha=0.12, linewidth=0)
        axes[0].axhline(tolerance, color="tab:green", linewidth=1.2, linestyle=(0, (2, 3)))
        axes[0].text(
            0.99,
            tolerance,
            f" throw_tolerance = {tolerance}",
            transform=axes[0].get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8,
            color="tab:green",
        )
        axes[0].set_ylabel("median |force − target_force|")
        axes[0].set_title("The learned quantity", fontsize=10)
        axes[1].axhline(1.0, color="grey", linewidth=1.2, linestyle=(0, (2, 3)))
        axes[1].text(
            0.99,
            1.0,
            " one throw per episode",
            transform=axes[1].get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8,
            color="grey",
        )
        axes[1].set_ylabel("greedy Throw attempts per throw episode")
        axes[1].set_title(
            "Retries available: none, at any competence (a miss costs the item)", fontsize=10
        )
        for ax in axes:
            ax.set_xlabel("Number of online transitions")
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.25, linewidth=0.6)
            ax.legend(fontsize=8, framealpha=0.9)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        fig.suptitle(
            "Tossing Room: EES learns the throw force (evaluation sweeps, mean ± stderr "
            "over seeds)",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(output, dpi=150)

    @staticmethod
    def print_table(*, arms: list[dict]) -> None:
        for arm in arms:
            print(f"\n{arm['label']}  ({len(arm['seeds'])} seeds)")
            header = (
                f"{'transitions':>12}{'solved':>10}{'%':>8}{'med err':>9}"
                f"{'within tol':>12}{'throws/ep':>11}{'no-op %':>9}"
            )
            print(header)
            print("-" * len(header))
            solved = TossingRoomThrowConvergence._solved_series(arm=arm)
            counts = TossingRoomThrowConvergence._solved_counts(arm=arm)
            median = TossingRoomThrowConvergence.sweep_series(arm=arm, key="median_error")
            within = TossingRoomThrowConvergence.sweep_series(arm=arm, key="frac_within_tolerance")
            attempts = TossingRoomThrowConvergence.sweep_series(arm=arm, key="attempts_per_episode")
            noop = TossingRoomThrowConvergence.sweep_series(arm=arm, key="noop_fraction")
            for transitions in sorted(solved):
                count = "{}/{}".format(*counts[transitions])
                print(
                    f"{transitions:>12}{count:>10}{solved[transitions][0]:>8.1f}"
                    f"{median.get(transitions, (float('nan'), 0))[0]:>9.3f}"
                    f"{100 * within.get(transitions, (float('nan'), 0))[0]:>11.0f}%"
                    f"{attempts.get(transitions, (float('nan'), 0))[0]:>11.2f}"
                    f"{100 * noop.get(transitions, (float('nan'), 0))[0]:>8.0f}%"
                )

    @staticmethod
    def _solved_counts(*, arm: dict) -> dict[int, tuple[int, int]]:
        """`{transitions: (solved, total)}` summed across seeds -- the episode counts the
        traces already carry, so the table can print the rate as what was measured rather
        than as a percentage a reader would have to multiply by the seed count to invert.
        `_solved_series` keeps the mean and standard error of the per-seed rates, which
        are spreads of rates and stay in points."""
        by_transitions: dict[int, tuple[int, int]] = {}
        for seed_run in arm["seeds"]:
            for sweep in seed_run["sweeps"]:
                solved, total = by_transitions.get(int(sweep["transitions"]), (0, 0))
                by_transitions[int(sweep["transitions"])] = (
                    solved + int(sweep["solved"]),
                    total + int(sweep["total"]),
                )
        return dict(sorted(by_transitions.items()))

    @staticmethod
    def _solved_series(*, arm: dict) -> dict[int, tuple[float, float]]:
        by_transitions: dict[int, list[float]] = {}
        for seed_run in arm["seeds"]:
            for sweep in seed_run["sweeps"]:
                by_transitions.setdefault(int(sweep["transitions"]), []).append(
                    100.0 * sweep["solved"] / sweep["total"]
                )
        return {
            transitions: (
                statistics.mean(values),
                statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0,
            )
            for transitions, values in sorted(by_transitions.items())
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traces",
        type=Path,
        action="append",
        required=True,
        help="Repeatable: one arm JSON from scripts/tossingroom_throw_traces.py.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--throw-tolerance", type=float, default=_DEFAULT_TOLERANCE)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    arms = [json.loads(path.read_text()) for path in args.traces]
    TossingRoomThrowConvergence.print_table(arms=arms)
    TossingRoomThrowConvergence.render(
        arms=arms, output=args.output, tolerance=args.throw_tolerance
    )


if __name__ == "__main__":
    main()

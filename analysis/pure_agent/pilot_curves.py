"""Post-run analysis for the pure-agent pilot on Tossing Room: **what does a coding agent
that writes the policy achieve, and what did the writing cost?**

**Post-run only.** It reads `stats.json` and `transcript.json` back out of directories
`hitl_pmp.cli` already wrote and never drives a `Method` itself.

**One seed, so no spread is established and no test is computed.** Every number below is a
single run's count. Nothing here compares two arms statistically, and nothing should: with
n=1 per arm there is no paired structure, no variance estimate and therefore no inference
available. The two prompt arms are plotted together because that is how you *see* whether
the plumbing carried both, not because the gap between them is a measurement of what the
hint is worth. `print_report` says so in its own output, so a reader who sees only the
console text still gets the caveat.

**Counts, never bare percentages.** Tossing Room's 30-task test set is 14 TRASH /
14 RECYCLING / **2** EMPTY, so an EMPTY column reading "100%" is `2/2` and supports
essentially nothing. Every figure label and every table cell here is `x/y`.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from analysis.practice_makes_perfect.goal_families import GoalFamilies  # noqa: E402

# The families in the order they are reported, which is descending denominator -- so the
# two-task EMPTY column is never the first thing a reader's eye lands on.
FAMILIES = ("TRASH", "RECYCLING", "EMPTY")


class PureAgentPilotCurves:
    """A static-method container, never instantiated, same as every other business-logic
    class in this project."""

    @staticmethod
    def style(*, index: int) -> tuple[str, str]:
        """Colour and linestyle for the `index`-th arm, in the order the caller passed
        them. By position rather than by name so the script carries no assumption about
        which arms exist -- the pilot plots four (the recipe's ladder plus EES at the same
        budget), and a later run may plot two or six.

        Solid for the first arm, then alternating dash patterns, so the figure survives
        being printed in greyscale."""
        colours = ("#1f4e9c", "#c2571a", "#2e7d4f", "#6a3d9a", "#a3243b")
        linestyles = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)))
        return colours[index % len(colours)], linestyles[index % len(linestyles)]

    @staticmethod
    def load_run(*, directory: Path) -> dict:
        """One run's `stats.json`, plus its `transcript.json` if the directory has one.

        The transcript is optional because a *replay* directory has no transcript of its
        own -- the transcript lives with the authoring run that produced it, and a caller
        may point at either."""
        stats = json.loads((directory / "stats.json").read_text())
        transcript_path = directory / "transcript.json"
        transcript = json.loads(transcript_path.read_text()) if transcript_path.is_file() else None
        return {"stats": stats, "transcript": transcript, "directory": directory}

    @staticmethod
    def curve(*, stats: dict, family: str | None) -> list[tuple[int, int, int]]:
        """`(transitions, solved, total)` per evaluation sweep, for one family or pooled.

        Read off `breakdowns` rather than `evaluations` whenever a family is asked for,
        because only the per-task detail can split the pooled count -- and the two are
        cross-checked in `print_report`, so a breakdown that disagreed with its own
        aggregate would surface rather than quietly redefine a denominator."""
        if family is None:
            return [
                (transitions, solved, total) for transitions, solved, total in stats["evaluations"]
            ]
        rows = []
        for breakdown in stats["breakdowns"]:
            outcomes = [
                outcome
                for outcome in breakdown["outcomes"]
                if GoalFamilies.classify(goal=outcome["goal"]) == family
            ]
            rows.append((
                breakdown["num_online_transitions"],
                sum(1 for outcome in outcomes if outcome["solved"]),
                len(outcomes),
            ))
        return rows

    @staticmethod
    def practice_totals(*, stats: dict) -> dict[str, tuple[int, int]]:
        """`{lifted skill: (successes, attempts)}` summed over every recorded window.

        The pure agent has no learned sampler, so only this overall pair is comparable to
        EES's -- the `SamplerConsultation` pools mean something different on this arm and
        are deliberately not surfaced here. See `PureAgentMethod`'s docstring."""
        totals: dict[str, tuple[int, int]] = {}
        for window in stats.get("practice_outcomes_per_cycle", []):
            for name, tally in window.items():
                successes, attempts = totals.get(name, (0, 0))
                totals[name] = (
                    successes + tally["num_successes"],
                    attempts + tally["num_attempts"],
                )
        return totals

    @staticmethod
    def print_report(*, runs: dict[str, dict]) -> None:
        print("\n=== pure-agent pilot, Tossing Room, ONE SEED ===")
        print(
            "One seed per arm. No per-seed spread is established and no significance "
            "test is computed or supported; these are single-run counts."
        )
        for label, run in runs.items():
            stats = run["stats"]
            pooled = PureAgentPilotCurves.curve(stats=stats, family=None)
            first, final = pooled[0], pooled[-1]
            print(f"\n--- {label} ---")
            print(f"  sweeps: {len(pooled)}  (sweep 0 is before any practice)")
            print(f"  sweep 0 (zero-feedback policy): {first[1]}/{first[2]} solved")
            print(f"  final sweep:                    {final[1]}/{final[2]} solved")
            for family in FAMILIES:
                rows = PureAgentPilotCurves.curve(stats=stats, family=family)
                print(f"    {family:<10} {rows[0][1]}/{rows[0][2]} -> {rows[-1][1]}/{rows[-1][2]}")
            # Cross-check: the per-family denominators must sum to the pooled one, or a
            # family rule has gone stale and every family row above is on a wrong base.
            family_total = sum(
                PureAgentPilotCurves.curve(stats=stats, family=family)[-1][2] for family in FAMILIES
            )
            if family_total != final[2]:
                raise ValueError(
                    f"{label}: family denominators sum to {family_total}, but the pooled "
                    f"sweep reports {final[2]} tasks. A goal-family rule is stale."
                )
            practice = PureAgentPilotCurves.practice_totals(stats=stats)
            for name in sorted(practice):
                successes, attempts = practice[name]
                print(f"    practice {name:<18} {successes}/{attempts} achieved add effects")
            PureAgentPilotCurves.print_cost(label=label, transcript=run["transcript"])

    @staticmethod
    def print_cost(*, label: str, transcript: dict | None) -> None:
        if transcript is None:
            print("    authoring cost: not recorded in this directory (replay run)")
            return
        rounds = transcript["rounds"]
        costs = [entry["total_cost_usd"] for entry in rounds]
        known = [cost for cost in costs if cost is not None]
        failed = sum(1 for entry in rounds if entry["load_error"] is not None)
        cut_off = sum(1 for entry in rounds if entry.get("query_error") is not None)
        print(f"    authoring rounds: {len(rounds)}")
        print(f"    rounds whose policy would not load: {failed}/{len(rounds)}")
        print(f"    rounds cut off by the budget cap:   {cut_off}/{len(rounds)}")
        print(
            f"    spend: ${sum(known):.4f} over {len(known)}/{len(rounds)} rounds that "
            "reported a cost (a LOWER bound if that is not all of them)"
        )
        print(
            f"    be-the-policy price: {transcript['num_decisions']} decisions, i.e. the "
            "API calls a variant querying inside the policy would have made"
        )
        print(
            f"    malformed decisions: {transcript['num_malformed_decisions']}/"
            f"{transcript['num_decisions']}"
        )
        del label

    @staticmethod
    def render(*, runs: dict[str, dict], output: Path, title: str) -> None:
        """Four panels: the pooled learning curve, then one per goal family.

        Split by family rather than pooled alone because on this domain the families fail
        for structurally different reasons -- a missed throw is merely expensive for TRASH
        (walk back to the pile) and terminal for RECYCLING (the one-way ledge has closed).
        A pooled curve cannot show which of the two an arm is losing.

        **No shaded band and no error bar anywhere**, deliberately: with one seed there is
        nothing to shade, and a band drawn from a single run would assert a spread that was
        never measured."""
        fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.4))
        panels = [(None, "all tasks")] + [(family, family) for family in FAMILIES]
        for axis, (family, name) in zip(axes, panels, strict=True):
            total = 0
            for index, (label, run) in enumerate(runs.items()):
                colour, linestyle = PureAgentPilotCurves.style(index=index)
                rows = PureAgentPilotCurves.curve(stats=run["stats"], family=family)
                xs = [transitions for transitions, _, _ in rows]
                ys = [solved for _, solved, _ in rows]
                total = rows[-1][2]
                axis.plot(
                    xs,
                    ys,
                    color=colour,
                    linestyle=linestyle,
                    linewidth=2.2,
                    marker="o",
                    markersize=4.5,
                    label=f"{label} — {rows[-1][1]}/{total}",
                )
            axis.set_title(f"{name} (x/{total})", fontsize=10)
            axis.set_xlabel("online transitions")
            axis.grid(alpha=0.25, linewidth=0.6)
            # Headroom for the legend, which carries each arm's final count and so must
            # not sit on top of the lines. Several arms here are flat on zero, so the
            # bottom of every panel is occupied and the legend has to go up.
            axis.set_ylim(-total * 0.06, total * 1.55)
            # Integer ticks, because every y value here is a count of tasks. Left to
            # matplotlib, the two-task EMPTY panel gets ticks at 0.5 -- "1.5 tasks
            # solved", which is not a thing.
            axis.yaxis.set_major_locator(MaxNLocator(integer=True))
            axis.legend(fontsize=8, loc="upper left", framealpha=0.95)
        axes[0].set_ylabel("test tasks solved (one seed)", fontsize=9)
        fig.suptitle(title, fontsize=10.5)
        fig.tight_layout()
        fig.savefig(output, dpi=160)
        print(f"wrote {output}")

    @staticmethod
    def parse_run(*, value: str) -> tuple[str, Path]:
        """`label=path`, so a caller names its own arms and the figure legend reads in the
        caller's vocabulary rather than in a directory name's."""
        label, _, path = value.partition("=")
        if not label or not path:
            raise argparse.ArgumentTypeError(f"expected label=path, got {value!r}")
        return label, Path(path)

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--run",
            action="append",
            required=True,
            type=lambda value: PureAgentPilotCurves.parse_run(value=value),
            help="label=DIR, repeatable. DIR holds a run's stats.json, and its "
            "transcript.json too if it was the authoring run.",
        )
        parser.add_argument("--output", type=Path, required=True, help="Figure path (.png).")
        parser.add_argument(
            "--title",
            type=str,
            default="Pure agent on Tossing Room — one seed, 30 test tasks",
            help="Figure suptitle.",
        )
        args = parser.parse_args()
        runs = {
            label: PureAgentPilotCurves.load_run(directory=directory)
            for label, directory in args.run
        }
        PureAgentPilotCurves.print_report(runs=runs)
        PureAgentPilotCurves.render(runs=runs, output=args.output, title=args.title)


if __name__ == "__main__":
    PureAgentPilotCurves.main()

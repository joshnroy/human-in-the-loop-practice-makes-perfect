"""Post-run analysis for the pure-agent feedback 2x2 on Tossing Room:
**{zero-shot, in-context} x {sonnet, opus}**, prompt arm held at `minimal`.

**Post-run only.** It reads `stats.json` (and the authoring `transcript.json`) back out of
directories `hitl_pmp.cli` already wrote, and never drives a `Method` itself.

**What the seed set is, and what it is not.** Each of the four cells was *authored once* --
authoring queries a real agent and is nondeterministic, so it cannot be repeated per seed
without repeating the money and losing reproducibility. That single transcript is then
replayed across a fixed seed set, which varies the task draw, the item weights and the
practice dynamics while holding the authored policy fixed.

So a paired test over those seeds answers **"does this particular authored policy solve
more tasks than that one, on this task distribution?"** -- a real and answerable question.
It does **not** answer "is the in-context arm better than the zero-shot arm", because at the
arm level n=1: one authoring run per cell, and authoring variance is entirely unmeasured.
`print_report` says both of those in its own output, so a reader who sees only the console
text still gets the distinction.

**Paired, because the arms share their seeds.** Every arm is replayed on the same seed set,
so the per-seed counts are paired data. An unpaired test would throw that structure away and
understate the evidence. Wilcoxon signed-rank rather than a paired t-test: n is small, the
outcome is a bounded count, and normality is not on offer.

**Counts, never bare percentages.** Tossing Room's 30-task test set is 14 TRASH /
14 RECYCLING / **2** EMPTY, so an EMPTY column reading "100%" is `2/2` and supports
essentially nothing. Every figure label and every table cell here is `x/y`.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from analysis.practice_makes_perfect.goal_families import GoalFamilies  # noqa: E402
from analysis.practice_makes_perfect.paired_tests import PairedTests  # noqa: E402

FAMILIES = ("TRASH", "RECYCLING", "EMPTY")
# The four cells, in the order they are reported and plotted.
ARMS = ("zero-shot-sonnet", "in-context-sonnet", "zero-shot-opus", "in-context-opus")


class FeedbackArm2x2:
    """A static-method container, never instantiated, same as every other business-logic
    class in this project."""

    @staticmethod
    def load_arm(*, root: Path) -> dict[int, dict]:
        """`{seed: stats}` for one arm, read out of `<root>/pure-agent/<seed>/stats.json`
        -- the layout `scripts/run_sweep.py` writes."""
        runs: dict[int, dict] = {}
        for directory in sorted((root / "pure-agent").glob("*")):
            if not (directory / "stats.json").is_file():
                continue
            runs[int(directory.name)] = json.loads((directory / "stats.json").read_text())
        return runs

    @staticmethod
    def curve(*, stats: dict, family: str | None) -> list[tuple[int, int, int]]:
        """`(transitions, solved, total)` per evaluation sweep, for one family or pooled."""
        if family is None:
            return [(t, solved, total) for t, solved, total in stats["evaluations"]]
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
    def final_solved(*, runs: dict[int, dict], family: str | None = None) -> dict[int, int]:
        """`{seed: tasks solved at the final sweep}`, the quantity every test below is on."""
        return {
            seed: FeedbackArm2x2.curve(stats=stats, family=family)[-1][1]
            for seed, stats in runs.items()
        }

    @staticmethod
    def best_solved(*, runs: dict[int, dict]) -> dict[int, int]:
        """`{seed: the best sweep's solved count}`.

        Reported beside the final sweep because a revise-loop is not monotone: an agent can
        author a better policy at round k than at round k+1, and a final-sweep-only report
        would score the arm on its last draft rather than its best one. Both are shown; the
        final sweep stays the headline, since picking the best sweep post hoc is a choice
        made with the test set in view."""
        return {
            seed: max(solved for _, solved, _ in FeedbackArm2x2.curve(stats=stats, family=None))
            for seed, stats in runs.items()
        }

    @staticmethod
    def paired_test(*, left: dict[int, int], right: dict[int, int], label: str) -> str:
        """Exact Wilcoxon signed-rank on the per-seed differences, plus the power question.

        Uses `analysis/practice_makes_perfect/paired_tests.py` rather than a local
        implementation: that module exists precisely because `PairedTests` was once defined
        twice and the copies diverged, and it is domain-agnostic (it takes a list of
        differences and nothing else). It is exact by enumeration, so no scipy and no normal
        approximation.

        **Never asserts an effect without a p-value.** And when p > 0.05 the minimum
        detectable effect is printed beside it, which is this project's standing requirement
        for a null result: it separates "no effect" from "no power"."""
        seeds = sorted(set(left) & set(right))
        if not seeds:
            return f"{label}: no shared seeds, no test"
        differences = [float(left[seed] - right[seed]) for seed in seeds]
        array = np.array(differences)
        summary = (
            f"{label}: n={len(seeds)} paired seeds, "
            f"median difference {np.median(array):+.1f}, "
            f"mean {array.mean():+.2f}, "
            f"wins {int((array > 0).sum())}/{len(seeds)}, "
            f"ties {int((array == 0).sum())}/{len(seeds)}, "
            f"losses {int((array < 0).sum())}/{len(seeds)}"
        )
        if all(value == 0.0 for value in differences):
            return summary + (
                "\n      no test: every paired difference is exactly 0, so the two arms "
                "solved the identical task set on every seed"
            )
        result = PairedTests.wilcoxon_signed_rank(differences=differences)
        line = (
            f"\n      Wilcoxon signed-rank (exact) W+={result.statistic:g}, "
            f"two-sided p = {result.p_value:.4g}, "
            f"{result.num_zero_differences}/{len(seeds)} zero differences dropped"
        )
        if result.p_value > 0.05:
            mde = PairedTests.minimum_detectable_effect(differences=differences)
            line += (
                f"\n      NOT established at alpha=0.05. Minimum detectable effect at this "
                f"spread and n: {mde:.2f} tasks (80% power) -- so a real effect smaller "
                "than that would not have been found here."
            )
        return summary + line

    @staticmethod
    def print_report(*, arms: dict[str, dict[int, dict]], transcripts: dict[str, dict]) -> None:
        print("\n=== pure-agent feedback 2x2, Tossing Room ===")
        print(
            "Each cell was AUTHORED ONCE (authoring is nondeterministic and paid) and that\n"
            "one transcript replayed across the seed set. So the seeds vary the task draw,\n"
            "the item weights and the practice dynamics -- NOT the authoring. A paired test\n"
            "below therefore answers 'does this authored policy beat that one on this task\n"
            "distribution', and NOT 'is this arm better than that arm': at the arm level\n"
            "n=1 authoring run per cell and authoring variance is unmeasured."
        )
        for arm in ARMS:
            runs = arms.get(arm, {})
            if not runs:
                print(f"\n--- {arm} --- (no runs found)")
                continue
            finals = FeedbackArm2x2.final_solved(runs=runs)
            bests = FeedbackArm2x2.best_solved(runs=runs)
            total = FeedbackArm2x2.curve(stats=next(iter(runs.values())), family=None)[-1][2]
            print(f"\n--- {arm} --- {len(runs)} seeds")
            per_seed = ", ".join(f"s{seed}:{finals[seed]}/{total}" for seed in sorted(finals))
            print(f"  final sweep, per seed: {per_seed}")
            print(
                f"  final sweep, pooled:   {sum(finals.values())}/{total * len(finals)} "
                f"(median {int(np.median(list(finals.values())))}/{total}, "
                f"min {min(finals.values())}/{total}, max {max(finals.values())}/{total})"
            )
            print(
                f"  best sweep, pooled:    {sum(bests.values())}/{total * len(bests)} "
                "(a revise loop is not monotone; the final sweep stays the headline)"
            )
            for family in FAMILIES:
                family_finals = FeedbackArm2x2.final_solved(runs=runs, family=family)
                family_total = FeedbackArm2x2.curve(stats=next(iter(runs.values())), family=family)[
                    -1
                ][2]
                print(
                    f"    {family:<10} {sum(family_finals.values())}/"
                    f"{family_total * len(family_finals)} pooled over seeds"
                )
            FeedbackArm2x2.print_cost(transcript=transcripts.get(arm))

        print("\n=== paired comparisons (same seeds, so paired data) ===")
        for model in ("sonnet", "opus"):
            left, right = f"in-context-{model}", f"zero-shot-{model}"
            if left in arms and right in arms:
                print(
                    "  "
                    + FeedbackArm2x2.paired_test(
                        left=FeedbackArm2x2.final_solved(runs=arms[left]),
                        right=FeedbackArm2x2.final_solved(runs=arms[right]),
                        label=f"in-context MINUS zero-shot, {model}",
                    )
                )
        for feedback in ("zero-shot", "in-context"):
            left, right = f"{feedback}-opus", f"{feedback}-sonnet"
            if left in arms and right in arms:
                print(
                    "  "
                    + FeedbackArm2x2.paired_test(
                        left=FeedbackArm2x2.final_solved(runs=arms[left]),
                        right=FeedbackArm2x2.final_solved(runs=arms[right]),
                        label=f"opus MINUS sonnet, {feedback}",
                    )
                )

    @staticmethod
    def print_cost(*, transcript: dict | None) -> None:
        if transcript is None:
            print("    authoring: no transcript supplied for this arm")
            return
        rounds = transcript["rounds"]
        known = [entry["total_cost_usd"] for entry in rounds if entry["total_cost_usd"] is not None]
        failed = sum(1 for entry in rounds if entry["load_error"] is not None)
        cut_off = sum(1 for entry in rounds if entry.get("query_error") is not None)
        print(f"    authoring rounds: {len(rounds)}")
        print(f"      policy would not load: {failed}/{len(rounds)}")
        print(f"      query cut off:         {cut_off}/{len(rounds)}")
        print(
            f"      reported spend: ${sum(known):.4f} over {len(known)}/{len(rounds)} "
            "rounds that reported a cost (a LOWER bound if that is not all of them). "
            "This is prpl-agent-utils' accounting, not a bill: the runs authenticate "
            "through a credential broker against a subscription."
        )

    @staticmethod
    def render(*, arms: dict[str, dict[int, dict]], output: Path, title: str) -> None:
        """Two panels of learning curves (one per model) plus one paired-difference panel.

        **Per-seed spread, not just a mean.** Each arm's individual seeds are drawn as thin
        translucent lines behind the median, because with ten seeds a bar chart of two means
        can hide one seed driving the whole effect. The third panel plots the paired
        per-seed differences directly, which is the quantity the test is computed on -- so
        the figure shows the same thing the p-value is about."""
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
        colours = {"zero-shot": "#1f4e9c", "in-context": "#c2571a"}
        total = 30
        for axis, model in zip(axes[:2], ("sonnet", "opus"), strict=True):
            for feedback in ("zero-shot", "in-context"):
                runs = arms.get(f"{feedback}-{model}", {})
                if not runs:
                    continue
                curves = [FeedbackArm2x2.curve(stats=stats, family=None) for stats in runs.values()]
                total = curves[0][-1][2]
                xs = [t for t, _, _ in curves[0]]
                ys = np.array([[solved for _, solved, _ in curve] for curve in curves])
                for row in ys:
                    axis.plot(xs, row, color=colours[feedback], alpha=0.22, linewidth=1.0)
                median = np.median(ys, axis=0)
                axis.plot(
                    xs,
                    median,
                    color=colours[feedback],
                    linewidth=2.4,
                    marker="o",
                    markersize=4.5,
                    label=f"{feedback} — median {int(median[-1])}/{total}",
                )
            axis.set_title(f"{model} (thin lines = individual seeds)", fontsize=10)
            axis.set_xlabel("online transitions")
            axis.set_ylabel(f"test tasks solved (x/{total})", fontsize=9)
            axis.grid(alpha=0.25, linewidth=0.6)
            axis.set_ylim(-total * 0.06, total * 1.12)
            axis.yaxis.set_major_locator(MaxNLocator(integer=True))
            axis.legend(fontsize=8, loc="upper left", framealpha=0.95)

        axis = axes[2]
        positions, labels = [], []
        for index, model in enumerate(("sonnet", "opus")):
            left, right = f"in-context-{model}", f"zero-shot-{model}"
            if left not in arms or right not in arms:
                continue
            left_finals = FeedbackArm2x2.final_solved(runs=arms[left])
            right_finals = FeedbackArm2x2.final_solved(runs=arms[right])
            seeds = sorted(set(left_finals) & set(right_finals))
            differences = [left_finals[seed] - right_finals[seed] for seed in seeds]
            jitter = np.linspace(-0.16, 0.16, len(differences)) if differences else []
            axis.scatter(
                [index + offset for offset in jitter],
                differences,
                color="#6a3d9a",
                alpha=0.75,
                s=34,
                zorder=3,
            )
            if differences:
                axis.hlines(
                    np.median(differences),
                    index - 0.28,
                    index + 0.28,
                    color="#a3243b",
                    linewidth=2.4,
                )
            positions.append(index)
            labels.append(f"{model}\n(n={len(seeds)} seeds)")
        axis.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, fontsize=9)
        axis.set_title("paired per-seed difference\n(in-context MINUS zero-shot)", fontsize=10)
        axis.set_ylabel("difference in tasks solved (x/30 scale)", fontsize=9)
        axis.grid(alpha=0.25, linewidth=0.6, axis="y")
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))

        fig.suptitle(title, fontsize=10.5)
        fig.tight_layout()
        fig.savefig(output, dpi=160)
        print(f"wrote {output}")

    @staticmethod
    def parse_arm(*, value: str) -> tuple[str, Path]:
        label, _, path = value.partition("=")
        if not label or not path:
            raise argparse.ArgumentTypeError(f"expected label=path, got {value!r}")
        return label, Path(path)

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--arm",
            action="append",
            required=True,
            type=lambda value: FeedbackArm2x2.parse_arm(value=value),
            help="label=RESULTS_ROOT, repeatable. RESULTS_ROOT is a run_sweep results root, "
            "so the runs are at RESULTS_ROOT/pure-agent/<seed>/.",
        )
        parser.add_argument(
            "--transcript",
            action="append",
            default=[],
            type=lambda value: FeedbackArm2x2.parse_arm(value=value),
            help="label=DIR, repeatable. DIR holds that arm's authoring transcript.json.",
        )
        parser.add_argument("--output", type=Path, required=True, help="Figure path (.png).")
        parser.add_argument(
            "--title",
            type=str,
            default=(
                "Pure agent on Tossing Room: feedback arm x model, prompt arm held at minimal"
            ),
        )
        args = parser.parse_args()
        arms = {label: FeedbackArm2x2.load_arm(root=root) for label, root in args.arm}
        transcripts = {
            label: json.loads((path / "transcript.json").read_text())
            for label, path in args.transcript
        }
        FeedbackArm2x2.print_report(arms=arms, transcripts=transcripts)
        FeedbackArm2x2.render(arms=arms, output=args.output, title=args.title)


if __name__ == "__main__":
    FeedbackArm2x2.main()

"""Post-run analysis for the Tossing Room EES bring-up: EES learning curves at
several sampler-iteration budgets against the random-skills lower bound, in the
paper's own view (% evaluation tasks solved vs. number of online transitions).

Reads only already-produced output (CLAUDE.md's analysis/ convention -- never runs a
simulation): `<root>/<method>/<seed>/stats.json` as written by `--output-dir`, the
layout `scripts/run_sweep.py` produces.

The skill-oracle upper bound is drawn as a flat reference line rather than read from a
sweep, because it is already pinned by CI rather than measured here:
`tests/environments/tossingroom/test_integration.py` asserts 30/30 on the mixed goal
distribution. Re-deriving it with a sweep would spend compute to reproduce an assertion.

`--arm "label=path"` is repeatable so the sampler-iteration grid (1k / 10k / 100k)
plots as three EES curves on one axis. Every arm must have been run over the same
transition budget, since the x-axis is shared -- that is exactly what makes the
curves comparable, and it is why `PracticeCycleCli` gives EES and random-skills the
same two protocol flags.

Arms share their seed set, so every comparison here is **paired** and is tested as
such -- an unpaired test on a paired design has already produced a wrong p-value in
this project. The test is an exact Wilcoxon signed-rank (all 2^n sign assignments
enumerated), not the normal approximation and not a t-test: n = 10 on a bounded,
ceiling-clipped percentage is neither large nor normal.
"""

import argparse
import itertools
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from hitl_pmp.core.problem.tasks.types import Task  # noqa: E402
from hitl_pmp.environments.tossingroom.environment import (  # noqa: E402
    TossingRoomEnvironment,
)
from hitl_pmp.environments.tossingroom.tasks import TossingRoomTasks  # noqa: E402

# Assigned in fixed order and never cycled, so an arm keeps its colour when another is
# added or dropped. Linestyle repeats the identity as a second channel, so the arms
# stay separable in greyscale and under colour-vision deficiency.
_ARM_STYLES: tuple[tuple[str, str], ...] = (
    ("tab:blue", "-"),
    ("tab:orange", "--"),
    ("tab:green", "-."),
    ("tab:purple", ":"),
)


class TossingRoomComparison:
    """A static-method container, never instantiated."""

    @staticmethod
    def per_seed_curves(*, root: Path, method: str) -> dict[str, dict[int, float]]:
        """seed -> {transitions: % solved}, one entry per stats.json under the root."""
        curves: dict[str, dict[int, float]] = {}
        for stats_path in sorted(root.glob(f"{method}/*/stats.json")):
            seed = stats_path.parent.name
            curves[seed] = {
                int(transitions): 100.0 * num_solved / num_total
                for transitions, num_solved, num_total in json.loads(stats_path.read_text())[
                    "evaluations"
                ]
            }
        return curves

    @staticmethod
    def per_seed_counts(*, root: Path, method: str) -> dict[str, dict[int, tuple[int, int]]]:
        """seed -> {transitions: (num_solved, num_total)}, copied verbatim off the same
        `evaluations` triples `per_seed_curves` divides.

        A success rate belongs in the log as the counts behind it, and those counts are
        recorded rather than reconstructed: `Metrics.record_evaluation` writes
        `num_solved`/`num_total` as the primary record (validating any per-task
        `breakdowns` against them), so nothing here has to multiply a rounded percentage
        by a seed count to get back to what was measured. 23.3% cannot be inverted to
        7/30 without knowing the denominator; 7/30 needs no inverting."""
        counts: dict[str, dict[int, tuple[int, int]]] = {}
        for stats_path in sorted(root.glob(f"{method}/*/stats.json")):
            counts[stats_path.parent.name] = {
                int(transitions): (num_solved, num_total)
                for transitions, num_solved, num_total in json.loads(stats_path.read_text())[
                    "evaluations"
                ]
            }
        return counts

    @staticmethod
    def pooled_endpoints(*, root: Path, method: str) -> dict[str, tuple[int, int]]:
        """`{"first": (solved, total), "final": (solved, total), "worst": ...}` --
        evaluation episodes summed across seeds, not per-seed rates averaged. Every seed
        runs the same number of test tasks here, so the pooled rate and the mean of the
        per-seed rates agree; the count is reported because it is what was measured."""
        counts = TossingRoomComparison.per_seed_counts(root=root, method=method)
        if not counts:
            return {}
        firsts = [curve[min(curve)] for curve in counts.values()]
        finals = [curve[max(curve)] for curve in counts.values()]
        return {
            "first": (sum(s for s, _ in firsts), sum(t for _, t in firsts)),
            "final": (sum(s for s, _ in finals), sum(t for _, t in finals)),
            "worst": min(finals, key=lambda pair: pair[0] / pair[1]),
        }

    @staticmethod
    def mean_curve(*, root: Path, method: str) -> dict[int, tuple[float, float]]:
        """transitions -> (mean %, stderr %) across seeds."""
        by_transitions: dict[int, list[float]] = {}
        for curve in TossingRoomComparison.per_seed_curves(root=root, method=method).values():
            for transitions, percent in curve.items():
                by_transitions.setdefault(transitions, []).append(percent)
        out: dict[int, tuple[float, float]] = {}
        for transitions, values in sorted(by_transitions.items()):
            stderr = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
            out[transitions] = (statistics.mean(values), stderr)
        return out

    @staticmethod
    def summarize(*, root: Path, method: str, threshold: float = 100.0) -> dict[str, float]:
        """The statistics this project reports alongside a mean -- variance and the
        worst seed, because a collapse-to-zero seed has repeatedly been the most
        informative signal here and a mean hides it entirely -- plus how quickly a
        typical seed reaches `threshold`, which is the only thing left to compare once
        the arms saturate at the endpoint."""
        curves = TossingRoomComparison.per_seed_curves(root=root, method=method)
        if not curves:
            return {}
        first = [curve[min(curve)] for curve in curves.values()]
        final = [curve[max(curve)] for curve in curves.values()]
        reached = [
            TossingRoomComparison.transitions_to_reach(curve=curve, threshold=threshold)
            for curve in curves.values()
        ]
        arrived = [value for value in reached if value is not None]
        downward = sum(
            1
            for curve in curves.values()
            for earlier, later in zip(sorted(curve), sorted(curve)[1:], strict=False)
            if curve[later] < curve[earlier]
        )
        return {
            "seeds": len(final),
            "first_mean": statistics.mean(first),
            "final_mean": statistics.mean(final),
            "final_sd": statistics.stdev(final) if len(final) > 1 else 0.0,
            "worst_seed": min(final),
            "seeds_at_zero": sum(1 for value in final if value == 0.0),
            "downward_steps": downward,
            # -1 rather than None so the row formatter stays a plain float format; the
            # companion "never" count is what makes the -1 readable.
            "median_reach": float(statistics.median(arrived)) if arrived else -1.0,
            "never_reached": float(len(reached) - len(arrived)),
        }

    @staticmethod
    def transitions_to_reach(*, curve: dict[int, float], threshold: float) -> int | None:
        """The first transition count at which this seed's curve reaches `threshold`,
        or None if it never does.

        The endpoint alone is a poor statistic on this domain: a run that saturates at
        100% partway through and one that only just gets there both score 100, and the
        interesting difference between sampler budgets is *how fast* the throw force is
        pinned down, not whether it eventually is. Reported alongside the endpoint, not
        instead of it -- a seed that never reaches the threshold shows up as None here
        and has to be reported as such rather than silently dropped."""
        for transitions in sorted(curve):
            if curve[transitions] >= threshold:
                return transitions
        return None

    @staticmethod
    def goal_family(*, task: Task) -> str:
        """Which goal family a drawn test task belongs to, read off the task's own goal
        rather than off the schedule that produced it -- so this stays a statement about
        the object the evaluation actually scores.

        `TossingRoomTasks.build_task` gives EMPTY a two-atom `BinEmpty` goal (both bins)
        and each throw family a single `ItemInBin` atom whose first object is the item.
        That is the whole vocabulary, so the item's name is the family."""
        atoms = sorted(task.goal.atoms, key=lambda atom: atom.predicate.name)
        if len(atoms) != 1:
            return "EMPTY"
        return atoms[0].objects[0].name.upper()

    @staticmethod
    def realised_test_composition(*, num_test_tasks: int, seeds: list[str]) -> dict[str, int]:
        """The goal-family composition of the test set every arm here was scored on,
        per seed, as goal-family name -> count.

        This is a **replication**, not a read-back: `stats.json` records only
        `(transitions, num_solved, num_total)`, with no per-task family, so there is
        nothing in a run's output to read the composition off. What this does instead is
        rebuild a `TossingRoomTasks` at that seed and draw its test tasks exactly as
        `PracticeLoop.run` does -- same class, same seeded stream, `sample_test_task`
        called `num_test_tasks` times -- then classify each `Task` by its own goal. It
        is worth doing anyway, because the composition is the denominator every
        percentage on this page is measured against, and it changed once already (it was
        sampled per seed before the fixed-composition change, so seed 0 got 16 TRASH /
        10 RECYCLING / 4 EMPTY where seed 1 got 11/12/7).

        `num_test_tasks` is the flag the arms were run with, and passing it is
        load-bearing: it is what `TossingRoomTasks` divides between the families, and
        leaving it at the field default while drawing more silently starts a second
        composition block (30 draws against a default of 10 realises 12/12/6).
        """
        composition: dict[str, int] = {}
        for seed in seeds:
            tasks = TossingRoomTasks(
                env=TossingRoomEnvironment(), seed=int(seed), num_test_tasks=num_test_tasks
            )
            counts: dict[str, int] = {}
            for _ in range(num_test_tasks):
                name = TossingRoomComparison.goal_family(task=tasks.sample_test_task())
                counts[name] = counts.get(name, 0) + 1
            if composition and counts != composition:
                raise ValueError(
                    f"test-set composition is not the same on every seed: {composition} "
                    f"vs {counts} at seed {seed}"
                )
            composition = counts
        return composition

    @staticmethod
    def print_test_composition(
        *, num_test_tasks: int, seeds: list[str], arms: list[tuple[str, Path]]
    ) -> None:
        """Print the realised composition, and fail loudly if it is not identical on
        every seed -- a per-seed-varying denominator is exactly the defect the fixed
        composition removed, and it must not come back unnoticed.

        Also cross-checks the replicated composition against the runs actually being
        read: every evaluation in every `stats.json` records its own `num_total`, and if
        that disagrees with the number of tasks this composition divides up then the
        arms were run with a different `--num-test-tasks` than the one passed here, and
        every percentage on the page is being explained by the wrong denominator. That
        is precisely the mismatch that silently gave two `scripts/` probes a 12/12/6 test
        set, so it is checked rather than assumed."""
        if not seeds:
            return
        composition = TossingRoomComparison.realised_test_composition(
            num_test_tasks=num_test_tasks, seeds=seeds
        )
        totals = {
            num_total
            for _, root in arms
            for stats_path in root.glob("*/*/stats.json")
            for _, _, num_total in json.loads(stats_path.read_text())["evaluations"]
        }
        if totals and totals != {num_test_tasks}:
            raise ValueError(
                f"--num-test-tasks {num_test_tasks} does not match the runs being read, "
                f"whose evaluations report num_total in {sorted(totals)}; the composition "
                "below would describe a different test set than the arms were scored on"
            )
        rendered = " / ".join(f"{count} {name}" for name, count in sorted(composition.items()))
        print(
            f"test set: {rendered} on every one of {len(seeds)} seeds "
            f"({num_test_tasks} tasks; replicated from TossingRoomTasks, and checked "
            f"against every run's own num_total)"
        )

    @staticmethod
    def wilcoxon_signed_rank(*, first: list[float], second: list[float]) -> dict:
        """Exact two-sided Wilcoxon signed-rank test on paired samples.

        Exact by full enumeration of all 2^n sign assignments rather than the normal
        approximation, because n here is 10: the approximation is not trustworthy at
        that size, and 2^10 = 1024 is free. Zero differences are dropped (the standard
        Pratt-vs-Wilcoxon choice made the conservative way), so the effective n is
        reported -- with a shared seed set and a saturating metric, ties are common and
        a test quietly run on three pairs must not be read as one run on ten.

        Note the floor this puts on any claim: with n non-zero pairs the smallest
        attainable two-sided p is 2 / 2^n, so n = 10 can reach p = 0.002 but n = 4
        cannot go below 0.125 no matter how large the effect."""
        differences = [a - b for a, b in zip(first, second, strict=True) if a != b]
        num_pairs = len(differences)
        if num_pairs == 0:
            return {"n": 0, "statistic": None, "p": 1.0}
        # Ranks of |d|, averaged over ties -- the standard signed-rank construction.
        order = sorted(range(num_pairs), key=lambda i: abs(differences[i]))
        ranks = [0.0] * num_pairs
        index = 0
        while index < num_pairs:
            stop = index
            while stop + 1 < num_pairs and abs(differences[order[stop + 1]]) == abs(
                differences[order[index]]
            ):
                stop += 1
            shared = (index + stop) / 2 + 1
            for position in range(index, stop + 1):
                ranks[order[position]] = shared
            index = stop + 1
        observed = sum(rank for rank, d in zip(ranks, differences, strict=True) if d > 0)
        total = sum(ranks)
        # Under the null the signs are exchangeable, so enumerate every assignment.
        extreme = 0
        for signs in itertools.product((0, 1), repeat=num_pairs):
            statistic = sum(rank for rank, sign in zip(ranks, signs, strict=True) if sign)
            if abs(statistic - total / 2) >= abs(observed - total / 2):
                extreme += 1
        return {"n": num_pairs, "statistic": observed, "p": extreme / 2**num_pairs}

    @staticmethod
    def print_paired_tests(*, arms: list[tuple[str, Path]], threshold: float) -> None:
        """Every arm pair, on both the endpoint and the transitions-to-threshold speed
        statistic, over the seeds the two arms actually share."""
        curves = {
            label: TossingRoomComparison.per_seed_curves(root=root, method="ees")
            for label, root in arms
        }
        print()
        print(f"paired comparisons (exact Wilcoxon signed-rank; threshold {threshold:.0f}%)")
        header = f"{'pair':<40}{'metric':<26}{'n':>4}{'median diff':>13}{'p':>9}"
        print(header)
        print("-" * len(header))
        for (first_label, _), (second_label, _) in itertools.combinations(arms, 2):
            shared = sorted(set(curves[first_label]) & set(curves[second_label]))
            if not shared:
                continue
            pair = f"{first_label} vs {second_label}"
            finals = [
                [curves[label][seed][max(curves[label][seed])] for seed in shared]
                for label in (first_label, second_label)
            ]
            speeds = [
                [
                    TossingRoomComparison.transitions_to_reach(
                        curve=curves[label][seed], threshold=threshold
                    )
                    for seed in shared
                ]
                for label in (first_label, second_label)
            ]
            TossingRoomComparison._print_test(
                pair=pair, metric="final % solved", first=finals[0], second=finals[1]
            )
            # A seed that never reaches the threshold has no speed to compare, so the
            # pair is dropped and n falls -- reported, never silently imputed.
            paired_speeds = [
                (a, b)
                for a, b in zip(speeds[0], speeds[1], strict=True)
                if a is not None and b is not None
            ]
            never = len(shared) - len(paired_speeds)
            TossingRoomComparison._print_test(
                pair=pair,
                metric=f"transitions to {threshold:.0f}%" + (f" ({never} never)" if never else ""),
                first=[float(a) for a, _ in paired_speeds],
                second=[float(b) for _, b in paired_speeds],
            )

    @staticmethod
    def _print_test(*, pair: str, metric: str, first: list[float], second: list[float]) -> None:
        if not first:
            print(f"{pair:<40}{metric:<26}{'--':>4}{'(no comparable seeds)':>22}")
            return
        result = TossingRoomComparison.wilcoxon_signed_rank(first=first, second=second)
        median_difference = statistics.median(a - b for a, b in zip(first, second, strict=True))
        print(
            f"{pair:<40}{metric:<26}{result['n']:>4}{median_difference:>13.1f}{result['p']:>9.3f}"
        )

    @staticmethod
    def _plot_curve(*, ax, curve: dict[int, tuple[float, float]], label: str, **kwargs) -> None:
        xs = sorted(curve)
        means = [curve[x][0] for x in xs]
        errs = [curve[x][1] for x in xs]
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
    def render(
        *, arms: list[tuple[str, Path]], random_skills_root: Path | None, output: Path, title: str
    ) -> None:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.axhline(
            100.0,
            color="grey",
            linestyle=(0, (2, 3)),
            linewidth=1.4,
            label="skill oracle (100%, pinned by CI)",
        )
        for index, (label, root) in enumerate(arms):
            color, linestyle = _ARM_STYLES[index % len(_ARM_STYLES)]
            TossingRoomComparison._plot_curve(
                ax=ax,
                curve=TossingRoomComparison.mean_curve(root=root, method="ees"),
                label=label,
                color=color,
                linestyle=linestyle,
            )
        if random_skills_root is not None:
            TossingRoomComparison._plot_curve(
                ax=ax,
                curve=TossingRoomComparison.mean_curve(
                    root=random_skills_root, method="random-skills"
                ),
                label="random skills",
                color="tab:red",
                linestyle=(0, (1, 1)),
            )
        ax.set_xlabel("Number of online transitions")
        ax.set_ylabel("% evaluation tasks solved")
        ax.set_title(title, fontsize=11)
        ax.set_ylim(-3, 107)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        fig.savefig(output, dpi=150)

    @staticmethod
    def render_grid(*, arms: list[tuple[str, Path]], output: Path, threshold: float) -> None:
        """The sampler-iteration grid: the curves *and* the per-seed endpoints, in one
        figure, because reporting only one of them misleads in opposite directions.

        The mean curves alone hide that the endpoint is **censored** -- nearly every
        seed is pinned at the oracle's 100%, so the arms cannot differ there no matter
        what the sampler budget does. The endpoints alone hide the shape of the run
        that produced them. Hence three panels: the curves, then the same seed's
        endpoint under each arm joined by a line (a paired design drawn as a paired
        figure), then the only statistic with any headroom left -- how many online
        transitions that seed needed before it first hit the threshold."""
        curves = {
            label: TossingRoomComparison.per_seed_curves(root=root, method="ees")
            for label, root in arms
        }
        labels = [label for label, _ in arms]
        seeds = sorted(set.intersection(*(set(curves[label]) for label in labels)), key=int)
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))

        axes[0].axhline(100.0, color="grey", linestyle=(0, (2, 3)), linewidth=1.4)
        for index, label in enumerate(labels):
            color, linestyle = _ARM_STYLES[index % len(_ARM_STYLES)]
            TossingRoomComparison._plot_curve(
                ax=axes[0],
                curve=TossingRoomComparison.mean_curve(root=dict(arms)[label], method="ees"),
                label=label,
                color=color,
                linestyle=linestyle,
            )
        axes[0].set_xlabel("Number of online transitions")
        axes[0].set_ylabel("% evaluation tasks solved")
        axes[0].set_title("Learning curves (mean ± stderr)", fontsize=10)
        axes[0].legend(loc="lower right", fontsize=8, framealpha=0.9)

        finals = {
            label: [curves[label][seed][max(curves[label][seed])] for seed in seeds]
            for label in labels
        }
        TossingRoomComparison._plot_paired(
            ax=axes[1], labels=labels, per_seed={k: list(v) for k, v in finals.items()}
        )
        axes[1].axhline(100.0, color="grey", linestyle=(0, (2, 3)), linewidth=1.4)
        axes[1].set_ylabel("% evaluation tasks solved (final sweep)")
        axes[1].set_title(f"Final result, per seed (n = {len(seeds)})", fontsize=10)

        speeds = {
            label: [
                TossingRoomComparison.transitions_to_reach(
                    curve=curves[label][seed], threshold=threshold
                )
                for seed in seeds
            ]
            for label in labels
        }
        # A seed that never reached the threshold under *any* arm has no line to draw;
        # dropping it is reported in the axis title rather than imputed to the budget.
        drawable = [
            index
            for index in range(len(seeds))
            if all(speeds[label][index] is not None for label in labels)
        ]
        TossingRoomComparison._plot_paired(
            ax=axes[2],
            labels=labels,
            per_seed={label: [float(speeds[label][i]) for i in drawable] for label in labels},
        )
        axes[2].set_ylabel(f"online transitions to first reach {threshold:.0f}%")
        axes[2].set_title(
            f"Speed to {threshold:.0f}%, per seed ({len(drawable)}/{len(seeds)} in every arm)",
            fontsize=10,
        )

        for ax in axes:
            ax.grid(True, alpha=0.25, linewidth=0.6)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        fig.tight_layout()
        fig.savefig(output, dpi=150)

    @staticmethod
    def _plot_paired(*, ax, labels: list[str], per_seed: dict[str, list[float]]) -> None:
        """One line per seed across the arms, plus each arm's mean as a heavy marker.

        Lines rather than separate box plots because the seeds are shared: the question
        is whether a *given* seed moves when the budget changes, which a marginal
        distribution cannot show. Points are jittered horizontally only, so no vertical
        value is ever displaced."""
        positions = list(range(len(labels)))
        count = len(next(iter(per_seed.values()))) if per_seed else 0
        for index in range(count):
            offsets = [position + (index - (count - 1) / 2) * 0.02 for position in positions]
            ax.plot(
                offsets,
                [per_seed[label][index] for label in labels],
                color="tab:grey",
                alpha=0.55,
                linewidth=1.0,
                marker="o",
                markersize=3.5,
            )
        for position, label in zip(positions, labels, strict=True):
            values = per_seed[label]
            if values:
                ax.plot(
                    [position],
                    [statistics.mean(values)],
                    marker="D",
                    markersize=8,
                    color=_ARM_STYLES[positions.index(position) % len(_ARM_STYLES)][0],
                    zorder=5,
                )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlim(-0.5, len(labels) - 0.5)

    @staticmethod
    def print_table(
        *, arms: list[tuple[str, Path]], random_skills_root: Path | None, threshold: float = 100.0
    ) -> None:
        rows = [(label, root, "ees") for label, root in arms]
        if random_skills_root is not None:
            rows.append(("random skills", random_skills_root, "random-skills"))
        header = (
            f"{'arm':<28}{'seeds':>6}{'first':>10}{'final':>10}"
            f"{'sd':>7}{'worst':>8}{'zeros':>7}{'down':>6}{'reach':>8}{'never':>7}"
        )
        print(header)
        print("-" * len(header))
        for label, root, method in rows:
            summary = TossingRoomComparison.summarize(root=root, method=method, threshold=threshold)
            if not summary:
                print(f"{label:<28}{'(no stats.json found)':>42}")
                continue
            pooled = TossingRoomComparison.pooled_endpoints(root=root, method=method)
            first_count = "{}/{}".format(*pooled["first"])
            final_count = "{}/{}".format(*pooled["final"])
            worst_count = "{}/{}".format(*pooled["worst"])
            print(
                f"{label:<28}{summary['seeds']:>6.0f}"
                f"{first_count:>10}{final_count:>10}"
                f"{summary['final_sd']:>7.1f}{worst_count:>8}"
                f"{summary['seeds_at_zero']:>7.0f}"
                f"{summary['downward_steps']:>6.0f}{summary['median_reach']:>8.0f}"
                f"{summary['never_reached']:>7.0f}"
            )
            first_pct = f"({summary['first_mean']:.1f}%)"
            final_pct = f"({summary['final_mean']:.1f}%)"
            worst_pct = f"({summary['worst_seed']:.1f}%)"
            print(f"{'':<28}{'':>6}{first_pct:>10}{final_pct:>10}{'':>7}{worst_pct:>8}")
        print("  first/final/worst are evaluation episodes solved -- first and final pooled")
        print("  across seeds, worst the single weakest seed. sd is the spread of the")
        print("  per-seed rates in points, not a binomial spread on the pooled count.")
        print(
            f"  reach = median transitions to first reach {threshold:.0f}% "
            "(-1 if no seed ever did); never = seeds that never reached it"
        )


def _parse_arm(*, raw: str) -> tuple[str, Path]:
    label, separator, path = raw.partition("=")
    if not separator or not label:
        raise ValueError(f"--arm must look like label=path, got {raw!r}")
    return label, Path(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        required=True,
        help='Repeatable, "label=results-root". One EES curve per arm.',
    )
    parser.add_argument("--random-skills-root", type=Path, default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=100.0,
        help=(
            "Success rate a seed must reach for the speed statistic. Defaults to 100 "
            "because this domain's arms saturate, which makes the endpoint blind."
        ),
    )
    parser.add_argument(
        "--num-test-tasks",
        type=int,
        default=30,
        help=(
            "The --num-test-tasks the arms were run with. Only used to report (and "
            "check) the realised goal-family composition of the test set every "
            "percentage here is measured against; see realised_test_composition."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--grid-output",
        type=Path,
        default=None,
        help=(
            "If given, also render the sampler-iteration grid figure there: the EES "
            "arms' curves plus their per-seed endpoints and speeds, paired."
        ),
    )
    parser.add_argument("--title", default="Tossing Room: EES vs. the random-skills lower bound")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    arms = [_parse_arm(raw=raw) for raw in args.arm]
    seeds: set[str] = set()
    for _, root in arms:
        seeds |= set(TossingRoomComparison.per_seed_curves(root=root, method="ees"))
    all_arms = arms + (
        [("random skills", args.random_skills_root)] if args.random_skills_root else []
    )
    TossingRoomComparison.print_test_composition(
        num_test_tasks=args.num_test_tasks, seeds=sorted(seeds), arms=all_arms
    )
    TossingRoomComparison.print_table(
        arms=arms, random_skills_root=args.random_skills_root, threshold=args.threshold
    )
    TossingRoomComparison.print_paired_tests(arms=arms, threshold=args.threshold)
    TossingRoomComparison.render(
        arms=arms,
        random_skills_root=args.random_skills_root,
        output=args.output,
        title=args.title,
    )
    if args.grid_output is not None:
        TossingRoomComparison.render_grid(
            arms=arms, output=args.grid_output, threshold=args.threshold
        )


if __name__ == "__main__":
    main()

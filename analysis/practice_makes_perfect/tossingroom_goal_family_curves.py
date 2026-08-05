"""Post-run analysis for Tossing Room: learning curves **split by goal family**, plus
the binomial noise floor and minimum detectable effect for whatever comparison the
resulting counts are used to draw.

Reads only already-produced output (CLAUDE.md's `analysis/` convention -- never runs a
simulation): `<root>/<method>/<seed>/stats.json` as written by `--output-dir`, the
layout `scripts/run_sweep.py` produces.

**Why a second script rather than a flag on `tossingroom_comparison.py`.** That one
answers "how do these arms compare", and it reads the pooled
`(transitions, num_solved, num_total)` triples, which carry no family information at
all. Since #74 the interesting question about this domain is no longer only how many
tasks are solved but *which*: `EMPTY` stopped being a free four-action walk and became
an **ordering** task -- the recycling button sits behind the one-way ledge, so it must
be pressed last, and pressing it early strands the robot with no `Throw` involved. A
pooled curve cannot show that, because `EMPTY` is 2 of 30 tasks and its failure is
worth at most 6.7 points of a number dominated by 28 throw tasks.

**Family comes from the record, not from a replication.** `tossingroom_comparison.py`
has to rebuild a `TossingRoomTasks` and re-draw the test set to recover the
composition, because at the time it was written `stats.json` held no per-task
information. It now holds `breakdowns`: one entry per evaluation sweep, each listing
every test task's index, goal string and solved flag, validated by
`Metrics.record_evaluation` against exactly the `num_solved`/`num_total` it also
records. So this reads the families off the same object that was scored, and a run
recorded without breakdowns is refused rather than approximated -- see
`pooled_family_counts`.

Counts, never bare percentages: a rate is always rendered `x/y (p%)` by
`format_count`, including on the figure's axis labels and annotations. `EMPTY 100%` is
really 2/2 on one seed and 20/20 pooled, and the difference between those two claims is
the entire reason the denominator is written down.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# Fixed per family and never cycled, so a family keeps its colour when the figure is
# regenerated with a different set. Linestyle repeats the identity as a second channel,
# so the panels stay separable in greyscale and under colour-vision deficiency.
_FAMILY_STYLES: dict[str, tuple[str, str]] = {
    "TRASH": ("tab:blue", "-"),
    "RECYCLING": ("tab:orange", "--"),
    "EMPTY": ("tab:green", "-."),
    "POOLED": ("black", "-"),
}

# z(0.975) + z(0.80): the standard-error multiple an 80%-power, two-sided 5% test needs.
_MDE_MULTIPLE = 2.8


class TossingRoomGoalFamilyCurves:
    """A static-method container, never instantiated."""

    @staticmethod
    def family_of(*, goal: str) -> str:
        """Which goal family a recorded goal string belongs to.

        `TossingRoomTasks.build_task` writes exactly three shapes, and this is the whole
        vocabulary: a throw family is one `ItemInBin(<item>, <bin>)` atom whose first
        object names the item, and `EMPTY` is the conjunction of one `BinEmpty` per bin
        (two of them since #74 gave each bin its own button, where it used to be a
        single atom).

        Anything else raises. Bucketing an unrecognised goal into a default would move
        tasks between denominators invisibly, which on a 14/14/2 composition is a
        finding-sized error; a domain change that adds a family should break this
        loudly."""
        if "BinEmpty" in goal:
            return "EMPTY"
        if goal.startswith("ItemInBin(") and ", " in goal:
            return goal[len("ItemInBin(") :].split(",")[0].strip().upper()
        raise ValueError(
            f"unrecognised goal {goal!r}: expected an ItemInBin throw goal or a BinEmpty "
            "conjunction. If the domain gained a goal family, teach this function about "
            "it rather than letting it fall into another family's denominator."
        )

    @staticmethod
    def per_seed_family_counts(
        *, root: Path, method: str
    ) -> dict[str, dict[str, dict[int, tuple[int, int]]]]:
        """seed -> family -> {transitions: (num_solved, num_total)}.

        Kept per seed rather than reduced, because with ten seeds a pooled mean hides
        one seed driving the whole effect -- which is exactly what happened to this
        domain's sampler-iteration grid, where four of ten seeds failed catastrophically
        behind a mean that read as a mild difference."""
        out: dict[str, dict[str, dict[int, tuple[int, int]]]] = {}
        for stats_path in sorted(root.glob(f"{method}/*/stats.json")):
            stats = json.loads(stats_path.read_text())
            breakdowns = stats.get("breakdowns") or []
            if not breakdowns:
                raise ValueError(
                    f"{stats_path} has no per-task breakdowns, so it carries no goal-family "
                    "information. Falling back to its pooled `evaluations` triples would "
                    "produce a curve that looks right and answers a different question."
                )
            per_family: dict[str, dict[int, tuple[int, int]]] = {}
            for breakdown in breakdowns:
                transitions = int(breakdown["num_online_transitions"])
                for outcome in breakdown["outcomes"]:
                    family = TossingRoomGoalFamilyCurves.family_of(goal=outcome["goal"])
                    solved, total = per_family.setdefault(family, {}).get(transitions, (0, 0))
                    per_family[family][transitions] = (
                        solved + (1 if outcome["solved"] else 0),
                        total + 1,
                    )
            out[stats_path.parent.name] = per_family
        return out

    @staticmethod
    def pooled_family_counts(*, root: Path, method: str) -> dict[str, dict[int, tuple[int, int]]]:
        """family -> {transitions: (num_solved, num_total)}, episodes summed across
        seeds rather than per-seed rates averaged.

        Summing episodes is the honest pooling here: every seed runs the same test set
        size, so it agrees with the mean of the rates, but it is what was measured and
        it stays a count."""
        pooled: dict[str, dict[int, tuple[int, int]]] = {}
        per_seed = TossingRoomGoalFamilyCurves.per_seed_family_counts(root=root, method=method)
        for families in per_seed.values():
            for family, curve in families.items():
                for transitions, (solved, total) in curve.items():
                    have_solved, have_total = pooled.setdefault(family, {}).get(transitions, (0, 0))
                    pooled[family][transitions] = (have_solved + solved, have_total + total)
        return pooled

    @staticmethod
    def pooled_overall_counts(*, root: Path, method: str) -> dict[int, tuple[int, int]]:
        """{transitions: (num_solved, num_total)} over every family at once -- the
        headline curve, summed from the same per-task records the family split uses so
        the two cannot drift apart."""
        overall: dict[int, tuple[int, int]] = {}
        for curve in TossingRoomGoalFamilyCurves.pooled_family_counts(
            root=root, method=method
        ).values():
            for transitions, (solved, total) in curve.items():
                have_solved, have_total = overall.get(transitions, (0, 0))
                overall[transitions] = (have_solved + solved, have_total + total)
        return overall

    @staticmethod
    def binomial_noise_floor(*, n_a: int, n_b: int) -> float:
        """`sqrt(0.25/n_a + 0.25/n_b)`, in percentage points -- the standard error of a
        difference of two proportions at the worst case p = 0.5.

        Worst case on purpose: `p(1-p)` is maximised at 0.5, so this is the largest the
        standard error can be at these sample sizes and any comparison smaller than a
        couple of these is noise whatever the point estimate says. It is quoted per
        comparison rather than once, because the family denominators differ by more than
        an order of magnitude -- 280 throw episodes against 20 `EMPTY` ones."""
        return 100.0 * math.sqrt(0.25 / n_a + 0.25 / n_b)

    @staticmethod
    def minimum_detectable_effect(*, n_a: int, n_b: int) -> float:
        """The smallest difference this design could detect at 80% power, two-sided 5%:
        `2.8` standard errors, since `z(0.975) + z(0.80) = 1.96 + 0.84`.

        Reported alongside every comparison so a null result can be read correctly. A
        design whose MDE is larger than the effect being looked for cannot resolve it,
        and saying so is not the same as saying there is no effect."""
        return _MDE_MULTIPLE * TossingRoomGoalFamilyCurves.binomial_noise_floor(n_a=n_a, n_b=n_b)

    @staticmethod
    def format_count(*, solved: int, total: int) -> str:
        """`x/y (p%)` -- the denominator first and always. A bare percentage hides how
        many episodes it summarises, and on this composition that is the difference
        between a result and an anecdote."""
        percent = 100.0 * solved / total if total else 0.0
        return f"{solved}/{total} ({percent:.1f}%)"

    @staticmethod
    def as_json(*, root: Path, method: str) -> dict:
        """Every count this script can produce, in one JSON-safe structure, so an
        experiment log's tables re-derive from a file committed beside it rather than
        from a transcription of a terminal.

        Counts only -- `[solved, total]` -- never rates: a committed percentage cannot
        be inverted to the count behind it without knowing a denominator that is exactly
        what changed last time this domain moved. Transition keys are strings because
        JSON object keys are, and pretending otherwise would break the round trip."""
        pooled = TossingRoomGoalFamilyCurves.pooled_family_counts(root=root, method=method)
        overall = TossingRoomGoalFamilyCurves.pooled_overall_counts(root=root, method=method)
        per_seed = TossingRoomGoalFamilyCurves.per_seed_family_counts(root=root, method=method)
        return {
            "pooled": {
                family: {str(t): list(counts) for t, counts in sorted(curve.items())}
                for family, curve in pooled.items()
            },
            "overall": {str(t): list(counts) for t, counts in sorted(overall.items())},
            "per_seed": {
                seed: {
                    family: {str(t): list(counts) for t, counts in sorted(curve.items())}
                    for family, curve in families.items()
                }
                for seed, families in per_seed.items()
            },
        }

    @staticmethod
    def count_ticks(*, total: int) -> tuple[list[float], list[str]]:
        """Y-axis tick positions (in percent, which is what the families can share) and
        their labels (as `x/total`, which is what a reader must actually see).

        The panels genuinely have to share a percentage scale -- EMPTY pools to 20
        episodes against TRASH's 140 -- but a bare `100` on the axis is the
        denominator-hiding this project's logs forbid, and it is precisely the axis on
        which "EMPTY 100%" would be read as a strong result rather than as 20/20.
        Counts are rounded to whole episodes; a `3.5/14` tick would be a fraction of an
        episode, which does not exist."""
        positions = [0.0, 25.0, 50.0, 75.0, 100.0]
        return positions, [f"{round(total * p / 100.0)}/{total}" for p in positions]

    @staticmethod
    def print_summary(*, root: Path, method: str, label: str) -> None:
        """Everything the log quotes, as counts, so the tables re-derive from run output
        rather than from a transcription."""
        pooled = TossingRoomGoalFamilyCurves.pooled_family_counts(root=root, method=method)
        overall = TossingRoomGoalFamilyCurves.pooled_overall_counts(root=root, method=method)
        per_seed = TossingRoomGoalFamilyCurves.per_seed_family_counts(root=root, method=method)
        checkpoints = sorted(overall)
        first, final = checkpoints[0], checkpoints[-1]
        print()
        print(f"=== {label} ({len(per_seed)} seeds, {len(checkpoints)} evaluation sweeps) ===")
        header = f"{'family':<12}{'pre-practice':>22}{'final':>22}{'noise floor':>14}{'MDE':>9}"
        print(header)
        print("-" * len(header))
        for family in ("TRASH", "RECYCLING", "EMPTY"):
            if family not in pooled:
                continue
            start, end = pooled[family][first], pooled[family][final]
            floor = TossingRoomGoalFamilyCurves.binomial_noise_floor(n_a=start[1], n_b=end[1])
            mde = TossingRoomGoalFamilyCurves.minimum_detectable_effect(n_a=start[1], n_b=end[1])
            print(
                f"{family:<12}"
                f"{TossingRoomGoalFamilyCurves.format_count(solved=start[0], total=start[1]):>22}"
                f"{TossingRoomGoalFamilyCurves.format_count(solved=end[0], total=end[1]):>22}"
                f"{floor:>13.2f}p{mde:>8.2f}p"
            )
        start, end = overall[first], overall[final]
        floor = TossingRoomGoalFamilyCurves.binomial_noise_floor(n_a=start[1], n_b=end[1])
        mde = TossingRoomGoalFamilyCurves.minimum_detectable_effect(n_a=start[1], n_b=end[1])
        print(
            f"{'POOLED':<12}"
            f"{TossingRoomGoalFamilyCurves.format_count(solved=start[0], total=start[1]):>22}"
            f"{TossingRoomGoalFamilyCurves.format_count(solved=end[0], total=end[1]):>22}"
            f"{floor:>13.2f}p{mde:>8.2f}p"
        )
        print()
        print("per-seed final solved, by family (a mean over ten seeds hides one seed)")
        for seed in sorted(per_seed, key=int):
            parts = []
            for family in ("TRASH", "RECYCLING", "EMPTY"):
                curve = per_seed[seed].get(family)
                if curve is None:
                    continue
                solved, total = curve[max(curve)]
                parts.append(f"{family[:4]} {solved}/{total}")
            seed_total = sum(
                per_seed[seed][family][max(per_seed[seed][family])][0] for family in per_seed[seed]
            )
            seed_denominator = sum(
                per_seed[seed][family][max(per_seed[seed][family])][1] for family in per_seed[seed]
            )
            print(f"  seed {seed:<3}{'  '.join(parts)}   all {seed_total}/{seed_denominator}")
        finals = [
            100.0
            * sum(per_seed[s][f][max(per_seed[s][f])][0] for f in per_seed[s])
            / sum(per_seed[s][f][max(per_seed[s][f])][1] for f in per_seed[s])
            for s in per_seed
        ]
        if len(finals) > 1:
            print(
                f"  spread of the ten per-seed final rates: sd {statistics.stdev(finals):.1f} "
                f"points, worst {min(finals):.1f}%, best {max(finals):.1f}%"
            )

    @staticmethod
    def _plot_family(
        *,
        ax,
        family: str,
        pooled: dict[int, tuple[int, int]],
        per_seed: list[dict[int, tuple[int, int]]],
        floor: dict[int, tuple[int, int]] | None = None,
    ) -> None:
        color, linestyle = _FAMILY_STYLES.get(family, ("tab:grey", "-"))
        # Every seed as a thin line first: the spread is the point, and a mean drawn
        # alone over ten seeds has repeatedly been the misleading view on this domain.
        for curve in per_seed:
            xs = sorted(curve)
            ax.plot(
                xs,
                [100.0 * curve[x][0] / curve[x][1] for x in xs],
                color=color,
                alpha=0.28,
                linewidth=0.9,
            )
        xs = sorted(pooled)
        ax.plot(
            xs,
            [100.0 * pooled[x][0] / pooled[x][1] for x in xs],
            color=color,
            linestyle=linestyle,
            linewidth=2.4,
        )
        if floor:
            floor_xs = sorted(floor)
            ax.plot(
                floor_xs,
                [100.0 * floor[x][0] / floor[x][1] for x in floor_xs],
                color="tab:red",
                linestyle=(0, (2, 3)),
                linewidth=1.3,
            )
        first, final = pooled[xs[0]], pooled[xs[-1]]
        ax.set_title(
            f"{family}: {TossingRoomGoalFamilyCurves.format_count(solved=first[0], total=first[1])}"
            f" → {TossingRoomGoalFamilyCurves.format_count(solved=final[0], total=final[1])}",
            fontsize=10,
        )
        ax.set_ylim(-4, 104)
        ax.grid(alpha=0.25, linewidth=0.6)
        # Counts on the y axis, never a bare percentage: the denominator is per family
        # and differs more than tenfold between them, so the ticks have to carry it.
        positions, labels = TossingRoomGoalFamilyCurves.count_ticks(total=final[1])
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=8)
        per_seed_denominator = final[1] // len(per_seed) if per_seed else final[1]
        ax.set_ylabel(
            f"solved — bold: x/{final[1]} pooled;  thin: one seed, x/{per_seed_denominator}",
            fontsize=8,
        )

    @staticmethod
    def _per_seed_overall(
        *, per_seed: dict[str, dict[str, dict[int, tuple[int, int]]]]
    ) -> list[dict[int, tuple[int, int]]]:
        """Each seed's own all-families curve, summed from that seed's per-family
        records so the thin lines on the pooled panel are the same episodes as the
        family panels' -- not a separately-read statistic that could drift from them."""
        return [
            {
                transitions: (
                    sum(seed_families[f][transitions][0] for f in seed_families),
                    sum(seed_families[f][transitions][1] for f in seed_families),
                )
                for transitions in sorted(next(iter(seed_families.values())))
            }
            for seed_families in per_seed.values()
        ]

    @staticmethod
    def render(
        *, root: Path, method: str, output: Path, title: str, floor_root: Path | None = None
    ) -> None:
        pooled_families = TossingRoomGoalFamilyCurves.pooled_family_counts(root=root, method=method)
        overall = TossingRoomGoalFamilyCurves.pooled_overall_counts(root=root, method=method)
        per_seed = TossingRoomGoalFamilyCurves.per_seed_family_counts(root=root, method=method)
        floor_families: dict[str, dict[int, tuple[int, int]]] = {}
        floor_overall: dict[int, tuple[int, int]] = {}
        if floor_root is not None:
            floor_families = TossingRoomGoalFamilyCurves.pooled_family_counts(
                root=floor_root, method="random-skills"
            )
            floor_overall = TossingRoomGoalFamilyCurves.pooled_overall_counts(
                root=floor_root, method="random-skills"
            )
        families = [f for f in ("TRASH", "RECYCLING", "EMPTY") if f in pooled_families]
        fig, axes = plt.subplots(1, len(families) + 1, figsize=(4.0 * (len(families) + 1), 4.4))
        for ax, family in zip(axes, families, strict=False):
            TossingRoomGoalFamilyCurves._plot_family(
                ax=ax,
                family=family,
                pooled=pooled_families[family],
                per_seed=[s[family] for s in per_seed.values() if family in s],
                floor=floor_families.get(family),
            )
        TossingRoomGoalFamilyCurves._plot_family(
            ax=axes[-1],
            family="POOLED",
            pooled=overall,
            per_seed=TossingRoomGoalFamilyCurves._per_seed_overall(per_seed=per_seed),
            floor=floor_overall or None,
        )
        for ax in axes:
            ax.set_xlabel("online transitions")
        if floor_root is not None:
            axes[-1].plot([], [], color="tab:red", linestyle=(0, (2, 3)), label="random skills")
            axes[-1].legend(fontsize=8, loc="lower right")
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        fig.savefig(output, dpi=160)
        print(f"wrote {output}")

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--results-root", type=Path, required=True)
        parser.add_argument("--method", type=str, default="ees")
        parser.add_argument("--label", type=str, default="EES")
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--title", type=str, default="Tossing Room, EES, by goal family")
        parser.add_argument(
            "--floor-root",
            type=Path,
            default=None,
            help="A random-skills results root, drawn as a dashed reference in every "
            "panel. Optional: the floor is a different arm, not part of this one.",
        )
        parser.add_argument(
            "--dump-json",
            type=Path,
            default=None,
            help="Write every count to this file, so an experiment log's tables "
            "re-derive from a committed record rather than from a transcription.",
        )
        args = parser.parse_args()
        if args.dump_json is not None:
            payload = {
                args.method: TossingRoomGoalFamilyCurves.as_json(
                    root=args.results_root, method=args.method
                )
            }
            if args.floor_root is not None:
                payload["random-skills"] = TossingRoomGoalFamilyCurves.as_json(
                    root=args.floor_root, method="random-skills"
                )
            args.dump_json.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
            print(f"wrote {args.dump_json}")
        TossingRoomGoalFamilyCurves.print_summary(
            root=args.results_root, method=args.method, label=args.label
        )
        if args.floor_root is not None:
            TossingRoomGoalFamilyCurves.print_summary(
                root=args.floor_root, method="random-skills", label="random skills (floor)"
            )
        TossingRoomGoalFamilyCurves.render(
            root=args.results_root,
            method=args.method,
            output=args.output,
            title=args.title,
            floor_root=args.floor_root,
        )


if __name__ == "__main__":
    TossingRoomGoalFamilyCurves.main()

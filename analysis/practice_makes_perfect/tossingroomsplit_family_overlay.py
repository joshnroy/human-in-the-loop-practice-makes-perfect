"""Post-run analysis for Tossing Room's split throws: **one axes** carrying the `TRASH`
and `RECYCLING` learning curves together, against online transitions, with every seed
drawn under the pooled line.

Reads only already-produced output (CLAUDE.md's `analysis/` convention -- never runs a
simulation): `<root>/<method>/<seed>/stats.json` as written by `--output-dir`, the
layout `scripts/run_sweep.py` produces.

**Why a single axes rather than the panel-per-family view that already exists.**
`tossingroom_goal_family_curves.py` renders one panel per family plus a pooled one,
which answers "what does each family do". The question here is different and narrower:
*how far apart are the two throw families, and when does that gap close* -- and a gap
read across two panels is reconstructed one number at a time, which is the work a figure
exists to remove. The two families share a denominator exactly (14 test tasks each per
seed, 140 pooled over ten), so they can honestly share a y axis; `shared_total` refuses
to draw if that ever stops being true.

Every count comes from `TossingRoomGoalFamilyCurves`, not from a second reader. That is
deliberate: the two figures must never disagree about what a run scored, and the way to
guarantee it is to have one implementation of "read the breakdowns and bucket them by
family" rather than two that agree today.

What is new here is only what sits on top of those counts -- the threshold crossings
("recycling reaches this level later"), the transition-weighted AUC, and the seed-aligned
endpoint pairing that a paired test needs. The families are measured *inside the same
runs*, so they are paired data and an unpaired test would throw that structure away.

Counts, never bare percentages: a rate is always rendered `x/y (p%)` by `format_count`,
including on the figure's axis labels, its legend and its annotations.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from analysis.practice_makes_perfect.tossingroom_goal_family_curves import (  # noqa: E402
    TossingRoomGoalFamilyCurves,
)
from analysis.practice_makes_perfect.tossingroom_reset_interval import PairedTests  # noqa: E402

# The two throw families this figure contrasts, in the order they are drawn. EMPTY is
# deliberately absent: it contains no throw, so neither sampler can touch it, and adding
# a flat line at the top would only invite it to be read as a third arm.
_FAMILIES = ("TRASH", "RECYCLING")

# Fixed per family and shared with tossingroom_goal_family_curves.py's own styling, so a
# family keeps its colour across every figure in this project. Linestyle repeats the
# identity as a second channel, so the two stay separable in greyscale and under
# colour-vision deficiency.
_STYLES: dict[str, tuple[str, str]] = {
    "TRASH": ("tab:blue", "-"),
    "RECYCLING": ("tab:orange", "--"),
}

# The levels the threshold table reports, as fractions of the shared denominator. These
# are the same four the 250-cycle run tabulated, so the two pages' tables have the same
# shape -- which is not the same as their numbers being comparable, and the log says so.
_THRESHOLD_FRACTIONS = (0.25, 0.5, 0.75, 0.9)


class TossingRoomSplitFamilyOverlay:
    """A static-method container, never instantiated."""

    @staticmethod
    def format_count(*, solved: int, total: int) -> str:
        """`x/y (p%)` -- the denominator first and always. Delegates to the family-curves
        implementation so the two figures render a count identically."""
        return TossingRoomGoalFamilyCurves.format_count(solved=solved, total=total)

    @staticmethod
    def shared_total(*, root: Path, method: str) -> int:
        """The denominator both families are measured over, or a raised error if they
        differ.

        The whole licence for putting two curves on one y axis is that a point at the
        same height means the same thing for both. Tossing Room's fixed test set is 14
        TRASH and 14 RECYCLING, so that holds by construction -- but it is checked rather
        than assumed, because a composition change would otherwise produce a figure whose
        two curves quietly mean different things."""
        pooled = TossingRoomGoalFamilyCurves.pooled_family_counts(root=root, method=method)
        totals = {}
        for family in _FAMILIES:
            if family not in pooled:
                raise ValueError(
                    f"no {family} tasks in {root}/{method}: this figure contrasts the two "
                    "throw families and cannot be drawn from a run that measured only one."
                )
            curve = pooled[family]
            totals[family] = curve[max(curve)][1]
        if len(set(totals.values())) != 1:
            raise ValueError(
                f"the two throw families have a different denominator ({totals}), so they "
                "cannot share a y axis: the same height would mean a different number of "
                "episodes for each. Draw them as separate panels instead."
            )
        return next(iter(totals.values()))

    @staticmethod
    def first_transitions_at_or_above(
        *, curve: dict[int, tuple[int, int]], solved: int
    ) -> int | None:
        """The earliest checkpoint whose solved count is at least `solved`, or `None` if
        the curve never gets there.

        `>=`, not `>`: "reaches 35/140" includes landing exactly on 35, and the off-by-one
        would report the crossing a whole practice period late. `None` rather than the
        final checkpoint, because "did not get there" and "got there at the end" are
        opposite findings -- and at a 60-cycle budget the top of the recycling curve is
        exactly the region that may not be reached."""
        for transitions in sorted(curve):
            if curve[transitions][0] >= solved:
                return transitions
        return None

    @staticmethod
    def mean_solved_rate(*, curve: dict[int, tuple[int, int]]) -> float:
        """Area under the solved-rate curve divided by the transition span -- a mean rate
        over the run, in [0, 1].

        Trapezoidal over *transitions*, not a mean of the checkpoint rates: checkpoints
        are only equally spaced while nothing thins them, and a mean of rates would
        silently reweight the curve if they ever were. This is the statistic behind
        "recycling still gets there later" once the endpoints have converged, which is
        the only task-level residue of the asymmetry that survives a large budget."""
        points = sorted(curve)
        if len(points) == 1:
            solved, total = curve[points[0]]
            return solved / total
        area = 0.0
        for left, right in zip(points, points[1:], strict=False):
            left_rate = curve[left][0] / curve[left][1]
            right_rate = curve[right][0] / curve[right][1]
            area += (right - left) * (left_rate + right_rate) / 2.0
        return area / (points[-1] - points[0])

    @staticmethod
    def paired_final_counts(
        *, root: Path, method: str
    ) -> list[tuple[str, tuple[int, int], tuple[int, int]]]:
        """`(seed, trash_final, recycling_final)` per seed, in numeric seed order.

        Seed-aligned on purpose. The two families are scored inside the *same* run, so
        they are paired data: seed 0 solving TRASH but not RECYCLING and seed 1 the
        reverse averages to no difference while having real per-seed spread, and only the
        pairing can tell that from both seeds sitting at the mean. A seed missing either
        family raises rather than being dropped, since dropping it would change a paired
        test's denominator without saying so."""
        per_seed = TossingRoomGoalFamilyCurves.per_seed_family_counts(root=root, method=method)
        out = []
        for seed in sorted(per_seed, key=int):
            families = per_seed[seed]
            for family in _FAMILIES:
                if family not in families:
                    raise ValueError(
                        f"seed {seed} has no {family} tasks, so it cannot be paired. A "
                        "paired comparison needs both families in every seed."
                    )
            finals = tuple(families[family][max(families[family])] for family in _FAMILIES)
            out.append((seed, finals[0], finals[1]))
        return out

    @staticmethod
    def thresholds(*, root: Path, method: str) -> dict[str, dict[str, int | None]]:
        """family -> level (as a solved count, stringified) -> the transition count at
        which the pooled curve first reached it.

        Keys are strings so the structure survives `json.dumps`, which integer keys
        silently would not."""
        pooled = TossingRoomGoalFamilyCurves.pooled_family_counts(root=root, method=method)
        total = TossingRoomSplitFamilyOverlay.shared_total(root=root, method=method)
        levels = sorted({max(1, round(total * fraction)) for fraction in _THRESHOLD_FRACTIONS})
        return {
            family: {
                str(level): TossingRoomSplitFamilyOverlay.first_transitions_at_or_above(
                    curve=pooled[family], solved=level
                )
                for level in levels
            }
            for family in _FAMILIES
        }

    @staticmethod
    def as_json(*, root: Path, method: str) -> dict:
        """Every count this script can produce, in one JSON-safe structure, so the
        experiment log's tables re-derive from a file committed beside it rather than
        from a transcription of a terminal.

        Counts only -- `[solved, total]` -- never rates: a committed percentage cannot be
        inverted to the count behind it without knowing a denominator, and the denominator
        is exactly what changes between budgets."""
        pooled = TossingRoomGoalFamilyCurves.pooled_family_counts(root=root, method=method)
        per_seed = TossingRoomGoalFamilyCurves.per_seed_family_counts(root=root, method=method)
        return {
            "pooled": {
                family: {str(t): list(pooled[family][t]) for t in sorted(pooled[family])}
                for family in _FAMILIES
                if family in pooled
            },
            "per_seed": {
                seed: {
                    family: {str(t): list(curve[t]) for t in sorted(curve)}
                    for family, curve in families.items()
                    if family in _FAMILIES
                }
                for seed, families in per_seed.items()
            },
            "thresholds": TossingRoomSplitFamilyOverlay.thresholds(root=root, method=method),
        }

    @staticmethod
    def print_report(*, root: Path, method: str) -> None:
        """The counts and the paired tests behind them, printed so the log quotes a
        derivation rather than an impression.

        No effect is asserted without a p-value, and the test is paired because the arms
        share seeds."""
        pooled = TossingRoomGoalFamilyCurves.pooled_family_counts(root=root, method=method)
        total = TossingRoomSplitFamilyOverlay.shared_total(root=root, method=method)
        paired = TossingRoomSplitFamilyOverlay.paired_final_counts(root=root, method=method)
        per_seed = TossingRoomGoalFamilyCurves.per_seed_family_counts(root=root, method=method)
        checkpoints = sorted(pooled[_FAMILIES[0]])
        print(f"=== {method}: {len(paired)} seeds, {len(checkpoints)} evaluation sweeps, ")
        print(f"    {checkpoints[0]} to {checkpoints[-1]} online transitions ===")
        for family in _FAMILIES:
            curve = pooled[family]
            first, final = curve[checkpoints[0]], curve[checkpoints[-1]]
            started = TossingRoomSplitFamilyOverlay.format_count(solved=first[0], total=first[1])
            ended = TossingRoomSplitFamilyOverlay.format_count(solved=final[0], total=final[1])
            auc = 100.0 * TossingRoomSplitFamilyOverlay.mean_solved_rate(curve=curve)
            print(f"  {family:<10} {started} -> {ended}   AUC {auc:.2f}pp")

        print("\nper-seed final solved (a mean over ten seeds hides one seed):")
        for seed, trash, recycling in paired:
            print(
                f"  seed {seed:>2}  TRASH {trash[0]}/{trash[1]}"
                f"   RECYCLING {recycling[0]}/{recycling[1]}"
            )

        endpoint_differences = [float(t[0] - r[0]) for _, t, r in paired]
        wilcoxon = PairedTests.wilcoxon_signed_rank(differences=endpoint_differences)
        num_equal = sum(1 for d in endpoint_differences if d == 0.0)
        print(
            f"\nendpoint TRASH - RECYCLING, paired over {len(paired)} seeds: "
            f"Wilcoxon p={wilcoxon.p_value:.4f} ({num_equal}/{len(paired)} seeds tied)"
        )

        auc_differences = [
            100.0
            * (
                TossingRoomSplitFamilyOverlay.mean_solved_rate(curve=per_seed[seed]["TRASH"])
                - TossingRoomSplitFamilyOverlay.mean_solved_rate(curve=per_seed[seed]["RECYCLING"])
            )
            for seed, _, _ in paired
        ]
        auc_test = PairedTests.wilcoxon_signed_rank(differences=auc_differences)
        print(
            f"AUC TRASH - RECYCLING, paired over {len(paired)} seeds: "
            f"mean {sum(auc_differences) / len(auc_differences):+.2f}pp, "
            f"Wilcoxon p={auc_test.p_value:.4f}"
        )

        print(f"\nfirst reaching a level, pooled over {len(paired)} seeds (x/{total}):")
        crossings = TossingRoomSplitFamilyOverlay.thresholds(root=root, method=method)
        for level in sorted(crossings[_FAMILIES[0]], key=int):
            cells = []
            for family in _FAMILIES:
                at = crossings[family][level]
                # "never" rather than the final checkpoint: at this budget the top of
                # recycling's curve is exactly the region that may not be reached, and
                # naming it as reached would be the one error worth guarding here.
                cells.append(f"{family} {'never' if at is None else at}")
            print(f"  {level}/{total}:  {'  '.join(cells)}")

    @staticmethod
    def render(*, root: Path, method: str, output: Path, title: str):
        """One axes, both families, every seed drawn under the pooled line."""
        pooled = TossingRoomGoalFamilyCurves.pooled_family_counts(root=root, method=method)
        per_seed = TossingRoomGoalFamilyCurves.per_seed_family_counts(root=root, method=method)
        total = TossingRoomSplitFamilyOverlay.shared_total(root=root, method=method)

        fig, ax = plt.subplots(1, 1, figsize=(9.0, 5.4))
        seed_total = 0
        for family in _FAMILIES:
            color, linestyle = _STYLES[family]
            # Every seed as a thin line first: the spread is the point, and on this
            # domain a mean drawn alone has repeatedly been the misleading view -- the
            # crossover is exactly where one seed can drive the whole picture.
            for families in per_seed.values():
                curve = families[family]
                xs = sorted(curve)
                seed_total = curve[xs[-1]][1]
                ax.plot(
                    xs,
                    [100.0 * curve[x][0] / curve[x][1] for x in xs],
                    color=color,
                    alpha=0.28,
                    linewidth=0.9,
                )
            curve = pooled[family]
            xs = sorted(curve)
            final = curve[xs[-1]]
            ax.plot(
                xs,
                [100.0 * curve[x][0] / curve[x][1] for x in xs],
                color=color,
                linestyle=linestyle,
                linewidth=2.4,
                label=(
                    f"{family} — pooled "
                    f"{TossingRoomSplitFamilyOverlay.format_count(solved=final[0], total=final[1])}"
                    f" at {xs[-1]} transitions"
                ),
            )

        ax.set_xlabel("online transitions")
        ax.set_ylim(-4, 104)
        ax.grid(alpha=0.25, linewidth=0.6)
        # Counts on the y axis, never a bare percentage. Both families share this
        # denominator -- shared_total refuses to draw otherwise -- so one axis is honest.
        positions, labels = TossingRoomGoalFamilyCurves.count_ticks(total=total)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_ylabel(
            f"test tasks solved — bold: x/{total} pooled;  thin: one seed, x/{seed_total}",
            fontsize=9,
        )
        ax.legend(fontsize=9, loc="lower right")
        ax.set_title(title, fontsize=11)
        fig.tight_layout()
        fig.savefig(output, dpi=160)
        print(f"wrote {output}")
        return fig

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--results-root", type=Path, required=True)
        parser.add_argument("--method", type=str, default="ees")
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument(
            "--title",
            type=str,
            default="Tossing Room (split throws), EES: TRASH vs RECYCLING",
        )
        parser.add_argument(
            "--dump-json",
            type=Path,
            default=None,
            help="Write every count to this file, so an experiment log's tables "
            "re-derive from a committed record rather than from a transcription.",
        )
        args = parser.parse_args()
        TossingRoomSplitFamilyOverlay.print_report(root=args.results_root, method=args.method)
        if args.dump_json is not None:
            args.dump_json.write_text(
                json.dumps(
                    TossingRoomSplitFamilyOverlay.as_json(
                        root=args.results_root, method=args.method
                    ),
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            print(f"wrote {args.dump_json}")
        TossingRoomSplitFamilyOverlay.render(
            root=args.results_root,
            method=args.method,
            output=args.output,
            title=args.title,
        )


if __name__ == "__main__":
    TossingRoomSplitFamilyOverlay.main()

"""Post-run analysis: redraws the **10x-budget** cells of
`docs/experiment-logs/2026-08-07-pickup-weight-cycle-budget-10x-runs/` (all four cells --
`oneway-scheduled`, `oneway-never`, `twoway-scheduled`, `twoway-never`) as training curves
that show the one-way `never` arm's bimodal population directly, rather than folding it
into one bold mean that describes neither of its two subgroups.

**Background.** `ResetFreeCycleBudget`'s own write-up
(`docs/experiment-logs/2026-08-07-pickup-weight-cycle-budget.md`, "Addendum: the one-way
`never` cell is a mixture of two populations") established that under the one-way ledge,
every reset-free seed strands -- but the *cycle* at which it strands is fixed by the seed:
six seeds take their last effective (pile-reaching) practice attempt at checkpoint 1 and
never gain another; four take theirs later and keep improving for several more cycles.
Pooling all ten into one mean, as `ResetFreeCycleBudget.render_curves` does, draws a line
that sits in the empty gap between the two groups' final scores -- exactly the failure
mode CLAUDE.md's per-seed-traces rule exists to make visible. Under the two-way ledge
stranding is structurally impossible (the domain's only irreversible action is removed),
so that arm keeps a single line -- but it says so explicitly in its own legend entry,
per CLAUDE.md's rule that a sibling panel with a split must not make an unsplit arm look
accidentally different.

This module is the **1x-dropped, split-added** redraw of
`ResetFreeCycleBudget.render_curves`: three rows (OVERALL / TRASH / RECYCLING) x two
columns (one-way / two-way ledge), 10x cells only, styled per CLAUDE.md's
"Training-curve style, fixed project-wide" section (added in #188). It reuses
`ResetFreeCycleBudget.load_cell` / `.series` / `.transitions` / `.format_count` rather
than re-deriving the loading and pooling arithmetic those already implement and test --
only the split, the subgroup mean and the six-panel layout are new here.

**Why a plain per-seed mean, not `ResetFreeCycleBudget.pooled_curve`'s summed x/y.**
`pooled_curve` sums solved and total separately across seeds, which matters when seeds
could have run different numbers of tasks. Here they can't: `load_cell` already asserts
every seed shares one `_COMPOSITION` (14 TRASH / 14 RECYCLING / 2 EMPTY) at every
checkpoint, so every seed in a subgroup has the identical per-checkpoint denominator and
a plain mean of per-seed solved-counts agrees with pooled-summed-then-divided exactly.
The plain mean is used anyway, deliberately, because it is what a reader compares the
faint per-seed traces against by eye -- the faint lines *are* per-seed counts, and the
bold line should be the same quantity averaged, not a differently-normalized one that
happens to coincide today.

Reads only already-produced output (CLAUDE.md's `analysis/` convention -- this never runs
a simulation or drives a `Method`).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from analysis.practice_makes_perfect.reset_free_cycle_budget import ResetFreeCycleBudget

_LEDGES = ("one-way", "two-way")
_SCHEDULED = "scheduled"
_NEVER = "never"

# Colour carries the arm's role (CLAUDE.md's training-curve-style section): blue is
# scheduled/env-resets, orange is never/reset-free. Never a third hue on this figure --
# the subgroup split within `never` is carried by linestyle instead.
_BLUE = "#0072B2"
_ORANGE = "#D55E00"

# Solid is the main/non-stuck population, or an arm's only line when it has no split;
# dashed is the stuck/stranded subgroup.
_SOLID = "-"
_DASHED = (0, (4, 2))

_SEED_ALPHA = 0.16
_SEED_WIDTH = 0.8
_MEAN_WIDTH = 2.3

# (family key, panel display name, per-seed denominator) -- the three rows, top to
# bottom. `None` reads the pooled-across-families series `ResetFreeCycleBudget.series`
# already returns as "overall".
_FAMILIES = (
    (None, "all test tasks", 30),
    ("TRASH", "TRASH tasks", 14),
    ("RECYCLING", "RECYCLING tasks", 14),
)

# The four 10x cells this figure draws, keyed the same way the committed run tree lays
# them out under `docs/experiment-logs/2026-08-07-pickup-weight-cycle-budget-10x-runs/`.
_CELL_DIRS = {
    ("one-way", _SCHEDULED): "oneway-scheduled",
    ("one-way", _NEVER): "oneway-never",
    ("two-way", _SCHEDULED): "twoway-scheduled",
    ("two-way", _NEVER): "twoway-never",
}


class ResetFreeCycleBudgetBimodal:
    """A static-method container, never instantiated, same as `ResetFreeCycleBudget`
    and every other business-logic class in this project."""

    # -------------------------------------------------------------- reading back

    @staticmethod
    def load_10x_cells(*, runs_root: Path) -> dict[tuple[str, str], dict]:
        """The four (ledge, policy) 10x cells this figure draws, read through
        `ResetFreeCycleBudget.load_cell` so the budget/manipulation/composition checks
        it already performs run here too."""
        return {
            (ledge, policy): ResetFreeCycleBudget.load_cell(
                directory=runs_root / directory, budget="10x", policy=policy
            )
            for (ledge, policy), directory in _CELL_DIRS.items()
        }

    # -------------------------------------------------------------- the split

    @staticmethod
    def stuck_split(*, cell: dict) -> tuple[list[int], list[int]]:
        """`(stuck seeds, non-stuck seeds)`, sorted.

        "Stuck" means the last increase in cumulative effective (pile-reaching) practice
        attempts happened at or before checkpoint 1 -- i.e. the seed never gained a
        single additional effective attempt after the first evaluation checkpoint that
        followed any practice at all.

        `max((...), default=0)` deliberately cannot distinguish two cases: a seed whose
        last gain really was at checkpoint 1, and a seed that gained nothing whatsoever
        (the generator expression is empty, so `default=0` fires). Both land in
        `last_increase <= 1` and therefore in the same "stuck" bucket. That collapse is
        the right call rather than an oversight: both describe a robot that picked up
        zero *additional* effective practice after checkpoint 0/1, which is exactly the
        thing "stuck" is meant to mean for this figure -- there is no further
        distinction a reader of this split needs between "stalled immediately" and
        "never started".
        """
        stuck: list[int] = []
        non_stuck: list[int] = []
        for seed in sorted(cell):
            attempts = cell[seed]["effective_attempts"]
            gains = (
                index for index in range(1, len(attempts)) if attempts[index] > attempts[index - 1]
            )
            last_increase = max(gains, default=0)
            (stuck if last_increase <= 1 else non_stuck).append(seed)
        return stuck, non_stuck

    # -------------------------------------------------------------- subgroup arithmetic

    @staticmethod
    def subgroup_mean_curve(*, cell: dict, seeds: list[int], family: str | None) -> list[float]:
        """The plain per-seed mean solved-count curve over just `seeds` -- not
        `ResetFreeCycleBudget.pooled_curve`'s summed x/y. See this module's docstring
        for why a plain mean is the deliberate choice here rather than an accidental
        divergence from the sibling module's convention: `load_cell` already asserts
        every seed shares the same per-checkpoint denominator, so the two are
        numerically identical here, and the plain mean is what a reader compares the
        faint per-seed traces against directly."""
        num_checkpoints = len(cell[seeds[0]]["transitions"])
        return [
            sum(
                ResetFreeCycleBudget.series(cell=cell, seed=seed, family=family)[index][0]
                for seed in seeds
            )
            / len(seeds)
            for index in range(num_checkpoints)
        ]

    @staticmethod
    def subgroup_pooled_final(
        *, cell: dict, seeds: list[int], family: str | None
    ) -> tuple[int, int]:
        """`(solved, total)` pooled -- summed, not scaled -- over just `seeds`' own
        final-checkpoint `series(...)[-1]`. This is the legend's `final x/y`: computed
        straight from each seed's own last entry rather than by rescaling
        `subgroup_mean_curve`'s output back up by a seed count, so a rounding step in
        the mean can never leak into the published count."""
        solved = 0
        total = 0
        for seed in seeds:
            entry = ResetFreeCycleBudget.series(cell=cell, seed=seed, family=family)[-1]
            solved += entry[0]
            total += entry[1]
        return solved, total

    # -------------------------------------------------------------- the figure

    @staticmethod
    def render(*, cells: dict[tuple[str, str], dict], output: Path, title: str) -> Figure:
        """Three rows (OVERALL / TRASH / RECYCLING) x two columns (one-way / two-way
        ledge), 10x cells only. The one-way `never` arm draws two bold subgroup means
        (stuck dashed, non-stuck solid) over its ten faint per-seed traces; every other
        arm draws one bold mean, and the two-way `never` arm's single line says
        explicitly that there is no split, so it cannot be mistaken for one that was
        never checked."""
        stuck_by_ledge = {
            ledge: ResetFreeCycleBudgetBimodal.stuck_split(cell=cells[(ledge, _NEVER)])
            for ledge in _LEDGES
        }

        figure, axes_grid = plt.subplots(3, 2, figsize=(13.6, 12.0), squeeze=False)
        for row, (family, family_name, total) in enumerate(_FAMILIES):
            for column, ledge in enumerate(_LEDGES):
                axes = axes_grid[row][column]

                scheduled = cells[(ledge, _SCHEDULED)]
                scheduled_seeds = sorted(scheduled)
                xs = ResetFreeCycleBudget.transitions(cell=scheduled)
                for seed in scheduled_seeds:
                    entry = ResetFreeCycleBudget.series(cell=scheduled, seed=seed, family=family)
                    axes.plot(
                        xs,
                        [solved for solved, _ in entry],
                        color=_BLUE,
                        alpha=_SEED_ALPHA,
                        linewidth=_SEED_WIDTH,
                    )
                scheduled_mean = ResetFreeCycleBudgetBimodal.subgroup_mean_curve(
                    cell=scheduled, seeds=scheduled_seeds, family=family
                )
                scheduled_final = ResetFreeCycleBudgetBimodal.subgroup_pooled_final(
                    cell=scheduled, seeds=scheduled_seeds, family=family
                )
                scheduled_count = ResetFreeCycleBudget.format_count(
                    solved=scheduled_final[0], total=scheduled_final[1]
                )
                axes.plot(
                    xs,
                    scheduled_mean,
                    color=_BLUE,
                    linestyle=_SOLID,
                    linewidth=_MEAN_WIDTH,
                    label=f"env resets — mean, n={len(scheduled_seeds)}, final {scheduled_count}",
                )

                never = cells[(ledge, _NEVER)]
                never_seeds = sorted(never)
                xs2 = ResetFreeCycleBudget.transitions(cell=never)
                for seed in never_seeds:
                    entry = ResetFreeCycleBudget.series(cell=never, seed=seed, family=family)
                    axes.plot(
                        xs2,
                        [solved for solved, _ in entry],
                        color=_ORANGE,
                        alpha=_SEED_ALPHA,
                        linewidth=_SEED_WIDTH,
                    )

                stuck, non_stuck = stuck_by_ledge[ledge]
                if stuck and non_stuck:
                    non_stuck_mean = ResetFreeCycleBudgetBimodal.subgroup_mean_curve(
                        cell=never, seeds=non_stuck, family=family
                    )
                    non_stuck_final = ResetFreeCycleBudgetBimodal.subgroup_pooled_final(
                        cell=never, seeds=non_stuck, family=family
                    )
                    non_stuck_count = ResetFreeCycleBudget.format_count(
                        solved=non_stuck_final[0], total=non_stuck_final[1]
                    )
                    axes.plot(
                        xs2,
                        non_stuck_mean,
                        color=_ORANGE,
                        linestyle=_SOLID,
                        linewidth=_MEAN_WIDTH,
                        label=(
                            f"never reset — non-stuck mean, n={len(non_stuck)}, "
                            f"final {non_stuck_count}"
                        ),
                    )
                    stuck_mean = ResetFreeCycleBudgetBimodal.subgroup_mean_curve(
                        cell=never, seeds=stuck, family=family
                    )
                    stuck_final = ResetFreeCycleBudgetBimodal.subgroup_pooled_final(
                        cell=never, seeds=stuck, family=family
                    )
                    stuck_count = ResetFreeCycleBudget.format_count(
                        solved=stuck_final[0], total=stuck_final[1]
                    )
                    axes.plot(
                        xs2,
                        stuck_mean,
                        color=_ORANGE,
                        linestyle=_DASHED,
                        linewidth=_MEAN_WIDTH,
                        label=f"never reset — stuck mean, n={len(stuck)}, final {stuck_count}",
                    )
                else:
                    never_mean = ResetFreeCycleBudgetBimodal.subgroup_mean_curve(
                        cell=never, seeds=never_seeds, family=family
                    )
                    never_final = ResetFreeCycleBudgetBimodal.subgroup_pooled_final(
                        cell=never, seeds=never_seeds, family=family
                    )
                    never_count = ResetFreeCycleBudget.format_count(
                        solved=never_final[0], total=never_final[1]
                    )
                    axes.plot(
                        xs2,
                        never_mean,
                        color=_ORANGE,
                        linestyle=_SOLID,
                        linewidth=_MEAN_WIDTH,
                        label=(
                            f"never reset — mean, n={len(never_seeds)}, final {never_count} "
                            "(no stranding here)"
                        ),
                    )

                axes.grid(alpha=0.25, linewidth=0.6)
                axes.set_ylim(-total * 0.04, total * 1.06)
                axes.set_xlabel("online transitions")
                axes.set_ylabel("solved per seed")
                axes.set_title(
                    f"{family_name} (of {total}) — {ledge} ledge, 100 cycles", fontsize=10
                )
                axes.legend(fontsize=7.5, loc="lower right", framealpha=0.95)

        figure.suptitle(title, fontsize=11.5)
        figure.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=150)
        return figure


def main() -> None:
    """CLI entrypoint. All flags named, no positional arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "docs"
            / "experiment-logs"
            / "2026-08-07-pickup-weight-cycle-budget-10x-runs"
        ),
        help=(
            "Directory holding oneway-scheduled/ oneway-never/ twoway-scheduled/ "
            "twoway-never/, each <seed>/stats.json. Defaults to this repo's own "
            "committed 10x run tree."
        ),
    )
    parser.add_argument("--output", type=Path, required=True, help="PNG to write.")
    args = parser.parse_args()

    cells = ResetFreeCycleBudgetBimodal.load_10x_cells(runs_root=args.runs_root)
    ResetFreeCycleBudgetBimodal.render(
        cells=cells,
        output=args.output,
        title=(
            "Tossing Room (split throws, weight drawn at pickup), EES, 10 fixed seeds\n"
            "100-cycle (10x) reset-free practice — one-way `never` split by stranding onset"
        ),
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

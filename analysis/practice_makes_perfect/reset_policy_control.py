"""The reset-policy control on Tossing Room: `scheduled` vs `never`, every goal family
plotted separately, in both of the domain's skill configurations.

**Why a new module rather than an argument to `reset_free_cycle_budget.py`.** That module
answers one narrower question -- "does the reset-free gap close when the cycle budget grows
tenfold?" -- and its shape is welded to it: it *requires* all eight cells of a fixed
budget x ledge x policy cube, and refuses to draw anything if one is missing, deliberately,
because a difference-of-differences with a hole in it is undefined. This module asks the
plainer question underneath ("what does turning the periodic reset off cost, per goal
family?"), so it takes whatever cells it is given and groups them, which is exactly the
flexibility the cube module gives up on purpose. Reading it back is still *its* job:
`load_cell` performs the budget, manipulation and composition checks, and they are the
reason a cell that is not what its directory name claims fails loudly here too.

Three things this draws that the cube module does not, each a requirement rather than a
preference:

* **The EMPTY family gets its own panel.** The cube module plots `all / TRASH / RECYCLING`
  and drops EMPTY. EMPTY is 2 of every seed's 30 tasks, so it is a real family and its
  absence from a "plot all tasks separately" figure would be a silent omission -- but it is
  also 20/20 in both arms of every cell measured so far, and a panel showing two flat lines
  on top of each other is the honest rendering of "this family supports no inference",
  which a missing panel is not.
* **`--unsplit-skills` is readable.** That flag renders the throw goal as a shared
  `ItemInBin(<kind>, <bin>)` rather than `TrashInBin`/`RecyclingInBin`, which
  `GoalFamilies.classify` did not recognise at all until the rule list grew to cover both.
* **Linestyle carries the within-arm subgroup, never the budget.** The cube module spends
  linestyle on `1x` vs `10x` because it puts both budgets on one panel. Here each budget is
  its own column, which frees linestyle for the thing CLAUDE.md reserves it for: the
  stuck/non-stuck split inside the reset-free arm. That split is the whole point on this
  domain -- the one-way `never` population is bimodal, so its pooled mean describes none of
  its seeds.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.practice_makes_perfect.paired_tests import PairedTests  # noqa: E402
from analysis.practice_makes_perfect.reset_free_cycle_budget import (  # noqa: E402
    ResetFreeCycleBudget,
)
from analysis.practice_makes_perfect.reset_free_cycle_budget_bimodal import (  # noqa: E402
    ResetFreeCycleBudgetBimodal,
)

_SCHEDULED = "scheduled"
_NEVER = "never"

# Colour carries the arm's role and nothing else: blue is the arm that HAS a reset
# mechanism, orange the arm that has none. See CLAUDE.md's training-curve-style section.
_BLUE = "#0072B2"
_ORANGE = "#D55E00"
_COLORS = {_SCHEDULED: _BLUE, _NEVER: _ORANGE}

# Linestyle carries the within-arm subgroup. Solid is the main population (or an arm's
# only line where it does not split); dashed is the stuck/stranded subgroup.
_SOLID = "-"
_DASHED = (0, (4, 2))

_SEED_ALPHA = 0.16
_SEED_WIDTH = 0.8
_MEAN_WIDTH = 2.3

# (family key, panel display name, per-seed denominator). `None` is the pooled row, kept
# alongside the per-family rows rather than instead of them.
_FAMILIES = (
    (None, "all test tasks", 30),
    ("TRASH", "TRASH tasks", 14),
    ("RECYCLING", "RECYCLING tasks", 14),
    ("EMPTY", "EMPTY tasks", 2),
)

_POLICY_DISPLAY = {_SCHEDULED: "env resets", _NEVER: "never reset"}


class ResetPolicyControl:
    """A static-method container, never instantiated, same as every other business-logic
    class in this project."""

    # ------------------------------------------------------------------ parsing input

    @staticmethod
    def parse_cells(*, raw: list[str]) -> dict[tuple[str, str, str, str], Path]:
        """`CONFIG:LEDGE:BUDGET:POLICY=DIR` entries into a keyed map.

        Four axes rather than the cube module's three, because the skill configuration
        (`split` vs `unsplit`) is a genuine second domain configuration and folding it into
        the ledge name would make `one-way-unsplit` sort next to `one-way` as if it were a
        third ledge.
        """
        cells: dict[tuple[str, str, str, str], Path] = {}
        for entry in raw:
            key, separator, directory = entry.partition("=")
            if not separator:
                raise ValueError(
                    f"--cell must look like CONFIG:LEDGE:BUDGET:POLICY=DIR, got {entry!r}"
                )
            parts = key.split(":")
            if len(parts) != 4:
                raise ValueError(f"--cell key must have four colon-separated parts, got {key!r}")
            config, ledge, budget, policy = parts
            if policy not in (_SCHEDULED, _NEVER):
                raise ValueError(f"policy must be {_SCHEDULED!r} or {_NEVER!r}, got {policy!r}")
            cells[(config, ledge, budget, policy)] = Path(directory)
        return cells

    @staticmethod
    def load(*, directories: dict[tuple[str, str, str, str], Path]) -> dict:
        """Every cell read back through the cube module's own guarded reader.

        Reusing `load_cell` rather than re-reading `stats.json` here is the whole point:
        it re-derives the cycle count from the run instead of trusting the path, checks
        `num_practice_resets` against the policy the directory name claims, and rejects a
        sweep whose goal-family composition is not the domain's fixed 14/14/2. Each of
        those has caught a real mis-pointed directory before.
        """
        loaded = {}
        for (config, ledge, budget, policy), directory in sorted(directories.items()):
            loaded[(config, ledge, budget, policy)] = ResetFreeCycleBudget.load_cell(
                directory=directory, budget=budget, policy=policy
            )
        return loaded

    @staticmethod
    def groups(*, cells: dict) -> list[tuple[str, str, str]]:
        """The (config, ledge, budget) triples that have BOTH policies present.

        A group with one arm is dropped rather than half-drawn: this module exists to
        report a paired difference, and a lone arm has no difference to report.
        """
        keys = {(config, ledge, budget) for config, ledge, budget, _ in cells}
        return sorted(
            group for group in keys if (*group, _SCHEDULED) in cells and (*group, _NEVER) in cells
        )

    # ------------------------------------------------------------------ the arithmetic

    @staticmethod
    def final_counts(*, cell: dict, family: str | None) -> tuple[int, int]:
        """The cell's pooled `(solved, total)` at the last checkpoint."""
        return ResetFreeCycleBudget.pooled_curve(cell=cell, family=family)[-1]

    @staticmethod
    def per_seed_finals(*, cell: dict, family: str | None) -> dict[int, int]:
        """Each seed's solved count at the last checkpoint."""
        return {
            seed: ResetFreeCycleBudget.series(cell=cell, seed=seed, family=family)[-1][0]
            for seed in sorted(cell)
        }

    @staticmethod
    def paired_differences(*, cells: dict, group: tuple[str, str, str], family: str | None):
        """`never - scheduled` per seed, on the seeds both arms actually ran.

        Intersected rather than zipped, so a partially-complete sweep produces a paired
        test over the seeds that really are paired instead of silently aligning seed 7 of
        one arm against seed 9 of the other.
        """
        scheduled = ResetPolicyControl.per_seed_finals(
            cell=cells[(*group, _SCHEDULED)], family=family
        )
        never = ResetPolicyControl.per_seed_finals(cell=cells[(*group, _NEVER)], family=family)
        seeds = sorted(set(scheduled) & set(never))
        return seeds, [float(never[seed] - scheduled[seed]) for seed in seeds]

    @staticmethod
    def format_count(*, solved: int, total: int) -> str:
        """`x/y (z%)` -- the count first and always, the percentage only ever beside it."""
        share = 100.0 * solved / total if total else 0.0
        return f"{solved}/{total} ({share:.0f}%)"

    # ------------------------------------------------------------------ the report

    @staticmethod
    def report(*, cells: dict) -> None:
        for group in ResetPolicyControl.groups(cells=cells):
            config, ledge, budget = group
            print(f"=== {config} skills / {ledge} ledge / {budget} budget ===")
            for family, family_name, _ in _FAMILIES:
                scheduled = ResetPolicyControl.final_counts(
                    cell=cells[(*group, _SCHEDULED)], family=family
                )
                never = ResetPolicyControl.final_counts(cell=cells[(*group, _NEVER)], family=family)
                seeds, differences = ResetPolicyControl.paired_differences(
                    cells=cells, group=group, family=family
                )
                test = PairedTests.sign_flip(differences=differences)
                worse = sum(1 for value in differences if value < 0)
                tied = sum(1 for value in differences if value == 0)
                scheduled_count = ResetPolicyControl.format_count(
                    solved=scheduled[0], total=scheduled[1]
                )
                never_count = ResetPolicyControl.format_count(solved=never[0], total=never[1])
                print(f"  {family_name}")
                print(f"    scheduled {scheduled_count}   never {never_count}")
                print(
                    f"    never worse on {worse}/{len(seeds)} seeds, tied on {tied}/{len(seeds)}; "
                    f"mean per-seed difference {test.statistic:+.2f}; "
                    f"exact paired sign-flip p = {test.p_value:.6f}"
                )
                if test.p_value > 0.05:
                    mde = PairedTests.minimum_detectable_effect(differences=differences)
                    if test.num_zero_differences == len(differences):
                        print(
                            "    null result: every seed's difference is exactly 0, so this "
                            "family supports no inference here"
                        )
                    else:
                        print(
                            "    null result; smallest per-seed difference this design had "
                            f"80% power to detect: {mde:.2f} tasks"
                        )
            print()

    # ------------------------------------------------------------------ the figures

    @staticmethod
    def subgroups(*, cell: dict, policy: str) -> list[tuple[str, list[int], object]]:
        """`(label, seeds, linestyle)` for one arm.

        The reset-free arm is split into stuck and non-stuck by
        `ResetFreeCycleBudgetBimodal.stuck_split`; the scheduled arm is not split at all,
        and says so in its legend entry rather than silently having one line where its
        sibling has two. A `never` arm that turns out not to split gets one solid line too
        -- the split is measured per cell, never assumed from the arm's name.
        """
        seeds = sorted(cell)
        if policy == _SCHEDULED:
            return [("no stranding here", seeds, _SOLID)]
        stuck, non_stuck = ResetFreeCycleBudgetBimodal.stuck_split(cell=cell)
        if not stuck or not non_stuck:
            return [("no split here", seeds, _SOLID)]
        return [("non-stuck", non_stuck, _SOLID), ("stuck", stuck, _DASHED)]

    @staticmethod
    def subgroup_mean(*, cell: dict, seeds: list[int], family: str | None) -> list[float]:
        """The mean per-seed solved count at each checkpoint, over one subgroup."""
        series = [
            ResetFreeCycleBudget.series(cell=cell, seed=seed, family=family) for seed in seeds
        ]
        return [
            sum(point[index][0] for point in series) / len(series)
            for index in range(len(series[0]))
        ]

    @staticmethod
    def render_curves(*, cells: dict, output: Path, title: str):
        """One row per goal family, one column per (config, ledge, budget) group.

        Faint per-seed traces are drawn first and are the point, not decoration: the
        one-way reset-free arm is bimodal on this domain, so a bold mean drawn over it
        describes none of its seeds, and the faint lines are what make that visible rather
        than asserted in prose.
        """
        groups = ResetPolicyControl.groups(cells=cells)
        if not groups:
            raise ValueError("no group has both policies present; nothing to draw")
        fig, axes_grid = plt.subplots(
            len(_FAMILIES),
            len(groups),
            figsize=(4.6 * len(groups), 3.3 * len(_FAMILIES)),
            squeeze=False,
        )
        for row, (family, family_name, denominator) in enumerate(_FAMILIES):
            for column, group in enumerate(groups):
                axes = axes_grid[row][column]
                config, ledge, budget = group
                for policy in (_SCHEDULED, _NEVER):
                    cell = cells[(*group, policy)]
                    xs = ResetFreeCycleBudget.transitions(cell=cell)
                    color = _COLORS[policy]
                    for seed in sorted(cell):
                        entry = ResetFreeCycleBudget.series(cell=cell, seed=seed, family=family)
                        axes.plot(
                            xs,
                            [solved for solved, _ in entry],
                            color=color,
                            alpha=_SEED_ALPHA,
                            linewidth=_SEED_WIDTH,
                        )
                    for label, seeds, linestyle in ResetPolicyControl.subgroups(
                        cell=cell, policy=policy
                    ):
                        means = ResetPolicyControl.subgroup_mean(
                            cell=cell, seeds=seeds, family=family
                        )
                        axes.plot(
                            xs,
                            means,
                            color=color,
                            linestyle=linestyle,
                            linewidth=_MEAN_WIDTH,
                            label=f"{_POLICY_DISPLAY[policy]} — {label} mean, n={len(seeds)}",
                        )
                axes.grid(alpha=0.25, linewidth=0.6)
                axes.set_ylim(-denominator * 0.04, denominator * 1.06)
                axes.set_xlabel("online transitions")
                # The bare quantity only: the denominator belongs in the panel title, which
                # is read once, not on every tick of a repeated axis label.
                axes.set_ylabel("solved per seed", fontsize=9)
                axes.set_title(
                    f"{family_name} (of {denominator}) — {config} skills, {ledge} ledge, {budget}",
                    fontsize=9.5,
                )
                axes.legend(fontsize=7.0, loc="lower right", framealpha=0.95)
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        fig.savefig(output, dpi=150)
        print(f"wrote {output}")
        return fig

    @staticmethod
    def render_paired(*, cells: dict, output: Path, title: str):
        """The paired view: one dot per seed, `never - scheduled` at the final checkpoint.

        A per-seed scatter rather than two bars, because with ten seeds a bar chart of two
        means hides one seed driving the whole effect -- and on this domain that is not
        hypothetical.
        """
        groups = ResetPolicyControl.groups(cells=cells)
        fig, axes_grid = plt.subplots(
            1, len(_FAMILIES), figsize=(4.4 * len(_FAMILIES), 4.6), squeeze=False
        )
        for column, (family, family_name, denominator) in enumerate(_FAMILIES):
            axes = axes_grid[0][column]
            for index, group in enumerate(groups):
                seeds, differences = ResetPolicyControl.paired_differences(
                    cells=cells, group=group, family=family
                )
                test = PairedTests.sign_flip(differences=differences)
                # Orange: every dot here is the reset-free arm's deficit against its own
                # paired scheduled seed, so the whole series belongs to the never arm.
                axes.scatter(
                    [index] * len(differences),
                    differences,
                    color=_ORANGE,
                    alpha=0.75,
                    s=38,
                    zorder=3,
                )
                mean = test.statistic
                axes.plot(
                    [index - 0.22, index + 0.22], [mean, mean], color=_BLUE, linewidth=_MEAN_WIDTH
                )
                # Anchored to the top of the axes in axes-fraction coordinates, not
                # offset from the mean line: offsetting from the data put the p-value on
                # top of the dots on exactly the panels where the effect is smallest and
                # the reader most needs to read both.
                axes.annotate(
                    f"p={test.p_value:.4f}\nn={len(seeds)}",
                    (index, 1.0),
                    xycoords=("data", "axes fraction"),
                    textcoords="offset points",
                    xytext=(0, -22),
                    ha="center",
                    va="top",
                    fontsize=7.0,
                )
            axes.axhline(0.0, color="#666666", linewidth=0.9, linestyle=(0, (3, 3)))
            axes.set_xticks(range(len(groups)))
            axes.set_xticklabels(
                [f"{config}\n{ledge}\n{budget}" for config, ledge, budget in groups], fontsize=7.5
            )
            axes.grid(alpha=0.25, linewidth=0.6, axis="y")
            axes.set_ylabel("never − scheduled, per seed", fontsize=9)
            axes.set_title(f"{family_name} (of {denominator})", fontsize=10)
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        fig.savefig(output, dpi=150)
        print(f"wrote {output}")
        return fig

    # ------------------------------------------------------------------ the aggregate

    @staticmethod
    def aggregate(*, cells: dict) -> dict:
        """Every number this module draws or prints, small enough to commit.

        A 100-cycle run's own `stats.json` is ~750 KB because it carries all 30 per-task
        outcomes at each of 101 checkpoints; sixty of those is ~45 MB, which is a lot of
        git history to add for figures that need only the per-family counts. This reduces
        each run to those counts -- the exact input `render_curves` and `report` consume --
        so both stay re-derivable from committed data without committing the raw sweeps.
        It is a projection of the runs, never a replacement for them: `stats.json` remains
        the artefact any byte-level reproducibility claim is made against.
        """
        return {
            ":".join(key): {
                str(seed): {
                    "transitions": cell[seed]["transitions"],
                    "overall": cell[seed]["overall"],
                    "families": cell[seed]["families"],
                }
                for seed in sorted(cell)
            }
            for key, cell in sorted(cells.items())
        }

    # ------------------------------------------------------------------ entrypoint

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--cell",
            action="append",
            required=True,
            metavar="CONFIG:LEDGE:BUDGET:POLICY=DIR",
            help="Repeatable. DIR holds <seed>/stats.json directly.",
        )
        parser.add_argument("--curves-output", type=Path, required=True)
        parser.add_argument("--paired-output", type=Path, required=True)
        parser.add_argument("--title", default="Tossing Room: periodic practice resets vs none")
        parser.add_argument(
            "--dump-json",
            type=Path,
            default=None,
            help="Write the per-seed, per-checkpoint family counts these figures are "
            "built from, so the log stays re-derivable without committing the raw sweeps.",
        )
        args = parser.parse_args()
        cells = ResetPolicyControl.load(directories=ResetPolicyControl.parse_cells(raw=args.cell))
        ResetPolicyControl.report(cells=cells)
        if args.dump_json is not None:
            args.dump_json.write_text(
                json.dumps(ResetPolicyControl.aggregate(cells=cells), indent=2, sort_keys=True)
            )
            print(f"wrote {args.dump_json}")
        ResetPolicyControl.render_curves(cells=cells, output=args.curves_output, title=args.title)
        ResetPolicyControl.render_paired(
            cells=cells, output=args.paired_output, title=f"{args.title} — paired per seed"
        )


if __name__ == "__main__":
    ResetPolicyControl.main()

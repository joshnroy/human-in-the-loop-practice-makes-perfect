"""Post-run analysis for reset-free practice on Tossing Room at **two cycle budgets**:
is the reset-free arm's deficit *starvation*, or an inability to learn?

**Background.** `--practice-reset-policy scheduled` puts the environment back to a
freshly-sampled train task at the top of each practice period; `never` lets practice state
run continuously, which is the real-robot condition. Under the one-way ledge the merged 1x
A/B measured `scheduled` 183/300 against `never` 112/300, and the mechanism it identified
was not a learning failure but a *supply* failure: the reset-free arm logged 207 effective
practice attempts against 1191, with 85/100 cycles attempting not one. If that starvation
is the whole story, then handing the reset-free arm ten times the cycles should buy back
most of the gap. If the gap survives at 10x, starvation is not sufficient.

There is direct precedent for the shape: on an earlier Tossing Room variant
`ThrowRecycling` went from 11/56 to 901/982 at 10x, which is what established
starved-not-unable there.

So this reads back a **cube** of budget (`1x` = 10 cycles, `10x` = 100) x ledge
(`one-way`, `two-way`) x policy (`scheduled`, `never`), and the quantity that carries is
the **change in the within-ledge gap between the two budgets** -- a difference of
differences, taken within a seed at every step because all eight cells ran the same fixed
seeds.

**Levels do not compare across ledge conditions, only gaps do.** `--two-way-ledge` also
drops EMPTY's shortest solve 10 -> 9 and the evaluation horizon 12 -> 11, and stops
RECYCLING being one-attempt-per-period, so the two-way world is a genuinely easier domain.
Nothing below ever subtracts a two-way count from a one-way one.

**Levels *do* compare across budgets**, which is the one comparison this module adds:
within a ledge and a policy, the only difference between the 1x and 10x cells is
`--num-cycles`. Everything else -- domain flags, seeds, test set, `--num-test-tasks 30` --
is identical, read out of the committed `config_snapshot.json` rather than reconstructed.

**Two figures.** `render_curves` puts all four (budget x policy) curves for one ledge on
one axes per family, on a shared **online-transitions** x axis so the 1x arm is visibly a
prefix of the 10x one's budget rather than a separate experiment. `render_gap` is the
figure the question actually asks: the per-seed `scheduled - never` gap at 1x joined to the
same seed's gap at 10x, so a gap that closes is ten lines sloping to zero and a gap that
does not is ten flat ones. A bar chart of two means would hide one seed carrying the whole
effect, which is exactly the failure mode this domain has already produced.

**Statistics.** Every comparison is paired -- the cells share seeds -- and the test is
`PairedTests.sign_flip`, exact by enumerating its null in full: no normal approximation, no
continuity or tie correction, no scipy. It is imported rather than reimplemented, as is
goal classification (`goal_families.GoalFamilies`); a second copy of either is how a sign
error or a drifted denominator gets published. Zero differences are kept, never dropped:
"every seed moved by exactly zero" is a finding.

Reads only already-produced output (CLAUDE.md's `analysis/` convention -- this never runs a
simulation or drives a `Method`). Each `--cell` points at the directory holding that cell's
`<seed>/stats.json`, because the committed sweeps nest differently (`<policy>/<seed>` for
one, `<policy>/ees/<seed>` for the others) and guessing between them is how the wrong sweep
gets read.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from analysis.practice_makes_perfect.goal_families import GoalFamilies  # noqa: E402
from analysis.practice_makes_perfect.paired_tests import PairedTests  # noqa: E402

# The cube, in the order every table, legend and report below uses: the incumbent first in
# each factor. `1x` is the merged protocol, `one-way` the ledge every banked number was
# measured under, `scheduled` the only reset policy that existed before this line of work.
_BUDGETS = ("1x", "10x")
_LEDGES = ("one-way", "two-way")
_POLICIES = ("scheduled", "never")

# What each budget means in cycles. Checked against every run's own `breakdowns` rather
# than trusted from a directory name: the two budgets differ *only* in `--num-cycles`, so
# nothing in a path stops the 1x sweep being read as the 10x one.
_CYCLES = {"1x": 10, "10x": 100}

# The composition the domain allocates for --num-test-tasks 30. `--two-way-ledge` does not
# move it, which is what keeps the ledge conditions' evaluation sets comparable in
# COMPOSITION even though the flag makes the two-way one easier.
_COMPOSITION = {"TRASH": 14, "RECYCLING": 14, "EMPTY": 2}
_NUM_TEST_TASKS = sum(_COMPOSITION.values())

# Colour carries the reset policy, linestyle the cycle budget -- so neither identity rests
# on hue alone on a panel that holds four lines. Validated for colourblind separation and
# for contrast against both a light and a dark chart surface.
_COLORS = {"scheduled": "#0072B2", "never": "#D55E00"}
_LINESTYLES = {"1x": (0, (4, 2)), "10x": "-"}

# Display names for the two `--practice-reset-policy` values. DISPLAY ONLY: the keys,
# directory names and flag values stay `scheduled`/`never`, because those are what the CLI
# accepts and what every committed `config_snapshot.json` already records.
_POLICY_DISPLAY = {
    "scheduled": "practice-session env resets",
    "never": "never env reset",
}

# Skills that require the robot to be at the item pile. A stranded robot can still walk
# (`MoveRoom`) and press buttons (`Press*`) for a whole period, so those are excluded --
# counting them would report a starved arm as busy.
_EFFECTIVE_PREFIXES = ("Pickup", "Throw")


class ResetFreeCycleBudget:
    """A static-method container, never instantiated, same as every other business-logic
    class in this project."""

    # ------------------------------------------------------------------ the cube

    @staticmethod
    def cells() -> tuple[tuple[str, str, str], ...]:
        """The eight (budget, ledge, policy) cells, in report order."""
        return tuple(
            (budget, ledge, policy)
            for budget in _BUDGETS
            for ledge in _LEDGES
            for policy in _POLICIES
        )

    @staticmethod
    def cycles_for(*, budget: str) -> int:
        """How many practice cycles this budget names. The only thing that differs
        between the two sweeps, and therefore the thing worth asserting."""
        return _CYCLES[budget]

    @staticmethod
    def style(*, budget: str, policy: str) -> tuple[str, object]:
        """The (colour, linestyle) this cell is drawn with, on every figure."""
        return _COLORS[policy], _LINESTYLES[budget]

    @staticmethod
    def label(*, budget: str, policy: str) -> str:
        """The legend entry. Uses the display name for the policy, never the flag
        value -- see `_POLICY_DISPLAY`."""
        cycles = _CYCLES[budget]
        return f"{cycles} cycles, {_POLICY_DISPLAY[policy]}"

    @staticmethod
    def format_count(*, solved: int, total: int) -> str:
        """`x/y`, never a bare percentage: the denominators here are small and uneven."""
        return f"{solved}/{total}"

    # ------------------------------------------------------------------ reading back

    @staticmethod
    def load_cells(
        *, directories: dict[tuple[str, str, str], Path]
    ) -> dict[tuple[str, str, str], dict]:
        """Every cell's per-seed, per-checkpoint family counts, with the checks first.

        The eight cells are a cube, not eight strings: "did the gap close?" is a
        difference of differences, so a missing cell leaves it undefined and a report
        that silently drew the halves it could still compute would read as a result.
        """
        missing = [cell for cell in ResetFreeCycleBudget.cells() if cell not in directories]
        if missing:
            names = ", ".join(f"{budget}/{ledge}/{policy}" for budget, ledge, policy in missing)
            raise ValueError(
                f"missing cell(s): {names}. This experiment is a budget x ledge x policy "
                "cube; with a cell absent the change in the within-ledge gap is not "
                "defined and no comparison here is meaningful."
            )
        return {
            cell: ResetFreeCycleBudget.load_cell(
                directory=directories[cell], budget=cell[0], policy=cell[2]
            )
            for cell in ResetFreeCycleBudget.cells()
        }

    @staticmethod
    def load_cell(*, directory: Path, budget: str, policy: str) -> dict:
        """One cell: `{seed: {"transitions": [...], "families": {...}, "overall": [...]}}`."""
        seeds = sorted(int(path.parent.name) for path in directory.glob("*/stats.json"))
        if not seeds:
            raise ValueError(f"no <seed>/stats.json under {directory}")
        expected_cycles = _CYCLES[budget]
        cell: dict[int, dict] = {}
        for seed in seeds:
            stats = json.loads((directory / str(seed) / "stats.json").read_text())
            # A cycle produces one evaluation sweep, plus the one taken before any
            # practice -- so cycles is checkpoints minus one, counted rather than named.
            num_cycles = len(stats["breakdowns"]) - 1
            if num_cycles != expected_cycles:
                raise ValueError(
                    f"{directory}/{seed}: ran {num_cycles} cycles, but this cell is the "
                    f"{budget} budget, which is {expected_cycles} cycles. The two budgets "
                    "differ only in --num-cycles, so nothing in the path would have "
                    "caught the wrong sweep being read here."
                )
            expected_resets = expected_cycles if policy == "scheduled" else 0
            resets = stats.get("num_practice_resets")
            if resets != expected_resets:
                raise ValueError(
                    f"{directory}/{seed}: num_practice_resets is {resets}, expected "
                    f"{expected_resets} for --practice-reset-policy {policy} at "
                    f"{expected_cycles} cycles. The cell is not what its name says."
                )
            transitions = []
            families: dict[str, list[tuple[int, int]]] = {family: [] for family in _COMPOSITION}
            overall: list[tuple[int, int]] = []
            for breakdown in stats["breakdowns"]:
                transitions.append(breakdown["num_online_transitions"])
                counts = ResetFreeCycleBudget.sweep_counts(outcomes=breakdown["outcomes"])
                composition = {family: total for family, (_, total) in counts.items()}
                if composition != _COMPOSITION:
                    raise ValueError(
                        f"{directory}/{seed}: sweep composition {composition} is not the "
                        f"domain's {_COMPOSITION}. A goal has been misfiled between "
                        "families, which moves tasks between denominators invisibly."
                    )
                for family, count in counts.items():
                    families[family].append(count)
                overall.append((sum(solved for solved, _ in counts.values()), _NUM_TEST_TASKS))
            cell[seed] = {
                "transitions": transitions,
                "families": families,
                "overall": overall,
                "effective_attempts": ResetFreeCycleBudget.cumulative_effective_attempts(
                    windows=stats["practice_outcomes_per_cycle"],
                    num_checkpoints=len(transitions),
                ),
            }
        return cell

    @staticmethod
    def sweep_counts(*, outcomes: list[dict]) -> dict[str, tuple[int, int]]:
        """One sweep's `(solved, total)` per family.

        Classification is `GoalFamilies.classify`, reused rather than recopied: it tests
        the `BinEmpty` predicate before the item names, because `Goal.describe()` renders
        EMPTY as "RecyclingBinEmpty(recycling_bin) & TrashBinEmpty(trash_bin)" -- it names
        BOTH bins, so a naive "does it mention recycling?" test swallows it and silently
        reports 16 RECYCLING / 0 EMPTY.
        """
        solved: Counter[str] = Counter()
        total: Counter[str] = Counter()
        for outcome in outcomes:
            family = GoalFamilies.classify(goal=outcome["goal"])
            total[family] += 1
            solved[family] += int(outcome["solved"])
        return {family: (solved[family], total[family]) for family in total}

    @staticmethod
    def cumulative_effective_attempts(*, windows: list[dict], num_checkpoints: int) -> list[int]:
        """Effective practice attempts accumulated *before* each evaluation checkpoint.

        "Effective" means a skill that needs the robot to be at the item pile -- the
        `Pickup*` and `Throw*` families. `MoveRoom` and the `Press*` skills are excluded
        deliberately: a stranded robot can still walk and press buttons all period, so
        counting those would report a starved arm as busy.

        Checkpoint 0 is taken before any practice, so it accumulates nothing; checkpoint
        `i` accumulates windows `0..i-1`. `practice_outcomes_per_cycle` carries one
        trailing window past the last cycle, which this slicing drops rather than
        mis-attributing to a checkpoint that never saw it.
        """
        cumulative = [0]
        running = 0
        for window in windows[: num_checkpoints - 1]:
            running += sum(
                record["num_attempts"]
                for name, record in window.items()
                if name.startswith(_EFFECTIVE_PREFIXES)
            )
            cumulative.append(running)
        return cumulative

    # ------------------------------------------------------------------ arithmetic

    @staticmethod
    def series(*, cell: dict, seed: int, family: str | None) -> list[tuple[int, int]]:
        """One seed's `(solved, total)` per checkpoint, for a family or pooled."""
        return cell[seed]["overall"] if family is None else cell[seed]["families"][family]

    @staticmethod
    def pooled_curve(*, cell: dict, family: str | None) -> list[tuple[int, int]]:
        """The cell's curve pooled over seeds: solved and total both SUMMED, per
        checkpoint.

        Summed rather than averaged, so `x/140` at ten seeds means what it says. A mean of
        per-seed rates would silently reweight a seed that ran a different number of tasks.
        """
        seeds = sorted(cell)
        num_checkpoints = len(cell[seeds[0]]["transitions"])
        pooled = []
        for index in range(num_checkpoints):
            solved = 0
            total = 0
            for seed in seeds:
                entry = ResetFreeCycleBudget.series(cell=cell, seed=seed, family=family)
                solved += entry[index][0]
                total += entry[index][1]
            pooled.append((solved, total))
        return pooled

    @staticmethod
    def pooled_final(*, cell: dict, family: str | None) -> tuple[int, int]:
        """The cell's final-checkpoint score, pooled over seeds, as `(solved, total)`."""
        return ResetFreeCycleBudget.pooled_curve(cell=cell, family=family)[-1]

    @staticmethod
    def transitions(*, cell: dict) -> list[int]:
        """The shared x axis: the checkpoint transition counts, which every seed shares."""
        grids = {tuple(cell[seed]["transitions"]) for seed in sorted(cell)}
        if len(grids) != 1:
            raise ValueError(
                f"seeds disagree on the evaluation checkpoints ({sorted(grids)}), so they "
                "cannot share an x axis."
            )
        return list(next(iter(grids)))

    @staticmethod
    def paired_gaps(*, cells: dict, budget: str, ledge: str, family: str | None) -> list[float]:
        """`scheduled` minus `never` at the final checkpoint, **within a seed**.

        The cells share seeds, so this is paired data. Zero differences are kept rather
        than dropped: "9/10 seeds differ by exactly zero" is a finding, and it is
        invisible if ties are discarded.
        """
        scheduled = cells[(budget, ledge, "scheduled")]
        never = cells[(budget, ledge, "never")]
        seeds = sorted(set(scheduled) & set(never))
        return [
            float(
                ResetFreeCycleBudget.series(cell=scheduled, seed=seed, family=family)[-1][0]
                - ResetFreeCycleBudget.series(cell=never, seed=seed, family=family)[-1][0]
            )
            for seed in seeds
        ]

    @staticmethod
    def paired_gap_changes(*, cells: dict, ledge: str, family: str | None) -> list[float]:
        """`gap at 1x` minus `gap at 10x`, **within a seed** -- positive means the
        reset-free deficit shrank when the budget grew, which is what the starvation
        hypothesis predicts.

        Both terms are already within-seed differences, so this is a difference of
        differences and the pairing has to survive both steps. Zipping two independently
        sorted lists would coincide whenever the cells happen to be flat, which is
        precisely when a broken pairing is hardest to notice.
        """
        at_1x = ResetFreeCycleBudget.paired_gaps(
            cells=cells, budget="1x", ledge=ledge, family=family
        )
        at_10x = ResetFreeCycleBudget.paired_gaps(
            cells=cells, budget="10x", ledge=ledge, family=family
        )
        return [before - after for before, after in zip(at_1x, at_10x, strict=True)]

    @staticmethod
    def starved_cycles(*, cell: dict) -> tuple[int, int]:
        """`(cycles attempting nothing pile-reaching, cycles in total)`, pooled over seeds.

        Returned as a pair so every caller reports it `x/y`. A rate here would hide that
        the 1x and 10x denominators differ by an order of magnitude, which is the whole
        manipulation.
        """
        starved = 0
        total = 0
        for seed in sorted(cell):
            attempts = cell[seed]["effective_attempts"]
            for before, after in zip(attempts[:-1], attempts[1:], strict=True):
                total += 1
                starved += int(after == before)
        return starved, total

    @staticmethod
    def effective_attempts(*, cell: dict) -> int:
        """Pile-reaching practice attempts over the whole run, pooled over seeds."""
        return sum(cell[seed]["effective_attempts"][-1] for seed in sorted(cell))

    # ------------------------------------------------------------------ the report

    @staticmethod
    def print_report(*, cells: dict) -> None:
        """Every number the write-up quotes, as `x/y`, re-derived here."""
        print("practice budget actually spent, per cell\n")
        for budget, ledge, policy in ResetFreeCycleBudget.cells():
            cell = cells[(budget, ledge, policy)]
            starved, total = ResetFreeCycleBudget.starved_cycles(cell=cell)
            print(
                f"  {budget:>3} {ledge:>8} {policy:>9}   "
                f"{ResetFreeCycleBudget.effective_attempts(cell=cell):>6} effective "
                f"attempts pooled   {starved}/{total} cycles with zero"
            )

        print("\nfinal-checkpoint scores, pooled over seeds, and the within-ledge gap\n")
        for family in (None, "TRASH", "RECYCLING", "EMPTY"):
            name = "OVERALL" if family is None else family
            print(f"  {name}")
            for ledge in _LEDGES:
                for budget in _BUDGETS:
                    cells_text = []
                    for policy in _POLICIES:
                        final = ResetFreeCycleBudget.pooled_final(
                            cell=cells[(budget, ledge, policy)], family=family
                        )
                        cells_text.append(
                            f"{policy} "
                            f"{ResetFreeCycleBudget.format_count(solved=final[0], total=final[1])}"
                        )
                    gaps = ResetFreeCycleBudget.paired_gaps(
                        cells=cells, budget=budget, ledge=ledge, family=family
                    )
                    test = PairedTests.sign_flip(differences=gaps)
                    worse = sum(1 for gap in gaps if gap > 0)
                    print(
                        f"    {ledge:>8} {budget:>3}  {'  '.join(cells_text)}"
                        f"   gap {int(sum(gaps))}"
                        f"   never worse on {worse}/{len(gaps)} seeds"
                        f"   (tied {test.num_zero_differences}/{len(gaps)})"
                        f"   exact paired sign-flip p = {test.p_value:.4g}"
                    )
                changes = ResetFreeCycleBudget.paired_gap_changes(
                    cells=cells, ledge=ledge, family=family
                )
                change_test = PairedTests.sign_flip(differences=changes)
                closed = sum(1 for change in changes if change > 0)
                print(
                    f"    {ledge:>8}  gap CHANGE (1x minus 10x): {int(sum(changes))} pooled"
                    f"   closed on {closed}/{len(changes)} seeds"
                    f"   (tied {change_test.num_zero_differences}/{len(changes)})"
                    f"   exact paired sign-flip p = {change_test.p_value:.4g}"
                )
                if change_test.p_value > 0.05:
                    mde = PairedTests.minimum_detectable_effect(differences=changes)
                    print(
                        f"    {ledge:>8}  null result on the change; smallest per-seed "
                        f"change this design had 80% power to detect: {mde:.2f} tasks"
                    )
            print()

    # ------------------------------------------------------------------ the figures

    @staticmethod
    def render_curves(*, cells: dict, output: Path, title: str):
        """Three rows (OVERALL / TRASH / RECYCLING) x two columns (one-way / two-way),
        four curves per panel, bold pooled mean over faint per-seed lines.

        The x axis is **online transitions**, shared within a row, so the 1x curve is
        visibly the first tenth of the 10x one's budget rather than a separate experiment.
        Per-seed lines are drawn first and are the point: on this domain the one-way
        reset-free arm is bimodal, so its mean describes none of its seeds.
        """
        families = ((None, "all test tasks"), ("TRASH", "TRASH tasks"), ("RECYCLING", "RECYCLING"))
        fig, axes_grid = plt.subplots(3, 2, figsize=(13.6, 12.0), squeeze=False)
        for row, (family, family_name) in enumerate(families):
            for column, ledge in enumerate(_LEDGES):
                axes = axes_grid[row][column]
                seed_total = 0
                for budget in _BUDGETS:
                    for policy in _POLICIES:
                        cell = cells[(budget, ledge, policy)]
                        color, linestyle = ResetFreeCycleBudget.style(budget=budget, policy=policy)
                        xs = ResetFreeCycleBudget.transitions(cell=cell)
                        seeds = sorted(cell)
                        for seed in seeds:
                            entry = ResetFreeCycleBudget.series(cell=cell, seed=seed, family=family)
                            seed_total = entry[-1][1]
                            axes.plot(
                                xs,
                                [solved for solved, _ in entry],
                                color=color,
                                alpha=0.16,
                                linewidth=0.8,
                            )
                        pooled = ResetFreeCycleBudget.pooled_curve(cell=cell, family=family)
                        # Per-seed lines are counts out of `seed_total`; the pooled line is
                        # out of the pooled total. Scaling it back onto the per-seed axis is
                        # what lets one y axis carry both without either meaning the wrong
                        # thing.
                        scale = seed_total / pooled[-1][1]
                        final = pooled[-1]
                        name = ResetFreeCycleBudget.label(budget=budget, policy=policy)
                        count = ResetFreeCycleBudget.format_count(solved=final[0], total=final[1])
                        axes.plot(
                            xs,
                            [solved * scale for solved, _ in pooled],
                            color=color,
                            linestyle=linestyle,
                            linewidth=2.3,
                            label=f"{name} — {count}",
                        )
                axes.grid(alpha=0.25, linewidth=0.6)
                axes.set_ylim(-seed_total * 0.04, seed_total * 1.06)
                axes.set_xlabel("online transitions")
                axes.set_ylabel(f"solved per seed (x/{seed_total})", fontsize=9)
                axes.set_title(f"{family_name} — {ledge} ledge", fontsize=10)
                axes.legend(fontsize=7.5, loc="lower right", framealpha=0.95)
        fig.suptitle(title, fontsize=11.5)
        fig.tight_layout()
        fig.savefig(output, dpi=150)
        print(f"wrote {output}")
        return fig

    @staticmethod
    def render_gap(*, cells: dict, output: Path, title: str):
        """The figure the question asks: does the reset-free gap close when the budget
        grows tenfold?

        One panel per (ledge, family). Each seed is one line joining its own
        `scheduled - never` gap at 1x to its gap at 10x, so a closing gap is ten lines
        sloping toward zero and a persistent one is ten flat lines. Per-seed rather than a
        bar chart of two means, because with ten seeds a mean can be carried entirely by
        one of them -- which has already happened on this domain.
        """
        families = (
            (None, "all tasks, x/30 per seed"),
            ("TRASH", "TRASH, x/14"),
            ("RECYCLING", "RECYCLING, x/14"),
        )
        fig, axes_grid = plt.subplots(2, 3, figsize=(13.6, 8.0), squeeze=False)
        for row, ledge in enumerate(_LEDGES):
            for column, (family, family_name) in enumerate(families):
                axes = axes_grid[row][column]
                at_1x = ResetFreeCycleBudget.paired_gaps(
                    cells=cells, budget="1x", ledge=ledge, family=family
                )
                at_10x = ResetFreeCycleBudget.paired_gaps(
                    cells=cells, budget="10x", ledge=ledge, family=family
                )
                for before, after in zip(at_1x, at_10x, strict=True):
                    axes.plot(
                        [0, 1],
                        [before, after],
                        color="#D55E00" if after < before else "#0072B2",
                        alpha=0.5,
                        linewidth=1.2,
                        marker="o",
                        markersize=4,
                    )
                mean_before = sum(at_1x) / len(at_1x)
                mean_after = sum(at_10x) / len(at_10x)
                changes = [b - a for b, a in zip(at_1x, at_10x, strict=True)]
                test = PairedTests.sign_flip(differences=changes)
                closed = sum(1 for change in changes if change > 0)
                axes.plot(
                    [0, 1],
                    [mean_before, mean_after],
                    color="black",
                    linewidth=2.6,
                    marker="s",
                    markersize=7,
                    label=f"mean, closed on {closed}/{len(changes)} seeds",
                )
                axes.axhline(0.0, color="#666666", linewidth=1.0, linestyle=":")
                axes.set_xticks([0, 1])
                axes.set_xticklabels(["10 cycles (1x)", "100 cycles (10x)"])
                axes.set_xlim(-0.25, 1.25)
                axes.grid(alpha=0.25, linewidth=0.6, axis="y")
                axes.set_ylabel("gap: scheduled minus never, per seed", fontsize=9)
                axes.set_title(
                    f"{ledge} ledge — {family_name}\nexact paired sign-flip on the change: "
                    f"p = {test.p_value:.4g}",
                    fontsize=9.5,
                )
                axes.legend(fontsize=8, loc="best", framealpha=0.95)
        fig.suptitle(title, fontsize=11.5)
        fig.tight_layout()
        fig.savefig(output, dpi=150)
        print(f"wrote {output}")
        return fig

    # ------------------------------------------------------------------ entry point

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--cell",
            action="append",
            required=True,
            metavar="BUDGET:LEDGE:POLICY=DIR",
            help="e.g. 10x:one-way:never=results/rf10x/oneway-never/ees . DIR holds "
            "<seed>/stats.json. All eight cells are required.",
        )
        parser.add_argument("--curves-output", type=Path, required=True)
        parser.add_argument("--gap-output", type=Path, required=True)
        args = parser.parse_args()

        directories = {}
        for spec in args.cell:
            key, _, path = spec.partition("=")
            budget, ledge, policy = key.split(":")
            directories[(budget, ledge, policy)] = Path(path)
        cells = ResetFreeCycleBudget.load_cells(directories=directories)
        ResetFreeCycleBudget.print_report(cells=cells)

        domain = "Tossing Room (split throws, weight drawn at pickup), EES, 10 fixed seeds"
        ResetFreeCycleBudget.render_gap(
            cells=cells,
            output=args.gap_output,
            title=f"{domain}\nDoes ten times the practice budget close the reset-free gap?",
        )
        ResetFreeCycleBudget.render_curves(
            cells=cells,
            output=args.curves_output,
            title=f"{domain}\nreset-free practice at 10 cycles (1x) against 100 cycles (10x)",
        )


if __name__ == "__main__":
    ResetFreeCycleBudget.main()

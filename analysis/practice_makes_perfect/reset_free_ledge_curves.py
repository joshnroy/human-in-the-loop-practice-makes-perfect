"""Post-run analysis for reset-free practice on Tossing Room: **does the domain's one
irreversible action explain why practice without a free reset is worse?**

**Background.** `--practice-reset-policy scheduled` puts the environment back to a
freshly-sampled train task at the top of each practice period; `never` lets practice
state run continuously across period boundaries, which is the real-robot condition -- a
robot practising in a lab is not teleported to a fresh start every few minutes. The A/B
between them was first measured with the ledge out of room 2 one-way: rooms 0-2 are
severed from the item pile in room 3, so a practice period that walks left once can never
pick anything up again, and under `never` that damage carries into every later period.
`--two-way-ledge` makes the ledge traversable rightward too, removing the domain's only
irreversible action and nothing else about the reset policy.

So this reads back a 2x2 of **ledge** (`one-way`, `two-way`) x **policy** (`scheduled`,
`never`), all on `tossingroomsplitpickupweight`, and the comparison that carries is the
**gap within a ledge condition**.

**Levels do not compare across ledge conditions, only gaps do.** Turning the flag on also
stops EMPTY being an ordering task (its shortest solve drops 10 -> 9, so the evaluation
horizon drops 12 -> 11) and stops RECYCLING being one-attempt-per-period, so the two-way
world is a genuinely easier domain. A two-way count placed beside a one-way one is
therefore not a like-for-like comparison. The reset-free penalty is a within-ledge
difference, so both of its terms carry the same domain difficulty and it cancels. Nothing
below ever subtracts a two-way count from a one-way one.

**Three figures, two panels each, all four arms on every panel.** Overall task success
(`x/300` pooled), then TRASH (`x/140`) and RECYCLING (`x/140`) separately -- because the
pooled curve cannot show *which* tasks a policy loses, and on this domain the two throw
families behave completely differently under the one-way ledge. Bold is the mean over the
ten fixed seeds, faint is one line per seed, because a mean over ten seeds can describe
none of them when an arm is bimodal.

**The second panel is effective practice attempts, not cycles, and that is a measured
choice.** A cycles axis was the first thing tried, on the theory that a stranded robot's
practice periods end early (the `Method` raises `InteractionComplete` and untaken steps
are not charged), giving equal cycles but fewer transitions. **That does not happen in
this data:** all 40 runs charge exactly 150.0 transitions per cycle, so cycles is
`transitions / 150` -- a pure rescale that would redraw the identical curve with different
tick labels, and would imply a corroboration it cannot supply. `mean_transitions_per_cycle`
computes the rate rather than assuming it, and `print_report` prints it, so the day a
period *does* end early this stops being true loudly rather than silently.

The starvation is real, just on a different axis: the one-way reset-free arm spends its
full 150 transitions per cycle **walking**. Counting only the skills that need the robot
at the pile (`Pickup*`, `Throw*`) it manages **207** attempts pooled against the
scheduled-reset arm's **1191**, with **85/100** cycles attempting not one. Under the
two-way ledge the two arms are level (4021 against 3982, 0/100 starved each). That is what
the second panel plots.

**EMPTY gets no figure.** It is 2 tasks per seed, 20 pooled per arm, and it is solved
20/20 in every one of the four arms -- a denominator that supports almost no inference,
and a ceiling that would read as a result if it were plotted beside the throw families.

**Statistics.** All four arms ran the same fixed seeds, so every comparison is *paired*
and the test is `PairedTests.sign_flip`, exact by enumerating its null in full -- no
normal approximation, no continuity or tie correction, no scipy. It is imported from
`paired_tests` rather than reimplemented: a second copy of a hand-rolled significance
test is exactly how a sign error gets published -- and there genuinely were two copies,
which had already diverged. The same goes for goal classification, which comes from
`goal_families.GoalFamilies` (adding another classifier is how a denominator drifts).

**No minimum-detectable-effect is reported here.** The two merged conventions for it on
this project -- pooled and unpooled -- disagree materially on these rows, and the paired
sign-flip p-values are convention-free, so the figures and the exact tests are quoted
instead.

Reads only already-produced output (CLAUDE.md's `analysis/` convention -- this never runs
a simulation or drives a `Method`). Each `--arm` points at the directory holding that
arm's `<seed>/stats.json`, because the two committed sweeps nest differently
(`<policy>/<seed>` for one, `<policy>/ees/<seed>` for the other) and guessing between
them is how the wrong sweep gets read.
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

# The 2x2, in the order every table, legend and report below uses: the incumbent first in
# each factor. `one-way` is the ledge every banked number on this project was measured
# under, and `scheduled` is the only reset policy that existed before this experiment.
_LEDGES = ("one-way", "two-way")
_POLICIES = ("scheduled", "never")

# The composition the domain allocates for --num-test-tasks 30. `--two-way-ledge` does
# not move it, which is what keeps the two ledge conditions' evaluation sets comparable
# in COMPOSITION even though the flag makes the two-way one easier. Asserted per sweep,
# because a goal misfiled between families moves tasks between denominators invisibly.
_COMPOSITION = {"TRASH": 14, "RECYCLING": 14, "EMPTY": 2}
_NUM_TEST_TASKS = sum(_COMPOSITION.values())

# `num_practice_resets` is a *measurement* of resets as they happened rather than a
# restatement of the flag, so it is the manipulation check.
_EXPECTED_RESETS = {"scheduled": 10, "never": 0}

# Four colours, validated (not eyeballed) for colourblind separation and for contrast
# against both a light and a dark chart surface. Linestyle carries the reset policy as
# well, so an arm's identity never rests on hue alone.
_COLORS = {
    ("one-way", "scheduled"): "#0072B2",
    ("one-way", "never"): "#D55E00",
    ("two-way", "scheduled"): "#009E73",
    ("two-way", "never"): "#785EF0",
}
_LINESTYLES = {"scheduled": "-", "never": "--"}

# Display names for the two `--practice-reset-policy` values. DISPLAY ONLY: the keys,
# directory names and flag values stay `scheduled`/`never`, because those are what the
# CLI accepts and what every committed `config_snapshot.json` already records.
_POLICY_DISPLAY = {
    "scheduled": "practice-session env resets",
    "never": "never env reset",
}

# Skills that require the robot to be at the item pile. A stranded robot can still walk
# (`MoveRoom`) and press buttons (`Press*`) for a whole period, so those are excluded --
# counting them would report a starved arm as busy.
_EFFECTIVE_PREFIXES = ("Pickup", "Throw")


class ResetFreeLedgeCurves:
    """A static-method container, never instantiated."""

    # ------------------------------------------------------------------ the square

    @staticmethod
    def arms() -> tuple[tuple[str, str], ...]:
        """The four (ledge, policy) cells, in report order."""
        return tuple((ledge, policy) for ledge in _LEDGES for policy in _POLICIES)

    @staticmethod
    def style(*, ledge: str, policy: str) -> tuple[str, str]:
        """The (colour, linestyle) this arm is drawn with, on every figure."""
        return _COLORS[(ledge, policy)], _LINESTYLES[policy]

    @staticmethod
    def label(*, ledge: str, policy: str) -> str:
        """The legend entry, which is where the ledge/policy pairing has to be legible.

        Uses the display name for the policy, never the flag value -- see
        `_POLICY_DISPLAY`."""
        return f"{ledge} ledge, {_POLICY_DISPLAY[policy]}"

    @staticmethod
    def format_count(*, solved: int, total: int) -> str:
        """`x/y`, never a bare percentage: the denominators here are small and uneven."""
        return f"{solved}/{total}"

    # ------------------------------------------------------------------ reading back

    @staticmethod
    def load_arms(*, directories: dict[tuple[str, str], Path]) -> dict[tuple[str, str], dict]:
        """Every arm's per-seed, per-checkpoint family counts, with the checks first.

        The four arms are a square, not four strings: a missing cell means the
        within-ledge gap is undefined under one of the two ledge conditions, and a report
        that silently printed the comparison it *could* still make would read as a result.
        """
        missing = [arm for arm in ResetFreeLedgeCurves.arms() if arm not in directories]
        if missing:
            names = ", ".join(f"{ledge}/{policy}" for ledge, policy in missing)
            raise ValueError(
                f"missing arm(s): {names}. This experiment is a 2x2 of ledge x policy; "
                "with a cell absent the within-ledge gap is not defined under both "
                "conditions and no comparison here is meaningful."
            )
        return {
            arm: ResetFreeLedgeCurves.load_arm(directory=directories[arm], policy=arm[1])
            for arm in ResetFreeLedgeCurves.arms()
        }

    @staticmethod
    def load_arm(*, directory: Path, policy: str) -> dict:
        """One arm: `{seed: {"transitions": [...], "families": {family: [(solved, total)]}}}`."""
        seeds = sorted(
            (int(path.parent.name) for path in directory.glob("*/stats.json")),
        )
        if not seeds:
            raise ValueError(f"no <seed>/stats.json under {directory}")
        arm: dict[int, dict] = {}
        for seed in seeds:
            stats = json.loads((directory / str(seed) / "stats.json").read_text())
            resets = stats.get("num_practice_resets")
            if resets != _EXPECTED_RESETS[policy]:
                raise ValueError(
                    f"{directory}/{seed}: num_practice_resets is {resets}, expected "
                    f"{_EXPECTED_RESETS[policy]} for --practice-reset-policy {policy}. "
                    "The arm is not what its name says."
                )
            transitions = []
            families: dict[str, list[tuple[int, int]]] = {family: [] for family in _COMPOSITION}
            overall: list[tuple[int, int]] = []
            for breakdown in stats["breakdowns"]:
                transitions.append(breakdown["num_online_transitions"])
                counts = ResetFreeLedgeCurves.sweep_counts(outcomes=breakdown["outcomes"])
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
            arm[seed] = {
                "transitions": transitions,
                "families": families,
                "overall": overall,
                "effective_attempts": ResetFreeLedgeCurves.cumulative_effective_attempts(
                    windows=stats["practice_outcomes_per_cycle"],
                    num_checkpoints=len(transitions),
                ),
            }
        return arm

    @staticmethod
    def cumulative_effective_attempts(*, windows: list[dict], num_checkpoints: int) -> list[int]:
        """Effective practice attempts accumulated *before* each evaluation checkpoint.

        "Effective" means a skill that needs the robot to be at the item pile -- the
        `Pickup*` and `Throw*` families. `MoveRoom` and the `Press*` skills are excluded
        deliberately: a stranded robot can still walk and press buttons all period, so
        counting those would report a starved arm as busy.

        Checkpoint 0 is taken before any practice, so it accumulates nothing; checkpoint
        `i` accumulates practice windows `0..i-1`. `practice_outcomes_per_cycle` carries
        one trailing window past the last cycle (it records zero attempts), which this
        slicing drops rather than mis-attributing to a checkpoint that never saw it.
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

    @staticmethod
    def mean_transitions_per_cycle(*, arm: dict) -> float:
        """Online transitions charged per practice cycle, averaged over seeds.

        This is the quantity a "cycles" axis would rest on. A period that ends early
        (the `Method` raises `InteractionComplete`) contributes only the steps it
        actually took, so a starved arm *could* show fewer transitions per cycle -- see
        `PracticeLoop`'s docstring. Whether it did is a measurement, not an assumption,
        which is why this is computed and printed rather than asserted."""
        rates = []
        for seed in sorted(arm):
            transitions = arm[seed]["transitions"]
            num_cycles = len(transitions) - 1
            rates.append((transitions[-1] - transitions[0]) / num_cycles)
        return sum(rates) / len(rates)

    @staticmethod
    def sweep_counts(*, outcomes: list[dict]) -> dict[str, tuple[int, int]]:
        """One sweep's `(solved, total)` per family.

        Classification is `GoalFamilies.classify`, reused rather than recopied. It
        tests the `BinEmpty` predicate before the item names, because `Goal.describe()`
        renders EMPTY as "RecyclingBinEmpty(recycling_bin) & TrashBinEmpty(trash_bin)" --
        it names BOTH bins, so a naive "does it mention recycling?" test swallows it and
        silently reports 16 RECYCLING / 0 EMPTY. An unrecognised goal raises there.
        """
        solved: Counter[str] = Counter()
        total: Counter[str] = Counter()
        for outcome in outcomes:
            family = GoalFamilies.classify(goal=outcome["goal"])
            total[family] += 1
            solved[family] += int(outcome["solved"])
        return {family: (solved[family], total[family]) for family in total}

    # ------------------------------------------------------------------ arithmetic

    @staticmethod
    def pooled_curve(*, arm: dict, family: str | None) -> list[tuple[int, int]]:
        """The arm's curve pooled over seeds: solved and total both SUMMED, per
        checkpoint.

        Summed rather than averaged, so `x/140` at ten seeds means what it says. A mean
        of per-seed rates would silently reweight a seed that ran a different number of
        tasks."""
        seeds = sorted(arm)
        key = "overall" if family is None else family
        num_checkpoints = len(arm[seeds[0]]["transitions"])
        pooled = []
        for index in range(num_checkpoints):
            solved = 0
            total = 0
            for seed in seeds:
                entry = arm[seed]["overall"] if family is None else arm[seed]["families"][key]
                solved += entry[index][0]
                total += entry[index][1]
            pooled.append((solved, total))
        return pooled

    @staticmethod
    def transitions(*, arm: dict) -> list[int]:
        """The shared x axis: the checkpoint transition counts, which every seed shares."""
        seeds = sorted(arm)
        grids = {tuple(arm[seed]["transitions"]) for seed in seeds}
        if len(grids) != 1:
            raise ValueError(
                f"seeds disagree on the evaluation checkpoints ({sorted(grids)}), so they "
                "cannot share an x axis."
            )
        return list(next(iter(grids)))

    @staticmethod
    def paired_final_differences(*, arms: dict, ledge: str, family: str | None) -> list[float]:
        """`scheduled` minus `never` at the final checkpoint, **within a seed**.

        The arms share seeds, so this is paired data. Zero differences are kept rather
        than dropped -- "9/10 seeds differ by exactly zero" is the two-way cell's whole
        headline and is invisible if ties are discarded."""
        scheduled = arms[(ledge, "scheduled")]
        never = arms[(ledge, "never")]
        seeds = sorted(set(scheduled) & set(never))
        differences = []
        for seed in seeds:
            key = "overall" if family is None else family
            left = (
                scheduled[seed]["overall"] if family is None else scheduled[seed]["families"][key]
            )
            right = never[seed]["overall"] if family is None else never[seed]["families"][key]
            differences.append(float(left[-1][0] - right[-1][0]))
        return differences

    # ------------------------------------------------------------------ the report

    @staticmethod
    def print_report(*, arms: dict) -> None:
        """Every number the write-up quotes, as `x/y`, re-derived here."""
        print("practice budget actually spent, per arm\n")
        for ledge, policy in ResetFreeLedgeCurves.arms():
            arm = arms[(ledge, policy)]
            rate = ResetFreeLedgeCurves.mean_transitions_per_cycle(arm=arm)
            effective = sum(arm[seed]["effective_attempts"][-1] for seed in sorted(arm))
            starved = sum(
                1
                for seed in sorted(arm)
                for before, after in zip(
                    arm[seed]["effective_attempts"][:-1],
                    arm[seed]["effective_attempts"][1:],
                    strict=True,
                )
                if after == before
            )
            cycles = sum(len(arm[seed]["effective_attempts"]) - 1 for seed in sorted(arm))
            print(
                f"  {ledge:>8} {policy:>9}   {rate:6.1f} transitions/cycle"
                f"   {effective:>5} effective attempts pooled"
                f"   {starved}/{cycles} cycles with zero"
            )
        print(
            "\n  A cycles axis would be transitions/rate. Every arm above charges the "
            "same\n  rate, so cycles is a pure rescale of transitions here and is not "
            "plotted.\n"
        )
        print("final-checkpoint scores, pooled over seeds\n")
        for family in (None, "TRASH", "RECYCLING", "EMPTY"):
            name = "OVERALL" if family is None else family
            print(f"  {name}")
            for ledge in _LEDGES:
                cells = []
                for policy in _POLICIES:
                    final = ResetFreeLedgeCurves.pooled_curve(
                        arm=arms[(ledge, policy)], family=family
                    )[-1]
                    cells.append(
                        f"{policy} "
                        f"{ResetFreeLedgeCurves.format_count(solved=final[0], total=final[1])}"
                    )
                differences = ResetFreeLedgeCurves.paired_final_differences(
                    arms=arms, ledge=ledge, family=family
                )
                test = PairedTests.sign_flip(differences=differences)
                worse = sum(1 for d in differences if d > 0)
                print(
                    f"    {ledge:>8}  {'  '.join(cells)}"
                    f"   gap {int(sum(differences))}"
                    f"   never worse on {worse}/{len(differences)} seeds"
                    f"   (tied {test.num_zero_differences}/{len(differences)})"
                    f"   exact paired sign-flip p = {test.p_value:.4g}"
                )
            print()

    # ------------------------------------------------------------------ the figures

    @staticmethod
    def render(*, arms: dict, family: str | None, output: Path, title: str, legend_loc: str):
        """Two panels sharing a y axis, all four arms on each, bold pooled mean over
        faint per-seed lines.

        **Left: online transitions. Right: cumulative effective practice attempts.**

        The right panel is deliberately *not* a cycles axis. Cycles were measured first,
        and in all 40 runs a cycle charges exactly 150 transitions -- no period ended
        early anywhere in this data -- so a cycles axis is `transitions / 150`, a pure
        rescale that would draw the identical curve with different tick labels. Putting
        that beside the transitions panel would imply a corroboration it cannot supply.

        What *is* true is that the arms are starved of **useful** experience rather than
        of steps: a stranded robot spends its full 150 transitions walking. So the second
        axis counts the practice attempts that need the robot at the pile, which is the
        starvation the equal-transitions grid hides."""
        fig, (ax_transitions, ax_effective) = plt.subplots(1, 2, figsize=(13.6, 5.4), sharey=True)
        seed_total = 0
        for ledge, policy in ResetFreeLedgeCurves.arms():
            arm = arms[(ledge, policy)]
            color, linestyle = ResetFreeLedgeCurves.style(ledge=ledge, policy=policy)
            xs = ResetFreeLedgeCurves.transitions(arm=arm)
            seeds = sorted(arm)
            # Every seed as a thin line first: the spread is the point. On this domain
            # the one-way `never` arm is bimodal, so its mean describes none of its seeds.
            for seed in seeds:
                entry = arm[seed]["overall"] if family is None else arm[seed]["families"][family]
                seed_total = entry[-1][1]
                ys = [solved for solved, _ in entry]
                ax_transitions.plot(xs, ys, color=color, alpha=0.22, linewidth=0.9)
                ax_effective.plot(
                    arm[seed]["effective_attempts"], ys, color=color, alpha=0.22, linewidth=0.9
                )
            pooled = ResetFreeLedgeCurves.pooled_curve(arm=arm, family=family)
            # Per-seed lines are counts out of `seed_total`; the pooled line is out of
            # `total`. Scaling the pooled line back onto the per-seed axis is what lets
            # one y axis carry both without either meaning the wrong thing.
            scale = seed_total / pooled[-1][1]
            pooled_ys = [solved * scale for solved, _ in pooled]
            final = pooled[-1]
            rate = ResetFreeLedgeCurves.mean_transitions_per_cycle(arm=arm)
            label = (
                f"{ResetFreeLedgeCurves.label(ledge=ledge, policy=policy)} — "
                f"{ResetFreeLedgeCurves.format_count(solved=final[0], total=final[1])}"
                f"  ({rate:.1f} transitions/cycle)"
            )
            ax_transitions.plot(
                xs, pooled_ys, color=color, linestyle=linestyle, linewidth=2.4, label=label
            )
            # The pooled line's x on the effective axis is the per-seed mean, so a point
            # means "at this many effective attempts on average, this many tasks solved".
            mean_effective = [
                sum(arm[seed]["effective_attempts"][i] for seed in seeds) / len(seeds)
                for i in range(len(xs))
            ]
            ax_effective.plot(
                mean_effective, pooled_ys, color=color, linestyle=linestyle, linewidth=2.4
            )

        ax_transitions.set_xlabel("online transitions")
        ax_effective.set_xlabel(
            "cumulative effective practice attempts (`Pickup*` + `Throw*`), mean over seeds"
        )
        for axes in (ax_transitions, ax_effective):
            axes.grid(alpha=0.25, linewidth=0.6)
            axes.set_ylim(-seed_total * 0.04, seed_total * 1.06)
        ax_transitions.set_ylabel(
            f"test tasks solved per seed (x/{seed_total});  legend gives the pooled count",
            fontsize=9,
        )
        ax_transitions.legend(fontsize=8.5, loc=legend_loc, framealpha=0.95)
        fig.suptitle(title, fontsize=10.5)
        fig.tight_layout()
        fig.savefig(output, dpi=160)
        print(f"wrote {output}")
        return fig

    # ------------------------------------------------------------------ entry point

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--arm",
            action="append",
            required=True,
            metavar="LEDGE:POLICY=DIR",
            help="e.g. one-way:never=docs/experiment-logs/.../never . DIR holds "
            "<seed>/stats.json. All four cells are required.",
        )
        parser.add_argument("--overall-output", type=Path, required=True)
        parser.add_argument("--trash-output", type=Path, required=True)
        parser.add_argument("--recycling-output", type=Path, required=True)
        args = parser.parse_args()

        directories = {}
        for spec in args.arm:
            key, _, path = spec.partition("=")
            ledge, _, policy = key.partition(":")
            directories[(ledge, policy)] = Path(path)
        arms = ResetFreeLedgeCurves.load_arms(directories=directories)
        ResetFreeLedgeCurves.print_report(arms=arms)

        domain = "Tossing Room (split throws, weight drawn at pickup), EES"
        # Figure order follows the argument, not the data model: the two throw families
        # carry the mechanism and the pooled curve is the summary of them.
        for family, output, name, legend_loc in (
            ("TRASH", args.trash_output, "TRASH tasks, x/140", "lower right"),
            ("RECYCLING", args.recycling_output, "RECYCLING tasks, x/140", "center left"),
            (None, args.overall_output, "all test tasks, x/300", "lower right"),
        ):
            ResetFreeLedgeCurves.render(
                arms=arms,
                family=family,
                output=output,
                title=f"{domain}\n{name}: scheduled-reset vs reset-free practice",
                legend_loc=legend_loc,
            )


if __name__ == "__main__":
    ResetFreeLedgeCurves.main()

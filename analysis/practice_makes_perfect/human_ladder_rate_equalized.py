"""Rate-equalized companion to `human_ladder_curves.py`: does `on-stuck` still beat
`at-random` once `at-random` is retuned to spend at `on-stuck`'s own rate?

**Background.** The human-in-the-loop ladder (`human_ladder_curves.py`,
`docs/experiment-logs/2026-08-07-human-ladder.md`) found `on-stuck` beating
`at-random` at the shared `--mean-steps-between-help-requests 150` default -- but
`on-stuck` spent 3.4x the rescues (347/337 vs 101/101 over ten seeds), so that
contrast was timing AND rate together, not timing alone. Priced per rescue, the
ranking **inverted**: `at-random-initial` bought 0.446 extra solves per rescue
against `stuck-initial`'s 0.331, so a claim that stuck-detection is a more
*efficient* use of a human was not supported by that data, even though its own
docstring said it was. This module re-derives the comparison with `at-random`
retuned so its expected rescue rate matches what `on-stuck` actually spent.

**How the matched rate was chosen.** `on-stuck` and `at-random` both draw from a
fixed budget of `num_cycles * max_steps_per_interaction` = 1500 policy calls per
seed, always -- a granted rescue consumes a call without producing a transition
(`PracticeLoop`'s `except HumanHelpRequested` continues), so a rescued seed's final
transition count plus its rescue count always sums to exactly 1500 (checked in the
original ladder's own data, not assumed). `on-stuck`'s realised rate is therefore
`total_rescues / (10 * 1500)`, and `--mean-steps-between-help-requests` is the
reciprocal of a per-call probability, so matching it is
`round(15000 / total_rescues)`:

    stuck-initial: 347 rescues over 15000 calls -> 15000/347 = 43.23 -> 43
    stuck-random:  337 rescues over 15000 calls -> 15000/337 = 44.51 -> 45

Two new arms carry those values: `at-random-initial-matched` (paired against
`stuck-initial`) and `at-random-random-matched` (paired against `stuck-random`).
Everything else about them is identical to the unmatched `at-random-initial`/
`at-random-random` arms (same `--human-reset-target`, same ten seeds, same
`--practice-reset-policy never` one-way world) -- only the request rate moved, so
a gap between matched and unmatched isolates exactly that.

**This does not touch `_ARMS` in `human_ladder_curves.py`.** That module's
`load_arms` requires all eight of its own named arms and is the reference report
for the original, unmatched ladder -- extending its registry to a ninth and tenth
arm would change what "the eight arms" means for every existing reader of that
file. This module instead calls `HumanLadderCurves.load_arm` (singular, already a
`directory + arm name -> per-seed dict` function with no dependency on the
eight-arm registry) once per arm it needs, and reuses `entry`,
`paired_final_differences` and `solves_per_rescue` unchanged -- same per-seed data
shape, same statistics, no reimplementation.

**Statistics.** Same exact paired sign-flip test as the original report
(`PairedTests.sign_flip`, imported rather than reimplemented), since all five arms
here share the same ten fixed seeds.

Reads only already-produced output (CLAUDE.md's `analysis/` convention -- this
never runs a simulation or drives a `Method`). Five `--arm` directories are
required: `no-human`, `stuck-initial`, `stuck-random` (reused unchanged from the
original ladder run -- their config does not read
`--mean-steps-between-help-requests` at all, so re-running them would reproduce
them byte-for-byte, which was checked directly) plus the two new matched arms.
"""

import argparse
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from analysis.practice_makes_perfect.human_ladder_curves import HumanLadderCurves  # noqa: E402
from analysis.practice_makes_perfect.paired_tests import PairedTests  # noqa: E402

_NUM_TEST_TASKS = 30

# (unmatched at-random arm, matched at-random arm, its stuck pair, the matched
# --mean-steps-between-help-requests value, --human-reset-target). Two pairs, one per
# cell of the {task-initial, random} axis -- see the module docstring for how the
# matched value was derived from the original ladder's own rescue counts.
_PAIRS = (
    ("at-random-initial", "at-random-initial-matched", "stuck-initial", 43, "task-initial"),
    ("at-random-random", "at-random-random-matched", "stuck-random", 45, "random"),
)

# Every arm this module ever reads. `no-human` is the shared control.
_ALL_ARMS = ("no-human",) + tuple(name for pair in _PAIRS for name in (pair[0], pair[1], pair[2]))

_COLORS = {
    "no-human": "#0072B2",
    "stuck-initial": "#D55E00",
    "stuck-random": "#009E73",
    "at-random-initial": "#E69F00",
    "at-random-random": "#56B4E9",
    "at-random-initial-matched": "#CC6600",
    "at-random-random-matched": "#3399CC",
}


class RateEqualizedComparison:
    """A static-method container, never instantiated -- same convention as
    `HumanLadderCurves`."""

    # ------------------------------------------------------------------ reading back

    @staticmethod
    def load_arms(*, directories: dict[str, Path]) -> dict[str, dict]:
        """Every arm this module needs, by `HumanLadderCurves.load_arm` unchanged.

        All of `_ALL_ARMS` are required -- with `no-human` absent there is no control
        to difference against, and with a matched or stuck arm absent the specific
        comparison it exists to make is undefined."""
        missing = [arm for arm in _ALL_ARMS if arm not in directories]
        if missing:
            raise ValueError(
                f"missing arm(s): {', '.join(missing)}. The rate-equalized comparison "
                "needs the control plus both stuck arms plus both matched at-random "
                "arms; with one absent, the comparison it exists to make is undefined."
            )
        return {
            arm: HumanLadderCurves.load_arm(directory=directories[arm], arm=arm)
            for arm in _ALL_ARMS
        }

    # ------------------------------------------------------------------ arithmetic

    @staticmethod
    def per_seed_solves_per_rescue(
        *, arms: dict, treatment: str, control: str
    ) -> tuple[list[float], list[int]]:
        """One ratio per seed that was rescued at least once: that seed's own
        (treatment final - control final) divided by that seed's own rescue count.

        A seed the treatment arm never rescued has an UNDEFINED ratio, not a zero --
        dividing by zero would be a bug, and silently reporting zero would understate
        the arm's efficiency on the seeds it never touched. Such seeds are excluded
        and returned separately (their seed numbers), so the exclusion is visible
        rather than quietly shrinking the sample.

        Per-seed rather than only the pooled `HumanLadderCurves.solves_per_rescue`
        ratio, because per-seed spread is what this project's figures are required to
        show (CLAUDE.md: "Plot per-seed spread rather than only a mean") -- a pooled
        ratio alone cannot distinguish ten seeds each buying a bit from one seed
        buying everything."""
        treat, ctrl = arms[treatment], arms[control]
        seeds = sorted(set(treat) & set(ctrl))
        ratios: list[float] = []
        excluded: list[int] = []
        for seed in seeds:
            rescues = treat[seed]["interventions"]
            if rescues == 0:
                excluded.append(seed)
                continue
            diff = (
                HumanLadderCurves.entry(arm=treat, seed=seed, family=None)[-1][0]
                - HumanLadderCurves.entry(arm=ctrl, seed=seed, family=None)[-1][0]
            )
            ratios.append(diff / rescues)
        return ratios, excluded

    # ------------------------------------------------------------------ the report

    @staticmethod
    def print_report(*, arms: dict) -> None:
        print("rate-equalized comparison: on-stuck vs at-random at MATCHED rescue rates\n")
        for unmatched, matched, stuck, mean_steps, target in _PAIRS:
            print(
                f"  target={target}  (matched vs {stuck}, "
                f"--mean-steps-between-help-requests {mean_steps})"
            )
            for label, arm_name in (
                (f"{stuck:>28}", stuck),
                (f"{unmatched:>28} (unmatched)", unmatched),
                (f"{matched:>28} (matched)", matched),
            ):
                seeds = sorted(arms[arm_name])
                rescues = [arms[arm_name][seed]["interventions"] for seed in seeds]
                pooled_solved, pooled_total = HumanLadderCurves.pooled_curve(
                    arm=arms[arm_name], family=None
                )[-1]
                ratio = HumanLadderCurves.solves_per_rescue(
                    arms=arms, treatment=arm_name, control="no-human"
                )
                shown = "n/a (never rescued)" if ratio is None else f"{ratio:.3f}"
                print(
                    f"    {label}  "
                    f"{HumanLadderCurves.format_count(solved=pooled_solved, total=pooled_total):>8}"
                    f"   {sum(rescues):>4} rescues (per-seed {min(rescues)}-{max(rescues)})"
                    f"   {shown} extra solves per rescue"
                )
            differences_vs_stuck = HumanLadderCurves.paired_final_differences(
                arms=arms, treatment=matched, control=stuck, family=None
            )
            test_vs_stuck = PairedTests.sign_flip(differences=differences_vs_stuck)
            better = sum(1 for d in differences_vs_stuck if d > 0)
            worse = sum(1 for d in differences_vs_stuck if d < 0)
            print(
                f"    {matched:>28} - {stuck:<28} gap {int(sum(differences_vs_stuck)):>+4}"
                f"   better {better}/10   worse {worse}/10"
                f"   tied {test_vs_stuck.num_zero_differences}/10   p = {test_vs_stuck.p_value:.4g}"
            )
            differences_vs_control = HumanLadderCurves.paired_final_differences(
                arms=arms, treatment=matched, control="no-human", family=None
            )
            test_vs_control = PairedTests.sign_flip(differences=differences_vs_control)
            better_c = sum(1 for d in differences_vs_control if d > 0)
            worse_c = sum(1 for d in differences_vs_control if d < 0)
            print(
                f"    {matched:>28} - {'no-human':<28} gap {int(sum(differences_vs_control)):>+4}"
                f"   better {better_c}/10   worse {worse_c}/10"
                f"   tied {test_vs_control.num_zero_differences}/10"
                f"   p = {test_vs_control.p_value:.4g}"
            )
            print()

    # ------------------------------------------------------------------ the figure

    @staticmethod
    def render(*, arms: dict, output: Path, title: str) -> None:
        """Two panels, one figure. **Left**: rescues charged per seed, for all six
        arms in `_PAIRS` order -- the manipulation check that the matched arms
        actually moved off the unmatched rate. **Right**: per-seed solves-per-rescue,
        the efficiency question itself, at the same six arms in the same order so a
        reader can look straight down between the two panels."""
        categories: list[tuple[str, str]] = []
        for unmatched, matched, stuck, _mean_steps, _target in _PAIRS:
            categories.append((stuck, stuck))
            categories.append((unmatched, f"{unmatched}\n(unmatched)"))
            categories.append((matched, f"{matched}\n(matched)"))

        fig, (ax_rescues, ax_ratio) = plt.subplots(1, 2, figsize=(14.5, 5.6))

        for index, (arm_name, _display) in enumerate(categories):
            seeds = sorted(arms[arm_name])
            counts = [arms[arm_name][seed]["interventions"] for seed in seeds]
            offsets = [(i % 5 - 2) * 0.06 for i in range(len(counts))]
            ax_rescues.scatter(
                [index + offset for offset in offsets],
                counts,
                color=_COLORS[arm_name],
                s=42,
                alpha=0.85,
                zorder=3,
            )
            ax_rescues.plot(
                [index - 0.26, index + 0.26],
                [statistics.mean(counts)] * 2,
                color=_COLORS[arm_name],
                linewidth=2.4,
            )
        ax_rescues.set_xticks(range(len(categories)))
        ax_rescues.set_xticklabels(
            [display for _, display in categories], rotation=20, ha="right", fontsize=7.5
        )
        ax_rescues.set_ylabel("human interventions charged to this seed", fontsize=9)
        ax_rescues.set_xlabel(
            "one dot per seed; bar is the per-seed mean\n"
            "manipulation check: does a *-matched arm actually sit near its stuck pair?",
            fontsize=8.5,
        )
        ax_rescues.grid(alpha=0.25, linewidth=0.6, axis="y")

        for index, (arm_name, _display) in enumerate(categories):
            ratios, excluded = RateEqualizedComparison.per_seed_solves_per_rescue(
                arms=arms, treatment=arm_name, control="no-human"
            )
            offsets = [(i % 5 - 2) * 0.06 for i in range(len(ratios))]
            ax_ratio.scatter(
                [index + offset for offset in offsets],
                ratios,
                color=_COLORS[arm_name],
                s=42,
                alpha=0.85,
                zorder=3,
            )
            if ratios:
                ax_ratio.plot(
                    [index - 0.26, index + 0.26],
                    [statistics.mean(ratios)] * 2,
                    color=_COLORS[arm_name],
                    linewidth=2.4,
                )
            if excluded:
                ax_ratio.annotate(
                    f"{len(excluded)} seed(s) never rescued (excluded)",
                    (index, 0.02),
                    xycoords=("data", "axes fraction"),
                    ha="center",
                    fontsize=6.5,
                    rotation=90,
                    va="bottom",
                )
        ax_ratio.axhline(0.0, color="black", linewidth=0.7, alpha=0.4)
        ax_ratio.set_xticks(range(len(categories)))
        ax_ratio.set_xticklabels(
            [display for _, display in categories], rotation=20, ha="right", fontsize=7.5
        )
        ax_ratio.set_ylabel(
            "extra test tasks solved per rescue, this seed\n"
            "(seeds never rescued by this arm are excluded, not zero)",
            fontsize=8.5,
        )
        ax_ratio.set_xlabel("one dot per seed; bar is the per-seed mean", fontsize=8.5)
        ax_ratio.grid(alpha=0.25, linewidth=0.6, axis="y")

        fig.suptitle(title, fontsize=10.5)
        fig.tight_layout()
        fig.savefig(output, dpi=160)
        print(f"wrote {output}")
        plt.close(fig)

    # ------------------------------------------------------------------ entry point

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--arm",
            action="append",
            required=True,
            metavar="NAME=DIR",
            help="e.g. stuck-initial=results/human-ladder/stuck-initial/ees . DIR holds "
            f"<seed>/stats.json. All of {', '.join(_ALL_ARMS)} are required.",
        )
        parser.add_argument("--output-dir", type=Path, required=True)
        args = parser.parse_args()

        directories = {}
        for spec in args.arm:
            name, _, path = spec.partition("=")
            directories[name] = Path(path)
        arms = RateEqualizedComparison.load_arms(directories=directories)
        RateEqualizedComparison.print_report(arms=arms)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        RateEqualizedComparison.render(
            arms=arms,
            output=args.output_dir / "human-ladder-rate-equalized.png",
            title=(
                "Tossing Room, reset-free practice\n"
                "on-stuck vs at-random, before and after matching the rescue rate"
            ),
        )


if __name__ == "__main__":
    RateEqualizedComparison.main()

"""Does vanilla EES plateau on Tossing3D, or keep climbing given five times the budget?

Post-run analysis only; it reads `--results-root` back in and never drives a `Method`.

## The question, and why the obvious reading of it is wrong

PR #133 measured `80/100` task success for `ees` on Tossing3D after **20** cycles. That
number cannot distinguish two very different worlds: the robot has reached the best it
will ever do, or it was still improving when the budget ran out. This reads a **100**-cycle
run to separate them.

The tempting way to answer is "compare the final sweep at 100 cycles to the final sweep at
20". That is unsound here, and measurably so. Tossing3D's per-seed score is **volatile
between adjacent sweeps** -- watched live on seed 0 of this very sweep, it went `8/10` at
sweep 10, `5/10` at sweep 13, `6/10` at sweep 15, with no learning event in between. A
single sweep is one draw of a noisy variable, so a final-sweep comparison measures the
draw as much as the policy.

## The rule, fixed before any final number was read

- **Score a window, not a sweep.** A seed's score in a window is its mean solved count
  over the `WINDOW` sweeps in it. Averaging ten sweeps of a settled process estimates its
  level with roughly `sqrt(10)` less noise than one sweep does.
- **`LATE` is the last `WINDOW` sweeps** -- where the robot ends up.
- **`REFERENCE` is the `WINDOW` sweeps starting at cycle 21**, i.e. the ten sweeps
  immediately after #133's whole budget. Same width as `LATE`, so the two are like for
  like, and positioned so the comparison asks exactly "did anything happen *after* the
  budget #133 had?".
- **Climb test:** exact paired sign-flip on the per-seed difference `LATE - REFERENCE`,
  over the ten fixed seeds. Paired because both windows come from the same run of the
  same seed; discarding that pairing would throw away the design.
- **A null result is reported as a null result, with its MDE beside it.** "Not
  significantly different" is not "identical", and without the minimum detectable effect a
  reader cannot tell a plateau from an underpowered test. Both are printed, always.

`LATE` deliberately understates the endpoint if the robot is genuinely still climbing --
it averages over the climb. That bias is in the conservative direction for a *plateau*
claim, and the final sweep and the per-seed maximum are reported alongside so a reader can
see the whole shape rather than one summary.
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.practice_makes_perfect.paired_tests import PairedTests  # noqa: E402

# Ten sweeps per window: wide enough to damp the measured sweep-to-sweep volatility,
# narrow enough that `REFERENCE` sits entirely inside the region #133 never saw.
WINDOW = 10
# 1-indexed cycle at which the reference window opens -- the sweep after #133's budget.
REFERENCE_START_CYCLE = 21

# PR #133's two non-learning arms, measured on this domain at 10 seeds x 10 test tasks.
# Drawn as horizontal REFERENCE LINES, never as curves: neither learns, so a "curve" for
# either would invite a reader to look for a trend in what is by construction a constant.
# random-skills consults no sampler at all and moved 26/100 -> 24/100 across #133's whole
# budget, i.e. flat within noise. Per seed, out of 10 tasks.
ORACLE_PER_SEED = 10.0
RANDOM_SKILLS_PER_SEED = 2.4
ORACLE_LABEL = "skill-oracle ceiling — 100/100 (#133)"
RANDOM_SKILLS_LABEL = "random-skills — 24/100 (#133)"

# CLAUDE.md's training-curve-style section (#188): reference/ceiling arms that are not the
# manipulation under test share one neutral grey, dotted -- distinguished from each other
# only by legend label/y-level, never by hue. `ees` (the one arm under test here) gets the
# spec blue; there is no reset-free counterpart on this figure (single policy, no reset
# axis), so orange is not used at all.
_EES_COLOUR = "#0072B2"
_REFERENCE_COLOUR = "#666666"


class Tossing3DPlateau:
    """A static-method container, never instantiated, same as every other
    business-logic class in this project."""

    @staticmethod
    def load_curves(*, results_root: Path) -> dict[int, list[tuple[int, int, int]]]:
        """`{seed: evaluations}`, where each entry is
        `(num_online_transitions, num_solved, num_total)` for one sweep.

        Read from `stats.json`, so only completed runs appear. A partially-finished run
        has no `stats.json` and is silently absent rather than contributing a truncated
        curve that would shorten every window computed from it.
        """
        curves: dict[int, list[tuple[int, int, int]]] = {}
        for path in sorted(results_root.glob("*/*/stats.json")):
            stats = json.loads(path.read_text())
            curves[int(path.parent.name)] = [
                (int(t), int(s), int(n)) for t, s, n in stats["evaluations"]
            ]
        return curves

    @staticmethod
    def window_scores(
        *, curves: dict[int, list[tuple[int, int, int]]], start: int, width: int
    ) -> dict[int, float]:
        """Each seed's mean solved count over `width` sweeps starting at index `start`.

        A float, not a count, because it is a mean of counts. Callers that need an `x/y`
        pool it back up against the same denominator -- see `pooled`.
        """
        return {
            seed: statistics.fmean(solved for _t, solved, _n in evaluations[start : start + width])
            for seed, evaluations in curves.items()
        }

    @staticmethod
    def pooled(*, scores: dict[int, float], num_total: int) -> tuple[float, int]:
        """`(x, y)` for an `x/y` over every seed: the summed mean-solved against the
        summed denominator. `x` stays a float because each seed's contribution is a
        window mean, and rounding it here would hide that."""
        return sum(scores.values()), num_total * len(scores)

    @staticmethod
    def report(*, results_root: Path) -> None:
        curves = Tossing3DPlateau.load_curves(results_root=results_root)
        if not curves:
            print(f"No completed runs under {results_root}")
            return
        num_sweeps = min(len(evaluations) for evaluations in curves.values())
        num_total = curves[next(iter(curves))][0][2]
        print(f"{len(curves)} seeds {sorted(curves)}, {num_sweeps} sweeps each, {num_total} tasks")

        late = Tossing3DPlateau.window_scores(
            curves=curves, start=num_sweeps - WINDOW, width=WINDOW
        )
        reference = Tossing3DPlateau.window_scores(
            curves=curves, start=REFERENCE_START_CYCLE, width=WINDOW
        )
        final = {seed: evaluations[-1][1] for seed, evaluations in curves.items()}
        best = {seed: max(s for _t, s, _n in evaluations) for seed, evaluations in curves.items()}

        late_x, late_y = Tossing3DPlateau.pooled(scores=late, num_total=num_total)
        ref_x, ref_y = Tossing3DPlateau.pooled(scores=reference, num_total=num_total)
        print(
            f"\nreference window (cycles {REFERENCE_START_CYCLE}-"
            f"{REFERENCE_START_CYCLE + WINDOW - 1}): {ref_x:.1f}/{ref_y}"
        )
        print(f"late window (last {WINDOW} sweeps):            {late_x:.1f}/{late_y}")
        print(
            f"final sweep alone:                           {sum(final.values())}/"
            f"{num_total * len(final)}"
        )
        print(
            f"best sweep per seed:                         {sum(best.values())}/"
            f"{num_total * len(best)}"
        )

        differences = [late[seed] - reference[seed] for seed in sorted(curves)]
        test = PairedTests.sign_flip(differences=differences)
        mde = PairedTests.minimum_detectable_effect(differences=differences)
        improved = sum(1 for d in differences if d > 0)
        worsened = sum(1 for d in differences if d < 0)
        print(f"\nclimb test (late - reference), paired over {len(differences)} seeds:")
        print(f"  improved {improved}/{len(differences)}, worsened {worsened}/{len(differences)}")
        print(f"  mean per-seed change {statistics.fmean(differences):+.2f} tasks")
        print(f"  exact paired sign-flip p = {test.p_value:.6g}")
        print(f"  minimum detectable effect at 80% power: {mde:.2f} tasks per seed")

        print("\nper seed (reference -> late, final, best):")
        for seed in sorted(curves):
            print(
                f"  seed {seed}: {reference[seed]:.1f} -> {late[seed]:.1f}  "
                f"final {final[seed]}/{num_total}  best {best[seed]}/{num_total}"
            )

    @staticmethod
    def render_curves(*, curves: dict[int, list[tuple[int, int, int]]], output: Path) -> None:
        """The learning curve: faint per-seed lines under a bold pooled mean, against
        cycles, with the two non-learning arms as horizontal reference lines.

        **Cycles, not online transitions, on the x axis.** Both are defensible, but the
        cycle is the *controlled* variable here -- every seed ran exactly 100 of them --
        while transitions vary per seed because a Tossing3D practice period ends early
        (`Toss` deletes `Reachable`, so nothing is applicable after a throw). Against
        cycles the seeds share a grid and the per-seed spread is readable; against
        transitions each seed sits on its own irregular one.

        The per-seed lines are the point, not decoration: the measured sweep-to-sweep
        swing is several tasks, so a bold mean alone would imply a smoothness the data
        does not have.
        """
        fig, axes = plt.subplots(1, 1, figsize=(11.0, 6.2))
        num_total = curves[next(iter(curves))][0][2]
        cycles = list(range(len(next(iter(curves.values())))))
        for seed in sorted(curves):
            axes.plot(
                cycles,
                [solved for _t, solved, _n in curves[seed]],
                color=_EES_COLOUR,
                alpha=0.20,
                linewidth=0.9,
            )
        pooled = [
            statistics.fmean(curves[seed][i][1] for seed in curves) for i in range(len(cycles))
        ]
        late_x, late_y = Tossing3DPlateau.pooled(
            scores=Tossing3DPlateau.window_scores(
                curves=curves, start=len(cycles) - WINDOW, width=WINDOW
            ),
            num_total=num_total,
        )
        axes.plot(
            cycles,
            pooled,
            color=_EES_COLOUR,
            linewidth=2.4,
            label=f"ees — last {WINDOW} sweeps {late_x:.1f}/{late_y}",
        )
        axes.axhline(
            ORACLE_PER_SEED,
            color=_REFERENCE_COLOUR,
            linestyle=":",
            linewidth=1.8,
            label=ORACLE_LABEL,
        )
        axes.axhline(
            RANDOM_SKILLS_PER_SEED,
            color=_REFERENCE_COLOUR,
            linestyle=":",
            linewidth=1.8,
            label=RANDOM_SKILLS_LABEL,
        )
        axes.axvline(20, color="#666666", linestyle="-.", linewidth=1.2)
        axes.annotate(
            "#133's whole budget",
            xy=(20, 0.4),
            xytext=(23, 0.4),
            fontsize=8.5,
            color="#666666",
        )
        axes.grid(alpha=0.25, linewidth=0.6)
        axes.set_ylim(-num_total * 0.04, num_total * 1.08)
        axes.set_xlabel("practice cycle")
        # No `(x/N)` suffix on the axis label (#188) -- the denominator moves into the title.
        axes.set_ylabel("test tasks solved per seed")
        axes.set_title(
            f"EES on Tossing3D over 100 cycles (of {num_total} test tasks) — {len(curves)} seeds, "
            f"bold pooled mean over faint per-seed lines",
            fontsize=10.5,
        )
        axes.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
        fig.tight_layout()
        fig.savefig(output, dpi=150)
        plt.close(fig)
        print(f"wrote {output}")

    @staticmethod
    def render_windows(*, curves: dict[int, list[tuple[int, int, int]]], output: Path) -> None:
        """One line per seed joining its REFERENCE-window score to its LATE-window score.

        A climbing robot is ten lines sloping up; a plateaued one is ten roughly flat
        lines. Plotted per seed rather than as two bars because with ten seeds a bar
        chart of two means hides one seed driving the whole movement -- which is exactly
        what happened on the sibling two-way-ledge panel in #166.
        """
        num_total = curves[next(iter(curves))][0][2]
        num_sweeps = len(next(iter(curves.values())))
        late = Tossing3DPlateau.window_scores(
            curves=curves, start=num_sweeps - WINDOW, width=WINDOW
        )
        reference = Tossing3DPlateau.window_scores(
            curves=curves, start=REFERENCE_START_CYCLE, width=WINDOW
        )
        differences = [late[s] - reference[s] for s in sorted(curves)]
        test = PairedTests.sign_flip(differences=differences)
        mde = PairedTests.minimum_detectable_effect(differences=differences)

        # Per-seed lines are coloured by direction of change (rose vs fell), not by arm
        # identity -- this is a paired before/after diff (like reset_free_cycle_budget.py's
        # render_gap), a different kind of chart from the learning curves the training-curve
        # style section governs, and there is only one arm (`ees`) here regardless. Left as
        # its pre-existing direction colours rather than remapped onto the arm palette, which
        # would misleadingly suggest this axis encodes an arm rather than a sign.
        fig, axes = plt.subplots(1, 1, figsize=(7.4, 6.2))
        for seed in sorted(curves):
            rose = late[seed] > reference[seed]
            axes.plot(
                [0, 1],
                [reference[seed], late[seed]],
                marker="o",
                markersize=5,
                color="#2166ac" if rose else "#b2182b",
                alpha=0.75,
                linewidth=1.4,
            )
        axes.axhline(
            ORACLE_PER_SEED,
            color=_REFERENCE_COLOUR,
            linestyle=":",
            linewidth=1.6,
            label=ORACLE_LABEL,
        )
        axes.axhline(
            RANDOM_SKILLS_PER_SEED,
            color=_REFERENCE_COLOUR,
            linestyle=":",
            linewidth=1.6,
            label=RANDOM_SKILLS_LABEL,
        )
        axes.set_xticks([0, 1])
        axes.set_xticklabels([
            f"cycles {REFERENCE_START_CYCLE}-{REFERENCE_START_CYCLE + WINDOW - 1}",
            f"last {WINDOW} sweeps",
        ])
        axes.set_xlim(-0.25, 1.25)
        axes.set_ylim(-num_total * 0.04, num_total * 1.08)
        # No `(x/N)` suffix on the axis label (#188) -- the denominator moves into the title.
        axes.set_ylabel("mean test tasks solved per seed")
        rose_count = sum(1 for d in differences if d > 0)
        axes.set_title(
            f"Does it still climb after #133's budget? (of {num_total} test tasks)\n"
            f"rose on {rose_count}/{len(differences)} seeds, "
            f"mean {statistics.fmean(differences):+.2f} tasks, "
            f"p = {test.p_value:.4g}, MDE {mde:.2f}",
            fontsize=10,
        )
        axes.grid(alpha=0.25, linewidth=0.6, axis="y")
        axes.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
        fig.tight_layout()
        fig.savefig(output, dpi=150)
        plt.close(fig)
        print(f"wrote {output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--curves-output", type=Path, default=None)
    parser.add_argument("--windows-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    Tossing3DPlateau.report(results_root=args.results_root)
    if args.curves_output is not None or args.windows_output is not None:
        curves = Tossing3DPlateau.load_curves(results_root=args.results_root)
        if args.curves_output is not None:
            Tossing3DPlateau.render_curves(curves=curves, output=args.curves_output)
        if args.windows_output is not None:
            Tossing3DPlateau.render_windows(curves=curves, output=args.windows_output)


if __name__ == "__main__":
    main()

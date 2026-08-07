"""Post-run analysis for the Tossing3D practice diagnosis: applies the decision rule
pre-registered in `docs/experiment-logs/2026-08-06-tossing3d-practice-diagnosis.md` to a
sweep's `practice_outcomes_per_cycle`, and renders the figure that log commits.

**What this decides.** PR #108 measured EES on Tossing3D at `19/100` pre-practice and
`21/100` at end of training and declined to say why, because `stats.json` recorded no
practice outcomes. It now does (PR #111). The question is starvation (too few labels)
against inability (labels present, cannot fit them) -- with a third registered outcome,
that the dichotomy itself is wrong.

**Why this is a structural test and not a statistical one, which is load-bearing here.**
Tossing3D currently has two independent reasons its *task-success* numbers are
provisional: it is not reproducible from `--seed` (a same-seed swing of at least 10 pp
has been measured), and #102 changed the no-op path underneath the existing results.
Every quantity this module decides on is instead a **count of skill executions** with a
denominator in the hundreds -- how many labels a sampler received, what fraction of them
were positive, and how many times its classifier ever discriminated among candidates.
None of those is an episode count, so neither defect moves them. Task success appears in
the figure for context and is never an input to the verdict; `verdict` does not read
`evaluations` at all.

**The positive control is not decoration.** "The classifier never made an informed draw"
and "the instrument cannot see informed draws" produce the same number, and only one of
them is a finding. `Pick` is the control: it is a different lifted skill, with its own
sampler, in the same runs and the same `stats.json`, and it does make informed draws. A
run in which `Pick` also shows zero informed draws is reporting an instrument fault, not
a result, and `verdict` says so rather than concluding.

Reads only already-produced `--output-dir` output (CLAUDE.md's `analysis/` convention).
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering -- no GUI backend needed/available in CI

import matplotlib.pyplot as plt  # noqa: E402

from analysis.practice_makes_perfect.practice_diagnostics import PracticeDiagnostics  # noqa: E402
from analysis.practice_makes_perfect.practice_verdict import PracticeVerdict  # noqa: E402
from hitl_pmp.core.method.types import SkillPracticeTally  # noqa: E402
from hitl_pmp.core.metrics.metrics import Metrics  # noqa: E402

# The standoff: Tossing3D's only meaningful learnable parameter, and the skill the
# diagnosis is about. Its add effect is NearBin.
THROW_POSE_SKILL = "MoveToThrowPose"
# The skill whose add effect is InGoalRegion -- the domain's actual success criterion --
# and which has param_dim = 0, so no sampler is ever fitted for it.
TOSS_SKILL = "Toss"
# The positive control: a different sampler, same runs, same file. See the module
# docstring for why a result without it is not a result.
CONTROL_SKILL = "Pick"

# The uniform-draw rate over THROW_STANDOFF_BOUNDS, measured through the CLI in PR #105
# as 543/2700. Carried as its two counts rather than a float, so the denominator stays
# visible wherever it is quoted.
UNIFORM_DRAW_SOLVED = 543
UNIFORM_DRAW_TOTAL = 2700

# The pre-registered thresholds. Named here so the code that decides and the log that
# registered the decision cannot drift apart silently.
MOSTLY_POSITIVE_LABELS = 0.90
NEVER_INFORMED = 0.05
INFORMED_IN_QUANTITY = 0.30
# The tolerance the `inability` cell asserts now lives on `PracticeVerdict`, where it also
# sets the power threshold -- see `PracticeVerdict.has_power` for why those are one number.


class Tossing3DPracticeDiagnosis:
    """A static-method container, never instantiated, same as every other
    business-logic class in this project."""

    @staticmethod
    def load(*, results_root: Path) -> list[Metrics]:
        """Every seed's `Metrics`, pooled over whatever methods the sweep contains.

        Flattened across method directories rather than keyed by method, because this
        sweep has exactly one arm (`ees`) -- the ceiling and uniform-draw references are
        PR #105's already-published counts, not arms re-run here."""
        runs: list[Metrics] = []
        for method_dir in sorted(results_root.iterdir()):
            if method_dir.is_dir():
                runs.extend(PracticeDiagnostics.load_runs(method_dir=method_dir))
        return runs

    @staticmethod
    def pooled(*, runs: Sequence[Metrics], skill_name: str) -> SkillPracticeTally:
        """One lifted skill's tally summed over every window of every seed.

        Pooled rather than averaged over seeds: the quantities the decision rule reads
        are fractions of *executions*, and a mean of per-seed fractions would weight a
        seed with one attempt like a seed with thirty."""
        total = SkillPracticeTally()
        for metrics in runs:
            tally = metrics.total_practice_outcomes().get(skill_name)
            if tally is not None:
                total = total.plus(other=tally)
        return total

    @staticmethod
    def informed_per_window(
        *, runs: Sequence[Metrics], skill_name: str, field: str = "num_informed_attempts"
    ) -> list[int]:
        """One informed pool per window, summed across seeds -- the series the rule's
        "flat at zero to the final cycle" clause reads attempts from, and the one its
        plateau check reads successes from.

        `field` defaults to attempts because that is what the starvation and H3 cells ask
        about ("was the classifier ever consulted"); the `inability` cell asks the
        different question of whether it was still *getting better*, which is a count of
        successes.

        Summed rather than per-seed because the clause is about whether the classifier
        *ever* discriminated: one informed draw in one seed's last cycle would falsify
        "flat at zero", and a per-seed mean could round it away.

        **Practice windows only.** The trailing bucket covers the final evaluation sweep
        alone and holds no practice by construction, so it is a structural zero rather
        than a measurement. Including it made the "rising" clause compare the last real
        cycle against an empty one and never fire -- caught by
        `test_the_starvation_cell_fires_when_informed_draws_rise_as_labels_accumulate`."""
        series = PracticeDiagnostics.per_seed_series(
            runs=runs, skill_name=skill_name, field=field
        )
        usable = [entry for entry in series if entry]
        if not usable:
            return []
        length = min(
            min(len(entry) for entry in usable),
            min(PracticeDiagnostics.practice_window_count(metrics=metrics) for metrics in runs),
        )
        return [sum(entry[index] for entry in usable) for index in range(length)]

    @staticmethod
    def _is_rising(*, series: Sequence[int]) -> bool:
        """The pre-registration's "rising across cycles", made precise: the later half
        of the run carries strictly more informed draws than the earlier half.

        Halves rather than last-versus-first, which on a series this sparse is decided by
        whichever single cycle happens to sit at each end. Operationalised here before the
        10-seed sweep finished, and stated in the log rather than chosen to fit a
        result."""
        if len(series) < 2:
            return False
        midpoint = len(series) // 2
        return sum(series[midpoint:]) > sum(series[:midpoint])

    @staticmethod
    def verdict(*, runs: Sequence[Metrics]) -> tuple[str, str]:
        """(cell of the pre-registered decision rule, the reasoning as x/y counts).

        Deliberately reads no episode counts. See the module docstring: task success on
        this domain is currently provisional for two independent reasons, and a verdict
        that depended on it would be overturned by a re-run."""
        target = Tossing3DPracticeDiagnosis.pooled(runs=runs, skill_name=THROW_POSE_SKILL)
        control = Tossing3DPracticeDiagnosis.pooled(runs=runs, skill_name=CONTROL_SKILL)
        counts = (
            f"{THROW_POSE_SKILL}: {target.num_successes}/{target.num_attempts} labelled a "
            f"success, {target.num_informed_attempts}/{target.num_attempts} informed. "
            f"Control {CONTROL_SKILL}: {control.num_informed_attempts}/"
            f"{control.num_attempts} informed."
        )
        if target.num_attempts == 0:
            return ("never asked", f"the skill was never practiced. {counts}")
        if control.num_informed_attempts == 0:
            # Before any conclusion: the control failing means the measurement failed.
            return (
                "instrument fault, no verdict",
                f"the control sampler also never made an informed draw, so zero informed "
                f"draws on {THROW_POSE_SKILL} is not evidence about {THROW_POSE_SKILL}. "
                f"{counts}",
            )
        positive_fraction = target.num_successes / target.num_attempts
        informed_fraction = target.num_informed_attempts / target.num_attempts
        informed_series = Tossing3DPracticeDiagnosis.informed_per_window(
            runs=runs, skill_name=THROW_POSE_SKILL
        )
        flat_at_zero = not any(informed_series)
        if (
            positive_fraction >= MOSTLY_POSITIVE_LABELS
            and informed_fraction <= NEVER_INFORMED
            and flat_at_zero
        ):
            return (
                "H3 -- neither: the label does not carry the signal",
                f"nearly every attempt is labelled a success, so the classifier never has "
                f"two classes to separate, and it never discriminates in any window. "
                f"{counts}",
            )
        if (
            positive_fraction < MOSTLY_POSITIVE_LABELS
            and informed_fraction <= NEVER_INFORMED
            and Tossing3DPracticeDiagnosis._is_rising(series=informed_series)
        ):
            return ("starvation", f"informed draws are rising as labels accumulate. {counts}")
        if informed_fraction >= INFORMED_IN_QUANTITY and target.num_informed_attempts:
            # Delegated to `PracticeVerdict` rather than decided here, which is the #131
            # amendment: this branch used to fire `inability` on the informed *share* plus
            # a tolerance, with no power requirement and no plateau check, and that
            # combination returns a verdict refuted by measurement on `tossingroomsplit`.
            # See that module's docstring. The uniform-draw counts stay the reference for
            # *this* domain -- Tossing3D's `MoveToThrowPose` has no epsilon-random control
            # large enough to serve as one -- and the successes trajectory it plateau-tests
            # is the same per-window series the starvation cell reads attempts from.
            cell, reasoning = PracticeVerdict.classify(
                informed_successes=target.num_informed_successes,
                informed_attempts=target.num_informed_attempts,
                control_successes=UNIFORM_DRAW_SOLVED,
                control_attempts=UNIFORM_DRAW_TOTAL,
                informed_success_trajectory=Tossing3DPracticeDiagnosis.informed_per_window(
                    runs=runs, skill_name=THROW_POSE_SKILL, field="num_informed_successes"
                ),
            )
            return (
                cell,
                f"{reasoning} The control is the uniform-draw rate over "
                f"THROW_STANDOFF_BOUNDS, {UNIFORM_DRAW_SOLVED}/{UNIFORM_DRAW_TOTAL}. "
                f"{counts}",
            )
        return (
            "undecided",
            f"the counts fall between the pre-registered cells, so no conclusion is "
            f"supported. {counts}",
        )

    @staticmethod
    def print_report(*, runs: Sequence[Metrics]) -> None:
        print(f"seeds: {len(runs)}")
        print(
            f"{'skill':<20}{'succeeded':>14}{'informed':>14}{'epsilon-random':>18}{'fallback':>14}"
        )
        for skill_name in (THROW_POSE_SKILL, TOSS_SKILL, CONTROL_SKILL):
            tally = Tossing3DPracticeDiagnosis.pooled(runs=runs, skill_name=skill_name)
            print(
                f"{skill_name:<20}"
                f"{f'{tally.num_successes}/{tally.num_attempts}':>14}"
                f"{f'{tally.num_informed_successes}/{tally.num_informed_attempts}':>14}"
                f"{f'{tally.num_random_successes}/{tally.num_random_attempts}':>18}"
                f"{f'{tally.num_fallback_successes()}/{tally.num_fallback_attempts()}':>14}"
            )
        pre = sum(metrics.evaluations[0][1] for metrics in runs if metrics.evaluations)
        post = sum(metrics.evaluations[-1][1] for metrics in runs if metrics.evaluations)
        total = sum(metrics.evaluations[0][2] for metrics in runs if metrics.evaluations)
        print(
            f"\ntask success (context only, NOT an input to the verdict): "
            f"{pre}/{total} pre-practice, {post}/{total} at end of training; "
            f"uniform draw {UNIFORM_DRAW_SOLVED}/{UNIFORM_DRAW_TOTAL}"
        )
        cell, reasoning = Tossing3DPracticeDiagnosis.verdict(runs=runs)
        print(f"\nverdict: {cell}\n  {reasoning}")

    @staticmethod
    def plot(*, runs: Sequence[Metrics], output_path: Path) -> None:
        """Three panels: the label the standoff sampler is graded on, whether its
        classifier ever discriminated (against the control that shows the instrument
        can see it when it does), and the signal that would have graded it instead."""
        fig, axes = plt.subplots(3, 1, figsize=(8.5, 10.5))
        try:
            Tossing3DPracticeDiagnosis._plot_labels(axis=axes[0], runs=runs)
            Tossing3DPracticeDiagnosis._plot_informed(axis=axes[1], runs=runs)
            Tossing3DPracticeDiagnosis._plot_toss(axis=axes[2], runs=runs)
            for axis in axes:
                axis.set_xlabel("Number of online transitions at the end of the window")
                axis.legend(fontsize=8)
                axis.margins(y=0.2)
            fig.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150)
        finally:
            plt.close(fig)

    @staticmethod
    def _series(*, runs: Sequence[Metrics], skill_name: str, field: str) -> list[list[int]]:
        return PracticeDiagnostics.per_seed_series(runs=runs, skill_name=skill_name, field=field)

    @staticmethod
    def _draw(
        *,
        axis: plt.Axes,
        runs: Sequence[Metrics],
        skill_name: str,
        field: str,
        color: str,
        label: str,
        width: float,
    ) -> None:
        series = Tossing3DPracticeDiagnosis._series(runs=runs, skill_name=skill_name, field=field)
        for metrics, entry in zip(runs, series, strict=True):
            transitions = PracticeDiagnostics.window_transitions(metrics=metrics)
            length = min(
                len(transitions),
                len(entry),
                PracticeDiagnostics.practice_window_count(metrics=metrics),
            )
            if length:
                axis.plot(transitions[:length], entry[:length], color=color, alpha=0.16, lw=1)
        mean = PracticeDiagnostics.mean_series(series=series)
        if mean and runs:
            transitions = PracticeDiagnostics.window_transitions(metrics=runs[0])
            length = min(
                len(transitions),
                len(mean),
                PracticeDiagnostics.practice_window_count(metrics=runs[0]),
            )
            axis.plot(
                transitions[:length],
                mean[:length],
                color=color,
                lw=width,
                marker="o",
                ms=3,
                label=label,
            )

    @staticmethod
    def _plot_labels(*, axis: plt.Axes, runs: Sequence[Metrics]) -> None:
        tally = Tossing3DPracticeDiagnosis.pooled(runs=runs, skill_name=THROW_POSE_SKILL)
        for field, color, label, width in (
            ("num_attempts", "tab:blue", "attempts", 3.2),
            ("num_successes", "tab:green", "labelled a success", 1.8),
        ):
            Tossing3DPracticeDiagnosis._draw(
                axis=axis,
                runs=runs,
                skill_name=THROW_POSE_SKILL,
                field=field,
                color=color,
                label=label,
                width=width,
            )
        axis.set_title(
            f"{THROW_POSE_SKILL} (the standoff): every draw is labelled a success\n"
            f"{tally.num_successes}/{tally.num_attempts} over {len(runs)} seeds "
            f"-- the two lines coincide",
            fontsize=11,
        )
        axis.set_ylabel("executions per window\n(count, not a rate)")

    @staticmethod
    def _plot_informed(*, axis: plt.Axes, runs: Sequence[Metrics]) -> None:
        target = Tossing3DPracticeDiagnosis.pooled(runs=runs, skill_name=THROW_POSE_SKILL)
        control = Tossing3DPracticeDiagnosis.pooled(runs=runs, skill_name=CONTROL_SKILL)
        Tossing3DPracticeDiagnosis._draw(
            axis=axis,
            runs=runs,
            skill_name=CONTROL_SKILL,
            field="num_informed_attempts",
            color="tab:orange",
            label=f"{CONTROL_SKILL} (control): {control.num_informed_attempts}"
            f"/{control.num_attempts} informed",
            width=2.4,
        )
        Tossing3DPracticeDiagnosis._draw(
            axis=axis,
            runs=runs,
            skill_name=THROW_POSE_SKILL,
            field="num_informed_attempts",
            color="tab:red",
            label=f"{THROW_POSE_SKILL}: {target.num_informed_attempts}"
            f"/{target.num_attempts} informed",
            width=2.4,
        )
        axis.set_title(
            "Informed draws: the standoff's classifier never discriminates\n"
            "(the control shows an informed draw is measurable in these same runs)",
            fontsize=11,
        )
        axis.set_ylabel("informed attempts per window\n(count, not a rate)")

    @staticmethod
    def _plot_toss(*, axis: plt.Axes, runs: Sequence[Metrics]) -> None:
        tally = Tossing3DPracticeDiagnosis.pooled(runs=runs, skill_name=TOSS_SKILL)
        for field, color, label, width in (
            ("num_attempts", "tab:blue", "attempts", 3.2),
            ("num_successes", "tab:purple", "cube in the goal region", 1.8),
        ):
            Tossing3DPracticeDiagnosis._draw(
                axis=axis,
                runs=runs,
                skill_name=TOSS_SKILL,
                field=field,
                color=color,
                label=label,
                width=width,
            )
        axis.set_title(
            f"{TOSS_SKILL}: the signal that does depend on the standoff, "
            f"{tally.num_successes}/{tally.num_attempts}\n"
            f"param_dim = 0, so no sampler is ever fitted to read it",
            fontsize=11,
        )
        axis.set_ylabel("executions per window\n(count, not a rate)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="Optional diagnosis PNG.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    runs = Tossing3DPracticeDiagnosis.load(results_root=args.results_root)
    Tossing3DPracticeDiagnosis.print_report(runs=runs)
    if args.output is not None:
        Tossing3DPracticeDiagnosis.plot(runs=runs, output_path=args.output)
        print(f"\nWrote plot to {args.output}")


if __name__ == "__main__":
    main()

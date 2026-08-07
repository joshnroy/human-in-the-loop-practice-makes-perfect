"""Post-run analysis for the two per-window diagnostic records in `stats.json`:
`practice_outcomes_per_cycle` (what practicing each lifted skill actually did) and
`planning_failures_per_cycle`/`planning_attempts_per_cycle` (how often the planner was
asked and how often it came back empty).

**The question this is for.** A learning curve says a method scored 21/100. It cannot
say *why*, and the candidate answers need different fixes: the samplers were never
given enough labels (starvation -- buy more transitions), or they have the labels and
cannot fit them (inability -- change the representation), or there was never a parameter
to learn at all (the domain is decomposed wrong). Those look identical in `evaluations`
and different here. Read the per-skill panels as: no line at all means the skill was
never practiced; a line whose *informed* attempts stay at zero means the sampler was
asked but never had a classifier that could rank its candidates; a healthy informed
count with a flat informed success count means it was asked and missed.

**The two dashed lines are the actionable distinction**, and they are the reason this
script changed after #111. *Never consultable* (dashed grey) counts executions of a
skill declaring `param_dim == 0`: no sampler exists, none can, and no amount of practice
will improve it -- the fix is to the domain's decomposition, moving the parameter or
fusing the skills. *Consulted, uninformative* (dotted purple) counts executions where a
sampler existed, was asked, and could not discriminate -- the fix is to the success
predicate, which is admitting parameters it should reject. #111 reported both as one
"fallback" number, which is how a Tossing3D design flaw survived two experiments: `Toss`
(`param_dim = 0`) and `MoveToThrowPose` (`param_dim = 1`, `NearBin` satisfied by every
standoff its sampler could draw) rendered identically. A panel that is entirely dashed
grey needs a different person to look at it than one that is entirely dotted purple.

Domain- and method-agnostic: it keys on nothing but the lifted skill names a run
happens to record, so the same script serves Light Switch, Tossing Room, Ball-Ring
and Tossing3D. That is the whole point -- the previous route to these numbers was a
per-domain collector script that imported one domain's `Environment` directly and could
not be pointed at another. #141 deleted it along with the fork it was written for, and
nothing replaced it: this module already answered the same question on every domain.

Never runs a simulation or drives a `Problem`/`Method` itself (CLAUDE.md's `analysis/`
convention): it reads back what `hitl_pmp.method_runner.MethodRunner.run` already wrote
under `--output-dir`, laid out as `<results-root>/<method>/<seed>/stats.json` -- the
layout `scripts/run_sweep.py` produces.

**Per-seed spread, not just a mean.** Every panel draws one faint line per seed under
the mean, because with ten seeds a mean can be one seed's behaviour: a single run that
practiced a skill 300 times while nine practiced it twice produces a mean of 32, which
describes nothing that happened.

**The x-axis offset is real and is handled here.** Both records are bucketed by
`PracticeLoop`'s `on_cycle_end`, which fires *before* each evaluation sweep, so bucket
`i` covers sweep `i` and the practice period after it -- it holds the practice that runs
between `evaluations[i]` and `evaluations[i+1]`. Plotting bucket `i` at
`evaluations[i]`'s transition count would shift a whole practice period one checkpoint
left. `window_transitions` puts each bucket at the transition count it *ends* at
instead, which is where the practice it describes actually happened. The trailing
bucket covers the final sweep alone and contains no practice at all -- see
`practice_window_count` for why the practice panels stop before it and the planning
panel does not.

Counts, never bare percentages: every printed cell and every axis label is `x/y`.
"""

import argparse
import statistics
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering -- no GUI backend needed/available in CI

import matplotlib.pyplot as plt  # noqa: E402

from hitl_pmp.core.method.types import PracticeTargetTally, SkillPracticeTally  # noqa: E402
from hitl_pmp.core.metrics.metrics import Metrics  # noqa: E402


class PracticeDiagnostics:
    """A static-method container, never instantiated, same as every other
    business-logic class in this project."""

    @staticmethod
    def load_runs(*, method_dir: Path) -> list[Metrics]:
        """Every seed's `Metrics` under one method directory, in seed-directory order.

        Reconstructed as `Metrics` rather than parsed as raw JSON so the per-run
        arithmetic (`total_practice_outcomes`, `total_planning_outcomes`) has exactly
        one implementation, in the class that owns the fields."""
        return [
            Metrics.model_validate_json(stats_path.read_text())
            for stats_path in sorted(method_dir.glob("*/stats.json"))
        ]

    @staticmethod
    def summarize(*, results_root: Path) -> dict[str, list[Metrics]]:
        return {
            method_dir.name: PracticeDiagnostics.load_runs(method_dir=method_dir)
            for method_dir in sorted(results_root.iterdir())
            if method_dir.is_dir()
        }

    @staticmethod
    def window_transitions(*, metrics: Metrics) -> list[int]:
        """The transition count each per-window bucket *ends* at.

        Bucket `i` covers evaluation sweep `i` and the practice period after it, so the
        practice it describes finished at `evaluations[i + 1]`'s transition count. The
        trailing bucket covers the final sweep alone and contains no practice, so it
        sits at the final transition count. Getting this wrong is a one-checkpoint
        shift, which on a short run is the whole effect."""
        transitions = [entry[0] for entry in metrics.evaluations]
        if not transitions:
            return []
        return [*transitions[1:], transitions[-1]]

    @staticmethod
    def skill_names(*, runs: Sequence[Metrics]) -> list[str]:
        """Every lifted skill any seed practiced, sorted. Taken over the union rather
        than over one seed, because a skill that only some seeds ever reached is
        precisely the kind of thing worth seeing."""
        names: set[str] = set()
        for metrics in runs:
            for window in metrics.practice_outcomes_per_cycle:
                names.update(window)
        return sorted(names)

    @staticmethod
    def per_seed_series(*, runs: Sequence[Metrics], skill_name: str, field: str) -> list[list[int]]:
        """One per-window series per seed, for one lifted skill and one counter.

        `field` names either a stored counter or a derived one -- the derived pools are
        zero-argument methods on `SkillPracticeTally`, so a callable attribute is called
        rather than plotted as itself. Resolving both here means a caller names a pool
        without having to know which of the two it is, which is the point of deriving
        one pool rather than storing it.

        A seed that never practiced the skill still contributes a series -- of zeros,
        the same length as its own bucket list -- because "this seed never touched it"
        is data, and dropping it would quietly shrink the denominator of the mean."""
        series: list[list[int]] = []
        for metrics in runs:
            series.append([
                PracticeDiagnostics.read_pool(
                    tally=window.get(skill_name, SkillPracticeTally()), field=field
                )
                for window in metrics.practice_outcomes_per_cycle
            ])
        return series

    @staticmethod
    def read_pool(*, tally: SkillPracticeTally, field: str) -> int:
        """One counter off a tally, whether it is stored or derived."""
        value = getattr(tally, field)
        return int(value() if callable(value) else value)

    @staticmethod
    def mean_series(*, series: Sequence[Sequence[int]]) -> list[float]:
        """Truncated to the shortest seed, so a partially-finished run cannot lengthen
        or skew the others."""
        usable = [entry for entry in series if entry]
        if not usable:
            return []
        length = min(len(entry) for entry in usable)
        return [
            statistics.mean([float(entry[index]) for entry in usable]) for index in range(length)
        ]

    @staticmethod
    def target_totals(*, runs: Sequence[Metrics]) -> dict[str, PracticeTargetTally]:
        """Every seed's practice-target decisions pooled per lifted skill. Pooled, not
        averaged, for the same reason `totals` is."""
        pooled: dict[str, PracticeTargetTally] = {}
        for metrics in runs:
            for name, tally in metrics.total_practice_target_outcomes().items():
                pooled[name] = pooled.get(name, PracticeTargetTally()).plus(other=tally)
        return {name: pooled[name] for name in sorted(pooled)}

    @staticmethod
    def target_totals_per_seed(
        *, runs: Sequence[Metrics], skill_name: str
    ) -> list[PracticeTargetTally]:
        """One whole-run tally per seed, for the per-seed spread the selection figure
        draws. A seed that never saw the skill contributes an all-zero tally rather than
        being dropped, so the denominator of a mean stays the seed count."""
        return [
            metrics.total_practice_target_outcomes().get(skill_name, PracticeTargetTally())
            for metrics in runs
        ]

    @staticmethod
    def print_target_table(*, summary: dict[str, list[Metrics]]) -> None:
        """Which skills EES chose to practice, and why it passed over the rest.

        Printed apart from the execution table rather than as extra columns on it,
        because the two have different denominators -- one row here is a count of
        scoring decisions over ground skills, one row there is a count of executions --
        and putting them side by side invites reading a ratio across them that means
        nothing.

        `declined perfect` is the row to read first. Nonzero there with a zero
        `selected` is a skill EES has stopped practicing because its measured success
        rate hit exactly 1.0, which is the state that let a Tossing3D design flaw
        survive two experiments."""
        for method, runs in summary.items():
            pooled = PracticeDiagnostics.target_totals(runs=runs)
            print(f"\n=== {method}: practice targets ({len(runs)} seeds) ===")
            if not pooled:
                print("targets: nothing recorded (this Method does not choose targets)")
                continue
            header = (
                f"{'skill':<26}{'selected':>12}{'scored':>12}"
                f"{'declined perfect':>20}{'unreachable':>14}"
            )
            print(header)
            for name, tally in pooled.items():
                print(
                    f"{name:<26}"
                    f"{tally.num_selected:>12}"
                    f"{tally.num_scored:>12}"
                    f"{tally.num_declined_perfect:>20}"
                    f"{tally.num_unreachable:>14}"
                )
            # The one line a reader should not have to derive by eye.
            never = [
                name
                for name, tally in pooled.items()
                if tally.num_selected == 0 and tally.num_declined_perfect > 0
            ]
            if never:
                print(
                    "NEVER SELECTED, declined as already-perfect: "
                    + ", ".join(sorted(never))
                    + "  <- skip_perfect dropped every grounding of these"
                )

    @staticmethod
    def totals(*, runs: Sequence[Metrics]) -> dict[str, SkillPracticeTally]:
        """Every seed's practice pooled per lifted skill -- the x/y counts the table
        prints. Pooled rather than averaged: a mean of rates over seeds with different
        attempt counts weights a two-attempt seed like a two-hundred-attempt one."""
        pooled: dict[str, SkillPracticeTally] = {}
        for metrics in runs:
            for name, tally in metrics.total_practice_outcomes().items():
                pooled[name] = pooled.get(name, SkillPracticeTally()).plus(other=tally)
        return {name: pooled[name] for name in sorted(pooled)}

    @staticmethod
    def print_table(*, summary: dict[str, list[Metrics]]) -> None:
        """Every cell an x/y, pooled across seeds, with the seed count stated so the
        denominators are recoverable."""
        for method, runs in summary.items():
            print(f"\n=== {method} ({len(runs)} seeds) ===")
            failures = sum(metrics.total_planning_outcomes()[0] for metrics in runs)
            attempts = sum(metrics.total_planning_outcomes()[1] for metrics in runs)
            print(f"planning: {failures}/{attempts} planner calls found no plan")
            pooled = PracticeDiagnostics.totals(runs=runs)
            if not pooled:
                print("practice: nothing recorded (this Method does not measure practice)")
                continue
            # The four pools are printed side by side, and "fallback" is deliberately
            # not among them: it is their union, and printing a union beside its parts
            # invites reading one number twice. Every row's four pools sum to its total.
            header = (
                f"{'skill':<22}{'succeeded':>14}{'informed':>14}"
                f"{'epsilon-random':>18}{'no sampler':>14}{'uninformative':>16}"
            )
            print(header)
            for name, tally in pooled.items():
                unparameterized = (
                    f"{tally.num_unparameterized_successes}/{tally.num_unparameterized_attempts}"
                )
                uninformative = (
                    f"{tally.num_uninformative_successes()}/{tally.num_uninformative_attempts()}"
                )
                print(
                    f"{name:<22}"
                    f"{f'{tally.num_successes}/{tally.num_attempts}':>14}"
                    f"{f'{tally.num_informed_successes}/{tally.num_informed_attempts}':>14}"
                    f"{f'{tally.num_random_successes}/{tally.num_random_attempts}':>18}"
                    f"{unparameterized:>14}"
                    f"{uninformative:>16}"
                )

    @staticmethod
    def _draw_panel(
        *,
        axis: plt.Axes,
        runs: Sequence[Metrics],
        skill_name: str,
        field: str,
        color: str,
        label: str,
        width: float,
        linestyle: str = "-",
    ) -> None:
        """One counter for one skill: a faint line per seed, the mean on top.

        The per-seed lines are the point of the panel rather than decoration -- see the
        module docstring on why a mean alone can describe nothing that happened.

        `width` descends across the series a caller draws, so a skill whose successes
        equal its attempts (every deterministic skill) shows both rather than hiding one
        exactly under the other. `linestyle` separates the two fallback pools from the
        four solid series for a stronger reason than crowding: they answer a different
        question (why was nothing learned) than the counts above them (how much was
        tried), and they are told apart from each other at a glance, which is what the
        reader has to act on."""
        series = PracticeDiagnostics.per_seed_series(runs=runs, skill_name=skill_name, field=field)
        for metrics, entry in zip(runs, series, strict=True):
            transitions = PracticeDiagnostics.window_transitions(metrics=metrics)
            length = min(
                len(transitions),
                len(entry),
                PracticeDiagnostics.practice_window_count(metrics=metrics),
            )
            if length:
                axis.plot(
                    transitions[:length],
                    entry[:length],
                    color=color,
                    alpha=0.18,
                    lw=1,
                    ls=linestyle,
                )
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
                ls=linestyle,
                marker="o",
                ms=3,
                label=label,
            )

    @staticmethod
    def practice_window_count(*, metrics: Metrics) -> int:
        """How many buckets describe a practice period.

        Every bucket but the last does. The trailing one covers the final evaluation
        sweep alone -- no practice runs in it *by construction*, and it sits at the same
        transition count as the bucket before it, since an evaluation sweep charges no
        transitions. Plotting it in a practice panel therefore draws a structural zero
        at a duplicated x, which renders as a cliff at the right-hand edge and reads as
        a collapse. It is excluded from the practice panels for that reason, and kept in
        the planning panel, where the final sweep really does make planner calls."""
        return max(len(metrics.practice_outcomes_per_cycle) - 1, 0)

    @staticmethod
    def target_window_count(*, metrics: Metrics) -> int:
        """`practice_window_count` for the selection buckets, counted off their own
        list rather than the execution one.

        A separate method rather than reusing that one, because the two lists are
        populated independently: a `Metrics` carrying only selection records has an
        empty `practice_outcomes_per_cycle`, and reading the count off that list
        silently truncates every selection panel to nothing. Real runs write both, so
        the bug is invisible on real data and appears only where the record is partial
        -- which is exactly the case this figure exists to read."""
        return max(len(metrics.practice_target_outcomes_per_cycle) - 1, 0)

    @staticmethod
    def plot(*, summary: dict[str, list[Metrics]], output_path: Path) -> None:
        """One row per (method, lifted skill) for practice, plus one row per method for
        planning. Every panel shares the transitions x-axis the learning curve uses, so
        a reader can lay this beside `analysis/practice_makes_perfect/ees.py`'s figure
        and read the two at the same checkpoints."""
        rows: list[tuple[str, list[Metrics], str | None]] = []
        for method, runs in summary.items():
            for name in PracticeDiagnostics.skill_names(runs=runs):
                rows.append((method, runs, name))
            rows.append((method, runs, None))
        if not rows:
            raise ValueError("nothing to plot: no stats.json under --results-root")
        fig, axes = plt.subplots(len(rows), 1, figsize=(8, 3.1 * len(rows)), squeeze=False)
        try:
            for axis, (method, runs, name) in zip(axes[:, 0], rows, strict=True):
                if name is None:
                    PracticeDiagnostics._draw_planning_panel(axis=axis, runs=runs)
                    axis.set_title(f"{method}: planning calls per window ({len(runs)} seeds)")
                    axis.set_ylabel("planner calls\n(failed / attempted)")
                else:
                    for field, color, label, width, linestyle in (
                        ("num_attempts", "tab:blue", "attempts", 3.2, "-"),
                        ("num_successes", "tab:green", "successes", 2.4, "-"),
                        ("num_informed_attempts", "tab:orange", "informed attempts", 1.8, "-"),
                        ("num_informed_successes", "tab:red", "informed successes", 1.2, "-"),
                        # The actionable split -- see the module docstring. Dashed and
                        # dotted so they read as a different kind of quantity from the
                        # four above, and as different from each other.
                        (
                            "num_unparameterized_attempts",
                            "tab:gray",
                            "never consultable (param_dim 0)",
                            2.0,
                            "--",
                        ),
                        (
                            "num_uninformative_attempts",
                            "tab:purple",
                            "consulted, uninformative",
                            2.0,
                            ":",
                        ),
                    ):
                        PracticeDiagnostics._draw_panel(
                            axis=axis,
                            runs=runs,
                            skill_name=name,
                            field=field,
                            color=color,
                            label=label,
                            width=width,
                            linestyle=linestyle,
                        )
                    axis.set_title(f"{method}: {name} practice per window ({len(runs)} seeds)")
                    axis.set_ylabel("executions per window\n(count, not a rate)")
                axis.set_xlabel("Number of online transitions at the end of the window")
                axis.legend(fontsize=8)
                axis.margins(y=0.15)
            fig.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150)
        finally:
            plt.close(fig)

    @staticmethod
    def target_skill_names(*, runs: Sequence[Metrics]) -> list[str]:
        """Every lifted skill any seed ever *scored*, sorted. Taken over the union for
        the same reason `skill_names` is -- a skill only some seeds ever made a
        candidate is exactly what this figure is for."""
        names: set[str] = set()
        for metrics in runs:
            for window in metrics.practice_target_outcomes_per_cycle:
                names.update(window)
        return sorted(names)

    @staticmethod
    def _target_series(*, runs: Sequence[Metrics], skill_name: str, field: str) -> list[list[int]]:
        return [
            [
                getattr(window.get(skill_name, PracticeTargetTally()), field)
                for window in metrics.practice_target_outcomes_per_cycle
            ]
            for metrics in runs
        ]

    @staticmethod
    def plot_targets(
        *,
        summary: dict[str, list[Metrics]],
        output_path: Path,
        skills: Sequence[str] | None = None,
    ) -> None:
        """One row per (method, lifted skill): was this skill chosen for practice, and
        if not, was it dropped as already-perfect?

        A **separate figure** from `plot`, not extra series on it, because the two
        count different events with different denominators -- executions there,
        scoring decisions over ground skills here. Sharing an axis would invite reading
        a ratio across them, and the ratio is meaningless.

        Drawn as lines over the same transitions x-axis rather than as bars over
        whole-run totals, because *when* a skill stopped being selected is the finding:
        `skip_perfect` fires the moment a measured rate touches 1.0, so the shape is a
        selection line falling to zero while a declined line rises to meet it. A bar of
        run totals shows the sum of those two and hides the crossing entirely.

        `skills` restricts the panels to a named subset. Ball-Ring has fifteen lifted
        skills and ten of them are `param_dim == 0`, where "never practiced" is the
        correct answer and not a finding; a figure that gives those ten equal space
        buries the five that can actually be learned. Unfiltered by default -- the
        subset is a presentation choice and must be made explicitly, never silently."""
        rows: list[tuple[str, list[Metrics], str]] = []
        for method, runs in summary.items():
            for name in PracticeDiagnostics.target_skill_names(runs=runs):
                if skills is None or name in skills:
                    rows.append((method, runs, name))
        if not rows:
            raise ValueError("nothing to plot: no practice-target records under --results-root")
        fig, axes = plt.subplots(len(rows), 1, figsize=(8, 3.1 * len(rows)), squeeze=False)
        try:
            for axis, (method, runs, name) in zip(axes[:, 0], rows, strict=True):
                # All four are counts of ground-skill decisions per window -- one unit,
                # so one axis. Widths descend so a skill whose scored and selected
                # counts coincide shows both rather than hiding one under the other.
                for field, color, label, width, linestyle in (
                    ("num_scored", "tab:blue", "scored (a live candidate)", 3.2, "-"),
                    ("num_selected", "tab:green", "selected (practiced on purpose)", 2.4, "-"),
                    (
                        "num_declined_perfect",
                        "tab:red",
                        "declined: already perfect (skip_perfect)",
                        1.8,
                        "--",
                    ),
                    ("num_unreachable", "tab:orange", "outranked but unreachable", 1.2, ":"),
                ):
                    series = PracticeDiagnostics._target_series(
                        runs=runs, skill_name=name, field=field
                    )
                    for metrics, entry in zip(runs, series, strict=True):
                        transitions = PracticeDiagnostics.window_transitions(metrics=metrics)
                        # Same truncation as the practice panels, for the same reason:
                        # the trailing bucket is the final evaluation sweep alone, in
                        # which no practice target is chosen by construction, and it
                        # sits at a duplicated x. Drawing it puts a structural zero at
                        # the right-hand edge that reads as selection collapsing.
                        length = min(
                            len(transitions),
                            len(entry),
                            PracticeDiagnostics.target_window_count(metrics=metrics),
                        )
                        if length:
                            axis.plot(
                                transitions[:length],
                                entry[:length],
                                color=color,
                                alpha=0.18,
                                lw=1,
                                ls=linestyle,
                            )
                    mean = PracticeDiagnostics.mean_series(series=series)
                    if mean and runs:
                        transitions = PracticeDiagnostics.window_transitions(metrics=runs[0])
                        length = min(
                            len(transitions),
                            len(mean),
                            PracticeDiagnostics.target_window_count(metrics=runs[0]),
                        )
                        axis.plot(
                            transitions[:length],
                            mean[:length],
                            color=color,
                            lw=width,
                            ls=linestyle,
                            marker="o",
                            ms=3,
                            label=label,
                        )
                axis.set_title(f"{method}: {name} practice-target decisions ({len(runs)} seeds)")
                axis.set_ylabel("ground-skill decisions\nper window (count)")
                axis.set_xlabel("Number of online transitions at the end of the window")
                axis.legend(fontsize=8)
                axis.margins(y=0.15)
            fig.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150)
        finally:
            plt.close(fig)

    @staticmethod
    def _draw_planning_panel(*, axis: plt.Axes, runs: Sequence[Metrics]) -> None:
        """#106's counters, which nothing plotted until now. Failures and attempts on
        one axis rather than a ratio: EES plans speculatively, so a healthy run reports
        a nonzero, workload-dependent failure count and only `failures` tracking
        `attempts` means anything is wrong -- a shape a ratio hides."""
        for field, color, label in (
            ("planning_attempts_per_cycle", "tab:blue", "attempts"),
            ("planning_failures_per_cycle", "tab:red", "failures"),
        ):
            for metrics in runs:
                transitions = PracticeDiagnostics.window_transitions(metrics=metrics)
                values = getattr(metrics, field)
                length = min(len(transitions), len(values))
                if length:
                    axis.plot(transitions[:length], values[:length], color=color, alpha=0.18, lw=1)
            series = [getattr(metrics, field) for metrics in runs]
            mean = PracticeDiagnostics.mean_series(series=series)
            if mean and runs:
                transitions = PracticeDiagnostics.window_transitions(metrics=runs[0])
                length = min(len(transitions), len(mean))
                axis.plot(
                    transitions[:length],
                    mean[:length],
                    color=color,
                    lw=3.2 if field.endswith("attempts_per_cycle") else 2.0,
                    marker="o",
                    ms=3,
                    label=label,
                )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Sweep directory laid out as <results-root>/<method>/<seed>/stats.json.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional diagnostics PNG.")
    parser.add_argument(
        "--target-output",
        type=Path,
        default=None,
        help="Optional practice-target (selection) PNG -- a separate figure, see plot_targets.",
    )
    parser.add_argument(
        "--target-skills",
        nargs="+",
        default=None,
        help="Restrict --target-output's panels to these lifted skill names.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = PracticeDiagnostics.summarize(results_root=args.results_root)
    PracticeDiagnostics.print_table(summary=summary)
    PracticeDiagnostics.print_target_table(summary=summary)
    if args.output is not None:
        PracticeDiagnostics.plot(summary=summary, output_path=args.output)
        print(f"\nWrote plot to {args.output}")
    if args.target_output is not None:
        PracticeDiagnostics.plot_targets(
            summary=summary, output_path=args.target_output, skills=args.target_skills
        )
        print(f"Wrote practice-target plot to {args.target_output}")


if __name__ == "__main__":
    main()

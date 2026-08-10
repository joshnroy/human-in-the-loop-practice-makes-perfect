"""Post-run analysis: the training curve behind `on-no-applicable-skill`'s headline
number -- 0/10 seeds ever asked, every shared `stats.json` field byte-identical to the
`no-human` control on all 10 seeds. `docs/experiment-logs/
2026-08-10-help-seeking-naive-trigger.md` already reports that as a final-score table
and a per-seed bar chart; neither shows the *shape* over training, which for this arm is
the whole point -- the two curves are not merely close, they are the same curve, drawn
twice.

**Reads only already-produced output** (CLAUDE.md's `analysis/` convention -- this never
runs a simulation or drives a `Method`). `on-no-applicable-skill`'s ten runs are committed
under `docs/experiment-logs/2026-08-10-help-seeking-naive-trigger-runs/`; the `no-human`
arm is not re-committed alongside them -- it is the same seed-matched control the log
already reuses rather than re-running, read back from its own existing home under
`docs/experiment-logs/2026-08-07-pickup-weight-reset-free-runs/never/ees/`.

**Three panels, not one.** OVERALL is `Metrics.evaluations` directly; TRASH and
RECYCLING are read from `Metrics.breakdowns`' per-task outcomes and classified by
`GoalFamilies.classify` (shared with every other Tossing Room analysis in this folder,
rather than a third copy of the rule). EMPTY is left out, matching every other
OVERALL/TRASH/RECYCLING figure in this project: the fixed test set carries only 2 EMPTY
tasks per seed, and CLAUDE.md's own counts-not-percentages rule is exactly the reason --
a 2-task denominator is almost no evidence on its own panel.

**One transition grid, so a second x-axis is exact rather than approximate.** Every
seed of both arms was evaluated at the same 11 checkpoints, 150 transitions apart
(`--num-cycles 10 --max-steps-per-interaction 150`), so `transitions = 150 * cycle`
exactly -- a top axis in cycles is the same measurement relabelled, not a second one.

**Per-seed faint lines under a bold mean**, the same convention
`reset_free_training_curves.py` uses and for the same reason: a mean alone would show
one flat line at the pooled value and hide that the flatness holds seed-for-seed, not
just on average.
"""

import argparse
from pathlib import Path

import matplotlib
from pydantic import BaseModel, ConfigDict

from hitl_pmp.core.metrics.metrics import Metrics

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from analysis.practice_makes_perfect.goal_families import GoalFamilies  # noqa: E402

# The ten fixed seeds both arms ran, and the `--num-test-tasks` each evaluation swept.
# Named rather than inlined so the published denominators (300 overall, 140 per throw
# family) are derived from these rather than restated.
_NUM_SEEDS = 10
_NUM_TEST_TASKS = 30

_OVERALL = "OVERALL"
_TRASH = "TRASH"
_RECYCLING = "RECYCLING"
# EMPTY excluded -- see module docstring: 2 tasks/seed is almost no evidence on its own
# panel, and no other OVERALL/TRASH/RECYCLING figure in this project plots it either.
_FAMILIES = (_OVERALL, _TRASH, _RECYCLING)

_ON_NO_APPLICABLE_SKILL = "on-no-applicable-skill"
_NO_HUMAN = "no-human"
# `no-human` first, `on-no-applicable-skill` second: the two curves are identical, so
# whichever is drawn last sits on top. Drawing the dashed arm last is what makes the
# overlap visible as a blue line with an orange dash pattern showing through it, rather
# than the dashed line being fully hidden under a later-drawn solid one.
_ARMS = (_NO_HUMAN, _ON_NO_APPLICABLE_SKILL)

_ARM_LABELS = {
    _ON_NO_APPLICABLE_SKILL: "on-no-applicable-skill (never fired: 0/10 seeds)",
    _NO_HUMAN: "no-human (control, reused)",
}

# `on-no-applicable-skill` is this entry's own new run, committed alongside it;
# `no-human` is the already-committed, seed-matched control the log reuses rather than
# re-running (2026-08-07-pickup-weight-reset-free-runs/never/ees). Both paths are
# relative to `--logs-root` (default docs/experiment-logs), matching every sibling
# module in this folder.
_ARM_DIRECTORIES = {
    _ON_NO_APPLICABLE_SKILL: (
        "2026-08-10-help-seeking-naive-trigger-runs/on-no-applicable-skill/ees"
    ),
    _NO_HUMAN: "2026-08-07-pickup-weight-reset-free-runs/never/ees",
}

# Okabe-Ito blue/orange, the same pair `reset_free_training_curves.py` uses. The control
# carries blue (the incumbent, drawn first); the treated arm carries orange -- even
# though the two curves land on top of each other, which is the finding.
_ARM_COLOR = {_NO_HUMAN: "#0072B2", _ON_NO_APPLICABLE_SKILL: "#D55E00"}
# The treated arm is drawn dashed on top of the control's solid line so a reader can see
# both are actually present at every point rather than one occluding the other.
_ARM_LINESTYLE = {_NO_HUMAN: "-", _ON_NO_APPLICABLE_SKILL: "--"}

_SEED_ALPHA = 0.18
_SEED_WIDTH = 1.0
_MEAN_WIDTH = 2.6

_CANVAS = "white"


class HelpSeekingNaiveTriggerCurves(BaseModel):
    """Reads the committed per-seed `stats.json` back and draws the training-curve
    figure. A static-method container: no per-run state to carry between calls, same
    rule CLAUDE.md applies to every other concrete business-logic class."""

    model_config = ConfigDict(frozen=True)

    @staticmethod
    def load_arm(*, logs_root: Path, arm: str) -> list[Metrics]:
        """The ten seeds of one arm, in seed order.

        Raises when any seed is absent, matching `reset_free_training_curves.py`'s
        `load_arm`: nine seeds would still produce a mean and a picture, but the
        published denominator (300, or 140 per throw family) would quietly stop being
        what the table reports."""
        runs: list[Metrics] = []
        directory = _ARM_DIRECTORIES[arm]
        for seed in range(_NUM_SEEDS):
            stats = logs_root / directory / str(seed) / "stats.json"
            if not stats.is_file():
                raise FileNotFoundError(
                    f"arm {arm!r} is missing seed {seed}: {stats} does not exist. All "
                    f"{_NUM_SEEDS} seeds are required, since the published denominator "
                    f"is {_NUM_SEEDS} x {_NUM_TEST_TASKS}."
                )
            runs.append(Metrics.model_validate_json(stats.read_text()))
        return runs

    @staticmethod
    def checkpoints(*, runs: list[Metrics]) -> list[int]:
        """The one transition grid every seed in the arm was evaluated on.

        Raises when the seeds disagree, for the same reason
        `reset_free_training_curves.py` does: a mean across misaligned grids would
        silently average unequal amounts of practice into one point."""
        grids = {tuple(transitions for transitions, _, _ in run.evaluations) for run in runs}
        if len(grids) != 1:
            raise ValueError(
                f"seeds in this arm were evaluated on {len(grids)} different transition "
                f"grids, so a per-checkpoint mean would average unequal practice: {grids}"
            )
        return list(next(iter(grids)))

    @staticmethod
    def per_seed_curve(*, runs: list[Metrics], family: str) -> list[list[int]]:
        """Each seed's full solved-count curve for one family, one row per seed.

        OVERALL reads `Metrics.evaluations` directly (the aggregate every checkpoint
        already carries). TRASH/RECYCLING recompute from `Metrics.breakdowns`' per-task
        outcomes, classified by the shared `GoalFamilies.classify` rule -- there is no
        aggregate-per-family field on `Metrics`, only the per-task detail behind it."""
        if family == _OVERALL:
            return [[solved for _, solved, _ in run.evaluations] for run in runs]
        curves: list[list[int]] = []
        for run in runs:
            if len(run.breakdowns) != len(run.evaluations):
                raise ValueError(
                    "breakdowns and evaluations must be the same length to plot a "
                    f"per-family curve; got {len(run.breakdowns)} vs {len(run.evaluations)}"
                )
            curves.append([
                sum(
                    1
                    for outcome in breakdown.outcomes
                    if outcome.solved and GoalFamilies.classify(goal=outcome.goal) == family
                )
                for breakdown in run.breakdowns
            ])
        return curves

    @staticmethod
    def family_denominator(*, runs: list[Metrics], family: str) -> int:
        """Tasks solved out of this many, pooled over the arm's seeds -- the x/y the
        figure's legend and `report()` publish. Read from the runs' own final
        breakdown rather than assumed, so a test-set change widens the denominator
        instead of silently mismatching a hardcoded one."""
        if family == _OVERALL:
            return sum(run.evaluations[-1][2] for run in runs)
        return sum(
            sum(
                1
                for outcome in run.breakdowns[-1].outcomes
                if GoalFamilies.classify(goal=outcome.goal) == family
            )
            for run in runs
        )

    @staticmethod
    def arm_total(*, runs: list[Metrics], family: str) -> int:
        """Tasks solved out of `family_denominator`, at the final checkpoint, pooled
        over the arm's seeds."""
        curves = HelpSeekingNaiveTriggerCurves.per_seed_curve(runs=runs, family=family)
        return sum(curve[-1] for curve in curves)

    @staticmethod
    def render(*, logs_root: Path, output: Path) -> Figure:
        """Draw the three-panel figure and write the PNG. Returns the figure so a test
        can assert on its structure without reopening the file."""
        figure, axes = plt.subplots(1, 3, figsize=(16.5, 4.6), dpi=150, facecolor=_CANVAS)
        arm_runs = {
            arm: HelpSeekingNaiveTriggerCurves.load_arm(logs_root=logs_root, arm=arm)
            for arm in _ARMS
        }
        # Both arms share one grid (this is the point being plotted), but each is
        # checked independently rather than assumed equal, matching `checkpoints`'
        # own no-silent-mismatch rule.
        grids = {
            arm: HelpSeekingNaiveTriggerCurves.checkpoints(runs=runs)
            for arm, runs in arm_runs.items()
        }
        if len(set(tuple(g) for g in grids.values())) != 1:
            raise ValueError(
                f"the two arms were evaluated on different transition grids: {grids}, "
                "so they cannot be drawn on one shared x-axis"
            )
        grid = grids[_NO_HUMAN]

        for axis, family in zip(axes, _FAMILIES, strict=True):
            denom = HelpSeekingNaiveTriggerCurves.family_denominator(
                runs=arm_runs[_NO_HUMAN], family=family
            )
            for arm in _ARMS:
                runs = arm_runs[arm]
                curves = HelpSeekingNaiveTriggerCurves.per_seed_curve(runs=runs, family=family)
                total = HelpSeekingNaiveTriggerCurves.arm_total(runs=runs, family=family)
                color = _ARM_COLOR[arm]
                style = _ARM_LINESTYLE[arm]
                for row in curves:
                    axis.plot(
                        grid,
                        row,
                        color=color,
                        alpha=_SEED_ALPHA,
                        linewidth=_SEED_WIDTH,
                        linestyle=style,
                    )
                mean = [sum(column) / len(column) for column in zip(*curves, strict=True)]
                axis.plot(
                    grid,
                    mean,
                    color=color,
                    linewidth=_MEAN_WIDTH,
                    linestyle=style,
                    label=f"{_ARM_LABELS[arm]} -- final {total}/{denom}",
                )
            axis.set_title(f"{family} (x/{denom})", fontsize=11, loc="left")
            axis.set_xlabel("online transitions")
            axis.grid(alpha=0.25, linewidth=0.6)
            axis.legend(frameon=False, loc="upper left", fontsize=8)
            axis.margins(x=0.06)
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
            # A second x-axis in cycles: transitions = 150 * cycle exactly (see module
            # docstring), so this is the same measurement, not a second one.
            cycle_axis = axis.secondary_xaxis(
                "top", functions=(lambda t: t / 150.0, lambda c: c * 150.0)
            )
            cycle_axis.set_xlabel("cycle", fontsize=8)

        axes[0].set_ylabel("tasks solved per evaluation")
        figure.suptitle(
            f"`on-no-applicable-skill` vs `no-human`, {_NUM_SEEDS} seeds each "
            "(bold = mean, faint = individual seeds, dashed = on-no-applicable-skill) "
            "-- the flat overlap is the finding: the trigger never fired",
            fontsize=11.5,
        )
        figure.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, bbox_inches="tight", facecolor=_CANVAS)
        return figure

    @staticmethod
    def report(*, logs_root: Path) -> str:
        """The numbers behind the figure, as x/y with per-seed finals beside them."""
        lines = []
        for family in _FAMILIES:
            for arm in _ARMS:
                runs = HelpSeekingNaiveTriggerCurves.load_arm(logs_root=logs_root, arm=arm)
                total = HelpSeekingNaiveTriggerCurves.arm_total(runs=runs, family=family)
                denom = HelpSeekingNaiveTriggerCurves.family_denominator(runs=runs, family=family)
                finals = [
                    curve[-1]
                    for curve in HelpSeekingNaiveTriggerCurves.per_seed_curve(
                        runs=runs, family=family
                    )
                ]
                lines.append(f"{family:>10s} / {arm:<22s} final {total}/{denom}  per-seed {finals}")
        return "\n".join(lines)


def main() -> None:
    """CLI entrypoint. All flags named, no positional arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "docs" / "experiment-logs",
        help="Directory holding the committed run trees. Defaults to this repo's own.",
    )
    parser.add_argument("--output", type=Path, required=True, help="PNG to write.")
    args = parser.parse_args()
    HelpSeekingNaiveTriggerCurves.render(logs_root=args.logs_root, output=args.output)
    print(HelpSeekingNaiveTriggerCurves.report(logs_root=args.logs_root))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

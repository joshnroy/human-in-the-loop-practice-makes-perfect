"""Post-run analysis: the training curve of `--practice-reset-policy scheduled` against
`never`, in each of the four Tossing Room variants this stack measured, on one figure.

**The fourth panel completes a 2x2** (variant x ledge) whose other three corners were run
first, so the figure is laid out as a square: reading down a column holds the ledge fixed,
reading across a row holds the variant fixed. It is the pickup-weight fork *with* the
two-way ledge -- both proposed mechanisms of the reset-free penalty removed at once -- and
it is the cell that decides between them. **It shows the gap closing**: `never` 287/300
against `scheduled` 300/300, where every other panel has the two arms visibly apart. It
also shows the effects are not additive, and that the interaction runs in *opposite*
directions in the two variants: the two-way ledge widens the gap on `tossingroomsplit`
(66/300 -> 132/300) and collapses it on the pickup-weight fork (71/300 -> 13/300). See
`docs/experiment-logs/2026-08-07-pickup-weight-two-way-ledge.md`, including its ceiling
caveat -- `scheduled` is at 300/300, so this world cannot resolve a small residual.

**Background.** PR #115 measured the reset-free A/B on `tossingroomsplit`, #122 repeated
it on `tossingroomsplitpickupweight` (weight drawn at pickup rather than at task build),
and #124/#125 added the `--two-way-ledge` positive control that removes the domain's only
irreversible action. Each of those reported *outcome counts* -- a final x/300 per arm,
plus penalties and an interaction -- and #115 and #122 each committed a curve of their
own. What no single figure showed is the three variants side by side, which is where two
things become visible that no table in the stack states.

**The first is the honest frame for the headline.** Opening the ledge lifted `scheduled`
from 151/300 to 276/300 and `never` from 85/300 to 144/300. Reset-free practice therefore
did **not** get worse in absolute terms when stranding was removed -- it improved. What
grew is the *gap*, because `scheduled` improved far more. That is exactly the ceiling
caveat #125 already flags, and the interaction statistic cannot show it: an interaction is
a difference of differences and is blind to both arms rising together.

**The second is that one arm's mean describes none of its seeds.** `pickup-weight / never`
finishes 18, 16, 5, 6, 7, 6, 21, 20, 7, 6 -- four seeds tracking the one-way arm and six
collapsed, with nothing in between. Its mean, 11.2/30, falls in the empty gap. That is the
stranding split of #122 appearing directly in task outcomes rather than in the
practice-tally diagnostic, and it is why every arm here is drawn as ten faint per-seed
lines under a bold mean rather than as a mean alone.

**Reads only already-produced output** (CLAUDE.md's `analysis/` convention -- this never
runs a simulation or drives a `Method`). Unlike the sibling reports in this folder it
needs no intermediate aggregate: all 60 runs' `stats.json` are committed under
`docs/experiment-logs/`, so the figure regenerates from the repository alone and
`--logs-root` exists only so a test can point it somewhere else.

**Counts, never rates.** `Metrics.task_training_curve` exists and returns a *fraction*;
this reads `Metrics.evaluations` directly instead, because the denominator (30 test tasks
per evaluation, 300 per arm) is small and uneven enough that a rate on an axis hides it.
The y-axis is tasks solved out of 30 and every legend entry carries its own x/300.
"""

import argparse
from pathlib import Path

import matplotlib
from pydantic import BaseModel, ConfigDict

from hitl_pmp.core.metrics.metrics import Metrics

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

# The ten fixed seeds every arm in this stack ran, and the `--num-test-tasks` each
# evaluation sweeps. Named rather than inlined because the published denominator (300)
# is their product and `load_arm` refuses an arm that does not have all ten.
_NUM_SEEDS = 10
_NUM_TEST_TASKS = 30

# Panel order, left to right: the incumbent world first, then the intervention that
# removes stranding from it, then the separate domain where the training distribution
# varies at pickup. That ordering is the stack's own dependency order (#115 -> #125 ->
# #122 is not it; #115 -> #122 -> #125 is), so the reader walks the panels in the order
# the experiments were run.
_ONE_WAY = "one-way"
_TWO_WAY = "two-way"
_PICKUP_WEIGHT = "pickup-weight"
# The fourth cell, added once it was run: the pickup-weight fork *with* the two-way
# ledge, i.e. both proposed mechanisms of the reset-free penalty removed at once. It
# completes the 2x2 the three panels above are three corners of, which is why the layout
# below is a square rather than a row -- reading down a column now holds the ledge fixed
# and reading across a row holds the variant fixed.
_PICKUP_WEIGHT_TWO_WAY = "pickup-weight-two-way"
_PANELS = (_ONE_WAY, _TWO_WAY, _PICKUP_WEIGHT, _PICKUP_WEIGHT_TWO_WAY)

_PANEL_TITLES = {
    _ONE_WAY: "tossingroomsplit -- one-way ledge\n(the pile is unreachable once crossed)",
    _TWO_WAY: "tossingroomsplit -- two-way ledge\n(no irreversible action at all)",
    _PICKUP_WEIGHT: "tossingroomsplitpickupweight\n(weight drawn at pickup, one-way ledge)",
    _PICKUP_WEIGHT_TWO_WAY: (
        "tossingroomsplitpickupweight -- two-way ledge\n(both mechanisms removed)"
    ),
}

# The two arms compared inside every panel. `scheduled` is the incumbent -- the only
# policy that existed before this stack -- so it is drawn first and carries the colour a
# reader will read as the baseline.
_SCHEDULED = "scheduled"
_NEVER = "never"
_POLICIES = (_SCHEDULED, _NEVER)

_POLICY_LABELS = {
    _SCHEDULED: "scheduled (reset each period)",
    _NEVER: "never (practice runs continuously)",
}

# Okabe-Ito blue/orange, the same pair the sibling reports in this folder use and the
# widest-separated two under deuteranopia and protanopia alike. Policy carries colour
# here (rather than world, as in `tossingroomsplit_two_way_ledge`) because policy is the
# comparison made *within* each panel; world is carried by the panels themselves, so the
# two factors are still never distinguished by colour alone.
_POLICY_COLOR = {_SCHEDULED: "#0072B2", _NEVER: "#D55E00"}

# Per-seed lines are what the figure exists for, but ten of them at full strength hide
# the mean. Low alpha and a thin stroke keep the spread legible as a band while the bold
# mean stays readable on top of it.
_SEED_ALPHA = 0.18
_SEED_WIDTH = 1.0
_MEAN_WIDTH = 2.6

# Figures are saved on an explicit white canvas rather than matplotlib's transparent
# default, so a PNG dropped into a dark-themed PR or Notion page keeps readable axes
# instead of black text on black.
_CANVAS = "white"


class ArmSpec(BaseModel):
    """One (panel, policy) cell and the committed directory its ten runs live in.

    Frozen and carried as data rather than assembled from a format string at each call
    site, because the two run trees this figure spans are *not* laid out the same way:
    the two-way-ledge sweep has a `--method`-named `ees/` level between the arm and the
    seed and the pickup-weight sweep does not. That asymmetry is a fact about what was
    committed, so it is recorded here once instead of being rediscovered by every reader
    who wonders why one path has an extra component."""

    model_config = ConfigDict(frozen=True)

    panel: str
    policy: str
    directory: str


# The six arms, spelled out. The `ees/` level in the first four is not a typo -- see
# `ArmSpec`.
_ARMS = (
    ArmSpec(
        panel=_ONE_WAY,
        policy=_SCHEDULED,
        directory="2026-08-06-reset-free-two-way-ledge-runs/one-way-scheduled/ees",
    ),
    ArmSpec(
        panel=_ONE_WAY,
        policy=_NEVER,
        directory="2026-08-06-reset-free-two-way-ledge-runs/one-way-never/ees",
    ),
    ArmSpec(
        panel=_TWO_WAY,
        policy=_SCHEDULED,
        directory="2026-08-06-reset-free-two-way-ledge-runs/two-way-scheduled/ees",
    ),
    ArmSpec(
        panel=_TWO_WAY,
        policy=_NEVER,
        directory="2026-08-06-reset-free-two-way-ledge-runs/two-way-never/ees",
    ),
    ArmSpec(
        panel=_PICKUP_WEIGHT,
        policy=_SCHEDULED,
        directory="2026-08-07-pickup-weight-reset-free-runs/scheduled",
    ),
    ArmSpec(
        panel=_PICKUP_WEIGHT,
        policy=_NEVER,
        directory="2026-08-07-pickup-weight-reset-free-runs/never",
    ),
    # The fourth cell's own sweep, which has the `ees/` level back -- it was driven by
    # `scripts/run_sweep.py` in the same layout as the two-way-ledge runs.
    ArmSpec(
        panel=_PICKUP_WEIGHT_TWO_WAY,
        policy=_SCHEDULED,
        directory="2026-08-07-pickup-weight-two-way-ledge-runs/scheduled/ees",
    ),
    ArmSpec(
        panel=_PICKUP_WEIGHT_TWO_WAY,
        policy=_NEVER,
        directory="2026-08-07-pickup-weight-two-way-ledge-runs/never/ees",
    ),
)


class ResetFreeTrainingCurves(BaseModel):
    """Reads the committed per-seed `stats.json` back and draws the three-panel figure.

    A static-method container: there is no per-run state to carry between calls, so
    nothing here is an instance method (CLAUDE.md's `core/` rule, applied to concrete
    helpers underneath it)."""

    @staticmethod
    def arm(*, panel: str, policy: str) -> ArmSpec:
        """The declared spec for one cell, so no caller re-spells a committed path."""
        for spec in _ARMS:
            if spec.panel == panel and spec.policy == policy:
                return spec
        raise KeyError(f"no arm declared for panel={panel!r} policy={policy!r}")

    @staticmethod
    def load_arm(*, logs_root: Path, arm: ArmSpec) -> list[Metrics]:
        """The ten seeds of one arm, in seed order.

        Raises when any seed is absent rather than returning what it found: nine seeds
        still produce a mean and a picture, but the denominator quietly stops being the
        published 300 and nothing downstream would notice. Read through
        `Metrics.model_validate_json` rather than as raw JSON so the schema is checked
        and the fields are the ones `Metrics` itself defines."""
        runs: list[Metrics] = []
        for seed in range(_NUM_SEEDS):
            stats = logs_root / arm.directory / str(seed) / "stats.json"
            if not stats.is_file():
                raise FileNotFoundError(
                    f"arm {arm.panel}/{arm.policy} is missing seed {seed}: {stats} does not "
                    f"exist. All {_NUM_SEEDS} seeds are required, since the published "
                    f"denominator is {_NUM_SEEDS} x {_NUM_TEST_TASKS}."
                )
            runs.append(Metrics.model_validate_json(stats.read_text()))
        return runs

    @staticmethod
    def per_seed_finals(*, runs: list[Metrics]) -> list[int]:
        """Each seed's tasks-solved at the LAST evaluation, in seed order.

        `evaluations[0]` is the pre-practice sweep at 0 transitions, so the final entry
        is `[-1]` and never `[0]` -- an off-by-one here reads as a plausible lower
        score for every arm rather than as an error."""
        return [run.evaluations[-1][1] for run in runs]

    @staticmethod
    def arm_total(*, runs: list[Metrics]) -> tuple[int, int]:
        """(solved, total) pooled over the arm's seeds at the final evaluation -- the
        x/y the tables publish. The denominator comes from each run's own recorded
        `num_total` rather than from the `_NUM_TEST_TASKS` constant, so a run that
        evaluated a different number of tasks widens the denominator instead of being
        silently divided by the expected one."""
        solved = sum(run.evaluations[-1][1] for run in runs)
        total = sum(run.evaluations[-1][2] for run in runs)
        return solved, total

    @staticmethod
    def checkpoints(*, runs: list[Metrics]) -> list[int]:
        """The one transition grid every seed in the arm was evaluated on.

        Raises when the seeds disagree. A mean taken across seeds whose checkpoints
        differ averages different amounts of practice into one point, which is invisible
        in the drawn curve."""
        grids = {tuple(transitions for transitions, _, _ in run.evaluations) for run in runs}
        if len(grids) != 1:
            raise ValueError(
                f"seeds in this arm were evaluated on {len(grids)} different transition "
                f"grids, so a per-checkpoint mean would average unequal practice: {grids}"
            )
        return list(next(iter(grids)))

    @staticmethod
    def per_seed_curves(*, runs: list[Metrics]) -> list[list[int]]:
        """Each seed's full solved-count curve, one row per seed."""
        return [[solved for _, solved, _ in run.evaluations] for run in runs]

    @staticmethod
    def render(*, logs_root: Path, output: Path) -> Figure:
        """Draw all three panels and write the PNG. Returns the figure so a test can
        assert on its structure without reopening the file."""
        # 2x2 rather than a row of four: the panels are a 2x2 design (variant x ledge),
        # so a square lets a reader hold one factor fixed by reading down a column or
        # across a row. A 1x4 row would be ~22 inches wide and would put the two cells
        # that differ only in ledge at opposite ends of the figure.
        figure, axes_grid = plt.subplots(
            2, 2, figsize=(13.5, 9.6), dpi=150, sharey=True, facecolor=_CANVAS
        )
        axes = list(axes_grid.flat)
        for axis, panel in zip(axes, _PANELS, strict=True):
            for policy in _POLICIES:
                arm = ResetFreeTrainingCurves.arm(panel=panel, policy=policy)
                runs = ResetFreeTrainingCurves.load_arm(logs_root=logs_root, arm=arm)
                grid = ResetFreeTrainingCurves.checkpoints(runs=runs)
                curves = ResetFreeTrainingCurves.per_seed_curves(runs=runs)
                solved, total = ResetFreeTrainingCurves.arm_total(runs=runs)
                color = _POLICY_COLOR[policy]
                for row in curves:
                    axis.plot(grid, row, color=color, alpha=_SEED_ALPHA, linewidth=_SEED_WIDTH)
                # `strict` because a ragged transpose would silently truncate the mean to
                # the shortest seed's curve; `checkpoints` has already proved they align.
                mean = [sum(column) / len(column) for column in zip(*curves, strict=True)]
                axis.plot(
                    grid,
                    mean,
                    color=color,
                    linewidth=_MEAN_WIDTH,
                    label=f"{_POLICY_LABELS[policy]} -- final {solved}/{total}",
                )
            axis.set_title(_PANEL_TITLES[panel], fontsize=11, loc="left")
            axis.set_xlabel("online transitions")
            axis.grid(alpha=0.25, linewidth=0.6)
            axis.legend(frameon=False, loc="upper left", fontsize=9)
            axis.margins(x=0.06)
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)

        # Both left-hand panels, since `sharey` hides the tick labels on the right column
        # but not the axis label.
        for axis in (axes[0], axes[2]):
            axis.set_ylabel(f"tasks solved per evaluation (x/{_NUM_TEST_TASKS})")
        figure.suptitle(
            "Reset-free practice across all four Tossing Room variants "
            f"(bold = mean of {_NUM_SEEDS} seeds, faint = individual seeds)",
            fontsize=12.5,
        )
        figure.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, bbox_inches="tight", facecolor=_CANVAS)
        return figure

    @staticmethod
    def report(*, logs_root: Path) -> str:
        """The numbers behind the figure, as x/y with per-seed finals beside them."""
        lines = []
        for panel in _PANELS:
            for policy in _POLICIES:
                arm = ResetFreeTrainingCurves.arm(panel=panel, policy=policy)
                runs = ResetFreeTrainingCurves.load_arm(logs_root=logs_root, arm=arm)
                solved, total = ResetFreeTrainingCurves.arm_total(runs=runs)
                finals = ResetFreeTrainingCurves.per_seed_finals(runs=runs)
                lines.append(
                    f"{panel:>14s} / {policy:<9s} final {solved}/{total}  per-seed {finals}"
                )
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
    ResetFreeTrainingCurves.render(logs_root=args.logs_root, output=args.output)
    print(ResetFreeTrainingCurves.report(logs_root=args.logs_root))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

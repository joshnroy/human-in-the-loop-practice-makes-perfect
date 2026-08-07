"""Post-run analysis for the completed 2x2x2: training curves for `scheduled` against
`never` in **all four** (variant x ledge) worlds, on one four-panel figure.

**Background.** Three PRs built a 2x2 and filled three of its cells. Reset-free practice
was measured worse than scheduled-reset practice on Tossing Room (PR #115), and two
independent mechanisms were proposed and then each removed on its own:

* **frozen sampler inputs** -- `reset_to_task` is the only thing that installs a task's
  initial state, so under `never` every per-task quantity the sampler reads stays at the
  one `hard_reset` value. `tossingroomsplitpickupweight` (PR #121/#122) removes this by
  drawing the item's weight at pickup off a pre-sampled per-task array.
* **stranding** -- the ledge out of room 2 is one-way and the pile (the only item source)
  is in room 3, so a practice period that walks left once can never pick anything up
  again. `--two-way-ledge` (PR #124/#125) removes this.

Each helped alone and neither closed the gap. The fourth cell -- the pickup-weight fork
*with* the two-way ledge, i.e. both mechanisms removed at once -- is what this reads back
alongside the other three.

**Why a curve per panel and not a bar of finals.** Two features of the banked data are
invisible in a mean, and both are load-bearing:

* `pickup-weight / never` (one-way) is **bimodal**: 4/10 seeds finish at 16-21 and 6/10
  at 5-7, matching the reported "6/10 seeds draw exactly one weight for the whole run"
  seed-for-seed. That low mode *is* stranding, showing up in task outcomes.
* `one-way / scheduled` seed 8 finishes 6/30, a clear outlier dragging that arm's mean
  and unremarkable in every other variant.

So every panel draws **faint per-seed lines under a bold mean**. A bar chart of two means
would hide a mode split behind a number no seed actually sits at, and `mode_split` below
reports the split as two counts rather than as the word "bimodal".

**Cross-world comparisons are never made on raw counts.** `--two-way-ledge` also makes
the domain easier -- EMPTY stops being an ordering task, so its shortest solve drops
10 -> 9 and the evaluation horizon 12 -> 11, and RECYCLING stops being one-shot. The
comparable quantity across worlds is the **within-world gap** between the two reset
policies, which carries the same difficulty in both of its terms and so cancels. Nothing
here subtracts a two-way count from a one-way one.

**Reads only already-produced output** (CLAUDE.md's `analysis/` convention -- this never
runs a simulation or drives a `Method`). It reads each run's `stats.json` `evaluations`
list, which is `[transitions, solved, total]` per evaluation sweep.

**Both committed directory layouts are handled.** PR #125's runs sit at
`<arm>/ees/<seed>/stats.json` and PR #122's at `<arm>/<seed>/stats.json`; a loader that
understood only one would silently read a 0-seed arm for the other. A missing seed raises
rather than being skipped, so a 9-seed arm can never be reported as a 10-seed one.

**Counts, never bare percentages.** `arm_total` returns `(successes, total)` and every
caption prints `x/y`; the denominator is carried from the data rather than assumed, so a
short sweep reports `x/60` instead of claiming a 300-task denominator it does not have.
"""

import argparse
import json
from pathlib import Path
from typing import ClassVar

import matplotlib
from pydantic import BaseModel

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


class PanelKey(BaseModel, frozen=True):
    """One panel of the 2x2: which domain variant, and which ledge. Frozen because it
    sits in a dict key."""

    variant: str
    ledge: str


class SeedCurve(BaseModel):
    """One run's evaluation curve: `(transitions, solved, total)` per sweep, in order."""

    seed: int
    points: list[tuple[int, int, int]]

    @property
    def final(self) -> int:
        return self.points[-1][1]

    @property
    def denominator(self) -> int:
        return self.points[-1][2]


class Arm(BaseModel):
    """One (variant, ledge, policy) cell: its per-seed curves, in seed order."""

    curves: list[SeedCurve]


class ModeSplit(BaseModel):
    """A per-seed final-score distribution described as two counts and the gap between
    the clusters, rather than as the word "bimodal".

    The split is taken at the **largest gap** between consecutive sorted finals. That is
    a description, not a test: with 10 seeds there is always a largest gap, so a big
    `gap` beside a balanced `low_count`/`high_count` is evidence of two modes and a small
    one is evidence against. The counts are what get reported."""

    low_count: int
    high_count: int
    total: int
    gap: int
    low_values: list[int]
    high_values: list[int]


class CrossVariantResetPolicyCurves:
    """Static-method container, per this project's convention for stateless business
    logic -- never instantiated."""

    variants: ClassVar[tuple[str, ...]] = ("tossingroomsplit", "tossingroomsplitpickupweight")
    ledges: ClassVar[tuple[str, ...]] = ("one-way", "two-way")
    policies: ClassVar[tuple[str, ...]] = ("scheduled", "never")
    # 95% two-sided, 80% power: z(0.975) + z(0.80) = 1.959964 + 0.841621.
    mde_z_sum: ClassVar[float] = 2.801585

    @staticmethod
    def panel_keys() -> tuple[PanelKey, ...]:
        """The four cells, derived from the (variant, ledge) product rather than written
        out as four strings -- so a caller that omits one gets a KeyError rather than a
        quietly absent panel."""
        return tuple(
            PanelKey(variant=variant, ledge=ledge)
            for variant in CrossVariantResetPolicyCurves.variants
            for ledge in CrossVariantResetPolicyCurves.ledges
        )

    @staticmethod
    def load_arm(*, results_root: Path, num_seeds: int) -> Arm:
        """Read `num_seeds` runs from either committed layout.

        Raises `FileNotFoundError` naming the seed when one is absent. Skipping it
        silently would report a short arm as a full one, which is the single easiest way
        to publish a wrong denominator."""
        curves: list[SeedCurve] = []
        for seed in range(num_seeds):
            candidates = [
                results_root / "ees" / str(seed) / "stats.json",
                results_root / str(seed) / "stats.json",
            ]
            found = next((path for path in candidates if path.exists()), None)
            if found is None:
                raise FileNotFoundError(
                    f"seed {seed} missing under {results_root} "
                    f"(looked for {' and '.join(str(c) for c in candidates)})"
                )
            evaluations = json.loads(found.read_text())["evaluations"]
            curves.append(
                SeedCurve(
                    seed=seed,
                    points=[(int(t), int(solved), int(total)) for t, solved, total in evaluations],
                )
            )
        return Arm(curves=curves)

    @staticmethod
    def final_scores(*, arm: Arm) -> list[int]:
        """Each seed's final evaluation score, in seed order."""
        return [curve.final for curve in arm.curves]

    @staticmethod
    def arm_total(*, arm: Arm) -> tuple[int, int]:
        """`(successes, total)` -- the arm's `x/y`. The denominator is summed from the
        runs' own reported totals rather than assumed to be 30 per seed."""
        return (
            sum(curve.final for curve in arm.curves),
            sum(curve.denominator for curve in arm.curves),
        )

    @staticmethod
    def minimum_detectable_effect(*, successes: tuple[int, int], totals: tuple[int, int]) -> float:
        """Smallest difference in proportions this comparison could have detected, at
        `2.801585 * sqrt(p_bar*(1-p_bar)*(1/n1 + 1/n2))`.

        Computed **per comparison from its own two denominators**, never once for the
        project: 300 pooled tasks per arm against 20 for EMPTY, so one shared MDE would
        flatter the small one."""
        pooled = (successes[0] + successes[1]) / (totals[0] + totals[1])
        return (
            CrossVariantResetPolicyCurves.mde_z_sum
            * (pooled * (1.0 - pooled) * (1.0 / totals[0] + 1.0 / totals[1])) ** 0.5
        )

    @staticmethod
    def mode_split(*, arm: Arm) -> ModeSplit:
        """Split the per-seed finals at their largest consecutive gap. See `ModeSplit`
        for why this describes rather than tests."""
        finals = sorted(CrossVariantResetPolicyCurves.final_scores(arm=arm))
        gaps = [(finals[i + 1] - finals[i], i) for i in range(len(finals) - 1)]
        gap, index = max(gaps)
        return ModeSplit(
            low_count=index + 1,
            high_count=len(finals) - index - 1,
            total=len(finals),
            gap=gap,
            low_values=finals[: index + 1],
            high_values=finals[index + 1 :],
        )

    @staticmethod
    def render(*, arms: dict[tuple[PanelKey, str], Arm], output: Path) -> None:
        """The four-panel figure: one panel per (variant, ledge), both policies in each,
        faint per-seed lines under a bold mean."""
        keys = CrossVariantResetPolicyCurves.panel_keys()
        fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4), sharey=True)
        colors = {"scheduled": "#1f77b4", "never": "#d62728"}
        for ax, key in zip(axes.flat, keys, strict=True):
            denominator = 0
            for policy in CrossVariantResetPolicyCurves.policies:
                arm = arms[(key, policy)]
                for curve in arm.curves:
                    ax.plot(
                        [p[0] for p in curve.points],
                        [p[1] for p in curve.points],
                        color=colors[policy],
                        alpha=0.22,
                        linewidth=1.0,
                        zorder=2,
                    )
                transitions = [p[0] for p in arm.curves[0].points]
                means = [
                    sum(curve.points[i][1] for curve in arm.curves) / len(arm.curves)
                    for i in range(len(transitions))
                ]
                successes, total = CrossVariantResetPolicyCurves.arm_total(arm=arm)
                denominator = total
                ax.plot(
                    transitions,
                    means,
                    color=colors[policy],
                    linewidth=2.8,
                    zorder=4,
                    label=f"{policy}: {successes}/{total}",
                )
            ax.set_title(f"{key.variant}  |  {key.ledge} ledge", fontsize=10)
            ax.set_xlabel("practice transitions")
            ax.set_ylabel(f"tasks solved / {arms[(key, 'never')].curves[0].denominator}")
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(alpha=0.25)
            del denominator
        fig.suptitle(
            "Reset-free vs scheduled-reset practice across both mechanisms\n"
            "bold = mean over seeds, faint = per-seed; arm totals are x/y over all seeds",
            fontsize=11,
        )
        fig.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
        plt.close(fig)

    @staticmethod
    def aggregate(*, arms: dict[tuple[PanelKey, str], Arm]) -> dict:
        """The committed record. Raw sweep directories live outside the repo and do not
        travel between machines, so this JSON is what survives and what the figure can be
        regenerated from."""
        return {
            f"{key.variant}|{key.ledge}|{policy}": {
                str(curve.seed): [list(point) for point in curve.points] for curve in arm.curves
            }
            for (key, policy), arm in arms.items()
        }

    @staticmethod
    def load_aggregate(*, path: Path) -> dict[tuple[PanelKey, str], Arm]:
        raw = json.loads(path.read_text())
        arms: dict[tuple[PanelKey, str], Arm] = {}
        for name, seeds in raw.items():
            variant, ledge, policy = name.split("|")
            arms[(PanelKey(variant=variant, ledge=ledge), policy)] = Arm(
                curves=[
                    SeedCurve(
                        seed=int(seed),
                        points=[(int(t), int(s), int(n)) for t, s, n in points],
                    )
                    for seed, points in sorted(seeds.items(), key=lambda kv: int(kv[0]))
                ]
            )
        return arms

    @staticmethod
    def report(*, arms: dict[tuple[PanelKey, str], Arm]) -> str:
        """The numbers the experiment log quotes, as text -- every one an `x/y`."""
        lines: list[str] = []
        for key in CrossVariantResetPolicyCurves.panel_keys():
            lines.append(f"{key.variant} | {key.ledge}")
            for policy in CrossVariantResetPolicyCurves.policies:
                arm = arms[(key, policy)]
                successes, total = CrossVariantResetPolicyCurves.arm_total(arm=arm)
                finals = CrossVariantResetPolicyCurves.final_scores(arm=arm)
                split = CrossVariantResetPolicyCurves.mode_split(arm=arm)
                lines.append(
                    f"  {policy:10s} {successes}/{total}  per-seed {finals}\n"
                    f"    largest-gap split: {split.low_count}/{split.total} low "
                    f"{split.low_values} | {split.high_count}/{split.total} high "
                    f"{split.high_values} | gap {split.gap}"
                )
            scheduled = CrossVariantResetPolicyCurves.arm_total(arm=arms[(key, "scheduled")])
            never = CrossVariantResetPolicyCurves.arm_total(arm=arms[(key, "never")])
            mde = CrossVariantResetPolicyCurves.minimum_detectable_effect(
                successes=(scheduled[0], never[0]), totals=(scheduled[1], never[1])
            )
            gap = scheduled[0] - never[0]
            lines.append(
                f"  within-world gap (scheduled - never): {gap}/{scheduled[1]}"
                f"   MDE {mde:.4f} (= {mde * scheduled[1]:.1f}/{scheduled[1]})"
            )
        return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="VARIANT|LEDGE|POLICY=DIR",
        help="Repeatable. Sweep directory for one of the eight cells.",
    )
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--aggregate-output", type=Path, default=None)
    parser.add_argument("--arms-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="The four-panel figure.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.arms_json is not None:
        arms = CrossVariantResetPolicyCurves.load_aggregate(path=args.arms_json)
    else:
        arms = {}
        for entry in args.arm:
            name, _, directory = entry.partition("=")
            variant, ledge, policy = name.split("|")
            arms[(PanelKey(variant=variant, ledge=ledge), policy)] = (
                CrossVariantResetPolicyCurves.load_arm(
                    results_root=Path(directory), num_seeds=args.num_seeds
                )
            )
    print(CrossVariantResetPolicyCurves.report(arms=arms))
    if args.aggregate_output is not None:
        args.aggregate_output.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate_output.write_text(
            json.dumps(CrossVariantResetPolicyCurves.aggregate(arms=arms), indent=1, sort_keys=True)
        )
    if args.output is not None:
        CrossVariantResetPolicyCurves.render(arms=arms, output=args.output)


if __name__ == "__main__":
    main()

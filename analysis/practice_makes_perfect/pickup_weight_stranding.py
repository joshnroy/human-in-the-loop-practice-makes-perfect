"""Post-run analysis for the pickup-weight reset-free A/B: when a reset-free practice
arm loses, is it because it stopped *reaching the pile*?

`docs/experiment-logs/2026-08-07-pickup-weight-reset-free-ab.md` holds the
pre-registration. The short version: the 2026-08-06 A/B on `tossingroomsplit` confounded
two mechanisms -- practice stranding itself behind the one-way ledge, and the training
distribution collapsing because `reset_to_task` is the only thing that installs a task's
continuous parameters and the reset-free arm never calls it. `tossingroom`
removes the second by construction (the item weight is drawn at pickup, off a per-task
pre-sampled array, and the bin distance is fixed), so what is left to measure is the
first.

**This reads `stats.json` and nothing else** (CLAUDE.md's `analysis/` convention -- it
never runs a simulation or drives a `Method`). That is possible only because PR #111 put
`practice_outcomes_per_cycle` -- per lifted skill, per window, attempts and successes --
into every run's own output. The 2026-08-06 experiment needed a bespoke per-domain
collector script and a second set of runs to answer the same question; this needs neither.

**The two quantities, and why each is a count.**

* **Pile access, per seed per period.** A period reached the pile iff it executed a
  `PickupTrash` or a `PickupRecycling`; both are applicable only in the pile's room.
  Reported as an `x/y` grid, one cell per (seed, period), not as a mean over onsets --
  the onsets in the 2026-08-06 data are bimodal (6/10 seeds at period 1, 2/10 at period
  3, 1/10 at period 4, 1/10 never), so a mean over them describes no seed.
* **Weight draws, per seed.** Under weight-at-pickup a run's weight sample size **is**
  its pickup count, exactly. That identity is the point: a seed that strands on its first
  `PickupRecycling` draws one weight for the entire run, which is n=1 unidentifiability
  rather than a sparse or a biased sample. Trash and recycling weights come from one law
  (`Uniform[0.5, 1.5)`) and one stream, so no weight region is systematically excluded;
  what stranding biases is the item *type*, which is why the two kinds are counted apart.

**Stranding is terminal-from-here, not "the first gap".** The onset is the first period
after which no period reaches the pile again. A run that misses the pile for one period
and comes back was never stranded, and calling it stranded would turn ordinary
exploration noise into the effect being claimed. A run that never reaches the pile at all
strands at period 0; a run that always reaches it reports `None`.

**A missing seed is an error, not a skip.** A reader that skips one silently reports a
9-seed result as a 10-seed one.
"""

import argparse
import json
from pathlib import Path
from typing import ClassVar

import matplotlib
from pydantic import BaseModel

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


class SeedStranding(BaseModel):
    """One run's stranding record, all counts, no rates."""

    seed: int
    # One entry per interaction period, in order: did that period reach the pile?
    pile_access: list[bool]
    # First period of the terminal run of no-access periods, or None if never stranded.
    stranding_onset: int | None
    num_trash_pickups: int
    num_recycling_pickups: int
    # Every lifted skill that executed at least once at or after the onset -- the
    # evidence for terminality being a real property of the layout rather than of a
    # short horizon.
    post_onset_skills: set[str]

    @property
    def is_stranded(self) -> bool:
        return self.stranding_onset is not None

    @property
    def num_weight_draws(self) -> int:
        """Pickups, which under this domain's dynamics is exactly the number of weights
        this run ever drew -- see the module docstring."""
        return self.num_trash_pickups + self.num_recycling_pickups

    @property
    def num_post_onset_periods(self) -> int:
        if self.stranding_onset is None:
            return 0
        return len(self.pile_access) - self.stranding_onset


class PickupWeightStranding:
    """A static-method container, never instantiated, same as every other
    business-logic class in this project."""

    # Applicable only in the pile's room, so executing one is proof of pile access.
    PICKUP_SKILLS: ClassVar[tuple[str, ...]] = ("PickupTrash", "PickupRecycling")

    @staticmethod
    def read_arm(*, root: Path, seeds: list[int]) -> list[SeedStranding]:
        """One arm's per-seed records, read from `root/ees/<seed>/stats.json`."""
        return [
            PickupWeightStranding.read_run(path=root / "ees" / str(seed) / "stats.json", seed=seed)
            for seed in seeds
        ]

    @staticmethod
    def read_run(*, path: Path, seed: int) -> SeedStranding:
        if not path.exists():
            raise FileNotFoundError(f"seed {seed}: no stats.json at {path}")
        stats = json.loads(path.read_text())
        periods = PickupWeightStranding.practice_periods(stats=stats)
        access = [PickupWeightStranding.reached_pile(period=period) for period in periods]
        onset = PickupWeightStranding.stranding_onset(access=access)
        return SeedStranding(
            seed=seed,
            pile_access=access,
            stranding_onset=onset,
            num_trash_pickups=sum(
                PickupWeightStranding.attempts(period=period, skill="PickupTrash")
                for period in periods
            ),
            num_recycling_pickups=sum(
                PickupWeightStranding.attempts(period=period, skill="PickupRecycling")
                for period in periods
            ),
            post_onset_skills={
                name
                for period in (periods[onset:] if onset is not None else [])
                for name, tally in period.items()
                if int(tally.get("num_attempts", 0)) > 0
            },
        )

    @staticmethod
    def practice_periods(*, stats: dict[str, object]) -> list[dict[str, dict[str, int]]]:
        """The windows that actually contain practice.

        `Metrics.record_practice_outcomes` appends one entry per window PLUS a final one
        covering the last evaluation sweep alone, which contains no practice at all. That
        trailing entry is dropped here rather than filtered on emptiness, because a real
        run writes it as every skill at zero rather than as `{}` -- filtering on
        emptiness would keep it and hand every run in both arms one phantom stranded
        period."""
        windows = stats.get("practice_outcomes_per_cycle") or []
        assert isinstance(windows, list)
        return list(windows[:-1])

    @staticmethod
    def attempts(*, period: dict[str, dict[str, int]], skill: str) -> int:
        return int(period.get(skill, {}).get("num_attempts", 0))

    @staticmethod
    def reached_pile(*, period: dict[str, dict[str, int]]) -> bool:
        return any(
            PickupWeightStranding.attempts(period=period, skill=skill) > 0
            for skill in PickupWeightStranding.PICKUP_SKILLS
        )

    @staticmethod
    def stranding_onset(*, access: list[bool]) -> int | None:
        """First index after which no period reaches the pile again, or None.

        Computed from the END backwards precisely so a recovered gap cannot be mistaken
        for an onset: the answer is the length of the terminal all-False suffix.

        **An onset at the LAST period is not evidence of stranding**, and the summary
        counts it separately for that reason. A run whose final period happens to take
        no pickup is indistinguishable, from this data alone, from one that stranded
        going into it -- there is no later period to fail to recover in. Measured on the
        `scheduled` arm, where the per-period reset makes stranding impossible by
        construction, 5/10 seeds still report an onset at period 9 and 0/10 report one
        anywhere earlier. `num_stranded_before_last_period` is the defensible count."""
        onset = len(access)
        while onset > 0 and not access[onset - 1]:
            onset -= 1
        return None if onset == len(access) else onset

    @staticmethod
    def summarise(*, records: list[SeedStranding]) -> dict[str, object]:
        """The arm's headline counts. Every one carries its denominator via
        `num_seeds`; nothing here is a rate."""
        return {
            "num_seeds": len(records),
            "num_stranded": sum(record.is_stranded for record in records),
            # The defensible count: an onset at the final period has no later period to
            # recover in, so it is not evidence -- see `stranding_onset`.
            "num_stranded_before_last_period": sum(
                record.stranding_onset is not None
                and record.stranding_onset < len(record.pile_access) - 1
                for record in records
            ),
            "num_seeds_with_one_weight_draw": sum(
                record.num_weight_draws == 1 for record in records
            ),
            "stranding_onsets": [record.stranding_onset for record in records],
            "weight_draws_per_seed": [record.num_weight_draws for record in records],
            "trash_pickups_per_seed": [record.num_trash_pickups for record in records],
            "recycling_pickups_per_seed": [record.num_recycling_pickups for record in records],
            "periods_with_pile_access_per_seed": [sum(record.pile_access) for record in records],
            "num_periods": max((len(record.pile_access) for record in records), default=0),
            "num_post_onset_periods": sum(record.num_post_onset_periods for record in records),
            "post_onset_skills": sorted({
                name for record in records for name in record.post_onset_skills
            }),
        }

    @staticmethod
    def report(*, arms: dict[str, dict[str, object]]) -> str:
        lines: list[str] = []
        for arm, summary in arms.items():
            seeds = summary["num_seeds"]
            lines.append(f"[{arm}]")
            lines.append(f"  stranded seeds:            {summary['num_stranded']}/{seeds}")
            lines.append(
                f"  ...before the last period: {summary['num_stranded_before_last_period']}/{seeds}"
            )
            lines.append(
                f"  seeds drawing 1 weight:    {summary['num_seeds_with_one_weight_draw']}/{seeds}"
            )
            lines.append(f"  stranding onsets:          {summary['stranding_onsets']}")
            lines.append(f"  weight draws per seed:     {summary['weight_draws_per_seed']}")
            lines.append(f"  trash pickups per seed:    {summary['trash_pickups_per_seed']}")
            lines.append(f"  recyc pickups per seed:    {summary['recycling_pickups_per_seed']}")
            lines.append(
                f"  periods with pile access:  {summary['periods_with_pile_access_per_seed']}"
                f" (of {summary['num_periods']} each)"
            )
            lines.append(
                f"  skills run post-onset:     {summary['post_onset_skills']}"
                f" over {summary['num_post_onset_periods']} periods"
            )
        return "\n".join(lines)

    # The project's figure palette, matched to the sibling reset-policy figures.
    SURFACE: ClassVar[str] = "#fcfcfb"
    INK: ClassVar[str] = "#22252a"
    MUTED: ClassVar[str] = "#7c828c"
    ABSENT: ClassVar[str] = "#e6e8ea"
    ARM_COLORS: ClassVar[dict[str, str]] = {"scheduled": "#3b7dd8", "never": "#e8833a"}

    @staticmethod
    def plot(*, arms: dict[str, list[SeedStranding]], output: Path) -> None:
        """Per-seed, per-period pile access, one panel per arm.

        A grid rather than a curve: each cell is one (seed, period) observation, so the
        reader sees the per-seed spread the pre-registration asked for and can count the
        denominators off the axis instead of trusting a mean over bimodal onsets."""
        fig, axes = plt.subplots(
            1, len(arms), figsize=(6.2 * len(arms), 4.9), facecolor=PickupWeightStranding.SURFACE
        )
        axes = axes if len(arms) > 1 else [axes]
        for ax, (arm, records) in zip(axes, arms.items(), strict=True):
            color = PickupWeightStranding.ARM_COLORS.get(arm, "#6f6f6f")
            ax.set_facecolor(PickupWeightStranding.SURFACE)
            num_periods = max(len(record.pile_access) for record in records)
            for row, record in enumerate(records):
                for index, reached in enumerate(record.pile_access):
                    ax.add_patch(
                        plt.Rectangle(
                            (index - 0.38, row - 0.38),
                            0.76,
                            0.76,
                            facecolor=color if reached else PickupWeightStranding.ABSENT,
                            edgecolor=PickupWeightStranding.SURFACE,
                            linewidth=2.0,
                            zorder=3 if reached else 2,
                        )
                    )
                ax.text(
                    num_periods + 0.15,
                    row,
                    f"{sum(record.pile_access)}/{len(record.pile_access)}",
                    va="center",
                    ha="left",
                    fontsize=9.5,
                    color=PickupWeightStranding.MUTED,
                )
            ax.set_xlim(-0.6, num_periods + 1.3)
            ax.set_ylim(len(records) - 0.4, -0.6)
            ax.set_xticks(range(num_periods))
            ax.set_yticks(range(len(records)))
            ax.set_yticklabels([f"seed {record.seed}" for record in records], fontsize=9)
            ax.set_xlabel("interaction period", fontsize=10, color=PickupWeightStranding.MUTED)
            ax.set_title(
                f"`{arm}`  -  periods that reached the pile",
                fontsize=12,
                color=PickupWeightStranding.INK,
                pad=10,
                loc="left",
            )
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(length=0, colors=PickupWeightStranding.MUTED)
        fig.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    marker="s",
                    linestyle="",
                    markersize=10,
                    color="#6f6f6f",
                    label="reached the pile (>=1 pickup)",
                ),
                Line2D(
                    [],
                    [],
                    marker="s",
                    linestyle="",
                    markersize=10,
                    color=PickupWeightStranding.ABSENT,
                    label="no pickup at all",
                ),
            ],
            loc="lower center",
            ncol=2,
            frameon=False,
            fontsize=9.5,
            labelcolor=PickupWeightStranding.MUTED,
        )
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160, facecolor=PickupWeightStranding.SURFACE)
        plt.close(fig)

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--arm",
            action="append",
            default=[],
            metavar="NAME=DIR",
            help="An arm's sweep root, i.e. the directory holding ees/<seed>/stats.json.",
        )
        parser.add_argument("--num-seeds", type=int, default=10)
        parser.add_argument("--aggregate-output", type=Path, default=None)
        parser.add_argument("--arms-json", type=Path, default=None)
        parser.add_argument("--output", type=Path, default=None, help="Pile-access figure.")
        args = parser.parse_args()

        seeds = list(range(args.num_seeds))
        if args.arm:
            arms = {
                name: PickupWeightStranding.read_arm(root=Path(root), seeds=seeds)
                for name, root in (entry.split("=", 1) for entry in args.arm)
            }
        elif args.arms_json is not None:
            raw = json.loads(args.arms_json.read_text())
            arms = {
                name: [SeedStranding(**record) for record in records]
                for name, records in raw.items()
            }
        else:
            parser.error("pass --arm NAME=DIR (to read a sweep) or --arms-json (to re-render)")

        print(
            PickupWeightStranding.report(
                arms={
                    name: PickupWeightStranding.summarise(records=records)
                    for name, records in arms.items()
                }
            )
        )
        if args.aggregate_output is not None:
            args.aggregate_output.parent.mkdir(parents=True, exist_ok=True)
            args.aggregate_output.write_text(
                json.dumps(
                    {
                        name: [
                            {
                                **record.model_dump(),
                                "post_onset_skills": sorted(record.post_onset_skills),
                            }
                            for record in records
                        ]
                        for name, records in arms.items()
                    },
                    indent=2,
                )
                + "\n"
            )
        if args.output is not None:
            PickupWeightStranding.plot(arms=arms, output=args.output)


if __name__ == "__main__":
    PickupWeightStranding.main()

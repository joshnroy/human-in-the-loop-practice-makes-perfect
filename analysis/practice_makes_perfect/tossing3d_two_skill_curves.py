"""Post-run reader for a **two-skill** Tossing3D sweep: `PickCube` and
`MoveToTossLocationAndToss`.

Post-run analysis only. It reads `--results-root` back in -- the
`<results-root>/<method>/<seed>/stats.json` layout `scripts/run_sweep.py` writes -- and
never constructs a `Problem`, `Method` or `Environment`.

## Why a new reader rather than repointing the old one

`tossing3d_ees_arms.py`, `tossing3d_practice_diagnosis.py` and
`tossing3d_reset_free_arms.py` key on the pre-migration names `Pick`, `MoveToThrowPose`
and `Toss`. Those names no longer exist. All three read their per-skill tallies with
`.get(skill_name)`, so pointed at a two-skill tree they find nothing, fall back to the
`{}` default, and report `0/0` -- **a confident empty plot, with no error**. That is the
worst available failure: it looks like a measurement.

Repointing the names would not have been enough. Those modules' arguments are about
`MoveToThrowPose`'s *informed* draws measured against a `Pick` control, and neither side
of that comparison survives the migration: the middle skill is gone, and the pick is now
the `param_dim == 0` one. So they are left as the record of what they measured (each
carries its own staleness banner) and this module reads the new decomposition.

## The guard, and the trap inside it

The obvious guard -- "raise if no skill shows learnable activity" -- **reintroduces the
bug it is meant to catch.** `PickCube` declares `param_dim == 0`. It has no sampler, none
can be constructed for it, and it therefore contributes nothing to any competence plot
*when everything is working correctly*. A guard that fires on the healthy case gets muted,
and a muted guard protects nothing.

So the two empty cases are separated by asking two independent questions, and a skill's
`status` is the answer to both together:

| status | name present? | data | verdict |
| --- | --- | --- | --- |
| `missing` | no | -- | **raises**: wrong domain, or a rename |
| `contradicts-declared-param-dim` | yes | disagrees with `skills.py` | **raises**: wrong vintage |
| `present-but-unpracticed` | yes | zero attempts | reported, does not raise |
| `unlearnable-by-construction` | yes | all draws unparameterized | **correct**, does not raise |
| `learnable` | yes | sampler was consultable | correct |

The distinction is drawn from **recorded data**, not from an assumption:
`SkillPracticeTally.num_unparameterized_attempts` counts exactly the executions where
`param_dim == 0` meant no sampler existed, so "correctly empty" is observable in
`stats.json` rather than inferred. `EXPECTED_SKILLS` carries the declared `param_dim`
beside each name so the two can be cross-checked, and
`tests/analysis/practice_makes_perfect/test_tossing3d_two_skill_curves.py` pins both
against `Tossing3DSkills` itself so this constant cannot drift away from `src/`.

`present-but-unpracticed` is kept apart from `unlearnable-by-construction` deliberately:
folding them together would either raise on a correct run that happened to practise
nothing, or call a genuinely empty run healthy.

## What it reports

Counts as `x/y`, never a bare percentage, everywhere -- printed cells, legend entries and
panel titles alike.

The figure follows CLAUDE.md's fixed training-curve style: `ees` in the spec blue with
faint per-seed traces under a bold mean; `skill-oracle` and `random-skills` as flat
horizontal grey reference lines, never curves, because neither learns.

**The per-seed traces are the point here, not decoration.** The pooled mean on this sweep
hides that every seed reaches `10/10` at some checkpoint and most end below their own
peak, which is a different claim about the method than a mean of `8.4` supports.

**Best-ever is reported beside final, and it is upward-biased.** It is a maximum over 21
noisy checkpoints, so it exceeds the final sweep even for an arm that learns nothing --
`random-skills` on this very sweep is the calibration for how much. `best_versus_final`
returns both arms' gaps for exactly that comparison; do not quote a best-ever number
without it.
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib
from pydantic import BaseModel

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hitl_pmp.core.metrics.metrics import Metrics  # noqa: E402

# The two lifted skills this domain declares, with the `param_dim` each one declares,
# copied from `hitl_pmp.environments.tossing3d.skills.Tossing3DSkills`. Pinned against
# that class by this module's own tests rather than imported, so the reader stays a pure
# data reader and still cannot drift from source.
EXPECTED_SKILLS: dict[str, int] = {
    "PickCube": 0,
    "MoveToTossLocationAndToss": 4,
}

# The one arm under test. The other two never learn and are drawn as flat lines.
LEARNING_ARM = "ees"
REFERENCE_ARMS = ("skill-oracle", "random-skills")

# CLAUDE.md's training-curve style. Blue is the arm with a learning mechanism; the
# reference/ceiling arms share one neutral grey, dotted, and are separated by their
# legend label and y-level rather than by hue. No third colour is introduced.
_EES_COLOUR = "#0072B2"
_REFERENCE_COLOUR = "#666666"
_SEED_TRACE_ALPHA = 0.16
_SEED_TRACE_WIDTH = 0.8
_MEAN_WIDTH = 2.3


class SkillCoverage(BaseModel):
    """What a results tree actually recorded for one *expected* lifted skill.

    Deliberately carries both halves of the guard's question -- whether the name appeared
    at all, and what the recorded draws say about it -- so that `status` can answer them
    together and a caller can never read one without the other.
    """

    skill_name: str
    declared_param_dim: int
    is_present: bool
    num_attempts: int
    num_unparameterized_attempts: int
    num_informed_attempts: int
    num_random_attempts: int

    @property
    def status(self) -> str:
        """One of the five rows in this module's docstring table."""
        if not self.is_present:
            return "missing"
        if self.num_attempts == 0:
            return "present-but-unpracticed"
        all_draws_unparameterized = self.num_unparameterized_attempts == self.num_attempts
        if self.declared_param_dim == 0:
            # Source says no sampler can exist. The data must agree, or these results
            # were produced by a different version of the domain under the same name.
            return (
                "unlearnable-by-construction"
                if all_draws_unparameterized
                else "contradicts-declared-param-dim"
            )
        # Source says a sampler exists. Data claiming every draw was unparameterized
        # means the name matched but the decomposition behind it did not.
        return "contradicts-declared-param-dim" if all_draws_unparameterized else "learnable"

    @property
    def is_correctly_empty(self) -> bool:
        """True exactly when a competence plot for this skill should be empty and that
        is the right answer -- never because the names failed to match."""
        return self.status == "unlearnable-by-construction"

    @property
    def is_fatal(self) -> bool:
        """Whether this entry means the tree cannot be read as a two-skill sweep."""
        return self.status in {"missing", "contradicts-declared-param-dim"}


class SeedRegression(BaseModel):
    """One seed that ended below its own best checkpoint."""

    seed: int
    final: int
    best: int
    peak_index: int
    num_total: int

    @property
    def drop(self) -> int:
        return self.final - self.best


class SkillNameMismatchError(RuntimeError):
    """Raised when a results tree cannot be read as a two-skill Tossing3D sweep.

    A distinct exception type rather than `ValueError` because the *only* correct
    response is to stop -- the numbers that would come back otherwise are zeros that look
    like measurements.
    """


class Tossing3DTwoSkillCurves:
    """A static-method container, never instantiated, same as every other
    business-logic class in this project."""

    @staticmethod
    def load_runs(*, results_root: Path, method: str) -> dict[int, Metrics]:
        """`{seed: Metrics}` for one arm, read from `stats.json`.

        Only completed runs appear: a run still in flight has no `stats.json` and is
        silently absent rather than contributing a truncated curve.
        """
        runs: dict[int, Metrics] = {}
        for path in sorted((results_root / method).glob("*/stats.json")):
            runs[int(path.parent.name)] = Metrics.model_validate(json.loads(path.read_text()))
        return runs

    @staticmethod
    def load_curves(*, results_root: Path, method: str) -> dict[int, list[tuple[int, int, int]]]:
        """`{seed: [(transitions, solved, total), ...]}` for one arm."""
        return {
            seed: [(int(t), int(s), int(n)) for t, s, n in metrics.evaluations]
            for seed, metrics in Tossing3DTwoSkillCurves.load_runs(
                results_root=results_root, method=method
            ).items()
        }

    @staticmethod
    def skill_coverage(*, results_root: Path, method: str) -> list[SkillCoverage]:
        """One `SkillCoverage` per *expected* skill, pooled over every cycle and seed.

        Pooled rather than per-cycle because the question is "does this tree describe the
        two-skill domain at all", which no single cycle answers -- a cycle can legitimately
        record nothing.
        """
        runs = Tossing3DTwoSkillCurves.load_runs(results_root=results_root, method=method)
        present: set[str] = set()
        totals: dict[str, dict[str, int]] = {
            name: {"attempts": 0, "unparameterized": 0, "informed": 0, "random": 0}
            for name in EXPECTED_SKILLS
        }
        for metrics in runs.values():
            for cycle in metrics.practice_outcomes_per_cycle:
                for name, tally in cycle.items():
                    present.add(name)
                    if name not in totals:
                        continue
                    totals[name]["attempts"] += tally.num_attempts
                    totals[name]["unparameterized"] += tally.num_unparameterized_attempts
                    totals[name]["informed"] += tally.num_informed_attempts
                    totals[name]["random"] += tally.num_random_attempts
        return [
            SkillCoverage(
                skill_name=name,
                declared_param_dim=param_dim,
                is_present=name in present,
                num_attempts=totals[name]["attempts"],
                num_unparameterized_attempts=totals[name]["unparameterized"],
                num_informed_attempts=totals[name]["informed"],
                num_random_attempts=totals[name]["random"],
            )
            for name, param_dim in EXPECTED_SKILLS.items()
        ]

    @staticmethod
    def observed_skill_names(*, results_root: Path, method: str) -> list[str]:
        """Every lifted skill name the tree actually recorded, expected or not. Named in
        the guard's error so a reader sees both sides without going digging."""
        observed: set[str] = set()
        for metrics in Tossing3DTwoSkillCurves.load_runs(
            results_root=results_root, method=method
        ).values():
            for cycle in metrics.practice_outcomes_per_cycle:
                observed.update(cycle)
        return sorted(observed)

    @staticmethod
    def check_skill_names(
        *, coverage: list[SkillCoverage], observed: list[str] | None = None
    ) -> None:
        """Raise unless this tree can be read as a two-skill sweep.

        Does **not** raise on `unlearnable-by-construction` or `present-but-unpracticed`:
        the first is the healthy state of `PickCube` and the second is a real, readable
        run that happened to practise nothing. See this module's docstring.
        """
        fatal = [entry for entry in coverage if entry.is_fatal]
        if not fatal:
            return
        lines = [
            "This results tree cannot be read as a two-skill Tossing3D sweep, so no "
            "number is reported rather than reporting zeros.",
            f"Expected skills (name -> declared param_dim): {EXPECTED_SKILLS}",
            f"Observed skill names: {observed if observed is not None else '(not collected)'}",
        ]
        for entry in fatal:
            if entry.status == "missing":
                lines.append(
                    f"  - {entry.skill_name!r}: never recorded. The pre-migration domain "
                    "used 'Pick', 'MoveToThrowPose' and 'Toss'; a tree from that domain "
                    "reads as 0/0 in every per-skill tally."
                )
            else:
                lines.append(
                    f"  - {entry.skill_name!r}: name matches but the recorded draws "
                    f"contradict its declared param_dim={entry.declared_param_dim} "
                    f"({entry.num_unparameterized_attempts}/{entry.num_attempts} attempts "
                    "were unparameterized). These results predate the migration."
                )
        lines.append(
            "Note: 'PickCube' recording only unparameterized attempts is CORRECT -- it "
            "declares param_dim=0, so it has no sampler and its competence panel is "
            "legitimately empty. That case is not an error and is not listed above."
        )
        raise SkillNameMismatchError("\n".join(lines))

    @staticmethod
    def pooled_final(*, curves: dict[int, list[tuple[int, int, int]]]) -> tuple[int, int]:
        """`(x, y)` over every seed's **last** evaluation sweep."""
        return (
            sum(evaluations[-1][1] for evaluations in curves.values()),
            sum(evaluations[-1][2] for evaluations in curves.values()),
        )

    @staticmethod
    def pooled_best(*, curves: dict[int, list[tuple[int, int, int]]]) -> tuple[int, int]:
        """`(x, y)` over each seed's own best sweep.

        The maximum is taken **per seed and then summed**, never after pooling: the
        latter would ask "was there one sweep where every seed did well simultaneously",
        which is a different and much smaller quantity.

        Upward-biased as an estimate of ability, because it is a maximum over many noisy
        checkpoints. Always read it against `best_versus_final`.
        """
        return (
            sum(max(solved for _t, solved, _n in evaluations) for evaluations in curves.values()),
            sum(evaluations[-1][2] for evaluations in curves.values()),
        )

    @staticmethod
    def mean_final_per_seed(*, curves: dict[int, list[tuple[int, int, int]]]) -> float:
        """The mean over seeds of each seed's final solved count -- the y-level a flat
        reference line is drawn at."""
        return statistics.fmean(evaluations[-1][1] for evaluations in curves.values())

    @staticmethod
    def regressed_seeds(*, curves: dict[int, list[tuple[int, int, int]]]) -> list[SeedRegression]:
        """Every seed whose final sweep sits below its own best sweep, in seed order.

        `peak_index` is the **first** checkpoint attaining the maximum, so a seed that
        held its peak for a while is reported as peaking when it got there.
        """
        regressions: list[SeedRegression] = []
        for seed in sorted(curves):
            solved = [s for _t, s, _n in curves[seed]]
            best = max(solved)
            if solved[-1] < best:
                regressions.append(
                    SeedRegression(
                        seed=seed,
                        final=solved[-1],
                        best=best,
                        peak_index=solved.index(best),
                        num_total=curves[seed][-1][2],
                    )
                )
        return regressions

    @staticmethod
    def best_versus_final(
        *, curves_by_arm: dict[str, dict[int, list[tuple[int, int, int]]]]
    ) -> dict[str, list[int]]:
        """Per-seed `best - final` for each arm.

        Exists to stop a best-ever number being quoted on its own. A maximum over many
        noisy checkpoints exceeds the final sweep for *any* arm, including one that
        cannot learn, so a non-learning arm's gaps are the calibration for how much of
        the learning arm's gap is selection rather than regression.
        """
        return {
            arm: [
                max(s for _t, s, _n in evaluations) - evaluations[-1][1]
                for _seed, evaluations in sorted(curves.items())
            ]
            for arm, curves in curves_by_arm.items()
        }

    @staticmethod
    def print_report(
        *,
        curves_by_arm: dict[str, dict[int, list[tuple[int, int, int]]]],
        coverage: list[SkillCoverage],
    ) -> None:
        """Every cell an `x/y`, never a bare percentage."""
        print("Skill coverage (pooled over every cycle and seed, arm = " + LEARNING_ARM + ")")
        for entry in coverage:
            informed = f"{entry.num_informed_attempts}/{entry.num_attempts}"
            unparameterized = f"{entry.num_unparameterized_attempts}/{entry.num_attempts}"
            print(
                f"  {entry.skill_name:<28} param_dim={entry.declared_param_dim}  "
                f"informed {informed:>10}  unparameterized {unparameterized:>10}  "
                f"-> {entry.status}"
            )
            if entry.is_correctly_empty:
                print(
                    "      (correctly empty: no sampler exists for this skill, so it has "
                    "no competence to plot)"
                )
        print()
        print("Task success, per arm")
        for arm, curves in curves_by_arm.items():
            final_x, final_y = Tossing3DTwoSkillCurves.pooled_final(curves=curves)
            best_x, best_y = Tossing3DTwoSkillCurves.pooled_best(curves=curves)
            per_seed = sorted(evaluations[-1][1] for evaluations in curves.values())
            print(
                f"  {arm:<15} final {final_x}/{final_y}   best-ever {best_x}/{best_y}   "
                f"n={len(curves)} seeds   per-seed final {per_seed}"
            )
        print()
        gaps = Tossing3DTwoSkillCurves.best_versus_final(curves_by_arm=curves_by_arm)
        print("Per-seed (best - final), the selection-bias calibration")
        for arm, values in gaps.items():
            print(f"  {arm:<15} {values}")
        print()
        learning = curves_by_arm.get(LEARNING_ARM, {})
        regressed = Tossing3DTwoSkillCurves.regressed_seeds(curves=learning)
        print(
            f"{LEARNING_ARM}: {len(regressed)}/{len(learning)} seeds ended below their own "
            "best checkpoint"
        )
        for entry in regressed:
            print(
                f"  seed {entry.seed}: peaked {entry.best}/{entry.num_total} at checkpoint "
                f"{entry.peak_index}, ended {entry.final}/{entry.num_total} ({entry.drop})"
            )

    @staticmethod
    def plot(
        *,
        curves_by_arm: dict[str, dict[int, list[tuple[int, int, int]]]],
        output_path: Path,
    ) -> None:
        """Two panels: the training curves, and each seed's peak against where it ended.

        The right panel exists because the left one's bold mean cannot show that every
        seed touched the ceiling -- a mean over ten seeds that peak at different
        checkpoints never reaches the peak any of them reached.
        """
        figure, (curve_axis, span_axis) = plt.subplots(1, 2, figsize=(13.5, 5.2))
        learning = curves_by_arm.get(LEARNING_ARM, {})
        num_total = next(iter(learning.values()))[-1][2] if learning else 0

        for evaluations in learning.values():
            curve_axis.plot(
                [t for t, _s, _n in evaluations],
                [s for _t, s, _n in evaluations],
                color=_EES_COLOUR,
                alpha=_SEED_TRACE_ALPHA,
                linewidth=_SEED_TRACE_WIDTH,
                zorder=2,
            )
        if learning:
            transitions = [t for t, _s, _n in next(iter(learning.values()))]
            means = [
                statistics.fmean(evaluations[i][1] for evaluations in learning.values())
                for i in range(len(transitions))
            ]
            final_x, final_y = Tossing3DTwoSkillCurves.pooled_final(curves=learning)
            curve_axis.plot(
                transitions,
                means,
                color=_EES_COLOUR,
                linewidth=_MEAN_WIDTH,
                zorder=4,
                label=f"ees — mean, n={len(learning)} (final {final_x}/{final_y})",
            )

        for arm in REFERENCE_ARMS:
            curves = curves_by_arm.get(arm)
            if not curves:
                continue
            level = Tossing3DTwoSkillCurves.mean_final_per_seed(curves=curves)
            pooled_x, pooled_y = Tossing3DTwoSkillCurves.pooled_final(curves=curves)
            curve_axis.axhline(
                level,
                color=_REFERENCE_COLOUR,
                linestyle=":",
                linewidth=1.8,
                zorder=3,
                label=f"{arm} — {pooled_x}/{pooled_y}, n={len(curves)}",
            )

        curve_axis.set_xlabel("online transitions")
        curve_axis.set_ylabel("solved per seed")
        curve_axis.set_title(f"Tossing3D two-skill, test tasks (of {num_total}) — every seed drawn")
        curve_axis.set_ylim(-0.4, num_total + 0.4)
        curve_axis.legend(loc="lower right", fontsize=8)
        curve_axis.grid(alpha=0.2)

        seeds = sorted(learning)
        for offset, seed in enumerate(seeds):
            solved = [s for _t, s, _n in learning[seed]]
            span_axis.plot(
                [offset, offset],
                [solved[-1], max(solved)],
                color=_EES_COLOUR,
                alpha=0.45,
                linewidth=1.6,
                zorder=2,
            )
        span_axis.scatter(
            range(len(seeds)),
            [max(s for _t, s, _n in learning[seed]) for seed in seeds],
            color=_EES_COLOUR,
            marker="^",
            zorder=4,
            label=f"best-ever checkpoint, n={len(seeds)}",
        )
        span_axis.scatter(
            range(len(seeds)),
            [learning[seed][-1][1] for seed in seeds],
            color=_EES_COLOUR,
            marker="o",
            facecolors="white",
            zorder=4,
            label=f"final checkpoint, n={len(seeds)}",
        )
        for arm in REFERENCE_ARMS:
            curves = curves_by_arm.get(arm)
            if not curves:
                continue
            pooled_x, pooled_y = Tossing3DTwoSkillCurves.pooled_final(curves=curves)
            span_axis.axhline(
                Tossing3DTwoSkillCurves.mean_final_per_seed(curves=curves),
                color=_REFERENCE_COLOUR,
                linestyle=":",
                linewidth=1.8,
                zorder=3,
                label=f"{arm} — {pooled_x}/{pooled_y}, n={len(curves)}",
            )
        span_axis.set_xticks(range(len(seeds)))
        span_axis.set_xticklabels([str(seed) for seed in seeds])
        span_axis.set_xlabel("seed")
        span_axis.set_ylabel("solved")
        span_axis.set_title(f"Peak vs final, test tasks (of {num_total}) — ees")
        span_axis.set_ylim(-0.4, num_total + 0.4)
        span_axis.legend(loc="lower right", fontsize=8)
        span_axis.grid(alpha=0.2)

        figure.tight_layout()
        figure.savefig(output_path, dpi=200)
        plt.close(figure)

    @staticmethod
    def plot_selection_bias(
        *,
        curves_by_arm: dict[str, dict[int, list[tuple[int, int, int]]]],
        output_path: Path,
    ) -> None:
        """Per-seed `best - final`, learning arm against non-learning control.

        The figure exists to stop one specific misreading. "Best-ever `100/100`, final
        `84/100`" invites the conclusion that the robot learned the task and then lost
        it. But a maximum over 21 noisy checkpoints exceeds the final checkpoint for
        *any* arm, and `random-skills` -- which consults no sampler and cannot improve --
        shows the same shape. Plotting the two side by side is the only way a reader can
        see how much of the learning arm's gap is regression rather than the maximum's
        own upward bias.
        """
        figure, axis = plt.subplots(figsize=(7.2, 5.0))
        arms = [arm for arm in (LEARNING_ARM, "random-skills") if curves_by_arm.get(arm)]
        gaps = Tossing3DTwoSkillCurves.best_versus_final(
            curves_by_arm={arm: curves_by_arm[arm] for arm in arms}
        )
        styles = {
            LEARNING_ARM: {"color": _EES_COLOUR, "marker": "o"},
            "random-skills": {"color": _REFERENCE_COLOUR, "marker": "s"},
        }
        for offset, arm in enumerate(arms):
            values = gaps[arm]
            jitter = (offset - 0.5) * 0.18
            axis.scatter(
                [jitter + i for i in range(len(values))],
                values,
                color=styles[arm]["color"],
                marker=styles[arm]["marker"],
                alpha=0.85,
                zorder=3,
                label=(
                    f"{arm} — median {statistics.median(values):.1f}, "
                    f"{sum(1 for v in values if v > 0)}/{len(values)} seeds ended below peak"
                ),
            )
            axis.axhline(
                statistics.fmean(values),
                color=styles[arm]["color"],
                linestyle=":" if arm != LEARNING_ARM else "-",
                linewidth=1.6,
                alpha=0.6,
                zorder=2,
            )
        axis.set_xticks(range(max(len(v) for v in gaps.values())))
        axis.set_xlabel("seed")
        axis.set_ylabel("best-ever minus final, in tasks")
        axis.set_title(
            "How much of the best-ever/final gap is regression?\n"
            "A non-learning arm shows the maximum's own upward bias"
        )
        axis.grid(alpha=0.2)
        axis.legend(loc="upper left", fontsize=8)
        figure.tight_layout()
        figure.savefig(output_path, dpi=200)
        plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Directory holding <method>/<seed>/stats.json.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Where to write the training-curve figure.",
    )
    parser.add_argument(
        "--selection-bias-path",
        type=Path,
        default=None,
        help="Where to write the best-versus-final calibration figure, if wanted.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    coverage = Tossing3DTwoSkillCurves.skill_coverage(
        results_root=args.results_root, method=LEARNING_ARM
    )
    observed = Tossing3DTwoSkillCurves.observed_skill_names(
        results_root=args.results_root, method=LEARNING_ARM
    )
    Tossing3DTwoSkillCurves.check_skill_names(coverage=coverage, observed=observed)
    curves_by_arm = {
        arm: Tossing3DTwoSkillCurves.load_curves(results_root=args.results_root, method=arm)
        for arm in (LEARNING_ARM, *REFERENCE_ARMS)
    }
    curves_by_arm = {arm: curves for arm, curves in curves_by_arm.items() if curves}
    Tossing3DTwoSkillCurves.print_report(curves_by_arm=curves_by_arm, coverage=coverage)
    Tossing3DTwoSkillCurves.plot(curves_by_arm=curves_by_arm, output_path=args.output_path)
    print(f"\nWrote {args.output_path}")
    if args.selection_bias_path is not None:
        Tossing3DTwoSkillCurves.plot_selection_bias(
            curves_by_arm=curves_by_arm, output_path=args.selection_bias_path
        )
        print(f"Wrote {args.selection_bias_path}")


if __name__ == "__main__":
    main()

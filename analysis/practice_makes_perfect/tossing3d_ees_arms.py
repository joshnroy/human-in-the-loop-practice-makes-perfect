"""Post-run analysis for the three-arm Tossing3D experiment: applies the decision rule
pre-registered in `docs/experiment-logs/2026-08-06-tossing3d-ees-first-real.md` and
renders the figure that log commits.

**What this decides.** Whether EES learns Tossing3D's throw standoff now that
`MoveToThrowPose`'s add effect can fail (#123). Every earlier number on this domain --
`24/90`, `33/90`, `19/100`, `21/100`, and the `543/2700` uniform reference -- was measured
where the standoff sampler was never consulted, so all three arms here are re-measured
rather than quoted: `ees`, a fresh uniform `random-skills`, and `skill-oracle` as the
ceiling.

**Why the primary endpoint is a practice count and not an episode count.** Tossing3D's
task-success axis is provisional twice over: it has a measured same-seed swing of at
least 10 pp, and at 10 seeds x 10 test tasks the MDE on a two-arm comparison is about
17 pp, so nothing smaller is detectable at all. `verdict` therefore reads only counts of
skill executions, with denominators in the hundreds, and never reads `evaluations` --
pinned by `test_the_verdict_does_not_read_task_success`. Task success is plotted and
tabulated for context and is explicitly not an input.

**The MDE is a gate inside the rule, not a footnote under it.** A rate gap is only called
learning when it exceeds `2.801585 * sqrt(pbar (1 - pbar) (1/n1 + 1/n2))` on its own two
denominators *and* clears a two-sided Fisher exact test. `test_a_modest_gap_clears_the_
mde_only_on_the_larger_denominator` feeds the identical gap at two denominators and
requires two different answers.

**There is no positive control skill in this experiment, and the code says so.** `Pick`
was #127's control, but post-#123 it ties at `57/60` and falls back to `0/0` informed --
the same tie-and-fall-back state on a different skill. So "the sampler never made an
informed draw" and "the instrument cannot see informed draws" are separated here by the
pre-flight probe alone, not by a control skill. `verdict` therefore reports `regressed`
and declines to conclude anything when informed draws are absent, rather than reading
zero as a finding.

**The uniform reference is within-run** (Amendment 1 in the log, registered before any
result was read). `RandomSkillsMethod` does not override `practice_outcomes()`, so it
returns the `{}` default and records no `MoveToThrowPose` tally at all -- the originally
registered cross-arm reference does not exist. EES's own non-informed draws serve
instead, and are strictly better: same code, same scene seeds, same cycle structure. The
`random-skills` arm remains the **task-success** baseline, which is what it can support.

**What a positive result would mean.** `bin_init_region` is degenerate -- the bin does not
move -- so the correct standoff is a **constant**. Learning it demonstrates a sampler
finding and memorising a constant, not representation learning, which would need the
target to be a function of observable state. The verdict string says "learns the
constant" for exactly that reason.

Reads only already-produced `--output-dir` output (CLAUDE.md's `analysis/` convention):
`<results-root>/<method>/<seed>/stats.json`, the layout `scripts/run_sweep.py` writes.
Counts, never bare percentages: every printed cell and every axis label is `x/y`.
"""

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering -- no GUI backend needed/available in CI

import matplotlib.pyplot as plt  # noqa: E402

from analysis.practice_makes_perfect.practice_diagnostics import PracticeDiagnostics  # noqa: E402
from analysis.practice_makes_perfect.tossingroom_reset_interval import PairedTests  # noqa: E402
from hitl_pmp.core.method.types import SkillPracticeTally  # noqa: E402
from hitl_pmp.core.metrics.metrics import Metrics  # noqa: E402

# The standoff: Tossing3D's only meaningful learnable parameter, and the skill this
# experiment is about. Post-#123 its add effect is RobotAtSuccessfulThrowPose, which can
# fail -- which is the whole reason there is anything to measure.
THROW_POSE_SKILL = "MoveToThrowPose"
# The skill whose add effect is InGoalRegion -- the domain's actual success criterion --
# and which has param_dim = 0, so no sampler is ever fitted for it.
TOSS_SKILL = "Toss"
# Not a control any more. Post-#123 Pick ties at 57/60 and falls back to 0/0 informed, so
# its counts cannot distinguish "no informed draw" from "instrument blind". Kept because
# it is still worth *showing* -- see the module docstring.
FORMER_CONTROL_SKILL = "Pick"

# z(0.975) + z(0.80): two-sided 5%, 80% power. Pre-registered; do not retune.
MDE_Z_SUM = 2.801585

# The pre-registered thresholds, named here so the code that decides and the log that
# registered the decision cannot drift apart silently.
INFORMED_IN_QUANTITY = 0.20
SIGNIFICANCE = 0.05

_ARM_ORDER = ("ees", "random-skills", "skill-oracle")
_ARM_COLOURS = {"ees": "#1f77b4", "random-skills": "#7f7f7f", "skill-oracle": "#2ca02c"}


class UnpairedTests:
    """Exact tests for the unpaired 2x2 count comparisons.

    Exact by enumeration rather than a chi-square approximation: the counts here are
    small and uneven, and no normal approximation is wanted. Hand-rolled because scipy
    is not a dependency of this project -- the paired half of this already lives in
    `PairedTests`, which this module imports rather than reimplements.
    """

    @staticmethod
    def fisher_exact(*, table: tuple[tuple[int, int], tuple[int, int]]) -> float:
        """Two-sided Fisher exact p for `((a, b), (c, d))`.

        Two-sided by the total-probability convention: sum the probability of every
        table with the same margins whose probability is no greater than the observed
        one. That is what `scipy.stats.fisher_exact` does, and it is the convention the
        pre-registration's `p < 0.05` was written against.
        """
        (a, b), (c, d) = table
        row_1, row_2 = a + b, c + d
        col_1 = a + c
        total = row_1 + row_2
        if total == 0 or row_1 == 0 or row_2 == 0 or col_1 == 0 or col_1 == total:
            return 1.0
        probabilities = {
            k: (math.comb(row_1, k) * math.comb(row_2, col_1 - k) / math.comb(total, col_1))
            for k in range(max(0, col_1 - row_2), min(row_1, col_1) + 1)
        }
        # A relative tolerance, because two genuinely equiprobable tables can differ in
        # the last bits after the binomial division and would otherwise be dropped.
        threshold = probabilities[a] * (1 + 1e-9)
        return min(1.0, sum(p for p in probabilities.values() if p <= threshold))


class Tossing3DEesArms:
    """A static-method container, never instantiated, same as every other business-logic
    class in this project."""

    @staticmethod
    def load(*, results_root: Path) -> dict[str, list[Metrics]]:
        """Every arm's per-seed `Metrics`, keyed by method directory name."""
        return PracticeDiagnostics.summarize(results_root=results_root)

    @staticmethod
    def minimum_detectable_effect(*, p_bar: float, n_1: int, n_2: int) -> float:
        """`2.801585 * sqrt(pbar (1 - pbar) (1/n1 + 1/n2))` -- the pre-registered form.

        Derived from its own two denominators every time it is quoted, rather than from
        a planning value carried over from the pre-registration, so the number reported
        is the one the realized data supports.
        """
        if n_1 <= 0 or n_2 <= 0:
            return math.inf
        return MDE_Z_SUM * math.sqrt(p_bar * (1 - p_bar) * (1 / n_1 + 1 / n_2))

    @staticmethod
    def pooled_tally(*, runs: Sequence[Metrics], skill_name: str) -> SkillPracticeTally:
        """One lifted skill's tally summed over every window of every seed."""
        pooled = SkillPracticeTally()
        for metrics in runs:
            tally = metrics.total_practice_outcomes().get(skill_name)
            if tally is not None:
                pooled = pooled.plus(other=tally)
        return pooled

    @staticmethod
    def per_seed_success(*, runs: Sequence[Metrics], index: int) -> list[tuple[int, int]]:
        """`(solved, total)` at one evaluation checkpoint, one entry per seed.

        Per seed rather than only pooled because a pooled count can describe no seed
        that was actually run: on Tossing Room ten seeds spanned `0/14` to `14/14`.
        """
        per_seed: list[tuple[int, int]] = []
        for metrics in runs:
            if not metrics.evaluations:
                continue
            _, solved, total = metrics.evaluations[index]
            per_seed.append((solved, total))
        return per_seed

    @staticmethod
    def pooled_success(*, runs: Sequence[Metrics], index: int) -> tuple[int, int]:
        per_seed = Tossing3DEesArms.per_seed_success(runs=runs, index=index)
        return (sum(s for s, _ in per_seed), sum(t for _, t in per_seed))

    @staticmethod
    def verdict(*, ees_runs: Sequence[Metrics]) -> tuple[str, str]:
        """The pre-registered decision rule, as amended before any result was read,
        applied to `MoveToThrowPose`.

        **The uniform reference is EES's own non-informed draws, not the `random-skills`
        arm.** `RandomSkillsMethod` does not override `practice_outcomes()`, so it returns
        the `{}` default and records no tally for this skill at all -- the originally
        registered `U` does not exist. Amendment 1 in the log replaces it with the
        within-run reference, which is strictly better anyway: same code, same scene
        seeds, same cycle structure, so the cross-arm confound disappears. A non-informed
        draw is a uniform draw over `THROW_STANDOFF_BOUNDS` by construction, whether it
        came from the epsilon-greedy coin flip or the uninformative fallback.

        Reads practice counts only. `evaluations` is never touched here -- see the module
        docstring for why that is load-bearing on this domain rather than fastidious.
        """
        ees = Tossing3DEesArms.pooled_tally(runs=ees_runs, skill_name=THROW_POSE_SKILL)
        attempts, successes = ees.num_attempts, ees.num_successes
        informed, informed_successes = ees.num_informed_attempts, ees.num_informed_successes
        uniform_attempts = attempts - informed
        uniform_successes = successes - informed_successes
        if attempts == 0:
            return (
                "undecided",
                f"{THROW_POSE_SKILL} was never practiced (0/0 attempts), so there is "
                "nothing to decide on.",
            )
        if informed == 0:
            return (
                "regressed",
                f"{THROW_POSE_SKILL} made 0/{attempts} informed draws. #123's own 3-seed "
                "probe measured 36/56, so this contradicts the merged fix rather than "
                "measuring it. No within-run control skill exists post-#123 to rule out "
                "an instrument fault, so this is a stop-and-report, not a finding.",
            )
        if uniform_attempts == 0:
            return (
                "undecided",
                f"{THROW_POSE_SKILL} made {informed}/{attempts} informed draws and no "
                "non-informed ones, so there is no uniform reference within the run to "
                "compare against -- the comparison would be a rate against itself.",
            )
        informed_rate = informed_successes / informed
        uniform_rate = uniform_successes / uniform_attempts
        gap = informed_rate - uniform_rate
        p_bar = successes / attempts
        mde = Tossing3DEesArms.minimum_detectable_effect(
            p_bar=p_bar, n_1=informed, n_2=uniform_attempts
        )
        p_value = UnpairedTests.fisher_exact(
            table=(
                (informed_successes, informed - informed_successes),
                (uniform_successes, uniform_attempts - uniform_successes),
            )
        )
        evidence = (
            f"{THROW_POSE_SKILL}: informed draws landed {informed_successes}/{informed}, "
            f"the same arm's uniform draws landed {uniform_successes}/{uniform_attempts}, "
            f"a gap of {gap * 100:+.1f} pp against an MDE of {mde * 100:.1f} pp on those "
            f"two denominators; Fisher exact p = {p_value:.4g}. Informed draws were "
            f"{informed}/{attempts} of all attempts; the skill succeeded "
            f"{successes}/{attempts} overall."
        )
        if informed / attempts >= INFORMED_IN_QUANTITY:
            if gap >= mde and p_value < SIGNIFICANCE:
                return ("learns the constant", evidence)
            if gap < mde:
                return ("consulted but no better than uniform", evidence)
            return ("undecided", evidence)
        return ("starved", evidence)

    @staticmethod
    def paired_change(*, runs: Sequence[Metrics]) -> tuple[PairedTests, list[float]]:
        """Exact paired Wilcoxon on each seed's pre-practice -> end-of-training change.

        Paired because both checkpoints come from the same run of the same seed; an
        unpaired test here would discard exactly the structure that makes ten seeds
        worth running.
        """
        before = Tossing3DEesArms.per_seed_success(runs=runs, index=0)
        after = Tossing3DEesArms.per_seed_success(runs=runs, index=-1)
        differences = [
            float(a_solved - b_solved)
            for (b_solved, _), (a_solved, _) in zip(before, after, strict=True)
        ]
        return (PairedTests.wilcoxon_signed_rank(differences=differences), differences)

    @staticmethod
    def print_report(*, arms: dict[str, list[Metrics]]) -> None:
        for arm in _ARM_ORDER:
            runs = arms.get(arm, [])
            if not runs:
                print(f"\n=== {arm}: no runs found ===")
                continue
            before_s, before_t = Tossing3DEesArms.pooled_success(runs=runs, index=0)
            after_s, after_t = Tossing3DEesArms.pooled_success(runs=runs, index=-1)
            print(f"\n=== {arm} ({len(runs)} seeds) ===")
            print(f"  task success, pre-practice     {before_s}/{before_t}")
            print(f"  task success, end of training  {after_s}/{after_t}")
            per_seed = Tossing3DEesArms.per_seed_success(runs=runs, index=-1)
            print("  per seed (end):                " + ", ".join(f"{s}/{t}" for s, t in per_seed))
            for skill in (THROW_POSE_SKILL, TOSS_SKILL, FORMER_CONTROL_SKILL):
                tally = Tossing3DEesArms.pooled_tally(runs=runs, skill_name=skill)
                print(
                    f"  {skill:<18} {tally.num_successes}/{tally.num_attempts} succeeded, "
                    f"{tally.num_informed_attempts}/{tally.num_attempts} informed "
                    f"({tally.num_informed_successes}/{tally.num_informed_attempts} of those "
                    f"succeeded), {tally.num_random_attempts}/{tally.num_attempts} "
                    f"epsilon-random, {tally.num_unparameterized_attempts}/"
                    f"{tally.num_attempts} unparameterized"
                )
        ees = arms.get("ees", [])
        if ees:
            label, evidence = Tossing3DEesArms.verdict(ees_runs=ees)
            print(f"\n=== VERDICT: {label} ===\n  {evidence}")
            wilcoxon, differences = Tossing3DEesArms.paired_change(runs=ees)
            print(
                f"  EES per-seed change: {[int(d) for d in differences]}, exact paired "
                f"Wilcoxon p = {wilcoxon.p_value:.4f} "
                f"({wilcoxon.num_zero_differences} zero differences)"
            )
        uniform = arms.get("random-skills", [])
        if ees and uniform:
            Tossing3DEesArms._print_task_success_comparison(ees=ees, uniform=uniform)

    @staticmethod
    def _print_task_success_comparison(
        *, ees: Sequence[Metrics], uniform: Sequence[Metrics]
    ) -> None:
        """H3, and it is reported with its own power. At 10 seeds x 10 test tasks the MDE
        is about 17 pp, so a non-significant result here is a statement about this
        design's power and not evidence of no effect. Said in the output, not only in the
        log, because the number is what gets quoted."""
        ees_solved, ees_total = Tossing3DEesArms.pooled_success(runs=ees, index=-1)
        uni_solved, uni_total = Tossing3DEesArms.pooled_success(runs=uniform, index=-1)
        p_bar = (ees_solved + uni_solved) / (ees_total + uni_total) if ees_total else 0.0
        mde = Tossing3DEesArms.minimum_detectable_effect(p_bar=p_bar, n_1=ees_total, n_2=uni_total)
        p_value = UnpairedTests.fisher_exact(
            table=(
                (ees_solved, ees_total - ees_solved),
                (uni_solved, uni_total - uni_solved),
            )
        )
        gap = (ees_solved / ees_total - uni_solved / uni_total) if ees_total and uni_total else 0.0
        print(
            f"\n=== H3, task success (context only, provisional) ===\n"
            f"  ees {ees_solved}/{ees_total} against random-skills {uni_solved}/{uni_total}: "
            f"gap {gap * 100:+.1f} pp, MDE {mde * 100:.1f} pp, Fisher exact p = {p_value:.4g}"
        )

    @staticmethod
    def plot(*, arms: dict[str, list[Metrics]], output_path: Path) -> None:
        """Three panels, all per seed: the standoff's label rate, its informed draws, and
        task success. Per-seed points over every bar, because a bar chart of two means
        hides one seed driving the whole effect."""
        figure, axes = plt.subplots(1, 3, figsize=(16, 5.2))
        Tossing3DEesArms._plot_skill_rate(axis=axes[0], arms=arms)
        Tossing3DEesArms._plot_informed(axis=axes[1], arms=arms)
        Tossing3DEesArms._plot_task_success(axis=axes[2], arms=arms)
        figure.suptitle(
            "Tossing3D, first run with the standoff sampler actually consulted "
            "(post-#118/#123/#119) — every label is x/y, dots are individual seeds",
            fontsize=11,
        )
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)

    @staticmethod
    def _present_arms(*, arms: dict[str, list[Metrics]]) -> list[str]:
        return [arm for arm in _ARM_ORDER if arms.get(arm)]

    @staticmethod
    def _plot_skill_rate(*, axis: plt.Axes, arms: dict[str, list[Metrics]]) -> None:
        """The verdict's own quantity: within EES, do informed draws of the standoff beat
        that same arm's uniform draws? One line per seed, so a pooled gap driven by one
        seed would be visible as a single crossing line rather than hidden in a bar."""
        runs = arms.get("ees", [])
        for metrics in runs:
            tally = metrics.total_practice_outcomes().get(THROW_POSE_SKILL)
            if tally is None or tally.num_attempts == 0:
                continue
            uniform_attempts = tally.num_attempts - tally.num_informed_attempts
            uniform_successes = tally.num_successes - tally.num_informed_successes
            if tally.num_informed_attempts == 0 or uniform_attempts == 0:
                continue
            axis.plot(
                [0, 1],
                [
                    uniform_successes / uniform_attempts,
                    tally.num_informed_successes / tally.num_informed_attempts,
                ],
                "-o",
                color="#1f77b4",
                alpha=0.45,
                markersize=4,
                linewidth=1.2,
            )
        pooled = Tossing3DEesArms.pooled_tally(runs=runs, skill_name=THROW_POSE_SKILL)
        uniform_attempts = pooled.num_attempts - pooled.num_informed_attempts
        uniform_successes = pooled.num_successes - pooled.num_informed_successes
        if pooled.num_informed_attempts and uniform_attempts:
            heights = [
                uniform_successes / uniform_attempts,
                pooled.num_informed_successes / pooled.num_informed_attempts,
            ]
            labels = [
                f"{uniform_successes}/{uniform_attempts}",
                f"{pooled.num_informed_successes}/{pooled.num_informed_attempts}",
            ]
            axis.plot([0, 1], heights, "-o", color="black", linewidth=2.6, markersize=8, zorder=5)
            for position, (height, label) in enumerate(zip(heights, labels, strict=True)):
                axis.annotate(
                    label,
                    (position, height),
                    textcoords="offset points",
                    xytext=(0, 12),
                    ha="center",
                    fontsize=10,
                    fontweight="bold",
                )
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["uniform draws\n(same runs)", "informed draws"])
        axis.set_xlim(-0.35, 1.35)
        axis.set_ylabel("MoveToThrowPose successes / attempts")
        axis.set_title(
            "Does the sampler's belief beat its own prior?\n(black = pooled, thin = one seed)"
        )
        axis.set_ylim(0, 1.05)
        axis.grid(axis="y", alpha=0.3)

    @staticmethod
    def _plot_informed(*, axis: plt.Axes, arms: dict[str, list[Metrics]]) -> None:
        """Learning curves, one faint line per seed per arm. The shape is the point: EES
        rises, uniform does not, and the oracle is flat at the ceiling."""
        for arm in Tossing3DEesArms._present_arms(arms=arms):
            for metrics in arms[arm]:
                if len(metrics.evaluations) < 2:
                    continue
                axis.plot(
                    [transitions for transitions, _, _ in metrics.evaluations],
                    [solved / total if total else 0.0 for _, solved, total in metrics.evaluations],
                    color=_ARM_COLOURS[arm],
                    alpha=0.35,
                    linewidth=1.0,
                )
        for arm in Tossing3DEesArms._present_arms(arms=arms):
            runs = [m for m in arms[arm] if len(m.evaluations) >= 2]
            if not runs:
                solved, total = Tossing3DEesArms.pooled_success(runs=arms[arm], index=-1)
                if total:
                    axis.axhline(
                        solved / total,
                        color=_ARM_COLOURS[arm],
                        linestyle="--",
                        linewidth=2.0,
                        label=f"{arm} {solved}/{total}",
                    )
                continue
            length = min(len(m.evaluations) for m in runs)
            axis.plot(
                [sum(m.evaluations[i][0] for m in runs) / len(runs) for i in range(length)],
                [
                    sum(m.evaluations[i][1] for m in runs) / sum(m.evaluations[i][2] for m in runs)
                    for i in range(length)
                ],
                color=_ARM_COLOURS[arm],
                linewidth=2.6,
                label=f"{arm} (n={len(runs)} seeds)",
            )
        axis.set_xlabel("online transitions")
        axis.set_ylabel("test tasks solved / attempted")
        axis.set_title("Learning curves, per seed")
        axis.set_ylim(0, 1.05)
        axis.legend(fontsize=8, loc="lower right")
        axis.grid(alpha=0.3)

    @staticmethod
    def _plot_task_success(*, axis: plt.Axes, arms: dict[str, list[Metrics]]) -> None:
        present = Tossing3DEesArms._present_arms(arms=arms)
        for position, arm in enumerate(present):
            runs = arms[arm]
            solved, total = Tossing3DEesArms.pooled_success(runs=runs, index=-1)
            rate = solved / total if total else 0.0
            axis.bar(position, rate, color=_ARM_COLOURS[arm], alpha=0.45, width=0.6)
            axis.annotate(
                f"{solved}/{total}",
                (position, rate),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=10,
                fontweight="bold",
            )
            per_seed = Tossing3DEesArms.per_seed_success(runs=runs, index=-1)
            for offset, (seed_solved, seed_total) in enumerate(per_seed):
                if seed_total:
                    axis.plot(
                        position + (offset - len(per_seed) / 2) * 0.045,
                        seed_solved / seed_total,
                        "o",
                        color="black",
                        markersize=4,
                        alpha=0.65,
                    )
        axis.set_xticks(range(len(present)))
        axis.set_xticklabels(present)
        axis.set_ylabel("test tasks solved at end of training")
        axis.set_title("Task success (context only — not an input to the verdict)")
        axis.set_ylim(0, 1.08)
        axis.grid(axis="y", alpha=0.3)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="Optional figure PNG.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    arms = Tossing3DEesArms.load(results_root=args.results_root)
    Tossing3DEesArms.print_report(arms=arms)
    if args.output is not None:
        Tossing3DEesArms.plot(arms=arms, output_path=args.output)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()

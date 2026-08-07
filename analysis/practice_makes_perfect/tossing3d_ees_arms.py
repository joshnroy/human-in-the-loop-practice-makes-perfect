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

**The learning curves are drawn against practice cycles, not online transitions.** Both
learning arms ran `--num-cycles 20`, so both have 21 evaluation checkpoints -- but a cycle
ends when the method raises `InteractionComplete`, and on Tossing3D `Toss` deletes
`Reachable`, so nothing is applicable after a throw and a practice period is effectively one
throw. The arms therefore spend very different numbers of transitions reaching the same
cycle: EES finishes at 69..101 transitions (mean 83.8) against `random-skills` at 92..174
(mean 144.0). On a transitions axis EES's curve stopped at about half the panel width and
read as *truncated* when it was in fact more efficient. Cycles are the controlled variable
and transitions are an outcome, so cycles are the axis; mean final transitions survive in
the legend. `cycle_grid` carries the full argument, including that the cycle axis is also
the only one the seeds *within* an arm share -- all ten seeds of each arm sit on ten
distinct transition grids.

**The figure is two panels.** It was three; the dropped one asked "does the sampler's
belief beat its own prior?", plotting EES's `48/275` uniform draws against its `117/206`
informed ones. That is still this experiment's headline result and is still reported --
by `verdict`, by `print_report`, and in the log -- but a two-point comparison reads better
as a sentence than as a chart.

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
from matplotlib.figure import Figure  # noqa: E402

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

# Okabe-Ito, the same palette every sibling report in this folder declares for itself
# (`reset_free_training_curves`, `tossingroom_reset_interval`, `tossingroom_reset_
# frequency`, `tossingroomsplit_reset_policy`, `tossingroomsplit_two_way_ledge`). It
# replaces matplotlib's default blue/grey/green, which is not colourblind-safe and did
# not match the sibling figures. Blue is the arm under test, vermillion the uniform
# baseline, bluish-green the ceiling -- and the three are also distinguished by role in
# the drawing (curve, curve, dashed horizontal line), so colour is never doing the work
# alone.
_ARM_COLOURS = {"ees": "#0072B2", "random-skills": "#D55E00", "skill-oracle": "#009E73"}

# Per-seed lines are what the figure exists for, but ten of them at full strength hide
# the mean. Matched to `reset_free_training_curves` so the two figures read as one system.
_SEED_ALPHA = 0.18
_SEED_WIDTH = 1.0
_MEAN_WIDTH = 2.6

# An explicit white canvas rather than matplotlib's transparent default, so a PNG dropped
# into a dark-themed PR or Notion page keeps readable axes instead of black text on black.
_CANVAS = "white"


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
    def cycle_grid(*, runs: Sequence[Metrics]) -> list[int]:
        """The shared x grid for a learning arm: the evaluation checkpoint index.

        **Checkpoint index is the controlled variable; online transitions are an outcome.**
        Every learning arm here ran `--num-cycles 20` and so has 21 evaluations -- index 0
        is the pre-practice sweep and index `i` is the state after `i` practice cycles.

        What differs between arms is how many transitions those cycles cost. A cycle ends
        when the method raises `InteractionComplete` -- nothing further worth practising --
        or when it exhausts `--max-steps-per-interaction`, whichever comes first, and
        untaken steps are not charged (`practice_loop.py` breaks out of the step loop; the
        count is data-driven rather than budget-driven). On Tossing3D the first case
        dominates: `Toss` deletes `Reachable`, so no skill is applicable after a throw and a
        practice period is effectively one throw. Measured over these runs, a period costs
        4.19 transitions for `ees` against 7.20 for `random-skills`. Plotting against
        transitions therefore stops EES's curve at roughly half the panel width and reads
        as truncation when it is really efficiency -- the axis penalises the arm that
        wastes fewer transitions.

        Not to be confused with the opposite mechanism: when the planner finds no plan EES
        emits a no-op, which *does* consume a step and *does* count as a transition (the
        path #102 corrected). The early exit is the method declaring nothing is worth
        practising, not the agent getting stuck.

        It is also the only axis the seeds *within* one arm share. All 10 EES seeds and all
        10 `random-skills` seeds were measured on 10 distinct transition grids each (they
        finish at 69..101 and 92..174 respectively), so a per-checkpoint mean on a
        transitions axis has to average the x positions too. On the cycle axis the grid is
        identical by construction.

        Raises when the seeds disagree on how many checkpoints they have. The replaced code
        took `min(len(m.evaluations) for m in runs)` and silently shortened the pooled
        curve to the shortest seed; every seed has 21 here so it never bit, but a mean that
        quietly drops a seed's tail is invisible in the drawn line. This is the same
        discipline `reset_free_training_curves.checkpoints` applies to its transition grid.
        """
        lengths = {len(metrics.evaluations) for metrics in runs}
        if len(lengths) != 1:
            raise ValueError(
                f"seeds in this arm have different numbers of evaluation checkpoints "
                f"({sorted(lengths)}), so a per-checkpoint mean would silently truncate to "
                f"the shortest seed."
            )
        return list(range(next(iter(lengths))))

    @staticmethod
    def per_seed_solved_curves(*, runs: Sequence[Metrics]) -> list[list[int]]:
        """Each seed's tasks-solved curve, one row per seed, on the cycle grid.

        Counts rather than rates, matching the y-axis: the denominator is 10 test tasks per
        evaluation and is small enough that a rate on an axis hides it.
        """
        return [[solved for _, solved, _ in metrics.evaluations] for metrics in runs]

    @staticmethod
    def mean_final_transitions(*, runs: Sequence[Metrics]) -> float:
        """Mean online transitions at the last checkpoint, over the arm's seeds.

        Transitions stopped being the x-axis (see `cycle_grid`) but they are still the
        efficiency story, so they are kept as a per-arm legend annotation rather than
        dropped: the same 20 cycles cost EES about 84 transitions and `random-skills`
        about 144, which is the point the old axis was accidentally making backwards.
        """
        finals = [metrics.evaluations[-1][0] for metrics in runs if metrics.evaluations]
        return sum(finals) / len(finals) if finals else 0.0

    @staticmethod
    def evaluation_size(*, runs: Sequence[Metrics]) -> int:
        """Test tasks per evaluation sweep -- the y-axis denominator.

        Read from the data rather than hardcoded, so an arm evaluated on a different number
        of tasks changes the printed denominator instead of being silently divided by the
        expected one.
        """
        sizes = {total for metrics in runs for _, _, total in metrics.evaluations}
        return next(iter(sizes)) if len(sizes) == 1 else 0

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
    def figure(*, arms: dict[str, list[Metrics]]) -> Figure:
        """Two panels: the learning curves and the task-success bars. Returns the figure
        so a test can assert on its structure without reopening a PNG.

        **Two panels, not three.** The third used to be "does the sampler's belief beat its
        own prior?", drawing EES's `48/275` uniform draws against its `117/206` informed
        ones. That comparison is still the headline result of this experiment -- it is
        reported by `verdict`, printed by `print_report`, and stated in the log and the PR
        body -- but it is a single two-point comparison, which a sentence carries better
        than a chart does.

        Per-seed detail under every mark, because a chart of two means hides one seed
        driving the whole effect: faint lines under the bold means on the left, individual
        dots over the bars on the right.
        """
        figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), dpi=150, facecolor=_CANVAS)
        Tossing3DEesArms._plot_learning_curves(axis=axes[0], arms=arms)
        Tossing3DEesArms._plot_task_success(axis=axes[1], arms=arms)
        figure.suptitle(
            "Tossing3D, first run with the standoff sampler actually consulted "
            "(post-#118/#123/#119) — every label is x/y, bold = mean over faint seeds",
            fontsize=12,
        )
        figure.tight_layout()
        return figure

    @staticmethod
    def plot(*, arms: dict[str, list[Metrics]], output_path: Path) -> None:
        """Render the figure and write the PNG."""
        figure = Tossing3DEesArms.figure(arms=arms)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, bbox_inches="tight", facecolor=_CANVAS)
        plt.close(figure)

    @staticmethod
    def _present_arms(*, arms: dict[str, list[Metrics]]) -> list[str]:
        return [arm for arm in _ARM_ORDER if arms.get(arm)]

    @staticmethod
    def _plot_learning_curves(*, axis: plt.Axes, arms: dict[str, list[Metrics]]) -> None:
        """Learning curves for every arm on one axes, against practice cycles.

        **The x-axis is cycles, not online transitions.** See `cycle_grid` for the full
        reason; the short form is that both learning arms ran the same 21 checkpoints but
        spent very different numbers of transitions reaching them, so a transitions axis
        made the *more* efficient arm look truncated. Cycles are what the experiment
        controlled. Mean final transitions survive in the legend, which is where the
        efficiency difference belongs -- as a number, not as a distortion of the axis.

        Bold mean over faint per-seed lines, matching `reset_free_training_curves`, because
        a mean can describe no seed that was actually run.
        """
        denominator = 0
        for arm in Tossing3DEesArms._present_arms(arms=arms):
            runs = [m for m in arms[arm] if len(m.evaluations) >= 2]
            colour = _ARM_COLOURS[arm]
            if not runs:
                # An arm with a single evaluation never practices -- `skill-oracle` is the
                # privileged ceiling, not a learner -- so it is a horizontal reference
                # line rather than a curve. Drawn dashed as well as coloured, so the
                # ceiling is distinguishable from a curve without relying on colour.
                solved, total = Tossing3DEesArms.pooled_success(runs=arms[arm], index=-1)
                size = Tossing3DEesArms.evaluation_size(runs=arms[arm])
                if total:
                    axis.axhline(
                        solved / total * size if size else 0.0,
                        color=colour,
                        linestyle="--",
                        linewidth=2.0,
                        label=f"{arm} (ceiling) — {solved}/{total}",
                    )
                continue
            grid = Tossing3DEesArms.cycle_grid(runs=runs)
            curves = Tossing3DEesArms.per_seed_solved_curves(runs=runs)
            denominator = Tossing3DEesArms.evaluation_size(runs=runs) or denominator
            for row in curves:
                axis.plot(grid, row, color=colour, alpha=_SEED_ALPHA, linewidth=_SEED_WIDTH)
            # `strict` because a ragged transpose would silently truncate the mean to the
            # shortest seed -- exactly what the old `min(len(...))` did. `cycle_grid` has
            # already proved the rows align, so this can only fire on a coding error.
            mean = [sum(column) / len(column) for column in zip(*curves, strict=True)]
            solved, total = Tossing3DEesArms.pooled_success(runs=runs, index=-1)
            transitions = Tossing3DEesArms.mean_final_transitions(runs=runs)
            axis.plot(
                grid,
                mean,
                color=colour,
                linewidth=_MEAN_WIDTH,
                label=(
                    f"{arm} (n={len(runs)} seeds) — final {solved}/{total}, "
                    f"{transitions:.0f} transitions mean"
                ),
            )
        axis.set_xlabel("practice cycles completed (checkpoint index; 0 = before any practice)")
        axis.set_ylabel(
            f"test tasks solved per evaluation (x/{denominator})"
            if denominator
            else "test tasks solved per evaluation"
        )
        axis.set_title("Learning curves, per seed", loc="left", fontsize=11)
        # Headroom above the oracle ceiling so the legend has somewhere to sit, and a
        # near-opaque white box behind it: the ceiling line runs the full width at the top
        # of the data, so a frameless legend in the usual upper-left corner would be read
        # through it. The legend has to be legible on its own.
        axis.set_ylim(0, (denominator or 10) * 1.3)
        # Integer ticks: a cycle count of 2.5 does not exist. Left to matplotlib the axis
        # labels every 2.5 checkpoints, which invites reading a fractional cycle.
        axis.set_xticks([cycle for cycle in Tossing3DEesArms._cycle_ticks(axis=axis)])
        axis.legend(frameon=True, framealpha=0.92, edgecolor="none", fontsize=8, loc="upper left")
        axis.grid(alpha=0.25, linewidth=0.6)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)

    @staticmethod
    def _cycle_ticks(*, axis: plt.Axes) -> list[int]:
        """Integer cycle ticks spanning whatever was plotted, at most about six of them."""
        left, right = axis.get_xlim()
        last = max(0, int(right))
        step = max(1, round(last / 5)) if last else 1
        return list(range(0, last + 1, step))

    @staticmethod
    def _plot_task_success(*, axis: plt.Axes, arms: dict[str, list[Metrics]]) -> None:
        present = Tossing3DEesArms._present_arms(arms=arms)
        for position, arm in enumerate(present):
            runs = arms[arm]
            solved, total = Tossing3DEesArms.pooled_success(runs=runs, index=-1)
            size = Tossing3DEesArms.evaluation_size(runs=runs) or 1
            # Bar height is the mean tasks-solved per seed, on the same x/10 count scale
            # as the curve panel beside it, rather than a 0..1 rate. A rate on an axis is
            # a bare percentage in everything but name; the count carries its denominator.
            height = solved / total * size if total else 0.0
            axis.bar(position, height, color=_ARM_COLOURS[arm], alpha=0.45, width=0.6)
            axis.annotate(
                f"{solved}/{total}",
                (position, height),
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
                        seed_solved,
                        "o",
                        color="black",
                        markersize=4,
                        alpha=0.65,
                    )
        size = Tossing3DEesArms.evaluation_size(runs=arms.get(present[0], [])) if present else 0
        axis.set_xticks(range(len(present)))
        axis.set_xticklabels(present)
        axis.set_ylabel(
            f"test tasks solved at end of training (x/{size})"
            if size
            else "test tasks solved at end of training"
        )
        axis.set_title(
            "Task success (context only — not an input to the verdict)", loc="left", fontsize=11
        )
        axis.set_ylim(0, (size or 10) * 1.12)
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)


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

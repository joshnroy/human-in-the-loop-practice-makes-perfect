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

**The learning curves are drawn twice, against both axes, as two separate graphs.** Both
learning arms ran `--num-cycles 20`, so both have 21 evaluation checkpoints -- but a cycle
ends when the method raises `InteractionComplete`, and on Tossing3D `Toss` deletes
`Reachable`, so nothing is applicable after a throw and a practice period is effectively one
throw. The arms therefore spend very different numbers of transitions reaching the same
cycle: EES finishes at 69..101 transitions (mean 83.8) against `random-skills` at 92..174
(mean 144.0), which is 4.19 transitions per practice period against 7.20 averaged over the
ten seeds.

That non-proportionality is what earns two graphs rather than one. **Against cycles the arms
align** -- the controlled variable, so like is compared with like. **Against transitions EES's
line ends earlier**, because it reached the same 21/21 checkpoints for fewer steps; that is
efficiency, and the reader should see it rather than have it flattened into a legend
annotation. Neither axis is a correction of the other. Contrast Tossing Room, where every run
charged exactly 150.0 transitions per cycle: there a cycles graph *is* the transitions graph
with relabelled ticks, and drawing both would be padding. Here every seed sits on its own
irregular grid -- per-cycle steps run 1..20 within a single seed -- so the two views genuinely
differ. `cycle_grid` and `mean_transition_grid` carry the details, including that the cycle
axis is the only one the seeds *within* an arm share.

**The prior-versus-belief panel stays gone.** It asked "does the sampler's belief beat its own
prior?", plotting EES's `48/275` uniform draws against its `117/206` informed ones. That is
still this experiment's headline result and is still reported -- by `verdict`, by
`print_report`, and in the log -- but a two-point comparison reads better as a sentence than
as a chart.

Reads only already-produced `--output-dir` output (CLAUDE.md's `analysis/` convention):
`<results-root>/<method>/<seed>/stats.json`, the layout `scripts/run_sweep.py` writes.
Counts, never bare percentages: every printed cell and every axis label is `x/y`.
"""

import argparse
import math
import textwrap
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering -- no GUI backend needed/available in CI

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from analysis.practice_makes_perfect.paired_tests import PairedTests  # noqa: E402
from analysis.practice_makes_perfect.practice_diagnostics import PracticeDiagnostics  # noqa: E402
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
# (`reset_free_training_curves`; the Tossing Room reports that also declared it were
# retired with their domains in #141). It
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

# Caption wrap width in characters, and the fraction of figure height reserved for it. Both
# are tuned to the 9.0in-wide curve figures: the caption must wrap inside the axes width, or
# `bbox_inches="tight"` widens the whole PNG to fit it.
_CAPTION_WRAP = 132
_CAPTION_HEIGHT = 0.23

# The captions. They state the cycle-ending mechanism on the figures themselves, because a
# PNG pinned into a PR body travels without the prose around it -- and because "why does the
# EES line stop earlier?" is the first question the transitions graph provokes.
#
# Both captions describe what the code does, which is *not* what an earlier revision of this
# log claimed. During practice a failure to plan raises `InteractionComplete` and ends the
# period (`EesPolicy.step`, ees_method.py:897-903). The no-op branch (`ees_method.py:908`) is
# reached only when `self._practicing` is false -- that is, inside an evaluation episode --
# and evaluation steps are deliberately never charged as online transitions
# (`PracticeLoop.run`, practice_loop.py:150-152). So no-ops enter neither axis, and nothing
# lengthens a practice cycle beyond `--max-steps-per-interaction`.
_MECHANISM = (
    "A practice cycle ends when the method raises InteractionComplete — nothing further worth "
    "practising — or when it exhausts --max-steps-per-interaction, whichever comes first; "
    "untaken steps are not charged (PracticeLoop.run, practice_loop.py:400-411). On Tossing3D "
    "the first case dominates: Toss deletes Reachable, so no skill is applicable after a throw "
    "and a practice period is effectively one throw."
)
_NO_OP_NOTE = (
    "EES's no-op-on-no-plan (ees_method.py:908) is an evaluation-only branch — during practice "
    "the same condition raises InteractionComplete instead — and evaluation steps are never "
    "charged as online transitions, so no-ops appear on neither axis."
)
_CYCLES_CAPTION = (
    f"{_MECHANISM} Cycles are therefore equal in count across arms but not in transitions, "
    f"which is why this graph has a companion drawn against transitions.\n{_NO_OP_NOTE}"
)
_TRANSITIONS_CAPTION = (
    "The EES line ends earlier because it reached the same 21/21 checkpoints for fewer steps — "
    "that is efficiency, not truncation. Seeds do not share a transition grid, so each bold "
    "mean averages the x positions as well as the y.\n"
    f"{_MECHANISM} {_NO_OP_NOTE}"
)
_TASK_SUCCESS_CAPTION = (
    "Context only — never an input to the verdict. At 10 seeds x 10 test tasks the MDE on a "
    "two-arm comparison is about 17 pp, and this domain has a measured same-seed swing of at "
    "least 10 pp, so small differences here are not detectable."
)


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
    def per_seed_transition_curves(*, runs: Sequence[Metrics]) -> list[list[int]]:
        """Each seed's online-transition count at each checkpoint, one row per seed.

        These rows are the x positions for the transitions graph, and they are the reason
        that graph needs its own treatment: unlike the cycle grid they are *not* shared, so
        every seed's line sits on a different x grid.
        """
        return [[online for online, _, _ in metrics.evaluations] for metrics in runs]

    @staticmethod
    def mean_transition_grid(*, runs: Sequence[Metrics]) -> list[float]:
        """The x grid for an arm's bold mean line on the transitions axis.

        **The x positions are averaged as well as the y.** The seeds do not share a
        transition grid, so there is no common x to read a per-checkpoint mean off. Taking
        one seed's grid, or the pooled maximum, would draw the mean at transition counts no
        seed actually reached. Averaging both coordinates keeps the mean a genuine centroid
        of the ten seed curves at each checkpoint -- which is exactly the extra step the
        cycle axis does not need, and is stated on the figure rather than left implicit.

        `strict` because a ragged transpose would silently truncate to the shortest seed;
        `cycle_grid` has already proved the rows align, so this can only fire on a coding
        error.
        """
        rows = Tossing3DEesArms.per_seed_transition_curves(runs=runs)
        return [sum(column) / len(column) for column in zip(*rows, strict=True)]

    @staticmethod
    def mean_transitions_per_cycle(*, runs: Sequence[Metrics]) -> float:
        """Mean transitions spent per practice cycle, over the arm's seeds.

        The mechanism number behind the two axes differing: 4.19 for `ees` against 7.20 for
        `random-skills` on this sweep. A cycle ends when the method raises
        `InteractionComplete` -- nothing further worth practising -- or when it exhausts
        `--max-steps-per-interaction`, whichever comes first, and untaken steps are not
        charged (`PracticeLoop.run`, `practice_loop.py:400-411`). On Tossing3D the first
        case dominates, so this sits far below the step budget.
        """
        rates = [
            metrics.evaluations[-1][0] / (len(metrics.evaluations) - 1)
            for metrics in runs
            if len(metrics.evaluations) > 1
        ]
        return sum(rates) / len(rates) if rates else 0.0

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
    def figure_cycles(*, arms: dict[str, list[Metrics]]) -> Figure:
        """Learning curves against practice cycles, as their own single-panel graph.

        Cycles are the controlled variable: both learning arms ran `--num-cycles 20`, so
        both span 0..20 and the arms are compared like with like.
        """
        figure, axis = plt.subplots(1, 1, figsize=(9.0, 5.6), dpi=150, facecolor=_CANVAS)
        Tossing3DEesArms._plot_learning_curves(axis=axis, arms=arms, against="cycles")
        figure.suptitle(
            "Tossing3D learning curves against practice cycles — the controlled variable\n"
            "every label is x/y, bold = mean over faint per-seed lines",
            fontsize=12,
        )
        Tossing3DEesArms._caption(figure=figure, text=_CYCLES_CAPTION)
        return figure

    @staticmethod
    def figure_transitions(*, arms: dict[str, list[Metrics]]) -> Figure:
        """Learning curves against online transitions, as their own single-panel graph.

        The companion view to `figure_cycles`, and a genuinely different one on this domain:
        the arms reach the same 21/21 checkpoints at very different transition costs, so EES's
        line ends earlier here. That is efficiency, not truncation.
        """
        figure, axis = plt.subplots(1, 1, figsize=(9.0, 5.6), dpi=150, facecolor=_CANVAS)
        Tossing3DEesArms._plot_learning_curves(axis=axis, arms=arms, against="transitions")
        figure.suptitle(
            "Tossing3D learning curves against online transitions — an outcome, not a control\n"
            "every label is x/y, bold = mean over faint per-seed lines",
            fontsize=12,
        )
        Tossing3DEesArms._caption(figure=figure, text=_TRANSITIONS_CAPTION)
        return figure

    @staticmethod
    def figure_task_success(*, arms: dict[str, list[Metrics]]) -> Figure:
        """End-of-training task success, unchanged: pooled bars with every seed drawn over
        them, because a chart of three means hides one seed driving the whole effect."""
        figure, axis = plt.subplots(1, 1, figsize=(7.0, 5.6), dpi=150, facecolor=_CANVAS)
        Tossing3DEesArms._plot_task_success(axis=axis, arms=arms)
        figure.suptitle(
            "Tossing3D end-of-training task success — every label is x/y, dots = seeds",
            fontsize=12,
        )
        Tossing3DEesArms._caption(figure=figure, text=_TASK_SUCCESS_CAPTION)
        return figure

    @staticmethod
    def _caption(*, figure: Figure, text: str) -> None:
        """A hard-wrapped caption under the axes.

        The mechanism belongs on the figure rather than only in the log: a PNG pinned into a
        PR body travels without its prose, and the cycle-ending mechanism is precisely what
        makes these two axes different rather than redundant.

        **Wrapped explicitly rather than left to matplotlib.** `savefig(bbox_inches="tight")`
        grows the canvas to contain every artist, so a single long unwrapped line stretches
        the PNG to several times the axes width and shrinks the plot to a sliver -- measured
        at 5041x848 before this wrapped. Matplotlib's own `wrap=True` does not help, because
        it wraps to the *figure* width that `bbox_inches` is itself deriving.
        """
        wrapped = "\n".join(
            line
            for paragraph in text.split("\n")
            for line in textwrap.wrap(paragraph, width=_CAPTION_WRAP)
        )
        figure.tight_layout(rect=(0.0, _CAPTION_HEIGHT, 1.0, 1.0))
        figure.text(0.01, 0.005, wrapped, ha="left", va="bottom", fontsize=7.4, color="#333333")

    @staticmethod
    def plot(
        *,
        arms: dict[str, list[Metrics]],
        cycles_path: Path,
        transitions_path: Path,
        task_success_path: Path,
    ) -> None:
        """Render all three figures and write their PNGs.

        Three separate files rather than three panels of one: the two learning-curve axes
        answer different questions and each earns a full canvas, and task success is context
        only -- explicitly not an input to the verdict -- so it should not sit beside a curve
        as though it were the same claim.
        """
        for figure, output_path in (
            (Tossing3DEesArms.figure_cycles(arms=arms), cycles_path),
            (Tossing3DEesArms.figure_transitions(arms=arms), transitions_path),
            (Tossing3DEesArms.figure_task_success(arms=arms), task_success_path),
        ):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output_path, bbox_inches="tight", facecolor=_CANVAS)
            plt.close(figure)

    @staticmethod
    def _present_arms(*, arms: dict[str, list[Metrics]]) -> list[str]:
        return [arm for arm in _ARM_ORDER if arms.get(arm)]

    @staticmethod
    def _plot_learning_curves(
        *, axis: plt.Axes, arms: dict[str, list[Metrics]], against: str
    ) -> None:
        """Learning curves for every arm on one axes, against `cycles` or `transitions`.

        One implementation for both graphs, because only the x coordinates and the labelling
        differ -- the arms, the palette, the per-seed lines, the bold mean and the oracle
        ceiling are identical, and drawing them twice would let the two views drift apart.

        **The two axes are not interchangeable on this domain.** Against `cycles` every seed
        of every arm shares one grid by construction, so a per-checkpoint mean is a mean in y
        alone. Against `transitions` no two seeds share a grid, so the mean averages the x
        positions too (`mean_transition_grid`) and the arms end at different x -- EES earlier,
        having reached the same checkpoints for fewer steps.

        Bold mean over faint per-seed lines, matching `reset_free_ledge_curves`, because a
        mean can describe no seed that was actually run.
        """
        by_cycles = against == "cycles"
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
            cycles = Tossing3DEesArms.cycle_grid(runs=runs)
            curves = Tossing3DEesArms.per_seed_solved_curves(runs=runs)
            seed_grids = Tossing3DEesArms.per_seed_transition_curves(runs=runs)
            denominator = Tossing3DEesArms.evaluation_size(runs=runs) or denominator
            for row, seed_grid in zip(curves, seed_grids, strict=True):
                axis.plot(
                    cycles if by_cycles else seed_grid,
                    row,
                    color=colour,
                    alpha=_SEED_ALPHA,
                    linewidth=_SEED_WIDTH,
                )
            # `strict` because a ragged transpose would silently truncate the mean to the
            # shortest seed -- exactly what the old `min(len(...))` did. `cycle_grid` has
            # already proved the rows align, so this can only fire on a coding error.
            mean = [sum(column) / len(column) for column in zip(*curves, strict=True)]
            solved, total = Tossing3DEesArms.pooled_success(runs=runs, index=-1)
            axis.plot(
                cycles if by_cycles else Tossing3DEesArms.mean_transition_grid(runs=runs),
                mean,
                color=colour,
                linewidth=_MEAN_WIDTH,
                label=Tossing3DEesArms._curve_label(
                    arm=arm, runs=runs, solved=solved, total=total, by_cycles=by_cycles
                ),
            )
        if by_cycles:
            axis.set_xlabel("practice cycles completed (checkpoint index; 0 = before any practice)")
            # Integer ticks: a cycle count of 2.5 does not exist. Left to matplotlib the axis
            # labels every 2.5 checkpoints, which invites reading a fractional cycle.
            axis.set_xticks(list(Tossing3DEesArms._cycle_ticks(axis=axis)))
        else:
            axis.set_xlabel("online transitions during practice (evaluation sweeps not charged)")
        axis.set_ylabel(
            f"test tasks solved per evaluation (x/{denominator})"
            if denominator
            else "test tasks solved per evaluation"
        )
        axis.set_title(
            "Learning curves, per seed — "
            + ("cycles are the controlled variable" if by_cycles else "transitions are an outcome"),
            loc="left",
            fontsize=11,
        )
        # Headroom above the oracle ceiling so the legend has somewhere to sit, and a
        # near-opaque white box behind it: the ceiling line runs the full width at the top
        # of the data, so a frameless legend in the usual upper-left corner would be read
        # through it. The legend has to be legible on its own.
        axis.set_ylim(0, (denominator or 10) * 1.3)
        axis.legend(frameon=True, framealpha=0.92, edgecolor="none", fontsize=8, loc="upper left")
        axis.grid(alpha=0.25, linewidth=0.6)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)

    @staticmethod
    def _curve_label(
        *, arm: str, runs: Sequence[Metrics], solved: int, total: int, by_cycles: bool
    ) -> str:
        """One arm's legend entry, carrying `x/y` counts on both graphs.

        Each graph annotates with the quantity the *other* axis would have shown, so neither
        view loses the efficiency story: the cycles graph reports mean final transitions, and
        the transitions graph reports the per-cycle cost that made the arms diverge.
        """
        if by_cycles:
            transitions = Tossing3DEesArms.mean_final_transitions(runs=runs)
            tail = f"{transitions:.0f} transitions mean"
        else:
            tail = f"{Tossing3DEesArms.mean_transitions_per_cycle(runs=runs):.2f} per cycle mean"
        return f"{arm} (n={len(runs)} seeds) — final {solved}/{total}, {tail}"

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
    """One explicit flag per figure, matching `reset_free_ledge_curves`.

    The single `--output` this replaced could only ever name one file, and there are now
    three. Naming each one rather than deriving siblings from a stem keeps the written paths
    greppable from the log that commits them.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--cycles-output", type=Path, default=None)
    parser.add_argument("--transitions-output", type=Path, default=None)
    parser.add_argument("--task-success-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    arms = Tossing3DEesArms.load(results_root=args.results_root)
    Tossing3DEesArms.print_report(arms=arms)
    outputs = (args.cycles_output, args.transitions_output, args.task_success_output)
    if all(output is not None for output in outputs):
        Tossing3DEesArms.plot(
            arms=arms,
            cycles_path=args.cycles_output,
            transitions_path=args.transitions_output,
            task_success_path=args.task_success_output,
        )
        for output in outputs:
            print(f"wrote {output}")
    elif any(output is not None for output in outputs):
        parser_error = (
            "pass all three of --cycles-output, --transitions-output and "
            "--task-success-output, or none of them"
        )
        raise SystemExit(parser_error)


if __name__ == "__main__":
    main()

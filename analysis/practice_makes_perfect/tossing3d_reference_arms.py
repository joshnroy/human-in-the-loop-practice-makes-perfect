"""What do the non-learning reference arms actually score on Tossing3D, at these pins?

Post-run analysis only; it reads `--results-root` back in and never drives a `Method`.

## Why this measurement exists

`docs/experiment-logs/2026-08-13-tossing3d-reset-policy-new-pin.md` measured reset-free
EES at `12.2/100` against `scheduled`'s `73.4/100`, and noted in passing that `12.2/100`
sits **below** the `24/100` `random-skills` line carried in
`tossing3d_reset_free_arms.py` as `RANDOM_SKILLS_PER_SEED`. That page was careful to call
the observation arithmetic rather than a test, and it was right to: the `24/100` comes
from #133, which ran **20 cycles** and predates #160's separate evaluation `Problem`. So
the most interesting comparison anyone would want to draw off that page -- "the learner is
worse than picking skills at random" -- rested on two numbers produced under different
conditions.

This module reads the arms that fix that: `random-skills` and `skill-oracle` re-run at the
*same* 100 cycles, the same 10 fixed seeds, the same separate evaluation `Problem` and the
same two KINDER pins as the EES sweep.

## What the reset-free `random-skills` arm can and cannot show

It is the matched comparator for reset-free EES, and it is **not** a test of whether the
evaluation score collapses without resets for this method -- that outcome is not available
to `RandomSkillsMethod` by construction, and reading a null there as "the domain is fine"
would be wrong. The method carries no learned state across cycles, and evaluation runs on
a separate `evaluation_problem` reset per episode, so the only channel from practice to
evaluation is the method's own `_rng` stream (`choose_ground_skill` serves both phases).

Where the domain effect *is* visible for this arm is the **practice** side: transitions
collected and idle cycles, which is why `report` prints those for every arm rather than
only for the ones whose score moved.

## The rule, taken from #178/#179 rather than invented here

- A seed's score is its **mean solved count over the last `WINDOW` sweeps** (`LATE`).
- **Paired across seeds**, because every arm ran the same fixed seed set.
- Exact paired sign-flip on per-seed differences, and **the MDE beside it, always**.

Counts are reported `x/y`, never as a bare percentage.
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib
from pydantic import BaseModel, ConfigDict

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.practice_makes_perfect.paired_tests import PairedTests  # noqa: E402
from analysis.practice_makes_perfect.tossing3d_reset_free_arms import (  # noqa: E402
    WINDOW,
    Tossing3DResetFree,
)

# Alpha for calling a paired comparison null. Fixed here rather than at each call site so
# the report and the figures cannot disagree about which results are null.
ALPHA = 0.05

# CLAUDE.md's training-curve style. Blue is the arm that HAS an intervention mechanism
# available (`scheduled`), orange the one that has none (`never`) -- whether or not it
# fires. Grey is reserved for reference arms that are not themselves being manipulated.
_SCHEDULED_COLOUR = "#0072B2"
_NEVER_COLOUR = "#D55E00"
_REFERENCE_COLOUR = "#666666"

# Faint per-seed traces under bold subgroup means, on every training curve.
_SEED_ALPHA = 0.16
_SEED_WIDTH = 0.8
_MEAN_WIDTH = 2.3


class PairedComparison(BaseModel):
    """One arm against another over the seeds they share.

    Frozen, because it is a finished reading of a fixed set of runs -- nothing downstream
    has any business adjusting a p-value in place.

    `num_right_lower` / `num_right_higher` are deliberately both carried rather than one
    plus a total: with ties possible they do not determine each other, and a reader
    checking `x/y` against the seed count should not have to infer the third number.
    """

    model_config = ConfigDict(frozen=True)

    seeds: list[int]
    mean_difference: float
    p_value: float
    minimum_detectable_effect: float
    seeds_for_80_percent_power: float
    num_right_lower: int
    num_right_higher: int
    num_tied: int

    @property
    def num_seeds(self) -> int:
        return len(self.seeds)

    @property
    def is_null(self) -> bool:
        """Whether the difference failed to reach significance. Named for what it is:
        this project writes "null result" in full and requires an MDE beside it, which is
        why that field is not optional on this model."""
        return self.p_value > ALPHA


class Tossing3DReferenceArms:
    """A static-method container, never instantiated, same as every other business-logic
    class in this project."""

    @staticmethod
    def load_runs(
        *, results_root: Path, policy: str, method: str
    ) -> dict[int, list[tuple[int, int, int]]]:
        """`{seed: evaluations}` for one arm, each entry
        `(num_online_transitions, num_solved, num_total)` for one sweep.

        **Keyed by `(policy, method)`, not by seed alone.** `scripts/run_sweep.py` writes
        `<results-root>/<policy>/<method>/<seed>/`, and this experiment is the first on
        this domain to run two methods under one policy. A loader that recursed from the
        policy directory and keyed on the containing directory's name -- which is what the
        sibling `Tossing3DResetFree.load_arms` does, correctly, for a tree that has no
        method level -- would read `random-skills/3` and `skill-oracle/3` into the same
        slot. Rooting the search at `policy/method` makes that unrepresentable.

        Read from `stats.json`, so only completed runs appear: a partially-finished run
        has none and is silently absent rather than contributing a truncated curve that
        would shorten every window computed from it.
        """
        runs: dict[int, list[tuple[int, int, int]]] = {}
        arm_root = results_root / policy / method
        if not arm_root.is_dir():
            return runs
        for path in sorted(arm_root.rglob("stats.json")):
            if not path.parent.name.isdigit():
                continue
            stats = json.loads(path.read_text())
            runs[int(path.parent.name)] = [
                (int(t), int(s), int(n)) for t, s, n in stats["evaluations"]
            ]
        return runs

    @staticmethod
    def late_scores(*, curves: dict[int, list[tuple[int, int, int]]]) -> dict[int, float]:
        """Each seed's mean solved count over the last `WINDOW` sweeps.

        Delegated to the sibling module rather than re-derived, so the two experiments'
        `LATE` cannot drift apart -- comparability with #247's `73.4/100` and `12.2/100`
        is the entire purpose of this analysis.

        A single-sweep arm (`skill-oracle`, which `SkillOracleCli` pins to `num_cycles=0`)
        is handled by the slice rather than specially: `[-WINDOW:]` of a one-element list
        is that element, so the arm scores its one sweep and nothing is padded.
        """
        return Tossing3DResetFree.late_scores(curves=curves)

    @staticmethod
    def pooled(*, scores: dict[int, float], num_total: int) -> tuple[float, int]:
        """`(x, y)` for an `x/y` over every seed. Delegated for the same reason as
        `late_scores`."""
        return Tossing3DResetFree.pooled(scores=scores, num_total=num_total)

    @staticmethod
    def transitions_per_cycle(*, evaluations: list[tuple[int, int, int]]) -> list[int]:
        """How many environment steps each practice period actually took. Delegated, so
        the stranding definition is shared with #179's."""
        return Tossing3DResetFree.transitions_per_cycle(evaluations=evaluations)

    @staticmethod
    def stranding_onset(*, transitions: list[int]) -> int | None:
        """First cycle of the **terminal** run of zero-transition cycles, or `None`.
        Delegated, so this experiment and #179's mean the same thing by "stranded"."""
        return Tossing3DResetFree.stranding_onset(transitions=transitions)

    @staticmethod
    def compare(*, left: dict[int, float], right: dict[int, float]) -> PairedComparison:
        """`right - left`, paired over the seeds both arms ran.

        Intersected rather than assumed equal: if one arm lost a run, pairing on the union
        would silently compare a seed against nothing. Refuses outright on an empty
        intersection, because a "comparison" of zero pairs would otherwise report
        `p = 1.0` and read as a null result rather than as an absent one.
        """
        seeds = sorted(set(left) & set(right))
        if not seeds:
            raise ValueError(
                f"no shared seeds between the two arms (left={sorted(left)}, "
                f"right={sorted(right)}) -- nothing to pair"
            )
        differences = [right[seed] - left[seed] for seed in seeds]
        test = PairedTests.sign_flip(differences=differences)
        return PairedComparison(
            seeds=seeds,
            mean_difference=statistics.fmean(differences),
            p_value=test.p_value,
            minimum_detectable_effect=PairedTests.minimum_detectable_effect(
                differences=differences
            ),
            seeds_for_80_percent_power=PairedTests.seeds_for_80_percent_power(
                differences=differences
            ),
            num_right_lower=sum(1 for d in differences if d < 0),
            num_right_higher=sum(1 for d in differences if d > 0),
            num_tied=sum(1 for d in differences if d == 0.0),
        )

    @staticmethod
    def describe(*, label: str, comparison: PairedComparison) -> str:
        """One comparison as the lines this project's reports are read in. Built here
        rather than inline in `report` so the figure captions and the printed report
        cannot state a result two different ways."""
        n = comparison.num_seeds
        lines = [
            f"{label}, paired over {n}/{n} shared seeds:",
            f"  second arm lower on {comparison.num_right_lower}/{n}, "
            f"higher on {comparison.num_right_higher}/{n}, tied on {comparison.num_tied}/{n}",
            f"  mean per-seed difference {comparison.mean_difference:+.2f} tasks",
            f"  exact paired sign-flip p = {comparison.p_value:.6g}",
            f"  minimum detectable effect at 80% power: "
            f"{comparison.minimum_detectable_effect:.2f} tasks per seed",
        ]
        if comparison.is_null:
            lines.append(
                f"  NULL RESULT at alpha = {ALPHA}; "
                f"{comparison.seeds_for_80_percent_power:.0f} seeds would be needed for "
                "80% power at the observed effect"
            )
        return "\n".join(lines)


def _arm_late(
    *, results_root: Path, policy: str, method: str
) -> tuple[dict[int, float], dict[int, list[tuple[int, int, int]]]]:
    curves = Tossing3DReferenceArms.load_runs(
        results_root=results_root, policy=policy, method=method
    )
    return Tossing3DReferenceArms.late_scores(curves=curves), curves


def report(*, results_root: Path, ees_root: Path | None, output_prefix: Path | None) -> None:
    """Print every arm's `x/y`, the paired comparisons the experiment was run to make,
    and the practice-side stranding counts; optionally render the figures."""
    arms = {
        ("scheduled", "random-skills"): None,
        ("never", "random-skills"): None,
        ("scheduled", "skill-oracle"): None,
    }
    late: dict[tuple[str, str], dict[int, float]] = {}
    curves: dict[tuple[str, str], dict[int, list[tuple[int, int, int]]]] = {}
    for policy, method in arms:
        late[(policy, method)], curves[(policy, method)] = _arm_late(
            results_root=results_root, policy=policy, method=method
        )
    if ees_root is not None:
        for policy in ("scheduled", "never"):
            late[(policy, "ees")], curves[(policy, "ees")] = _arm_late(
                results_root=ees_root, policy=policy, method="ees"
            )

    present = {key: value for key, value in late.items() if value}
    if not present:
        print(f"No completed runs under {results_root}")
        return

    print(f"LATE window: mean solved over the last {WINDOW} sweeps (or all, if fewer)\n")
    for key, scores in present.items():
        policy, method = key
        num_total = curves[key][sorted(curves[key])[0]][0][2]
        x, y = Tossing3DReferenceArms.pooled(scores=scores, num_total=num_total)
        sweeps = min(len(c) for c in curves[key].values())
        print(
            f"  {method:>14} / {policy:<9}  {x:>5.1f}/{y:<4}  "
            f"({len(scores)} seeds, {sweeps} sweeps each)"
        )

    for label, left_key, right_key in (
        (
            "random-skills: never - scheduled",
            ("scheduled", "random-skills"),
            ("never", "random-skills"),
        ),
        (
            "reset-free: ees - random-skills",
            ("never", "random-skills"),
            ("never", "ees"),
        ),
        (
            "scheduled: ees - random-skills",
            ("scheduled", "random-skills"),
            ("scheduled", "ees"),
        ),
    ):
        if left_key not in present or right_key not in present:
            continue
        comparison = Tossing3DReferenceArms.compare(
            left=present[left_key], right=present[right_key]
        )
        print()
        print(Tossing3DReferenceArms.describe(label=label, comparison=comparison))

    print("\nper-seed LATE (the numbers every pooled x/y above is the column sum of):")
    columns = [key for key in present]
    print("  seed | " + " | ".join(f"{m}/{p}"[:22].rjust(22) for p, m in columns))
    shared = sorted(set.intersection(*(set(present[key]) for key in columns)))
    for seed in shared:
        print(f"  {seed:>4} | " + " | ".join(f"{present[key][seed]:>22.1f}" for key in columns))
    print("   SUM | " + " | ".join(f"{sum(present[key].values()):>22.1f}" for key in columns))

    print("\npractice actually taken (transitions per cycle):")
    for key, seed_curves in curves.items():
        if not seed_curves:
            continue
        policy, method = key
        per_seed = {
            seed: Tossing3DReferenceArms.transitions_per_cycle(evaluations=evaluations)
            for seed, evaluations in seed_curves.items()
        }
        num_cycles = len(next(iter(per_seed.values())))
        if num_cycles == 0:
            print(f"  {method} / {policy}: no practice cycles (num_cycles=0 by construction)")
            continue
        onsets = {
            seed: Tossing3DReferenceArms.stranding_onset(transitions=steps)
            for seed, steps in per_seed.items()
        }
        stranded = sum(1 for onset in onsets.values() if onset is not None)
        idle = sum(sum(1 for t in steps if t == 0) for steps in per_seed.values())
        total = sum(sum(steps) for steps in per_seed.values())
        print(f"  {method} / {policy}:")
        print(f"    total transitions      {total} over {len(per_seed)} seeds")
        print(f"    seeds ever stranded    {stranded}/{len(per_seed)}")
        print(f"    idle cycles (0 steps)  {idle}/{num_cycles * len(per_seed)}")
        for seed in sorted(per_seed):
            onset = onsets[seed]
            where = "never stranded" if onset is None else f"stranded from cycle {onset}"
            print(f"      seed {seed}: {sum(per_seed[seed])} transitions, {where}")

    if output_prefix is not None:
        render_curves(
            late=present,
            curves=curves,
            output=output_prefix.with_name(output_prefix.name + "-curves.png"),
        )
        render_paired(
            late=present, output=output_prefix.with_name(output_prefix.name + "-paired.png")
        )
        render_practice(
            curves=curves, output=output_prefix.with_name(output_prefix.name + "-practice.png")
        )


def render_practice(
    *,
    curves: dict[tuple[str, str], dict[int, list[tuple[int, int, int]]]],
    output: Path,
) -> None:
    """Cumulative practice transitions per cycle, one panel per method.

    This is the figure that carries the domain claim. A robot that keeps practising is a
    line that keeps rising; a stranded one goes flat and stays flat. Putting EES and
    `random-skills` in adjacent panels on a shared y axis is the whole argument that
    stranding belongs to the *domain* rather than to the learner: two methods with nothing
    in common but the operators collapse to the same handful of transitions.

    `skill-oracle` is absent rather than empty-panelled -- it runs `num_cycles=0`, so it
    has no practice to plot and a blank panel would suggest it practised and did nothing.
    """
    panels = [
        ("ees", "EES"),
        ("random-skills", "random-skills"),
    ]
    drawable = [
        (method, title)
        for method, title in panels
        if any(
            (policy, method) in curves and curves[(policy, method)]
            for policy in ("scheduled", "never")
        )
    ]
    if not drawable:
        return
    figure, axes_list = plt.subplots(
        1, len(drawable), figsize=(5.4 * len(drawable), 4.8), sharey=True
    )
    if len(drawable) == 1:
        axes_list = [axes_list]
    for axes, (method, title) in zip(axes_list, drawable, strict=True):
        for policy, colour in (("scheduled", _SCHEDULED_COLOUR), ("never", _NEVER_COLOUR)):
            key = (policy, method)
            if key not in curves or not curves[key]:
                continue
            seed_curves = curves[key]
            totals = []
            for evaluations in seed_curves.values():
                steps = Tossing3DReferenceArms.transitions_per_cycle(evaluations=evaluations)
                cumulative = []
                running = 0
                for step in steps:
                    running += step
                    cumulative.append(running)
                totals.append(running)
                axes.plot(
                    range(1, len(cumulative) + 1),
                    cumulative,
                    color=colour,
                    alpha=_SEED_ALPHA * 2.2,
                    linewidth=_SEED_WIDTH,
                )
            stranded = sum(
                1
                for evaluations in seed_curves.values()
                if Tossing3DReferenceArms.stranding_onset(
                    transitions=Tossing3DReferenceArms.transitions_per_cycle(
                        evaluations=evaluations
                    )
                )
                is not None
            )
            label = "env resets" if policy == "scheduled" else "never reset"
            axes.plot(
                [],
                [],
                color=colour,
                linewidth=_MEAN_WIDTH,
                label=(
                    f"{label} — {sum(totals)} transitions over {len(seed_curves)} seeds, "
                    f"{stranded}/{len(seed_curves)} stranded"
                ),
            )
        axes.set_yscale("symlog", linthresh=10)
        axes.set_xlabel("practice cycle")
        axes.set_title(f"{title} (100 cycles x 10 seeds)", fontsize=10)
        axes.legend(fontsize=8, loc="upper left")
        axes.grid(alpha=0.25)
    axes_list[0].set_ylabel("cumulative practice transitions")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    print(f"wrote {output}")


def render_curves(
    *,
    late: dict[tuple[str, str], dict[int, float]],
    curves: dict[tuple[str, str], dict[int, list[tuple[int, int, int]]]],
    output: Path,
) -> None:
    """Learning curves for the arms that learn, reference arms as flat lines.

    **Reference/non-manipulated arms are horizontal lines, never curves** -- neither
    `skill-oracle` nor `random-skills` learns, so a wandering line would invite a reader
    to hunt for a trend in a constant. `random-skills` appears twice here (once per reset
    policy) and both are drawn flat at their own pooled level for the same reason.
    """
    del late
    figure, axes = plt.subplots(figsize=(9.0, 5.4))
    learners = [
        (("scheduled", "ees"), _SCHEDULED_COLOUR, "-", "EES, env resets"),
        (("never", "ees"), _NEVER_COLOUR, "-", "EES, never reset"),
    ]
    for key, colour, style, label in learners:
        if key not in curves or not curves[key]:
            continue
        seed_curves = curves[key]
        for evaluations in seed_curves.values():
            axes.plot(
                range(len(evaluations)),
                [solved for _t, solved, _n in evaluations],
                color=colour,
                alpha=_SEED_ALPHA,
                linewidth=_SEED_WIDTH,
            )
        length = min(len(c) for c in seed_curves.values())
        means = [
            statistics.fmean(seed_curves[seed][index][1] for seed in seed_curves)
            for index in range(length)
        ]
        axes.plot(
            range(length),
            means,
            color=colour,
            linestyle=style,
            linewidth=_MEAN_WIDTH,
            label=f"{label} — mean, n={len(seed_curves)}",
        )

    references = [
        (("scheduled", "skill-oracle"), (0, (1, 1.6)), "skill-oracle"),
        (("scheduled", "random-skills"), (0, (5, 2)), "random-skills, env resets"),
        (("never", "random-skills"), (0, (2, 2)), "random-skills, never reset"),
    ]
    for key, style, label in references:
        if key not in curves or not curves[key]:
            continue
        scores = Tossing3DReferenceArms.late_scores(curves=curves[key])
        level = statistics.fmean(scores.values())
        num_total = curves[key][sorted(curves[key])[0]][0][2]
        x, y = Tossing3DReferenceArms.pooled(scores=scores, num_total=num_total)
        axes.axhline(
            level,
            color=_REFERENCE_COLOUR,
            linestyle=style,
            linewidth=1.7,
            label=f"{label} — {x:.1f}/{y}, n={len(scores)}",
        )

    axes.set_xlabel("practice cycle")
    axes.set_ylabel("solved per seed")
    axes.set_title("Tossing3D reference arms against EES (of 10 test tasks per sweep)")
    axes.set_ylim(bottom=0)
    axes.legend(fontsize=8, loc="upper left", framealpha=0.9)
    axes.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    print(f"\nwrote {output}")


def render_paired(*, late: dict[tuple[str, str], dict[int, float]], output: Path) -> None:
    """Per-seed slopes between the arms being compared, rather than two bars.

    With ten seeds a bar chart of two means hides one seed driving the whole effect; the
    per-seed line is what makes a unanimous move visible as unanimous.
    """
    pairs = [
        (
            ("never", "random-skills"),
            ("never", "ees"),
            "The question this experiment was run to answer",
            ("random-skills\n(never reset)", "EES\n(never reset)"),
        ),
        (
            ("scheduled", "random-skills"),
            ("never", "random-skills"),
            "The control: does random-skills collapse too?",
            ("random-skills\n(env resets)", "random-skills\n(never reset)"),
        ),
    ]
    drawable = [p for p in pairs if p[0] in late and p[1] in late]
    if not drawable:
        return
    figure, axes_list = plt.subplots(1, len(drawable), figsize=(5.0 * len(drawable), 5.2))
    if len(drawable) == 1:
        axes_list = [axes_list]
    for axes, (left_key, right_key, title, ticks) in zip(axes_list, drawable, strict=True):
        seeds = sorted(set(late[left_key]) & set(late[right_key]))
        comparison = Tossing3DReferenceArms.compare(left=late[left_key], right=late[right_key])
        for seed in seeds:
            axes.plot(
                [0, 1],
                [late[left_key][seed], late[right_key][seed]],
                # Orange wherever the reset-free arm is the subject, blue where the
                # comparison spans the reset policy itself -- the same role rule the
                # curves figure uses, so the two read as one report.
                color=_NEVER_COLOUR if left_key[0] == "never" else _SCHEDULED_COLOUR,
                alpha=0.55,
                linewidth=1.2,
                marker="o",
                markersize=3.5,
            )
        verdict = (
            f"null result, p = {comparison.p_value:.3g}"
            if comparison.is_null
            else f"p = {comparison.p_value:.3g}"
        )
        axes.set_xticks([0, 1])
        axes.set_xticklabels(list(ticks), fontsize=9)
        axes.set_xlim(-0.3, 1.3)
        axes.set_ylim(bottom=0)
        axes.set_ylabel("solved per seed")
        axes.set_title(
            f"{title}\n(of 10 test tasks, n={len(seeds)})\n"
            f"higher on {comparison.num_right_higher}/{len(seeds)}, "
            f"lower on {comparison.num_right_lower}/{len(seeds)}, "
            f"tied on {comparison.num_tied}/{len(seeds)} — {verdict}",
            fontsize=9,
        )
        axes.grid(alpha=0.25, axis="y")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--ees-root",
        type=Path,
        default=None,
        help="The EES reset-policy sweep, same <policy>/<method>/<seed> layout. Omit to "
        "report the reference arms alone.",
    )
    parser.add_argument("--output-prefix", type=Path, default=None)
    args = parser.parse_args()
    report(
        results_root=args.results_root,
        ees_root=args.ees_root,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    main()

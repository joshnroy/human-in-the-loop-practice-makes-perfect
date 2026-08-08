"""What does a genuinely reset-free arm cost on Tossing3D?

Post-run analysis only; it reads `--results-root` back in and never drives a `Method`.

## Why this measurement did not exist before

#178 ran `--practice-reset-policy never` against `scheduled` at 100 cycles x 10 seeds and
found the two arms byte-identical in every `stats.json` field except
`num_practice_resets`. The flag was a no-op on this domain: `PracticeLoop` sampled the
train task before the reset-policy branch, and `Tossing3DTasks.build_task` could only
build an initial `State` by rebuilding the MuJoCo scene, so the scene was rebuilt every
cycle whatever the flag said. The reset-free condition was never realised, so there is no
earlier number here to compare against -- this is the first measurement of the arm, not a
re-analysis of one.

## What "reset-free" now means on this domain, which is not what it means elsewhere

After the fix, a reset-free run practises in **one scene for its whole length** --
whatever `hard_reset` left behind. That is not an implementation shortcut. On Tossing3D,
handing the robot a new scene and resetting it are the *same physical act*: the only way
to obtain a new initial state is `env.reset(seed=...)`. So the arm is "no reset" and "no
scene variety" inseparably, and a result about it is a result about both together. Any
reading that attributes the whole difference to the missing rescue is over-claiming.

## The rule, taken from #178 rather than invented here

Tossing3D's per-seed score is volatile between adjacent sweeps -- several tasks, with no
learning event in between -- so a single final sweep is one draw of a noisy variable.
#178 fixed a windowed rule before reading its numbers, and it is reused unchanged
because the volatility is a property of the domain, not of that experiment:

- A seed's score is its **mean solved count over the last `WINDOW` sweeps** (`LATE`).
- **Paired across seeds**, because both arms ran the same fixed seed set. An unpaired
  test would discard exactly the structure the design bought.
- Exact paired sign-flip on per-seed `never - scheduled`, and **the MDE beside it,
  always**: a null result without one cannot be told apart from an underpowered test.

Counts are reported `x/y`, never as a bare percentage.
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.practice_makes_perfect.paired_tests import PairedTests  # noqa: E402

# Same window as #178, for the same reason: wide enough to damp the measured
# sweep-to-sweep volatility, and the two experiments stay comparable.
WINDOW = 10

# The arm directory names under `--results-root`, in the order they are reported and
# plotted. `scheduled` first because it is the incumbent every committed number sits on.
SCHEDULED = "scheduled"
NEVER = "never"

# #133's two non-learning arms on this domain, per seed out of 10 tasks. Horizontal
# REFERENCE LINES, never curves: neither learns, so a curve invites a reader to look for
# a trend in a constant.
ORACLE_PER_SEED = 10.0
RANDOM_SKILLS_PER_SEED = 2.4
ORACLE_LABEL = "skill-oracle ceiling — 100/100 (#133)"
RANDOM_SKILLS_LABEL = "random-skills — 24/100 (#133)"

_ARM_COLOURS = {SCHEDULED: "#2166ac", NEVER: "#d6604d"}


class Tossing3DResetFree:
    """A static-method container, never instantiated, same as every other business-logic
    class in this project."""

    @staticmethod
    def load_arms(*, results_root: Path) -> dict[str, dict[int, list[tuple[int, int, int]]]]:
        """`{arm: {seed: evaluations}}`, each entry
        `(num_online_transitions, num_solved, num_total)` for one sweep.

        Read from `stats.json`, so only completed runs appear. A partially-finished run
        has no `stats.json` and is silently absent rather than contributing a truncated
        curve that would shorten every window computed from it.

        **Two layouts, both accepted, because the tree moves between them.**
        `scripts/run_sweep.py` writes `<arm>/<method>/<seed>/`, while a committed
        `docs/experiment-logs/` tree drops the method level and is `<arm>/<seed>/`. A
        glob fixed to one depth finds nothing under the other and `report` then prints
        "No completed runs" and exits 0 -- a silent wrong answer, not a failure. So the
        search is recursive and the **seed is the containing directory's name**, which is
        true of both.

        Deliberately rooted at `results_root / arm` rather than at `results_root`: keying
        on the containing directory alone would collide `scheduled/0` with `never/0` into
        one entry. That collision is invisible for as long as two arms agree, which is
        exactly the condition this whole experiment exists because of.
        """
        arms: dict[str, dict[int, list[tuple[int, int, int]]]] = {}
        for arm in (SCHEDULED, NEVER):
            curves: dict[int, list[tuple[int, int, int]]] = {}
            for path in sorted((results_root / arm).rglob("stats.json")):
                if not path.parent.name.isdigit():
                    continue
                stats = json.loads(path.read_text())
                curves[int(path.parent.name)] = [
                    (int(t), int(s), int(n)) for t, s, n in stats["evaluations"]
                ]
            if curves:
                arms[arm] = curves
        return arms

    @staticmethod
    def transitions_per_cycle(*, evaluations: list[tuple[int, int, int]]) -> list[int]:
        """How many environment steps each practice period actually took.

        `evaluations[i][0]` is `num_online_transitions` as of sweep `i`, and sweep 0
        happens before any practice, so cycle `i`'s cost is the difference between
        consecutive sweeps. A **zero** means the period took no step at all.

        That is the quantity a reset-free arm on this domain lives or dies by, and it is
        not the same question as task success. A tossed cube ends up past an immovable
        barrier and no skill brings it back, while `Toss` deletes `Reachable` -- so after
        a throw nothing is applicable. With a per-period reset the next period starts
        fresh; with none, the robot can simply stop accumulating experience. An arm that
        scores lower because it practised less is a different finding from one that
        scores lower because it learned less, and only this separates them.
        """
        return [evaluations[i + 1][0] - evaluations[i][0] for i in range(len(evaluations) - 1)]

    @staticmethod
    def stranding_onset(*, transitions: list[int]) -> int | None:
        """The first cycle of the **terminal** run of zero-transition cycles, or `None`.

        Terminal-from-here, not "the first gap" -- the same definition
        `pickup_weight_stranding.py` uses on Tossing Room, kept identical so the two
        experiments can be read side by side. A run that takes no step for one period and
        then resumes was never stranded, and calling it stranded would promote ordinary
        exploration noise into the effect being claimed. A run that never moves at all
        strands at cycle 0; a run that moves in its last period reports `None`.
        """
        onset: int | None = None
        for index in range(len(transitions) - 1, -1, -1):
            if transitions[index] != 0:
                break
            onset = index
        return onset

    @staticmethod
    def late_scores(*, curves: dict[int, list[tuple[int, int, int]]]) -> dict[int, float]:
        """Each seed's mean solved count over the last `WINDOW` sweeps.

        A float, not a count, because it is a mean of counts. Callers that need an `x/y`
        pool it back up against the same denominator -- see `pooled`.
        """
        return {
            seed: statistics.fmean(solved for _t, solved, _n in evaluations[-WINDOW:])
            for seed, evaluations in curves.items()
        }

    @staticmethod
    def pooled(*, scores: dict[int, float], num_total: int) -> tuple[float, int]:
        """`(x, y)` for an `x/y` over every seed: the summed mean-solved against the
        summed denominator. `x` stays a float because each seed's contribution is a
        window mean, and rounding it here would hide that."""
        return sum(scores.values()), num_total * len(scores)

    @staticmethod
    def shared_seeds(*, arms: dict[str, dict[int, list[tuple[int, int, int]]]]) -> list[int]:
        """Seeds present in BOTH arms, which is what a paired test may use.

        Intersected rather than assumed equal: if one arm lost a run, pairing on the
        union would silently compare a seed against nothing.
        """
        return sorted(set(arms[SCHEDULED]) & set(arms[NEVER]))

    @staticmethod
    def report(*, results_root: Path) -> None:
        arms = Tossing3DResetFree.load_arms(results_root=results_root)
        missing = [arm for arm in (SCHEDULED, NEVER) if arm not in arms]
        if missing:
            print(f"No completed runs for {missing} under {results_root}")
            return
        seeds = Tossing3DResetFree.shared_seeds(arms=arms)
        num_total = arms[SCHEDULED][seeds[0]][0][2]
        for arm in (SCHEDULED, NEVER):
            sweeps = min(len(c) for c in arms[arm].values())
            print(f"{arm:>9}: {len(arms[arm])} seeds {sorted(arms[arm])}, {sweeps} sweeps each")
        print(f"paired on {len(seeds)}/{len(seeds)} shared seeds, {num_total} test tasks each")

        late = {arm: Tossing3DResetFree.late_scores(curves=arms[arm]) for arm in arms}
        print(f"\nLATE window (last {WINDOW} sweeps):")
        for arm in (SCHEDULED, NEVER):
            x, y = Tossing3DResetFree.pooled(
                scores={s: late[arm][s] for s in seeds}, num_total=num_total
            )
            print(f"  {arm:>9}: {x:.1f}/{y}")

        differences = [late[NEVER][s] - late[SCHEDULED][s] for s in seeds]
        test = PairedTests.sign_flip(differences=differences)
        mde = PairedTests.minimum_detectable_effect(differences=differences)
        better = sum(1 for d in differences if d > 0)
        worse = sum(1 for d in differences if d < 0)
        print(f"\nnever - scheduled, paired over {len(differences)} seeds:")
        print(f"  never higher on {better}/{len(differences)}, lower on {worse}/{len(differences)}")
        print(f"  mean per-seed difference {statistics.fmean(differences):+.2f} tasks")
        print(f"  exact paired sign-flip p = {test.p_value:.6g}")
        print(f"  minimum detectable effect at 80% power: {mde:.2f} tasks per seed")

        print(f"\nper seed ({SCHEDULED} -> {NEVER}, difference):")
        for seed in seeds:
            d = late[NEVER][seed] - late[SCHEDULED][seed]
            print(
                f"  seed {seed}: {late[SCHEDULED][seed]:.1f} -> {late[NEVER][seed]:.1f}  ({d:+.1f})"
            )

        Tossing3DResetFree.report_stranding(arms=arms, seeds=seeds)

    @staticmethod
    def report_stranding(
        *, arms: dict[str, dict[int, list[tuple[int, int, int]]]], seeds: list[int]
    ) -> None:
        """Did the reset-free arm learn less, or did it stop practising altogether?

        Reported before any interpretation of the score gap, because the two support
        very different claims and the score alone cannot tell them apart.
        """
        print("\npractice actually taken (transitions per cycle):")
        for arm in (SCHEDULED, NEVER):
            per_seed = {
                seed: Tossing3DResetFree.transitions_per_cycle(evaluations=arms[arm][seed])
                for seed in seeds
            }
            num_cycles = len(next(iter(per_seed.values())))
            idle = {seed: sum(1 for t in steps if t == 0) for seed, steps in per_seed.items()}
            onsets = {
                seed: Tossing3DResetFree.stranding_onset(transitions=steps)
                for seed, steps in per_seed.items()
            }
            stranded = sum(1 for onset in onsets.values() if onset is not None)
            total = sum(sum(steps) for steps in per_seed.values())
            print(f"  {arm}:")
            print(f"    total transitions      {total} over {len(seeds)} seeds")
            print(f"    seeds ever stranded    {stranded}/{len(seeds)}")
            print(
                f"    idle cycles (0 steps)  "
                f"{sum(idle.values())}/{num_cycles * len(seeds)} across all seeds"
            )
            for seed in seeds:
                onset = onsets[seed]
                where = "never stranded" if onset is None else f"stranded from cycle {onset}"
                print(
                    f"      seed {seed}: {sum(per_seed[seed])} transitions, "
                    f"{idle[seed]}/{num_cycles} idle cycles, {where}"
                )

    @staticmethod
    def render_curves(
        *, arms: dict[str, dict[int, list[tuple[int, int, int]]]], output: Path
    ) -> None:
        """Both arms' learning curves: faint per-seed lines under a bold pooled mean.

        **Cycles, not online transitions, on the x axis.** The cycle is the controlled
        variable -- every seed ran exactly 100 -- while transitions vary per seed because
        a Tossing3D practice period ends early. Against cycles the seeds share a grid and
        the per-seed spread is readable.

        The per-seed haze is the point, not decoration: the measured sweep-to-sweep swing
        is several tasks, so a bold mean alone would imply a smoothness the data does not
        have, and with ten seeds a gap can be one seed wide.
        """
        fig, axes = plt.subplots(1, 1, figsize=(11.0, 6.2))
        num_total = arms[SCHEDULED][sorted(arms[SCHEDULED])[0]][0][2]
        for arm in (SCHEDULED, NEVER):
            curves = arms[arm]
            cycles = list(range(min(len(c) for c in curves.values())))
            colour = _ARM_COLOURS[arm]
            for seed in sorted(curves):
                axes.plot(
                    cycles,
                    [curves[seed][i][1] for i in cycles],
                    color=colour,
                    alpha=0.18,
                    linewidth=0.9,
                )
            pooled_curve = [statistics.fmean(curves[seed][i][1] for seed in curves) for i in cycles]
            x, y = Tossing3DResetFree.pooled(
                scores=Tossing3DResetFree.late_scores(curves=curves), num_total=num_total
            )
            axes.plot(
                cycles,
                pooled_curve,
                color=colour,
                linewidth=2.4,
                label=f"{arm} — last {WINDOW} sweeps {x:.1f}/{y}",
            )
        axes.axhline(
            ORACLE_PER_SEED, color="#1a9850", linestyle="--", linewidth=1.8, label=ORACLE_LABEL
        )
        axes.axhline(
            RANDOM_SKILLS_PER_SEED,
            color="#762a83",
            linestyle=":",
            linewidth=1.8,
            label=RANDOM_SKILLS_LABEL,
        )
        axes.grid(alpha=0.25, linewidth=0.6)
        axes.set_ylim(-num_total * 0.04, num_total * 1.08)
        axes.set_xlabel("practice cycle")
        axes.set_ylabel(f"test tasks solved per seed (x/{num_total})")
        axes.set_title(
            "Reset-free against scheduled practice on Tossing3D — "
            "bold pooled mean over faint per-seed lines",
            fontsize=10.5,
        )
        axes.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
        fig.tight_layout()
        fig.savefig(output, dpi=150)
        plt.close(fig)
        print(f"wrote {output}")

    @staticmethod
    def render_paired(
        *, arms: dict[str, dict[int, list[tuple[int, int, int]]]], output: Path
    ) -> None:
        """One line per seed joining its `scheduled` LATE score to its `never` LATE score.

        Plotted per seed rather than as two bars because with ten seeds a bar chart of
        two means hides one seed driving the whole movement. Ten near-flat lines is a
        null result; ten lines sloping the same way is a real one.
        """
        seeds = Tossing3DResetFree.shared_seeds(arms=arms)
        num_total = arms[SCHEDULED][seeds[0]][0][2]
        late = {arm: Tossing3DResetFree.late_scores(curves=arms[arm]) for arm in arms}
        differences = [late[NEVER][s] - late[SCHEDULED][s] for s in seeds]
        test = PairedTests.sign_flip(differences=differences)
        mde = PairedTests.minimum_detectable_effect(differences=differences)

        fig, axes = plt.subplots(1, 1, figsize=(7.4, 6.2))
        for seed in seeds:
            fell = late[NEVER][seed] < late[SCHEDULED][seed]
            axes.plot(
                [0, 1],
                [late[SCHEDULED][seed], late[NEVER][seed]],
                marker="o",
                markersize=5,
                color="#b2182b" if fell else "#2166ac",
                alpha=0.75,
                linewidth=1.4,
            )
        axes.axhline(
            ORACLE_PER_SEED, color="#1a9850", linestyle="--", linewidth=1.6, label=ORACLE_LABEL
        )
        axes.axhline(
            RANDOM_SKILLS_PER_SEED,
            color="#762a83",
            linestyle=":",
            linewidth=1.6,
            label=RANDOM_SKILLS_LABEL,
        )
        axes.set_xticks([0, 1])
        axes.set_xticklabels([
            f"{SCHEDULED}\n(reset each cycle)",
            f"{NEVER}\n(one scene, no reset)",
        ])
        axes.set_xlim(-0.25, 1.25)
        axes.set_ylim(-num_total * 0.04, num_total * 1.08)
        axes.set_ylabel(
            f"mean test tasks solved per seed over last {WINDOW} sweeps (x/{num_total})"
        )
        fell_count = sum(1 for d in differences if d < 0)
        axes.set_title(
            f"What does a genuinely reset-free arm cost?\n"
            f"fell on {fell_count}/{len(differences)} seeds, "
            f"mean {statistics.fmean(differences):+.2f} tasks, "
            f"p = {test.p_value:.4g}, MDE {mde:.2f}",
            fontsize=10,
        )
        axes.grid(alpha=0.25, linewidth=0.6, axis="y")
        axes.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
        fig.tight_layout()
        fig.savefig(output, dpi=150)
        plt.close(fig)
        print(f"wrote {output}")

    @staticmethod
    def render_practice(
        *, arms: dict[str, dict[int, list[tuple[int, int, int]]]], output: Path
    ) -> None:
        """Cumulative practice transitions against cycle, per seed, for both arms.

        The figure that separates "learned less" from "practised less". A robot that
        keeps practising is a line that keeps rising; a stranded one is a line that goes
        flat and stays flat, and the cycle it flattens at is its stranding onset read
        straight off the axis. Per seed rather than pooled, because a mean over ten seeds
        with different onsets describes no seed.
        """
        fig, axes = plt.subplots(1, 1, figsize=(9.0, 6.0))
        for arm in (SCHEDULED, NEVER):
            curves = arms[arm]
            colour = _ARM_COLOURS[arm]
            for index, seed in enumerate(sorted(curves)):
                cumulative = [t for t, _s, _n in curves[seed]]
                axes.plot(
                    range(len(cumulative)),
                    cumulative,
                    color=colour,
                    alpha=0.55,
                    linewidth=1.2,
                    label=arm if index == 0 else None,
                )
        axes.grid(alpha=0.25, linewidth=0.6)
        axes.set_xlabel("practice cycle")
        axes.set_ylabel("cumulative practice transitions")
        axes.set_title(
            "Did the reset-free arm learn less, or practise less?\n"
            "one line per seed; a flat line is a robot that has stopped acting",
            fontsize=10.5,
        )
        axes.legend(fontsize=9, loc="upper left", framealpha=0.95)
        fig.tight_layout()
        fig.savefig(output, dpi=150)
        plt.close(fig)
        print(f"wrote {output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--curves-output", type=Path, default=None)
    parser.add_argument("--paired-output", type=Path, default=None)
    parser.add_argument("--practice-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    Tossing3DResetFree.report(results_root=args.results_root)
    outputs = (args.curves_output, args.paired_output, args.practice_output)
    if any(output is not None for output in outputs):
        arms = Tossing3DResetFree.load_arms(results_root=args.results_root)
        if args.curves_output is not None:
            Tossing3DResetFree.render_curves(arms=arms, output=args.curves_output)
        if args.paired_output is not None:
            Tossing3DResetFree.render_paired(arms=arms, output=args.paired_output)
        if args.practice_output is not None:
            Tossing3DResetFree.render_practice(arms=arms, output=args.practice_output)


if __name__ == "__main__":
    main()
